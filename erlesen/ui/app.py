################################################################################
#
# ErLeSen synthetic data generation
################################################################################
import os
from pathlib import Path

import gradio as gr
import spacy
import textstat
import torch
from huggingface_hub import InferenceClient
import threading
from evaluate import load
from textstat.textstat import textstatistics
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from erlesen.evaluation.grammar_checker import grammar_evaluation

from erlesen import prompts

# Configuration
base_url = os.getenv("TGI_URL")
if not base_url:
    base_url = "http://127.0.0.1:8080"
client = InferenceClient(base_url=base_url)

# A threading Event to signal stopping text generation
stop_event = threading.Event()
# Load SARI metric
sari = load("sari")
# Load readability model and tokenizer
preference_model_name = "agentlans/mdeberta-v3-base-readability"

max_generation_tokens = 2048
max_chars = 3000

preference_tokenizer = AutoTokenizer.from_pretrained(preference_model_name)
preference_model = AutoModelForSequenceClassification.from_pretrained(preference_model_name)

nlp = spacy.load("de_dep_news_trf")

device = torch.device("cpu")
preference_model = preference_model.to(device)

system_message = (Path(prompts.__file__).parent / "system_message.txt").read_text()

# -------------------------------------------------------------------------
# Simplify Function
# Description:
#   Given a complex input text, sends it to a local TGI server for a
#   simplified version. Uses streaming to update the output in real time
#   until either complete or a stop signal is triggered.
# -------------------------------------------------------------------------
def simplify_text(complex_text):
    global stop_event
    response = ""
    stop_event.clear()  # Reset stop event at the beginning

    output = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": complex_text},
        ],
        stream=True,
        max_tokens=max_generation_tokens,
    )

    # Stream the generated text pieces
    for chunk in output:
        if stop_event.is_set():
            break
        response += chunk.choices[0].delta.content
        yield response


# -------------------------------------------------------------------------
# Function: highlight_violations
# Description:
#   Highlight detected violations
# -------------------------------------------------------------------------
def highlight_violations_nested(text: str, violations: list[dict]) -> str:
    """
    Erstellt verschachteltes HTML-Markup für alle Verstöße in `violations`.
    Bei überlappenden Bereichen werden Spans ineinander verschachtelt,
    statt textuell dupliziert.
    """

    # Beispiel-Farbtabelle
    color_map = {
        "genitive":           "rgba(255, 245, 157, 0.8)",  # helles Gelb, 80% Deckkraft
        "passive":            "rgba(129, 212, 250, 0.8)",  # helles Blau
        "konjunktiv":         "rgba(239, 154, 154, 0.8)",
        "subclause":          "rgba(179, 157, 219, 0.7)",
        "complex_subclause":  "rgba(179, 157, 219, 0.7)",
        "no_svo":             "rgba(255, 204, 128, 0.8)",
        "long_noun_phrase":   "rgba(255, 213, 79, 0.8)",
        "long_sentence":      "rgba(255, 171, 145, 0.8)",
        "long_word":          "rgba(165, 214, 167, 0.8)",
    }

    # Alle Start/End-Ereignisse sammeln
    events = []  # (pos, event_type, violation_type)

    for v in violations:
        vtype = v["type"]
        for (start, end) in v["index"]:
            # Start-Ereignis
            events.append((start, "start", vtype))
            # End-Ereignis
            events.append((end, "end", vtype))

    # Sortieren: zuerst nach position,
    # bei Gleichstand "end" vor "start",
    # damit sich Spans sauber schließen/öffnen
    # (also kein Null-Width-Span übrig bleibt).
    events.sort(key=lambda e: (e[0], 0 if e[1] == "end" else 1))

    # Durch den Text laufen und Spans öffnen/schließen
    output = []
    stack = []  # hält die offenen Violation-Typen
    current_pos = 0

    def open_span(vtype):
        """Span für einen Verstoßtyp öffnen."""
        color = color_map.get(vtype, "rgba(255, 205, 210, 0.8)")
        return f'<span style="background-color:{color}" data-type="{vtype}">'

    def close_span(vtype):
        """Span schließen (entspricht open_span)."""
        return "</span>"

    for (pos, etype, vtype) in events:
        # Text zwischen current_pos und pos unverändert einfügen
        if pos > current_pos:
            # Dieser Teil des Texts ist "innen" in allen aktuell offenen Spans
            snippet = text[current_pos:pos]
            output.append(snippet)
            current_pos = pos

        if etype == "start":
            # Span öffnen
            stack.append(vtype)
            output.append(open_span(vtype))
        else:  # etype == "end"
            # Möglicherweise ist der Span noch in der Mitte des Stacks
            # => wir müssen zurück bis wir ihn finden
            if vtype in stack:
                # Schließe Spans bis zum passenden
                while stack:
                    top = stack.pop()
                    output.append(close_span(top))
                    if top == vtype:
                        break
                # Danach öffnen wir alles wieder,
                # was eigentlich noch offen sein sollte (danach),
                # damit die Reihenfolge passt:
                reopens = []
                while events and stack:
                    reopens.append(stack.pop())
                # ^ In vielen Fällen bräuchte man hier
                #   die restlichen Events nicht, sondern
                #   merkt sich separat, was man geöffnet hatte.
                #   Hier ein vereinfachtes Beispiel.
                #   Korrekte Handhabung von verschachtelten Overlaps
                #   kann etwas mehr Logik erfordern.

                # In einem einfachen (strictly nested) Szenario
                # könnte man direkt ab hier neu öffnen:
                for t in reversed(reopens):
                    output.append(open_span(t))
                    stack.append(t)

            # Wenn der Span nicht mehr existiert (z.B. doppelt geschlossen),
            # ignorieren wir es.

    # Am Ende restlichen Text anfügen
    if current_pos < len(text):
        output.append(text[current_pos:])

    # Noch offene Spans schließen
    while stack:
        top = stack.pop()
        output.append(close_span(top))

    # Insgesamt zusammenfügen
    highlighted = "".join(output)
    return f'<div style="white-space: pre-wrap;">{highlighted}</div>'

def get_color_legend():
    color_map = {
        "genitive":           "rgba(255, 245, 157, 0.8)",
        "passive":            "rgba(129, 212, 250, 0.8)",
        "konjunktiv":         "rgba(239, 154, 154, 0.8)",
        "subclause":          "rgba(179, 157, 219, 0.7)",
        "complex_subclause":  "rgba(179, 157, 219, 0.7)",
        "no_svo":             "rgba(255, 204, 128, 0.8)",
        "long_noun_phrase":   "rgba(255, 213, 79, 0.8)",
        "long_sentence":      "rgba(255, 171, 145, 0.8)",
        "long_word":          "rgba(165, 214, 167, 0.8)",
    }

    # Du kannst hier dynamisch oder statisch schreiben.
    # Beispielhaft bauen wir ein HTML-Grid.
    legend_html_parts = []
    legend_html_parts.append('<div style="display: flex; flex-wrap: wrap; gap: 1rem;">')

    for vtype, color in color_map.items():
        block = (
            f'<div style="display: flex; align-items: center; gap: 0.25em;">'
            f'  <div style="width: 1em; height: 1em; background-color:{color};"></div>'
            f'  <span>{vtype}</span>'
            f'</div>'
        )
        legend_html_parts.append(block)

    legend_html_parts.append("</div>")
    return "".join(legend_html_parts)

# -------------------------------------------------------------------------
# Function: stop_generation
# Description:
#   Sets the stop_event to True, which signals the simplify_text generator
#   to stop yielding new tokens.
# -------------------------------------------------------------------------
def stop_generation():
    global stop_event
    stop_event.set()
    return "Generierung gestoppt."


# -------------------------------------------------------------------------
# Evaluation: SARI & Readability Metrics
# Description:
#   Takes a reference text, the simplified text, and the source text, then
#   computes the SARI score. Also computes additional readability metrics
#   like Flesch Reading Ease, LIX, and a custom model-based "Readability."
# -------------------------------------------------------------------------
def evaluate_sari(simplified, source):
    metrics = []

    # --- 1) Lesbarkeitsmetriken (wie bisher) ---
    stst = textstatistics()
    stst.set_lang("de")

    # Flesch Reading Ease (German)
    fre_score = stst.flesch_reading_ease(simplified)
    metrics.append(["flesch_reading_ease [0 (hard), ..., 100 (easy)]", fre_score])

    # LIX score (rough measure of difficulty)
    lix_score = stst.lix(simplified)
    metrics.append(["lix ... 40 (children's lit) ... 60 (technical)", lix_score])

    # Model-based readability
    inputs = preference_tokenizer(simplified, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = preference_model(**inputs).logits.squeeze().cpu()
    metrics.append(["Readability (Modell)", round(logits.item(), 4)])

    # --- 2) Grammar-Check ---
    grammar_score, violations = grammar_evaluation(simplified, nlp)
    metrics.append(["Grammar-Score [0..1]", round(grammar_score, 4)])

    # Annotierten Text erzeugen (HTML)
    annotated_html = highlight_violations_nested(simplified, violations)
    legend_html = get_color_legend()

    return metrics, annotated_html, legend_html


# -------------------------------------------------------------------------
# Utility: count_text_stats
# Description:
#   Given a text, returns a string summarizing character and word counts.
# -------------------------------------------------------------------------
def count_text_stats(text):
    chars = len(text)
    words = len(text.split())
    return f"**Characters** {chars}/{max_chars} **Words** {words}"


# -------------------------------------------------------------------------
# Main Gradio Interface
# Description:
#   Defines the UI layout and connects functions to buttons/inputs.
# -------------------------------------------------------------------------
def interface():
    theme = gr.themes.Monochrome(
        text_size="lg",
        font=[gr.themes.GoogleFont("Verdana"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        body_text_color_subdued_dark="*neutral_600",
        body_text_weight="600",
        button_large_radius="*radius_lg",
        button_small_radius="*radius_lg",
        button_small_text_size="*text_md",
        button_medium_radius="*radius_lg",
        button_medium_text_size="*text_lg",
    )

    with gr.Blocks(theme=theme) as demo:
        gr.Markdown("# Automatische Textvereinfachung")
        gr.Markdown("Geben Sie Text ein und lassen Sie ihn automatisch vereinfachen.")

        with gr.Row():
            with gr.Column(elem_id="input_div"):
                input_box = gr.Textbox(
                    label="Ausgangstext",
                    info="Geben Sie hier den Text ein, der vereinfacht werden soll.",
                    placeholder="Geben Sie hier den komplexen Text ein...",
                    lines=15,
                    max_lines=15,
                    elem_id="input_box",
                    autofocus=True,
                    scale=2,
                    max_length=max_chars,
                )

                with gr.Row(elem_id="input_div"):
                    stats = gr.Markdown(
                        value=f"**Characters** -/{max_chars} **Words** -",
                        elem_id="stats"
                    )

                    simplify_button = gr.Button(
                        "Vereinfachen",
                        elem_id="simplify_button"
                    )
                    stop_button = gr.Button(
                        "Stop",
                        elem_id="stop_button"
                    )

            with gr.Column():
                output_box = gr.Textbox(
                    label="Vereinfachung",
                    info="Hier erscheint die KI-Ausgabe",
                    placeholder="Der vereinfachte Text erscheint hier...",
                    lines=15,
                    max_lines=15,
                    elem_id="output_box",
                    interactive=True,
                    show_copy_button=True,
                    scale=2,
                    max_length=max_chars,
                )

        with gr.Accordion("Metriken", open=False):
            evaluate_button = gr.Button("Evaluieren")
            evaluation_output = gr.Dataframe(
                headers=["Metrik", "Wert"],
                elem_id="evaluation_output",
                interactive=False,
            )
            grammar_annotated_html = gr.HTML(
                label="Grammatische Analyse (Hervorhebungen)"
            )
            grammar_legend = gr.HTML(
                label="Legende"
            )

        # Wire UI components to functions
        simplify_button.click(
            fn=simplify_text,
            inputs=[input_box],
            outputs=[output_box]
        )
        stop_button.click(
            fn=stop_generation,
            inputs=[],
            outputs=[output_box]
        )
        input_box.change(
            fn=count_text_stats,
            inputs=[input_box],
            outputs=[stats]
        )
        evaluate_button.click(
            fn=evaluate_sari,
            inputs=[output_box, input_box],
            outputs=[evaluation_output, grammar_annotated_html, grammar_legend],
        )

    demo.launch(server_name="0.0.0.0")


# -------------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    interface()
