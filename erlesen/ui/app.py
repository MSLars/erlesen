################################################################################
#
# ErLeSen synthetic data generation
################################################################################
import os
from pathlib import Path
import threading

import gradio as gr
import spacy
import textstat
import torch
from huggingface_hub import InferenceClient
from evaluate import load
from textstat.textstat import textstatistics
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import language_tool_python  # Für Rechtschreib- und Grammatikprüfung

# Initialisiere die beiden LanguageTool-Instanzen
tool_grammar = language_tool_python.LanguageTool('de-DE')
tool_simple = language_tool_python.LanguageTool('de-DE-x-simple-language')

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

    # Stream der generierten Textteile
    for chunk in output:
        if stop_event.is_set():
            break
        response += chunk.choices[0].delta.content
        yield response


# -------------------------------------------------------------------------
# Neue Funktion: highlight_language_errors
# Description:
#   Hebt mithilfe von LanguageTool erkannte Rechtschreib- und Grammatikfehler hervor.
#   Fehlerhafte Textstellen werden unterstrichen und zusätzlich mit einem blass markierten
#   Hintergrund in der entsprechenden Farbe hervorgehoben.
# -------------------------------------------------------------------------
def highlight_language_errors(text: str, matches) -> str:
    # Farbtabelle für starke Farbe (für die Unterstreichung)
    color_map = {
        "TYPOS": "#1500a0",
        "GRAMMAR": "#2f8f25",
        "MISC": "#1500a0",
        "STYLE": "#27cfe5",
        "DIFFICULT_WORDS": "#0986a7"
    }
    # Blassere Farben für den Hintergrund
    lighter_color_map = {
        "TYPOS": "#e6d9ff",
        "GRAMMAR": "#d4eacb",
        "MISC": "#e6d9ff",
        "STYLE": "#ccf2f9",
        "DIFFICULT_WORDS": "#cceef5"
    }

    events = []
    for match in matches:
        # Optional: Bestimmte Fälle überspringen (z. B. Bindestrich-Fälle)
        if match.category == "TYPOS":
            if "-" in match.matchedText:
                if match.matchedText.replace("-", "").lower() in [r.lower() for r in match.replacements]:
                    continue

        start = match.offset
        end = match.offset + match.errorLength
        category = str(match.category)
        message = match.message
        events.append((start, "start", category, message))
        events.append((end, "end", category, message))
    # Sortiere: Bei gleicher Position kommt "end" vor "start"
    events.sort(key=lambda e: (e[0], 0 if e[1] == "end" else 1))

    output = []
    stack = []
    current_pos = 0

    def open_span(category, message):
        color = color_map.get(category, "#FFA500")
        lighter_color = lighter_color_map.get(category, "#FFF5CC")
        return (
            f'<span style="'
            f'background-color: {lighter_color}; '
            f'text-decoration: underline; '
            f'text-decoration-color: {color}; '
            f'cursor: help;" '
            f'title="{message}">'
        )

    def close_span():
        return "</span>"

    for pos, etype, category, message in events:
        if pos > current_pos:
            snippet = text[current_pos:pos]
            output.append(snippet)
            current_pos = pos

        if etype == "start":
            output.append(open_span(category, message))
            stack.append(category)
        elif etype == "end":
            if stack:
                stack.pop()
                output.append(close_span())
    if current_pos < len(text):
        output.append(text[current_pos:])
    while stack:
        stack.pop()
        output.append(close_span())
    highlighted = "".join(output)
    return f'<div style="white-space: pre-wrap;">{highlighted}</div>'


# -------------------------------------------------------------------------
# Funktion: stop_generation
# -------------------------------------------------------------------------
def stop_generation():
    global stop_event
    stop_event.set()
    return "Generierung gestoppt."


# -------------------------------------------------------------------------
# Evaluation: SARI & Readability Metrics
# -------------------------------------------------------------------------
def evaluate_sari(simplified, source):
    metrics = []
    stst = textstatistics()
    stst.set_lang("de")

    # Flesch Reading Ease (German)
    fre_score = stst.flesch_reading_ease(simplified)
    metrics.append(["flesch_reading_ease [0 (schwer), ..., 100 (einfach)]", fre_score])

    # LIX score
    lix_score = stst.lix(simplified)
    metrics.append(["lix ... 40 (Kinderliteratur) ... 60 (Fachtexte)", lix_score])

    # Modell-basierte Readability
    inputs = preference_tokenizer(simplified, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = preference_model(**inputs).logits.squeeze().cpu()
    metrics.append(["Readability (Modell)", round(logits.item(), 4)])

    return metrics


# -------------------------------------------------------------------------
# Neue Hilfsfunktion: calculate_grammar_score
# Description:
#   Berechnet den Grammar Score unter Einbeziehung der Textlänge. Dabei wird
#   die Fehlerdichte (Fehleranzahl pro Wort) bestimmt und mittels der Formel:
#
#       Grammar Score = 100 * (1 - (error_density / (error_density + smoothing)))
#
#   berechnet. Der Glättungsfaktor (hier 0,03) sorgt dafür, dass bei wenigen Fehlern
#   oder langen Texten der Score nicht zu stark sinkt.
# -------------------------------------------------------------------------
def calculate_grammar_score(text: str, matches) -> float:
    error_count = len(matches)
    words = len(text.split())
    if words == 0:
        return 100.0
    error_density = error_count / words
    smoothing = 0.03  # Glättungsfaktor (anpassbar)
    score = 100 * (1 - (error_density / (error_density + smoothing)))
    return round(score, 2)


# -------------------------------------------------------------------------
# Neue Hilfsfunktion: calculate_easy_score
# Description:
#   Berechnet den Easy Score unter Einbeziehung der Textlänge. Analog zum Grammar Score
#   wird hier die Fehlerdichte basierend auf den Fehlern aus dem "Leichte Sprache"
#   LanguageTool bestimmt.
#
#       Easy Score = 100 * (1 - (error_density / (error_density + smoothing)))
#
#   Dabei wird ein Glättungsfaktor (hier 0,03) verwendet.
# -------------------------------------------------------------------------
def calculate_easy_score(text: str, matches) -> float:
    error_count = len(matches)
    words = len(text.split())
    if words == 0:
        return 100.0
    error_density = error_count / words
    smoothing = 0.03  # Glättungsfaktor (anpassbar)
    score = 100 * (1 - (error_density / (error_density + smoothing)))
    return round(score, 2)


# -------------------------------------------------------------------------
# Neue Evaluations-Funktion: evaluate_text
# Description:
#   Führt die bisherigen Metriken sowie beide LanguageTool-Prüfungen (Grammatik
#   und Leichte Sprache) durch, kombiniert die Ergebnisse zu einer einzigen
#   Fehlerübersicht und berechnet sowohl den Grammar Score als auch den Easy Score.
# -------------------------------------------------------------------------
def evaluate_text(simplified, source):
    metrics = evaluate_sari(simplified, source)

    # Prüfe mit de-DE (Grammatik) und de-DE-x-simple-language (Leichte Sprache)
    matches_grammar = tool_grammar.check(simplified)
    matches_simple = tool_simple.check(simplified)

    # Kombiniere die beiden Ergebnislisten zu einer einheitlichen Fehlerübersicht
    combined_matches = matches_grammar + matches_simple
    error_html = highlight_language_errors(simplified, combined_matches)

    # Berechne den Grammar Score (auf Basis der de-DE-Prüfung)
    grammar_score = calculate_grammar_score(simplified, matches_grammar)
    metrics.append(["Grammar Score", grammar_score])

    # Berechne den Easy Score (auf Basis der de-DE-x-simple-language-Prüfung)
    easy_score = calculate_easy_score(simplified, matches_simple)
    metrics.append(["Easy Score", easy_score])

    return metrics, error_html


# -------------------------------------------------------------------------
# Utility: count_text_stats
# -------------------------------------------------------------------------
def count_text_stats(text):
    chars = len(text)
    words = len(text.split())
    return f"**Characters** {chars}/{max_chars} **Words** {words}"


# -------------------------------------------------------------------------
# Main Gradio Interface
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
                    simplify_button = gr.Button("Vereinfachen", elem_id="simplify_button")
                    stop_button = gr.Button("Stop", elem_id="stop_button")

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
            # Eine kombinierte Fehlerübersicht
            evaluation_html = gr.HTML(elem_id="evaluation_html")

        # Wire UI components to functions
        simplify_button.click(fn=simplify_text, inputs=[input_box], outputs=[output_box])
        stop_button.click(fn=stop_generation, inputs=[], outputs=[output_box])
        input_box.change(fn=count_text_stats, inputs=[input_box], outputs=[stats])
        evaluate_button.click(
            fn=evaluate_text,
            inputs=[output_box, input_box],
            outputs=[evaluation_output, evaluation_html],
        )

    demo.launch(server_name="0.0.0.0")


# -------------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    interface()
