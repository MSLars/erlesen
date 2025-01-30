################################################################################
#   ____             _   _
#  / ___| _   _ _ __| |_| |__
#  \___ \| | | | '__| __| '_ \
#   ___) | |_| | |  | |_| | | |
#  |____/ \__,_|_|   \__|_| |_|
#
# ErLeSen synthetic data generation
################################################################################

import gradio as gr
import textstat
import torch
from huggingface_hub import InferenceClient
import threading
from evaluate import load
from textstat.textstat import textstatistics
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configuration
base_url = "http://127.0.0.1:8080"  # TGI server URL
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

device = torch.device("cpu")
preference_model = preference_model.to(device)

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
            {"role": "system", "content": "Vereinfache Texte."},
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
def evaluate_sari(reference, simplified, source):
    metrics = []

    # Only compute SARI if a reference is provided
    if reference:
        sari_score = sari.compute(
            sources=[source],
            predictions=[simplified],
            references=[[reference]],
        )
        metrics.append(["SARI", round(sari_score["sari"], 4)])

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
    metrics.append(["Readability", round(logits.item(), 4)])

    return metrics


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

        with gr.Accordion("Evaluation anhand eines Referenztexts", open=False):
            ref_input_box = gr.Textbox(
                label="Referenztext",
                info="Geben Sie hier den Text an, der bei der Bewertung als Referenz dienen soll.",
                placeholder="Geben Sie hier den Referenztext ein...",
                lines=10,
                max_lines=10,
                elem_id="ref_input_box",
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
            inputs=[ref_input_box, output_box, input_box],
            outputs=[evaluation_output],
        )

    demo.launch()


# -------------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    interface()
