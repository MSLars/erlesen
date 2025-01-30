import gradio as gr
import textstat
import torch
from huggingface_hub import InferenceClient
import threading
from evaluate import load
from textstat.textstat import textstatistics
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Konfiguration
base_url = "http://127.0.0.1:8080"  # URL des TGI-Servers
client = InferenceClient(base_url=base_url)

# Flag zum Stoppen der Generierung
stop_event = threading.Event()

# SARI-Loader
sari = load("sari")

model_name="agentlans/mdeberta-v3-base-readability"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("cpu")
model = model.to(device)


# Simplify-Funktion
def simplify_text(complex_text):
    global stop_event
    response = ""
    stop_event.clear()  # Reset Stop-Event

    output = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Vereinfache Texte."},
            {"role": "user", "content": complex_text},
        ],
        stream=True,
        max_tokens=4096,
    )
    for chunk in output:
        if stop_event.is_set():
            break  # Abbrechen der Generierung
        response += chunk.choices[0].delta.content
        yield response


# Funktion zum Stoppen der Generierung
def stop_generation():
    global stop_event
    stop_event.set()  # Setze das Stop-Event
    return "Generierung gestoppt."


# Evaluation: Berechne nur SARI
def evaluate_sari(reference, simplified, source):
    metrics = []
    if reference:
        sari_score = sari.compute(
            sources=[source],
            predictions=[simplified],
            references=[[reference]],
        )
        metrics.append(["SARI", round(sari_score["sari"], 4)])

    stst = textstatistics()

    stst.set_lang("de")

    metrics.append(["flesch_reading_ease [0 (hard), ..., 100 (easy)]", stst.flesch_reading_ease(simplified)])
    metrics.append(["lix ... 40 (literature for children) ... 60 (specialist literature)",stst.lix(simplified)])

    inputs = tokenizer(simplified, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze().cpu()
    metrics.append(["Readability", round(logits.item(), 4)])

    # Return a list of lists for Dataframe compatibility
    return metrics


max_chars = 2048

# Funktion für Zeichenzähler
def count_text_stats(text):
    chars = len(text)
    words = len(text.split())
    return f"**Characters** {chars}/{max_chars} **Words** {words}"


# Gradio-Interface
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
                    max_length=2048,
                )

                with gr.Row(elem_id="input_div"):
                    stats = gr.Markdown(value=f"**Characters** -/{max_chars} **Words** -", elem_id="stats")

                    simplify_button = gr.Button(
                        "Vereinfachen", elem_id="simplify_button"
                    )
                    stop_button = gr.Button(
                        "Stop", elem_id="stop_button"
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
                    max_length=2048,
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
                max_length=2048,
            )

        with gr.Accordion("Metriken", open=False):
            evaluate_button = gr.Button("Evaluieren")
            evaluation_output = gr.Dataframe(
                headers=["Metrik", "Wert"],
                elem_id="evaluation_output",
                interactive=False,
            )

        simplify_button.click(
            fn=simplify_text, inputs=[input_box], outputs=[output_box]
        )
        stop_button.click(
            fn=stop_generation, inputs=[], outputs=[output_box]
        )
        input_box.change(fn=count_text_stats, inputs=[input_box], outputs=[stats])
        evaluate_button.click(
            fn=evaluate_sari,
            inputs=[ref_input_box, output_box, input_box],
            outputs=[evaluation_output],
        )

    demo.launch()


if __name__ == "__main__":
    interface()
