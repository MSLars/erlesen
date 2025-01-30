import json
import logging
import os
import random
from pathlib import Path

import srsly
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from erlesen import prompts, data

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def make_request(system, user, model):
    api_key = os.environ.get("OPENAI_API_KEY")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    client = OpenAI(
        # This is the default and can be omitted
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    return response.choices[0].message.content


class LLMParsingError(Exception):
    pass

def parse_llm_response(response: str):
    # Plan extrahieren
    plan_start = response.find('# 3. Plan des neuen Textes')
    if plan_start == -1:
        raise LLMParsingError("Plan section not found")
    plan_start = response.find('```', plan_start)
    if plan_start == -1:
        raise LLMParsingError("Markdown block for Plan not found")
    plan_start += len('```')
    plan_end = response.find('```', plan_start)
    if plan_end == -1:
        raise LLMParsingError("End of markdown block for Plan not found")
    plan = response[plan_start:plan_end].strip("```").strip("markdown").strip("\n").strip()

    # Langen Text extrahieren
    long_text_start = response.find('# 4. Neuer Zeitungsartikel in Leichter Sprache')
    if long_text_start == -1:
        raise LLMParsingError("Long text section not found")
    long_text_start = response.find('```text', long_text_start)
    if long_text_start == -1:
        raise LLMParsingError("Text block for Long text not found")
    long_text_start += len('```')
    long_text_end = response.find('```', long_text_start)
    if long_text_end == -1:
        raise LLMParsingError("End of text block for Long text not found")
    long_text = response[long_text_start:long_text_end].strip("```").strip("text").strip("\n").strip()

    # Kurzen Text extrahieren
    short_text_start = response.find('# 5. Kurzer Text in Leichter Sprache')
    if short_text_start == -1:
        raise LLMParsingError("Short text section not found")
    short_text_start = response.find('```', short_text_start)
    if short_text_start == -1:
        raise LLMParsingError("Text block for Short text not found")
    short_text_start += len('```text')
    short_text_end = response.find('```', short_text_start)
    if short_text_end == -1:
        raise LLMParsingError("End of text block for Short text not found")
    short_text = response[short_text_start:short_text_end].strip("```").strip("text").strip("\n").strip()

    # Entfernen von Artefakten wie leading/trailing Whitespace
    plan = plan.strip()
    long_text = long_text.strip()
    short_text = short_text.strip()

    # Validierung der Länge von Long und Short Text
    if len(long_text) < 2 * len(short_text):
        raise LLMParsingError("Long text is not at least twice as long as short text")

    return plan, long_text, short_text


def parse_long_reconstruction(response: str):
    # Plan extrahieren
    text_start = response.find('# 6. Sprachlicher Feinschliff')
    if text_start == -1:
        raise LLMParsingError("Text section not found")
    text_start = response.find('```', text_start)
    if text_start == -1:
        raise LLMParsingError("Markdown block for Text not found")
    text_start += len('```')
    text_end = response.find('```', text_start)
    if text_end == -1:
        raise LLMParsingError("End of markdown block for Text not found")
    text = response[text_start:text_end].strip()
    if text.startswith("markdown"):
        text = text[len("markdown"):]
        text = text.strip()

    if text.startswith("text"):
        text = text[len("text"):]
        text = text.strip()

    return text


def parse_short_reconstruction(response: str):
    # Plan extrahieren
    text_start = response.find('```')
    if text_start == -1:
        raise LLMParsingError("Text block for Text not found")
    text_start += len('```')
    text_end = response.find('```', text_start)
    if text_end == -1:
        raise LLMParsingError("End of Text block for Text not found")
    text = response[text_start:text_end].strip()

    text = response[text_start:text_end].strip()
    if text.startswith("markdown"):
        text = text[len("markdown"):]
        text = text.strip()

    if text.startswith("text"):
        text = text[len("text"):]
        text = text.strip()

    return text



if __name__ == "__main__":
    logging.info("Start des Skripts...")
    openai_model = "gpt-4o"

    reconstr_Prompt_path = Path(prompts.__file__).parent / "reconstruction_prompt.txt"
    reconstr_prompt = reconstr_Prompt_path.read_text()

    topic_path = Path(data.__file__).parent / "topics.json"
    topics = json.load(open(topic_path))

    hurraki_samples_path = Path(data.__file__).parent / "hurraki_crawl.jsonl"
    hurraki_samples = list(srsly.read_jsonl(hurraki_samples_path))

    linguistic_challenges_path = Path(prompts.__file__).parent / "linguistic_challanges.json"
    linguistic_challenges = json.load(open(linguistic_challenges_path))

    categories = ["Zeitungsartikel", "Blogbeitrag", "Fachtext"]
    length = ["kurz", "mittellang", "lang"]

    train_out_path = Path(data.__file__).parent / "train_v1.jsonl"
    test_out_path = Path(data.__file__).parent / "test_v1.jsonl"

    n_samples = 1500

    samples_for_prompt = []
    logging.info("Erstelle Beispiele für LS aus Hurraki Samples")
    for idx in range(0, n_samples):
        batch_elements = random.sample(hurraki_samples, 5)
        batch_text_areas = [f"## Beispiel {i + 1}\n```\n{be['text']}\n```\n\n" for (i, be) in enumerate(batch_elements)]
        samples_for_prompt.append("".join(batch_text_areas))

    synth_easy_data = []
    logging.info("Erstelle Leichte Sprache Texte")
    for index in tqdm(range(0, n_samples)):

        joined_batch_samples = random.choice(samples_for_prompt)
        category = random.choice(categories)
        topic = random.choice(topics)
        length = random.choice(length)

        try:
            system = reconstr_prompt.format(easy_read_samples=joined_batch_samples, category=category, topic=topic, length=length)
            user = f"Einen realistischen Text der Kategorie {category} in Leichter Sprache zum Thema {topic['topic']}."
            reconstructed_article = make_request(system, user, openai_model)
            plan, long, short = parse_llm_response(reconstructed_article)
            synth_easy_data.append({"index": topic["index"], "plan": plan, "easy_long": long, "easy_short": short, "cat": category})
        except LLMParsingError as e:
            print(e)

    train_samples = []

    original_prompt = (Path(prompts.__file__).parent / "original_reconstruction_prompt.txt").read_text()
    short_prompt = (Path(prompts.__file__).parent / "short_reconstruction_prompt.txt").read_text()
    logging.info("Rekonstruiere komplexe Texte")
    for synth_elem in tqdm(synth_easy_data):

        if synth_elem["index"] in {t["index"] for t in train_samples}:
            continue
        try:

            ling_challenge = random.choice([v for (k,v) in linguistic_challenges.items()])

            prompt = original_prompt.format(ling_challenge=ling_challenge, category=synth_elem["cat"])
            reconstructed_article = make_request(prompt, synth_elem["easy_long"], openai_model)
            reconstructed_article = parse_long_reconstruction(reconstructed_article)

            prompt = short_prompt.format(ling_challenge=ling_challenge)
            reconstructed_short = make_request(prompt, synth_elem["easy_short"], openai_model)
            reconstructed_short = parse_short_reconstruction(reconstructed_short)

            train_samples.append(
                synth_elem | {"complex_short": reconstructed_short, "complex_long": reconstructed_article})
        except LLMParsingError as e:
            print(e)
            continue

    logging.info("reconstruction finished, save data")
    test = list(random.sample(train_samples, 1))
    train = [t for t in train_samples if t not in test]

    print(len(test), len(train), len(test) + len(train), len(train_samples))

    srsly.write_jsonl(train_out_path, train)
    srsly.write_jsonl(test_out_path, test)
