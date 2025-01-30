################################################################################
# ErLeSen synthetic data generation
################################################################################

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

# Load environment variables (like the OpenAI API key) from a .env file
load_dotenv()

# Set up logging so we can get meaningful console output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


################################################################################
# Function: make_request
# Description:
#   This function composes a chat request to the OpenAI API using a system
#   message (the 'system' argument) and a user message (the 'user' argument).
#   It then makes a call to the OpenAI ChatCompletion endpoint with a chosen
#   model and returns the response text.
################################################################################
def make_request(system, user, model):
    api_key = os.environ.get("OPENAI_API_KEY")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    client = OpenAI(
        # If OPENAI_API_KEY is set in your environment, you can omit api_key=...
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    return response.choices[0].message.content


################################################################################
# Exception: LLMParsingError
# Description:
#   A custom exception signaling that the AI's response could not be parsed
#   because certain sections or formatting markers were missing.
################################################################################
class LLMParsingError(Exception):
    pass


################################################################################
# Function: parse_llm_response
# Description:
#   This function locates and extracts specific sections in the AI's output
#   (using markdown headers and backticks):
#     - The plan of the new text.
#     - A longer text in easy language.
#     - A shorter text in easy language.
#
#   It also checks if the long text is at least twice the length of the short one.
################################################################################
def parse_llm_response(response: str):
    # Extract the plan
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

    # Extract the long text
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

    # Extract the short text
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

    # Clean up whitespace
    plan = plan.strip()
    long_text = long_text.strip()
    short_text = short_text.strip()

    # Make sure the long text is at least twice the length of the short text
    if len(long_text) < 2 * len(short_text):
        raise LLMParsingError("Long text is not at least twice as long as short text")

    return plan, long_text, short_text


################################################################################
# Function: parse_long_reconstruction
# Description:
#   Extracts the revised ("complex") text from the '# 6. Sprachlicher Feinschliff'
#   section. It locates the text between triple backticks and strips any
#   prefixes like "markdown" or "text".
################################################################################
def parse_long_reconstruction(response: str):
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


################################################################################
# Function: parse_short_reconstruction
# Description:
#   Extracts the shorter revised ("complex") text from the first triple-backtick
#   block in the response. Also handles removal of "markdown" or "text" prefixes.
################################################################################
def parse_short_reconstruction(response: str):
    text_start = response.find('```')
    if text_start == -1:
        raise LLMParsingError("Text block for Text not found")
    text_start += len('```')
    text_end = response.find('```', text_start)
    if text_end == -1:
        raise LLMParsingError("End of Text block for Text not found")
    text = response[text_start:text_end].strip()

    if text.startswith("markdown"):
        text = text[len("markdown"):]
        text = text.strip()
    if text.startswith("text"):
        text = text[len("text"):]
        text = text.strip()

    return text


################################################################################
# Main Program
# Description:
#   1. Creates texts in "Leichte Sprache" by referencing existing easy text
#      samples from Hurraki.
#   2. Reconstructs more complex versions of those texts.
#   3. Splits the data into training and test sets, then saves them to JSONL.
################################################################################
if __name__ == "__main__":
    logging.info("Starting the script...")

    ################################################################################
    # START Definition of relevant parameter (edit here!)
    ################################################################################

    openai_model = "gpt-4o"

    # Load the reconstruction prompt for generating easy texts
    reconstr_Prompt_path = Path(prompts.__file__).parent / "reconstruction_prompt.txt"
    reconstr_prompt = reconstr_Prompt_path.read_text()

    # Load topics from a JSON file
    topic_path = Path(data.__file__).parent / "topics.json"
    topics = json.load(open(topic_path))

    # Load Hurraki samples (easy German data) from a JSONL file
    hurraki_samples_path = Path(data.__file__).parent / "hurraki_crawl.jsonl"
    hurraki_samples = list(srsly.read_jsonl(hurraki_samples_path))

    # Load linguistic challenges to incorporate in the reconstructed texts
    linguistic_challenges_path = Path(prompts.__file__).parent / "linguistic_challanges.json"
    linguistic_challenges = json.load(open(linguistic_challenges_path))

    # Define some categories and lengths
    categories = ["Zeitungsartikel", "Blogbeitrag", "Fachtext"]
    length = ["kurz", "mittellang", "lang"]

    # Paths to output our final training and testing data
    train_out_path = Path(data.__file__).parent / "train_v1.jsonl"
    test_out_path = Path(data.__file__).parent / "test_v1.jsonl"

    n_samples = 1500
    n_test = 150

    ################################################################################
    # END Definition of relevant parameter (do NOT edit following code!)
    ################################################################################

    # Prepare examples for prompts by sampling from Hurraki data
    samples_for_prompt = []
    logging.info("Creating examples for Leichte Sprache from Hurraki samples...")
    for idx in range(0, n_samples):
        batch_elements = random.sample(hurraki_samples, 5)
        batch_text_areas = [
            f"## Example {i + 1}\n```\n{be['text']}\n```\n\n"
            for (i, be) in enumerate(batch_elements)
        ]
        samples_for_prompt.append("".join(batch_text_areas))

    # Generate easy-language texts
    synth_easy_data = []
    logging.info("Generating Leichte Sprache texts...")
    for index in tqdm(range(0, n_samples)):

        joined_batch_samples = random.choice(samples_for_prompt)
        category = random.choice(categories)
        topic = random.choice(topics)
        length_choice = random.choice(length)

        try:
            system = reconstr_prompt.format(
                easy_read_samples=joined_batch_samples,
                category=category,
                topic=topic,
                length=length_choice
            )
            user = f"A realistic text of type {category} in Leichte Sprache about the topic {topic['topic']}."
            reconstructed_article = make_request(system, user, openai_model)
            plan, long_text, short_text = parse_llm_response(reconstructed_article)

            synth_easy_data.append({
                "index": topic["index"],
                "plan": plan,
                "easy_long": long_text,
                "easy_short": short_text,
                "cat": category
            })
        except LLMParsingError as e:
            print(e)

    # Reconstruct complex texts from the easy texts
    train_samples = []
    original_prompt = (Path(prompts.__file__).parent / "original_reconstruction_prompt.txt").read_text()
    short_prompt = (Path(prompts.__file__).parent / "short_reconstruction_prompt.txt").read_text()
    logging.info("Reconstructing complex texts...")

    for synth_elem in tqdm(synth_easy_data):
        # Ensure uniqueness by checking if we already processed this index
        if synth_elem["index"] in {t["index"] for t in train_samples}:
            continue

        try:
            ling_challenge = random.choice([v for (k, v) in linguistic_challenges.items()])

            # Use the original prompt for a full-length reconstruction
            prompt = original_prompt.format(ling_challenge=ling_challenge, category=synth_elem["cat"])
            reconstructed_article = make_request(prompt, synth_elem["easy_long"], openai_model)
            reconstructed_article = parse_long_reconstruction(reconstructed_article)

            # Use the short prompt for a shorter reconstruction
            prompt = short_prompt.format(ling_challenge=ling_challenge)
            reconstructed_short = make_request(prompt, synth_elem["easy_short"], openai_model)
            reconstructed_short = parse_short_reconstruction(reconstructed_short)

            # Combine the easy text with the complex reconstructions
            train_samples.append(
                synth_elem | {
                    "complex_short": reconstructed_short,
                    "complex_long": reconstructed_article
                }
            )
        except LLMParsingError as e:
            print(e)
            continue

    logging.info("Reconstruction finished. Saving data...")

    # Example of a simple split: 1 random sample into test, the rest into train
    test = list(random.sample(train_samples, n_test))
    train = [t for t in train_samples if t not in test]

    print(len(test), len(train), len(test) + len(train), len(train_samples))

    # Write to JSONL files
    srsly.write_jsonl(train_out_path, train)
    srsly.write_jsonl(test_out_path, test)
