################################################################################
# ErLeSen synthetic data generation
################################################################################

"""
Finetunes a large language model with configurable parameters.
"""

import sys
import random
import logging
import platform
import subprocess
from time import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import datasets
from dotenv import load_dotenv
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    logging as transformers_logging
)

from erlesen import prompts, data

load_dotenv()

# ------------------------------------------------------------------------------------------
# Configure logging
# ------------------------------------------------------------------------------------------

logging.getLogger("transformers.trainer").setLevel(logging.WARNING)
transformers_logging.disable_default_handler()  # Disable default Transformers logs


# Custom logging to control console output
class LogFilter:
    def __init__(self):
        self.original_stdout = sys.stdout

    def write(self, message):
        # Only allow messages that match the format "[YYYY-MM-DD HH:MM:SS] LEVEL - message"
        if message.strip().startswith("[") and " - " in message:
            self.original_stdout.write(message)
            self.original_stdout.write("\n")

    def flush(self):
        self.original_stdout.flush()


def log(msg, level="INFO"):
    """A simple log function that prints a timestamp, log level, and message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level} - {msg}", flush=True)


# Redirect sys.stdout to our custom LogFilter
sys.stdout = LogFilter()


# ------------------------------------------------------------------------------------------
# Data Loading & Processing
# ------------------------------------------------------------------------------------------

def create_dataloaders(
        tokenizer,
        data_path,
        llama_template,
        source_key,
        target_key,
        max_seq_length,
        batch_size,
        n_max_train_samples,
        system_message,
        log_function=print):
    """
    Loads data from a JSONL file (or JSON), tokenizes it using a template, and
    returns a HuggingFace Dataset object suitable for training.

    :param tokenizer: Tokenizer object (AutoTokenizer)
    :param data_path: Path to the dataset file
    :param llama_template: A template string used to generate prompt text
    :param source_key: List of keys to retrieve from each JSON example (source text)
    :param target_key: List of keys to retrieve from each JSON example (target text)
    :param max_seq_length: Maximum sequence length for the model
    :param batch_size: How many examples per batch
    :param n_max_train_samples: If > 0, limit the number of examples used
    :param log_function: A logging function (by default, `print`)
    :return: A HuggingFace Dataset
    """
    import srsly

    log_function("Loading raw data from JSONL file...")
    try:
        raw_data = list(srsly.read_jsonl(data_path))
    except:
        raw_data = list(srsly.read_json(data_path))

    input_ids = []
    attention_masks = []
    logit_masks = []
    output_labels = []

    for example in tqdm(raw_data, desc="Tokenizing examples"):
        # We may have multiple source/target pairs
        for sk, tk in zip(source_key, target_key):
            source_text = example.get(sk, "")
            target_text = "\n\n" + example.get(tk, "") + "<|eot_id|><|end_of_text|>"

            # Format the conversation
            formatted_text = llama_template.format(
                system=system_message,
                user=source_text
            )

            input_ids_complex, _ = tokenizer(formatted_text, add_special_tokens=False).values()
            target_ids, _ = tokenizer(target_text, add_special_tokens=False).values()
            token_ids, attention_mask = tokenizer(
                formatted_text + target_text,
                add_special_tokens=False
            ).values()

            # Skip if sequence is too long
            if len(token_ids) > max_seq_length:
                continue

            complex_count = len(input_ids_complex)

            # Sanity checks
            assert token_ids[:complex_count] == input_ids_complex
            assert token_ids[complex_count:complex_count + len(target_ids)] == target_ids
            assert token_ids[complex_count + len(target_ids) - 1] == tokenizer.eos_token_id

            logit_mask = np.array(attention_mask)
            logit_mask[:complex_count] = 0

            labels = np.array(token_ids)
            labels[:complex_count] = -100
            labels[~np.array(attention_mask, dtype=bool)] = -100
            assert [l for l in labels.tolist() if l >= 0] == target_ids

            if batch_size == 1:
                seq_length = len(token_ids)
            else:
                seq_length = max_seq_length

            # Pad sequences to seq_length
            token_ids = token_ids + [tokenizer.pad_token_id] * (seq_length - len(token_ids))
            attention_mask = attention_mask + [0] * (seq_length - len(attention_mask))
            logit_mask = list(logit_mask) + [0] * (seq_length - len(logit_mask))
            labels = list(labels) + [-100] * (seq_length - len(labels))

            # Final length check
            assert len(token_ids) == len(attention_mask) == len(logit_mask) == len(labels)

            input_ids.append(token_ids)
            attention_masks.append(attention_mask)
            logit_masks.append(logit_mask)
            output_labels.append(labels)

            if n_max_train_samples > 0 and len(input_ids) >= n_max_train_samples:
                break

    log_function("Converting data to Dataset object...")
    log(f"len(input_ids) = {len(input_ids)}")
    log(f"len(output_labels) = {len(output_labels)}")

    if len(input_ids) > 0:
        log(f"len(input_ids[0]) = {len(input_ids[0])}")
        log(f"len(output_labels[0]) = {len(output_labels[0])}")
    else:
        raise ValueError("The length of the dataset is 0, which is invalid.")

    dataset = datasets.Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": output_labels
    })

    if len(input_ids) > 0:
        log(f"len(dataset['input_ids'][0]) = {len(dataset['input_ids'][0])}")
        log(f"len(dataset['labels'][0]) = {len(dataset['labels'][0])}")

    return dataset


def dec(indx, tokenizer):
    """
    Custom decoding function to handle invalid or out-of-range token IDs.
    """
    if indx == "...":
        return "..."
    try:
        return tokenizer.decode([indx]).replace("\n", "<NL>")
    except Exception:
        return "[INV]"


def format_table(*lists):
    """
    Takes multiple lists (columns) and arranges them in a neat text-based table.
    """
    # Transpose the list of lists to group columns
    columns = list(zip(*lists))

    # Determine column widths based on the maximum element length in each column
    column_widths = [max(len(str(item)) for item in column) for column in columns]

    # Create the horizontal border based on column widths
    horizontal_border = "+" + "+".join("-" * (width + 2) for width in column_widths) + "+"

    # Create a row formatter string
    row_format = "| " + " | ".join(f"{{:<{width}}}" for width in column_widths) + " |"

    # Format all rows
    rows = [row_format.format(*[str(item) for item in row]) for row in lists]

    # Assemble the final table
    table = f"{horizontal_border}\n" + f"\n{horizontal_border}\n".join(rows) + f"\n{horizontal_border}"

    return table


def log_random_samples(dataset, tokenizer, num_samples=3):
    """
    Logs details of `num_samples` random elements from the dataset in a tabular format.
    """
    random_indices = random.sample(range(len(dataset)), num_samples)

    for index in random_indices:
        elem = dataset[index]
        input_ids = elem["input_ids"]
        labels = elem["labels"]

        # Find index for first label >= 0 (start of target) and last label < 0
        idx_first_change = next((i for i, label in enumerate(labels) if label >= 0), -1)
        idx_last_change = next(
            (i for i, label in enumerate(labels[idx_first_change:], idx_first_change) if label < 0),
            len(labels) - 1
        )

        log((idx_first_change, idx_last_change, len(labels), len(input_ids)))

        idx_list = [0, 1, 2, "..."] + [
            idx_first_change - 8,
            idx_first_change - 7,
            idx_first_change - 6,
            idx_first_change - 5,
            idx_first_change - 4,
            idx_first_change - 3,
            idx_first_change - 2,
            idx_first_change - 1,
            idx_first_change,
            idx_first_change + 1,
            idx_first_change + 2,
            idx_first_change + 3,
            "...",
            idx_last_change - 3,
            idx_last_change - 2,
            idx_last_change - 1
        ]

        for i in idx_list:
            if i == "...":
                continue
            try:
                input_ids[i]
            except:
                log(f"{i} not in range of input_ids")

        if idx_last_change < len(labels):
            idx_list += [idx_last_change]
        if idx_last_change < len(labels) - 1:
            idx_list += [idx_last_change + 1]
        if idx_last_change < len(labels) - 2:
            idx_list += [idx_last_change + 2]

        inp_list = [dec(input_ids[x], tokenizer) if x != "..." else "..." for x in idx_list]
        lab_list = [dec(labels[x], tokenizer) if x != "..." else "..." for x in idx_list]
        lists = [idx_list, inp_list, lab_list]

        log("\n" + format_table(*lists))


class TabularLoggingCallback(TrainerCallback):
    """
    A custom TrainerCallback to log training progress in a neat table format.
    """

    def __init__(self, logging_steps, log_function):
        self.logging_steps = logging_steps
        self.log_function = log_function
        self.header_printed = False
        self.start_time = None
        self.last_log_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time()
        self.last_log_time = self.start_time
        self.log_function("╔════════════════════════════════════════════╗", "INFO")
        self.log_function(
            f"║ Training Started: Epochs = {args.num_train_epochs}, Steps = {state.max_steps} ║",
            "INFO"
        )
        self.log_function("╚════════════════════════════════════════════╝", "INFO")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.global_step % self.logging_steps != 0:
            return

        # Print header once
        if not self.header_printed:
            self.log_function(
                "+-------+-------+-------+---------------+-----------+-----------+-----------+-----------+",
                "INFO")
            self.log_function(
                "| Step  | Epoch | Loss  | Learning Rate | Grad Norm | Total Time| Time/Step | Est. Time |",
                "INFO")
            self.log_function(
                "+-------+-------+-------+---------------+-----------+-----------+-----------+-----------+",
                "INFO")
            self.header_printed = True

        current_time = time()
        total_time = current_time - self.start_time
        log_interval_time = current_time - self.last_log_time
        self.last_log_time = current_time
        time_per_step = log_interval_time / self.logging_steps

        total_epochs = args.num_train_epochs
        current_epoch = state.epoch if state.epoch is not None else 0.0
        progress = current_epoch / total_epochs if total_epochs > 0 else 0.0
        estimated_total_time = total_time / progress if progress > 0 else float("inf")
        estimated_remaining_time = estimated_total_time - total_time

        loss = logs.get("loss", float("nan"))
        learning_rate = logs.get("learning_rate", float("nan"))
        grad_norm = logs.get("grad_norm", float("nan"))

        formatted_lr = f"{float(learning_rate):.4e}" if learning_rate != "N/A" else "N/A"
        formatted_grad_norm = f"{float(grad_norm):.2f}" if grad_norm != "N/A" else "N/A"
        formatted_total_time = f"{total_time:.2f}s"
        formatted_time_per_step = f"{time_per_step:.2f}s"
        formatted_est_remaining = (
            f"{estimated_remaining_time:.2f}s" if estimated_remaining_time != float("inf") else "N/A"
        )

        self.log_function(
            f"| {state.global_step:<5} | {current_epoch:<5.2f} | {loss:<5.3f} | {formatted_lr:<13} | {formatted_grad_norm:<9} | {formatted_total_time:<9} | {formatted_time_per_step:<9} | {formatted_est_remaining:<9} |",
            "INFO"
        )

    def on_train_end(self, args, state, control, **kwargs):
        self.log_function(
            "+-------+-------+-------+---------------+-----------+-----------+-----------+-----------+",
            "INFO"
        )
        self.log_function("╔════════════════════════════════════════════╗", "INFO")
        self.log_function(
            f"║ Training Completed: Total Steps = {state.global_step}, Final Epoch = {state.epoch:.2f}  ║",
            "INFO"
        )
        self.log_function("╚════════════════════════════════════════════╝", "INFO")


# ------------------------------------------------------------------------------------------
# Main Training Function
# ------------------------------------------------------------------------------------------

def train_llm(
        dataset_file: str | Path,
        model_output_dir: str | Path,
        model_name_or_path: str,
        max_seq_length: int,
        batch_size: int,
        learning_rate: float,
        num_epochs: int,
        gradient_accumulation_steps: int,
        logging_steps: int,
        source_key: str,
        target_key: str,
        weight_decay: float,
        gradient_checkpointing: bool,
        n_max_train_samples: int,
        template_path: str,
        system_message,
):
    """
    Trains a Causal Language Model using HuggingFace Transformers.

    :param dataset_file: Path to the dataset in JSON/JSONL
    :param model_output_dir: Directory to save the fine-tuned model
    :param model_name_or_path: Base model name or path from HuggingFace
    :param max_seq_length: Max sequence length for the model
    :param batch_size: Training batch size per device
    :param learning_rate: Learning rate for the optimizer
    :param num_epochs: Number of total training epochs
    :param gradient_accumulation_steps: Steps to accumulate before backward/update
    :param logging_steps: Steps interval for logging
    :param source_key: JSON key(s) for the source text
    :param target_key: JSON key(s) for the target text
    :param weight_decay: Weight decay coefficient for regularization
    :param gradient_checkpointing: Whether to use gradient checkpointing
    :param n_max_train_samples: If >0, limit the total training samples
    :param template_path: The template string or file path for prompt generation
    """

    # Handle multiple source/target keys
    if "," in source_key:
        source_key = [sk.strip() for sk in source_key.split(",")]
    else:
        source_key = [source_key]

    if "," in target_key:
        target_key = [tk.strip() for tk in target_key.split(",")]
    else:
        target_key = [target_key]

    log(f"source_key={str(source_key)}, target_key={str(target_key)}")
    if len(source_key) != len(target_key):
        raise ValueError(f"len(source_key)={len(source_key)} != len(target_key)={len(target_key)}")

    # For demonstration, we directly use template_path as the actual template text
    llama_template = template_path
    log("╔════ GPU-Check Start ════")

    # Basic system info
    system_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'processor': platform.processor() or "Not available"
    }

    cuda_available = torch.cuda.is_available()
    gpu_info = {
        'cuda_available': cuda_available,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda if cuda_available else "N/A",
        'device_count': torch.cuda.device_count() if cuda_available else 0,
        'system_info': system_info
    }

    for k, v in gpu_info.items():
        log(f"║ GPU Info: {k}: {v}")

    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi']).decode('utf-8')
        log(nvidia_smi)
    except:
        log("nvidia-smi not available")

    log("╚════ GPU-Check End ════")
    log("╔════ Model Training Start ════")

    log("║ Start loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        token=True,
    )

    log("║ Tokenizer loaded, start loading model")

    # Load dataset
    training_file_path = Path(dataset_file)
    log(f"║ Loading dataset from {str(training_file_path)}, exists={training_file_path.exists()}")

    train_dataset = create_dataloaders(
        tokenizer,
        training_file_path,
        llama_template,
        source_key,
        target_key,
        max_seq_length,
        batch_size,
        n_max_train_samples,
        system_message
    )
    # For simplicity, we can just reassign train_dataset as eval_dataset here
    eval_dataset = train_dataset

    log(f"║ Dataset loaded with {len(train_dataset)} entries")
    log_random_samples(train_dataset, tokenizer)

    # Prepare training arguments
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        use_cpu=False,
        accelerator_config={'split_batches': False},
        per_device_train_batch_size=batch_size,
        disable_tqdm=False,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_dir=f"{model_output_dir}/logs",
        logging_steps=logging_steps,
        save_strategy="no",
        weight_decay=weight_decay,
        bf16=True,
        push_to_hub=False,
        gradient_checkpointing=gradient_checkpointing,
        warmup_steps=max(int(len(train_dataset) / 20) + 1, 5),
        optim="adamw_bnb_8bit",
        lr_scheduler_type="cosine",
        log_level="error",  # Suppress internal trainer logs
    )

    log("║ loading the model, this may take several minutes")
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=True,
    )
    log(model.hf_device_map)

    # Ensure use_cache is disabled and enable gradient checkpointing
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    log("║ Model loaded")
    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi']).decode('utf-8')
        log(nvidia_smi)
    except:
        log("nvidia-smi not available")

    # Pad token settings
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[TabularLoggingCallback(logging_steps=logging_steps, log_function=log)]
    )

    # Start training
    trainer.train()

    # Save final model artifacts
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)


if __name__ == "__main__":
    log("Start training script")

    ################################################################################
    # START Definition of relevant parameter (edit here!)
    ################################################################################

    dataset_file = Path(data.__file__).parent / "train.jsonl"
    model_output_dir = "models"
    model_name_or_path = "meta-llama/Llama-3.2-1B"
    max_seq_length = 4096
    batch_size = 1
    learning_rate = 5e-5
    num_epochs = 4
    gradient_accumulation_steps = 2
    logging_steps = 20
    source_key = "complex_long,complex_short"
    target_key = "easy_long,easy_short"
    weight_decay = 0.01
    gradient_checkpointing = True
    n_max_train_samples = -1
    template_path = (Path(prompts.__file__).parent / "llama_template.txt").read_text()
    system_message = (Path(prompts.__file__).parent / "system_message.txt").read_text()

    ################################################################################
    # END Definition of relevant parameter (do NOT edit following code!)
    ################################################################################

    train_llm(
        dataset_file=dataset_file,
        model_output_dir=model_output_dir,
        model_name_or_path=model_name_or_path,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        source_key=source_key,
        target_key=target_key,
        weight_decay=weight_decay,
        gradient_checkpointing=gradient_checkpointing,
        n_max_train_samples=n_max_train_samples,
        template_path=template_path,
        system_message=system_message,
    )
