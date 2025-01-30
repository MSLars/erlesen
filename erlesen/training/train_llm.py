"""Finetunes a large language model with configurable parameters."""
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

from pathlib import Path
from tqdm import tqdm
import numpy as np
import datasets
import sys
from transformers import TrainerCallback
import random

import torch
import subprocess
import platform

from time import time
import logging

from erlesen import prompts

logging.getLogger("transformers.trainer").setLevel(logging.WARNING)

from datetime import datetime

from transformers import logging as transformers_logging

# Disable default transformers handlers
transformers_logging.disable_default_handler()

import os

"""Finetunes a large language model with configurable parameters."""


class LogFilter:
    def __init__(self):
        self.original_stdout = sys.stdout

    def write(self, message):
        # Only allow messages that match the format of your log function
        if message.strip().startswith("[") and " - " in message:
            self.original_stdout.write(message)
            self.original_stdout.write("\n")

    def flush(self):
        self.original_stdout.flush()

def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level} - {msg}", flush=True)

# Redirect sys.stdout to your custom filter
sys.stdout = LogFilter()

def create_dataloaders(
        tokenizer,
        data_path,
        llama_template,
        source_key,
        target_key,
        max_seq_length,
        batch_size,
        n_max_train_samples,
        log_function=print):
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

        for sk, tk in zip(source_key, target_key):

            source_text = example.get(sk, "")
            target_text = "\n\n" + example.get(tk, "") + "<|eot_id|>"

            # formatted_text = llama_template.format(system=f"Vereinfache Texte im Stil {example['domain']}.", user=source_text)
            formatted_text = llama_template.format(system=f"Vereinfache Texte.",
                                                   user=source_text)
            # log(formatted_text, level="INFO")
            # log(target_text, level="INFO")
            input_ids_complex, _ = tokenizer(formatted_text, add_special_tokens=False).values()
            target_ids, _ = tokenizer(target_text, add_special_tokens=False).values()
            # log(formatted_text + target_text, level="INFO")
            token_ids, attention_mask = tokenizer(formatted_text + target_text, add_special_tokens=False).values()

            if len(token_ids) > max_seq_length:
                continue

            complex_count = len(input_ids_complex)

            # check format of token_ids
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

            token_ids = token_ids + [tokenizer.pad_token_id] * (seq_length - len(token_ids))
            attention_mask = attention_mask + [0] * (seq_length - len(attention_mask))
            logit_mask = list(logit_mask) + [0] * (seq_length - len(logit_mask))
            labels = list(labels) + [-100] * (seq_length - len(labels))

            assert len(token_ids) == len(attention_mask) == len(logit_mask) == len(labels)

            input_ids.append(token_ids)
            attention_masks.append(attention_mask)
            logit_masks.append(logit_mask)
            output_labels.append(labels)

            if n_max_train_samples > 0:
                if len(input_ids) >= n_max_train_samples:
                    break

    log_function("Converting data to Dataset object...")
    log(f"len(input_ids) = {len(input_ids)}")
    log(f"len(output_labels) = {len(output_labels)}")

    if len(input_ids) > 0:
        log(f"len(input_ids[0]) = {len(input_ids[0])}")
        log(f"len(output_labels[0]) = {len(output_labels[0])}")
    else:
        raise ValueError(f"The length of the dataset should not be 0.")

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

        # Find idx_first_change and idx_last_change
        idx_first_change = next((i for i, label in enumerate(labels) if label >= 0), -1)
        idx_last_change = next(
            (i for i, label in enumerate(labels[idx_first_change:], idx_first_change) if label < 0), len(labels)-1)

        log((idx_first_change, idx_last_change, len(labels), len(input_ids)))

        idx_list = [0, 1, 2, "..."] + [idx_first_change -8,idx_first_change -7,idx_first_change -6,idx_first_change -5,idx_first_change -4,idx_first_change -3, idx_first_change -2, idx_first_change -1, idx_first_change, idx_first_change+1, idx_first_change+2, idx_first_change+3, "...", idx_last_change-3, idx_last_change-2, idx_last_change-1]

        for i in idx_list:
            if i == "...":
                continue
            try:
                input_ids[i]
            except:
                log(f"{i} in not in {str(input_ids)}")

        if idx_last_change < len(labels):
            idx_list += [idx_last_change]
        if idx_last_change < len(labels)-1:
            idx_list += [idx_last_change +1]
        if idx_last_change < len(labels) - 2:
            idx_list += [idx_last_change + 2]

        inp_list = [dec(input_ids[x], tokenizer) if x != "..." else "..." for x in idx_list]
        lab_list = [dec(labels[x], tokenizer) if x != "..." else "..." for x in idx_list]
        lists = [idx_list, inp_list, lab_list]
        log("\n" + format_table(*lists))


class TabularLoggingCallback(TrainerCallback):
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
        self.log_function(f"║ Training Started: Epochs = {args.num_train_epochs}, Steps = {state.max_steps} ║",
                          "INFO")
        self.log_function("╚════════════════════════════════════════════╝", "INFO")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.global_step % self.logging_steps != 0:
            return

        # Print header if not already done
        if not self.header_printed:
            self.log_function(
                "+-------+-------+-------+---------------+-----------+-----------+-----------+-----------+", "INFO")
            self.log_function(
                "| Step  | Epoch | Loss  | Learning Rate | Grad Norm | Total Time| Time/Step | Est. Time |", "INFO")
            self.log_function(
                "+-------+-------+-------+---------------+-----------+-----------+-----------+-----------+", "INFO")
            self.header_printed = True

        # Metrics and time calculations
        current_time = time()
        total_time = current_time - self.start_time
        log_interval_time = current_time - self.last_log_time
        self.last_log_time = current_time

        # Time per step based on logging_steps
        time_per_step = log_interval_time / self.logging_steps

        # Progress as a fraction of total epochs
        total_epochs = args.num_train_epochs
        current_epoch = state.epoch if state.epoch is not None else 0.0
        progress = current_epoch / total_epochs if total_epochs > 0 else 0.0

        # Estimated remaining time
        estimated_total_time = total_time / progress if progress > 0 else float("inf")
        estimated_remaining_time = estimated_total_time - total_time

        # Format times and logs
        loss = logs.get("loss", float("nan"))
        learning_rate = logs.get("learning_rate", float("nan"))
        grad_norm = logs.get("grad_norm", float("nan"))

        formatted_lr = f"{float(learning_rate):.4e}" if learning_rate != "N/A" else "N/A"
        formatted_grad_norm = f"{float(grad_norm):.2f}" if grad_norm != "N/A" else "N/A"
        formatted_total_time = f"{total_time:.2f}s"
        formatted_time_per_step = f"{time_per_step:.2f}s"
        formatted_est_remaining = f"{estimated_remaining_time:.2f}s" if estimated_remaining_time != float(
            "inf") else "N/A"

        # Log the row
        self.log_function(
            f"| {state.global_step:<5} | {current_epoch:<5.2f} | {loss:<5.3f} | {formatted_lr:<13} | {formatted_grad_norm:<9} | {formatted_total_time:<9} | {formatted_time_per_step:<9} | {formatted_est_remaining:<9} |",
            "INFO"
        )

    def on_train_end(self, args, state, control, **kwargs):
        self.log_function(
            "+-------+-------+-------+---------------+-----------+-----------+-----------+-----------+", "INFO")
        self.log_function("╔════════════════════════════════════════════╗", "INFO")
        self.log_function(
            f"║ Training Completed: Total Steps = {state.global_step}, Final Epoch = {state.epoch:.2f}  ║", "INFO"
        )
        self.log_function("╚════════════════════════════════════════════╝", "INFO")


def train_llm(
        dataset_file: str,
        model_output_dir: str,
        model_name_or_path: str,
        max_seq_length: int,
        batch_size: int,
        learning_rate: float,
        num_epochs: int,
        gradient_accumulation_steps: int,
        logging_steps: int,
        source_key: str,
        target_key: str,
        weight_decay:float,
        gradient_checkpointing: bool,
        n_max_train_samples: int,
        template_path: str
):

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

    # llama_template = (Path(prompts.__file__).parent / "llama_template.txt").read_text()
    llama_template = template_path
    log("╔════ GPU-Check Start ════")

    # System Informationen sammeln
    system_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'processor': platform.processor() or "Nicht verfügbar"
    }

    # Basis GPU-Informationen sammeln
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
        log("nvidia-smi nicht verfügbar")


    log("╚════ GPU-Check Ende ════")

    log("╔════ Model Training Start ════")

    log("║ Start loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        token=True,
    )
    log("║ Tokenizer loaded, start loading model")

    # Load dataset
    training_file_path = Path(dataset_file)
    log(f"║ load training data from file {str(training_file_path)}, training_file_path.exists()={training_file_path.exists()}")
    train_dataset, eval_dataset = create_dataloaders(tokenizer,
                                                     training_file_path,
                                                     llama_template,
                                                     source_key,
                                                     target_key,
                                                     max_seq_length,
                                                     batch_size,
                                                     n_max_train_samples,)

    log(f"║ loaded dataset with {len(train_dataset)} entries")
    log_random_samples(train_dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        use_cpu=False,
        accelerator_config={'split_batches':False},
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
        warmup_steps=max(int(len(train_dataset)/20)+1, 5),
        optim="adamw_bnb_8bit",
        lr_scheduler_type="cosine",
        log_level="error",  # Suppress internal Trainer logs
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map = "auto",
        torch_dtype=torch.bfloat16,  # Use appropriate dtype for your setup
        token=True,
    )
    log(model.hf_device_map)
    model.config.use_cache = False  # Ensure use_cache is disabled
    model.gradient_checkpointing_enable()  # Enable gradient checkpointing

    log("║ Model loaded")
    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi']).decode('utf-8')
        log(nvidia_smi)
    except:
        log("nvidia-smi nicht verfügbar")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[TabularLoggingCallback(logging_steps=logging_steps, log_function=log)]  # Pass your log function
    )

    trainer.train()

    # Save final model
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
