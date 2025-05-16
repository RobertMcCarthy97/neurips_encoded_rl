# %%

# pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
# pip install transformers trl==0.14.0 datasets wandb

assert False, "This script is buggy"

# %%

EXP_NAME = "sft_arth_direct_3B"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATASET_NAME = "arth_direct_answer_instruct"

BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 8
N_EVAL_SAMPLES = 200
LR = 2e-5
WARMUP_STEPS = 20
MAX_STEPS = 100
MAX_EPOCHS = 10
WEIGHT_DECAY = 0.01
# MAX_LENGTH = 256

EVAL_STEPS = 1000
EVAL_ACCUMULATION_STEPS = 4
EVAL_N_SAMPLES = 64

RESPONSE_TEMPLATE = "<|im_start|>assistant\nAnswer:" # ["<|im_start|>assistant", "Answer:"]


# %%


import wandb
import os
import json
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments, DataCollatorForSeq2Seq, AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import torch
from tqdm import tqdm
import numpy as np
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk

# %%

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# %%

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# %%

response_template = RESPONSE_TEMPLATE
collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

# %%

import json
import os
import random
from pathlib import Path
import requests
import argparse
from datasets import Dataset
from tqdm import tqdm
from typing import NamedTuple, Literal, List

# --- Definitions from the sycophancy dataset code ---


# Adjust this path as necessary. Here we assume the parquet file is saved at data/train.parquet.
parquet_file = os.path.join("/workspace/TinyZero/data/" + DATASET_NAME, "train.parquet")
# Load the dataset from the parquet file
ds = Dataset.from_parquet(parquet_file)

dataset = []

for data in ds:
    text = data["prompt"][0]["content"]
    text += str(data["reward_model"]["ground_truth"]) + "<|im_end|>"
    dataset.append({"text": text, "label": str(data["reward_model"]["ground_truth"])})

dataset = Dataset.from_list(dataset)

# Split dataset into train and test sets (80/20 split)
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Tokenize the test dataset
test_dataset = test_dataset.map(lambda x: tokenizer(x['text'], padding=True, truncation=True), batched=True)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

print(f"Train example: {train_dataset[0]}")

# %%

import numpy as np

class ArthTrainer(SFTTrainer):
    def __init__(self, *args, eval_dataset=None, eval_sample_size=EVAL_N_SAMPLES, **kwargs):
        super().__init__(*args, eval_dataset=eval_dataset, **kwargs)
        self.eval_dataset = eval_dataset
        self.eval_sample_size = eval_sample_size
    
    # def get_eval_dataloader(self, eval_dataset=None):
    #     '''
    #     Samples the evaluation dataset and returns a subset 
    #     of size self.eval_sample_size.
    #     '''
    #     if eval_dataset is None:
    #         eval_dataset = self.eval_dataset
    #     idxs = random.sample(range(len(eval_dataset)), self.eval_sample_size)
    #     eval_subset = eval_dataset.select(idxs)
    #     return super().get_eval_dataloader(eval_subset)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # print("in evaluate")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

# %%

def evaluate_and_log_model(model, tokenizer, test_dataset, n_samples, batch_size):
    """
    Evaluate the model on a subset of the test dataset using batch generation.
    
    For each sample:
      - Extract the prompt (everything before "Answer:" in sample["text"])
      - Generate a continuation using the model.
      - Extract the answer from the generated text.
      - Compare it (here via exact match) to the ground truth answer (sample["label"]).
    
    Logs a wandb table with the following columns:
      - Models prompt
      - Models generation
      - GT answer
      - Extracted answer
      - accuracy (1 if the extracted answer matches the GT answer, 0 otherwise)
      
    Returns:
      The overall evaluation accuracy (mean of the per-example accuracy).
    """
    import random
    import torch
    import wandb

    # # Helper function to extract the answer from the generated text.
    # def extract_answer(generated_text):
    #     # If the generated text contains the delimiter "Answer:",
    #     # return the text following it.
    #     if "Answer:" in generated_text:
    #         return generated_text.split("Answer:", 1)[1].strip()
    #     else:
    #         return "INVALID"

    # Get the device from the model.
    device = next(model.parameters()).device

    total_samples = len(test_dataset)
    # Sample indices (if n_samples exceeds total available, use all)
    indices = random.sample(range(total_samples), min(n_samples, total_samples))

    num_correct = 0
    # Create a table with the required columns.
    wandb_table = wandb.Table(columns=[
        "Models prompt", "Models generation", "GT answer", "Extracted answer", "accuracy"
    ])

    # Process the selected samples in batches.
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        prompts = []
        gt_answers = []
        # For each sample, extract the prompt and ground truth answer.
        for idx in batch_indices:
            sample = test_dataset[idx]
            # sample["text"] contains the full text that was created as:
            # prompt_content + "\nAnswer: " + ground_truth.
            text = sample["text"]
            template_pos = text.find(RESPONSE_TEMPLATE)
            prompt = text[:template_pos + len(RESPONSE_TEMPLATE)]
            prompts.append(prompt)
            # Ground truth answer is stored in the "label" field.
            assert "label" in sample and sample["label"] is not None
            gt = sample["label"].strip()
            gt_answers.append(gt)

        # Tokenize the list of prompt strings.
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate outputs from the model (adjust max_new_tokens as needed).
        outputs = model.generate(**inputs, max_new_tokens=10)
        # Decode the outputs.
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        print(prompts[0])
        print()
        print(generated_texts[0])
        generated_texts = [text.split(RESPONSE_TEMPLATE)[1].strip() for text in generated_texts]

        # Process each generated output.
        for j in range(len(prompts)):
            full_generation = generated_texts[j]
            # strip special tokens
            extracted_ans = tokenizer.decode(tokenizer.encode(full_generation), skip_special_tokens=True).strip()
            # Compare extracted answer with ground-truth (case-insensitive exact match).
            correct = int(extracted_ans.strip().lower() == gt_answers[j].strip().lower())
            num_correct += correct

            wandb_table.add_data(prompts[j], full_generation, gt_answers[j], extracted_ans, correct)

    overall_accuracy = num_correct / len(indices) if len(indices) > 0 else 0.0

    # Log the table and overall accuracy to wandb.
    wandb.log({"eval_table": wandb_table, "eval_accuracy": overall_accuracy})
    return overall_accuracy

# %%

# Modify your training setup to use the custom trainer
trainer = ArthTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,  # Add the test dataset for evaluation
    data_collator=collator,
    # compute_metrics=compute_metrics, # GRRRR
    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,  # Add eval batch size
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=MAX_EPOCHS,
        max_steps=MAX_STEPS,
        learning_rate=LR,
        logging_steps=1,
        eval_steps=EVAL_STEPS,  # Evaluate every 100 steps
        evaluation_strategy="steps",  # Enable periodic evaluation
        save_strategy="no",  # Save checkpoints at the same time
        optim="adamw_torch",
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="finetune_outputs",
        report_to="wandb",
        run_name=EXP_NAME,
        # eval_accumulation_steps=EVAL_ACCUMULATION_STEPS,
    ),
)

# %%

trainer_stats = trainer.train()

# %%

# Run evaluation
accuracy = evaluate_and_log_model(model, tokenizer, test_dataset, n_samples=N_EVAL_SAMPLES, batch_size=BATCH_SIZE)

# %%


print("Final Evaluation Accuracy:", accuracy)


# %%

wandb.finish()
# %%
