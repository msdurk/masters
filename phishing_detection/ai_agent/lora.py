import os, sys, json, re
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import f1_score 
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, default_data_collator
import evaluate
import numpy as np  ### CHANGED: needed for compute_metrics

# --- Config ---
LOCAL_DIR = Path("/fp/projects01/ec12/mathisdu/llama/Llama-3.1-8B-Instruct")

data_files = {
    "train": "/fp/homes01/u01/ec-mathiassd/phishing_detection/ai_agent/data/ephish/train.csv",
    "validation": "/fp/homes01/u01/ec-mathiassd/phishing_detection/ai_agent/data/ephish/val.csv",
    "test": "/fp/homes01/u01/ec-mathiassd/phishing_detection/ai_agent/data/ephish/test.csv",
}
raw_datasets = load_dataset("csv", data_files=data_files)

tok = AutoTokenizer.from_pretrained(LOCAL_DIR)

model = AutoModelForCausalLM.from_pretrained(
    LOCAL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

tok.padding_side = "right"
pad_id = tok.pad_token_id

model.config.pad_token_id = pad_id

tok_len = len(tok)
emb_len = model.get_input_embeddings().weight.shape[0]
if tok_len != emb_len:
    model.resize_token_embeddings(tok_len)

# LoRA config – for Llama, target q_proj/v_proj (and optionally k/o)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"],  # you can add "k_proj", "o_proj" too
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

max_len = 512  # or 1024 if you can afford it

def build_prompt(email_text: str) -> str:
    return f"""You are an email security classifier.
Classify the following email as either "phish" or "ham".
Respond with exactly one word: phish or ham.

Email:
{email_text}

Answer:"""

label_to_word = {0: "ham", 1: "phish"}

# 1) Add prompt + target strings
def add_prompt_and_target(example):
    prompt = build_prompt(example["text"])
    label_word = label_to_word[int(example["label"])]
    example["prompt"] = prompt
    example["target"] = label_word
    return example

ds = raw_datasets.map(add_prompt_and_target)

max_len = 512  # or 1024 if you want and have VRAM

def tokenize_example(example):
    # prompt
    prompt_ids = tok(
        example["prompt"],
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    )["input_ids"]          # list[int]

    # target: "phish" or "ham"
    target_ids = tok(
        example["target"],
        add_special_tokens=False,
    )["input_ids"]          # list[int]

    # concat
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids

    # truncate if longer than max_len
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]

    attention_mask = [1] * len(input_ids)

    # manual padding up to max_len
    pad_len = max_len - len(input_ids)
    if pad_len > 0:
        input_ids += [pad_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len

    assert len(input_ids) == max_len
    assert len(attention_mask) == max_len
    assert len(labels) == max_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

cols_to_remove = ds["train"].column_names  # remove text/label/prompt/target etc.

tokenized = ds.map(
    tokenize_example,
    batched=False,
    remove_columns=cols_to_remove,
)
vocab_size = model.config.vocab_size

def fix_labels(example):
    fixed = []
    for t in example["labels"]:
        if t == -100:
            fixed.append(t)
        elif 0 <= t < vocab_size:
            fixed.append(t)
        else:
            # anything weird gets ignored instead of crashing CUDA
            fixed.append(-100)
    example["labels"] = fixed
    return example

tokenized = tokenized.map(fix_labels, batched=False)


print("tokenizer length:", len(tok))
print("model vocab_size:", model.config.vocab_size)
ex = tokenized["train"][0]
print("\nSample from train:")
print("  input_ids len:", len(ex["input_ids"]))
print("  labels len   :", len(ex["labels"]))
print("  labels min/max:", min(ex["labels"]), max(ex["labels"]))


# IMPORTANT: let Trainer/Collator handle padding & tensor conversion
# tokenized.set_format(type="torch")  ### CHANGED: remove this line

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

# We’ll decode and check if it produced "phish" or "ham"
id2label = {0: "ham", 1: "phish"}
label2id = {"ham": 0, "phish": 1}

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # If HF gave us a tuple, take the first element
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    # After preprocess_logits_for_metrics, logits is [batch, vocab]
    preds_token_ids = np.argmax(logits, axis=-1)

    # Map predicted token id -> token -> ham/phish via simple match
    pred_words = [tok.decode([int(tid)]).strip().lower() for tid in preds_token_ids]

    def map_word(w):
        if "phish" in w:
            return 1
        if "ham" in w:
            return 0
        return 1  # fallback: treat unknown as phish

    preds = np.array([map_word(w) for w in pred_words])

    # Recover gold labels: decode last non -100 token and map to 0/1
    gold = []
    for row in labels:
        idx = np.where(row != -100)[0]
        last_token_id = row[idx[-1]]
        gold_word = tok.decode([int(last_token_id)]).strip().lower()
        gold.append(map_word(gold_word))
    gold = np.array(gold)

    return {
        "accuracy": accuracy_metric.compute(predictions=preds, references=gold)["accuracy"],
        "f1": f1_metric.compute(predictions=preds, references=gold, average="binary")["f1"],
    }


training_args = TrainingArguments(
    output_dir="/fp/projects01/ec12/mathisdu/llama/llama-3.3-8b-phish-lora",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=50,
    eval_strategy="epoch",  ### CHANGED: correct name
    save_strategy="epoch",
    remove_unused_columns=False,   # keeps labels
    eval_accumulation_steps=1,     # avoids giant buffer
    load_best_model_at_end=True,
    bf16=True,   # if supported by your GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tok,
    data_collator=default_data_collator,
    preprocess_logits_for_metrics=lambda logits, labels: logits[:, -1, :],
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("./llama-3.1-8b-phish-lora-adapter")
