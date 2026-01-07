import os
import sys
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,  # Changed to AutoModel for best compatibility
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
import evaluate

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
# Ensure this path is correct for your 4B model
MODEL_ID = "/fp/projects01/ec12/mathisdu/gemma/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"
OUTPUT_DIR = "/fp/projects01/ec12/mathisdu/gemma/gemma-3-4b-output"

data_files = {
    "train": "/fp/homes01/u01/ec-mathiassd/phishing_detection/ai_agent/data/ephish/train.csv",
    "validation": "/fp/homes01/u01/ec-mathiassd/phishing_detection/ai_agent/data/ephish/val.csv",
}

SYSTEM_INSTRUCTION = """You are a strict email security classifier.
Task: Analyze the email and determine if it is safe or phishing.
Output: Respond ONLY with '0' for safe or '1' for phishing. Do not provide explanations.
"""

# ---------------------------------------------------------
# 2. MODEL & TOKENIZER SETUP
# ---------------------------------------------------------
print(f">>> Loading Tokenizer from: {MODEL_ID}")

try:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
except OSError:
    # Fallback to cache search if direct load fails
    cache_root = os.getenv("HF_HOME")
    if cache_root:
        print(">>> Standard load failed, searching cache snapshots...")
        snapshot_dir = Path(cache_root) / f"models--{MODEL_ID.replace('/', '--')}" / "snapshots"
        if snapshot_dir.exists():
            latest_snap = sorted(snapshot_dir.iterdir(), key=os.path.getmtime)[-1]
            print(f">>> Found manual snapshot: {latest_snap}")
            tok = AutoTokenizer.from_pretrained(latest_snap)
        else:
            raise

tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print(f">>> Loading Model (AutoModelForCausalLM)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Switch to AutoModelForCausalLM for better architecture handling
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    attn_implementation="sdpa", 
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
)

model.gradient_checkpointing_enable()                
model = prepare_model_for_kbit_training(model)       

# ---------------------------------------------------------
# 3. LORA CONFIGURATION
# ---------------------------------------------------------
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------------------------------------------------------
# 4. DATA PROCESSING
# ---------------------------------------------------------
max_len = 512
label_to_text = {0: "0", 1: "1"}

def format_and_tokenize(example):
    full_user_content = f"{SYSTEM_INSTRUCTION}\n\nEmail Content:\n{example['text']}"
    messages = [{"role": "user", "content": full_user_content}]

    # 1. Create full text
    full_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + label_to_text[int(example["label"])] + tok.eos_token

    # 2. Tokenize everything at once
    tokenized_full = tok(full_text, truncation=True, max_length=max_len, padding=False, add_special_tokens=False)
    input_ids = tokenized_full["input_ids"]
    labels = input_ids.copy()

    # 3. Calculate prompt length
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_ids)

    # 4. Mask the prompt so we only train on the completion (0 or 1)
    for i in range(len(labels)):
        if i < prompt_len:
            labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": tokenized_full["attention_mask"],
        "labels": labels
    }

print(">>> Processing datasets...")
raw_datasets = load_dataset("csv", data_files=data_files)
tokenized_datasets = raw_datasets.map(format_and_tokenize, batched=False)
tokenized_datasets = tokenized_datasets.remove_columns(["text", "label"])

# ---------------------------------------------------------
# 5. METRICS & TRAINING
# ---------------------------------------------------------
accuracy = evaluate.load("accuracy")

ID_0 = tok.convert_tokens_to_ids("0")
ID_1 = tok.convert_tokens_to_ids("1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    true_labels = []
    pred_labels = []

    for i in range(len(labels)):
        valid_indices = np.where(labels[i] != -100)[0]
        if len(valid_indices) > 0:
            target_idx = valid_indices[-1] 
            
            pred_id = predictions[i][target_idx]
            true_id = labels[i][target_idx]
            
            p = 1 if pred_id == ID_1 else 0 
            t = 1 if true_id == ID_1 else 0
            
            pred_labels.append(p)
            true_labels.append(t)
            
    return accuracy.compute(predictions=pred_labels, references=true_labels)

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple): logits = logits[0]
    return logits.argmax(dim=-1)

# Optimized training args for 4B model
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    # INCREASED: 4B is small, we can handle more examples per step
    per_device_train_batch_size=8, 
    # DECREASED: To balance the increased batch size (Effective batch = 8 * 2 = 16)
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=8,
    eval_accumulation_steps=1,
    learning_rate=2e-4,
    num_train_epochs=1,
    bf16=True, 
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets.get("validation"),
    tokenizer=tok,
    data_collator=DataCollatorForSeq2Seq(tok, pad_to_multiple_of=8),
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

print(">>> Starting Training...")
trainer.train()
trainer.save_model(os.path.join(OUTPUT_DIR, "gemma-3-4b-finetuned"))