import os
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    Gemma3ForCausalLM,  # Specific class for Gemma 3
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
import evaluate

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
# We use the Model ID since you downloaded it to the cache in the previous step
MODEL_ID = "/fp/projects01/ec12/mathisdu/gemma/models--google--gemma-3-27b-it" 
# If you are using the 4B model, change above to: "google/gemma-3-4b-it"

OUTPUT_DIR = "/fp/projects01/ec12/mathisdu/gemma"

# Update these paths to your actual data locations
data_files = {
    "train": "/fp/homes01/u01/ec-mathiassd/phishing_detection/ai_agent/data/ephish/train.csv",
    "validation": "/fp/homes01/u01/ec-mathiassd/phishing_detection/ai_agent/data/ephish/val.csv"
}

# Gemma doesn't support "System" roles natively, so we prepend this to the User prompt later
SYSTEM_INSTRUCTION = """You are a strict email security classifier.
Task: Analyze the email and determine if it is safe or phishing.
Output: Respond ONLY with '0' for safe or '1' for phishing. Do not provide explanations.
"""

# ---------------------------------------------------------
# 2. MODEL & TOKENIZER SETUP
# ---------------------------------------------------------
print(f">>> Loading Model: {MODEL_ID}...")

# 1. Load Tokenizer
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "right"

# Fix Padding: Gemma usually has a pad token, but if not, use EOS
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# 2. Load Model (Optimized for A100)
# If you have the 40GB A100, use 4-bit loading. If 80GB, you can try setting load_in_4bit=False
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = Gemma3ForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2", # <--- A100 Superpower
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# ---------------------------------------------------------
# 3. LORA CONFIGURATION
# ---------------------------------------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    # Gemma uses standard projection names, targeting all gives best results
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
    # A. MERGE SYSTEM PROMPT FOR GEMMA
    # Gemma template is strict: <start_of_turn>user ... <end_of_turn>
    # It rejects "system" roles. We merge instructions into the user turn.
    full_user_content = f"{SYSTEM_INSTRUCTION}\n\nEmail Content:\n{example['text']}"

    messages = [
        {"role": "user", "content": full_user_content},
    ]
    
    # B. Apply Template (Prompt only)
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # C. Target Text
    target_text = label_to_text[int(example["label"])]
    
    # D. Full Text + EOS
    full_text = prompt_text + target_text + tok.eos_token
    
    # E. Tokenize
    tokenized = tok(
        full_text,
        truncation=True,
        max_length=max_len,
        padding=False,
        add_special_tokens=False
    )
    
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    labels = input_ids.copy()

    # F. MASKING (Loss only on the answer '0' or '1')
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_ids)

    # Set prompt tokens to -100 so model isn't trained to memorize the email
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

print(">>> Processing datasets...")
raw_datasets = load_dataset("csv", data_files=data_files)
tokenized_datasets = raw_datasets.map(format_and_tokenize, batched=False)
tokenized_datasets = tokenized_datasets.remove_columns(["text", "label"])

# ---------------------------------------------------------
# 5. METRICS
# ---------------------------------------------------------
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    true_labels = []
    pred_labels = []
    
    for i in range(len(labels)):
        valid_indices = np.where(labels[i] != -100)[0]
        if len(valid_indices) > 0:
            target_idx = valid_indices[-1]
            pred_token_id = predictions[i][target_idx]
            true_token_id = labels[i][target_idx]
            
            pred_txt = tok.decode([pred_token_id]).strip()
            true_txt = tok.decode([true_token_id]).strip()
            
            p = 1 if pred_txt == "1" else 0
            l = 1 if true_txt == "1" else 0
            
            pred_labels.append(p)
            true_labels.append(l)

    return accuracy.compute(predictions=pred_labels, references=true_labels)

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

# ---------------------------------------------------------
# 6. TRAINER
# ---------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4, 
    gradient_accumulation_steps=4, # Adjusted for 27B model size
    per_device_eval_batch_size=4,
    eval_accumulation_steps=1,
    learning_rate=2e-4,
    num_train_epochs=1,
    bf16=True,             # Essential for A100
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets.get("validation"), # Handle if val is missing
    tokenizer=tok,
    data_collator=DataCollatorForSeq2Seq(tok, pad_to_multiple_of=8),
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

print(">>> Starting Training...")
trainer.train()

print(">>> Saving Adapter...")
trainer.save_model(os.path.join(OUTPUT_DIR, "final_adapter"))