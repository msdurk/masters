import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
BASE_MODEL_PATH = "/fp/projects01/ec12/mathisdu/llama/Llama-3.1-8B-Instruct"
# Point this to the specific folder where the adapter was saved
ADAPTER_PATH = "/fp/projects01/ec12/mathisdu/llama/llama-3.1-8b-phish-lora-classifier/final_adapter"
TEST_DATA_PATH = "/fp/homes01/u01/ec-mathiassd/phishing_detection/ai_agent/data/ephish/test.csv"
OUTPUT_CSV = "test_results_full.csv"

# ---------------------------------------------------------
# LOAD MODEL & TOKENIZER
# ---------------------------------------------------------
print(">>> Loading Base Model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print(f">>> Loading Adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# ---------------------------------------------------------
# CLASSIFICATION FUNCTION
# ---------------------------------------------------------
SYSTEM_PROMPT = """You are a strict email security classifier.
Task: Analyze the email and determine if it is safe or phishing.
Output: Respond ONLY with '0' for safe or '1' for phishing. Do not provide explanations."""

def classify_email(email_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": email_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Extract only the newly generated token
    generated_token = outputs[:, inputs.input_ids.shape[1]:]
    result = tokenizer.decode(generated_token[0], skip_special_tokens=True).strip()
    return result

# ---------------------------------------------------------
# MAIN EVALUATION LOOP
# ---------------------------------------------------------
print(">>> Loading Test Data...")
df = pd.read_csv(TEST_DATA_PATH)
# Ensure labels are integers
df['label'] = df['label'].astype(int)

print(f">>> Starting Inference on {len(df)} emails...")

y_true = []
y_pred = []
predictions_text = []

# Use tqdm for a progress bar
for index, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
    text = row['text']
    true_label = row['label']
    
    # Run Inference
    try:
        pred_text = classify_email(text)
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        pred_text = "-1" # Error flag

    # Convert text prediction ("0"/"1") to integer
    # If the model outputs garbage (rare), we default to safe (0) or mark as -1
    if pred_text == "1":
        pred_int = 1
    elif pred_text == "0":
        pred_int = 0
    else:
        pred_int = 0 # Fallback: assume safe if unsure, or separate class
        
    y_true.append(true_label)
    y_pred.append(pred_int)
    predictions_text.append(pred_text)

# ---------------------------------------------------------
# METRICS & SAVING
# ---------------------------------------------------------
print("\n" + "="*30)
print("FINAL RESULTS")
print("="*30)

acc = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
cm = confusion_matrix(y_true, y_pred)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\nConfusion Matrix:")
print(f"True Negatives (Safe correctly identified): {cm[0][0]}")
print(f"False Positives (Safe marked as Phish):     {cm[0][1]}")
print(f"False Negatives (Phish marked as Safe):     {cm[1][0]}")
print(f"True Positives (Phish correctly identified):{cm[1][1]}")

# Save detailed results to CSV
df['predicted_label'] = y_pred
df['raw_output'] = predictions_text
df['is_correct'] = df['label'] == df['predicted_label']

df.to_csv(OUTPUT_CSV, index=False)
print(f"\n>>> Detailed results saved to {OUTPUT_CSV}")