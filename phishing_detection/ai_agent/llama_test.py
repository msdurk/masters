from transformers import AutoTokenizer, AutoModelForCausalLM

folder = "/fp/projects01/ec12/mathisdu/lllama/Llama-3.1-8B-Instruct"

tok = AutoTokenizer.from_pretrained(folder, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(folder, device_map="auto")

print("Model loaded successfully!")
