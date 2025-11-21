from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

policy_name = gen_name
tok = gen_tok

dset = Dataset.from_list([{"prompt": d["prompt"], "chosen": d["chosen"], "rejected": d["rejected"]} for d in pairs])

base = AutoModelForCausalLM.from_pretrained(policy_name, torch_dtype="auto", device_map="auto")
lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
base = get_peft_model(base, lora_cfg)

cfg = DPOConfig(
    beta=0.1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_train_epochs=2,
    max_prompt_length=512,
    max_length=768,
    bf16=True,
    logging_steps=20,
)

trainer = DPOTrainer(
    model=base,
    ref_model=None,              # internal ref snapshot from model weights
    args=cfg,
    tokenizer=tok,
    train_dataset=dset
)
trainer.train()
