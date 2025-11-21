import torch, math
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

gen_name = "/fp/projects01/ec12/mathisdu/lllama/Llama-3.1-8B-Instruct"
clf_name = "/fp/projects01/ec12/mathisdu/models/deberta-phishy"

gen_tok = AutoTokenizer.from_pretrained(gen_name, use_fast=True)
gen_tok.pad_token = gen_tok.eos_token
gen_model = AutoModelForCausalLM.from_pretrained(gen_name, torch_dtype="auto", device_map="auto")

clf_tok = AutoTokenizer.from_pretrained(clf_name, use_fast=True)
clf_model = AutoModelForSequenceClassification.from_pretrained(clf_name, torch_dtype="auto", device_map="auto")
clf_model.eval()

def build_scoring_input(opening: str, original_text: str) -> str:
    return f"[GENERATED OPENING]\n{opening.strip()}\n\n[ORIGINAL TEXT]\n{original_text.strip()}"

def sample_openings(prompt, k=12, max_new_tokens=140):
    inp = gen_tok(prompt, return_tensors="pt").to(gen_model.device)
    outs = gen_model.generate(
        **inp,
        max_new_tokens=max_new_tokens,
        do_sample=True, top_p=0.95, temperature=0.95,
        repetition_penalty=1.05,
        eos_token_id=gen_tok.eos_token_id, pad_token_id=gen_tok.eos_token_id
    )
    resp_ids = outs[:, inp["input_ids"].shape[1]:]
    texts = gen_tok.batch_decode(resp_ids, skip_special_tokens=True)
    # keep only the first paragraph
    return [t.strip().split("\n\n")[0].strip() for t in texts]

@torch.no_grad()
def score_probs(texts):
    batch = clf_tok(texts, padding=True, truncation=True, return_tensors="pt").to(clf_model.device)
    logits = clf_model(**batch).logits.squeeze(-1).float()
    p = torch.sigmoid(logits).cpu().tolist()
    return p

def make_pair(prompt: str, original_text: str, k=12, prob_gap_min=0.15, logit_gap_min=0.75):
    cands = sample_openings(prompt, k=k)
    if len(cands) < 2:
        return None
    scoring_inputs = [build_scoring_input(c, original_text) for c in cands]
    p = score_probs(scoring_inputs)

    # convert to logits for margin checks
    def logit(x):
        eps = 1e-6; x = min(max(x, eps), 1-eps); return math.log(x/(1-x))
    s = [logit(x) for x in p]

    # we want to MINIMIZE score -> chosen has the LOWEST s
    order = sorted(range(len(cands)), key=lambda i: s[i])  # ascending
    best = order[0]
    worst = order[-1]

    if (abs(p[worst]-p[best]) < prob_gap_min) or (abs(s[worst]-s[best]) < logit_gap_min):
        return None

    return {
        "prompt": prompt,
        "chosen": cands[best],     # lower-scoring opening
        "rejected": cands[worst],  # higher-scoring opening
        "p_chosen": p[best],
        "p_rejected": p[worst],
        "logit_gap": s[worst]-s[best],
    }
