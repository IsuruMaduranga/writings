"""Route a support ticket with Qwen2.5-0.5B-Instruct by reading three answer probabilities.

Needs: pip install torch transformers
Swap in Qwen/Qwen2.5-1.5B-Instruct or Qwen/Qwen2.5-3B-Instruct for better accuracy.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
labels = ["billing", "technical", "account"]
ticket = "I was billed twice this month, please refund one of the charges."
prompt = ("A support ticket is routed to one team: billing, technical, or account.\n"
          f"Ticket: {ticket}\nWhich team? Answer with one word.")
chat = [{"role": "user", "content": prompt}]
inputs = tok.apply_chat_template(                  # wrap the prompt in the chat format the model was trained on
    chat, add_generation_prompt=True, return_tensors="pt", return_dict=True)

with torch.inference_mode():
    p = model(**inputs).logits[0, -1].softmax(-1)

scores = []
for label in labels:
    # "billing", "Billing", " billing" and " Billing" are four different tokens; count them all
    variants = [label, label.title(), " " + label, " " + label.title()]
    ids = [tok.encode(v, add_special_tokens=False)[0] for v in variants]
    scores.append(p[ids].sum())
scores = torch.stack(scores) / sum(scores)         # rescale the three so they add up to 1
print(labels[scores.argmax().item()], {l: round(s, 4) for l, s in zip(labels, scores.tolist())})
