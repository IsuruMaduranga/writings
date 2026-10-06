"""Route a support ticket with GPT-2 small by reading three answer probabilities.

Needs: pip install torch transformers
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "openai-community/gpt2"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).eval()
labels = ["billing", "technical", "account"]
ticket = "I was billed twice this month, please refund one of the charges."
prompt = ("A support ticket is routed to one team: billing, technical, or account.\n"
          f"Ticket: {ticket}\nTeam:")
inputs = tok(prompt, return_tensors="pt")

with torch.inference_mode():
    p = model(**inputs).logits[0, -1].softmax(-1)    # one run: a probability for each of 50,257 tokens

ids = [tok.encode(" " + label)[0] for label in labels]   # after "Team:" the answer starts with a space
scores = p[ids] / p[ids].sum()                          # rescale the three so they add up to 1
print(labels[scores.argmax().item()], {l: round(s, 4) for l, s in zip(labels, scores.tolist())})
