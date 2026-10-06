"""The same read-the-label-probabilities trick as gpt2_logits.py, on a small instruction-tuned model.

    python instruct_logits.py [Qwen/Qwen2.5-3B-Instruct] [--lower-only]
"""
import sys, time
from collections import Counter
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tickets import TICKETS, LABELS

MODEL = next((a for a in sys.argv[1:] if "/" in a), "Qwen/Qwen2.5-0.5B-Instruct")
LOWER_ONLY = "--lower-only" in sys.argv              # read only "billing", as my first version did
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
print(f"{MODEL}: {sum(p.numel() for p in model.parameters()):,} parameters")

# The model may write "billing", "Billing" or " billing"; each spelling is a different token.
SPELLINGS = [[tok.encode(v)[0] for v in (l, l.capitalize(), " " + l, " " + l.capitalize())] for l in LABELS]
print("label tokens:", [[tok.decode([i]) for i in ids] for ids in SPELLINGS])
if LOWER_ONLY:
    SPELLINGS = [ids[:1] for ids in SPELLINGS]

PROMPT = ("A support ticket is routed to one team: billing, technical, or account.\n"
          "Ticket: {t}\nWhich team? Answer with one word.")

def ids_for(text):
    msgs = [{"role": "user", "content": PROMPT.format(t=text)}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]

@torch.inference_mode()
def next_token_probs(text):
    return torch.softmax(model(ids_for(text)).logits[0, -1], -1).numpy()   # one forward pass

def classify(text):
    p_all = next_token_probs(text)
    p = np.array([p_all[ids].sum() for ids in SPELLINGS])
    return LABELS[int(p.argmax())], p / p.sum(), p.sum(), tok.decode([int(p_all.argmax())])

def table(text, k=5):
    p = next_token_probs(text)
    print("  all tokens, top 5:", "  ".join(f"{tok.decode([int(i)])!r} {p[i]:.3f}" for i in np.argsort(-p)[:k]))

print("next-token table for", repr(TICKETS[0][0]))
table(TICKETS[0][0])

classify(TICKETS[0][0])                                         # warm-up
correct, times, rows = 0, [], []
for text, gold in TICKETS:
    t = time.time(); pred, p, mass, top = classify(text); times.append(time.time() - t)
    correct += pred == gold
    rows.append((gold, pred, p.max(), mass, top))
print(f"zero-shot accuracy: {correct}/{len(TICKETS)}")
print(f"median time per decision: {np.median(times)*1000:.0f} ms (one forward pass, CPU)")
print("predicted:", Counter(r[1] for r in rows))
print("mean prob mass on the three label tokens:", f"{np.mean([r[3] for r in rows]):.3f}")
print("unrestricted top token was an answer:", sum(r[4].strip().lower() in LABELS for r in rows), "of", len(rows))
for (text, gold), (g, pr, c, m, top) in zip(TICKETS, rows):
    mark = "  " if g == pr else "X "
    print(f"{mark}{gold:9} -> {pr:9} conf {c:.2f} mass {m:.2f} | {text[:55]}")

empty = classify("N/A")
print("\nempty ticket (N/A):", dict(zip(LABELS, np.round(empty[1], 3))))

ids = ids_for(TICKETS[0][0])
with torch.inference_mode():
    t = time.time()
    out = model.generate(ids, max_new_tokens=20, do_sample=False)
    dt = time.time() - t
print(f"generate (greedy, up to 20 tokens): {dt*1000:.0f} ms ->",
      repr(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)))
