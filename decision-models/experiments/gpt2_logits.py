"""GPT-2 small as a classifier: one forward pass, read three token probabilities."""
import sys, time, os
import numpy as np
from gpt2_numpy import load_safetensors, gpt2, softmax, generate
from gpt2_tokenizer import GPT2Tokenizer, GPT2_DIR
from tickets import TICKETS, LABELS

W = load_safetensors(os.path.join(GPT2_DIR, "model.safetensors"))
tok = GPT2Tokenizer()
label_ids = [tok.encode(" " + l)[0] for l in LABELS]
print("label first tokens:", [tok.decode([i]) for i in label_ids], [len(tok.encode(" " + l)) for l in LABELS])

PROMPT = ("A support ticket is routed to one team: billing, technical, or account.\n"
          "Ticket: {t}\nTeam:")

def classify(text):
    logits = gpt2(np.array(tok.encode(PROMPT.format(t=text))), W)[-1]
    p_all = softmax(logits)
    p = p_all[label_ids]
    return LABELS[int(p.argmax())], p / p.sum(), p.sum()

def table(text, k=5):
    """The full next-token table, then the same table with only the three answers kept."""
    p = softmax(gpt2(np.array(tok.encode(PROMPT.format(t=text))), W)[-1])
    print("  all 50,257 tokens, top 5:", "  ".join(f"{tok.decode([int(i)])!r} {p[i]:.3f}" for i in np.argsort(-p)[:k]))
    print("  only the three answers:  ", "  ".join(f"{tok.decode([i])!r} {p[i]:.4f}" for i in label_ids),
          "-> rescaled:", "  ".join(f"{l} {v:.2f}" for l, v in zip(LABELS, p[label_ids] / p[label_ids].sum())))
print("next-token table after 'Team:' for", repr(TICKETS[0][0]))
table(TICKETS[0][0])

correct, times, rows = 0, [], []
for text, gold in TICKETS:
    t = time.time(); pred, p, mass = classify(text); times.append(time.time() - t)
    correct += pred == gold
    rows.append((gold, pred, p.max(), mass))
print(f"zero-shot accuracy: {correct}/{len(TICKETS)}")
print(f"median time per decision: {np.median(times)*1000:.0f} ms (one forward pass, NumPy CPU)")
from collections import Counter
print("predicted:", Counter(r[1] for r in rows))
print("mean prob mass on the three label tokens:", f"{np.mean([r[3] for r in rows]):.3f}")
for (text, gold), (g, pr, c, m) in list(zip(TICKETS, rows))[:6]:
    print(f"  {gold:9} -> {pr:9} conf {c:.2f}  | {text[:50]}")

# The generate-then-parse way, for comparison
t = time.time()
out = generate(PROMPT.format(t=TICKETS[0][0]), W, tok, n=5)
print(f"\ngenerate 5 tokens then parse: {(time.time()-t)*1000:.0f} ms ->", repr(out.split('Team:')[-1]))
