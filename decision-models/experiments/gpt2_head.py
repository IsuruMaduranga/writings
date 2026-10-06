"""Two patches on frozen GPT-2: (1) divide out its bias, (2) train a tiny head on its last hidden state."""
import os, time
import numpy as np
from gpt2_numpy import load_safetensors, layers, attention, mlp, layer_norm, softmax, gpt2
from gpt2_tokenizer import GPT2Tokenizer, GPT2_DIR
from tickets import TICKETS, LABELS, GPT2_PROMPT as PROMPT

W = load_safetensors(os.path.join(GPT2_DIR, "model.safetensors"))
tok = GPT2Tokenizer()
label_ids = [tok.encode(" " + l)[0] for l in LABELS]
y = np.array([LABELS.index(g) for _, g in TICKETS])

# Patch 1: contextual calibration. Ask with an empty ticket, divide that bias out.
def label_probs(text):
    p = softmax(gpt2(np.array(tok.encode(PROMPT.format(t=text))), W)[-1])[label_ids]
    return p / p.sum()
bias = label_probs("N/A")
print("label probs for an empty ticket:", dict(zip(LABELS, bias.round(3))))
P = np.array([label_probs(t) for t, _ in TICKETS])
print(f"raw accuracy {np.mean(P.argmax(1) == y):.2f}, after dividing out the bias {np.mean((P / bias).argmax(1) == y):.2f}")

# Patch 2: a linear head on the final hidden vector (the thing just before unembedding)
def hidden(text):
    ids = np.array(tok.encode("Ticket: " + text))
    x = W["wte.weight"][ids] + W["wpe.weight"][np.arange(len(ids))]
    for p in layers(W):
        x = x + attention(layer_norm(x, p["ln_1.weight"], p["ln_1.bias"]), p)
        x = x + mlp(layer_norm(x, p["ln_2.weight"], p["ln_2.bias"]), p)
    return layer_norm(x, W["ln_f.weight"], W["ln_f.bias"]).mean(0)   # average over tokens: one 768-number vector

X = np.array([hidden(t) for t, _ in TICKETS])

def train_head(X, y, steps=500, lr=0.1, l2=1e-2):
    Wh = np.zeros((X.shape[1], len(LABELS))); b = np.zeros(len(LABELS))
    Y = np.eye(len(LABELS))[y]
    for _ in range(steps):
        p = softmax(X @ Wh + b)
        g = (p - Y) / len(X)
        Wh -= lr * (X.T @ g + l2 * Wh); b -= lr * g.sum(0)
    return Wh, b

# leave-one-out: train on 29, test on the one held out, 30 times
preds, confs = [], []
for i in range(len(X)):
    m = np.arange(len(X)) != i
    mu, sd = X[m].mean(0), X[m].std(0) + 1e-6           # scale with the training 29 only
    Wh, b = train_head((X[m] - mu) / sd, y[m])
    p = softmax(((X[i:i+1] - mu) / sd) @ Wh + b)[0]
    preds.append(p.argmax()); confs.append(p.max())
preds, confs = np.array(preds), np.array(confs)
print(f"head on frozen GPT-2, leave-one-out: {np.sum(preds == y)}/{len(y)} correct")
print(f"head size: {768*3 + 3} numbers, trained on 29 examples")
print(f"mean confidence when right {confs[preds == y].mean():.2f}, when wrong {confs[preds != y].mean() if (preds != y).any() else float('nan'):.2f}")
for i in np.where(preds != y)[0]:
    print(f"  wrong: gold {LABELS[y[i]]:9} pred {LABELS[preds[i]]:9} conf {confs[i]:.2f} | {TICKETS[i][0]}")
t = time.time(); [hidden(TICKETS[0][0]) for _ in range(5)]; print(f"time per decision: {(time.time()-t)/5*1000:.0f} ms")
print(f"right answers below 0.75 confidence: {np.sum((preds == y) & (confs < 0.75))} of {np.sum(preds == y)}")
