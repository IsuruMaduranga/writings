"""Every number in 'How the LLMs you know work': tokens, one earlier word changing the next token, greedy decoding."""
import os, time
import numpy as np
from gpt2_numpy import load_safetensors, gpt2, softmax
from gpt2_tokenizer import GPT2Tokenizer, GPT2_DIR

W = load_safetensors(os.path.join(GPT2_DIR, "model.safetensors"))
tok = GPT2Tokenizer()
print(f"vocabulary: {W['wte.weight'].shape[0]:,} tokens, parameters: {sum(v.size for v in W.values()):,}")
for s in ["I was billed twice", "Eiffel"]:
    print(f"{s!r} -> {[tok.decode([i]) for i in tok.encode(s)]}")

def top(text, k=5):
    p = softmax(gpt2(np.array(tok.encode(text)), W)[-1])         # one run of the model
    order = np.argsort(-p)[:k]
    return [(tok.decode([int(i)]), float(p[i])) for i in order]

print("\nthe same sentence with one earlier word changed:")
for s in ["The Eiffel Tower is located in the city of",
          "The CN Tower is located in the city of",
          "The Tower is located in the city of",
          "I took a photo of the Eiffel"]:
    print(f"  {s!r:48} " + "  ".join(f"{t!r} {p:.3f}" for t, p in top(s)))

print("\ngreedy decoding, one token per run of the model:")
ids = tok.encode("The Eiffel Tower is located in the city of")
t = time.time()
for step in range(4):
    p = softmax(gpt2(np.array(ids), W)[-1])
    nxt = int(p.argmax())                                          # greedy: take the most likely token
    print(f"  step {step + 1}: {len(ids):2} tokens in, picked {tok.decode([nxt])!r} at {p[nxt]:.3f}")
    ids.append(nxt)
print(f"  {(time.time() - t) * 1000:.0f} ms for 4 tokens -> {tok.decode(ids)!r}")

call = '{"name": "escalate", "arguments": {"team": "billing"}}'
print(f"\na tool call written as text: {call} = {len(tok.encode(call))} GPT-2 tokens")
