#!/usr/bin/env python3
"""Fetch GPT-2 small (124M) from Hugging Face into experiments/gpt2_weights/.

Three files: the weights (safetensors, ~548 MB), and the two tokenizer
files (vocab.json, merges.txt). No SDK, one urllib call per file.

    python3 download_gpt2.py
"""
import os, urllib.request

BASE = "https://huggingface.co/openai-community/gpt2/resolve/main/"
FILES = ["model.safetensors", "vocab.json", "merges.txt"]
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpt2_weights")

os.makedirs(DEST, exist_ok=True)
for name in FILES:
    path = os.path.join(DEST, name)
    if os.path.exists(path):
        print(f"have {name}")
        continue
    print(f"fetching {name} ...")
    urllib.request.urlretrieve(BASE + name, path)
    print(f"  {os.path.getsize(path)/1e6:.1f} MB")
print("done ->", DEST)
