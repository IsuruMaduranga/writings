#!/usr/bin/env python3
"""Figures for the decision-models post. Numbers are copied from the runs in ../experiments
(the confidence chart reads ../experiments/laya_confidence.json directly).

Regenerate: python3 gen_diagrams.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, RED, GREEN, GREY = "#2471a3", "#c0392b", "#1e8449", "#7f8c8d"

# 1. causal vs bidirectional attention
toks = ["[MASK]", "billing", "[MASK]", "account", "billed", "twice"]
n = len(toks)
fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), dpi=150)
for ax, causal, title in [(axes[0], True, "GPT-2 (causal): each token sees only earlier ones"),
                          (axes[1], False, "Laya (bidirectional): every token sees every token")]:
    m = np.tril(np.ones((n, n))) if causal else np.ones((n, n))
    ax.imshow(m, cmap=matplotlib.colors.ListedColormap(["#f2f3f4", BLUE if causal else GREEN]), vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(toks, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(toks, fontsize=8)
    ax.set_xlabel("can look at this token", fontsize=8); ax.set_ylabel("this token", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(np.arange(-.5, n), minor=True); ax.set_yticks(np.arange(-.5, n), minor=True)
    ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="minor", length=0)
axes[0].annotate("the billing slot\ncannot see the ticket", xy=(4.5, 0), xytext=(3.3, -1.9),
                 fontsize=8, color=RED, ha="center", annotation_clip=False,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1))
fig.tight_layout()
fig.savefig(os.path.join(HERE, "attention_masks.png"), bbox_inches="tight")
plt.close(fig)

# 2. accuracy of each approach on the same 30 tickets
ORANGE = "#ca6f1e"
rows = [("GPT-2 small (124M),\nread the answer probabilities", 10, BLUE),
        ("Qwen2.5-Instruct 0.5B,\nsame trick", 11, BLUE),
        ("Qwen2.5-Instruct 1.5B,\nsame trick", 17, BLUE),
        ("Qwen2.5-Instruct 3B,\nsame trick", 26, BLUE),
        ("GPT-2 + trained head\n(2,307 numbers, leave-one-out)", 27, ORANGE),
        ("Laya (421M), question\nin the input, no training", 29, GREEN)]
fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
y = np.arange(len(rows))[::-1]
ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.6)
for yi, (_, v, _) in zip(y, rows):
    ax.text(v + 0.3, yi, f"{v}/30", va="center", fontsize=9)
ax.axvline(10, color=GREY, ls="--", lw=1)
ax.text(10.3, y[0] + 0.45, "always guessing one team = 10/30", fontsize=8, color=GREY)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
ax.set_xlim(0, 32); ax.set_xlabel("tickets routed correctly (30 tickets, 10 per team)")
ax.set_title("Same 30 support tickets: read an LLM, train a head, or ask Laya", fontsize=11)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "accuracy.png"), bbox_inches="tight")
plt.close(fig)

# 3. Laya's confidence: real tickets vs tickets that fit no team
c = json.load(open(os.path.join(HERE, "..", "experiments", "laya_confidence.json")))
wrong = {29}                                           # "Transfer ownership..." was the one mistake
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(8, 3.2), dpi=150)
lab = np.array(c["labelled"]); nt = np.array(c["no_team"])
ok = [i for i in range(len(lab)) if i not in wrong]
ax.scatter(lab[ok], 1 + rng.uniform(-.12, .12, len(ok)), color=GREEN, s=28, label="real ticket, routed correctly")
ax.scatter(lab[list(wrong)], [1] * len(wrong), color=RED, s=40, marker="x", label="real ticket, routed wrongly")
ax.scatter(nt, 0 + rng.uniform(-.12, .12, len(nt)), color=GREY, s=28, label="ticket that fits no team")
ax.axvspan(nt.min(), nt.max(), color=GREY, alpha=0.12)
ax.set_yticks([0, 1]); ax.set_yticklabels(["no-team\ntickets (4)", "real\ntickets (30)"], fontsize=9)
ax.set_xlim(0.3, 1.02); ax.set_ylim(-0.5, 1.6)
ax.set_xlabel("Laya's probability for its top answer")
ax.set_title("Low confidence flags the misfits, and some good answers too", fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.02))
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "confidence.png"), bbox_inches="tight")
plt.close(fig)
print("wrote attention_masks.png, accuracy.png, confidence.png")
