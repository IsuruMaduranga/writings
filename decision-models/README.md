# Decision Models: An LLM With the Talking Taken Out

This folder holds the post and every experiment behind it. The post explains
decision models, such as TypeSafe's Jev and the open Laya, to a reader who
has only used a chat model. It builds up from how an LLM works to reading
answer probabilities from GPT-2 and Qwen2.5, a trained head, and Laya's
internals.

Read the post: [`article.md`](article.md).

## Try it on your own text

[`experiments/examples/`](experiments/examples/) holds three short scripts
that route one support ticket. Each is self-contained, so you can copy it
into any folder and run it. Edit `ticket` to route another ticket, or
the prompt and the team names to try your own task.

| Script | Method | Install |
|---|---|---|
| `gpt2_probabilities.py` | Read three answer probabilities from GPT-2 small | `pip install torch transformers` |
| `qwen_probabilities.py` | The same with Qwen2.5-0.5B-Instruct, summing each answer's spellings | `pip install torch transformers` |
| `laya_route.py` | Ask Laya, which takes the question as input | `pip install laya==0.3.27` |

Each script prints the chosen team and its probabilities, for example
`billing {'billing': 0.9872, 'technical': 0.0068, 'account': 0.006}`.

## Rerun the experiments

The experiments ran on a MacBook with an Apple M4 Pro chip (14 CPU cores,
20 GPU cores) and Python 3.13. The Laya experiments use the chip's GPU
through PyTorch's Metal Performance Shaders (MPS) backend. Everything else
runs on its CPU cores.

1. Create a virtual environment and install the dependencies:

   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   ```

1. Download GPT-2 small from Hugging Face, about 550 MB, into
   `experiments/gpt2_weights/`:

   ```bash
   cd experiments
   ../.venv/bin/python download_gpt2.py
   ```

1. From the `experiments/` folder, run any script in the following table.
   Each one prints the numbers the post quotes.

   | Script | What it prints |
   |---|---|
   | `gpt2_basics.py` | Tokens, the Eiffel Tower and CN Tower examples, greedy decoding, the token count of a tool call |
   | `gpt2_logits.py` | GPT-2 reading the three answer probabilities: 10 of 30 tickets correct |
   | `gpt2_head.py` | Dividing out GPT-2's bias (9 of 30), and a trained head on frozen GPT-2 (27 of 30) |
   | `instruct_logits.py` | The same trick on Qwen2.5-Instruct; pass a model name such as `Qwen/Qwen2.5-3B-Instruct` |
   | `laya_run.py` | Laya on the 30 tickets: 29 of 30, about 25 ms each |
   | `laya_inside.py` | Laya's token sequence, parameter counts, raw scores, option order, label wording, and batching |
   | `laya_more.py` | Three questions at once, tickets that fit no team, and a planted instruction |
   | `laya_other.py` | The trade-off of adding an `other` option |
   | `laya_confidence.py` | Writes `laya_confidence.json`, the data for the confidence chart |

   For example:

   ```bash
   ../.venv/bin/python gpt2_logits.py
   ```

The Laya and Qwen2.5 scripts download their model weights from Hugging Face
on the first run. The Qwen2.5-Instruct models take about 1 GB, 3 GB, and
6 GB for the 0.5B, 1.5B, and 3B sizes. Timings depend on your hardware.

## Rebuild the figures

The three charts in `diagrams/` come from `diagrams/gen_diagrams.py`. The
accuracy numbers are copied into the script by hand from the runs. The
confidence chart reads `experiments/laya_confidence.json`.

```bash
cd diagrams
../.venv/bin/python gen_diagrams.py
```

## Files

- `article.md`: the post. Mechanisms are Mermaid diagrams; measured results
  are the PNG files in `diagrams/`.
- `experiments/tickets.py`: the 30 hand-labelled support tickets, 10 per
  team, the four tickets that fit no team, and the GPT-2 prompt and Laya
  question the scripts share.
- `experiments/gpt2_numpy.py` and `experiments/gpt2_tokenizer.py`: the
  GPT-2 forward pass and tokenizer in plain NumPy, running OpenAI's released
  weights.
