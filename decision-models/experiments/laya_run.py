import time, inspect
import numpy as np
import laya
from laya import Router
from tickets import TICKETS, LABELS

Q = {"team": {"type": "choice", "instructions": "Which team should handle this support ticket?",
              "criteria": {"billing": "payments, invoices, refunds, plans, prices",
                           "technical": "bugs, crashes, errors, things not working",
                           "account": "login, password, profile, users, account settings"}}}
router = Router()
t = time.time(); r = router.predict({"body": TICKETS[0][0]}, Q); print(f"first call (loads model): {time.time()-t:.1f}s")
print("raw result:", r)
right, times, confs = 0, [], []
for text, gold in TICKETS:
    t = time.time(); r = router.predict({"body": text}, Q); times.append(time.time() - t)
    a = r["answers"]["team"]; pred = a["choice"]
    probs = a.get("probabilities") or a.get("probs") or a
    right += pred == gold
    confs.append((pred == gold, probs))
    if pred != gold: print(f"  wrong: gold {gold:9} pred {pred:9} {probs} | {text}")
print(f"laya zero-shot: {right}/{len(TICKETS)}, median {np.median(times)*1000:.0f} ms per decision")
