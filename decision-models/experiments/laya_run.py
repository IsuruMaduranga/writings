"""Laya on the 30 tickets: accuracy, the wrong answers, and time per decision."""
import time
import numpy as np
from laya import Router
from tickets import TICKETS, Q

router = Router()
t = time.time(); r = router.predict({"body": TICKETS[0][0]}, Q); print(f"first call (loads model): {time.time()-t:.1f}s")
print("raw result:", r)
right, times = 0, []
for text, gold in TICKETS:
    t = time.time(); a = router.predict({"body": text}, Q)["answers"]["team"]; times.append(time.time() - t)
    right += a["choice"] == gold
    if a["choice"] != gold: print(f"  wrong: gold {gold:9} pred {a['choice']:9} {a['probabilities']} | {text}")
print(f"laya zero-shot: {right}/{len(TICKETS)}, median {np.median(times)*1000:.0f} ms per decision")
