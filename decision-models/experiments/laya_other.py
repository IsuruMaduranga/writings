"""Does an explicit 'other' option catch tickets that fit no team, without hurting the 30 real ones?"""
import json
import numpy as np
from laya import Router
from tickets import TICKETS, NO_TEAM, Q

with open("laya_confidence.json") as f:                      # written by laya_confidence.py
    conf = json.load(f)["labelled"]
print(f"labelled tickets with top probability <= 0.51: {sum(c <= 0.51 for c in conf)} of 30; median {np.median(conf):.2f}")
QO = {"team": dict(Q["team"], criteria=dict(Q["team"]["criteria"], other="anything that is not billing, technical or account"))}
router = Router()
right = sum(router.predict({"body": t}, QO)["answers"]["team"]["choice"] == g for t, g in TICKETS)
print(f"with an 'other' option: {right}/30 real tickets correct")
for t in NO_TEAM:
    a = router.predict({"body": t}, QO)["answers"]["team"]
    print(f"  {a['choice']:9} {a['probabilities']} | {t}")
