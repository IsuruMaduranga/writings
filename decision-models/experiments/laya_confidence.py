"""Top-answer probability for the 30 labelled tickets and the 4 no-team tickets; writes laya_confidence.json."""
import json
from laya import Router
from tickets import TICKETS, NO_TEAM, Q

router = Router()
top = lambda t: max(router.predict({"body": t}, Q)["answers"]["team"]["probabilities"].values())
out = {"labelled": [top(t) for t, _ in TICKETS], "no_team": [top(t) for t in NO_TEAM]}
with open("laya_confidence.json", "w") as f:
    json.dump(out, f, indent=1)
print({k: [round(v, 3) for v in vs] for k, vs in out.items()})
