"""Top-answer probability for the 30 labelled tickets and the 4 no-team tickets; writes laya_confidence.json."""
import json
from laya import Router
from tickets import TICKETS
from laya_run import Q
NO_TEAM = ["What are your office hours?", "Do you have a job opening for a designer?",
           "asdf qwer zxcv", "I love your product, thank you!"]
router = Router()
top = lambda t: max(router.predict({"body": t}, Q)["answers"]["team"]["probabilities"].values())
out = {"labelled": [top(t) for t, _ in TICKETS], "no_team": [top(t) for t in NO_TEAM]}
json.dump(out, open("laya_confidence.json", "w"), indent=1)
print({k: [round(v, 3) for v in vs] for k, vs in out.items()})
