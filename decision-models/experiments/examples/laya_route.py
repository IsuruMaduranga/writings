"""Route a support ticket with Laya, a decision model that takes the question as input.

Needs: pip install laya==0.3.27
"""
from laya import Router

router = Router()
Q = {"team": {"type": "choice", "instructions": "Which team should handle this support ticket?",
              "criteria": {"billing": "payments, invoices, refunds, plans, prices",
                           "technical": "bugs, crashes, errors, things not working",
                           "account": "login, password, profile, users, account settings"}}}
ticket = "I was billed twice this month, please refund one of the charges."

answer = router.predict({"body": ticket}, Q)["answers"]["team"]
print(answer["choice"], answer["probabilities"])
