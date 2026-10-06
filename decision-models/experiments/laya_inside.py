import torch, numpy as np, time
from laya import Router
from laya.common import build_sequence
from tickets import TICKETS
Q = {"team": {"type": "choice", "instructions": "Which team should handle this support ticket?",
              "criteria": {"billing": "payments, invoices, refunds, plans, prices",
                           "technical": "bugs, crashes, errors, things not working",
                           "account": "login, password, profile, users, account settings"}}}
router = Router()
router.predict({"body": "warm up"}, Q)
agent = router._agents["english"]
m, tok = agent.model, agent.tok
count = lambda mod: sum(p.numel() for p in mod.parameters())
print(f"encoder {count(m.encoder):,}  head layers {count(m.head):,}  scorer {count(m.scorer):,}  act head {count(m.act_head):,}  total {count(m):,}")
print("encoder layers:", m.encoder.config.num_hidden_layers, "hidden size:", m.encoder.config.hidden_size, "head layers:", len(m.head.layers))
print("temperature buffer:", m.temperature.tolist())
print("cfg:", {k: agent.cfg.get(k) for k in ("max_len", "head_max_len", "encoder")})

text = "I was billed twice this month, please refund one of the charges."
internal = {"team": agent._to_internal(Q["team"])}
items = agent._encode_state(text, ["team"], internal)
ids, markers = items[0]["ids"], items[0]["markers"]
print(f"\nsequence: {len(ids)} tokens, option markers at {markers}")
print(tok.decode(ids))

# raw scores, straight from the model
b = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
     "marker_pos": torch.tensor([markers]), "marker_mask": torch.ones(1, len(markers), dtype=torch.bool),
     "qtype": torch.tensor([0])}
with torch.no_grad():
    logits, act = m(*(v.to(agent.device) for v in b.values()))
lg = logits[0].float().cpu().numpy()
print("raw option scores:", lg.round(2), " softmax:", (np.exp(lg - lg.max()) / np.exp(lg - lg.max()).sum()).round(4))
print("act head softmax:", torch.softmax(act[0].float(), -1).cpu().numpy().round(4))

# position bias: same ticket, same options, different order
for order in [None, [2, 1, 0], [1, 2, 0]]:
    q = {"team": dict(Q["team"], **({"option_order": order} if order else {}))}
    print("order", order, router.predict({"body": "Notifications arrive two hours late."}, q)["answers"]["team"]["probabilities"])

# the labels carry meaning: rename them to opaque letters
QA = {"team": {"type": "choice", "instructions": Q["team"]["instructions"],
               "criteria": {"A": Q["team"]["criteria"]["billing"], "B": Q["team"]["criteria"]["technical"], "C": Q["team"]["criteria"]["account"]}}}
QN = {"team": {"type": "choice", "instructions": Q["team"]["instructions"],
               "criteria": {"billing": "", "technical": "", "account": ""}}}
amap = {"A": "billing", "B": "technical", "C": "account"}
for name, q, f in [("letters + descriptions", QA, lambda c: amap[c]), ("names only, no descriptions", QN, lambda c: c)]:
    right = sum(f(router.predict({"body": t}, q)["answers"]["team"]["choice"]) == g for t, g in TICKETS)
    print(f"{name}: {right}/30")

# batching: 30 tickets one by one vs one batch
t = time.time(); [router.predict({"body": x}, Q) for x, _ in TICKETS]; one = time.time() - t
t = time.time(); agent.predict_batch([{"body": x} for x, _ in TICKETS], Q); batch = time.time() - t
print(f"30 tickets one at a time {one*1000:.0f} ms, as one batch {batch*1000:.0f} ms, device {agent.device}")
