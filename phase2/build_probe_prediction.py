#!/usr/bin/env python3
"""Emit the online-probe (cascade) router's prediction file.

Best config from phase2/online_probe.py: probe each query with qwen + coder; if their
boxed answers AGREE, keep qwen's answer (cheap, ~87% correct); else escalate to deepseek.
generated_result is the FINAL model's cached output. Cost note: the RouterArena
single-model-per-row format bills only the final model at its own price, so v3's recomputed
cost UNDERSTATES a cascade (it cannot charge probe tokens at the probe models' prices).
The honest cascade cost (probes + escalation) is computed by online_probe.py; use that
number, not v3's cost, for arena-S. This file exists for the accuracy confirmation and as a
submission artifact.

Out: router_inference/predictions/cruq-probe-router.json
"""
import json, re
def norm(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())
BOXED = re.compile(r"\\boxed\{+([^{}]*)\}+")
def boxed(gr):
    ga = gr.get("generated_answer") if isinstance(gr, dict) else gr
    ga = ga if isinstance(ga, str) else ""
    ms = BOXED.findall(ga)
    return norm(ms[-1]) if ms else norm(ga[-40:])

Q = json.load(open("router_inference/predictions/cruq-single-qwen.json"))
C = {r["global index"]: r for r in json.load(open("router_inference/predictions/cruq-single-coder.json"))}
D = {r["global index"]: r for r in json.load(open("router_inference/predictions/cruq-single-deepseek.json"))}

out = []
import collections
dist = collections.Counter()
for qr in Q:
    gi = qr["global index"]
    cr, dr = C.get(gi), D.get(gi)
    if cr is None or dr is None:
        continue
    agree = boxed(qr.get("generated_result")) and boxed(qr.get("generated_result")) == boxed(cr.get("generated_result"))
    src = qr if agree else dr
    dist["qwen" if agree else "deepseek"] += 1
    row = dict(qr)
    row["prediction"] = src["prediction"]
    row["generated_result"] = src["generated_result"]
    row["accuracy"] = None
    row["cost"] = None
    out.append(row)

json.dump(out, open("router_inference/predictions/cruq-probe-router.json", "w"))
print(f"wrote {len(out)} rows -> cruq-probe-router.json")
print("final-model routing:", dict(dist))
