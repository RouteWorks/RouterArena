#!/usr/bin/env python3
"""Rebuild the self-consistency router with a chosen escalation model.

Kept queries (qwen self-consistency >= tau) keep qwen's majority vote (from
cruq-sc-router.json, K=4/tau=0.6). Escalated queries (the 130 in esc_gis.json) use the
candidate escalation model's answer from phase2/data/escalation.jsonl instead of deepseek.

Env: ESC_MODEL (slug), ESC_LABEL (short name) -> writes cruq-sc-<label>.json
Cost note: v3 bills only the final model; honest cost also adds the K=4 qwen probe tokens on
the escalated queries. compute_honest_cost.py handles that separately.
"""
import json, os

ESC_MODEL = os.environ["ESC_MODEL"]
ESC_LABEL = os.environ["ESC_LABEL"]
ESC_GIS = "/private/tmp/claude-502/-Users-nabaruns-work-cruq-ai-cruq-ai/063e8a4a-b7ad-4b1d-9cd2-2cffe81c3e8b/scratchpad/esc_gis.json"

esc = set(json.load(open(ESC_GIS)))
cand = {}
for line in open("phase2/data/escalation.jsonl"):
    r = json.loads(line)
    if r["model"] == ESC_MODEL:
        cand[r["gi"]] = r
base = json.load(open("router_inference/predictions/cruq-sc-router.json"))  # K=4/tau=0.6 (qwen kept, deepseek escalated)

out, missing = [], 0
for r in base:
    gi = r["global index"]
    row = dict(r)
    if gi in esc:
        c = cand.get(gi)
        if c is None:
            missing += 1  # keep deepseek fallback if candidate call missing
        else:
            row["prediction"] = ESC_MODEL
            row["generated_result"] = {"generated_answer": c["response"], "success": True,
                                        "token_usage": {"input_tokens": c["in"], "output_tokens": c["out"],
                                                        "total_tokens": c["in"] + c["out"]}}
    row["accuracy"] = None; row["cost"] = None
    out.append(row)
json.dump(out, open(f"router_inference/predictions/cruq-sc-{ESC_LABEL}.json", "w"))
print(f"wrote cruq-sc-{ESC_LABEL}.json (escalation={ESC_MODEL}, {len(esc)} escalated, {missing} missing->deepseek)")
