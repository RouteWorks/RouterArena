#!/usr/bin/env python3
"""Honest arena-S for each escalation-swap variant: v3 accuracy + honest cost
(K=4 qwen probes on every scored query + the candidate's cost on the 130 escalated)."""
import json, math, collections

COST = json.load(open("model_cost/model_cost.json"))
def price(slug): c = COST[slug]; return c["input_token_price_per_million"], c["output_token_price_per_million"]
def arena(a, c1k, b=.1, cmin=.0044, cmax=200.):
    if c1k <= 0: return 0.
    C = max(0., min(1., (math.log2(cmax) - math.log2(c1k)) / (math.log2(cmax) - math.log2(cmin))))
    den = b * a + C
    return 0. if den == 0 else (1 + b) * a * C / den

# qwen K=4 probe cost per gi
qi, qo = price("qwen/qwen3-235b-a22b-2507")
raw = collections.defaultdict(list)
for line in open("phase2/data/qwen_sc.jsonl"):
    r = json.loads(line); raw[r["gi"]].append((r["s"], r["in"], r["out"]))
qprobe = {}
for g, rows in raw.items():
    qprobe[g] = sum((i * qi + o * qo) / 1e6 for s, i, o in sorted(rows)[:4])

# candidate cost per gi (escalated only)
esc = json.load(open("/private/tmp/claude-502/-Users-nabaruns-work-cruq-ai-cruq-ai/063e8a4a-b7ad-4b1d-9cd2-2cffe81c3e8b/scratchpad/esc_gis.json"))
escset = set(esc)
candtok = collections.defaultdict(dict)
for line in open("phase2/data/escalation.jsonl"):
    r = json.loads(line); candtok[r["model"]][r["gi"]] = (r["in"], r["out"])

CAND = {"dspro": "deepseek/deepseek-v4-pro", "gempro": "google/gemini-2.5-pro", "sonnet": "anthropic/claude-sonnet-4.5"}
print(f"\n{'escalation model':30}{'v3-acc':>8}{'cost/1k':>10}{'arena-S':>9}")
# baseline: current router (deepseek escalation)
base = json.load(open("router_inference/predictions/cruq-sc-router.json"))
gis = [r["global index"] for r in base if r.get("accuracy") is not None]
for lbl, slug in CAND.items():
    d = json.load(open(f"router_inference/predictions/cruq-sc-{lbl}.json"))
    accs = {r["global index"]: r.get("accuracy") for r in d}
    scored = [g for g in gis if accs.get(g) is not None]
    N = len(scored)
    A = sum(accs[g] for g in scored) / N
    ci, co = price(slug)
    cost = 0.0
    for g in scored:
        cost += qprobe.get(g, 0.0)
        if g in escset and g in candtok[slug]:
            ti, to = candtok[slug][g]; cost += (ti * ci + to * co) / 1e6
    c1k = cost / N * 1000
    print(f"  {slug:28}{A:8.4f}{c1k:10.3f}{arena(A, c1k):9.4f}")
print(f"  {'deepseek-v4-flash (current)':28}{0.7590:8.4f}{0.192:10.3f}{arena(0.7590,0.192):9.4f}")
