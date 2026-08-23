#!/usr/bin/env python3
"""Aggregate OFFICIAL per-model scores (from the 5 scored prediction files) into the
true per-dataset best model, oracle, and domain-routing ceiling, on the real metric
(including code/translation/QANTA the lightweight grader could not score)."""
import json, collections, math
from datasets import load_from_disk

COST = json.load(open("model_cost/model_cost.json"))
MODELS = {  # label -> (prediction file, cost slug)
    "qwen3-235b": ("cruq-single-qwen", "qwen/qwen3-235b-a22b-2507"),
    "coder-next": ("cruq-single-coder", "Qwen/Qwen3-Coder-Next"),
    "deepseek": ("cruq-single-deepseek", "deepseek/deepseek-v4-flash"),
    "gpt-4o-mini": ("cruq-single-gpt4omini", "openai/gpt-4o-mini"),
    "gemini-flash-lite": ("cruq-gemini25fl", "google/gemini-2.5-flash-lite"),
}
def blended(slug):
    c = COST[slug]; return .5*c["input_token_price_per_million"] + .5*c["output_token_price_per_million"]
ORDER = sorted(MODELS, key=lambda m: blended(MODELS[m][1]))

ds = load_from_disk("./dataset/routerarena_10")
dn = {r["Global Index"]: (r.get("Dataset name") or r["Global Index"].split("_")[0]) for r in ds}

# per model: gi -> (accuracy, cost)
acc = {}
for lab, (fn, slug) in MODELS.items():
    d = json.load(open(f"router_inference/predictions/{fn}.json"))
    acc[lab] = {r.get("global index"): (r.get("accuracy"), r.get("cost")) for r in d
                if r.get("accuracy") is not None}

# common set scored by all 5
gis = set(dn)
for lab in MODELS: gis &= set(acc[lab].keys())
gis = sorted(gis)
N = len(gis)

def mean_acc(lab, items): return sum(acc[lab][g][0] for g in items) / len(items)
def mean_cost1k(lab, items):
    cs = [acc[lab][g][1] for g in items if acc[lab][g][1] is not None]
    return (sum(cs)/len(cs))*1000 if cs else 0.0
def arena(a, c1k, beta=.1, cmin=.0044, cmax=200.):
    if c1k <= 0: return 0.
    C = max(0., min(1., (math.log2(cmax)-math.log2(c1k))/(math.log2(cmax)-math.log2(cmin))))
    den = beta*a + C
    return 0. if den==0 else (1+beta)*a*C/den

print(f"OFFICIAL scores on {N} commonly-scored sub_10 items\n")
print(f"{'model':20}{'acc':>8}{'cost/1k':>10}{'arena-S':>9}")
for m in ORDER:
    a, c = mean_acc(m, gis), mean_cost1k(m, gis)
    print(f"  {m:18}{a:8.4f}{c:10.4f}{arena(a,c):9.4f}")

# per-query oracle (cheapest-correct)
oa = oc = 0
for g in gis:
    picks = [m for m in ORDER if acc[m][g][0] and acc[m][g][0] > 0.5]
    if picks:
        oa += 1; oc += acc[picks[0]][g][1] or 0
print(f"\n  {'per-query ORACLE':18}{oa/N:8.4f}{oc/N*1000:10.4f}{arena(oa/N, oc/N*1000):9.4f}")

# domain-routing ceiling (best single model per dataset)
byds = collections.defaultdict(list)
for g in gis: byds[dn[g]].append(g)
dom_correct = dom_cost = 0
best_by_ds = {}
for dsn, items in byds.items():
    accs = {m: mean_acc(m, items) for m in ORDER}
    bm = max(ORDER, key=lambda m: accs[m])
    best_by_ds[dsn] = (bm, accs[bm])
    dom_correct += sum(acc[bm][g][0] for g in items)
    dom_cost += sum((acc[bm][g][1] or 0) for g in items)
print(f"  {'DOMAIN ceiling':18}{dom_correct/N:8.4f}{dom_cost/N*1000:10.4f}{arena(dom_correct/N, dom_cost/N*1000):9.4f}")

# where does each model uniquely win a whole dataset (n>=7)?
print("\n=== per-dataset best model (official, n>=7) ===")
for dsn in sorted(byds, key=lambda d: -len(byds[d])):
    if len(byds[dsn]) >= 7:
        bm, ba = best_by_ds[dsn]
        print(f"  {dsn:34} n={len(byds[dsn]):3} best={bm:16} acc={ba:.3f}")
