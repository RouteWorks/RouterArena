#!/usr/bin/env python3
"""Low-memory lightweight grader for cruq pool experiments.

Streams each model's cached_results jsonl (no pandas), extracts the \\boxed{X}
answer, compares to the gold answer from the sub_10 arrow, and reports per-model
accuracy, the oracle (cheapest-correct) ceiling, and the CruqRouter's simulated
pick. This is an APPROXIMATE grade over letter/short-answer datasets, not the full
RouterArena harness (which also handles code/translation scorers).
"""
import json
import re
import sys

sys.path.insert(0, ".")
from datasets import load_from_disk  # noqa: E402
from router_inference.router.cruq_router import _difficulty, _load_model_costs  # noqa: E402

POOL = [
    "qwen/qwen3-235b-a22b-2507",
    "Qwen/Qwen3-Coder-Next",
    "deepseek/deepseek-v4-flash",
]
CACHE = {
    "qwen/qwen3-235b-a22b-2507": "cached_results/qwen_qwen3-235b-a22b-2507.jsonl",
    "Qwen/Qwen3-Coder-Next": "cached_results/Qwen_Qwen3-Coder-Next.jsonl",
    "deepseek/deepseek-v4-flash": "cached_results/deepseek_deepseek-v4-flash.jsonl",
}

costs = _load_model_costs()
w = 0.5


def blended(m):
    i, o = costs.get(m, (1.0, 2.0))
    return (1 - w) * i + w * o


ordered = sorted(POOL, key=blended)  # cheapest -> priciest

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def extract(ans):
    if not ans:
        return None
    ms = BOXED.findall(ans)
    if ms:
        return ms[-1].strip()
    return None


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# gold + difficulty class from the arrow (809 rows, tiny)
ds = load_from_disk("./dataset/routerarena_10")
gold = {}
for r in ds:
    gi = r["Global Index"]
    gold[gi] = {
        "answer": r["Answer"],
        "difficulty": (r.get("Difficulty") or "").lower(),
        "dataset": r.get("Dataset name") or gi.split("_")[0],
    }


def scorable(gi):
    a = str(gold[gi]["answer"]).strip()
    # letter MCQ, or short (<=15 char) numeric/token answer -> lightweight-scorable
    return len(a) <= 15


def correct(gi, pred_boxed):
    if pred_boxed is None:
        return False
    g = norm(gold[gi]["answer"])
    p = norm(pred_boxed)
    if not g:
        return False
    # Some MCQ datasets store the gold as a numeric option INDEX (0=A,1=B,...)
    # while the model answers with the letter. Reconcile both directions.
    pl = p[0] if p else ""
    if g.isdigit() and 0 <= int(g) <= 25:
        letter = chr(ord("a") + int(g))
        if pl == letter:  # gold index vs model letter
            return True
        if p == g:  # both numeric (e.g. GSM8K)
            return True
    if len(g) == 1:  # gold is a single letter/char
        return p == g or pl == g
    return p == g or g in p


# stream each model -> per (gi) correctness + cost
res = {m: {} for m in POOL}   # m -> gi -> (correct_bool, cost_usd)
boxed_hit = {m: 0 for m in POOL}
for m in POOL:
    ci, co = costs.get(m, (1.0, 2.0))
    for line in open(CACHE[m]):
        try:
            r = json.loads(line)
        except Exception:
            continue
        gi = r.get("global_index") or r.get("global index")
        if gi not in gold:
            continue
        ans = r.get("generated_answer") or ""
        pb = extract(ans)
        if pb is not None:
            boxed_hit[m] += 1
        tu = r.get("token_usage") or {}
        ti = tu.get("input_tokens", 0) or 0
        to = tu.get("output_tokens", 0) or 0
        cost = (ti * ci + to * co) / 1e6
        res[m][gi] = (correct(gi, pb), cost)

gis = [gi for gi in gold if scorable(gi) and all(gi in res[m] for m in POOL)]
N = len(gis)
print(f"scorable queries with all 3 models: {N} / {len(gold)}")
print(f"pool cheapest->priciest: {ordered}  (blended $/M: {[round(blended(m),3) for m in ordered]})")
print()

print("=== per-model (lightweight boxed-match) ===")
for m in ordered:
    acc = sum(res[m][gi][0] for gi in gis) / N
    cph = sum(res[m][gi][1] for gi in gis) / N * 1000
    print(f"  {m:34} acc={acc:.4f}  cost/1k=${cph:.3f}  boxed_hit={boxed_hit[m]}/809")

# oracle: cheapest model that is correct
orc_acc = orc_cost = 0
for gi in gis:
    picks = [m for m in ordered if res[m][gi][0]]
    if picks:
        orc_acc += 1
        orc_cost += res[picks[0]][gi][1]
print(f"\n=== oracle (cheapest-correct) ===\n  acc={orc_acc/N:.4f}  cost/1k=${orc_cost/N*1000:.3f}")

# router simulation
prompts = {r["global index"]: (r.get("prompt_formatted") or r.get("prompt"))
           for r in json.load(open("dataset/router_data_10.json"))}
thr = [i / len(ordered) for i in range(1, len(ordered))]


def route(gi):
    d = _difficulty(prompts.get(gi, ""))
    idx = 0
    for t in thr:
        if d >= t:
            idx += 1
    return ordered[min(idx, len(ordered) - 1)]


r_acc = r_cost = 0
dist = {m: 0 for m in ordered}
opt_sel = 0
for gi in gis:
    m = route(gi)
    dist[m] += 1
    r_acc += res[m][gi][0]
    r_cost += res[m][gi][1]
    # optimal selection: did the router pick the cheapest correct model (if any correct)?
    picks = [x for x in ordered if res[x][gi][0]]
    if picks and m == picks[0]:
        opt_sel += 1
print("\n=== CruqRouter (difficulty-tiered) ===")
print(f"  acc={r_acc/N:.4f}  cost/1k=${r_cost/N*1000:.3f}  opt_sel={opt_sel/N:.4f}")
print(f"  routing: " + ", ".join(f"{m.split('/')[-1]}={dist[m]}" for m in ordered))
