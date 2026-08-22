#!/usr/bin/env python3
"""FREE cache-only analysis: does any already-run candidate model add oracle lift?

For every fully-cached model on the sub_10 split, compute lightweight boxed-match
accuracy, and the MARGINAL oracle lift each candidate adds when appended to the
3-model base pool. This answers 'is there a reachable complementary model' without
spending a cent, by mining the caches already on disk.
"""
import json
import re
import sys

sys.path.insert(0, ".")
from datasets import load_from_disk

COST = json.load(open("model_cost/model_cost.json"))

# model slug -> cached_results filename
CACHE = {
    "qwen/qwen3-235b-a22b-2507": "cached_results/qwen_qwen3-235b-a22b-2507.jsonl",
    "Qwen/Qwen3-Coder-Next": "cached_results/Qwen_Qwen3-Coder-Next.jsonl",
    "deepseek/deepseek-v4-flash": "cached_results/deepseek_deepseek-v4-flash.jsonl",
    "z-ai/glm-4.7": "cached_results/z-ai_glm-4.7.jsonl",
    "qwen/qwen3-next-80b-a3b-instruct": "cached_results/qwen_qwen3-next-80b-a3b-instruct.jsonl",
    "claude-3-haiku-20240307": "cached_results/claude-3-haiku-20240307.jsonl",
    "gemini-2.0-flash-001": "cached_results/gemini-2.0-flash-001.jsonl",
    "gpt-4o-mini": "cached_results/gpt-4o-mini.jsonl",
}
BASE = ["qwen/qwen3-235b-a22b-2507", "Qwen/Qwen3-Coder-Next", "deepseek/deepseek-v4-flash"]

W = 0.5


def cost_pair(m):
    c = COST.get(m, {"input_token_price_per_million": 1.0, "output_token_price_per_million": 2.0})
    return c["input_token_price_per_million"], c["output_token_price_per_million"]


def blended(m):
    i, o = cost_pair(m)
    return (1 - W) * i + W * o


BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def extract(ans):
    if not ans:
        return None
    ms = BOXED.findall(ans)
    return ms[-1].strip() if ms else None


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


ds = load_from_disk("./dataset/routerarena_10")
gold = {}
for r in ds:
    gold[r["Global Index"]] = str(r["Answer"]).strip()


def scorable(gi):
    return len(gold[gi]) <= 15


def correct(gi, pb):
    if pb is None:
        return False
    g = norm(gold[gi])
    p = norm(pb)
    if not g:
        return False
    pl = p[0] if p else ""
    if g.isdigit() and 0 <= int(g) <= 25:
        if pl == chr(ord("a") + int(g)):
            return True
        if p == g:
            return True
    if len(g) == 1:
        return p == g or pl == g
    return p == g or g in p


# grade every model
res = {}       # m -> gi -> (correct, cost_usd)
for m, path in CACHE.items():
    res[m] = {}
    ci, co = cost_pair(m)
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        gi = r.get("global_index") or r.get("global index")
        if gi not in gold:
            continue
        ans = r.get("generated_answer") or ""
        tu = r.get("token_usage") or {}
        ti = tu.get("input_tokens", 0) or 0
        to = tu.get("output_tokens", 0) or 0
        res[m][gi] = (correct(gi, extract(ans)), (ti * ci + to * co) / 1e6)

# common scorable set present in EVERY cached model
gis = [gi for gi in gold if scorable(gi) and all(gi in res[m] for m in CACHE)]
N = len(gis)
print(f"common scorable queries across all {len(CACHE)} models: {N}\n")

print("=== per-model accuracy (lightweight boxed-match) ===")
for m in sorted(CACHE, key=blended):
    acc = sum(res[m][gi][0] for gi in gis) / N
    cph = sum(res[m][gi][1] for gi in gis) / N * 1000
    print(f"  {m:36} acc={acc:.4f}  cost/1k=${cph:.3f}  blended$/M={blended(m):.3f}")


def oracle(models):
    """cheapest-correct oracle over `models` (order by blended cost)."""
    order = sorted(models, key=blended)
    acc = cost = 0
    for gi in gis:
        picks = [m for m in order if res[m][gi][0]]
        if picks:
            acc += 1
            cost += res[picks[0]][gi][1]
    return acc / N, cost / N * 1000


ba, bc = oracle(BASE)
print(f"\n=== 3-model BASE oracle ===\n  acc={ba:.4f}  cost/1k=${bc:.3f}")

print("\n=== marginal oracle lift: BASE + one candidate ===")
cands = [m for m in CACHE if m not in BASE]
rows = []
for m in cands:
    a, c = oracle(BASE + [m])
    rows.append((a - ba, a, c, m))
for lift, a, c, m in sorted(rows, reverse=True):
    print(f"  +{m:34} oracle_acc={a:.4f}  (+{lift*100:.2f} pts)  cost/1k=${c:.3f}")

# best full-pool oracle
allm = list(CACHE)
fa, fc = oracle(allm)
print(f"\n=== oracle over ALL {len(allm)} cached models ===\n  acc={fa:.4f}  cost/1k=${fc:.3f}")

# how many BASE-unsolved queries does each candidate rescue?
base_wrong = [gi for gi in gis if not any(res[m][gi][0] for m in BASE)]
print(f"\n=== queries ALL 3 base models fail: {len(base_wrong)} of {N} ===")
for m in sorted(cands, key=blended):
    rescued = sum(res[m][gi][0] for gi in base_wrong)
    print(f"  {m:36} rescues {rescued:3d}/{len(base_wrong)}  ({rescued/max(len(base_wrong),1)*100:.1f}%)")
