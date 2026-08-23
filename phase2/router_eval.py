#!/usr/bin/env python3
"""End-to-end eval of the learned 5-model router on sub_10.

Loads the trained per-model P(correct) heads (predictor.pkl), embeds each sub_10
query with the same MiniLM encoder used at training, and applies the
cheapest-sufficient policy: pick the cheapest pool model whose predicted
P(correct) >= tau, else the argmax-P model. Scores the router's pick against that
model's cached sub_10 result (lightweight boxed-match), sweeps tau, and prints the
5-model oracle ceiling for comparison. Cache-only apart from the embedder; no API.
"""
import json
import pickle
import re
import sys

sys.path.insert(0, ".")
from datasets import load_from_disk

COST = json.load(open("model_cost/model_cost.json"))
POOL = [
    "qwen/qwen3-235b-a22b-2507",
    "Qwen/Qwen3-Coder-Next",
    "deepseek/deepseek-v4-flash",
    "google/gemini-2.5-flash-lite",
    "openai/gpt-4o-mini",
]
CACHE = {
    "qwen/qwen3-235b-a22b-2507": "cached_results/qwen_qwen3-235b-a22b-2507.jsonl",
    "Qwen/Qwen3-Coder-Next": "cached_results/Qwen_Qwen3-Coder-Next.jsonl",
    "deepseek/deepseek-v4-flash": "cached_results/deepseek_deepseek-v4-flash.jsonl",
    "google/gemini-2.5-flash-lite": "cached_results/google_gemini-2.5-flash-lite.jsonl",
    "openai/gpt-4o-mini": "cached_results/gpt-4o-mini.jsonl",
}
W = 0.5


def cost_pair(m):
    c = COST[m]
    return c["input_token_price_per_million"], c["output_token_price_per_million"]


def blended(m):
    i, o = cost_pair(m)
    return (1 - W) * i + W * o


ORDER = sorted(POOL, key=blended)  # cheapest -> priciest
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def extract(a):
    ms = BOXED.findall(a or "")
    return ms[-1].strip() if ms else None


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


ds = load_from_disk("./dataset/routerarena_10")
gold = {r["Global Index"]: str(r["Answer"]).strip() for r in ds}


def scorable(gi):
    return len(gold[gi]) <= 15


def correct(gi, pb):
    if pb is None:
        return False
    g, p = norm(gold[gi]), norm(pb)
    if not g:
        return False
    pl = p[0] if p else ""
    if g.isdigit() and 0 <= int(g) <= 25:
        if pl == chr(ord("a") + int(g)) or p == g:
            return True
    if len(g) == 1:
        return p == g or pl == g
    return p == g or g in p


# grade every pool model from cache
res = {m: {} for m in POOL}
for m in POOL:
    ci, co = cost_pair(m)
    for line in open(CACHE[m]):
        try:
            r = json.loads(line)
        except Exception:
            continue
        gi = r.get("global_index") or r.get("global index")
        if gi not in gold:
            continue
        tu = r.get("token_usage") or {}
        ti = tu.get("input_tokens", 0) or 0
        to = tu.get("output_tokens", 0) or 0
        res[m][gi] = (correct(gi, extract(r.get("generated_answer") or "")), (ti * ci + to * co) / 1e6)

gis = [gi for gi in gold if scorable(gi) and all(gi in res[m] for m in POOL)]
N = len(gis)
print(f"scorable queries with all 5 models cached: {N}")
print(f"pool cheapest->priciest: {[m.split('/')[-1] for m in ORDER]}")
print(f"  blended $/M: {[round(blended(m), 3) for m in ORDER]}\n")

print("=== per-model (lightweight boxed-match) ===")
for m in ORDER:
    acc = sum(res[m][gi][0] for gi in gis) / N
    cph = sum(res[m][gi][1] for gi in gis) / N * 1000
    print(f"  {m:34} acc={acc:.4f}  cost/1k=${cph:.3f}")


def oracle():
    a = c = 0
    for gi in gis:
        picks = [m for m in ORDER if res[m][gi][0]]
        if picks:
            a += 1
            c += res[picks[0]][gi][1]
    return a / N, c / N * 1000


oa, oc = oracle()
print(f"\n=== 5-model oracle (cheapest-correct) ===\n  acc={oa:.4f}  cost/1k=${oc:.3f}")

# ---- learned router ----
from sentence_transformers import SentenceTransformer  # noqa: E402

_bundle = pickle.load(open("phase2/data/predictor.pkl", "rb"))
heads = _bundle["heads"]  # {model: LogisticRegression}; bundle also has "embedder"
enc = SentenceTransformer(_bundle.get("embedder", "sentence-transformers/all-MiniLM-L6-v2"))
prompts = {r["global index"]: (r.get("prompt_formatted") or r.get("prompt"))
           for r in json.load(open("dataset/router_data_10.json"))}
emb = {gi: enc.encode([prompts[gi]])[0] for gi in gis}


def pcorrect(m, gi):
    h = heads.get(m)
    if h is None:
        return 0.0
    return float(h.predict_proba([emb[gi]])[0][1])


print("\n=== learned router (cheapest model with P>=tau, else argmax P) ===")
print(f"{'tau':>5} {'acc':>7} {'cost/1k':>9} {'opt_sel':>8}  routing(cheap..exp)")
for tau in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    a = c = opt = 0
    dist = {m: 0 for m in ORDER}
    for gi in gis:
        ps = {m: pcorrect(m, gi) for m in ORDER}
        pick = next((m for m in ORDER if ps[m] >= tau), max(ORDER, key=lambda m: ps[m]))
        dist[pick] += 1
        a += res[pick][gi][0]
        c += res[pick][gi][1]
        picks = [m for m in ORDER if res[m][gi][0]]
        if picks and pick == picks[0]:
            opt += 1
    routing = "/".join(str(dist[m]) for m in ORDER)
    print(f"{tau:5.2f} {a/N:7.4f} ${c/N*1000:8.3f} {opt/N:8.4f}  {routing}")
print(f"\noracle ceiling: acc={oa:.4f}  (base-3 oracle was 0.806)")
