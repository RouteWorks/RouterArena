#!/usr/bin/env python3
"""Phase 3: domain-aware router — train on external corpus, evaluate on sub_10.

Complementarity in the pool is domain-structured (a code model wins code, a math
model wins math, cheap qwen wins knowledge MCQ), and domain is far more
predictable from the query than per-query correctness. So instead of five
per-model P(correct) heads, this:

  1. trains a DOMAIN classifier (query embedding -> domain) on the external corpus,
  2. builds a per-domain best-model table from the corpus labels (acc-max, and a
     cost-aware variant: cheapest model within a margin of the best),
  3. routes each sub_10 query by its predicted domain and grades from cache.

Everything is learned on the external corpus (never sub_10). Reports the domain
classifier's CV accuracy, the routing table, and sub_10 accuracy / cost / arena-S
against the oracle and the prior per-model-P router (0.706 / 0.717).
Cache-only apart from the MiniLM embedder.
"""
import json
import re
import collections
import numpy as np

from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sentence_transformers import SentenceTransformer

COST = json.load(open("model_cost/model_cost.json"))
POOL = ["qwen/qwen3-235b-a22b-2507", "Qwen/Qwen3-Coder-Next", "deepseek/deepseek-v4-flash",
        "google/gemini-2.5-flash-lite", "openai/gpt-4o-mini"]
CACHE = {"qwen/qwen3-235b-a22b-2507": "cached_results/qwen_qwen3-235b-a22b-2507.jsonl",
         "Qwen/Qwen3-Coder-Next": "cached_results/Qwen_Qwen3-Coder-Next.jsonl",
         "deepseek/deepseek-v4-flash": "cached_results/deepseek_deepseek-v4-flash.jsonl",
         "google/gemini-2.5-flash-lite": "cached_results/google_gemini-2.5-flash-lite.jsonl",
         "openai/gpt-4o-mini": "cached_results/gpt-4o-mini.jsonl"}


def cp(m):
    c = COST[m]
    return c["input_token_price_per_million"], c["output_token_price_per_million"]


def blended(m):
    i, o = cp(m)
    return .5 * i + .5 * o


ORDER = sorted(POOL, key=blended)
BETA = .1
L2MIN, L2MAX = np.log2(.0044), np.log2(200.)


def arena_S(a, c1k):
    c1k = max(c1k, 1e-9)
    C = float(np.clip((L2MAX - np.log2(c1k)) / (L2MAX - L2MIN), 0, 1))
    d = BETA * a + C
    return 0. if d == 0 else (1 + BETA) * a * C / d


BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def extract(a):
    ms = BOXED.findall(a or "")
    return ms[-1].strip() if ms else None


def nm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def correct_ans(gold, pb):
    if pb is None:
        return False
    g, p = nm(gold), nm(pb)
    if not g:
        return False
    pl = p[0] if p else ""
    if g.isdigit() and 0 <= int(g) <= 25 and (pl == chr(ord("a") + int(g)) or p == g):
        return True
    if len(g) == 1:
        return p == g or pl == g
    return p == g or g in p


# ---- external corpus: embeddings, domains, per-model correctness ----
corpus = [json.loads(x) for x in open("phase2/data/corpus.jsonl")]
lab = collections.defaultdict(dict)
for line in open("phase2/data/labels.jsonl"):
    r = json.loads(line)
    lab[r["model"]][r["id"]] = r["correct"]

enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
Ecorp = enc.encode([c["prompt"] for c in corpus], batch_size=64)
domains = [c["domain"] for c in corpus]

# domain classifier CV accuracy
clf_dom = LogisticRegression(max_iter=2000, C=2.0)
cv = cross_val_score(clf_dom, Ecorp, domains, cv=5)
print(f"domain classifier CV accuracy: {cv.mean():.3f} (+/-{cv.std():.3f}) over {len(set(domains))} domains")
clf_dom.fit(Ecorp, domains)

# per-domain best-model table from corpus labels
by_dom = collections.defaultdict(list)
for i, c in enumerate(corpus):
    by_dom[c["domain"]].append(c["id"])


def dom_acc(dom):
    ids = by_dom[dom]
    return {m: np.mean([lab[m][i] for i in ids if i in lab[m]]) if any(i in lab[m] for i in ids) else 0.0
            for m in POOL}


table_acc, table_cost, table_robust = {}, {}, {}   # acc-max, cost-aware, margin-gated
MARGIN = 0.05
DEFAULT = "deepseek/deepseek-v4-flash"     # strong default; override only on real evidence
OVR = 0.08                                 # a model must beat the default by this on corpus
for dom in by_dom:
    a = dom_acc(dom)
    best = max(a, key=a.get)
    table_acc[dom] = best
    within = [m for m in POOL if a[m] >= a[best] - MARGIN]
    table_cost[dom] = min(within, key=blended)
    # margin-gated: keep default unless a cheaper-or-equal model clearly wins the domain.
    # candidates that beat the default by >= OVR; among them take the cheapest.
    cands = [m for m in POOL if a[m] >= a[DEFAULT] + OVR]
    table_robust[dom] = min(cands, key=blended) if cands else DEFAULT

print("\n=== per-domain routing table (corpus) ===")
print(f"{'domain':26}{'acc-max model':>22}{'cost-aware model':>22}")
for dom in sorted(by_dom):
    print(f"  {dom:24}{table_acc[dom].split('/')[-1]:>22}{table_cost[dom].split('/')[-1]:>22}")

# ---- sub_10 test set: gold, cached grade+cost, prompts ----
ds = load_from_disk("./dataset/routerarena_10")
gold = {r["Global Index"]: str(r["Answer"]).strip() for r in ds}
dsname = {r["Global Index"]: (r.get("Dataset name") or r["Global Index"].split("_")[0]) for r in ds}


def scorable(gi):
    return len(gold[gi]) <= 15


res = {m: {} for m in POOL}
for m in POOL:
    ci, co = cp(m)
    for line in open(CACHE[m]):
        try:
            r = json.loads(line)
        except Exception:
            continue
        gi = r.get("global_index") or r.get("global index")
        if gi not in gold:
            continue
        tu = r.get("token_usage") or {}
        ti, to = tu.get("input_tokens", 0) or 0, tu.get("output_tokens", 0) or 0
        res[m][gi] = (correct_ans(gold[gi], extract(r.get("generated_answer") or "")), (ti * ci + to * co) / 1e6)

gis = [gi for gi in gold if scorable(gi) and all(gi in res[m] for m in POOL)]
N = len(gis)
prompts = {r["global index"]: (r.get("prompt_formatted") or r.get("prompt"))
           for r in json.load(open("dataset/router_data_10.json"))}
Etest = enc.encode([prompts[gi] for gi in gis], batch_size=64)
pred_dom = clf_dom.predict(Etest)


def eval_router(pick):
    a = c = 0
    dist = collections.Counter()
    for i, gi in enumerate(gis):
        m = pick(i, gi)
        dist[m] += 1
        a += res[m][gi][0]
        c += res[m][gi][1]
    return a / N, c / N * 1000, dist


# oracle + single-model baselines
oa = oc = 0
for gi in gis:
    pk = [m for m in ORDER if res[m][gi][0]]
    if pk:
        oa += 1
        oc += res[pk[0]][gi][1]
print(f"\nsub_10 scorable N={N}")
print(f"{'policy':34}{'acc':>8}{'cost/1k':>10}{'arena-S':>9}")
for m in ORDER:
    aa = sum(res[m][gi][0] for gi in gis) / N
    cc = sum(res[m][gi][1] for gi in gis) / N * 1000
    print(f"  {'single: ' + m.split('/')[-1]:32}{aa:8.4f}{cc:10.3f}{arena_S(aa, cc):9.4f}")

for label, tbl in [("domain-router (acc-max)", table_acc), ("domain-router (cost-aware)", table_cost),
                   ("domain-router (margin-gated)", table_robust)]:
    a, c, dist = eval_router(lambda i, gi, t=tbl: t.get(pred_dom[i], "deepseek/deepseek-v4-flash"))
    routing = " ".join(f"{m.split('/')[-1]}={dist[m]}" for m in ORDER if dist[m])
    print(f"  {label:32}{a:8.4f}{c:10.3f}{arena_S(a, c):9.4f}   {routing}")

print(f"  {'ORACLE (cheapest-correct)':32}{oa / N:8.4f}{oc / N * 1000:10.3f}{arena_S(oa / N, oc / N * 1000):9.4f}")
print(f"\nprior per-model-P routers: raw tau=0.706, calibrated per-model tau=0.717")
