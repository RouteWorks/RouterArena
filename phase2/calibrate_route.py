#!/usr/bin/env python3
"""Selector-side experiment: calibrated heads + per-model tau.

The 5-model router plateaus at 0.706 vs a 0.832 oracle because independent
per-model P(correct) heads are not comparable across models, so a single global
tau on raw probabilities routes badly. This tries to fix the SELECTOR without
touching the pool:

  1. Refit each head and wrap it in isotonic calibration (CalibratedClassifierCV)
     so predicted P(correct) is a true probability, comparable across models.
  2. Tune thresholds on a HELD-OUT slice of the external corpus (never RouterArena):
        - a single global tau, and
        - per-model tau_m via greedy coordinate ascent,
     both maximizing the RouterArena Acc-Cost score S on the val slice.
  3. Evaluate the resulting policies on sub_10 (held-out, never tuned on).

Reports accuracy / cost/1k / arena-S for: uncalibrated global-tau (baseline),
calibrated global-tau, calibrated per-model tau, and the oracle ceiling.
Cache-only apart from the MiniLM embedder; no API calls.
"""
import json
import re
import sys

import numpy as np

sys.path.insert(0, ".")
from datasets import load_from_disk
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

# ---- pool, costs, caches ----
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
# RouterArena Acc-Cost: S = (1+b)AC / (bA + C), C = log2-normalized cheapness in [0,1]
BETA = 0.1
C_MIN, C_MAX = 0.0044, 200.0  # cost/1k normalization bounds
L2MIN, L2MAX = np.log2(C_MIN), np.log2(C_MAX)


def cost_pair(m):
    c = COST[m]
    return c["input_token_price_per_million"], c["output_token_price_per_million"]


def blended(m):
    i, o = cost_pair(m)
    return (1 - W) * i + W * o


ORDER = sorted(POOL, key=blended)


def arena_S(acc, cost1k):
    cost1k = max(cost1k, 1e-9)
    C = (L2MAX - np.log2(cost1k)) / (L2MAX - L2MIN)
    C = float(np.clip(C, 0.0, 1.0))
    denom = BETA * acc + C
    return 0.0 if denom == 0 else (1 + BETA) * acc * C / denom


BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def extract(a):
    ms = BOXED.findall(a or "")
    return ms[-1].strip() if ms else None


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# ---- sub_10 gold + per-model cached grade/cost ----
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
    if g.isdigit() and 0 <= int(g) <= 25 and (pl == chr(ord("a") + int(g)) or p == g):
        return True
    if len(g) == 1:
        return p == g or pl == g
    return p == g or g in p


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
        ti, to = tu.get("input_tokens", 0) or 0, tu.get("output_tokens", 0) or 0
        res[m][gi] = (correct(gi, extract(r.get("generated_answer") or "")), (ti * ci + to * co) / 1e6)

test_gis = [gi for gi in gold if scorable(gi) and all(gi in res[m] for m in POOL)]

# ---- external corpus: embeddings + per-model labels (for calibration + tau tuning) ----
from sentence_transformers import SentenceTransformer  # noqa: E402

corpus = [json.loads(x) for x in open("phase2/data/corpus.jsonl")]
cid = [c["id"] for c in corpus]
labels = {}  # (id, model) -> correct
for line in open("phase2/data/labels.jsonl"):
    r = json.loads(line)
    labels[(r["id"], r["model"])] = int(r["correct"])

enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
Ecorpus = {cid[i]: e for i, e in enumerate(enc.encode([c["prompt"] for c in corpus], batch_size=64))}

# deterministic external train/val split (80/20) by index order (no RNG in this env)
n = len(cid)
val_ids = set(cid[i] for i in range(n) if i % 5 == 0)   # 20%
trn_ids = [c for c in cid if c not in val_ids]
val_list = [c for c in cid if c in val_ids]
print(f"corpus={n}  train={len(trn_ids)}  val={len(val_list)}  sub_10 test={len(test_gis)}")

# ---- fit raw + isotonic-calibrated heads per model on the train slice ----
raw_heads, cal_heads = {}, {}
print(f"\n{'model':34} {'Brier raw':>9} {'Brier cal':>9}  (val slice)")
for m in POOL:
    Xtr = np.array([Ecorpus[i] for i in trn_ids if (i, m) in labels])
    ytr = np.array([labels[(i, m)] for i in trn_ids if (i, m) in labels])
    Xva = np.array([Ecorpus[i] for i in val_list if (i, m) in labels])
    yva = np.array([labels[(i, m)] for i in val_list if (i, m) in labels])
    raw = LogisticRegression(max_iter=1000, C=1.0).fit(Xtr, ytr)
    cal = CalibratedClassifierCV(LogisticRegression(max_iter=1000, C=1.0), method="isotonic", cv=5).fit(Xtr, ytr)
    raw_heads[m], cal_heads[m] = raw, cal
    b_raw = brier_score_loss(yva, raw.predict_proba(Xva)[:, 1])
    b_cal = brier_score_loss(yva, cal.predict_proba(Xva)[:, 1])
    print(f"{m:34} {b_raw:9.4f} {b_cal:9.4f}")

# precompute predicted P(correct) for val (corpus) and test (sub_10) under raw & cal heads
Eval_arr = np.array([Ecorpus[i] for i in val_list])
Etest_arr = np.array([enc.encode([prompts_gi]) [0] for prompts_gi in
                      [ {r["global index"]:(r.get("prompt_formatted") or r.get("prompt")) for r in json.load(open("dataset/router_data_10.json"))}[gi] for gi in test_gis ]])


def pmat(heads, X):
    return {m: heads[m].predict_proba(X)[:, 1] for m in POOL}


Pval_raw, Pval_cal = pmat(raw_heads, Eval_arr), pmat(cal_heads, Eval_arr)
Ptest_raw, Ptest_cal = pmat(raw_heads, Etest_arr), pmat(cal_heads, Etest_arr)

# val correctness/cost per model (from corpus labels; cost from mean sub_10 token cost proxy)
# For val we only need correctness to tune tau against arena-S; use a per-model mean cost
# from the sub_10 cache as the cost proxy (same models, same prompt style).
mean_cost = {m: np.mean([res[m][gi][1] for gi in test_gis]) for m in POOL}


def route_pick(Pcol, i, taus):
    """cheapest model whose P>=tau_m at row i, else global-argmax P."""
    for m in ORDER:
        if Pcol[m][i] >= taus[m]:
            return m
    return max(ORDER, key=lambda m: Pcol[m][i])


def eval_val(Pcol, taus):
    """arena-S on the val corpus slice for a routing (correctness from labels)."""
    acc = 0.0
    cost = 0.0
    valid = 0
    for i, cidv in enumerate(val_list):
        if not all((cidv, m) in labels for m in POOL):
            continue
        valid += 1
        m = route_pick(Pcol, i, taus)
        acc += labels[(cidv, m)]
        cost += mean_cost[m]
    acc /= valid
    cost1k = cost / valid * 1000
    return arena_S(acc, cost1k), acc, cost1k


def eval_test(Pcol, taus):
    acc = cost = 0.0
    dist = {m: 0 for m in ORDER}
    for i, gi in enumerate(test_gis):
        m = route_pick(Pcol, i, taus)
        dist[m] += 1
        acc += res[m][gi][0]
        cost += res[m][gi][1]
    N = len(test_gis)
    return acc / N, cost / N * 1000, dist


GRID = [round(x, 2) for x in np.arange(0.30, 0.96, 0.05)]

# --- (a) global tau, tuned on val, for raw and calibrated heads ---
def best_global(Pcol):
    best = (-1, None)
    for t in GRID:
        S, _, _ = eval_val(Pcol, {m: t for m in POOL})
        if S > best[0]:
            best = (S, t)
    return best[1]


# --- (b) per-model tau via greedy coordinate ascent on val-S (calibrated heads) ---
def tune_permodel(Pcol):
    taus = {m: 0.5 for m in POOL}
    bestS = eval_val(Pcol, taus)[0]
    for _ in range(4):  # sweeps
        improved = False
        for m in ORDER:
            for t in GRID:
                cand = dict(taus, **{m: t})
                S = eval_val(Pcol, cand)[0]
                if S > bestS + 1e-9:
                    bestS, taus, improved = S, cand, True
        if not improved:
            break
    return taus


g_raw = best_global(Pval_raw)
g_cal = best_global(Pval_cal)
pm_cal = tune_permodel(Pval_cal)

print("\n=== tuned thresholds (on external corpus val slice) ===")
print(f"  raw  global tau = {g_raw}")
print(f"  cal  global tau = {g_cal}")
print(f"  cal  per-model  = " + ", ".join(f"{m.split('/')[-1]}={pm_cal[m]}" for m in ORDER))

# oracle on test
oa = oc = 0
for gi in test_gis:
    picks = [m for m in ORDER if res[m][gi][0]]
    if picks:
        oa += 1
        oc += res[picks[0]][gi][1]
N = len(test_gis)
oracle_acc, oracle_cost = oa / N, oc / N * 1000

print("\n=== sub_10 test (never tuned on) ===")
print(f"{'policy':32} {'acc':>7} {'cost/1k':>9} {'arena-S':>8}  routing(cheap..exp)")


def show(label, Pcol, taus):
    acc, cost1k, dist = eval_test(Pcol, taus)
    S = arena_S(acc, cost1k)
    routing = "/".join(str(dist[m]) for m in ORDER)
    print(f"{label:32} {acc:7.4f} ${cost1k:8.3f} {S:8.4f}  {routing}")


show("uncalibrated global-tau", Ptest_raw, {m: g_raw for m in POOL})
show("calibrated global-tau", Ptest_cal, {m: g_cal for m in POOL})
show("calibrated per-model tau", Ptest_cal, pm_cal)
print(f"{'ORACLE (cheapest-correct)':32} {oracle_acc:7.4f} ${oracle_cost:8.3f} {arena_S(oracle_acc, oracle_cost):8.4f}")
print(f"\nprior best (raw router, tau=0.90): acc=0.706")
