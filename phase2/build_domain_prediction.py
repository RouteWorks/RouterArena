#!/usr/bin/env python3
"""Build a RouterArena prediction file for the domain-aware router.

Trains the query->domain classifier + cost-aware per-domain best-model table on the
external corpus (never sub_10), routes each sub_10 query to its predicted domain's
model, and emits a prediction file whose generated_result is that selected model's
cached output. Score the result with the official evaluator (deploy/routerarena-eval,
v3). Cache-only apart from the MiniLM embedder.

Out: router_inference/predictions/cruq-domain-router.json
"""
import json
import collections
import numpy as np
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

COST = json.load(open("model_cost/model_cost.json"))
# label -> (single-model prediction file, cost slug)
MODELS = {
    "qwen3-235b": ("cruq-single-qwen", "qwen/qwen3-235b-a22b-2507"),
    "coder-next": ("cruq-single-coder", "Qwen/Qwen3-Coder-Next"),
    "deepseek": ("cruq-single-deepseek", "deepseek/deepseek-v4-flash"),
    "gpt-4o-mini": ("cruq-single-gpt4omini", "openai/gpt-4o-mini"),
    "gemini-flash-lite": ("cruq-gemini25fl", "google/gemini-2.5-flash-lite"),
}
def blended(slug):
    c = COST[slug]; return .5*c["input_token_price_per_million"] + .5*c["output_token_price_per_million"]

# ---- train domain classifier + cost-aware table on external corpus ----
corpus = [json.loads(x) for x in open("phase2/data/corpus.jsonl")]
lab = collections.defaultdict(dict)
for line in open("phase2/data/labels.jsonl"):
    r = json.loads(line); lab[r["model"]][r["id"]] = r["correct"]
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
Ecorp = enc.encode([c["prompt"] for c in corpus], batch_size=64)
domains = [c["domain"] for c in corpus]
clf = LogisticRegression(max_iter=2000, C=2.0).fit(Ecorp, domains)

by_dom = collections.defaultdict(list)
for c in corpus:
    by_dom[c["domain"]].append(c["id"])
# slug used in labels for each model label
LABEL_SLUG = {"qwen3-235b": "qwen/qwen3-235b-a22b-2507", "coder-next": "Qwen/Qwen3-Coder-Next",
              "deepseek": "deepseek/deepseek-v4-flash", "gpt-4o-mini": "openai/gpt-4o-mini",
              "gemini-flash-lite": "google/gemini-2.5-flash-lite"}
import os
MODE = os.getenv("ROUTE_MODE", "costaware")  # costaware | accmax
MARGIN = 0.05
table = {}  # domain -> model label
for dom, ids in by_dom.items():
    acc = {}
    for m, slug in LABEL_SLUG.items():
        v = [lab[slug][i] for i in ids if i in lab[slug]]
        acc[m] = sum(v)/len(v) if v else 0.0
    best = max(acc, key=acc.get)
    if MODE == "accmax":
        table[dom] = best
    else:  # cost-aware: cheapest within MARGIN of best acc
        within = [m for m in MODELS if acc[m] >= acc[best] - MARGIN]
        table[dom] = min(within, key=lambda m: blended(LABEL_SLUG[m]))
print(f"[{MODE}] domain->model table:")
for d in sorted(table): print(f"  {d:26} -> {table[d]}")

# ---- load the 5 single-model prediction files (aligned, 809 rows) ----
preds = {m: json.load(open(f"router_inference/predictions/{fn}.json")) for m, (fn, _) in MODELS.items()}
ref = preds["deepseek"]
prompts = [r["prompt"] for r in ref]
Etest = enc.encode(prompts, batch_size=64)
pred_dom = clf.predict(Etest)

DEFAULT = "deepseek"
out = []
route_dist = collections.Counter()
for i, base in enumerate(ref):
    m = table.get(pred_dom[i], DEFAULT)
    route_dist[m] += 1
    src = preds[m][i]  # same index == same global index (verified aligned)
    row = dict(base)
    row["prediction"] = src["prediction"]
    row["generated_result"] = src["generated_result"]
    row["accuracy"] = None
    row["cost"] = None
    out.append(row)

outname = "cruq-domain-router" if MODE == "costaware" else "cruq-domain-accmax"
json.dump(out, open(f"router_inference/predictions/{outname}.json", "w"))
print(f"\nwrote {len(out)} rows -> {outname}.json")
print("routing:", {m: route_dist[m] for m in MODELS})
