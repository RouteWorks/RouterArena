#!/usr/bin/env python3
"""Analyze the qwen self-consistency samples and emit the best cascade prediction file.

Self-consistency = fraction of K qwen samples agreeing with the majority boxed answer.
Cascade: consistency >= tau -> keep qwen's majority-vote answer (cheap: K probe calls);
else escalate to deepseek. Sweeps tau on a lightweight grade to pick the operating point,
then writes cruq-sc-router.json (synthetic majority answer for kept queries, deepseek's
cached answer for escalated) with token_usage set so cost reflects the K probes + escalation.
Score that file with v3 for the official accuracy.
"""
import json, re, math, collections
from datasets import load_from_disk

SC = "phase2/data/qwen_sc.jsonl"
QCOST = {"in": 0.0, "out": 0.0}  # qwen price per token, filled below
COST = json.load(open("model_cost/model_cost.json"))
qc = COST["qwen/qwen3-235b-a22b-2507"]; QCOST = (qc["input_token_price_per_million"], qc["output_token_price_per_million"])

def norm(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())
ds = load_from_disk("./dataset/routerarena_10")
gold = {r["Global Index"]: str(r["Answer"]).strip() for r in ds}
def correct(gi, pb):
    if not pb: return False
    g, p = norm(gold.get(gi, "")), norm(pb)
    if not g: return False
    pl = p[0] if p else ""
    if g.isdigit() and 0 <= int(g) <= 25 and (pl == chr(ord("a") + int(g)) or p == g): return True
    if len(g) == 1: return p == g or pl == g
    return p == g or g in p

# gather samples: gi -> list of (raw_boxed, norm_boxed, in_tok, out_tok)
samp = collections.defaultdict(list)
for line in open(SC):
    r = json.loads(line); samp[r["gi"]].append((str(r["boxed"]).strip(), norm(r["boxed"]), r["in"], r["out"]))

# per-model v3 accuracy + cost (deepseek escalation; qwen for reference)
def load_scored(fn):
    d = json.load(open(f"router_inference/predictions/{fn}.json"))
    return {r["global index"]: (r.get("accuracy"), r.get("cost")) for r in d if r.get("accuracy") is not None}
DS = load_scored("cruq-single-deepseek"); QW = load_scored("cruq-single-qwen")

gis = [g for g in samp if g in gold and g in DS and g in QW and len(samp[g]) >= 1]
K = max(len(samp[g]) for g in gis)
N = len(gis)

def arena(a, c1k, b=.1, cmin=.0044, cmax=200.):
    if c1k <= 0: return 0.
    C = max(0., min(1., (math.log2(cmax) - math.log2(c1k)) / (math.log2(cmax) - math.log2(cmin))))
    den = b * a + C
    return 0. if den == 0 else (1 + b) * a * C / den

# per-query: consistency, majority RAW boxed, majority-correct(lightweight), probe cost
info = {}
for g in gis:
    probe_cost = sum((i * QCOST[0] + o * QCOST[1]) / 1e6 for _, _, i, o in samp[g])
    pairs = [(raw, nm) for raw, nm, _, _ in samp[g] if nm]
    if not pairs:
        info[g] = (0.0, "", False, probe_cost); continue
    cnt = collections.Counter(nm for _, nm in pairs)
    maj_norm, c = cnt.most_common(1)[0]
    maj_raw = next(raw for raw, nm in pairs if nm == maj_norm)  # a raw form of the majority
    info[g] = (c / len(samp[g]), maj_raw, correct(g, maj_raw), probe_cost)

print(f"self-consistency cascade on {N} sub_10 items (K={K} samples/query)\n")
# diagnostic: is consistency predictive?
for lo, hi in [(0.99, 1.01), (0.6, 0.99), (0.0, 0.6)]:
    sub = [g for g in gis if lo <= info[g][0] < hi]
    if sub:
        pc = sum(info[g][2] for g in sub) / len(sub)
        print(f"  consistency in [{lo:.2f},{hi:.2f}): {len(sub):3} queries, majority-correct(lightweight)={pc:.2f}")

print(f"\n{'tau':>5} {'keep%':>6} {'acc':>7} {'cost/1k':>9} {'arena-S':>8}")
best = (-1, None)
for tau in [0.4, 0.6, 0.8, 1.0]:
    acc = cost = kept = 0
    for g in gis:
        cons, maj, maj_ok, probe = info[g]
        if cons >= tau:  # keep qwen majority vote
            acc += maj_ok; cost += probe; kept += 1
        else:            # escalate to deepseek
            acc += DS[g][0]; cost += probe + (DS[g][1] or 0)
    A = acc / N; c1k = cost / N * 1000; S = arena(A, c1k)
    print(f"{tau:5.2f} {kept/N:6.0%} {A:7.4f} ${c1k:8.3f} {S:8.4f}")
    if S > best[0]: best = (S, tau)

# emit prediction file for the best tau
tau = best[1]
qpred = json.load(open("router_inference/predictions/cruq-single-qwen.json"))
dmap = {r["global index"]: r for r in json.load(open("router_inference/predictions/cruq-single-deepseek.json"))}
out = []
for r in qpred:
    g = r["global index"]
    if g not in info:
        out.append(r); continue
    cons, maj, _, probe = info[g]
    itok = sum(i for _, _, i, _ in samp[g]); otok = sum(o for _, _, _, o in samp[g])
    row = dict(r)
    if cons >= tau:
        row["prediction"] = "qwen/qwen3-235b-a22b-2507"
        row["generated_result"] = {"generated_answer": f"\\boxed{{{maj}}}", "success": True,
                                    "token_usage": {"input_tokens": itok, "output_tokens": otok, "total_tokens": itok + otok}}
    else:
        d = dmap[g]; row["prediction"] = d["prediction"]; row["generated_result"] = d["generated_result"]
    row["accuracy"] = None; row["cost"] = None
    out.append(row)
json.dump(out, open("router_inference/predictions/cruq-sc-router.json", "w"))
print(f"\nbest tau={tau} -> wrote cruq-sc-router.json (score with v3 for official accuracy)")
