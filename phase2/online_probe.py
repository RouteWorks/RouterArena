#!/usr/bin/env python3
"""Online-probe (cascade) router — cache-only prototype on the v3-official metric.

The one lever the no-train rule leaves is an INFERENCE-TIME signal. Here: probe a
query with two cheap models; if their answers AGREE, trust the cheap answer (easy
query); if they DISAGREE (uncertainty), escalate to the strong model. Cost counts
every call made (both probes, plus the escalation model when used) — the RouterArena
way for cascades. Measured on the 5 single-model prediction files already scored by
the fixed v3 evaluator (per-query accuracy + cost are official).

This is a measurement harness to pick the best probe config; the winning config is
then emitted as a real prediction file by build_probe_prediction.py.
"""
import json, re, math, itertools

MODELS = {
    "qwen": ("cruq-single-qwen", "qwen/qwen3-235b-a22b-2507"),
    "coder": ("cruq-single-coder", "Qwen/Qwen3-Coder-Next"),
    "deepseek": ("cruq-single-deepseek", "deepseek/deepseek-v4-flash"),
    "gpt4omini": ("cruq-single-gpt4omini", "openai/gpt-4o-mini"),
    "gemini": ("cruq-gemini25fl", "google/gemini-2.5-flash-lite"),
}
BOXED = re.compile(r"\\boxed\{+([^{}]*)\}+")
def norm(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())
def boxed(gr):
    ga = gr.get("generated_answer") if isinstance(gr, dict) else gr
    ga = ga if isinstance(ga, str) else ""
    ms = BOXED.findall(ga)
    return norm(ms[-1]) if ms else norm(ga[-40:])  # fallback: tail

# load per-model: gi -> (accuracy, cost, boxed_answer)
M = {}
for m, (fn, _) in MODELS.items():
    d = json.load(open(f"router_inference/predictions/{fn}.json"))
    M[m] = {}
    for r in d:
        if r.get("accuracy") is None:
            continue
        M[m][r["global index"]] = (r["accuracy"], r.get("cost") or 0.0, boxed(r.get("generated_result")))

gis = set(M["qwen"])
for m in MODELS:
    gis &= set(M[m])
gis = sorted(gis)
N = len(gis)

def arena(a, c1k, b=.1, cmin=.0044, cmax=200.):
    if c1k <= 0: return 0.
    C = max(0., min(1., (math.log2(cmax) - math.log2(c1k)) / (math.log2(cmax) - math.log2(cmin))))
    den = b * a + C
    return 0. if den == 0 else (1 + b) * a * C / den

def report(label, acc, cost_usd):
    A = acc / N; c1k = cost_usd / N * 1000
    print(f"  {label:38} acc={A:.4f} cost/1k=${c1k:.3f} arena-S={arena(A, c1k):.4f}")

print(f"online-probe cascades on {N} v3-scored sub_10 items\n")
# baselines
for m in ["qwen", "deepseek"]:
    a = sum(M[m][g][0] for g in gis); c = sum(M[m][g][1] for g in gis)
    report(f"single: {m}", a, c)
print()

# --- dual-probe agreement cascades: probe {p1,p2}; agree->cheaper of the two, else escalate ---
CHEAP = ["qwen", "gpt4omini", "coder", "gemini"]
COSTORD = sorted(MODELS, key=lambda m: {"qwen":.03,"gpt4omini":.08,"coder":.19,"gemini":.25,"deepseek":.21}[m])
def cheaper(a, b):  # by blended proxy already in COSTORD
    return a if COSTORD.index(a) <= COSTORD.index(b) else b
for p1, p2 in itertools.combinations(CHEAP, 2):
    for esc in ["deepseek"]:
        acc = cost = agree_n = agree_correct = 0
        keep = cheaper(p1, p2)
        for g in gis:
            probe_cost = M[p1][g][1] + M[p2][g][1]
            if M[p1][g][2] and M[p1][g][2] == M[p2][g][2]:  # agree (non-empty)
                acc += M[keep][g][0]; cost += probe_cost
                agree_n += 1; agree_correct += M[keep][g][0]
            else:
                acc += M[esc][g][0]; cost += probe_cost + M[esc][g][1]
        rate = agree_n / N
        pc = agree_correct / agree_n if agree_n else 0
        report(f"probe({p1}+{p2})->{keep}|else {esc}  [agree {rate:.0%}, P(correct|agree)={pc:.2f}]", acc, cost)

# --- 3-way vote: qwen+coder+gpt4omini; unanimous->cheapest, 2-agree->majority, else deepseek ---
print()
tri = ["qwen", "coder", "gpt4omini"]
acc = cost = 0
for g in gis:
    ans = [M[t][g][2] for t in tri]
    probe_cost = sum(M[t][g][1] for t in tri)
    from collections import Counter
    cnt = Counter(a for a in ans if a)
    if cnt and cnt.most_common(1)[0][1] >= 2:
        top = cnt.most_common(1)[0][0]
        # pick cheapest model that produced the majority answer
        winners = [t for t in tri if M[t][g][2] == top]
        keep = min(winners, key=lambda m: COSTORD.index(m))
        acc += M[keep][g][0]; cost += probe_cost
    else:
        acc += M["deepseek"][g][0]; cost += probe_cost + M["deepseek"][g][1]
report("vote(qwen,coder,gpt4omini)->maj|else deepseek", acc, cost)

print(f"\n  reference: domain-router 0.737 | domain ceiling 0.778 | oracle 0.842")
