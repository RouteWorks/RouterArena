import json, sys, collections, math
from datasets import load_from_disk
router = sys.argv[1] if len(sys.argv) > 1 else "cruq-gemini25fl"
d = json.load(open(f"router_inference/predictions/{router}.json"))
ds = load_from_disk("./dataset/routerarena_10")
dn = {r["Global Index"]: (r.get("Dataset name") or r["Global Index"].split("_")[0]) for r in ds}
acc = [(r.get("global index"), r.get("accuracy"), r.get("cost")) for r in d]
scored = [(gi, a, c) for gi, a, c in acc if a is not None]
if not scored:
    print("SUMMARY: 0 rows scored"); sys.exit(0)
A = sum(a for _, a, _ in scored) / len(scored)
costs = [c for _, _, c in scored if c is not None]
cost1k = (sum(costs) / len(costs) * 1000) if costs else 0.0
def arena(a, c1k, beta=0.1, cmin=0.0044, cmax=200.0):
    if c1k <= 0: return 0.0
    C = max(0.0, min(1.0, (math.log2(cmax) - math.log2(c1k)) / (math.log2(cmax) - math.log2(cmin))))
    den = beta * a + C
    return 0.0 if den == 0 else (1 + beta) * a * C / den
print("="*60)
print(f"OFFICIAL RESULT  router={router}")
print(f"  rows scored : {len(scored)}/{len(d)}")
print(f"  accuracy    : {A:.4f}")
print(f"  cost/1k     : ${cost1k:.4f}")
print(f"  arena-S     : {arena(A, cost1k):.4f}")
byds = collections.defaultdict(list)
for gi, a, _ in scored:
    if gi in dn: byds[dn[gi]].append(a)
print("  --- per-dataset (previously-unscorable in bold-worthy) ---")
for name in sorted(byds):
    v = byds[name]
    print(f"    {name:34} n={len(v):3} acc={sum(v)/len(v):.3f}")
print("="*60)
