#!/usr/bin/env python3
"""Assemble the full-split (8400) self-consistency prediction file for submission.

Per query, using the first K qwen probes from phase2/data/qwen_sc_full.jsonl:
  consistency = fraction of the K samples agreeing on the majority \\boxed answer.
  consistency >= tau  -> KEEP qwen: emit the raw majority \\boxed answer, token_usage
                          = sum of the K probe calls, priced at qwen (fully honest).
  consistency <  tau  -> ESCALATE to deepseek-v4-flash: emit deepseek's answer, and
                          fold the K qwen probe tokens into token_usage priced at
                          deepseek. deepseek is pricier per-token than qwen, so this
                          slightly OVER-counts the probe tokens: conservative, never
                          understating the router's true multi-call cost.

Deepseek answers: reused from cruq-single-deepseek.json (sub_10) where available,
else from phase2/data/escalation_full.jsonl.

Writes router_inference/predictions/cruq-router.json (8400 regular entries).
Env: K (default 4), TAU (default 0.6).
"""
import json, os, re, collections

K = int(os.getenv("K", "4"))
TAU = float(os.getenv("TAU", "0.6"))
QWEN = "qwen/qwen3-235b-a22b-2507"
DEEPSEEK = "deepseek/deepseek-v4-flash"


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


data = json.load(open("dataset/router_data.json"))

# gather qwen probes: gi -> [(s, raw_boxed, norm_boxed, in, out)]; also the sample-0 raw text
raw = collections.defaultdict(list)
qwen_text = {}
for line in open("phase2/data/qwen_sc_full.jsonl"):
    try:
        r = json.loads(line)
    except Exception:
        continue
    raw[r["gi"]].append((r["s"], str(r["boxed"]).strip(), norm(r["boxed"]), r["in"], r["out"]))
    if r.get("s") == 0 and r.get("text"):
        qwen_text[r["gi"]] = r["text"]
samp = {}
for gi, rows in raw.items():
    samp[gi] = [(rb, nb, i, o) for _, rb, nb, i, o in sorted(rows)[:K]]

# deepseek answers: sub_10 cache (full generated_result) + full-split escalation cache
ds_sub10 = {r["global index"]: r["generated_result"]
            for r in json.load(open("router_inference/predictions/cruq-single-deepseek.json"))}
ds_full = {}
if os.path.exists("phase2/data/escalation_full.jsonl"):
    for line in open("phase2/data/escalation_full.jsonl"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        ds_full[r["gi"]] = r

kept = freeform = escalated = missing_probe = missing_ds = 0
out = []
for e in data:
    gi = e["global index"]
    prompt = e["prompt_formatted"]
    rows = samp.get(gi, [])
    if len(rows) < 1:
        missing_probe += 1
    itok = sum(i for _, _, i, _ in rows)
    otok = sum(o for _, _, _, o in rows)
    qtu = {"input_tokens": itok, "output_tokens": otok, "total_tokens": itok + otok}
    pairs = [(rb, nb) for rb, nb, _, _ in rows if nb]
    if pairs:
        cnt = collections.Counter(nb for _, nb in pairs)
        maj_norm, c = cnt.most_common(1)[0]
        maj_raw = next(rb for rb, nb in pairs if nb == maj_norm)
        consistency = c / len(rows)
    else:
        maj_raw, consistency = "", 0.0

    if not pairs:
        # free-form dataset (no \boxed answer, e.g. code/translation/long-form QA):
        # self-consistency can't apply, so keep qwen's actual answer rather than pay to
        # escalate. Cost = the K qwen probe calls, priced at qwen (honest).
        freeform += 1
        gr = {"generated_answer": qwen_text.get(gi, ""), "success": True,
              "token_usage": qtu, "model_used": QWEN, "provider": "openrouter", "error": None}
        out.append({"global index": gi, "prompt": prompt, "prediction": QWEN,
                    "generated_result": gr, "cost": None, "accuracy": None, "for_optimality": False})
        continue
    elif consistency >= TAU:
        kept += 1
        gr = {"generated_answer": f"\\boxed{{{maj_raw}}}", "success": True,
              "token_usage": qtu,
              "model_used": QWEN, "provider": "openrouter", "error": None}
        prediction = QWEN
    else:
        escalated += 1
        prediction = DEEPSEEK
        dsr = ds_sub10.get(gi)
        if dsr is not None:
            ans = dsr.get("generated_answer")
            dtu = dsr.get("token_usage") or {}
            din = dtu.get("input_tokens", 0) or 0
            dout = dtu.get("output_tokens", 0) or 0
        elif gi in ds_full:
            ans = ds_full[gi]["response"]
            din = ds_full[gi]["in"]; dout = ds_full[gi]["out"]
        else:
            # no deepseek answer available -> fall back to qwen majority so nothing is empty
            missing_ds += 1
            kept += 1; escalated -= 1
            gr = {"generated_answer": f"\\boxed{{{maj_raw}}}", "success": True,
                  "token_usage": qtu,
                  "model_used": QWEN, "provider": "openrouter", "error": None}
            out.append({"global index": gi, "prompt": prompt, "prediction": QWEN,
                        "generated_result": gr, "cost": None, "accuracy": None, "for_optimality": False})
            continue
        # fold the K qwen probe tokens into the escalated cost (priced at deepseek: conservative)
        tin = din + itok; tout = dout + otok
        gr = {"generated_answer": ans, "success": True,
              "token_usage": {"input_tokens": tin, "output_tokens": tout, "total_tokens": tin + tout},
              "model_used": DEEPSEEK, "provider": "openrouter", "error": None}

    out.append({"global index": gi, "prompt": prompt, "prediction": prediction,
                "generated_result": gr, "cost": None, "accuracy": None, "for_optimality": False})

json.dump(out, open("router_inference/predictions/cruq-router.json", "w"))
N = len(out)
print(f"wrote cruq-router.json: {N} entries | kept_boxed={kept} freeform_qwen={freeform} "
      f"escalated={escalated} ({escalated/N:.1%}) | qwen_total={kept + freeform} ({(kept + freeform)/N:.1%}) "
      f"| missing_probe={missing_probe} missing_ds_fallback={missing_ds}")
