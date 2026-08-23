#!/usr/bin/env python3
"""Phase 3, step 1: build a DOMAIN-DIVERSE external corpus mirroring RouterArena.

The original corpus (business MMLU-Pro + MATH only) was domain-narrow, so the
per-model heads were out-of-distribution on the 71 domains of sub_10. This builds
a balanced corpus across the largest RouterArena domains, from the datasets' OWN
public splits, deduped against RouterArena, each item tagged with a `domain` so a
domain-aware router can be trained. Covered domains map directly onto sub_10:

  mmlupro_<cat> (14 cats)  -> MMLUPro_<cat>
  math                     -> MATH / AIME / GSM8K / AsDiv / MMLUPro_math
  medical_pubmed           -> PubMedQA
  medical_mcq              -> MedMCQA
  science                  -> ArcMMLU (science half)
  knowledge                -> ArcMMLU (MMLU half) / OpenTDB / misc MCQ

Not sourced here (no clean public split / special scorer): code (LiveCodeBench),
translation (WMT19), reading (NarrativeQA/QANTA), trivia (OpenTDB), chess, music.
Queries in those domains fall back to the router's default at eval time.

Output: phase2/data/corpus.jsonl  ({id, source, domain, prompt, gold})
Free: only downloads public datasets; no model calls.
"""
import argparse
import json
import os
import re

from dotenv import load_dotenv

load_dotenv(os.path.join(os.getcwd(), ".env"))  # HF_HOME on the SSD + token
from datasets import load_dataset  # noqa: E402

OUT_DIR = "phase2/data"
LETTERS = "ABCDEFGHIJ"
MCQ_INSTR = (
    "\n\nProvide the correct letter choice in \\boxed{X}, where X is the correct "
    "letter choice. Keep the explanation within 3 sentences."
)
NUM_INSTR = (
    "\n\nProvide the final numeric answer in \\boxed{X}. Keep the explanation "
    "within 3 sentences."
)


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())[:200]


def _load_routerarena_keys():
    keys = set()
    path = "dataset/router_data.json"
    if not os.path.exists(path):
        print(f"[corpus] WARN: {path} missing; dedup disabled")
        return keys
    for r in json.load(open(path)):
        p = r.get("prompt_formatted") or r.get("prompt") or ""
        m = re.search(r"Question:\s*(.+)", p)
        keys.add(_norm(m.group(1) if m else p))
    return keys


def _mcq(question, options):
    body = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return f"{question}\n\nOptions:\n{body}{MCQ_INSTR}"


def add(out, seen, ra, item, cap):
    k = _norm(item["_q"])
    if len(out) >= cap or k in ra or k in seen:
        return False
    seen.add(k)
    del item["_q"]
    item["id"] = f"{item['domain']}_{len([o for o in out if o['domain']==item['domain']])}"
    out.append(item)
    return True


def build(ra, per_domain):
    out, seen = [], set()

    # MMLU-Pro across ALL 14 categories (fixes the all-business bug)
    cap_pc = {}
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", streaming=True)
    for row in ds:
        cat = (row.get("category") or "other").lower().replace(" ", "_")
        dom = f"mmlupro_{cat}"
        if cap_pc.get(dom, 0) >= per_domain:
            continue
        opts = [o for o in row["options"] if o and o != "N/A"]
        ans = row.get("answer")
        if not ans or ans not in LETTERS[: len(opts)]:
            continue
        if add(out, seen, ra, {"_q": row["question"], "source": "MMLU-Pro", "domain": dom,
                               "prompt": _mcq(row["question"], opts), "gold": ans}, 10**9):
            cap_pc[dom] = cap_pc.get(dom, 0) + 1

    # MATH-500 (math)
    n0 = len(out)
    for row in load_dataset("HuggingFaceH4/MATH-500", split="test"):
        q = row.get("problem") or ""
        g = str(row.get("answer") or "").strip()
        if g:
            add(out, seen, ra, {"_q": q, "source": "MATH", "domain": "math",
                                "prompt": q + NUM_INSTR, "gold": g}, n0 + per_domain)

    # GSM8K (math word) -> extend the math domain
    for row in load_dataset("openai/gsm8k", "main", split="test"):
        q = row["question"]
        g = row["answer"].split("####")[-1].strip().replace(",", "")
        add(out, seen, ra, {"_q": q, "source": "GSM8K", "domain": "math",
                            "prompt": q + NUM_INSTR, "gold": g},
            n0 + per_domain + per_domain // 2)

    # PubMedQA (medical yes/no/maybe)
    cnt = 0
    opts = ["yes", "no", "maybe"]
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    for row in ds:
        if cnt >= per_domain:
            break
        dec = row.get("final_decision")
        if dec not in opts:
            continue
        q = row["question"]
        ctx = " ".join(row["context"]["contexts"]) if isinstance(row.get("context"), dict) else ""
        prompt = (f"{q}\n\nContext: {ctx[:1500]}\n\nOptions:\n"
                  + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts)) + MCQ_INSTR)
        if add(out, seen, ra, {"_q": q, "source": "PubMedQA", "domain": "medical_pubmed",
                               "prompt": prompt, "gold": LETTERS[opts.index(dec)]}, 10**9):
            cnt += 1

    # MedMCQA (medical MCQ)
    cnt = 0
    ds = load_dataset("openlifescienceai/medmcqa", split="validation", streaming=True)
    for row in ds:
        if cnt >= per_domain:
            break
        opts = [row["opa"], row["opb"], row["opc"], row["opd"]]
        cop = row.get("cop")
        if cop is None or not (0 <= cop < 4):
            continue
        if add(out, seen, ra, {"_q": row["question"], "source": "MedMCQA", "domain": "medical_mcq",
                               "prompt": _mcq(row["question"], opts), "gold": LETTERS[cop]}, 10**9):
            cnt += 1

    # ARC-Challenge (science)
    n0 = len(out)
    for row in load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"):
        labels, texts, gold = row["choices"]["label"], row["choices"]["text"], row["answerKey"]
        if gold not in labels:
            continue
        add(out, seen, ra, {"_q": row["question"], "source": "ARC", "domain": "science",
                            "prompt": _mcq(row["question"], texts), "gold": LETTERS[labels.index(gold)]},
            n0 + per_domain)

    # MMLU (broad knowledge)
    n0 = len(out)
    ds = load_dataset("cais/mmlu", "all", split="test", streaming=True)
    for row in ds:
        add(out, seen, ra, {"_q": row["question"], "source": "MMLU", "domain": "knowledge",
                            "prompt": _mcq(row["question"], row["choices"]), "gold": LETTERS[int(row["answer"])]},
            n0 + per_domain)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=40)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    ra = _load_routerarena_keys()
    print(f"[corpus] RouterArena dedup keys: {len(ra)}")
    corpus = build(ra, args.per_domain)
    import collections
    dist = collections.Counter(c["domain"] for c in corpus)
    for d, n in sorted(dist.items()):
        print(f"  {d:26} {n}")
    out_path = os.path.join(OUT_DIR, "corpus.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in corpus:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[corpus] wrote {len(corpus)} items across {len(dist)} domains -> {out_path}")


if __name__ == "__main__":
    main()
