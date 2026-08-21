#!/usr/bin/env python3
"""Phase 2, step 1: build an EXTERNAL labeling corpus.

The learned router must not train on RouterArena data. This builds a corpus from
the public benchmark datasets' OWN splits (not RouterArena's sampled items), then
dedups every item against RouterArena's 8,400 prompts so no test question leaks
into training. Each item is formatted with the same "answer in \\boxed{X}"
instruction the router/grader use, so labeling and grading stay consistent.

Output: phase2/data/corpus.jsonl  ({id, source, domain, prompt, gold})

Run:  uv run python phase2/build_corpus.py --per-source 800
Free: this only downloads public datasets; no model calls happen here.
"""
import argparse
import json
import os
import re

from datasets import load_dataset

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
    """Normalized question prefixes of every RouterArena prompt, for dedup."""
    keys = set()
    path = "dataset/router_data.json"
    if not os.path.exists(path):
        print(f"[corpus] WARN: {path} missing; dedup disabled (run prep_datasets first)")
        return keys
    for r in json.load(open(path)):
        p = r.get("prompt_formatted") or r.get("prompt") or ""
        # index on the Question line to catch the same item across formats
        m = re.search(r"Question:\s*(.+)", p)
        keys.add(_norm(m.group(1) if m else p))
    return keys


def _mcq_prompt(question, options):
    body = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return f"{question}\n\nOptions:\n{body}{MCQ_INSTR}"


def build_mmlu(n, ra_keys, seen):
    out = []
    ds = load_dataset("cais/mmlu", "all", split="test", streaming=True)
    for row in ds:
        if len(out) >= n:
            break
        q, ch, ans = row["question"], row["choices"], row["answer"]
        k = _norm(q)
        if k in ra_keys or k in seen:
            continue
        seen.add(k)
        out.append({
            "id": f"mmlu_{len(out)}", "source": "MMLU", "domain": row.get("subject", ""),
            "prompt": _mcq_prompt(q, ch), "gold": LETTERS[int(ans)],
        })
    return out


def build_arc(n, ra_keys, seen):
    out = []
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    for row in ds:
        if len(out) >= n:
            break
        q = row["question"]
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        gold = row["answerKey"]
        if gold not in labels:
            continue
        k = _norm(q)
        if k in ra_keys or k in seen:
            continue
        seen.add(k)
        # remap arbitrary labels (A-D or 1-4) to positional letters
        gi = labels.index(gold)
        out.append({
            "id": f"arc_{len(out)}", "source": "ARC", "domain": "science",
            "prompt": _mcq_prompt(q, texts), "gold": LETTERS[gi],
        })
    return out


def build_gsm8k(n, ra_keys, seen):
    out = []
    ds = load_dataset("openai/gsm8k", "main", split="test")
    for row in ds:
        if len(out) >= n:
            break
        q = row["question"]
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        k = _norm(q)
        if k in ra_keys or k in seen:
            continue
        seen.add(k)
        out.append({
            "id": f"gsm8k_{len(out)}", "source": "GSM8K", "domain": "math",
            "prompt": q + NUM_INSTR, "gold": gold,
        })
    return out


def build_mmlu_pro(n, ra_keys, seen):
    """MMLU-Pro: 10-option, markedly harder than MMLU; where models diverge."""
    out = []
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", streaming=True)
    for row in ds:
        if len(out) >= n:
            break
        q = row["question"]
        opts = [o for o in row["options"] if o and o != "N/A"]
        ans = row.get("answer")  # letter
        if not ans or ans not in LETTERS[: len(opts)]:
            continue
        k = _norm(q)
        if k in ra_keys or k in seen:
            continue
        seen.add(k)
        out.append({
            "id": f"mmlupro_{len(out)}", "source": "MMLU-Pro",
            "domain": row.get("category", ""),
            "prompt": _mcq_prompt(q, opts), "gold": ans,
        })
    return out


def build_math(n, ra_keys, seen):
    """MATH-500: hard competition math; boxed numeric/expression answers."""
    out = []
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    for row in ds:
        if len(out) >= n:
            break
        q = row.get("problem") or ""
        gold = str(row.get("answer") or "").strip()
        if not gold:
            continue
        k = _norm(q)
        if k in ra_keys or k in seen:
            continue
        seen.add(k)
        out.append({
            "id": f"math_{len(out)}", "source": "MATH", "domain": "math",
            "prompt": q + NUM_INSTR, "gold": gold,
        })
    return out


BUILDERS = {
    "mmlu": build_mmlu, "arc": build_arc, "gsm8k": build_gsm8k,
    "mmlu_pro": build_mmlu_pro, "math": build_math,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=800)
    ap.add_argument("--sources", nargs="+", default=list(BUILDERS))
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    ra_keys = _load_routerarena_keys()
    print(f"[corpus] RouterArena dedup keys: {len(ra_keys)}")

    seen, corpus = set(), []
    for s in args.sources:
        items = BUILDERS[s](args.per_source, ra_keys, seen)
        print(f"[corpus] {s}: {len(items)} items (after dedup)")
        corpus.extend(items)

    out_path = os.path.join(OUT_DIR, "corpus.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in corpus:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[corpus] wrote {len(corpus)} items -> {out_path}")


if __name__ == "__main__":
    main()
