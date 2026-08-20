#!/usr/bin/env python3
"""Phase 2, step 2: label the external corpus with each pool model.

For every (corpus item, pool model) pair, call the model and score correctness
(\\boxed{X} extraction, reconciling numeric-index vs letter gold). Results are the
training signal for the per-model P(correct) predictor. Idempotent: skips pairs
already in the label cache, so an interrupted run (e.g. an OpenRouter weekly-limit
403) resumes without repeating spend.

Output: phase2/data/labels.jsonl  ({id, model, correct, in_tok, out_tok})

Run:  uv run python phase2/label_corpus.py --pool phase2/pool.json
COST: this makes len(corpus) * len(pool) live API calls. Size the corpus and
      raise the OpenRouter weekly cap accordingly before running.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, ".")
from dotenv import load_dotenv  # noqa: E402

load_dotenv()
from llm_inference.model_inference import ModelInference  # noqa: E402

CORPUS = "phase2/data/corpus.jsonl"
LABELS = "phase2/data/labels.jsonl"
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
LETTERS = "abcdefghij"


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def is_correct(gold, answer):
    ms = BOXED.findall(answer or "")
    if not ms:
        return False
    p = _norm(ms[-1])
    g = _norm(gold)
    if not g or not p:
        return False
    pl = p[0]
    if g.isdigit() and 0 <= int(g) <= 25:
        if pl == LETTERS[int(g)] or p == g:
            return True
    if len(g) == 1:
        return p == g or pl == g
    return p == g or g in p


def load_done():
    done = set()
    if os.path.exists(LABELS):
        for line in open(LABELS):
            try:
                r = json.loads(line)
                done.add((r["id"], r["model"]))
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="phase2/pool.json")
    ap.add_argument("--limit", type=int, default=0, help="cap corpus items (0=all)")
    args = ap.parse_args()

    pool = json.load(open(args.pool))["models"]
    corpus = [json.loads(x) for x in open(CORPUS)]
    if args.limit:
        corpus = corpus[: args.limit]
    done = load_done()
    mi = ModelInference()

    todo = [(it, m) for it in corpus for m in pool if (it["id"], m) not in done]
    print(f"[label] pool={len(pool)} corpus={len(corpus)} pending_calls={len(todo)}")

    written = 0
    with open(LABELS, "a", encoding="utf-8") as f:
        for it, m in todo:
            r = mi.infer(m, it["prompt"])
            if not r.get("success"):
                err = str(r.get("error"))[:80]
                if "Key limit" in err or "429" in err:
                    print(f"[label] STOPPING: rate/key limit hit after {written} new labels: {err}")
                    break
                continue
            tu = r.get("token_usage") or {}
            rec = {
                "id": it["id"], "model": m,
                "correct": bool(is_correct(it["gold"], r.get("response"))),
                "in_tok": tu.get("input_tokens", 0) or 0,
                "out_tok": tu.get("output_tokens", 0) or 0,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            written += 1
            if written % 100 == 0:
                print(f"[label] {written}/{len(todo)} new labels")
    print(f"[label] done: {written} new labels -> {LABELS}")


if __name__ == "__main__":
    main()
