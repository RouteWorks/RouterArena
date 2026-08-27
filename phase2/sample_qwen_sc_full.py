#!/usr/bin/env python3
"""Full-split (8400) self-consistency sampler for the RouterArena submission.

Samples qwen K times per query at temperature>0 over dataset/router_data.json
(the official full split, prompt_formatted is the exact prompt the validator checks).
Answer-agreement across samples is the confidence signal for the cascade
(high consistency -> keep qwen majority vote; low -> escalate to deepseek).

Reuses the sub_10 samples already in phase2/data/qwen_sc.jsonl (all 809 sub_10
indices are a subset of the full split), so only the ~7591 remaining queries are
sampled fresh.

Idempotent: caches each (global_index, sample_idx) to phase2/data/qwen_sc_full.jsonl
and resumes, so a weekly-limit 403 loses no work. Parallel with backoff.

Run:  LABEL_WORKERS=10 uv run python phase2/sample_qwen_sc_full.py --k 4
"""
import argparse, json, os, re, sys, time
import concurrent.futures as cf
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

MODEL = "qwen/qwen3-235b-a22b-2507"
DATASET = "dataset/router_data.json"
SEED = "phase2/data/qwen_sc.jsonl"       # existing sub_10 samples to reuse
OUT = "phase2/data/qwen_sc_full.jsonl"   # full-split cache
BOXED = re.compile(r"\\boxed\{+([^{}]*)\}+")


def boxed(text):
    ms = BOXED.findall(text or "")
    return ms[-1].strip() if ms else ""


def _take(it, n):
    """Pull up to n items from an iterator (for the rolling submission window)."""
    out = []
    for _ in range(n):
        try:
            out.append(next(it))
        except StopIteration:
            break
    return out


MAX_TOKENS = int(os.getenv("SC_MAX_TOKENS", "6000"))  # cap runaway reasoning/code generations


_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"),
                         default_headers={"HTTP-Referer": "https://cruq.ai/", "X-Title": "Cruq AI"})
    return _CLIENT


def sample_once(prompt, temp):
    client = _client()
    for a in range(5):
        try:
            # sort providers by throughput: OpenRouter otherwise routes qwen to slow
            # providers (latency tail up to ~28s); throughput sort caps it near ~5s.
            r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                               temperature=temp, max_tokens=MAX_TOKENS,
                                               extra_body={"provider": {"sort": "throughput"}})
            u = r.usage
            return {"ok": True, "text": r.choices[0].message.content,
                    "in": getattr(u, "prompt_tokens", 0) or 0, "out": getattr(u, "completion_tokens", 0) or 0}
        except Exception as e:
            err = str(e)
            if ("402" in err or "Insufficient credits" in err):
                # hard credit stop: no point retrying, propagate so the run halts cleanly
                return {"ok": False, "err": "CREDITS: " + err[:120]}
            if ("Key limit" in err or "429" in err or "rate" in err.lower()) and a < 4:
                time.sleep(3 * (2 ** a)); continue
            return {"ok": False, "err": err[:80]}
    return {"ok": False, "err": "exhausted"}


def seed_full_cache(k):
    """Copy the first k sub_10 samples per gi into the full cache if not already present.

    For sample 0 attach qwen's actual answer (from cruq-single-qwen.json) as `text`, so
    free-form (unboxed) sub_10 queries have a real qwen answer to emit, matching the
    behaviour of the freshly-sampled full-split queries.
    """
    if not os.path.exists(SEED):
        return
    qans = {}
    qpath = "router_inference/predictions/cruq-single-qwen.json"
    if os.path.exists(qpath):
        for r in json.load(open(qpath)):
            gr = r.get("generated_result") or {}
            qans[r["global index"]] = gr.get("generated_answer") or ""
    have = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line); have.add((r["gi"], r["s"]))
            except Exception:
                pass
    added = 0
    with open(OUT, "a", encoding="utf-8") as f:
        for line in open(SEED):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r["s"] < k and (r["gi"], r["s"]) not in have:
                rec = {"gi": r["gi"], "s": r["s"], "boxed": r["boxed"], "in": r["in"], "out": r["out"]}
                if r["s"] == 0:
                    rec["text"] = qans.get(r["gi"], "")
                f.write(json.dumps(rec) + "\n")
                have.add((r["gi"], r["s"])); added += 1
    print(f"[seed] copied {added} sub_10 samples into {OUT}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.7)
    args = ap.parse_args()

    data = json.load(open(DATASET))
    queries = [(e["global index"], e["prompt_formatted"]) for e in data]
    print(f"[sc-full] dataset={len(queries)} queries k={args.k}", flush=True)

    seed_full_cache(args.k)

    done = set()
    for line in open(OUT):
        try:
            r = json.loads(line); done.add((r["gi"], r["s"]))
        except Exception:
            pass
    todo = [(gi, p, s) for gi, p in queries for s in range(args.k) if (gi, s) not in done]
    print(f"[sc-full] pending={len(todo)} (already have {len(done)})", flush=True)

    workers = int(os.getenv("LABEL_WORKERS", "10"))
    written, stop = 0, False
    # Continuous pipelining: keep the pool saturated with a rolling window of futures
    # instead of chunked batches (a chunk otherwise waits on its slowest call).
    window = workers * 4
    it = iter(todo)
    with open(OUT, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(max_workers=workers) as ex:
        inflight = {}
        for gi, p, s in _take(it, window):
            inflight[ex.submit(sample_once, p, args.temp)] = (gi, s)
        while inflight:
            for fut in cf.as_completed(list(inflight)):
                gi, s = inflight.pop(fut)
                r = fut.result()
                if r.get("ok"):
                    rec = {"gi": gi, "s": s, "boxed": boxed(r["text"]), "in": r["in"], "out": r["out"]}
                    # keep the raw response for sample 0 only: the qwen answer used verbatim
                    # on free-form (unboxed) datasets, where self-consistency can't apply.
                    if s == 0:
                        rec["text"] = r["text"]
                    f.write(json.dumps(rec) + "\n"); f.flush(); written += 1
                    if written % 500 == 0:
                        print(f"[sc-full] {written} new (of {len(todo)} pending)", flush=True)
                else:
                    e = str(r.get("err"))
                    if "CREDITS" in e or "Key limit" in e or "429" in e:
                        if not stop:
                            print(f"[sc-full] HALTING on fatal error: {e}", flush=True)
                        stop = True
                # top up the window unless halting
                if not stop:
                    for nxt in _take(it, 1):
                        inflight[ex.submit(sample_once, nxt[1], args.temp)] = (nxt[0], nxt[2])
                break  # re-evaluate as_completed over the refreshed inflight set
            if stop:
                for fut in list(inflight):
                    fut.cancel()
                break
    print(f"[sc-full] done: {written} new samples -> {OUT}" + (" (stopped on limit)" if stop else ""), flush=True)


if __name__ == "__main__":
    main()
