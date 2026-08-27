#!/usr/bin/env python3
"""Self-consistency sampler: sample the cheapest model (qwen) K times per sub_10 query
at temperature>0, so answer-agreement across samples can serve as a confidence signal
for a cascade (high consistency -> keep qwen; low -> escalate to deepseek).

Idempotent: caches each (global_index, sample_idx) to phase2/data/qwen_sc.jsonl and
resumes, so a weekly-limit 403 loses no work. Parallel with backoff.

Run:  LABEL_WORKERS=4 uv run python phase2/sample_qwen_sc.py --k 5
COST: K * |sub_10| qwen calls (~809*K). qwen is the cheapest model.
"""
import argparse, json, os, re, sys, time
import concurrent.futures as cf
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

MODEL = "qwen/qwen3-235b-a22b-2507"
OUT = "phase2/data/qwen_sc.jsonl"
BOXED = re.compile(r"\\boxed\{+([^{}]*)\}+")
def boxed(text):
    ms = BOXED.findall(text or "")
    return ms[-1].strip() if ms else ""

def sample_once(prompt, temp):
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"),
                    default_headers={"HTTP-Referer": "https://cruq.ai/", "X-Title": "Cruq AI"})
    for a in range(5):
        try:
            r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=temp)
            u = r.usage
            return {"ok": True, "text": r.choices[0].message.content,
                    "in": getattr(u, "prompt_tokens", 0) or 0, "out": getattr(u, "completion_tokens", 0) or 0}
        except Exception as e:
            err = str(e)
            if ("Key limit" in err or "429" in err or "rate" in err.lower()) and a < 4:
                time.sleep(3 * (2 ** a)); continue
            return {"ok": False, "err": err[:80]}
    return {"ok": False, "err": "exhausted"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.7)
    args = ap.parse_args()

    preds = json.load(open("router_inference/predictions/cruq-single-qwen.json"))
    queries = [(r["global index"], r["prompt"]) for r in preds]

    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line); done.add((r["gi"], r["s"]))
            except Exception:
                pass
    todo = [(gi, p, s) for gi, p in queries for s in range(args.k) if (gi, s) not in done]
    print(f"[sc] k={args.k} temp={args.temp} queries={len(queries)} pending={len(todo)}", flush=True)

    workers = int(os.getenv("LABEL_WORKERS", "4"))
    batch = workers * 6
    written, stop = 0, False
    with open(OUT, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for i in range(0, len(todo), batch):
            if stop:
                break
            chunk = todo[i:i + batch]
            futs = {ex.submit(sample_once, p, args.temp): (gi, s) for gi, p, s in chunk}
            for fut in cf.as_completed(futs):
                gi, s = futs[fut]
                r = fut.result()
                if not r.get("ok"):
                    if "Key limit" in str(r.get("err")) or "429" in str(r.get("err")):
                        stop = True
                    continue
                rec = {"gi": gi, "s": s, "boxed": boxed(r["text"]), "in": r["in"], "out": r["out"]}
                f.write(json.dumps(rec) + "\n"); f.flush(); written += 1
            print(f"[sc] {written} written ({min(i + batch, len(todo))}/{len(todo)} attempted)", flush=True)
    print(f"[sc] done: {written} new samples -> {OUT}" + (" (stopped on limit)" if stop else ""), flush=True)

if __name__ == "__main__":
    main()
