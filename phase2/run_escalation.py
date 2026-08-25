#!/usr/bin/env python3
"""Run stronger candidate models on ONLY the escalated (hard) queries, to pick a better
escalation tier for the self-consistency router. Deepseek scores just 0.476 on the K=4/tau=0.6
escalation set (130 queries), so a stronger model there can lift overall accuracy cheaply.

Idempotent cache: phase2/data/escalation.jsonl ({model, gi, response, in, out}). Parallel+backoff.
Run: LABEL_WORKERS=4 uv run python phase2/run_escalation.py
"""
import json, os, sys, time
import concurrent.futures as cf
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()
from llm_inference.model_inference import ModelInference

CANDIDATES = ["deepseek/deepseek-v4-pro", "google/gemini-2.5-pro", "anthropic/claude-sonnet-4.5"]
ESC = "/private/tmp/claude-502/-Users-nabaruns-work-cruq-ai-cruq-ai/063e8a4a-b7ad-4b1d-9cd2-2cffe81c3e8b/scratchpad/esc_gis.json"
OUT = "phase2/data/escalation.jsonl"

def infer_retry(mi, m, prompt, tries=5):
    for a in range(tries):
        try:
            r = mi.infer(m, prompt)
        except Exception as e:
            return {"success": False, "error": str(e)}
        if r.get("success"):
            return r
        err = str(r.get("error"))
        if ("Key limit" in err or "429" in err or "rate" in err.lower()) and a < tries - 1:
            time.sleep(3 * (2 ** a)); continue
        return r
    return r

def main():
    esc = set(json.load(open(ESC)))
    prompts = {r["global index"]: r["prompt"] for r in json.load(open("router_inference/predictions/cruq-single-qwen.json"))}
    todo_items = [(gi, prompts[gi]) for gi in esc if gi in prompts]

    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line); done.add((r["model"], r["gi"]))
            except Exception:
                pass
    todo = [(m, gi, p) for m in CANDIDATES for gi, p in todo_items if (m, gi) not in done]
    print(f"[esc] candidates={len(CANDIDATES)} esc_queries={len(todo_items)} pending={len(todo)}", flush=True)

    mi = ModelInference()
    workers = int(os.getenv("LABEL_WORKERS", "4"))
    batch = workers * 6
    written, stop = 0, False
    with open(OUT, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for i in range(0, len(todo), batch):
            if stop:
                break
            chunk = todo[i:i + batch]
            futs = {ex.submit(infer_retry, mi, m, p): (m, gi) for m, gi, p in chunk}
            for fut in cf.as_completed(futs):
                m, gi = futs[fut]
                r = fut.result()
                if not r.get("success"):
                    if "Key limit" in str(r.get("error")) or "429" in str(r.get("error")):
                        stop = True
                    continue
                tu = r.get("token_usage") or {}
                rec = {"model": m, "gi": gi, "response": r.get("response"),
                       "in": tu.get("input_tokens", 0) or 0, "out": tu.get("output_tokens", 0) or 0}
                f.write(json.dumps(rec) + "\n"); f.flush(); written += 1
            print(f"[esc] {written} written ({min(i + batch, len(todo))}/{len(todo)})", flush=True)
    print(f"[esc] done: {written} new" + (" (stopped on limit)" if stop else ""), flush=True)

if __name__ == "__main__":
    main()
