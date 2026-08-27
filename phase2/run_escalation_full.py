#!/usr/bin/env python3
"""Run deepseek-v4-flash on the FULL-split escalated queries for the submission.

The escalation set = queries whose qwen self-consistency (K=4) is below tau=0.6,
computed from phase2/data/qwen_sc_full.jsonl. Answers already cached for sub_10
(router_inference/predictions/cruq-single-deepseek.json) are reused; only the rest
are called fresh.

Idempotent cache: phase2/data/escalation_full.jsonl ({gi, response, in, out}).
Run AFTER sampling is complete. Parallel + backoff.

Run:  LABEL_WORKERS=6 uv run python phase2/run_escalation_full.py
"""
import json, os, sys, time, re, collections
import concurrent.futures as cf
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

MODEL = "deepseek/deepseek-v4-flash"
K = 4
TAU = 0.6
SC = "phase2/data/qwen_sc_full.jsonl"
OUT = "phase2/data/escalation_full.jsonl"
# Pin deepseek to DeepInfra: it serves concise answers (~50-200 tokens) matching the
# validated sub_10 deepseek profile. OpenRouter's default/throughput routing sometimes
# picks a verbose reasoning provider (Alibaba, 2k-19k tokens) that is slow and inflates cost.
MAX_TOKENS = int(os.getenv("ESC_MAX_TOKENS", "8000"))  # safety cap if a fallback provider is used
PROVIDER = {"order": ["deepinfra"], "allow_fallbacks": True}
_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"),
                         default_headers={"HTTP-Referer": "https://cruq.ai/", "X-Title": "Cruq AI"})
    return _CLIENT


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def escalated_gis():
    samp = collections.defaultdict(list)
    for line in open(SC):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r["s"] < K:
            samp[r["gi"]].append(norm(r["boxed"]))
    esc = []
    for gi, boxes in samp.items():
        bx = [b for b in boxes if b]
        if not bx:
            # free-form (no \boxed answer): kept on qwen, never escalated.
            continue
        top = collections.Counter(bx).most_common(1)[0][1]
        if top / len(boxes) < TAU:
            esc.append(gi)
    return set(esc)


def infer_retry(mi, prompt, tries=5):
    client = _client()
    for a in range(tries):
        try:
            r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                               max_tokens=MAX_TOKENS, extra_body={"provider": PROVIDER})
            u = r.usage
            return {"success": True, "response": r.choices[0].message.content,
                    "token_usage": {"input_tokens": getattr(u, "prompt_tokens", 0) or 0,
                                    "output_tokens": getattr(u, "completion_tokens", 0) or 0}}
        except Exception as e:
            err = str(e)
            if "402" in err or "Insufficient credits" in err:
                return {"success": False, "error": "CREDITS: " + err[:120]}
            if ("Key limit" in err or "429" in err or "rate" in err.lower()) and a < tries - 1:
                time.sleep(3 * (2 ** a)); continue
            return {"success": False, "error": err[:120]}
    return {"success": False, "error": "exhausted"}


def main():
    esc = escalated_gis()
    prompts = {e["global index"]: e["prompt_formatted"] for e in json.load(open("dataset/router_data.json"))}

    # already have deepseek answers for sub_10 (reused by the builder, not re-run here)
    sub10_done = {r["global index"] for r in json.load(open("router_inference/predictions/cruq-single-deepseek.json"))}
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                done.add(json.loads(line)["gi"])
            except Exception:
                pass

    todo = [gi for gi in esc if gi not in sub10_done and gi not in done and gi in prompts]
    print(f"[esc-full] escalated={len(esc)} sub10_cached={len(esc & sub10_done)} already={len(done)} pending={len(todo)}", flush=True)

    workers = int(os.getenv("LABEL_WORKERS", "6"))
    batch = workers * 6
    written, stop = 0, False
    with open(OUT, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for i in range(0, len(todo), batch):
            if stop:
                break
            chunk = todo[i:i + batch]
            futs = {ex.submit(infer_retry, None, prompts[gi]): gi for gi in chunk}
            for fut in cf.as_completed(futs):
                gi = futs[fut]
                r = fut.result()
                if not r.get("success"):
                    e = str(r.get("error"))
                    if "CREDITS" in e or "Key limit" in e or "429" in e:
                        if not stop:
                            print(f"[esc-full] HALTING on fatal error: {e}", flush=True)
                        stop = True
                    continue
                tu = r.get("token_usage") or {}
                rec = {"gi": gi, "response": r.get("response"),
                       "in": tu.get("input_tokens", 0) or 0, "out": tu.get("output_tokens", 0) or 0}
                f.write(json.dumps(rec) + "\n"); f.flush(); written += 1
            print(f"[esc-full] {written} written ({min(i + batch, len(todo))}/{len(todo)})", flush=True)
    print(f"[esc-full] done: {written} new" + (" (stopped on limit)" if stop else ""), flush=True)


if __name__ == "__main__":
    main()
