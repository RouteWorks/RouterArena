#!/usr/bin/env python3
"""Build the robustness prediction file for the cruq self-consistency router.

Robustness only measures the model-selection flip ratio after prompt noise, so no
generated_result is needed: just the router's model choice per query. For the SC
cascade that means sampling qwen K=4 on each of the 420 robustness prompts and
choosing qwen (consistency >= tau) or deepseek (escalate), same rule as the full split.

Idempotent probe cache: phase2/data/qwen_sc_robustness.jsonl.
Writes router_inference/predictions/cruq-router-robustness.json.

Run:  LABEL_WORKERS=8 uv run python phase2/build_robustness.py
"""
import json, os, re, sys, time, collections
import concurrent.futures as cf
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

MODEL = "qwen/qwen3-235b-a22b-2507"
K, TAU = 4, 0.6
QWEN, DEEPSEEK = "qwen/qwen3-235b-a22b-2507", "deepseek/deepseek-v4-flash"
DATASET = "dataset/router_robustness.json"
OUT = "phase2/data/qwen_sc_robustness.jsonl"
BOXED = re.compile(r"\\boxed\{+([^{}]*)\}+")


def boxed(t):
    ms = BOXED.findall(t or "")
    return ms[-1].strip() if ms else ""


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def sample_once(prompt, temp=0.7):
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"),
                    default_headers={"HTTP-Referer": "https://cruq.ai/", "X-Title": "Cruq AI"})
    for a in range(5):
        try:
            r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=temp,
                                               max_tokens=6000, extra_body={"provider": {"sort": "throughput"}})
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
    data = json.load(open(DATASET))
    queries = [(e["global index"], e["prompt_formatted"]) for e in data]

    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line); done.add((r["gi"], r["s"]))
            except Exception:
                pass
    todo = [(gi, p, s) for gi, p in queries for s in range(K) if (gi, s) not in done]
    print(f"[rob] queries={len(queries)} pending={len(todo)}", flush=True)

    workers = int(os.getenv("LABEL_WORKERS", "8"))
    batch = workers * 6
    with open(OUT, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            futs = {ex.submit(sample_once, p): (gi, s) for gi, p, s in chunk}
            for fut in cf.as_completed(futs):
                gi, s = futs[fut]
                r = fut.result()
                if not r.get("ok"):
                    continue
                f.write(json.dumps({"gi": gi, "s": s, "boxed": boxed(r["text"]), "in": r["in"], "out": r["out"]}) + "\n")
                f.flush()
            print(f"[rob] {min(i + batch, len(todo))}/{len(todo)} attempted", flush=True)

    # build predictions
    raw = collections.defaultdict(list)
    for line in open(OUT):
        r = json.loads(line); raw[r["gi"]].append((r["s"], norm(r["boxed"])))
    out = []
    esc = 0
    for e in data:
        gi = e["global index"]
        rows = sorted(raw.get(gi, []))[:K]
        boxes = [nb for _, nb in rows if nb]
        if boxes and len(rows) >= 1:
            top = collections.Counter(boxes).most_common(1)[0][1]
            cons = top / len(rows)
        else:
            cons = 0.0
        pred = QWEN if cons >= TAU else DEEPSEEK
        if pred == DEEPSEEK:
            esc += 1
        out.append({"global index": gi, "prompt": e["prompt_formatted"], "prediction": pred,
                    "generated_result": None, "cost": None, "accuracy": None, "for_optimality": False})
    json.dump(out, open("router_inference/predictions/cruq-router-robustness.json", "w"))
    print(f"[rob] wrote cruq-router-robustness.json: {len(out)} entries, escalated={esc} ({esc/len(out):.1%})")


if __name__ == "__main__":
    main()
