# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Anthropic Message Batches inference (Anthropic-only, 50%-off async batch API).

Adds a new model (e.g. Opus 4.8) as a candidate by running batch inference over a
list of `global_index` queries and writing results into a per-model JSONL whose
schema matches the existing 9-model cache (prepare_data/train_cached_results*):
    global_index, question, llm_selected, generated_answer, token_usage,
    success, provider, error, run_number, evaluation_result(=None; filled by llm_evaluation later)

Four subcommands (run in order):
  prepare  build + validate the batch requests, write prepared_requests.jsonl + batch_meta.json
           (NO API call, NO spend) — inspect before submitting.
  create   submit the batch to Anthropic (needs ANTHROPIC_API_KEY); records batch_id in meta.
  status   poll processing_status + request_counts.
  fetch    stream results once ended → write <out_name>.jsonl into the cache dir.

Example:
  uv run python llm_inference/anthropic_batch_inference.py prepare \
      --sample /scratch/yl231/henry-shan/opus_sample_2000.txt \
      --model claude-opus-4-8 --max-tokens 2048
  uv run python llm_inference/anthropic_batch_inference.py create  --workdir <wd>
  uv run python llm_inference/anthropic_batch_inference.py status  --workdir <wd>
  uv run python llm_inference/anthropic_batch_inference.py fetch   --workdir <wd>
"""

import argparse
import json
import os
import re
import sys
import time
import datetime
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("anthropic_batch")

CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

DEFAULT_PROMPT_SOURCE = (
    "/home/yl231/prepare_data/train_cached_results_300k/deepseek_deepseek-v3.2.jsonl"
)
DEFAULT_CACHE_DIR = (
    "/scratch/yl231/henry-shan/prepare_data/train_cached_results_300k_plus"
)
DEFAULT_WORKROOT = "/scratch/yl231/henry-shan/opus_batch"


# ----------------------------- helpers -----------------------------
def load_sample_ids(path):
    ids = [x.strip() for x in open(path, encoding="utf-8") if x.strip()]
    # de-dup, preserve order
    seen, out = set(), []
    for g in ids:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def load_prompts(ids, prompt_source):
    """Map each global_index -> its prompt text (the `question` field) by streaming
    an existing per-model JSONL. Guarantees identical prompt to what other models saw."""
    want = set(ids)
    id_re = re.compile(rb'"global_index":\s*"([^"]*)"')
    got = {}
    with open(prompt_source, "rb") as f:
        for line in f:
            m = id_re.search(line)
            if not m:
                continue
            g = m.group(1).decode()
            if g in want and g not in got:
                got[g] = json.loads(line).get("question", "")
                if len(got) == len(want):
                    break
    missing = [g for g in ids if g not in got]
    return got, missing


def build_prepared(ids, id2prompt):
    """Build the list of prepared request rows with safe custom_ids.
    custom_id = req_<i> (index in sample order) — always matches ^[A-Za-z0-9_-]{1,64}$;
    global_index can contain spaces/dots so it cannot be used directly."""
    prepared = []
    for i, g in enumerate(ids):
        cid = f"req_{i}"
        assert CUSTOM_ID_RE.match(cid), cid
        prepared.append({"custom_id": cid, "global_index": g, "question": id2prompt[g]})
    return prepared


def write_meta(workdir, meta):
    with open(os.path.join(workdir, "batch_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def read_meta(workdir):
    with open(os.path.join(workdir, "batch_meta.json")) as f:
        return json.load(f)


def read_prepared(workdir):
    rows = []
    with open(os.path.join(workdir, "prepared_requests.jsonl")) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def anthropic_requests(prepared, model, max_tokens):
    """Turn prepared rows into Anthropic batch Request objects."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    reqs = []
    for row in prepared:
        reqs.append(
            Request(
                custom_id=row["custom_id"],
                params=MessageCreateParamsNonStreaming(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": row["question"]}],
                ),
            )
        )
    return reqs


# ----------------------------- subcommands -----------------------------
def cmd_prepare(args):
    ids = load_sample_ids(args.sample)
    logger.info(f"loaded {len(ids)} sample ids from {args.sample}")
    id2prompt, missing = load_prompts(ids, args.prompt_source)
    if missing:
        logger.warning(
            f"{len(missing)} ids had no prompt in source; dropping them. "
            f"e.g. {missing[:3]}"
        )
        ids = [g for g in ids if g in id2prompt]
    prepared = build_prepared(ids, id2prompt)

    os.makedirs(args.workdir, exist_ok=True)
    with open(os.path.join(args.workdir, "prepared_requests.jsonl"), "w") as f:
        for row in prepared:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

    meta = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "out_name": args.out_name or f"anthropic_{args.model}",
        "cache_dir": args.cache_dir,
        "prompt_source": args.prompt_source,
        "sample_file": os.path.abspath(args.sample),
        "n_requests": len(prepared),
        "batch_id": None,
        "created_at": None,
        "prepared_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    write_meta(args.workdir, meta)

    # quick token sanity (chars/4 heuristic — just a sanity print, not billing)
    approx_in = sum(len(r["question"]) for r in prepared) / 4
    logger.info("=" * 60)
    logger.info(f"PREPARED {len(prepared)} requests (NO API call made).")
    logger.info(f"  model={args.model}  max_tokens={args.max_tokens}")
    logger.info(f"  out file (on fetch): {meta['cache_dir']}/{meta['out_name']}.jsonl")
    logger.info(f"  workdir: {args.workdir}")
    logger.info(f"  ~input tokens (chars/4 heuristic): {approx_in:,.0f}")
    logger.info("  custom_ids validated against ^[A-Za-z0-9_-]{1,64}$ (req_<i>).")
    logger.info(
        "Next: set ANTHROPIC_API_KEY in .env, then run `create --workdir <wd>`."
    )
    logger.info("=" * 60)


def _client():
    import anthropic

    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        logger.error(
            "ANTHROPIC_API_KEY not set (put it in RouterArena/.env). Aborting."
        )
        sys.exit(2)
    return anthropic.Anthropic(api_key=key)


def cmd_create(args):
    meta = read_meta(args.workdir)
    if meta.get("batch_id"):
        logger.error(f"batch already created: {meta['batch_id']} (delete meta to redo)")
        sys.exit(1)
    prepared = read_prepared(args.workdir)
    logger.info(
        f"submitting {len(prepared)} requests as one Message Batch "
        f"(model={meta['model']}, max_tokens={meta['max_tokens']})"
    )
    client = _client()
    reqs = anthropic_requests(prepared, meta["model"], meta["max_tokens"])
    batch = client.messages.batches.create(requests=reqs)
    meta["batch_id"] = batch.id
    meta["created_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    write_meta(args.workdir, meta)
    logger.info(f"CREATED batch {batch.id}  status={batch.processing_status}")
    logger.info(f"poll with: status --workdir {args.workdir}")


def cmd_status(args):
    meta = read_meta(args.workdir)
    if not meta.get("batch_id"):
        logger.error("no batch_id in meta; run create first.")
        sys.exit(1)
    client = _client()
    b = client.messages.batches.retrieve(meta["batch_id"])
    logger.info(f"batch {b.id}: processing_status={b.processing_status}")
    logger.info(f"  request_counts={b.request_counts}")
    if args.wait:
        while b.processing_status != "ended":
            logger.info("  still processing… sleeping 60s")
            time.sleep(60)
            b = client.messages.batches.retrieve(meta["batch_id"])
        logger.info(f"ENDED. counts={b.request_counts}")


def cmd_fetch(args):
    meta = read_meta(args.workdir)
    if not meta.get("batch_id"):
        logger.error("no batch_id in meta; run create first.")
        sys.exit(1)
    prepared = read_prepared(args.workdir)
    cid2row = {r["custom_id"]: r for r in prepared}
    client = _client()

    out_dir = meta["cache_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{meta['out_name']}.jsonl")
    n_ok = n_err = n_other = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for result in client.messages.batches.results(meta["batch_id"]):
            row = cid2row.get(result.custom_id, {})
            gid = row.get("global_index", result.custom_id)
            question = row.get("question", "")
            rtype = result.result.type
            rec = {
                "global_index": gid,
                "question": question,
                "llm_selected": meta["out_name"],
                "generated_answer": "",
                "token_usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
                "success": False,
                "provider": "anthropic",
                "error": None,
                "run_number": 1,
                "evaluation_result": None,
            }
            if rtype == "succeeded":
                msg = result.result.message
                text = "".join(
                    getattr(b, "text", "")
                    for b in msg.content
                    if getattr(b, "type", None) == "text"
                )
                u = msg.usage
                in_tok = getattr(u, "input_tokens", 0)
                out_tok = getattr(u, "output_tokens", 0)
                rec.update(
                    success=True,
                    generated_answer=text,
                    token_usage={
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "total_tokens": in_tok + out_tok,
                    },
                )
                n_ok += 1
            elif rtype == "errored":
                rec["error"] = str(getattr(result.result, "error", "errored"))
                n_err += 1
            else:  # expired / canceled
                rec["error"] = rtype
                n_other += 1
            json.dump(rec, out, ensure_ascii=False)
            out.write("\n")
    logger.info(f"WROTE {n_ok + n_err + n_other} records -> {out_path}")
    logger.info(f"  succeeded={n_ok}  errored={n_err}  expired/canceled={n_other}")
    logger.info("Next: run llm_evaluation to fill evaluation_result (score/metric).")


def main():
    p = argparse.ArgumentParser(description="Anthropic-only Message Batches inference")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="build + validate requests (no API call)")
    pp.add_argument("--sample", required=True, help="file: one global_index per line")
    pp.add_argument("--model", default="claude-opus-4-8")
    pp.add_argument("--max-tokens", type=int, default=2048)
    pp.add_argument(
        "--prompt-source",
        default=DEFAULT_PROMPT_SOURCE,
        help="existing per-model jsonl to read `question` from",
    )
    pp.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="output folder for the per-model jsonl (on fetch)",
    )
    pp.add_argument(
        "--out-name",
        default=None,
        help="jsonl basename / llm_selected (default anthropic_<model>)",
    )
    pp.add_argument("--workdir", default=None)
    pp.set_defaults(func=cmd_prepare)

    for name, fn, helptxt in [
        ("create", cmd_create, "submit the batch (needs key)"),
        ("status", cmd_status, "poll status"),
        ("fetch", cmd_fetch, "download results -> jsonl"),
    ]:
        sp = sub.add_parser(name, help=helptxt)
        sp.add_argument("--workdir", required=True)
        if name == "status":
            sp.add_argument("--wait", action="store_true", help="poll until ended")
        sp.set_defaults(func=fn)

    args = p.parse_args()

    # default workdir per sample/model for prepare
    if args.cmd == "prepare" and not args.workdir:
        base = os.path.splitext(os.path.basename(args.sample))[0]
        args.workdir = os.path.join(DEFAULT_WORKROOT, f"{base}__{args.model}")

    # load .env so ANTHROPIC_API_KEY is available for create/status/fetch
    try:
        from dotenv import load_dotenv

        load_dotenv(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
        )
    except Exception:
        pass

    args.func(args)


if __name__ == "__main__":
    main()
