#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0
"""Generate Kilo Auto Efficient submissions without training or local routing.

Run from any directory; paths are relative to this repository. Credentials and
full completion URL come only from KILO_API_KEY and KILO_API_URL. No API calls
are made by --help. Response parsing uses the OpenAI chat-completion contract;
a successful live Kilo response must be verified by the operator before a run.

An optional order seed shuffles scheduling only, never the prompts or output.
Keep that seed unchanged on every resume. Explicit HTTP429/500/502/503/504
transport failures are journaled without inventing a model or generation.
Nonadaptive batches stop scheduling, drain active requests, and return 75 for a
lower-concurrency resume. Adaptive mode instead halves concurrency on the first
such error, stops ramping, and continues unrelated requests with timed windows.
Its final "adaptive: " JSON line reports the concurrency to retain on resume.
Other transport failures are fatal and block automatic journal replay.
Successful and non-provider-error results are never eligible for regeneration.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import tempfile
import threading
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ModelNameManager = importlib.import_module("universal_model_names").ModelNameManager

ROUTER = "kilo-auto-efficient"
MODEL = "kilo-auto/efficient"
RETRYABLE_HTTP_STATUSES = (429, 500, 502, 503, 504)
SPLITS = {
    "full": ("router_data.json", 8400),
    "robustness": ("router_robustness.json", 420),
}


class CompletionTransportError(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the bearer credential to a redirect destination.
        return None


def fetch(endpoint, key, row, timeout, split):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": row["prompt_formatted"]}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    organization_id = os.environ.get("KILO_ORGANIZATION_ID")
    if organization_id:
        headers["x-kilocode-organizationid"] = organization_id
    run_id = os.environ["KILO_RUN_ID"]
    request_id = f"{run_id}:{uuid.uuid4().hex}"
    headers["x-kilo-request"] = request_id
    headers["x-kilo-session"] = f"{run_id}:{split}:{row['global index']}"
    headers["User-Agent"] = f"RouterArena/{run_id}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.monotonic()
    body = None
    try:
        with urllib.request.build_opener(NoRedirect).open(
            request, timeout=timeout
        ) as response:
            if split == "robustness":
                event_lines = []
                for raw_line in response:
                    line = raw_line.decode("utf-8")
                    if line.startswith("data:"):
                        event_lines.append(line[5:].strip())
                    elif not line.strip() and event_lines:
                        data = "\n".join(event_lines)
                        event_lines = []
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)
                        if isinstance(chunk.get("model"), str) and not chunk[
                            "model"
                        ].startswith("kilo-auto/"):
                            body = json.dumps(chunk)
                            break
                        if chunk.get("error"):
                            raise ValueError(
                                "Stream failed before exposing a selected model"
                            )
                else:
                    raise ValueError("Stream ended before exposing a selected model")
                if not body:
                    raise ValueError("Stream ended before exposing a selected model")
            else:
                body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.close()
        raise CompletionTransportError(
            f"Completion request failed: HTTP {status}", status
        ) from None
    except Exception as exc:
        # Exception text and server error bodies may contain sensitive headers.
        raise CompletionTransportError(
            f"Completion transport failed ({type(exc).__name__})"
        ) from None
    return {
        "global_index": row["global index"],
        "request_id": request_id,
        "response_body": body,
        "transport": "stream-model-selection" if split == "robustness" else "stream",
        "completion_duration_seconds": time.monotonic() - started,
    }


def completion_response(body):
    if body.lstrip().startswith("{"):
        return json.loads(body)
    response: dict[str, Any] = {"object": "chat.completion", "choices": []}
    content = []
    finish_reason = None
    done = False
    event_lines = []
    for line in body.splitlines() + [""]:
        if line.startswith("data:"):
            event_lines.append(line[5:].lstrip())
        elif not line and event_lines:
            data = "\n".join(event_lines)
            event_lines = []
            if data == "[DONE]":
                done = True
                break
            chunk = json.loads(data)
            if not isinstance(chunk, dict):
                raise ValueError("Streaming completion event must be a JSON object")
            if chunk.get("error"):
                response["error"] = chunk["error"]
                break
            for name in ("id", "model", "created"):
                value = chunk.get(name)
                if value is not None:
                    if (
                        name in ("id", "model")
                        and name in response
                        and response[name] != value
                    ):
                        raise ValueError(f"Streaming completion changed its {name}")
                    response[name] = value
            if chunk.get("usage") is not None:
                response["usage"] = chunk["usage"]
            for choice in chunk.get("choices", []):
                if choice.get("index", 0) != 0:
                    continue
                delta = choice.get("delta") or {}
                text = delta.get("content")
                if text is not None:
                    if not isinstance(text, str):
                        raise ValueError("Streaming content must be text")
                    content.append(text)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
    if not response.get("error") and (not done or finish_reason is None):
        raise ValueError("Streaming completion ended without a final completion event")
    response["choices"] = [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "".join(content)},
            "finish_reason": finish_reason,
        }
    ]
    return response


def prediction(row, event, split, allowed):
    response = completion_response(event["response_body"])
    if not isinstance(response, dict):
        raise ValueError("Completion response must be a JSON object")
    raw_model = response.get("model")
    if (
        not isinstance(raw_model, str)
        or not raw_model
        or raw_model.startswith("kilo-auto/")
    ):
        raise ValueError(
            "Endpoint did not expose a concrete selected model in response.model"
        )
    # Only aliases already recorded by RouterArena are permitted; never infer
    # equivalence from a provider prefix, version suffix, or reasoning variant.
    try:
        selected = ModelNameManager.get_universal_name(raw_model)
    except ValueError:
        raise ValueError(
            "response.model is not registered; no model normalization was guessed"
        ) from None
    if selected not in allowed:
        raise ValueError("response.model is outside the configured Kilo model pool")
    result = {
        "global_index": row["global index"],
        "prompt": row["prompt_formatted"],
        "prediction": selected,
    }
    metadata = {
        "requested_model": MODEL,
        "response_model": raw_model,
        "response_id": response.get("id"),
        "completion_duration_seconds": event["completion_duration_seconds"],
        "usage_raw": response.get("usage"),
        "transport": event.get("transport", "non-streaming"),
        **({"generation_error": response["error"]} if response.get("error") else {}),
    }
    if split == "robustness":
        # Selection is visible before generation ends. Any early inference work
        # can still incur cost; this is not routing-only latency.
        result["routing_metadata"] = metadata
        return result
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("Completion response has no first choice")
    message = choices[0].get("message")
    answer = message.get("content") if isinstance(message, dict) else None
    usage = response.get("usage")
    valid_usage = isinstance(usage, dict) and all(
        isinstance(usage.get(name), int)
        and not isinstance(usage[name], bool)
        and usage[name] >= 0
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
    )
    token_usage = None
    if isinstance(usage, dict) and valid_usage and usage["completion_tokens"] > 0:
        token_usage = {
            "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
        }
    generation_error = response.get("error")
    if not generation_error and (not isinstance(answer, str) or not answer.strip()):
        generation_error = {
            "type": "empty_answer",
            "message": "Provider completed without an answer",
        }
    if not generation_error and token_usage is None:
        generation_error = {
            "type": "missing_usage",
            "message": "Provider omitted usable token accounting",
        }
    result["generated_result"] = {
        "generated_answer": answer if isinstance(answer, str) else "",
        "success": not bool(generation_error),
        "token_usage": token_usage,
        "model_used": selected,
        "provider": "kilo",
        "finish_reason": choices[0].get("finish_reason"),
        **metadata,
        **({"generation_error": generation_error} if generation_error else {}),
    }
    return result


def append_event(journal, event):
    journal.write(json.dumps(event, ensure_ascii=False) + "\n")
    journal.flush()
    os.fsync(journal.fileno())


def export_predictions(path, rows, completed):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        delete=False,
    ) as output:
        temporary = Path(output.name)
        try:
            json.dump(
                [
                    completed[row["global index"]]
                    for row in rows
                    if row["global index"] in completed
                ],
                output,
                ensure_ascii=False,
                indent=2,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def provider_error_retryable(result):
    generation = result.get("generated_result") or {}
    error = generation.get("generation_error") or {}
    return (
        generation.get("success") is False
        and isinstance(error, dict)
        and (
            error.get("type") == "timeout"
            or error.get("code") in (429, 500, 502, 503, 504)
        )
    )


class AdaptiveConcurrency:
    def __init__(self, concurrency, capacity, fallback_model):
        self.concurrency = min(concurrency, capacity)
        self.capacity = capacity
        self.fallback_model = fallback_model
        self.ramping = self.concurrency < capacity
        self.limit_reason = None
        self.transport_limited = False
        self.previous_throughput = None
        self.doubling_from = None
        self.low_gain_windows = 0
        self.window_started = time.monotonic()
        self.window_completed = 0
        self.provider_errors = 0
        self.fallback_responses = 0

    def record(self, result=None, transport_status=None):
        self.window_completed += 1
        if result is not None:
            self.provider_errors += int(provider_error_retryable(result))
            self.fallback_responses += int(
                self.fallback_model is not None
                and result["prediction"] == self.fallback_model
            )
        elif transport_status in RETRYABLE_HTTP_STATUSES:
            self.provider_errors += 1

    def throughput(self, now):
        return self.window_completed * 60 / max(now - self.window_started, 1e-9)

    def emit(self, completed_rows, previous_concurrency=None, now=None):
        now = time.monotonic() if now is None else now
        print(
            "adaptive: "
            + json.dumps(
                {
                    "concurrency": self.concurrency,
                    "previousConcurrency": (
                        self.concurrency
                        if previous_concurrency is None
                        else previous_concurrency
                    ),
                    "completedRows": completed_rows,
                    "windowCompleted": self.window_completed,
                    "throughputPerMinute": round(self.throughput(now), 3),
                    "providerErrors": self.provider_errors,
                    "fallbackResponses": self.fallback_responses,
                    "limitReason": self.limit_reason,
                    "ramping": self.ramping,
                }
            ),
            flush=True,
        )

    def wait_timeout(self):
        if self.window_completed >= 8:
            return max(0, 45 - (time.monotonic() - self.window_started))
        return None

    def transport_limit(self, status, completed_rows):
        if self.transport_limited:
            return
        previous = self.concurrency
        self.concurrency = max(1, previous // 2)
        self.ramping = False
        self.transport_limited = True
        self.limit_reason = f"transport_http_{status}"
        self.emit(completed_rows, previous)

    def evaluate(self, completed_rows):
        now = time.monotonic()
        if self.window_completed < 8 or now - self.window_started < 45:
            return
        previous = self.concurrency
        throughput = self.throughput(now)
        if self.provider_errors / self.window_completed >= 0.05:
            self.limit_reason = "provider_error_rate"
            self.concurrency = max(1, previous // 2)
            self.ramping = False
        elif (
            self.fallback_responses >= 3
            and self.fallback_responses / self.window_completed >= 0.03
        ):
            self.limit_reason = "fallback_rate"
            self.concurrency = max(1, previous // 2)
            self.ramping = False
        elif self.ramping:
            if self.previous_throughput is not None:
                self.low_gain_windows = (
                    self.low_gain_windows + 1
                    if throughput < self.previous_throughput * 1.1
                    else 0
                )
            if self.low_gain_windows >= 2:
                self.concurrency = self.doubling_from
                self.ramping = False
                self.limit_reason = "throughput_plateau"
            else:
                self.doubling_from = previous
                self.previous_throughput = throughput
                self.concurrency = min(self.capacity, previous * 2)
                if self.concurrency == self.capacity:
                    self.ramping = False
                    self.limit_reason = "concurrency_capacity"
        self.emit(completed_rows, previous, now)
        self.window_started = now
        self.window_completed = 0
        self.provider_errors = 0
        self.fallback_responses = 0


def run(args):
    endpoint = os.environ.get("KILO_API_URL", "")
    key = os.environ.get("KILO_API_KEY", "")
    run_id = os.environ.get("KILO_RUN_ID")
    if not run_id:
        raise ValueError("Set KILO_RUN_ID to a unique identifier for this run")
    url = urllib.parse.urlsplit(endpoint)
    if (
        not key
        or url.scheme != "https"
        or not url.hostname
        or url.username
        or url.password
    ):
        raise ValueError(
            "Set KILO_API_KEY and KILO_API_URL (full HTTPS chat/completions URL) in the environment"
        )
    dataset_file, expected = SPLITS[args.split]
    dataset_bytes = (ROOT / "dataset" / dataset_file).read_bytes()
    rows = json.loads(dataset_bytes)
    if not isinstance(rows, list) or len(rows) != expected:
        raise ValueError(f"Prepared {args.split} dataset must contain {expected} rows")
    by_index = {}
    for row in rows:
        index = row.get("global index")
        prompt = row.get("prompt_formatted")
        if not isinstance(index, str) or not index or index in by_index:
            raise ValueError("Dataset has a missing or duplicate global index")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"Dataset prompt is empty for {index}")
        by_index[index] = row
    config = json.loads(
        (ROOT / "router_inference" / "config" / f"{ROUTER}.json").read_text()
    )
    allowed = {
        ModelNameManager.get_universal_name(model)
        for model in config["pipeline_params"]["models"]
    }
    if not allowed:
        raise ValueError("Configured model pool is empty")
    header = {
        "format_version": 1,
        "split": args.split,
        "run_id": run_id,
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "endpoint": endpoint,
        "request_model": MODEL,
        "request_parameters": "one unchanged user message; no generation overrides",
        "allowed_models": sorted(allowed),
    }
    scheduled_rows = rows
    if args.order_seed is not None:
        header["order_seed"] = args.order_seed
        scheduled_rows = rows.copy()
        random.Random(args.order_seed).shuffle(scheduled_rows)
    checkpoint = ROOT / "cached_results" / ROUTER / f"{args.split}.jsonl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    filename = ROUTER + ("-robustness" if args.split == "robustness" else "") + ".json"
    output = ROOT / "router_inference" / "predictions" / filename
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    completed: dict[str, dict[str, Any]] = {}
    attempts: dict[str, int] = {}
    failed = False
    transport_failed = False
    with checkpoint.open("a+", encoding="utf-8") as journal:
        try:
            fcntl.flock(journal, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(
                "Another driver is using this split's checkpoint"
            ) from None
        journal.seek(0)
        first = journal.readline()
        if first:
            previous_header = json.loads(first)
            comparable_header = dict(previous_header)
            fallback_model = config["pipeline_params"].get("fallback_model")
            # A declared static fallback extends acceptance, not gateway routing.
            # Keep the original header and every captured response unchanged.
            if fallback_model and comparable_header.get("allowed_models") == sorted(
                allowed - {fallback_model}
            ):
                comparable_header["allowed_models"] = sorted(allowed)
            if comparable_header != header:
                raise ValueError(
                    "Checkpoint dataset, endpoint, request, model pool, or order seed differs; do not mix runs"
                )
            for line in journal:
                if not line.endswith("\n"):
                    raise ValueError(
                        "Checkpoint has an incomplete final line; preserve it and repair before resuming"
                    )
                event = json.loads(line)
                index = event["global_index"]
                attempt = event.get("attempt", 1)
                if (
                    index not in by_index
                    or not isinstance(attempt, int)
                    or isinstance(attempt, bool)
                ):
                    raise ValueError("Checkpoint contains an invalid index or attempt")
                if attempt != attempts.get(index, 0) + 1:
                    raise ValueError("Checkpoint attempts are not sequential")
                if index in completed and not provider_error_retryable(
                    completed[index]
                ):
                    raise ValueError(
                        "Checkpoint retries a successful or non-provider-error result"
                    )
                if "transport_error" in event:
                    transport_error = event["transport_error"]
                    if (
                        not isinstance(transport_error, dict)
                        or transport_error.get("http_status")
                        not in RETRYABLE_HTTP_STATUSES
                    ):
                        raise ValueError(
                            "Checkpoint contains a fatal transport error; investigate before resuming"
                        )
                    if index in completed:
                        completed[index]["generated_result"].setdefault(
                            "retry_transport_errors", []
                        ).append(transport_error)
                else:
                    completed[index] = prediction(
                        by_index[index], event, args.split, allowed
                    )
                attempts[index] = attempt
                if args.split == "full" and index in completed:
                    completed[index]["generated_result"]["captured_attempts"] = attempt
        else:
            if output.exists():
                raise ValueError(
                    "Prediction file exists without a checkpoint; refusing to overwrite it"
                )
            append_event(journal, header)
        if args.retry_provider_errors:
            if len(completed) != expected:
                raise ValueError(
                    "Finish the first pass before retrying provider errors"
                )
            remaining = [
                row
                for row in scheduled_rows
                if provider_error_retryable(completed[row["global index"]])
            ]
        else:
            remaining = [
                row for row in scheduled_rows if row["global index"] not in completed
            ]
        if args.limit is not None:
            remaining = remaining[: args.limit]
        print(
            f"{args.split}: {len(completed)}/{expected} checkpointed; {len(remaining)} requests scheduled",
            flush=True,
        )
        if args.split == "robustness":
            print(
                "Robustness streams close after model selection; partial inference can still incur cost.",
                flush=True,
            )
        work = iter(remaining)
        pending: dict[Future[dict[str, Any]], dict[str, Any]] = {}
        adaptive = None
        if args.adaptive_concurrency:
            fallback_model = config["pipeline_params"].get("fallback_model")
            adaptive = AdaptiveConcurrency(
                args.concurrency,
                expected,
                ModelNameManager.get_universal_name(fallback_model)
                if fallback_model
                else None,
            )
        with ThreadPoolExecutor(
            max_workers=expected if adaptive is not None else args.concurrency
        ) as executor:
            while True:
                concurrency = (
                    adaptive.concurrency if adaptive is not None else args.concurrency
                )
                while not stop.is_set() and len(pending) < concurrency:
                    row = next(work, None)
                    if row is None:
                        break
                    future = executor.submit(
                        fetch, endpoint, key, row, args.timeout, args.split
                    )
                    pending[future] = row
                if not pending:
                    break
                done, _ = wait(
                    pending,
                    timeout=adaptive.wait_timeout() if adaptive is not None else None,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    row = pending.pop(future)
                    try:
                        event = future.result()
                        event["attempt"] = attempts.get(row["global index"], 0) + 1
                        # Persist even malformed/unsupported responses before parsing.
                        # A parsing failure must be investigated, not paid for again.
                        append_event(journal, event)
                        completed[row["global index"]] = prediction(
                            row, event, args.split, allowed
                        )
                        attempts[row["global index"]] = event["attempt"]
                        if args.split == "full":
                            completed[row["global index"]]["generated_result"][
                                "captured_attempts"
                            ] = event["attempt"]
                        if adaptive is not None:
                            adaptive.record(completed[row["global index"]])
                    except Exception as exc:
                        if adaptive is not None:
                            adaptive.record(
                                transport_status=exc.status
                                if isinstance(exc, CompletionTransportError)
                                else None
                            )
                        if isinstance(exc, CompletionTransportError):
                            index = row["global index"]
                            attempt = attempts.get(index, 0) + 1
                            transport_error = {
                                "message": str(exc),
                                "http_status": exc.status,
                            }
                            append_event(
                                journal,
                                {
                                    "global_index": index,
                                    "attempt": attempt,
                                    "transport_error": transport_error,
                                },
                            )
                            attempts[index] = attempt
                            if index in completed:
                                generation = completed[index]["generated_result"]
                                generation["captured_attempts"] = attempt
                                generation.setdefault(
                                    "retry_transport_errors", []
                                ).append(transport_error)
                            if exc.status in RETRYABLE_HTTP_STATUSES:
                                transport_failed = True
                                if adaptive is not None:
                                    adaptive.transport_limit(exc.status, len(completed))
                                    message = "leaving query unfinished; unrelated scheduling continues"
                                else:
                                    stop.set()
                                    message = (
                                        "scheduling stopped, draining active requests"
                                    )
                                print(
                                    f"{index}: {exc}; {message}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                continue
                        failed = True
                        stop.set()
                        print(
                            f"{row['global index']}: {exc}", file=sys.stderr, flush=True
                        )
                if adaptive is not None:
                    adaptive.evaluate(len(completed))
                if len(completed) % 25 == 0:
                    export_predictions(output, rows, completed)
        export_predictions(output, rows, completed)
    print(f"Wrote {len(completed)}/{expected} rows to {output}", flush=True)
    if len(completed) != expected:
        print(
            "INCOMPLETE: not ready for submission. Rerun to resume; inspect any response-contract failures first.",
            flush=True,
        )
    if adaptive is not None:
        adaptive.emit(len(completed))
    if failed:
        return 1
    if transport_failed:
        return 75
    return 130 if stop.is_set() else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=SPLITS)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Maximum concurrent paid requests (default: 4)",
    )
    parser.add_argument(
        "--adaptive-concurrency",
        action="store_true",
        help="Ramp with 45-second outcome windows; retain all in-flight responses",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="Per-request network timeout in seconds",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run at most this many unfinished rows; output will be partial",
    )
    parser.add_argument(
        "--order-seed",
        type=int,
        help="Deterministically shuffle request scheduling; required unchanged on resume",
    )
    parser.add_argument(
        "--retry-provider-errors",
        action="store_true",
        help="Retry only explicit provider errors; keep every successful generation",
    )
    args = parser.parse_args()
    if args.retry_provider_errors and args.split != "full":
        parser.error("Generation retries apply only to the full split")
    if (
        args.concurrency < 1
        or args.timeout <= 0
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("concurrency, timeout, and limit must be positive")
    try:
        return run(args)
    except Exception as exc:
        print(f"Submission stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
