# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""Refit the local-model price rule (model_cost/local_pricing.yaml) from market data.

Reads every OpenRouter model that names its Hugging Face weights, its per-provider
endpoints, and the parameter count from Hugging Face. Each model's standard price is the
median over its standard-tier hosts at 8 bits or more, with promotions undone. Then:

* size curve: log10(price) = log10(coef) + exponent * log10(active parameters, billions),
  fit by least squares, for input and output separately. Mixture-of-experts models count
  only when their id names the active size (e.g. ``-a22b``);
* quantization: for models served at both precisions, the ratio of the median 4-bit
  host price to the median 8/16-bit one, and of 8-bit to 16-bit.

It prints the fitted numbers; a maintainer copies them into local_pricing.yaml (the
quantization factors are a policy choice informed by the ratios). Takes a few minutes.

Usage:
    python scripts/pricing/fit_local_rule.py [--save raw.jsonl] [--load raw.jsonl]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

BITS = {"bf16": 16, "fp16": 16, "fp32": 16, "fp8": 8, "int8": 8}
BITS.update({q: 4 for q in ("fp4", "nvfp4", "mxfp4", "int4")})
MOE_TYPES = {
    "deepseek_v3",
    "deepseek_v4",
    "gpt_oss",
    "mixtral",
    "llama4",
    "dbrx",
    "jamba",
    "glm_moe_dsa",
    "glm4_moe",
    "minimax_m2",
    "minimax_m3",
    "minimax_m3_vl",
    "kimi_k2",
    "qwen3_moe",
    "qwen3_next",
}


def get(url: str) -> Any:
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "routerarena-local-price-fit"}
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except Exception:
            time.sleep(2 * (attempt + 1))
    return {}


def collect() -> List[Dict[str, Any]]:
    models = get("https://openrouter.ai/api/v1/models").get("data", [])
    records = []
    for m in models:
        if not m.get("hugging_face_id") or m["id"].endswith(":free"):
            continue
        endpoints = (
            get(f"https://openrouter.ai/api/v1/models/{m['id']}/endpoints").get("data")
            or {}
        ).get("endpoints", [])
        hf = get(f"https://huggingface.co/api/models/{m['hugging_face_id']}")
        records.append(
            {
                "id": m["id"],
                "params_total": (hf.get("safetensors") or {}).get("total"),
                "model_type": (hf.get("config") or {}).get("model_type"),
                "endpoints": [
                    {
                        "quantization": e.get("quantization"),
                        "tier": e.get("service_tier"),
                        "status": e.get("status"),
                        "prompt": (e.get("pricing") or {}).get("prompt"),
                        "completion": (e.get("pricing") or {}).get("completion"),
                        "discount": (e.get("pricing") or {}).get("discount"),
                    }
                    for e in endpoints
                ],
            }
        )
        time.sleep(0.3)
    return records


def active_params_b(record: Dict[str, Any]) -> Optional[float]:
    named = re.search(r"[-_]a(\d+(?:\.\d+)?)b", record["id"].lower())
    if named:
        return float(named.group(1))
    model_type = (record.get("model_type") or "").lower()
    if model_type in MOE_TYPES or "moe" in model_type or not record.get("params_total"):
        return None
    return record["params_total"] / 1e9


def standard_endpoints(record: Dict[str, Any]) -> List[tuple]:
    out = []
    for e in record["endpoints"]:
        if e.get("tier") not in (None, "standard"):
            continue
        try:
            p_in, p_out = float(e["prompt"]) * 1e6, float(e["completion"]) * 1e6
        except (TypeError, ValueError):
            continue
        if p_in <= 0 or p_out <= 0:
            continue
        discount = float(e.get("discount") or 0)
        factor = 1 / (1 - discount) if 0 < discount < 1 else 1.0
        active = (e.get("status") or 0) >= 0
        out.append(
            (
                BITS.get(str(e.get("quantization"))),
                p_in * factor,
                p_out * factor,
                active,
            )
        )
    return [e for e in out if e[3]] or out


def fit(records: List[Dict[str, Any]]) -> None:
    points: List[Tuple[float, float, float]] = []
    ratio4: List[Tuple[float, float]] = []
    ratio8: List[Tuple[float, float]] = []
    for r in records:
        eps = standard_endpoints(r)
        full = [e for e in eps if e[0] != 4]
        size = active_params_b(r)
        if full and size:
            points.append(
                (
                    size,
                    statistics.median(e[1] for e in full),
                    statistics.median(e[2] for e in full),
                )
            )
        for low_bits, high_bits, store in (
            (4, (8, 16, None), ratio4),
            (8, (16,), ratio8),
        ):
            low = [e for e in eps if e[0] == low_bits]
            high = [e for e in eps if e[0] in high_bits]
            if low and high:
                store.append(
                    (
                        statistics.median(e[1] for e in low)
                        / statistics.median(e[1] for e in high),
                        statistics.median(e[2] for e in low)
                        / statistics.median(e[2] for e in high),
                    )
                )
    print(f"{len(points)} models with a parameter count and a full-precision price")
    print("size_curve:")
    for k, name in ((1, "input"), (2, "output")):
        xs = [math.log10(p[0]) for p in points]
        ys = [math.log10(p[k]) for p in points]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum(
            (x - mx) ** 2 for x in xs
        )
        a = my - b * mx
        spread = statistics.pstdev([y - (a + b * x) for x, y in zip(xs, ys)])
        print(
            f"  {name}: {{coef: {10**a:.4f}, exponent: {b:.3f}}}  # residual x{10**spread:.2f}"
        )
    for label, ratios in (("4-bit vs 8/16-bit", ratio4), ("8-bit vs 16-bit", ratio8)):
        if ratios:
            print(
                f"{label}: {len(ratios)} models, median ratio "
                f"input {statistics.median(r[0] for r in ratios):.2f}, "
                f"output {statistics.median(r[1] for r in ratios):.2f}"
            )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save", help="also save the collected data (JSON lines)")
    parser.add_argument("--load", help="fit from saved data instead of collecting")
    args = parser.parse_args(argv)
    if args.load:
        with open(args.load, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
    else:
        records = collect()
        if args.save:
            with open(args.save, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r) + "\n" for r in records)
    fit(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
