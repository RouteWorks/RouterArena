# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""Router calls: LLM calls a router makes itself, besides the call that produces the answer.

A router that uses an LLM to decide (a routing model, a classifier, voters, a critic, the
earlier steps of a cascade) records each such call in its prediction row:

    "router_usage": [
        {"model": "openai/gpt-4o-mini", "input_tokens": 812, "output_tokens": 37,
         "purpose": "routing"}
    ]

``output_tokens`` counts every generated token, reasoning included, as the provider bills
it. ``purpose`` is optional free text. Each call is priced like an answer: the model's
price profile in ``model_cost/model_profiles.yaml``, or the table's highest price if the
model has none. The row's cost is the answer's cost plus the cost of its router calls.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

__all__ = [
    "ROUTER_USAGE_FIELD",
    "validate_router_usage",
    "router_usage_models",
    "router_usage_cost",
]

ROUTER_USAGE_FIELD = "router_usage"

# (model name, token usage) -> USD; ModelEvaluator.calculate_inference_cost fits.
PriceFn = Callable[[str, Dict[str, int]], float]


def validate_router_usage(value: Any) -> List[str]:
    """Return a list of problems with a row's ``router_usage`` (empty if it is valid)."""
    if not isinstance(value, list):
        return [f"{ROUTER_USAGE_FIELD} must be a list of calls"]
    errors: List[str] = []
    for i, call in enumerate(value):
        where = f"{ROUTER_USAGE_FIELD}[{i}]"
        if not isinstance(call, dict):
            errors.append(f"{where} must be an object")
            continue
        if not isinstance(call.get("model"), str) or not call["model"].strip():
            errors.append(f"{where}.model must be a non-empty string")
        for field in ("input_tokens", "output_tokens"):
            count = call.get(field)
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                errors.append(f"{where}.{field} must be an integer >= 0")
        if "purpose" in call and not isinstance(call["purpose"], str):
            errors.append(f"{where}.purpose must be a string")
    return errors


def router_usage_models(value: Any) -> List[str]:
    """Model names used by a row's router calls (for price checks)."""
    if not isinstance(value, list):
        return []
    return [
        c["model"]
        for c in value
        if isinstance(c, dict) and isinstance(c.get("model"), str)
    ]


def router_usage_cost(value: Any, price: PriceFn) -> float:
    """Total cost of a row's router calls. A missing field costs nothing.

    Raises ValueError for a malformed field, so a bad record is never priced as free."""
    if value is None:
        return 0.0
    errors = validate_router_usage(value)
    if errors:
        raise ValueError("; ".join(errors))
    return sum(
        price(
            call["model"],
            {
                "input_tokens": call["input_tokens"],
                "output_tokens": call["output_tokens"],
                "total_tokens": call["input_tokens"] + call["output_tokens"],
            },
        )
        for call in value
    )
