# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""Utilities for computing robustness metrics across scripts."""

from __future__ import annotations

from typing import Any, Optional

from universal_model_names import ModelNameManager

__all__ = ["compute_robustness_score"]


def _normalize_model_name(
    model_name: Optional[str], name_manager: ModelNameManager
) -> Optional[str]:
    """Convert a model name to its universal form, falling back gracefully."""
    if model_name is None:
        return None
    try:
        return name_manager.get_universal_name(model_name)
    except Exception:
        return model_name


def compute_robustness_score(
    full_predictions: list[dict[str, Any]],
    robustness_predictions: list[dict[str, Any]],
    *,
    name_manager: ModelNameManager | None = None,
) -> Optional[float]:
    """
    Compute the robustness flip ratio between full and robustness prediction sets.

    Args:
        full_predictions: Router predictions for the full/sub_10 split.
        robustness_predictions: Predictions collected from the robustness split.
        name_manager: Optional shared instance to reuse universal name cache.

    Returns:
        A float in [0, 1] representing stability (1 - flip ratio),
        or ``None`` if no overlapping entries were found.
    """

    manager = name_manager or ModelNameManager()

    # Build a lookup of router selections from the full split.
    full_map: dict[str, dict[str, Any]] = {}
    for entry in full_predictions:
        if not isinstance(entry, dict):
            continue
        if entry.get("for_optimality", False):
            continue
        global_index = entry.get("global index") or entry.get("global_index")
        if global_index is None:
            continue
        key = str(global_index)
        if key not in full_map:
            full_map[key] = entry

    if not full_map:
        return None

    matched = 0
    flips = 0
    for entry in robustness_predictions:
        if not isinstance(entry, dict):
            continue
        global_index = entry.get("global index") or entry.get("global_index")
        if global_index is None:
            continue
        key = str(global_index)
        full_entry = full_map.get(key)
        if not full_entry:
            continue

        full_model = full_entry.get("prediction")
        robust_model = entry.get("prediction")
        if not full_model or not robust_model:
            continue

        matched += 1
        full_model_norm = _normalize_model_name(str(full_model), manager)
        robust_model_norm = _normalize_model_name(str(robust_model), manager)
        if full_model_norm != robust_model_norm:
            flips += 1

    if matched == 0:
        return None

    return 1.0 - flips / matched
