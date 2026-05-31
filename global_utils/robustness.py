# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""Utilities for computing robustness metrics across scripts."""

from __future__ import annotations

from typing import Any, Optional

from universal_model_names import ModelNameManager

__all__ = ["compute_robustness_score", "compute_flip_labels"]


def _normalize_model_name(
    model_name: Optional[str], name_manager: ModelNameManager
) -> Optional[str]:
    """Convert a model name to its universal form, falling back gracefully."""
    if model_name is None:
        return None
    try:
        return name_manager.get_universal_name(model_name)
    except ValueError:
        return model_name


def compute_flip_labels(
    full_predictions: list[dict[str, Any]],
    robustness_predictions: list[dict[str, Any]],
    *,
    name_manager: Optional[ModelNameManager] = None,
) -> list[dict[str, Any]]:
    """
    Compute per-query robustness flip labels.

    A query "flips" when the router selects a different model for the perturbed
    (robustness) prompt than it did for the original prompt.

    Args:
        full_predictions: Router predictions for the full/sub_10 split.
        robustness_predictions: Predictions collected from the robustness split.
        name_manager: Optional shared instance to reuse universal name cache.

    Returns:
        A list of ``{"global index": <id>, "flip": 0|1}`` for every robustness
        query that has a matching full-split selection, sorted by global index.
    """

    manager = name_manager or ModelNameManager()

    def get_index(entry: dict[str, Any]) -> Optional[str]:
        """Extract a normalized global index from an entry."""
        value = entry.get("global index") or entry.get("global_index")
        return str(value) if value is not None else None

    def normalize(name: object) -> Optional[str]:
        """Normalize model names through the shared name manager."""
        if not name:
            return None
        return _normalize_model_name(str(name), manager)

    # Build a lookup of router selections from the full split.
    full_map = {
        key: entry
        for entry in full_predictions
        if isinstance(entry, dict)
        and not entry.get("for_optimality", False)
        and (key := get_index(entry)) is not None
    }

    labels: list[dict[str, Any]] = []
    for entry in robustness_predictions:
        if not isinstance(entry, dict):
            continue
        key = get_index(entry)
        if key is None or key not in full_map:
            continue
        full_model = full_map[key].get("prediction")
        robust_model = entry.get("prediction")
        if not full_model or not robust_model:
            continue
        flipped = normalize(full_model) != normalize(robust_model)
        labels.append({"global index": key, "flip": 1 if flipped else 0})

    labels.sort(key=lambda item: str(item["global index"]))
    return labels


def compute_robustness_score(
    full_predictions: list[dict[str, Any]],
    robustness_predictions: list[dict[str, Any]],
    *,
    name_manager: Optional[ModelNameManager] = None,
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

    labels = compute_flip_labels(
        full_predictions, robustness_predictions, name_manager=name_manager
    )
    if not labels:
        return None

    flips = sum(label["flip"] for label in labels)
    return 1.0 - flips / len(labels)
