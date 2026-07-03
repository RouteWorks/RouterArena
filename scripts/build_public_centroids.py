#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0
"""Build cheapest-correct centroids from public HuggingFace datasets.

Runs 4 RouterArena models on public prompts, grades answers, finds the
cheapest model that gets each prompt right, then rebuilds chuzom-centroids.npz
from those labeled examples.

Usage:
    OPENROUTER_API_KEY=xxx uv run python3 scripts/build_public_centroids.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not API_KEY:
    sys.exit("ERROR: OPENROUTER_API_KEY not set")

MODELS = [
    "google/gemini-3.1-flash-lite",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3-next-80b-a3b-instruct",
    "qwen/qwen3-235b-a22b-2507",
]
# Actual cost rank for typical RA query (~100 input / 75 output tokens):
# qwen3-235b=$0.0000121  deepseek=$0.0000531  flash-lite=$0.0001375  qwen3-next=$0.0003725
# Flash-lite is expensive because output tokens cost $1.5/M (vs qwen3-235b $0.10/M).
MODEL_COST_RANK = {
    "qwen/qwen3-235b-a22b-2507": 0,  # cheapest at typical prompt lengths
    "deepseek/deepseek-v4-flash": 1,
    "google/gemini-3.1-flash-lite": 2,  # expensive on output-heavy prompts
    "qwen/qwen3-next-80b-a3b-instruct": 3,  # most expensive
}

PROMPTS_PER_DATASET = 500
MAX_WORKERS = 20
MAX_TOKENS = 256
LABELS_FILE = ROOT / "data" / "public_centroid_labels.jsonl"
CENTROIDS_FILE = ROOT / "router_inference" / "config" / "chuzom-v3-centroids.npz"

# ── Dataset loaders ───────────────────────────────────────────────────────────


def load_arc(split: str = "test", n: int = PROMPTS_PER_DATASET) -> list[dict]:
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset("ai2_arc", "ARC-Challenge", split=split, trust_remote_code=False)
    items = []
    for row in ds.shuffle(seed=42).select(range(min(n, len(ds)))):
        choices = row["choices"]
        options = "\n".join(
            f"{label}. {text}" for label, text in zip(choices["label"], choices["text"])
        )
        prompt = f"{row['question']}\n\n{options}\n\nAnswer with just the letter."
        items.append(
            {
                "prompt": prompt,
                "answer": row["answerKey"],
                "type": "mcq",
                "source": "arc",
            }
        )
    return items


def load_mmlu(split: str = "test", n: int = PROMPTS_PER_DATASET) -> list[dict]:
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset("cais/mmlu", "all", split=split, trust_remote_code=False)
    items = []
    labels = ["A", "B", "C", "D"]
    for row in ds.shuffle(seed=42).select(range(min(n, len(ds)))):
        choices = row["choices"]
        options = "\n".join(f"{labels[i]}. {c}" for i, c in enumerate(choices))
        prompt = f"{row['question']}\n\n{options}\n\nAnswer with just the letter."
        gold = labels[row["answer"]]
        items.append(
            {"prompt": prompt, "answer": gold, "type": "mcq", "source": "mmlu"}
        )
    return items


def load_gsm8k(split: str = "test", n: int = PROMPTS_PER_DATASET) -> list[dict]:
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset("openai/gsm8k", "main", split=split, trust_remote_code=False)
    items = []
    for row in ds.shuffle(seed=42).select(range(min(n, len(ds)))):
        prompt = f"{row['question']}\n\nShow your work and end with 'The answer is X.'"
        # Gold answer is the last number after ####
        match = re.search(r"####\s*([\d,\.\-]+)", row["answer"])
        gold = match.group(1).replace(",", "") if match else None
        if gold:
            items.append(
                {"prompt": prompt, "answer": gold, "type": "math", "source": "gsm8k"}
            )
    return items[:n]


def load_squad(split: str = "validation", n: int = PROMPTS_PER_DATASET) -> list[dict]:
    """SQuAD: reading comprehension → builds NarrativeQA/PubMedQA-relevant centroids."""
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset("rajpurkar/squad", split=split, trust_remote_code=False)
    items = []
    for row in ds.shuffle(seed=42).select(range(min(n * 2, len(ds)))):
        ctx = row["context"][:1500]
        question = row["question"]
        gold = row["answers"]["text"][0] if row["answers"]["text"] else None
        if not gold:
            continue
        prompt = f"Read the following passage and answer the question.\n\nPassage: {ctx}\n\nQuestion: {question}\n\nAnswer:"
        items.append(
            {
                "prompt": prompt,
                "answer": gold.lower(),
                "type": "reading",
                "source": "squad",
            }
        )
        if len(items) >= n:
            break
    return items


def load_humaneval(n: int = PROMPTS_PER_DATASET) -> list[dict]:
    """HumanEval: coding tasks → builds LiveCodeBench-relevant centroids."""
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=False)
    items = []
    for row in ds.shuffle(seed=42).select(range(min(n, len(ds)))):
        prompt = row["prompt"]
        canonical = row.get("canonical_solution", "")
        items.append(
            {
                "prompt": f"Complete the following Python function:\n\n{prompt}",
                "answer": canonical[:50],
                "type": "code",
                "source": "humaneval",
                "canonical": canonical,
            }
        )
        if len(items) >= n:
            break
    return items


def grade_reading(response: str, gold: str) -> bool:
    """Loose reading comprehension grading: gold substring in response."""
    response_lower = response.lower()
    gold_lower = gold.lower()
    if gold_lower in response_lower:
        return True
    gold_words = set(gold_lower.split())
    if len(gold_words) >= 2:
        return (
            sum(1 for w in gold_words if w in response_lower) >= len(gold_words) * 0.8
        )
    return False


def grade_code(response: str, item: dict) -> bool:
    """Check if response defines a valid function body."""
    return "def " in response and "return" in response


def load_math(split: str = "test", n: int = PROMPTS_PER_DATASET) -> list[dict]:
    from datasets import load_dataset, concatenate_datasets  # type: ignore[import]

    subsets = [
        "algebra",
        "number_theory",
        "counting_and_probability",
        "intermediate_algebra",
    ]
    parts = []
    per_subset = max(1, n // len(subsets))
    for subset in subsets:
        try:
            ds = load_dataset(
                "EleutherAI/hendrycks_math",
                subset,
                split=split,
                trust_remote_code=False,
            )
            parts.append(ds.shuffle(seed=42).select(range(min(per_subset, len(ds)))))
        except Exception:
            pass
    if not parts:
        return []
    combined = concatenate_datasets(parts).shuffle(seed=42)
    items = []
    for row in combined:
        prompt = (
            f"Solve the following math problem. Show your work.\n\n{row['problem']}"
        )
        gold_raw = row.get("solution", "")
        match = re.search(r"\\boxed\{([^}]+)\}", gold_raw)
        gold = match.group(1).strip() if match else None
        if gold:
            items.append(
                {
                    "prompt": prompt,
                    "answer": gold,
                    "type": "math_hard",
                    "source": "math",
                }
            )
        if len(items) >= n:
            break
    return items


# ── Graders ───────────────────────────────────────────────────────────────────

_MCQ_RE = re.compile(r"\b([A-D])\b")


def grade_mcq(response: str, gold: str) -> bool:
    response = response.strip()
    # Direct letter match at start
    if response.upper().startswith(gold.upper()):
        return True
    # Find all A/B/C/D mentions, take last one
    matches = _MCQ_RE.findall(response.upper())
    return bool(matches and matches[-1] == gold.upper())


def _extract_number(text: str) -> float | None:
    # Look for "The answer is X" pattern first
    m = re.search(r"answer\s+is\s+([\d,\.\-]+)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    # Last number in text
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text.replace(",", ""))
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            pass
    return None


def grade_math(response: str, gold: str) -> bool:
    # Try exact string match first (for symbolic answers)
    if gold.strip().lower() in response.lower():
        return True
    # Try numeric comparison
    try:
        gold_n = float(gold.replace(",", ""))
        resp_n = _extract_number(response)
        if resp_n is not None:
            return abs(gold_n - resp_n) < 1e-3 * max(1.0, abs(gold_n))
    except (ValueError, TypeError):
        pass
    return False


def grade_math_hard(response: str, gold: str) -> bool:
    # Check if gold expression appears in response (boxed or plain)
    gold_clean = gold.strip()
    if gold_clean in response:
        return True
    # Try numeric
    return grade_math(response, gold_clean)


def grade(response: str, item: dict) -> bool:
    t = item["type"]
    if t == "mcq":
        return grade_mcq(response, item["answer"])
    if t == "math":
        return grade_math(response, item["answer"])
    if t == "math_hard":
        return grade_math_hard(response, item["answer"])
    if t == "reading":
        return grade_reading(response, item["answer"])
    if t == "code":
        return grade_code(response, item)
    return False


# ── Inference ─────────────────────────────────────────────────────────────────


def call_model(model: str, prompt: str) -> str | None:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
        }
    ).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
    except Exception:
        return None


def run_all_models(item: dict) -> dict | None:
    """Run all 4 models on one prompt. Returns labeled record or None if none correct."""
    prompt = item["prompt"]
    responses: dict[str, str | None] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(call_model, m, prompt): m for m in MODELS}
        for f in as_completed(futures):
            m = futures[f]
            responses[m] = f.result()

    correct_models = [
        m for m in MODELS if (resp := responses.get(m)) and grade(resp, item)
    ]
    if not correct_models:
        return None

    cheapest_correct = min(correct_models, key=lambda m: MODEL_COST_RANK[m])
    return {
        "prompt": prompt,
        "source": item["source"],
        "type": item["type"],
        "answer": item["answer"],
        "cheapest_correct": cheapest_correct,
        "correct_models": correct_models,
        "responses": {m: responses[m] for m in MODELS},
    }


# ── Centroid builder ──────────────────────────────────────────────────────────


def embed_texts(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # type: ignore[import]

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embeddings = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64
    )
    return np.array(embeddings, dtype=np.float32)


def build_centroids(labels: list[dict]) -> np.ndarray:
    """One centroid per model = mean embedding of cheapest-correct examples."""
    centroid_rows = []
    for model in MODELS:
        prompts = [r["prompt"] for r in labels if r["cheapest_correct"] == model]
        if not prompts:
            print(
                f"  WARNING: no cheapest-correct examples for {model}, using zero centroid"
            )
            centroid_rows.append(np.zeros(384, dtype=np.float32))
            continue
        print(f"  Embedding {len(prompts)} examples for {model}...")
        embs = embed_texts(prompts)
        centroid = embs.mean(axis=0)
        centroid /= np.linalg.norm(centroid)  # normalize
        centroid_rows.append(centroid)
    return np.stack(centroid_rows)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    LABELS_FILE.parent.mkdir(exist_ok=True)

    # Load public datasets
    print("Loading public datasets...")
    all_items: list[dict] = []
    loaders: list[tuple[str, Callable[[], list[dict]]]] = [
        ("ARC-Challenge", load_arc),
        ("MMLU", load_mmlu),
        ("GSM8K", load_gsm8k),
        ("MATH", load_math),
        ("SQuAD", load_squad),
        ("HumanEval", load_humaneval),
    ]
    for name, loader in loaders:
        try:
            items = loader()
            all_items.extend(items)
            print(f"  {name}: {len(items)} prompts")
        except Exception as ex:
            print(f"  {name}: FAILED — {ex}")

    print(f"\nTotal prompts: {len(all_items)}")

    # Run inference + grading
    print(f"\nRunning inference on {len(all_items)} prompts ({MAX_WORKERS} workers)...")
    labels: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_all_models, item): item for item in all_items}
        for f in as_completed(futures):
            result = f.result()
            done += 1
            if result:
                labels.append(result)
            if done % 50 == 0:
                print(f"  {done}/{len(all_items)} | {len(labels)} labeled")

    print(f"\nLabeled: {len(labels)}/{len(all_items)} (at least one model correct)")

    # Save labels
    with open(LABELS_FILE, "w") as fp:
        for record in labels:
            fp.write(json.dumps(record) + "\n")
    print(f"Labels saved to {LABELS_FILE}")

    # Distribution
    print("\nCheapest-correct distribution:")
    for model in MODELS:
        n = sum(1 for r in labels if r["cheapest_correct"] == model)
        print(f"  {n:4d} ({n / len(labels) * 100:.1f}%)  {model}")

    # Build and save centroids
    print("\nBuilding centroids from cheapest-correct examples...")
    centroids = build_centroids(labels)

    # Backup existing file
    if CENTROIDS_FILE.exists():
        backup = CENTROIDS_FILE.with_suffix(".npz.bak")
        import shutil

        shutil.copy(CENTROIDS_FILE, backup)
        print(f"Backed up existing centroids to {backup}")

    # Build final arrays preserving model order from the existing file; rows for
    # models not retrained here (e.g. gemini-2.0-flash-001) are kept as-is.
    existing = np.load(CENTROIDS_FILE)
    existing_models = [str(m) for m in existing["models"]]
    final_models = existing_models
    final_centroids = existing["centroids"].astype(np.float32).copy()
    for new_idx, model in enumerate(MODELS):
        if model in existing_models:
            existing_idx = existing_models.index(model)
            final_centroids[existing_idx] = centroids[new_idx]

    np.savez(CENTROIDS_FILE, centroids=final_centroids, models=np.array(final_models))
    print(f"Saved updated centroids to {CENTROIDS_FILE}")
    print(f"Shape: {final_centroids.shape}")


if __name__ == "__main__":
    main()
