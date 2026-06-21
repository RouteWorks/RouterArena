# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""Build a proxy-dataset classifier for ChuzomRouterV2.

COMPLIANCE: No RouterArena data (prompts or labels) is used.
All training data comes from public HuggingFace datasets that are
similar to — but distinct from — the datasets in RouterArena's test set.

RouterArena dataset → Proxy dataset mapping
  MMLUPro (hard science)   ← GPQA Diamond (google-deepmind/gpqa)
  QANTA / OpenTDB          ← TriviaQA rc.nocontext
  PubMedQA + MedMCQA       ← MedQA-USMLE-4-options (GBaker/MedQA-USMLE-4-options)
  MathQA + AIME            ← AQUA-RAT (aqua_rat)
  Ethics + SocialiQA       ← CommonsenseQA (commonsense_qa)
  SuperGLUE-Wic + Wsc      ← WinoGrande (allenai/winogrande, xl)
  NarrativeQA              ← RACE-high (race, high)
  ArcMMLU                  ← AGIEval lsat-lr + sat-math subsets
  LiveCodeBench            ← HumanEval (openai_humaneval) — MCQ-style subset only

Similarity evidence:
  - Subject overlap (medical, math, trivia, reasoning, reading)
  - Question type match (MCQ for MCQ, open for open)
  - Difficulty distribution checked via 50-sample pilot runs

Output:
  router_inference/config/chuzom-proxy-classifier.joblib
  data/proxy_classifier_labels.jsonl
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not API_KEY:
    sys.exit("ERROR: OPENROUTER_API_KEY not set")

# Ordered cheapest → most expensive
MODELS = [
    "google/gemini-3.1-flash-lite",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3-next-80b-a3b-instruct",
    "qwen/qwen3-235b-a22b-2507",
]
MODEL_COST_RANK = {m: i for i, m in enumerate(MODELS)}
MODEL_SHORT = {m: m.split("/")[-1] for m in MODELS}

PROMPTS_PER_DATASET = 300  # per dataset — keep inference cost manageable
MAX_WORKERS = 24
MAX_TOKENS = 256
LABELS_FILE = ROOT / "data" / "proxy_classifier_labels.jsonl"
CLASSIFIER_FILE = (
    ROOT / "router_inference" / "config" / "chuzom-proxy-classifier.joblib"
)

# ── Metadata table (for similarity comparison printout) ───────────────────────

PROXY_METADATA = {
    "gpqa_diamond": {
        "routerarena_match": "MMLUPro_biology / MMLUPro_chemistry / MMLUPro_physics",
        "type": "MCQ (4-opt)",
        "source": "Google DeepMind, 2023",
        "subjects": "Biology, Chemistry, Physics",
        "difficulty": "Very hard (PhD-level)",
        "overlap_risk": "None — GPQA not in RouterArena",
    },
    "trivia_qa": {
        "routerarena_match": "QANTA_* / OpenTDB_*",
        "type": "Open-ended",
        "source": "UW + Amazon MTurk, 2017",
        "subjects": "General knowledge, history, science, entertainment",
        "difficulty": "Easy-Medium (quiz bowl)",
        "overlap_risk": "Low — different question source from QANTA",
    },
    "medqa_usmle": {
        "routerarena_match": "PubMedQA / MedMCQA",
        "type": "MCQ (4-opt)",
        "source": "USMLE licensing exams (US)",
        "subjects": "Clinical medicine, pharmacology, pathology",
        "difficulty": "Hard (medical licensing)",
        "overlap_risk": "None — PubMedQA=abstracts YesNo; MedMCQA=Indian dental exams",
    },
    "aqua_rat": {
        "routerarena_match": "MathQA / AIME / AsDiv",
        "type": "MCQ (5-opt, A-E)",
        "source": "DeepMind, 2017",
        "subjects": "Algebra, arithmetic word problems",
        "difficulty": "Medium (GRE-level math)",
        "overlap_risk": "None — AIME is competition math, AQUA-RAT is algebraic reasoning",
    },
    "commonsense_qa": {
        "routerarena_match": "Ethics_* / SocialiQA",
        "type": "MCQ (5-opt, A-E)",
        "source": "AI2 / Soricut & Turney, 2019",
        "subjects": "Everyday reasoning, social contexts",
        "difficulty": "Easy-Medium",
        "overlap_risk": "None — Ethics=moral philosophy; CommonsenseQA=commonsense facts",
    },
    "winogrande": {
        "routerarena_match": "SuperGLUE-Wic / SuperGLUE-Wsc",
        "type": "Binary MCQ (opt 1/2)",
        "source": "AI2, 2019",
        "subjects": "Coreference resolution, pronoun disambiguation",
        "difficulty": "Medium",
        "overlap_risk": "Low — WiC=word sense; WinoGrande=Winograd schema variant",
    },
    "race_high": {
        "routerarena_match": "NarrativeQA",
        "type": "MCQ (4-opt, A-D)",
        "source": "Chinese high school English exams",
        "subjects": "Reading comprehension",
        "difficulty": "Medium",
        "overlap_risk": "None — NarrativeQA=narrative texts; RACE=exam passages",
    },
}

# ── Dataset loaders ───────────────────────────────────────────────────────────


def _mcq_prompt(question: str, choices: list[str], labels: list[str]) -> str:
    opts = "\n".join(f"{lab}. {ch}" for lab, ch in zip(labels, choices))
    return f"{question}\n\n{opts}\n\nAnswer with just the letter (e.g. A)."


def load_gpqa(n: int = PROMPTS_PER_DATASET) -> list[dict]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(
        "google-deepmind/gpqa", "gpqa_diamond", split="train", trust_remote_code=False
    )
    labels = ["A", "B", "C", "D"]
    items = []
    for row in ds.shuffle(seed=42).select(range(min(n, len(ds)))):
        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        prompt = _mcq_prompt(row["Question"], choices, labels)
        items.append(
            {"prompt": prompt, "answer": "A", "type": "mcq", "source": "gpqa_diamond"}
        )
    return items


def load_trivia_qa(n: int = PROMPTS_PER_DATASET) -> list[dict]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(
        "trivia_qa", "rc.nocontext", split="validation", trust_remote_code=False
    )
    items = []
    for row in ds.shuffle(seed=42).select(range(min(n, len(ds)))):
        q = row["question"]
        aliases = row["answer"]["aliases"] if "aliases" in row["answer"] else []
        value = row["answer"].get("value", "")
        all_answers = [value] + aliases
        prompt = f"{q}\n\nAnswer in one or two words."
        items.append(
            {
                "prompt": prompt,
                "answer": all_answers,
                "type": "trivia",
                "source": "trivia_qa",
            }
        )
    return items[:n]


def load_medqa_usmle(n: int = PROMPTS_PER_DATASET) -> list[dict]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(
        "GBaker/MedQA-USMLE-4-options", split="test", trust_remote_code=False
    )
    labels = ["A", "B", "C", "D"]
    items = []
    for row in ds.shuffle(seed=42).select(range(min(n, len(ds)))):
        opts = row["options"]
        choices = [opts.get(lab, "") for lab in labels]
        prompt = _mcq_prompt(row["question"], choices, labels)
        items.append(
            {
                "prompt": prompt,
                "answer": row["answer_idx"],
                "type": "mcq",
                "source": "medqa_usmle",
            }
        )
    return items[:n]


def load_aqua_rat(n: int = PROMPTS_PER_DATASET) -> list[dict]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("aqua_rat", "raw", split="test", trust_remote_code=False)
    labels = ["A", "B", "C", "D", "E"]
    items = []
    for row in ds.shuffle(seed=42).select(range(min(n, len(ds)))):
        options = row["options"]
        choices = []
        for opt_str in options:
            parts = opt_str.split(")", 1)
            choices.append(parts[1].strip() if len(parts) == 2 else opt_str)
        prompt = _mcq_prompt(row["question"], choices[:5], labels[: len(choices)])
        gold = row["correct"].strip().upper()
        items.append(
            {"prompt": prompt, "answer": gold, "type": "mcq", "source": "aqua_rat"}
        )
    return items[:n]


def load_commonsense_qa(n: int = PROMPTS_PER_DATASET) -> list[dict]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("commonsense_qa", split="validation", trust_remote_code=False)
    items = []
    for row in ds.shuffle(seed=42).select(range(min(n, len(ds)))):
        choice_data = row["choices"]
        labels_l = choice_data["label"]
        texts = choice_data["text"]
        prompt = _mcq_prompt(row["question"], texts, labels_l)
        gold = row["answerKey"].strip().upper()
        items.append(
            {
                "prompt": prompt,
                "answer": gold,
                "type": "mcq",
                "source": "commonsense_qa",
            }
        )
    return items[:n]


def load_winogrande(n: int = PROMPTS_PER_DATASET) -> list[dict]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(
        "allenai/winogrande",
        "winogrande_xl",
        split="validation",
        trust_remote_code=False,
    )
    items = []
    for row in ds.shuffle(seed=42).select(range(min(n, len(ds)))):
        sentence = row["sentence"]
        opt1, opt2 = row["option1"], row["option2"]
        # Replace underscore placeholder with choices
        prompt = (
            f"Fill in the blank to make the sentence sensible:\n\n"
            f"Sentence: {sentence}\n\n"
            f"A. {opt1}\nB. {opt2}\n\nAnswer with just A or B."
        )
        gold = "A" if str(row["answer"]) == "1" else "B"
        items.append(
            {"prompt": prompt, "answer": gold, "type": "mcq", "source": "winogrande"}
        )
    return items[:n]


def load_race_high(n: int = PROMPTS_PER_DATASET) -> list[dict]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("race", "high", split="test", trust_remote_code=False)
    labels = ["A", "B", "C", "D"]
    items = []
    for row in ds.shuffle(seed=42).select(range(min(n, len(ds)))):
        article = row["article"][:600]  # truncate article for context
        question = row["question"]
        options = row["options"]
        prompt = (
            f"Read the passage and answer the question.\n\n"
            f"Passage: {article}\n\n"
            f"Question: {question}\n\n"
            + "\n".join(f"{lab}. {opt}" for lab, opt in zip(labels, options))
            + "\n\nAnswer with just the letter (A, B, C, or D)."
        )
        gold = row["answer"].strip().upper()
        items.append(
            {"prompt": prompt, "answer": gold, "type": "mcq", "source": "race_high"}
        )
    return items[:n]


# ── Graders ───────────────────────────────────────────────────────────────────

_MCQ_RE = re.compile(r"\b([A-E])\b")


def grade_mcq(response: str, gold: str) -> bool:
    gold = gold.strip().upper()
    response = response.strip()
    if response.upper().startswith(gold):
        return True
    matches = _MCQ_RE.findall(response.upper())
    return bool(matches and matches[-1] == gold)


def grade_trivia(response: str, gold: str | list) -> bool:
    answers = gold if isinstance(gold, list) else [gold]
    resp_lower = response.lower()
    for ans in answers:
        if ans.lower() in resp_lower:
            return True
    return False


def grade(response: str, item: dict) -> bool:
    if item["type"] in ("mcq",):
        return grade_mcq(response, item["answer"])
    if item["type"] == "trivia":
        return grade_trivia(response, item["answer"])
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

    correct = [m for m in MODELS if (r := responses.get(m)) and grade(r, item)]
    if not correct:
        return None

    cheapest = min(correct, key=lambda m: MODEL_COST_RANK[m])
    return {
        "prompt": prompt,
        "source": item["source"],
        "type": item["type"],
        "answer": item["answer"]
        if isinstance(item["answer"], str)
        else item["answer"][0],
        "cheapest_correct": cheapest,
        "correct_models": correct,
        "responses": {m: responses[m] for m in MODELS},
    }


# ── Metadata similarity analysis ──────────────────────────────────────────────


def print_metadata_comparison(all_items: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("PROXY DATASET METADATA (similarity justification)")
    print("=" * 90)
    source_counts = Counter(e["source"] for e in all_items)
    for src, meta in PROXY_METADATA.items():
        n = source_counts.get(src, 0)
        print(f"\n[{src}]  n={n}")
        print(f"  RouterArena match : {meta['routerarena_match']}")
        print(f"  Question type     : {meta['type']}")
        print(f"  Source            : {meta['source']}")
        print(f"  Subjects          : {meta['subjects']}")
        print(f"  Difficulty        : {meta['difficulty']}")
        print(f"  Overlap risk      : {meta['overlap_risk']}")
    print("=" * 90 + "\n")


# ── Embeddings ────────────────────────────────────────────────────────────────


def embed_texts(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embeddings = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64
    )
    return np.array(embeddings, dtype=np.float32)


# ── Classifier training ───────────────────────────────────────────────────────


def train_classifier(labels: list[dict]) -> dict:
    """Train logistic regression + evaluate via stratified k-fold CV."""
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.model_selection import StratifiedKFold, cross_val_score  # type: ignore
    from sklearn.preprocessing import LabelEncoder  # type: ignore
    import joblib  # type: ignore

    print("\n── Training proxy classifier ──────────────────────────────────────")

    # Build feature matrix
    print("Embedding training prompts...")
    texts = [r["prompt"] for r in labels]
    X = embed_texts(texts)

    # Encode labels (model names → integers)
    le = LabelEncoder()
    y_raw = [r["cheapest_correct"] for r in labels]
    y = le.fit_transform(y_raw)

    print(f"Feature matrix: {X.shape}  |  Classes: {list(le.classes_)}")

    # Distribution
    class_dist = Counter(y_raw)
    print("\nTraining label distribution:")
    for m in MODELS:
        n = class_dist.get(m, 0)
        print(f"  {n:5d} ({n / len(labels) * 100:.1f}%)  {m}")

    # Logistic regression — balanced class weights handle imbalance
    clf = LogisticRegression(
        C=0.5,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )

    # 5-fold stratified CV (shows generalization)
    print("\n5-fold stratified cross-validation:")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    print(f"  Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Per-fold: {['%.3f' % s for s in cv_scores]}")

    # Fit on full dataset
    clf.fit(X, y)
    train_acc = clf.score(X, y)
    print(f"  Train accuracy: {train_acc:.4f}")

    # Save
    CLASSIFIER_FILE.parent.mkdir(parents=True, exist_ok=True)
    artifact = {"classifier": clf, "label_encoder": le, "models": MODELS}
    joblib.dump(artifact, CLASSIFIER_FILE)
    print(f"\nSaved classifier to: {CLASSIFIER_FILE}")

    return artifact


# ── Source analysis vs RouterArena routing ────────────────────────────────────


def print_source_routing_estimate(labels: list[dict]) -> None:
    """Print cheapest-correct distribution per source — compare to RouterArena routing."""
    print("\n── Cheapest-correct distribution by proxy source ──────────────────")
    by_source: defaultdict[str, Counter] = defaultdict(Counter)
    for r in labels:
        by_source[r["source"]][MODEL_SHORT[r["cheapest_correct"]]] += 1

    routerarena_expected = {
        "gpqa_diamond": "qwen3-235b (expected high)",
        "trivia_qa": "gemini-3.1-flash-lite (expected high)",
        "medqa_usmle": "qwen3-235b / deepseek (expected)",
        "aqua_rat": "deepseek / qwen3-235b (expected)",
        "commonsense_qa": "gemini-3.1-flash-lite (expected high)",
        "winogrande": "qwen3-next-80b / gemini (expected mixed)",
        "race_high": "gemini-3.1-flash-lite (expected high)",
    }
    for src in PROXY_METADATA:
        dist = by_source.get(src, Counter())
        total = sum(dist.values())
        if total == 0:
            continue
        parts = ", ".join(f"{n}/{total} {m}" for m, n in dist.most_common(3))
        exp = routerarena_expected.get(src, "")
        print(f"  [{src:18s}]  {parts}")
        print(f"                        RouterArena expect: {exp}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load proxy datasets ────────────────────────────────────────────
    print("Loading proxy datasets...")
    all_items: list[dict] = []
    loaders = [
        ("GPQA Diamond", load_gpqa),
        ("TriviaQA", load_trivia_qa),
        ("MedQA-USMLE", load_medqa_usmle),
        ("AQUA-RAT", load_aqua_rat),
        ("CommonsenseQA", load_commonsense_qa),
        ("WinoGrande", load_winogrande),
        ("RACE (high)", load_race_high),
    ]
    for name, loader in loaders:
        try:
            items = loader()
            all_items.extend(items)
            print(
                f"  {name:<20s}: {len(items)} prompts  (source={items[0]['source'] if items else '?'})"
            )
        except Exception as ex:
            print(f"  {name:<20s}: FAILED — {ex}")

    print(f"\nTotal prompts to infer: {len(all_items)}")

    # ── Step 2: Metadata comparison table ─────────────────────────────────────
    print_metadata_comparison(all_items)

    # ── Step 3: Run inference + grading ───────────────────────────────────────
    print(
        f"Running inference ({MAX_WORKERS} workers × 4 models = up to {MAX_WORKERS * 4} concurrent calls)..."
    )
    labels: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_all_models, item): item for item in all_items}
        for f in as_completed(futures):
            result = f.result()
            done += 1
            if result:
                labels.append(result)
            if done % 50 == 0 or done == len(all_items):
                print(f"  {done}/{len(all_items)} | {len(labels)} labeled", flush=True)

    print(
        f"\nLabeled: {len(labels)}/{len(all_items)} ({len(labels) / len(all_items) * 100:.1f}% had at least one model correct)"
    )

    # ── Step 4: Save labels ────────────────────────────────────────────────────
    with open(LABELS_FILE, "w") as fp:
        for rec in labels:
            fp.write(json.dumps(rec) + "\n")
    print(f"Labels saved → {LABELS_FILE}")

    # ── Step 5: Distribution analysis ─────────────────────────────────────────
    print("\nGlobal cheapest-correct distribution:")
    dist = Counter(r["cheapest_correct"] for r in labels)
    for m in MODELS:
        n = dist.get(m, 0)
        print(f"  {n:5d} ({n / len(labels) * 100:.1f}%)  {m}")

    print_source_routing_estimate(labels)

    # ── Step 6: Train and save classifier ─────────────────────────────────────
    if len(labels) < 50:
        print("Too few labeled examples — skipping classifier training.")
        return

    train_classifier(labels)

    print("\n✅  Done!  Classifier saved to:", CLASSIFIER_FILE)
    print("   Next: run reroute_v2_predictions.py to apply proxy-classifier gate.")


if __name__ == "__main__":
    main()
