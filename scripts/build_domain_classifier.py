#!/usr/bin/env python3
"""
Build an enhanced domain classifier for Gate 0 of ChuzomRouterV2.

Strategy:
  1. Download ~29k labeled examples from HuggingFace across 31 dataset configs
  2. Embed each prompt using BGE-small-en-v1.5 (same encoder as Gate 1)
  3. Train a 3-class MLP: FLASH / DEEPSEEK / QWEN235B
  4. Evaluate via 5-fold CV and compare against current LogReg Gate 0
  5. Save model + label encoder to router_inference/config/

Usage:
    cd RouterArena
    uv run python3 scripts/build_domain_classifier.py

Outputs:
    data/domain_classifier_labels.jsonl      — raw labeled corpus
    router_inference/config/chuzom-domain-classifier.joblib  — MLP pipeline
    router_inference/config/chuzom-domain-label-encoder.joblib
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression  # baseline comparison

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from scripts.domain_dataset_map import DOMAIN_MAP

LABELS_PATH = ROOT / "data" / "domain_classifier_labels.jsonl"
# Saves in the same artifact format as build_proxy_classifier.py so Gate 0 needs no changes:
# {"classifier": mlp, "label_encoder": le, "models": [...]}
MODEL_PATH = ROOT / "router_inference" / "config" / "chuzom-domain-classifier.joblib"
PROXY_PATH = ROOT / "router_inference" / "config" / "chuzom-proxy-classifier.joblib"  # Gate 0 slot
BGE_MODEL = "BAAI/bge-small-en-v1.5"

# ── Format functions ──────────────────────────────────────────────────────────

def _options_block(choices: list[str]) -> str:
    letters = "ABCDEFGHIJ"
    return "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))


def fmt_arc_mcq(entry: dict) -> str | None:
    q = entry.get("question", "")
    choices = entry.get("choices", {})
    if isinstance(choices, dict):
        texts = choices.get("text", [])
    else:
        texts = choices
    if not q or not texts:
        return None
    return (
        f"Context: None\n\nQuestion: {q}\n\nOptions:\n{_options_block(texts)}\n\n"
        "Provide the correct letter choice."
    )


def fmt_mmlupro_mcq(entry: dict) -> str | None:
    q = entry.get("question", "")
    options = entry.get("options", [])
    if not q or not options:
        return None
    return (
        f"Context: None\n\nQuestion: {q}\n\nOptions:\n{_options_block(options)}\n\n"
        "Provide the correct letter choice."
    )


def fmt_mmlu_mcq(entry: dict) -> str | None:
    q = entry.get("question", "")
    choices = entry.get("choices", [])
    if not q or not choices:
        return None
    return (
        f"Context: None\n\nQuestion: {q}\n\nOptions:\n{_options_block(choices)}\n\n"
        "Provide the correct letter choice."
    )


def fmt_medqa_mcq(entry: dict) -> str | None:
    q = entry.get("question", "")
    options = entry.get("options", {})
    if isinstance(options, dict):
        # Preserve letter ordering (A/B/C/D) if keys are letter labels
        keys = sorted(options.keys())
        opts_str = "\n".join(f"{k}. {options[k]}" for k in keys)
    elif isinstance(options, list):
        opts_str = _options_block(options)
    else:
        return None
    if not q or not opts_str:
        return None
    return (
        f"Context: None\n\nQuestion: {q}\n\nOptions:\n{opts_str}\n\n"
        "Provide the correct letter choice."
    )


def fmt_medmcqa_mcq(entry: dict) -> str | None:
    q = entry.get("question", "")
    opts = [entry.get(k, "") for k in ("opa", "opb", "opc", "opd") if entry.get(k)]
    if not q or not opts:
        return None
    return (
        f"Context: None\n\nQuestion: {q}\n\nOptions:\n{_options_block(opts)}\n\n"
        "Provide the correct letter choice."
    )


def fmt_triviaqa(entry: dict) -> str | None:
    q = entry.get("question", "")
    if not q:
        return None
    return f"Context: None\n\nQuestion: {q}\n\nAnswer in one or two words."


def fmt_qanta_qa(entry: dict) -> str | None:
    q = (
        entry.get("text") or entry.get("question") or
        entry.get("first_sentence") or ""
    )
    if not q:
        return None
    return f"Context: None\n\nQuestion: {q}\n\nAnswer in one or two words."


def fmt_ethics_binary(entry: dict) -> str | None:
    scenario = entry.get("input", "")
    if not scenario:
        return None
    return (
        f"Context: None\n\nQuestion: Is the following morally acceptable?\n\n"
        f"{scenario}\n\nOptions:\nA. Yes\nB. No\n\nProvide the correct letter choice."
    )


def fmt_commonsenseqa_mcq(entry: dict) -> str | None:
    q = entry.get("question", "")
    choices = entry.get("choices", {})
    if isinstance(choices, dict):
        texts = choices.get("text", [])
    else:
        texts = [c.get("text", "") for c in choices] if choices else []
    if not q or not texts:
        return None
    return (
        f"Context: None\n\nQuestion: {q}\n\nOptions:\n{_options_block(texts)}\n\n"
        "Provide the correct letter choice."
    )


def fmt_narrative_qa(entry: dict) -> str | None:
    q = (
        entry.get("question", {}).get("text", "") if isinstance(entry.get("question"), dict)
        else entry.get("question", "")
    )
    doc = entry.get("document", {})
    if isinstance(doc, dict):
        summary = doc.get("summary", {})
        ctx = summary.get("text", "") if isinstance(summary, dict) else str(summary)
    else:
        ctx = ""
    if not q:
        return None
    ctx_block = f"Context: {ctx[:300]}..." if ctx else "Context: None"
    return f"{ctx_block}\n\nQuestion: {q}\n\nAnswer in one sentence."


def fmt_gsm8k_math(entry: dict) -> str | None:
    q = entry.get("question", "")
    if not q:
        return None
    return f"Context: None\n\nQuestion: {q}\n\nSolve step by step."


def fmt_mathqa_mcq(entry: dict) -> str | None:
    q = entry.get("Problem", "") or entry.get("problem", "")
    opts_str = entry.get("options", "")
    if not q:
        return None
    return (
        f"Context: None\n\nQuestion: {q}\n\nOptions: {opts_str}\n\n"
        "Provide the correct letter choice."
    )


def fmt_competition_math(entry: dict) -> str | None:
    q = entry.get("problem", "")
    if not q:
        return None
    return f"Context: None\n\nQuestion: {q}\n\nSolve and provide the final answer."


def fmt_aime_math(entry: dict) -> str | None:
    # AI-MO/aimo-validation-aime uses "problem" key
    q = entry.get("problem", "") or entry.get("Problem", "")
    if not q:
        return None
    return f"Context: None\n\nQuestion: {q}\n\nSolve and provide the final numerical answer."


def fmt_humaneval_code(entry: dict) -> str | None:
    prompt = entry.get("prompt", "")
    if not prompt:
        return None
    return f"Complete the following Python function:\n\n```python\n{prompt}\n```"


def fmt_mbpp_code(entry: dict) -> str | None:
    q = entry.get("text", "")
    if not q:
        return None
    return f"Write a Python function: {q}"


def fmt_superglue_wic(entry: dict) -> str | None:
    word = entry.get("word", "")
    s1 = entry.get("sentence1", "")
    s2 = entry.get("sentence2", "")
    if not word or not s1 or not s2:
        return None
    return (
        f"Context: None\n\nQuestion: Is the word '{word}' used in the same sense "
        f"in both sentences?\n\nSentence 1: {s1}\nSentence 2: {s2}\n\n"
        "Options:\nA. Yes\nB. No\n\nProvide the correct letter choice."
    )


def fmt_superglue_record(entry: dict) -> str | None:
    passage = entry.get("passage", "")
    query = entry.get("query", "")
    if not passage or not query:
        return None
    return (
        f"Context: {passage[:400]}\n\nQuestion: {query}\n\n"
        "Fill in the blank with the correct entity."
    )


def fmt_superglue_multirc(entry: dict) -> str | None:
    para = entry.get("paragraph", "")
    question = entry.get("question", "")
    answer = entry.get("answer", "")
    if not question:
        return None
    ctx = para[:300] if para else "None"
    return (
        f"Context: {ctx}\n\nQuestion: {question}\nCandidate answer: {answer}\n\n"
        "Is the answer correct? Options:\nA. Yes\nB. No\n\nProvide the correct letter choice."
    )


def fmt_superglue_copa(entry: dict) -> str | None:
    premise = entry.get("premise", "")
    choice1 = entry.get("choice1", "")
    choice2 = entry.get("choice2", "")
    q_type = entry.get("question", "cause")
    if not premise or not choice1 or not choice2:
        return None
    q = f"What was the {q_type} of '{premise}'?"
    return (
        f"Context: None\n\nQuestion: {q}\n\nOptions:\nA. {choice1}\nB. {choice2}\n\n"
        "Provide the correct letter choice."
    )


def fmt_wmt_translation(entry: dict) -> str | None:
    trans = entry.get("translation", {})
    if not trans:
        return None
    pairs = list(trans.items())
    if len(pairs) < 2:
        return None
    src_lang, src_text = pairs[0]
    tgt_lang, _ = pairs[1]
    return (
        f"Context: None\n\nTranslate the following from {src_lang} to {tgt_lang}:\n\n"
        f"{src_text[:300]}"
    )


def fmt_chess_generic(entry: dict) -> str | None:
    # Lichess puzzles: FEN + Moves fields
    fen = entry.get("FEN") or entry.get("fen") or ""
    moves = entry.get("Moves") or entry.get("moves") or ""
    text = entry.get("instruction") or entry.get("text") or entry.get("input") or ""
    if fen:
        return (
            f"Chess position (FEN): {fen}\n\n"
            f"Find the best move sequence. Previous moves: {moves}"
        )
    if not text:
        return None
    return f"Chess position analysis:\n\n{str(text)[:400]}"


def fmt_finqa_generic(entry: dict) -> str | None:
    # ChanceFocus/flare-finqa uses "query" and "text" (financial context passage)
    q = entry.get("query", "") or entry.get("question", "")
    ctx_text = entry.get("text", "") or entry.get("context", "")
    table = entry.get("table", [])
    if not q:
        return None
    if ctx_text:
        ctx = ctx_text[:400]
    elif table and isinstance(table, list):
        ctx = "Table:\n" + "\n".join(" | ".join(str(c) for c in row) for row in table[:5])
    else:
        ctx = "None"
    return f"Context: {ctx}\n\nQuestion: {q}\n\nProvide the numerical answer."


def fmt_truthfulqa_mcq(entry: dict) -> str | None:
    # truthful_qa/multiple_choice: keys = question, mc1_targets (dict with choices+labels)
    # mc1_targets can have >10 choices — cap at 10 to stay within _options_block's A-J range
    q = entry.get("question", "")
    mc = entry.get("mc1_targets", {})
    choices = mc.get("choices", []) if isinstance(mc, dict) else []
    if not q or not choices:
        return None
    return (
        f"Context: None\n\nQuestion: {q}\n\nOptions:\n{_options_block(choices[:10])}\n\n"
        "Provide the correct letter choice."
    )


def fmt_winogrande(entry: dict) -> str | None:
    sentence = entry.get("sentence", "")
    opt1 = entry.get("option1", "")
    opt2 = entry.get("option2", "")
    if not sentence or not opt1 or not opt2:
        return None
    return (
        f"Context: None\n\nFill in the blank:\n{sentence}\n\n"
        f"Options:\nA. {opt1}\nB. {opt2}\n\nProvide the correct letter choice."
    )


def fmt_generic_mcq(entry: dict) -> str | None:
    q = entry.get("question", "") or entry.get("Question", "")
    choices = (
        entry.get("choices") or entry.get("options") or
        entry.get("Options") or []
    )
    if not q:
        return None
    if choices:
        if isinstance(choices, dict):
            texts = list(choices.values())
        elif isinstance(choices, list):
            texts = [c if isinstance(c, str) else str(c) for c in choices]
        else:
            texts = []
        opts = _options_block(texts) if texts else ""
        return (
            f"Context: None\n\nQuestion: {q}\n\nOptions:\n{opts}\n\n"
            "Provide the correct letter choice."
        )
    return f"Context: None\n\nQuestion: {q}\n\nAnswer briefly."


FORMAT_REGISTRY = {
    "arc_mcq": fmt_arc_mcq,
    "mmlupro_mcq": fmt_mmlupro_mcq,
    "mmlu_mcq": fmt_mmlu_mcq,
    "medqa_mcq": fmt_medqa_mcq,
    "medmcqa_mcq": fmt_medmcqa_mcq,
    "triviaqa": fmt_triviaqa,
    "qanta_qa": fmt_qanta_qa,
    "ethics_binary": fmt_ethics_binary,
    "commonsenseqa_mcq": fmt_commonsenseqa_mcq,
    "narrative_qa": fmt_narrative_qa,
    "gsm8k_math": fmt_gsm8k_math,
    "mathqa_mcq": fmt_mathqa_mcq,
    "competition_math": fmt_competition_math,
    "aime_math": fmt_aime_math,
    "humaneval_code": fmt_humaneval_code,
    "mbpp_code": fmt_mbpp_code,
    "superglue_wic": fmt_superglue_wic,
    "superglue_record": fmt_superglue_record,
    "superglue_multirc": fmt_superglue_multirc,
    "superglue_copa": fmt_superglue_copa,
    "wmt_translation": fmt_wmt_translation,
    "chess_generic": fmt_chess_generic,
    "finqa_generic": fmt_finqa_generic,
    "winogrande": fmt_winogrande,
    "truthfulqa_mcq": fmt_truthfulqa_mcq,
    "generic_mcq": fmt_generic_mcq,
}

# ── Download and format ───────────────────────────────────────────────────────

def download_and_format() -> list[dict]:
    records: list[dict] = []

    for cfg in DOMAIN_MAP:
        hf_path = cfg["hf_path"]
        hf_name = cfg.get("hf_name")
        split = cfg["split"]
        sample_n = cfg["sample_n"]
        label = cfg["label"]
        fmt_key = cfg["format_fn"]
        fmt_fn = FORMAT_REGISTRY.get(fmt_key, fmt_generic_mcq)

        print(f"  Loading {hf_path}/{hf_name or ''} [{split}] → {label} ...", flush=True)
        try:
            ds = load_dataset(
                hf_path,
                hf_name,
                split=split,
            )
        except Exception as e:
            print(f"    SKIP: {e}", flush=True)
            continue

        added = 0
        for i, entry in enumerate(ds):
            if added >= sample_n:
                break
            prompt = fmt_fn(dict(entry))
            if not prompt or len(prompt) < 20:
                continue
            records.append({
                "prompt": prompt,
                "label": label,
                "source": f"{hf_path}/{hf_name or ''}",
                "ra_datasets": cfg["ra_datasets"],
            })
            added += 1

        print(f"    → {added} examples", flush=True)

    return records


# ── Embed + train ─────────────────────────────────────────────────────────────

def embed_prompts(prompts: list[str], model: SentenceTransformer) -> np.ndarray:
    print(f"  Embedding {len(prompts)} prompts with BGE-small ...", flush=True)
    return model.encode(
        prompts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


def train_and_evaluate(X: np.ndarray, y: np.ndarray, label_names: list[str]) -> MLPClassifier:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nBaseline — LogisticRegression:", flush=True)
    lr = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced", random_state=42)
    lr_scores = cross_val_score(lr, X, y, cv=cv, scoring="accuracy")
    print(f"  CV accuracy: {lr_scores.mean():.4f} ± {lr_scores.std():.4f}", flush=True)

    print("\nMLP classifier:", flush=True)
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size=128,
        learning_rate="adaptive",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=False,
    )
    mlp_scores = cross_val_score(mlp, X, y, cv=cv, scoring="accuracy")
    print(f"  CV accuracy: {mlp_scores.mean():.4f} ± {mlp_scores.std():.4f}", flush=True)

    # Per-class breakdown
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import classification_report
    y_pred = cross_val_predict(mlp, X, y, cv=cv)
    print("\nClassification report (MLP, CV):", flush=True)
    print(classification_report(y, y_pred, target_names=label_names), flush=True)

    # Final fit on full data
    print("Fitting final MLP on full dataset ...", flush=True)
    mlp.fit(X, y)
    return mlp, float(mlp_scores.mean()), float(mlp_scores.std())


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Download (or load cache)
    if LABELS_PATH.exists():
        print(f"Loading cached labels from {LABELS_PATH} ...", flush=True)
        with open(LABELS_PATH) as f:
            records = [json.loads(l) for l in f]
        print(f"  {len(records)} examples loaded", flush=True)
    else:
        print("Downloading datasets ...", flush=True)
        records = download_and_format()
        print(f"\nTotal examples: {len(records)}", flush=True)
        with open(LABELS_PATH, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"Saved to {LABELS_PATH}", flush=True)

    from collections import Counter
    label_dist = Counter(r["label"] for r in records)
    print(f"\nLabel distribution: {dict(label_dist)}", flush=True)

    # Step 2: Embed
    prompts = [r["prompt"] for r in records]
    labels_raw = [r["label"] for r in records]

    bge = SentenceTransformer(BGE_MODEL)
    X = embed_prompts(prompts, bge)

    le = LabelEncoder()
    y = le.fit_transform(labels_raw)
    print(f"\nClasses: {list(le.classes_)}", flush=True)

    # Step 3: Train
    print("\n── Training ─────────────────────────────────────────────────────\n", flush=True)
    mlp, mlp_cv_mean, mlp_cv_std = train_and_evaluate(X, y, list(le.classes_))

    # Step 4: Save — compatible artifact format for Gate 0 in chuzom_router_v2.py
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    ROUTING_MODELS = [
        "google/gemini-3.1-flash-lite",
        "deepseek/deepseek-v4-flash",
        "qwen/qwen3-235b-a22b-2507",
        "qwen/qwen3-next-80b-a3b-instruct",
    ]

    # Map classifier classes (FLASH/DEEPSEEK/QWEN235B) → routing model names
    CLASS_TO_MODEL = {
        "FLASH": "google/gemini-3.1-flash-lite",
        "DEEPSEEK": "deepseek/deepseek-v4-flash",
        "QWEN235B": "qwen/qwen3-235b-a22b-2507",
    }

    # Build a label encoder that returns routing model names (not class labels)
    from sklearn.preprocessing import LabelEncoder as LE
    model_le = LE()
    model_names_per_sample = [CLASS_TO_MODEL.get(lbl, lbl) for lbl in labels_raw]
    model_le.fit(model_names_per_sample)

    # Retrain MLP with model-name labels for direct compatibility
    y_model = model_le.transform(model_names_per_sample)
    mlp_final = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size=128,
        learning_rate="adaptive",
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=False,
    )
    mlp_final.fit(X, y_model)

    artifact = {
        "classifier": mlp_final,
        "label_encoder": model_le,
        "models": ROUTING_MODELS,
        "metadata": {
            "version": "domain-classifier-v1.0",
            "n_train": len(records),
            "classes": list(le.classes_),
            "cv_accuracy_mlp": mlp_cv_mean,
            "cv_std_mlp": mlp_cv_std,
        },
    }
    joblib.dump(artifact, MODEL_PATH)
    # Also write to proxy slot so Gate 0 picks it up automatically
    joblib.dump(artifact, PROXY_PATH)
    print(f"\nSaved domain classifier → {MODEL_PATH}", flush=True)
    print(f"Installed as Gate 0    → {PROXY_PATH}", flush=True)

    # Step 5: Sanity check
    print("\n── Sanity checks ────────────────────────────────────────────────\n", flush=True)
    test_cases = [
        ("Context: None\n\nQuestion: What is the capital of France?\n\nOptions:\nA. Berlin\nB. Paris\nC. Rome\nD. Madrid\n\nProvide the correct letter choice.", "FLASH"),
        ("Complete the following Python function:\n\n```python\ndef two_sum(nums, target):\n    # return indices\n```", "FLASH"),
        ("Context: None\n\nQuestion: Find all positive integers n such that n^2 + 2 is divisible by n + 1.\n\nSolve and provide the final numerical answer.", "DEEPSEEK"),
        ("Is the following morally acceptable?\n\nI returned the extra change the cashier mistakenly gave me.", "FLASH"),
    ]
    for prompt, expected in test_cases:
        emb = bge.encode([prompt], normalize_embeddings=True)
        probs = mlp.predict_proba(emb)[0]
        pred_idx = np.argmax(probs)
        pred_label = le.classes_[pred_idx]
        conf = probs[pred_idx]
        status = "✓" if pred_label == expected else "✗"
        print(f"  {status} Predicted: {pred_label} ({conf:.2f}) | Expected: {expected}", flush=True)
        print(f"    Prompt: {prompt[:80].strip()!r}", flush=True)


if __name__ == "__main__":
    main()
