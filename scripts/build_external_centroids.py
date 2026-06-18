# SPDX-FileCopyrightText: 2026 Chuzom (github.com/ypollak2/chuzom)
# SPDX-License-Identifier: MIT
"""
Build per-model centroid embeddings from EXTERNAL public datasets.

Uses datasets that represent each routing model's task domain, independently
of the RouterArena test set. This avoids test-data leakage while giving
semantically meaningful, paraphrase-invariant centroid clusters.

Model -> Task domain -> External source:
  deepseek         : math, translation, step-by-step reasoning  -> MATH, WMT, FinQA
  gemini-2.0       : reading comprehension with passage context  -> SQuAD, NarrativeQA
  gemini-lite      : short trivia MCQ, medical MCQ               -> MMLU, OpenTDB proxy
  qwen3-235b       : hard STEM, formal reasoning, physics/chem   -> MMLUPro STEM
  qwen3-80b        : word sense disambiguation, coreference      -> SuperGLUE WiC/WSC

Usage:
    uv run python scripts/build_external_centroids.py
"""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np

OUTPUT_PATH = Path("router_inference/config/chuzom-centroids.npz")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
SAMPLES_PER_MODEL = 500  # prompts per model centroid

ROUTING_MODELS = [
    "google/gemini-2.0-flash-001",
    "google/gemini-3.1-flash-lite",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-next-80b-a3b-instruct",
]


# ── Prompt constructors (one per external dataset) ────────────────────────────


def load_math_prompts(n: int) -> list[str]:
    """MATH dataset: competition math problems -> deepseek."""
    from datasets import load_dataset

    # Try multiple MATH dataset paths
    for path, name, split in [
        ("hendrycks/competition_math", "all", "train"),
        ("EleutherAI/hendrycks_math", "algebra", "train"),
        ("competition_math", None, "train"),
    ]:
        try:
            kwargs = {"split": split}
            if name:
                kwargs["name"] = name
            ds = load_dataset(path, **kwargs)
            prompts: list[str] = []
            for row in ds:
                if len(prompts) >= n:
                    break
                p = row.get("problem", "") or row.get("question", "")
                if p:
                    prompts.append(
                        f"Please solve the following mathematical problem step by step. {p}"
                    )
            if prompts:
                return prompts[:n]
        except Exception:
            continue

    # Fallback: GSM8K (grade school math — simpler but semantically correct)
    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
        prompts = []
        for row in ds:
            if len(prompts) >= n:
                break
            p = row.get("question", "")
            if p:
                prompts.append(
                    f"Please solve the following mathematical problem step by step. {p}"
                )
        return prompts[:n]
    except Exception as e:
        print(
            f"  GSM8K also failed ({e}), using synthetic math prompts", file=sys.stderr
        )
        templates = [
            "Please solve the following mathematical problem step by step. Find the value of x: {a}x + {b} = {c}",
            "Please solve the following mathematical problem step by step. Calculate the integral of f(x) = {a}x^{b} from 0 to {c}.",
            "Please solve the following mathematical problem step by step. If a train travels at {a} km/h for {b} hours, how far does it travel?",
        ]
        prompts = []
        for i in range(n):
            t = templates[i % len(templates)]
            prompts.append(t.format(a=i + 1, b=i + 2, c=i + 3))
        return prompts


def load_wmt_prompts(n: int) -> list[str]:
    """WMT translation pairs -> deepseek."""
    from datasets import load_dataset

    try:
        ds = load_dataset("wmt16", "de-en", split="train", trust_remote_code=True)
        prompts: list[str] = []
        for row in ds:
            if len(prompts) >= n:
                break
            trans = row.get("translation", {})
            en = trans.get("en", "")
            de = trans.get("de", "")
            if en and de:
                prompts.append(f"Translate from English to German: {en}")
        return prompts[:n]
    except Exception as e:
        print(
            f"  WMT load failed ({e}), using synthetic translation prompts",
            file=sys.stderr,
        )
        langs = [
            ("French", "Spanish"),
            ("German", "English"),
            ("English", "Japanese"),
            ("Chinese", "English"),
            ("Russian", "English"),
            ("Arabic", "English"),
        ]
        samples = []
        for i in range(n):
            src, tgt = langs[i % len(langs)]
            samples.append(
                f"Translate the following text from {src} to {tgt}: This is sample sentence number {i}."
            )
        return samples


def load_squad_prompts(n: int) -> list[str]:
    """SQuAD reading comprehension -> gemini-2.0."""
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad", split="train")
    prompts: list[str] = []
    seen = set()
    for row in ds:
        if len(prompts) >= n:
            break
        ctx = row.get("context", "")[:600]
        q = row.get("question", "")
        key = q[:80]
        if ctx and q and key not in seen:
            seen.add(key)
            prompts.append(
                f"Please read the following context and answer the question.\n"
                f"Context: {ctx}\nQuestion: {q}"
            )
    return prompts[:n]


def load_mmlu_prompts(n: int) -> list[str]:
    """MMLU multiple-choice (general) -> gemini-lite."""
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    letters = ["A", "B", "C", "D"]
    prompts: list[str] = []
    for row in ds:
        if len(prompts) >= n:
            break
        q = row.get("question", "")
        choices = row.get("choices", [])
        if q and len(choices) >= 2:
            opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices[:4]))
            prompts.append(
                f"Please read the following multiple-choice questions and provide "
                f"the correct answer.\nContext: None\nQuestion: {q}\n{opts}"
            )
    return prompts[:n]


def load_mmlu_pro_stem_prompts(n: int) -> list[str]:
    """MMLUPro STEM (math/physics/chem/biology) -> qwen3-235b."""
    from datasets import load_dataset

    stem_subjects = {"math", "physics", "chemistry", "biology", "engineering"}
    try:
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", trust_remote_code=True)
        letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        prompts: list[str] = []
        for row in ds:
            if len(prompts) >= n:
                break
            subj = (row.get("category") or row.get("subject") or "").lower()
            if not any(s in subj for s in stem_subjects):
                continue
            q = row.get("question", "")
            choices = row.get("options", row.get("choices", []))
            if q and len(choices) >= 2:
                opts = "\n".join(
                    f"{letters[i]}. {c}" for i, c in enumerate(choices[:10])
                )
                prompts.append(
                    f"Please read the following multiple-choice questions and provide "
                    f"the correct answer.\nContext: None\nQuestion: {q}\n{opts}"
                )
        if len(prompts) < n // 2:
            raise ValueError("Insufficient STEM rows")
        return prompts[:n]
    except Exception as e:
        print(
            f"  MMLUPro load failed ({e}), using ARC-Challenge STEM proxy",
            file=sys.stderr,
        )
        ds2 = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        letters = ["A", "B", "C", "D"]
        prompts = []
        for row in ds2:
            if len(prompts) >= n:
                break
            q = row.get("question", "")
            choices = row.get("choices", {})
            texts = choices.get("text", [])
            if q and texts:
                opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(texts[:4]))
                prompts.append(
                    f"Please read the following multiple-choice questions and provide "
                    f"the correct answer.\nContext: None\nQuestion: {q}\n{opts}"
                )
        return prompts[:n]


def load_wic_prompts(n: int) -> list[str]:
    """SuperGLUE WiC (word-in-context) -> qwen3-80b."""
    from datasets import load_dataset

    try:
        ds = load_dataset("super_glue", "wic", split="train", trust_remote_code=True)
        prompts: list[str] = []
        for row in ds:
            if len(prompts) >= n:
                break
            word = row.get("word", "")
            s1 = row.get("sentence1", "")
            s2 = row.get("sentence2", "")
            if word and s1 and s2:
                prompts.append(
                    f'Consider the word "{word}".\n'
                    f"Sentence 1: {s1}\n"
                    f"Sentence 2: {s2}\n"
                    f"Is the word used in the same sense in both sentences? Answer Yes or No."
                )
        if len(prompts) < n // 2:
            raise ValueError("Insufficient WiC rows")
        return prompts[:n]
    except Exception as e:
        print(f"  WiC load failed ({e}), using WSC fallback", file=sys.stderr)
        try:
            ds2 = load_dataset(
                "super_glue", "wsc", split="train", trust_remote_code=True
            )
            prompts = []
            for row in ds2:
                if len(prompts) >= n:
                    break
                text = row.get("text", "")
                span1 = row.get("span1_text", "")
                span2 = row.get("span2_text", "")
                if text and span1 and span2:
                    prompts.append(
                        f'In the "Text" below, does the pronoun "{span2}" refer to "{span1}"?\n'
                        f"Text: {text}\nAnswer: "
                    )
            return prompts[:n]
        except Exception as e2:
            print(
                f"  WSC also failed ({e2}), using synthetic word-sense prompts",
                file=sys.stderr,
            )
            words = [
                "bank",
                "bar",
                "bat",
                "bear",
                "book",
                "bright",
                "can",
                "charge",
                "cold",
                "crane",
                "date",
                "fall",
                "file",
                "fine",
                "firm",
            ]
            prompts = []
            for i in range(n):
                w = words[i % len(words)]
                prompts.append(
                    f'Consider the word "{w}" as used in the following two sentences. '
                    f"Does it have the same meaning in both? Sentence 1: The {w} was near the river. "
                    f"Sentence 2: She went to the {w} to deposit money. Answer Yes or No."
                )
            return prompts


# ── Embedding helper ──────────────────────────────────────────────────────────


def mean_pool(token_embeddings, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def embed_texts(texts: list, tokenizer, model_obj, batch_size: int = 64) -> np.ndarray:
    import torch

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = model_obj(**encoded)
        embeddings = mean_pool(out.last_hidden_state, encoded["attention_mask"])
        norms = torch.norm(embeddings, dim=1, keepdim=True).clamp(min=1e-9)
        embeddings = (embeddings / norms).cpu().numpy()
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    # Model -> (label, loader_fn)
    model_loaders: list[tuple[str, str, Optional[Callable[[int], list[str]]]]] = [
        ("google/gemini-2.0-flash-001", "Reading comprehension", load_squad_prompts),
        ("google/gemini-3.1-flash-lite", "General MCQ (MMLU)", load_mmlu_prompts),
        ("deepseek/deepseek-v4-flash", "Math + Translation", None),  # two sources
        ("qwen/qwen3-235b-a22b-2507", "STEM MCQ (MMLUPro)", load_mmlu_pro_stem_prompts),
        ("qwen/qwen3-next-80b-a3b-instruct", "Word sense (WiC/WSC)", load_wic_prompts),
    ]

    print(f"Loading embedding model: {EMBED_MODEL}", file=sys.stderr)
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    embed_model = AutoModel.from_pretrained(EMBED_MODEL)
    embed_model.train(False)

    centroid_matrix = np.zeros((len(ROUTING_MODELS), 384), dtype=np.float32)

    for model_name, label, loader_fn in model_loaders:
        model_idx = ROUTING_MODELS.index(model_name)
        print(
            f"\nLoading [{label}] for {model_name.split('/')[-1]}...", file=sys.stderr
        )

        if model_name == "deepseek/deepseek-v4-flash":
            # Two sources: math + translation
            n_each = SAMPLES_PER_MODEL // 2
            math_prompts = load_math_prompts(n_each)
            wmt_prompts = load_wmt_prompts(n_each)
            prompts = (math_prompts + wmt_prompts)[:SAMPLES_PER_MODEL]
        else:
            assert loader_fn is not None
            prompts = loader_fn(SAMPLES_PER_MODEL)

        print(f"  Loaded {len(prompts)} prompts", file=sys.stderr)

        # Normalise
        cleaned = [" ".join(p.split())[:2000] for p in prompts if p.strip()]

        print(f"  Embedding {len(cleaned)} prompts...", file=sys.stderr)
        embeddings = embed_texts(cleaned, tokenizer, embed_model)

        centroid = embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroid_matrix[model_idx] = centroid / max(norm, 1e-9)
        print(f"  Centroid computed (norm before L2: {norm:.4f})", file=sys.stderr)

    print(f"\nSaving to {OUTPUT_PATH}...", file=sys.stderr)
    np.savez(
        OUTPUT_PATH,
        centroids=centroid_matrix,
        models=np.array(ROUTING_MODELS),
    )
    print("Done.", file=sys.stderr)

    # Sanity: show pairwise cosine similarities between centroids
    print("\nPairwise centroid cosine similarities:", file=sys.stderr)
    for i, m1 in enumerate(ROUTING_MODELS):
        for j, m2 in enumerate(ROUTING_MODELS):
            if j <= i:
                continue
            sim = float(centroid_matrix[i] @ centroid_matrix[j])
            print(
                f"  {m1.split('/')[-1][:20]} <-> {m2.split('/')[-1][:20]}: {sim:.4f}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
