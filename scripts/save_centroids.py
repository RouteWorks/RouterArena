"""
Compute per-model centroid embeddings from v4 routing decisions and save to npz.

Uses BAAI/bge-small-en-v1.5 to embed all routing prompts, then averages
embeddings per model to build centroid vectors. These centroids are loaded
by chuzom_router.py at evaluation time for paraphrase-invariant routing.

Usage:
    uv run python scripts/save_centroids.py
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

DECISIONS_PATH = Path("router_inference/config/chuzom-llm-routing-decisions.json")
PREDICTIONS_PATH = Path("router_inference/predictions/chuzom-llm-router.json")
OUTPUT_PATH = Path("router_inference/config/chuzom-centroids.npz")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

ROUTING_MODELS = [
    "google/gemini-2.0-flash-001",
    "google/gemini-3.1-flash-lite",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-next-80b-a3b-instruct",
]


def extract_prompt_text(prompt: str) -> str:
    prompt = re.sub(
        r"Please read the following multiple-choice questions.*?(?=Context:)",
        "",
        prompt,
        flags=re.DOTALL,
    )
    return " ".join(prompt.split())[:2000]


def mean_pool(token_embeddings, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def embed_texts(texts: list, tokenizer, model_obj, batch_size: int = 128) -> np.ndarray:
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
        # L2 normalize
        norms = torch.norm(embeddings, dim=1, keepdim=True).clamp(min=1e-9)
        embeddings = (embeddings / norms).cpu().numpy()
        all_embeddings.append(embeddings)
        if i % (batch_size * 10) == 0:
            print(f"  Embedded {i}/{len(texts)}", file=sys.stderr)
    return np.vstack(all_embeddings)


def main() -> None:
    print("Loading routing decisions...", file=sys.stderr)
    with open(DECISIONS_PATH) as f:
        decisions = json.load(f)  # sha256 -> model

    print("Loading predictions...", file=sys.stderr)
    with open(PREDICTIONS_PATH) as f:
        predictions = json.load(f)

    # Build prompt -> model map using sha256 keys
    routing_entries = []
    for entry in predictions:
        if entry.get("for_optimality"):
            continue
        prompt = entry.get("prompt", "")
        h = hashlib.sha256(prompt.encode()).hexdigest()
        model = decisions.get(h)
        if model:
            routing_entries.append({"prompt": prompt, "model": model})

    print(f"Routing entries with decisions: {len(routing_entries)}", file=sys.stderr)

    texts = [extract_prompt_text(e["prompt"]) for e in routing_entries]

    print(f"Loading embedding model: {EMBED_MODEL}", file=sys.stderr)
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model_obj = AutoModel.from_pretrained(EMBED_MODEL)
    model_obj.train(False)  # set to inference mode

    print(f"Embedding {len(texts)} prompts...", file=sys.stderr)
    embeddings = embed_texts(texts, tokenizer, model_obj)

    print("Computing centroids...", file=sys.stderr)
    centroid_matrix = np.zeros((len(ROUTING_MODELS), embeddings.shape[1]))
    for i, m in enumerate(ROUTING_MODELS):
        indices = [j for j, e in enumerate(routing_entries) if e["model"] == m]
        if indices:
            centroid_matrix[i] = embeddings[indices].mean(axis=0)
            print(f"  {m}: {len(indices)} prompts", file=sys.stderr)
        else:
            centroid_matrix[i] = embeddings.mean(axis=0)
            print(f"  {m}: no prompts -- using global mean", file=sys.stderr)

    # L2-normalize
    norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
    centroid_matrix = centroid_matrix / np.maximum(norms, 1e-9)

    print(f"Saving centroids to {OUTPUT_PATH}...", file=sys.stderr)
    np.savez(
        OUTPUT_PATH,
        centroids=centroid_matrix,
        models=np.array(ROUTING_MODELS),
    )
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
