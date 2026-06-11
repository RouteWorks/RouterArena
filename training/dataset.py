import json
import re
from pathlib import Path
import numpy as np

MODEL_NAME_MAP = {
    "235b": "qwen/qwen3-235b-a22b-2507",
    "30b": "qwen/qwen3-30b-a3b-instruct-2507",
    "ministral-3b": "mistralai/ministral-3-3b-2512",
}

SKIP_FOLDERS = {
    "gemma-3n-e4b", # wrong schema — no accuracy/cost fields
    "glm-5", # concise only, wrong schema
    "gemini-3-pro", # concise only, wrong schema
    "gpt-5.2", # concise only, wrong schema
    "gpt4o", # concise only, wrong schema
    "haiku", # concise only, no budget sweep
    "gemini-flash", # concise only, no budget sweep
}

def _get_global_index(entry: dict) -> str:
    return entry.get("global_index") or entry.get("global index", "")

def load_r2bench(data_dir, model_name_map) -> list[dict]:
    """
    Walk data_dir/budget_sweep/. For each subfolder, check if it's in model_name_map. Skip if not.
    For each matched folder, read every budget_<N>.json file. Parse the integer N from the filename.
    Read all records, extract global_index, prompt, accuracy, cost. Append with model = canonical name, budget = N.
    Also read concise.json if present. Treat budget as None.
    Return the full flat list.
    """
    records = []

    budget_sweep_dir = Path(data_dir) / "budget_sweep"

    for subfolder in sorted(budget_sweep_dir.iterdir()):
        if not subfolder.is_dir():
            continue
        folder_name = subfolder.name
        if folder_name not in model_name_map:
            continue
        if folder_name in SKIP_FOLDERS:
            continue
        canonical_name = model_name_map[folder_name]

        for json_file in sorted(subfolder.glob("budget_*.json")):
            match = re.fullmatch(r"budget_(\d+)\.json", json_file.name)
            if not match:
                continue
            budget = int(match.group(1))
            
            with open(json_file) as file:
                data = json.load(file)
            for entry in data:
                accuracy = entry.get("accuracy")
                if accuracy is None:
                    continue
                records.append({
                    "global_index": _get_global_index(entry),
                    "prompt": entry["prompt"],
                    "model": canonical_name,
                    "budget": budget,
                    "accuracy": float(accuracy),
                    "cost": float(entry.get("cost", 0.0)),
                })
        
        unlimited_file = subfolder / "budget_unlimited.json"
        concise_file = subfolder / "concise.json"
        source_file = unlimited_file if unlimited_file.exists() else concise_file if concise_file.exists() else None

        if source_file:
            with open(source_file) as file:
                data = json.load(file)
            for entry in data:
                accuracy = entry.get("accuracy")
                if accuracy is None:
                    continue
                records.append({
                    "global_index": _get_global_index(entry),
                    "prompt": entry["prompt"],
                    "model": canonical_name,
                    "budget": None,
                    "accuracy": float(accuracy),
                    "cost": float(entry.get("cost", 0.0)),
                })
        
    return records

def embed_in_chunks(encoder, prompts, chunk_size=256):
    chunks = [encoder.encode(prompts[i: i + chunk_size]) for i in range(0, len(prompts), chunk_size)]
    return np.concatenate(chunks, axis=0)