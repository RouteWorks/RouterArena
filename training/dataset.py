import json
import re
from pathlib import Path
import numpy as np

MODEL_NAME_MAP = {
    "235b": "qwen/qwen3-235b-a22b-2507",
    "Qwen3-Coder-Next": "Qwen/Qwen3-Coder-Next",
    "gemini-flash": "gemini-2.0-flash-001",
}

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
        canonical_name = model_name_map[folder_name]

        for json_file in sorted(subfolder.glob("budget_*.json")):
            match = re.fullmatch(r"budget_(\d+)\.json", json_file.name)
            if not match:
                continue
            budget = int(match.group(1))
            
            with open(json_file) as file:
                data = json.load(file)
            for entry in data:
                records.append({
                    "global_index": entry["global index"],
                    "prompt": entry["prompt"],
                    "model": canonical_name,
                    "budget": budget,
                    "accuracy": entry["accuracy"],
                    "cost": entry["cost"],
                })
        
        unlimited_file = subfolder / "budget_unlimited.json"
        concise_file = subfolder / "concise.json"
        source_file = unlimited_file if unlimited_file.exists() else concise_file if concise_file.exists() else None

        if source_file:
            with open(source_file) as file:
                data = json.load(file)
            for entry in data:
                records.append({
                    "global_index": entry["global index"],
                    "prompt": entry["prompt"],
                    "model": canonical_name,
                    "budget": None,
                    "accuracy": entry["accuracy"],
                    "cost": entry["cost"],
                })
        
    return records

def embed_in_chunks(encoder, prompts, chunk_size=256):
    chunks = [encoder.encode(prompts[i: i + chunk_size]) for i in range(0, len(prompts), chunk_size)]
    return np.concatenate(chunks, axis=0)