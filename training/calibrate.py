from hybrid_router.calibration import TemperatureScaler
from training.dataset import embed_in_chunks
import numpy as np
import torch
from scipy.optimize import minimize_scalar

def calibrate(records, encoder, collection) -> TemperatureScaler:
    """
    Filter to numeric-budget records only.
    For each model, collect (raw_logit, true_accuracy) pairs. The raw logit is the pre-simgoid ouput of the head.
    We get this by running the embedding through head.network[:-1] (all layers except the final sigmoid) or equivalently 
    by computing torch.logit(torch.clamp(tensor, 1e-6, 1-1e-6)) on the sigmoid output.
    Find T that minimizes binary cross-entropy between sigmoid(logit / T) and the true label. 
    Use scipy.optimize.minimize_scaler with bounds (0.05, 10.0).
    """
    filtered_records = [r for r in records if r["budget"] is not None]
    temperatures = {}

    for model in collection.heads:
        head = collection.heads[model]
        model_records = [r for r in filtered_records if r["model"] == model]
        if not model_records:
            continue

        prompts = [r["prompt"] for r in model_records]
        labels = np.array([r["accuracy"] for r in model_records], dtype=np.float32)

        embeddings = embed_in_chunks(encoder, prompts)

        x = torch.tensor(embeddings, dtype=torch.float32)
        head.eval()
        with torch.no_grad():
            sigmoid_outputs = head(x)
            logits = torch.logit(
                torch.clamp(sigmoid_outputs, 1e-6, 1 - 1e-6)
            ).squeeze(1).numpy()

        logits_tensor = torch.tensor(logits)
        def BCE(T, logits=logits_tensor, y=labels):
            probs = torch.sigmoid(logits / T).numpy()
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            return -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))

        result = minimize_scalar(BCE, bounds=(0.05, 10.0), method="bounded")
        temperatures[model] = float(result.x)

    return TemperatureScaler(temperatures)