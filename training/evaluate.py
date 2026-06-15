from hybrid_router.budget_curves import BudgetCurves
from hybrid_router.calibration import TemperatureScaler
from hybrid_router.cost_model import CostModel
from hybrid_router.model_heads import ModelHeadCollection
from training.dataset import embed_in_chunks
import numpy as np
from collections import defaultdict
import torch
import math


def evaluate(records, encoder, collection, scaler, curves, cost_model) -> dict:
    """
    Runs the full router logic offline against the R2-Bench records and returns metrics.
    Computes the following metrics:
        - mean_accuracy (the average accuracy of the records the router would have selected)
        - mean_cost (average cost of those selections)
        - accuracy_at_budget (a dict mapping each anchor budget to mean accuracy across records routed to that budget)
        - model_distribution (dict mapping each model name to fraction of queries routed to it)
    The input records should be the unlimited-budget records only (budget=None)
    """
    unlimited = [r for r in records if r["budget"] is None]
    if not unlimited:
        return {}

    ground_truth = defaultdict(dict)
    for r in unlimited:
        ground_truth[r["global_index"]][r["model"]] = r["accuracy"]

    seen = set()
    queries = []
    for r in unlimited:
        if r["global_index"] not in seen:
            seen.add(r["global_index"])
            queries.append({"global_index": r["global_index"], "prompt": r["prompt"]})

    prompts = [q["prompt"] for q in queries]
    embeddings = embed_in_chunks(encoder, prompts)

    model_names = list(collection.heads.keys())
    anchor_budgets = list(curves.budgets)

    chosen_models = []
    chosen_budgets = []
    chosen_accuracies = []
    chosen_costs = []

    for idx, query in enumerate(queries):
        emb = embeddings[idx]
        global_index = query["global_index"]

        best_score = -1.0
        best_model = model_names[0]
        best_budget = anchor_budgets[0]

        for model in model_names:
            with torch.no_grad():
                emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
                sigmoid_out = collection.heads[model](emb_tensor).item()

            raw_logit = math.log(max(sigmoid_out, 1e-9) / max(1 - sigmoid_out, 1e-9))
            calibrated = scaler.apply(model, raw_logit)

            for budget in anchor_budgets:
                curve_quality = curves.quality_at_budget(model, budget)
                blended = 0.5 * calibrated + 0.5 * curve_quality
                cost = cost_model.estimate(model, len(query["prompt"].split()), budget)
                score = blended / (cost + 1e-9)

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_budget = budget

        true_acc = ground_truth[global_index].get(best_model, 0.0)
        est_cost = cost_model.estimate(
            best_model, len(query["prompt"].split()), best_budget
        )

        chosen_models.append(best_model)
        chosen_budgets.append(best_budget)
        chosen_accuracies.append(true_acc)
        chosen_costs.append(est_cost)

    accuracy_at_budget = defaultdict(list)
    for acc, bud in zip(chosen_accuracies, chosen_budgets):
        accuracy_at_budget[bud].append(acc)
    accuracy_at_budget = {
        b: float(np.mean(accs)) for b, accs in accuracy_at_budget.items()
    }

    model_distribution = defaultdict(int)
    for m in chosen_models:
        model_distribution[m] += 1
    total = len(chosen_models)
    model_distribution = {m: c / total for m, c in model_distribution.items()}

    return {
        "mean_accuracy": float(np.mean(chosen_accuracies)),
        "mean_cost": float(np.mean(chosen_costs)),
        "accuracy_at_budget": accuracy_at_budget,
        "model_distribution": model_distribution,
    }


if __name__ == "__main__":
    import os
    import torch
    from training.dataset import load_r2bench, MODEL_NAME_MAP
    from hybrid_router.encoder import HybridRouterEncoder
    from hybrid_router.model_heads import ModelHeadCollection
    from hybrid_router.budget_curves import BudgetCurves
    from hybrid_router.calibration import TemperatureScaler
    from hybrid_router.cost_model import CostModel

    records = load_r2bench("./data/r2bench", MODEL_NAME_MAP)
    encoder = HybridRouterEncoder()
    model_names = list(MODEL_NAME_MAP.values())
    ckpt = "./checkpoints/hybrid-router"

    collection = ModelHeadCollection(model_names)
    heads_dir = os.path.join(ckpt, "heads")
    for name in model_names:
        filename = name.replace("/", "_") + ".pt"
        state = torch.load(
            os.path.join(heads_dir, filename), map_location="cpu", weights_only=True
        )
        collection.heads[name].load_state_dict(state)
        collection.heads[name].eval()

    curves = BudgetCurves.load(os.path.join(ckpt, "curves.npz"))
    scaler = TemperatureScaler.load(os.path.join(ckpt, "temperatures.json"))
    cost_model = CostModel.from_json("./model_cost/model_cost.json")

    metrics = evaluate(records, encoder, collection, scaler, curves, cost_model)
    if not metrics:
        print("No unlimited-budget records found for evaluation.")
        print(
            "Check that MODEL_NAME_MAP folders have budget_unlimited.json or concise.json"
        )
    else:
        print(f"mean_accuracy : {metrics['mean_accuracy']:.4f}")
        print(f"mean_cost : {metrics['mean_cost']:.8f}")
        print("model_distribution:")
        for m, frac in metrics["model_distribution"].items():
            print(f"  {m}: {frac:.2%}")
        print("accuracy_at_budget:")
        for b, acc in sorted(metrics["accuracy_at_budget"].items()):
            print(f"  budget={b}: {acc:.4f}")
