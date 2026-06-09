from hybrid_router.budget_curves import BudgetCurves
from collections import defaultdict
import numpy as np

def fit_budget_curves(records, anchor_budgets=None) -> BudgetCurves:
    """
    Default anchor_budgets is [80, 150, 200, 400, 800, 1500] if not provided.
    Group records by model, then by budget. For each (model, budget) group, compute the mean accuracy.
    For each model, build the 6-element anchor array by looking up the mean accuracy at each anchor budget.
    If an anchor budget has no data (e.g. gemini-flash which only has budget=None), fill it with the mean accuracy from the budget=None records for that model.
    Return BudgetCurves(anchors).
    """
    anchor_budgets = anchor_budgets or [80, 150, 200, 400, 800, 1500]
    buckets = defaultdict(lambda: defaultdict(list))
    for rec in records:
        buckets[rec["model"]][rec["budget"]].append(rec["accuracy"])
    
    anchors = {}
    for model, budget_map in buckets.items():
        fallback = float(np.mean(budget_map[None])) if budget_map[None] else 0.5

        anchor_array = np.zeros(len(anchor_budgets))
        for i, b in enumerate(anchor_budgets):
            if budget_map[b]:
                anchor_array[i] = float(np.mean(budget_map[b]))
            else:
                anchor_array[i] = fallback
        
        anchors[model] = anchor_array
    
    return BudgetCurves(anchors)

if __name__ == "__main__":
    import os
    from training.dataset import load_r2bench, MODEL_NAME_MAP

    records = load_r2bench("./data/r2bench", MODEL_NAME_MAP)
    curves = fit_budget_curves(records)

    os.makedirs("./checkpoints/hybrid-router", exist_ok=True)
    curves.save("./checkpoints/hybrid-router/curves.npz")
    print("Saved curves.npz")
    for model, arr in curves.anchors.items():
        print(f"  {model}: {[round(v, 3) for v in arr.tolist()]}")