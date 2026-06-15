from hybrid_router.budget_curves import BudgetCurves
import numpy as np


def test_interpolated_value():
    curve = BudgetCurves({"model1": np.array([0.4, 0.55, 0.65, 0.72, 0.76, 0.78])})
    quality = curve.quality_at_budget("model1", 300)
    assert 0.65 <= quality <= 0.72


def test_clamp_bounds():
    curve = BudgetCurves({"model1": np.array([0.4, 0.55, 0.65, 0.72, 0.76, 0.78])})
    upper_quality = curve.quality_at_budget("model1", 9999)
    lower_quality = curve.quality_at_budget("model1", 1)
    assert upper_quality <= 1.0 and lower_quality >= 0.0


def test_save_load_roundtrip():
    curve = BudgetCurves({"model1": np.array([0.4, 0.55, 0.65, 0.72, 0.76, 0.78])})
    quality_before = curve.quality_at_budget("model1", 300)
    curve.save("/tmp/test_curves.npz")
    loaded_curve = BudgetCurves.load("/tmp/test_curves.npz")
    quality_after = loaded_curve.quality_at_budget("model1", 300)
    assert quality_before == quality_after
