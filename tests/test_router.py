from hybrid_router.router import HybridRouter
from unittest.mock import MagicMock
import numpy as np

def test_route_returns_tuple():
    models = ['model1', 'model2']
    budget_candidates = [50, 100, 200]

    encoder = MagicMock()
    encoder.encode.return_value = np.random.randn(768).astype('float32')

    heads = MagicMock()
    heads.heads = {m: None for m in models}
    heads.predict.return_value = 0.7

    scaler = MagicMock()
    curves = MagicMock()
    curves.quality_at_budget.return_value = 0.6

    cost_model = MagicMock()
    cost_model.estimate.return_value = 0.001

    router = HybridRouter(encoder, heads, scaler, curves, cost_model, budget_candidates)
    result = router.route("test query")
    assert len(result) == 2 and isinstance(result[0], str) and isinstance(result[1], int)

def test_route_returns_valid_model():
    models = ['model1', 'model2']
    budget_candidates = [50, 100, 200]

    encoder = MagicMock()
    encoder.encode.return_value = np.random.randn(768).astype('float32')

    heads = MagicMock()
    heads.heads = {m: None for m in models}
    heads.predict.return_value = 0.7

    scaler = MagicMock()
    curves = MagicMock()
    curves.quality_at_budget.return_value = 0.6

    cost_model = MagicMock()
    cost_model.estimate.return_value = 0.001

    router = HybridRouter(encoder, heads, scaler, curves, cost_model, budget_candidates)
    result = router.route("test query")
    assert result[0] in models

def test_route_returns_valid_budget():
    models = ['model1', 'model2']
    budget_candidates = [50, 100, 200]

    encoder = MagicMock()
    encoder.encode.return_value = np.random.randn(768).astype('float32')

    heads = MagicMock()
    heads.heads = {m: None for m in models}
    heads.predict.return_value = 0.7

    scaler = MagicMock()
    curves = MagicMock()
    curves.quality_at_budget.return_value = 0.6

    cost_model = MagicMock()
    cost_model.estimate.return_value = 0.001
    
    router = HybridRouter(encoder, heads, scaler, curves, cost_model, budget_candidates)
    result = router.route("test query")
    assert result[1] in budget_candidates