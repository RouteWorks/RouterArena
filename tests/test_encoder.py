from hybrid_router.encoder import HybridRouterEncoder
import numpy as np

def test_single_encode():
    encoder = HybridRouterEncoder()
    result = encoder.encode("hello world")
    assert result.shape == (768,)

def test_batch_encode():
    encoder = HybridRouterEncoder()
    result = encoder.encode(["a", "b"])
    assert result.shape == (2, 768)

def test_deterministic():
    encoder = HybridRouterEncoder()
    result1 = encoder.encode("hello world")
    result2 = encoder.encode("hello world")
    assert np.array_equal(result1, result2)