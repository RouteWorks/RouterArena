# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

from hybrid_router.model_heads import ModelHeadCollection
import numpy as np


def test_single_predict():
    model_collection = ModelHeadCollection(["model1"])
    embedding = np.random.randn(768).astype("float32")
    result = model_collection.predict("model1", embedding)
    assert isinstance(result, float) and 0.0 <= result <= 1.0


def test_predict_all_keys():
    model_collection = ModelHeadCollection(["model1", "model2"])
    embedding = np.random.randn(768).astype("float32")
    results = model_collection.predict_all(embedding)

    assert "model1" in results and "model2" in results


def test_predict_all_values():
    model_collection = ModelHeadCollection(["model1", "model2"])
    embedding = np.random.randn(768).astype("float32")
    results = model_collection.predict_all(embedding)

    assert all(0.0 <= value <= 1.0 for value in results.values())
