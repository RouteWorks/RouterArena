# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from scipy.interpolate import PchipInterpolator


class BudgetCurves:
    def __init__(self, anchors: dict[str, np.ndarray]):
        """
        Takes anchors: dict[str, np.ndarray]. Each value is a numpy array of shape (6,)
        representing the mean quality at each of the 6 anchor budgets. Stores it as self.anchors,
        and stores the fixed budget list as self.budgets = np.array([80, 150, 200, 400, 800, 1500]).
        """
        self.anchors = anchors
        self.budgets = np.array([80, 150, 200, 400, 800, 1500])

    def quality_at_budget(self, model_name: str, budget: float) -> float:
        """
        Builds a PCHIP interpolator from scipy.interpolate.PchipInterpolator using self.budgets as
        x and self.anchors[model_name] as y, then evaluates it at budget. Clamps the result to [0.0, 1.0]
        using float(np.clip(..., 0.0, 1.0)) before returning.
        """
        interpolator = PchipInterpolator(self.budgets, self.anchors[model_name])
        evaluation = interpolator(budget)
        clamp = float(np.clip(evaluation, 0.0, 1.0))

        return clamp

    def save(self, path: str):
        """
        Saves self.anchors to an .npz file using np.savez. The keyword arguments to np.savez should be the
        model names as keys and the arrays as values (but model names contain hyphens which aren't valid Python keyword arguments).
        Thus, we'll need to use np.savez(path, **self.anchors) to unpack the dict.
        """
        np.savez(path, **self.anchors)

    @classmethod
    def load(cls, path: str):
        """
        A class method that loads the .npz file with np.load and reconstructs the dict.
        """
        data = np.load(path)

        anchors = {k: v for k, v in data.items()}
        return cls(anchors)
