# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""AINative Router adapter.

Content-only routing: the question body (first/last paragraph dropped, field labels and option letters stripped) is
classified into a content category by a classifier trained only on external calibration data, and each category maps
to a model chosen on that data. The router reads no RouterArena files; policy and classifier are hash-frozen
(artifacts/FREEZE.json) and verified at load.
"""

import os
import sys

from router_inference.router.base_router import BaseRouter

_PROJECT_ROOT = os.environ.get(
    "ARENA_ROUTER_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")),
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class AINativeRouter(BaseRouter):
    def __init__(self, router_name: str):
        super().__init__(router_name)
        from arena_router.router import ArenaRouter as _Impl

        self._impl = _Impl()
        unknown = set(self._impl.policy.models) - set(self.models)
        if unknown:
            raise ValueError(f"policy models not in config: {unknown}")

    def _get_prediction(self, query: str) -> str:
        return self._impl.route(query)
