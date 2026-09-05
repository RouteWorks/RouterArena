# SPDX-License-Identifier: Apache-2.0
"""Shunt's real router engine as a RouterArena ``BaseRouter``.

Thin adapter over the shipped Shunt package (https://github.com/KookaS/shunt,
affiliation @KookaS, open-source): it builds the engine with
``RouterPolicy(strategy="'knn_semantic_cascade'", models=[...])`` over the menu's live
shunt ids and routes each prompt through ``engine.decide``. RouterArena's
stateless single-turn / no-verification regime has no verified-outcome channel,
so Shunt's verified-failure escalation ladder is inert. With the default
``knn_semantic_cascade`` the engine consults a prior-seeded outcome index
(``_PRIOR``: the spec's active verified-outcome corpus — ``qacalib`` QA-domain
cells by default, ``swebench`` SWE-bench cells as an option); with
``session_cascade`` it reduces to the pool's cheapest live model.

Reproducibility: this module runs against Shunt pinned at ``SHUNT_PIN``
(installed from git — Shunt is not on PyPI). When the seeded ``qacalib`` corpus
is shipped next to this file under ``router_inference/router/shunt_seed/`` (the
PR bundles it), the engine reads the seed from THERE — so a clean checkout of
this repo alone can rebuild the index and regenerate identical predictions,
with no env vars and no other checkout. Without the bundled directory the
module falls back to the env-resolved prior root (``SHUNT_ARENA_BENCH_ROOT``,
else the parent of ``SHUNT_SRC``) that the deployer uses for its own runs.

Imports: if the ``SHUNT_SRC`` env var is set, that directory is prepended to
``sys.path`` before ``import shunt`` (so the arena runner can drive a checkout);
otherwise a normal ``import shunt`` (an installed Shunt, at ``SHUNT_PIN``) is
attempted.
"""

import itertools
import os
import sys
from typing import Any

from router_inference.router.base_router import BaseRouter

# The Shunt commit this submission was generated with (install: ``pip install
# "git+https://github.com/KookaS/shunt@<SHUNT_PIN>"``).
SHUNT_PIN = "7ee66af5ddcbd78886a58e5336b21ffc5291da56"

# The full menu (spec `models`, cheapest-first): shunt ids the engine may route
# to (the live pool restricts to those the Shunt registry serves) and the map
# from the engine's shunt id back to the arena name predictions must carry.
_MENU_SHUNT_IDS = ["deepseek-v4-flash", "deepseek-v4-pro"]
_MODEL_TO_ARENA = {
    "deepseek-v4-flash": "deepseek/deepseek-v4-flash",
    "deepseek-v4-pro": "deepseek/deepseek-v4-pro",
}
# The strategy this module was rendered with and, under `knn_semantic_cascade`,
# the prior config (the spec's `prior:` block — source choice + per-source paths
# relative to the resolved prior root) that seeds the engine's outcome index.
# Values are either strings or small nested dicts of strings, so `Any` is the
# honest annotation (a tighter literal type would need the renderer to emit
# typing per corpus).
_STRATEGY = "knn_semantic_cascade"
_PRIOR: dict[str, Any] = {
    "source": "qacalib",
    "swebench": {
        "results_csv": "benchmark/routing/results.csv",
        "challenges_json": "benchmark/routing/data/challenges.json",
        "default_arms": {"deepseek-v4-flash": "high", "deepseek-v4-pro": "high"},
    },
    "qacalib": {
        "dir": "qacalib",
        "sample": "sample.json",
        "results": "results.json",
        "embeddings": "embeddings.json",
    },
}


def _import_shunt():
    src = os.environ.get("SHUNT_SRC")
    if src:
        sys.path.insert(0, src)
    try:
        import shunt  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - re-raised with context for the owner
        raise RuntimeError(
            "could not import the Shunt package (SHUNT_SRC="
            + (repr(src) if src else "unset")
            + "). Point SHUNT_SRC at a Shunt src/ checkout, or install the pinned "
            + "commit: pip install 'git+https://github.com/KookaS/shunt@"
            + SHUNT_PIN
            + "'."
        ) from exc
    return sys.modules["shunt"]


class ShuntSessionCascade(BaseRouter):
    """Route every prompt through the real Shunt engine ('knn_semantic_cascade' strategy)."""

    def __init__(self, router_name):
        super().__init__(router_name)
        self._engine = None
        self._counter = itertools.count(1)

    def _get_prediction(self, query):
        if self._engine is None:
            self._engine = self._build_engine()
        session_id = "shunt-arena-%06d" % next(self._counter)
        chosen, _reason, _provenance = self._engine.decide(
            session_id=session_id, prompt_text=query
        )
        if chosen not in _MODEL_TO_ARENA:
            raise ValueError("Shunt chose model %r which has no arena mapping" % chosen)
        return _MODEL_TO_ARENA[chosen]

    def _build_engine(self):
        # The REAL shipped Shunt engine, wired per the offline recipe used across
        # Shunt's own evaluation, over the LIVE menu models (menu shunt ids the
        # Shunt registry actually serves) and the strategy pinned in _STRATEGY.
        # `knn_semantic_cascade` consults a benchmark-seeded outcome index (see
        # _build_seeded_index below); `session_cascade` is the fixed
        # cheapest-first pick and never embeds or queries an index, so its null
        # session manager / null outcome index are never reached.
        _import_shunt()  # import-for-effect: fails fast with a pinned-install hint
        from shunt.models import ModelPool
        from shunt.router.engine import RouterEngine
        from shunt.router.policy import RouterPolicy
        from shunt.router.selection import SelectionRule
        from shunt.router.strategies import build_strategy

        def _bundled_seed_dir():
            # The PR-bundled QA seed (router_inference/router/shunt_seed/) is the
            # reproduction path: a clean checkout of THIS repo rebuilds the index
            # from it with no env vars. Present -> prefer it over the env root.
            from pathlib import Path

            bundled = Path(__file__).resolve().parent / "shunt_seed"
            return bundled if bundled.is_dir() else None

        def _benchmark_root():
            # Legacy/env fallback for the seeded corpus: the arena artifact dir
            # for the qacalib (QA) source, a Shunt checkout holding
            # benchmark/routing for the swebench source. SHUNT_ARENA_BENCH_ROOT
            # pins an explicit root; otherwise SHUNT_SRC (always set by the
            # deployer) points at <shunt>/src.
            import os

            from pathlib import Path

            root = os.environ.get("SHUNT_ARENA_BENCH_ROOT")
            if root:
                return Path(root)
            src = os.environ.get("SHUNT_SRC")
            if src:
                return Path(src).parent
            raise RuntimeError(
                "knn_semantic_cascade needs the seeded prior: set "
                "SHUNT_ARENA_BENCH_ROOT, or SHUNT_SRC to a checkout whose parent "
                "holds the prior corpus."
            )

        def _build_seeded_index(embedder):
            from router_inference.router.shunt_bench_prior import (
                build_prior_index,
                embedding_map_from_json,
                load_prior_cells,
                load_qa_prior_cells,
            )

            source = _PRIOR.get("source") or "swebench"
            if source == "qacalib":
                qa = _PRIOR["qacalib"]
                qadir = _bundled_seed_dir() or (_benchmark_root() / qa["dir"])
                pairs = [(arena, shunt) for shunt, arena in _MODEL_TO_ARENA.items()]
                report = load_qa_prior_cells(qadir, pairs)
                cells = report.cells
                if not cells:
                    raise RuntimeError(
                        "knn_semantic_cascade: the QA prior under %r holds no "
                        "menu-model cells (rows seen: %d, nonterminal skipped: "
                        "%d); refusing to seed an empty outcome index (re-run "
                        "qacalib for the menu's models, or deploy with --strategy "
                        "session_cascade)"
                        % (qadir, report.rows_seen, report.skipped_nonterminal)
                    )
                vectors = embedding_map_from_json(qadir / qa["embeddings"])
                return build_prior_index(
                    cells, embedder, vectors=vectors, session_label="qa"
                )
            root = _benchmark_root()
            if source == "swebench":
                swe = _PRIOR["swebench"]
                report = load_prior_cells(
                    root / swe["results_csv"],
                    root / swe["challenges_json"],
                    _MENU_SHUNT_IDS,
                    swe["default_arms"],
                )
                cells = report.cells
                if not cells:
                    raise RuntimeError(
                        "knn_semantic_cascade: the benchmark prior holds no "
                        "default-arm cells for menu %r; refusing to seed an empty "
                        "outcome index (re-run the benchmark for the menu's default "
                        "arms, or deploy with --strategy session_cascade)"
                        % (_MENU_SHUNT_IDS,)
                    )
                return build_prior_index(cells, embedder)
            raise RuntimeError(
                "knn_semantic_cascade: unknown prior source %r" % (source,)
            )

        class _NullSessionManager:
            def get_session(self, session_id):
                return None

        class _NullOutcomeIndex:
            def count_labeled(self):
                raise NotImplementedError

            def count_total_labeled(self):
                raise NotImplementedError

            def effective_labeled(self):
                raise NotImplementedError

            def effective_tier2(self):
                raise NotImplementedError

            def model_priors(self):
                raise NotImplementedError

            def query(self, embedding, k=20):
                raise NotImplementedError

        pool = ModelPool()
        live = [name for name in _MENU_SHUNT_IDS if pool.get_model(name) is not None]
        if not live:
            raise RuntimeError(
                "none of the menu shunt ids %r are present in the Shunt "
                "registry (%r)" % (_MENU_SHUNT_IDS, pool.model_names())
            )
        pool.restrict_to_live(live)
        policy = RouterPolicy(strategy=_STRATEGY, models=live)
        selection_rule = SelectionRule(
            min_success_rate=policy.policy.success_rate_threshold,
            min_samples=policy.policy.min_samples,
        )
        strategy = build_strategy(policy.strategy, selection_rule)
        # ONE shared Embedder instance serves both the seeded corpus and every
        # arena prompt when the engine consults neighbours, so both live in the
        # same (lazily-loaded) embedding space. Fixed strategies never embed, so
        # nothing is constructed for them (the engine's own default stays inert).
        embedder = None
        if _STRATEGY == "knn_semantic_cascade":
            from shunt.router.embedder import Embedder

            embedder = Embedder()
            outcome_index = _build_seeded_index(embedder)
        else:
            outcome_index = _NullOutcomeIndex()
        return RouterEngine(
            model_pool=pool,
            session_manager=_NullSessionManager(),
            outcome_index=outcome_index,
            embedder=embedder,
            selection_rule=selection_rule,
            strategy=strategy,
            strategy_id=policy.strategy,
            neighbor_k=policy.policy.k,
            exploration=None,
            escalation=policy.escalation.to_config(),
            task_key_resolver=None,
        )
