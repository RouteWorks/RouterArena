"""Verified-outcome priors for the arena router's seeded kNN index.

The seeded-kNN prior is a corpus of measured (prompt, model) outcome cells — the
deployer MENU's models over a task family, each cell carrying a verified pass/
fail and a real cost — served to the shipped RouterEngine through the
``OutcomeIndex`` protocol it already speaks (``count_labeled`` /
``count_total_labeled`` / ``effective_labeled`` / ``effective_tier2`` /
``model_priors`` / ``query``). The router's spec ``prior.source`` selects which
loader builds the cells:

* ``swebench`` — Shunt's SWE-bench benchmark outcome rows
  (``benchmark/routing/results.csv``) joined to their task text via the
  benchmark's ``challenges.json`` (``load_prior_cells``);
* ``qacalib`` — the QA-domain calibration corpus (MMLU + GSM8K, labeled,
  deterministically graded): per-(prompt, model) pass/cost rows recorded by
  ``scripts/arenas/qacalib.py`` under ``qacalib/sample.json`` + ``results.json``
  (``load_qa_prior_cells``). This is RouterArena's domain, and the arena default.

Cell embeddings come from the shipped ``shunt.router.embedder.Embedder`` at
index-build time; for ``qacalib`` the precomputed ``embeddings.json`` vectors
(keyed by prompt digest) are reused when a prompt's digest is present, falling
back to the Embedder otherwise. The query side is a brute-force cosine scan over
the loaded cells (hundreds, not millions), so the arena router picks up no extra
index dependency.

Two trees, one source: this file is imported by the deployer (``build_engine``,
tests) AND copied verbatim into the arena clone as
``router_inference/router/shunt_bench_prior.py`` so the generated router module
runs standalone inside the arena generator. Keep it free of wrapper imports —
stdlib + numpy + the shunt package only, numpy imported lazily so the pure
loader stays importable where numpy is absent.

Only CANONICAL default-arm rows seed the index. A menu model's measured default
arm (``models.yaml`` ``reasoning.default_arm``; for ``deepseek-v4-flash`` /
``deepseek-v4-pro`` that is ``high``) is the behaviour the arena actually serves
when it routes to that model, so an off-arm measurement (``nothink``, ``max``,
``medium``, ...) is a DIFFERENT behaviour priced under the same id and must not
vote as if it were the default.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# Confidence granted to a MEASURED default-arm benchmark cell. results.csv rows
# are verified Tier-2 outcomes, never monotone-ladder fills, so every cell votes
# at full confidence (mirrors the benchmark's MatrixOutcomeIndex).
_VERIFICATION_CONFIDENCE: float = 1.0

# qacalib corpus identity for the session_id prefix (vs "bench" for SWE cells).
_QA_CORPUS_LABEL = "qa"


@dataclass(frozen=True)
class PriorCell:
    """One verified prior cell: (task, model) measured outcome.

    ``model`` is always the menu model's SHUNT id — the id the engine's pool is
    built over — whatever label the underlying corpus used to record the row.
    """

    task_id: str
    model: str
    routing_text: str
    outcome: bool
    cost: float


@dataclass(frozen=True)
class PriorReport:
    """Loader result + honest counts: what seeded vs what was skipped and why."""

    cells: list[PriorCell]
    rows_seen: int
    skipped_no_text: int
    skipped_unknown_reasoning: int
    # qacalib: menu rows whose `pass` is not a terminal bool (infra-error rows a
    # resume re-measures) — never a verified outcome, never seeded.
    skipped_nonterminal: int = 0
    # Which loader produced the cells (drives error messages/session prefixes).
    source: str = "swebench"


class BenchmarkOutcomeIndex:
    """Implements the RouterEngine ``OutcomeIndex`` protocol over benchmark cells.

    Every seeded cell is a verified Tier-2 outcome at confidence 1.0, so the
    effective sample size equals the cell count and the engine's cold-start gate
    (n_e_tier2 >= 20 OR n_e_labeled >= 50) is INACTIVE on a seeded corpus — the
    engine consults neighbours instead of returning the cold-start default.
    """

    def __init__(self, cells: list[PriorCell], embeddings: Any, *, session_label: str = "bench") -> None:
        import numpy as np

        self._session_label = session_label
        self._cells = list(cells)
        self._task_ids = [c.task_id for c in self._cells]
        self._models = [c.model for c in self._cells]
        self._outcomes = [bool(c.outcome) for c in self._cells]
        self._costs = [float(c.cost) for c in self._cells]
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(self._cells):
            raise ValueError(
                f"benchmark prior embeddings shape {matrix.shape} does not match "
                f"{len(self._cells)} cells (expected 2-D with one row per cell)"
            )
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0
        self._embeddings = matrix / norms[:, None]

    def count_labeled(self) -> int:
        """Number of cells with a labeled outcome (all are Tier-1)."""
        return len(self._cells)

    def count_total_labeled(self) -> int:
        """Number of cells with any labeled outcome (same corpus)."""
        return len(self._cells)

    def effective_labeled(self) -> float:
        """Effective sample size over all labeled outcomes (confidence-1 cells)."""
        return float(len(self._cells))

    def effective_tier2(self) -> float:
        """Effective sample size over verified (Tier-2) outcomes (all cells)."""
        return float(len(self._cells))

    def model_priors(self) -> dict[str, tuple[float, float]]:
        """Per-model ``(pass_rate, cell_count)`` offline aggregates for prior seeding."""
        counts: dict[str, int] = defaultdict(int)
        passes: dict[str, int] = defaultdict(int)
        for model, outcome in zip(self._models, self._outcomes, strict=True):
            counts[model] += 1
            passes[model] += 1 if outcome else 0
        return {
            model: (passes[model] / counts[model], float(counts[model]))
            for model in counts
        }

    def query(self, embedding: Any, k: int = 20) -> list[Any]:
        """Return the *k* nearest cells as engine ``NeighborResult`` objects.

        Cosine distance = ``1 - cos_sim`` over the L2-normalised vectors, matching
        the benchmark's HNSW ``space="cosine"`` semantics. Every measured cell
        carries verification_confidence 1.0.
        """
        import numpy as np

        from shunt.router.selection import NeighborResult

        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if query.shape[0] != self._embeddings.shape[1]:
            raise RuntimeError(
                f"query embedding dim {query.shape[0]} != benchmark prior dim "
                f"{self._embeddings.shape[1]} — the prompt embedder differs from "
                "the embedder that built the seeded corpus"
            )
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        distances = 1.0 - (self._embeddings @ query)
        limit = len(self._cells)
        k = min(int(k), limit) if int(k) > 0 else limit
        if k < limit:
            order = np.argpartition(distances, k - 1)[:k]
        else:
            order = np.arange(limit)
        order = order[np.argsort(distances[order])]
        results = []
        for index in order:
            cell = self._cells[int(index)]
            results.append(
                NeighborResult(
                    model=cell.model,
                    outcome=self._outcomes[int(index)],
                    cost=self._costs[int(index)],
                    verification_confidence=_VERIFICATION_CONFIDENCE,
                    distance=float(distances[int(index)]),
                    session_id=f"{self._session_label}:{cell.task_id}:{cell.model}",
                    truncation_rate=0.0,
                )
            )
        return results


def load_prior_cells(
    results_csv: str | Path,
    challenges_json: str | Path,
    menu_shunt_ids: list[str],
    default_arms: dict[str, str],
) -> PriorReport:
    """Load canonical default-arm benchmark cells for the menu shunt ids.

    ``results_csv`` is the source of truth: one row per measured (challenge,
    model, reasoning-arm) cell. Only rows whose ``model`` is in ``menu_shunt_ids``
    AND whose ``reasoning`` column equals that model's ``default_arms`` value are
    kept (a canonical default-arm measurement — see the module docstring). The
    routing text for each kept row is that challenge's task text from
    ``challenges_json`` (``problem_statement``, else ``description``, else the id
    — mirroring ``benchmark.routing.strategies.routing_text``). Cost is the
    provider-billed ``real_cost`` when present, else the ``cost`` estimate.
    """
    wanted = set(menu_shunt_ids)
    text_of = _task_text_reader(challenges_json)
    cells: list[PriorCell] = []
    skipped_no_text = 0
    skipped_unknown_reasoning = 0
    rows_seen = 0
    with Path(results_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model = str(row.get("model") or "")
            if model not in wanted:
                continue
            reasoning = str(row.get("reasoning") or "").strip()
            canonical = str(default_arms.get(model) or "")
            if not canonical:
                raise ValueError(
                    f"menu model {model!r} has no default-arm entry in the arena "
                    "spec `prior.default_arms` — the loader cannot tell its "
                    "canonical measurements from off-arm ones"
                )
            if reasoning != canonical:
                skipped_unknown_reasoning += 1
                continue
            rows_seen += 1
            challenge_id = str(row.get("challenge_id") or "")
            text = text_of(challenge_id)
            if not text:
                skipped_no_text += 1
                continue
            cells.append(
                PriorCell(
                    task_id=challenge_id,
                    model=model,
                    routing_text=text,
                    outcome=_row_bool(row.get("pass")),
                    cost=_cell_cost(row),
                )
            )
    return PriorReport(
        cells=cells,
        rows_seen=rows_seen,
        skipped_no_text=skipped_no_text,
        skipped_unknown_reasoning=skipped_unknown_reasoning,
    )


def load_qa_prior_cells(
    qacalib_dir: str | Path,
    menu_pairs: Sequence[tuple[str, str]],
) -> PriorReport:
    """Load measured QA-domain cells for the MENU from a qacalib results.json.

    ``qacalib_dir`` holds the QA-calibration files written by
    ``scripts/arenas/qacalib.py`` (``sample.json`` = the labeled MMLU/GSM8K
    prompts, ``results.json`` = one per-(prompt, model) graded row
    ``{prompt_id, prompt, model, pass, cost, ...}``). *menu_pairs* is the MENU as
    ``(arena_name, shunt_id)`` pairs (spec ``models``); a results row whose
    ``model`` is not one of those arena names is not a rung this arena can route
    to and is skipped, matching the SWE loader's non-menu handling.

    Cells are built for rows whose ``pass`` is a TERMINAL bool (a deterministic
    grade, never an LLM-judge score). ``routing_text`` is the prompt text the
    prompt was measured on (the sample prompt when the row's ``prompt_id``
    resolves, else the row's own ``prompt``); the cell's ``model`` is the row's
    MENU SHUNT id (what the engine's pool routes on); ``outcome`` is the graded
    pass and ``cost`` the billed USD for that exact call. Every QA row is a
    verified outcome at full confidence (the corpus has no monotone-ladder
    fills), exactly like the SWE default-arm cells.
    """
    base = Path(qacalib_dir)
    sample_path = base / "sample.json"
    results_path = base / "results.json"
    for required, name in ((sample_path, "sample.json"), (results_path, "results.json")):
        if not required.exists():
            raise FileNotFoundError(
                f"qacalib prior missing {name} under {base} — run the qacalib "
                "`fetch`+`run` steps (scripts/arenas/qacalib.py) to measure the "
                "QA-domain corpus"
            )
    prompt_by_id: dict[str, str] = {}
    with sample_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    for prompt in payload.get("prompts", []) if isinstance(payload, dict) else []:
        if isinstance(prompt, dict) and prompt.get("prompt"):
            prompt_by_id[str(prompt.get("id"))] = str(prompt["prompt"])

    shunt_of = {arena: shunt for arena, shunt in menu_pairs}
    wanted_arena = set(shunt_of)
    with results_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    cells: list[PriorCell] = []
    rows_seen = 0
    skipped_no_text = 0
    skipped_nonterminal = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        arena_model = str(row.get("model") or "")
        if arena_model not in wanted_arena:
            continue
        passed = row.get("pass")
        if not isinstance(passed, bool):
            # Infra/error rows (pass null) are retried by a qacalib resume, never
            # a verified outcome — a QA cell must not vote on a failed call.
            skipped_nonterminal += 1
            continue
        rows_seen += 1
        prompt_text = prompt_by_id.get(str(row.get("prompt_id") or ""))
        if not prompt_text:
            prompt_text = str(row.get("prompt") or "")
        if not prompt_text:
            skipped_no_text += 1
            continue
        cells.append(
            PriorCell(
                task_id=str(row.get("prompt_id") or ""),
                model=shunt_of[arena_model],
                routing_text=prompt_text,
                outcome=bool(passed),
                cost=_qa_row_cost(row),
            )
        )
    return PriorReport(
        cells=cells,
        rows_seen=rows_seen,
        skipped_no_text=skipped_no_text,
        skipped_unknown_reasoning=0,
        skipped_nonterminal=skipped_nonterminal,
        source="qacalib",
    )


def _qa_row_cost(row: dict[str, Any]) -> float:
    """The billed USD of a qacalib row (measured cost), else 0.0."""
    cost = _as_float(row, "cost")
    return cost if cost is not None and cost > 0.0 else 0.0


def embedding_map_from_json(embeddings_json: str | Path) -> dict[str, Any]:
    """``prompt digest -> vector`` from a qacalib embeddings.json (else empty).

    The qacalib embeddings file maps an exact prompt's sha256-[:20] digest to
    the shipped-Embedder vector recorded for it. A missing/absent file yields an
    empty map (callers then embed every cell at index build), so the kNN QA
    prior never hard-requires precomputed vectors.
    """
    path = Path(embeddings_json)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get("entries", {}) if isinstance(payload, dict) else {}
    out: dict[str, Any] = {}
    for digest, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        vector = entry.get("vec")
        if vector is not None:
            out[str(digest)] = vector
    return out


def prompt_text_digest(prompt: str) -> str:
    """The 20-hex identity a qacalib embeddings.json keys prompts by."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:20]


def _task_text_reader(challenges_json: str | Path):
    """Return ``challenge_id -> routing text`` from the benchmark challenges file."""
    with Path(challenges_json).open(encoding="utf-8") as handle:
        data = json.load(handle)
    tasks = data.get("tasks", {}) if isinstance(data, dict) else {}

    def text_of(challenge_id: str) -> str:
        meta = tasks.get(challenge_id, {}) if isinstance(tasks, dict) else {}
        if not isinstance(meta, dict):
            meta = {}
        statement = str(meta.get("problem_statement") or "").strip()
        if statement:
            return statement
        description = str(meta.get("description") or "").strip()
        return description or challenge_id

    return text_of


def _cell_cost(row: dict[str, str]) -> float:
    """The provider-billed real_cost, else the cost estimate, else 0.0."""
    real = _as_float(row, "real_cost")
    if real is not None and real > 0.0:
        return real
    estimated = _as_float(row, "cost")
    return estimated if estimated is not None else 0.0


def _as_float(row: dict[str, str], key: str) -> float | None:
    raw = str(row.get(key) or "").strip()
    try:
        return float(raw)
    except ValueError:
        return None


def _row_bool(raw: Any) -> bool:
    """CSV booleans come back as ``'True'``/``'False'`` strings."""
    return str(raw or "").strip().lower() in ("1", "true", "yes")


def build_prior_index(
    cells: list[PriorCell],
    embedder: Any = None,
    *,
    vectors: dict[str, Any] | None = None,
    session_label: str = "bench",
) -> BenchmarkOutcomeIndex:
    """Wrap *cells* in an outcome index, embedding each task's routing text once.

    The shipped ``Embedder`` is used unless ``embedder`` is injected (tests /
    stub). ``vectors`` optionally pre-resolves cell vectors: a ``prompt digest ->
    vector`` map (see ``embedding_map_from_json``) for a corpus whose prompts
    were embedded in advance. A cell whose routing-text digest is present reuses
    that vector; a cell whose digest is missing is embedded through the embedder
    at index build (the fallback that keeps a ``qacalib`` prior working when the
    embeddings file is partial or absent). ``session_label`` names the corpus in
    query session ids (``bench`` for SWE cells, ``qa`` for qacalib cells).

    When no ``vectors`` are supplied every cell is embedded, exactly as before.
    The model is lazy-loaded; when it is unavailable (no cached ONNX model, no
    network) the failure is re-raised with the arena context so a
    ``knn_semantic_cascade`` deploy cannot silently degrade to a fake index.
    """
    if not cells:
        raise ValueError(
            "cannot seed the kNN outcome index from an empty benchmark prior"
        )
    vector_by_digest = {str(k): v for k, v in (vectors or {}).items()}
    by_text: dict[str, Any] = {}
    missing: list[str] = []
    for cell in cells:
        if cell.routing_text in by_text or cell.routing_text in missing:
            continue
        vector = vector_by_digest.get(prompt_text_digest(cell.routing_text))
        if vector is not None:
            by_text[cell.routing_text] = vector
        else:
            missing.append(cell.routing_text)
    if missing:
        embedded = _embed_many(missing, embedder)
        by_text.update(zip(missing, embedded, strict=True))
    embeddings = [by_text[cell.routing_text] for cell in cells]
    return BenchmarkOutcomeIndex(
        cells, embeddings, session_label=session_label
    )


def _embed_many(texts: list[str], embedder: Any) -> list[Any]:
    """Embed *texts* one at a time through ``embed`` (batch only as a fallback).

    Single-embed on purpose, mirroring ``benchmark/routing/build_seed_bundle.py``
    (the shipped corpus builder): a full-corpus ``embed_batch`` pads every row to
    the longest sequence in one ONNX call and blew a ~20 GB attention allocation
    on a 365-cell coding corpus. One call per unique text keeps the allocation
    per-text and still lazy-loads the model exactly once per embedder instance.
    """
    import numpy as np

    if embedder is None:
        embedder = _shipped_embedder()
    embed = getattr(embedder, "embed", None)
    batch = getattr(embedder, "embed_batch", None)
    if not callable(embed) and not callable(batch):
        raise RuntimeError(
            "knn_semantic_cascade: the embedding backend exposes neither "
            "embed(text) nor embed_batch(texts); cannot embed the benchmark prior"
        )
    vectors: list[Any] = []
    for text in texts:
        try:
            if callable(embed):
                raw = embed(text)
            else:
                raw = list(batch([text]))[0]
        except Exception as exc:  # noqa: BLE001 - surfaced with arena context
            raise RuntimeError(
                "knn_semantic_cascade index build failed while embedding the "
                "benchmark prior with the shipped Embedder. The embedding model "
                "(jina v2 base code, ~600MB) must be downloadable/cached "
                "(SHUNT_EMBED_CACHE_DIR) or a stub embedder must be injected."
            ) from exc
        vectors.append(np.asarray(raw, dtype=np.float32).reshape(-1))
    return vectors


def _shipped_embedder() -> Any:
    """The shipped router Embedder (ONNX model lazy-loads on first embed)."""
    from shunt.router.embedder import Embedder

    return Embedder()
