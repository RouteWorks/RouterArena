# cruq LLM Router: RouterArena findings (parked 2026-08-22)

A cruq-built LLM router for the [RouterArena](https://github.com/RouteWorks/RouterArena)
leaderboard, and a reusable routing engine for cruq's Model Gateway. This is the
state at pause: what works, what the data proved, and the one lever left to pull.

## Goal

For each query, pick the cheapest model from a pool that will still answer correctly.
RouterArena ranks routers by **Acc-Cost Arena** = a weighted harmonic mean of accuracy
`A` and log-normalized cost `C` (`S = (1+β)·A·C / (β·A + C)`, β=0.1, cost normalized
log2 between $0.0044 and $200). The router sees only the raw prompt string
(`_get_prediction(query: str) -> str`) and may not train on RouterArena data.

## What was built (on branch `cruq-router`)

- **`CruqRouter`** (`router_inference/router/cruq_router.py`) — Phase 1, training-free:
  a lexical difficulty score (length, math/code markers, reasoning cues, MCQ discount)
  mapped onto a pool sorted cheapest→strongest by price. Thresholds live in config.
- **Phase 2 learned pipeline** (`phase2/`): `build_corpus.py` (external corpus from
  public benchmark splits, deduped vs RouterArena), `label_corpus.py` (parallel labeling
  with backoff retry), `train_predictor.py` (MiniLM embedding → per-model logistic
  P(correct) heads), and **`CruqLearnedRouter`** (embed query → predict P(correct) per
  model → pick cheapest over τ).
- **Low-memory grader** (`router_evaluation/lightweight_grade.py`): streams cached
  results, `\boxed{}` extraction with numeric-index↔letter reconciliation. Used because
  the official pandas evaluator OOMs the dev machine.

## Results (all on the `sub_10` split, lightweight boxed-match grade)

### The 3-model OpenRouter pool

| Model | Accuracy | Cost/1K |
|---|---|---|
| qwen/qwen3-235b-a22b-2507 | 0.691 | $0.034 (cheapest) |
| Qwen/Qwen3-Coder-Next | 0.650 | $0.127 |
| deepseek/deepseek-v4-flash | 0.716 | $0.266 (best) |
| **Oracle (cheapest-correct)** | **0.806** | $0.061 |

### Predictor: hard training data is decisive

The learned predictor is only as good as the difficulty of its training corpus. On an
easy corpus (MMLU/ARC/GSM8K, models score ~90%) the per-model P(correct) heads were at
the noise floor. On a hard corpus (MMLU-Pro + MATH-500, base accuracy ~0.76-0.80) they
became genuinely predictive:

| Model | CV AUC (easy corpus) | CV AUC (hard corpus) |
|---|---|---|
| qwen3-235b | 0.349 | **0.708** |
| Qwen3-Coder-Next | 0.448 | **0.681** |
| deepseek-v4-flash | 0.512 | **0.774** |

### Router: it escalates, but the pool caps the gain

With the hard-trained predictor, sweeping τ shows the router moving traffic off the
cheapest model and lifting accuracy, but it plateaus far below the oracle:

| τ | Accuracy | Cost/1K | Opt-Sel | Routing (cheap/mid/exp) |
|---|---|---|---|---|
| 0.50 | 0.691 | $0.034 | 0.688 | 728 / 2 / 1 |
| 0.70 | 0.696 | $0.108 | 0.594 | 595 / 54 / 82 |
| 0.80 | 0.707 | $0.216 | 0.274 | 241 / 80 / 410 |
| 0.90 | 0.709 | $0.232 | 0.166 | 139 / 74 / 518 |
| oracle | **0.806** | — | — | — |

(τ here is analysis on `sub_10`; a submission τ must be set on external held-out data.)

## Diagnosis

1. **The router engine works.** Both the lexical and learned selectors run, and the
   learned one beats the lexical one on the cost-weighted metric.
2. **Predictor quality is a training-data-difficulty problem, and it is solved.** Hard,
   divergent training data (MMLU-Pro + MATH) lifts AUC from ~0.35 to ~0.71. More easy
   data actively hurt (AUC fell); harder data fixed it.
3. **The remaining gap to the oracle is pool spread, not the selector.** The three pool
   models are near-equal (all ~0.65-0.72 even on hard MATH), so escalating between them
   recovers little. glm-4.7 (0.70 on MATH) was not a real step up either. The oracle's
   0.806 is complementarity these near-twins cannot be routed into.

## The one lever left: a genuine strong tier

Add a genuinely more capable, pricier model to escalate *to*, then re-run the τ frontier.
`openai/gpt-5-mini` is **wired but unvalidated**: added to `_get_provider` (as an
openrouter slug), `model_cost.json` ($0.25/$2), and `universal_model_names.py`. The
decisive experiment (run it across `sub_10`, recompute the 4-model oracle) did not
complete, the run was killed at zero cached results. If the 4-model oracle jumps well
above 0.806, label gpt-5-mini on the hard corpus, retrain the 4-head predictor, and the
router should finally capture real accuracy gains. If it also lands ~0.72, the pool's
ceiling is intrinsic and no reachable model helps.

## Reproduce

```
uv sync
# dataset (writes dataset/router_data*.json with canonical \boxed prompts):
uv run python scripts/process_datasets/prep_datasets.py
# Phase 1 lexical router on sub_10 (uses the 3 cached models = free):
uv run python router_inference/generate_prediction_file.py cruq-router-or3 sub_10
uv run python llm_inference/run.py cruq-router-or3
uv run python router_evaluation/lightweight_grade.py
# Phase 2 (needs OPENROUTER_API_KEY in .env):
uv run python phase2/build_corpus.py --sources mmlu_pro math --per-source 600
LABEL_WORKERS=4 uv run python phase2/label_corpus.py --pool phase2/pool.json --limit 350
HF_HUB_OFFLINE=1 uv run python phase2/train_predictor.py
```

## Environment notes

- The OpenRouter key hits a low weekly/burst limit under sustained load; `label_corpus`
  has backoff retry, but big runs still stall. Raise the weekly cap before labeling.
- HF caches live on the external SSD (`/Volumes/Extreme SSD/cruq-cache`, set in `.env`).
- `transformers` 5.x flakes on an online adapter-config lookup, so training/embedding
  needs `HF_HUB_OFFLINE=1` (MiniLM is already cached locally).
- The official `llm_evaluation` pandas evaluator OOMs the dev machine; use the
  lightweight grader for local iteration, run the official one only with headroom.
- Data artifacts (`cached_results/`, `phase2/data/`, predictions) are regenerable and
  gitignored; only source is committed.
