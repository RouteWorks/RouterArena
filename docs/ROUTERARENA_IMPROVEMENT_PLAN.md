# RouterArena Submission Improvement Plan

> Legitimate optimization roadmap for the `llm-router` submission.
>
> **Background**: An earlier optimization attempt (Lever #3, commit `401ad54`)
> used per-query accuracy data from the prior evaluation to decide which
> queries to reroute to a stronger model. The maintainer correctly flagged
> this as test-set leakage. The branch was reset to baseline (`065cca5`)
> and the score was invalidated.
>
> This plan documents what **is** allowed and the concrete paths to legitimately
> improve the submission beyond the 0.7139 honest baseline.

## Submission policy — allowed vs forbidden signals

### ✅ Allowed (uses prompt content or a-priori knowledge)

| Signal | Examples |
|---|---|
| Prompt text | keywords, length, language, presence of `\boxed{}`, code blocks |
| `Global Index` prefix | `MMLUPro_physics_*`, `AIME_*`, `LiveCodeBench_*` |
| `Dataset name` / `Category` from the dataset | "Mathematics", "Code", "Music" — these are dataset metadata, not labels |
| Generated answer characteristics | length, formatting, refusal patterns |
| Inference success/failure | `success=False`, empty `generated_answer`, API error code |
| Cross-model agreement | two models produce the same letter → high confidence (ensemble/self-consistency) |

### ❌ Forbidden (test-set leakage)

| Signal | Why it leaks |
|---|---|
| `prediction.accuracy` | Direct ground-truth label |
| `prediction.cost` filtered by correctness | Indirect — cost-of-correct varies by model accuracy |
| `optimality entries` to select per-query winners | Uses observed accuracy across models |
| Routing chain ordered by per-query historical accuracy | Same as Lever #3 — even via heuristics |

**Rule of thumb**: if a routing decision could not have been made without first
running the evaluation, the decision uses the test set.

## Roadmap

Phases ordered by ROI (expected score lift per $ + engineering hour).

### Tier 1 — Free wins (no cost, ~1-3 hours each)

#### 1A. Self-consistency for multiple-choice queries
- **Mechanism**: detect MC by prompt pattern (presence of `A.`, `B.`, ..., `Provide the correct letter choice`). For each MC query, run 3 sampling passes from the same cheap model with `temperature=0.7`. Majority-vote the `\boxed{X}` answer.
- **Why legitimate**: only uses prompt format detection + model outputs; never reads accuracy.
- **Expected lift**: +0.02 to +0.04 accuracy on the ~80% of queries that are MC.
- **Cost**: 3x inference on MC queries — still <$2 total with current model pool.
- **Risk**: low. If self-consistency disagrees, the majority is usually right.

#### 1B. Better system prompts per task family
- **Mechanism**: currently no system prompt is set. Add explicit CoT/format guidance per dataset prefix:
  - Math (`AIME`, `MATH`, `MathQA`, `MMLUPro_physics`) → "Think step by step. Show your reasoning. Put the final numerical answer in `\boxed{X}`."
  - MC knowledge (`MMLUPro_*`, `ArcMMLU_*`) → "Read the question, eliminate clearly wrong options, then state the answer in `\boxed{X}`."
  - Code (`LiveCodeBench_*`) → "Implement the function correctly. Return only working code in the requested format."
- **Why legitimate**: dataset prefix is metadata, not a label.
- **Expected lift**: +0.01 to +0.03 accuracy.
- **Cost**: $0 (just code).
- **Risk**: very low.

#### 1C. Investigate robustness drop
- The last evaluation showed robustness score 0.30 → 0.236 even though `predictions-robustness.json` wasn't modified. Investigate:
  - Are the robustness queries scored against the main `predictions.json` instead of the dedicated robustness file?
  - Does the deepseek-v3.2 model output format affect robustness scoring differently?
- **Expected lift**: potentially recover +0.06 if it was a routing-side artifact.

### Tier 2 — Modest cost ($5-15 OpenRouter)

#### 2A. Per-dataset specialist routing
- **Mechanism**: based on the `Dataset name` field from the dataset (a-priori knowledge of which models excel at which subjects, NOT measured from this evaluation), route:
  - `LiveCodeBench` → `Qwen3-Coder-Next` (already a specialist in the existing adapter)
  - `AIME`, `MATH` → strong reasoning model (DeepSeek-V3.2 or Sonnet 4)
  - `MusicTheoryBench` → strong general (Gemini Flash, Sonnet 4)
  - `QANTA_*` → strong knowledge (Sonnet 4 if affordable, else best workhorse)
- **Why legitimate**: a-priori knowledge of dataset difficulty + public benchmark performance, NOT measured from RouterArena labels.
- **Expected lift**: +0.02 to +0.04.
- **Cost**: ~$5-10 depending on model mix.
- **Risk**: medium — the a-priori reasoning has to be defensible (publish your rationale).

#### 2B. Confidence-based fallback
- **Mechanism**: run cheap model first. If output is:
  - shorter than 50 chars, OR
  - lacks `\boxed{}` when expected, OR
  - contains refusal patterns ("I don't know", "I cannot determine"), OR
  - has reasoning_tokens but empty content (thinking-model failure)
  → automatically retry the same query on a stronger model.
- **Why legitimate**: trigger uses OUTPUT features, never accuracy.
- **Expected lift**: +0.01 to +0.02.
- **Cost**: ~10-20% of queries escalate at ~$0.005 each = $4-8.

### Tier 3 — Real engineering (higher cost, higher lift)

#### 3A. Two-model ensemble vote on math/code/reasoning categories
- **Mechanism**: for `AIME`, `MATH`, `LiveCodeBench`, `MMLUPro_physics`, `MMLUPro_chemistry`, `MMLUPro_computer science`, run two different models in parallel. Compare extracted `\boxed{X}`. If they agree → use it. If disagree → run a third model as tiebreaker.
- **Why legitimate**: dataset-category selection only; agreement is across model outputs, not against ground truth.
- **Expected lift**: +0.02 to +0.05 (these categories are ~20% of queries).
- **Cost**: ~2.5x inference on those categories = $5-8 total.

#### 3B. Add Claude Sonnet 4 / GPT-4o to the model pool
- **Mechanism**: add to config; route based on category (Tier 2A) or as confidence fallback (Tier 2B).
- **Why legitimate**: just adding a model. The routing rules that use it must follow Tier 1/2 principles.
- **Expected lift**: +0.03 to +0.05 if combined with smart per-category routing.
- **Cost**: $10-20 for 1500-2000 routed queries.

## Realistic ceiling

| Combination | Expected legitimate score |
|---|---|
| Baseline (current) | 0.7139 |
| Tier 1A + 1B + 1C | 0.74-0.76 |
| + Tier 2A + 2B | 0.76-0.78 |
| + Tier 3A + 3B | 0.78-0.81 |

Sqwish (#1) is at 0.7527. Tier 1 alone can beat it honestly.

## Sequencing recommendation

1. **Start with Tier 1B + 1C** (free, fast, low risk). Re-eval. If we land in 0.72-0.74 territory, great.
2. **Then Tier 1A (self-consistency)**. Re-eval. If we cross Sqwish honestly, take a victory lap and document the strategy.
3. **Then Tier 2A** if more headroom is wanted.
4. **Tier 3** only if pushing for #1 with margin.

After each step:
- Run `scripts/check_submission_integrity.py` before pushing
- Open an audit log in `docs/submission_audit.md` noting what changed and why it's legitimate
- Only force-push to the PR branch when the integrity check passes
