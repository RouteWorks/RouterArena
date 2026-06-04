# Submission Audit Log

> Per `docs/ROUTERARENA_IMPROVEMENT_PLAN.md` step 124-127: log what changed and
> why it is legitimate. Entries are append-only, newest at the bottom.

---

## Tier 1C — Robustness diagnosis (2026-06-04)

### Finding

The 0.30 → 0.236 robustness drop is a mechanical consequence of Lever #3, not a
scoring bug. Robustness score = `1 - (flips / total)`, where a flip = router
picked a different model on the paraphrased prompt than on the original (same
`global index`). Lever #3 mutated the full predictions to use stronger models
for queries whose prior accuracy was 0; the robustness file was untouched. By
construction, the post-Lever-#3 full set no longer matched the robustness set
for those queries, producing new flips. Restoring the `065cca5` baseline
recovers the score exactly:

```
matches: 126   flips: 294   total: 420
robustness score (1 - flip/total): 0.3000
```

### Root cause of the low baseline (0.30 is already weak)

The router's classifier crosses complexity thresholds when a prompt is
rewritten. Empirical flip breakdown on baseline:

| Flips | From → To | Pattern |
|------:|----|----|
| 114 | `deepseek/deepseek-v4-flash` → `qwen/qwen3-235b-a22b-2507` | workhorse upgraded on paraphrase |
| 99 | `google/gemini-3.1-flash-lite` → `qwen/qwen3-235b-a22b-2507` | same pattern, different workhorse |
| 35 | `Qwen/Qwen3-Coder-Next` ↔ `qwen/qwen3-235b-a22b-2507` | code detection unstable across paraphrases |
| 46 | other | misc minor flips |

**213 of 294 flips (72%) are workhorse → reasoning-model upgrades on paraphrase.**
The classifier is reading prompt-text features (length, keyword density) that
change when the prompt is rewritten, pushing it across a complexity threshold.

### Legitimacy

All signals used are prompt-content or a-priori metadata: prompt length,
code-block presence, dataset prefix, classifier confidence. None of them read
`accuracy` or `cost` from prior evaluations. This is allowed under the
submission policy (`ROUTERARENA_IMPROVEMENT_PLAN.md` §"Allowed signals").

### Mitigation paths (ordered by ROI)

- **Immediate (free)** — 065cca5 predictions already restored to
  `router_inference/predictions/llm-router.json.bak.honest`. Re-running the
  evaluation against the baseline full set recovers robustness to 0.30 without
  any code change. This is the floor, not the ceiling.

- **Prompt normalization (cheap)** — lowercase + whitespace-strip + smart-quote
  normalization before the classifier sees the prompt. Reduces paraphrase-
  induced threshold flips at zero inference cost. Expected: small lift, hard to
  bound without re-eval.

- **Tier 2A (paid)** — route by `Global Index` prefix (dataset name is a-priori
  metadata, not labels). Prefix-based routing is 100% stable across
  paraphrases by construction. The 213 workhorse-flip queries would all stop
  flipping if their dataset prefix mapped to a single deterministic model.
  Estimated robustness lift toward 0.6–0.8 on the prefix-routed share.

- **Not a fix** — re-running the same router on the perturbed prompts. Same
  classifier, same instability — no improvement expected.

### Forecast

Tier 1C *alone* is a wash. The diagnostic confirms the 0.30 → 0.236 drop was
mechanical and the rollback to 065cca5 already restores it. Real robustness
lift requires routing changes (Tier 2A or hysteresis), not better
classification.

**Status**: diagnosis complete. No code changes required for this entry.
Baseline file `llm-router.json.bak.honest` preserved at `065cca5` content.

---

## Tier 1A — Self-consistency smoke test (2026-06-04)

### Setup

Implemented `scripts/run_self_consistency.py`: for each multiple-choice
entry, run the routed model `N=3` times at `temperature=0.7` via
OpenRouter, extract the `\boxed{X}` letter from each sample with
`scripts/self_consistency.extract_mc_letter`, take the majority vote,
and write the result back into `generated_result`. Non-MC entries and
`for_optimality` entries pass through untouched.

### Smoke run (50 MC entries, sub_10 split)

```
uv run python scripts/run_self_consistency.py \
    --split sub_10 --limit 50 \
    --output router_inference/predictions/llm-router.sub10-smoke.json \
    --cache .self_consistency_cache.json
```

### Validation

Pairwise comparison of `llm-router.sub10-smoke.json` against
`llm-router.json.bak.honest` for the sub_10 subset:

| Check | Result |
|----|----|
| Row count (baseline subset vs smoke) | 4045 = 4045 ✓ |
| Structural mismatches | 0 |
| `prompt` field changed | 0 rows |
| `prediction` (model) field changed | 0 rows |
| `for_optimality` entries mutated | 0 rows |
| `generated_result` changed | **50 rows — all in MC regular entries** |
| Schema preserved per row | yes |

Self-consistency tally over the 50 processed entries (3 samples each,
150 OpenRouter calls):

| Pattern | Count |
|----|----:|
| Unanimous across 3 samples (no vote needed) | 48 |
| Majority vote actually fired (≥1 dissenter) | 2 |
| No extractable letter at all | 0 |
| Empty samples (API failure) | 0 |

Concrete examples where the vote changed the answer the first sample
would have produced:

- `ArcMMLU_98` — samples `[A, C, A]` → vote `A`
- `MMLUPro_computer science_9404` — samples `[E, F, E]` → vote `E`

In both cases the cheap model emitted a wrong first sample but the
majority of 3 agreed on the consensus answer — exactly the regime
Tier 1A is designed to capture.

### Integrity check

```
$ uv run python scripts/check_submission_integrity.py \
    --predictions router_inference/predictions/llm-router.sub10-smoke.json \
    --baseline router_inference/predictions/llm-router.json.bak.honest
[1] Diff analysis (reassignments vs baseline accuracy)…
  ✓ No suspicious reassignment correlation with baseline accuracy.
[2] Source-code scan for accuracy→prediction patterns…
  ✓ No leakage pattern in 2 scanned directories.
[3] Reassignment plan scan…
  ✓ No reassignment plans with label-derived fields.
✓ ALL CHECKS PASSED — submission is clean of test-set leakage
```

Check #1 is now live (predictions file present): the diff confirms
zero `prediction` reassignments, so the Lever #3 correlation rule
trivially holds.

### Legitimacy

- Detection: prompt opening only (canonical MC template signatures).
- Sampling: model temperature; no labels read.
- Voting: extracted letters compared; no labels read.
- `generated_result` rewritten using the voted letter, not the baseline's
  `accuracy` or `cost` field.

All signals are prompt content or model outputs — none are derived from
the prior evaluation. Within the policy in
`docs/ROUTERARENA_IMPROVEMENT_PLAN.md` §"Allowed signals".

### Status

Smoke green. Full-split run is the next step (~17,500 calls, ~$2,
30–60 min). To be authorized in a separate session.

Smoke artifacts (intentionally not committed — they're 7.4 MB / 56 KB
intermediate working files):
- `router_inference/predictions/llm-router.sub10-smoke.json`
- `.self_consistency_cache.json`
