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
   recovers little. The oracle's 0.806 is complementarity these near-twins cannot be
   routed into.

## Update (2026-08-22): the reachable lever is cheap diversity, not a strong tier

The parked conclusion assumed the only way to lift the oracle was to add a genuinely
more capable, pricier model to escalate *to* (`openai/gpt-5-mini`). A zero-cost pass
over the caches already on disk **refutes that**. Every model the base harness had
already run on `sub_10` was graded and its marginal oracle lift measured
(`phase2/oracle_lift.py`, no API spend). The lift comes from *cheap-model diversity*:

| Pool | Oracle acc | Cost/1k | Δ vs base |
|---|---|---|---|
| base 3-model | 0.806 | $0.061 | — |
| base3 + gemini-2.0-flash | **0.832** | $0.062 | +2.6 pts |
| base3 + gemini + gpt-4o-mini | **0.840** | $0.062 | +3.4 pts |
| 7-model ensemble (all real caches) | 0.852 | $0.135 | +4.7 pts |

gemini-2.0-flash-001 alone rescues **19 of the 142** queries all three base models fail,
at essentially unchanged cost (the oracle picks cheapest-correct, and gemini-flash is
cheap). Adding gpt-4o-mini on top recovers more, still at ~$0.062/1k. The expensive
candidates are the *worse* bets: glm-4.7 (+1.23 pts at $0.204/1k, and only 389/809 of its
answers even emit `\boxed{}` — the rest are reasoning-runaway truncations at up to 65k
output tokens), and gpt-5-mini was never validated — its 216 cached rows are all
`401 Incorrect API key` (the bare slug `gpt-5-mini` resolves to the OpenAI provider and
was called with the OpenRouter key; the openrouter-routed slug is `openai/gpt-5-mini`).

## Update (2026-08-23): experiment run to completion — the bottleneck is now the selector

The spend-gated step above was executed via OpenRouter slugs. `gemini-2.0-flash-001` is
aged out of OpenRouter (404), so the live same-price stand-in `google/gemini-2.5-flash-lite`
was used, plus `openai/gpt-4o-mini`. Both were labeled on the full 1066-item corpus and a
consistent 5-head predictor retrained (`phase2/train_predictor.py`); `gemini-2.5-flash-lite`
was then run across `sub_10` and the learned router evaluated end-to-end
(`phase2/router_eval.py`).

**5-head predictor CV AUC** (all genuinely predictive, well above the 0.5 floor):

| Model | corpus base acc | CV AUC |
|---|---|---|
| deepseek/deepseek-v4-flash | 0.806 | 0.761 |
| qwen/qwen3-235b-a22b-2507 | 0.753 | 0.750 |
| Qwen/Qwen3-Coder-Next | 0.737 | 0.723 |
| google/gemini-2.5-flash-lite | 0.755 | 0.711 |
| openai/gpt-4o-mini | 0.580 | 0.701 |

**End-to-end router vs oracle on `sub_10` (731 scorable, 5-model pool):**

| | Accuracy | Cost/1k |
|---|---|---|
| 5-model oracle (cheapest-correct) | **0.832** | $0.064 |
| Learned router, best τ (0.90) | **0.706** | $0.270 |
| (base-3 learned router, prior finding) | 0.709 | — |

**The result is a clean negative, and it relocates the bottleneck.** Adding the two
diverse-cheap models lifted the *oracle* (0.806 → 0.832) but did **nothing** for the
*router* (0.706, statistically identical to the base-3 router's 0.709). The τ sweep shows
why: even at τ=0.90 the router sends only 25/731 queries to gemini and 2 to gpt-4o-mini —
it escalates *within the base-3* (582 queries onto deepseek) rather than into the niche
models. The queries these cheap models uniquely rescue are exactly the hard ones where
every head is uncertain (AUC ~0.72), so a per-model P(correct) selector cannot confidently
route to them. The gap to the oracle is now **12.6 points and it lives in the selector**,
not the pool. (Note `gemini-2.5-flash-lite` scores only 0.529 standalone on `sub_10`; its
entire value is complementary — precisely the value this selector cannot harvest.)

## The lever that's left: fix the selector, not the pool

Pool changes raise the ceiling; they do not move the router. The remaining levers are all
selector-side: (a) **calibrate** the independent heads (Platt/isotonic) and use per-model τ
so probabilities are comparable across models; (b) **learn to route directly** — train one
model whose target is "the cheapest model that is actually correct," instead of five
independent correctness heads; (c) **richer query features** than a 384-dim MiniLM
embedding; (d) **output-side confidence** (self-consistency or logprobs from a cheap first
pass) rather than predicting correctness from the prompt text alone. Until per-query
model-correctness on hard items is more predictable, the oracle's complementarity stays
out of reach regardless of what is added to the pool.

### Superseded next step (now done)
The 2026-08-22 update proposed running exactly this experiment against an oracle target of
0.840. Executed: the reachable oracle on the live pool is 0.832, and the router captured
none of the lift. The actionable direction is no longer "add models" but "fix the selector."

## Update (2026-08-23): selector lever (a) tried — calibration + per-model tau

Lever (a) above was implemented cache-only (`phase2/calibrate_route.py`): each head refit
and wrapped in isotonic calibration (`CalibratedClassifierCV`), then thresholds tuned to
maximize RouterArena Acc-Cost score `S` on a held-out 20% slice of the *external corpus*
(never sub_10), as a single global tau and as per-model tau via greedy coordinate ascent.
Evaluated on sub_10:

| Policy (sub_10, never tuned on) | Acc | Cost/1k | Arena-S | Routing (cheap→exp) |
|---|---|---|---|---|
| uncalibrated global-tau (prior) | 0.709 | $0.251 | 0.700 | 92/27/599/12/1 |
| calibrated global-tau | 0.709 | $0.240 | 0.700 | 86/20/584/22/19 |
| calibrated per-model tau | **0.717** | $0.266 | 0.707 | 2/5/**723**/1/0 |
| oracle | 0.832 | $0.064 | 0.824 | — |

Two findings: (1) **calibration was not the missing piece** — val Brier scores barely moved
(deepseek 0.1222→0.1200), the raw logistic heads were already well-calibrated, so
cross-model comparability was not actually broken. (2) **Per-model tau gains +1.1 pts
(0.706→0.717) but by collapsing to the single strongest model, not by exploiting
diversity**: it routes 723/731 queries to deepseek and only 1 to gemini, 0 to gpt-4o-mini,
landing at deepseek's standalone accuracy. The threshold search's best learnable policy is
"always use the strongest base model." The ~11.5-pt gap to the oracle is therefore not a
calibration/thresholding problem; per-query model-correctness on the hard items is close to
unpredictable from a MiniLM embedding. Remaining untried levers: (b) direct learn-to-route,
(c) richer features, (d) output-side confidence — all bet on making that prediction better,
which is the actual bottleneck.

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

## Update (2026-08-23b): domain-aware routing — best arena-S so far, but capped by coverage

Complementarity was confirmed domain-structured (a per-dataset analysis gave a
domain-routing ceiling of 0.772 acc / 0.762 arena-S vs 0.716/0.705 for the best single
model). Root cause of the earlier router's failure was also found: the corpus was
domain-narrow (all-`business` MMLU-Pro + MATH, a sampling bug in `build_corpus.py`), so the
heads were out-of-distribution on 69 of sub_10's 71 domains. Fixes shipped:
`phase2/build_corpus_domains.py` builds a 780-item corpus across 19 domains (all 14 MMLU-Pro
categories, math, medical x2, science, knowledge), all 5 pool models relabeled on it, and
`phase2/domain_router_eval.py` trains a query->domain classifier + a per-domain best-model
table and routes sub_10 by predicted domain. Transfer was validated first (corpus-best model
per domain matches sub_10-best on the domains we already had).

| Policy (sub_10, 731 scorable) | Acc | Cost/1k | Arena-S |
|---|---|---|---|
| single: deepseek (best single) | 0.7155 | $0.266 | 0.7053 |
| single: qwen3-235b (cheapest) | 0.6908 | $0.034 | 0.7002 |
| prior calibrated per-model tau | 0.717 | $0.266 | 0.707 |
| domain-router (acc-max) | 0.6922 | $0.244 | 0.6856 |
| **domain-router (cost-aware)** | **0.7073** | **$0.090** | **0.7082** |
| domain-router (margin-gated) | 0.7155 | $0.266 | 0.7053 |
| oracle | 0.8317 | $0.064 | 0.8236 |

The cost-aware domain router is the **best arena-S found** (0.7082): it matches deepseek's
accuracy within ~1 pt at one-third the cost, by defaulting to cheap qwen and escalating to
deepseek only where a domain clearly needs it. The acc-max variant *regresses* (0.686) —
it trusts the corpus top-1 and misroutes on transfer noise (corpus said gemini wins
mmlupro_math; gemini is weak on sub_10). Margin-gating collapses back to deepseek.

Domain routing thus works directionally but stays ~5 pts of arena-S below the 0.762 ceiling.
Three binding limits, in order of size: (1) **domain coverage** — ~40% of sub_10 (code,
translation, reading, trivia, chess, music, ethics, SuperGLUE) is absent from the corpus and
gets misrouted to the nearest covered domain; (2) the domain classifier is only 76% accurate
over 19 domains; (3) per-domain table transfer noise (mitigated by the cost-aware/robust
variants).

### The real blocker
Every result here is on the **731 scorable (boxed MCQ/numeric) items only**. The actual
RouterArena leaderboard scores all 8,400 items with format-specific scorers (code execution,
translation BLEU, reading-comprehension), and the complementarity that would move the needle
toward the ceiling lives disproportionately in exactly those unscored domains (e.g. the pool
carries Qwen3-Coder-Next specifically for code, invisible in a boxed-match grade). On the
scorable subset deepseek is already a near-best generalist, so selector work here has hit
diminishing returns. Making real leaderboard progress needs the **official multi-scorer
evaluator** run over the full set — which OOMs the dev machine, so it is the natural first
use of an off-laptop (k8s) run. That, not more selector tuning, is the next lever.

## Update (2026-08-23c): official evaluator on k8s — and why local official numbers are NOT trusted

Built the official RouterArena evaluator as an off-laptop job (`deploy/routerarena-eval/`:
Dockerfile + summarize + k8s Job). The official `llm_evaluation/run.py` crashes on macOS
(multiprocessing / code-sandbox `resource_tracker` leak) but runs on Linux; the container/Job
completes. A k8s Job in the `cruq` namespace succeeded end-to-end. So far so good.

**But the local reproduction does not faithfully match RouterArena's official scoring, and this
was caught by validating against a known reference.** Scoring the current #1 leaderboard router
(`Paix2-router.json`, official 79.69% on full) through this harness gives **0.475 on sub_10**.
The cause: the sub_10 prediction files' `global index` keys do not all match the evaluator's
full-arrow `all_data`, so `_get_ground_truth` silently returns `None` and entire dataset families
(AsDiv, FinQA, QANTA, WMT19, SuperGLUE, ...) score **0.000 for EVERY router**, including Paix2.
`math_metric` scores those answers correctly in isolation; the failure is data plumbing, not the
scorer. (Also learned along the way: the double-brace `\boxed{{}}` prompt is RouterArena's OWN
canonical prompt from its eval config, not our bug.)

**Consequences / what to trust:**
- The "true official" per-model / oracle / domain-ceiling numbers computed locally are
  **contaminated** and are discarded. Do not cite them.
- The lightweight boxed-match grader remains a fair proxy for the ~590 MCQ items, and the
  domain-router result on that metric (best arena-S 0.7082, cost-aware) stands.
- The ONLY trustworthy leaderboard-comparable number is RouterArena's own `/evaluate` PR
  workflow (their infra scores the submission correctly). A test submission is the next step
  for a real number; local harness hardening (fix the sub_10↔full global-index join) is the
  alternative if we want to iterate off-leaderboard.

Method note: validating a new measurement harness against a known-good reference (here, the #1
router's public score) before trusting its outputs is the check that prevented reporting
contaminated numbers as truth.

## Update (2026-08-23d): local harness FIXED — trustworthy official numbers

The local-eval unreliability from the previous update was root-caused and fixed. The bug:
the eval image OMITTED `config/eval_config/`, so `load_eval_config_for_dataset` found no
metrics and the evaluator silently fell back to `mcq_accuracy` for EVERY dataset — numeric
(AsDiv/FinQA/MATH/AIME), translation (WMT19), and word-sense (SuperGLUE) answers were graded
as multiple-choice and scored 0 for ALL routers, including Paix2. Fixed by baking
`config/eval_config/` into the image (`talentreviewai/routerarena-eval:v3`,
`deploy/routerarena-eval/`). Validation against the reference: Paix2 sub_10 0.475 → 0.52+,
with AsDiv 0→0.43, FinQA 0→0.14, WMT19 0→0.41, SuperGLUE-Wic 0→0.50 recovering. (QANTA zeros
are genuine — models ramble instead of naming the gold entity.)

**Corrected TRUE official numbers (v3, sub_10, 771 commonly-scored items):**

| Model / policy | Acc | Cost/1k | Arena-S |
|---|---|---|---|
| deepseek | 0.749 | $0.238 | 0.736 |
| qwen3-235b (cheapest) | 0.729 | $0.030 | 0.736 |
| coder-next | 0.689 | $0.096 | 0.691 |
| gemini-2.5-flash-lite | 0.642 | $0.166 | 0.643 |
| gpt-4o-mini | 0.625 | $0.078 | 0.634 |
| **domain-routing ceiling** | **0.790** | $0.141 | **0.778** |
| per-query oracle | 0.853 | $0.067 | 0.842 |

These supersede the "true official" numbers in update 23b/c (which were contaminated by the
missing-config bug) and also come out somewhat ABOVE the lightweight-grader proxy (deepseek
0.716→0.749, oracle 0.832→0.853), because the official scorers award partial credit
(meteor/rouge) and score datasets the boxed-match grader skipped. Domain routing is worth
**+5.4 pts arena-S** on the true metric (best single 0.736 → ceiling 0.778), with the oracle a
further +6.4 above that. The trustworthy iteration metric now exists; next is to build the
domain router's full prediction file and score it under v3, then push toward the ceiling.

## Update (2026-08-24): domain router scored on the FIXED (v3) official metric

Built the domain router's prediction file (`phase2/build_domain_prediction.py`: train the
query->domain classifier + per-domain table on the external corpus, route each sub_10 query
to the selected model, emit that model's cached output) and scored it with the fixed v3
evaluator.

| Router (sub_10, official v3) | Acc | Cost/1k | Arena-S |
|---|---|---|---|
| best single (deepseek / qwen) | 0.749 / 0.729 | $0.238 / $0.030 | 0.736 |
| **domain router (cost-aware)** | **0.740** | **$0.094** | **0.737** |
| domain router (acc-max) | 0.733 | $0.213 | 0.723 |
| domain-routing ceiling | 0.790 | $0.141 | 0.778 |
| per-query oracle | 0.853 | $0.067 | 0.842 |

The cost-aware domain router lands at **arena-S 0.737 — a marginal win over the best single
model, at deepseek-level accuracy for 40% of the cost** (routes 528/809 to cheap qwen, 257 to
deepseek, 24 to coder). The acc-max table is WORSE (0.723): per-domain top-1 estimated on the
corpus carries transfer noise onto sub_10, and it costs more. The router does NOT reach the
0.778 ceiling; the cap is the same on the true metric as on the lightweight proxy: (1) the
domain classifier is ~76% accurate, (2) ~40% of sub_10 domains (code, translation, reading,
trivia, chess, music, ethics, SuperGLUE) are absent from the 19-domain corpus and get
misrouted, (3) per-domain best-model transfer noise. Closing the gap needs broader domain
coverage in the corpus + a better classifier, not a different routing policy (both policies
tested land at/below the best single model).

## Update (2026-08-24b): widening corpus coverage did NOT help — transfer is the real cap

Widened the corpus from 19 to 23 domains (`phase2/build_corpus_domains.py` now also builds
`trivia` (TriviaQA), `science_qa` (SciQ), `commonsense` (CommonsenseQA), `word_sense`
(SuperGLUE-Wic) — the largest sourceable, boxed-match-labelable sub_10 domains, ~158 OpenTDB/
QANTA trivia items among them). Labeled the 160 new items x 5 models, retrained the classifier
+ table, rebuilt and re-scored the router under v3.

**Result: arena-S 0.7325 — slightly WORSE than the 19-domain router's 0.7372.** Widening
coverage regressed. Diagnosis: the new `trivia` domain routes to deepseek, because on the
external trivia set (TriviaQA) deepseek is the per-domain best; but sub_10's actual trivia
(OpenTDB) is better and far more cheaply served by qwen. The external per-domain best-model
does not match RouterArena's per-domain best-model **even within the same nominal domain**,
because the specific dataset/distribution differs (TriviaQA vs OpenTDB, SciQ vs OpenTDB-Science,
CommonsenseQA vs SocialiQA). This is transfer failure, and it is a direct consequence of the
"no training on RouterArena data" rule.

**Conclusion for the domain-routing line of attack.** Across every variant tried — lightweight
vs true metric, 19 vs 23 domains, cost-aware vs acc-max — the domain router lands at
**arena-S ~0.737, tied with the best single model**, and never approaches the 0.778 per-dataset
ceiling (let alone the 0.842 oracle). Adding coverage did not move it; the cap is not coverage
but transfer: you cannot learn RouterArena's per-dataset model preferences from external data
because even same-named domains differ in distribution. The 19-domain cost-aware router is kept
as canonical (`cruq-domain-router.json`, acc 0.740 / cost $0.094 / arena-S 0.737: best-single
accuracy at 40% of the cost — a real efficiency win, not a ceiling-capturing one).

The honest state of the RouterArena effort: pool is not the bottleneck (oracle 0.85); the
selector is, and every selector we can build under the no-train rule (lexical, per-model-P,
calibrated, domain) tops out at best-single arena-S. Capturing the pool's complementarity would
require per-query/per-dataset model-preference signal that the rules forbid learning — or an
online signal (a cheap first-pass probe / self-consistency) not yet built. That, not more pool
or corpus work, is the only remaining lever.
