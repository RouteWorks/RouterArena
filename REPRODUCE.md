# Reproducing the Shunt submission

This repository state reproduces the Shunt RouterArena submission
(`router_inference/predictions/shunt.json` for the base split and
`router_inference/predictions/shunt-robustness.json` for the robustness split)
**from this repo alone**: the router module
(`router_inference/router/shunt_session_cascade.py`) imports Shunt pinned at a
single commit and seeds its kNN outcome index from the QA corpus bundled here
under `router_inference/router/shunt_seed/`, so no other checkout and no env
vars are needed.

## Pinned Shunt

Shunt is not on PyPI; the exact commit this submission's predictions were
generated with is pinned as `SHUNT_PIN` in the router module:

- commit: `7ee66af5ddcbd78886a58e5336b21ffc5291da56`
- install: `python -m pip install "git+https://github.com/KookaS/shunt@7ee66af5ddcbd78886a58e5336b21ffc5291da56"`

## Seeded prior (bundled in `router_inference/router/shunt_seed/`)

The router runs Shunt's `knn_semantic_cascade` strategy over a menu of
2 models (deepseek/deepseek-v4-flash, deepseek/deepseek-v4-pro), seeded from a
measured QA-domain corpus. RouterArena's stateless single-turn regime gives
Shunt no verified-outcome channel (no repo to re-test, no confirmation step),
so the verified-failure escalation ladder is inert; the routing signal is the
semantic neighbourhood — a prompt whose neighbours lack evidence that the cheap
rung succeeds is escalated to the rung the corpus evidence supports.

Seed provenance:

- **Sample**: 200 labeled single-turn QA prompts —
  120 MMLU + 80 GSM8K —
  selected with the fixed seed `20260904`
  (`sample.json` carries the full prompts + gold answers).
- **Grading**: deterministic string/letter grading against the gold answers — no
  LLM judge. Rows record `pass` (terminal bool) and the billed `cost` per
  (prompt, model) call (`results.json`).
- **Measured**: `2026-09-04T08:16:59+00:00` (results.json, 1000 measured (prompt, model) rows across all calibration candidates; 400 of them seed the index) over the sample created `2026-09-04T08:16:46+00:00`.
- **Embedder**: the shipped Shunt `Embedder` (`jinaai/jina-embeddings-v2-base-code`, dim 768);
  the bundled `embeddings.json` holds its vectors for every sample prompt
  (keyed by prompt digest), so index rebuild needs no model download.
- **Menu model prices** (RouterArena `model_cost` table, USD per 1M tokens):

| model | input $/1M | output $/1M |
|---|---|---|
| deepseek/deepseek-v4-flash | 0.14 | 0.28 |
| deepseek/deepseek-v4-pro | 0.435 | 0.87 |

Measured pass rate and cost on the sample (the rows that seed the index):

| model | pass | cost (USD) | pass rate |
|---|---|---|---|
| deepseek/deepseek-v4-flash | 167/200 | $0.0166 | 83.5% |
| deepseek/deepseek-v4-pro | 179/200 | $0.0633 | 89.5% |

## Regenerating the predictions

The router module resolves the seed from `router_inference/router/shunt_seed/`
first (when present), falling back to the deployer's env/artifact paths. A
regeneration therefore needs only the pinned Shunt install above, from this
repo's root:

```bash
# choices for the base split and the robustness split (free, no API calls):
python router_inference/generate_prediction_file.py shunt sub_10 --no-optimality
python router_inference/generate_prediction_file.py shunt robustness --no-optimality
```

A choice re-run writes `generated_result: null` rows — it does not re-produce
the recorded provider answers already in the submitted file (those are measured
data, not router output). To verify byte-identical reproduction of the
**submitted** files, save the recorded answers aside before regenerating and
re-attach them by `global index`, then diff. The re-attach mirrors each
committed file's own trailing-newline convention (the base file was rewritten by
the arena's `run.py`, which adds one; the robustness file is the generator's
raw output, which does not), so an identical re-run leaves a truly empty diff:

```bash
git show HEAD:router_inference/predictions/shunt.json > /tmp/shunt-committed.json
git show HEAD:router_inference/predictions/shunt-robustness.json > /tmp/shunt-robustness-committed.json
python router_inference/generate_prediction_file.py shunt sub_10 --no-optimality
python router_inference/generate_prediction_file.py shunt robustness --no-optimality
python - <<'PY'
import json
from pathlib import Path

def reattach(committed_path, regenerated_path):
    committed = {row["global index"]: row for row in json.loads(Path(committed_path).read_text())}
    rows = json.loads(Path(regenerated_path).read_text())
    for row in rows:
        if row["global index"] in committed:
            row["generated_result"] = committed[row["global index"]]["generated_result"]
    newline = Path(committed_path).read_bytes().endswith(b"\n")
    text = json.dumps(rows, ensure_ascii=False, indent=2)
    Path(regenerated_path).write_text(text + ("\n" if newline else ""))

reattach(Path("/tmp/shunt-committed.json"), Path("router_inference/predictions/shunt.json"))
reattach(Path("/tmp/shunt-robustness-committed.json"), Path("router_inference/predictions/shunt-robustness.json"))
PY
git diff --exit-code -- router_inference/predictions/shunt.json router_inference/predictions/shunt-robustness.json
```

An empty diff proves the committed router module + bundled seed + pinned Shunt
reproduce the submitted predictions exactly. If it is not empty, the diff shows
which prompts the reproduction routes differently — investigate before trusting
a re-run.
