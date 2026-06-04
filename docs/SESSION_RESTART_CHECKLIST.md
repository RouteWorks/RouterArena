# Session Restart Checklist

> Use this when starting a new Claude Code session to continue RouterArena
> optimization work, after the v10.1.1 submission cleanup and the llm-router
> hook + dashboard fixes.

## 1. Pre-flight (run outside Claude Code)

### 1.1 Restart the llm-router MCP server with new config

The currently-running MCP server (PID was 2100 in the cleanup session) was
started before:
- `qwen2.5:7b` was installed (faster Ollama model)
- `~/.llm-router/config.yaml` had `ollama_budget_models` updated
- `~/.llm-router/policies/subscription_first.yaml` was written
- llm-router PR #21 (dashboard + enforce-route fixes) was merged

Restart it so a new Claude Code session picks up the new config:

```bash
# Find and kill the existing MCP server
ps aux | grep llm-router-sse | grep -v grep
# Replace 2100 with the actual PID from the previous line
kill 2100
sleep 2

# Optional: activate the subscription-first policy
export LLM_ROUTER_POLICY=subscription_first

# Restart
cd /Users/yali.pollak/Projects/llm-router && uv run llm-router-sse 17891 &
disown
```

### 1.2 Verify the new config is live

```bash
# Should show qwen2.5:7b ahead of qwen3.5
grep ollama_budget_models ~/.llm-router/config.yaml

# Should show the subscription_first policy file
ls ~/.llm-router/policies/

# Should respond fast (~2s for short prompts)
curl -s --max-time 10 http://localhost:11434/api/generate \
  -d '{"model":"qwen2.5:7b","prompt":"What is 2+2?","stream":false}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"latency: {d.get(\"total_duration\",0)/1e9:.2f}s")'
```

### 1.3 Pull the latest llm-router improvements

If llm-router PR #21 is merged:
```bash
cd /Users/yali.pollak/Projects/llm-router
git checkout main
git pull
```

If still open, work off the PR branch:
```bash
cd /Users/yali.pollak/Projects/llm-router
git checkout fix/dashboard-and-enforce-route
git pull
```

## 2. Start a new Claude Code session

Open a fresh terminal, `cd /Users/yali.pollak/Projects/RouterArena`, run `claude`.

The new session should have:
- ✅ Faster Ollama (qwen2.5:7b at 2-3s warm vs the old 12-18s)
- ✅ Fixed enforce-route hook (loop-detection auto-pivot, read-only Bash allowlist, correct messaging)
- ✅ Dashboard tracking (claude_usage table writes from session_spend)

## 3. Brief Claude on the situation

Open the new session with this context (paste as the first message):

> Continue the RouterArena optimization work for PR #132. The previous session's
> Lever #3 was rejected as test-set leakage and the branch was reset to baseline
> `065cca5`. Read `docs/ROUTERARENA_IMPROVEMENT_PLAN.md` and pick up at Tier 1
> (free wins: better system prompts + self-consistency + robustness investigation).
> Before any submission, run `uv run python scripts/check_submission_integrity.py`
> and ensure it exits 0.

## 4. Before any submission to the PR

Run the integrity check:

```bash
cd /Users/yali.pollak/Projects/RouterArena
uv run python scripts/check_submission_integrity.py \
  --predictions router_inference/predictions/llm-router.json \
  --baseline router_inference/predictions/llm-router.json.bak.honest \
  --scripts-dir scripts --scripts-dir router_inference/router \
  --plan /tmp/reassignment_plan.json
```

It must exit 0 ("ALL CHECKS PASSED"). If it exits 1, fix the leak before pushing.

## 5. Working with the optimization branch

The legitimate-improvement plan lives on branch `feat/legitimate-improvement-plan`.
Build new optimization work as additional commits on top, OR create a child branch:

```bash
git checkout feat/legitimate-improvement-plan
git checkout -b feat/tier-1-system-prompts  # example
# ...do work...
uv run pytest tests/test_submission_integrity.py  # verify checks still pass
uv run python scripts/check_submission_integrity.py  # verify state is clean
```

When ready to submit:

```bash
git push fork HEAD:submit-llm-router-v10.1.1 --force-with-lease
gh pr comment 132 --repo RouteWorks/RouterArena --body "/evaluate"
```

## 6. Quick reference — what changed since last session

| File | What |
|---|---|
| `docs/ROUTERARENA_IMPROVEMENT_PLAN.md` | Tiered roadmap of legitimate optimizations |
| `docs/SESSION_RESTART_CHECKLIST.md` | This file |
| `scripts/check_submission_integrity.py` | Pre-submission leakage detector |
| `tests/test_submission_integrity.py` | 13 tests proving the detector catches the Lever #3 pattern |
| `~/.llm-router/config.yaml` | qwen2.5:7b prioritized over slow qwen3.5 |
| `~/.llm-router/policies/subscription_first.yaml` | Chain: gemini_cli → codex → claude haiku → ollama |
| llm-router PR #21 | Dashboard savings tracking + enforce-route deadlock recovery |
| RouterArena PR #132 | Reset to clean baseline (`065cca5` → `792bd52`) |

## 7. Known follow-ups

- Robustness score drop (0.30 → 0.236 in the prior submission). The robustness
  predictions file wasn't modified, so the change came from another input.
  Worth understanding before the next submission cycle.
- The llm-router enforce-route patches are not yet deployed to the live hook
  at `~/.claude/hooks/llm-router-enforce-route.py` — they live in
  `src/llm_router/hooks/enforce-route.py` and need to be copied/symlinked
  on PR #21 merge.
