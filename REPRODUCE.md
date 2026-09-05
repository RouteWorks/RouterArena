# Shunt submission — RouterArena (sub_10) — how to reproduce

Router **shunt** (@KookaS, https://github.com/KookaS/shunt, open-source) is a thin
adapter over Shunt's real router engine: every prompt is routed through
`RouterEngine.decide` under Shunt's shipped `session_cascade` strategy over
Shunt's live pool (`deepseek-v4-flash`, `deepseek-v4-pro`). RouterArena's
stateless single-turn / no-verification regime provides no verified-outcome
channel, so Shunt's verified-failure escalation ladder never fires and the
router deterministically picks its cheapest live model
(`deepseek/deepseek-v4-flash`).

## Pin

The router code was generated against Shunt at this commit (Shunt is not on
PyPI — install from git):

```
SHUNT_PIN = 7ee66af5ddcbd78886a58e5336b21ffc5291da56
pip install "git+https://github.com/KookaS/shunt@7ee66af5ddcbd78886a58e5336b21ffc5291da56"
```

or point `SHUNT_SRC` at a Shunt `src/` checkout of that commit.

## Regenerate the routing decisions

From this repository root (a Shunt install or `SHUNT_SRC` must be available):

```
uv run python router_inference/generate_prediction_file.py shunt sub_10 --no-optimality
uv run python router_inference/generate_prediction_file.py shunt robustness --no-optimality
```

The recorded answers in `predictions/shunt.json` are the models' own outputs
(provider generations are not byte-reproducible); the router CHOICES are
deterministic. The two prediction files are required by RouterArena's workflow
(exactly one base file + one `-robustness` file per PR).

## Honest framing

This is a model baseline produced by Shunt's real `session_cascade` machinery:
without a verification channel the router stays on its cheapest live model.
Training/tuning on RouterArena data was not used — the engine routes from Shunt's
shipped configuration only.
