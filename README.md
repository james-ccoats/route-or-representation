# route-or-representation

**Route or representation?** In mixture-of-experts (MoE) language models, when
few-shot examples from a *source* domain improve accuracy on a *target* domain,
how much of the gain comes from the router selecting **different experts**
(routing), and how much from the **same experts processing shifted hidden
states** (representation)?

We answer this with a routing-replay intervention: run the few-shot prompt but
force the zero-shot routing on the target-question tokens, and measure how much
of the gain survives. A reverse check forces few-shot routing into the
zero-shot prompt.

## Hypothesis

[RARE](docs/related_work.md#rare) steers MoE models by changing hidden states
in the null space of the router, so routing stays fixed while behavior changes.
That shows hidden-state shifts *alone* can carry behavioral change. Motivated by
this, we hypothesize that a **substantial share of cross-domain few-shot gain is
representational**: forcing zero-shot routing will retain a large fraction of
the gain, and forcing few-shot routing into a zero-shot prompt will recover
only a small fraction. We expect the routing share to grow with domain
distance. These are hypotheses, not results; see
[docs/research_plan.md](docs/research_plan.md) for falsification criteria.

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Setup: load OLMoE, score MC by option log-probs, record top-k routing | scaffolding |
| 1 | Data: MMLU domain pairs, four prompt conditions | not started |
| 2 | Pilot: few-shot beats zero-shot? relevant examples shift routing more? | not started |
| 3 | Main experiment: forced routing, reverse check, sanity checks | not started |
| 4 | Extensions: layer-by-layer forcing, second MoE model, distance/direction | not started |


## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync                      # create .venv and install dependencies
uv run pytest                # run tests
uv run ruff check .          # lint

# Phase 0 smoke test (will stop at NotImplementedError until Phase 0 is built)
uv run python scripts/00_smoke_test.py --config configs/models/olmoe.yaml
```

Experiment scripts, in order:

```bash
uv run python scripts/01_build_prompts.py  --config configs/experiments/pilot.yaml
uv run python scripts/02_run_pilot.py      --config configs/experiments/pilot.yaml
uv run python scripts/03_force_routing.py  --config configs/experiments/force_routing.yaml
uv run python scripts/04_reverse_check.py  --config configs/experiments/reverse_check.yaml
uv run python scripts/05_make_figures.py   --config configs/experiments/force_routing.yaml
```

## Layout

```
configs/      YAML configs (models, domain pairs, experiments)
docs/         research plan, design decisions, experiment log, related work
src/rr/       package: config, models, data, prompts, scoring, analysis, routing/
scripts/      numbered entry points, one per phase step; all take --config
tests/        pytest suite
data/ results/ figures/ notebooks/ paper/   working directories (contents gitignored where large)
```

## Documentation

- [Research plan](docs/research_plan.md): question, hypotheses, phases, success criteria
- [Design decisions](docs/design_decisions.md): running log of methodological choices
- [Experiment log](docs/experiment_log.md): dated entries (template)
- [Related work](docs/related_work.md): RARE, routing-drift paper

## Conventions

Python 3.11+, `src/` layout, package name `rr`. All dependencies are managed
with uv.
