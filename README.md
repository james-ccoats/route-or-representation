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
| 0 | Setup: load OLMoE, score MC by option log-probs, record top-k routing | scoring, routing recorder and structure inspection done and tested on a tiny MoE; `load_model` and the real-model check pending |
| 1 | Data: MMLU domain pairs, four prompt conditions | pilot and ladder prompts built (`data/prompts/pilot`, `data/prompts/ladder`); main-run prompts not yet built |
| 2 | Pilot: few-shot beats zero-shot? relevant examples shift routing more? | pipeline built and tested on a tiny MoE; not yet run on OLMoE |
| 3 | Main experiment: forced routing, reverse check, sanity checks | forcing, both scripts and the gain-fraction analysis built and tested on a tiny MoE; not yet run on OLMoE; `05_make_figures.py` still a stub |
| 4 | Extensions: layer-by-layer forcing, second MoE model, distance/direction | not started |

"Tested on a tiny MoE" means a real `OlmoeForCausalLM` with random weights built in
the test suite, so the whole pipeline is exercised without downloads or a GPU. No
result from the real model exists yet, and nothing here is a scientific finding.


## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync                      # create .venv and install dependencies
uv run pytest                # run tests
uv run ruff check .          # lint

# Phase 0 smoke test (stops at NotImplementedError in `load_model` until it is written)
uv run python scripts/00_smoke_test.py --config configs/models/olmoe.yaml
```

Experiment scripts, in order:

```bash
uv run python scripts/01_build_prompts.py  --config configs/experiments/pilot.yaml
uv run python scripts/02_run_pilot.py      --config configs/experiments/pilot.yaml
uv run python scripts/06_measure_distance.py --config configs/pairs/mmlu_ladder.yaml  # no model or GPU
uv run python scripts/01_build_prompts.py  --config configs/experiments/ladder_pilot.yaml
uv run python scripts/02_run_pilot.py      --config configs/experiments/ladder_pilot.yaml
uv run python scripts/01_build_prompts.py  --config configs/experiments/force_routing.yaml
uv run python scripts/03_force_routing.py  --config configs/experiments/force_routing.yaml
uv run python scripts/04_reverse_check.py  --config configs/experiments/reverse_check.yaml
uv run python scripts/05_make_figures.py   --config configs/experiments/force_routing.yaml
```

`ladder_pilot` is the fixed-target distance ladder: one target (`college_biology`) and
five sources at increasing distance, so the target questions are identical across rungs
(see D12). `06_measure_distance.py` prints a model-free lexical distance beside each
run's expected label; it is a sanity check on the labels, not the explanatory variable.

`03` and `04` read `data/prompts/main`, so build those prompts first (the second
`01` command above; `reverse_check.yaml` points at the same directory). Building
prompts needs only the tokenizer and MMLU, not the model weights.

Scripts `02`-`04` write one JSONL file per run (domain pair and direction) to the
config's `output_dir`, one record per question x condition x intervention, and print
a summary table when they finish. They are **resumable**: re-running the same command
after a crash or preemption continues where it stopped. Each results file has a
`<run>.manifest.json` beside it recording the model, forcing options and plans, and
a hash of the prompt file; resuming after any of those changed is refused with the
differences listed. To change a setting, delete that run's results and manifest and
start it over.

`03` runs each few-shot prompt with zero-shot routing forced on the target-question
tokens (fraction of gain *retained*) plus two sanity controls (forcing a prompt's own
routing back onto itself must change nothing; forcing random experts must hurt). `04`
forces few-shot routing into the zero-shot prompt (fraction of gain *recovered*).

## Layout

```
configs/      YAML configs (models, domain pairs, experiments)
docs/         research plan, design decisions, experiment log, related work
src/rr/       package (see below)
scripts/      numbered entry points, one per phase step; all take --config
tests/        pytest suite (runs on a tiny random-weight OLMoE; no downloads or GPU)
data/ results/ figures/ notebooks/ paper/   working directories (contents gitignored where large)
```

Package modules (`src/rr/`):

```
config, data, prompts       configs, MMLU loading/sampling, the four prompt conditions
distance                    model-free lexical distance between MMLU subjects (labels check)
models, scoring             MoE structure inspection, option log-prob scoring
routing/                    record (hooks), force (hooks), align (target tokens), metrics (Jaccard)
results, runner, manifest   result-record schema + JSONL I/O, resumable per-file runner, run manifests
pilot                       Phase 2: score conditions, routing shift vs zero-shot
force_experiment, force_cli Phase 3: forcing plans, evaluation, shared script driver
analysis, force_analysis    pilot contrasts; gain fractions with bootstrap intervals, sanity checks
```

## Documentation

- [Research plan](docs/research_plan.md): question, hypotheses, phases, success criteria
- [Design decisions](docs/design_decisions.md): running log of methodological choices
- [Experiment log](docs/experiment_log.md): dated entries (template)
- [Related work](docs/related_work.md): RARE, routing-drift paper

## Conventions

Python 3.11+, `src/` layout, package name `rr`. All dependencies are managed
with uv.
