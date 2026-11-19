# route-or-representation

**Route or representation?** In mixture-of-experts (MoE) language models, when
few-shot examples from a question's own subject improve accuracy, how much of the
gain comes from the router selecting **different experts** (routing), and how much
from the **same experts processing shifted hidden states** (representation)?

We answer this with a routing-replay intervention, in one forward pass, on the
target-question tokens:

- **Routing-removed:** run the few-shot prompt but force the zero-shot routing.
  *Gain kept* = (forced - zero-shot) / (few-shot - zero-shot).
- **Routing-added:** run the zero-shot prompt but force the few-shot routing.
  *Gain added* uses the same formula.

The slide deck (`docs/presentation.html`) is the current statement of the design;
the plan, decisions and related work in `docs/` follow it.

## Design in brief

- **Model and task:** `allenai/OLMoE-1B-7B-0924` (8 of 64 experts, 16 MoE layers); MMLU
  multiple choice scored by answer-option log-probs.
- **Primary condition: same-subject few-shot.** Each subject is both example source and
  target (examples exclude the scored question). Cross-domain examples, and whether the
  routing share depends on domain distance, are a Phase 4 extension with no hypothesis.
- **Prompt conditions:** zero-shot; same-subject few-shot; length-matched irrelevant
  examples from another subject (control). k = 1 to 5 (nested), 3 example-set replicates.
- **Controls and damage anchors.** Replay mixes the routing's value with mismatch damage
  (routing and hidden state were not computed together). Anchors: identity check, matched
  replay (examples from another same-subject set), graded self-perturbations (next-best
  swap, random experts in k of 16 layers), a random-expert floor, and a below-baseline
  rule. A high gain kept is robust to damage; a low gain kept or added is read against
  the anchors.
- **Example-token replay (planned, main design).** Headline runs leave the example tokens'
  own routing in place. A separate run perturbs it to test whether example-token routing
  is needed.
- **Irrelevant-prompt forcing (planned).** The same two forced runs on the irrelevant
  prompt show how much of the split is generic context and length, not content.

## Hypotheses

- **H1:** same-subject few-shot beats zero-shot for some subjects.
- **H2a:** few-shot examples shift routing on question tokens beyond noise and beyond the
  irrelevant control (size of shift, not direction).
- **H2b:** the shift is content-driven: routing under one same-subject example set is
  closer to routing under another set than to zero-shot or irrelevant routing.
- **H3:** routing-removed keeps a large fraction of the gain. Motivated by
  [RARE](docs/related_work.md#rare), which shows hidden-state changes with routing held
  fixed can steer behavior.
- **H4:** routing-added adds only a small fraction of the gain.

H3 and H4 are the main claims; H1, H2a and H2b are preconditions checked in the pilot.
If both hold, the claim is that routing at the tested positions is neither necessary nor
sufficient; attributing the gain to representation (the same experts processing shifted
hidden states) follows by elimination. The design does not isolate which non-routing path
carries it (attention versus experts, example-token versus question-token states). A
finding that routing carries most of the gain is as valid as the reverse. These are
hypotheses, not results; see [docs/research_plan.md](docs/research_plan.md).

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Setup: load OLMoE, score MC by option log-probs, record top-k routing | scoring, routing recorder and structure inspection done and tested on a tiny MoE; `load_model` and the real-model check pending |
| 1 | Data: MMLU subjects, same-subject prompt conditions with a length-matched irrelevant control, k = 1 to 5, three example-set replicates | done: `data/prompts/pilot` and `main` rebuilt for four subjects (college biology, chemistry, physics, computer science), 60 files each; `ladder` is the old cross-domain set kept for the Phase 4 extension |
| 2 | Pilot: few-shot beats zero-shot? relevant examples shift routing more? | pipeline built and tested on a tiny MoE; not yet run on OLMoE |
| 3 | Main experiment: forced routing, reverse check, sanity checks | forcing, both scripts and the gain-fraction analysis built and tested on a tiny MoE; not yet run on OLMoE; `05_make_figures.py` still a stub |
| 4 | Extensions: layer-by-layer forcing, second MoE model, cross-domain examples and distance/direction | not started |

"Tested on a tiny MoE" means a real `OlmoeForCausalLM` with random weights built in
the test suite, so the whole pipeline is exercised without downloads or a GPU. No
result from the real model exists yet, and nothing here is a scientific finding.

**Design versus code.** Implemented for the same-subject primary design (D15): subject
config (`configs/pairs/mmlu_subjects.yaml`), same-subject prompts with the irrelevant control
matched to the same-subject block, pilot contrasts for H1 and H2a, and the forcing runs on
the same-subject prompt. The cross-domain pair and ladder code (D12) remains as the Phase 4
extension. **Not yet built:** the H2b measure, forcing of the irrelevant prompt,
example-token replay, matched replay and the graded self-perturbation anchors, and the
mismatch-size measurement. Also not done: pinning the model revision, numeric
pre-registered thresholds, and the compute plan.


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
uv run python scripts/01_build_prompts.py  --config configs/experiments/force_routing.yaml
uv run python scripts/03_force_routing.py  --config configs/experiments/force_routing.yaml
uv run python scripts/04_reverse_check.py  --config configs/experiments/reverse_check.yaml
uv run python scripts/05_make_figures.py   --config configs/experiments/force_routing.yaml

# Phase 4 extension (cross-domain distance ladder), not part of the primary design:
uv run python scripts/06_measure_distance.py --config configs/pairs/mmlu_ladder.yaml  # no model or GPU
uv run python scripts/01_build_prompts.py  --config configs/experiments/ladder_pilot.yaml
uv run python scripts/02_run_pilot.py      --config configs/experiments/ladder_pilot.yaml
```

To run these in order, use `scripts/run_all.py --stage {pilot,main,all}`
(`--dry-run` prints the commands). `all` stops after the pilot so the Phase 2 criteria are
checked before the main experiments; `--continue-past-gates` runs on. The extension stages
`ladder` and `ladder-forcing` run only when named.

For the ladder, `ladder_force_routing.yaml` and `ladder_reverse_check.yaml` run `03` and
`04` on the prompts that `ladder_pilot.yaml` built (no second `01` run needed).

`ladder_pilot` is the fixed-target distance ladder: one target (`college_biology`) and
five sources at increasing distance, so the target questions are identical across rungs
(see D12). `06_measure_distance.py` prints a model-free lexical distance beside each
run's expected label; it is a sanity check on the labels, not the explanatory variable.

With `record_hidden_states: true` (set in both pilot configs), `02` also records the
hidden state each router receives on the target-question tokens and saves per-layer
summaries of how far it moved from the zero-shot prompt's (D13). This adds no forward
passes. It is supporting evidence about the representation, not a causal test, and it
only means something next to its chance baseline and the irrelevant-example control.

`03` and `04` read `data/prompts/main` (same-subject prompts, 100 questions per subject), so build those prompts first (the `01` command for
`force_routing.yaml`; `reverse_check.yaml` points at the same directory). Building
prompts needs only the tokenizer and MMLU, not the model weights.

**Shots and replicates (D14).** Every experiment runs k = 1, 2, 3, 4, 5 examples, not just
zero-shot against k = 5. Each of 3 replicates draws its own set of 5 examples per subject
from `dev` and `validation` together (after dropping any that also appear in `test`), and the k-shot prompt uses the first k of them. Each
(subject, replicate, k) is a separate prompt file and run, named like
`college_biology_r1_k3`, with every question seeing the same examples within a run. `01`
refuses to write into a directory that holds prompt files it does not produce, so
rebuilding after changing these settings needs the old files removed first. The summaries
printed by `02`-`04` then show gain against k, with the range across replicates.

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
stages                      ordered experiment stages behind scripts/run_all.py
shots, prompt_build         run names for (subject, replicate, k); prompts for every replicate and k
shot_curves                 gain / routing shift / gain fractions against k, spread across replicates
models, scoring             MoE structure inspection, option log-prob scoring
routing/                    record (hooks), force (hooks), align (target tokens), metrics (Jaccard)
results, runner, manifest   result-record schema + JSONL I/O, resumable per-file runner, run manifests
pilot                       Phase 2: score conditions, routing shift vs zero-shot
force_experiment, force_cli Phase 3: forcing plans, evaluation, shared script driver
analysis, force_analysis    pilot contrasts; gain fractions with bootstrap intervals, sanity checks
```

## Documentation

- [Slide deck](docs/presentation.html): the proposal, 19 slides (open in a browser; arrow keys to move)
- [Research plan](docs/research_plan.md): question, hypotheses, phases, success criteria
- [Design decisions](docs/design_decisions.md): running log of methodological choices
- [Experiment log](docs/experiment_log.md): dated entries (template)
- [Related work](docs/related_work.md): RARE, routing drift, expert specialization, polysemantic experts, routing interventions, EMO; unread candidates listed

## Conventions

Python 3.11+, `src/` layout, package name `rr`. All dependencies are managed
with uv.
