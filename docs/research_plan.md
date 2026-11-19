# Research plan

## Question

In mixture-of-experts (MoE) language models, when few-shot examples from a
question's own subject improve accuracy, how much of the gain comes from

- **routing**: the router selecting different experts for the target-question
  tokens, versus
- **representation**: the same experts processing shifted hidden states?

## Setup

- **Model:** `allenai/OLMoE-1B-7B-0924` first; a second MoE model in Phase 4.
- **Task:** MMLU multiple choice, scored by comparing answer-option log-probs
  (no text generation).
- **Subjects:** four MMLU science subjects (college biology, chemistry, physics,
  computer science). Each subject is both example source and target; examples
  exclude the question being scored. Cross-domain examples (source subject
  differs from target) are a Phase 4 extension, not the primary condition.
- **Shots:** k = 1, 2, 3, 4, 5 (nested), repeated over 3 independently drawn example sets
  (replicates), with examples pooled from the `dev` and `validation` splits (design_decisions.md D14).
- **Prompt conditions per target question** (at each k and replicate):
  1. zero-shot
  2. few-shot with examples from the question's own subject (primary)
  3. irrelevant examples from another subject, matched on k and token length (control)

## Hypotheses

- **H1 (few-shot helps):** For at least some subjects, few-shot accuracy with
  same-subject examples exceeds zero-shot.
- **H2a (routing shifts):** Few-shot examples shift routing on target-question
  tokens beyond noise and beyond a length-matched irrelevant prompt (lower
  per-layer Jaccard of top-k expert sets vs. zero-shot). Size of shift only, not
  direction.
- **H2b (the shift is content-driven):** Routing under one example set of a
  subject is closer to routing under another example set of the same subject
  than zero-shot and irrelevant-example routing are.
- **H3 (representation carries much of the gain):** Forcing zero-shot routing
  on the target-question tokens of a few-shot prompt (routing-removed) keeps a
  large fraction of the gain. Motivated by RARE, which shows that hidden-state changes with
  routing held fixed can steer behavior.
- **H4 (routing alone is insufficient):** Forcing few-shot routing into a
  zero-shot prompt (routing-added) adds only a small fraction of the gain.
H3 and H4 are the main claims; H1, H2a and H2b are preconditions checked in the
pilot. Whether the routing share depends on domain distance or transfer direction
(cross-domain examples) is an exploratory Phase 4 extension with no hypothesis.

## Metrics

- **Accuracy** and **log-prob of the correct answer** (smoother, more
  sensitive). See design_decisions.md.
- **Routing shift:** per-layer Jaccard overlap of top-k expert sets on aligned
  target-question tokens.
- **Hidden-state shift (supporting, pilot):** per-layer size of the change in the
  state each router receives on those tokens versus zero-shot, and the share of that
  change in the router's row space against its chance level (design_decisions.md D13).
  Descriptive only; the causal evidence is the forcing experiment.
- **Gain against k:** the metrics above at each k, averaged over replicates, with the
  range across replicates shown separately from the question-level interval (D14).
- **Gain kept** (routing-removed; `retained` in the code) = (forced − zero_shot) / (few_shot − zero_shot),
  where *forced* is the few-shot prompt with zero-shot routing forced.
- **Gain added** (routing-added; `recovered` in the code) = (forced − zero_shot) / (few_shot − zero_shot),
  where *forced* is the zero-shot prompt with few-shot routing forced in.

## Phases

### Phase 0: Setup
Load the model; verify layer count, expert count, top-k and router module names
by inspection (not assumption); score MC questions by option log-probs; record
per-token top-k routing with forward hooks; confirm hooks leave outputs
unchanged and are removed on exit.
**Done when:** the smoke test runs and recorded routing has shape
`[num_layers, num_tokens, top_k]` and matches what the model actually uses.

### Phase 1: Data
Load MMLU subjects, build the three prompt conditions with an identical
target-question block, and an irrelevant control matched to the few-shot block
on k and token length.
**Done when:** prompts are written to `data/prompts/` and checks confirm
no shot/eval overlap and length matching within tolerance.

### Phase 2: Pilot
**Done when:** (a) same-subject few-shot beats zero-shot for some subjects (H1),
and (b) the examples shift routing beyond the irrelevant control (H2a), in a
content-driven direction (H2b). If (a) fails for all subjects, revisit subjects,
shots or model before proceeding; if (b) fails, the routing story is weak and the
project should be reframed.

### Phase 3: Main experiment
Forced-routing experiment, reverse check, and sanity checks:
- forcing zero-shot routing onto the zero-shot prompt changes nothing;
- forcing random experts hurts accuracy.
**Done when:** both sanity checks pass and gain fractions with confidence
intervals exist for every subject with a meaningful gain.

Irrelevant-prompt forcing (planned): run the same two forced runs on the
length-matched irrelevant prompt, to show how much of the gain, and of its
routing/representation split, is generic context and length rather than content.

Example-token replay (planned, main design): in the few-shot prompt, perturb
the routing of the example tokens (next-best swap, random experts in k of 16
layers, routing from an equal-token-length example set) and measure how much of
the gain survives, read against the damage anchors. The zero-shot prompt has no
example tokens, so there is no zero-shot routing to copy onto them; this tests
whether example-token routing is needed, not whether it is removed.

### Phase 4: Extensions
Layer-by-layer forcing; a second MoE model; cross-domain examples, relating
results to domain distance and transfer direction.

## Success criteria

- Sanity checks pass (identity check exact up to numerics; random forcing hurts).
  As implemented in `rr.force_analysis`: the identity check passes if forcing the
  zero-shot prompt's own routing changes no option log-prob by more than
  `IDENTITY_TOLERANCE` (1e-3, a placeholder to tighten once the real model's bf16
  numerics are known); the random check passes if random forcing lowers accuracy for
  every few-shot condition tested. Neither is a statistical test.
- A subject is only interpreted if its few-shot gain is clearly nonzero
  (the ratio is undefined otherwise). The analysis reports such fractions as
  "undefined" and shows the share of bootstrap resamples with no defined ratio.
- **Not yet pre-registered:** what size of pilot effect counts as H1/H2 holding
  (Phase 2 "done when"), and what counts as "a large fraction" kept or "only a
  small fraction" added (H3/H4), and the cutoffs for graded anchors that "lose much
  less" or "about as much" as routing-removed at the same routing difference. Write
  numeric thresholds here before the main run, so the interpretation is not chosen
  after seeing the results.
- The headline result is reported as gain fractions with uncertainty across
  subjects, whichever direction it points. A finding that routing carries most of
  the gain is as valid as the reverse.

## Threats to validity

- Forcing routing for the target tokens only still lets later layers and
  non-forced positions respond (see design_decisions.md). Example-token replay
  (Phase 3) addresses the example positions.
- Forced routing may be out-of-distribution for the hidden states it is
  applied to; the random-expert control bounds this but does not remove it.
- Small per-subject question counts give wide intervals on a ratio.
- Findings on one model may not transfer; Phase 4 adds a second.
