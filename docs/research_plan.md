# Research plan

## Question

In mixture-of-experts (MoE) language models, when source-domain few-shot
examples improve accuracy on a target domain, how much of the gain comes from

- **routing**: the router selecting different experts for the target-question
  tokens, versus
- **representation**: the same experts processing shifted hidden states?

## Setup

- **Model:** `allenai/OLMoE-1B-7B-0924` first; a second MoE model in Phase 4.
- **Task:** MMLU multiple choice, scored by comparing answer-option log-probs
  (no text generation).
- **Domains:** MMLU science subjects. Three to four pairs at varying distance
  (e.g. biology→medicine, chemistry→physics, computer science→biology), each
  run in both directions.
- **Prompt conditions per target question:**
  1. zero-shot
  2. cross-domain few-shot (source-domain examples)
  3. same-domain few-shot (upper reference)
  4. irrelevant examples of matched token length (control)

## Hypotheses

- **H1 (transfer exists):** For at least some pairs, cross-domain few-shot
  accuracy exceeds zero-shot.
- **H2 (relevance shifts routing):** Relevant examples shift routing on
  target-question tokens more than irrelevant ones (lower per-layer Jaccard of
  top-k expert sets vs. zero-shot).
- **H3 (representation carries much of the gain):** Forcing zero-shot routing
  on the target-question tokens of a few-shot prompt retains a large fraction
  of the gain. Motivated by RARE, which shows that hidden-state changes with
  routing held fixed can steer behavior.
- **H4 (routing alone is insufficient):** Forcing few-shot routing into a
  zero-shot prompt recovers only a small fraction of the gain.
- **H5 (distance and direction):** The routing share of the gain grows with
  domain distance and may differ by transfer direction.

H3 and H4 are the main claims; H1 and H2 are preconditions checked in the pilot.

## Metrics

- **Accuracy** and **log-prob of the correct answer** (smoother, more
  sensitive). See design_decisions.md.
- **Routing shift:** per-layer Jaccard overlap of top-k expert sets on aligned
  target-question tokens.
- **Fraction of gain retained** = (forced − zero_shot) / (few_shot − zero_shot),
  where *forced* is the few-shot prompt with zero-shot routing forced.
- **Fraction of gain recovered** = (forced − zero_shot) / (few_shot − zero_shot),
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
Load MMLU subjects, choose pairs, build the four prompt conditions with an
identical target-question block, token-length-matched irrelevant control.
**Done when:** prompts are written to `data/prompts/` and checks confirm
no shot/eval overlap and length matching within tolerance.

### Phase 2: Pilot
**Done when:** (a) cross-domain few-shot beats zero-shot for some pairs, and
(b) relevant examples shift routing more than irrelevant ones. If (a) fails for
all pairs, revisit pairs, shots or model before proceeding; if (b) fails, the
routing story is weak and the project should be reframed.

### Phase 3: Main experiment
Forced-routing experiment, reverse check, and sanity checks:
- forcing zero-shot routing onto the zero-shot prompt changes nothing;
- forcing random experts hurts accuracy.
**Done when:** both sanity checks pass and gain fractions with confidence
intervals exist for every pair and direction with a meaningful gain.

### Phase 4: Extensions
Layer-by-layer forcing; a second MoE model; relating results to domain distance
and transfer direction.

## Success criteria

- Sanity checks pass (identity check exact up to numerics; random forcing hurts).
- A pair/direction is only interpreted if its few-shot gain is clearly nonzero
  (the ratio is undefined otherwise).
- The headline result is reported as gain fractions with uncertainty across
  pairs, whichever direction it points. A finding that routing carries most of
  the gain is as valid as the reverse.

## Threats to validity

- Forcing routing for the target tokens only still lets later layers and
  non-forced positions respond (see design_decisions.md).
- Forced routing may be out-of-distribution for the hidden states it is
  applied to; the random-expert control bounds this but does not remove it.
- Small per-pair question counts give wide intervals on a ratio.
- Findings on one model may not transfer; Phase 4 adds a second.
