# Design decisions

Running log of methodological choices. Add new entries at the bottom; when a
decision is resolved, change its status, state the choice and the reason, and
date it. Do not delete the options that were rejected.

---

## D1. Force only expert selection, or selection plus mixing weights?

**Status:** open

**Options:**
- A. Force only *which* experts are selected; mixing weights come from the
  router's own probabilities over the forced experts.
- B. Force both selection and mixing weights (full routing replay).

**Considerations:** B transplants the complete routing decision and is the
cleaner "routing" intervention. A leaves part of the routing signal (weights)
tied to the current hidden state. Possibly report both. Depends on whether
the model renormalizes top-k probabilities (verify in Phase 0).

---

## D2. Where to force routing

**Status:** open (proposed default below)

**Options:**
- A. Only on target-question tokens; example tokens route naturally.
  **Proposed default.**
- B. On all tokens of the prompt.
- C. Only on the final (answer-position) token.

**Considerations:** A isolates the effect of routing on the tokens being
answered while letting the examples do whatever they do. B conflates example
processing with the question. C may be too narrow since earlier question tokens
feed later ones through attention.

---

## D3. Metric

**Status:** open (proposed default below)

**Proposed:** report both accuracy and log-prob of the correct answer.

**Considerations:** accuracy is coarse and noisy at small question counts;
log-prob is smoother but harder to interpret. Gain fractions can be computed on
either; report both and flag disagreement.

---

## D4. Number of shots and questions per target domain

**Status:** open

**Options:**
- Shots: 1, 3, 5, or more.
- Questions per target domain: ~100 (pilot) up to all available (main).

**Considerations:** MMLU `dev` has only 5 examples per subject, so more than 5
shots requires drawing from another split and excluding them from eval. More
shots increase the gain (a better signal) and prompt length. Question count
sets the width of the confidence interval on the gain ratio; choose it with a
power estimate from the pilot.

---

## D5. What counts as "irrelevant" examples

**Status:** open

**Options:** MC questions from unrelated subjects (e.g. history); non-MC text of
matched token length; shuffled-answer or shuffled-text versions of source
examples.

**Considerations:** the control should match length and format while carrying
no domain signal. MC-format unrelated questions control for format-learning;
non-MC text controls only for length.

**Resolved for Phase 1 (2026-11-06):** the control matches BOTH the number of
examples (k, same as the cross-domain shots) and the rendered token length of the
cross-domain example block. Reason: matching only tokens let k drift (e.g. 2
control examples vs 5 relevant ones), and matching only k lets length and the
absolute position of the target tokens differ, which could change routing for
reasons unrelated to relevance (H2). Rejected: token-only matching; k-only
matching; running both as separate conditions (kept as an option if the choice
turns out to matter).

Implementation: MC-format questions from unrelated subjects, drawn from the
`test` split (548 questions for history + philosophy; these subjects are never
eval targets, enforced in `load_pairs_config`). `select_irrelevant_examples`
picks exactly k questions within 16 tokens (either direction) of the budget,
using random starts plus swap improvement; it raises if the pool cannot do so.
On the pilot configs all six directions matched exactly (0-token gap). The
achieved `example_tokens` is saved with every prompt so the match can be audited.
The earlier `dev`-split pool (10 questions) is too small to match both k and length.

---

## D6. How to score options

**Status:** open

**Options:** compare log-probs of the letter tokens (A/B/C/D) at the answer
position (one forward pass), or score full option text (length normalization
needed).

**Considerations:** letter scoring is cheap and standard for MMLU but sensitive
to label bias; the zero-shot prompt may have a strong letter prior that
few-shot examples change, which is itself a representational effect.

---

## D7. How to implement forcing

**Status:** open

**Options:** override router logits so the desired experts win top-k; or patch
the top-k selection inside the MoE block.

**Considerations:** the logit approach is simple but couples selection and
weights (affects D1). Decide after inspecting the installed `transformers`
MoE implementation in Phase 0.

---

## D8. Pinning the dataset and recording provenance

**Status:** open

**Context:** `load_mmlu_subject` accepts a `revision` and passes it to
`load_dataset`, but no config sets it yet, and `model.revision` is still `null`.
Sampling indexes into dataset order, so a Hub-side change to `cais/mmlu` would
silently change which questions are sampled.

**Options:**
- A. Add `dataset_revision` to the pairs config and pin it after first download.
- B. Also write a provenance file next to each prompt JSONL (seed, dataset and
  tokenizer revisions, counts) so a prompt set can be traced and reproduced.

**Considerations:** both are cheap. B is what a reviewer would ask for when
checking that two prompt sets are comparable.

---

## D9. Shot sampling and seeds

**Status:** open

**Context:** the pilot uses `shot_split: dev` (5 questions per subject) with
`num_shots: 5`, so `sample_questions` selects every dev question and the seed only
permutes their order. Shot selection is therefore not random at these settings.
Separately, one `seed` is reused for the source shots, target shots and eval
sample, which correlates those permutations.

**Options:**
- A. Accept it for the pilot: with 5 shots from dev, the shot *set* is fixed and
  only order varies. Report that shot order is the only sampled factor.
- B. Draw shots from a larger split (and exclude them from eval) so the seed
  matters, or use fewer shots than the pool size.
- C. Derive per-purpose seeds (e.g. `default_rng([seed, purpose_id])`).

**Considerations:** `load_pairs_config` now requires `shot_split != eval_split`,
so shot questions can never be evaluated. If conclusions should not depend on one
particular shot set, B is needed; consider repeating the pilot over several shot
draws.
