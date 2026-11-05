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
