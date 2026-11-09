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

**Implemented (2026-11-08), still open as a research choice:** both are available
through `force_mixing_weights` in the experiment config (`false` = A, the default;
`true` = B). With A, the weights of the forced experts are the router's own softmax
probabilities for them, renormalized only if the router renormalizes. In the installed
`transformers` (5.22.1) `OlmoeTopKRouter.norm_topk_prob` follows the model config
(`false` by default for OLMoE); read it off the real model with `inspect_moe_structure`
rather than assuming. Consequence for the random control: with A and no renormalization,
forcing random experts also shrinks the MoE output (their router probabilities are
small), so it is a harsher intervention than "random experts with typical weights".
Decide which of A/B is the headline before the main run, and report the other as a check.

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

**Implemented (2026-11-08):** only A. `parse_forcing_options` rejects any
`force_scope` other than `target_question_tokens`. Target positions come from
`rr.routing.align` (offset mappings), and alignment raises if a token straddles the
example/target boundary or the target token ids differ between prompts. B and C are
not implemented; add them only if A's result needs a robustness check.

---

## D3. Metric

**Status:** open (proposed default below)

**Proposed:** report both accuracy and log-prob of the correct answer.

**Considerations:** accuracy is coarse and noisy at small question counts;
log-prob is smoother but harder to interpret. Gain fractions can be computed on
either; report both and flag disagreement.

**Implemented (2026-11-08):** `rr.force_analysis` and `rr.analysis` report both
metrics side by side (`accuracy`, `correct_logprob`). Nothing yet flags disagreement
automatically; compare the two columns by eye on the pilot before deciding which is
primary.

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

**Implemented as the default (2026-11-08), not yet verified on the real tokenizer:**
letter scoring in one forward pass. Each label is encoded with a leading space
(`" A"`), matching how in-context examples render `Answer: B`, and `score_mc` raises
if a label is not a single token. Option log-probs come from the full-vocabulary
log-softmax at the answer position and are *not* renormalized over A-D. Still open:
whether to renormalize over the four options, and a label-bias check (e.g. permuting
option order). Verify `tokenizer(" A", add_special_tokens=False)` on the OLMoE
tokenizer when it is first loaded.

---

## D7. How to implement forcing

**Status:** resolved (2026-11-08)

**Options:** override router logits so the desired experts win top-k; or patch
the top-k selection inside the MoE block.

**Considerations:** the logit approach is simple but couples selection and
weights (affects D1). Decide after inspecting the installed `transformers`
MoE implementation in Phase 0.

**Choice:** neither as written. In the installed `transformers` (5.22.1) each layer's
router is an `OlmoeTopKRouter` whose forward returns
`(router_logits, router_scores, router_indices)`, and `OlmoeSparseMoeBlock` consumes
the scores and indices. `RoutingForcer` is a forward hook on that router that replaces
`router_scores` and `router_indices` at the chosen token rows and leaves the logits
untouched; no model code is patched. Reason: it keeps selection and weights separate,
which is what makes D1 a free choice, and it needs no modified `transformers` source.
Rejected: the logit override (couples selection and weights, and cannot force an
arbitrary expert set exactly) and patching the block (invasive, breaks on upgrades).

**Verified on a tiny random-weight `OlmoeForCausalLM` (no downloads):** recorded top-k
matches `output_router_logits`; forcing a run's own routing back onto itself leaves
logits unchanged (with and without forced weights, with and without
`norm_topk_prob`); hooks are removed on errors; batch size > 1 is rejected.

**Caveats:** this relies on the router module returning that tuple. Re-run the test
suite whenever the `transformers` version changes, and expect a second model family
(Phase 4) to need its own adapter. Not yet exercised on the real 7B model, whose dtype
(bf16) and device placement could change the identity-check numerics.

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

**Extended by the 2026-11-08 code review:** the same gap existed for *results*.
Resumed runs (D11) checked the run name and the question count, but not the settings
that produced the existing records (`force_mixing_weights`, `layers`, random seed,
planned interventions, model name/revision/dtype). Resuming after editing the config
would silently mix records produced under different settings in one file, because
the intervention label does not encode them.

**Results half implemented (2026-11-08), option C:** each results file gets a
`<run>.manifest.json` next to it (`rr.manifest`) holding the run name, the SHA-256 of
the prompt file, and the settings above (model name, revision and dtype; forcing
plans and options). It is written before the first question is evaluated and compared
on every resume; any difference is refused, with each differing key listed. Results
that exist without a manifest are also refused. The device is deliberately not
recorded, so resuming on a different GPU is allowed. Still open here: A and B for the
*prompt* sets, and pinning `model.revision` (still `null`), which the manifest records
but cannot invent.

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

---

## D10. Uncertainty for the gain fractions

**Status:** implemented (2026-11-08), open to revision after the pilot

**Context:** the headline quantities are ratios of aggregate scores,
`(forced - Z) / (F - Z)`. Ratios are unstable when `F - Z` is small, and an interval
that ignores that would be misleading.

**Choice:** a percentile bootstrap over *questions*, resampling the same questions
jointly for Z, F and the forced score (they come from the same questions, so they are
paired). 2000 resamples, 95% interval, seeded. A resample whose few-shot gain is ~0
has no defined ratio and is dropped; `undefined_fraction` is reported next to the
interval, and a fraction whose point gain is ~0 is reported as undefined with no
interval. Paired differences for the pilot use the same bootstrap.

**Limits to keep in mind:**
- The interval covers question sampling only. With `shot_split: dev` and 5 shots the
  shot set is fixed (D9), so shot-choice variance is not in the interval.
- No multiple-comparison correction across pairs, directions, conditions or metrics.
- A large `undefined_fraction` means the fraction should not be over-read.

**Rejected:** averaging per-question ratios (unstable; `rr.analysis` states this
explicitly) and a delta-method interval (poor when the denominator can be near zero).

---

## D11. Batch size and resumable runs

**Status:** implemented (2026-11-08); the resume-settings gap is tracked in D8

**Choice:** all routing hooks assume batch size 1. Routers see flattened
`[batch * tokens, ...]` rows, so token positions are only meaningful for a single
sequence; `RoutingRecorder` and `RoutingForcer` raise `ValueError` for any larger
batch rather than record or force the wrong rows. Throughput is therefore one prompt
per forward pass; revisit only if the main run is too slow.

Experiments append results one question at a time to `<output_dir>/<run>.jsonl`
(one record per question x condition x intervention). On restart a torn final line is
dropped, a half-finished question is re-evaluated, and only its missing records are
appended. Scores are deterministic (no sampling), so re-evaluation reproduces the
lost values. The runner refuses to resume if the settings or prompt file changed since
the run started (the manifest, D8), if the file holds another run's records, repeats a
record, or has more questions than the prompt file. It also refuses an output path equal
to the prompt file, which would append results into the prompts.

Config validation rejects duplicate entries in `conditions`; they would plan the same
intervention twice and write duplicate records.

**Rejected:** one file per question (too many files on a cluster filesystem) and
atomic whole-run output (loses everything on a preemption).

---

## D12. Which domain pairs, and the distance ladder

**Status:** pilot pairs unchanged; ladder added (2026-11-09), open to revision after the pilot

**Context:** the original three pairs (biology/medicine near, chemistry/physics medium,
computer science/biology far) came from the initial scaffold with no recorded
rationale beyond "science subjects at varying distance" for H5. Distance was an
assumption. Three disjoint pairs also have different target questions, so differences
between pairs mix distance with everything else about the target.

**Measured distance (`rr.distance`, `scripts/06_measure_distance.py`):** each subject is
one document of question and answer-choice text from the `test` split; words are
TF-IDF weighted with IDF over all 57 subjects (a word in every subject has no weight);
distance = 1 - cosine similarity. It is model-free, deterministic and cheap. It is
lexical: it sees shared terminology, not shared skills or routing.

What it showed on 2026-11-09:
- **Near is resolved, far is compressed.** From `college_biology`, `high_school_biology`
  is at 0.685 and `college_medicine` at 0.835, but nearly every other subject lies in
  0.89-0.99. `college_chemistry` is 0.935 and `college_computer_science` 0.966, only
  0.03 apart, and `philosophy` and `high_school_world_history` (the irrelevant pool) are
  at 0.969 and 0.953, about as far as computer science.
- **The pilot's near/medium labels are not separated.** `bio_med` measures 0.835 and
  `chem_phys` 0.827, a 0.008 difference. Only `cs_bio` (0.966) is clearly farther. Do
  not read a near-versus-medium difference in the pilot as a distance effect.

**Choice:** a fixed-target ladder, `configs/pairs/mmlu_ladder.yaml` with
`configs/experiments/ladder_pilot.yaml`. One target (`college_biology`, all 144 test
questions, identical across rungs because subject, seed and pool are the same) and five
sources, one direction each: `high_school_biology` (same_field, 0.685), `college_medicine`
(near, 0.835), `virology` (near, 0.897), `college_chemistry` (medium, 0.935),
`college_computer_science` (far, 0.966). Rungs were picked from the measurement to
spread across its range; the labels remain hypotheses from domain knowledge, and the
script flags any label ordering that the measurement contradicts (none for the ladder).
`same_field` was added to the allowed labels for the high-school-to-college rung, which
separates "same domain" from "same difficulty and style".

All five rungs build with k = 5 and an exact 0-token gap between the irrelevant control
and the cross-domain examples. Cross-domain example tokens are 274 (virology), 393
(medicine), 415 (high school biology), 442 (chemistry) and 714 (computer science).

**Known limits:**
- **Length is partly confounded with distance.** The longest prompt is the farthest rung
  (rank correlation of example length with distance across the rungs is about 0.6).
  The length-matched irrelevant control is the within-rung guard; compare
  cross-domain minus irrelevant, not cross-domain alone, across rungs.
- **Five rungs, one target.** It tests distance within biology as a target; it does not
  show the pattern generalizes to other targets. The original pairs and a second ladder
  on another target would be the replication.
- **Lexical, not representational.** A routing- or hidden-state-based distance needs the
  model; if it disagrees with the lexical ordering, prefer it and re-pick the rungs.
- **Rung choice used the data.** Choosing sources by measured distance is fine for
  spreading them, but do not then report the measurement as independent confirmation of
  the labels.
- **Shots are fixed.** Each source contributes its 5 `dev` questions (D9), so a rung is
  one shot set; shot-choice variance is not in the intervals (D10).

**Rejected:** picking sources by intuition alone (the labels did not separate in the
pilot); deriving distance from the zero-shot accuracy matrix (needs the model and
conflates distance with difficulty).

---

## D13. Measuring the hidden-state shift directly

**Status:** implemented (2026-11-09) as an opt-in pilot measurement; not yet run on the
real model

**Context:** the forcing experiment holds target-token *routing* fixed and asks how much
of the few-shot gain survives. That infers a hidden-state ("representation") effect as
the residual; nothing measured it. A high retained fraction means the gain does not need
the routing change, which is weaker than "the hidden states carry it".

**What is measured (`rr.hidden`):** at the aligned target-question tokens, the state each
router receives in a few-shot prompt versus the zero-shot prompt. In OLMoE that state is
`post_attention_layernorm(residual after attention)`, the RMS-normalised residual stream,
which is exactly what the router multiplies by its weights (`logits = W h`). Recorded
inputs were checked against the model's own router logits on the tiny model. Per layer:
- `relative_l2`: mean over tokens of `|delta| / |h0|`;
- `cosine_distance`: mean over tokens of `1 - cos(h0, h1)`;
- `router_visible_share`: the fraction of the shift's squared norm in the row space of
  `W` (directions that can change the routing logits), pooled over tokens; the rest is
  in the null space of `W` and cannot change which experts are chosen;
- `router_visible_chance`: `rank(W) / hidden_size`, the share a random direction would have.

Only these summaries are stored (in `ResultRecord.hidden_shift`), never raw vectors.
Enable with `record_hidden_states: true` in the pilot configs (set for both); the setting
is recorded in the run manifest (D8), so toggling it on resume is refused. It adds no
forward passes. Memory is transient: `layers * tokens * hidden * 4` bytes per pass, about
105 MB for an 800-token OLMoE prompt.

**How to read it:**
- **Compare the visible share with chance, not with zero.** `W` has at most one row per
  expert (64) against about 2,048 hidden dimensions, so a random shift puts only about 3%
  of its energy in the row space. A small visible share is the default.
- **Compare with the irrelevant control.** Any extra context shifts the state. The pilot
  contrasts report `hidden_shift` and `visible_share_excess` for cross-domain minus
  irrelevant (and same-domain minus irrelevant for the first), which is the length-matched
  baseline.
- **It is descriptive, not causal.** A large shift does not show it caused the gain. The
  causal evidence still has to come from interventions.

**Limits:**
- The zero-shot state is a reference, not a counterfactual: the few-shot prompt differs at
  every earlier token, so the states differ through attention by construction.
- Target tokens only, one forward pass each, so the downstream effect of forced routing on
  later layers is not captured here.
- The layer average in the summary table hides structure; per-layer values are in the
  records for Phase 4.
- Multiple comparisons are not corrected (D10).

**Not done:** the matching causal test, which applies only the router-invisible part of
the shift with routing held fixed (the complement of what forcing does). Build it only if
the pilot shows a shift worth testing. A routing-based distance between subjects
(D12) would reuse the same recorded router inputs.
