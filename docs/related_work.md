# Related work

Notes on the closest prior work. Fill in details after reading each paper in
full; the summaries below come from the project brief and are provisional.

## RARE

<a id="rare"></a>
**Link:** https://arxiv.org/abs/2608.21236

**Idea (as understood):** steers MoE models by changing hidden states in the
*null space of the router*, so the router's output (and hence expert routing)
is unchanged while model behavior still changes.

**Relevance:** it is existence proof that representation-only changes, with
routing held fixed, can alter behavior. This motivates our hypothesis that part
of few-shot transfer is representational. The null-space construction is also a
possible alternative to our forced-routing intervention: instead of replaying
routing, change hidden states without touching routing.

**Open questions:** does the intervention work for natural prompt-induced
shifts, or only for constructed steering vectors? Which layers does it target?

## Routing Drift Alone Does Not Diagnose Failure in Merged MoE LLMs

**Link:** https://arxiv.org/abs/2609.32821

**Read status:** sections 1 to 7 read in full via a subagent's verbatim report;
appendices A.2 onward came only as summaries and the last ~16.5k characters were
not read. Re-check anything below marked [summary] against the paper.

**Idea:** studies routing drift in merged MoE LLMs (DeepSeekMoE-16B, OLMoE-7B,
Qwen3-30B; Average, Task Arithmetic, TIES and WUDI merges). Attribution swaps
router parameters and router inputs and counts expert-set changes, without running
the model. The task-level intervention is source-route replay: restore
source-derived expert selections, with native routing weights, into the merged
model's forward pass, parameters fixed. Conclusion: routing drift alone is
insufficient evidence of routing failure; most expert reassignments are
attributed to input shifts, not parameter changes at the same layer.

**Relevance:** closest methodological precedent for our forcing experiments, and
it includes OLMoE. Their replay pairs source routing with merged hidden states,
a mismatch like ours. On OLMoE science, source-route replay changed accuracy by
+1.17 pp [-1.56, +3.91] (N=256); on ARC by -0.39 to +0.39 pp; no displayed
interval is strictly positive. This is some evidence that mismatched replay is
not destructive when the mismatch is small. It is not a claim of harmlessness:
the authors write that an interval containing zero "establishes neither
harmlessness nor optimality".

**Differences from our design (answers our earlier open question):**
- Weights: they restore selections with native routing weights (ambiguous whether
  the source run's or recomputed), closest to our selection-only variant (D1
  option A). Our headline is full replay with forced weights.
- Layers: task-level replay and their router repair use the final five sparse
  layers; we replay all 16.
- Tokens: question-choice tokens, as in ours.
- Damage controls: a random-route control appears only for mixture-cosine
  comparisons, and no random-route control was found in the task-level replays.
  Identity or round-trip checks are described in appendix summaries only
  [summary]. Our no-benefit anchors go beyond this.

**Evidence on damage from large corruption:** permuting expert logits in 4 or all
16 OLMoE layers drops accuracy sharply, and replaying clean routes recovers about
12 to 33 pp (k=1 layer gives no consistent recovery). This supports using a
random-expert extreme as the damage ceiling.

**Code:** the abstract links https://github.com/wyy-code/SRR; the reproducibility
statement links a different anonymous mirror. Neither was visited. Reported scope:
crossed-intervention attribution, token-local and task-level route replay,
controlled corruption. Check it for reusable replay and corruption code before
extending ours.

**Open questions:** does "native routing weights" mean the source run's weights?
Which link is current? Does the appendix describe a same-run replay or identity
control with numbers?

## The Myth of Expert Specialization in MoEs

**Link:** https://arxiv.org/abs/2604.09780

**Idea (from the abstract only; full text not yet read):** MoE routers are linear
maps, so hidden-state similarity is necessary and sufficient to explain
expert-usage similarity. Specialization is argued to be an emergent property of
the representation space, not of the routing architecture. Reported across five
pre-trained models (not named in the abstract): expert overlap between models
answering the same question is about 60%, no higher than between different
questions; deeper layers show near-identical expert activation across unrelated
inputs.

**Relevance:** the strongest published support for the representational side.
It motivates (does not establish) our expectation in H3 that hidden states carry
much of the gain, and it is consistent with the interaction point that routing is
itself computed from the shifted hidden states.

**Limits:** the result is about similarity of expert usage, not output or
accuracy, so it does not show routing is causally unimportant. As far as the
abstract says it is observational, with no forcing or replay and no few-shot
setting. It also strengthens the out-of-distribution concern for replay:
routing and hidden state are normally consistent.

**Open questions:** which five models, and are any intervention experiments in
the full text?

## Polysemantic Experts, Monosemantic Paths: Routing as Control in MoEs

**Link:** https://arxiv.org/abs/2604.17837 (Ye, Yuan, Sharkey; v1, 20 Apr 2026)

**Read status:** abstract and sections 1 to 5 read in full via a subagent; the rest of
appendix B and appendices C to F came as summaries; appendices G and H unread.

**Idea:** splits each layer's hidden state into a router-visible part (projection onto
the row space of the router matrix, via SVD) and an orthogonal router-blind part.
By construction only the visible part can change expert selection. Probes on cached
activations; six MoEs including OLMoE (model list from an appendix summary).

**Relevance:** the same row-space idea behind our hidden-state shift measure (design
decision D13). It reports no measurement of how much of a prompt-induced shift lies in
the row space compared with chance, and no replay or forcing, so D13's use looks distinct
(from the read parts; appendices G and H unread). It supports the interaction point:
routing is computed from the visible component of the hidden state.

**Limits for us:** observational (probes, cosine similarity, clustering); about 2M
tokens from C4 and HPLT; no downstream tasks; no in-context or few-shot prompts; no
interventions. The authors list perturbing control as future work.

## What MoE Routing Controls: Causal Interventions on Expert Trajectories

**Link:** https://neurips.cc/virtual/2026/loc/sydney/165994 (abstract only; OpenReview
forum mcR9et16wu not fetched). NeurIPS 2026 interpretability workshop poster
(Athmaram, Rao, Ma).

**Read status:** abstract only.

**Idea (from the abstract):** adds a sustained bias to one expert's router logit at each
layer during generation, targeting the modal top-1 expert for a concept or role. Reports
that it biases output toward the concept, with limited reach (changes generated text for
some targets, only next-token distributions for others). Controls for generic corruption,
injected vocabulary and gate-weight inflation.

**Relevance:** a causal routing intervention, so it weakens any claim that intervening on
routing is new. It biases logits rather than replaying recorded routing, names only
Qwen3-30B-A3B, and mentions no few-shot setting. Whether it has random-expert or identity
controls is unclear from the abstract.

## EMO: Pretraining Mixture of Experts for Emergent Modularity

**Link:** https://arxiv.org/abs/2605.06663 (Wang, Bhagia, Min; v2, May 2026)

**Read status:** about 40k of 117k characters of the HTML read verbatim (abstract,
sections 1 to 4, start of 5.2); the rest, including appendices, came as a tool summary
only. The PDF fetch returned nothing usable.

**Idea:** a pretraining method that makes MoEs modular: tokens in a document route within
a shared expert pool. Per domain, a fixed expert subset is chosen by average routing
probability over validation tokens, and routing is restricted to it.

**Relevance:** low overlap. "even a single few-shot example is sufficient to select an
effective expert subset" (section 4.2) concerns using few-shot prompts as data for choosing
a static subset, not in-context examples changing routing inside the model. No comparison
of zero-shot and few-shot routing, no forcing or replay of routing between prompts, no
separation of routing from hidden-state effects, and no interventions on OLMoE itself
(the released OLMoE is a reference row only). A random-expert-subset sanity check
appears (summary only). Unread parts could still contain something; a chunked full read
would confirm.

## Candidates not yet read

Found by search, unread:
- Awakening Dormant Experts: Counterfactual Routing, https://arxiv.org/pdf/2604.14246
- Multilingual Routing in Mixture-of-Experts, https://arxiv.org/html/2510.04694
  (intercepts router logits to boost or suppress experts)
- RARE (above): only the abstract has been seen for its router-visible-removal wording.

Search status: three web searches found no paper that applies routing replay to
in-context learning or compares OLMoE routing under zero-shot and few-shot prompts.
That is not a literature review; keep the novelty claim as "to our knowledge".

## How this project differs

The first two papers intervene on routing or hidden states for a different purpose
(steering; diagnosing merges), and the intervention paper above steers generation by
biasing router logits. We apply a routing-replay intervention to
decompose the effect of in-context examples on accuracy.
