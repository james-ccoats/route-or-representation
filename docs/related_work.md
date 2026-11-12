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

**Idea (as understood):** uses a routing-replay intervention in a
model-merging setting, to test whether routing changes explain failures of
merged MoE models.

**Relevance:** closest methodological precedent for our forcing experiments.
Its conclusion that routing drift alone does not explain failure parallels our
question in a different setting. We should compare how they replay routing
(selection vs. weights, which tokens, which layers) with our D1, D2 and D7.

**Open questions:** what replay variant do they use, and how do they handle
the out-of-distribution concern for replayed routing?

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

## How this project differs

The first two papers intervene on routing or hidden states for a different purpose
(steering; diagnosing merges). We apply a routing-replay intervention to
decompose the effect of in-context examples on domain transfer.
