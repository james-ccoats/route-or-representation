"""Multiple-choice scoring by answer-option log-probabilities.

Role in the project: evaluates accuracy without text generation. Given a
prompt, compare the model's log-probs for each answer option and take the
argmax. Used in Phase 0 (smoke test) and every phase after.

Methodological notes:

* No sampling or ``generate()``: the prediction is a deterministic argmax.
* Besides accuracy, keep the log-prob of the correct option; it is a smoother
  signal than 0/1 accuracy (see docs/design_decisions.md).
* Whether options are scored as single letter tokens (one forward pass) or as
  full option text is an open decision; the default sketch is letter tokens.
* When routing is recorded or forced (:mod:`rr.routing`), scoring must run
  inside the context manager, on exactly the same token ids as the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from rr.data import MCQuestion

DEFAULT_OPTION_LABELS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class MCScore:
    """Result of scoring one multiple-choice prompt.

    Attributes:
        option_logprobs: Log-prob of each option, in option order.
        prediction: Index of the highest-scoring option.
        is_correct: Whether ``prediction`` equals the correct index.
        correct_logprob: Log-prob assigned to the correct option.
    """

    option_logprobs: tuple[float, ...]
    prediction: int
    is_correct: bool
    correct_logprob: float


def format_mc_prompt(
    question: MCQuestion,
    *,
    include_answer: bool = False,
    option_labels: tuple[str, ...] = DEFAULT_OPTION_LABELS,
) -> str:
    """Render one question block.

    Args:
        question: The question to render.
        include_answer: If True, append the correct label (used for in-context
            examples); if False, end with the ``Answer:`` cue (target question).
        option_labels: Labels for the options, e.g. ``("A", "B", "C", "D")``.

    Returns:
        The rendered block as a string.

    Methodological note:
        Target blocks must be byte-identical across conditions so token
        alignment (:mod:`rr.routing.align`) is exact.
    """
    if len(option_labels) != len(question.choices):
        raise ValueError(
            f"got {len(option_labels)} option labels for {len(question.choices)} choices"
        )
    option_lines = [
        f"{label}. {choice}" for label, choice in zip(option_labels, question.choices, strict=True)
    ]
    block = "\n".join([f"Question: {question.question}", *option_lines, "Answer:"])
    if include_answer:
        return f"{block} {option_labels[question.answer_index]}"
    return block


def score_mc(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    correct_index: int,
    option_labels: tuple[str, ...] = DEFAULT_OPTION_LABELS,
) -> MCScore:
    """Score one prompt by comparing option log-probs at the answer position.

    Args:
        model: Loaded causal LM.
        tokenizer: Matching tokenizer.
        prompt: Full prompt ending with the ``Answer:`` cue.
        correct_index: Index of the correct option.
        option_labels: Option labels whose continuation tokens are compared.

    Returns:
        An :class:`MCScore`.

    Methodological note:
        Check how the tokenizer encodes " A" vs "A" after "Answer:"; compare
        log-probs over the option tokens only (not renormalized over the
        vocabulary unless decided otherwise).
    """
    # TODO(Phase 0): one forward pass under torch.inference_mode(), take logits
    # at the last position, log_softmax, gather option-label token ids, argmax.
    raise NotImplementedError("TODO(Phase 0): option log-prob scoring")
