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

import torch

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
    if not 0 <= correct_index < len(option_labels):
        raise ValueError(
            f"correct_index {correct_index} out of range for {len(option_labels)} option labels"
        )
    label_ids = _option_token_ids(tokenizer, option_labels)
    input_ids = torch.tensor([tokenizer(prompt)["input_ids"]], device=model.device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits[0, -1]
        logprobs = logits.float().log_softmax(dim=-1)[list(label_ids)].tolist()
    prediction = max(range(len(logprobs)), key=logprobs.__getitem__)
    return MCScore(
        option_logprobs=tuple(logprobs),
        prediction=prediction,
        is_correct=prediction == correct_index,
        correct_logprob=logprobs[correct_index],
    )


def _option_token_ids(
    tokenizer: PreTrainedTokenizerBase, option_labels: tuple[str, ...]
) -> tuple[int, ...]:
    """Token id of each label as it follows the ``Answer:`` cue (i.e. with a leading space)."""
    ids: list[int] = []
    for label in option_labels:
        label_ids = tokenizer(f" {label}", add_special_tokens=False)["input_ids"]
        if len(label_ids) != 1:
            raise ValueError(
                f"option label {label!r} must encode to a single token, got {len(label_ids)}"
            )
        ids.append(label_ids[0])
    return tuple(ids)
