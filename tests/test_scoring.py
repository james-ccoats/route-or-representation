"""Tests for rr.scoring: prompt formatting and option log-prob scoring."""

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from conftest import WhitespaceTokenizer
from rr.data import MCQuestion
from rr.scoring import format_mc_prompt, score_mc

TINY_VOCAB_SIZE = 4096  # must exceed the largest WhitespaceTokenizer id
PROMPT = "Question: What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:"


@pytest.fixture(scope="module")
def tiny_lm() -> GPT2LMHeadModel:
    torch.manual_seed(0)
    config = GPT2Config(vocab_size=TINY_VOCAB_SIZE, n_positions=64, n_embd=16, n_layer=1, n_head=2)
    return GPT2LMHeadModel(config).eval()


def test_format_mc_prompt_ends_with_answer_cue() -> None:
    question = MCQuestion("toy", "What is 2 + 2?", ("3", "4", "5", "6"), 1)

    block = format_mc_prompt(question)

    assert block == PROMPT


def test_format_mc_prompt_with_answer_appends_correct_label() -> None:
    question = MCQuestion("toy", "What is 2 + 2?", ("3", "4", "5", "6"), 1)

    assert format_mc_prompt(question, include_answer=True).endswith("\nAnswer: B")


def test_format_mc_prompt_rejects_label_count_mismatch() -> None:
    question = MCQuestion("toy", "Q?", ("x", "y"), 0)

    with pytest.raises(ValueError, match="labels"):
        format_mc_prompt(question)


def test_score_mc_prediction_is_argmax_of_option_logprobs(tiny_lm) -> None:
    score = score_mc(tiny_lm, WhitespaceTokenizer(), PROMPT, correct_index=1)

    assert score.prediction == max(range(4), key=lambda i: score.option_logprobs[i])


def test_score_mc_reports_correctness_and_correct_logprob(tiny_lm) -> None:
    score = score_mc(tiny_lm, WhitespaceTokenizer(), PROMPT, correct_index=2)

    assert score.correct_logprob == score.option_logprobs[2]
    assert score.is_correct == (score.prediction == 2)


def test_score_mc_logprobs_are_full_vocabulary_log_softmax_values(tiny_lm) -> None:
    tokenizer = WhitespaceTokenizer()
    input_ids = torch.tensor([tokenizer(PROMPT)["input_ids"]])
    with torch.inference_mode():
        expected = tiny_lm(input_ids=input_ids).logits[0, -1].log_softmax(-1)

    score = score_mc(tiny_lm, tokenizer, PROMPT, correct_index=0)

    label_ids = [
        tokenizer(f" {label}", add_special_tokens=False)["input_ids"][0] for label in "ABCD"
    ]
    assert score.option_logprobs == pytest.approx([expected[i].item() for i in label_ids], abs=1e-5)
    assert all(lp < 0 for lp in score.option_logprobs)


def test_score_mc_is_deterministic(tiny_lm) -> None:
    first = score_mc(tiny_lm, WhitespaceTokenizer(), PROMPT, correct_index=0)
    second = score_mc(tiny_lm, WhitespaceTokenizer(), PROMPT, correct_index=0)

    assert first == second


def test_score_mc_leaves_the_model_in_its_original_mode(tiny_lm) -> None:
    score_mc(tiny_lm, WhitespaceTokenizer(), PROMPT, correct_index=0)

    assert tiny_lm.training is False


def test_score_mc_rejects_out_of_range_correct_index(tiny_lm) -> None:
    with pytest.raises(ValueError, match="correct_index"):
        score_mc(tiny_lm, WhitespaceTokenizer(), PROMPT, correct_index=4)


def test_score_mc_rejects_labels_that_are_not_single_tokens(tiny_lm) -> None:
    with pytest.raises(ValueError, match="single token"):
        score_mc(
            tiny_lm,
            WhitespaceTokenizer(),
            PROMPT,
            correct_index=0,
            option_labels=("A B", "C", "D", "E"),
        )
