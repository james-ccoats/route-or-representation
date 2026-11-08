"""Tests for rr.prompts: rendering, spans, length-matched control, validation, JSONL."""

from __future__ import annotations

import json

import pytest

from conftest import WhitespaceTokenizer, make_question
from rr.prompts import (
    LENGTH_MATCH_TOLERANCE_TOKENS,
    BuiltPrompt,
    PromptCondition,
    build_condition_prompts,
    build_prompt,
    count_example_tokens,
    load_prompts,
    save_prompts,
    select_irrelevant_examples,
    verify_target_tokens,
)
from rr.scoring import format_mc_prompt

ALL_CONDITIONS = tuple(PromptCondition)


def _varied_pool(subject: str, size: int = 40) -> list:
    """Questions whose lengths grow steadily, so many k-subsets can hit a token budget."""
    return [make_question(subject, i, words=2 + i) for i in range(size)]


def _span_text(prompt: BuiltPrompt) -> str:
    start, end = prompt.target_char_span
    return prompt.text[start:end]


def _build_all(make_questions, tokenizer, *, source_words: int = 3, target=None):
    target = target or make_questions("bio", 1)[0]
    source = make_questions("chem", 3, words=source_words)
    irrelevant = select_irrelevant_examples(
        _varied_pool("hist"),
        len(source),
        count_example_tokens(source, tokenizer),
        tokenizer,
        seed=0,
    )
    return build_condition_prompts(
        target,
        source_examples=source,
        same_domain_examples=make_questions("bio", 4)[1:],
        irrelevant_examples=irrelevant,
        conditions=ALL_CONDITIONS,
        tokenizer=tokenizer,
    )


# --- build_prompt -----------------------------------------------------------------


def test_zero_shot_prompt_is_just_the_target_block(make_questions) -> None:
    target = make_questions("bio", 1)[0]

    prompt = build_prompt(PromptCondition.ZERO_SHOT, target, [])

    assert prompt.text == format_mc_prompt(target)
    assert prompt.target_char_span == (0, len(prompt.text))
    assert prompt.num_examples == 0


def test_examples_are_rendered_with_answers_before_the_target(make_questions) -> None:
    examples = make_questions("chem", 2)
    target = make_questions("bio", 1)[0]

    prompt = build_prompt(PromptCondition.CROSS_DOMAIN, target, examples)

    expected_prefix = (
        "\n\n".join(format_mc_prompt(e, include_answer=True) for e in examples) + "\n\n"
    )
    assert prompt.text.startswith(expected_prefix)
    assert prompt.num_examples == 2


def test_target_span_covers_target_block_through_answer_cue(make_questions) -> None:
    examples = make_questions("chem", 3)
    target = make_questions("bio", 1)[0]

    prompt = build_prompt(PromptCondition.CROSS_DOMAIN, target, examples)

    assert _span_text(prompt) == format_mc_prompt(target)
    assert _span_text(prompt).endswith("Answer:")
    assert prompt.target_char_span[1] == len(prompt.text)


# --- build_condition_prompts ------------------------------------------------------


def test_target_block_is_identical_across_conditions(make_questions, tokenizer) -> None:
    target = make_questions("bio", 1)[0]

    prompts = _build_all(make_questions, tokenizer, target=target)

    assert set(prompts) == set(ALL_CONDITIONS)
    assert {_span_text(p) for p in prompts.values()} == {format_mc_prompt(target)}


def test_condition_example_sources_and_counts(make_questions, tokenizer) -> None:
    prompts = _build_all(make_questions, tokenizer)

    assert prompts[PromptCondition.ZERO_SHOT].num_examples == 0
    assert prompts[PromptCondition.CROSS_DOMAIN].num_examples == 3
    assert prompts[PromptCondition.SAME_DOMAIN].num_examples == 3
    assert "chem" in prompts[PromptCondition.CROSS_DOMAIN].text
    assert "hist" not in prompts[PromptCondition.CROSS_DOMAIN].text
    control_text = prompts[PromptCondition.IRRELEVANT].text
    assert "hist" in control_text
    assert "chem" not in control_text


def test_only_requested_conditions_are_built(make_questions, tokenizer) -> None:
    prompts = build_condition_prompts(
        make_questions("bio", 1)[0],
        source_examples=make_questions("chem", 3),
        same_domain_examples=[],
        irrelevant_examples=[],
        conditions=[PromptCondition.ZERO_SHOT, PromptCondition.CROSS_DOMAIN],
        tokenizer=tokenizer,
    )

    assert set(prompts) == {PromptCondition.ZERO_SHOT, PromptCondition.CROSS_DOMAIN}


def test_example_tokens_are_recorded_and_matched_for_the_control(make_questions, tokenizer) -> None:
    prompts = _build_all(make_questions, tokenizer, source_words=12)

    cross = prompts[PromptCondition.CROSS_DOMAIN]
    control = prompts[PromptCondition.IRRELEVANT]
    prefix_tokens = len(tokenizer(cross.text[: cross.target_char_span[0]])["input_ids"])
    assert prompts[PromptCondition.ZERO_SHOT].example_tokens == 0
    assert cross.example_tokens == count_example_tokens(make_questions("chem", 3, 12), tokenizer)
    assert prefix_tokens == cross.example_tokens
    assert control.num_examples == cross.num_examples
    assert abs(cross.example_tokens - control.example_tokens) <= LENGTH_MATCH_TOLERANCE_TOKENS


@pytest.mark.parametrize("leaky_set", ["source", "same_domain", "irrelevant"])
def test_target_in_any_example_set_is_rejected(make_questions, tokenizer, leaky_set) -> None:
    target = make_questions("bio", 1)[0]
    sets = {"source": [], "same_domain": [], "irrelevant": []}
    sets[leaky_set] = [target]

    with pytest.raises(ValueError, match="target"):
        build_condition_prompts(
            target,
            source_examples=sets["source"],
            same_domain_examples=sets["same_domain"],
            irrelevant_examples=sets["irrelevant"],
            conditions=[PromptCondition.ZERO_SHOT],
            tokenizer=tokenizer,
        )


# --- select_irrelevant_examples ---------------------------------------------------


def test_select_irrelevant_returns_exactly_k_examples_near_the_budget(tokenizer) -> None:
    pool = _varied_pool("hist")
    budget = 150

    selected = select_irrelevant_examples(pool, 4, budget, tokenizer, seed=3)

    assert len(selected) == 4
    assert len(set(selected)) == 4
    gap = abs(count_example_tokens(selected, tokenizer) - budget)
    assert gap <= LENGTH_MATCH_TOLERANCE_TOKENS


def test_select_irrelevant_is_deterministic_for_a_seed_and_varies_across_seeds(tokenizer) -> None:
    pool = _varied_pool("hist")

    first = select_irrelevant_examples(pool, 4, 150, tokenizer, seed=3)

    assert first == select_irrelevant_examples(pool, 4, 150, tokenizer, seed=3)
    others = {tuple(select_irrelevant_examples(pool, 4, 150, tokenizer, seed=s)) for s in range(6)}
    assert len(others) > 1


def test_select_irrelevant_raises_when_budget_is_unreachable_with_k(
    make_questions, tokenizer
) -> None:
    pool = make_questions("hist", 10, words=5)

    with pytest.raises(ValueError, match="budget"):
        select_irrelevant_examples(pool, 3, 500, tokenizer, seed=0)


def test_select_irrelevant_raises_when_pool_is_smaller_than_k(make_questions, tokenizer) -> None:
    with pytest.raises(ValueError, match="pool"):
        select_irrelevant_examples(make_questions("hist", 2), 3, 50, tokenizer, seed=0)


def test_select_irrelevant_with_k_zero_requires_zero_budget(make_questions, tokenizer) -> None:
    pool = make_questions("hist", 5)

    assert select_irrelevant_examples(pool, 0, 0, tokenizer, seed=0) == []
    with pytest.raises(ValueError, match="budget"):
        select_irrelevant_examples(pool, 0, 40, tokenizer, seed=0)


def test_irrelevant_examples_must_match_the_source_example_count(make_questions, tokenizer) -> None:
    with pytest.raises(ValueError, match="count"):
        build_condition_prompts(
            make_questions("bio", 1)[0],
            source_examples=make_questions("chem", 3),
            same_domain_examples=[],
            irrelevant_examples=make_questions("hist", 2),
            conditions=[PromptCondition.IRRELEVANT],
            tokenizer=tokenizer,
        )


# --- verify_target_tokens ---------------------------------------------------------


class _StartSensitiveTokenizer(WhitespaceTokenizer):
    """Tokenizes the very first word of a text differently (like a leading-space marker)."""

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
        encoded = super().__call__(text, add_special_tokens, return_offsets_mapping)
        encoded["input_ids"][0] += 100_000
        return encoded


class _SingleTokenTokenizer(WhitespaceTokenizer):
    """Merges the whole text into one token, so it straddles the target boundary."""

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
        return {"input_ids": [1], "offset_mapping": [(0, len(text))]}


def test_verify_target_tokens_accepts_aligned_prompts(make_questions, tokenizer) -> None:
    prompts = _build_all(make_questions, tokenizer)

    verify_target_tokens(list(prompts.values()), tokenizer)


def test_verify_target_tokens_rejects_differing_target_token_ids(make_questions, tokenizer) -> None:
    prompts = _build_all(make_questions, tokenizer)

    with pytest.raises(ValueError, match="differ"):
        verify_target_tokens(list(prompts.values()), _StartSensitiveTokenizer())


def test_verify_target_tokens_rejects_tokens_straddling_the_boundary(
    make_questions, tokenizer
) -> None:
    prompts = _build_all(make_questions, tokenizer)

    with pytest.raises(ValueError, match="straddle"):
        verify_target_tokens(list(prompts.values()), _SingleTokenTokenizer())


def test_verify_target_tokens_with_the_real_olmoe_tokenizer(make_questions) -> None:
    transformers = pytest.importorskip("transformers")
    try:
        real = transformers.AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
    except OSError:
        pytest.skip("OLMoE tokenizer unavailable (offline)")
    prompts = _build_all(make_questions, real)

    verify_target_tokens(list(prompts.values()), real)


# --- JSONL ------------------------------------------------------------------------


def test_prompts_round_trip_through_jsonl(make_questions, tokenizer, tmp_path) -> None:
    built = list(_build_all(make_questions, tokenizer).values())
    path = tmp_path / "nested" / "prompts.jsonl"

    save_prompts(built, path)

    assert load_prompts(path) == built
    assert [p.name for p in path.parent.iterdir()] == ["prompts.jsonl"]


def test_saving_no_prompts_writes_an_empty_file(tmp_path) -> None:
    path = tmp_path / "empty.jsonl"

    save_prompts([], path)

    assert path.read_text() == ""
    assert load_prompts(path) == []


def _saved_record(make_questions, tokenizer, tmp_path, condition) -> dict:
    path = tmp_path / "one.jsonl"
    save_prompts([_build_all(make_questions, tokenizer)[condition]], path)
    return json.loads(path.read_text())


def _write_record(tmp_path, record: dict):
    path = tmp_path / "tampered.jsonl"
    path.write_text(json.dumps(record) + "\n")
    return path


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("span end short of text", lambda r: r.update(text=r["text"] + " trailing")),
        ("zero-shot with examples", lambda r: r.update(num_examples=2)),
        ("example_tokens mismatch", lambda r: r.update(example_tokens=5)),
        ("answer_index out of range", lambda r: r["target"].update(answer_index=9)),
        ("unknown condition", lambda r: r.update(condition="bogus")),
    ],
)
def test_load_prompts_rejects_inconsistent_records(
    make_questions, tokenizer, tmp_path, field, mutate
) -> None:
    record = _saved_record(make_questions, tokenizer, tmp_path, PromptCondition.ZERO_SHOT)
    mutate(record)

    with pytest.raises(ValueError, match="line 1"):
        load_prompts(_write_record(tmp_path, record))


def test_load_prompts_rejects_examples_missing_from_a_few_shot_record(
    make_questions, tokenizer, tmp_path
) -> None:
    record = _saved_record(make_questions, tokenizer, tmp_path, PromptCondition.CROSS_DOMAIN)
    record["num_examples"] = 0

    with pytest.raises(ValueError, match="line 1"):
        load_prompts(_write_record(tmp_path, record))


def test_load_prompts_rejects_corrupt_lines(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"condition": "zero_shot"}\n')

    with pytest.raises(ValueError, match="line 1"):
        load_prompts(path)
