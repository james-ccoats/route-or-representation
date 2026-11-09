"""Tests for rr.shots (run naming) and rr.prompt_build (prompts for every replicate and k)."""

from __future__ import annotations

import pytest

from rr.prompt_build import build_direction_prompts
from rr.prompts import PromptCondition, verify_target_tokens
from rr.shots import ShotRun, parse_run_stem, run_stem

ALL = tuple(PromptCondition)


def test_run_stem_replaces_the_pair_separator_and_appends_replicate_and_shots() -> None:
    assert run_stem("bio_med:a2b", replicate=2, shots=3) == "bio_med_a2b_r2_k3"


def test_parse_run_stem_round_trips() -> None:
    parsed = parse_run_stem("bio_med_a2b_r2_k3")

    assert parsed == ShotRun(direction="bio_med_a2b", replicate=2, shots=3)
    assert parsed.stem == "bio_med_a2b_r2_k3"


@pytest.mark.parametrize("stem", ["bio_med_a2b", "bio_med_a2b_r1", "bio_k3", "x_rA_k3"])
def test_parse_run_stem_returns_none_for_other_names(stem: str) -> None:
    assert parse_run_stem(stem) is None


def test_parse_run_stem_keeps_underscores_in_the_direction() -> None:
    assert parse_run_stem("hsbio_to_bio_a2b_r0_k5") == ShotRun("hsbio_to_bio_a2b", 0, 5)


@pytest.fixture
def pools(make_questions):
    return {
        "source_pool": make_questions("chem", 14),
        "target_pool": make_questions("bio", 14),
        "eval_pool": make_questions("bioeval", 30),
        "irrelevant_pool": make_questions("hist", 60, words=4),
    }


def _build(pools, tokenizer, **overrides):
    settings = {
        "direction": "chem_bio:a2b",
        "tokenizer": tokenizer,
        "conditions": ALL,
        "num_shots": 4,
        "shot_counts": (1, 2, 4),
        "num_replicates": 2,
        "num_questions": 5,
        "seed": 0,
        **pools,
        **overrides,
    }
    return build_direction_prompts(**settings)


def test_one_prompt_list_per_replicate_and_shot_count(pools, tokenizer) -> None:
    built = _build(pools, tokenizer)

    assert sorted(built) == [f"chem_bio_a2b_r{r}_k{k}" for r in (0, 1) for k in (1, 2, 4)]


def test_each_run_has_every_condition_for_every_question(pools, tokenizer) -> None:
    built = _build(pools, tokenizer)

    for prompts in built.values():
        assert len(prompts) == 5 * len(ALL)
        assert {p.condition for p in prompts} == set(ALL)


def test_example_counts_match_k_for_every_few_shot_condition(pools, tokenizer) -> None:
    built = _build(pools, tokenizer)

    for stem, prompts in built.items():
        k = parse_run_stem(stem).shots
        for prompt in prompts:
            expected = 0 if prompt.condition is PromptCondition.ZERO_SHOT else k
            assert prompt.num_examples == expected


def test_k_shot_examples_are_a_prefix_of_the_larger_prompt_within_a_replicate(
    pools, tokenizer
) -> None:
    built = _build(pools, tokenizer)

    def cross_prefix(stem: str) -> str:
        prompt = next(p for p in built[stem] if p.condition is PromptCondition.CROSS_DOMAIN)
        return prompt.text[: prompt.target_char_span[0]]

    small, large = cross_prefix("chem_bio_a2b_r0_k2"), cross_prefix("chem_bio_a2b_r0_k4")
    assert large.startswith(small.rstrip("\n"))


def test_replicates_use_different_cross_domain_examples(pools, tokenizer) -> None:
    built = _build(pools, tokenizer)

    def examples(stem: str) -> str:
        prompt = next(p for p in built[stem] if p.condition is PromptCondition.CROSS_DOMAIN)
        return prompt.text[: prompt.target_char_span[0]]

    assert examples("chem_bio_a2b_r0_k4") != examples("chem_bio_a2b_r1_k4")


def test_every_run_asks_the_same_questions_in_the_same_order(pools, tokenizer) -> None:
    built = _build(pools, tokenizer)

    def targets(prompts):
        return [p.target for p in prompts if p.condition is PromptCondition.ZERO_SHOT]

    reference = targets(built["chem_bio_a2b_r0_k1"])
    assert all(targets(prompts) == reference for prompts in built.values())


def test_irrelevant_examples_match_the_cross_domain_token_count(pools, tokenizer) -> None:
    built = _build(pools, tokenizer)

    for prompts in built.values():
        tokens = {p.condition: p.example_tokens for p in prompts}
        assert abs(tokens[PromptCondition.IRRELEVANT] - tokens[PromptCondition.CROSS_DOMAIN]) <= 16


def test_eval_questions_never_appear_as_shots(pools, tokenizer) -> None:
    built = _build(pools, tokenizer)

    for prompts in built.values():
        for prompt in prompts:
            if prompt.condition is not PromptCondition.ZERO_SHOT:
                prefix = prompt.text[: prompt.target_char_span[0]]
                assert prompt.target.question not in prefix


def test_target_tokens_match_across_conditions(pools, tokenizer) -> None:
    built = _build(pools, tokenizer)

    for prompts in built.values():
        by_target = {}
        for prompt in prompts:
            by_target.setdefault(prompt.target, []).append(prompt)
        for group in by_target.values():
            verify_target_tokens(group, tokenizer)


def test_building_is_deterministic(pools, tokenizer) -> None:
    assert _build(pools, tokenizer) == _build(pools, tokenizer)


def test_without_the_irrelevant_condition_no_control_is_built(pools, tokenizer) -> None:
    conditions = (PromptCondition.ZERO_SHOT, PromptCondition.CROSS_DOMAIN)

    built = _build(pools, tokenizer, conditions=conditions, irrelevant_pool=[])

    assert all(len(prompts) == 5 * 2 for prompts in built.values())


def test_shot_count_above_num_shots_is_rejected(pools, tokenizer) -> None:
    with pytest.raises(ValueError, match="shot_counts"):
        _build(pools, tokenizer, shot_counts=(1, 5))


def test_pool_too_small_for_a_set_is_rejected(pools, tokenizer, make_questions) -> None:
    with pytest.raises(ValueError, match="pool"):
        _build(pools, tokenizer, source_pool=make_questions("chem", 3))


# --- writing prompt files -------------------------------------------------------------------


def test_write_runs_saves_one_file_per_run_and_loads_back(pools, tokenizer, tmp_path) -> None:
    from rr.prompt_build import write_runs
    from rr.prompts import load_prompts

    built = _build(pools, tokenizer)

    write_runs(built, tmp_path / "prompts")

    files = sorted(p.stem for p in (tmp_path / "prompts").glob("*.jsonl"))
    assert files == sorted(built)
    assert (
        load_prompts(tmp_path / "prompts" / "chem_bio_a2b_r0_k1.jsonl")
        == built["chem_bio_a2b_r0_k1"]
    )


def test_write_runs_refuses_when_other_prompt_files_are_present(pools, tokenizer, tmp_path) -> None:
    from rr.prompt_build import write_runs

    directory = tmp_path / "prompts"
    directory.mkdir()
    (directory / "old_a2b.jsonl").write_text("{}\n")

    with pytest.raises(ValueError, match="old_a2b"):
        write_runs(_build(pools, tokenizer), directory)

    assert sorted(p.name for p in directory.iterdir()) == ["old_a2b.jsonl"]


def test_write_runs_overwrites_its_own_earlier_files(pools, tokenizer, tmp_path) -> None:
    from rr.prompt_build import write_runs

    built = _build(pools, tokenizer)
    write_runs(built, tmp_path)

    write_runs(built, tmp_path)

    assert len(list(tmp_path.glob("*.jsonl"))) == len(built)


def test_without_overlap_drops_pool_questions_that_are_also_evaluation_questions(
    make_questions,
) -> None:
    from rr.prompt_build import without_overlap

    pool = make_questions("bio", 6)
    eval_pool = pool[1:3] + make_questions("other", 4)

    kept = without_overlap(pool, eval_pool)

    assert kept == [pool[0], *pool[3:]]
