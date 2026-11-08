"""Shared test fixtures: small question factories and a download-free tokenizer."""

from __future__ import annotations

import re
from typing import Any

import pytest

from rr.data import MCQuestion

WORD = re.compile(r"\S+")


class WhitespaceTokenizer:
    """Minimal stand-in for a Hugging Face tokenizer: one token per whitespace word."""

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        matches = list(WORD.finditer(text))
        encoded: dict[str, Any] = {"input_ids": [self._token_id(m.group()) for m in matches]}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [m.span() for m in matches]
        return encoded

    @staticmethod
    def _token_id(word: str) -> int:
        return sum(ord(c) for c in word)


def make_question(subject: str, index: int, words: int = 3) -> MCQuestion:
    stem = " ".join(f"{subject}{index}w{i}" for i in range(words))
    return MCQuestion(
        subject=subject,
        question=f"{stem}?",
        choices=tuple(f"{subject}{index}{letter}" for letter in "abcd"),
        answer_index=index % 4,
    )


@pytest.fixture
def tokenizer() -> WhitespaceTokenizer:
    return WhitespaceTokenizer()


@pytest.fixture
def make_questions():
    def _make(subject: str, count: int, words: int = 3) -> list[MCQuestion]:
        return [make_question(subject, i, words) for i in range(count)]

    return _make
