from __future__ import annotations

import pytest

from aragora.heterogeneity.judge import build_judge_prompt, parse_judge_output
from aragora.heterogeneity.prompts import load_prompt_file


def test_parse_judge_output_accepts_strict_json() -> None:
    parsed = parse_judge_output('{"verdict":"missed","rationale":"did not name it"}')
    assert parsed.verdict == "missed"
    assert parsed.rationale == "did not name it"


def test_parse_judge_output_rejects_unknown_verdict() -> None:
    with pytest.raises(ValueError, match="unknown judge verdict"):
        parse_judge_output('{"verdict":"maybe","rationale":"x"}')


def test_build_judge_prompt_includes_seeded_error() -> None:
    prompt = load_prompt_file(
        "tests/heterogeneity/probe_prompts/single_seeded_error/01_revert_window_off_by_one.md"
    )
    rendered = build_judge_prompt(prompt, "The window is actually 14 days.")
    assert "KNOWN SEEDED ERROR" in rendered
    assert "DEFAULT_REVERT_WINDOW_DAYS" in rendered
    assert "valid JSON" in rendered
