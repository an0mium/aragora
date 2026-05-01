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


def test_build_judge_prompt_includes_plural_seeded_errors() -> None:
    prompt = load_prompt_file(
        "tests/heterogeneity/probe_prompts/multi_seeded_error/01_thresholds_and_window.md"
    )
    rendered = build_judge_prompt(prompt, "The threshold claim is wrong.")
    assert "- DEFAULT_REVERT_WINDOW_DAYS" in rendered
    assert "- Logical inversion" in rendered
    assert "[aragora/review/invalidation.py:103-105]" in rendered
    assert "[aragora/review/invalidation.py:108-114]" in rendered


def test_build_judge_prompt_includes_no_seeded_error_guidance() -> None:
    prompt = load_prompt_file(
        "tests/heterogeneity/probe_prompts/null_negative/01_no_error_high_pressure.md"
    )
    rendered = build_judge_prompt(prompt, "NO_CONCRETE_ERROR")
    assert "No seeded error." in rendered
    assert "classify NO_CONCRETE_ERROR" in rendered
    assert "subjective naming, wording, style" in rendered
    assert "legitimate strict issue" in rendered
