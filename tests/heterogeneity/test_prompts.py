from __future__ import annotations

import re
from pathlib import Path

from aragora.heterogeneity.prompts import (
    DEFAULT_PILOT_CLASS_QUOTAS,
    build_panel_prompt,
    load_prompt_file,
    load_prompt_set,
    select_pilot_prompts,
)

ROOT = Path(__file__).resolve().parents[2]


def _read_status_metric(markdown: str, label: str) -> str:
    match = re.search(rf"\| {re.escape(label)} \| ([^|]+) \|", markdown)
    assert match is not None
    return match.group(1).strip()


def test_load_prompt_set_reads_50_authored_prompts() -> None:
    prompts = load_prompt_set("tests/heterogeneity/probe_prompts")
    assert len(prompts) == 50
    assert {prompt.prompt_class for prompt in prompts} == set(DEFAULT_PILOT_CLASS_QUOTAS)


def test_load_prompt_file_reads_seeded_error() -> None:
    prompt = load_prompt_file(
        "tests/heterogeneity/probe_prompts/single_seeded_error/01_revert_window_off_by_one.md"
    )
    assert prompt.prompt_id == "sse_01_revert_window_off_by_one"
    assert prompt.seeded_error is not None
    assert "14" in prompt.seeded_error.description
    assert len(prompt.seeded_errors) == 1


def test_load_prompt_file_reads_plural_seeded_errors() -> None:
    prompt = load_prompt_file(
        "tests/heterogeneity/probe_prompts/multi_seeded_error/01_thresholds_and_window.md"
    )
    assert prompt.prompt_id == "mse_01_thresholds_and_window"
    assert prompt.seeded_error is not None
    assert len(prompt.seeded_errors) == 2
    assert "30 days" in prompt.seeded_errors[0].description
    assert "MORE permissive" in prompt.seeded_errors[1].description


def test_select_pilot_prompts_satisfies_minimum_gate() -> None:
    prompts = load_prompt_set("tests/heterogeneity/probe_prompts")
    selected = select_pilot_prompts(prompts)
    counts = {}
    for prompt in selected:
        counts[prompt.prompt_class] = counts.get(prompt.prompt_class, 0) + 1
    assert len(selected) == sum(DEFAULT_PILOT_CLASS_QUOTAS.values())
    assert counts == DEFAULT_PILOT_CLASS_QUOTAS
    assert all(count >= 2 for count in counts.values())
    assert counts["correlated_priming"] >= 6


def test_build_panel_prompt_calibrates_no_error_cases() -> None:
    prompt = load_prompt_file(
        "tests/heterogeneity/probe_prompts/null_negative/01_no_error_high_pressure.md"
    )
    rendered = build_panel_prompt(prompt)

    assert "may contain an error, multiple errors, or no error at all" in rendered
    assert "do not infer an error from the framing alone" in rendered
    assert "NO_CONCRETE_ERROR" in rendered
    assert "CRITICAL: There is a subtle error" in rendered


def test_build_panel_prompt_is_class_agnostic() -> None:
    clean = load_prompt_file(
        "tests/heterogeneity/probe_prompts/clean_neutral/01_invalidation_signals.md"
    )
    seeded = load_prompt_file(
        "tests/heterogeneity/probe_prompts/single_seeded_error/01_revert_window_off_by_one.md"
    )

    clean_preamble = build_panel_prompt(clean).split("PROMPT TO REVIEW:", 1)[0]
    seeded_preamble = build_panel_prompt(seeded).split("PROMPT TO REVIEW:", 1)[0]

    assert clean_preamble == seeded_preamble
    assert "DEFAULT_REVERT_WINDOW_DAYS is incorrectly stated" not in seeded_preamble


def test_clean_h1_readiness_prompt_matches_status_doc() -> None:
    status_doc = (ROOT / "docs/status/H1_01_REV4_PROMOTION_READINESS.md").read_text(
        encoding="utf-8"
    )
    prompt = load_prompt_file(
        ROOT / "tests/heterogeneity/probe_prompts/clean_neutral/05_h1_readiness_gate.md"
    )

    status_match = re.search(r"- Status: `([^`]+)`", status_doc)
    assert status_match is not None
    status = status_match.group(1)
    dispatched = _read_status_metric(
        status_doc, "Staged issues with dispatch evidence (any source)"
    )
    missing = _read_status_metric(status_doc, "Staged issues still missing dispatch evidence")
    needed = _read_status_metric(status_doc, "Additional dispatches needed")

    assert f"verdict is `{status}`" in prompt.body
    assert f"{dispatched} staged issues with dispatch evidence" in prompt.body
    assert f"{missing} still missing" in prompt.body
    assert f"{needed} more dispatches" in prompt.body
