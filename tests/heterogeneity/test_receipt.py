from __future__ import annotations

from aragora.heterogeneity.probe import (
    PanelistClassification,
    PromptProbeResult,
    build_probe_receipt,
)
from aragora.heterogeneity.prompts import load_prompt_file
from aragora.heterogeneity.receipt import compute_receipt_id, write_receipt


def test_receipt_id_excludes_produced_at(tmp_path) -> None:
    result = PromptProbeResult(
        prompt_id="p1",
        prompt_class="single_seeded_error",
        seeded_error="seed",
        classifications=(
            PanelistClassification(agent="a", verdict="flagged_correctly"),
            PanelistClassification(agent="b", verdict="missed"),
        ),
    )
    first = build_probe_receipt(
        run_id="run",
        results=[result],
        panel_models=["a", "b"],
        judge_model="fixture",
        produced_at="2026-04-30T00:00:00+00:00",
    )
    second = {**first, "produced_at": "2026-05-01T00:00:00+00:00"}
    assert compute_receipt_id(first) == compute_receipt_id(second)

    path = write_receipt(first, tmp_path)
    assert path.exists()
    assert path.name == f"{first['receipt_id']}.json"
    assert first["metrics"]["independent_flag_successes"] == 1
    assert first["metrics"]["independent_flag_trials"] == 2
    assert first["metrics"]["partial_multi_seeded_successes"] == 0
    assert first["metrics"]["partial_multi_seeded_trials"] == 0


def test_receipt_preserves_plural_seeded_errors() -> None:
    prompt = load_prompt_file(
        "tests/heterogeneity/probe_prompts/multi_seeded_error/01_thresholds_and_window.md"
    )
    result = PromptProbeResult.from_prompt(
        prompt,
        (
            PanelistClassification(agent="a", verdict="flagged_correctly"),
            PanelistClassification(agent="b", verdict="partial_multi_seeded"),
        ),
    )
    receipt = build_probe_receipt(
        run_id="run",
        results=[result],
        panel_models=["a", "b"],
        judge_model="fixture",
        produced_at="2026-04-30T00:00:00+00:00",
    )
    breakdown = receipt["per_prompt_breakdown"][0]
    assert len(breakdown["seeded_errors"]) == 2
    assert breakdown["seeded_error"] == breakdown["seeded_errors"][0]
    assert receipt["metrics"]["independent_flag_successes"] == 1
    assert receipt["metrics"]["partial_multi_seeded_successes"] == 1
    assert receipt["metrics"]["partial_multi_seeded_trials"] == 2
