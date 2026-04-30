from __future__ import annotations

from aragora.heterogeneity.probe import (
    PanelistClassification,
    PromptProbeResult,
    build_probe_receipt,
)
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
