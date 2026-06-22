from __future__ import annotations

import json
from pathlib import Path

import scripts.aft_harness as mod


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _rows() -> list[dict]:
    return [
        {
            "schema_version": "aft-pr-triage/0.1",
            "pr_number": 1,
            "decision": "merged_fast",
            "rationale_seeds": [
                "branch_namespace=dependabot",
                "diff_size=20",
                "file_count=1",
            ],
            "title": "chore(deps): bump small package",
            "tier_hint": "tier_1_or_2",
        },
        {
            "schema_version": "aft-pr-triage/0.1",
            "pr_number": 2,
            "decision": "closed_no_merge",
            "rationale_seeds": [
                "branch_namespace=codex",
                "diff_size=10",
                "file_count=1",
            ],
            "title": "preflight patch-equivalent probe",
            "tier_hint": "tier_1_or_2",
        },
    ]


def test_harness_writes_summary_with_stubbed_counts(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    holdout = tmp_path / "holdout.jsonl"
    output_dir = tmp_path / "results"
    _write_jsonl(train, _rows())
    _write_jsonl(holdout, _rows())

    rc = mod.main(
        [
            "--holdout",
            str(holdout),
            "--train",
            str(train),
            "--results-dir",
            str(output_dir),
            "--conditions",
            "baseline_random",
            "local_advocate",
        ]
    )

    assert rc == 0
    summary_paths = list(output_dir.glob("aft_summary_*.json"))
    assert len(summary_paths) == 1
    summary = json.loads(summary_paths[0].read_text(encoding="utf-8"))
    assert summary["n_tasks"] == 2
    assert set(summary["conditions"]) == {"baseline_random", "local_advocate_stubbed"}
    local = summary["conditions"]["local_advocate_stubbed"]
    assert local["stubbed_predictions"] == 2
    assert local["mock_predictions"] == 2
    assert local["real_predictions"] == 0
    assert local["error_predictions"] == 0
    assert local["stubbed"] is True


def test_summarize_reports_accuracy_and_truthfulness_counts() -> None:
    predictions = [
        mod.Prediction(
            pr_number=1,
            condition="frontier_rules",
            prediction="merged_fast",
            confidence=0.9,
            latency_ms=1,
            cost_usd_estimate=0,
        ),
        mod.Prediction(
            pr_number=2,
            condition="frontier_rules",
            prediction="merged_fast",
            confidence=0.2,
            latency_ms=1,
            cost_usd_estimate=0,
        ),
    ]
    labels = {1: "merged_fast", 2: "closed_no_merge"}

    summary = mod.summarize({"frontier_rules": predictions}, labels)

    frontier = summary["conditions"]["frontier_rules"]
    assert frontier["accuracy"] == 0.5
    assert frontier["stubbed_predictions"] == 0
    assert frontier["mock_predictions"] == 0
    assert frontier["real_predictions"] == 2
    assert frontier["error_predictions"] == 0
    assert frontier["stubbed"] is False
