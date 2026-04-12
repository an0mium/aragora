from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_crash_patterns import (
    load_prompt_rows,
    low_success_categories,
    render_report,
    terminal_class_counts,
)


def test_load_prompt_rows_filters_prompt_versions(tmp_path: Path) -> None:
    metrics_path = tmp_path / "boss_metrics.jsonl"
    metrics_path.write_text(
        "\n".join(
            [
                json.dumps({"issue_number": 1, "prompt_chars": 0, "worker_status": "completed"}),
                json.dumps({"issue_number": 2, "prompt_chars": 10, "worker_status": "completed"}),
            ]
        )
        + "\n"
    )

    rows = load_prompt_rows(metrics_path)

    assert [row["issue_number"] for row in rows] == [2]


def test_terminal_class_counts_and_report(tmp_path: Path) -> None:
    rows = [
        {
            "issue_number": 101,
            "issue_title": "Narrow broad except Exception in foo.py",
            "prompt_chars": 1000,
            "worker_status": "completed",
            "worker_outcome": "pr_adopted",
            "publish_action": "pr_created",
            "elapsed_seconds": 120.0,
            "files_changed": 1,
            "has_deliverable": True,
        },
        {
            "issue_number": 102,
            "issue_title": "Narrow broad except Exception in bar.py",
            "prompt_chars": 1000,
            "worker_status": "failed",
            "worker_outcome": "worker_crash",
            "publish_action": "",
            "elapsed_seconds": 180.0,
            "files_changed": 0,
            "has_deliverable": False,
        },
    ]

    counts = terminal_class_counts(rows)
    category_stats = {
        "broad_exception": {
            "total": 2.0,
            "success_rate": 0.5,
            "crash_rate": 0.5,
            "avg_elapsed_seconds": 150.0,
        }
    }

    report = render_report(rows, counts, category_stats, threshold=0.6)

    assert counts["deliverable_pr_created"] == 1
    assert counts["rescue_worker_crash"] == 1
    assert (
        "broad_exception: success_rate=0.500 crash_rate=0.500 avg_elapsed=150.0s total=2" in report
    )
    assert "Categories below success threshold (0.60):" in report
    assert "  - broad_exception" in report


def test_low_success_categories_sorts_threshold_breaches() -> None:
    stats = {
        "broad_exception": {"success_rate": 0.9},
        "handler_validation": {"success_rate": 0.1},
        "type_annotation": {"success_rate": 0.2},
    }

    assert low_success_categories(stats, threshold=0.3) == [
        "handler_validation",
        "type_annotation",
    ]


def test_render_report_handles_missing_category_stats() -> None:
    rows = [
        {
            "issue_number": 201,
            "prompt_chars": 1000,
            "worker_status": "failed",
            "worker_outcome": "worker_crash",
            "publish_action": "",
            "elapsed_seconds": 33.0,
            "files_changed": 0,
            "has_deliverable": False,
        }
    ]
    report = render_report(
        rows,
        terminal_class_counts(rows),
        {},
        threshold=0.3,
    )

    assert "No category stats available" in report
