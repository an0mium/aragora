from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.measure_b0_progress import (
    is_b0_cohort_row,
    load_metrics_rows,
    measure_b0_progress,
    render_table,
    report_to_json,
)
from scripts.rotate_boss_metrics import archive_path_for, rotate_metrics_file


FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "fixtures" / "b0_metrics" / "mixed_metrics.jsonl"
)


def test_measure_b0_progress_cohorts_and_unique_issue_aggregation() -> None:
    rows = load_metrics_rows(FIXTURE_PATH)

    report = measure_b0_progress(rows)

    all_summary = report["all"]
    assert all_summary.rows == 6
    assert all_summary.unique_issues_attempted == 5
    assert all_summary.unique_issues_with_mergeable_pr_signal == 1
    assert all_summary.unique_issues_with_completed_iteration == 3
    assert all_summary.deferred_publish_issue_count == 1
    assert all_summary.deferred_publish_event_count == 1
    assert all_summary.deferred_publish_issue_rate == 0.2
    assert all_summary.average_iterations_per_issue == 1.2
    assert all_summary.success_rate == 0.2

    v2_summary = report["v2"]
    assert v2_summary.rows == 5
    assert v2_summary.unique_issues_attempted == 4
    assert v2_summary.unique_issues_with_mergeable_pr_signal == 1
    assert v2_summary.unique_issues_with_completed_iteration == 2
    assert v2_summary.deferred_publish_issue_count == 1
    assert v2_summary.average_iterations_per_issue == 1.25

    decomposed_summary = report["decomposed"]
    assert decomposed_summary.rows == 2
    assert decomposed_summary.unique_issues_attempted == 1
    assert decomposed_summary.unique_issues_with_mergeable_pr_signal == 1
    assert decomposed_summary.unique_issues_with_completed_iteration == 1
    assert decomposed_summary.average_iterations_per_issue == 2.0


def test_b0_tagged_cohort_detects_title_and_metadata_tags() -> None:
    rows = load_metrics_rows(FIXTURE_PATH)

    tagged_numbers = [row["issue_number"] for row in rows if is_b0_cohort_row(row)]
    report = measure_b0_progress(rows)
    tagged_summary = report["b0_tagged"]

    assert tagged_numbers == [101, 101, 102]
    assert tagged_summary.rows == 3
    assert tagged_summary.unique_issues_attempted == 2
    assert tagged_summary.unique_issues_with_mergeable_pr_signal == 1
    assert tagged_summary.deferred_publish_issue_count == 1
    assert tagged_summary.average_iterations_per_issue == 1.5


def test_b0_tagged_cohort_detects_explicit_cohort_tag_field() -> None:
    rows = [
        {
            "issue_number": 777,
            "cohort_tag": "B0-cohort",
            "prompt_chars": 123,
            "publish_action": "pr_created",
            "worker_status": "completed",
            "terminal_class": "deliverable_pr_created",
        }
    ]

    assert is_b0_cohort_row(rows[0]) is True

    report = measure_b0_progress(rows)
    tagged_summary = report["b0_tagged"]

    assert tagged_summary.rows == 1
    assert tagged_summary.unique_issues_attempted == 1
    assert tagged_summary.unique_issues_with_mergeable_pr_signal == 1


def test_terminal_class_distribution_uses_existing_or_fallback_classification() -> None:
    rows = load_metrics_rows(FIXTURE_PATH)

    report = measure_b0_progress(rows)
    distribution = report["all"].terminal_class_distribution

    assert distribution["deliverable_pr_created"] == 1
    assert distribution["deliverable_branch_pushed"] == 1
    assert distribution["rescue_publish_deferred"] == 1
    assert distribution["rescue_worker_crash"] == 1
    assert distribution["blocked_sanitation_failed"] == 1
    assert distribution["rescue_no_deliverable"] == 1


def test_json_output_contains_machine_readable_report() -> None:
    rows = load_metrics_rows(FIXTURE_PATH)
    report = measure_b0_progress(rows)

    payload = json.loads(report_to_json(FIXTURE_PATH, report))

    assert payload["metrics_file"].endswith("mixed_metrics.jsonl")
    assert payload["cohorts"]["all"]["unique_issues_attempted"] == 5
    assert payload["cohorts"]["b0_tagged"]["unique_issues_with_mergeable_pr_signal"] == 1


def test_table_output_contains_comparison_rows_and_terminal_classes() -> None:
    rows = load_metrics_rows(FIXTURE_PATH)
    report = measure_b0_progress(rows)

    table = render_table(report)

    assert "Cohort" in table
    assert "all" in table
    assert "b0_tagged" in table
    assert "Terminal class distribution:" in table
    assert "deliverable_pr_created" in table
    assert "rescue_publish_deferred" in table


def test_rotate_metrics_file_dry_run_and_archive_path() -> None:
    metrics_path = Path("/tmp/example-boss-metrics.jsonl")
    now = datetime(2026, 4, 12, 19, 30, tzinfo=timezone.utc)

    archive_path = archive_path_for(metrics_path, now=now)
    summary = rotate_metrics_file(metrics_path, dry_run=True, now=now)

    assert archive_path.name == "example-boss-metrics.20260412T193000Z.jsonl"
    assert summary["archived_existing_file"] is False
    assert summary["created_fresh_file"] is True
    assert summary["archive_path"].endswith("example-boss-metrics.20260412T193000Z.jsonl")


def test_rotate_metrics_file_moves_existing_file_and_creates_empty_replacement(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "boss_metrics.jsonl"
    metrics_path.write_text('{"issue_number": 1}\n', encoding="utf-8")
    now = datetime(2026, 4, 12, 19, 30, tzinfo=timezone.utc)

    summary = rotate_metrics_file(metrics_path, dry_run=False, now=now)
    archived_path = Path(summary["archive_path"])

    assert archived_path.exists()
    assert archived_path.read_text(encoding="utf-8") == '{"issue_number": 1}\n'
    assert metrics_path.exists()
    assert metrics_path.read_text(encoding="utf-8") == ""
