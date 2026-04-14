from __future__ import annotations

import json
import sys
from pathlib import Path

_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import render_benchmark_truth_status as mod  # noqa: E402


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_render_status_markdown_includes_metrics_and_paths(tmp_path: Path) -> None:
    corpus_path = _write_json(
        tmp_path / "corpus.json",
        {
            "corpus_id": "tw-01-bounded-execution-v1",
            "revision": 1,
            "recorded_on": "2026-04-14",
            "success_contract": "mergeable_pr_or_merged_pr",
            "issues": [{"issue_id": 1064, "title": "Issue A"}],
        },
    )
    latest_paths = mod.resolve_latest_paths(
        corpus_path=corpus_path,
        truth_root=tmp_path / "truth",
        scorecard_root=tmp_path / "scorecards",
    )
    markdown = mod.render_status_markdown(
        corpus_path=corpus_path,
        truth_path=latest_paths["truth_corpus_latest"],
        scorecard_path=latest_paths["scorecard_corpus_latest"],
        latest_paths=latest_paths,
        truth_payload={
            "generated_at": "2026-04-14T19:00:00Z",
            "corpus": {
                "corpus_id": "tw-01-bounded-execution-v1",
                "revision": 1,
                "recorded_on": "2026-04-14",
                "success_contract": "mergeable_pr_or_merged_pr",
                "issue_count": 1,
            },
        },
        scorecard_payload={
            "generated_at": "2026-04-14T19:05:00Z",
            "corpus": {
                "corpus_id": "tw-01-bounded-execution-v1",
                "revision": 1,
                "recorded_on": "2026-04-14",
                "success_contract": "mergeable_pr_or_merged_pr",
                "issue_count": 1,
            },
            "coverage": {
                "status": "complete",
                "attempted_issue_count": 1,
                "missing_issue_numbers": [],
            },
            "truth_metrics": {
                "truth_success_rate": 1.0,
                "no_rescue_truth_success_rate": 1.0,
                "merged_only_rate": 0.0,
            },
            "proxy_metrics": {
                "no_rescue_success_rate": 1.0,
                "unique_issues_attempted": 1,
                "unique_issues_succeeded": 1,
                "unique_issues_failed": 0,
                "total_ticks": 1,
            },
            "failure_class_distribution": {},
            "rescue_counts_by_type": {},
            "previous_artifact": {
                "path": "docs/status/generated/benchmark_scorecards/tw-01-bounded-execution-v1/rev-1/scorecard-20260407T190500Z.json",
                "generated_at": "2026-04-07T19:05:00Z",
            },
            "deltas": {
                "truth_success_rate": 0.25,
                "proxy_no_rescue_success_rate": 0.5,
            },
        },
    )

    assert "B0 Benchmark Truth Status" in markdown
    assert f"`{latest_paths['truth_corpus_latest']}`" in markdown
    assert f"`{latest_paths['scorecard_corpus_latest']}`" in markdown
    assert "| Truth success rate | 100.0% |" in markdown
    assert "## Proxy Metrics" in markdown
    assert "## Deltas" in markdown
    assert "`truth_success_rate`: 0.2500" in markdown


def test_main_writes_markdown_from_latest_paths(tmp_path: Path, capsys) -> None:
    corpus_path = _write_json(
        tmp_path / "docs" / "benchmarks" / "corpus.json",
        {
            "corpus_id": "tw-01-bounded-execution-v1",
            "revision": 2,
            "recorded_on": "2026-04-14",
            "success_contract": "mergeable_pr_or_merged_pr",
            "issues": [{"issue_id": 1064, "title": "Issue A"}],
        },
    )
    truth_root = tmp_path / "docs" / "status" / "generated" / "benchmark_truth_artifacts"
    scorecard_root = tmp_path / "docs" / "status" / "generated" / "benchmark_scorecards"
    _write_json(
        truth_root / "tw-01-bounded-execution-v1" / "latest.json",
        {
            "generated_at": "2026-04-14T20:00:00Z",
            "corpus": {
                "corpus_id": "tw-01-bounded-execution-v1",
                "revision": 2,
                "recorded_on": "2026-04-14",
                "success_contract": "mergeable_pr_or_merged_pr",
                "issue_count": 1,
            },
        },
    )
    _write_json(
        scorecard_root / "tw-01-bounded-execution-v1" / "latest.json",
        {
            "generated_at": "2026-04-14T20:05:00Z",
            "corpus": {
                "corpus_id": "tw-01-bounded-execution-v1",
                "revision": 2,
                "recorded_on": "2026-04-14",
                "success_contract": "mergeable_pr_or_merged_pr",
                "issue_count": 1,
            },
            "coverage": {
                "status": "incomplete",
                "attempted_issue_count": 0,
                "missing_issue_numbers": [1064],
            },
            "truth_metrics": {
                "truth_success_rate": 0.0,
                "no_rescue_truth_success_rate": 0.0,
                "merged_only_rate": 0.0,
            },
            "proxy_metrics": {
                "no_rescue_success_rate": 0.0,
                "unique_issues_attempted": 0,
                "unique_issues_succeeded": 0,
                "unique_issues_failed": 0,
                "total_ticks": 0,
            },
            "failure_class_distribution": {"blocked_auth_failure": 1},
            "rescue_counts_by_type": {},
        },
    )
    output_path = tmp_path / "docs" / "status" / "B0_BENCHMARK_TRUTH_STATUS.md"

    exit_code = mod.main(
        [
            "--corpus",
            str(corpus_path),
            "--truth-root",
            str(truth_root),
            "--scorecard-root",
            str(scorecard_root),
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == str(output_path)
    content = output_path.read_text(encoding="utf-8")
    assert "Coverage status: `incomplete`" in content
    assert "Missing corpus issues: `1064`" in content
    assert "`blocked_auth_failure`: 1" in content
