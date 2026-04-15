from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import render_benchmark_truth_status as mod  # noqa: E402


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _truth_payload(
    *, revision: int, generated_at: str = "2026-04-14T20:00:00Z"
) -> dict[str, object]:
    return {
        "generated_at": generated_at,
        "corpus": {
            "corpus_id": "tw-01-bounded-execution-v1",
            "revision": revision,
            "recorded_on": "2026-04-14",
            "success_contract": "mergeable_pr_or_merged_pr",
            "issue_count": 1,
        },
    }


def _scorecard_payload(
    *, revision: int, generated_at: str = "2026-04-14T20:05:00Z"
) -> dict[str, object]:
    return {
        "generated_at": generated_at,
        "corpus": {
            "corpus_id": "tw-01-bounded-execution-v1",
            "revision": revision,
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
            "unique_issues_neutral": 0,
            "total_ticks": 0,
            "neutral_classes": {},
        },
        "failure_class_distribution": {"blocked_auth_failure": 1},
        "rescue_counts_by_type": {},
    }


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
                "unique_issues_neutral": 0,
                "total_ticks": 1,
                "neutral_classes": {},
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
    assert "| Proxy no-rescue success rate | 100.0% |" in markdown
    assert "## Deltas" in markdown
    assert "`truth_success_rate`: 0.2500" in markdown


def test_render_status_markdown_surfaces_proxy_neutral_issue_classes(tmp_path: Path) -> None:
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
        truth_payload=_truth_payload(revision=1),
        scorecard_payload={
            **_scorecard_payload(revision=1),
            "proxy_metrics": {
                "no_rescue_success_rate": 0.0,
                "unique_issues_attempted": 1,
                "unique_issues_succeeded": 0,
                "unique_issues_failed": 0,
                "unique_issues_neutral": 1,
                "total_ticks": 1,
                "neutral_classes": {"issue_already_resolved": 1},
            },
        },
    )

    assert "| Unique issues neutral | 1 |" in markdown
    assert "Proxy note: neutral issue outcomes" in markdown
    assert "## Proxy Neutral Class Distribution" in markdown
    assert "`issue_already_resolved`: 1" in markdown


def test_render_status_markdown_backfills_legacy_proxy_neutral_fields(tmp_path: Path) -> None:
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
        truth_payload=_truth_payload(revision=1),
        scorecard_payload={
            **_scorecard_payload(revision=1),
            "proxy_metrics": {
                "no_rescue_success_rate": 0.0,
                "unique_issues_attempted": 5,
                "unique_issues_succeeded": 0,
                "unique_issues_failed": 1,
                "total_ticks": 6,
                "terminal_class_distribution": {
                    "blocked_auth_failure": 2,
                    "issue_already_resolved": 4,
                },
            },
        },
    )

    assert "| Unique issues neutral | 4 |" in markdown
    assert "## Proxy Neutral Class Distribution" in markdown
    assert "`issue_already_resolved`: 4" in markdown


def test_render_status_markdown_surfaces_stale_closed_corpus_issues(tmp_path: Path) -> None:
    corpus_path = _write_json(
        tmp_path / "corpus.json",
        {
            "corpus_id": "tw-01-bounded-execution-v1",
            "revision": 1,
            "recorded_on": "2026-04-14",
            "success_contract": "mergeable_pr_or_merged_pr",
            "issues": [{"issue_id": 1733, "title": "Issue A"}],
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
            **_truth_payload(revision=1),
            "corpus_freshness": {
                "status": "stale_closed_issues_detected",
                "stale_closed_issue_count": 1,
                "stale_closed_issue_numbers": [1733],
                "stale_closed_issues": [
                    {
                        "issue_number": 1733,
                        "issue_title": "Detached worker cleanup",
                        "issue_closed_at": "2026-03-31T23:45:29Z",
                        "issue_state_reason": "COMPLETED",
                        "truth_state": "no_linked_pr",
                    }
                ],
            },
        },
        scorecard_payload=_scorecard_payload(revision=1),
    )

    assert "## Corpus Freshness Alerts" in markdown
    assert "Closed issues without linked PR truth" in markdown
    assert "`#1733` `Detached worker cleanup`" in markdown
    assert "truth `no_linked_pr`" in markdown


def test_render_status_markdown_surfaces_corpus_freshness_follow_up(tmp_path: Path) -> None:
    corpus_path = _write_json(
        tmp_path / "corpus.json",
        {
            "corpus_id": "tw-01-bounded-execution-v1",
            "revision": 1,
            "recorded_on": "2026-04-14",
            "success_contract": "mergeable_pr_or_merged_pr",
            "issues": [{"issue_id": 1733, "title": "Issue A"}],
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
            **_truth_payload(revision=1),
            "corpus_freshness": {
                "status": "stale_closed_issues_detected",
                "stale_closed_issue_count": 1,
                "stale_closed_issue_numbers": [1733],
                "stale_closed_issues": [
                    {
                        "issue_number": 1733,
                        "issue_title": "Detached worker cleanup",
                        "issue_closed_at": "2026-03-31T23:45:29Z",
                        "issue_state_reason": "COMPLETED",
                        "truth_state": "no_linked_pr",
                    }
                ],
                "issue_map_path": "docs/benchmarks/benchmark_corpus_freshness.json",
                "linked_issues": [
                    {
                        "target": "#6001",
                        "title": "[TW-02] Restock stale issues in tw-01-bounded-execution-v1 rev-1",
                        "url": "https://github.com/synaptent/aragora/issues/6001",
                    }
                ],
                "issue_linkage_results": [
                    {
                        "action": "linked_existing_issue",
                        "target": "#6001",
                        "url": "https://github.com/synaptent/aragora/issues/6001",
                    }
                ],
                "issue_drafts": [],
            },
        },
        scorecard_payload=_scorecard_payload(revision=1),
    )

    assert "## Corpus Freshness Follow-Up" in markdown
    assert "Freshness map: `docs/benchmarks/benchmark_corpus_freshness.json`" in markdown
    assert "[#6001](https://github.com/synaptent/aragora/issues/6001)" in markdown
    assert (
        "`linked_existing_issue` -> [#6001](https://github.com/synaptent/aragora/issues/6001)"
        in markdown
    )


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
        _truth_payload(revision=2),
    )
    _write_json(
        truth_root / "tw-01-bounded-execution-v1" / "rev-2" / "latest.json",
        _truth_payload(revision=2),
    )
    _write_json(
        scorecard_root / "tw-01-bounded-execution-v1" / "latest.json",
        _scorecard_payload(revision=2),
    )
    _write_json(
        scorecard_root / "tw-01-bounded-execution-v1" / "rev-2" / "latest.json",
        _scorecard_payload(revision=2),
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


def test_main_requires_revision_scoped_latest_paths(tmp_path: Path) -> None:
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
        truth_root / "tw-01-bounded-execution-v1" / "latest.json", _truth_payload(revision=2)
    )
    _write_json(
        scorecard_root / "tw-01-bounded-execution-v1" / "latest.json",
        _scorecard_payload(revision=2),
    )
    output_path = tmp_path / "docs" / "status" / "B0_BENCHMARK_TRUTH_STATUS.md"

    with pytest.raises(SystemExit, match="truth artifact revision latest.json not found"):
        mod.main(
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

    assert not output_path.exists()


def test_main_rejects_stale_corpus_latest_payload_revision(tmp_path: Path) -> None:
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
        truth_root / "tw-01-bounded-execution-v1" / "latest.json", _truth_payload(revision=1)
    )
    _write_json(
        truth_root / "tw-01-bounded-execution-v1" / "rev-2" / "latest.json",
        _truth_payload(revision=2),
    )
    _write_json(
        scorecard_root / "tw-01-bounded-execution-v1" / "latest.json",
        _scorecard_payload(revision=1),
    )
    _write_json(
        scorecard_root / "tw-01-bounded-execution-v1" / "rev-2" / "latest.json",
        _scorecard_payload(revision=2),
    )
    output_path = tmp_path / "docs" / "status" / "B0_BENCHMARK_TRUTH_STATUS.md"

    with pytest.raises(SystemExit, match="truth artifact latest.json revision mismatch"):
        mod.main(
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

    assert not output_path.exists()
