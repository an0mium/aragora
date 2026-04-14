from __future__ import annotations

import json
import sys
from pathlib import Path

_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import measure_b0_scorecard as mod  # noqa: E402


def _write_metrics(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_main_ci_passes_at_threshold(tmp_path: Path, capsys) -> None:
    metrics_path = _write_metrics(
        tmp_path / "boss_metrics.jsonl",
        [
            {"issue_number": 1001, "terminal_class": "deliverable_pr_created"},
            {"issue_number": 1002, "terminal_class": "rescue_worker_crash"},
        ],
    )

    exit_code = mod.main(["--metrics", str(metrics_path), "--ci", "--threshold", "0.5"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert (
        captured.out.strip()
        == "status=pass scorecard_status=active success_rate=0.500 threshold=0.500 "
        "total_ticks=2 unique_issues_attempted=2 unique_issues_succeeded=1 unique_issues_failed=1"
    )


def test_main_ci_fails_below_threshold(tmp_path: Path, capsys) -> None:
    metrics_path = _write_metrics(
        tmp_path / "boss_metrics.jsonl",
        [
            {"issue_number": 1001, "terminal_class": "deliverable_pr_created"},
            {"issue_number": 1002, "terminal_class": "rescue_worker_crash"},
        ],
    )

    exit_code = mod.main(["--metrics", str(metrics_path), "--ci", "--threshold", "0.75"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert (
        captured.out.strip()
        == "status=fail scorecard_status=active success_rate=0.500 threshold=0.750 "
        "total_ticks=2 unique_issues_attempted=2 unique_issues_succeeded=1 unique_issues_failed=1"
    )


def test_main_json_mode_keeps_json_output(tmp_path: Path, capsys) -> None:
    metrics_path = _write_metrics(
        tmp_path / "boss_metrics.jsonl",
        [
            {"issue_number": 1001, "terminal_class": "deliverable_pr_created"},
        ],
    )

    exit_code = mod.main(["--metrics", str(metrics_path), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "active"
    assert payload["no_rescue_success_rate"] == 1.0
    assert payload["unique_issues_attempted"] == 1


def test_build_published_scorecard_links_truth_artifact_and_previous_delta(
    tmp_path: Path,
) -> None:
    metrics_path = _write_metrics(
        tmp_path / "boss_metrics.jsonl",
        [
            {"issue_number": 1001, "terminal_class": "deliverable_pr_created"},
            {"issue_number": 1002, "terminal_class": "rescue_worker_crash"},
        ],
    )
    truth_artifact_path = _write_json(
        tmp_path / "truth-artifact.json",
        {
            "generated_at": "2026-04-14T14:00:00Z",
            "corpus": {
                "path": "docs/benchmarks/corpus.json",
                "corpus_id": "tw-01-bounded-execution-v1",
                "revision": 3,
                "recorded_on": "2026-04-14",
                "success_contract": "mergeable_pr_or_merged_pr",
                "issue_count": 2,
            },
            "primary_metrics": {
                "truth_success_rate": 0.5,
                "no_rescue_truth_success_rate": 0.5,
                "merged_only_rate": 0.5,
            },
            "coverage": {"is_complete": True, "missing_issue_numbers": []},
            "failure_class_distribution": {"rescue_worker_crash": 1},
            "rescue_counts_by_type": {"rescue_worker_crash": 1},
        },
    )
    previous_dir = tmp_path / "published" / "tw-01-bounded-execution-v1" / "rev-3"
    previous_dir.mkdir(parents=True)
    _write_json(
        previous_dir / "scorecard-20260407T120000Z.json",
        {
            "generated_at": "2026-04-07T12:00:00Z",
            "truth_metrics": {
                "truth_success_rate": 0.25,
                "no_rescue_truth_success_rate": 0.25,
                "merged_only_rate": 0.25,
            },
            "proxy_metrics": {
                "no_rescue_success_rate": 0.0,
                "unique_issues_attempted": 1,
            },
        },
    )

    published = mod.build_published_scorecard(
        scorecard=mod.compute_scorecard(mod.load_metrics(metrics_path)),
        metrics_path=metrics_path,
        truth_artifact_path=truth_artifact_path,
        publish_dir=tmp_path / "published",
        generated_at="2026-04-14T15:16:17Z",
    )

    assert published["generated_at"] == "2026-04-14T15:16:17Z"
    assert published["corpus"]["corpus_id"] == "tw-01-bounded-execution-v1"
    assert published["truth_metrics"]["truth_success_rate"] == 0.5
    assert published["proxy_metrics"]["no_rescue_success_rate"] == 0.5
    assert published["previous_artifact"]["path"].endswith(
        "tw-01-bounded-execution-v1/rev-3/scorecard-20260407T120000Z.json"
    )
    assert published["deltas"]["truth_success_rate"] == 0.25
    assert published["deltas"]["proxy_no_rescue_success_rate"] == 0.5
    assert published["deltas"]["unique_issues_attempted"] == 1


def test_resolve_available_published_scorecard_path_avoids_same_second_collision(
    tmp_path: Path,
) -> None:
    publish_dir = tmp_path / "published"
    published_scorecard = {
        "generated_at": "2026-04-14T15:16:17Z",
        "corpus": {"corpus_id": "tw-01-bounded-execution-v1", "revision": 3},
    }
    first_path = mod.resolve_published_scorecard_path(
        publish_dir=publish_dir,
        published_scorecard=published_scorecard,
    )
    first_path.parent.mkdir(parents=True)
    first_path.write_text("{}", encoding="utf-8")

    reserved_path = mod.resolve_available_published_scorecard_path(
        publish_dir=publish_dir,
        published_scorecard=published_scorecard,
    )

    assert reserved_path.name == "scorecard-20260414T151617Z-2.json"


def test_main_publish_dir_writes_timestamped_artifact_and_prints_path(
    tmp_path: Path, capsys
) -> None:
    metrics_path = _write_metrics(
        tmp_path / "boss_metrics.jsonl",
        [{"issue_number": 1001, "terminal_class": "deliverable_pr_created"}],
    )
    truth_artifact_path = _write_json(
        tmp_path / "truth-artifact.json",
        {
            "generated_at": "2026-04-14T14:00:00Z",
            "corpus": {
                "path": "docs/benchmarks/corpus.json",
                "corpus_id": "tw-01-bounded-execution-v1",
                "revision": 4,
                "recorded_on": "2026-04-14",
                "success_contract": "mergeable_pr_or_merged_pr",
                "issue_count": 1,
            },
            "primary_metrics": {"truth_success_rate": 1.0},
            "coverage": {"is_complete": True},
            "failure_class_distribution": {},
            "rescue_counts_by_type": {},
        },
    )

    exit_code = mod.main(
        [
            "--metrics",
            str(metrics_path),
            "--publish-dir",
            str(tmp_path / "published"),
            "--truth-artifact",
            str(truth_artifact_path),
        ]
    )

    captured = capsys.readouterr()
    written_path = Path(captured.out.strip())
    assert exit_code == 0
    assert written_path.parent == tmp_path / "published" / "tw-01-bounded-execution-v1" / "rev-4"
    payload = json.loads(written_path.read_text(encoding="utf-8"))
    assert payload["truth_artifact_path"] == str(truth_artifact_path)
    assert payload["proxy_metrics"]["no_rescue_success_rate"] == 1.0
    assert captured.err == ""


def test_main_publish_dir_with_json_keeps_stdout_json_and_reports_path_on_stderr(
    tmp_path: Path, capsys
) -> None:
    metrics_path = _write_metrics(
        tmp_path / "boss_metrics.jsonl",
        [{"issue_number": 1001, "terminal_class": "deliverable_pr_created"}],
    )
    truth_artifact_path = _write_json(
        tmp_path / "truth-artifact.json",
        {
            "generated_at": "2026-04-14T14:00:00Z",
            "corpus": {
                "path": "docs/benchmarks/corpus.json",
                "corpus_id": "tw-01-bounded-execution-v1",
                "revision": 5,
                "recorded_on": "2026-04-14",
                "success_contract": "mergeable_pr_or_merged_pr",
                "issue_count": 1,
            },
            "primary_metrics": {"truth_success_rate": 1.0},
            "coverage": {"is_complete": True},
            "failure_class_distribution": {},
            "rescue_counts_by_type": {},
        },
    )

    exit_code = mod.main(
        [
            "--metrics",
            str(metrics_path),
            "--publish-dir",
            str(tmp_path / "published"),
            "--truth-artifact",
            str(truth_artifact_path),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    reported_path = Path(captured.err.strip())
    assert exit_code == 0
    assert payload["corpus"]["revision"] == 5
    assert payload["proxy_metrics"]["unique_issues_attempted"] == 1
    assert reported_path.parent == tmp_path / "published" / "tw-01-bounded-execution-v1" / "rev-5"


def test_main_publish_mode_requires_truth_artifact(tmp_path: Path) -> None:
    metrics_path = _write_metrics(
        tmp_path / "boss_metrics.jsonl",
        [{"issue_number": 1001, "terminal_class": "deliverable_pr_created"}],
    )

    try:
        mod.main(["--metrics", str(metrics_path), "--publish"])
    except SystemExit as exc:
        assert str(exc) == "truth artifact required for publish mode: --truth-artifact PATH"
    else:
        raise AssertionError("publish mode should require a truth artifact")
