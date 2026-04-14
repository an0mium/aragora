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
