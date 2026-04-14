"""Tests for scripts/report_code_quality.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import report_code_quality  # noqa: E402


def _write_boss_metrics(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    metrics_path = tmp_path / ".aragora" / "overnight" / "boss_metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return metrics_path


def test_default_file_loc_ratchet_is_tighter_than_legacy_ceiling() -> None:
    assert report_code_quality.RATCHET["max_file_loc"] <= 5400


def test_check_ratchet_flags_files_above_file_loc_ceiling() -> None:
    violations = report_code_quality.check_ratchet(
        {"except_exception": 0, "type_ignore": 0, "noqa": 0},
        [
            {
                "top5_largest": [
                    {
                        "file": "aragora/nomic/dev_coordination.py",
                        "loc": report_code_quality.RATCHET["max_file_loc"] + 1,
                    }
                ]
            }
        ],
    )

    assert violations == [
        "aragora/nomic/dev_coordination.py: "
        f"{report_code_quality.RATCHET['max_file_loc'] + 1} LOC > "
        f"{report_code_quality.RATCHET['max_file_loc']}"
    ]


def test_check_ratchet_allows_files_at_file_loc_ceiling() -> None:
    violations = report_code_quality.check_ratchet(
        {"except_exception": 0, "type_ignore": 0, "noqa": 0},
        [
            {
                "top5_largest": [
                    {
                        "file": "aragora/swarm/boss_loop.py",
                        "loc": report_code_quality.RATCHET["max_file_loc"],
                    }
                ]
            }
        ],
    )

    assert violations == []


def test_build_comparison_reports_delta_from_last_recorded_baseline() -> None:
    comparison = report_code_quality.build_comparison(
        {"except_exception": 768, "type_ignore": 625, "noqa": 2568}
    )

    assert comparison["baseline_date"] == "2026-04-12"
    assert comparison["global_suppressions"]["except_exception"] == {
        "baseline": 770,
        "current": 768,
        "delta": -2,
    }


def test_build_comparison_reports_positive_delta_when_regressed() -> None:
    comparison = report_code_quality.build_comparison(
        {"except_exception": 771, "type_ignore": 625, "noqa": 2568}
    )

    assert comparison["global_suppressions"]["except_exception"]["delta"] == 1


@pytest.mark.parametrize(
    ("rows", "expected_completed", "expected_rate"),
    [
        (
            [
                {"issue_number": 1, "prompt_chars": 100, "worker_status": "completed"},
                {"issue_number": 1, "prompt_chars": 100, "worker_status": "failed"},
            ],
            0,
            0.0,
        ),
        (
            [
                {"issue_number": 1, "prompt_chars": 100, "worker_status": "failed"},
                {"issue_number": 1, "prompt_chars": 100, "worker_status": "completed"},
            ],
            1,
            1.0,
        ),
    ],
)
def test_scan_boss_metrics_uses_latest_issue_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, object]],
    expected_completed: int,
    expected_rate: float,
) -> None:
    _write_boss_metrics(tmp_path, rows)
    monkeypatch.setattr(report_code_quality, "REPO_ROOT", tmp_path)

    metrics = report_code_quality.scan_boss_metrics()

    assert metrics["available"] is True
    assert metrics["total_iterations"] == len(rows)
    assert metrics["unique_issues"] == 1
    assert metrics["issues_completed"] == expected_completed
    assert metrics["per_issue_success_rate"] == expected_rate


def test_main_json_compare_includes_baseline_comparison(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["report_code_quality.py", "--json", "--compare"],
    )
    monkeypatch.setattr(
        report_code_quality,
        "scan_all_aragora",
        lambda: {"except_exception": 768, "type_ignore": 0, "noqa": 0, "todo": 0, "fixme": 0},
    )
    monkeypatch.setattr(report_code_quality, "scan_boss_metrics", lambda: {"available": False})
    monkeypatch.setattr(report_code_quality, "scan_subsystem", lambda name, path: {"name": name})

    report_code_quality.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["baseline_comparison"]["baseline_date"] == "2026-04-12"
    assert payload["baseline_comparison"]["global_suppressions"]["except_exception"]["delta"] == -2


def test_main_text_compare_prints_baseline_section(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["report_code_quality.py", "--compare"])
    monkeypatch.setattr(
        report_code_quality,
        "scan_all_aragora",
        lambda: {"except_exception": 771, "type_ignore": 0, "noqa": 0, "todo": 0, "fixme": 0},
    )
    monkeypatch.setattr(report_code_quality, "scan_boss_metrics", lambda: {"available": False})
    monkeypatch.setattr(
        report_code_quality,
        "scan_subsystem",
        lambda name, path: {
            "name": name,
            "files": 0,
            "loc": 0,
            "test_ratio": 0.0,
            "suppressions": {"except_exception": 0, "noqa": 0},
            "top5_largest": [],
        },
    )

    report_code_quality.main()

    output = capsys.readouterr().out
    assert "Baseline Comparison:" in output
    assert "baseline_date: 2026-04-12" in output
    assert "except_exception" in output
    assert "delta=+1" in output
