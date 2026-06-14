"""Tests for scripts/score_benchmark.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import score_benchmark  # noqa: E402


def _fixture_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "worker_status": "completed",
        "worker_outcome": "pr_created",
        "elapsed_seconds": 12.5,
        "files_changed": 2,
        "has_deliverable": True,
        "publish_action": "opened_pr",
        "expected_class": "deliverable_pr_created",
    }
    row.update(overrides)
    return row


def _write_fixture(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def test_score_fixtures_accepts_unique_examples(tmp_path: Path, monkeypatch) -> None:
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    _write_fixture(fixtures_dir / "cases.json", [_fixture_row()])

    monkeypatch.setattr(
        score_benchmark,
        "classify_from_metrics",
        lambda row: SimpleNamespace(value=row["expected_class"]),
    )

    all_passed, report = score_benchmark.score_fixtures(fixtures_dir)

    assert all_passed is True
    assert "Terminal-truth benchmark: PASS" in report
    assert "Examples: 1" in report
    assert "Pass: 1" in report


def test_score_fixtures_rejects_duplicate_examples_in_one_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    row = _fixture_row()
    _write_fixture(fixtures_dir / "cases.json", [row, dict(row)])

    monkeypatch.setattr(
        score_benchmark,
        "classify_from_metrics",
        lambda row: SimpleNamespace(value=row["expected_class"]),
    )

    all_passed, report = score_benchmark.score_fixtures(fixtures_dir)

    assert all_passed is False
    assert "Terminal-truth benchmark: FAIL" in report
    assert "duplicate benchmark example also seen in cases.json[0]" in report
    assert "Pass: 1" in report
    assert "Fail: 1" in report


def test_score_fixtures_rejects_duplicate_examples_across_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    row = _fixture_row()
    _write_fixture(fixtures_dir / "a.json", [row])
    _write_fixture(fixtures_dir / "b.json", [dict(row)])

    monkeypatch.setattr(
        score_benchmark,
        "classify_from_metrics",
        lambda row: SimpleNamespace(value=row["expected_class"]),
    )

    all_passed, report = score_benchmark.score_fixtures(fixtures_dir)

    assert all_passed is False
    assert "duplicate benchmark example also seen in a.json[0]" in report
