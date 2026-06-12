"""Tests for scripts/generate_contract_drift_backlog.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.generate_contract_drift_backlog as backlog


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_build_backlog_counts_valid_source_lists(monkeypatch, tmp_path: Path) -> None:
    baselines = tmp_path / "scripts" / "baselines"
    _write_json(
        baselines / "verify_sdk_contracts.json",
        {
            "python_sdk_drift": ["GET /api/v1/auth/sessions"],
            "typescript_sdk_drift": ["POST /api/v1/debates"],
        },
    )
    _write_json(
        baselines / "validate_openapi_routes.json",
        {
            "missing_in_spec": ["GET /api/v1/chat/messages"],
            "orphaned_in_spec": ["GET /api/v1/billing/invoices"],
        },
    )
    _write_json(
        baselines / "check_sdk_parity.json",
        {"missing_from_both_sdks": ["GET /api/v1/knowledge/items"]},
    )
    monkeypatch.setattr(backlog, "PROJECT_ROOT", tmp_path)

    result = backlog.build_backlog()

    assert result["total_items"] == 5
    assert result["counts_by_source"] == {
        "verify_python_sdk_drift": 1,
        "verify_typescript_sdk_drift": 1,
        "routes_missing_in_spec": 1,
        "routes_orphaned_in_spec": 1,
        "sdk_missing_from_both": 1,
    }
    assert {ticket["domain"] for ticket in result["tickets"]} == {
        "auth",
        "billing",
        "chat",
        "debates",
        "knowledge",
    }


def test_build_backlog_rejects_non_list_source_field(monkeypatch, tmp_path: Path) -> None:
    baseline = tmp_path / "scripts" / "baselines" / "verify_sdk_contracts.json"
    _write_json(baseline, {"python_sdk_drift": "GET /api/v1/auth/sessions"})
    monkeypatch.setattr(backlog, "PROJECT_ROOT", tmp_path)

    try:
        backlog.build_backlog()
    except backlog.BacklogSourceError as exc:
        assert "field 'python_sdk_drift' must be a list" in str(exc)
        assert str(baseline) in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("non-list source field should fail closed")


def test_build_backlog_rejects_non_string_source_item(monkeypatch, tmp_path: Path) -> None:
    baseline = tmp_path / "scripts" / "baselines" / "check_sdk_parity.json"
    _write_json(baseline, {"missing_from_both_sdks": ["GET /api/v1/auth/sessions", 42]})
    monkeypatch.setattr(backlog, "PROJECT_ROOT", tmp_path)

    try:
        backlog.build_backlog()
    except backlog.BacklogSourceError as exc:
        assert "field 'missing_from_both_sdks' item 1 must be a non-empty string" in str(exc)
        assert str(baseline) in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("non-string source item should fail closed")


def test_main_reports_untrusted_sources_without_writing_outputs(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    baseline = tmp_path / "scripts" / "baselines" / "verify_sdk_contracts.json"
    md_out = tmp_path / "backlog.md"
    json_out = tmp_path / "backlog.json"
    _write_json(baseline, {"python_sdk_drift": [""]})
    monkeypatch.setattr(backlog, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_contract_drift_backlog.py",
            "--markdown-out",
            str(md_out),
            "--json-out",
            str(json_out),
        ],
    )

    try:
        backlog.main()
    except SystemExit as exc:
        assert str(exc) == (
            f"Contract drift baseline field 'python_sdk_drift' item 0 "
            f"must be a non-empty string: {baseline}"
        )
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("untrusted source field should stop report generation")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert not md_out.exists()
    assert not json_out.exists()
