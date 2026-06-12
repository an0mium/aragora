"""Tests for scripts/check_contract_drift_ratchet.py."""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

import scripts.check_contract_drift_ratchet as ratchet


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _seed_files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    verify = tmp_path / "verify.json"
    routes = tmp_path / "routes.json"
    parity = tmp_path / "parity.json"
    program = tmp_path / "program.json"

    _write_json(
        verify,
        {
            "python_sdk_drift": ["a", "b"],
            "typescript_sdk_drift": ["x", "y", "z"],
            "missing_stable": [],
        },
    )
    _write_json(
        routes,
        {
            "missing_in_spec": ["m1", "m2"],
            "orphaned_in_spec": ["o1"],
        },
    )
    _write_json(parity, {"missing_from_both_sdks": ["p1", "p2"]})

    return verify, routes, parity, program


def test_strict_passes_on_program_start(monkeypatch, tmp_path: Path):
    verify, routes, parity, program = _seed_files(tmp_path)
    today = date.today().isoformat()

    _write_json(
        program,
        {
            "start_date": today,
            "start_total_items": 10,
            "weekly_reduction": 0.1,
            "grace_weeks": 0,
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_contract_drift_ratchet.py",
            "--strict",
            "--program-baseline",
            str(program),
            "--verify-baseline",
            str(verify),
            "--routes-baseline",
            str(routes),
            "--parity-baseline",
            str(parity),
            "--as-of",
            today,
        ],
    )

    assert ratchet.main() == 0


def test_strict_fails_when_above_target(monkeypatch, tmp_path: Path):
    verify, routes, parity, program = _seed_files(tmp_path)
    today = date.today()
    as_of = (today + timedelta(days=8)).isoformat()

    _write_json(
        program,
        {
            "start_date": today.isoformat(),
            "start_total_items": 10,
            "weekly_reduction": 0.1,
            "grace_weeks": 0,
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_contract_drift_ratchet.py",
            "--strict",
            "--program-baseline",
            str(program),
            "--verify-baseline",
            str(verify),
            "--routes-baseline",
            str(routes),
            "--parity-baseline",
            str(parity),
            "--as-of",
            as_of,
        ],
    )

    assert ratchet.main() == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start_total_items", "10"),
        ("start_total_items", True),
        ("start_total_items", -1),
        ("grace_weeks", "0"),
        ("grace_weeks", False),
        ("grace_weeks", -1),
    ],
)
def test_build_ratchet_result_rejects_invalid_program_integers(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    verify, routes, parity, program = _seed_files(tmp_path)
    payload: dict[str, object] = {
        "start_date": date.today().isoformat(),
        "start_total_items": 10,
        "weekly_reduction": 0.1,
        "grace_weeks": 0,
    }
    payload[field] = value
    _write_json(program, payload)

    with pytest.raises(ValueError, match=f"invalid '{field}'"):
        ratchet.build_ratchet_result(
            program_baseline=program,
            verify_baseline=verify,
            routes_baseline=routes,
            parity_baseline=parity,
            as_of=date.today(),
        )


def test_build_ratchet_result_allows_zero_program_integers(tmp_path: Path) -> None:
    verify, routes, parity, program = _seed_files(tmp_path)
    today = date.today()
    _write_json(
        program,
        {
            "start_date": today.isoformat(),
            "start_total_items": 0,
            "weekly_reduction": 0.1,
            "grace_weeks": 0,
        },
    )

    result = ratchet.build_ratchet_result(
        program_baseline=program,
        verify_baseline=verify,
        routes_baseline=routes,
        parity_baseline=parity,
        as_of=today,
    )

    assert result["program"]["start_total_items"] == 0
    assert result["program"]["grace_weeks"] == 0
    assert result["target"]["max_open_items"] == 0
