"""Governance test for Tier 4 settlement-helper timeout reporting.

This is the regression target for
``docs/specs/TIER4_SETTLEMENT_PROBE_TIMEOUT_REPORTING.md``. It pins the
fail-closed JSON contract for live probe timeouts without changing the
settlement authorization rules.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_settler() -> Any:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "settle_tier4_pr.py"
    spec = importlib.util.spec_from_file_location("settle_tier4_pr_governance", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_json_check_reports_live_probe_timeout_as_structured_blocker(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settler = _load_settler()

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout"))

    monkeypatch.setattr(settler.subprocess, "run", fake_run)

    exit_code = settler.main(
        [
            "--check",
            "--pr",
            "7423",
            "--head",
            "57c740022e3c432718462efa12ca79f1df4f674d",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["ok"] is False
    assert payload["error"].startswith("gh pr view 7423 --repo synaptent/aragora --json ")
    assert payload["error"].endswith(" timed out after 120s")
