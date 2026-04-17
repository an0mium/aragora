"""Integration tests: SpecUpgrader wired into dispatch_followups."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest  # noqa: F401 -- used by xfail decorators in future

from aragora.swarm.dispatch_followups import (
    upgrade_on_contract_drift,
    upgrade_unbounded_spec,
)
from aragora.swarm.spec import SwarmSpec


def _make_unbounded_spec():
    return SwarmSpec(
        raw_goal="Improve thing",
        refined_goal="Improve thing",
        acceptance_criteria=[],
        constraints=[],
        file_scope_hints=[],
        work_orders=[],
    )


def test_followup_routes_through_spec_upgrader(tmp_path, monkeypatch):
    (tmp_path / "aragora" / "swarm").mkdir(parents=True)
    (tmp_path / "aragora" / "swarm" / "boss_loop.py").write_text("")
    monkeypatch.chdir(tmp_path)
    metrics = tmp_path / "boss_metrics.jsonl"

    spec = _make_unbounded_spec()
    issue_body = "Improve `aragora/swarm/boss_loop.py`."
    issue_title = "[TW-02] Improve boss_loop"

    with patch("aragora.swarm.spec_upgrader.AuditPersistence") as AP:
        AP.return_value.read_attempt_count.return_value = (0, True)
        AP.return_value.upsert.return_value = True
        result = upgrade_unbounded_spec(
            spec,
            issue_number=5898,
            issue_title=issue_title,
            issue_body=issue_body,
            repo_root=Path(tmp_path),
            metrics_path=metrics,
            llm_client=None,
        )
    assert result is not None
    assert result.is_dispatch_bounded()


def test_seam_b_upgrades_on_contract_drift(tmp_path):
    """When contract gate fails with drift, SpecUpgrader enriches with a scoping criterion."""
    (tmp_path / "aragora" / "swarm").mkdir(parents=True)
    (tmp_path / "aragora" / "swarm" / "boss_loop.py").write_text("")
    metrics = tmp_path / "boss_metrics.jsonl"

    # Already bounded spec; drift is the reason to upgrade.
    spec = SwarmSpec(
        raw_goal="Improve",
        refined_goal="Improve",
        acceptance_criteria=["Do the thing"],
        constraints=[],
        file_scope_hints=["aragora/swarm/boss_loop.py"],
        work_orders=[],
    )
    drift = {
        "expected": {"files": ["aragora/swarm/boss_loop.py"]},
        "actual": {"files": ["aragora/swarm/boss_loop.py", "unrelated.py"]},
    }

    with patch("aragora.swarm.spec_upgrader.AuditPersistence") as AP:
        AP.return_value.read_attempt_count.return_value = (0, True)
        AP.return_value.upsert.return_value = True
        result = upgrade_on_contract_drift(
            spec,
            issue_number=5898,
            issue_title="[TW-02] Fix",
            issue_body="Fix `aragora/swarm/boss_loop.py`.",
            preflight_diff=drift,
            repo_root=Path(tmp_path),
            metrics_path=metrics,
            llm_client=None,
        )
    assert result is not None
    # Scoping criterion synthesized from drift should be present.
    assert any("scope" in c.lower() for c in result.acceptance_criteria)


_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "spec_upgrader"


def test_e2e_issue_5898_gets_bounded(tmp_path, monkeypatch):
    """Frozen-fixture regression: #5898 can be upgraded to a bounded spec."""
    fixture = json.loads((_FIXTURES_DIR / "issue_5898.json").read_text())

    # Realistic repo layout subset needed for path validation.
    (tmp_path / "aragora" / "swarm").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "swarm").mkdir(parents=True, exist_ok=True)
    (tmp_path / "aragora" / "swarm" / "proof_first_queue.py").write_text("")
    (tmp_path / "aragora" / "swarm" / "__init__.py").write_text("")

    monkeypatch.chdir(tmp_path)
    metrics = tmp_path / "boss_metrics.jsonl"
    spec = SwarmSpec(
        raw_goal=fixture["title"],
        refined_goal=fixture["title"],
        acceptance_criteria=[],
        constraints=[],
        file_scope_hints=[],
        work_orders=[],
    )

    with patch("aragora.swarm.spec_upgrader.AuditPersistence") as AP:
        AP.return_value.read_attempt_count.return_value = (0, True)
        AP.return_value.upsert.return_value = True
        result = upgrade_unbounded_spec(
            spec,
            issue_number=fixture["number"],
            issue_title=fixture["title"],
            issue_body=fixture.get("body") or "",
            repo_root=Path(tmp_path),
            metrics_path=metrics,
            llm_client=None,
        )

    assert result is not None, (
        "Tier 1 should succeed for #5898 given its body references a real file"
    )
    assert result.is_dispatch_bounded()


def test_e2e_issue_5903_gets_bounded(tmp_path, monkeypatch):
    fixture = json.loads((_FIXTURES_DIR / "issue_5903.json").read_text())

    (tmp_path / "aragora" / "swarm").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "swarm").mkdir(parents=True, exist_ok=True)
    (tmp_path / "aragora" / "swarm" / "roadmap_priority.py").write_text("")
    (tmp_path / "aragora" / "swarm" / "__init__.py").write_text("")

    monkeypatch.chdir(tmp_path)
    metrics = tmp_path / "boss_metrics.jsonl"
    spec = SwarmSpec(
        raw_goal=fixture["title"],
        refined_goal=fixture["title"],
        acceptance_criteria=[],
        constraints=[],
        file_scope_hints=[],
        work_orders=[],
    )

    with patch("aragora.swarm.spec_upgrader.AuditPersistence") as AP:
        AP.return_value.read_attempt_count.return_value = (0, True)
        AP.return_value.upsert.return_value = True
        result = upgrade_unbounded_spec(
            spec,
            issue_number=fixture["number"],
            issue_title=fixture["title"],
            issue_body=fixture.get("body") or "",
            repo_root=Path(tmp_path),
            metrics_path=metrics,
            llm_client=None,
        )
    assert result is not None, "Tier 1 should bound #5903 via CS-01 track hints and body file refs"
    assert result.is_dispatch_bounded()
