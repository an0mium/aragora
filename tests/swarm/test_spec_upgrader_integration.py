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
