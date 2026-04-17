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


# -------- Seam B v1.1 integration tests --------


def _gate_failure_with_checksum_drift() -> dict:
    """Realistic ``dispatch_contract_gate()`` failure return with drift signals."""
    return {
        "status": "needs_human",
        "outcome": "blocked",
        "reasons": ["Issue failed contract admission"],
        "next_actions": ["retry"],
        "dispatch_contract": {
            "target_agent": "codex",
            "contract_valid": True,
            "missing_slices": [],
            "credential_envelope": {},
            "required_receipts": ["scratch"],
            "preflight_receipts": [
                {
                    "check_type": "scratch",
                    "receipt_id": "rcpt-1",
                    "passed": False,
                    "expires_at": "",
                    "failure_terminal_class": "BLOCKED_NOT_DISPATCH_BOUNDED",
                    "failed_checks": [
                        {
                            "name": "worker_contract_checksum",
                            "detail": "worker_checksum != expected (drifted)",
                        }
                    ],
                }
            ],
        },
    }


def test_maybe_upgrade_on_contract_drift_extracts_and_calls(tmp_path):
    """``maybe_upgrade_on_contract_drift`` must extract drift then call
    ``upgrade_on_contract_drift`` with the drift diagnostic populated."""
    from aragora.swarm.dispatch_followups import maybe_upgrade_on_contract_drift

    (tmp_path / "aragora" / "swarm").mkdir(parents=True)
    (tmp_path / "aragora" / "swarm" / "boss_loop.py").write_text("")
    metrics = tmp_path / "boss_metrics.jsonl"

    spec = SwarmSpec(
        raw_goal="Fix a thing",
        refined_goal="Fix a thing",
        acceptance_criteria=["Do it"],
        constraints=[],
        file_scope_hints=["aragora/swarm/boss_loop.py"],
        work_orders=[],
    )
    gate = _gate_failure_with_checksum_drift()

    sentinel_spec = SwarmSpec(
        raw_goal="Fix a thing",
        refined_goal="Fix a thing",
        acceptance_criteria=["Do it", "Scope criterion"],
        constraints=[],
        file_scope_hints=["aragora/swarm/boss_loop.py"],
        work_orders=[],
    )

    with patch(
        "aragora.swarm.dispatch_followups.upgrade_on_contract_drift",
        return_value=sentinel_spec,
    ) as mock_upgrade:
        result = maybe_upgrade_on_contract_drift(
            gate_result=gate,
            spec=spec,
            issue_number=5898,
            issue_title="[TW-02] Fix",
            issue_body="Fix `aragora/swarm/boss_loop.py`.",
            repo_root=Path(tmp_path),
            metrics_path=metrics,
            llm_client=None,
        )

    assert result is sentinel_spec
    mock_upgrade.assert_called_once()
    call_kwargs = mock_upgrade.call_args.kwargs
    passed_diff = call_kwargs["preflight_diff"]
    assert passed_diff["drift_signals"]  # extractor produced signals
    assert any(s["check"] == "worker_contract_checksum" for s in passed_diff["drift_signals"])
    # Expected.files seeded from spec.file_scope_hints by the helper.
    assert passed_diff["expected"]["files"] == ["aragora/swarm/boss_loop.py"]


def test_maybe_upgrade_on_contract_drift_no_drift_returns_none(tmp_path):
    """When the gate failure has no drift signals, the helper returns None without calling upgrader."""
    from aragora.swarm.dispatch_followups import maybe_upgrade_on_contract_drift

    spec = SwarmSpec(
        raw_goal="x",
        refined_goal="x",
        acceptance_criteria=["do"],
        constraints=[],
        file_scope_hints=["x.py"],
        work_orders=[],
    )
    gate_no_drift = {
        "status": "needs_human",
        "outcome": "blocked_auth_failure",
        "dispatch_contract": {
            "missing_slices": ["runner"],
            "preflight_receipts": [],
        },
    }
    with patch("aragora.swarm.dispatch_followups.upgrade_on_contract_drift") as mock_upgrade:
        result = maybe_upgrade_on_contract_drift(
            gate_result=gate_no_drift,
            spec=spec,
            issue_number=5898,
            issue_title="X",
            issue_body="",
            repo_root=Path(tmp_path),
            metrics_path=tmp_path / "metrics.jsonl",
        )
    assert result is None
    mock_upgrade.assert_not_called()


def test_maybe_upgrade_on_contract_drift_handles_unavailable(tmp_path):
    """Transient LLM infrastructure failure must swallow cleanly to None."""
    from aragora.swarm.dispatch_followups import (
        SpecUpgraderUnavailable,
        maybe_upgrade_on_contract_drift,
    )

    spec = SwarmSpec(
        raw_goal="x",
        refined_goal="x",
        acceptance_criteria=["do"],
        constraints=[],
        file_scope_hints=["x.py"],
        work_orders=[],
    )
    gate = _gate_failure_with_checksum_drift()
    with patch(
        "aragora.swarm.dispatch_followups.upgrade_on_contract_drift",
        side_effect=SpecUpgraderUnavailable("llm down"),
    ):
        result = maybe_upgrade_on_contract_drift(
            gate_result=gate,
            spec=spec,
            issue_number=5898,
            issue_title="X",
            issue_body="",
            repo_root=Path(tmp_path),
            metrics_path=tmp_path / "metrics.jsonl",
        )
    assert result is None


def test_seam_b_retry_flow_end_to_end(tmp_path):
    """Simulate the boss_worker_lifecycle double-gate flow: first gate returns drift,
    upgrader bounds spec, second gate passes (None). This mirrors the wiring."""
    from aragora.swarm.dispatch_followups import maybe_upgrade_on_contract_drift

    (tmp_path / "aragora" / "swarm").mkdir(parents=True)
    (tmp_path / "aragora" / "swarm" / "boss_loop.py").write_text("")
    metrics = tmp_path / "boss_metrics.jsonl"

    spec = SwarmSpec(
        raw_goal="Fix",
        refined_goal="Fix",
        acceptance_criteria=["Do it"],
        constraints=[],
        file_scope_hints=["aragora/swarm/boss_loop.py"],
        work_orders=[],
    )

    # Mocked contract gate: first call returns drift, second call passes (returns None).
    gate_returns = [_gate_failure_with_checksum_drift(), None]
    gate_calls: list[tuple] = []

    def fake_gate(*args, **kwargs):
        gate_calls.append((args, kwargs))
        return gate_returns.pop(0) if gate_returns else None

    # Run the equivalent wiring inline.
    with patch("aragora.swarm.spec_upgrader.AuditPersistence") as AP:
        AP.return_value.read_attempt_count.return_value = (0, True)
        AP.return_value.upsert.return_value = True

        first_gate_result = fake_gate(spec)
        assert first_gate_result is not None  # drift surfaced
        upgraded = maybe_upgrade_on_contract_drift(
            gate_result=first_gate_result,
            spec=spec,
            issue_number=5898,
            issue_title="[TW-02] Fix",
            issue_body="Fix `aragora/swarm/boss_loop.py`.",
            repo_root=Path(tmp_path),
            metrics_path=metrics,
            llm_client=None,
        )

    assert upgraded is not None, "Upgrader should have bounded the drift-feedback spec"
    # Scoping criterion should be synthesized from seeded expected.files.
    assert any("aragora/swarm/boss_loop.py" in c for c in upgraded.acceptance_criteria)

    # Second gate call passes -> dispatch proceeds.
    second_gate_result = fake_gate(upgraded)
    assert second_gate_result is None
    assert len(gate_calls) == 2


def test_seam_b_wiring_integrated_in_boss_worker_lifecycle():
    """Smoke-level check: ``boss_worker_lifecycle`` imports the helper and
    dispatch_followups exports it. Prevents silent wiring regressions."""
    from aragora.swarm import boss_worker_lifecycle, dispatch_followups

    assert hasattr(dispatch_followups, "maybe_upgrade_on_contract_drift")
    source = Path(boss_worker_lifecycle.__file__).read_text()
    assert "maybe_upgrade_on_contract_drift" in source, (
        "boss_worker_lifecycle.py must call maybe_upgrade_on_contract_drift (Seam B)"
    )
