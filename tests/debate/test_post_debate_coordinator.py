"""Tests for PostDebateCoordinator pipeline (C4).

Verifies that:
1. Steps run in order: explain -> plan -> notify -> execute
2. Plan receives explanation context
3. Notification includes approval URL when plan exists
4. Failure in one step doesn't cascade to others
5. Configuration toggles steps on/off
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.post_debate_coordinator import (
    PostDebateConfig,
    PostDebateCoordinator,
    PostDebateResult,
)
from aragora.pipeline.execution_mode import ExecutionMode
from aragora.pipeline.plan_store import PlanStore


def _make_debate_result(consensus=True, confidence=0.85, task="Test task"):
    """Create a mock debate result."""
    result = MagicMock()
    result.consensus = "majority" if consensus else None
    result.consensus_reached = consensus
    result.confidence = confidence
    result.task = task
    result.domain = "general"
    result.messages = []
    result.winner = "claude"
    result.agents = ["claude", "gpt4"]
    result.participants = ["claude", "gpt4"]
    result.debate_id = "test-debate"
    result.final_answer = "test answer"
    result.metadata = {}
    return result


def _patch_plan_store(monkeypatch, tmp_path):
    store = PlanStore(db_path=str(tmp_path / "post_debate_backbone.db"))
    monkeypatch.setattr("aragora.pipeline.plan_store._store", store)
    return store


class TestPostDebateResult:
    """Test PostDebateResult dataclass."""

    def test_defaults(self):
        r = PostDebateResult()
        assert r.debate_id == ""
        assert r.run_id is None
        assert r.explanation is None
        assert r.plan is None
        assert r.notification_sent is False
        assert r.execution_result is None
        assert r.errors == []

    def test_success_when_no_errors(self):
        r = PostDebateResult(debate_id="d1")
        assert r.success is True

    def test_not_success_when_errors(self):
        r = PostDebateResult(debate_id="d1", errors=["step failed"])
        assert r.success is False


class TestPostDebateConfig:
    """Test PostDebateConfig defaults."""

    def test_defaults(self):
        config = PostDebateConfig()
        assert config.execution_mode == ExecutionMode.AUTONOMOUS
        assert config.auto_explain is True
        assert config.auto_create_plan is True
        assert config.auto_notify is True
        assert config.auto_execute_plan is False
        assert config.plan_min_confidence == 0.7


class TestStepOrdering:
    """Test that pipeline steps execute in correct order."""

    def test_all_steps_disabled_returns_empty_result(self):
        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=False,
            auto_notify=False,
            auto_execute_plan=False,
            auto_persist_receipt=False,
            auto_execution_bridge=False,
        )
        coordinator = PostDebateCoordinator(config=config)
        result = coordinator.run("d1", _make_debate_result())

        assert result.debate_id == "d1"
        assert result.explanation is None
        assert result.plan is None
        assert result.notification_sent is False
        assert result.execution_result is None

    def test_explain_only(self):
        config = PostDebateConfig(
            auto_explain=True,
            auto_create_plan=False,
            auto_notify=False,
        )
        coordinator = PostDebateCoordinator(config=config)

        with patch.object(coordinator, "_step_explain", return_value={"test": True}) as mock:
            result = coordinator.run("d1", _make_debate_result(), task="test task")
            mock.assert_called_once()
            assert result.explanation == {"test": True}

    def test_plan_skipped_below_confidence(self):
        config = PostDebateConfig(
            auto_create_plan=True,
            plan_min_confidence=0.9,
            auto_notify=False,
        )
        coordinator = PostDebateCoordinator(config=config)

        with patch.object(coordinator, "_step_create_plan") as mock:
            # confidence=0.5 < threshold 0.9
            result = coordinator.run("d1", _make_debate_result(), confidence=0.5)
            mock.assert_not_called()
            assert result.plan is None

    def test_plan_runs_above_confidence(self):
        config = PostDebateConfig(
            auto_create_plan=True,
            plan_min_confidence=0.7,
            auto_notify=False,
        )
        coordinator = PostDebateCoordinator(config=config)

        with patch.object(coordinator, "_step_create_plan", return_value={"plan": "test"}) as mock:
            result = coordinator.run("d1", _make_debate_result(), confidence=0.85)
            mock.assert_called_once()
            assert result.plan == {"plan": "test"}

    def test_execute_requires_plan(self):
        config = PostDebateConfig(
            auto_execute_plan=True,
            auto_notify=False,
        )
        coordinator = PostDebateCoordinator(config=config)

        # No plan created → no execution
        result = coordinator.run("d1", _make_debate_result())
        assert result.execution_result is None


class TestPlanReceivesExplanation:
    """Test that plan creation receives explanation context."""

    def test_plan_gets_explanation_when_available(self):
        config = PostDebateConfig(
            auto_explain=True,
            auto_create_plan=True,
            plan_min_confidence=0.5,
            auto_notify=False,
        )
        coordinator = PostDebateCoordinator(config=config)

        explanation = {"explanation": "test reason", "debate_id": "d1", "task": "t"}

        with patch.object(coordinator, "_step_explain", return_value=explanation):
            with patch.object(coordinator, "_step_create_plan") as plan_mock:
                plan_mock.return_value = {"plan": "test"}
                coordinator.run("d1", _make_debate_result(), confidence=0.8, task="t")
                # Check that explanation was passed to plan step
                call_args = plan_mock.call_args
                assert call_args[0][3] == explanation  # 4th positional arg is explanation


class TestNotificationContext:
    """Test that notifications include context from prior steps."""

    def test_notification_called_with_defaults(self):
        config = PostDebateConfig(auto_notify=True)
        coordinator = PostDebateCoordinator(config=config)

        with patch.object(coordinator, "_step_notify", return_value=True) as mock:
            result = coordinator.run("d1", _make_debate_result())
            mock.assert_called_once()
            assert result.notification_sent is True

    def test_notification_receives_explanation_and_plan(self):
        config = PostDebateConfig(
            auto_explain=True,
            auto_create_plan=True,
            auto_notify=True,
            plan_min_confidence=0.5,
        )
        coordinator = PostDebateCoordinator(config=config)

        expl = {"explanation": "reason"}
        plan = {"plan": "test_plan"}

        with patch.object(coordinator, "_step_explain", return_value=expl):
            with patch.object(coordinator, "_step_create_plan", return_value=plan):
                with patch.object(coordinator, "_step_notify") as notify_mock:
                    notify_mock.return_value = True
                    coordinator.run("d1", _make_debate_result(), confidence=0.9)
                    # Notification receives explanation and plan
                    call_args = notify_mock.call_args
                    assert call_args[0][2] == expl  # explanation
                    assert call_args[0][3] == plan  # plan


class TestFailureIsolation:
    """Test that failures in one step don't cascade."""

    def test_explain_failure_doesnt_prevent_notify(self):
        config = PostDebateConfig(
            auto_explain=True,
            auto_notify=True,
        )
        coordinator = PostDebateCoordinator(config=config)

        with patch.object(coordinator, "_step_explain", return_value=None):
            with patch.object(coordinator, "_step_notify", return_value=True) as notify:
                result = coordinator.run("d1", _make_debate_result())
                notify.assert_called_once()
                assert result.notification_sent is True
                assert result.explanation is None

    def test_plan_failure_doesnt_prevent_notify(self):
        config = PostDebateConfig(
            auto_create_plan=True,
            auto_notify=True,
            plan_min_confidence=0.5,
        )
        coordinator = PostDebateCoordinator(config=config)

        with patch.object(coordinator, "_step_create_plan", return_value=None):
            with patch.object(coordinator, "_step_notify", return_value=True) as notify:
                result = coordinator.run("d1", _make_debate_result(), confidence=0.9)
                notify.assert_called_once()
                assert result.plan is None

    def test_execute_skipped_without_plan(self):
        config = PostDebateConfig(
            auto_create_plan=True,
            auto_execute_plan=True,
            auto_notify=False,
            plan_min_confidence=0.5,
        )
        coordinator = PostDebateCoordinator(config=config)

        with patch.object(coordinator, "_step_create_plan", return_value=None):
            with patch.object(coordinator, "_step_execute_plan") as exec_mock:
                result = coordinator.run("d1", _make_debate_result(), confidence=0.9)
                exec_mock.assert_not_called()


class TestStepImplementations:
    """Test individual step implementations with mocked dependencies."""

    def test_step_explain_import_error(self):
        coordinator = PostDebateCoordinator()
        with patch.dict("sys.modules", {"aragora.explainability.builder": None}):
            result = coordinator._step_explain("d1", _make_debate_result(), "task")
            assert result is None

    def test_step_explain_uses_async_builder_signature(self):
        coordinator = PostDebateCoordinator()

        class _Decision:
            def to_dict(self):
                return {"decision_id": "dec-1"}

        class _Builder:
            async def build(self, result, context=None, include_counterfactuals=True):
                assert result is not None
                return _Decision()

            def generate_summary(self, decision):
                assert isinstance(decision, _Decision)
                return "summary"

        with patch("aragora.explainability.builder.ExplanationBuilder", _Builder):
            result = coordinator._step_explain("d1", _make_debate_result(), "task")

        assert result is not None
        assert result["explanation"] == "summary"
        assert result["decision"] == {"decision_id": "dec-1"}

    def test_run_async_callable_when_loop_already_running(self):
        coordinator = PostDebateCoordinator()

        async def _returns_value(value):
            return value

        async def _inside_loop():
            return coordinator._run_async_callable(_returns_value, 7)

        assert asyncio.run(_inside_loop()) == 7

    def test_step_notify_import_error(self):
        coordinator = PostDebateCoordinator()
        with patch.dict("sys.modules", {"aragora.notifications.service": None}):
            result = coordinator._step_notify("d1", _make_debate_result(), None, None)
            assert result is False

    def test_step_execute_import_error(self):
        coordinator = PostDebateCoordinator()
        with patch.dict("sys.modules", {"aragora.pipeline.executor": None}):
            result = coordinator._step_execute_plan({"plan": MagicMock()}, None)
            assert result is None

    def test_step_execute_skips_non_approved(self):
        coordinator = PostDebateCoordinator()
        plan_obj = MagicMock()
        plan_obj.status = "draft"
        plan_data = {"plan": plan_obj}

        with patch("aragora.pipeline.executor.PlanExecutor", create=True):
            result = coordinator._step_execute_plan(plan_data, None)
            assert result is not None
            assert result.get("skipped") is True

    def test_default_config(self):
        coordinator = PostDebateCoordinator()
        assert coordinator.config.auto_explain is True
        assert coordinator.config.auto_notify is True


class TestExecutionSafetyGate:
    """Test execution gate wiring inside PostDebateCoordinator.run()."""

    def test_execution_plan_blocked_when_gate_denies(self):
        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=True,
            auto_notify=False,
            auto_execute_plan=True,
            auto_persist_receipt=False,
            auto_execution_bridge=False,
            plan_min_confidence=0.5,
            enforce_execution_safety_gate=True,
        )
        coordinator = PostDebateCoordinator(config=config)
        plan_obj = MagicMock()
        plan_obj.status = "approved"

        with patch.object(coordinator, "_step_create_plan", return_value={"plan": plan_obj}):
            with patch.object(coordinator, "_step_execution_gate") as gate_mock:
                gate_mock.return_value = {
                    "allow_auto_execution": False,
                    "reason_codes": ["provider_diversity_below_minimum"],
                }
                with patch.object(coordinator, "_step_execute_plan") as exec_mock:
                    result = coordinator.run("d1", _make_debate_result(), confidence=0.9)

        exec_mock.assert_not_called()
        assert result.execution_result is not None
        assert result.execution_result.get("skipped") is True
        assert result.execution_result.get("reason") == "execution_gate_blocked"

    def test_plan_forced_to_human_approval_when_gate_denies(self):
        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=True,
            auto_notify=False,
            auto_execute_plan=False,
            auto_persist_receipt=False,
            auto_execution_bridge=False,
            plan_min_confidence=0.5,
            enforce_execution_safety_gate=True,
        )
        coordinator = PostDebateCoordinator(config=config)

        class _Plan:
            def __init__(self):
                self.metadata = {}
                self.approval_mode = None
                self.status = None

        plan_obj = _Plan()
        with patch.object(coordinator, "_step_create_plan", return_value={"plan": plan_obj}):
            with patch.object(coordinator, "_step_execution_gate") as gate_mock:
                gate_mock.return_value = {
                    "allow_auto_execution": False,
                    "reason_codes": ["tainted_context_detected"],
                }
                result = coordinator.run("d1", _make_debate_result(), confidence=0.9)

        assert result.plan is not None
        assert plan_obj.metadata.get("execution_gate") is not None
        assert str(plan_obj.status).lower().endswith("awaiting_approval")

    def test_execution_gate_fails_closed_when_evaluation_errors_or_is_unavailable(self):
        coordinator = PostDebateCoordinator()
        debate_result = _make_debate_result()

        with patch(
            "aragora.debate.execution_safety.evaluate_auto_execution_safety",
            side_effect=RuntimeError("boom"),
        ):
            errored_gate = coordinator._step_execution_gate(debate_result)

        assert errored_gate == {
            "allow_auto_execution": False,
            "reason_codes": ["gate_evaluation_failed"],
        }

        with patch.dict(sys.modules, {"aragora.debate.execution_safety": None}):
            unavailable_gate = coordinator._step_execution_gate(debate_result)

        assert unavailable_gate == {
            "allow_auto_execution": False,
            "reason_codes": ["gate_unavailable"],
        }


class TestBackboneWiring:
    """Tests for backbone ledger wiring in PostDebateCoordinator."""

    def test_backbone_run_seeded_when_no_run_id(self, tmp_path, monkeypatch):
        store = _patch_plan_store(monkeypatch, tmp_path)
        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=False,
            auto_notify=False,
            auto_execute_plan=False,
            auto_persist_receipt=False,
            auto_execution_bridge=False,
            enforce_execution_safety_gate=False,
        )
        coordinator = PostDebateCoordinator(config=config)

        result = coordinator.run("d1", _make_debate_result(confidence=0.9), confidence=0.9)
        run = store.get_run(result.run_id or "")

        assert result.run_id is not None
        assert run is not None
        assert run.entrypoint == "post_debate_coordinator.run"
        assert run.debate_id == "d1"
        assert any(
            event.stage == "intake" and event.status == "completed" for event in run.stage_events
        )
        assert any(
            event.stage == "deliberation" and event.status == "completed"
            for event in run.stage_events
        )

    def test_post_debate_wires_backbone_when_run_id_present(self):
        """Coordinator should wire to backbone when debate has backbone_run_id."""
        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=True,
            auto_notify=False,
            auto_execute_plan=False,
            auto_persist_receipt=False,
            auto_execution_bridge=False,
            enforce_execution_safety_gate=False,
            plan_min_confidence=0.5,
        )
        coordinator = PostDebateCoordinator(config=config)

        debate_result = _make_debate_result(confidence=0.9)
        debate_result.metadata = {"backbone_run_id": "boss-run-123-issue42"}

        appended_events: list[dict] = []
        updated_calls: list[dict] = []

        class MockRuntime:
            def append_stage_event(self, run_id, stage, *, status, artifact_ref=""):
                appended_events.append(
                    {
                        "run_id": run_id,
                        "stage": stage,
                        "status": status,
                        "artifact_ref": artifact_ref,
                    }
                )

            def update_run(self, run_id, **kw):
                updated_calls.append({"run_id": run_id, **kw})

        fake_plan = {"id": "plan-abc", "steps": []}

        with (
            patch.object(coordinator, "_step_create_plan", return_value=fake_plan),
            patch(
                "aragora.pipeline.backbone_runtime.BackboneRuntime",
                MockRuntime,
            ),
        ):
            result = coordinator.run("d1", debate_result, confidence=0.9)

        assert result.plan is not None
        assert result.run_id == "boss-run-123-issue42"
        assert len(appended_events) == 1
        assert appended_events[0]["run_id"] == "boss-run-123-issue42"
        assert appended_events[0]["stage"] == "plan"
        assert appended_events[0]["status"] == "created"
        assert appended_events[0]["artifact_ref"] == "plan-abc"
        assert len(updated_calls) == 1
        assert updated_calls[0]["status"] == "plan_ready"
        assert updated_calls[0]["plan_id"] == "plan-abc"

    def test_backbone_records_receipt_and_execution_events(self, tmp_path, monkeypatch):
        store = _patch_plan_store(monkeypatch, tmp_path)
        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=True,
            auto_notify=False,
            auto_execute_plan=True,
            auto_persist_receipt=True,
            auto_execution_bridge=False,
            enforce_execution_safety_gate=False,
            plan_min_confidence=0.5,
        )
        coordinator = PostDebateCoordinator(config=config)
        fake_plan = {"id": "plan-xyz", "steps": []}

        with (
            patch.object(coordinator, "_step_create_plan", return_value=fake_plan),
            patch.object(coordinator, "_step_persist_signed_receipt", return_value="receipt-123"),
            patch.object(
                coordinator,
                "_step_execute_plan",
                return_value={"url": "https://github.com/owner/repo/issues/42"},
            ),
        ):
            result = coordinator.run("d1", _make_debate_result(confidence=0.9), confidence=0.9)

        run = store.get_run(result.run_id or "")

        assert run is not None
        assert run.receipt_id == "receipt-123"
        execution_events = [event for event in run.stage_events if event.stage == "execution"]
        assert [event.status for event in execution_events] == ["requested", "completed"]
        assert any(
            event.stage == "receipt" and event.status == "approved" for event in run.stage_events
        )

    def test_backbone_failure_does_not_block_pipeline(self):
        """Backbone runtime errors must not prevent post-debate pipeline from completing."""
        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=True,
            auto_notify=False,
            auto_execute_plan=False,
            auto_persist_receipt=False,
            auto_execution_bridge=False,
            enforce_execution_safety_gate=False,
            plan_min_confidence=0.5,
        )
        coordinator = PostDebateCoordinator(config=config)
        debate_result = _make_debate_result(confidence=0.9)
        debate_result.metadata = {"backbone_run_id": "boss-run-999-issue7"}

        class CrashingRuntime:
            def append_stage_event(self, *a, **kw):
                raise RuntimeError("backbone unavailable")

            def update_run(self, *a, **kw):
                raise RuntimeError("backbone unavailable")

        fake_plan = {"id": "plan-crash", "steps": []}

        with (
            patch.object(coordinator, "_step_create_plan", return_value=fake_plan),
            patch(
                "aragora.pipeline.backbone_runtime.BackboneRuntime",
                CrashingRuntime,
            ),
        ):
            # Should NOT raise despite backbone failures
            result = coordinator.run("d1", debate_result, confidence=0.9)

        assert result.plan is not None
        assert result.plan["id"] == "plan-crash"

    def test_interactive_mode_raises_when_backbone_runtime_unavailable(self):
        config = PostDebateConfig(
            execution_mode=ExecutionMode.INTERACTIVE,
            auto_explain=False,
            auto_create_plan=False,
            auto_notify=False,
            auto_execute_plan=False,
            auto_persist_receipt=False,
            auto_execution_bridge=False,
            enforce_execution_safety_gate=False,
        )
        coordinator = PostDebateCoordinator(config=config)
        coordinator._backbone_runtime_failed = True

        with pytest.raises(RuntimeError, match="backbone runtime unavailable"):
            coordinator.run("d1", _make_debate_result(confidence=0.9), confidence=0.9)

    def test_interactive_mode_raises_when_backbone_write_fails(self):
        config = PostDebateConfig(
            execution_mode=ExecutionMode.INTERACTIVE,
            auto_explain=False,
            auto_create_plan=True,
            auto_notify=False,
            auto_execute_plan=False,
            auto_persist_receipt=False,
            auto_execution_bridge=False,
            enforce_execution_safety_gate=False,
            plan_min_confidence=0.5,
        )
        coordinator = PostDebateCoordinator(config=config)
        debate_result = _make_debate_result(confidence=0.9)
        debate_result.metadata = {"backbone_run_id": "boss-run-123-issue42"}

        class CrashingRuntime:
            def append_stage_event(self, *a, **kw):
                raise RuntimeError("backbone unavailable")

            def update_run(self, *a, **kw):
                raise RuntimeError("backbone unavailable")

        fake_plan = {"id": "plan-crash", "steps": []}

        with (
            patch.object(coordinator, "_step_create_plan", return_value=fake_plan),
            patch("aragora.pipeline.backbone_runtime.BackboneRuntime", CrashingRuntime),
            pytest.raises(RuntimeError, match="backbone plan record failed"),
        ):
            coordinator.run("d1", debate_result, confidence=0.9)


class TestRunLedgerIntegration:
    """Test that PostDebateCoordinator creates and populates a RunLedger."""

    def test_ledger_created_on_run(self):
        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=False,
            auto_notify=False,
            auto_execute_plan=False,
            auto_persist_receipt=False,
            auto_execution_bridge=False,
            enforce_execution_safety_gate=False,
        )
        coordinator = PostDebateCoordinator(config=config)
        coordinator.run("d1", _make_debate_result())

        ledger = coordinator._ledger
        assert ledger is not None
        assert ledger.entrypoint == "post_debate_coordinator"
        assert ledger.debate_id == "d1"
        assert ledger.run_id.startswith("post-debate-")

    def test_ledger_plan_created_event(self):
        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=True,
            auto_notify=False,
            auto_execute_plan=False,
            auto_persist_receipt=False,
            auto_execution_bridge=False,
            enforce_execution_safety_gate=False,
            plan_min_confidence=0.5,
        )
        coordinator = PostDebateCoordinator(config=config)

        plan_obj = MagicMock()
        plan_obj.plan_id = "plan-123"
        with patch.object(coordinator, "_step_create_plan", return_value={"plan": plan_obj}):
            coordinator.run("d1", _make_debate_result(), confidence=0.9)

        ledger = coordinator._ledger
        assert ledger is not None
        assert ledger.plan_id == "plan-123"
        plan_events = [e for e in ledger.stage_events if e.status == "plan_created"]
        assert len(plan_events) == 1
        assert plan_events[0].stage == "plan"
        assert plan_events[0].details["debate_id"] == "d1"
        assert plan_events[0].details["plan_id"] == "plan-123"

    def test_ledger_execution_events(self):
        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=True,
            auto_notify=False,
            auto_execute_plan=True,
            auto_persist_receipt=False,
            auto_execution_bridge=False,
            enforce_execution_safety_gate=False,
            plan_min_confidence=0.5,
        )
        coordinator = PostDebateCoordinator(config=config)

        plan_obj = MagicMock()
        plan_obj.plan_id = "plan-456"
        plan_obj.status = "approved"

        with patch.object(coordinator, "_step_create_plan", return_value={"plan": plan_obj}):
            with patch.object(coordinator, "_step_execute_plan", return_value={"ok": True}):
                coordinator.run("d1", _make_debate_result(), confidence=0.9)

        ledger = coordinator._ledger
        assert ledger is not None
        statuses = [e.status for e in ledger.stage_events]
        assert "execution_requested" in statuses
        assert "execution_completed" in statuses
        exec_events = [e for e in ledger.stage_events if e.stage == "execution"]
        assert len(exec_events) == 2
        for ev in exec_events:
            assert ev.details["debate_id"] == "d1"
            assert ev.details["plan_id"] == "plan-456"

    def test_ledger_no_plan_event_when_plan_none(self):
        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=True,
            auto_notify=False,
            auto_execute_plan=False,
            auto_persist_receipt=False,
            auto_execution_bridge=False,
            enforce_execution_safety_gate=False,
            plan_min_confidence=0.5,
        )
        coordinator = PostDebateCoordinator(config=config)

        with patch.object(coordinator, "_step_create_plan", return_value=None):
            coordinator.run("d1", _make_debate_result(), confidence=0.9)

        ledger = coordinator._ledger
        assert ledger is not None
        plan_events = [e for e in ledger.stage_events if e.status == "plan_created"]
        assert len(plan_events) == 0
