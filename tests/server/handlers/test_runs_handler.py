"""Comprehensive tests for the runs endpoints on the prompt-engine handler.

Covers:
- GET /api/prompt-engine/runs       (list runs with filters, pagination)
- GET /api/prompt-engine/runs/{id}  (single run retrieval, not-found, field validation)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.pipeline.backbone_contracts import (
    BackboneStage,
    RunLedger,
    RunStageEvent,
)
from aragora.pipeline.plan_store import PlanStore
from aragora.server.handlers.prompt_engine.handler import PromptEngineHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# All fields that RunLedger.to_dict() must return.
REQUIRED_LEDGER_FIELDS = {
    "run_id",
    "entrypoint",
    "status",
    "intake_bundle",
    "spec_bundle",
    "goal_refs",
    "deliberation_bundle",
    "plan_id",
    "debate_id",
    "execution_id",
    "receipt_id",
    "receipt_envelope",
    "feedback_record",
    "attestation",
    "taint_flags",
    "stage_events",
    "metadata",
    "created_at",
    "updated_at",
}


def _parse(result: Any) -> dict[str, Any]:
    """Parse a HandlerResult tuple into {status, data}."""
    if hasattr(result, "to_dict"):
        data = result.to_dict()
        return {"status": data["status"], "data": data["body"]}

    body, status, _headers = result
    if isinstance(body, (bytes, bytearray)):
        parsed_body = json.loads(body.decode("utf-8"))
    elif isinstance(body, str):
        parsed_body = json.loads(body)
    else:
        parsed_body = body
    return {"status": status, "data": parsed_body}


def _make_run(
    run_id: str = "run-test",
    entrypoint: str = "prompt_engine.run",
    status: str = "received",
    plan_id: str = "",
    debate_id: str = "",
    execution_id: str = "",
    events: list[tuple[BackboneStage, str]] | None = None,
) -> RunLedger:
    """Create a RunLedger with optional stage events."""
    run = RunLedger(
        run_id=run_id,
        entrypoint=entrypoint,
        status=status,
        plan_id=plan_id,
        debate_id=debate_id,
        execution_id=execution_id,
    )
    for stage, evt_status in events or []:
        run.add_event(RunStageEvent.create(stage, status=evt_status))
    return run


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler() -> PromptEngineHandler:
    return PromptEngineHandler({})


@pytest.fixture(autouse=True)
def isolated_plan_store(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> PlanStore:
    store = PlanStore(db_path=str(tmp_path / "runs_handler_test.db"))
    monkeypatch.setattr("aragora.pipeline.plan_store.get_plan_store", lambda: store)
    return store


# ---------------------------------------------------------------------------
# GET /api/prompt-engine/runs — list runs
# ---------------------------------------------------------------------------


class TestListRuns:
    """Tests for GET /api/prompt-engine/runs."""

    def test_empty_store_returns_empty_list(
        self,
        handler: PromptEngineHandler,
    ) -> None:
        req = MagicMock()
        req.path = "/api/prompt-engine/runs"
        result = handler.handle_GET(req, {})
        parsed = _parse(result)

        assert parsed["status"] == 200
        assert parsed["data"]["runs"] == []

    def test_returns_persisted_ledgers(
        self,
        handler: PromptEngineHandler,
        isolated_plan_store: PlanStore,
    ) -> None:
        isolated_plan_store.create_run(
            _make_run("run-a", status="spec_ready", events=[(BackboneStage.INTAKE, "received")])
        )
        isolated_plan_store.create_run(
            _make_run("run-b", status="plan_ready", events=[(BackboneStage.PLAN, "completed")])
        )

        req = MagicMock()
        req.path = "/api/prompt-engine/runs"
        result = handler.handle_GET(req, {})
        parsed = _parse(result)

        assert parsed["status"] == 200
        run_ids = {r["run_id"] for r in parsed["data"]["runs"]}
        assert run_ids == {"run-a", "run-b"}

    def test_filter_by_status(
        self,
        handler: PromptEngineHandler,
        isolated_plan_store: PlanStore,
    ) -> None:
        isolated_plan_store.create_run(_make_run("run-1", status="spec_ready"))
        isolated_plan_store.create_run(_make_run("run-2", status="plan_ready"))
        isolated_plan_store.create_run(_make_run("run-3", status="spec_ready"))

        req = MagicMock()
        req.path = "/api/prompt-engine/runs"
        result = handler.handle_GET(req, {"status": "spec_ready"})
        parsed = _parse(result)

        assert parsed["status"] == 200
        assert len(parsed["data"]["runs"]) == 2
        assert all(r["status"] == "spec_ready" for r in parsed["data"]["runs"])

    def test_filter_by_plan_id(
        self,
        handler: PromptEngineHandler,
        isolated_plan_store: PlanStore,
    ) -> None:
        isolated_plan_store.create_run(_make_run("run-1", plan_id="plan-abc"))
        isolated_plan_store.create_run(_make_run("run-2", plan_id="plan-xyz"))

        req = MagicMock()
        req.path = "/api/prompt-engine/runs"
        result = handler.handle_GET(req, {"plan_id": "plan-abc"})
        parsed = _parse(result)

        assert parsed["status"] == 200
        assert len(parsed["data"]["runs"]) == 1
        assert parsed["data"]["runs"][0]["run_id"] == "run-1"

    def test_filter_by_debate_id(
        self,
        handler: PromptEngineHandler,
        isolated_plan_store: PlanStore,
    ) -> None:
        isolated_plan_store.create_run(_make_run("run-1", debate_id="deb-100"))
        isolated_plan_store.create_run(_make_run("run-2", debate_id="deb-200"))

        req = MagicMock()
        req.path = "/api/prompt-engine/runs"
        result = handler.handle_GET(req, {"debate_id": "deb-100"})
        parsed = _parse(result)

        assert parsed["status"] == 200
        assert len(parsed["data"]["runs"]) == 1
        assert parsed["data"]["runs"][0]["debate_id"] == "deb-100"

    def test_filter_by_execution_id(
        self,
        handler: PromptEngineHandler,
        isolated_plan_store: PlanStore,
    ) -> None:
        isolated_plan_store.create_run(_make_run("run-1", execution_id="exec-a"))
        isolated_plan_store.create_run(_make_run("run-2", execution_id="exec-b"))

        req = MagicMock()
        req.path = "/api/prompt-engine/runs"
        result = handler.handle_GET(req, {"execution_id": "exec-a"})
        parsed = _parse(result)

        assert parsed["status"] == 200
        assert len(parsed["data"]["runs"]) == 1
        assert parsed["data"]["runs"][0]["execution_id"] == "exec-a"

    def test_pagination_limit(
        self,
        handler: PromptEngineHandler,
        isolated_plan_store: PlanStore,
    ) -> None:
        for i in range(5):
            isolated_plan_store.create_run(_make_run(f"run-{i}"))

        req = MagicMock()
        req.path = "/api/prompt-engine/runs"
        result = handler.handle_GET(req, {"limit": 2})
        parsed = _parse(result)

        assert parsed["status"] == 200
        assert len(parsed["data"]["runs"]) == 2

    def test_pagination_offset(
        self,
        handler: PromptEngineHandler,
        isolated_plan_store: PlanStore,
    ) -> None:
        for i in range(5):
            isolated_plan_store.create_run(_make_run(f"run-{i}"))

        req = MagicMock()
        req.path = "/api/prompt-engine/runs"
        # Get all runs first to know the order
        all_result = handler.handle_GET(req, {})
        all_parsed = _parse(all_result)
        all_ids = [r["run_id"] for r in all_parsed["data"]["runs"]]

        # Get with offset=2
        result = handler.handle_GET(req, {"offset": 2})
        parsed = _parse(result)

        assert parsed["status"] == 200
        offset_ids = [r["run_id"] for r in parsed["data"]["runs"]]
        assert offset_ids == all_ids[2:]

    def test_combined_filters(
        self,
        handler: PromptEngineHandler,
        isolated_plan_store: PlanStore,
    ) -> None:
        isolated_plan_store.create_run(_make_run("run-1", status="spec_ready", plan_id="plan-a"))
        isolated_plan_store.create_run(_make_run("run-2", status="spec_ready", plan_id="plan-b"))
        isolated_plan_store.create_run(_make_run("run-3", status="plan_ready", plan_id="plan-a"))

        req = MagicMock()
        req.path = "/api/prompt-engine/runs"
        result = handler.handle_GET(req, {"status": "spec_ready", "plan_id": "plan-a"})
        parsed = _parse(result)

        assert parsed["status"] == 200
        assert len(parsed["data"]["runs"]) == 1
        assert parsed["data"]["runs"][0]["run_id"] == "run-1"

    def test_invalid_limit_uses_default(
        self,
        handler: PromptEngineHandler,
        isolated_plan_store: PlanStore,
    ) -> None:
        isolated_plan_store.create_run(_make_run("run-1"))

        req = MagicMock()
        req.path = "/api/prompt-engine/runs"
        result = handler.handle_GET(req, {"limit": "not-a-number"})
        parsed = _parse(result)

        assert parsed["status"] == 200
        assert len(parsed["data"]["runs"]) == 1

    def test_invalid_offset_uses_default(
        self,
        handler: PromptEngineHandler,
        isolated_plan_store: PlanStore,
    ) -> None:
        isolated_plan_store.create_run(_make_run("run-1"))

        req = MagicMock()
        req.path = "/api/prompt-engine/runs"
        result = handler.handle_GET(req, {"offset": "bad"})
        parsed = _parse(result)

        assert parsed["status"] == 200
        assert len(parsed["data"]["runs"]) == 1

    def test_empty_status_filter_ignored(
        self,
        handler: PromptEngineHandler,
        isolated_plan_store: PlanStore,
    ) -> None:
        isolated_plan_store.create_run(_make_run("run-1"))

        req = MagicMock()
        req.path = "/api/prompt-engine/runs"
        result = handler.handle_GET(req, {"status": ""})
        parsed = _parse(result)

        assert parsed["status"] == 200
        assert len(parsed["data"]["runs"]) == 1

    def test_nonexistent_status_returns_empty(
        self,
        handler: PromptEngineHandler,
        isolated_plan_store: PlanStore,
    ) -> None:
        isolated_plan_store.create_run(_make_run("run-1", status="received"))

        req = MagicMock()
        req.path = "/api/prompt-engine/runs"
        result = handler.handle_GET(req, {"status": "nonexistent_status"})
        parsed = _parse(result)

        assert parsed["status"] == 200
        assert parsed["data"]["runs"] == []

    def test_all_required_fields_in_list_response(
        self,
        handler: PromptEngineHandler,
        isolated_plan_store: PlanStore,
    ) -> None:
        isolated_plan_store.create_run(
            _make_run(
                "run-full",
                status="spec_ready",
                plan_id="plan-1",
                debate_id="deb-1",
                execution_id="exec-1",
                events=[(BackboneStage.INTAKE, "received"), (BackboneStage.SPEC, "completed")],
            )
        )

        req = MagicMock()
        req.path = "/api/prompt-engine/runs"
        result = handler.handle_GET(req, {})
        parsed = _parse(result)

        assert parsed["status"] == 200
        assert len(parsed["data"]["runs"]) == 1
        run_data = parsed["data"]["runs"][0]
        assert REQUIRED_LEDGER_FIELDS.issubset(run_data.keys()), (
            f"Missing fields: {REQUIRED_LEDGER_FIELDS - set(run_data.keys())}"
        )


# ---------------------------------------------------------------------------
# GET /api/prompt-engine/runs/{run_id} — single run
# ---------------------------------------------------------------------------


class TestGetRun:
    """Tests for GET /api/prompt-engine/runs/{run_id}."""

    def test_returns_single_ledger(
        self,
        handler: PromptEngineHandler,
        isolated_plan_store: PlanStore,
    ) -> None:
        isolated_plan_store.create_run(
            _make_run("run-42", status="plan_ready", events=[(BackboneStage.PLAN, "completed")])
        )

        req = MagicMock()
        req.path = "/api/prompt-engine/runs/run-42"
        result = handler.handle_GET(req)
        parsed = _parse(result)

        assert parsed["status"] == 200
        assert parsed["data"]["run"]["run_id"] == "run-42"
        assert parsed["data"]["run"]["status"] == "plan_ready"

    def test_not_found_returns_404(
        self,
        handler: PromptEngineHandler,
    ) -> None:
        req = MagicMock()
        req.path = "/api/prompt-engine/runs/nonexistent-run"
        result = handler.handle_GET(req)
        parsed = _parse(result)

        assert parsed["status"] == 404

    def test_empty_run_id_returns_400(
        self,
        handler: PromptEngineHandler,
    ) -> None:
        req = MagicMock()
        req.path = "/api/prompt-engine/runs/"
        result = handler.handle_GET(req)
        parsed = _parse(result)

        assert parsed["status"] == 400

    def test_all_required_fields_present(
        self,
        handler: PromptEngineHandler,
        isolated_plan_store: PlanStore,
    ) -> None:
        isolated_plan_store.create_run(
            _make_run(
                "run-fields",
                status="received",
                plan_id="p1",
                debate_id="d1",
                execution_id="e1",
                events=[(BackboneStage.INTAKE, "received")],
            )
        )

        req = MagicMock()
        req.path = "/api/prompt-engine/runs/run-fields"
        result = handler.handle_GET(req)
        parsed = _parse(result)

        assert parsed["status"] == 200
        run_data = parsed["data"]["run"]
        assert REQUIRED_LEDGER_FIELDS.issubset(run_data.keys()), (
            f"Missing fields: {REQUIRED_LEDGER_FIELDS - set(run_data.keys())}"
        )

    def test_stage_events_returned_correctly(
        self,
        handler: PromptEngineHandler,
        isolated_plan_store: PlanStore,
    ) -> None:
        isolated_plan_store.create_run(
            _make_run(
                "run-events",
                events=[
                    (BackboneStage.INTAKE, "received"),
                    (BackboneStage.SPEC, "completed"),
                    (BackboneStage.PLAN, "in_progress"),
                ],
            )
        )

        req = MagicMock()
        req.path = "/api/prompt-engine/runs/run-events"
        result = handler.handle_GET(req)
        parsed = _parse(result)

        assert parsed["status"] == 200
        events = parsed["data"]["run"]["stage_events"]
        assert len(events) == 3
        assert events[0]["stage"] == BackboneStage.INTAKE.value
        assert events[1]["stage"] == BackboneStage.SPEC.value
        assert events[2]["stage"] == BackboneStage.PLAN.value

    def test_field_values_match_created_run(
        self,
        handler: PromptEngineHandler,
        isolated_plan_store: PlanStore,
    ) -> None:
        isolated_plan_store.create_run(
            _make_run(
                "run-vals",
                entrypoint="custom_entry",
                status="execution_ready",
                plan_id="plan-99",
                debate_id="deb-99",
                execution_id="exec-99",
            )
        )

        req = MagicMock()
        req.path = "/api/prompt-engine/runs/run-vals"
        result = handler.handle_GET(req)
        parsed = _parse(result)

        assert parsed["status"] == 200
        run_data = parsed["data"]["run"]
        assert run_data["run_id"] == "run-vals"
        assert run_data["entrypoint"] == "custom_entry"
        assert run_data["status"] == "execution_ready"
        assert run_data["plan_id"] == "plan-99"
        assert run_data["debate_id"] == "deb-99"
        assert run_data["execution_id"] == "exec-99"

    def test_default_nullable_fields_are_none(
        self,
        handler: PromptEngineHandler,
        isolated_plan_store: PlanStore,
    ) -> None:
        isolated_plan_store.create_run(_make_run("run-defaults"))

        req = MagicMock()
        req.path = "/api/prompt-engine/runs/run-defaults"
        result = handler.handle_GET(req)
        parsed = _parse(result)

        assert parsed["status"] == 200
        run_data = parsed["data"]["run"]
        assert run_data["intake_bundle"] is None
        assert run_data["spec_bundle"] is None
        assert run_data["deliberation_bundle"] is None
        assert run_data["receipt_envelope"] is None
        assert run_data["feedback_record"] is None

    def test_default_collection_fields_are_empty(
        self,
        handler: PromptEngineHandler,
        isolated_plan_store: PlanStore,
    ) -> None:
        isolated_plan_store.create_run(_make_run("run-empty-colls"))

        req = MagicMock()
        req.path = "/api/prompt-engine/runs/run-empty-colls"
        result = handler.handle_GET(req)
        parsed = _parse(result)

        assert parsed["status"] == 200
        run_data = parsed["data"]["run"]
        assert run_data["goal_refs"] == []
        assert run_data["taint_flags"] == []
        assert run_data["attestation"] == {}
        assert run_data["metadata"] == {}

    def test_timestamps_present_and_non_empty(
        self,
        handler: PromptEngineHandler,
        isolated_plan_store: PlanStore,
    ) -> None:
        isolated_plan_store.create_run(_make_run("run-ts"))

        req = MagicMock()
        req.path = "/api/prompt-engine/runs/run-ts"
        result = handler.handle_GET(req)
        parsed = _parse(result)

        assert parsed["status"] == 200
        run_data = parsed["data"]["run"]
        assert run_data["created_at"]
        assert run_data["updated_at"]
        assert isinstance(run_data["created_at"], str)
        assert isinstance(run_data["updated_at"], str)


# ---------------------------------------------------------------------------
# Route matching (can_handle)
# ---------------------------------------------------------------------------


class TestCanHandleRuns:
    """Verify can_handle correctly matches runs endpoints."""

    def test_matches_runs_list_get(self, handler: PromptEngineHandler) -> None:
        assert handler.can_handle("/api/prompt-engine/runs", "GET")

    def test_matches_single_run_get(self, handler: PromptEngineHandler) -> None:
        assert handler.can_handle("/api/prompt-engine/runs/some-id", "GET")

    def test_rejects_runs_post(self, handler: PromptEngineHandler) -> None:
        assert not handler.can_handle("/api/prompt-engine/runs", "POST")

    def test_rejects_unrelated_path(self, handler: PromptEngineHandler) -> None:
        assert not handler.can_handle("/api/other/runs", "GET")

    def test_backward_compat_arg_order(self, handler: PromptEngineHandler) -> None:
        # can_handle(method, path) should also work
        assert handler.can_handle("GET", "/api/prompt-engine/runs")
