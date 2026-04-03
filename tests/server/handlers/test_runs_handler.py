"""Tests for run ledger handlers."""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from aragora.pipeline.backbone_contracts import RunLedger, RunStageEvent
from aragora.server.handlers.runs import get_runs, get_run_by_id, _run_summary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_run(
    run_id: str = "run-1",
    status: str = "completed",
    execution_id: str = "exec-1",
    receipt_id: str = "rcpt-1",
    safety_mode: str = "standard",
    stages: list[str] | None = None,
) -> RunLedger:
    events = [
        RunStageEvent(stage=s, status="done") for s in (stages or ["intake", "specification"])
    ]
    return RunLedger(
        run_id=run_id,
        entrypoint="cli",
        status=status,
        execution_id=execution_id,
        receipt_id=receipt_id,
        metadata={"safety_mode": safety_mode},
        stage_events=events,
    )


def _fake_request(query: dict | None = None, match_info: dict | None = None):
    req = types.SimpleNamespace()
    req.query = query or {}
    req.match_info = match_info or {}
    return req


# ---------------------------------------------------------------------------
# _run_summary unit tests
# ---------------------------------------------------------------------------


class TestRunSummary:
    def test_extracts_fields(self):
        run = _make_run()
        summary = _run_summary(run.to_dict())
        assert summary["run_id"] == "run-1"
        assert summary["status"] == "completed"
        assert summary["execution_id"] == "exec-1"
        assert summary["receipt_id"] == "rcpt-1"
        assert summary["safety_mode"] == "standard"
        assert summary["stages"] == ["intake", "specification"]

    def test_missing_safety_mode(self):
        run = _make_run()
        run.metadata = {}
        summary = _run_summary(run.to_dict())
        assert summary["safety_mode"] == ""

    def test_empty_stages(self):
        run = _make_run(stages=[])
        summary = _run_summary(run.to_dict())
        assert summary["stages"] == []


# ---------------------------------------------------------------------------
# get_runs handler tests
# ---------------------------------------------------------------------------


class TestGetRuns:
    @pytest.mark.asyncio
    async def test_returns_runs(self):
        runs = [_make_run("r1"), _make_run("r2", status="running")]
        with patch("aragora.server.handlers.runs.PlanStore") as MockStore:
            MockStore.return_value.list_runs.return_value = runs
            result = await get_runs(_fake_request())

        body = result.get("body") if isinstance(result, dict) else None
        if body is None:
            # HandlerResult is (body, status) or web.Response-like
            if isinstance(result, tuple):
                body = result[0]
            else:
                body = result
        # Unpack depending on handler return shape
        import json as _json

        if isinstance(body, str):
            body = _json.loads(body)
        if hasattr(body, "body"):
            body = _json.loads(body.body)
        assert len(body["runs"]) == 2
        assert body["runs"][0]["run_id"] == "r1"

    @pytest.mark.asyncio
    async def test_passes_filters(self):
        with patch("aragora.server.handlers.runs.PlanStore") as MockStore:
            MockStore.return_value.list_runs.return_value = []
            await get_runs(_fake_request(query={"status": "running", "limit": "10"}))
            call_kwargs = MockStore.return_value.list_runs.call_args
            assert (
                call_kwargs.kwargs.get("status") == "running"
                or call_kwargs[1].get("status") == "running"
            )

    @pytest.mark.asyncio
    async def test_clamps_limit(self):
        with patch("aragora.server.handlers.runs.PlanStore") as MockStore:
            MockStore.return_value.list_runs.return_value = []
            await get_runs(_fake_request(query={"limit": "999"}))
            call_kwargs = MockStore.return_value.list_runs.call_args
            limit_val = (
                call_kwargs.kwargs.get("limit")
                if call_kwargs.kwargs.get("limit") is not None
                else call_kwargs[1].get("limit")
            )
            assert limit_val <= 200


# ---------------------------------------------------------------------------
# get_run_by_id handler tests
# ---------------------------------------------------------------------------


class TestGetRunById:
    @pytest.mark.asyncio
    async def test_returns_run(self):
        run = _make_run("run-42")
        with patch("aragora.server.handlers.runs.PlanStore") as MockStore:
            MockStore.return_value.get_run.return_value = run
            result = await get_run_by_id(_fake_request(match_info={"run_id": "run-42"}))

        import json as _json

        body = result
        if isinstance(body, tuple):
            body = body[0]
        if hasattr(body, "body"):
            body = _json.loads(body.body)
        elif isinstance(body, str):
            body = _json.loads(body)
        assert body["run_id"] == "run-42"

    @pytest.mark.asyncio
    async def test_missing_run_id(self):
        result = await get_run_by_id(_fake_request(match_info={}))
        # Should be a 400 error
        import json as _json

        if isinstance(result, tuple):
            assert result[1] == 400 or (hasattr(result[0], "status") and result[0].status == 400)
        elif hasattr(result, "status"):
            assert result.status == 400

    @pytest.mark.asyncio
    async def test_not_found(self):
        with patch("aragora.server.handlers.runs.PlanStore") as MockStore:
            MockStore.return_value.get_run.return_value = None
            result = await get_run_by_id(_fake_request(match_info={"run_id": "nope"}))

        import json as _json

        if isinstance(result, tuple):
            assert result[1] == 404 or (hasattr(result[0], "status") and result[0].status == 404)
        elif hasattr(result, "status"):
            assert result.status == 404


# ---------------------------------------------------------------------------
# FastAPI route tests
# ---------------------------------------------------------------------------


class TestFastAPIRoutes:
    def test_router_has_runs_routes(self):
        from aragora.server.fastapi.routes.runs import router

        paths = [r.path for r in router.routes]
        assert "/runs" in paths
        assert "/runs/{run_id}" in paths

    def test_summary_model_fields(self):
        from aragora.server.fastapi.routes.runs import RunSummary

        fields = RunSummary.model_fields
        for key in ("run_id", "status", "stages", "execution_id", "receipt_id", "safety_mode"):
            assert key in fields, f"Missing field: {key}"
