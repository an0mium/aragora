"""Tests for the runs handler (aiohttp) and FastAPI route."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Lightweight stand-ins for RunLedger / RunStageEvent so the test suite stays
# self-contained (no heavy imports from the pipeline package).
# ---------------------------------------------------------------------------


@dataclass
class _StubEvent:
    stage: str = "intake"
    status: str = "completed"
    event_id: str = "evt-abc"
    artifact_ref: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    created_at: str = "2026-04-03T00:00:00Z"

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "event_id": self.event_id,
            "artifact_ref": self.artifact_ref,
            "details": self.details,
            "created_at": self.created_at,
        }


@dataclass
class _StubLedger:
    run_id: str = "run-001"
    entrypoint: str = "cli"
    status: str = "completed"
    execution_id: str = "exec-1"
    receipt_id: str = "rcpt-1"
    stage_events: list[Any] = field(default_factory=lambda: [_StubEvent()])
    metadata: dict[str, Any] = field(default_factory=lambda: {"safety_mode": "strict"})


def _make_store(ledgers: list[_StubLedger] | None = None) -> MagicMock:
    """Return a mock PlanStore with get_run / list_runs wired up."""
    store = MagicMock()
    ledgers = ledgers if ledgers is not None else [_StubLedger()]
    _by_id = {l.run_id: l for l in ledgers}

    store.get_run.side_effect = lambda run_id: _by_id.get(run_id)
    store.list_runs.side_effect = lambda **kw: [
        l for l in ledgers if kw.get("status") is None or l.status == kw["status"]
    ][kw.get("offset", 0) : kw.get("offset", 0) + kw.get("limit", 50)]
    return store


# ============================================================================
# aiohttp handler tests
# ============================================================================


class TestAiohttpHandlers:
    """Test the aiohttp handler functions directly."""

    def _make_request(
        self,
        store: Any,
        match_info: dict[str, str] | None = None,
        query: dict[str, str] | None = None,
    ) -> MagicMock:
        req = MagicMock()
        req.app = {"context": {"plan_store": store}}
        req.match_info = match_info or {}
        req.query = query or {}
        return req

    @pytest.mark.asyncio
    async def test_list_runs(self) -> None:
        from aragora.server.handlers.runs import handle_list_runs

        store = _make_store()
        req = self._make_request(store)
        resp = await handle_list_runs(req)
        body = json.loads(resp.body)
        assert body["total"] == 1
        assert body["runs"][0]["run_id"] == "run-001"
        assert body["runs"][0]["safety_mode"] == "strict"

    @pytest.mark.asyncio
    async def test_list_runs_with_status_filter(self) -> None:
        from aragora.server.handlers.runs import handle_list_runs

        ledgers = [
            _StubLedger(run_id="r1", status="completed"),
            _StubLedger(run_id="r2", status="running"),
        ]
        store = _make_store(ledgers)
        req = self._make_request(store, query={"status": "running"})
        resp = await handle_list_runs(req)
        body = json.loads(resp.body)
        assert body["total"] == 1
        assert body["runs"][0]["run_id"] == "r2"

    @pytest.mark.asyncio
    async def test_get_run_found(self) -> None:
        from aragora.server.handlers.runs import handle_get_run

        store = _make_store()
        req = self._make_request(store, match_info={"run_id": "run-001"})
        resp = await handle_get_run(req)
        body = json.loads(resp.body)
        assert body["run_id"] == "run-001"
        assert body["status"] == "completed"
        assert body["execution_id"] == "exec-1"
        assert body["receipt_id"] == "rcpt-1"
        assert body["safety_mode"] == "strict"
        assert len(body["stages"]) == 1
        assert body["stages"][0]["stage"] == "intake"

    @pytest.mark.asyncio
    async def test_get_run_not_found(self) -> None:
        from aiohttp import web

        from aragora.server.handlers.runs import handle_get_run

        store = _make_store()
        req = self._make_request(store, match_info={"run_id": "no-such"})
        with pytest.raises(web.HTTPNotFound):
            await handle_get_run(req)

    @pytest.mark.asyncio
    async def test_no_plan_store(self) -> None:
        from aiohttp import web

        from aragora.server.handlers.runs import handle_list_runs

        req = MagicMock()
        req.app = {"context": {}}
        req.query = {}
        with pytest.raises(web.HTTPServiceUnavailable):
            await handle_list_runs(req)

    @pytest.mark.asyncio
    async def test_safety_mode_defaults_to_standard(self) -> None:
        from aragora.server.handlers.runs import handle_get_run

        ledger = _StubLedger(metadata={})
        store = _make_store([ledger])
        req = self._make_request(store, match_info={"run_id": "run-001"})
        resp = await handle_get_run(req)
        body = json.loads(resp.body)
        assert body["safety_mode"] == "standard"


# ============================================================================
# FastAPI route tests
# ============================================================================


class TestFastAPIRoutes:
    """Test the FastAPI router using TestClient."""

    @pytest.fixture()
    def client(self) -> Any:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from aragora.server.fastapi.routes.runs import _get_plan_store, router

        app = FastAPI()
        app.include_router(router)

        store = _make_store()
        app.dependency_overrides[_get_plan_store] = lambda: store

        return TestClient(app)

    def test_list_runs(self, client: Any) -> None:
        resp = client.get("/api/v2/runs")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["runs"][0]["run_id"] == "run-001"

    def test_get_run(self, client: Any) -> None:
        resp = client.get("/api/v2/runs/run-001")
        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == "run-001"
        assert body["safety_mode"] == "strict"

    def test_get_run_not_found(self, client: Any) -> None:
        resp = client.get("/api/v2/runs/missing")
        assert resp.status_code == 404
