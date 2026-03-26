from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.pipeline.decision_plan.core import DecisionPlan
from aragora.server.handlers.base import HandlerResult
from aragora.server.handlers.debates.bridge import DebateDecisionBridgeHandler
from aragora.server.handlers.debates.handler import DebatesHandler
from aragora.server.router import RequestRouter


def _body(result: HandlerResult) -> dict:
    raw = result.body
    if isinstance(raw, bytes):
        return json.loads(raw.decode("utf-8"))
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _http_handler(body: dict | None = None) -> MagicMock:
    handler = MagicMock()
    encoded = json.dumps(body or {}).encode("utf-8")
    handler.headers = {"Content-Length": str(len(encoded))}
    handler.rfile.read.return_value = encoded
    return handler


@pytest.fixture(autouse=True)
def _allow_permissions(monkeypatch: pytest.MonkeyPatch) -> None:
    checker = SimpleNamespace(
        check_permission=lambda context, permission_key, resource_id=None: SimpleNamespace(
            allowed=True
        )
    )
    monkeypatch.setattr("aragora.rbac.decorators.get_permission_checker", lambda: checker)


class TestCanHandle:
    def test_accepts_bridge_route(self) -> None:
        handler = DebateDecisionBridgeHandler()
        assert handler.can_handle("/api/v1/debates/debate-123/bridge") is True

    def test_rejects_non_bridge_routes(self) -> None:
        handler = DebateDecisionBridgeHandler()
        assert handler.can_handle("/api/v1/debates/debate-123/package") is False
        assert handler.can_handle("/api/v1/debates/debate-123/bridge/extra") is False


class TestRouterResolution:
    def test_router_dispatches_bridge_route_to_bridge_handler(self) -> None:
        router = RequestRouter()
        router.register(DebatesHandler({}))
        router.register(DebateDecisionBridgeHandler())

        resolved = router.get_handler_for_path("/api/v1/debates/debate-123/bridge", method="POST")

        assert isinstance(resolved, DebateDecisionBridgeHandler)


class TestHandlePost:
    def test_requires_target(self) -> None:
        handler = DebateDecisionBridgeHandler()
        result = handler.handle_post("/api/v1/debates/debate-123/bridge", {}, _http_handler({}))

        assert result is not None
        assert result.status_code == 400
        assert _body(result)["error"] == "target is required"

    def test_rejects_unsupported_target(self) -> None:
        handler = DebateDecisionBridgeHandler()
        result = handler.handle_post(
            "/api/v1/debates/debate-123/bridge",
            {},
            _http_handler({"target": "asana"}),
        )

        assert result is not None
        assert result.status_code == 400
        assert "Unsupported target" in _body(result)["error"]

    def test_returns_404_when_no_plan_exists(self) -> None:
        handler = DebateDecisionBridgeHandler()
        store = MagicMock()
        store.list.return_value = []

        with patch("aragora.server.handlers.debates.bridge._get_plan_store", return_value=store):
            result = handler.handle_post(
                "/api/v1/debates/debate-404/bridge",
                {},
                _http_handler({"target": "n8n"}),
            )

        assert result is not None
        assert result.status_code == 404
        assert "No decision plan found" in _body(result)["error"]

    def test_returns_409_when_task_backed_targets_have_no_tasks(self) -> None:
        handler = DebateDecisionBridgeHandler()
        store = MagicMock()
        store.list.return_value = [
            DecisionPlan(id="plan-1", debate_id="debate-123", task="Ship it")
        ]

        with patch("aragora.server.handlers.debates.bridge._get_plan_store", return_value=store):
            result = handler.handle_post(
                "/api/v1/debates/debate-123/bridge",
                {},
                _http_handler({"target": "jira"}),
            )

        assert result is not None
        assert result.status_code == 409
        assert "has no implementation tasks" in _body(result)["error"]

    def test_uses_latest_plan_and_requested_target(self) -> None:
        handler = DebateDecisionBridgeHandler()
        store = MagicMock()
        latest_plan = DecisionPlan(
            id="plan-new",
            debate_id="debate-123",
            task="Ship it",
            metadata={"integrations": ["jira", "linear"]},
        )
        latest_plan.implement_plan = SimpleNamespace(tasks=[SimpleNamespace(title="Task A")])
        store.list.return_value = [latest_plan]

        bridge_result = SimpleNamespace(
            jira_issues=[],
            linear_issues=[{"id": "LIN-1"}],
            n8n_triggered=False,
            errors=[],
            to_dict=lambda: {
                "jira_issues": [],
                "linear_issues": [{"id": "LIN-1"}],
                "n8n_triggered": False,
                "errors": [],
            },
        )
        bridge = MagicMock()
        bridge.handle_decision_plan = AsyncMock(return_value=bridge_result)

        with (
            patch("aragora.server.handlers.debates.bridge._get_plan_store", return_value=store),
            patch("aragora.integrations.decision_bridge.DecisionBridge", return_value=bridge),
        ):
            result = handler.handle_post(
                "/api/v1/debates/debate-123/bridge",
                {},
                _http_handler({"target": "linear"}),
            )

        assert result is not None
        assert result.status_code == 200
        data = _body(result)
        assert data["success"] is True
        assert data["plan_id"] == "plan-new"
        assert data["target"] == "linear"
        assert data["result"]["linear_issues"] == [{"id": "LIN-1"}]
        bridge.handle_decision_plan.assert_awaited_once()
        bridged_plan = bridge.handle_decision_plan.await_args.args[0]
        assert bridged_plan.metadata["integrations"] == ["linear"]

    def test_returns_502_when_bridge_produces_no_artifact(self) -> None:
        handler = DebateDecisionBridgeHandler()
        store = MagicMock()
        plan = DecisionPlan(id="plan-n8n", debate_id="debate-123", task="Trigger workflow")
        store.list.return_value = [plan]

        bridge_result = SimpleNamespace(
            jira_issues=[],
            linear_issues=[],
            n8n_triggered=False,
            errors=["n8n webhook not configured"],
            to_dict=lambda: {
                "jira_issues": [],
                "linear_issues": [],
                "n8n_triggered": False,
                "errors": ["n8n webhook not configured"],
            },
        )
        bridge = MagicMock()
        bridge.handle_decision_plan = AsyncMock(return_value=bridge_result)

        with (
            patch("aragora.server.handlers.debates.bridge._get_plan_store", return_value=store),
            patch("aragora.integrations.decision_bridge.DecisionBridge", return_value=bridge),
        ):
            result = handler.handle_post(
                "/api/v1/debates/debate-123/bridge",
                {},
                _http_handler({"target": "n8n"}),
            )

        assert result is not None
        assert result.status_code == 502
        data = _body(result)
        assert data["success"] is False
        assert data["error"] == "n8n webhook not configured"
