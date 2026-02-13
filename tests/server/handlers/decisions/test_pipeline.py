"""Tests for decision pipeline handler validation and RBAC dispatch."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from aragora.server.handlers.base import error_response
from aragora.server.handlers.decisions.pipeline import (
    DECISION_CREATE_PERMISSION,
    DECISION_UPDATE_PERMISSION,
    DecisionPipelineHandler,
)


def _make_http_handler(body: dict) -> MagicMock:
    """Create a mock HTTP handler with a JSON body."""
    payload = json.dumps(body).encode("utf-8")
    handler = MagicMock()
    handler.headers = {"Content-Length": str(len(payload))}
    handler.rfile = io.BytesIO(payload)
    return handler


def _parse_body(result) -> dict:
    """Parse HandlerResult body into a dictionary."""
    body = result.body
    if isinstance(body, bytes):
        return json.loads(body.decode("utf-8"))
    return json.loads(body)


def test_handle_post_create_requires_create_permission() -> None:
    """Create-plan route should enforce decisions:create permission."""
    handler = DecisionPipelineHandler({})
    request = _make_http_handler({"debate_id": "deb-123"})
    user = SimpleNamespace(user_id="user-1")
    permission_error = error_response("forbidden", 403)

    with (
        patch.object(handler, "_check_circuit_breaker", return_value=None),
        patch.object(handler, "require_auth_or_error", return_value=(user, None)),
        patch.object(
            handler,
            "require_permission_or_error",
            return_value=(None, permission_error),
        ) as mock_permission,
    ):
        result = handler.handle_post("/api/v1/decisions/plans", {}, request)

    assert result.status_code == 403
    mock_permission.assert_called_once_with(request, DECISION_CREATE_PERMISSION)


def test_handle_post_execute_requires_update_permission() -> None:
    """Approve/reject/execute routes should enforce decisions:update permission."""
    handler = DecisionPipelineHandler({})
    request = _make_http_handler({})
    user = SimpleNamespace(user_id="user-1")
    permission_error = error_response("forbidden", 403)

    with (
        patch.object(handler, "_check_circuit_breaker", return_value=None),
        patch.object(handler, "require_auth_or_error", return_value=(user, None)),
        patch.object(
            handler,
            "require_permission_or_error",
            return_value=(None, permission_error),
        ) as mock_permission,
    ):
        result = handler.handle_post("/api/v1/decisions/plans/plan-1/execute", {}, request)

    assert result.status_code == 403
    mock_permission.assert_called_once_with(request, DECISION_UPDATE_PERMISSION)


def test_create_plan_rejects_invalid_approval_mode() -> None:
    """Invalid approval_mode should fail fast with 400."""
    handler = DecisionPipelineHandler({})
    request = _make_http_handler(
        {
            "debate_id": "deb-123",
            "approval_mode": "invalid-mode",
        }
    )

    mock_loop = MagicMock()
    mock_loop.run_until_complete.return_value = object()

    with (
        patch("asyncio.get_event_loop", return_value=mock_loop),
        patch(
            "aragora.server.handlers.decisions.pipeline._load_debate_result",
            return_value=object(),
        ),
        patch("aragora.pipeline.decision_plan.DecisionPlanFactory.from_debate_result") as mock_build,
    ):
        result = handler._handle_create_plan(request, SimpleNamespace(user_id="user-1"))

    assert result.status_code == 400
    assert "Invalid approval_mode" in _parse_body(result)["error"]
    mock_build.assert_not_called()


def test_create_plan_rejects_invalid_max_auto_risk() -> None:
    """Invalid max_auto_risk should fail fast with 400."""
    handler = DecisionPipelineHandler({})
    request = _make_http_handler(
        {
            "debate_id": "deb-123",
            "max_auto_risk": "catastrophic",
        }
    )

    mock_loop = MagicMock()
    mock_loop.run_until_complete.return_value = object()

    with (
        patch("asyncio.get_event_loop", return_value=mock_loop),
        patch(
            "aragora.server.handlers.decisions.pipeline._load_debate_result",
            return_value=object(),
        ),
        patch("aragora.pipeline.decision_plan.DecisionPlanFactory.from_debate_result") as mock_build,
    ):
        result = handler._handle_create_plan(request, SimpleNamespace(user_id="user-1"))

    assert result.status_code == 400
    assert "Invalid max_auto_risk" in _parse_body(result)["error"]
    mock_build.assert_not_called()


def test_create_plan_rejects_invalid_budget_limit_usd() -> None:
    """Invalid budget_limit_usd should fail with 400 instead of silently defaulting."""
    handler = DecisionPipelineHandler({})
    request = _make_http_handler(
        {
            "debate_id": "deb-123",
            "budget_limit_usd": "not-a-number",
        }
    )

    mock_loop = MagicMock()
    mock_loop.run_until_complete.return_value = object()

    with (
        patch("asyncio.get_event_loop", return_value=mock_loop),
        patch(
            "aragora.server.handlers.decisions.pipeline._load_debate_result",
            return_value=object(),
        ),
        patch("aragora.pipeline.decision_plan.DecisionPlanFactory.from_debate_result") as mock_build,
    ):
        result = handler._handle_create_plan(request, SimpleNamespace(user_id="user-1"))

    assert result.status_code == 400
    assert "budget_limit_usd" in _parse_body(result)["error"]
    mock_build.assert_not_called()


def test_create_plan_rejects_non_object_metadata() -> None:
    """metadata must be a JSON object."""
    handler = DecisionPipelineHandler({})
    request = _make_http_handler(
        {
            "debate_id": "deb-123",
            "metadata": ["not", "an", "object"],
        }
    )

    mock_loop = MagicMock()
    mock_loop.run_until_complete.return_value = object()

    with (
        patch("asyncio.get_event_loop", return_value=mock_loop),
        patch(
            "aragora.server.handlers.decisions.pipeline._load_debate_result",
            return_value=object(),
        ),
        patch("aragora.pipeline.decision_plan.DecisionPlanFactory.from_debate_result") as mock_build,
    ):
        result = handler._handle_create_plan(request, SimpleNamespace(user_id="user-1"))

    assert result.status_code == 400
    assert "metadata must be an object" in _parse_body(result)["error"]
    mock_build.assert_not_called()


def test_execute_plan_accepts_execution_overrides() -> None:
    """Execute endpoint should forward execution overrides to PlanExecutor."""
    handler = DecisionPipelineHandler({})
    request = _make_http_handler(
        {
            "execution_mode": "hybrid",
            "parallel_execution": True,
            "max_parallel": 4,
        }
    )
    user = SimpleNamespace(
        user_id="user-1",
        role="member",
        roles=["member"],
        permissions=["decisions:update"],
    )

    mock_plan = MagicMock()
    mock_plan.id = "plan-1"
    mock_plan.to_dict.return_value = {"id": "plan-1"}

    mock_outcome = MagicMock()
    mock_outcome.success = True
    mock_outcome.to_dict.return_value = {"success": True}

    mock_executor = MagicMock()
    mock_executor.execute.return_value = "coro"

    mock_loop = MagicMock()
    mock_loop.run_until_complete.return_value = mock_outcome

    with (
        patch("aragora.pipeline.executor.get_plan", return_value=mock_plan),
        patch("aragora.pipeline.executor.PlanExecutor", return_value=mock_executor) as mock_exec_cls,
        patch("asyncio.get_event_loop", return_value=mock_loop),
    ):
        result = handler._handle_execute_plan("plan-1", request, user)

    assert result.status_code == 200
    mock_exec_cls.assert_called_once_with(max_parallel=4)
    kwargs = mock_executor.execute.call_args.kwargs
    assert kwargs["execution_mode"] == "hybrid"
    assert kwargs["parallel_execution"] is True


def test_execute_plan_rejects_invalid_execution_mode() -> None:
    """Execute endpoint should validate execution_mode values."""
    handler = DecisionPipelineHandler({})
    request = _make_http_handler({"execution_mode": "warp-drive"})
    user = SimpleNamespace(user_id="user-1")
    mock_plan = MagicMock()
    mock_plan.id = "plan-1"

    with patch("aragora.pipeline.executor.get_plan", return_value=mock_plan):
        result = handler._handle_execute_plan("plan-1", request, user)

    assert result.status_code == 400
    assert "Invalid execution_mode" in _parse_body(result)["error"]


def test_execute_plan_rejects_invalid_parallel_settings() -> None:
    """Execute endpoint should validate parallel settings."""
    handler = DecisionPipelineHandler({})
    user = SimpleNamespace(user_id="user-1")
    mock_plan = MagicMock()
    mock_plan.id = "plan-1"

    with patch("aragora.pipeline.executor.get_plan", return_value=mock_plan):
        result_parallel = handler._handle_execute_plan(
            "plan-1",
            _make_http_handler({"parallel_execution": "yes"}),
            user,
        )
        result_max_parallel = handler._handle_execute_plan(
            "plan-1",
            _make_http_handler({"max_parallel": 0}),
            user,
        )

    assert result_parallel.status_code == 400
    assert "parallel_execution must be a boolean" in _parse_body(result_parallel)["error"]
    assert result_max_parallel.status_code == 400
    assert "max_parallel must be >= 1" in _parse_body(result_max_parallel)["error"]


def test_execute_plan_normalizes_execution_mode_alias() -> None:
    """Execute endpoint should normalize known execution-mode aliases."""
    handler = DecisionPipelineHandler({})
    request = _make_http_handler({"execution_mode": "workflow_execute"})
    user = SimpleNamespace(user_id="user-1", role="member", roles=["member"], permissions=[])

    mock_plan = MagicMock()
    mock_plan.id = "plan-1"
    mock_plan.to_dict.return_value = {"id": "plan-1"}

    mock_outcome = MagicMock()
    mock_outcome.success = True
    mock_outcome.to_dict.return_value = {"success": True}

    mock_executor = MagicMock()
    mock_executor.execute.return_value = "coro"

    mock_loop = MagicMock()
    mock_loop.run_until_complete.return_value = mock_outcome

    with (
        patch("aragora.pipeline.executor.get_plan", return_value=mock_plan),
        patch("aragora.pipeline.executor.PlanExecutor", return_value=mock_executor),
        patch("asyncio.get_event_loop", return_value=mock_loop),
    ):
        result = handler._handle_execute_plan("plan-1", request, user)

    assert result.status_code == 200
    kwargs = mock_executor.execute.call_args.kwargs
    assert kwargs["execution_mode"] == "workflow"


def test_create_plan_normalizes_profile_execution_mode_alias() -> None:
    """Create-plan endpoint should normalize execution-mode aliases in profile payload."""
    handler = DecisionPipelineHandler({})
    request = _make_http_handler(
        {
            "debate_id": "deb-123",
            "execution_mode": "computer-use",
        }
    )

    mock_plan = MagicMock()
    mock_plan.id = "plan-1"
    mock_plan.debate_id = "deb-123"
    mock_plan.requires_human_approval = False
    mock_plan.status = SimpleNamespace(value="approved")
    mock_plan.to_dict.return_value = {"id": "plan-1"}

    mock_loop = MagicMock()
    mock_loop.run_until_complete.return_value = object()

    with (
        patch("asyncio.get_event_loop", return_value=mock_loop),
        patch(
            "aragora.server.handlers.decisions.pipeline._load_debate_result",
            return_value=object(),
        ),
        patch("aragora.pipeline.decision_plan.DecisionPlanFactory.from_debate_result", return_value=mock_plan) as mock_build,
        patch("aragora.pipeline.executor.store_plan"),
    ):
        result = handler._handle_create_plan(request, SimpleNamespace(user_id="user-1"))

    assert result.status_code == 201
    kwargs = mock_build.call_args.kwargs
    assert kwargs["implementation_profile"]["execution_mode"] == "computer_use"
