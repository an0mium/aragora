"""Tests for workflow handler (aragora/server/handlers/workflows/handler.py).

Covers all routes and behavior of the WorkflowHandler class:
- can_handle() routing for all ROUTES
- GET  /api/v1/workflows                                - List workflows
- POST /api/v1/workflows                                - Create workflow
- GET  /api/v1/workflows/{id}                           - Get workflow details
- PATCH /api/v1/workflows/{id}                          - Update workflow
- DELETE /api/v1/workflows/{id}                         - Delete workflow
- POST /api/v1/workflows/{id}/execute                   - Execute workflow
- POST /api/v1/workflows/{id}/simulate                  - Dry-run workflow
- GET  /api/v1/workflows/{id}/status                    - Get execution status
- GET  /api/v1/workflows/{id}/versions                  - Get version history
- POST /api/v1/workflows/{id}/versions/{v}/restore      - Restore version
- GET  /api/v1/workflow-templates                        - List templates
- GET  /api/v1/workflow-approvals                        - List pending approvals
- POST /api/v1/workflow-approvals/{id}/resolve           - Resolve approval
- GET  /api/v1/workflow-executions                       - List executions
- GET  /api/v1/workflow-executions/{id}                  - Get execution details
- DELETE /api/v1/workflow-executions/{id}                - Terminate execution
- PUT  /api/v1/workflows/{id}                           - Update (via PUT)
- Error handling (not found, invalid data, storage errors)
- Edge cases (path rewriting, _normalize_execute_inputs)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.workflows.handler import (
    WorkflowHandler,
    WorkflowHandlers,
    _normalize_execute_inputs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Mock HTTP handler for request simulation."""

    def __init__(
        self,
        body: dict | None = None,
        command: str = "GET",
    ):
        self.command = command
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}
        self.rfile = MagicMock()
        self.client_address = ("127.0.0.1", 12345)
        self.path = ""

        if body is not None:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
            self.headers["Content-Type"] = "application/json"
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"
            self.headers["Content-Type"] = "application/json"


def _sync_run_async(coro):
    """Synchronously resolve an async coroutine.

    AsyncMock coroutines resolve immediately via send(None).
    """
    if asyncio.iscoroutine(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value
        finally:
            coro.close()
    return coro


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

import aragora.server.handlers.workflows.handler as _handler_mod
import aragora.server.handlers.workflows as _wf_pkg


@pytest.fixture
def handler():
    """Create a WorkflowHandler with empty context."""
    return WorkflowHandler(ctx={})


@pytest.fixture(autouse=True)
def _disable_rbac_and_patch_run_async(monkeypatch):
    """Disable RBAC checks and patch _run_async for all tests.

    The WorkflowHandler checks the package-level RBAC_AVAILABLE flag.
    We set it to False so tests don't need JWT tokens.
    We also patch _run_async so coroutines resolve synchronously.
    """
    import aragora.server.handlers.workflows.core as wf_core

    monkeypatch.setattr(_wf_pkg, "RBAC_AVAILABLE", False)
    monkeypatch.setattr(wf_core, "RBAC_AVAILABLE", False)

    # Patch _run_async at the package level so _run_async_fn() picks it up
    monkeypatch.setattr(_wf_pkg, "_run_async", _sync_run_async)


# ============================================================================
# can_handle() routing
# ============================================================================


class TestCanHandle:
    """Verify that can_handle correctly accepts or rejects paths."""

    def test_workflows_root(self, handler):
        assert handler.can_handle("/api/v1/workflows")

    def test_workflows_with_id(self, handler):
        assert handler.can_handle("/api/v1/workflows/wf_abc123")

    def test_workflows_execute(self, handler):
        assert handler.can_handle("/api/v1/workflows/wf_abc123/execute")

    def test_workflows_simulate(self, handler):
        assert handler.can_handle("/api/v1/workflows/wf_abc123/simulate")

    def test_workflows_status(self, handler):
        assert handler.can_handle("/api/v1/workflows/wf_abc123/status")

    def test_workflows_versions(self, handler):
        assert handler.can_handle("/api/v1/workflows/wf_abc123/versions")

    def test_workflow_templates(self, handler):
        assert handler.can_handle("/api/v1/workflow-templates")

    def test_workflows_templates_alias(self, handler):
        assert handler.can_handle("/api/v1/workflows/templates")

    def test_workflow_approvals(self, handler):
        assert handler.can_handle("/api/v1/workflow-approvals")

    def test_workflow_approvals_resolve(self, handler):
        assert handler.can_handle("/api/v1/workflow-approvals/req_123/resolve")

    def test_workflow_executions(self, handler):
        assert handler.can_handle("/api/v1/workflow-executions")

    def test_workflow_executions_with_id(self, handler):
        assert handler.can_handle("/api/v1/workflow-executions/exec_abc")

    def test_workflows_executions_alias(self, handler):
        assert handler.can_handle("/api/v1/workflows/executions")

    def test_templates_registry(self, handler):
        assert handler.can_handle("/api/v1/templates/registry")

    def test_rejects_unrelated_path(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_rejects_billing_path(self, handler):
        assert not handler.can_handle("/api/v1/billing/plans")

    def test_rejects_v2_path(self, handler):
        assert not handler.can_handle("/api/v2/workflows")


# ============================================================================
# _normalize_execute_inputs
# ============================================================================


class TestNormalizeExecuteInputs:
    """Tests for the _normalize_execute_inputs helper."""

    def test_none_payload(self):
        inputs, err = _normalize_execute_inputs(None)
        assert inputs == {}
        assert err is None

    def test_non_dict_payload(self):
        inputs, err = _normalize_execute_inputs("not a dict")
        assert inputs == {}
        assert err == "Request body must be a JSON object"

    def test_flat_payload_becomes_inputs(self):
        inputs, err = _normalize_execute_inputs({"key": "value", "count": 5})
        assert err is None
        assert inputs == {"key": "value", "count": 5}

    def test_nested_inputs_extracted(self):
        inputs, err = _normalize_execute_inputs({"inputs": {"key": "value"}})
        assert err is None
        assert inputs == {"key": "value"}

    def test_non_dict_inputs_rejected(self):
        inputs, err = _normalize_execute_inputs({"inputs": "bad"})
        assert inputs == {}
        assert err == "inputs must be an object"

    def test_compat_keys_merged_into_inputs(self):
        payload = {
            "inputs": {"task": "test"},
            "channel_targets": ["slack:#general"],
            "thread_id": "t123",
        }
        inputs, err = _normalize_execute_inputs(payload)
        assert err is None
        assert inputs["task"] == "test"
        assert inputs["channel_targets"] == ["slack:#general"]
        assert inputs["thread_id"] == "t123"

    def test_compat_keys_not_overwritten(self):
        """If input already has the compat key, don't overwrite it."""
        payload = {
            "inputs": {"thread_id": "from_inputs"},
            "thread_id": "from_top_level",
        }
        inputs, err = _normalize_execute_inputs(payload)
        assert err is None
        assert inputs["thread_id"] == "from_inputs"

    def test_empty_dict_payload(self):
        inputs, err = _normalize_execute_inputs({})
        assert inputs == {}
        assert err is None


# ============================================================================
# GET /api/v1/workflows (list)
# ============================================================================


class TestListWorkflows:
    """Tests for GET /api/v1/workflows."""

    def test_list_workflows_returns_json(self, handler, monkeypatch):
        mock_result = {"workflows": [], "total_count": 0, "limit": 50, "offset": 0}
        monkeypatch.setattr(_handler_mod, "list_workflows", AsyncMock(return_value=mock_result))
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflows", {}, http)
        assert result is not None
        body = _body(result)
        assert "workflows" in body

    def test_list_workflows_with_query_params(self, handler, monkeypatch):
        mock_result = {"workflows": [], "total_count": 0, "limit": 10, "offset": 5}
        monkeypatch.setattr(_handler_mod, "list_workflows", AsyncMock(return_value=mock_result))
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/workflows",
            {"category": "code", "search": "test", "limit": "10", "offset": "5"},
            http,
        )
        assert result is not None
        body = _body(result)
        assert "workflows" in body

    def test_list_workflows_storage_error(self, handler, monkeypatch):
        monkeypatch.setattr(
            _handler_mod, "list_workflows", AsyncMock(side_effect=OSError("disk full"))
        )
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflows", {}, http)
        assert _status(result) == 503


# ============================================================================
# GET /api/v1/workflows/{id}
# ============================================================================


class TestGetWorkflow:
    """Tests for GET /api/v1/workflows/{id}."""

    def test_get_workflow_found(self, handler, monkeypatch):
        mock_wf = {"id": "wf_123", "name": "Test Workflow"}
        monkeypatch.setattr(_handler_mod, "get_workflow", AsyncMock(return_value=mock_wf))
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflows/wf_123", {}, http)
        assert result is not None
        body = _body(result)
        assert body["id"] == "wf_123"

    def test_get_workflow_not_found(self, handler, monkeypatch):
        monkeypatch.setattr(_handler_mod, "get_workflow", AsyncMock(return_value=None))
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflows/wf_missing", {}, http)
        assert _status(result) == 404

    def test_get_workflow_data_error(self, handler, monkeypatch):
        monkeypatch.setattr(
            _handler_mod, "get_workflow", AsyncMock(side_effect=KeyError("bad key"))
        )
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflows/wf_123", {}, http)
        assert _status(result) == 500


# ============================================================================
# POST /api/v1/workflows (create)
# ============================================================================


class TestCreateWorkflow:
    """Tests for POST /api/v1/workflows."""

    def test_create_workflow_success(self, handler, monkeypatch):
        mock_result = {"id": "wf_new", "name": "New Workflow"}
        monkeypatch.setattr(_handler_mod, "create_workflow", AsyncMock(return_value=mock_result))
        body_data = {"name": "New Workflow", "steps": []}
        http = MockHTTPHandler(body=body_data, command="POST")
        result = handler.handle_post("/api/v1/workflows", {}, http)
        assert result is not None
        body = _body(result)
        assert body["id"] == "wf_new"
        assert _status(result) == 201

    def test_create_workflow_invalid_data(self, handler, monkeypatch):
        monkeypatch.setattr(
            _handler_mod, "create_workflow", AsyncMock(side_effect=ValueError("missing name"))
        )
        http = MockHTTPHandler(body={"bad": "data"}, command="POST")
        result = handler.handle_post("/api/v1/workflows", {}, http)
        assert _status(result) == 400

    def test_create_workflow_storage_error(self, handler, monkeypatch):
        monkeypatch.setattr(
            _handler_mod, "create_workflow", AsyncMock(side_effect=OSError("disk full"))
        )
        http = MockHTTPHandler(body={"name": "test"}, command="POST")
        result = handler.handle_post("/api/v1/workflows", {}, http)
        assert _status(result) == 503

    def test_create_workflow_data_error(self, handler, monkeypatch):
        monkeypatch.setattr(
            _handler_mod, "create_workflow", AsyncMock(side_effect=KeyError("missing"))
        )
        http = MockHTTPHandler(body={"name": "test"}, command="POST")
        result = handler.handle_post("/api/v1/workflows", {}, http)
        assert _status(result) == 500


# ============================================================================
# PATCH /api/v1/workflows/{id} (update)
# ============================================================================


class TestUpdateWorkflow:
    """Tests for PATCH /api/v1/workflows/{id}."""

    def test_update_workflow_success(self, handler, monkeypatch):
        mock_result = {"id": "wf_123", "name": "Updated"}
        monkeypatch.setattr(_handler_mod, "update_workflow", AsyncMock(return_value=mock_result))
        http = MockHTTPHandler(body={"name": "Updated"}, command="PATCH")
        result = handler.handle_patch("/api/v1/workflows/wf_123", {}, http)
        assert result is not None
        body = _body(result)
        assert body["name"] == "Updated"

    def test_update_workflow_not_found(self, handler, monkeypatch):
        monkeypatch.setattr(_handler_mod, "update_workflow", AsyncMock(return_value=None))
        http = MockHTTPHandler(body={"name": "test"}, command="PATCH")
        result = handler.handle_patch("/api/v1/workflows/wf_missing", {}, http)
        assert _status(result) == 404

    def test_update_workflow_invalid_data(self, handler, monkeypatch):
        monkeypatch.setattr(
            _handler_mod, "update_workflow", AsyncMock(side_effect=ValueError("bad"))
        )
        http = MockHTTPHandler(body={"name": ""}, command="PATCH")
        result = handler.handle_patch("/api/v1/workflows/wf_123", {}, http)
        assert _status(result) == 400

    def test_update_workflow_storage_error(self, handler, monkeypatch):
        monkeypatch.setattr(_handler_mod, "update_workflow", AsyncMock(side_effect=OSError("disk")))
        http = MockHTTPHandler(body={"name": "test"}, command="PATCH")
        result = handler.handle_patch("/api/v1/workflows/wf_123", {}, http)
        assert _status(result) == 503

    def test_update_workflow_data_error(self, handler, monkeypatch):
        monkeypatch.setattr(
            _handler_mod, "update_workflow", AsyncMock(side_effect=AttributeError("attr"))
        )
        http = MockHTTPHandler(body={"name": "test"}, command="PATCH")
        result = handler.handle_patch("/api/v1/workflows/wf_123", {}, http)
        assert _status(result) == 500


# ============================================================================
# PUT /api/v1/workflows/{id} (update via PUT, delegates to PATCH)
# ============================================================================


class TestPutWorkflow:
    """Tests for PUT /api/v1/workflows/{id}."""

    def test_put_delegates_to_patch(self, handler, monkeypatch):
        mock_result = {"id": "wf_123", "name": "Via PUT"}
        monkeypatch.setattr(_handler_mod, "update_workflow", AsyncMock(return_value=mock_result))
        http = MockHTTPHandler(body={"name": "Via PUT"}, command="PUT")
        result = handler.handle_put("/api/v1/workflows/wf_123", {}, http)
        assert result is not None
        body = _body(result)
        assert body["name"] == "Via PUT"


# ============================================================================
# DELETE /api/v1/workflows/{id}
# ============================================================================


class TestDeleteWorkflow:
    """Tests for DELETE /api/v1/workflows/{id}."""

    def test_delete_workflow_success(self, handler, monkeypatch):
        monkeypatch.setattr(_handler_mod, "delete_workflow", AsyncMock(return_value=True))
        http = MockHTTPHandler(command="DELETE")
        result = handler.handle_delete("/api/v1/workflows/wf_123", {}, http)
        assert result is not None
        body = _body(result)
        assert body["deleted"] is True
        assert body["id"] == "wf_123"

    def test_delete_workflow_not_found(self, handler, monkeypatch):
        monkeypatch.setattr(_handler_mod, "delete_workflow", AsyncMock(return_value=False))
        http = MockHTTPHandler(command="DELETE")
        result = handler.handle_delete("/api/v1/workflows/wf_missing", {}, http)
        assert _status(result) == 404

    def test_delete_workflow_storage_error(self, handler, monkeypatch):
        monkeypatch.setattr(_handler_mod, "delete_workflow", AsyncMock(side_effect=OSError("boom")))
        http = MockHTTPHandler(command="DELETE")
        result = handler.handle_delete("/api/v1/workflows/wf_123", {}, http)
        assert _status(result) == 503

    def test_delete_workflow_data_error(self, handler, monkeypatch):
        monkeypatch.setattr(
            _handler_mod, "delete_workflow", AsyncMock(side_effect=AttributeError("x"))
        )
        http = MockHTTPHandler(command="DELETE")
        result = handler.handle_delete("/api/v1/workflows/wf_123", {}, http)
        assert _status(result) == 500


# ============================================================================
# POST /api/v1/workflows/{id}/execute
# ============================================================================


class TestExecuteWorkflow:
    """Tests for POST /api/v1/workflows/{id}/execute."""

    def test_execute_success(self, handler, monkeypatch):
        mock_result = {"id": "exec_001", "status": "completed"}
        monkeypatch.setattr(_handler_mod, "execute_workflow", AsyncMock(return_value=mock_result))
        http = MockHTTPHandler(body={"inputs": {"key": "val"}}, command="POST")
        result = handler.handle_post("/api/v1/workflows/wf_123/execute", {}, http)
        assert result is not None
        body = _body(result)
        assert body["status"] == "completed"

    def test_execute_not_found(self, handler, monkeypatch):
        monkeypatch.setattr(
            _handler_mod,
            "execute_workflow",
            AsyncMock(side_effect=ValueError("Workflow not found: wf_missing")),
        )
        http = MockHTTPHandler(body={}, command="POST")
        result = handler.handle_post("/api/v1/workflows/wf_missing/execute", {}, http)
        assert _status(result) == 404

    def test_execute_invalid_inputs(self, handler, monkeypatch):
        """Passing non-dict inputs should yield 400."""
        http = MockHTTPHandler(body={"inputs": "bad_string"}, command="POST")
        result = handler.handle_post("/api/v1/workflows/wf_123/execute", {}, http)
        assert _status(result) == 400

    def test_execute_connection_error(self, handler, monkeypatch):
        monkeypatch.setattr(
            _handler_mod,
            "execute_workflow",
            AsyncMock(side_effect=ConnectionError("timeout")),
        )
        http = MockHTTPHandler(body={}, command="POST")
        result = handler.handle_post("/api/v1/workflows/wf_123/execute", {}, http)
        assert _status(result) == 503

    def test_execute_timeout_error(self, handler, monkeypatch):
        monkeypatch.setattr(
            _handler_mod,
            "execute_workflow",
            AsyncMock(side_effect=TimeoutError("timed out")),
        )
        http = MockHTTPHandler(body={}, command="POST")
        result = handler.handle_post("/api/v1/workflows/wf_123/execute", {}, http)
        assert _status(result) == 503

    def test_execute_storage_error(self, handler, monkeypatch):
        monkeypatch.setattr(
            _handler_mod, "execute_workflow", AsyncMock(side_effect=OSError("disk"))
        )
        http = MockHTTPHandler(body={}, command="POST")
        result = handler.handle_post("/api/v1/workflows/wf_123/execute", {}, http)
        assert _status(result) == 503

    def test_execute_data_error(self, handler, monkeypatch):
        monkeypatch.setattr(_handler_mod, "execute_workflow", AsyncMock(side_effect=KeyError("x")))
        http = MockHTTPHandler(body={}, command="POST")
        result = handler.handle_post("/api/v1/workflows/wf_123/execute", {}, http)
        assert _status(result) == 500

    def test_execute_flat_payload(self, handler, monkeypatch):
        """Flat payload (no 'inputs' key) should be treated as inputs."""
        mock_result = {"id": "exec_002", "status": "completed"}
        monkeypatch.setattr(_handler_mod, "execute_workflow", AsyncMock(return_value=mock_result))
        http = MockHTTPHandler(body={"task": "do stuff"}, command="POST")
        result = handler.handle_post("/api/v1/workflows/wf_123/execute", {}, http)
        assert result is not None
        body = _body(result)
        assert body["status"] == "completed"

    def test_execute_with_event_emitter_in_ctx(self, handler, monkeypatch):
        """Execute should pass event_emitter from ctx."""
        mock_result = {"id": "exec_x", "status": "completed"}
        monkeypatch.setattr(_handler_mod, "execute_workflow", AsyncMock(return_value=mock_result))
        emitter = MagicMock()
        handler.ctx = {"event_emitter": emitter}
        http = MockHTTPHandler(body={}, command="POST")
        result = handler.handle_post("/api/v1/workflows/wf_123/execute", {}, http)
        assert result is not None


# ============================================================================
# POST /api/v1/workflows/{id}/simulate
# ============================================================================


class TestSimulateWorkflow:
    """Tests for POST /api/v1/workflows/{id}/simulate (dry-run)."""

    def _make_mock_workflow(self):
        """Create a mock WorkflowDefinition with validate() and steps."""
        wf = MagicMock()
        wf.validate.return_value = (True, [])
        wf.steps = [MagicMock()]
        wf.entry_step = "step_1"

        step = MagicMock()
        step.id = "step_1"
        step.name = "First Step"
        step.step_type = "agent"
        step.optional = False
        step.timeout_seconds = 60
        step.next_steps = []
        wf.get_step.return_value = step
        return wf

    def test_simulate_success(self, handler, monkeypatch):
        mock_wf_dict = {"id": "wf_123", "name": "Test"}
        monkeypatch.setattr(_handler_mod, "get_workflow", AsyncMock(return_value=mock_wf_dict))
        mock_wf = self._make_mock_workflow()
        monkeypatch.setattr(
            _handler_mod.WorkflowDefinition,
            "from_dict",
            MagicMock(return_value=mock_wf),
        )
        http = MockHTTPHandler(body={}, command="POST")
        result = handler.handle_post("/api/v1/workflows/wf_123/simulate", {}, http)
        assert result is not None
        body = _body(result)
        assert body["workflow_id"] == "wf_123"
        assert body["is_valid"] is True
        assert "execution_plan" in body
        assert len(body["execution_plan"]) == 1

    def test_simulate_workflow_not_found(self, handler, monkeypatch):
        monkeypatch.setattr(_handler_mod, "get_workflow", AsyncMock(return_value=None))
        http = MockHTTPHandler(body={}, command="POST")
        result = handler.handle_post("/api/v1/workflows/wf_missing/simulate", {}, http)
        assert _status(result) == 404

    def test_simulate_with_validation_errors(self, handler, monkeypatch):
        mock_wf_dict = {"id": "wf_123", "name": "Bad"}
        monkeypatch.setattr(_handler_mod, "get_workflow", AsyncMock(return_value=mock_wf_dict))
        mock_wf = self._make_mock_workflow()
        mock_wf.validate.return_value = (False, ["missing entry step"])
        monkeypatch.setattr(
            _handler_mod.WorkflowDefinition,
            "from_dict",
            MagicMock(return_value=mock_wf),
        )
        http = MockHTTPHandler(body={}, command="POST")
        result = handler.handle_post("/api/v1/workflows/wf_123/simulate", {}, http)
        assert result is not None
        body = _body(result)
        assert body["is_valid"] is False
        assert "missing entry step" in body["validation_errors"]

    def test_simulate_storage_error(self, handler, monkeypatch):
        monkeypatch.setattr(_handler_mod, "get_workflow", AsyncMock(side_effect=OSError("boom")))
        http = MockHTTPHandler(body={}, command="POST")
        result = handler.handle_post("/api/v1/workflows/wf_123/simulate", {}, http)
        assert _status(result) == 503

    def test_simulate_data_error(self, handler, monkeypatch):
        monkeypatch.setattr(_handler_mod, "get_workflow", AsyncMock(side_effect=TypeError("bad")))
        http = MockHTTPHandler(body={}, command="POST")
        result = handler.handle_post("/api/v1/workflows/wf_123/simulate", {}, http)
        assert _status(result) == 500


# ============================================================================
# GET /api/v1/workflows/{id}/status
# ============================================================================


class TestGetWorkflowStatus:
    """Tests for GET /api/v1/workflows/{id}/status."""

    def test_status_with_execution(self, handler, monkeypatch):
        mock_exec = [{"id": "exec_001", "status": "running"}]
        monkeypatch.setattr(_handler_mod, "list_executions", AsyncMock(return_value=mock_exec))
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflows/wf_123/status", {}, http)
        assert result is not None
        body = _body(result)
        assert body["status"] == "running"

    def test_status_no_executions(self, handler, monkeypatch):
        monkeypatch.setattr(_handler_mod, "list_executions", AsyncMock(return_value=[]))
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflows/wf_123/status", {}, http)
        assert result is not None
        body = _body(result)
        assert body["status"] == "no_executions"

    def test_status_storage_error(self, handler, monkeypatch):
        monkeypatch.setattr(
            _handler_mod, "list_executions", AsyncMock(side_effect=OSError("db down"))
        )
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflows/wf_123/status", {}, http)
        assert _status(result) == 503

    def test_status_data_error(self, handler, monkeypatch):
        monkeypatch.setattr(_handler_mod, "list_executions", AsyncMock(side_effect=KeyError("x")))
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflows/wf_123/status", {}, http)
        assert _status(result) == 500


# ============================================================================
# GET /api/v1/workflows/{id}/versions
# ============================================================================


class TestGetVersions:
    """Tests for GET /api/v1/workflows/{id}/versions."""

    def test_get_versions_success(self, handler, monkeypatch):
        mock_versions = [{"version": "1.0.0"}, {"version": "1.0.1"}]
        monkeypatch.setattr(
            _handler_mod,
            "get_workflow_versions",
            AsyncMock(return_value=mock_versions),
        )
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflows/wf_123/versions", {}, http)
        assert result is not None
        body = _body(result)
        assert body["workflow_id"] == "wf_123"
        assert len(body["versions"]) == 2

    def test_get_versions_storage_error(self, handler, monkeypatch):
        monkeypatch.setattr(
            _handler_mod,
            "get_workflow_versions",
            AsyncMock(side_effect=OSError("disk")),
        )
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflows/wf_123/versions", {}, http)
        assert _status(result) == 503

    def test_get_versions_data_error(self, handler, monkeypatch):
        monkeypatch.setattr(
            _handler_mod,
            "get_workflow_versions",
            AsyncMock(side_effect=TypeError("bad")),
        )
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflows/wf_123/versions", {}, http)
        assert _status(result) == 500


# ============================================================================
# POST /api/v1/workflows/{id}/versions/{v}/restore
# ============================================================================


class TestRestoreVersion:
    """Tests for POST /api/v1/workflows/{id}/versions/{v}/restore."""

    def test_restore_version_success(self, handler, monkeypatch):
        mock_result = {"id": "wf_123", "version": "1.0.0"}
        monkeypatch.setattr(
            _handler_mod,
            "restore_workflow_version",
            AsyncMock(return_value=mock_result),
        )
        http = MockHTTPHandler(body={}, command="POST")
        result = handler.handle_post("/api/v1/workflows/wf_123/versions/1.0.0/restore", {}, http)
        assert result is not None
        body = _body(result)
        assert body["restored"] is True

    def test_restore_version_not_found(self, handler, monkeypatch):
        monkeypatch.setattr(
            _handler_mod,
            "restore_workflow_version",
            AsyncMock(return_value=None),
        )
        http = MockHTTPHandler(body={}, command="POST")
        result = handler.handle_post("/api/v1/workflows/wf_123/versions/0.0.0/restore", {}, http)
        assert _status(result) == 404

    def test_restore_version_storage_error(self, handler, monkeypatch):
        monkeypatch.setattr(
            _handler_mod,
            "restore_workflow_version",
            AsyncMock(side_effect=OSError("disk")),
        )
        http = MockHTTPHandler(body={}, command="POST")
        result = handler.handle_post("/api/v1/workflows/wf_123/versions/1.0.0/restore", {}, http)
        assert _status(result) == 503

    def test_restore_version_data_error(self, handler, monkeypatch):
        monkeypatch.setattr(
            _handler_mod,
            "restore_workflow_version",
            AsyncMock(side_effect=TypeError("bad")),
        )
        http = MockHTTPHandler(body={}, command="POST")
        result = handler.handle_post("/api/v1/workflows/wf_123/versions/1.0.0/restore", {}, http)
        assert _status(result) == 500


# ============================================================================
# GET /api/v1/workflow-templates
# ============================================================================


class TestListTemplates:
    """Tests for GET /api/v1/workflow-templates."""

    def test_list_templates_success(self, handler, monkeypatch):
        mock_templates = [{"id": "tpl_1", "name": "Contract Review"}]
        monkeypatch.setattr(_handler_mod, "list_templates", AsyncMock(return_value=mock_templates))
        # Also mock the catalog import to avoid real imports
        with patch(
            "aragora.workflow.templates.list_templates",
            return_value=[],
            create=True,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/workflow-templates", {}, http)
        assert result is not None
        body = _body(result)
        assert "templates" in body
        assert body["count"] >= 1

    def test_list_templates_storage_error(self, handler, monkeypatch):
        monkeypatch.setattr(_handler_mod, "list_templates", AsyncMock(side_effect=OSError("db")))
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflow-templates", {}, http)
        assert _status(result) == 503

    def test_list_templates_data_error(self, handler, monkeypatch):
        monkeypatch.setattr(
            _handler_mod, "list_templates", AsyncMock(side_effect=AttributeError("bad"))
        )
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflow-templates", {}, http)
        assert _status(result) == 500

    def test_list_templates_via_alias_path(self, handler, monkeypatch):
        """GET /api/v1/workflows/templates should rewrite to /api/v1/workflow-templates."""
        mock_templates = [{"id": "tpl_2"}]
        monkeypatch.setattr(_handler_mod, "list_templates", AsyncMock(return_value=mock_templates))
        with patch(
            "aragora.workflow.templates.list_templates",
            return_value=[],
            create=True,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/workflows/templates", {}, http)
        assert result is not None
        body = _body(result)
        assert "templates" in body


# ============================================================================
# GET /api/v1/workflow-approvals
# ============================================================================


class TestListApprovals:
    """Tests for GET /api/v1/workflow-approvals."""

    def test_list_approvals_success(self, handler, monkeypatch):
        mock_approvals = [{"id": "apr_1", "status": "pending"}]
        mock_fn = AsyncMock(return_value=mock_approvals)
        monkeypatch.setattr(_handler_mod, "list_pending_approvals", mock_fn)
        # Also override at package level since _list_pending_approvals_fn checks there
        monkeypatch.setattr(_wf_pkg, "list_pending_approvals", mock_fn)
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflow-approvals", {}, http)
        assert result is not None
        body = _body(result)
        assert "approvals" in body
        assert body["count"] == 1

    def test_list_approvals_storage_error(self, handler, monkeypatch):
        mock_fn = AsyncMock(side_effect=OSError("db"))
        monkeypatch.setattr(_handler_mod, "list_pending_approvals", mock_fn)
        monkeypatch.setattr(_wf_pkg, "list_pending_approvals", mock_fn)
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflow-approvals", {}, http)
        assert _status(result) == 503

    def test_list_approvals_data_error(self, handler, monkeypatch):
        mock_fn = AsyncMock(side_effect=TypeError("bad"))
        monkeypatch.setattr(_handler_mod, "list_pending_approvals", mock_fn)
        monkeypatch.setattr(_wf_pkg, "list_pending_approvals", mock_fn)
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflow-approvals", {}, http)
        assert _status(result) == 500


# ============================================================================
# POST /api/v1/workflow-approvals/{id}/resolve
# ============================================================================


class TestResolveApproval:
    """Tests for POST /api/v1/workflow-approvals/{id}/resolve."""

    def test_resolve_approval_success(self, handler, monkeypatch):
        monkeypatch.setattr(_handler_mod, "resolve_approval", AsyncMock(return_value=True))
        http = MockHTTPHandler(body={"status": "approved", "notes": "LGTM"}, command="POST")
        result = handler.handle_post("/api/v1/workflow-approvals/req_001/resolve", {}, http)
        assert result is not None
        body = _body(result)
        assert body["resolved"] is True

    def test_resolve_approval_not_found(self, handler, monkeypatch):
        monkeypatch.setattr(_handler_mod, "resolve_approval", AsyncMock(return_value=False))
        http = MockHTTPHandler(body={"status": "approved"}, command="POST")
        result = handler.handle_post("/api/v1/workflow-approvals/req_missing/resolve", {}, http)
        assert _status(result) == 404

    def test_resolve_approval_invalid_status(self, handler, monkeypatch):
        monkeypatch.setattr(
            _handler_mod,
            "resolve_approval",
            AsyncMock(side_effect=ValueError("Invalid status: WRONG")),
        )
        http = MockHTTPHandler(body={"status": "WRONG"}, command="POST")
        result = handler.handle_post("/api/v1/workflow-approvals/req_001/resolve", {}, http)
        assert _status(result) == 400

    def test_resolve_approval_storage_error(self, handler, monkeypatch):
        monkeypatch.setattr(_handler_mod, "resolve_approval", AsyncMock(side_effect=OSError("db")))
        http = MockHTTPHandler(body={"status": "approved"}, command="POST")
        result = handler.handle_post("/api/v1/workflow-approvals/req_001/resolve", {}, http)
        assert _status(result) == 503

    def test_resolve_approval_data_error(self, handler, monkeypatch):
        monkeypatch.setattr(_handler_mod, "resolve_approval", AsyncMock(side_effect=KeyError("x")))
        http = MockHTTPHandler(body={"status": "approved"}, command="POST")
        result = handler.handle_post("/api/v1/workflow-approvals/req_001/resolve", {}, http)
        assert _status(result) == 500


# ============================================================================
# GET /api/v1/workflow-executions
# ============================================================================


class TestListExecutions:
    """Tests for GET /api/v1/workflow-executions."""

    def test_list_executions_success(self, handler, monkeypatch):
        mock_execs = [
            {"id": "exec_1", "status": "completed"},
            {"id": "exec_2", "status": "running"},
        ]
        monkeypatch.setattr(_handler_mod, "list_executions", AsyncMock(return_value=mock_execs))
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflow-executions", {}, http)
        assert result is not None
        body = _body(result)
        assert body["count"] == 2

    def test_list_executions_with_status_filter(self, handler, monkeypatch):
        mock_execs = [
            {"id": "exec_1", "status": "completed"},
            {"id": "exec_2", "status": "running"},
        ]
        monkeypatch.setattr(_handler_mod, "list_executions", AsyncMock(return_value=mock_execs))
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflow-executions", {"status": "running"}, http)
        assert result is not None
        body = _body(result)
        assert body["count"] == 1
        assert body["executions"][0]["status"] == "running"

    def test_list_executions_storage_error(self, handler, monkeypatch):
        monkeypatch.setattr(_handler_mod, "list_executions", AsyncMock(side_effect=OSError("db")))
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflow-executions", {}, http)
        assert _status(result) == 503

    def test_list_executions_via_alias_path(self, handler, monkeypatch):
        """GET /api/v1/workflows/executions -> rewritten to /api/v1/workflow-executions."""
        mock_execs = [{"id": "exec_1", "status": "completed"}]
        monkeypatch.setattr(_handler_mod, "list_executions", AsyncMock(return_value=mock_execs))
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflows/executions", {}, http)
        assert result is not None
        body = _body(result)
        assert body["count"] == 1

    def test_list_executions_data_error(self, handler, monkeypatch):
        monkeypatch.setattr(
            _handler_mod, "list_executions", AsyncMock(side_effect=AttributeError("bad"))
        )
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflow-executions", {}, http)
        assert _status(result) == 500


# ============================================================================
# GET /api/v1/workflow-executions/{id}
# ============================================================================


class TestGetExecution:
    """Tests for GET /api/v1/workflow-executions/{id}."""

    def test_get_execution_found(self, handler, monkeypatch):
        mock_exec = {"id": "exec_001", "status": "completed", "outputs": {}}
        monkeypatch.setattr(_handler_mod, "get_execution", AsyncMock(return_value=mock_exec))
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflow-executions/exec_001", {}, http)
        assert result is not None
        body = _body(result)
        assert body["id"] == "exec_001"

    def test_get_execution_not_found(self, handler, monkeypatch):
        monkeypatch.setattr(_handler_mod, "get_execution", AsyncMock(return_value=None))
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflow-executions/exec_missing", {}, http)
        assert _status(result) == 404

    def test_get_execution_storage_error(self, handler, monkeypatch):
        monkeypatch.setattr(_handler_mod, "get_execution", AsyncMock(side_effect=OSError("db")))
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflow-executions/exec_001", {}, http)
        assert _status(result) == 503

    def test_get_execution_data_error(self, handler, monkeypatch):
        monkeypatch.setattr(
            _handler_mod, "get_execution", AsyncMock(side_effect=AttributeError("bad"))
        )
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflow-executions/exec_001", {}, http)
        assert _status(result) == 500


# ============================================================================
# DELETE /api/v1/workflow-executions/{id} (terminate)
# ============================================================================


class TestTerminateExecution:
    """Tests for DELETE /api/v1/workflow-executions/{id}."""

    def test_terminate_success(self, handler, monkeypatch):
        monkeypatch.setattr(_handler_mod, "terminate_execution", AsyncMock(return_value=True))
        http = MockHTTPHandler(command="DELETE")
        result = handler.handle_delete("/api/v1/workflow-executions/exec_001", {}, http)
        assert result is not None
        body = _body(result)
        assert body["terminated"] is True
        assert body["execution_id"] == "exec_001"

    def test_terminate_cannot_terminate(self, handler, monkeypatch):
        monkeypatch.setattr(_handler_mod, "terminate_execution", AsyncMock(return_value=False))
        http = MockHTTPHandler(command="DELETE")
        result = handler.handle_delete("/api/v1/workflow-executions/exec_001", {}, http)
        assert _status(result) == 400

    def test_terminate_storage_error(self, handler, monkeypatch):
        monkeypatch.setattr(
            _handler_mod, "terminate_execution", AsyncMock(side_effect=OSError("db"))
        )
        http = MockHTTPHandler(command="DELETE")
        result = handler.handle_delete("/api/v1/workflow-executions/exec_001", {}, http)
        assert _status(result) == 503

    def test_terminate_data_error(self, handler, monkeypatch):
        monkeypatch.setattr(
            _handler_mod, "terminate_execution", AsyncMock(side_effect=KeyError("x"))
        )
        http = MockHTTPHandler(command="DELETE")
        result = handler.handle_delete("/api/v1/workflow-executions/exec_001", {}, http)
        assert _status(result) == 500


# ============================================================================
# _extract_id helper
# ============================================================================


class TestExtractId:
    """Tests for the _extract_id path helper."""

    def test_extract_id_from_workflows_path(self, handler):
        assert handler._extract_id("/api/v1/workflows/wf_abc123") == "wf_abc123"

    def test_extract_id_with_suffix(self, handler):
        assert (
            handler._extract_id("/api/v1/workflows/wf_123/execute", suffix="/execute") == "wf_123"
        )

    def test_extract_id_with_versions_suffix(self, handler):
        assert (
            handler._extract_id("/api/v1/workflows/wf_123/versions", suffix="/versions") == "wf_123"
        )

    def test_extract_id_returns_none_for_short_path(self, handler):
        assert handler._extract_id("/api/v1/workflows") is None

    def test_extract_id_returns_none_for_wrong_prefix(self, handler):
        assert handler._extract_id("/api/v1/billing/abc") is None


# ============================================================================
# Handler initialization
# ============================================================================


class TestHandlerInit:
    """Tests for handler initialization."""

    def test_default_context_is_empty_dict(self):
        handler = WorkflowHandler()
        assert handler.ctx == {}

    def test_context_passed_through(self):
        ctx = {"event_emitter": MagicMock()}
        handler = WorkflowHandler(ctx=ctx)
        assert handler.ctx is ctx

    def test_routes_list_populated(self, handler):
        assert len(handler.ROUTES) > 0
        assert "/api/v1/workflows" in handler.ROUTES
        assert "/api/v1/workflow-templates" in handler.ROUTES

    def test_handle_returns_none_for_unhandled_path(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/debates", {}, http)
        assert result is None

    def test_handle_post_returns_none_for_unhandled_path(self, handler):
        http = MockHTTPHandler(body={}, command="POST")
        result = handler.handle_post("/api/v1/debates", {}, http)
        assert result is None

    def test_handle_patch_returns_none_for_unhandled_path(self, handler):
        http = MockHTTPHandler(body={}, command="PATCH")
        result = handler.handle_patch("/api/v1/debates", {}, http)
        assert result is None

    def test_handle_delete_returns_none_for_unhandled_path(self, handler):
        http = MockHTTPHandler(command="DELETE")
        result = handler.handle_delete("/api/v1/debates", {}, http)
        assert result is None


# ============================================================================
# WorkflowHandlers (legacy interface)
# ============================================================================


class TestWorkflowHandlersLegacy:
    """Tests for the WorkflowHandlers legacy static method interface."""

    @pytest.fixture(autouse=True)
    def _patch_workflows_module(self, monkeypatch):
        """Patch the workflows package module-level functions for legacy handlers."""
        self.mock_list = AsyncMock(return_value={"workflows": [], "total_count": 0})
        self.mock_get = AsyncMock(return_value={"id": "wf_1"})
        self.mock_create = AsyncMock(return_value={"id": "wf_new"})
        self.mock_update = AsyncMock(return_value={"id": "wf_1"})
        self.mock_delete = AsyncMock(return_value=True)
        self.mock_execute = AsyncMock(return_value={"id": "exec_1"})
        self.mock_templates = AsyncMock(return_value=[])
        self.mock_approvals = AsyncMock(return_value=[])
        self.mock_resolve = AsyncMock(return_value=True)

        import aragora.server.handlers.workflows as wf_pkg

        monkeypatch.setattr(wf_pkg, "list_workflows", self.mock_list)
        monkeypatch.setattr(wf_pkg, "get_workflow", self.mock_get)
        monkeypatch.setattr(wf_pkg, "create_workflow", self.mock_create)
        monkeypatch.setattr(wf_pkg, "update_workflow", self.mock_update)
        monkeypatch.setattr(wf_pkg, "delete_workflow", self.mock_delete)
        monkeypatch.setattr(wf_pkg, "execute_workflow", self.mock_execute)
        monkeypatch.setattr(wf_pkg, "list_templates", self.mock_templates)
        monkeypatch.setattr(wf_pkg, "list_pending_approvals", self.mock_approvals)
        monkeypatch.setattr(wf_pkg, "resolve_approval", self.mock_resolve)

    @pytest.mark.asyncio
    async def test_handle_list_workflows(self):
        result = await WorkflowHandlers.handle_list_workflows({"limit": 10})
        assert "workflows" in result

    @pytest.mark.asyncio
    async def test_handle_get_workflow(self):
        result = await WorkflowHandlers.handle_get_workflow("wf_1", {})
        assert result["id"] == "wf_1"

    @pytest.mark.asyncio
    async def test_handle_create_workflow(self):
        result = await WorkflowHandlers.handle_create_workflow({"name": "test"}, {})
        assert result["id"] == "wf_new"

    @pytest.mark.asyncio
    async def test_handle_update_workflow(self):
        result = await WorkflowHandlers.handle_update_workflow("wf_1", {"name": "upd"}, {})
        assert result["id"] == "wf_1"

    @pytest.mark.asyncio
    async def test_handle_delete_workflow(self):
        result = await WorkflowHandlers.handle_delete_workflow("wf_1", {})
        assert result is True

    @pytest.mark.asyncio
    async def test_handle_execute_workflow(self):
        result = await WorkflowHandlers.handle_execute_workflow(
            "wf_1", {"inputs": {"key": "val"}}, {}
        )
        assert result["id"] == "exec_1"

    @pytest.mark.asyncio
    async def test_handle_list_templates(self):
        result = await WorkflowHandlers.handle_list_templates({})
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_handle_list_approvals(self):
        result = await WorkflowHandlers.handle_list_approvals({})
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_handle_resolve_approval(self):
        result = await WorkflowHandlers.handle_resolve_approval("req_1", {"status": "approved"}, {})
        assert result is True


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and corner scenarios."""

    def test_handle_get_with_root_path_lists(self, handler, monkeypatch):
        """Requesting /api/v1/workflows with no ID should list, not get."""
        mock_result = {"workflows": [], "total_count": 0, "limit": 50, "offset": 0}
        monkeypatch.setattr(_handler_mod, "list_workflows", AsyncMock(return_value=mock_result))
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/workflows", {}, http)
        assert result is not None
        body = _body(result)
        assert "workflows" in body

    def test_handle_delete_no_id_returns_none(self, handler):
        """DELETE /api/v1/workflows (no ID) should return None (no match)."""
        http = MockHTTPHandler(command="DELETE")
        result = handler.handle_delete("/api/v1/workflows", {}, http)
        assert result is None

    def test_handle_patch_no_id_returns_none(self, handler):
        """PATCH /api/v1/workflows (no ID) should return None."""
        http = MockHTTPHandler(body={}, command="PATCH")
        result = handler.handle_patch("/api/v1/workflows", {}, http)
        assert result is None

    def test_delete_via_executions_alias(self, handler, monkeypatch):
        """DELETE /api/v1/workflows/executions/{id} should rewrite and terminate."""
        monkeypatch.setattr(_handler_mod, "terminate_execution", AsyncMock(return_value=True))
        http = MockHTTPHandler(command="DELETE")
        result = handler.handle_delete("/api/v1/workflows/executions/exec_001", {}, http)
        assert result is not None
        body = _body(result)
        assert body["terminated"] is True
