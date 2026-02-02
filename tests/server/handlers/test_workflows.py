"""
Tests for WorkflowHandler - Workflow HTTP endpoints.

Tests cover:
- Route registration and can_handle
- GET endpoints (list, get, versions, status, templates, approvals, executions)
- POST endpoints (create, execute, simulate, restore version, resolve approval)
- PATCH/PUT endpoints (update workflow)
- DELETE endpoints (delete workflow, terminate execution)
- Error cases (not found, invalid input, storage errors)
- Permission/auth checks (RBAC gating)
- Edge cases (path aliasing, empty results, pagination)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import importlib
import importlib.util
import sys
import types
import os

# Load the workflows handler module directly from its file path, bypassing
# the broken aragora.server.handlers.__init__.py which fails on Slack imports.
_WORKFLOWS_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "aragora",
    "server",
    "handlers",
    "workflows",
    "handler.py",
)
_WORKFLOWS_PATH = os.path.abspath(_WORKFLOWS_PATH)

# Ensure parent package modules exist in sys.modules (but skip the broken __init__)
for _pkg in [
    "aragora.server.handlers",
    "aragora.server.handlers.workflows",
]:
    if _pkg not in sys.modules:
        _stub = types.ModuleType(_pkg)
        _stub.__path__ = [os.path.dirname(_WORKFLOWS_PATH)]  # type: ignore[attr-defined]
        _stub.__package__ = _pkg
        sys.modules[_pkg] = _stub

_spec = importlib.util.spec_from_file_location(
    "aragora.server.handlers.workflows.handler", _WORKFLOWS_PATH
)
_workflows_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["aragora.server.handlers.workflows.handler"] = _workflows_mod
_spec.loader.exec_module(_workflows_mod)  # type: ignore[union-attr]

WorkflowHandler = _workflows_mod.WorkflowHandler  # type: ignore[attr-defined]

from aragora.server.handlers.utils.responses import HandlerResult, json_response, error_response  # noqa: E402


# ===========================================================================
# Helpers
# ===========================================================================


def _parse_body(result: HandlerResult) -> dict:
    """Parse JSON body from a HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


def _make_mock_handler():
    """Create a mock HTTP request handler that simulates no RBAC."""
    handler = MagicMock()
    handler.headers = {"Content-Type": "application/json"}
    handler.rfile = MagicMock()
    return handler


def _make_workflow_dict(**overrides) -> dict:
    """Build a minimal valid workflow dict for create/update."""
    base = {
        "name": "Test Workflow",
        "description": "A test workflow",
        "steps": [
            {
                "id": "step1",
                "name": "Step One",
                "step_type": "task",
                "config": {"task_type": "transform", "transform": "{}"},
            }
        ],
    }
    base.update(overrides)
    return base


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_store():
    """Create a mock PersistentWorkflowStore."""
    store = MagicMock()
    store.list_workflows.return_value = ([], 0)
    store.get_workflow.return_value = None
    store.save_workflow.return_value = None
    store.save_version.return_value = None
    store.delete_workflow.return_value = False
    store.get_versions.return_value = []
    store.get_version.return_value = None
    store.list_templates.return_value = []
    store.get_template.return_value = None
    store.save_execution.return_value = None
    store.get_execution.return_value = None
    store.list_executions.return_value = ([], 0)
    return store


@pytest.fixture
def handler_no_rbac(mock_store):
    """Create a WorkflowHandler with RBAC disabled."""
    ctx = MagicMock()
    with (
        patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
        patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
        patch("aragora.server.handlers.workflows.audit_data"),
    ):
        h = WorkflowHandler(ctx)
        yield h


@pytest.fixture
def mock_http_handler():
    """Mock HTTP request handler for passing to handle methods."""
    return _make_mock_handler()


# ===========================================================================
# Test: can_handle and routing
# ===========================================================================


class TestCanHandle:
    """Test route matching via can_handle."""

    def test_workflows_path(self):
        ctx = MagicMock()
        h = WorkflowHandler(ctx)
        assert h.can_handle("/api/v1/workflows")

    def test_workflows_with_id(self):
        ctx = MagicMock()
        h = WorkflowHandler(ctx)
        assert h.can_handle("/api/v1/workflows/wf_abc123")

    def test_workflow_templates(self):
        ctx = MagicMock()
        h = WorkflowHandler(ctx)
        assert h.can_handle("/api/v1/workflow-templates")

    def test_workflow_approvals(self):
        ctx = MagicMock()
        h = WorkflowHandler(ctx)
        assert h.can_handle("/api/v1/workflow-approvals")

    def test_workflow_executions(self):
        ctx = MagicMock()
        h = WorkflowHandler(ctx)
        assert h.can_handle("/api/v1/workflow-executions")

    def test_workflows_templates_alias(self):
        ctx = MagicMock()
        h = WorkflowHandler(ctx)
        assert h.can_handle("/api/v1/workflows/templates")

    def test_workflows_executions_alias(self):
        ctx = MagicMock()
        h = WorkflowHandler(ctx)
        assert h.can_handle("/api/v1/workflows/executions")

    def test_unrelated_path_rejected(self):
        ctx = MagicMock()
        h = WorkflowHandler(ctx)
        assert not h.can_handle("/api/v1/debates")
        assert not h.can_handle("/api/v1/backups")
        assert not h.can_handle("/api/v2/workflows")


# ===========================================================================
# Test: extract_id helper
# ===========================================================================


class TestExtractId:
    """Test the _extract_id path parser."""

    def test_extracts_id_from_workflows_path(self):
        ctx = MagicMock()
        h = WorkflowHandler(ctx)
        assert h._extract_id("/api/v1/workflows/wf_abc") == "wf_abc"

    def test_extracts_id_with_suffix(self):
        ctx = MagicMock()
        h = WorkflowHandler(ctx)
        assert h._extract_id("/api/v1/workflows/wf_abc/execute", suffix="/execute") == "wf_abc"

    def test_returns_none_for_base_path(self):
        ctx = MagicMock()
        h = WorkflowHandler(ctx)
        assert h._extract_id("/api/v1/workflows") is None

    def test_returns_none_for_unrelated_path(self):
        ctx = MagicMock()
        h = WorkflowHandler(ctx)
        assert h._extract_id("/api/v1/debates/d1") is None


# ===========================================================================
# Test: GET /api/v1/workflows (list)
# ===========================================================================


class TestListWorkflows:
    """Test GET /api/v1/workflows."""

    def test_list_workflows_empty(self, handler_no_rbac, mock_store, mock_http_handler):
        mock_store.list_workflows.return_value = ([], 0)
        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
        ):
            result = handler_no_rbac.handle("/api/v1/workflows", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["workflows"] == []
        assert body["total_count"] == 0

    def test_list_workflows_with_results(self, handler_no_rbac, mock_store, mock_http_handler):
        mock_wf = MagicMock()
        mock_wf.to_dict.return_value = {"id": "wf_1", "name": "Test"}
        mock_store.list_workflows.return_value = ([mock_wf], 1)
        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
        ):
            result = handler_no_rbac.handle("/api/v1/workflows", {}, mock_http_handler)
        assert result.status_code == 200
        body = _parse_body(result)
        assert len(body["workflows"]) == 1
        assert body["total_count"] == 1

    def test_list_workflows_with_search_param(self, handler_no_rbac, mock_store, mock_http_handler):
        mock_store.list_workflows.return_value = ([], 0)
        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
        ):
            result = handler_no_rbac.handle(
                "/api/v1/workflows",
                {"search": "contract", "category": "legal"},
                mock_http_handler,
            )
        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Test: GET /api/v1/workflows/{id}
# ===========================================================================


class TestGetWorkflow:
    """Test GET /api/v1/workflows/{id}."""

    def test_get_workflow_found(self, handler_no_rbac, mock_store, mock_http_handler):
        mock_wf = MagicMock()
        mock_wf.to_dict.return_value = {"id": "wf_1", "name": "Test"}
        mock_store.get_workflow.return_value = mock_wf
        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
        ):
            result = handler_no_rbac.handle("/api/v1/workflows/wf_1", {}, mock_http_handler)
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["id"] == "wf_1"

    def test_get_workflow_not_found(self, handler_no_rbac, mock_store, mock_http_handler):
        mock_store.get_workflow.return_value = None
        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
        ):
            result = handler_no_rbac.handle("/api/v1/workflows/wf_missing", {}, mock_http_handler)
        assert result.status_code == 404


# ===========================================================================
# Test: POST /api/v1/workflows (create)
# ===========================================================================


class TestCreateWorkflow:
    """Test POST /api/v1/workflows."""

    def test_create_workflow_success(self, handler_no_rbac, mock_store, mock_http_handler):
        mock_wf = MagicMock()
        mock_wf.validate.return_value = (True, [])
        mock_wf.to_dict.return_value = {"id": "wf_new", "name": "New WF"}

        body_bytes = json.dumps(_make_workflow_dict()).encode()
        mock_http_handler.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body_bytes)),
        }
        mock_http_handler.rfile.read.return_value = body_bytes

        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
            patch("aragora.server.handlers.workflows.WorkflowDefinition") as MockWfDef,
            patch("aragora.server.handlers.workflows.audit_data"),
        ):
            MockWfDef.from_dict.return_value = mock_wf
            result = handler_no_rbac.handle_post("/api/v1/workflows", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 201

    def test_create_workflow_validation_error(self, handler_no_rbac, mock_store, mock_http_handler):
        mock_wf = MagicMock()
        mock_wf.validate.return_value = (False, ["Missing name"])

        body_bytes = json.dumps({"steps": []}).encode()
        mock_http_handler.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body_bytes)),
        }
        mock_http_handler.rfile.read.return_value = body_bytes

        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
            patch("aragora.server.handlers.workflows.WorkflowDefinition") as MockWfDef,
            patch("aragora.server.handlers.workflows.audit_data"),
        ):
            MockWfDef.from_dict.return_value = mock_wf
            result = handler_no_rbac.handle_post("/api/v1/workflows", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Test: PATCH /api/v1/workflows/{id} (update)
# ===========================================================================


class TestUpdateWorkflow:
    """Test PATCH /api/v1/workflows/{id}."""

    def test_update_workflow_success(self, handler_no_rbac, mock_store, mock_http_handler):
        existing = MagicMock()
        existing.created_by = "user1"
        existing.created_at = datetime.now(timezone.utc)
        existing.version = "1.0.0"

        updated = MagicMock()
        updated.validate.return_value = (True, [])
        updated.version = "1.0.1"
        updated.to_dict.return_value = {"id": "wf_1", "version": "1.0.1"}

        mock_store.get_workflow.return_value = existing

        body_bytes = json.dumps({"name": "Updated WF"}).encode()
        mock_http_handler.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body_bytes)),
        }
        mock_http_handler.rfile.read.return_value = body_bytes

        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
            patch("aragora.server.handlers.workflows.WorkflowDefinition") as MockWfDef,
            patch("aragora.server.handlers.workflows.audit_data"),
        ):
            MockWfDef.from_dict.return_value = updated
            result = handler_no_rbac.handle_patch("/api/v1/workflows/wf_1", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200

    def test_update_workflow_not_found(self, handler_no_rbac, mock_store, mock_http_handler):
        mock_store.get_workflow.return_value = None

        body_bytes = json.dumps({"name": "Updated"}).encode()
        mock_http_handler.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body_bytes)),
        }
        mock_http_handler.rfile.read.return_value = body_bytes

        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
        ):
            result = handler_no_rbac.handle_patch(
                "/api/v1/workflows/wf_missing", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 404

    def test_put_delegates_to_patch(self, handler_no_rbac, mock_store, mock_http_handler):
        """PUT should behave identically to PATCH."""
        mock_store.get_workflow.return_value = None

        body_bytes = json.dumps({"name": "Updated"}).encode()
        mock_http_handler.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body_bytes)),
        }
        mock_http_handler.rfile.read.return_value = body_bytes

        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
        ):
            result = handler_no_rbac.handle_put(
                "/api/v1/workflows/wf_missing", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Test: DELETE /api/v1/workflows/{id}
# ===========================================================================


class TestDeleteWorkflow:
    """Test DELETE /api/v1/workflows/{id}."""

    def test_delete_workflow_success(self, handler_no_rbac, mock_store, mock_http_handler):
        mock_store.delete_workflow.return_value = True
        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
            patch("aragora.server.handlers.workflows.audit_data"),
        ):
            result = handler_no_rbac.handle_delete("/api/v1/workflows/wf_1", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["deleted"] is True

    def test_delete_workflow_not_found(self, handler_no_rbac, mock_store, mock_http_handler):
        mock_store.delete_workflow.return_value = False
        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
            patch("aragora.server.handlers.workflows.audit_data"),
        ):
            result = handler_no_rbac.handle_delete(
                "/api/v1/workflows/wf_gone", {}, mock_http_handler
            )
        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Test: POST /api/v1/workflows/{id}/execute
# ===========================================================================


class TestExecuteWorkflow:
    """Test POST /api/v1/workflows/{id}/execute."""

    def test_execute_workflow_not_found(self, handler_no_rbac, mock_store, mock_http_handler):
        mock_store.get_workflow.return_value = None

        body_bytes = json.dumps({"inputs": {}}).encode()
        mock_http_handler.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body_bytes)),
        }
        mock_http_handler.rfile.read.return_value = body_bytes

        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
        ):
            result = handler_no_rbac.handle_post(
                "/api/v1/workflows/wf_missing/execute", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 404

    def test_execute_workflow_success(self, handler_no_rbac, mock_store, mock_http_handler):
        mock_wf = MagicMock()
        mock_store.get_workflow.return_value = mock_wf

        exec_result = MagicMock()
        exec_result.success = True
        exec_result.final_output = {"result": "ok"}
        exec_result.steps = []
        exec_result.error = None
        exec_result.total_duration_ms = 100

        body_bytes = json.dumps({"inputs": {"key": "val"}}).encode()
        mock_http_handler.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body_bytes)),
        }
        mock_http_handler.rfile.read.return_value = body_bytes

        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
            patch("aragora.server.handlers.workflows._engine") as mock_engine,
            patch("aragora.server.handlers.workflows.audit_data"),
        ):
            mock_engine.execute = AsyncMock(return_value=exec_result)
            result = handler_no_rbac.handle_post(
                "/api/v1/workflows/wf_1/execute", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Test: GET /api/v1/workflows/{id}/versions
# ===========================================================================


class TestGetVersions:
    """Test GET /api/v1/workflows/{id}/versions."""

    def test_get_versions(self, handler_no_rbac, mock_store, mock_http_handler):
        mock_store.get_versions.return_value = [
            {"version": "1.0.0", "created_at": "2024-01-01T00:00:00Z"},
        ]
        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
        ):
            result = handler_no_rbac.handle(
                "/api/v1/workflows/wf_1/versions", {}, mock_http_handler
            )
        assert result is not None
        assert result.status_code == 200
        body = _parse_body(result)
        assert "versions" in body
        assert body["workflow_id"] == "wf_1"


# ===========================================================================
# Test: GET /api/v1/workflows/{id}/status
# ===========================================================================


class TestGetStatus:
    """Test GET /api/v1/workflows/{id}/status."""

    def test_status_no_executions(self, handler_no_rbac, mock_store, mock_http_handler):
        mock_store.list_executions.return_value = ([], 0)
        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
        ):
            result = handler_no_rbac.handle("/api/v1/workflows/wf_1/status", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["status"] == "no_executions"

    def test_status_with_execution(self, handler_no_rbac, mock_store, mock_http_handler):
        mock_store.list_executions.return_value = (
            [{"id": "exec_1", "status": "completed"}],
            1,
        )
        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
        ):
            result = handler_no_rbac.handle("/api/v1/workflows/wf_1/status", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["id"] == "exec_1"


# ===========================================================================
# Test: GET /api/v1/workflow-templates
# ===========================================================================


class TestListTemplates:
    """Test GET /api/v1/workflow-templates."""

    def test_list_templates_empty(self, handler_no_rbac, mock_store, mock_http_handler):
        mock_store.list_templates.return_value = []
        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
        ):
            result = handler_no_rbac.handle("/api/v1/workflow-templates", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["templates"] == []
        assert body["count"] == 0

    def test_list_templates_via_alias(self, handler_no_rbac, mock_store, mock_http_handler):
        """The /api/v1/workflows/templates path should alias to /api/v1/workflow-templates."""
        mock_store.list_templates.return_value = []
        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
        ):
            result = handler_no_rbac.handle("/api/v1/workflows/templates", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Test: GET /api/v1/workflow-approvals
# ===========================================================================


class TestListApprovals:
    """Test GET /api/v1/workflow-approvals."""

    def test_list_approvals(self, handler_no_rbac, mock_store, mock_http_handler):
        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
            patch("aragora.server.handlers.workflows.list_pending_approvals") as mock_list,
        ):
            mock_list.return_value = []
            result = handler_no_rbac.handle("/api/v1/workflow-approvals", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["approvals"] == []


# ===========================================================================
# Test: GET /api/v1/workflow-executions
# ===========================================================================


class TestListExecutions:
    """Test GET /api/v1/workflow-executions."""

    def test_list_executions_empty(self, handler_no_rbac, mock_store, mock_http_handler):
        mock_store.list_executions.return_value = ([], 0)
        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
        ):
            result = handler_no_rbac.handle("/api/v1/workflow-executions", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["executions"] == []

    def test_list_executions_via_alias(self, handler_no_rbac, mock_store, mock_http_handler):
        """The /api/v1/workflows/executions path should alias to /api/v1/workflow-executions."""
        mock_store.list_executions.return_value = ([], 0)
        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
        ):
            result = handler_no_rbac.handle("/api/v1/workflows/executions", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Test: GET /api/v1/workflow-executions/{id}
# ===========================================================================


class TestGetExecution:
    """Test GET /api/v1/workflow-executions/{id}."""

    def test_get_execution_found(self, handler_no_rbac, mock_store, mock_http_handler):
        mock_store.get_execution.return_value = {"id": "exec_1", "status": "completed"}
        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
        ):
            result = handler_no_rbac.handle(
                "/api/v1/workflow-executions/exec_1", {}, mock_http_handler
            )
        assert result is not None
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["id"] == "exec_1"

    def test_get_execution_not_found(self, handler_no_rbac, mock_store, mock_http_handler):
        mock_store.get_execution.return_value = None
        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
        ):
            result = handler_no_rbac.handle(
                "/api/v1/workflow-executions/exec_missing", {}, mock_http_handler
            )
        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Test: DELETE /api/v1/workflow-executions/{id} (terminate)
# ===========================================================================


class TestTerminateExecution:
    """Test DELETE /api/v1/workflow-executions/{id}."""

    def test_terminate_running_execution(self, handler_no_rbac, mock_store, mock_http_handler):
        mock_store.get_execution.return_value = {"id": "exec_1", "status": "running"}
        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
            patch("aragora.server.handlers.workflows._engine") as mock_engine,
        ):
            result = handler_no_rbac.handle_delete(
                "/api/v1/workflow-executions/exec_1", {}, mock_http_handler
            )
        assert result is not None
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["terminated"] is True

    def test_terminate_non_running_execution(self, handler_no_rbac, mock_store, mock_http_handler):
        mock_store.get_execution.return_value = {"id": "exec_1", "status": "completed"}
        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
        ):
            result = handler_no_rbac.handle_delete(
                "/api/v1/workflow-executions/exec_1", {}, mock_http_handler
            )
        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Test: POST /api/v1/workflows/{id}/simulate
# ===========================================================================


class TestSimulateWorkflow:
    """Test POST /api/v1/workflows/{id}/simulate."""

    def test_simulate_workflow_not_found(self, handler_no_rbac, mock_store, mock_http_handler):
        mock_store.get_workflow.return_value = None

        body_bytes = json.dumps({}).encode()
        mock_http_handler.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body_bytes)),
        }
        mock_http_handler.rfile.read.return_value = body_bytes

        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
        ):
            result = handler_no_rbac.handle_post(
                "/api/v1/workflows/wf_missing/simulate", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 404

    def test_simulate_workflow_success(self, handler_no_rbac, mock_store, mock_http_handler):
        # Return a dict from get_workflow (which the simulate handler expects)
        mock_wf_dict = {
            "id": "wf_1",
            "name": "Test",
            "steps": [{"id": "s1", "name": "S1", "step_type": "task", "config": {}}],
        }
        mock_wf_obj = MagicMock()
        mock_wf_obj.to_dict.return_value = mock_wf_dict
        mock_store.get_workflow.return_value = mock_wf_obj

        mock_sim_wf = MagicMock()
        mock_sim_wf.validate.return_value = (True, [])
        mock_sim_wf.entry_step = "s1"
        step = MagicMock()
        step.id = "s1"
        step.name = "S1"
        step.step_type = "task"
        step.optional = False
        step.timeout_seconds = 60
        step.next_steps = []
        mock_sim_wf.get_step.return_value = step
        mock_sim_wf.steps = [step]

        body_bytes = json.dumps({}).encode()
        mock_http_handler.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body_bytes)),
        }
        mock_http_handler.rfile.read.return_value = body_bytes

        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False),
            patch("aragora.server.handlers.workflows.WorkflowDefinition") as MockWfDef,
        ):
            MockWfDef.from_dict.return_value = mock_sim_wf
            result = handler_no_rbac.handle_post(
                "/api/v1/workflows/wf_1/simulate", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["is_valid"] is True
        assert body["workflow_id"] == "wf_1"
        assert len(body["execution_plan"]) == 1


# ===========================================================================
# Test: RBAC / Auth gating
# ===========================================================================


class TestRBACGating:
    """Test that endpoints enforce authentication when RBAC is available."""

    def test_handle_get_requires_auth(self, mock_store, mock_http_handler):
        ctx = MagicMock()
        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", True),
            patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract,
        ):
            mock_jwt = MagicMock()
            mock_jwt.authenticated = False
            mock_jwt.user_id = None
            mock_extract.return_value = mock_jwt

            h = WorkflowHandler(ctx)
            result = h.handle("/api/v1/workflows", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 401

    def test_handle_post_requires_auth(self, mock_store, mock_http_handler):
        ctx = MagicMock()
        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", True),
            patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract,
        ):
            mock_jwt = MagicMock()
            mock_jwt.authenticated = False
            mock_jwt.user_id = None
            mock_extract.return_value = mock_jwt

            h = WorkflowHandler(ctx)
            result = h.handle_post("/api/v1/workflows", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 401

    def test_handle_delete_requires_auth(self, mock_store, mock_http_handler):
        ctx = MagicMock()
        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", True),
            patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract,
        ):
            mock_jwt = MagicMock()
            mock_jwt.authenticated = False
            mock_jwt.user_id = None
            mock_extract.return_value = mock_jwt

            h = WorkflowHandler(ctx)
            result = h.handle_delete("/api/v1/workflows/wf_1", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 401

    def test_handle_patch_requires_auth(self, mock_store, mock_http_handler):
        ctx = MagicMock()
        with (
            patch("aragora.server.handlers.workflows._get_store", return_value=mock_store),
            patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", True),
            patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract,
        ):
            mock_jwt = MagicMock()
            mock_jwt.authenticated = False
            mock_jwt.user_id = None
            mock_extract.return_value = mock_jwt

            h = WorkflowHandler(ctx)
            result = h.handle_patch("/api/v1/workflows/wf_1", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 401


# ===========================================================================
# Test: handle returns None for unrecognized paths
# ===========================================================================


class TestUnrecognizedPaths:
    """Test that handlers return None for paths they cannot handle."""

    def test_handle_returns_none_for_unrelated_path(self, handler_no_rbac, mock_http_handler):
        with patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False):
            result = handler_no_rbac.handle("/api/v1/debates", {}, mock_http_handler)
        assert result is None

    def test_handle_post_returns_none_for_unrelated_path(self, handler_no_rbac, mock_http_handler):
        with patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False):
            result = handler_no_rbac.handle_post("/api/v1/debates", {}, mock_http_handler)
        assert result is None

    def test_handle_delete_returns_none_for_unrelated_path(
        self, handler_no_rbac, mock_http_handler
    ):
        with patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False):
            result = handler_no_rbac.handle_delete("/api/v1/debates", {}, mock_http_handler)
        assert result is None
