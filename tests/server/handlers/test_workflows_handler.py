"""
Tests for the WorkflowHandler module.

Tests cover:
- Handler routing for all workflow endpoints
- ID extraction from paths
- Route handling and can_handle method
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from aragora.server.handlers.workflows import WorkflowHandler


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None}


class TestWorkflowHandlerRouting:
    """Tests for handler routing."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    def test_can_handle_workflows(self, handler):
        """Handler can handle workflows base path."""
        assert handler.can_handle("/api/workflows")

    def test_can_handle_workflow_by_id(self, handler):
        """Handler can handle workflow by ID."""
        assert handler.can_handle("/api/workflows/wf_123")

    def test_can_handle_workflow_execute(self, handler):
        """Handler can handle workflow execute."""
        assert handler.can_handle("/api/workflows/wf_123/execute")

    def test_can_handle_workflow_simulate(self, handler):
        """Handler can handle workflow simulate."""
        assert handler.can_handle("/api/workflows/wf_123/simulate")

    def test_can_handle_workflow_versions(self, handler):
        """Handler can handle workflow versions."""
        assert handler.can_handle("/api/workflows/wf_123/versions")

    def test_can_handle_workflow_status(self, handler):
        """Handler can handle workflow status."""
        assert handler.can_handle("/api/workflows/wf_123/status")

    def test_can_handle_templates(self, handler):
        """Handler can handle templates."""
        assert handler.can_handle("/api/workflow-templates")

    def test_can_handle_approvals(self, handler):
        """Handler can handle approvals."""
        assert handler.can_handle("/api/workflow-approvals")

    def test_can_handle_executions(self, handler):
        """Handler can handle executions."""
        assert handler.can_handle("/api/workflow-executions")

    def test_cannot_handle_other_paths(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/other")
        assert not handler.can_handle("/api/debates")


class TestWorkflowHandlerIdExtraction:
    """Tests for ID extraction from paths."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    def test_extract_id_basic(self, handler):
        """Extract ID from basic path."""
        id_ = handler._extract_id("/api/workflows/wf_123")
        assert id_ == "wf_123"

    def test_extract_id_with_suffix(self, handler):
        """Extract ID with suffix removal."""
        id_ = handler._extract_id("/api/workflows/wf_123/execute", suffix="/execute")
        assert id_ == "wf_123"

    def test_extract_id_no_id(self, handler):
        """Extract ID returns None for base path."""
        id_ = handler._extract_id("/api/workflows")
        assert id_ is None

    def test_extract_id_with_versions_suffix(self, handler):
        """Extract ID with versions suffix."""
        id_ = handler._extract_id("/api/workflows/wf_abc123/versions", suffix="/versions")
        assert id_ == "wf_abc123"


class TestWorkflowHandlerRouteDispatch:
    """Tests for route dispatch logic."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    def test_handle_dispatches_list_workflows(self, handler):
        """Handle dispatches /api/workflows to list handler."""
        mock_http = MagicMock()

        result = handler.handle("/api/workflows", {}, mock_http)

        # Result should be returned
        assert result is not None

    def test_handle_dispatches_templates(self, handler):
        """Handle dispatches /api/workflow-templates to template handler."""
        mock_http = MagicMock()

        result = handler.handle("/api/workflow-templates", {}, mock_http)

        assert result is not None

    def test_handle_dispatches_approvals(self, handler):
        """Handle dispatches /api/workflow-approvals to approval handler."""
        mock_http = MagicMock()

        result = handler.handle("/api/workflow-approvals", {}, mock_http)

        assert result is not None

    def test_handle_dispatches_executions(self, handler):
        """Handle dispatches /api/workflow-executions to execution handler."""
        mock_http = MagicMock()

        result = handler.handle("/api/workflow-executions", {}, mock_http)

        assert result is not None


class TestWorkflowHandlerUnknownPath:
    """Tests for unknown path handling."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    def test_unhandled_post_path(self, handler):
        """Unhandled POST path returns None."""
        mock_http = MagicMock()
        mock_http.rfile = MagicMock()
        mock_http.headers = {"Content-Length": "2", "Content-Type": "application/json"}
        mock_http.rfile.read.return_value = b"{}"

        result = handler.handle_post("/api/other", {}, mock_http)

        assert result is None

    def test_unhandled_delete_path(self, handler):
        """Unhandled DELETE path returns None."""
        mock_http = MagicMock()

        result = handler.handle_delete("/api/other", {}, mock_http)

        assert result is None

    def test_unhandled_patch_path(self, handler):
        """Unhandled PATCH path returns None."""
        mock_http = MagicMock()
        mock_http.rfile = MagicMock()
        mock_http.headers = {"Content-Length": "2", "Content-Type": "application/json"}
        mock_http.rfile.read.return_value = b"{}"

        result = handler.handle_patch("/api/other", {}, mock_http)

        assert result is None
