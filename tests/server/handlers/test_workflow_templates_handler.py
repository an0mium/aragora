"""
Tests for the WorkflowTemplatesHandler module.

Tests cover:
- Handler routing for template endpoints
- Template listing with filters
- Template details and package retrieval
- Categories and patterns endpoints
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from aragora.server.handlers.workflow_templates import (
    WorkflowTemplatesHandler,
    WorkflowCategoriesHandler,
    WorkflowPatternsHandler,
)


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None}


# =============================================================================
# WorkflowTemplatesHandler Routing Tests
# =============================================================================


class TestWorkflowTemplatesHandlerRouting:
    """Tests for template handler routing."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowTemplatesHandler(mock_server_context)

    def test_can_handle_templates_base(self, handler):
        """Handler can handle templates base path."""
        assert handler.can_handle("/api/workflow/templates")

    def test_can_handle_templates_v1(self, handler):
        """Handler can handle v1 templates path."""
        assert handler.can_handle("/api/v1/workflow/templates")

    def test_can_handle_template_by_id(self, handler):
        """Handler can handle template by ID."""
        assert handler.can_handle("/api/workflow/templates/legal/contract-review")

    def test_can_handle_template_package(self, handler):
        """Handler can handle template package path."""
        assert handler.can_handle("/api/workflow/templates/legal/contract-review/package")

    def test_can_handle_template_run(self, handler):
        """Handler can handle template run path."""
        assert handler.can_handle("/api/workflow/templates/general/research/run")

    def test_cannot_handle_other_paths(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/other")
        assert not handler.can_handle("/api/debates")
        assert not handler.can_handle("/api/workflows")  # Different from templates

    def test_routes_defined(self, handler):
        """Handler has expected routes defined."""
        assert "/api/workflow/templates" in handler.ROUTES
        assert "/api/workflow/templates/*" in handler.ROUTES
        assert "/api/v1/workflow/templates" in handler.ROUTES


# =============================================================================
# WorkflowCategoriesHandler Tests
# =============================================================================


class TestWorkflowCategoriesHandler:
    """Tests for categories handler."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowCategoriesHandler(mock_server_context)

    def test_can_handle_categories(self, handler):
        """Handler can handle categories path."""
        assert handler.can_handle("/api/workflow/categories")

    def test_can_handle_categories_v1(self, handler):
        """Handler can handle v1 categories path."""
        assert handler.can_handle("/api/v1/workflow/categories")

    def test_cannot_handle_other_paths(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/workflow/templates")
        assert not handler.can_handle("/api/other")

    def test_routes_defined(self, handler):
        """Handler has expected routes defined."""
        assert "/api/workflow/categories" in handler.ROUTES
        assert "/api/v1/workflow/categories" in handler.ROUTES


# =============================================================================
# WorkflowPatternsHandler Tests
# =============================================================================


class TestWorkflowPatternsHandler:
    """Tests for patterns handler."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowPatternsHandler(mock_server_context)

    def test_can_handle_patterns(self, handler):
        """Handler can handle patterns path."""
        assert handler.can_handle("/api/workflow/patterns")

    def test_can_handle_patterns_v1(self, handler):
        """Handler can handle v1 patterns path."""
        assert handler.can_handle("/api/v1/workflow/patterns")

    def test_cannot_handle_other_paths(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/workflow/templates")
        assert not handler.can_handle("/api/other")


# =============================================================================
# Handler Response Tests (with mocked dependencies)
# =============================================================================


class TestWorkflowTemplatesHandlerResponses:
    """Tests for handler response generation."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowTemplatesHandler(mock_server_context)

    @pytest.fixture
    def mock_handler_request(self):
        """Create mock HTTP handler."""
        handler = MagicMock()
        handler.command = "GET"
        handler.headers = {"Content-Length": "0"}
        return handler

    def _parse_result(self, result):
        """Parse HandlerResult into body, status, content_type."""
        import json
        body = json.loads(result.body.decode("utf-8"))
        return body, result.status_code, result.content_type

    def test_list_templates_returns_json(self, handler, mock_handler_request):
        """List templates returns JSON response."""
        result = handler.handle(
            "/api/workflow/templates",
            {},
            mock_handler_request,
        )

        assert result is not None
        body, status, content_type = self._parse_result(result)
        assert status == 200
        assert content_type == "application/json"
        assert "templates" in body
        assert "total" in body

    def test_list_templates_with_limit(self, handler, mock_handler_request):
        """List templates respects limit parameter."""
        result = handler.handle(
            "/api/workflow/templates",
            {"limit": ["5"]},
            mock_handler_request,
        )

        assert result is not None
        body, status, _ = self._parse_result(result)
        assert status == 200
        assert body["limit"] == 5

    def test_list_templates_with_offset(self, handler, mock_handler_request):
        """List templates respects offset parameter."""
        result = handler.handle(
            "/api/workflow/templates",
            {"offset": ["10"]},
            mock_handler_request,
        )

        assert result is not None
        body, status, _ = self._parse_result(result)
        assert status == 200
        assert body["offset"] == 10

    def test_list_templates_with_category_filter(self, handler, mock_handler_request):
        """List templates filters by category."""
        result = handler.handle(
            "/api/workflow/templates",
            {"category": ["legal"]},
            mock_handler_request,
        )

        assert result is not None
        body, status, _ = self._parse_result(result)
        assert status == 200

    def test_get_template_not_found(self, handler, mock_handler_request):
        """Get non-existent template returns 404."""
        result = handler.handle(
            "/api/workflow/templates/nonexistent-template",
            {},
            mock_handler_request,
        )

        assert result is not None
        body, status, _ = self._parse_result(result)
        assert status == 404

    def test_method_not_allowed(self, handler, mock_handler_request):
        """Invalid method returns 405."""
        mock_handler_request.command = "DELETE"
        result = handler.handle(
            "/api/workflow/templates",
            {},
            mock_handler_request,
        )

        assert result is not None
        body, status, _ = self._parse_result(result)
        assert status == 405


class TestWorkflowCategoriesHandlerResponses:
    """Tests for categories handler responses."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowCategoriesHandler(mock_server_context)

    @pytest.fixture
    def mock_handler_request(self):
        handler = MagicMock()
        handler.command = "GET"
        return handler

    def _parse_result(self, result):
        """Parse HandlerResult into body, status, content_type."""
        import json
        body = json.loads(result.body.decode("utf-8"))
        return body, result.status_code, result.content_type

    def test_list_categories_returns_json(self, handler, mock_handler_request):
        """List categories returns JSON response."""
        result = handler.handle(
            "/api/workflow/categories",
            {},
            mock_handler_request,
        )

        assert result is not None
        body, status, content_type = self._parse_result(result)
        assert status == 200
        assert content_type == "application/json"
        assert "categories" in body


class TestWorkflowPatternsHandlerResponses:
    """Tests for patterns handler responses."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowPatternsHandler(mock_server_context)

    @pytest.fixture
    def mock_handler_request(self):
        handler = MagicMock()
        handler.command = "GET"
        return handler

    def _parse_result(self, result):
        """Parse HandlerResult into body, status, content_type."""
        import json
        body = json.loads(result.body.decode("utf-8"))
        return body, result.status_code, result.content_type

    def test_list_patterns_returns_json(self, handler, mock_handler_request):
        """List patterns returns JSON response."""
        result = handler.handle(
            "/api/workflow/patterns",
            {},
            mock_handler_request,
        )

        assert result is not None
        body, status, content_type = self._parse_result(result)
        assert status == 200
        assert content_type == "application/json"
        assert "patterns" in body


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestWorkflowTemplatesRateLimiting:
    """Tests for rate limiting behavior."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowTemplatesHandler(mock_server_context)

    def test_rate_limiter_configured(self, handler):
        """Rate limiter is configured."""
        from aragora.server.handlers.workflow_templates import _template_limiter

        assert _template_limiter is not None
        assert _template_limiter.rpm == 60
