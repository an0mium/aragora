"""Tests for pipeline template REST endpoints.

Tests GET /api/v1/canvas/pipeline/templates and
POST /api/v1/canvas/pipeline/from-template.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from aragora.server.handlers.canvas_pipeline import CanvasPipelineHandler


@pytest.fixture
def handler():
    return CanvasPipelineHandler()


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.get.return_value = None
    store.save.return_value = None
    with patch(
        "aragora.server.handlers.canvas_pipeline._get_store", return_value=store,
    ):
        yield store


def _parse_result(result):
    """Parse a HandlerResult into status_code and body dict."""
    status = result.status_code
    body = json.loads(result.body) if result.body else {}
    return status, body


class TestListTemplates:
    """Test GET /api/v1/canvas/pipeline/templates."""

    @pytest.mark.asyncio
    async def test_list_templates_returns_all(self, handler):
        result = await handler.handle_list_templates({})
        status, body = _parse_result(result)
        assert status == 200
        assert "templates" in body
        assert body["count"] == 5

    @pytest.mark.asyncio
    async def test_list_templates_filter_by_category(self, handler):
        result = await handler.handle_list_templates({"category": "compliance"})
        status, body = _parse_result(result)
        assert status == 200
        assert body["count"] == 1
        assert body["templates"][0]["name"] == "compliance_audit"

    @pytest.mark.asyncio
    async def test_list_templates_unknown_category_returns_empty(self, handler):
        result = await handler.handle_list_templates({"category": "xyz"})
        status, body = _parse_result(result)
        assert status == 200
        assert body["count"] == 0
        assert body["templates"] == []

    @pytest.mark.asyncio
    async def test_list_templates_no_params(self, handler):
        result = await handler.handle_list_templates(None)
        status, body = _parse_result(result)
        assert status == 200
        assert body["count"] == 5


class TestFromTemplate:
    """Test POST /api/v1/canvas/pipeline/from-template."""

    @pytest.mark.asyncio
    async def test_from_template_missing_name(self, handler, mock_store):
        result = await handler.handle_from_template({})
        status, body = _parse_result(result)
        assert status == 400

    @pytest.mark.asyncio
    async def test_from_template_unknown_template(self, handler, mock_store):
        result = await handler.handle_from_template({"template_name": "nonexistent"})
        status, body = _parse_result(result)
        assert status == 404

    @pytest.mark.asyncio
    async def test_from_template_creates_pipeline(self, handler, mock_store):
        result = await handler.handle_from_template({
            "template_name": "hiring_decision",
            "auto_advance": False,
        })
        status, body = _parse_result(result)
        assert status == 201
        assert "pipeline_id" in body
        assert body["pipeline_id"].startswith("pipe-hiring_decision-")

    @pytest.mark.asyncio
    async def test_from_template_persists_result(self, handler, mock_store):
        await handler.handle_from_template({
            "template_name": "product_launch",
            "auto_advance": False,
        })
        assert mock_store.save.called

    @pytest.mark.asyncio
    async def test_from_template_response_includes_template_info(self, handler, mock_store):
        result = await handler.handle_from_template({
            "template_name": "vendor_selection",
            "auto_advance": False,
        })
        status, body = _parse_result(result)
        assert status == 201
        assert body["template"]["name"] == "vendor_selection"
        assert body["template"]["category"] == "procurement"

    @pytest.mark.asyncio
    async def test_from_template_response_includes_goals_count(self, handler, mock_store):
        result = await handler.handle_from_template({
            "template_name": "market_entry",
            "auto_advance": False,
        })
        status, body = _parse_result(result)
        assert status == 201
        assert "goals_count" in body
        assert body["goals_count"] > 0


class TestCanHandleRoutes:
    """Test route matching for template endpoints."""

    def test_can_handle_canvas_pipeline_path(self, handler):
        assert handler.can_handle("/api/v1/canvas/pipeline/templates")

    def test_routes_include_templates(self, handler):
        assert "GET /api/v1/canvas/pipeline/templates" in handler.ROUTES
        assert "POST /api/v1/canvas/pipeline/from-template" in handler.ROUTES

    def test_handle_dispatches_templates_get(self, handler):
        mock_handler = MagicMock()
        mock_handler.request.body = b"{}"
        result = handler.handle("/api/v1/canvas/pipeline/templates", {}, mock_handler)
        # Should return a coroutine (async handler)
        import asyncio
        assert asyncio.iscoroutine(result)
        # Clean up the coroutine to avoid warning
        result.close()
