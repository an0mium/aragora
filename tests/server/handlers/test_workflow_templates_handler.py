"""
Tests for the WorkflowTemplatesHandler module.

Tests cover:
- Handler routing for template endpoints
- Template listing with filters (category, tag, search, pagination)
- Template details and package retrieval
- Template execution (POST /run)
- Specific template execution (POST /:id/run)
- Categories and patterns endpoints
- Pattern template listing, detail, and instantiation
- Template recommendations based on use case
- SME workflow listing, info, creation, and execution
- Rate limiting across all handlers
- Async workflow execution helper functions
- Error handling (404s, 405s, 400s, 500s, invalid JSON)
"""

from __future__ import annotations

import asyncio
import io
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
import pytest

from aragora.server.handlers.workflow_templates import (
    WorkflowTemplatesHandler,
    WorkflowCategoriesHandler,
    WorkflowPatternsHandler,
    WorkflowPatternTemplatesHandler,
    TemplateRecommendationsHandler,
    SMEWorkflowsHandler,
    USE_CASE_TEMPLATES,
    _execute_workflow_async,
    _start_workflow_execution,
    _template_limiter,
)


# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None}


@pytest.fixture
def mock_handler_get():
    """Create mock HTTP handler for GET requests."""
    handler = MagicMock()
    handler.command = "GET"
    handler.headers = {"Content-Length": "0"}
    handler.client_address = ("127.0.0.1", 12345)
    return handler


@pytest.fixture
def mock_handler_post():
    """Create mock HTTP handler for POST requests with configurable body."""

    def _factory(body: dict | None = None):
        handler = MagicMock()
        handler.command = "POST"
        body_bytes = json.dumps(body or {}).encode("utf-8")
        handler.headers = {"Content-Length": str(len(body_bytes))}
        handler.rfile = io.BytesIO(body_bytes)
        handler.client_address = ("127.0.0.1", 12345)
        return handler

    return _factory


def parse_result(result):
    """Parse HandlerResult into (body_dict, status_code, content_type)."""
    body = json.loads(result.body.decode("utf-8"))
    return body, result.status_code, result.content_type


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
        assert handler.can_handle("/api/v1/workflow/templates")

    def test_can_handle_template_by_id(self, handler):
        """Handler can handle template by ID."""
        assert handler.can_handle("/api/v1/workflow/templates/legal/contract-review")

    def test_can_handle_template_package(self, handler):
        """Handler can handle template package path."""
        assert handler.can_handle("/api/v1/workflow/templates/legal/contract-review/package")

    def test_can_handle_template_run(self, handler):
        """Handler can handle template run path."""
        assert handler.can_handle("/api/v1/workflow/templates/general/research/run")

    def test_cannot_handle_other_paths(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/v1/other")
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/v1/workflows")

    def test_routes_defined(self, handler):
        """Handler has expected routes defined."""
        assert "/api/v1/workflow/templates" in handler.ROUTES
        assert "/api/v1/workflow/templates/*" in handler.ROUTES


# =============================================================================
# WorkflowTemplatesHandler Response Tests
# =============================================================================


class TestWorkflowTemplatesHandlerResponses:
    """Tests for handler response generation with mocked dependencies."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowTemplatesHandler(mock_server_context)

    def test_list_templates_returns_json(self, handler, mock_handler_get):
        """List templates returns JSON response with expected structure."""
        result = handler.handle("/api/v1/workflow/templates", {}, mock_handler_get)
        assert result is not None
        body, status, content_type = parse_result(result)
        assert status == 200
        assert content_type == "application/json"
        assert "templates" in body
        assert "total" in body
        assert "limit" in body
        assert "offset" in body

    def test_list_templates_with_limit(self, handler, mock_handler_get):
        """List templates respects limit parameter."""
        result = handler.handle("/api/v1/workflow/templates", {"limit": ["5"]}, mock_handler_get)
        assert result is not None
        body, status, _ = parse_result(result)
        assert status == 200
        assert body["limit"] == 5

    def test_list_templates_with_offset(self, handler, mock_handler_get):
        """List templates respects offset parameter."""
        result = handler.handle("/api/v1/workflow/templates", {"offset": ["10"]}, mock_handler_get)
        assert result is not None
        body, status, _ = parse_result(result)
        assert status == 200
        assert body["offset"] == 10

    def test_list_templates_with_category_filter(self, handler, mock_handler_get):
        """List templates filters by category."""
        result = handler.handle(
            "/api/v1/workflow/templates", {"category": ["legal"]}, mock_handler_get
        )
        assert result is not None
        body, status, _ = parse_result(result)
        assert status == 200
        assert isinstance(body["templates"], list)

    def test_list_templates_with_tag_filter(self, handler, mock_handler_get):
        """List templates filters by tag."""
        result = handler.handle(
            "/api/v1/workflow/templates", {"tag": ["decision"]}, mock_handler_get
        )
        assert result is not None
        body, status, _ = parse_result(result)
        assert status == 200
        # All returned templates must have the requested tag
        for t in body["templates"]:
            assert "decision" in t.get("tags", [])

    def test_list_templates_with_search_filter(self, handler, mock_handler_get):
        """List templates filters by search query matching name or description."""
        result = handler.handle(
            "/api/v1/workflow/templates", {"search": ["review"]}, mock_handler_get
        )
        assert result is not None
        body, status, _ = parse_result(result)
        assert status == 200

    def test_list_templates_enriched_metadata(self, handler, mock_handler_get):
        """List templates includes enriched metadata like steps_count and pattern."""
        result = handler.handle("/api/v1/workflow/templates", {}, mock_handler_get)
        assert result is not None
        body, status, _ = parse_result(result)
        assert status == 200
        for t in body["templates"]:
            assert "steps_count" in t
            assert "pattern" in t
            assert "estimated_duration" in t

    def test_list_templates_pagination_slices_correctly(self, handler, mock_handler_get):
        """Pagination correctly slices the template list."""
        # First get total
        result_all = handler.handle("/api/v1/workflow/templates", {}, mock_handler_get)
        body_all, _, _ = parse_result(result_all)
        total = body_all["total"]

        if total > 1:
            # Fetch with offset=1, limit=1
            result_page = handler.handle(
                "/api/v1/workflow/templates",
                {"offset": ["1"], "limit": ["1"]},
                mock_handler_get,
            )
            body_page, status, _ = parse_result(result_page)
            assert status == 200
            assert len(body_page["templates"]) <= 1
            assert body_page["total"] == total  # total stays the same

    def test_get_template_not_found(self, handler, mock_handler_get):
        """Get non-existent template returns 404."""
        result = handler.handle(
            "/api/v1/workflow/templates/nonexistent-template", {}, mock_handler_get
        )
        assert result is not None
        body, status, _ = parse_result(result)
        assert status == 404

    def test_get_template_found(self, handler, mock_handler_get):
        """Get existing template returns full details."""
        with patch("aragora.workflow.templates.get_template") as mock_get:
            mock_get.return_value = {
                "name": "Quick Decision",
                "description": "Fast decision making",
                "pattern": "debate",
                "steps": [{"id": "s1", "name": "step1"}],
                "inputs": {"task": "string"},
                "outputs": {"result": "string"},
                "estimated_duration": 5,
                "recommended_agents": ["claude", "gpt"],
                "tags": ["quick", "decision"],
            }
            result = handler._get_template("general/quick-decision")
            assert result is not None
            body, status, _ = parse_result(result)
            assert status == 200
            assert body["name"] == "Quick Decision"
            assert body["category"] == "general"
            assert len(body["steps"]) == 1
            assert body["tags"] == ["quick", "decision"]

    def test_get_package_not_found(self, handler, mock_handler_get):
        """Get package for non-existent template returns 404."""
        result = handler.handle(
            "/api/v1/workflow/templates/nonexistent/package", {}, mock_handler_get
        )
        assert result is not None
        body, status, _ = parse_result(result)
        assert status == 404

    def test_get_package_found(self, handler, mock_handler_get):
        """Get package for existing template returns full package data."""
        with patch("aragora.workflow.templates.get_template") as mock_get_tpl:
            mock_get_tpl.return_value = {
                "name": "Test Template",
                "description": "A test",
                "steps": [],
            }
            mock_package = MagicMock()
            mock_package.to_dict.return_value = {
                "template": {"name": "Test Template"},
                "version": "1.0.0",
                "author": {"name": "Aragora Team"},
            }
            with patch(
                "aragora.workflow.templates.package.create_package",
                return_value=mock_package,
            ):
                result = handler._get_package("general/test")
                assert result is not None
                body, status, _ = parse_result(result)
                assert status == 200
                assert body["version"] == "1.0.0"

    def test_method_not_allowed(self, handler, mock_handler_get):
        """Invalid method on base path returns 405."""
        mock_handler_get.command = "DELETE"
        result = handler.handle("/api/v1/workflow/templates", {}, mock_handler_get)
        assert result is not None
        body, status, _ = parse_result(result)
        assert status == 405

    def test_post_base_path_routes_to_run_template(self, handler, mock_handler_post):
        """POST to base path routes to _run_template."""
        post_handler = mock_handler_post({"template_id": "nonexistent/template"})
        result = handler.handle("/api/v1/workflow/templates", {}, post_handler)
        assert result is not None
        body, status, _ = parse_result(result)
        # Template not found
        assert status == 404

    def test_run_template_missing_template_id(self, handler, mock_handler_post):
        """POST without template_id returns 400."""
        post_handler = mock_handler_post({"inputs": {}})
        result = handler._run_template(post_handler)
        assert result is not None
        body, status, _ = parse_result(result)
        assert status == 400
        assert "template_id is required" in body.get("error", "")

    def test_run_template_invalid_json(self, handler):
        """POST with invalid JSON returns 400."""
        bad_handler = MagicMock()
        bad_handler.headers = {"Content-Length": "10"}
        bad_handler.rfile = io.BytesIO(b"not json!!")
        bad_handler.client_address = ("127.0.0.1", 12345)
        result = handler._run_template(bad_handler)
        assert result is not None
        body, status, _ = parse_result(result)
        assert status == 400
        assert "Invalid JSON" in body.get("error", "")

    def test_run_template_not_found(self, handler, mock_handler_post):
        """Run template returns 404 when template does not exist."""
        post_handler = mock_handler_post({"template_id": "nonexistent"})
        result = handler._run_template(post_handler)
        assert result is not None
        body, status, _ = parse_result(result)
        assert status == 404

    def test_run_template_execution_failure(self, handler, mock_handler_post):
        """Run template returns 500 on execution failure."""
        post_handler = mock_handler_post({"template_id": "test/tmpl", "inputs": {}})
        with patch("aragora.workflow.templates.get_template") as mock_get:
            mock_get.return_value = {"name": "Test", "steps": []}
            with patch("aragora.workflow.engine.WorkflowEngine") as mock_engine_cls:
                mock_engine = MagicMock()
                mock_engine_cls.return_value = mock_engine
                with patch(
                    "aragora.server.handlers.workflow_templates.WorkflowDefinition.from_dict",
                    side_effect=Exception("Parse failed"),
                ):
                    result = handler._run_template(post_handler)
                    assert result is not None
                    body, status, _ = parse_result(result)
                    assert status == 500
                    assert body["status"] == "failed"

    def test_run_specific_template_returns_accepted(self, handler, mock_handler_post):
        """Run specific template returns 202 accepted."""
        post_handler = mock_handler_post({"task": "Analyze data"})
        with patch("aragora.workflow.templates.get_template") as mock_get:
            mock_get.return_value = {"name": "Test", "steps": []}
            result = handler._run_specific_template("test/tmpl", post_handler)
            assert result is not None
            body, status, _ = parse_result(result)
            assert status == 202
            assert body["status"] == "accepted"
            assert body["template_id"] == "test/tmpl"

    def test_run_specific_template_not_found(self, handler, mock_handler_post):
        """Run specific template returns 404 when not found."""
        post_handler = mock_handler_post({})
        result = handler._run_specific_template("nonexistent", post_handler)
        assert result is not None
        body, status, _ = parse_result(result)
        assert status == 404

    def test_run_specific_template_invalid_json(self, handler):
        """Run specific template returns 400 on invalid JSON body."""
        bad_handler = MagicMock()
        bad_handler.headers = {"Content-Length": "5"}
        bad_handler.rfile = io.BytesIO(b"xxxxx")
        bad_handler.client_address = ("127.0.0.1", 12345)
        with patch("aragora.workflow.templates.get_template") as mock_get:
            mock_get.return_value = {"name": "Test", "steps": []}
            result = handler._run_specific_template("test/tmpl", bad_handler)
            assert result is not None
            body, status, _ = parse_result(result)
            assert status == 400

    def test_handle_invalid_path(self, handler, mock_handler_get):
        """Handle returns error for paths with no template ID segments."""
        # A path that starts with the prefix but has no meaningful segments beyond base
        result = handler.handle("/api/v1/workflow/templates", {}, mock_handler_get)
        # This should still work (list templates)
        assert result is not None
        body, status, _ = parse_result(result)
        assert status == 200


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
        assert handler.can_handle("/api/v1/workflow/categories")

    def test_cannot_handle_other_paths(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/v1/workflow/templates")
        assert not handler.can_handle("/api/v1/other")

    def test_routes_defined(self, handler):
        """Handler has expected routes defined."""
        assert "/api/v1/workflow/categories" in handler.ROUTES

    def test_list_categories_returns_json(self, handler, mock_handler_get):
        """List categories returns JSON with categories array."""
        result = handler.handle("/api/v1/workflow/categories", {}, mock_handler_get)
        assert result is not None
        body, status, content_type = parse_result(result)
        assert status == 200
        assert content_type == "application/json"
        assert "categories" in body

    def test_categories_have_expected_fields(self, handler, mock_handler_get):
        """Each category entry has id, name, and template_count."""
        result = handler.handle("/api/v1/workflow/categories", {}, mock_handler_get)
        body, status, _ = parse_result(result)
        assert status == 200
        for cat in body["categories"]:
            assert "id" in cat
            assert "name" in cat
            assert "template_count" in cat
            assert cat["template_count"] > 0

    def test_categories_only_includes_nonempty(self, handler, mock_handler_get):
        """Categories only includes those with at least one template."""
        result = handler.handle("/api/v1/workflow/categories", {}, mock_handler_get)
        body, _, _ = parse_result(result)
        for cat in body["categories"]:
            assert cat["template_count"] > 0


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
        assert handler.can_handle("/api/v1/workflow/patterns")

    def test_cannot_handle_other_paths(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/v1/workflow/templates")
        assert not handler.can_handle("/api/v1/other")

    def test_list_patterns_returns_json(self, handler, mock_handler_get):
        """List patterns returns JSON with patterns array."""
        result = handler.handle("/api/v1/workflow/patterns", {}, mock_handler_get)
        assert result is not None
        body, status, content_type = parse_result(result)
        assert status == 200
        assert content_type == "application/json"
        assert "patterns" in body

    def test_patterns_have_expected_fields(self, handler, mock_handler_get):
        """Each pattern has id, name, description, and available flag."""
        result = handler.handle("/api/v1/workflow/patterns", {}, mock_handler_get)
        body, _, _ = parse_result(result)
        for p in body["patterns"]:
            assert "id" in p
            assert "name" in p
            assert "description" in p
            assert "available" in p
            assert isinstance(p["available"], bool)


# =============================================================================
# WorkflowPatternTemplatesHandler Tests
# =============================================================================


class TestWorkflowPatternTemplatesHandler:
    """Tests for pattern template handler."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowPatternTemplatesHandler(mock_server_context)

    def test_can_handle_pattern_templates(self, handler):
        """Handler can handle pattern-templates path."""
        assert handler.can_handle("/api/v1/workflow/pattern-templates")

    def test_can_handle_specific_pattern(self, handler):
        """Handler can handle specific pattern template path."""
        assert handler.can_handle("/api/v1/workflow/pattern-templates/hive-mind")

    def test_can_handle_instantiate(self, handler):
        """Handler can handle instantiate sub-path."""
        assert handler.can_handle("/api/v1/workflow/pattern-templates/hive-mind/instantiate")

    def test_cannot_handle_unrelated(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/v1/workflow/templates")
        assert not handler.can_handle("/api/v1/other")

    def test_list_pattern_templates(self, handler, mock_handler_get):
        """List pattern templates returns JSON response."""
        mock_handler_get.command = "GET"
        result = handler.handle("/api/v1/workflow/pattern-templates", {}, mock_handler_get)
        assert result is not None
        body, status, _ = parse_result(result)
        assert status == 200
        assert "pattern_templates" in body
        assert "total" in body

    def test_get_pattern_template_not_found(self, handler, mock_handler_get):
        """Get non-existent pattern template returns 404."""
        with patch(
            "aragora.workflow.templates.patterns.get_pattern_template",
            return_value=None,
        ):
            result = handler._get_pattern_template("nonexistent")
            assert result is not None
            body, status, _ = parse_result(result)
            assert status == 404

    def test_get_pattern_template_found(self, handler, mock_handler_get):
        """Get existing pattern template returns full details."""
        mock_template = {
            "id": "hive-mind",
            "name": "Hive Mind",
            "description": "Collaborative pattern",
            "pattern": "hive_mind",
            "version": "2.0.0",
            "config": {"agents": 5},
            "inputs": {"task": "string"},
            "outputs": {"result": "string"},
            "tags": ["collaborative"],
        }
        with patch(
            "aragora.workflow.templates.patterns.get_pattern_template",
            return_value=mock_template,
        ):
            result = handler._get_pattern_template("hive-mind")
            body, status, _ = parse_result(result)
            assert status == 200
            assert body["name"] == "Hive Mind"
            assert body["version"] == "2.0.0"

    def test_get_pattern_template_tries_prefix(self, handler):
        """Handler tries pattern/ prefix when direct lookup fails."""
        call_count = [0]

        def side_effect(pid):
            call_count[0] += 1
            if call_count[0] == 1:
                return None  # First call returns None
            return {"id": f"pattern/{pid.split('/')[-1]}", "name": "Found"}

        with patch(
            "aragora.workflow.templates.patterns.get_pattern_template",
            side_effect=side_effect,
        ):
            result = handler._get_pattern_template("test-pattern")
            body, status, _ = parse_result(result)
            assert status == 200
            assert call_count[0] == 2  # Called twice

    def test_method_not_allowed_for_pattern_templates(self, handler):
        """Non-GET method on list endpoint returns 405."""
        mock_h = MagicMock()
        mock_h.command = "DELETE"
        mock_h.client_address = ("127.0.0.1", 12345)
        result = handler.handle("/api/v1/workflow/pattern-templates", {}, mock_h)
        assert result is not None
        body, status, _ = parse_result(result)
        assert status == 405

    def test_instantiate_hive_mind(self, handler, mock_handler_post):
        """Instantiate hive-mind pattern returns 201 with workflow."""
        mock_workflow = MagicMock()
        mock_workflow.id = "wf_123"
        mock_workflow.name = "Test Hive Mind"
        mock_workflow.description = "A hive mind workflow"
        mock_workflow.steps = []
        mock_workflow.entry_step = "start"
        mock_workflow.tags = ["hive-mind"]
        mock_workflow.metadata = {}

        post_h = mock_handler_post({"name": "Test Hive Mind", "task": "Analyze data"})
        with patch(
            "aragora.workflow.templates.patterns.create_hive_mind_workflow",
            return_value=mock_workflow,
        ):
            result = handler._instantiate_pattern("hive-mind", post_h)
            body, status, _ = parse_result(result)
            assert status == 201
            assert body["status"] == "created"
            assert body["workflow"]["id"] == "wf_123"

    def test_instantiate_map_reduce(self, handler, mock_handler_post):
        """Instantiate map-reduce pattern returns 201."""
        mock_workflow = MagicMock()
        mock_workflow.id = "wf_456"
        mock_workflow.name = "Map Reduce"
        mock_workflow.description = "A map reduce workflow"
        mock_workflow.steps = []
        mock_workflow.entry_step = "map"
        mock_workflow.tags = []
        mock_workflow.metadata = {}

        post_h = mock_handler_post({"name": "Map Reduce", "task": "Process data"})
        with patch(
            "aragora.workflow.templates.patterns.create_map_reduce_workflow",
            return_value=mock_workflow,
        ):
            result = handler._instantiate_pattern("map-reduce", post_h)
            body, status, _ = parse_result(result)
            assert status == 201

    def test_instantiate_review_cycle(self, handler, mock_handler_post):
        """Instantiate review-cycle pattern returns 201."""
        mock_workflow = MagicMock()
        mock_workflow.id = "wf_789"
        mock_workflow.name = "Review Cycle"
        mock_workflow.description = "A review cycle workflow"
        mock_workflow.steps = []
        mock_workflow.entry_step = "review"
        mock_workflow.tags = []
        mock_workflow.metadata = {}

        post_h = mock_handler_post({"name": "Review Cycle", "task": "Review code"})
        with patch(
            "aragora.workflow.templates.patterns.create_review_cycle_workflow",
            return_value=mock_workflow,
        ):
            result = handler._instantiate_pattern("review-cycle", post_h)
            body, status, _ = parse_result(result)
            assert status == 201

    def test_instantiate_unknown_pattern(self, handler, mock_handler_post):
        """Instantiate unknown pattern returns 404."""
        post_h = mock_handler_post({"task": "Do something"})
        result = handler._instantiate_pattern("unknown-pattern", post_h)
        body, status, _ = parse_result(result)
        assert status == 404
        assert "Unknown pattern" in body.get("error", "")

    def test_instantiate_underscore_variant(self, handler, mock_handler_post):
        """Instantiate pattern using underscore variant (hive_mind) works."""
        mock_workflow = MagicMock()
        mock_workflow.id = "wf_u1"
        mock_workflow.name = "Hive Mind"
        mock_workflow.description = ""
        mock_workflow.steps = []
        mock_workflow.entry_step = "start"
        mock_workflow.tags = []
        mock_workflow.metadata = {}

        post_h = mock_handler_post({"name": "Hive Mind", "task": "Test"})
        with patch(
            "aragora.workflow.templates.patterns.create_hive_mind_workflow",
            return_value=mock_workflow,
        ):
            result = handler._instantiate_pattern("hive_mind", post_h)
            body, status, _ = parse_result(result)
            assert status == 201

    def test_instantiate_invalid_json(self, handler):
        """Instantiate pattern with invalid JSON returns 400."""
        bad_handler = MagicMock()
        bad_handler.headers = {"Content-Length": "5"}
        bad_handler.rfile = io.BytesIO(b"oops!")
        bad_handler.client_address = ("127.0.0.1", 12345)
        result = handler._instantiate_pattern("hive-mind", bad_handler)
        body, status, _ = parse_result(result)
        assert status == 400

    def test_instantiate_factory_error(self, handler, mock_handler_post):
        """Instantiate returns 500 when factory function raises."""
        post_h = mock_handler_post({"task": "Fail"})
        with patch(
            "aragora.workflow.templates.patterns.create_hive_mind_workflow",
            side_effect=ValueError("Invalid configuration"),
        ):
            result = handler._instantiate_pattern("hive-mind", post_h)
            body, status, _ = parse_result(result)
            assert status == 500
            assert "Failed to instantiate" in body.get("error", "")


# =============================================================================
# TemplateRecommendationsHandler Tests
# =============================================================================


class TestTemplateRecommendationsHandler:
    """Tests for template recommendations handler."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return TemplateRecommendationsHandler(mock_server_context)

    def test_can_handle_recommended(self, handler):
        """Handler can handle recommended templates path."""
        assert handler.can_handle("/api/v1/templates/recommended")

    def test_can_handle_requires_get(self, handler):
        """Handler only accepts GET method."""
        assert handler.can_handle("/api/v1/templates/recommended", "GET")
        assert not handler.can_handle("/api/v1/templates/recommended", "POST")

    def test_cannot_handle_other_paths(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/v1/workflow/templates")

    def test_get_recommendations_default_use_case(self, handler, mock_handler_get):
        """Get recommendations without use_case returns general recommendations."""
        result = handler.handle("/api/v1/templates/recommended", {}, mock_handler_get)
        assert result is not None
        body, status, _ = parse_result(result)
        assert status == 200
        assert body["use_case"] == "general"
        assert "recommendations" in body
        assert "available_use_cases" in body
        assert "total" in body

    def test_get_recommendations_specific_use_case(self, handler, mock_handler_get):
        """Get recommendations for a specific use case."""
        result = handler.handle(
            "/api/v1/templates/recommended",
            {"use_case": ["team_decisions"]},
            mock_handler_get,
        )
        body, status, _ = parse_result(result)
        assert status == 200
        assert body["use_case"] == "team_decisions"

    def test_get_recommendations_unknown_use_case_falls_back(self, handler, mock_handler_get):
        """Unknown use case falls back to general recommendations."""
        result = handler.handle(
            "/api/v1/templates/recommended",
            {"use_case": ["unknown_use_case"]},
            mock_handler_get,
        )
        body, status, _ = parse_result(result)
        assert status == 200
        # Falls back to general
        assert len(body["recommendations"]) > 0

    def test_get_recommendations_with_limit(self, handler, mock_handler_get):
        """Recommendations respect limit parameter."""
        result = handler.handle(
            "/api/v1/templates/recommended",
            {"limit": ["2"]},
            mock_handler_get,
        )
        body, status, _ = parse_result(result)
        assert status == 200
        assert len(body["recommendations"]) <= 2

    def test_recommendations_include_enriched_fields(self, handler, mock_handler_get):
        """Recommendations include enriched fields like agents_count and rounds."""
        result = handler.handle("/api/v1/templates/recommended", {}, mock_handler_get)
        body, status, _ = parse_result(result)
        assert status == 200
        for rec in body["recommendations"]:
            assert "id" in rec
            assert "name" in rec
            assert "description" in rec
            assert "agents_count" in rec
            assert "rounds" in rec
            assert "estimated_duration_minutes" in rec
            assert "use_case" in rec
            assert "category" in rec

    def test_available_use_cases_match_constant(self, handler, mock_handler_get):
        """Available use cases returned match USE_CASE_TEMPLATES keys."""
        result = handler.handle("/api/v1/templates/recommended", {}, mock_handler_get)
        body, _, _ = parse_result(result)
        assert set(body["available_use_cases"]) == set(USE_CASE_TEMPLATES.keys())

    @pytest.mark.parametrize(
        "use_case",
        [
            "team_decisions",
            "project_planning",
            "vendor_selection",
            "policy_review",
            "technical_decisions",
            "general",
        ],
    )
    def test_all_use_cases_return_results(self, handler, mock_handler_get, use_case):
        """Every documented use case returns at least one recommendation."""
        result = handler.handle(
            "/api/v1/templates/recommended",
            {"use_case": [use_case]},
            mock_handler_get,
        )
        body, status, _ = parse_result(result)
        assert status == 200
        assert len(body["recommendations"]) > 0


# =============================================================================
# SMEWorkflowsHandler Tests
# =============================================================================


class TestSMEWorkflowsHandler:
    """Tests for SME workflows handler."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return SMEWorkflowsHandler(mock_server_context)

    def test_can_handle_sme_workflows(self, handler):
        """Handler can handle SME workflows base path."""
        assert handler.can_handle("/api/v1/sme/workflows")

    def test_can_handle_sme_workflow_type(self, handler):
        """Handler can handle specific SME workflow type path."""
        assert handler.can_handle("/api/v1/sme/workflows/invoice")

    def test_cannot_handle_other(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/v1/workflows")
        assert not handler.can_handle("/api/v1/other")

    def test_list_sme_workflows(self, handler, mock_handler_get):
        """List SME workflows returns all workflow types."""
        result = handler.handle("/api/v1/sme/workflows", {}, mock_handler_get)
        assert result is not None
        body, status, _ = parse_result(result)
        assert status == 200
        assert "workflows" in body
        assert "total" in body
        assert body["total"] == 4
        ids = [w["id"] for w in body["workflows"]]
        assert "invoice" in ids
        assert "followup" in ids
        assert "inventory" in ids
        assert "report" in ids

    def test_list_sme_workflows_structure(self, handler, mock_handler_get):
        """Each SME workflow has expected fields."""
        result = handler.handle("/api/v1/sme/workflows", {}, mock_handler_get)
        body, _, _ = parse_result(result)
        for w in body["workflows"]:
            assert "id" in w
            assert "name" in w
            assert "description" in w
            assert "icon" in w
            assert "category" in w
            assert "inputs" in w
            assert "features" in w

    def test_method_not_allowed_on_list(self, handler):
        """Non-GET/POST method on base path returns 405."""
        mock_h = MagicMock()
        mock_h.command = "DELETE"
        mock_h.client_address = ("127.0.0.1", 12345)
        result = handler.handle("/api/v1/sme/workflows", {}, mock_h)
        assert result is not None
        body, status, _ = parse_result(result)
        assert status == 405

    @pytest.mark.parametrize("workflow_type", ["invoice", "followup", "inventory", "report"])
    def test_get_sme_workflow_info(self, handler, workflow_type):
        """Each SME workflow type returns schema info."""
        result = handler._get_sme_workflow_info(workflow_type)
        assert result is not None
        body, status, _ = parse_result(result)
        assert status == 200
        assert body["id"] == workflow_type
        assert "name" in body
        assert "description" in body
        assert "inputs" in body

    def test_get_sme_workflow_info_unknown_type(self, handler):
        """Unknown SME workflow type returns 404."""
        result = handler._get_sme_workflow_info("nonexistent")
        assert result is not None
        body, status, _ = parse_result(result)
        assert status == 404

    def test_get_invoice_schema_has_required_inputs(self, handler):
        """Invoice schema includes required inputs like customer_id and items."""
        result = handler._get_sme_workflow_info("invoice")
        body, _, _ = parse_result(result)
        assert "customer_id" in body["inputs"]
        assert body["inputs"]["customer_id"]["required"] is True
        assert "items" in body["inputs"]
        assert body["inputs"]["items"]["required"] is True

    def test_create_sme_invoice_workflow(self, handler, mock_handler_post):
        """POST to create invoice workflow returns 201."""
        mock_workflow = MagicMock()
        mock_workflow.id = "wf_inv_001"
        mock_workflow.name = "Invoice Workflow"
        mock_workflow.steps = [MagicMock(), MagicMock()]

        post_h = mock_handler_post(
            {
                "customer_id": "cust_123",
                "items": [{"name": "Widget", "quantity": 1, "unit_price": 10.0}],
            }
        )
        with patch(
            "aragora.workflow.templates.sme.create_invoice_workflow",
            return_value=mock_workflow,
        ):
            result = handler._create_sme_workflow("invoice", post_h)
            body, status, _ = parse_result(result)
            assert status == 201
            assert body["workflow_type"] == "invoice"
            assert body["status"] == "created"
            assert body["steps_count"] == 2

    def test_create_sme_followup_workflow(self, handler, mock_handler_post):
        """POST to create followup workflow returns 201."""
        mock_workflow = MagicMock()
        mock_workflow.id = "wf_fu_001"
        mock_workflow.name = "Followup"
        mock_workflow.steps = [MagicMock()]

        post_h = mock_handler_post({"followup_type": "post_sale"})
        with patch(
            "aragora.workflow.templates.sme.create_followup_workflow",
            return_value=mock_workflow,
        ):
            result = handler._create_sme_workflow("followup", post_h)
            body, status, _ = parse_result(result)
            assert status == 201
            assert body["workflow_type"] == "followup"

    def test_create_sme_inventory_workflow(self, handler, mock_handler_post):
        """POST to create inventory workflow returns 201."""
        mock_workflow = MagicMock()
        mock_workflow.id = "wf_inv_alert"
        mock_workflow.name = "Inventory Alerts"
        mock_workflow.steps = []

        post_h = mock_handler_post({"alert_threshold": 10})
        with patch(
            "aragora.workflow.templates.sme.create_inventory_alert_workflow",
            return_value=mock_workflow,
        ):
            result = handler._create_sme_workflow("inventory", post_h)
            body, status, _ = parse_result(result)
            assert status == 201

    def test_create_sme_report_workflow(self, handler, mock_handler_post):
        """POST to create report workflow returns 201."""
        mock_workflow = MagicMock()
        mock_workflow.id = "wf_rpt"
        mock_workflow.name = "Report"
        mock_workflow.steps = [MagicMock()]

        post_h = mock_handler_post({"report_type": "sales"})
        with patch(
            "aragora.workflow.templates.sme.create_report_workflow",
            return_value=mock_workflow,
        ):
            result = handler._create_sme_workflow("report", post_h)
            body, status, _ = parse_result(result)
            assert status == 201

    def test_create_sme_unknown_type(self, handler, mock_handler_post):
        """POST for unknown SME workflow type returns 400."""
        post_h = mock_handler_post({})
        result = handler._create_sme_workflow("unknown_type", post_h)
        body, status, _ = parse_result(result)
        assert status == 400

    def test_create_sme_invalid_json(self, handler):
        """POST with invalid JSON returns 400."""
        bad_handler = MagicMock()
        bad_handler.headers = {"Content-Length": "5"}
        bad_handler.rfile = io.BytesIO(b"badjson")
        bad_handler.client_address = ("127.0.0.1", 12345)
        result = handler._create_sme_workflow("invoice", bad_handler)
        body, status, _ = parse_result(result)
        assert status == 400

    def test_create_sme_workflow_with_execution(self, handler, mock_handler_post):
        """POST with execute=True starts async execution."""
        mock_workflow = MagicMock()
        mock_workflow.id = "wf_exec"
        mock_workflow.name = "Exec Workflow"
        mock_workflow.steps = []

        post_h = mock_handler_post(
            {
                "customer_id": "cust_x",
                "items": [],
                "execute": True,
            }
        )
        with patch(
            "aragora.workflow.templates.sme.create_invoice_workflow",
            return_value=mock_workflow,
        ):
            with patch(
                "aragora.server.handlers.workflow_templates._start_workflow_execution",
                return_value="exec_abc123",
            ) as mock_start:
                result = handler._create_sme_workflow("invoice", post_h)
                body, status, _ = parse_result(result)
                assert status == 201
                assert body["status"] == "running"
                assert body["execution_id"] == "exec_abc123"
                assert "poll_url" in body
                mock_start.assert_called_once()

    def test_create_sme_workflow_factory_error(self, handler, mock_handler_post):
        """POST returns 500 when factory raises exception."""
        post_h = mock_handler_post({"customer_id": "x", "items": []})
        with patch(
            "aragora.workflow.templates.sme.create_invoice_workflow",
            side_effect=RuntimeError("Factory exploded"),
        ):
            result = handler._create_sme_workflow("invoice", post_h)
            body, status, _ = parse_result(result)
            assert status == 500
            assert "Failed to create workflow" in body.get("error", "")


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestRateLimiting:
    """Tests for rate limiting behavior across handlers."""

    def test_rate_limiter_configured(self):
        """Rate limiter is configured with 60 requests per minute."""
        assert _template_limiter is not None
        assert _template_limiter.rpm == 60

    def test_templates_handler_rate_limited(self, mock_server_context, mock_handler_get):
        """WorkflowTemplatesHandler returns 429 when rate limited."""
        handler = WorkflowTemplatesHandler(mock_server_context)
        with patch(
            "aragora.server.handlers.workflow_templates._template_limiter.is_allowed",
            return_value=False,
        ):
            result = handler.handle("/api/v1/workflow/templates", {}, mock_handler_get)
            assert result is not None
            body, status, _ = parse_result(result)
            assert status == 429

    def test_pattern_templates_handler_rate_limited(self, mock_server_context, mock_handler_get):
        """WorkflowPatternTemplatesHandler returns 429 when rate limited."""
        handler = WorkflowPatternTemplatesHandler(mock_server_context)
        with patch(
            "aragora.server.handlers.workflow_templates._template_limiter.is_allowed",
            return_value=False,
        ):
            result = handler.handle("/api/v1/workflow/pattern-templates", {}, mock_handler_get)
            assert result is not None
            body, status, _ = parse_result(result)
            assert status == 429

    def test_recommendations_handler_rate_limited(self, mock_server_context, mock_handler_get):
        """TemplateRecommendationsHandler returns 429 when rate limited."""
        handler = TemplateRecommendationsHandler(mock_server_context)
        with patch(
            "aragora.server.handlers.workflow_templates._template_limiter.is_allowed",
            return_value=False,
        ):
            result = handler.handle("/api/v1/templates/recommended", {}, mock_handler_get)
            assert result is not None
            body, status, _ = parse_result(result)
            assert status == 429

    def test_sme_workflows_handler_rate_limited(self, mock_server_context, mock_handler_get):
        """SMEWorkflowsHandler returns 429 when rate limited."""
        handler = SMEWorkflowsHandler(mock_server_context)
        with patch(
            "aragora.server.handlers.workflow_templates._template_limiter.is_allowed",
            return_value=False,
        ):
            result = handler.handle("/api/v1/sme/workflows", {}, mock_handler_get)
            assert result is not None
            body, status, _ = parse_result(result)
            assert status == 429


# =============================================================================
# Async Workflow Execution Tests
# =============================================================================


class TestAsyncWorkflowExecution:
    """Tests for async workflow execution helper functions."""

    def test_start_workflow_execution_creates_record(self):
        """_start_workflow_execution creates execution record and returns ID."""
        mock_store = MagicMock()
        mock_workflow = MagicMock()
        mock_workflow.id = "wf_test"
        mock_workflow.name = "Test Workflow"

        with patch(
            "aragora.server.handlers.workflow_templates._get_workflow_store",
            return_value=mock_store,
        ):
            with patch(
                "aragora.server.handlers.workflow_templates.asyncio.get_running_loop"
            ) as mock_loop_fn:
                mock_loop = MagicMock()
                mock_loop_fn.return_value = mock_loop

                exec_id = _start_workflow_execution(mock_workflow, {"task": "test"})

                assert exec_id.startswith("exec_")
                mock_store.save_execution.assert_called_once()
                saved = mock_store.save_execution.call_args[0][0]
                assert saved["id"] == exec_id
                assert saved["workflow_id"] == "wf_test"
                assert saved["status"] == "running"
                assert saved["inputs"] == {"task": "test"}

    def test_start_workflow_execution_no_running_loop(self):
        """_start_workflow_execution handles no running event loop."""
        mock_store = MagicMock()
        mock_workflow = MagicMock()
        mock_workflow.id = "wf_noloop"
        mock_workflow.name = "No Loop"

        with patch(
            "aragora.server.handlers.workflow_templates._get_workflow_store",
            return_value=mock_store,
        ):
            with patch(
                "aragora.server.handlers.workflow_templates.asyncio.get_running_loop",
                side_effect=RuntimeError("No running loop"),
            ):
                with patch(
                    "aragora.server.handlers.workflow_templates.asyncio.create_task"
                ) as mock_create:
                    exec_id = _start_workflow_execution(mock_workflow)
                    assert exec_id.startswith("exec_")
                    mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_workflow_async_success(self):
        """_execute_workflow_async updates store on successful execution."""
        mock_store = MagicMock()
        mock_execution = {"id": "exec_1", "status": "running"}
        mock_store.get_execution.return_value = mock_execution

        mock_engine = AsyncMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.final_output = {"answer": "42"}
        mock_result.steps = []
        mock_result.error = None
        mock_result.total_duration_ms = 1500
        mock_engine.execute.return_value = mock_result

        with patch(
            "aragora.server.handlers.workflow_templates._get_workflow_store",
            return_value=mock_store,
        ):
            with patch(
                "aragora.server.handlers.workflow_templates._get_workflow_engine",
                return_value=mock_engine,
            ):
                await _execute_workflow_async(MagicMock(), "exec_1", {"task": "test"})

                mock_store.save_execution.assert_called_once()
                assert mock_execution["status"] == "completed"
                assert mock_execution["outputs"] == {"answer": "42"}

    @pytest.mark.asyncio
    async def test_execute_workflow_async_failure(self):
        """_execute_workflow_async updates store with failed status on exception."""
        mock_store = MagicMock()
        mock_execution = {"id": "exec_2", "status": "running"}
        mock_store.get_execution.return_value = mock_execution

        mock_engine = AsyncMock()
        mock_engine.execute.side_effect = RuntimeError("Boom")

        with patch(
            "aragora.server.handlers.workflow_templates._get_workflow_store",
            return_value=mock_store,
        ):
            with patch(
                "aragora.server.handlers.workflow_templates._get_workflow_engine",
                return_value=mock_engine,
            ):
                await _execute_workflow_async(MagicMock(), "exec_2", {})

                mock_store.save_execution.assert_called_once()
                assert mock_execution["status"] == "failed"
                assert "Boom" in mock_execution["error"]

    @pytest.mark.asyncio
    async def test_execute_workflow_async_result_failed(self):
        """_execute_workflow_async marks status as failed when result.success is False."""
        mock_store = MagicMock()
        mock_execution = {"id": "exec_3", "status": "running"}
        mock_store.get_execution.return_value = mock_execution

        mock_engine = AsyncMock()
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.final_output = None
        mock_result.steps = []
        mock_result.error = "Validation failed"
        mock_result.total_duration_ms = 200
        mock_engine.execute.return_value = mock_result

        with patch(
            "aragora.server.handlers.workflow_templates._get_workflow_store",
            return_value=mock_store,
        ):
            with patch(
                "aragora.server.handlers.workflow_templates._get_workflow_engine",
                return_value=mock_engine,
            ):
                await _execute_workflow_async(MagicMock(), "exec_3", {})

                mock_store.save_execution.assert_called_once()
                assert mock_execution["status"] == "failed"

    @pytest.mark.asyncio
    async def test_execute_workflow_async_no_execution_record(self):
        """_execute_workflow_async handles missing execution record gracefully."""
        mock_store = MagicMock()
        mock_store.get_execution.return_value = None

        mock_engine = AsyncMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.final_output = {}
        mock_result.steps = []
        mock_result.error = None
        mock_result.total_duration_ms = 100
        mock_engine.execute.return_value = mock_result

        with patch(
            "aragora.server.handlers.workflow_templates._get_workflow_store",
            return_value=mock_store,
        ):
            with patch(
                "aragora.server.handlers.workflow_templates._get_workflow_engine",
                return_value=mock_engine,
            ):
                # Should not raise even if execution record is missing
                await _execute_workflow_async(MagicMock(), "exec_missing", {})
                mock_store.save_execution.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_workflow_async_exception_no_record(self):
        """_execute_workflow_async handles exception when no execution record exists."""
        mock_store = MagicMock()
        mock_store.get_execution.return_value = None

        mock_engine = AsyncMock()
        mock_engine.execute.side_effect = RuntimeError("Error")

        with patch(
            "aragora.server.handlers.workflow_templates._get_workflow_store",
            return_value=mock_store,
        ):
            with patch(
                "aragora.server.handlers.workflow_templates._get_workflow_engine",
                return_value=mock_engine,
            ):
                # Should not raise
                await _execute_workflow_async(MagicMock(), "exec_no_rec", {})
                mock_store.save_execution.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_workflow_async_with_steps(self):
        """_execute_workflow_async serializes step results correctly."""
        mock_store = MagicMock()
        mock_execution = {"id": "exec_steps", "status": "running"}
        mock_store.get_execution.return_value = mock_execution

        mock_step = MagicMock()
        mock_step.step_id = "s1"
        mock_step.step_name = "Step One"
        mock_step.status = MagicMock()
        mock_step.status.value = "completed"
        mock_step.duration_ms = 500
        mock_step.error = None

        mock_engine = AsyncMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.final_output = {"data": "result"}
        mock_result.steps = [mock_step]
        mock_result.error = None
        mock_result.total_duration_ms = 600
        mock_engine.execute.return_value = mock_result

        with patch(
            "aragora.server.handlers.workflow_templates._get_workflow_store",
            return_value=mock_store,
        ):
            with patch(
                "aragora.server.handlers.workflow_templates._get_workflow_engine",
                return_value=mock_engine,
            ):
                await _execute_workflow_async(MagicMock(), "exec_steps", {})

                mock_store.save_execution.assert_called_once()
                steps = mock_execution["steps"]
                assert len(steps) == 1
                assert steps[0]["step_id"] == "s1"
                assert steps[0]["step_name"] == "Step One"
                assert steps[0]["status"] == "completed"
                assert steps[0]["duration_ms"] == 500


# =============================================================================
# USE_CASE_TEMPLATES Constant Tests
# =============================================================================


class TestUseCaseTemplatesConstant:
    """Tests for the USE_CASE_TEMPLATES constant structure."""

    def test_all_expected_use_cases_present(self):
        """All expected use cases are present in the mapping."""
        expected = {
            "team_decisions",
            "project_planning",
            "vendor_selection",
            "policy_review",
            "technical_decisions",
            "general",
        }
        assert set(USE_CASE_TEMPLATES.keys()) == expected

    def test_each_use_case_has_templates(self):
        """Each use case has at least one template recommendation."""
        for use_case, templates in USE_CASE_TEMPLATES.items():
            assert len(templates) > 0, f"Use case '{use_case}' has no templates"

    def test_template_recommendations_have_required_fields(self):
        """Each template recommendation has id, name, and description."""
        for use_case, templates in USE_CASE_TEMPLATES.items():
            for t in templates:
                assert "id" in t, f"Missing 'id' in {use_case}"
                assert "name" in t, f"Missing 'name' in {use_case}"
                assert "description" in t, f"Missing 'description' in {use_case}"

    def test_template_ids_follow_category_format(self):
        """Template IDs follow category/name format."""
        for use_case, templates in USE_CASE_TEMPLATES.items():
            for t in templates:
                assert "/" in t["id"], (
                    f"Template ID '{t['id']}' in '{use_case}' does not follow category/name format"
                )
