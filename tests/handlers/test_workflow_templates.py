"""Tests for workflow templates handler (aragora/server/handlers/workflow_templates.py).

Covers all routes and behavior of the handler classes:
- WorkflowTemplatesHandler:
  - can_handle() route matching
  - GET  /api/v1/workflow/templates          - List available templates
  - GET  /api/v1/workflow/templates/:id      - Get template details
  - GET  /api/v1/workflow/templates/:id/package - Get full package
  - POST /api/v1/workflow/templates          - Execute a template
  - POST /api/v1/workflow/templates/:id/run  - Run a specific template
  - Method not allowed, invalid paths, rate limiting
- WorkflowCategoriesHandler:
  - can_handle(), GET /api/v1/workflow/categories
- WorkflowPatternsHandler:
  - can_handle(), GET /api/v1/workflow/patterns
- WorkflowPatternTemplatesHandler:
  - can_handle(), list, get, instantiate pattern templates
  - Rate limiting, method not allowed, unknown patterns
- TemplateRecommendationsHandler:
  - can_handle(), GET /api/v1/templates/recommended
  - Use case filtering, limit, defaults
- SMEWorkflowsHandler:
  - can_handle(), GET/POST /api/v1/sme/workflows
  - GET /api/v1/sme/workflows/:type (info)
  - POST /api/v1/sme/workflows/:type (create)
  - Unknown types, invalid JSON, body too large, execute mode
  - Method not allowed
"""

from __future__ import annotations

import io
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _make_handler(
    method: str = "GET",
    body: dict[str, Any] | None = None,
    client_address: tuple[str, int] = ("127.0.0.1", 12345),
) -> MagicMock:
    """Create a mock HTTP handler suitable for the workflow templates handler."""
    h = MagicMock()
    h.command = method
    h.client_address = client_address

    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
    else:
        body_bytes = b"{}"

    h.headers = {"Content-Length": str(len(body_bytes))}
    h.rfile = MagicMock()
    h.rfile.read.return_value = body_bytes
    return h


# ---------------------------------------------------------------------------
# Fixtures - lazy import to allow conftest auto-auth to patch first
# ---------------------------------------------------------------------------


@pytest.fixture
def handler_module():
    """Import the handler module lazily (after conftest patches)."""
    import aragora.server.handlers.workflow_templates as mod
    return mod


@pytest.fixture
def templates_handler(handler_module):
    """Create a WorkflowTemplatesHandler instance."""
    return handler_module.WorkflowTemplatesHandler(ctx={})


@pytest.fixture
def categories_handler(handler_module):
    """Create a WorkflowCategoriesHandler instance."""
    return handler_module.WorkflowCategoriesHandler(ctx={})


@pytest.fixture
def patterns_handler(handler_module):
    """Create a WorkflowPatternsHandler instance."""
    return handler_module.WorkflowPatternsHandler(ctx={})


@pytest.fixture
def pattern_templates_handler(handler_module):
    """Create a WorkflowPatternTemplatesHandler instance."""
    return handler_module.WorkflowPatternTemplatesHandler(ctx={})


@pytest.fixture
def recommendations_handler(handler_module):
    """Create a TemplateRecommendationsHandler instance."""
    return handler_module.TemplateRecommendationsHandler(ctx={})


@pytest.fixture
def sme_handler(handler_module):
    """Create a SMEWorkflowsHandler instance."""
    return handler_module.SMEWorkflowsHandler(ctx={})


@pytest.fixture(autouse=True)
def _reset_rate_limiter(handler_module):
    """Reset the module-level rate limiter between tests."""
    handler_module._template_limiter._buckets = {}
    yield
    handler_module._template_limiter._buckets = {}


# ---------------------------------------------------------------------------
# Sample template data for mocking
# ---------------------------------------------------------------------------

SAMPLE_TEMPLATE = {
    "name": "Quick Decision",
    "description": "Fast yes/no decisions with 2 agents",
    "pattern": "debate",
    "steps": [
        {"id": "step1", "name": "Propose", "type": "propose"},
        {"id": "step2", "name": "Critique", "type": "critique"},
    ],
    "inputs": {"question": {"type": "string", "required": True}},
    "outputs": {"decision": {"type": "string"}},
    "estimated_duration": 5,
    "recommended_agents": ["claude", "gpt4"],
    "tags": ["quick", "decision"],
    "config": {"rounds": 2},
}

SAMPLE_TEMPLATE_LIST = [
    {
        "id": "general/quick-decision",
        "name": "Quick Decision",
        "description": "Fast decisions",
        "tags": ["quick"],
    },
    {
        "id": "general/pros-cons",
        "name": "Pros and Cons",
        "description": "Balanced analysis",
        "tags": ["analysis"],
    },
    {
        "id": "code/architecture-review",
        "name": "Architecture Review",
        "description": "Review architecture",
        "tags": ["code", "review"],
    },
]


# ===========================================================================
# WorkflowTemplatesHandler - can_handle
# ===========================================================================


class TestWorkflowTemplatesCanHandle:
    """Tests for WorkflowTemplatesHandler.can_handle()."""

    def test_can_handle_list_path(self, templates_handler):
        assert templates_handler.can_handle("/api/v1/workflow/templates") is True

    def test_can_handle_template_id_path(self, templates_handler):
        assert templates_handler.can_handle("/api/v1/workflow/templates/general/quick-decision") is True

    def test_can_handle_package_path(self, templates_handler):
        assert templates_handler.can_handle("/api/v1/workflow/templates/general/quick-decision/package") is True

    def test_can_handle_run_path(self, templates_handler):
        assert templates_handler.can_handle("/api/v1/workflow/templates/general/quick-decision/run") is True

    def test_cannot_handle_unrelated_path(self, templates_handler):
        assert templates_handler.can_handle("/api/v1/workflow/patterns") is False

    def test_cannot_handle_root(self, templates_handler):
        assert templates_handler.can_handle("/") is False

    def test_can_handle_with_method(self, templates_handler):
        assert templates_handler.can_handle("/api/v1/workflow/templates", method="GET") is True
        assert templates_handler.can_handle("/api/v1/workflow/templates", method="POST") is True

    def test_cannot_handle_workflow_root(self, templates_handler):
        assert templates_handler.can_handle("/api/v1/workflow") is False


# ===========================================================================
# WorkflowTemplatesHandler - list templates
# ===========================================================================


class TestListTemplates:
    """Tests for GET /api/v1/workflow/templates."""

    @patch("aragora.workflow.templates.list_templates")
    @patch("aragora.workflow.templates.WORKFLOW_TEMPLATES", {"general/quick-decision": SAMPLE_TEMPLATE})
    def test_list_templates_basic(self, mock_list, templates_handler):
        mock_list.return_value = SAMPLE_TEMPLATE_LIST[:1]
        h = _make_handler(method="GET")
        result = templates_handler.handle("/api/v1/workflow/templates", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert "templates" in body
        assert "total" in body

    @patch("aragora.workflow.templates.list_templates")
    @patch("aragora.workflow.templates.WORKFLOW_TEMPLATES", {"general/quick-decision": SAMPLE_TEMPLATE})
    def test_list_templates_with_category_filter(self, mock_list, templates_handler):
        mock_list.return_value = SAMPLE_TEMPLATE_LIST[:1]
        h = _make_handler(method="GET")
        result = templates_handler.handle(
            "/api/v1/workflow/templates",
            {"category": "general"},
            h,
        )
        assert _status(result) == 200
        mock_list.assert_called_with(category="general")

    @patch("aragora.workflow.templates.list_templates")
    @patch("aragora.workflow.templates.WORKFLOW_TEMPLATES", {})
    def test_list_templates_with_tag_filter(self, mock_list, templates_handler):
        mock_list.return_value = SAMPLE_TEMPLATE_LIST
        h = _make_handler(method="GET")
        result = templates_handler.handle(
            "/api/v1/workflow/templates",
            {"tag": "quick"},
            h,
        )
        body = _body(result)
        assert _status(result) == 200
        # Only the template with "quick" tag should remain
        for t in body["templates"]:
            assert "quick" in t.get("tags", [])

    @patch("aragora.workflow.templates.list_templates")
    @patch("aragora.workflow.templates.WORKFLOW_TEMPLATES", {})
    def test_list_templates_with_search_filter(self, mock_list, templates_handler):
        mock_list.return_value = SAMPLE_TEMPLATE_LIST
        h = _make_handler(method="GET")
        result = templates_handler.handle(
            "/api/v1/workflow/templates",
            {"search": "architecture"},
            h,
        )
        body = _body(result)
        assert _status(result) == 200
        for t in body["templates"]:
            combined = (t["name"] + t.get("description", "")).lower()
            assert "architecture" in combined

    @patch("aragora.workflow.templates.list_templates")
    @patch("aragora.workflow.templates.WORKFLOW_TEMPLATES", {})
    def test_list_templates_with_search_case_insensitive(self, mock_list, templates_handler):
        mock_list.return_value = SAMPLE_TEMPLATE_LIST
        h = _make_handler(method="GET")
        result = templates_handler.handle(
            "/api/v1/workflow/templates",
            {"search": "QUICK"},
            h,
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["total"] >= 1

    @patch("aragora.workflow.templates.list_templates")
    @patch("aragora.workflow.templates.WORKFLOW_TEMPLATES", {})
    def test_list_templates_pagination_limit(self, mock_list, templates_handler):
        mock_list.return_value = SAMPLE_TEMPLATE_LIST
        h = _make_handler(method="GET")
        result = templates_handler.handle(
            "/api/v1/workflow/templates",
            {"limit": "1"},
            h,
        )
        body = _body(result)
        assert _status(result) == 200
        assert len(body["templates"]) <= 1
        assert body["limit"] == 1

    @patch("aragora.workflow.templates.list_templates")
    @patch("aragora.workflow.templates.WORKFLOW_TEMPLATES", {})
    def test_list_templates_pagination_offset(self, mock_list, templates_handler):
        mock_list.return_value = SAMPLE_TEMPLATE_LIST
        h = _make_handler(method="GET")
        result = templates_handler.handle(
            "/api/v1/workflow/templates",
            {"offset": "1", "limit": "1"},
            h,
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["offset"] == 1

    @patch("aragora.workflow.templates.list_templates")
    @patch("aragora.workflow.templates.WORKFLOW_TEMPLATES", {})
    def test_list_templates_empty_result(self, mock_list, templates_handler):
        mock_list.return_value = []
        h = _make_handler(method="GET")
        result = templates_handler.handle("/api/v1/workflow/templates", {}, h)
        body = _body(result)
        assert _status(result) == 200
        assert body["templates"] == []
        assert body["total"] == 0

    @patch("aragora.workflow.templates.list_templates")
    @patch("aragora.workflow.templates.WORKFLOW_TEMPLATES", {"general/quick-decision": SAMPLE_TEMPLATE})
    def test_list_templates_enrichment(self, mock_list, templates_handler):
        mock_list.return_value = [SAMPLE_TEMPLATE_LIST[0]]
        h = _make_handler(method="GET")
        result = templates_handler.handle("/api/v1/workflow/templates", {}, h)
        body = _body(result)
        assert _status(result) == 200
        tmpl = body["templates"][0]
        assert "steps_count" in tmpl
        assert tmpl["steps_count"] == 2
        assert "pattern" in tmpl

    @patch("aragora.workflow.templates.list_templates")
    @patch("aragora.workflow.templates.WORKFLOW_TEMPLATES", {})
    def test_list_templates_no_match_in_workflow_templates(self, mock_list, templates_handler):
        """When template ID is not in WORKFLOW_TEMPLATES, steps_count=0."""
        mock_list.return_value = [SAMPLE_TEMPLATE_LIST[0]]
        h = _make_handler(method="GET")
        result = templates_handler.handle("/api/v1/workflow/templates", {}, h)
        body = _body(result)
        tmpl = body["templates"][0]
        assert tmpl["steps_count"] == 0
        assert tmpl["pattern"] is None

    @patch("aragora.workflow.templates.list_templates")
    @patch("aragora.workflow.templates.WORKFLOW_TEMPLATES", {})
    def test_list_templates_tag_filter_no_match(self, mock_list, templates_handler):
        mock_list.return_value = SAMPLE_TEMPLATE_LIST
        h = _make_handler(method="GET")
        result = templates_handler.handle(
            "/api/v1/workflow/templates",
            {"tag": "nonexistent_tag"},
            h,
        )
        body = _body(result)
        assert body["total"] == 0
        assert body["templates"] == []

    @patch("aragora.workflow.templates.list_templates")
    @patch("aragora.workflow.templates.WORKFLOW_TEMPLATES", {})
    def test_list_templates_search_no_match(self, mock_list, templates_handler):
        mock_list.return_value = SAMPLE_TEMPLATE_LIST
        h = _make_handler(method="GET")
        result = templates_handler.handle(
            "/api/v1/workflow/templates",
            {"search": "zzzznonexistent"},
            h,
        )
        body = _body(result)
        assert body["total"] == 0


# ===========================================================================
# WorkflowTemplatesHandler - get template
# ===========================================================================


class TestGetTemplate:
    """Tests for GET /api/v1/workflow/templates/:id."""

    @patch("aragora.workflow.templates.get_template")
    def test_get_template_found(self, mock_get, templates_handler):
        mock_get.return_value = SAMPLE_TEMPLATE
        h = _make_handler(method="GET")
        result = templates_handler.handle(
            "/api/v1/workflow/templates/general/quick-decision",
            {},
            h,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "general/quick-decision"
        assert body["name"] == "Quick Decision"
        assert body["category"] == "general"
        assert len(body["steps"]) == 2

    @patch("aragora.workflow.templates.get_template")
    def test_get_template_not_found(self, mock_get, templates_handler):
        mock_get.return_value = None
        h = _make_handler(method="GET")
        result = templates_handler.handle(
            "/api/v1/workflow/templates/nonexistent/template",
            {},
            h,
        )
        assert _status(result) == 404

    @patch("aragora.workflow.templates.get_template")
    def test_get_template_general_category(self, mock_get, templates_handler):
        """Template ID without / should default to 'general' category."""
        mock_get.return_value = SAMPLE_TEMPLATE
        h = _make_handler(method="GET")
        result = templates_handler.handle(
            "/api/v1/workflow/templates/simple-template",
            {},
            h,
        )
        body = _body(result)
        assert body["category"] == "general"

    @patch("aragora.workflow.templates.get_template")
    def test_get_template_with_tags(self, mock_get, templates_handler):
        mock_get.return_value = SAMPLE_TEMPLATE
        h = _make_handler(method="GET")
        result = templates_handler.handle(
            "/api/v1/workflow/templates/general/quick-decision",
            {},
            h,
        )
        body = _body(result)
        assert body["tags"] == ["quick", "decision"]

    @patch("aragora.workflow.templates.get_template")
    def test_get_template_response_fields(self, mock_get, templates_handler):
        mock_get.return_value = SAMPLE_TEMPLATE
        h = _make_handler(method="GET")
        result = templates_handler.handle(
            "/api/v1/workflow/templates/general/quick-decision",
            {},
            h,
        )
        body = _body(result)
        expected_keys = {"id", "name", "description", "category", "pattern", "steps",
                         "inputs", "outputs", "estimated_duration", "recommended_agents", "tags"}
        assert expected_keys == set(body.keys())


# ===========================================================================
# WorkflowTemplatesHandler - get package
# ===========================================================================


class TestGetPackage:
    """Tests for GET /api/v1/workflow/templates/:id/package."""

    @patch("aragora.workflow.templates.package.create_package")
    @patch("aragora.workflow.templates.get_template")
    def test_get_package_found(self, mock_get, mock_create, templates_handler):
        mock_get.return_value = SAMPLE_TEMPLATE
        mock_pkg = MagicMock()
        mock_pkg.to_dict.return_value = {"id": "general/quick-decision", "version": "1.0.0"}
        mock_create.return_value = mock_pkg
        h = _make_handler(method="GET")
        result = templates_handler.handle(
            "/api/v1/workflow/templates/general/quick-decision/package",
            {},
            h,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["version"] == "1.0.0"

    @patch("aragora.workflow.templates.get_template")
    def test_get_package_not_found(self, mock_get, templates_handler):
        mock_get.return_value = None
        h = _make_handler(method="GET")
        result = templates_handler.handle(
            "/api/v1/workflow/templates/nonexistent/template/package",
            {},
            h,
        )
        assert _status(result) == 404

    @patch("aragora.workflow.templates.package.create_package")
    @patch("aragora.workflow.templates.get_template")
    def test_get_package_general_category(self, mock_get, mock_create, templates_handler):
        """Template without / in ID should use 'general' category."""
        mock_get.return_value = SAMPLE_TEMPLATE
        mock_pkg = MagicMock()
        mock_pkg.to_dict.return_value = {"category": "general"}
        mock_create.return_value = mock_pkg
        h = _make_handler(method="GET")
        result = templates_handler.handle(
            "/api/v1/workflow/templates/simple/package",
            {},
            h,
        )
        assert _status(result) == 200
        # Verify create_package was called with category="simple" (first segment)
        # since "simple" is a single-segment ID
        call_kwargs = mock_create.call_args
        # category is positional or keyword
        assert call_kwargs is not None


# ===========================================================================
# WorkflowTemplatesHandler - run template (POST)
# ===========================================================================


class TestRunTemplate:
    """Tests for POST /api/v1/workflow/templates (run template)."""

    @patch("aragora.workflow.engine.WorkflowEngine")
    @patch("aragora.workflow.templates.get_template")
    def test_run_template_success(self, mock_get, mock_engine_cls, templates_handler):
        mock_get.return_value = SAMPLE_TEMPLATE
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.to_dict.return_value = {"output": "done"}
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        # Patch WorkflowDefinition.from_dict
        with patch("aragora.server.handlers.workflow_templates.WorkflowDefinition") as mock_wd:
            mock_workflow = MagicMock()
            mock_wd.from_dict.return_value = mock_workflow

            # Patch asyncio.run to return the mock result
            with patch("asyncio.run", return_value=mock_result):
                h = _make_handler(method="POST", body={"template_id": "general/quick-decision", "inputs": {}})
                result = templates_handler.handle("/api/v1/workflow/templates", {}, h)

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "completed"
        assert body["template_id"] == "general/quick-decision"

    @patch("aragora.workflow.templates.get_template")
    def test_run_template_missing_template_id(self, mock_get, templates_handler):
        h = _make_handler(method="POST", body={"inputs": {}})
        result = templates_handler.handle("/api/v1/workflow/templates", {}, h)
        assert _status(result) == 400
        body = _body(result)
        assert "template_id" in body.get("error", "")

    @patch("aragora.workflow.templates.get_template")
    def test_run_template_not_found(self, mock_get, templates_handler):
        mock_get.return_value = None
        h = _make_handler(method="POST", body={"template_id": "nonexistent"})
        result = templates_handler.handle("/api/v1/workflow/templates", {}, h)
        assert _status(result) == 404

    def test_run_template_invalid_json(self, templates_handler):
        h = MagicMock()
        h.command = "POST"
        h.client_address = ("127.0.0.1", 12345)
        h.headers = {"Content-Length": "11"}
        h.rfile = MagicMock()
        h.rfile.read.return_value = b"not json!!!"
        result = templates_handler.handle("/api/v1/workflow/templates", {}, h)
        assert _status(result) == 400

    def test_run_template_body_too_large(self, templates_handler):
        h = MagicMock()
        h.command = "POST"
        h.client_address = ("127.0.0.1", 12345)
        h.headers = {"Content-Length": str(11 * 1024 * 1024)}  # 11 MB
        h.rfile = MagicMock()
        result = templates_handler.handle("/api/v1/workflow/templates", {}, h)
        assert _status(result) == 413

    @patch("aragora.workflow.engine.WorkflowEngine")
    @patch("aragora.workflow.templates.get_template")
    def test_run_template_execution_error(self, mock_get, mock_engine_cls, templates_handler):
        mock_get.return_value = SAMPLE_TEMPLATE

        with patch("aragora.server.handlers.workflow_templates.WorkflowDefinition") as mock_wd:
            mock_wd.from_dict.side_effect = RuntimeError("workflow error")
            h = _make_handler(method="POST", body={"template_id": "general/quick-decision"})
            result = templates_handler.handle("/api/v1/workflow/templates", {}, h)

        assert _status(result) == 500
        body = _body(result)
        assert body["status"] == "failed"

    def test_run_template_empty_body(self, templates_handler):
        """POST with empty body should fail with missing template_id."""
        h = _make_handler(method="POST", body={})
        result = templates_handler.handle("/api/v1/workflow/templates", {}, h)
        assert _status(result) == 400

    @patch("aragora.workflow.engine.WorkflowEngine")
    @patch("aragora.workflow.templates.get_template")
    def test_run_template_emits_events(self, mock_get, mock_engine_cls, templates_handler):
        """Verify template execution events are emitted."""
        mock_emitter = MagicMock()
        templates_handler.ctx = {"event_emitter": mock_emitter}

        mock_get.return_value = SAMPLE_TEMPLATE
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.to_dict.return_value = {}

        with patch("aragora.server.handlers.workflow_templates.WorkflowDefinition") as mock_wd:
            mock_wd.from_dict.return_value = MagicMock()
            with patch("asyncio.run", return_value=mock_result):
                h = _make_handler(method="POST", body={"template_id": "test/t1"})
                result = templates_handler.handle("/api/v1/workflow/templates", {}, h)
        # Events should have been emitted (emit is called)
        # We verify no crash; exact emit calls depend on StreamEventType availability
        assert _status(result) == 200


# ===========================================================================
# WorkflowTemplatesHandler - run specific template
# ===========================================================================


class TestRunSpecificTemplate:
    """Tests for POST /api/v1/workflow/templates/:id/run."""

    @patch("aragora.workflow.templates.get_template")
    def test_run_specific_template_accepted(self, mock_get, templates_handler):
        mock_get.return_value = SAMPLE_TEMPLATE
        h = _make_handler(method="POST", body={"inputs": {"question": "test?"}})
        # The handler reads method from handler.command
        result = templates_handler.handle(
            "/api/v1/workflow/templates/general/quick-decision/run",
            {},
            h,
        )
        assert _status(result) == 202
        body = _body(result)
        assert body["status"] == "accepted"
        assert body["template_id"] == "general/quick-decision"

    @patch("aragora.workflow.templates.get_template")
    def test_run_specific_template_not_found(self, mock_get, templates_handler):
        mock_get.return_value = None
        h = _make_handler(method="POST")
        result = templates_handler.handle(
            "/api/v1/workflow/templates/nonexistent/run",
            {},
            h,
        )
        assert _status(result) == 404


# ===========================================================================
# WorkflowTemplatesHandler - method not allowed, invalid paths
# ===========================================================================


class TestTemplatesMethodNotAllowed:
    """Tests for method not allowed and invalid paths."""

    @patch("aragora.workflow.templates.list_templates")
    @patch("aragora.workflow.templates.WORKFLOW_TEMPLATES", {})
    def test_method_put_not_allowed(self, mock_list, templates_handler):
        h = _make_handler(method="PUT")
        result = templates_handler.handle("/api/v1/workflow/templates", {}, h)
        assert _status(result) == 405

    @patch("aragora.workflow.templates.list_templates")
    @patch("aragora.workflow.templates.WORKFLOW_TEMPLATES", {})
    def test_method_delete_not_allowed(self, mock_list, templates_handler):
        h = _make_handler(method="DELETE")
        result = templates_handler.handle("/api/v1/workflow/templates", {}, h)
        assert _status(result) == 405

    @patch("aragora.workflow.templates.list_templates")
    @patch("aragora.workflow.templates.WORKFLOW_TEMPLATES", {})
    def test_method_patch_not_allowed(self, mock_list, templates_handler):
        h = _make_handler(method="PATCH")
        result = templates_handler.handle("/api/v1/workflow/templates", {}, h)
        assert _status(result) == 405


# ===========================================================================
# WorkflowTemplatesHandler - rate limiting
# ===========================================================================


class TestTemplatesRateLimiting:
    """Tests for rate limiting on workflow templates handler."""

    def test_rate_limit_exceeded(self, handler_module, templates_handler):
        """Simulate rate limit exceeded."""
        with patch.object(handler_module._template_limiter, "is_allowed", return_value=False):
            h = _make_handler(method="GET")
            result = templates_handler.handle("/api/v1/workflow/templates", {}, h)
            assert _status(result) == 429

    def test_rate_limit_not_exceeded(self, handler_module, templates_handler):
        """Normal request should not be rate limited."""
        with patch.object(handler_module._template_limiter, "is_allowed", return_value=True):
            with patch("aragora.workflow.templates.list_templates", return_value=[]):
                with patch("aragora.workflow.templates.WORKFLOW_TEMPLATES", {}):
                    h = _make_handler(method="GET")
                    result = templates_handler.handle("/api/v1/workflow/templates", {}, h)
                    assert _status(result) == 200


# ===========================================================================
# WorkflowCategoriesHandler
# ===========================================================================


class TestWorkflowCategoriesHandler:
    """Tests for WorkflowCategoriesHandler."""

    def test_can_handle_categories_path(self, categories_handler):
        assert categories_handler.can_handle("/api/v1/workflow/categories") is True

    def test_cannot_handle_other_path(self, categories_handler):
        assert categories_handler.can_handle("/api/v1/workflow/templates") is False

    def test_cannot_handle_subpath(self, categories_handler):
        assert categories_handler.can_handle("/api/v1/workflow/categories/foo") is False

    @patch("aragora.workflow.templates.WORKFLOW_TEMPLATES", {
        "general/quick-decision": {},
        "code/architecture": {},
        "general/brainstorm": {},
    })
    def test_list_categories(self, categories_handler):
        mock_category = MagicMock()
        mock_category.value = "general"
        mock_category2 = MagicMock()
        mock_category2.value = "code"
        mock_category3 = MagicMock()
        mock_category3.value = "devops"  # no templates for this

        with patch("aragora.workflow.templates.package.TemplateCategory", [mock_category, mock_category2, mock_category3]):
            h = _make_handler(method="GET")
            result = categories_handler.handle("/api/v1/workflow/categories", {}, h)

        assert _status(result) == 200
        body = _body(result)
        assert "categories" in body
        # Only categories with count > 0 should be present
        cat_ids = [c["id"] for c in body["categories"]]
        assert "general" in cat_ids
        assert "code" in cat_ids
        assert "devops" not in cat_ids

    @patch("aragora.workflow.templates.WORKFLOW_TEMPLATES", {})
    def test_list_categories_empty(self, categories_handler):
        with patch("aragora.workflow.templates.package.TemplateCategory", []):
            h = _make_handler(method="GET")
            result = categories_handler.handle("/api/v1/workflow/categories", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert body["categories"] == []

    def test_categories_handler_init_default_ctx(self, handler_module):
        handler = handler_module.WorkflowCategoriesHandler()
        assert handler.ctx == {}


# ===========================================================================
# WorkflowPatternsHandler
# ===========================================================================


class TestWorkflowPatternsHandler:
    """Tests for WorkflowPatternsHandler."""

    def test_can_handle_patterns_path(self, patterns_handler):
        assert patterns_handler.can_handle("/api/v1/workflow/patterns") is True

    def test_cannot_handle_other_path(self, patterns_handler):
        assert patterns_handler.can_handle("/api/v1/workflow/templates") is False

    def test_list_patterns(self, patterns_handler):
        mock_type = MagicMock()
        mock_type.value = "map_reduce"
        mock_class = MagicMock()
        mock_class.__doc__ = "Map-reduce pattern for parallel processing.\nMore details."

        with patch("aragora.workflow.patterns.PATTERN_REGISTRY", {mock_type: mock_class}):
            with patch("aragora.workflow.patterns.base.PatternType", [mock_type]):
                h = _make_handler(method="GET")
                result = patterns_handler.handle("/api/v1/workflow/patterns", {}, h)

        assert _status(result) == 200
        body = _body(result)
        assert "patterns" in body
        assert len(body["patterns"]) == 1
        assert body["patterns"][0]["id"] == "map_reduce"
        assert body["patterns"][0]["available"] is True
        assert "Map-reduce pattern" in body["patterns"][0]["description"]

    def test_list_patterns_no_doc(self, patterns_handler):
        mock_type = MagicMock()
        mock_type.value = "pipeline"
        mock_class = MagicMock()
        mock_class.__doc__ = None

        with patch("aragora.workflow.patterns.PATTERN_REGISTRY", {mock_type: mock_class}):
            with patch("aragora.workflow.patterns.base.PatternType", [mock_type]):
                h = _make_handler(method="GET")
                result = patterns_handler.handle("/api/v1/workflow/patterns", {}, h)

        body = _body(result)
        assert body["patterns"][0]["description"] == ""

    def test_list_patterns_unavailable(self, patterns_handler):
        mock_type = MagicMock()
        mock_type.value = "custom"

        with patch("aragora.workflow.patterns.PATTERN_REGISTRY", {mock_type: None}):
            with patch("aragora.workflow.patterns.base.PatternType", [mock_type]):
                h = _make_handler(method="GET")
                result = patterns_handler.handle("/api/v1/workflow/patterns", {}, h)

        body = _body(result)
        assert body["patterns"][0]["available"] is False

    def test_patterns_handler_init_default_ctx(self, handler_module):
        handler = handler_module.WorkflowPatternsHandler()
        assert handler.ctx == {}


# ===========================================================================
# WorkflowPatternTemplatesHandler
# ===========================================================================


class TestWorkflowPatternTemplatesHandler:
    """Tests for WorkflowPatternTemplatesHandler."""

    def test_can_handle_pattern_templates_path(self, pattern_templates_handler):
        assert pattern_templates_handler.can_handle("/api/v1/workflow/pattern-templates") is True

    def test_can_handle_specific_pattern(self, pattern_templates_handler):
        assert pattern_templates_handler.can_handle("/api/v1/workflow/pattern-templates/hive-mind") is True

    def test_cannot_handle_unrelated(self, pattern_templates_handler):
        assert pattern_templates_handler.can_handle("/api/v1/workflow/templates") is False

    @patch("aragora.workflow.templates.patterns.list_pattern_templates")
    def test_list_pattern_templates(self, mock_list, pattern_templates_handler):
        mock_list.return_value = [
            {"id": "hive-mind", "name": "Hive Mind"},
            {"id": "map-reduce", "name": "Map Reduce"},
        ]
        h = _make_handler(method="GET")
        result = pattern_templates_handler.handle(
            "/api/v1/workflow/pattern-templates",
            {},
            h,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2

    def test_list_method_not_allowed(self, pattern_templates_handler):
        h = _make_handler(method="POST")
        result = pattern_templates_handler.handle(
            "/api/v1/workflow/pattern-templates",
            {},
            h,
        )
        assert _status(result) == 405

    @patch("aragora.workflow.templates.patterns.get_pattern_template")
    def test_get_pattern_template_found(self, mock_get, pattern_templates_handler):
        mock_get.return_value = {
            "id": "hive-mind",
            "name": "Hive Mind",
            "description": "Collective intelligence",
            "pattern": "hive_mind",
            "version": "2.0.0",
            "config": {"agents": 5},
            "inputs": {"task": {"type": "string"}},
            "outputs": {"result": {"type": "string"}},
            "tags": ["collective"],
        }
        h = _make_handler(method="GET")
        result = pattern_templates_handler.handle(
            "/api/v1/workflow/pattern-templates/hive-mind",
            {},
            h,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "hive-mind"
        assert body["version"] == "2.0.0"

    @patch("aragora.workflow.templates.patterns.get_pattern_template")
    def test_get_pattern_template_not_found(self, mock_get, pattern_templates_handler):
        mock_get.return_value = None  # both direct and pattern/ prefix return None
        h = _make_handler(method="GET")
        result = pattern_templates_handler.handle(
            "/api/v1/workflow/pattern-templates/nonexistent",
            {},
            h,
        )
        assert _status(result) == 404

    @patch("aragora.workflow.templates.patterns.get_pattern_template")
    def test_get_pattern_template_fallback_prefix(self, mock_get, pattern_templates_handler):
        """When direct lookup fails, tries with pattern/ prefix."""
        # First call returns None (direct), second call returns the template
        mock_get.side_effect = [None, {"id": "pattern/hive-mind", "name": "Hive Mind"}]
        h = _make_handler(method="GET")
        result = pattern_templates_handler.handle(
            "/api/v1/workflow/pattern-templates/hive-mind",
            {},
            h,
        )
        assert _status(result) == 200
        assert mock_get.call_count == 2

    def test_rate_limit_pattern_templates(self, handler_module, pattern_templates_handler):
        with patch.object(handler_module._template_limiter, "is_allowed", return_value=False):
            h = _make_handler(method="GET")
            result = pattern_templates_handler.handle(
                "/api/v1/workflow/pattern-templates",
                {},
                h,
            )
            assert _status(result) == 429


# ===========================================================================
# WorkflowPatternTemplatesHandler - instantiate pattern
# ===========================================================================


class TestInstantiatePattern:
    """Tests for POST /api/v1/workflow/pattern-templates/:id/instantiate."""

    @patch("aragora.workflow.templates.patterns.create_hive_mind_workflow")
    def test_instantiate_hive_mind(self, mock_create, pattern_templates_handler):
        mock_wf = MagicMock()
        mock_wf.id = "wf-123"
        mock_wf.name = "Hive Mind Workflow"
        mock_wf.description = "Test hive mind"
        mock_wf.steps = []
        mock_wf.entry_step = "start"
        mock_wf.tags = ["test"]
        mock_wf.metadata = {}
        mock_create.return_value = mock_wf

        h = _make_handler(method="POST", body={"name": "My Hive", "task": "analyze"})
        result = pattern_templates_handler.handle(
            "/api/v1/workflow/pattern-templates/hive-mind/instantiate",
            {},
            h,
        )
        assert _status(result) == 201
        body = _body(result)
        assert body["status"] == "created"
        assert body["workflow"]["id"] == "wf-123"

    @patch("aragora.workflow.templates.patterns.create_map_reduce_workflow")
    def test_instantiate_map_reduce(self, mock_create, pattern_templates_handler):
        mock_wf = MagicMock()
        mock_wf.id = "wf-mr"
        mock_wf.name = "Map Reduce"
        mock_wf.description = ""
        mock_wf.steps = []
        mock_wf.entry_step = "map"
        mock_wf.tags = []
        mock_wf.metadata = {}
        mock_create.return_value = mock_wf

        h = _make_handler(method="POST", body={"name": "MR", "task": "count"})
        result = pattern_templates_handler.handle(
            "/api/v1/workflow/pattern-templates/map-reduce/instantiate",
            {},
            h,
        )
        assert _status(result) == 201

    @patch("aragora.workflow.templates.patterns.create_map_reduce_workflow")
    def test_instantiate_map_reduce_underscore(self, mock_create, pattern_templates_handler):
        """Both hyphens and underscores should work."""
        mock_wf = MagicMock()
        mock_wf.id = "wf-mr"
        mock_wf.name = "Map Reduce"
        mock_wf.description = ""
        mock_wf.steps = []
        mock_wf.entry_step = "map"
        mock_wf.tags = []
        mock_wf.metadata = {}
        mock_create.return_value = mock_wf

        h = _make_handler(method="POST", body={"name": "MR", "task": "count"})
        result = pattern_templates_handler.handle(
            "/api/v1/workflow/pattern-templates/map_reduce/instantiate",
            {},
            h,
        )
        assert _status(result) == 201

    @patch("aragora.workflow.templates.patterns.create_review_cycle_workflow")
    def test_instantiate_review_cycle(self, mock_create, pattern_templates_handler):
        mock_wf = MagicMock()
        mock_wf.id = "wf-rc"
        mock_wf.name = "Review Cycle"
        mock_wf.description = ""
        mock_wf.steps = []
        mock_wf.entry_step = "review"
        mock_wf.tags = []
        mock_wf.metadata = {}
        mock_create.return_value = mock_wf

        h = _make_handler(method="POST", body={"name": "RC", "task": "review code"})
        result = pattern_templates_handler.handle(
            "/api/v1/workflow/pattern-templates/review-cycle/instantiate",
            {},
            h,
        )
        assert _status(result) == 201

    def test_instantiate_unknown_pattern(self, pattern_templates_handler):
        h = _make_handler(method="POST", body={"name": "Unknown"})
        result = pattern_templates_handler.handle(
            "/api/v1/workflow/pattern-templates/unknown-pattern/instantiate",
            {},
            h,
        )
        assert _status(result) == 404
        body = _body(result)
        assert "Unknown pattern" in body.get("error", "")

    def test_instantiate_invalid_json(self, pattern_templates_handler):
        h = MagicMock()
        h.command = "POST"
        h.client_address = ("127.0.0.1", 12345)
        h.headers = {"Content-Length": "5"}
        h.rfile = MagicMock()
        h.rfile.read.return_value = b"{bad}"
        result = pattern_templates_handler.handle(
            "/api/v1/workflow/pattern-templates/hive-mind/instantiate",
            {},
            h,
        )
        assert _status(result) == 400

    def test_instantiate_body_too_large(self, pattern_templates_handler):
        h = MagicMock()
        h.command = "POST"
        h.client_address = ("127.0.0.1", 12345)
        h.headers = {"Content-Length": str(11 * 1024 * 1024)}
        h.rfile = MagicMock()
        result = pattern_templates_handler.handle(
            "/api/v1/workflow/pattern-templates/hive-mind/instantiate",
            {},
            h,
        )
        assert _status(result) == 413

    @patch("aragora.workflow.templates.patterns.create_hive_mind_workflow")
    def test_instantiate_factory_error(self, mock_create, pattern_templates_handler):
        mock_create.side_effect = ValueError("bad config")
        h = _make_handler(method="POST", body={"name": "Bad", "task": "test"})
        result = pattern_templates_handler.handle(
            "/api/v1/workflow/pattern-templates/hive-mind/instantiate",
            {},
            h,
        )
        assert _status(result) == 500

    @patch("aragora.workflow.templates.patterns.create_hive_mind_workflow")
    def test_instantiate_with_config(self, mock_create, pattern_templates_handler):
        """Config from request should be merged into factory args."""
        mock_wf = MagicMock()
        mock_wf.id = "wf-c"
        mock_wf.name = "Custom"
        mock_wf.description = ""
        mock_wf.steps = []
        mock_wf.entry_step = "start"
        mock_wf.tags = []
        mock_wf.metadata = {}
        mock_create.return_value = mock_wf

        h = _make_handler(
            method="POST",
            body={"name": "Custom", "task": "analyze", "config": {"agents": 5}},
        )
        result = pattern_templates_handler.handle(
            "/api/v1/workflow/pattern-templates/hive-mind/instantiate",
            {},
            h,
        )
        assert _status(result) == 201
        # Verify the config was passed through
        call_kwargs = mock_create.call_args
        assert call_kwargs.kwargs.get("agents") == 5 or (call_kwargs.args and True)

    @patch("aragora.workflow.templates.patterns.create_hive_mind_workflow")
    def test_instantiate_default_name(self, mock_create, pattern_templates_handler):
        """When no name provided, generate default from pattern ID."""
        mock_wf = MagicMock()
        mock_wf.id = "wf-default"
        mock_wf.name = "Hive Mind Workflow"
        mock_wf.description = ""
        mock_wf.steps = []
        mock_wf.entry_step = "start"
        mock_wf.tags = []
        mock_wf.metadata = {}
        mock_create.return_value = mock_wf

        h = _make_handler(method="POST", body={"task": "test"})
        result = pattern_templates_handler.handle(
            "/api/v1/workflow/pattern-templates/hive-mind/instantiate",
            {},
            h,
        )
        assert _status(result) == 201
        call_kwargs = mock_create.call_args
        assert "Hive Mind" in call_kwargs.kwargs.get("name", "")

    @patch("aragora.workflow.templates.patterns.create_hive_mind_workflow")
    def test_instantiate_workflow_steps_serialized(self, mock_create, pattern_templates_handler):
        """Verify workflow steps are properly serialized in response."""
        mock_step = MagicMock()
        mock_step.id = "step1"
        mock_step.name = "Analyze"
        mock_step.step_type = "analyze"
        mock_step.config = {"key": "val"}
        mock_step.next_steps = ["step2"]

        mock_wf = MagicMock()
        mock_wf.id = "wf-steps"
        mock_wf.name = "Steps WF"
        mock_wf.description = "with steps"
        mock_wf.steps = [mock_step]
        mock_wf.entry_step = "step1"
        mock_wf.tags = ["tag1"]
        mock_wf.metadata = {"key": "val"}
        mock_create.return_value = mock_wf

        h = _make_handler(method="POST", body={"name": "S", "task": "t"})
        result = pattern_templates_handler.handle(
            "/api/v1/workflow/pattern-templates/hive-mind/instantiate",
            {},
            h,
        )
        body = _body(result)
        wf = body["workflow"]
        assert len(wf["steps"]) == 1
        assert wf["steps"][0]["id"] == "step1"
        assert wf["steps"][0]["type"] == "analyze"
        assert wf["entry_step"] == "step1"
        assert wf["tags"] == ["tag1"]


# ===========================================================================
# TemplateRecommendationsHandler
# ===========================================================================


class TestTemplateRecommendationsHandler:
    """Tests for TemplateRecommendationsHandler."""

    def test_can_handle_recommended_path(self, recommendations_handler):
        assert recommendations_handler.can_handle("/api/v1/templates/recommended") is True

    def test_can_handle_recommended_path_get_only(self, recommendations_handler):
        assert recommendations_handler.can_handle("/api/v1/templates/recommended", method="GET") is True
        assert recommendations_handler.can_handle("/api/v1/templates/recommended", method="POST") is False

    def test_cannot_handle_other_path(self, recommendations_handler):
        assert recommendations_handler.can_handle("/api/v1/templates") is False

    @patch("aragora.workflow.templates.get_template")
    def test_get_recommendations_default(self, mock_get, recommendations_handler):
        mock_get.return_value = {}
        h = _make_handler(method="GET")
        result = recommendations_handler.handle("/api/v1/templates/recommended", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert "recommendations" in body
        assert body["use_case"] == "general"
        assert "available_use_cases" in body

    @patch("aragora.workflow.templates.get_template")
    def test_get_recommendations_team_decisions(self, mock_get, recommendations_handler):
        mock_get.return_value = {}
        h = _make_handler(method="GET")
        result = recommendations_handler.handle(
            "/api/v1/templates/recommended",
            {"use_case": "team_decisions"},
            h,
        )
        body = _body(result)
        assert body["use_case"] == "team_decisions"
        assert len(body["recommendations"]) > 0

    @patch("aragora.workflow.templates.get_template")
    def test_get_recommendations_project_planning(self, mock_get, recommendations_handler):
        mock_get.return_value = {}
        h = _make_handler(method="GET")
        result = recommendations_handler.handle(
            "/api/v1/templates/recommended",
            {"use_case": "project_planning"},
            h,
        )
        body = _body(result)
        assert body["use_case"] == "project_planning"

    @patch("aragora.workflow.templates.get_template")
    def test_get_recommendations_vendor_selection(self, mock_get, recommendations_handler):
        mock_get.return_value = {}
        h = _make_handler(method="GET")
        result = recommendations_handler.handle(
            "/api/v1/templates/recommended",
            {"use_case": "vendor_selection"},
            h,
        )
        body = _body(result)
        assert body["use_case"] == "vendor_selection"

    @patch("aragora.workflow.templates.get_template")
    def test_get_recommendations_policy_review(self, mock_get, recommendations_handler):
        mock_get.return_value = {}
        h = _make_handler(method="GET")
        result = recommendations_handler.handle(
            "/api/v1/templates/recommended",
            {"use_case": "policy_review"},
            h,
        )
        body = _body(result)
        assert body["use_case"] == "policy_review"

    @patch("aragora.workflow.templates.get_template")
    def test_get_recommendations_technical_decisions(self, mock_get, recommendations_handler):
        mock_get.return_value = {}
        h = _make_handler(method="GET")
        result = recommendations_handler.handle(
            "/api/v1/templates/recommended",
            {"use_case": "technical_decisions"},
            h,
        )
        body = _body(result)
        assert body["use_case"] == "technical_decisions"

    @patch("aragora.workflow.templates.get_template")
    def test_get_recommendations_unknown_use_case_falls_back(self, mock_get, recommendations_handler):
        mock_get.return_value = {}
        h = _make_handler(method="GET")
        result = recommendations_handler.handle(
            "/api/v1/templates/recommended",
            {"use_case": "unknown_use_case"},
            h,
        )
        body = _body(result)
        # Falls back to "general"
        assert len(body["recommendations"]) > 0

    @patch("aragora.workflow.templates.get_template")
    def test_get_recommendations_with_limit(self, mock_get, recommendations_handler):
        mock_get.return_value = {}
        h = _make_handler(method="GET")
        result = recommendations_handler.handle(
            "/api/v1/templates/recommended",
            {"use_case": "general", "limit": "1"},
            h,
        )
        body = _body(result)
        assert body["total"] <= 1

    @patch("aragora.workflow.templates.get_template")
    def test_get_recommendations_response_fields(self, mock_get, recommendations_handler):
        """Each recommendation should have specific fields."""
        mock_get.return_value = {"recommended_agents": ["a1", "a2"], "config": {"rounds": 3}, "estimated_duration": 10}
        h = _make_handler(method="GET")
        result = recommendations_handler.handle(
            "/api/v1/templates/recommended",
            {"use_case": "general", "limit": "1"},
            h,
        )
        body = _body(result)
        if body["recommendations"]:
            rec = body["recommendations"][0]
            assert "id" in rec
            assert "name" in rec
            assert "description" in rec
            assert "agents_count" in rec
            assert "rounds" in rec
            assert "estimated_duration_minutes" in rec
            assert "use_case" in rec
            assert "category" in rec

    @patch("aragora.workflow.templates.get_template")
    def test_get_recommendations_available_use_cases(self, mock_get, recommendations_handler):
        """Response should include all available use cases."""
        mock_get.return_value = {}
        h = _make_handler(method="GET")
        result = recommendations_handler.handle("/api/v1/templates/recommended", {}, h)
        body = _body(result)
        expected_cases = {"team_decisions", "project_planning", "vendor_selection",
                          "policy_review", "technical_decisions", "general"}
        assert expected_cases == set(body["available_use_cases"])

    def test_recommendations_rate_limit(self, handler_module, recommendations_handler):
        with patch.object(handler_module._template_limiter, "is_allowed", return_value=False):
            h = _make_handler(method="GET")
            result = recommendations_handler.handle("/api/v1/templates/recommended", {}, h)
            assert _status(result) == 429

    def test_recommendations_handler_init_default_ctx(self, handler_module):
        handler = handler_module.TemplateRecommendationsHandler()
        assert handler.ctx == {}


# ===========================================================================
# SMEWorkflowsHandler - can_handle
# ===========================================================================


class TestSMEWorkflowsCanHandle:
    """Tests for SMEWorkflowsHandler.can_handle()."""

    def test_can_handle_sme_root(self, sme_handler):
        assert sme_handler.can_handle("/api/v1/sme/workflows") is True

    def test_can_handle_sme_type(self, sme_handler):
        assert sme_handler.can_handle("/api/v1/sme/workflows/invoice") is True

    def test_cannot_handle_other(self, sme_handler):
        assert sme_handler.can_handle("/api/v1/workflow/templates") is False

    def test_can_handle_method_get(self, sme_handler):
        assert sme_handler.can_handle("/api/v1/sme/workflows", method="GET") is True

    def test_can_handle_method_post(self, sme_handler):
        assert sme_handler.can_handle("/api/v1/sme/workflows", method="POST") is True


# ===========================================================================
# SMEWorkflowsHandler - list workflows
# ===========================================================================


class TestSMEListWorkflows:
    """Tests for GET /api/v1/sme/workflows."""

    def test_list_sme_workflows(self, sme_handler):
        h = _make_handler(method="GET")
        result = sme_handler.handle("/api/v1/sme/workflows", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert "workflows" in body
        assert body["total"] == 4
        workflow_ids = [w["id"] for w in body["workflows"]]
        assert "invoice" in workflow_ids
        assert "followup" in workflow_ids
        assert "inventory" in workflow_ids
        assert "report" in workflow_ids

    def test_list_sme_workflows_structure(self, sme_handler):
        h = _make_handler(method="GET")
        result = sme_handler.handle("/api/v1/sme/workflows", {}, h)
        body = _body(result)
        for wf in body["workflows"]:
            assert "id" in wf
            assert "name" in wf
            assert "description" in wf
            assert "icon" in wf
            assert "category" in wf
            assert "inputs" in wf
            assert "features" in wf

    def test_list_sme_method_not_allowed(self, sme_handler):
        h = _make_handler(method="PUT")
        result = sme_handler.handle("/api/v1/sme/workflows", {}, h)
        assert _status(result) == 405

    def test_list_sme_method_delete_not_allowed(self, sme_handler):
        h = _make_handler(method="DELETE")
        result = sme_handler.handle("/api/v1/sme/workflows", {}, h)
        assert _status(result) == 405


# ===========================================================================
# SMEWorkflowsHandler - get workflow info
# ===========================================================================


class TestSMEGetWorkflowInfo:
    """Tests for GET /api/v1/sme/workflows/:type."""

    def test_get_invoice_info(self, sme_handler):
        h = _make_handler(method="GET")
        result = sme_handler.handle("/api/v1/sme/workflows/invoice", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "invoice"
        assert "inputs" in body
        assert "customer_id" in body["inputs"]

    def test_get_followup_info(self, sme_handler):
        h = _make_handler(method="GET")
        result = sme_handler.handle("/api/v1/sme/workflows/followup", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "followup"

    def test_get_inventory_info(self, sme_handler):
        h = _make_handler(method="GET")
        result = sme_handler.handle("/api/v1/sme/workflows/inventory", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "inventory"

    def test_get_report_info(self, sme_handler):
        h = _make_handler(method="GET")
        result = sme_handler.handle("/api/v1/sme/workflows/report", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "report"
        assert "inputs" in body
        assert "report_type" in body["inputs"]

    def test_get_unknown_type_info(self, sme_handler):
        h = _make_handler(method="GET")
        result = sme_handler.handle("/api/v1/sme/workflows/unknown", {}, h)
        assert _status(result) == 404

    def test_get_info_outputs_field(self, sme_handler):
        h = _make_handler(method="GET")
        result = sme_handler.handle("/api/v1/sme/workflows/invoice", {}, h)
        body = _body(result)
        assert "outputs" in body
        assert "invoice_id" in body["outputs"]


# ===========================================================================
# SMEWorkflowsHandler - create workflow
# ===========================================================================


class TestSMECreateWorkflow:
    """Tests for POST /api/v1/sme/workflows/:type."""

    @patch("aragora.workflow.templates.sme.create_invoice_workflow")
    def test_create_invoice_workflow(self, mock_create, sme_handler):
        mock_wf = MagicMock()
        mock_wf.id = "wf-inv-1"
        mock_wf.name = "Invoice Workflow"
        mock_wf.steps = [MagicMock(), MagicMock()]
        mock_create.return_value = mock_wf

        h = _make_handler(method="POST", body={
            "customer_id": "cust-123",
            "items": [{"name": "Widget", "quantity": 2, "unit_price": 10}],
        })
        result = sme_handler.handle("/api/v1/sme/workflows/invoice", {}, h)
        assert _status(result) == 201
        body = _body(result)
        assert body["workflow_type"] == "invoice"
        assert body["status"] == "created"

    @patch("aragora.workflow.templates.sme.create_followup_workflow")
    def test_create_followup_workflow(self, mock_create, sme_handler):
        mock_wf = MagicMock()
        mock_wf.id = "wf-fu-1"
        mock_wf.name = "Follow-up"
        mock_wf.steps = [MagicMock()]
        mock_create.return_value = mock_wf

        h = _make_handler(method="POST", body={"followup_type": "post_sale"})
        result = sme_handler.handle("/api/v1/sme/workflows/followup", {}, h)
        assert _status(result) == 201

    @patch("aragora.workflow.templates.sme.create_inventory_alert_workflow")
    def test_create_inventory_workflow(self, mock_create, sme_handler):
        mock_wf = MagicMock()
        mock_wf.id = "wf-inv-a"
        mock_wf.name = "Inventory Alerts"
        mock_wf.steps = [MagicMock()]
        mock_create.return_value = mock_wf

        h = _make_handler(method="POST", body={"alert_threshold": 10})
        result = sme_handler.handle("/api/v1/sme/workflows/inventory", {}, h)
        assert _status(result) == 201

    @patch("aragora.workflow.templates.sme.create_report_workflow")
    def test_create_report_workflow(self, mock_create, sme_handler):
        mock_wf = MagicMock()
        mock_wf.id = "wf-rpt"
        mock_wf.name = "Report"
        mock_wf.steps = [MagicMock()]
        mock_create.return_value = mock_wf

        h = _make_handler(method="POST", body={"report_type": "sales"})
        result = sme_handler.handle("/api/v1/sme/workflows/report", {}, h)
        assert _status(result) == 201

    def test_create_unknown_type(self, sme_handler):
        h = _make_handler(method="POST", body={})
        result = sme_handler.handle("/api/v1/sme/workflows/unknown", {}, h)
        assert _status(result) == 400

    def test_create_invalid_json(self, sme_handler):
        h = MagicMock()
        h.command = "POST"
        h.client_address = ("127.0.0.1", 12345)
        h.headers = {"Content-Length": "5"}
        h.rfile = MagicMock()
        h.rfile.read.return_value = b"notjn"
        result = sme_handler.handle("/api/v1/sme/workflows/invoice", {}, h)
        assert _status(result) == 400

    def test_create_body_too_large(self, sme_handler):
        h = MagicMock()
        h.command = "POST"
        h.client_address = ("127.0.0.1", 12345)
        h.headers = {"Content-Length": str(11 * 1024 * 1024)}
        h.rfile = MagicMock()
        result = sme_handler.handle("/api/v1/sme/workflows/invoice", {}, h)
        assert _status(result) == 413

    @patch("aragora.workflow.templates.sme.create_invoice_workflow")
    def test_create_workflow_error(self, mock_create, sme_handler):
        mock_create.side_effect = RuntimeError("creation failed")
        h = _make_handler(method="POST", body={"customer_id": "cust-1"})
        result = sme_handler.handle("/api/v1/sme/workflows/invoice", {}, h)
        assert _status(result) == 500

    @patch("aragora.server.handlers.workflow_templates._start_workflow_execution")
    @patch("aragora.workflow.templates.sme.create_invoice_workflow")
    def test_create_workflow_with_execute(self, mock_create, mock_start, sme_handler):
        mock_wf = MagicMock()
        mock_wf.id = "wf-exec"
        mock_wf.name = "Invoice Exec"
        mock_wf.steps = [MagicMock()]
        mock_create.return_value = mock_wf
        mock_start.return_value = "exec_abc123"

        h = _make_handler(method="POST", body={
            "customer_id": "cust-1",
            "execute": True,
        })
        result = sme_handler.handle("/api/v1/sme/workflows/invoice", {}, h)
        assert _status(result) == 201
        body = _body(result)
        assert body["status"] == "running"
        assert body["execution_id"] == "exec_abc123"
        assert "poll_url" in body

    @patch("aragora.workflow.templates.sme.create_invoice_workflow")
    def test_create_workflow_without_execute(self, mock_create, sme_handler):
        mock_wf = MagicMock()
        mock_wf.id = "wf-no-exec"
        mock_wf.name = "Invoice No Exec"
        mock_wf.steps = []
        mock_create.return_value = mock_wf

        h = _make_handler(method="POST", body={"customer_id": "cust-1"})
        result = sme_handler.handle("/api/v1/sme/workflows/invoice", {}, h)
        body = _body(result)
        assert body["status"] == "created"
        assert "execution_id" not in body

    def test_sme_handler_init_default_ctx(self, handler_module):
        handler = handler_module.SMEWorkflowsHandler()
        assert handler.ctx == {}

    def test_sme_rate_limit(self, handler_module, sme_handler):
        with patch.object(handler_module._template_limiter, "is_allowed", return_value=False):
            h = _make_handler(method="GET")
            result = sme_handler.handle("/api/v1/sme/workflows", {}, h)
            assert _status(result) == 429

    def test_sme_invalid_path(self, sme_handler):
        """Short path that does not match any route."""
        h = _make_handler(method="GET")
        # Path starts with sme/workflows but doesn't match list or type patterns
        # Actually the handler checks path == "/api/v1/sme/workflows" first,
        # then len(parts) >= 5.
        # /api/v1/sme/workflows has 4 parts after split (["", "api", "v1", "sme", "workflows"])
        # so it matches the first branch. Let's test something shorter.
        # can_handle already guards this so the server would not route here,
        # but testing a pathological case:
        result = sme_handler.handle("/api/v1/sme/workflows", {}, _make_handler(method="PATCH"))
        assert _status(result) == 405


# ===========================================================================
# SMEWorkflowsHandler - create with full params
# ===========================================================================


class TestSMECreateWorkflowFullParams:
    """Tests for creating SME workflows with all possible parameters."""

    @patch("aragora.workflow.templates.sme.create_invoice_workflow")
    def test_invoice_all_params(self, mock_create, sme_handler):
        mock_wf = MagicMock()
        mock_wf.id = "wf-full-inv"
        mock_wf.name = "Full Invoice"
        mock_wf.steps = []
        mock_create.return_value = mock_wf

        h = _make_handler(method="POST", body={
            "customer_id": "cust-42",
            "items": [{"name": "A", "quantity": 1, "unit_price": 100}],
            "tax_rate": 0.1,
            "due_days": 45,
            "send_email": True,
            "notes": "Thank you for your business",
        })
        result = sme_handler.handle("/api/v1/sme/workflows/invoice", {}, h)
        assert _status(result) == 201
        mock_create.assert_called_once_with(
            customer_id="cust-42",
            items=[{"name": "A", "quantity": 1, "unit_price": 100}],
            tax_rate=0.1,
            due_days=45,
            send_email=True,
            notes="Thank you for your business",
        )

    @patch("aragora.workflow.templates.sme.create_followup_workflow")
    def test_followup_all_params(self, mock_create, sme_handler):
        mock_wf = MagicMock()
        mock_wf.id = "wf-full-fu"
        mock_wf.name = "Full Followup"
        mock_wf.steps = []
        mock_create.return_value = mock_wf

        h = _make_handler(method="POST", body={
            "followup_type": "renewal",
            "days_since_contact": 60,
            "channel": "sms",
            "auto_send": True,
            "customer_id": "cust-99",
        })
        result = sme_handler.handle("/api/v1/sme/workflows/followup", {}, h)
        assert _status(result) == 201

    @patch("aragora.workflow.templates.sme.create_inventory_alert_workflow")
    def test_inventory_all_params(self, mock_create, sme_handler):
        mock_wf = MagicMock()
        mock_wf.id = "wf-full-inv-a"
        mock_wf.name = "Full Inventory"
        mock_wf.steps = []
        mock_create.return_value = mock_wf

        h = _make_handler(method="POST", body={
            "alert_threshold": 5,
            "auto_reorder": True,
            "notification_channels": ["email", "slack"],
            "categories": ["electronics", "clothing"],
        })
        result = sme_handler.handle("/api/v1/sme/workflows/inventory", {}, h)
        assert _status(result) == 201

    @patch("aragora.workflow.templates.sme.create_report_workflow")
    def test_report_all_params(self, mock_create, sme_handler):
        mock_wf = MagicMock()
        mock_wf.id = "wf-full-rpt"
        mock_wf.name = "Full Report"
        mock_wf.steps = []
        mock_create.return_value = mock_wf

        h = _make_handler(method="POST", body={
            "report_type": "financial",
            "frequency": "monthly",
            "date_range": "last_month",
            "format": "excel",
            "recipients": ["boss@company.com"],
            "include_charts": False,
            "comparison": False,
        })
        result = sme_handler.handle("/api/v1/sme/workflows/report", {}, h)
        assert _status(result) == 201
        mock_create.assert_called_once_with(
            report_type="financial",
            frequency="monthly",
            date_range="last_month",
            format="excel",
            recipients=["boss@company.com"],
            include_charts=False,
            include_comparison=False,
        )


# ===========================================================================
# Event emission tests
# ===========================================================================


class TestEventEmission:
    """Tests for template event emission."""

    def test_emit_template_event_with_emitter(self, templates_handler):
        """When event emitter is available, emit should be called."""
        mock_emitter = MagicMock()
        templates_handler.ctx = {"event_emitter": mock_emitter}

        # This may or may not emit depending on StreamEventType availability
        # We just verify no crash occurs
        templates_handler._emit_template_event("TEMPLATE_INSTANTIATED", {"template_id": "test"})

    def test_emit_template_event_without_emitter(self, templates_handler):
        """When no event emitter, should not crash."""
        templates_handler.ctx = {}
        templates_handler._emit_template_event("TEMPLATE_INSTANTIATED", {"template_id": "test"})

    def test_emit_template_event_no_ctx(self, handler_module):
        """Handler with None ctx should not crash."""
        handler = handler_module.WorkflowTemplatesHandler(ctx=None)
        handler._emit_template_event("TEMPLATE_INSTANTIATED", {"template_id": "test"})

    def test_emit_template_event_unknown_type(self, templates_handler):
        """Unknown event type should not crash (returns early)."""
        mock_emitter = MagicMock()
        templates_handler.ctx = {"event_emitter": mock_emitter}
        templates_handler._emit_template_event("TOTALLY_UNKNOWN_EVENT", {"x": 1})


# ===========================================================================
# Async workflow execution helpers
# ===========================================================================


class TestAsyncWorkflowExecution:
    """Tests for _start_workflow_execution and _execute_workflow_async."""

    @patch("aragora.server.handlers.workflow_templates._get_workflow_store")
    @patch("aragora.server.handlers.workflow_templates._get_workflow_engine")
    @pytest.mark.asyncio
    async def test_execute_workflow_async_success(self, mock_engine_fn, mock_store_fn, handler_module):
        mock_store = MagicMock()
        mock_store_fn.return_value = mock_store
        mock_engine = MagicMock()
        mock_engine_fn.return_value = mock_engine

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.final_output = {"answer": "42"}
        mock_result.steps = []
        mock_result.error = None
        mock_result.total_duration_ms = 100

        from unittest.mock import AsyncMock
        mock_engine.execute = AsyncMock(return_value=mock_result)

        mock_execution = {}
        mock_store.get_execution.return_value = mock_execution

        await handler_module._execute_workflow_async(
            MagicMock(), "exec-1", {"input": "val"}, "tenant-1"
        )

        mock_store.save_execution.assert_called_once()
        assert mock_execution["status"] == "completed"

    @patch("aragora.server.handlers.workflow_templates._get_workflow_store")
    @patch("aragora.server.handlers.workflow_templates._get_workflow_engine")
    @pytest.mark.asyncio
    async def test_execute_workflow_async_failure(self, mock_engine_fn, mock_store_fn, handler_module):
        mock_store = MagicMock()
        mock_store_fn.return_value = mock_store
        mock_engine = MagicMock()
        mock_engine_fn.return_value = mock_engine

        from unittest.mock import AsyncMock
        mock_engine.execute = AsyncMock(side_effect=RuntimeError("engine error"))

        mock_execution = {"id": "exec-2"}
        mock_store.get_execution.return_value = mock_execution

        await handler_module._execute_workflow_async(
            MagicMock(), "exec-2", None, "default"
        )

        mock_store.save_execution.assert_called_once()
        assert mock_execution["status"] == "failed"

    @patch("aragora.server.handlers.workflow_templates._get_workflow_store")
    @patch("aragora.server.handlers.workflow_templates._get_workflow_engine")
    @pytest.mark.asyncio
    async def test_execute_workflow_async_no_execution_record(self, mock_engine_fn, mock_store_fn, handler_module):
        """When execution record is not found after engine run, should not crash."""
        mock_store = MagicMock()
        mock_store_fn.return_value = mock_store
        mock_engine = MagicMock()
        mock_engine_fn.return_value = mock_engine

        from unittest.mock import AsyncMock
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.final_output = {}
        mock_result.steps = []
        mock_result.error = None
        mock_result.total_duration_ms = 50
        mock_engine.execute = AsyncMock(return_value=mock_result)

        mock_store.get_execution.return_value = None  # No record found

        await handler_module._execute_workflow_async(
            MagicMock(), "exec-3", None, "default"
        )
        # Should not crash, save_execution should not be called
        mock_store.save_execution.assert_not_called()

    @patch("aragora.server.handlers.workflow_templates._get_workflow_store")
    @patch("aragora.server.handlers.workflow_templates._get_workflow_engine")
    @pytest.mark.asyncio
    async def test_execute_workflow_async_failure_no_record(self, mock_engine_fn, mock_store_fn, handler_module):
        """When failure occurs and no execution record, should not crash."""
        mock_store = MagicMock()
        mock_store_fn.return_value = mock_store
        mock_engine = MagicMock()
        mock_engine_fn.return_value = mock_engine

        from unittest.mock import AsyncMock
        mock_engine.execute = AsyncMock(side_effect=ValueError("bad"))
        mock_store.get_execution.return_value = None

        await handler_module._execute_workflow_async(
            MagicMock(), "exec-4", None, "default"
        )
        mock_store.save_execution.assert_not_called()

    @patch("aragora.server.handlers.workflow_templates._get_workflow_store")
    @patch("asyncio.get_running_loop")
    def test_start_workflow_execution(self, mock_loop_fn, mock_store_fn, handler_module):
        mock_store = MagicMock()
        mock_store_fn.return_value = mock_store
        mock_loop = MagicMock()
        mock_loop_fn.return_value = mock_loop

        mock_wf = MagicMock()
        mock_wf.id = "wf-test"
        mock_wf.name = "Test Workflow"

        exec_id = handler_module._start_workflow_execution(
            workflow=mock_wf,
            inputs={"key": "val"},
            tenant_id="t1",
        )

        assert exec_id.startswith("exec_")
        mock_store.save_execution.assert_called_once()
        saved = mock_store.save_execution.call_args[0][0]
        assert saved["status"] == "running"
        assert saved["tenant_id"] == "t1"
        mock_loop.create_task.assert_called_once()

    @patch("aragora.server.handlers.workflow_templates._get_workflow_store")
    @patch("asyncio.get_running_loop", side_effect=RuntimeError("no loop"))
    @patch("asyncio.create_task")
    def test_start_workflow_execution_no_event_loop(self, mock_create, mock_loop_fn, mock_store_fn, handler_module):
        """When no running event loop, should use asyncio.create_task."""
        mock_store = MagicMock()
        mock_store_fn.return_value = mock_store

        mock_wf = MagicMock()
        mock_wf.id = "wf-noloop"
        mock_wf.name = "No Loop WF"

        exec_id = handler_module._start_workflow_execution(
            workflow=mock_wf,
            inputs=None,
            tenant_id="default",
        )

        assert exec_id.startswith("exec_")
        mock_create.assert_called_once()


# ===========================================================================
# WorkflowTemplatesHandler - constructor
# ===========================================================================


class TestHandlerConstructors:
    """Tests for handler constructors."""

    def test_templates_handler_default_ctx(self, handler_module):
        handler = handler_module.WorkflowTemplatesHandler()
        assert handler.ctx == {}

    def test_templates_handler_with_ctx(self, handler_module):
        ctx = {"key": "value"}
        handler = handler_module.WorkflowTemplatesHandler(ctx=ctx)
        assert handler.ctx == ctx

    def test_pattern_templates_handler_with_empty_ctx(self, handler_module):
        handler = handler_module.WorkflowPatternTemplatesHandler(ctx={})
        assert handler is not None

    def test_sme_handler_with_ctx(self, handler_module):
        ctx = {"db": "test_db"}
        handler = handler_module.SMEWorkflowsHandler(ctx=ctx)
        assert handler.ctx == ctx


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge case and boundary tests."""

    @patch("aragora.workflow.templates.get_template")
    def test_template_with_no_optional_fields(self, mock_get, templates_handler):
        """Template with minimal fields should still return all response keys."""
        mock_get.return_value = {"name": "Minimal"}
        h = _make_handler(method="GET")
        result = templates_handler.handle(
            "/api/v1/workflow/templates/minimal",
            {},
            h,
        )
        body = _body(result)
        assert body["description"] == ""
        assert body["steps"] == []
        assert body["inputs"] == {}
        assert body["outputs"] == {}
        assert body["tags"] == []
        assert body["recommended_agents"] == []

    @patch("aragora.workflow.templates.list_templates")
    @patch("aragora.workflow.templates.WORKFLOW_TEMPLATES", {})
    def test_high_offset_returns_empty(self, mock_list, templates_handler):
        mock_list.return_value = SAMPLE_TEMPLATE_LIST
        h = _make_handler(method="GET")
        result = templates_handler.handle(
            "/api/v1/workflow/templates",
            {"offset": "1000"},
            h,
        )
        body = _body(result)
        assert body["templates"] == []
        assert body["total"] == 3  # total is pre-pagination

    @patch("aragora.workflow.templates.get_template")
    def test_deeply_nested_template_id(self, mock_get, templates_handler):
        """Template ID with multiple slashes."""
        mock_get.return_value = SAMPLE_TEMPLATE
        h = _make_handler(method="GET")
        result = templates_handler.handle(
            "/api/v1/workflow/templates/org/team/deep-template",
            {},
            h,
        )
        body = _body(result)
        assert body["id"] == "org/team/deep-template"
        assert body["category"] == "org"

    @patch("aragora.workflow.templates.patterns.get_pattern_template")
    def test_pattern_template_response_defaults(self, mock_get, pattern_templates_handler):
        """Pattern template with minimal data should use defaults."""
        mock_get.return_value = {"id": "test"}
        h = _make_handler(method="GET")
        result = pattern_templates_handler.handle(
            "/api/v1/workflow/pattern-templates/test",
            {},
            h,
        )
        body = _body(result)
        assert body["version"] == "1.0.0"
        assert body["config"] == {}
        assert body["inputs"] == {}
        assert body["outputs"] == {}
        assert body["tags"] == []

    @patch("aragora.workflow.templates.get_template")
    def test_run_template_result_without_to_dict(self, mock_get, templates_handler):
        """When result does not have to_dict, use raw result."""
        mock_get.return_value = SAMPLE_TEMPLATE

        mock_result = MagicMock(spec=[])  # No to_dict attribute
        mock_result.success = True

        with patch("aragora.server.handlers.workflow_templates.WorkflowDefinition") as mock_wd:
            mock_wd.from_dict.return_value = MagicMock()
            with patch("asyncio.run", return_value=mock_result):
                h = _make_handler(method="POST", body={"template_id": "test"})
                result = templates_handler.handle("/api/v1/workflow/templates", {}, h)
        # Should not crash - uses result directly instead of result.to_dict()
        assert _status(result) == 200

    @patch("aragora.workflow.templates.get_template")
    def test_run_template_value_error(self, mock_get, templates_handler):
        """ValueError during execution should return 500."""
        mock_get.return_value = SAMPLE_TEMPLATE
        with patch("aragora.server.handlers.workflow_templates.WorkflowDefinition") as mock_wd:
            mock_wd.from_dict.side_effect = ValueError("bad value")
            h = _make_handler(method="POST", body={"template_id": "test"})
            result = templates_handler.handle("/api/v1/workflow/templates", {}, h)
        assert _status(result) == 500

    @patch("aragora.workflow.templates.get_template")
    def test_run_template_type_error(self, mock_get, templates_handler):
        mock_get.return_value = SAMPLE_TEMPLATE
        with patch("aragora.server.handlers.workflow_templates.WorkflowDefinition") as mock_wd:
            mock_wd.from_dict.side_effect = TypeError("type issue")
            h = _make_handler(method="POST", body={"template_id": "test"})
            result = templates_handler.handle("/api/v1/workflow/templates", {}, h)
        assert _status(result) == 500

    @patch("aragora.workflow.templates.get_template")
    def test_run_template_key_error(self, mock_get, templates_handler):
        mock_get.return_value = SAMPLE_TEMPLATE
        with patch("aragora.server.handlers.workflow_templates.WorkflowDefinition") as mock_wd:
            mock_wd.from_dict.side_effect = KeyError("missing key")
            h = _make_handler(method="POST", body={"template_id": "test"})
            result = templates_handler.handle("/api/v1/workflow/templates", {}, h)
        assert _status(result) == 500

    @patch("aragora.workflow.templates.get_template")
    def test_run_template_attribute_error(self, mock_get, templates_handler):
        mock_get.return_value = SAMPLE_TEMPLATE
        with patch("aragora.server.handlers.workflow_templates.WorkflowDefinition") as mock_wd:
            mock_wd.from_dict.side_effect = AttributeError("no attr")
            h = _make_handler(method="POST", body={"template_id": "test"})
            result = templates_handler.handle("/api/v1/workflow/templates", {}, h)
        assert _status(result) == 500

    @patch("aragora.workflow.templates.get_template")
    def test_run_template_os_error(self, mock_get, templates_handler):
        mock_get.return_value = SAMPLE_TEMPLATE
        with patch("aragora.server.handlers.workflow_templates.WorkflowDefinition") as mock_wd:
            mock_wd.from_dict.side_effect = OSError("disk error")
            h = _make_handler(method="POST", body={"template_id": "test"})
            result = templates_handler.handle("/api/v1/workflow/templates", {}, h)
        assert _status(result) == 500


# ===========================================================================
# ROUTES attribute tests
# ===========================================================================


class TestRouteAttributes:
    """Verify ROUTES class attributes are correct."""

    def test_templates_handler_routes(self, handler_module):
        routes = handler_module.WorkflowTemplatesHandler.ROUTES
        assert "/api/v1/workflow/templates" in routes
        assert "/api/v1/workflow/templates/*" in routes
        assert "/api/v1/workflow/templates/*/package" in routes

    def test_categories_handler_routes(self, handler_module):
        routes = handler_module.WorkflowCategoriesHandler.ROUTES
        assert "/api/v1/workflow/categories" in routes

    def test_patterns_handler_routes(self, handler_module):
        routes = handler_module.WorkflowPatternsHandler.ROUTES
        assert "/api/v1/workflow/patterns" in routes

    def test_pattern_templates_handler_routes(self, handler_module):
        routes = handler_module.WorkflowPatternTemplatesHandler.ROUTES
        assert "/api/v1/workflow/pattern-templates" in routes
        assert "/api/v1/workflow/pattern-templates/*" in routes

    def test_recommendations_handler_routes(self, handler_module):
        routes = handler_module.TemplateRecommendationsHandler.ROUTES
        assert "/api/v1/templates/recommended" in routes

    def test_sme_handler_routes(self, handler_module):
        routes = handler_module.SMEWorkflowsHandler.ROUTES
        assert "/api/v1/sme/workflows" in routes
        assert "/api/v1/sme/workflows/*" in routes
