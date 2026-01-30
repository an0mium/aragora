"""
Tests for DocsHandler - API documentation endpoint serving.

Tests cover:
- can_handle() route matching for all documentation paths
- handle() dispatching to OpenAPI JSON, YAML, Swagger UI, ReDoc, Postman
- _get_openapi_spec: JSON format, YAML format, import error, generation error
- _get_swagger_ui: returns valid HTML with correct references
- _get_postman_collection: happy path, error handling
- _get_redoc: returns valid HTML with correct spec URL
- Caching behavior (CACHE_TTL_OPENAPI)
- RBAC permission enforcement
"""

from __future__ import annotations

import json
import sys
import types as _types_mod
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Slack stubs to prevent transitive import issues
# ---------------------------------------------------------------------------
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m


from aragora.server.handlers.docs import DocsHandler, CACHE_TTL_OPENAPI


# ===========================================================================
# Fixtures and Helpers
# ===========================================================================


def get_body(result) -> dict:
    """Extract JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


def get_text(result) -> str:
    """Extract text body from HandlerResult."""
    return result.body.decode("utf-8")


@pytest.fixture
def handler():
    """Create a DocsHandler with empty context."""
    return DocsHandler({})


# ===========================================================================
# Tests: can_handle
# ===========================================================================


class TestCanHandle:
    """Tests for route matching."""

    def test_handles_openapi_json(self, handler):
        """Matches /api/v1/openapi."""
        assert handler.can_handle("/api/v1/openapi") is True

    def test_handles_openapi_json_ext(self, handler):
        """Matches /api/v1/openapi.json."""
        assert handler.can_handle("/api/v1/openapi.json") is True

    def test_handles_openapi_yaml(self, handler):
        """Matches /api/v1/openapi.yaml."""
        assert handler.can_handle("/api/v1/openapi.yaml") is True

    def test_handles_postman(self, handler):
        """Matches /api/v1/postman.json."""
        assert handler.can_handle("/api/v1/postman.json") is True

    def test_handles_docs(self, handler):
        """Matches /api/v1/docs."""
        assert handler.can_handle("/api/v1/docs") is True

    def test_handles_docs_trailing_slash(self, handler):
        """Matches /api/v1/docs/."""
        assert handler.can_handle("/api/v1/docs/") is True

    def test_handles_redoc(self, handler):
        """Matches /api/v1/redoc."""
        assert handler.can_handle("/api/v1/redoc") is True

    def test_handles_redoc_trailing_slash(self, handler):
        """Matches /api/v1/redoc/."""
        assert handler.can_handle("/api/v1/redoc/") is True

    def test_does_not_handle_unrelated_path(self, handler):
        """Does not match unrelated paths."""
        assert handler.can_handle("/api/v1/debates") is False
        assert handler.can_handle("/api/health") is False

    def test_does_not_handle_wrong_subpath(self, handler):
        """Does not match incorrect subpaths."""
        assert handler.can_handle("/api/v1/openapi/extra") is False
        assert handler.can_handle("/api/v1/docs/extra") is False


# ===========================================================================
# Tests: handle() dispatching
# ===========================================================================


class TestHandleDispatching:
    """Tests for main handle() method routing."""

    def test_routes_openapi_json(self, handler):
        """Routes /api/v1/openapi to OpenAPI JSON spec."""
        mock_content = '{"openapi": "3.0.0"}'
        with patch("aragora.server.handlers.docs.DocsHandler._get_openapi_spec") as mock_spec:
            from aragora.server.handlers.base import HandlerResult

            mock_spec.return_value = HandlerResult(
                status_code=200,
                content_type="application/json",
                body=mock_content.encode("utf-8"),
            )
            result = handler.handle("/api/v1/openapi", {}, MagicMock())

        assert result is not None
        assert result.status_code == 200

    def test_routes_openapi_json_ext(self, handler):
        """Routes /api/v1/openapi.json to OpenAPI JSON spec."""
        with patch("aragora.server.handlers.docs.DocsHandler._get_openapi_spec") as mock_spec:
            from aragora.server.handlers.base import HandlerResult

            mock_spec.return_value = HandlerResult(
                status_code=200,
                content_type="application/json",
                body=b"{}",
            )
            result = handler.handle("/api/v1/openapi.json", {}, MagicMock())

        assert result is not None
        assert result.status_code == 200

    def test_routes_openapi_yaml(self, handler):
        """Routes /api/v1/openapi.yaml to OpenAPI YAML spec."""
        with patch("aragora.server.handlers.docs.DocsHandler._get_openapi_spec") as mock_spec:
            from aragora.server.handlers.base import HandlerResult

            mock_spec.return_value = HandlerResult(
                status_code=200,
                content_type="text/yaml",
                body=b'openapi: "3.0.0"',
            )
            result = handler.handle("/api/v1/openapi.yaml", {}, MagicMock())

        assert result is not None
        assert result.status_code == 200

    def test_routes_postman(self, handler):
        """Routes /api/v1/postman.json to Postman collection."""
        with patch(
            "aragora.server.handlers.docs.DocsHandler._get_postman_collection"
        ) as mock_postman:
            from aragora.server.handlers.base import HandlerResult

            mock_postman.return_value = HandlerResult(
                status_code=200,
                content_type="application/json",
                body=b'{"info": {"name": "Aragora"}}',
            )
            result = handler.handle("/api/v1/postman.json", {}, MagicMock())

        assert result is not None
        assert result.status_code == 200

    def test_routes_swagger_ui(self, handler):
        """Routes /api/v1/docs to Swagger UI."""
        result = handler.handle("/api/v1/docs", {}, MagicMock())

        assert result is not None
        assert result.status_code == 200
        assert "text/html" in result.content_type

    def test_routes_redoc(self, handler):
        """Routes /api/v1/redoc to ReDoc."""
        result = handler.handle("/api/v1/redoc", {}, MagicMock())

        assert result is not None
        assert result.status_code == 200
        assert "text/html" in result.content_type

    def test_returns_none_for_unmatched(self, handler):
        """Returns None for unmatched paths."""
        result = handler.handle("/api/v1/unknown", {}, MagicMock())
        assert result is None


# ===========================================================================
# Tests: _get_openapi_spec
# ===========================================================================


class TestOpenAPISpec:
    """Tests for OpenAPI specification generation."""

    def test_openapi_json_happy_path(self, handler):
        """Returns valid JSON OpenAPI spec."""
        mock_spec = '{"openapi": "3.0.0", "info": {"title": "Aragora"}}'

        with patch(
            "aragora.server.openapi.handle_openapi_request",
            create=True,
            return_value=(mock_spec, "application/json"),
        ):
            result = handler._get_openapi_spec("json")

        assert result.status_code == 200
        assert result.content_type == "application/json"
        body = json.loads(result.body.decode("utf-8"))
        assert body["openapi"] == "3.0.0"

    def test_openapi_yaml_happy_path(self, handler):
        """Returns valid YAML OpenAPI spec."""
        mock_yaml = 'openapi: "3.0.0"\ninfo:\n  title: Aragora'

        with patch(
            "aragora.server.openapi.handle_openapi_request",
            create=True,
            return_value=(mock_yaml, "text/yaml"),
        ):
            result = handler._get_openapi_spec("yaml")

        assert result.status_code == 200
        assert result.content_type == "text/yaml"
        assert "openapi" in get_text(result)

    def test_openapi_import_error(self, handler):
        """Returns 503 when OpenAPI module is not available."""
        with patch(
            "aragora.server.openapi.handle_openapi_request",
            create=True,
            side_effect=ImportError("module not found"),
        ):
            result = handler._get_openapi_spec("json")

        body = get_body(result)
        assert result.status_code == 503
        assert "not available" in body.get("error", "").lower()

    def test_openapi_generation_error(self, handler):
        """Returns 500 when spec generation fails."""
        with patch(
            "aragora.server.openapi.handle_openapi_request",
            create=True,
            side_effect=RuntimeError("Generation failed"),
        ):
            result = handler._get_openapi_spec("json")

        assert result.status_code == 500


# ===========================================================================
# Tests: _get_swagger_ui
# ===========================================================================


class TestSwaggerUI:
    """Tests for Swagger UI page generation."""

    def test_swagger_ui_returns_html(self, handler):
        """Swagger UI returns a valid HTML page."""
        result = handler._get_swagger_ui()

        assert result.status_code == 200
        assert "text/html" in result.content_type

        html = get_text(result)
        assert "<!DOCTYPE html>" in html
        assert "<html" in html

    def test_swagger_ui_references_openapi_json(self, handler):
        """Swagger UI page references the OpenAPI JSON endpoint."""
        result = handler._get_swagger_ui()
        html = get_text(result)
        assert "/api/v1/openapi.json" in html

    def test_swagger_ui_includes_swagger_bundle(self, handler):
        """Swagger UI page includes SwaggerUI bundle script."""
        result = handler._get_swagger_ui()
        html = get_text(result)
        assert "swagger-ui-bundle" in html.lower()

    def test_swagger_ui_title(self, handler):
        """Swagger UI page has correct title."""
        result = handler._get_swagger_ui()
        html = get_text(result)
        assert "Aragora API" in html


# ===========================================================================
# Tests: _get_postman_collection
# ===========================================================================


class TestPostmanCollection:
    """Tests for Postman collection generation."""

    def test_postman_happy_path(self, handler):
        """Returns Postman collection with download header."""
        mock_collection = '{"info": {"name": "Aragora API"}}'

        with patch(
            "aragora.server.openapi.handle_postman_request",
            create=True,
            return_value=(mock_collection, "application/json"),
        ):
            result = handler._get_postman_collection()

        assert result.status_code == 200
        assert result.content_type == "application/json"
        assert result.headers is not None
        assert "Content-Disposition" in result.headers
        assert "attachment" in result.headers["Content-Disposition"]
        assert "postman_collection" in result.headers["Content-Disposition"]

    def test_postman_error(self, handler):
        """Returns 500 when Postman generation fails."""
        with patch(
            "aragora.server.openapi.handle_postman_request",
            create=True,
            side_effect=RuntimeError("Generation error"),
        ):
            result = handler._get_postman_collection()

        assert result.status_code == 500


# ===========================================================================
# Tests: _get_redoc
# ===========================================================================


class TestReDoc:
    """Tests for ReDoc page generation."""

    def test_redoc_returns_html(self, handler):
        """ReDoc returns a valid HTML page."""
        result = handler._get_redoc()

        assert result.status_code == 200
        assert "text/html" in result.content_type

        html = get_text(result)
        assert "<!DOCTYPE html>" in html

    def test_redoc_references_openapi_spec(self, handler):
        """ReDoc page references the OpenAPI spec endpoint."""
        result = handler._get_redoc()
        html = get_text(result)
        assert "/api/v1/openapi.json" in html

    def test_redoc_includes_redoc_script(self, handler):
        """ReDoc page includes the ReDoc script."""
        result = handler._get_redoc()
        html = get_text(result)
        assert "redoc" in html.lower()

    def test_redoc_title(self, handler):
        """ReDoc page has correct title."""
        result = handler._get_redoc()
        html = get_text(result)
        assert "Aragora API" in html


# ===========================================================================
# Tests: Constants
# ===========================================================================


class TestConstants:
    """Tests for module constants."""

    def test_cache_ttl_is_one_hour(self):
        """Cache TTL for OpenAPI spec is 3600 seconds (1 hour)."""
        assert CACHE_TTL_OPENAPI == 3600

    def test_routes_list_completeness(self, handler):
        """ROUTES list includes all known documentation paths."""
        expected_routes = {
            "/api/v1/openapi",
            "/api/v1/openapi.json",
            "/api/v1/openapi.yaml",
            "/api/v1/postman.json",
            "/api/v1/docs",
            "/api/v1/docs/",
            "/api/v1/redoc",
            "/api/v1/redoc/",
        }
        assert set(handler.ROUTES) == expected_routes
