"""
Tests for DocsHandler endpoints.

Endpoints tested:
- GET /api/openapi - OpenAPI 3.0 JSON specification
- GET /api/openapi.json - OpenAPI 3.0 JSON specification
- GET /api/openapi.yaml - OpenAPI 3.0 YAML specification
- GET /api/postman.json - Postman collection export
- GET /api/docs - Swagger UI interactive documentation
- GET /api/redoc - ReDoc API documentation viewer
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from aragora.server.handlers.docs import DocsHandler, CACHE_TTL_OPENAPI
from aragora.server.handlers.base import clear_cache


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def docs_handler():
    """Create a DocsHandler with minimal context."""
    ctx = {
        "storage": None,
        "elo_system": None,
    }
    return DocsHandler(ctx)


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before and after each test."""
    clear_cache()
    yield
    clear_cache()


# ============================================================================
# Route Matching Tests
# ============================================================================


class TestDocsRouting:
    """Tests for route matching."""

    def test_can_handle_openapi(self, docs_handler):
        """Handler can handle /api/openapi."""
        assert docs_handler.can_handle("/api/v1/openapi") is True

    def test_can_handle_openapi_json(self, docs_handler):
        """Handler can handle /api/openapi.json."""
        assert docs_handler.can_handle("/api/v1/openapi.json") is True

    def test_can_handle_openapi_yaml(self, docs_handler):
        """Handler can handle /api/openapi.yaml."""
        assert docs_handler.can_handle("/api/v1/openapi.yaml") is True

    def test_can_handle_postman(self, docs_handler):
        """Handler can handle /api/postman.json."""
        assert docs_handler.can_handle("/api/v1/postman.json") is True

    def test_can_handle_docs(self, docs_handler):
        """Handler can handle /api/docs."""
        assert docs_handler.can_handle("/api/v1/docs") is True

    def test_can_handle_docs_trailing_slash(self, docs_handler):
        """Handler can handle /api/docs/."""
        assert docs_handler.can_handle("/api/v1/docs/") is True

    def test_can_handle_redoc(self, docs_handler):
        """Handler can handle /api/redoc."""
        assert docs_handler.can_handle("/api/v1/redoc") is True

    def test_can_handle_redoc_trailing_slash(self, docs_handler):
        """Handler can handle /api/redoc/."""
        assert docs_handler.can_handle("/api/v1/redoc/") is True

    def test_cannot_handle_unrelated_routes(self, docs_handler):
        """Handler doesn't handle unrelated routes."""
        assert docs_handler.can_handle("/api/v1/debates") is False
        assert docs_handler.can_handle("/api/v1/agents") is False
        assert docs_handler.can_handle("/api/v1/replays") is False
        assert docs_handler.can_handle("/api/v1/openapi/extra") is False


# ============================================================================
# GET /api/docs (Swagger UI) Tests
# ============================================================================


class TestSwaggerUI:
    """Tests for GET /api/docs endpoint."""

    def test_swagger_ui_returns_html(self, docs_handler):
        """Returns HTML content for Swagger UI."""
        result = docs_handler.handle("/api/docs", {}, None)

        assert result is not None
        assert result.status_code == 200
        assert result.content_type == "text/html; charset=utf-8"
        assert isinstance(result.body, bytes)

    def test_swagger_ui_contains_required_elements(self, docs_handler):
        """Swagger UI HTML contains necessary elements."""
        result = docs_handler.handle("/api/docs", {}, None)

        html = result.body.decode("utf-8")
        assert "swagger-ui" in html
        assert "SwaggerUIBundle" in html
        assert "/api/openapi.json" in html
        assert "Aragora API" in html

    def test_swagger_ui_trailing_slash(self, docs_handler):
        """Swagger UI works with trailing slash."""
        result = docs_handler.handle("/api/docs/", {}, None)

        assert result is not None
        assert result.status_code == 200
        assert "swagger-ui" in result.body.decode("utf-8")


# ============================================================================
# GET /api/redoc Tests
# ============================================================================


class TestRedoc:
    """Tests for GET /api/redoc endpoint."""

    def test_redoc_returns_html(self, docs_handler):
        """Returns HTML content for ReDoc."""
        result = docs_handler.handle("/api/redoc", {}, None)

        assert result is not None
        assert result.status_code == 200
        assert result.content_type == "text/html; charset=utf-8"
        assert isinstance(result.body, bytes)

    def test_redoc_contains_required_elements(self, docs_handler):
        """ReDoc HTML contains necessary elements."""
        result = docs_handler.handle("/api/redoc", {}, None)

        html = result.body.decode("utf-8")
        assert "redoc" in html
        assert "/api/openapi.json" in html
        assert "Aragora API" in html

    def test_redoc_trailing_slash(self, docs_handler):
        """ReDoc works with trailing slash."""
        result = docs_handler.handle("/api/redoc/", {}, None)

        assert result is not None
        assert result.status_code == 200
        assert "redoc" in result.body.decode("utf-8")


# ============================================================================
# GET /api/openapi Tests
# ============================================================================


class TestOpenAPISpec:
    """Tests for GET /api/openapi endpoints."""

    def test_openapi_json_success(self, docs_handler):
        """Returns OpenAPI JSON spec when module is available."""
        mock_content = '{"openapi": "3.0.0", "info": {"title": "Test"}}'
        mock_content_type = "application/json"

        with patch(
            "aragora.server.openapi.handle_openapi_request",
            return_value=(mock_content, mock_content_type),
        ):
            result = docs_handler.handle("/api/openapi", {}, None)

            assert result is not None
            assert result.status_code == 200
            assert result.content_type == "application/json"
            assert result.body == mock_content.encode("utf-8")

    def test_openapi_json_explicit(self, docs_handler):
        """Returns OpenAPI JSON spec for /api/openapi.json."""
        mock_content = '{"openapi": "3.0.0"}'
        mock_content_type = "application/json"

        with patch(
            "aragora.server.openapi.handle_openapi_request",
            return_value=(mock_content, mock_content_type),
        ):
            result = docs_handler.handle("/api/openapi.json", {}, None)

            assert result is not None
            assert result.status_code == 200
            assert result.content_type == "application/json"

    def test_openapi_yaml_success(self, docs_handler):
        """Returns OpenAPI YAML spec for /api/openapi.yaml."""
        mock_content = "openapi: 3.0.0\ninfo:\n  title: Test"
        mock_content_type = "application/x-yaml"

        with patch(
            "aragora.server.openapi.handle_openapi_request",
            return_value=(mock_content, mock_content_type),
        ):
            result = docs_handler.handle("/api/openapi.yaml", {}, None)

            assert result is not None
            assert result.status_code == 200
            assert result.content_type == "application/x-yaml"
            assert result.body == mock_content.encode("utf-8")

    def test_openapi_import_error(self, docs_handler):
        """Returns 503 when OpenAPI module is not available."""
        import sys

        # Temporarily remove the module to simulate ImportError
        original_modules = {}
        modules_to_remove = [k for k in sys.modules if k.startswith("aragora.server.openapi")]
        for mod in modules_to_remove:
            original_modules[mod] = sys.modules.pop(mod, None)

        with patch.dict(
            sys.modules,
            {"aragora.server.openapi": None},
        ):
            clear_cache()
            result = docs_handler.handle("/api/openapi", {}, None)

            assert result is not None
            assert result.status_code == 503
            data = json.loads(result.body)
            assert "error" in data
            assert "not available" in data["error"].lower()

        # Restore modules
        for mod, val in original_modules.items():
            if val is not None:
                sys.modules[mod] = val

    def test_openapi_generation_error(self, docs_handler):
        """Returns 500 on OpenAPI generation error."""
        with patch(
            "aragora.server.openapi.handle_openapi_request",
            side_effect=ValueError("Generation failed"),
        ):
            clear_cache()
            result = docs_handler.handle("/api/openapi", {}, None)

            assert result is not None
            assert result.status_code == 500
            data = json.loads(result.body)
            assert "error" in data


# ============================================================================
# GET /api/postman.json Tests
# ============================================================================


class TestPostmanCollection:
    """Tests for GET /api/postman.json endpoint."""

    def test_postman_success(self, docs_handler):
        """Returns Postman collection JSON."""
        mock_content = '{"info": {"name": "Aragora"}}'
        mock_content_type = "application/json"

        with patch(
            "aragora.server.openapi.handle_postman_request",
            return_value=(mock_content, mock_content_type),
        ):
            result = docs_handler.handle("/api/postman.json", {}, None)

            assert result is not None
            assert result.status_code == 200
            assert result.content_type == "application/json"
            assert result.body == mock_content.encode("utf-8")

    def test_postman_has_download_header(self, docs_handler):
        """Postman collection includes Content-Disposition header."""
        mock_content = '{"info": {"name": "Aragora"}}'
        mock_content_type = "application/json"

        with patch(
            "aragora.server.openapi.handle_postman_request",
            return_value=(mock_content, mock_content_type),
        ):
            result = docs_handler.handle("/api/postman.json", {}, None)

            assert result is not None
            assert result.headers is not None
            assert "Content-Disposition" in result.headers
            assert "aragora.postman_collection.json" in result.headers["Content-Disposition"]

    def test_postman_error(self, docs_handler):
        """Returns 500 on Postman generation error."""
        with patch(
            "aragora.server.openapi.handle_postman_request",
            side_effect=Exception("Generation failed"),
        ):
            result = docs_handler.handle("/api/postman.json", {}, None)

            assert result is not None
            assert result.status_code == 500
            data = json.loads(result.body)
            assert "error" in data


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestDocsErrorHandling:
    """Tests for error handling."""

    def test_handle_returns_none_for_unhandled(self, docs_handler):
        """Returns None for unhandled routes."""
        result = docs_handler.handle("/api/other/endpoint", {}, None)
        assert result is None

    def test_handles_post_not_implemented(self, docs_handler):
        """DocsHandler doesn't implement handle_post."""
        assert (
            not hasattr(docs_handler, "handle_post")
            or docs_handler.handle_post("/api/docs", {}, None) is None
        )


# ============================================================================
# Handler Import Tests
# ============================================================================


class TestDocsHandlerImport:
    """Test DocsHandler import and export."""

    def test_handler_importable(self):
        """DocsHandler can be imported from handlers package."""
        from aragora.server.handlers import DocsHandler

        assert DocsHandler is not None

    def test_handler_in_all_exports(self):
        """DocsHandler is in __all__ exports."""
        from aragora.server.handlers import __all__

        assert "DocsHandler" in __all__

    def test_cache_ttl_constant_exported(self):
        """CACHE_TTL_OPENAPI constant is exported."""
        from aragora.server.handlers.docs import CACHE_TTL_OPENAPI

        assert CACHE_TTL_OPENAPI == 3600


# ============================================================================
# Constants Tests
# ============================================================================


class TestDocsConstants:
    """Tests for module constants."""

    def test_cache_ttl_value(self):
        """CACHE_TTL_OPENAPI has correct value (1 hour)."""
        assert CACHE_TTL_OPENAPI == 3600

    def test_routes_list(self, docs_handler):
        """ROUTES list contains expected paths."""
        expected_routes = [
            "/api/openapi",
            "/api/openapi.json",
            "/api/openapi.yaml",
            "/api/postman.json",
            "/api/docs",
            "/api/docs/",
            "/api/redoc",
            "/api/redoc/",
        ]
        assert docs_handler.ROUTES == expected_routes
