"""
Tests for aragora.server.handlers.docs - API Documentation HTTP Handler.

Tests cover:
- DocsHandler: instantiation, ROUTES, can_handle
- handle routing: openapi json, openapi yaml, postman, swagger, redoc, unmatched
- _get_openapi_spec: json success, yaml success, import error, general error
- _get_swagger_ui: returns HTML with correct content
- _get_redoc: returns HTML with correct content
- _get_postman_collection: success, error
- CACHE_TTL_OPENAPI constant
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.docs import CACHE_TTL_OPENAPI, DocsHandler
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helpers
# ===========================================================================


def _parse_body(result: HandlerResult) -> dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


def _make_mock_handler(
    method: str = "GET",
    body: bytes = b"",
) -> MagicMock:
    """Create a mock HTTP handler object."""
    handler = MagicMock()
    handler.command = method
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": "application/json",
        "Host": "localhost:8080",
    }
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = body
    return handler


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def handler():
    """Create a DocsHandler with empty context."""
    return DocsHandler(ctx={})


# ===========================================================================
# Test Instantiation and Basics
# ===========================================================================


class TestDocsHandlerBasics:
    """Basic instantiation and attribute tests."""

    def test_instantiation(self, handler):
        assert handler is not None
        assert isinstance(handler, DocsHandler)

    def test_routes_defined(self, handler):
        assert len(handler.ROUTES) == 8

    def test_can_handle_openapi(self, handler):
        assert handler.can_handle("/api/v1/openapi") is True

    def test_can_handle_openapi_json(self, handler):
        assert handler.can_handle("/api/v1/openapi.json") is True

    def test_can_handle_openapi_yaml(self, handler):
        assert handler.can_handle("/api/v1/openapi.yaml") is True

    def test_can_handle_postman(self, handler):
        assert handler.can_handle("/api/v1/postman.json") is True

    def test_can_handle_docs(self, handler):
        assert handler.can_handle("/api/v1/docs") is True

    def test_can_handle_docs_slash(self, handler):
        assert handler.can_handle("/api/v1/docs/") is True

    def test_can_handle_redoc(self, handler):
        assert handler.can_handle("/api/v1/redoc") is True

    def test_can_handle_redoc_slash(self, handler):
        assert handler.can_handle("/api/v1/redoc/") is True

    def test_cannot_handle_other(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_cannot_handle_partial(self, handler):
        assert handler.can_handle("/api/v1/doc") is False

    def test_cache_ttl_constant(self):
        assert CACHE_TTL_OPENAPI == 3600


# ===========================================================================
# Test handle() Routing
# ===========================================================================


class TestHandleRouting:
    """Tests for the top-level handle() routing method."""

    def test_route_openapi_json_via_openapi(self, handler):
        mock_handler = _make_mock_handler()

        with patch(
            "aragora.server.handlers.docs.handle_openapi_request",
            create=True,
        ):
            with patch.object(handler, "_get_openapi_spec") as mock_spec:
                mock_spec.return_value = HandlerResult(
                    status_code=200,
                    content_type="application/json",
                    body=b'{"openapi":"3.0.0"}',
                )
                result = handler.handle("/api/v1/openapi", {}, mock_handler)
                assert result is not None
                assert result.status_code == 200
                mock_spec.assert_called_once_with("json")

    def test_route_openapi_json(self, handler):
        mock_handler = _make_mock_handler()

        with patch.object(handler, "_get_openapi_spec") as mock_spec:
            mock_spec.return_value = HandlerResult(
                status_code=200,
                content_type="application/json",
                body=b'{"openapi":"3.0.0"}',
            )
            result = handler.handle("/api/v1/openapi.json", {}, mock_handler)
            assert result is not None
            mock_spec.assert_called_once_with("json")

    def test_route_openapi_yaml(self, handler):
        mock_handler = _make_mock_handler()

        with patch.object(handler, "_get_openapi_spec") as mock_spec:
            mock_spec.return_value = HandlerResult(
                status_code=200,
                content_type="application/x-yaml",
                body=b"openapi: '3.0.0'",
            )
            result = handler.handle("/api/v1/openapi.yaml", {}, mock_handler)
            assert result is not None
            mock_spec.assert_called_once_with("yaml")

    def test_route_postman(self, handler):
        mock_handler = _make_mock_handler()

        with patch.object(handler, "_get_postman_collection") as mock_postman:
            mock_postman.return_value = HandlerResult(
                status_code=200,
                content_type="application/json",
                body=b'{"info":{}}',
            )
            result = handler.handle("/api/v1/postman.json", {}, mock_handler)
            assert result is not None
            mock_postman.assert_called_once()

    def test_route_swagger_ui(self, handler):
        mock_handler = _make_mock_handler()

        result = handler.handle("/api/v1/docs", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200
        assert "text/html" in result.content_type

    def test_route_redoc(self, handler):
        mock_handler = _make_mock_handler()

        result = handler.handle("/api/v1/redoc", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200
        assert "text/html" in result.content_type

    def test_route_unmatched_returns_none(self, handler):
        mock_handler = _make_mock_handler()
        result = handler.handle("/api/v1/unknown", {}, mock_handler)
        assert result is None


# ===========================================================================
# Test _get_openapi_spec
# ===========================================================================


class TestGetOpenAPISpec:
    """Tests for the OpenAPI spec endpoint."""

    def test_openapi_json_success(self, handler):
        with patch(
            "aragora.server.openapi.handle_openapi_request",
            return_value=('{"openapi":"3.0.0"}', "application/json"),
        ):
            result = handler._get_openapi_spec.__wrapped__(handler, "json")
            assert result.status_code == 200
            assert result.content_type == "application/json"

    def test_openapi_yaml_success(self, handler):
        with patch(
            "aragora.server.openapi.handle_openapi_request",
            return_value=("openapi: '3.0.0'", "application/yaml"),
        ):
            result = handler._get_openapi_spec.__wrapped__(handler, "yaml")
            assert result.status_code == 200
            assert result.content_type == "application/yaml"

    def test_openapi_import_error(self, handler):
        with patch(
            "aragora.server.openapi.handle_openapi_request",
            side_effect=ImportError("No module"),
        ):
            result = handler._get_openapi_spec.__wrapped__(handler, "json")
            assert result.status_code == 503

    def test_openapi_general_error(self, handler):
        with patch(
            "aragora.server.openapi.handle_openapi_request",
            side_effect=RuntimeError("Spec generation failed"),
        ):
            result = handler._get_openapi_spec.__wrapped__(handler, "json")
            assert result.status_code == 500


# ===========================================================================
# Test _get_swagger_ui
# ===========================================================================


class TestGetSwaggerUI:
    """Tests for the Swagger UI endpoint."""

    def test_returns_html(self, handler):
        result = handler._get_swagger_ui()
        assert result.status_code == 200
        assert "text/html" in result.content_type

    def test_contains_swagger_ui_elements(self, handler):
        result = handler._get_swagger_ui()
        html = result.body.decode("utf-8")
        assert "swagger-ui" in html
        assert "SwaggerUIBundle" in html
        assert "/api/v1/openapi.json" in html

    def test_contains_title(self, handler):
        result = handler._get_swagger_ui()
        html = result.body.decode("utf-8")
        assert "Aragora" in html


# ===========================================================================
# Test _get_redoc
# ===========================================================================


class TestGetRedoc:
    """Tests for the ReDoc endpoint."""

    def test_returns_html(self, handler):
        result = handler._get_redoc()
        assert result.status_code == 200
        assert "text/html" in result.content_type

    def test_contains_redoc_elements(self, handler):
        result = handler._get_redoc()
        html = result.body.decode("utf-8")
        assert "redoc" in html.lower()
        assert "/api/v1/openapi.json" in html

    def test_contains_title(self, handler):
        result = handler._get_redoc()
        html = result.body.decode("utf-8")
        assert "Aragora" in html


# ===========================================================================
# Test _get_postman_collection
# ===========================================================================


class TestGetPostmanCollection:
    """Tests for the Postman collection endpoint."""

    def test_postman_success(self, handler):
        with patch(
            "aragora.server.openapi.handle_postman_request",
            return_value=('{"info":{}}', "application/json"),
        ):
            result = handler._get_postman_collection()
            assert result.status_code == 200
            assert result.content_type == "application/json"
            assert result.headers is not None
            assert "Content-Disposition" in result.headers

    def test_postman_error(self, handler):
        with patch(
            "aragora.server.openapi.handle_postman_request",
            side_effect=RuntimeError("Export failed"),
        ):
            result = handler._get_postman_collection()
            assert result.status_code == 500
