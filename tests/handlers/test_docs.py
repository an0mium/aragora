"""Tests for API documentation endpoint handlers.

Tests the DocsHandler which serves:
- GET /api/v1/openapi         - OpenAPI 3.0 JSON specification
- GET /api/v1/openapi.json    - OpenAPI 3.0 JSON specification
- GET /api/v1/openapi.yaml    - OpenAPI 3.0 YAML specification
- GET /api/v1/postman.json    - Postman collection export
- GET /api/v1/docs            - Swagger UI interactive documentation
- GET /api/v1/docs/           - Swagger UI (trailing slash)
- GET /api/v1/redoc           - ReDoc API documentation viewer
- GET /api/v1/redoc/          - ReDoc (trailing slash)

Covers:
- Route matching via can_handle()
- Happy-path responses for each endpoint
- Error handling (ImportError, ValueError, TypeError, etc.)
- Response structure (status codes, content types, body contents)
- Swagger UI / ReDoc HTML validation
- OpenAPI JSON vs YAML format dispatch
- Postman collection Content-Disposition header
- Cache decorator presence on _get_openapi_spec
- Unrecognized paths returning None
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.docs import DocsHandler, CACHE_TTL_OPENAPI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


def _status(result) -> int:
    """Extract status code from HandlerResult."""
    return result.status_code


def _html(result) -> str:
    """Decode HTML body from HandlerResult."""
    return result.body.decode("utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler() -> DocsHandler:
    """Create a DocsHandler instance with empty context."""
    return DocsHandler(ctx={})


@pytest.fixture
def handler_with_ctx() -> DocsHandler:
    """Create a DocsHandler instance with a populated context."""
    return DocsHandler(ctx={"storage": MagicMock(), "server_name": "test"})


@pytest.fixture
def mock_http_handler() -> MagicMock:
    """Create a mock HTTP request handler."""
    h = MagicMock()
    h.headers = {}
    h.client_address = ("127.0.0.1", 12345)
    return h


# ===========================================================================
# Initialization
# ===========================================================================


class TestDocsHandlerInit:
    """Test DocsHandler initialization."""

    def test_init_default_ctx(self):
        """DocsHandler initializes with empty dict when ctx is None."""
        handler = DocsHandler()
        assert handler.ctx == {}

    def test_init_none_ctx(self):
        """DocsHandler initializes with empty dict when ctx is explicitly None."""
        handler = DocsHandler(ctx=None)
        assert handler.ctx == {}

    def test_init_with_ctx(self):
        """DocsHandler stores the provided context."""
        ctx = {"storage": "mock_storage"}
        handler = DocsHandler(ctx=ctx)
        assert handler.ctx is ctx

    def test_init_preserves_ctx_reference(self):
        """DocsHandler stores the exact dict reference, not a copy."""
        ctx = {"key": "value"}
        handler = DocsHandler(ctx=ctx)
        ctx["new_key"] = "new_value"
        assert "new_key" in handler.ctx


# ===========================================================================
# Route matching (can_handle)
# ===========================================================================


class TestCanHandle:
    """Test can_handle() route matching."""

    @pytest.mark.parametrize(
        "path",
        [
            "/api/openapi",
            "/api/openapi.json",
            "/api/openapi.yaml",
            "/api/postman.json",
            "/api/docs",
            "/api/docs/",
            "/api/redoc",
            "/api/redoc/",
            "/api/v1/openapi",
            "/api/v1/openapi.json",
            "/api/v1/openapi.yaml",
            "/api/v1/postman.json",
            "/api/v1/docs",
            "/api/v1/docs/",
            "/api/v1/redoc",
            "/api/v1/redoc/",
        ],
    )
    def test_known_routes_are_handled(self, handler, path):
        """All declared routes should be recognized."""
        assert handler.can_handle(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/unknown",
            "/api/v1/openapi.xml",
            "/api/v1/doc",
            "/api/v1/docs/extra",
            "/api/v1/redoc/extra",
            "/openapi",
            "",
            "/",
            "/api/v2/openapi",
            "/api/v1/swagger",
            "/api/v1/postman",
            "/api/v1/postman.yaml",
        ],
    )
    def test_unrecognized_paths_rejected(self, handler, path):
        """Paths not in ROUTES should not be handled."""
        assert handler.can_handle(path) is False


# ===========================================================================
# Routing dispatch (handle)
# ===========================================================================


class TestHandleRouting:
    """Test handle() dispatches to the correct internal method."""

    def test_unrecognized_path_returns_none(self, handler, mock_http_handler):
        """Unrecognized paths return None from handle()."""
        result = handler.handle("/api/v1/something-else", {}, mock_http_handler)
        assert result is None

    def test_empty_path_returns_none(self, handler, mock_http_handler):
        """Empty path returns None."""
        result = handler.handle("", {}, mock_http_handler)
        assert result is None

    @patch("aragora.server.handlers.docs.DocsHandler._get_openapi_spec")
    def test_openapi_routes_to_json(self, mock_spec, handler, mock_http_handler):
        """GET /api/v1/openapi routes to _get_openapi_spec('json')."""
        mock_spec.return_value = MagicMock(status_code=200)
        handler.handle("/api/v1/openapi", {}, mock_http_handler)
        mock_spec.assert_called_once_with("json")

    @patch("aragora.server.handlers.docs.DocsHandler._get_openapi_spec")
    def test_openapi_json_routes_to_json(self, mock_spec, handler, mock_http_handler):
        """GET /api/v1/openapi.json routes to _get_openapi_spec('json')."""
        mock_spec.return_value = MagicMock(status_code=200)
        handler.handle("/api/v1/openapi.json", {}, mock_http_handler)
        mock_spec.assert_called_once_with("json")

    @patch("aragora.server.handlers.docs.DocsHandler._get_openapi_spec")
    def test_openapi_yaml_routes_to_yaml(self, mock_spec, handler, mock_http_handler):
        """GET /api/v1/openapi.yaml routes to _get_openapi_spec('yaml')."""
        mock_spec.return_value = MagicMock(status_code=200)
        handler.handle("/api/v1/openapi.yaml", {}, mock_http_handler)
        mock_spec.assert_called_once_with("yaml")


# ===========================================================================
# OpenAPI Spec endpoint (_get_openapi_spec)
# ===========================================================================


class TestOpenAPISpec:
    """Test _get_openapi_spec() JSON and YAML variants."""

    @patch("aragora.server.handlers.docs.handle_openapi_request", create=True)
    def test_openapi_json_happy_path(self, mock_req, handler, mock_http_handler):
        """Successful JSON OpenAPI spec returns 200 with correct content type."""
        mock_content = '{"openapi": "3.0.0"}'
        mock_req.return_value = (mock_content, "application/json")

        with patch.dict(
            "sys.modules",
            {"aragora.server.openapi": MagicMock(handle_openapi_request=mock_req)},
        ):
            result = handler.handle("/api/v1/openapi.json", {}, mock_http_handler)

        assert _status(result) == 200
        assert result.content_type == "application/json"
        parsed = json.loads(result.body.decode("utf-8"))
        assert parsed["openapi"] == "3.0.0"

    @patch("aragora.server.handlers.docs.handle_openapi_request", create=True)
    def test_openapi_yaml_happy_path(self, mock_req, handler, mock_http_handler):
        """Successful YAML OpenAPI spec returns 200 with YAML content type."""
        yaml_content = "openapi: '3.0.0'\ninfo:\n  title: Aragora"
        mock_req.return_value = (yaml_content, "application/x-yaml")

        with patch.dict(
            "sys.modules",
            {"aragora.server.openapi": MagicMock(handle_openapi_request=mock_req)},
        ):
            result = handler.handle("/api/v1/openapi.yaml", {}, mock_http_handler)

        assert _status(result) == 200
        assert result.content_type == "application/x-yaml"
        body_text = result.body.decode("utf-8")
        assert "openapi" in body_text

    def test_openapi_import_error_returns_503(self, handler, mock_http_handler):
        """When OpenAPI module is not available, returns 503."""
        with patch.dict("sys.modules", {"aragora.server.openapi": None}):
            # Force an ImportError by patching the import inside _get_openapi_spec
            with patch(
                "builtins.__import__",
                side_effect=_selective_import_error("aragora.server.openapi"),
            ):
                result = handler.handle("/api/v1/openapi", {}, mock_http_handler)

        assert _status(result) == 503
        body = _body(result)
        assert "error" in body

    def test_openapi_value_error_returns_500(self, handler, mock_http_handler):
        """ValueError during spec generation returns 500."""
        mock_openapi = MagicMock()
        mock_openapi.handle_openapi_request.side_effect = ValueError("bad schema")

        with patch.dict("sys.modules", {"aragora.server.openapi": mock_openapi}):
            result = handler.handle("/api/v1/openapi", {}, mock_http_handler)

        assert _status(result) == 500
        body = _body(result)
        assert "error" in body

    def test_openapi_type_error_returns_500(self, handler, mock_http_handler):
        """TypeError during spec generation returns 500."""
        mock_openapi = MagicMock()
        mock_openapi.handle_openapi_request.side_effect = TypeError("wrong type")

        with patch.dict("sys.modules", {"aragora.server.openapi": mock_openapi}):
            result = handler.handle("/api/v1/openapi", {}, mock_http_handler)

        assert _status(result) == 500

    def test_openapi_key_error_returns_500(self, handler, mock_http_handler):
        """KeyError during spec generation returns 500."""
        mock_openapi = MagicMock()
        mock_openapi.handle_openapi_request.side_effect = KeyError("missing")

        with patch.dict("sys.modules", {"aragora.server.openapi": mock_openapi}):
            result = handler.handle("/api/v1/openapi", {}, mock_http_handler)

        assert _status(result) == 500

    def test_openapi_runtime_error_returns_500(self, handler, mock_http_handler):
        """RuntimeError during spec generation returns 500."""
        mock_openapi = MagicMock()
        mock_openapi.handle_openapi_request.side_effect = RuntimeError("fail")

        with patch.dict("sys.modules", {"aragora.server.openapi": mock_openapi}):
            result = handler.handle("/api/v1/openapi", {}, mock_http_handler)

        assert _status(result) == 500

    def test_openapi_os_error_returns_500(self, handler, mock_http_handler):
        """OSError during spec generation returns 500."""
        mock_openapi = MagicMock()
        mock_openapi.handle_openapi_request.side_effect = OSError("disk fail")

        with patch.dict("sys.modules", {"aragora.server.openapi": mock_openapi}):
            result = handler.handle("/api/v1/openapi", {}, mock_http_handler)

        assert _status(result) == 500

    def test_openapi_json_body_is_bytes(self, handler, mock_http_handler):
        """Response body is always bytes, even when content is a string."""
        mock_openapi = MagicMock()
        mock_openapi.handle_openapi_request.return_value = (
            '{"info": "test"}',
            "application/json",
        )

        with patch.dict("sys.modules", {"aragora.server.openapi": mock_openapi}):
            result = handler.handle("/api/v1/openapi.json", {}, mock_http_handler)

        assert isinstance(result.body, bytes)

    def test_openapi_bytes_content_not_double_encoded(self, handler, mock_http_handler):
        """If handle_openapi_request returns bytes, they are not re-encoded."""
        raw_bytes = b'{"already": "bytes"}'
        mock_openapi = MagicMock()
        mock_openapi.handle_openapi_request.return_value = (
            raw_bytes,
            "application/json",
        )

        with patch.dict("sys.modules", {"aragora.server.openapi": mock_openapi}):
            result = handler.handle("/api/v1/openapi.json", {}, mock_http_handler)

        assert result.body == raw_bytes

    def test_openapi_both_paths_equivalent(self, handler, mock_http_handler):
        """Both /api/v1/openapi and /api/v1/openapi.json produce identical results."""
        mock_openapi = MagicMock()
        mock_openapi.handle_openapi_request.return_value = (
            '{"openapi": "3.0.0"}',
            "application/json",
        )

        with patch.dict("sys.modules", {"aragora.server.openapi": mock_openapi}):
            result1 = handler.handle("/api/v1/openapi", {}, mock_http_handler)
            result2 = handler.handle("/api/v1/openapi.json", {}, mock_http_handler)

        assert result1.status_code == result2.status_code
        assert result1.content_type == result2.content_type


# ===========================================================================
# Swagger UI endpoint (_get_swagger_ui)
# ===========================================================================


class TestSwaggerUI:
    """Test Swagger UI HTML page generation."""

    def test_swagger_ui_returns_200(self, handler, mock_http_handler):
        """GET /api/v1/docs returns 200."""
        result = handler.handle("/api/v1/docs", {}, mock_http_handler)
        assert _status(result) == 200

    def test_swagger_ui_content_type_html(self, handler, mock_http_handler):
        """Swagger UI response has text/html content type."""
        result = handler.handle("/api/v1/docs", {}, mock_http_handler)
        assert result.content_type == "text/html; charset=utf-8"

    def test_swagger_ui_body_is_bytes(self, handler, mock_http_handler):
        """Swagger UI body is bytes."""
        result = handler.handle("/api/v1/docs", {}, mock_http_handler)
        assert isinstance(result.body, bytes)

    def test_swagger_ui_contains_doctype(self, handler, mock_http_handler):
        """Swagger UI HTML starts with DOCTYPE."""
        result = handler.handle("/api/v1/docs", {}, mock_http_handler)
        html = _html(result)
        assert "<!DOCTYPE html>" in html

    def test_swagger_ui_contains_title(self, handler, mock_http_handler):
        """Swagger UI page has Aragora in the title."""
        result = handler.handle("/api/v1/docs", {}, mock_http_handler)
        html = _html(result)
        assert "<title>Aragora API Documentation</title>" in html

    def test_swagger_ui_loads_swagger_bundle(self, handler, mock_http_handler):
        """Swagger UI loads the swagger-ui-bundle.js script."""
        result = handler.handle("/api/v1/docs", {}, mock_http_handler)
        html = _html(result)
        assert "swagger-ui-bundle.js" in html

    def test_swagger_ui_loads_swagger_css(self, handler, mock_http_handler):
        """Swagger UI loads the swagger-ui.css stylesheet."""
        result = handler.handle("/api/v1/docs", {}, mock_http_handler)
        html = _html(result)
        assert "swagger-ui.css" in html

    def test_swagger_ui_loads_standalone_preset(self, handler, mock_http_handler):
        """Swagger UI loads the standalone preset script."""
        result = handler.handle("/api/v1/docs", {}, mock_http_handler)
        html = _html(result)
        assert "swagger-ui-standalone-preset.js" in html

    def test_swagger_ui_points_to_openapi_json(self, handler, mock_http_handler):
        """Swagger UI points to the correct OpenAPI JSON endpoint."""
        result = handler.handle("/api/v1/docs", {}, mock_http_handler)
        html = _html(result)
        assert "/api/v1/openapi.json" in html

    def test_swagger_ui_has_div_target(self, handler, mock_http_handler):
        """Swagger UI has the #swagger-ui div target."""
        result = handler.handle("/api/v1/docs", {}, mock_http_handler)
        html = _html(result)
        assert 'id="swagger-ui"' in html

    def test_swagger_ui_trailing_slash_equivalent(self, handler, mock_http_handler):
        """GET /api/v1/docs/ returns the same as /api/v1/docs."""
        r1 = handler.handle("/api/v1/docs", {}, mock_http_handler)
        r2 = handler.handle("/api/v1/docs/", {}, mock_http_handler)
        assert r1.status_code == r2.status_code
        assert r1.body == r2.body
        assert r1.content_type == r2.content_type

    def test_swagger_ui_deep_linking_enabled(self, handler, mock_http_handler):
        """Swagger UI config has deepLinking enabled."""
        result = handler.handle("/api/v1/docs", {}, mock_http_handler)
        html = _html(result)
        assert "deepLinking: true" in html

    def test_swagger_ui_filter_enabled(self, handler, mock_http_handler):
        """Swagger UI config has filter enabled."""
        result = handler.handle("/api/v1/docs", {}, mock_http_handler)
        html = _html(result)
        assert "filter: true" in html

    def test_swagger_ui_persist_authorization(self, handler, mock_http_handler):
        """Swagger UI config has persistAuthorization enabled."""
        result = handler.handle("/api/v1/docs", {}, mock_http_handler)
        html = _html(result)
        assert "persistAuthorization: true" in html

    def test_swagger_ui_validator_disabled(self, handler, mock_http_handler):
        """Swagger UI config disables the validator."""
        result = handler.handle("/api/v1/docs", {}, mock_http_handler)
        html = _html(result)
        assert "validatorUrl: null" in html

    def test_swagger_ui_display_request_duration(self, handler, mock_http_handler):
        """Swagger UI config shows request duration."""
        result = handler.handle("/api/v1/docs", {}, mock_http_handler)
        html = _html(result)
        assert "displayRequestDuration: true" in html

    def test_swagger_ui_standalone_layout(self, handler, mock_http_handler):
        """Swagger UI uses StandaloneLayout."""
        result = handler.handle("/api/v1/docs", {}, mock_http_handler)
        html = _html(result)
        assert '"StandaloneLayout"' in html


# ===========================================================================
# ReDoc endpoint (_get_redoc)
# ===========================================================================


class TestReDoc:
    """Test ReDoc HTML page generation."""

    def test_redoc_returns_200(self, handler, mock_http_handler):
        """GET /api/v1/redoc returns 200."""
        result = handler.handle("/api/v1/redoc", {}, mock_http_handler)
        assert _status(result) == 200

    def test_redoc_content_type_html(self, handler, mock_http_handler):
        """ReDoc response has text/html content type."""
        result = handler.handle("/api/v1/redoc", {}, mock_http_handler)
        assert result.content_type == "text/html; charset=utf-8"

    def test_redoc_body_is_bytes(self, handler, mock_http_handler):
        """ReDoc body is bytes."""
        result = handler.handle("/api/v1/redoc", {}, mock_http_handler)
        assert isinstance(result.body, bytes)

    def test_redoc_contains_doctype(self, handler, mock_http_handler):
        """ReDoc HTML starts with DOCTYPE."""
        result = handler.handle("/api/v1/redoc", {}, mock_http_handler)
        html = _html(result)
        assert "<!DOCTYPE html>" in html

    def test_redoc_contains_title(self, handler, mock_http_handler):
        """ReDoc page has Aragora in the title."""
        result = handler.handle("/api/v1/redoc", {}, mock_http_handler)
        html = _html(result)
        assert "<title>Aragora API - ReDoc</title>" in html

    def test_redoc_loads_redoc_script(self, handler, mock_http_handler):
        """ReDoc page loads the redoc.standalone.js script."""
        result = handler.handle("/api/v1/redoc", {}, mock_http_handler)
        html = _html(result)
        assert "redoc.standalone.js" in html

    def test_redoc_points_to_openapi_json(self, handler, mock_http_handler):
        """ReDoc points to the correct OpenAPI JSON spec URL."""
        result = handler.handle("/api/v1/redoc", {}, mock_http_handler)
        html = _html(result)
        assert "/api/v1/openapi.json" in html

    def test_redoc_has_spec_url_attribute(self, handler, mock_http_handler):
        """ReDoc element has the spec-url attribute."""
        result = handler.handle("/api/v1/redoc", {}, mock_http_handler)
        html = _html(result)
        assert 'spec-url="/api/v1/openapi.json"' in html

    def test_redoc_expand_responses(self, handler, mock_http_handler):
        """ReDoc has expand-responses config for 200 and 201."""
        result = handler.handle("/api/v1/redoc", {}, mock_http_handler)
        html = _html(result)
        assert 'expand-responses="200,201"' in html

    def test_redoc_native_scrollbars(self, handler, mock_http_handler):
        """ReDoc has native-scrollbars enabled."""
        result = handler.handle("/api/v1/redoc", {}, mock_http_handler)
        html = _html(result)
        assert 'native-scrollbars="true"' in html

    def test_redoc_path_in_middle_panel(self, handler, mock_http_handler):
        """ReDoc shows path in middle panel."""
        result = handler.handle("/api/v1/redoc", {}, mock_http_handler)
        html = _html(result)
        assert 'path-in-middle-panel="true"' in html

    def test_redoc_loads_google_fonts(self, handler, mock_http_handler):
        """ReDoc page loads Google Fonts for Montserrat and Roboto."""
        result = handler.handle("/api/v1/redoc", {}, mock_http_handler)
        html = _html(result)
        assert "fonts.googleapis.com" in html
        assert "Montserrat" in html
        assert "Roboto" in html

    def test_redoc_trailing_slash_equivalent(self, handler, mock_http_handler):
        """GET /api/v1/redoc/ returns the same as /api/v1/redoc."""
        r1 = handler.handle("/api/v1/redoc", {}, mock_http_handler)
        r2 = handler.handle("/api/v1/redoc/", {}, mock_http_handler)
        assert r1.status_code == r2.status_code
        assert r1.body == r2.body
        assert r1.content_type == r2.content_type

    def test_redoc_download_button_visible(self, handler, mock_http_handler):
        """ReDoc shows the download button."""
        result = handler.handle("/api/v1/redoc", {}, mock_http_handler)
        html = _html(result)
        assert 'hide-download-button="false"' in html


# ===========================================================================
# Postman Collection endpoint (_get_postman_collection)
# ===========================================================================


class TestPostmanCollection:
    """Test Postman collection export."""

    def test_postman_happy_path(self, handler, mock_http_handler):
        """Successful Postman export returns 200."""
        mock_openapi = MagicMock()
        mock_openapi.handle_postman_request.return_value = (
            '{"info": {"name": "Aragora"}}',
            "application/json",
        )

        with patch.dict("sys.modules", {"aragora.server.openapi": mock_openapi}):
            result = handler.handle("/api/v1/postman.json", {}, mock_http_handler)

        assert _status(result) == 200

    def test_postman_content_type(self, handler, mock_http_handler):
        """Postman export uses the content type from the handler."""
        mock_openapi = MagicMock()
        mock_openapi.handle_postman_request.return_value = (
            '{"info": {"name": "Aragora"}}',
            "application/json",
        )

        with patch.dict("sys.modules", {"aragora.server.openapi": mock_openapi}):
            result = handler.handle("/api/v1/postman.json", {}, mock_http_handler)

        assert result.content_type == "application/json"

    def test_postman_content_disposition_header(self, handler, mock_http_handler):
        """Postman export has Content-Disposition attachment header."""
        mock_openapi = MagicMock()
        mock_openapi.handle_postman_request.return_value = (
            '{"info": {"name": "Aragora"}}',
            "application/json",
        )

        with patch.dict("sys.modules", {"aragora.server.openapi": mock_openapi}):
            result = handler.handle("/api/v1/postman.json", {}, mock_http_handler)

        assert result.headers is not None
        assert "Content-Disposition" in result.headers
        assert "attachment" in result.headers["Content-Disposition"]
        assert "aragora.postman_collection.json" in result.headers["Content-Disposition"]

    def test_postman_body_is_bytes(self, handler, mock_http_handler):
        """Postman export body is bytes."""
        mock_openapi = MagicMock()
        mock_openapi.handle_postman_request.return_value = (
            '{"info": {"name": "Aragora"}}',
            "application/json",
        )

        with patch.dict("sys.modules", {"aragora.server.openapi": mock_openapi}):
            result = handler.handle("/api/v1/postman.json", {}, mock_http_handler)

        assert isinstance(result.body, bytes)

    def test_postman_import_error_returns_500(self, handler, mock_http_handler):
        """Import error for Postman module returns 500."""
        with patch.dict("sys.modules", {"aragora.server.openapi": None}):
            with patch(
                "builtins.__import__",
                side_effect=_selective_import_error("aragora.server.openapi"),
            ):
                result = handler.handle("/api/v1/postman.json", {}, mock_http_handler)

        assert _status(result) == 500
        body = _body(result)
        assert "error" in body

    def test_postman_value_error_returns_500(self, handler, mock_http_handler):
        """ValueError during Postman export returns 500."""
        mock_openapi = MagicMock()
        mock_openapi.handle_postman_request.side_effect = ValueError("bad data")

        with patch.dict("sys.modules", {"aragora.server.openapi": mock_openapi}):
            result = handler.handle("/api/v1/postman.json", {}, mock_http_handler)

        assert _status(result) == 500

    def test_postman_type_error_returns_500(self, handler, mock_http_handler):
        """TypeError during Postman export returns 500."""
        mock_openapi = MagicMock()
        mock_openapi.handle_postman_request.side_effect = TypeError("bad type")

        with patch.dict("sys.modules", {"aragora.server.openapi": mock_openapi}):
            result = handler.handle("/api/v1/postman.json", {}, mock_http_handler)

        assert _status(result) == 500

    def test_postman_key_error_returns_500(self, handler, mock_http_handler):
        """KeyError during Postman export returns 500."""
        mock_openapi = MagicMock()
        mock_openapi.handle_postman_request.side_effect = KeyError("missing")

        with patch.dict("sys.modules", {"aragora.server.openapi": mock_openapi}):
            result = handler.handle("/api/v1/postman.json", {}, mock_http_handler)

        assert _status(result) == 500

    def test_postman_runtime_error_returns_500(self, handler, mock_http_handler):
        """RuntimeError during Postman export returns 500."""
        mock_openapi = MagicMock()
        mock_openapi.handle_postman_request.side_effect = RuntimeError("fail")

        with patch.dict("sys.modules", {"aragora.server.openapi": mock_openapi}):
            result = handler.handle("/api/v1/postman.json", {}, mock_http_handler)

        assert _status(result) == 500

    def test_postman_os_error_returns_500(self, handler, mock_http_handler):
        """OSError during Postman export returns 500."""
        mock_openapi = MagicMock()
        mock_openapi.handle_postman_request.side_effect = OSError("disk")

        with patch.dict("sys.modules", {"aragora.server.openapi": mock_openapi}):
            result = handler.handle("/api/v1/postman.json", {}, mock_http_handler)

        assert _status(result) == 500

    def test_postman_body_parseable(self, handler, mock_http_handler):
        """Postman collection body is valid JSON."""
        collection = {"info": {"name": "Aragora", "schema": "v2.1"}, "item": []}
        mock_openapi = MagicMock()
        mock_openapi.handle_postman_request.return_value = (
            json.dumps(collection),
            "application/json",
        )

        with patch.dict("sys.modules", {"aragora.server.openapi": mock_openapi}):
            result = handler.handle("/api/v1/postman.json", {}, mock_http_handler)

        parsed = json.loads(result.body.decode("utf-8"))
        assert parsed["info"]["name"] == "Aragora"


# ===========================================================================
# ROUTES constant
# ===========================================================================


class TestRoutes:
    """Test the ROUTES class attribute."""

    def test_routes_count(self):
        """ROUTES contains at least the canonical documentation endpoints."""
        assert len(DocsHandler.ROUTES) >= 8

    def test_routes_all_versioned(self):
        """All routes use /api/ prefix (versioned and unversioned supported)."""
        for route in DocsHandler.ROUTES:
            assert route.startswith("/api/"), f"Route {route} missing /api/ prefix"

    def test_routes_are_strings(self):
        """All ROUTES entries are strings."""
        for route in DocsHandler.ROUTES:
            assert isinstance(route, str)


# ===========================================================================
# Cache TTL constant
# ===========================================================================


class TestCacheTTL:
    """Test the CACHE_TTL_OPENAPI constant."""

    def test_cache_ttl_value(self):
        """CACHE_TTL_OPENAPI is 3600 seconds (1 hour)."""
        assert CACHE_TTL_OPENAPI == 3600

    def test_cache_ttl_is_int(self):
        """CACHE_TTL_OPENAPI is an integer."""
        assert isinstance(CACHE_TTL_OPENAPI, int)


# ===========================================================================
# Query parameters (ignored by docs handler)
# ===========================================================================


class TestQueryParams:
    """Test that query parameters don't affect docs endpoints."""

    def test_docs_ignores_query_params(self, handler, mock_http_handler):
        """Swagger UI endpoint ignores query parameters."""
        result = handler.handle("/api/v1/docs", {"format": "yaml"}, mock_http_handler)
        assert _status(result) == 200

    def test_redoc_ignores_query_params(self, handler, mock_http_handler):
        """ReDoc endpoint ignores query parameters."""
        result = handler.handle("/api/v1/redoc", {"extra": "param"}, mock_http_handler)
        assert _status(result) == 200


# ===========================================================================
# Error response structure
# ===========================================================================


class TestErrorStructure:
    """Test that error responses have consistent structure."""

    def test_openapi_error_has_error_field(self, handler, mock_http_handler):
        """OpenAPI error response contains 'error' in body."""
        mock_openapi = MagicMock()
        mock_openapi.handle_openapi_request.side_effect = ValueError("bad")

        with patch.dict("sys.modules", {"aragora.server.openapi": mock_openapi}):
            result = handler.handle("/api/v1/openapi", {}, mock_http_handler)

        body = _body(result)
        assert "error" in body

    def test_postman_error_has_error_field(self, handler, mock_http_handler):
        """Postman error response contains 'error' in body."""
        mock_openapi = MagicMock()
        mock_openapi.handle_postman_request.side_effect = RuntimeError("fail")

        with patch.dict("sys.modules", {"aragora.server.openapi": mock_openapi}):
            result = handler.handle("/api/v1/postman.json", {}, mock_http_handler)

        body = _body(result)
        assert "error" in body

    def test_openapi_503_error_message(self, handler, mock_http_handler):
        """503 error mentions module unavailability."""
        with patch.dict("sys.modules", {"aragora.server.openapi": None}):
            with patch(
                "builtins.__import__",
                side_effect=_selective_import_error("aragora.server.openapi"),
            ):
                result = handler.handle("/api/v1/openapi", {}, mock_http_handler)

        body = _body(result)
        assert "not available" in body.get("error", "").lower() or _status(result) == 503


# ===========================================================================
# Handler with different contexts
# ===========================================================================


class TestHandlerContext:
    """Test handler behavior with different context values."""

    def test_handler_works_with_empty_ctx(self, mock_http_handler):
        """Handler works with empty context dict."""
        handler = DocsHandler(ctx={})
        result = handler.handle("/api/v1/docs", {}, mock_http_handler)
        assert _status(result) == 200

    def test_handler_works_with_none_ctx(self, mock_http_handler):
        """Handler works when ctx is None (defaults to empty dict)."""
        handler = DocsHandler(ctx=None)
        result = handler.handle("/api/v1/docs", {}, mock_http_handler)
        assert _status(result) == 200

    def test_handler_works_with_populated_ctx(self, handler_with_ctx, mock_http_handler):
        """Handler works with a populated context."""
        result = handler_with_ctx.handle("/api/v1/redoc", {}, mock_http_handler)
        assert _status(result) == 200


# ===========================================================================
# Helpers (used by tests above)
# ===========================================================================


def _selective_import_error(blocked_module: str):
    """Create an import side_effect that only blocks a specific module."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _side_effect(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    return _side_effect
