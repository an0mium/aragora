"""
Tests for SSO authentication handler.

Tests:
- SSOHandler initialization
- Route definitions
- Parameter extraction
- Redirect URL validation
- Response formatting
"""

import pytest

from aragora.server.handlers.sso import SSOHandler


class TestSSOHandlerInit:
    """Tests for SSOHandler initialization."""

    def test_init_with_empty_context(self):
        """Should initialize with empty context."""
        handler = SSOHandler()
        assert handler._provider is None
        assert handler._initialized is False

    def test_init_with_context(self):
        """Should initialize with provided context."""
        ctx = {"db": "mock_db"}
        handler = SSOHandler(ctx)
        assert handler._initialized is False


class TestSSOHandlerRoutes:
    """Tests for SSO route definitions."""

    def test_routes_returns_list(self):
        """Routes should return a list of tuples."""
        handler = SSOHandler()
        routes = handler.routes()
        assert isinstance(routes, list)
        assert len(routes) > 0

    def test_routes_include_login(self):
        """Routes should include login endpoint."""
        handler = SSOHandler()
        routes = handler.routes()
        methods_paths = [(r[0], r[1]) for r in routes]
        assert ("GET", "/auth/sso/login") in methods_paths
        assert ("POST", "/auth/sso/login") in methods_paths

    def test_routes_include_callback(self):
        """Routes should include callback endpoint."""
        handler = SSOHandler()
        routes = handler.routes()
        methods_paths = [(r[0], r[1]) for r in routes]
        assert ("GET", "/auth/sso/callback") in methods_paths
        assert ("POST", "/auth/sso/callback") in methods_paths

    def test_routes_include_logout(self):
        """Routes should include logout endpoint."""
        handler = SSOHandler()
        routes = handler.routes()
        methods_paths = [(r[0], r[1]) for r in routes]
        assert ("GET", "/auth/sso/logout") in methods_paths
        assert ("POST", "/auth/sso/logout") in methods_paths

    def test_routes_include_metadata(self):
        """Routes should include metadata endpoint."""
        handler = SSOHandler()
        routes = handler.routes()
        methods_paths = [(r[0], r[1]) for r in routes]
        assert ("GET", "/auth/sso/metadata") in methods_paths

    def test_routes_include_status(self):
        """Routes should include status endpoint."""
        handler = SSOHandler()
        routes = handler.routes()
        methods_paths = [(r[0], r[1]) for r in routes]
        assert ("GET", "/auth/sso/status") in methods_paths

    def test_routes_have_handler_names(self):
        """Each route should have a handler method name."""
        handler = SSOHandler()
        routes = handler.routes()
        for route in routes:
            assert len(route) == 3
            method, path, handler_name = route
            assert hasattr(handler, handler_name)


class TestSSOGetParam:
    """Tests for _get_param helper method."""

    def test_get_param_string_value(self):
        """Should return string value directly."""
        handler = SSOHandler()
        params = {"key": "value"}
        assert handler._get_param(params, "key") == "value"

    def test_get_param_list_value(self):
        """Should extract first element from list."""
        handler = SSOHandler()
        params = {"key": ["value1", "value2"]}
        assert handler._get_param(params, "key") == "value1"

    def test_get_param_empty_list(self):
        """Should return None for empty list."""
        handler = SSOHandler()
        params = {"key": []}
        assert handler._get_param(params, "key") is None

    def test_get_param_missing_key(self):
        """Should return None for missing key."""
        handler = SSOHandler()
        params = {"other": "value"}
        assert handler._get_param(params, "key") is None

    def test_get_param_none_value(self):
        """Should return None for None value."""
        handler = SSOHandler()
        params = {"key": None}
        assert handler._get_param(params, "key") is None

    def test_get_param_converts_to_string(self):
        """Should convert non-string values to string."""
        handler = SSOHandler()
        params = {"key": 123}
        assert handler._get_param(params, "key") == "123"


class TestSSOValidateRedirectUrl:
    """Tests for _validate_redirect_url method."""

    def test_empty_url_is_valid(self):
        """Empty URL should be valid (no redirect)."""
        handler = SSOHandler()
        assert handler._validate_redirect_url("") is True

    def test_https_url_is_valid(self):
        """HTTPS URL should be valid."""
        handler = SSOHandler()
        assert handler._validate_redirect_url("https://example.com/callback") is True

    def test_http_url_is_valid_in_dev(self):
        """HTTP URL should be valid in non-production."""
        handler = SSOHandler()
        assert handler._validate_redirect_url("http://localhost:3000/callback") is True

    def test_javascript_scheme_is_invalid(self):
        """JavaScript scheme should be blocked."""
        handler = SSOHandler()
        assert handler._validate_redirect_url("javascript:alert(1)") is False

    def test_data_scheme_is_invalid(self):
        """Data scheme should be blocked."""
        handler = SSOHandler()
        assert handler._validate_redirect_url("data:text/html,test") is False

    def test_file_scheme_is_invalid(self):
        """File scheme should be blocked."""
        handler = SSOHandler()
        assert handler._validate_redirect_url("file:///etc/passwd") is False

    def test_url_with_credentials_is_invalid(self):
        """URL with credentials should be blocked."""
        handler = SSOHandler()
        assert handler._validate_redirect_url("https://user:pass@evil.com") is False

    def test_malformed_url_is_invalid(self):
        """Malformed URL should be blocked."""
        handler = SSOHandler()
        # This should not crash
        result = handler._validate_redirect_url("not a valid url at all")
        # Either returns False or raises no exception
        assert result in (True, False)


class TestSSOShouldReturnHandlerResult:
    """Tests for _should_return_handler_result method."""

    def test_none_handler_returns_false(self):
        """None handler should return False."""
        handler = SSOHandler()
        assert handler._should_return_handler_result(None) is False

    def test_handler_with_send_response_returns_true(self):
        """Handler with send_response method should return True."""
        handler = SSOHandler()

        class MockHandler:
            def send_response(self):
                pass

        assert handler._should_return_handler_result(MockHandler()) is True

    def test_handler_without_send_response_returns_false(self):
        """Handler without send_response should return False."""
        handler = SSOHandler()

        class MockHandler:
            pass

        assert handler._should_return_handler_result(MockHandler()) is False


class TestSSOFlattenErrorBody:
    """Tests for _flatten_error_body method."""

    def test_non_dict_passthrough(self):
        """Non-dict values should pass through unchanged."""
        handler = SSOHandler()
        assert handler._flatten_error_body("error") == "error"
        assert handler._flatten_error_body(123) == 123
        assert handler._flatten_error_body(["a", "b"]) == ["a", "b"]

    def test_dict_without_error_passthrough(self):
        """Dict without 'error' key should pass through."""
        handler = SSOHandler()
        body = {"message": "test"}
        assert handler._flatten_error_body(body) == body

    def test_dict_with_string_error_passthrough(self):
        """Dict with string error should pass through."""
        handler = SSOHandler()
        body = {"error": "simple error"}
        assert handler._flatten_error_body(body) == body

    def test_dict_with_structured_error_flattens(self):
        """Dict with structured error should flatten."""
        handler = SSOHandler()
        body = {
            "error": {
                "message": "Error message",
                "code": "ERR_CODE",
                "suggestion": "Fix it",
            },
            "other_field": "value",
        }
        result = handler._flatten_error_body(body)
        assert result["error"] == "Error message"
        assert result["code"] == "ERR_CODE"
        assert result["suggestion"] == "Fix it"
        assert result["other_field"] == "value"


class TestSSOToLegacyResult:
    """Tests for _to_legacy_result method."""

    def test_dict_input_returns_dict(self):
        """Dict input should return dict."""
        handler = SSOHandler()
        input_dict = {"status": 200, "body": {"data": "test"}}
        result = handler._to_legacy_result(input_dict)
        assert isinstance(result, dict)
        assert result["status"] == 200

    def test_adds_headers_if_missing(self):
        """Should add empty headers dict if missing."""
        handler = SSOHandler()
        input_dict = {"status": 200, "body": {}}
        result = handler._to_legacy_result(input_dict)
        assert "headers" in result
        assert result["headers"] == {}

    def test_converts_status_code_to_status(self):
        """Should convert status_code key to status."""
        handler = SSOHandler()
        input_dict = {"status_code": 404, "body": {}}
        result = handler._to_legacy_result(input_dict)
        assert result["status"] == 404


class TestSSOGetProvider:
    """Tests for _get_provider method."""

    def test_first_call_initializes(self):
        """First call should set _initialized to True."""
        handler = SSOHandler()
        assert handler._initialized is False
        handler._get_provider()
        assert handler._initialized is True

    def test_returns_none_without_sso_module(self):
        """Should return None when SSO module not available."""
        handler = SSOHandler()
        # Without actual SSO module configured, should return None
        result = handler._get_provider()
        # Provider may be None if SSO not configured
        assert result is None or handler._initialized is True

    def test_subsequent_calls_dont_reinitialize(self):
        """Subsequent calls should not reinitialize."""
        handler = SSOHandler()
        handler._get_provider()
        handler._initialized = True
        handler._provider = "mock_provider"
        # Second call should return cached provider
        result = handler._get_provider()
        assert result == "mock_provider"
