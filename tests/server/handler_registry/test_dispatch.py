"""Tests for handler registry dispatch and mixin."""

from __future__ import annotations

import io
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handler_registry import (
    HANDLER_REGISTRY,
    HANDLERS_AVAILABLE,
    HandlerRegistryMixin,
)


# ---------------------------------------------------------------------------
# Fake handler for dispatch tests
# ---------------------------------------------------------------------------


class _FakeResult:
    """Minimal handler result."""

    def __init__(
        self,
        status_code: int = 200,
        content_type: str = "application/json",
        body: bytes = b"{}",
        headers: dict[str, str] | None = None,
    ):
        self.status_code = status_code
        self.content_type = content_type
        self.body = body
        self.headers = headers or {}


class _FakeHandler:
    """A minimal handler for dispatch testing."""

    ROUTES = ["/api/fake"]

    def __init__(self, ctx: dict | None = None) -> None:
        pass

    def can_handle(self, path: str) -> bool:
        return path.startswith("/api/fake")

    def handle(self, path: str, query: dict, request_handler: Any) -> _FakeResult:
        return _FakeResult(body=json.dumps({"ok": True}).encode())

    def handle_post(self, path: str, query: dict, request_handler: Any) -> _FakeResult:
        return _FakeResult(status_code=201, body=json.dumps({"created": True}).encode())

    def handle_delete(self, path: str, query: dict, request_handler: Any) -> _FakeResult:
        return _FakeResult(body=json.dumps({"deleted": True}).encode())


class _ErrorHandler:
    """Handler that raises during handle."""

    ROUTES = ["/api/error"]

    def __init__(self, ctx: dict | None = None) -> None:
        pass

    def can_handle(self, path: str) -> bool:
        return path.startswith("/api/error")

    def handle(self, path: str, query: dict, request_handler: Any) -> Any:
        raise ValueError("something broke")


class _PermissionErrorHandler:
    """Handler that raises a permission error."""

    ROUTES = ["/api/perm"]

    def __init__(self, ctx: dict | None = None) -> None:
        pass

    def can_handle(self, path: str) -> bool:
        return path.startswith("/api/perm")

    def handle(self, path: str, query: dict, request_handler: Any) -> Any:
        raise PermissionError("AuthorizationContext required")


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_mixin_instance(
    handler: Any = None,
    handler_attr: str = "_fake_handler",
    method: str = "GET",
) -> HandlerRegistryMixin:
    """Create a mixin instance with mocked HTTP infrastructure."""

    class _TestMixin(HandlerRegistryMixin):
        _handlers_initialized = True
        storage = None
        elo_system = None
        debate_embeddings = None
        document_store = None
        nomic_state_file = None
        critique_store = None
        persona_manager = None
        position_ledger = None

    instance = _TestMixin()
    instance.command = method
    instance.headers = {}
    instance.wfile = io.BytesIO()
    instance.send_response = MagicMock()
    instance.send_header = MagicMock()
    instance.end_headers = MagicMock()
    instance._add_cors_headers = MagicMock()
    instance._add_security_headers = MagicMock()
    instance._add_trace_headers = MagicMock()
    instance._auth_context = None

    if handler:
        setattr(instance, handler_attr, handler)

    return instance


# ---------------------------------------------------------------------------
# Module-level attribute tests
# ---------------------------------------------------------------------------


class TestModuleAttributes:
    """Test module-level exports."""

    def test_handler_registry_is_list(self) -> None:
        assert isinstance(HANDLER_REGISTRY, list)

    def test_handler_registry_entries_are_tuples(self) -> None:
        for entry in HANDLER_REGISTRY[:5]:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            assert isinstance(entry[0], str)

    def test_handlers_available_is_bool(self) -> None:
        assert isinstance(HANDLERS_AVAILABLE, bool)

    def test_handler_registry_attr_names_prefixed(self) -> None:
        for attr_name, _ in HANDLER_REGISTRY:
            assert attr_name.startswith("_"), f"{attr_name} should start with _"
            assert attr_name.endswith("_handler"), f"{attr_name} should end with _handler"


# ---------------------------------------------------------------------------
# HandlerRegistryMixin tests
# ---------------------------------------------------------------------------


class TestHandlerRegistryMixin:
    """Tests for the HandlerRegistryMixin class."""

    def test_handlers_initialized_default_false(self) -> None:
        assert HandlerRegistryMixin._handlers_initialized is not None

    def test_get_handler_stats_not_initialized(self) -> None:
        instance = _make_mixin_instance()
        instance._handlers_initialized = False
        stats = instance._get_handler_stats()
        assert stats["initialized"] is False
        assert stats["count"] == 0

    def test_get_handler_stats_initialized(self) -> None:
        instance = _make_mixin_instance()
        instance._handlers_initialized = True
        stats = instance._get_handler_stats()
        assert stats["initialized"] is True


class TestTryModularHandler:
    """Tests for _try_modular_handler dispatch."""

    def _setup_dispatch(
        self,
        handler: Any,
        path: str = "/api/fake",
        method: str = "GET",
    ) -> HandlerRegistryMixin:
        """Set up mixin with route index pointing to handler."""
        instance = _make_mixin_instance(handler=handler, method=method)

        # Mock the route index to return our handler
        mock_index = MagicMock()
        mock_index.get_handler = MagicMock(return_value=("_fake_handler", handler))

        return instance, mock_index

    @patch("aragora.server.handler_registry.HANDLERS_AVAILABLE", True)
    @patch("aragora.server.handler_registry.get_route_index")
    @patch("aragora.server.handler_registry.extract_version", return_value=("v1", False))
    @patch("aragora.server.handler_registry.strip_version_prefix", side_effect=lambda p: p)
    @patch("aragora.server.handler_registry.normalize_path_version", side_effect=lambda p, v: p)
    @patch("aragora.server.handler_registry.version_response_headers", return_value={})
    def test_dispatch_get_success(self, mock_vrh, mock_npv, mock_svp, mock_ev, mock_gri) -> None:
        handler = _FakeHandler()
        instance = _make_mixin_instance(handler=handler, method="GET")
        mock_index = MagicMock()
        mock_index.get_handler = MagicMock(return_value=("_fake_handler", handler))
        mock_gri.return_value = mock_index

        with patch(
            "aragora.server.handler_registry.should_apply_default_rate_limit", return_value=False
        ):
            result = instance._try_modular_handler("/api/fake", {})

        assert result is True
        instance.send_response.assert_called_once_with(200)

    @patch("aragora.server.handler_registry.HANDLERS_AVAILABLE", True)
    @patch("aragora.server.handler_registry.get_route_index")
    @patch("aragora.server.handler_registry.extract_version", return_value=("v1", False))
    @patch("aragora.server.handler_registry.strip_version_prefix", side_effect=lambda p: p)
    @patch("aragora.server.handler_registry.normalize_path_version", side_effect=lambda p, v: p)
    @patch("aragora.server.handler_registry.version_response_headers", return_value={})
    def test_dispatch_post_uses_handle_post(
        self, mock_vrh, mock_npv, mock_svp, mock_ev, mock_gri
    ) -> None:
        handler = _FakeHandler()
        instance = _make_mixin_instance(handler=handler, method="POST")
        mock_index = MagicMock()
        mock_index.get_handler = MagicMock(return_value=("_fake_handler", handler))
        mock_gri.return_value = mock_index

        with patch(
            "aragora.server.handler_registry.should_apply_default_rate_limit", return_value=False
        ):
            result = instance._try_modular_handler("/api/fake", {})

        assert result is True
        instance.send_response.assert_called_once_with(201)

    @patch("aragora.server.handler_registry.HANDLERS_AVAILABLE", True)
    @patch("aragora.server.handler_registry.get_route_index")
    @patch("aragora.server.handler_registry.extract_version", return_value=("v1", False))
    @patch("aragora.server.handler_registry.strip_version_prefix", side_effect=lambda p: p)
    @patch("aragora.server.handler_registry.normalize_path_version", side_effect=lambda p, v: p)
    @patch("aragora.server.handler_registry.version_response_headers", return_value={})
    def test_dispatch_delete_uses_handle_delete(
        self, mock_vrh, mock_npv, mock_svp, mock_ev, mock_gri
    ) -> None:
        handler = _FakeHandler()
        instance = _make_mixin_instance(handler=handler, method="DELETE")
        mock_index = MagicMock()
        mock_index.get_handler = MagicMock(return_value=("_fake_handler", handler))
        mock_gri.return_value = mock_index

        with patch(
            "aragora.server.handler_registry.should_apply_default_rate_limit", return_value=False
        ):
            result = instance._try_modular_handler("/api/fake", {})

        assert result is True
        instance.send_response.assert_called_once_with(200)

    @patch("aragora.server.handler_registry.HANDLERS_AVAILABLE", False)
    def test_returns_false_when_handlers_unavailable(self) -> None:
        instance = _make_mixin_instance()
        result = instance._try_modular_handler("/api/fake", {})
        assert result is False

    @patch("aragora.server.handler_registry.HANDLERS_AVAILABLE", True)
    @patch("aragora.server.handler_registry.get_route_index")
    @patch("aragora.server.handler_registry.extract_version", return_value=("v1", False))
    @patch("aragora.server.handler_registry.strip_version_prefix", side_effect=lambda p: p)
    @patch("aragora.server.handler_registry.normalize_path_version", side_effect=lambda p, v: p)
    def test_no_match_returns_false(self, mock_npv, mock_svp, mock_ev, mock_gri) -> None:
        instance = _make_mixin_instance(method="GET")
        mock_index = MagicMock()
        mock_index.get_handler = MagicMock(return_value=None)
        mock_gri.return_value = mock_index

        result = instance._try_modular_handler("/api/unknown", {})
        assert result is False

    @patch("aragora.server.handler_registry.HANDLERS_AVAILABLE", True)
    @patch("aragora.server.handler_registry.get_route_index")
    @patch("aragora.server.handler_registry.extract_version", return_value=("v1", False))
    @patch("aragora.server.handler_registry.strip_version_prefix", side_effect=lambda p: p)
    @patch("aragora.server.handler_registry.normalize_path_version", side_effect=lambda p, v: p)
    @patch("aragora.server.handler_registry.version_response_headers", return_value={})
    def test_handler_error_returns_500(
        self, mock_vrh, mock_npv, mock_svp, mock_ev, mock_gri
    ) -> None:
        handler = _ErrorHandler()
        instance = _make_mixin_instance(handler=handler, handler_attr="_fake_handler", method="GET")
        mock_index = MagicMock()
        mock_index.get_handler = MagicMock(return_value=("_fake_handler", handler))
        mock_gri.return_value = mock_index

        with patch(
            "aragora.server.handler_registry.should_apply_default_rate_limit", return_value=False
        ):
            result = instance._try_modular_handler("/api/error", {})

        assert result is True
        instance.send_response.assert_called_once_with(500)
        # Verify error body
        body = instance.wfile.getvalue()
        data = json.loads(body)
        assert data["code"] == "handler_error"

    @patch("aragora.server.handler_registry.HANDLERS_AVAILABLE", True)
    @patch("aragora.server.handler_registry.get_route_index")
    @patch("aragora.server.handler_registry.extract_version", return_value=("v1", False))
    @patch("aragora.server.handler_registry.strip_version_prefix", side_effect=lambda p: p)
    @patch("aragora.server.handler_registry.normalize_path_version", side_effect=lambda p, v: p)
    @patch("aragora.server.handler_registry.version_response_headers", return_value={})
    def test_permission_error_returns_403(
        self, mock_vrh, mock_npv, mock_svp, mock_ev, mock_gri
    ) -> None:
        handler = _PermissionErrorHandler()
        instance = _make_mixin_instance(handler=handler, handler_attr="_fake_handler", method="GET")
        mock_index = MagicMock()
        mock_index.get_handler = MagicMock(return_value=("_fake_handler", handler))
        mock_gri.return_value = mock_index

        with patch(
            "aragora.server.handler_registry.should_apply_default_rate_limit", return_value=False
        ):
            result = instance._try_modular_handler("/api/perm", {})

        assert result is True
        instance.send_response.assert_called_once_with(403)
        body = instance.wfile.getvalue()
        data = json.loads(body)
        assert data["code"] == "forbidden"

    @patch("aragora.server.handler_registry.HANDLERS_AVAILABLE", True)
    @patch("aragora.server.handler_registry.get_route_index")
    @patch("aragora.server.handler_registry.extract_version", return_value=("v1", False))
    @patch("aragora.server.handler_registry.strip_version_prefix", side_effect=lambda p: p)
    @patch("aragora.server.handler_registry.normalize_path_version", side_effect=lambda p, v: p)
    @patch("aragora.server.handler_registry.version_response_headers", return_value={})
    def test_rate_limit_returns_429(self, mock_vrh, mock_npv, mock_svp, mock_ev, mock_gri) -> None:
        handler = _FakeHandler()
        instance = _make_mixin_instance(handler=handler, method="GET")
        mock_index = MagicMock()
        mock_index.get_handler = MagicMock(return_value=("_fake_handler", handler))
        mock_gri.return_value = mock_index

        mock_rate_result = MagicMock()
        mock_rate_result.allowed = False
        mock_rate_result.retry_after = 30.0
        mock_rate_result.limit = 60
        mock_rate_result.remaining = 0

        with (
            patch(
                "aragora.server.handler_registry.should_apply_default_rate_limit", return_value=True
            ),
            patch(
                "aragora.server.handler_registry.check_default_rate_limit",
                return_value=mock_rate_result,
            ),
        ):
            result = instance._try_modular_handler("/api/fake", {})

        assert result is True
        instance.send_response.assert_called_once_with(429)
        body = instance.wfile.getvalue()
        data = json.loads(body)
        assert data["code"] == "rate_limit_exceeded"

    @patch("aragora.server.handler_registry.HANDLERS_AVAILABLE", True)
    @patch("aragora.server.handler_registry.get_route_index")
    @patch("aragora.server.handler_registry.extract_version", return_value=("v1", False))
    @patch("aragora.server.handler_registry.strip_version_prefix", side_effect=lambda p: p)
    @patch("aragora.server.handler_registry.normalize_path_version", side_effect=lambda p, v: p)
    @patch("aragora.server.handler_registry.version_response_headers", return_value={})
    def test_query_list_values_collapsed(
        self, mock_vrh, mock_npv, mock_svp, mock_ev, mock_gri
    ) -> None:
        """Query params with single-element lists should be collapsed."""
        handler = _FakeHandler()
        instance = _make_mixin_instance(handler=handler, method="GET")
        mock_index = MagicMock()
        mock_index.get_handler = MagicMock(return_value=("_fake_handler", handler))
        mock_gri.return_value = mock_index

        with patch(
            "aragora.server.handler_registry.should_apply_default_rate_limit", return_value=False
        ):
            instance._try_modular_handler("/api/fake", {"key": ["value"]})

        assert instance.send_response.called

    @patch("aragora.server.handler_registry.HANDLERS_AVAILABLE", True)
    @patch("aragora.server.handler_registry.get_route_index")
    @patch("aragora.server.handler_registry.extract_version", return_value=("v1", False))
    @patch("aragora.server.handler_registry.strip_version_prefix", side_effect=lambda p: p)
    @patch("aragora.server.handler_registry.normalize_path_version", side_effect=lambda p, v: p)
    @patch("aragora.server.handler_registry.version_response_headers", return_value={})
    def test_cors_and_security_headers_added(
        self, mock_vrh, mock_npv, mock_svp, mock_ev, mock_gri
    ) -> None:
        handler = _FakeHandler()
        instance = _make_mixin_instance(handler=handler, method="GET")
        mock_index = MagicMock()
        mock_index.get_handler = MagicMock(return_value=("_fake_handler", handler))
        mock_gri.return_value = mock_index

        with patch(
            "aragora.server.handler_registry.should_apply_default_rate_limit", return_value=False
        ):
            instance._try_modular_handler("/api/fake", {})

        instance._add_cors_headers.assert_called_once()
        instance._add_security_headers.assert_called_once()
