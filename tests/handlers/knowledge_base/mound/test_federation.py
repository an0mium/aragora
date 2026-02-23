"""Tests for FederationOperationsMixin (aragora/server/handlers/knowledge_base/mound/federation.py).

Covers all seven handler methods on the mixin:
- _handle_register_region       (POST /api/knowledge/mound/federation/regions)
- _handle_unregister_region     (DELETE /api/knowledge/mound/federation/regions/:id)
- _handle_sync_to_region        (POST /api/knowledge/mound/federation/sync/push)
- _handle_pull_from_region      (POST /api/knowledge/mound/federation/sync/pull)
- _handle_sync_all_regions      (POST /api/knowledge/mound/federation/sync/all)
- _handle_get_federation_status (GET /api/knowledge/mound/federation/status)
- _handle_list_regions          (GET /api/knowledge/mound/federation/regions)

Each method is tested for:
- Success with valid inputs
- Mound not available (503)
- Missing required parameters (400)
- Invalid parameter values (400)
- Internal errors from mound operations (500)
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.knowledge.mound.ops.federation import FederationMode, SyncScope
from aragora.server.handlers.knowledge_base.mound.federation import (
    FederationOperationsMixin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return -1
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _make_http_handler(body: dict | None = None) -> MagicMock:
    """Create a mock HTTP handler with headers and rfile."""
    handler = MagicMock()
    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.headers = {"Content-Length": str(len(body_bytes))}
        handler.rfile.read.return_value = body_bytes
    else:
        handler.headers = {"Content-Length": "0"}
        handler.rfile.read.return_value = b""
    return handler


# ---------------------------------------------------------------------------
# Mock domain objects
# ---------------------------------------------------------------------------


@dataclass
class MockFederatedRegion:
    region_id: str = "region-us-east"
    endpoint_url: str = "https://federation.example.com/api"
    api_key: str = "secret-key"
    mode: FederationMode = FederationMode.BIDIRECTIONAL
    sync_scope: SyncScope = SyncScope.SUMMARY
    enabled: bool = True


@dataclass
class MockSyncResult:
    region_id: str = "region-us-east"
    direction: str = "push"
    nodes_synced: int = 42
    nodes_skipped: int = 3
    nodes_failed: int = 0
    duration_ms: float = 150.5
    success: bool = True
    error: str | None = None


# ---------------------------------------------------------------------------
# Concrete test class combining the mixin with stubs
# ---------------------------------------------------------------------------


class FederationTestHandler(FederationOperationsMixin):
    """Concrete handler for testing the federation mixin."""

    def __init__(self, mound=None):
        self._mound = mound

    def _get_mound(self):
        return self._mound

    def require_auth_or_error(self, handler):
        return MagicMock(), None

    def require_admin_or_error(self, handler):
        return MagicMock(), None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound with federation methods."""
    mound = MagicMock()
    mound.register_federated_region = AsyncMock(return_value=MockFederatedRegion())
    mound.unregister_federated_region = AsyncMock(return_value=True)
    mound.sync_to_region = AsyncMock(return_value=MockSyncResult())
    mound.pull_from_region = AsyncMock(return_value=MockSyncResult(direction="pull"))
    mound.sync_all_regions = AsyncMock(
        return_value=[
            MockSyncResult(region_id="region-us-east", direction="push"),
            MockSyncResult(
                region_id="region-eu-west", direction="push", success=False, error="timeout"
            ),
        ]
    )
    mound.get_federation_status = AsyncMock(
        return_value={
            "region-us-east": {"enabled": True, "healthy": True, "last_sync": "2025-01-15"},
            "region-eu-west": {"enabled": True, "healthy": False, "last_sync": "2025-01-14"},
            "region-ap-south": {"enabled": False, "healthy": False},
        }
    )
    return mound


@pytest.fixture
def handler(mock_mound):
    """Create a FederationTestHandler with a mocked mound."""
    return FederationTestHandler(mound=mock_mound)


@pytest.fixture
def handler_no_mound():
    """Create a FederationTestHandler with no mound (None)."""
    return FederationTestHandler(mound=None)


@contextmanager
def _mock_metrics():
    """Patch federation metrics context managers and functions."""

    # track_federation_sync is a context manager that yields a mutable dict
    @contextmanager
    def mock_sync_tracker(region_id, direction):
        ctx = {"status": "success", "nodes_synced": 0}
        yield ctx

    with (
        patch(
            "aragora.server.handlers.knowledge_base.mound.federation.track_federation_sync",
            side_effect=mock_sync_tracker,
        ),
        patch(
            "aragora.server.handlers.knowledge_base.mound.federation.track_federation_regions",
        ),
        patch(
            "aragora.server.handlers.knowledge_base.mound.federation.validate_webhook_url",
            return_value=(True, ""),
        ),
        patch(
            "aragora.server.handlers.knowledge_base.mound.federation._run_async",
            side_effect=lambda coro: __import__("asyncio").run(coro)
            if hasattr(coro, "__await__")
            else coro,
        ),
    ):
        yield


@contextmanager
def _mock_metrics_with_ssrf_fail(error_msg="Blocked private IP"):
    """Patch metrics but make validate_webhook_url fail."""

    @contextmanager
    def mock_sync_tracker(region_id, direction):
        ctx = {"status": "success", "nodes_synced": 0}
        yield ctx

    with (
        patch(
            "aragora.server.handlers.knowledge_base.mound.federation.track_federation_sync",
            side_effect=mock_sync_tracker,
        ),
        patch(
            "aragora.server.handlers.knowledge_base.mound.federation.track_federation_regions",
        ),
        patch(
            "aragora.server.handlers.knowledge_base.mound.federation.validate_webhook_url",
            return_value=(False, error_msg),
        ),
        patch(
            "aragora.server.handlers.knowledge_base.mound.federation._run_async",
            side_effect=lambda coro: __import__("asyncio").run(coro)
            if hasattr(coro, "__await__")
            else coro,
        ),
    ):
        yield


# ============================================================================
# Tests: _handle_register_region
# ============================================================================


class TestRegisterRegion:
    """Test _handle_register_region (POST /api/knowledge/mound/federation/regions)."""

    def test_register_region_success(self, handler, mock_mound):
        """Successfully registering a region returns 201 with region data."""
        body = {
            "region_id": "region-us-east",
            "endpoint_url": "https://federation.example.com/api",
            "api_key": "secret-key",
            "mode": "bidirectional",
            "sync_scope": "summary",
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_register_region(mock_http)
        assert _status(result) == 201
        data = _body(result)
        assert data["success"] is True
        assert data["region"]["region_id"] == "region-us-east"
        assert data["region"]["mode"] == "bidirectional"
        assert data["region"]["sync_scope"] == "summary"
        assert data["region"]["enabled"] is True

    def test_register_region_mound_called_with_correct_args(self, handler, mock_mound):
        """Mound.register_federated_region is called with correct args."""
        body = {
            "region_id": "region-eu-west",
            "endpoint_url": "https://eu.example.com/api",
            "api_key": "eu-key",
            "mode": "push",
            "sync_scope": "full",
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            handler._handle_register_region(mock_http)
        mock_mound.register_federated_region.assert_called_once()
        call_kwargs = mock_mound.register_federated_region.call_args.kwargs
        assert call_kwargs["region_id"] == "region-eu-west"
        assert call_kwargs["endpoint_url"] == "https://eu.example.com/api"
        assert call_kwargs["api_key"] == "eu-key"
        assert call_kwargs["mode"] == FederationMode.PUSH
        assert call_kwargs["sync_scope"] == SyncScope.FULL

    def test_register_region_default_mode_and_scope(self, handler, mock_mound):
        """Default mode is bidirectional and default scope is summary."""
        body = {
            "region_id": "region-1",
            "endpoint_url": "https://example.com/api",
            "api_key": "key",
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            handler._handle_register_region(mock_http)
        call_kwargs = mock_mound.register_federated_region.call_args.kwargs
        assert call_kwargs["mode"] == FederationMode.BIDIRECTIONAL
        assert call_kwargs["sync_scope"] == SyncScope.SUMMARY

    def test_register_region_no_body_returns_400(self, handler):
        """Empty body returns 400."""
        mock_http = _make_http_handler(None)
        with _mock_metrics():
            result = handler._handle_register_region(mock_http)
        assert _status(result) == 400
        assert "body" in _body(result)["error"].lower()

    def test_register_region_missing_region_id_returns_400(self, handler):
        """Missing region_id returns 400."""
        body = {
            "endpoint_url": "https://example.com/api",
            "api_key": "key",
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_register_region(mock_http)
        assert _status(result) == 400
        assert "region_id" in _body(result)["error"].lower()

    def test_register_region_missing_endpoint_url_returns_400(self, handler):
        """Missing endpoint_url returns 400."""
        body = {
            "region_id": "region-1",
            "api_key": "key",
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_register_region(mock_http)
        assert _status(result) == 400
        assert "endpoint_url" in _body(result)["error"].lower()

    def test_register_region_missing_api_key_returns_400(self, handler):
        """Missing api_key returns 400."""
        body = {
            "region_id": "region-1",
            "endpoint_url": "https://example.com/api",
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_register_region(mock_http)
        assert _status(result) == 400
        assert "api_key" in _body(result)["error"].lower()

    def test_register_region_ssrf_protection_blocks_private_ip(self, handler):
        """SSRF protection blocks private IP endpoint URLs."""
        body = {
            "region_id": "region-1",
            "endpoint_url": "http://192.168.1.1/api",
            "api_key": "key",
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics_with_ssrf_fail("Blocked private IP"):
            result = handler._handle_register_region(mock_http)
        assert _status(result) == 400
        assert "invalid endpoint url" in _body(result)["error"].lower()

    def test_register_region_invalid_mode_returns_400(self, handler):
        """Invalid federation mode returns 400 with valid modes listed."""
        body = {
            "region_id": "region-1",
            "endpoint_url": "https://example.com/api",
            "api_key": "key",
            "mode": "invalid_mode",
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_register_region(mock_http)
        assert _status(result) == 400
        error_msg = _body(result)["error"].lower()
        assert "invalid mode" in error_msg

    def test_register_region_invalid_sync_scope_returns_400(self, handler):
        """Invalid sync scope returns 400 with valid scopes listed."""
        body = {
            "region_id": "region-1",
            "endpoint_url": "https://example.com/api",
            "api_key": "key",
            "sync_scope": "invalid_scope",
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_register_region(mock_http)
        assert _status(result) == 400
        error_msg = _body(result)["error"].lower()
        assert "invalid sync_scope" in error_msg

    def test_register_region_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        body = {
            "region_id": "region-1",
            "endpoint_url": "https://example.com/api",
            "api_key": "key",
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler_no_mound._handle_register_region(mock_http)
        assert _status(result) == 503
        assert "not available" in _body(result)["error"].lower()

    def test_register_region_connection_error_returns_500(self, handler, mock_mound):
        """ConnectionError from mound returns 500."""
        mock_mound.register_federated_region = AsyncMock(side_effect=ConnectionError("refused"))
        body = {
            "region_id": "region-1",
            "endpoint_url": "https://example.com/api",
            "api_key": "key",
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_register_region(mock_http)
        assert _status(result) == 500

    def test_register_region_timeout_error_returns_500(self, handler, mock_mound):
        """TimeoutError from mound returns 500."""
        mock_mound.register_federated_region = AsyncMock(side_effect=TimeoutError("timed out"))
        body = {
            "region_id": "region-1",
            "endpoint_url": "https://example.com/api",
            "api_key": "key",
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_register_region(mock_http)
        assert _status(result) == 500

    def test_register_region_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.register_federated_region = AsyncMock(side_effect=OSError("disk full"))
        body = {
            "region_id": "region-1",
            "endpoint_url": "https://example.com/api",
            "api_key": "key",
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_register_region(mock_http)
        assert _status(result) == 500

    def test_register_region_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.register_federated_region = AsyncMock(side_effect=ValueError("bad data"))
        body = {
            "region_id": "region-1",
            "endpoint_url": "https://example.com/api",
            "api_key": "key",
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_register_region(mock_http)
        assert _status(result) == 500

    def test_register_region_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.register_federated_region = AsyncMock(side_effect=TypeError("wrong type"))
        body = {
            "region_id": "region-1",
            "endpoint_url": "https://example.com/api",
            "api_key": "key",
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_register_region(mock_http)
        assert _status(result) == 500

    def test_register_region_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.register_federated_region = AsyncMock(side_effect=KeyError("missing"))
        body = {
            "region_id": "region-1",
            "endpoint_url": "https://example.com/api",
            "api_key": "key",
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_register_region(mock_http)
        assert _status(result) == 500

    def test_register_region_invalid_json_returns_400(self, handler):
        """Invalid JSON body returns 400."""
        mock_http = MagicMock()
        mock_http.headers = {"Content-Length": "5"}
        mock_http.rfile.read.return_value = b"notjs"
        with _mock_metrics():
            result = handler._handle_register_region(mock_http)
        assert _status(result) == 400
        assert "invalid" in _body(result)["error"].lower()

    def test_register_region_all_valid_modes(self, handler, mock_mound):
        """All valid FederationMode values are accepted."""
        for mode in FederationMode:
            mock_mound.register_federated_region = AsyncMock(
                return_value=MockFederatedRegion(mode=mode)
            )
            body = {
                "region_id": "region-1",
                "endpoint_url": "https://example.com/api",
                "api_key": "key",
                "mode": mode.value,
            }
            mock_http = _make_http_handler(body)
            with _mock_metrics():
                result = handler._handle_register_region(mock_http)
            assert _status(result) == 201, f"Mode {mode.value} should be accepted"

    def test_register_region_all_valid_sync_scopes(self, handler, mock_mound):
        """All valid SyncScope values are accepted."""
        for scope in SyncScope:
            mock_mound.register_federated_region = AsyncMock(
                return_value=MockFederatedRegion(sync_scope=scope)
            )
            body = {
                "region_id": "region-1",
                "endpoint_url": "https://example.com/api",
                "api_key": "key",
                "sync_scope": scope.value,
            }
            mock_http = _make_http_handler(body)
            with _mock_metrics():
                result = handler._handle_register_region(mock_http)
            assert _status(result) == 201, f"Scope {scope.value} should be accepted"

    def test_register_region_response_endpoint_url(self, handler, mock_mound):
        """Response includes endpoint_url from the registered region."""
        mock_mound.register_federated_region = AsyncMock(
            return_value=MockFederatedRegion(endpoint_url="https://custom.example.com/api")
        )
        body = {
            "region_id": "region-1",
            "endpoint_url": "https://custom.example.com/api",
            "api_key": "key",
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_register_region(mock_http)
        data = _body(result)
        assert data["region"]["endpoint_url"] == "https://custom.example.com/api"


# ============================================================================
# Tests: _handle_unregister_region
# ============================================================================


class TestUnregisterRegion:
    """Test _handle_unregister_region (DELETE /api/knowledge/mound/federation/regions/:id)."""

    def test_unregister_region_success(self, handler, mock_mound):
        """Successfully unregistering a region returns success."""
        mock_http = _make_http_handler()
        with _mock_metrics():
            result = handler._handle_unregister_region("region-us-east", mock_http)
        assert _status(result) == 200
        data = _body(result)
        assert data["success"] is True
        assert data["region_id"] == "region-us-east"

    def test_unregister_region_mound_called(self, handler, mock_mound):
        """Mound.unregister_federated_region is called with the region_id."""
        mock_http = _make_http_handler()
        with _mock_metrics():
            handler._handle_unregister_region("region-eu-west", mock_http)
        mock_mound.unregister_federated_region.assert_called_once_with("region-eu-west")

    def test_unregister_region_not_found_returns_404(self, handler, mock_mound):
        """When mound returns False (not found), returns 404."""
        mock_mound.unregister_federated_region = AsyncMock(return_value=False)
        mock_http = _make_http_handler()
        with _mock_metrics():
            result = handler._handle_unregister_region("nonexistent", mock_http)
        assert _status(result) == 404
        assert "not found" in _body(result)["error"].lower()

    def test_unregister_region_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        mock_http = _make_http_handler()
        with _mock_metrics():
            result = handler_no_mound._handle_unregister_region("region-1", mock_http)
        assert _status(result) == 503

    def test_unregister_region_connection_error_returns_500(self, handler, mock_mound):
        """ConnectionError returns 500."""
        mock_mound.unregister_federated_region = AsyncMock(side_effect=ConnectionError("refused"))
        mock_http = _make_http_handler()
        with _mock_metrics():
            result = handler._handle_unregister_region("region-1", mock_http)
        assert _status(result) == 500

    def test_unregister_region_timeout_error_returns_500(self, handler, mock_mound):
        """TimeoutError returns 500."""
        mock_mound.unregister_federated_region = AsyncMock(side_effect=TimeoutError("timed out"))
        mock_http = _make_http_handler()
        with _mock_metrics():
            result = handler._handle_unregister_region("region-1", mock_http)
        assert _status(result) == 500

    def test_unregister_region_os_error_returns_500(self, handler, mock_mound):
        """OSError returns 500."""
        mock_mound.unregister_federated_region = AsyncMock(side_effect=OSError("disk error"))
        mock_http = _make_http_handler()
        with _mock_metrics():
            result = handler._handle_unregister_region("region-1", mock_http)
        assert _status(result) == 500

    def test_unregister_region_value_error_returns_500(self, handler, mock_mound):
        """ValueError returns 500."""
        mock_mound.unregister_federated_region = AsyncMock(side_effect=ValueError("bad"))
        mock_http = _make_http_handler()
        with _mock_metrics():
            result = handler._handle_unregister_region("region-1", mock_http)
        assert _status(result) == 500

    def test_unregister_region_type_error_returns_500(self, handler, mock_mound):
        """TypeError returns 500."""
        mock_mound.unregister_federated_region = AsyncMock(side_effect=TypeError("bad type"))
        mock_http = _make_http_handler()
        with _mock_metrics():
            result = handler._handle_unregister_region("region-1", mock_http)
        assert _status(result) == 500

    def test_unregister_region_key_error_returns_500(self, handler, mock_mound):
        """KeyError returns 500."""
        mock_mound.unregister_federated_region = AsyncMock(side_effect=KeyError("missing"))
        mock_http = _make_http_handler()
        with _mock_metrics():
            result = handler._handle_unregister_region("region-1", mock_http)
        assert _status(result) == 500

    def test_unregister_region_404_includes_region_id(self, handler, mock_mound):
        """404 error message includes the region_id that was not found."""
        mock_mound.unregister_federated_region = AsyncMock(return_value=False)
        mock_http = _make_http_handler()
        with _mock_metrics():
            result = handler._handle_unregister_region("specific-region", mock_http)
        assert "specific-region" in _body(result)["error"]


# ============================================================================
# Tests: _handle_sync_to_region
# ============================================================================


class TestSyncToRegion:
    """Test _handle_sync_to_region (POST /api/knowledge/mound/federation/sync/push)."""

    def test_sync_to_region_success(self, handler, mock_mound):
        """Successfully syncing to a region returns sync result."""
        body = {"region_id": "region-us-east"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_sync_to_region(mock_http)
        assert _status(result) == 200
        data = _body(result)
        assert data["success"] is True
        assert data["region_id"] == "region-us-east"
        assert data["direction"] == "push"
        assert data["nodes_synced"] == 42
        assert data["nodes_skipped"] == 3
        assert data["nodes_failed"] == 0
        assert data["duration_ms"] == 150.5
        assert data["error"] is None

    def test_sync_to_region_with_all_params(self, handler, mock_mound):
        """All optional parameters are forwarded to mound."""
        body = {
            "region_id": "region-1",
            "workspace_id": "ws-123",
            "since": "2025-01-15T10:00:00Z",
            "visibility_levels": ["public", "team"],
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            handler._handle_sync_to_region(mock_http)
        mock_mound.sync_to_region.assert_called_once()
        call_kwargs = mock_mound.sync_to_region.call_args.kwargs
        assert call_kwargs["region_id"] == "region-1"
        assert call_kwargs["workspace_id"] == "ws-123"
        assert call_kwargs["since"] is not None
        assert call_kwargs["visibility_levels"] == ["public", "team"]

    def test_sync_to_region_no_body_returns_400(self, handler):
        """Empty body returns 400."""
        mock_http = _make_http_handler(None)
        with _mock_metrics():
            result = handler._handle_sync_to_region(mock_http)
        assert _status(result) == 400

    def test_sync_to_region_missing_region_id_returns_400(self, handler):
        """Missing region_id returns 400."""
        body = {"workspace_id": "ws-1"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_sync_to_region(mock_http)
        assert _status(result) == 400
        assert "region_id" in _body(result)["error"].lower()

    def test_sync_to_region_invalid_since_returns_400(self, handler):
        """Invalid since timestamp returns 400."""
        body = {"region_id": "region-1", "since": "not-a-date"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_sync_to_region(mock_http)
        assert _status(result) == 400
        assert "since" in _body(result)["error"].lower() or "iso" in _body(result)["error"].lower()

    def test_sync_to_region_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        body = {"region_id": "region-1"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler_no_mound._handle_sync_to_region(mock_http)
        assert _status(result) == 503

    def test_sync_to_region_connection_error_returns_500(self, handler, mock_mound):
        """ConnectionError returns 500."""
        mock_mound.sync_to_region = AsyncMock(side_effect=ConnectionError("refused"))
        body = {"region_id": "region-1"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_sync_to_region(mock_http)
        assert _status(result) == 500

    def test_sync_to_region_timeout_error_returns_500(self, handler, mock_mound):
        """TimeoutError returns 500."""
        mock_mound.sync_to_region = AsyncMock(side_effect=TimeoutError("timed out"))
        body = {"region_id": "region-1"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_sync_to_region(mock_http)
        assert _status(result) == 500

    def test_sync_to_region_os_error_returns_500(self, handler, mock_mound):
        """OSError returns 500."""
        mock_mound.sync_to_region = AsyncMock(side_effect=OSError("disk full"))
        body = {"region_id": "region-1"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_sync_to_region(mock_http)
        assert _status(result) == 500

    def test_sync_to_region_value_error_returns_500(self, handler, mock_mound):
        """ValueError returns 500."""
        mock_mound.sync_to_region = AsyncMock(side_effect=ValueError("bad"))
        body = {"region_id": "region-1"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_sync_to_region(mock_http)
        assert _status(result) == 500

    def test_sync_to_region_key_error_returns_500(self, handler, mock_mound):
        """KeyError returns 500."""
        mock_mound.sync_to_region = AsyncMock(side_effect=KeyError("missing"))
        body = {"region_id": "region-1"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_sync_to_region(mock_http)
        assert _status(result) == 500

    def test_sync_to_region_type_error_returns_500(self, handler, mock_mound):
        """TypeError returns 500."""
        mock_mound.sync_to_region = AsyncMock(side_effect=TypeError("wrong type"))
        body = {"region_id": "region-1"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_sync_to_region(mock_http)
        assert _status(result) == 500

    def test_sync_to_region_since_z_suffix_parsed(self, handler, mock_mound):
        """ISO date with Z suffix is correctly parsed."""
        body = {"region_id": "region-1", "since": "2025-01-15T10:00:00Z"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            handler._handle_sync_to_region(mock_http)
        call_kwargs = mock_mound.sync_to_region.call_args.kwargs
        assert isinstance(call_kwargs["since"], datetime)

    def test_sync_to_region_since_offset_parsed(self, handler, mock_mound):
        """ISO date with offset is correctly parsed."""
        body = {"region_id": "region-1", "since": "2025-01-15T10:00:00+05:30"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            handler._handle_sync_to_region(mock_http)
        call_kwargs = mock_mound.sync_to_region.call_args.kwargs
        assert isinstance(call_kwargs["since"], datetime)

    def test_sync_to_region_no_since_passes_none(self, handler, mock_mound):
        """No since parameter passes None to mound."""
        body = {"region_id": "region-1"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            handler._handle_sync_to_region(mock_http)
        call_kwargs = mock_mound.sync_to_region.call_args.kwargs
        assert call_kwargs["since"] is None

    def test_sync_to_region_invalid_json_returns_400(self, handler):
        """Invalid JSON body returns 400."""
        mock_http = MagicMock()
        mock_http.headers = {"Content-Length": "5"}
        mock_http.rfile.read.return_value = b"{bad}"
        with _mock_metrics():
            result = handler._handle_sync_to_region(mock_http)
        assert _status(result) == 400

    def test_sync_to_region_failed_result(self, handler, mock_mound):
        """Sync result with success=False is returned as-is."""
        mock_mound.sync_to_region = AsyncMock(
            return_value=MockSyncResult(
                success=False,
                error="Region unavailable",
                nodes_synced=0,
                nodes_failed=5,
            )
        )
        body = {"region_id": "region-1"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_sync_to_region(mock_http)
        assert _status(result) == 200
        data = _body(result)
        assert data["success"] is False
        assert data["error"] == "Region unavailable"
        assert data["nodes_failed"] == 5


# ============================================================================
# Tests: _handle_pull_from_region
# ============================================================================


class TestPullFromRegion:
    """Test _handle_pull_from_region (POST /api/knowledge/mound/federation/sync/pull)."""

    def test_pull_from_region_success(self, handler, mock_mound):
        """Successfully pulling from a region returns sync result."""
        body = {"region_id": "region-us-east"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_pull_from_region(mock_http)
        assert _status(result) == 200
        data = _body(result)
        assert data["success"] is True
        assert data["region_id"] == "region-us-east"
        assert data["direction"] == "pull"

    def test_pull_from_region_with_all_params(self, handler, mock_mound):
        """All optional parameters are forwarded to mound."""
        body = {
            "region_id": "region-1",
            "workspace_id": "ws-456",
            "since": "2025-02-01T00:00:00Z",
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            handler._handle_pull_from_region(mock_http)
        mock_mound.pull_from_region.assert_called_once()
        call_kwargs = mock_mound.pull_from_region.call_args.kwargs
        assert call_kwargs["region_id"] == "region-1"
        assert call_kwargs["workspace_id"] == "ws-456"
        assert isinstance(call_kwargs["since"], datetime)

    def test_pull_from_region_no_body_returns_400(self, handler):
        """Empty body returns 400."""
        mock_http = _make_http_handler(None)
        with _mock_metrics():
            result = handler._handle_pull_from_region(mock_http)
        assert _status(result) == 400

    def test_pull_from_region_missing_region_id_returns_400(self, handler):
        """Missing region_id returns 400."""
        body = {"workspace_id": "ws-1"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_pull_from_region(mock_http)
        assert _status(result) == 400
        assert "region_id" in _body(result)["error"].lower()

    def test_pull_from_region_invalid_since_returns_400(self, handler):
        """Invalid since timestamp returns 400."""
        body = {"region_id": "region-1", "since": "not-a-date"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_pull_from_region(mock_http)
        assert _status(result) == 400

    def test_pull_from_region_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        body = {"region_id": "region-1"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler_no_mound._handle_pull_from_region(mock_http)
        assert _status(result) == 503

    def test_pull_from_region_connection_error_returns_500(self, handler, mock_mound):
        """ConnectionError returns 500."""
        mock_mound.pull_from_region = AsyncMock(side_effect=ConnectionError("refused"))
        body = {"region_id": "region-1"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_pull_from_region(mock_http)
        assert _status(result) == 500

    def test_pull_from_region_timeout_error_returns_500(self, handler, mock_mound):
        """TimeoutError returns 500."""
        mock_mound.pull_from_region = AsyncMock(side_effect=TimeoutError("timed out"))
        body = {"region_id": "region-1"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_pull_from_region(mock_http)
        assert _status(result) == 500

    def test_pull_from_region_os_error_returns_500(self, handler, mock_mound):
        """OSError returns 500."""
        mock_mound.pull_from_region = AsyncMock(side_effect=OSError("disk full"))
        body = {"region_id": "region-1"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_pull_from_region(mock_http)
        assert _status(result) == 500

    def test_pull_from_region_value_error_returns_500(self, handler, mock_mound):
        """ValueError returns 500."""
        mock_mound.pull_from_region = AsyncMock(side_effect=ValueError("bad"))
        body = {"region_id": "region-1"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_pull_from_region(mock_http)
        assert _status(result) == 500

    def test_pull_from_region_key_error_returns_500(self, handler, mock_mound):
        """KeyError returns 500."""
        mock_mound.pull_from_region = AsyncMock(side_effect=KeyError("missing"))
        body = {"region_id": "region-1"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_pull_from_region(mock_http)
        assert _status(result) == 500

    def test_pull_from_region_type_error_returns_500(self, handler, mock_mound):
        """TypeError returns 500."""
        mock_mound.pull_from_region = AsyncMock(side_effect=TypeError("wrong type"))
        body = {"region_id": "region-1"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_pull_from_region(mock_http)
        assert _status(result) == 500

    def test_pull_from_region_no_since_passes_none(self, handler, mock_mound):
        """No since parameter passes None to mound."""
        body = {"region_id": "region-1"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            handler._handle_pull_from_region(mock_http)
        call_kwargs = mock_mound.pull_from_region.call_args.kwargs
        assert call_kwargs["since"] is None

    def test_pull_from_region_invalid_json_returns_400(self, handler):
        """Invalid JSON body returns 400."""
        mock_http = MagicMock()
        mock_http.headers = {"Content-Length": "3"}
        mock_http.rfile.read.return_value = b"abc"
        with _mock_metrics():
            result = handler._handle_pull_from_region(mock_http)
        assert _status(result) == 400

    def test_pull_from_region_response_fields(self, handler, mock_mound):
        """Response includes all expected fields from SyncResult."""
        custom_result = MockSyncResult(
            region_id="region-eu",
            direction="pull",
            nodes_synced=100,
            nodes_failed=2,
            duration_ms=500.0,
            success=True,
            error=None,
        )
        mock_mound.pull_from_region = AsyncMock(return_value=custom_result)
        body = {"region_id": "region-eu"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_pull_from_region(mock_http)
        data = _body(result)
        assert data["nodes_synced"] == 100
        assert data["nodes_failed"] == 2
        assert data["duration_ms"] == 500.0


# ============================================================================
# Tests: _handle_sync_all_regions
# ============================================================================


class TestSyncAllRegions:
    """Test _handle_sync_all_regions (POST /api/knowledge/mound/federation/sync/all)."""

    def test_sync_all_regions_success(self, handler, mock_mound):
        """Successfully syncing all regions returns aggregated results."""
        mock_http = _make_http_handler({})
        with _mock_metrics():
            result = handler._handle_sync_all_regions(mock_http)
        assert _status(result) == 200
        data = _body(result)
        assert data["total_regions"] == 2
        assert data["successful"] == 1
        assert data["failed"] == 1
        assert len(data["results"]) == 2

    def test_sync_all_regions_result_fields(self, handler, mock_mound):
        """Each result entry includes expected fields."""
        mock_http = _make_http_handler({})
        with _mock_metrics():
            result = handler._handle_sync_all_regions(mock_http)
        data = _body(result)
        first_result = data["results"][0]
        assert "region_id" in first_result
        assert "direction" in first_result
        assert "success" in first_result
        assert "nodes_synced" in first_result
        assert "nodes_failed" in first_result
        assert "error" in first_result

    def test_sync_all_regions_with_workspace(self, handler, mock_mound):
        """workspace_id is forwarded to mound."""
        body = {"workspace_id": "ws-prod"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            handler._handle_sync_all_regions(mock_http)
        call_kwargs = mock_mound.sync_all_regions.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws-prod"

    def test_sync_all_regions_with_since(self, handler, mock_mound):
        """since parameter is parsed and forwarded."""
        body = {"since": "2025-01-15T10:00:00Z"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            handler._handle_sync_all_regions(mock_http)
        call_kwargs = mock_mound.sync_all_regions.call_args.kwargs
        assert isinstance(call_kwargs["since"], datetime)

    def test_sync_all_regions_empty_body_ok(self, handler, mock_mound):
        """Empty body (no params) is accepted."""
        mock_http = _make_http_handler(None)
        # Content-Length 0 means empty body which is allowed
        mock_http.headers = {"Content-Length": "0"}
        with _mock_metrics():
            result = handler._handle_sync_all_regions(mock_http)
        assert _status(result) == 200

    def test_sync_all_regions_invalid_since_returns_400(self, handler):
        """Invalid since timestamp returns 400."""
        body = {"since": "not-a-date"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_sync_all_regions(mock_http)
        assert _status(result) == 400

    def test_sync_all_regions_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        mock_http = _make_http_handler({})
        with _mock_metrics():
            result = handler_no_mound._handle_sync_all_regions(mock_http)
        assert _status(result) == 503

    def test_sync_all_regions_connection_error_returns_500(self, handler, mock_mound):
        """ConnectionError returns 500."""
        mock_mound.sync_all_regions = AsyncMock(side_effect=ConnectionError("refused"))
        mock_http = _make_http_handler({})
        with _mock_metrics():
            result = handler._handle_sync_all_regions(mock_http)
        assert _status(result) == 500

    def test_sync_all_regions_timeout_error_returns_500(self, handler, mock_mound):
        """TimeoutError returns 500."""
        mock_mound.sync_all_regions = AsyncMock(side_effect=TimeoutError("timed out"))
        mock_http = _make_http_handler({})
        with _mock_metrics():
            result = handler._handle_sync_all_regions(mock_http)
        assert _status(result) == 500

    def test_sync_all_regions_os_error_returns_500(self, handler, mock_mound):
        """OSError returns 500."""
        mock_mound.sync_all_regions = AsyncMock(side_effect=OSError("disk"))
        mock_http = _make_http_handler({})
        with _mock_metrics():
            result = handler._handle_sync_all_regions(mock_http)
        assert _status(result) == 500

    def test_sync_all_regions_value_error_returns_500(self, handler, mock_mound):
        """ValueError returns 500."""
        mock_mound.sync_all_regions = AsyncMock(side_effect=ValueError("bad"))
        mock_http = _make_http_handler({})
        with _mock_metrics():
            result = handler._handle_sync_all_regions(mock_http)
        assert _status(result) == 500

    def test_sync_all_regions_key_error_returns_500(self, handler, mock_mound):
        """KeyError returns 500."""
        mock_mound.sync_all_regions = AsyncMock(side_effect=KeyError("missing"))
        mock_http = _make_http_handler({})
        with _mock_metrics():
            result = handler._handle_sync_all_regions(mock_http)
        assert _status(result) == 500

    def test_sync_all_regions_type_error_returns_500(self, handler, mock_mound):
        """TypeError returns 500."""
        mock_mound.sync_all_regions = AsyncMock(side_effect=TypeError("wrong"))
        mock_http = _make_http_handler({})
        with _mock_metrics():
            result = handler._handle_sync_all_regions(mock_http)
        assert _status(result) == 500

    def test_sync_all_regions_empty_results(self, handler, mock_mound):
        """No regions returns empty results."""
        mock_mound.sync_all_regions = AsyncMock(return_value=[])
        mock_http = _make_http_handler({})
        with _mock_metrics():
            result = handler._handle_sync_all_regions(mock_http)
        data = _body(result)
        assert data["total_regions"] == 0
        assert data["successful"] == 0
        assert data["failed"] == 0
        assert data["results"] == []

    def test_sync_all_regions_all_successful(self, handler, mock_mound):
        """All regions succeeding returns correct counts."""
        mock_mound.sync_all_regions = AsyncMock(
            return_value=[
                MockSyncResult(region_id="r1", success=True),
                MockSyncResult(region_id="r2", success=True),
                MockSyncResult(region_id="r3", success=True),
            ]
        )
        mock_http = _make_http_handler({})
        with _mock_metrics():
            result = handler._handle_sync_all_regions(mock_http)
        data = _body(result)
        assert data["total_regions"] == 3
        assert data["successful"] == 3
        assert data["failed"] == 0

    def test_sync_all_regions_all_failed(self, handler, mock_mound):
        """All regions failing returns correct counts."""
        mock_mound.sync_all_regions = AsyncMock(
            return_value=[
                MockSyncResult(region_id="r1", success=False, error="err"),
                MockSyncResult(region_id="r2", success=False, error="err"),
            ]
        )
        mock_http = _make_http_handler({})
        with _mock_metrics():
            result = handler._handle_sync_all_regions(mock_http)
        data = _body(result)
        assert data["total_regions"] == 2
        assert data["successful"] == 0
        assert data["failed"] == 2

    def test_sync_all_regions_invalid_json_returns_400(self, handler):
        """Invalid JSON body returns 400."""
        mock_http = MagicMock()
        mock_http.headers = {"Content-Length": "5"}
        mock_http.rfile.read.return_value = b"notjs"
        with _mock_metrics():
            result = handler._handle_sync_all_regions(mock_http)
        assert _status(result) == 400


# ============================================================================
# Tests: _handle_get_federation_status
# ============================================================================


class TestGetFederationStatus:
    """Test _handle_get_federation_status (GET /api/knowledge/mound/federation/status)."""

    def test_get_status_success(self, handler, mock_mound):
        """Successfully getting federation status returns region data."""
        with _mock_metrics():
            result = handler._handle_get_federation_status({})
        assert _status(result) == 200
        data = _body(result)
        assert data["total_regions"] == 3
        assert data["enabled_regions"] == 2
        assert "regions" in data
        assert "region-us-east" in data["regions"]

    def test_get_status_region_details(self, handler, mock_mound):
        """Status includes per-region detail data."""
        with _mock_metrics():
            result = handler._handle_get_federation_status({})
        data = _body(result)
        us_east = data["regions"]["region-us-east"]
        assert us_east["enabled"] is True
        assert us_east["healthy"] is True

    def test_get_status_enabled_count(self, handler, mock_mound):
        """Enabled count reflects the number of enabled regions."""
        with _mock_metrics():
            result = handler._handle_get_federation_status({})
        data = _body(result)
        # us-east enabled, eu-west enabled, ap-south disabled = 2 enabled
        assert data["enabled_regions"] == 2

    def test_get_status_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        with _mock_metrics():
            result = handler_no_mound._handle_get_federation_status({})
        assert _status(result) == 503

    def test_get_status_connection_error_returns_500(self, handler, mock_mound):
        """ConnectionError returns 500."""
        mock_mound.get_federation_status = AsyncMock(side_effect=ConnectionError("refused"))
        with _mock_metrics():
            result = handler._handle_get_federation_status({})
        assert _status(result) == 500

    def test_get_status_timeout_error_returns_500(self, handler, mock_mound):
        """TimeoutError returns 500."""
        mock_mound.get_federation_status = AsyncMock(side_effect=TimeoutError("timed out"))
        with _mock_metrics():
            result = handler._handle_get_federation_status({})
        assert _status(result) == 500

    def test_get_status_os_error_returns_500(self, handler, mock_mound):
        """OSError returns 500."""
        mock_mound.get_federation_status = AsyncMock(side_effect=OSError("disk"))
        with _mock_metrics():
            result = handler._handle_get_federation_status({})
        assert _status(result) == 500

    def test_get_status_value_error_returns_500(self, handler, mock_mound):
        """ValueError returns 500."""
        mock_mound.get_federation_status = AsyncMock(side_effect=ValueError("bad"))
        with _mock_metrics():
            result = handler._handle_get_federation_status({})
        assert _status(result) == 500

    def test_get_status_key_error_returns_500(self, handler, mock_mound):
        """KeyError returns 500."""
        mock_mound.get_federation_status = AsyncMock(side_effect=KeyError("missing"))
        with _mock_metrics():
            result = handler._handle_get_federation_status({})
        assert _status(result) == 500

    def test_get_status_type_error_returns_500(self, handler, mock_mound):
        """TypeError returns 500."""
        mock_mound.get_federation_status = AsyncMock(side_effect=TypeError("wrong type"))
        with _mock_metrics():
            result = handler._handle_get_federation_status({})
        assert _status(result) == 500

    def test_get_status_empty_regions(self, handler, mock_mound):
        """Empty region set returns zero counts."""
        mock_mound.get_federation_status = AsyncMock(return_value={})
        with _mock_metrics():
            result = handler._handle_get_federation_status({})
        data = _body(result)
        assert data["total_regions"] == 0
        assert data["enabled_regions"] == 0
        assert data["regions"] == {}

    def test_get_status_all_enabled_and_healthy(self, handler, mock_mound):
        """All regions enabled and healthy."""
        mock_mound.get_federation_status = AsyncMock(
            return_value={
                "r1": {"enabled": True, "healthy": True},
                "r2": {"enabled": True, "healthy": True},
            }
        )
        with _mock_metrics():
            result = handler._handle_get_federation_status({})
        data = _body(result)
        assert data["total_regions"] == 2
        assert data["enabled_regions"] == 2

    def test_get_status_all_disabled(self, handler, mock_mound):
        """All regions disabled returns 0 enabled."""
        mock_mound.get_federation_status = AsyncMock(
            return_value={
                "r1": {"enabled": False, "healthy": False},
                "r2": {"enabled": False, "healthy": False},
            }
        )
        with _mock_metrics():
            result = handler._handle_get_federation_status({})
        data = _body(result)
        assert data["enabled_regions"] == 0

    def test_get_status_tracks_metrics(self, handler, mock_mound):
        """Federation region metrics tracking function is called."""
        with (
            patch(
                "aragora.server.handlers.knowledge_base.mound.federation.track_federation_regions",
            ) as mock_track,
            patch(
                "aragora.server.handlers.knowledge_base.mound.federation.track_federation_sync",
            ),
            patch(
                "aragora.server.handlers.knowledge_base.mound.federation._run_async",
                side_effect=lambda coro: __import__("asyncio").run(coro)
                if hasattr(coro, "__await__")
                else coro,
            ),
        ):
            handler._handle_get_federation_status({})
        mock_track.assert_called_once_with(
            enabled=2,
            disabled=1,
            healthy=1,
            unhealthy=2,
        )


# ============================================================================
# Tests: _handle_list_regions
# ============================================================================


class TestListRegions:
    """Test _handle_list_regions (GET /api/knowledge/mound/federation/regions)."""

    def test_list_regions_success(self, handler, mock_mound):
        """Successfully listing regions returns region list."""
        with _mock_metrics():
            result = handler._handle_list_regions({})
        assert _status(result) == 200
        data = _body(result)
        assert data["count"] == 3
        assert len(data["regions"]) == 3

    def test_list_regions_includes_region_id(self, handler, mock_mound):
        """Each region entry includes region_id from the status dict key."""
        with _mock_metrics():
            result = handler._handle_list_regions({})
        data = _body(result)
        region_ids = [r["region_id"] for r in data["regions"]]
        assert "region-us-east" in region_ids
        assert "region-eu-west" in region_ids
        assert "region-ap-south" in region_ids

    def test_list_regions_includes_region_data(self, handler, mock_mound):
        """Each region entry includes the status data fields."""
        with _mock_metrics():
            result = handler._handle_list_regions({})
        data = _body(result)
        # Find the us-east region
        us_east = next(r for r in data["regions"] if r["region_id"] == "region-us-east")
        assert us_east["enabled"] is True
        assert us_east["healthy"] is True

    def test_list_regions_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        with _mock_metrics():
            result = handler_no_mound._handle_list_regions({})
        assert _status(result) == 503

    def test_list_regions_connection_error_returns_500(self, handler, mock_mound):
        """ConnectionError returns 500."""
        mock_mound.get_federation_status = AsyncMock(side_effect=ConnectionError("refused"))
        with _mock_metrics():
            result = handler._handle_list_regions({})
        assert _status(result) == 500

    def test_list_regions_timeout_error_returns_500(self, handler, mock_mound):
        """TimeoutError returns 500."""
        mock_mound.get_federation_status = AsyncMock(side_effect=TimeoutError("timed out"))
        with _mock_metrics():
            result = handler._handle_list_regions({})
        assert _status(result) == 500

    def test_list_regions_os_error_returns_500(self, handler, mock_mound):
        """OSError returns 500."""
        mock_mound.get_federation_status = AsyncMock(side_effect=OSError("disk"))
        with _mock_metrics():
            result = handler._handle_list_regions({})
        assert _status(result) == 500

    def test_list_regions_value_error_returns_500(self, handler, mock_mound):
        """ValueError returns 500."""
        mock_mound.get_federation_status = AsyncMock(side_effect=ValueError("bad"))
        with _mock_metrics():
            result = handler._handle_list_regions({})
        assert _status(result) == 500

    def test_list_regions_key_error_returns_500(self, handler, mock_mound):
        """KeyError returns 500."""
        mock_mound.get_federation_status = AsyncMock(side_effect=KeyError("missing"))
        with _mock_metrics():
            result = handler._handle_list_regions({})
        assert _status(result) == 500

    def test_list_regions_type_error_returns_500(self, handler, mock_mound):
        """TypeError returns 500."""
        mock_mound.get_federation_status = AsyncMock(side_effect=TypeError("wrong"))
        with _mock_metrics():
            result = handler._handle_list_regions({})
        assert _status(result) == 500

    def test_list_regions_empty(self, handler, mock_mound):
        """Empty region set returns count=0 and empty list."""
        mock_mound.get_federation_status = AsyncMock(return_value={})
        with _mock_metrics():
            result = handler._handle_list_regions({})
        data = _body(result)
        assert data["count"] == 0
        assert data["regions"] == []

    def test_list_regions_single_region(self, handler, mock_mound):
        """Single region is returned correctly."""
        mock_mound.get_federation_status = AsyncMock(
            return_value={
                "only-region": {"enabled": True, "healthy": True, "mode": "push"},
            }
        )
        with _mock_metrics():
            result = handler._handle_list_regions({})
        data = _body(result)
        assert data["count"] == 1
        assert data["regions"][0]["region_id"] == "only-region"
        assert data["regions"][0]["mode"] == "push"


# ============================================================================
# Tests: edge cases and cross-cutting concerns
# ============================================================================


class TestFederationEdgeCases:
    """Test edge cases across federation operations."""

    def test_register_region_admin_auth_required(self, mock_mound):
        """Register region calls require_admin_or_error."""
        handler = FederationTestHandler(mound=mock_mound)
        handler.require_admin_or_error = MagicMock(
            return_value=(None, MagicMock(status_code=403, body=b'{"error":"forbidden"}'))
        )
        body = {
            "region_id": "r1",
            "endpoint_url": "https://example.com/api",
            "api_key": "key",
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_register_region(mock_http)
        assert _status(result) == 403

    def test_unregister_region_admin_auth_required(self, mock_mound):
        """Unregister region calls require_admin_or_error."""
        handler = FederationTestHandler(mound=mock_mound)
        handler.require_admin_or_error = MagicMock(
            return_value=(None, MagicMock(status_code=403, body=b'{"error":"forbidden"}'))
        )
        mock_http = _make_http_handler()
        with _mock_metrics():
            result = handler._handle_unregister_region("region-1", mock_http)
        assert _status(result) == 403

    def test_sync_to_region_auth_required(self, mock_mound):
        """Sync to region calls require_auth_or_error."""
        handler = FederationTestHandler(mound=mock_mound)
        handler.require_auth_or_error = MagicMock(
            return_value=(None, MagicMock(status_code=401, body=b'{"error":"unauthorized"}'))
        )
        body = {"region_id": "r1"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_sync_to_region(mock_http)
        assert _status(result) == 401

    def test_pull_from_region_auth_required(self, mock_mound):
        """Pull from region calls require_auth_or_error."""
        handler = FederationTestHandler(mound=mock_mound)
        handler.require_auth_or_error = MagicMock(
            return_value=(None, MagicMock(status_code=401, body=b'{"error":"unauthorized"}'))
        )
        body = {"region_id": "r1"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_pull_from_region(mock_http)
        assert _status(result) == 401

    def test_sync_all_regions_auth_required(self, mock_mound):
        """Sync all regions calls require_auth_or_error."""
        handler = FederationTestHandler(mound=mock_mound)
        handler.require_auth_or_error = MagicMock(
            return_value=(None, MagicMock(status_code=401, body=b'{"error":"unauthorized"}'))
        )
        mock_http = _make_http_handler({})
        with _mock_metrics():
            result = handler._handle_sync_all_regions(mock_http)
        assert _status(result) == 401

    def test_register_region_push_mode(self, handler, mock_mound):
        """Push mode is correctly passed through."""
        mock_mound.register_federated_region = AsyncMock(
            return_value=MockFederatedRegion(mode=FederationMode.PUSH)
        )
        body = {
            "region_id": "r1",
            "endpoint_url": "https://example.com/api",
            "api_key": "key",
            "mode": "push",
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_register_region(mock_http)
        data = _body(result)
        assert data["region"]["mode"] == "push"

    def test_register_region_pull_mode(self, handler, mock_mound):
        """Pull mode is correctly passed through."""
        mock_mound.register_federated_region = AsyncMock(
            return_value=MockFederatedRegion(mode=FederationMode.PULL)
        )
        body = {
            "region_id": "r1",
            "endpoint_url": "https://example.com/api",
            "api_key": "key",
            "mode": "pull",
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_register_region(mock_http)
        data = _body(result)
        assert data["region"]["mode"] == "pull"

    def test_register_region_none_mode(self, handler, mock_mound):
        """None mode is correctly passed through (defaults to bidirectional via FederationMode)."""
        mock_mound.register_federated_region = AsyncMock(
            return_value=MockFederatedRegion(mode=FederationMode.NONE)
        )
        body = {
            "region_id": "r1",
            "endpoint_url": "https://example.com/api",
            "api_key": "key",
            "mode": "none",
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_register_region(mock_http)
        data = _body(result)
        assert data["region"]["mode"] == "none"

    def test_register_region_metadata_scope(self, handler, mock_mound):
        """Metadata sync scope is accepted."""
        mock_mound.register_federated_region = AsyncMock(
            return_value=MockFederatedRegion(sync_scope=SyncScope.METADATA)
        )
        body = {
            "region_id": "r1",
            "endpoint_url": "https://example.com/api",
            "api_key": "key",
            "sync_scope": "metadata",
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_register_region(mock_http)
        data = _body(result)
        assert data["region"]["sync_scope"] == "metadata"

    def test_register_region_full_scope(self, handler, mock_mound):
        """Full sync scope is accepted."""
        mock_mound.register_federated_region = AsyncMock(
            return_value=MockFederatedRegion(sync_scope=SyncScope.FULL)
        )
        body = {
            "region_id": "r1",
            "endpoint_url": "https://example.com/api",
            "api_key": "key",
            "sync_scope": "full",
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            result = handler._handle_register_region(mock_http)
        data = _body(result)
        assert data["region"]["sync_scope"] == "full"

    def test_sync_to_region_visibility_levels_forwarded(self, handler, mock_mound):
        """visibility_levels parameter is forwarded to mound."""
        body = {
            "region_id": "r1",
            "visibility_levels": ["public", "internal"],
        }
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            handler._handle_sync_to_region(mock_http)
        call_kwargs = mock_mound.sync_to_region.call_args.kwargs
        assert call_kwargs["visibility_levels"] == ["public", "internal"]

    def test_sync_to_region_no_visibility_passes_none(self, handler, mock_mound):
        """No visibility_levels parameter passes None to mound."""
        body = {"region_id": "r1"}
        mock_http = _make_http_handler(body)
        with _mock_metrics():
            handler._handle_sync_to_region(mock_http)
        call_kwargs = mock_mound.sync_to_region.call_args.kwargs
        assert call_kwargs["visibility_levels"] is None

    def test_sync_all_regions_no_since_passes_none(self, handler, mock_mound):
        """No since parameter passes None to mound in sync_all_regions."""
        mock_http = _make_http_handler({})
        with _mock_metrics():
            handler._handle_sync_all_regions(mock_http)
        call_kwargs = mock_mound.sync_all_regions.call_args.kwargs
        assert call_kwargs["since"] is None

    def test_sync_all_regions_no_workspace_passes_none(self, handler, mock_mound):
        """No workspace_id parameter passes None to mound in sync_all_regions."""
        mock_http = _make_http_handler({})
        with _mock_metrics():
            handler._handle_sync_all_regions(mock_http)
        call_kwargs = mock_mound.sync_all_regions.call_args.kwargs
        assert call_kwargs["workspace_id"] is None
