"""
Tests for Knowledge Mound Federation handler endpoints.

Tests federation operations:
- POST /api/knowledge/mound/federation/regions - Register a federated region
- DELETE /api/knowledge/mound/federation/regions/:id - Unregister a region
- POST /api/knowledge/mound/federation/sync/push - Sync to a region
- POST /api/knowledge/mound/federation/sync/pull - Pull from a region
- POST /api/knowledge/mound/federation/sync/all - Sync with all regions
- GET /api/knowledge/mound/federation/status - Get federation status
- GET /api/knowledge/mound/federation/regions - List federated regions
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.knowledge_base.mound.federation import (
    FederationOperationsMixin,
)


class MockFederationMode(str, Enum):
    """Mock federation mode for testing."""

    BIDIRECTIONAL = "bidirectional"
    PUSH_ONLY = "push_only"
    PULL_ONLY = "pull_only"


class MockSyncScope(str, Enum):
    """Mock sync scope for testing."""

    SUMMARY = "summary"
    FULL = "full"
    METADATA_ONLY = "metadata_only"


@dataclass
class MockFederatedRegion:
    """Mock federated region for testing."""

    region_id: str = "region-123"
    endpoint_url: str = "https://remote.example.com/api"
    mode: MockFederationMode = MockFederationMode.BIDIRECTIONAL
    sync_scope: MockSyncScope = MockSyncScope.SUMMARY
    enabled: bool = True


@dataclass
class MockSyncResult:
    """Mock sync result for testing."""

    success: bool = True
    region_id: str = "region-123"
    direction: str = "push"
    nodes_synced: int = 50
    nodes_skipped: int = 5
    nodes_failed: int = 0
    duration_ms: float = 250.5
    error: str | None = None


class MockMound:
    """Mock KnowledgeMound for testing."""

    def __init__(self):
        self.registered_regions = {}

    async def register_federated_region(self, **kwargs):
        region = MockFederatedRegion(
            region_id=kwargs.get("region_id", "region-123"),
            endpoint_url=kwargs.get("endpoint_url", "https://remote.example.com/api"),
            mode=kwargs.get("mode", MockFederationMode.BIDIRECTIONAL),
            sync_scope=kwargs.get("sync_scope", MockSyncScope.SUMMARY),
        )
        self.registered_regions[region.region_id] = region
        return region

    async def unregister_federated_region(self, region_id: str):
        if region_id in self.registered_regions:
            del self.registered_regions[region_id]
            return True
        return False

    async def sync_to_region(self, **kwargs):
        return MockSyncResult(
            region_id=kwargs.get("region_id", "region-123"),
            direction="push",
        )

    async def pull_from_region(self, **kwargs):
        return MockSyncResult(
            region_id=kwargs.get("region_id", "region-123"),
            direction="pull",
        )

    async def sync_all_regions(self, **kwargs):
        return [
            MockSyncResult(region_id="region-1", direction="push"),
            MockSyncResult(region_id="region-2", direction="push"),
        ]

    async def get_federation_status(self):
        return {
            "region-1": {"enabled": True, "healthy": True},
            "region-2": {"enabled": True, "healthy": False},
            "region-3": {"enabled": False, "healthy": True},
        }


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, body_data: dict | None = None):
        self.headers = {"Content-Length": "0"}
        self.rfile = io.BytesIO(b"")
        if body_data is not None:  # Allow empty dict {}
            body_bytes = json.dumps(body_data).encode("utf-8")
            self.headers = {"Content-Length": str(len(body_bytes))}
            self.rfile = io.BytesIO(body_bytes)


class MockFederationHandler(FederationOperationsMixin):
    """Handler for testing FederationOperationsMixin."""

    def __init__(self):
        self.mound = MockMound()

    def _get_mound(self):
        return self.mound

    def require_auth_or_error(self, handler):
        return {"user_id": "test-user"}, None

    def require_admin_or_error(self, handler):
        return {"user_id": "admin-user", "is_admin": True}, None


class MockFederationHandlerNoMound(FederationOperationsMixin):
    """Handler with no mound available."""

    def _get_mound(self):
        return None

    def require_auth_or_error(self, handler):
        return {"user_id": "test-user"}, None

    def require_admin_or_error(self, handler):
        return {"user_id": "admin-user"}, None


class MockFederationHandlerAuthFail(FederationOperationsMixin):
    """Handler that fails auth."""

    def __init__(self):
        self.mound = MockMound()

    def _get_mound(self):
        return self.mound

    def require_auth_or_error(self, handler):
        from aragora.server.handlers.base import error_response

        return None, error_response("Unauthorized", 401)

    def require_admin_or_error(self, handler):
        from aragora.server.handlers.base import error_response

        return None, error_response("Admin required", 403)


def parse_json_response(result):
    """Parse JSON response from HandlerResult dataclass."""
    body = result.body
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    return json.loads(body)


@pytest.fixture
def handler():
    """Create test handler with mocked mound."""
    return MockFederationHandler()


@pytest.fixture
def handler_no_mound():
    """Create test handler without mound."""
    return MockFederationHandlerNoMound()


@pytest.fixture
def handler_auth_fail():
    """Create test handler that fails auth."""
    return MockFederationHandlerAuthFail()


# Mock the decorators to bypass RBAC and rate limiting for tests
@pytest.fixture(autouse=True)
def mock_decorators():
    """Mock RBAC, rate limit decorators, and metrics."""
    with (
        patch(
            "aragora.server.handlers.knowledge_base.mound.federation.require_permission",
            lambda perm: lambda fn: fn,
        ),
        patch(
            "aragora.server.handlers.knowledge_base.mound.federation.rate_limit",
            lambda **kwargs: lambda fn: fn,
        ),
        patch(
            "aragora.server.handlers.knowledge_base.mound.federation.handle_errors",
            lambda msg: lambda fn: fn,
        ),
        patch(
            "aragora.server.handlers.knowledge_base.mound.federation.track_federation_sync",
            MagicMock(return_value=MagicMock(__enter__=lambda s: {}, __exit__=lambda s, *a: None)),
        ),
        patch(
            "aragora.server.handlers.knowledge_base.mound.federation.track_federation_regions",
            MagicMock(),
        ),
        patch(
            "aragora.server.handlers.knowledge_base.mound.federation._run_async",
            lambda coro: __import__("asyncio").get_event_loop().run_until_complete(coro)
            if hasattr(coro, "__await__")
            else coro,
        ),
    ):
        yield


class TestRegisterRegion:
    """Tests for _handle_register_region endpoint."""

    def test_register_success(self, handler):
        """Test successful region registration."""
        with (
            patch(
                "aragora.knowledge.mound.ops.federation.FederationMode",
                MockFederationMode,
            ),
            patch(
                "aragora.knowledge.mound.ops.federation.SyncScope",
                MockSyncScope,
            ),
        ):
            mock_handler = MockHandler(
                body_data={
                    "region_id": "new-region",
                    "endpoint_url": "https://new.example.com/api",
                    "api_key": "secret-key",
                    "mode": "bidirectional",
                    "sync_scope": "summary",
                }
            )
            result = handler._handle_register_region(mock_handler)

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 201
        assert data["success"] is True
        assert data["region"]["region_id"] == "new-region"

    def test_register_missing_region_id(self, handler):
        """Test registration fails without region_id."""
        with patch(
            "aragora.knowledge.mound.ops.federation.FederationMode",
            MockFederationMode,
        ):
            mock_handler = MockHandler(
                body_data={
                    "endpoint_url": "https://new.example.com/api",
                    "api_key": "secret-key",
                }
            )
            result = handler._handle_register_region(mock_handler)

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "region_id is required" in data.get("error", "")

    def test_register_missing_endpoint_url(self, handler):
        """Test registration fails without endpoint_url."""
        with patch(
            "aragora.knowledge.mound.ops.federation.FederationMode",
            MockFederationMode,
        ):
            mock_handler = MockHandler(
                body_data={
                    "region_id": "new-region",
                    "api_key": "secret-key",
                }
            )
            result = handler._handle_register_region(mock_handler)

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "endpoint_url is required" in data.get("error", "")

    def test_register_missing_api_key(self, handler):
        """Test registration fails without api_key."""
        with patch(
            "aragora.knowledge.mound.ops.federation.FederationMode",
            MockFederationMode,
        ):
            mock_handler = MockHandler(
                body_data={
                    "region_id": "new-region",
                    "endpoint_url": "https://new.example.com/api",
                }
            )
            result = handler._handle_register_region(mock_handler)

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "api_key is required" in data.get("error", "")

    def test_register_invalid_mode(self, handler):
        """Test registration fails with invalid mode."""
        with patch(
            "aragora.knowledge.mound.ops.federation.FederationMode",
            MockFederationMode,
        ):
            mock_handler = MockHandler(
                body_data={
                    "region_id": "new-region",
                    "endpoint_url": "https://new.example.com/api",
                    "api_key": "secret-key",
                    "mode": "invalid_mode",
                }
            )
            result = handler._handle_register_region(mock_handler)

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "Invalid mode" in data.get("error", "")

    def test_register_no_body(self, handler):
        """Test registration fails without body."""
        mock_handler = MockHandler()
        result = handler._handle_register_region(mock_handler)

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "body required" in data.get("error", "")

    def test_register_mound_unavailable(self, handler_no_mound):
        """Test registration when mound is unavailable."""
        with (
            patch(
                "aragora.knowledge.mound.ops.federation.FederationMode",
                MockFederationMode,
            ),
            patch(
                "aragora.knowledge.mound.ops.federation.SyncScope",
                MockSyncScope,
            ),
        ):
            mock_handler = MockHandler(
                body_data={
                    "region_id": "new-region",
                    "endpoint_url": "https://new.example.com/api",
                    "api_key": "secret-key",
                }
            )
            result = handler_no_mound._handle_register_region(mock_handler)

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 503
        assert "not available" in data.get("error", "")

    def test_register_admin_required(self, handler_auth_fail):
        """Test registration fails without admin."""
        mock_handler = MockHandler(
            body_data={
                "region_id": "new-region",
                "endpoint_url": "https://new.example.com/api",
                "api_key": "secret-key",
            }
        )
        result = handler_auth_fail._handle_register_region(mock_handler)

        status_code = result.status_code
        assert status_code == 403


class TestUnregisterRegion:
    """Tests for _handle_unregister_region endpoint."""

    def test_unregister_success(self, handler):
        """Test successful region unregistration."""
        # First register a region
        handler.mound.registered_regions["region-123"] = MockFederatedRegion()

        mock_handler = MockHandler()
        result = handler._handle_unregister_region("region-123", mock_handler)

        data = parse_json_response(result)
        assert data["success"] is True
        assert data["region_id"] == "region-123"

    def test_unregister_not_found(self, handler):
        """Test unregistration of non-existent region."""
        mock_handler = MockHandler()
        result = handler._handle_unregister_region("nonexistent", mock_handler)

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 404
        assert "not found" in data.get("error", "")

    def test_unregister_mound_unavailable(self, handler_no_mound):
        """Test unregistration when mound is unavailable."""
        mock_handler = MockHandler()
        result = handler_no_mound._handle_unregister_region("region-123", mock_handler)

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 503
        assert "not available" in data.get("error", "")


class TestSyncToRegion:
    """Tests for _handle_sync_to_region endpoint."""

    def test_sync_push_success(self, handler):
        """Test successful sync push."""
        mock_handler = MockHandler(
            body_data={
                "region_id": "region-123",
            }
        )
        result = handler._handle_sync_to_region(mock_handler)

        data = parse_json_response(result)
        assert data["success"] is True
        assert data["direction"] == "push"

    def test_sync_push_missing_region_id(self, handler):
        """Test sync push fails without region_id."""
        mock_handler = MockHandler(body_data={})
        result = handler._handle_sync_to_region(mock_handler)

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "region_id is required" in data.get("error", "")

    def test_sync_push_no_body(self, handler):
        """Test sync push fails without body."""
        mock_handler = MockHandler()
        result = handler._handle_sync_to_region(mock_handler)

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "body required" in data.get("error", "")

    def test_sync_push_mound_unavailable(self, handler_no_mound):
        """Test sync push when mound is unavailable."""
        mock_handler = MockHandler(body_data={"region_id": "region-123"})
        result = handler_no_mound._handle_sync_to_region(mock_handler)

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 503
        assert "not available" in data.get("error", "")


class TestPullFromRegion:
    """Tests for _handle_pull_from_region endpoint."""

    def test_pull_success(self, handler):
        """Test successful pull."""
        mock_handler = MockHandler(
            body_data={
                "region_id": "region-123",
            }
        )
        result = handler._handle_pull_from_region(mock_handler)

        data = parse_json_response(result)
        assert data["success"] is True
        assert data["direction"] == "pull"

    def test_pull_missing_region_id(self, handler):
        """Test pull fails without region_id."""
        mock_handler = MockHandler(body_data={})
        result = handler._handle_pull_from_region(mock_handler)

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "region_id is required" in data.get("error", "")

    def test_pull_mound_unavailable(self, handler_no_mound):
        """Test pull when mound is unavailable."""
        mock_handler = MockHandler(body_data={"region_id": "region-123"})
        result = handler_no_mound._handle_pull_from_region(mock_handler)

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 503
        assert "not available" in data.get("error", "")


class TestSyncAllRegions:
    """Tests for _handle_sync_all_regions endpoint."""

    def test_sync_all_success(self, handler):
        """Test successful sync all regions."""
        mock_handler = MockHandler(body_data={})
        result = handler._handle_sync_all_regions(mock_handler)

        data = parse_json_response(result)
        assert data["total_regions"] == 2
        assert data["successful"] == 2
        assert data["failed"] == 0

    def test_sync_all_with_workspace(self, handler):
        """Test sync all with workspace filter."""
        mock_handler = MockHandler(
            body_data={
                "workspace_id": "workspace-123",
            }
        )
        result = handler._handle_sync_all_regions(mock_handler)

        data = parse_json_response(result)
        assert data["total_regions"] == 2

    def test_sync_all_mound_unavailable(self, handler_no_mound):
        """Test sync all when mound is unavailable."""
        mock_handler = MockHandler(body_data={})
        result = handler_no_mound._handle_sync_all_regions(mock_handler)

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 503
        assert "not available" in data.get("error", "")


class TestGetFederationStatus:
    """Tests for _handle_get_federation_status endpoint."""

    def test_get_status_success(self, handler):
        """Test successful status retrieval."""
        result = handler._handle_get_federation_status({})

        data = parse_json_response(result)
        assert data["total_regions"] == 3
        assert data["enabled_regions"] == 2
        assert "regions" in data

    def test_get_status_mound_unavailable(self, handler_no_mound):
        """Test status when mound is unavailable."""
        result = handler_no_mound._handle_get_federation_status({})

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 503
        assert "not available" in data.get("error", "")


class TestListRegions:
    """Tests for _handle_list_regions endpoint."""

    def test_list_regions_success(self, handler):
        """Test successful region listing."""
        result = handler._handle_list_regions({})

        data = parse_json_response(result)
        assert data["count"] == 3
        assert len(data["regions"]) == 3

    def test_list_regions_mound_unavailable(self, handler_no_mound):
        """Test listing when mound is unavailable."""
        result = handler_no_mound._handle_list_regions({})

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 503
        assert "not available" in data.get("error", "")
