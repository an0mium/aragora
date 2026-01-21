"""Tests for enterprise connectors handler.

Tests the connectors API endpoints including:
- GET /api/connectors - list connectors
- GET /api/connectors/{id} - get connector details
- POST /api/connectors - create connector
- PUT /api/connectors/{id} - update connector
- DELETE /api/connectors/{id} - remove connector
- POST /api/connectors/{id}/sync - start sync
- POST /api/connectors/sync/{sync_id}/cancel - cancel sync
- POST /api/connectors/test - test connection
- GET /api/connectors/sync-history - sync history
- GET /api/connectors/stats - aggregate stats
- GET /api/connectors/types - list connector types
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.connectors import (
    ConnectorsHandler,
    CONNECTOR_TYPES,
    _connectors,
    _sync_jobs,
    _sync_history,
)


@dataclass
class MockRequest:
    """Mock async HTTP request."""

    method: str = "GET"
    path: str = "/"
    query: Dict[str, str] = None
    _body: Dict[str, Any] = None

    def __post_init__(self):
        if self.query is None:
            self.query = {}

    async def json(self) -> Dict[str, Any]:  # noqa: F811 - intentional method name
        return self._body or {}


@pytest.fixture(autouse=True)
def clear_global_state():
    """Clear global state before each test."""
    import aragora.server.handlers.features.connectors as conn_module

    _connectors.clear()
    _sync_jobs.clear()
    _sync_history.clear()
    # Reset the global store to force in-memory usage
    conn_module._store = None
    yield
    _connectors.clear()
    _sync_jobs.clear()
    _sync_history.clear()
    conn_module._store = None


@pytest.fixture(autouse=True)
def disable_persistent_store(monkeypatch):
    """Disable persistent store to use in-memory fallback."""
    import aragora.server.handlers.features.connectors as conn_module

    # Mock _get_store to always return None (use in-memory)
    async def mock_get_store():
        return None

    monkeypatch.setattr(conn_module, "_get_store", mock_get_store)


@pytest.fixture
def connectors_handler():
    """Create connectors handler instance."""
    ctx = {}
    return ConnectorsHandler(ctx)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestConnectorsHandlerInit:
    """Tests for handler initialization."""

    def test_routes_defined(self, connectors_handler):
        """Test that handler routes are defined."""
        assert hasattr(connectors_handler, "ROUTES")
        assert len(connectors_handler.ROUTES) > 0

    def test_connector_types_defined(self):
        """Test that connector types are defined."""
        assert len(CONNECTOR_TYPES) > 0
        assert "github" in CONNECTOR_TYPES
        assert "s3" in CONNECTOR_TYPES
        assert "postgresql" in CONNECTOR_TYPES


# =============================================================================
# List Connectors Tests
# =============================================================================


class TestListConnectors:
    """Tests for GET /api/connectors endpoint."""

    @pytest.mark.asyncio
    async def test_list_empty_connectors(self, connectors_handler):
        """Test listing when no connectors configured."""
        request = MockRequest(method="GET", path="/api/connectors")
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 200
        body = result["body"]
        assert body["total"] == 0
        assert body["connectors"] == []

    @pytest.mark.asyncio
    async def test_list_connectors_with_data(self, connectors_handler):
        """Test listing configured connectors."""
        # Add a connector directly to state
        _connectors["test-1"] = {
            "id": "test-1",
            "type": "github",
            "name": "Test GitHub",
            "status": "connected",
            "config": {},
        }

        request = MockRequest(method="GET", path="/api/connectors")
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 200
        body = result["body"]
        assert body["total"] == 1
        assert body["connected"] == 1

    @pytest.mark.asyncio
    async def test_list_connectors_with_status_filter(self, connectors_handler):
        """Test filtering connectors by status."""
        _connectors["test-1"] = {
            "id": "test-1",
            "type": "github",
            "name": "Test GitHub",
            "status": "connected",
            "config": {},
        }
        _connectors["test-2"] = {
            "id": "test-2",
            "type": "s3",
            "name": "Test S3",
            "status": "error",
            "config": {},
        }

        request = MockRequest(
            method="GET",
            path="/api/connectors",
            query={"status": "connected"},
        )
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 200

    @pytest.mark.asyncio
    async def test_list_connectors_with_category_filter(self, connectors_handler):
        """Test filtering connectors by category."""
        _connectors["test-1"] = {
            "id": "test-1",
            "type": "github",
            "name": "Test GitHub",
            "status": "connected",
            "config": {},
        }
        _connectors["test-2"] = {
            "id": "test-2",
            "type": "s3",
            "name": "Test S3",
            "status": "connected",
            "config": {},
        }

        request = MockRequest(
            method="GET",
            path="/api/connectors",
            query={"category": "git"},
        )
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 200
        body = result["body"]
        # Should only include github (category: git)
        assert body["total"] == 1


# =============================================================================
# Get Connector Tests
# =============================================================================


class TestGetConnector:
    """Tests for GET /api/connectors/{id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_existing_connector(self, connectors_handler):
        """Test getting an existing connector."""
        _connectors["test-1"] = {
            "id": "test-1",
            "type": "github",
            "name": "Test GitHub",
            "status": "connected",
            "config": {"repo": "test/repo"},
            "items_synced": 100,
            "last_sync": "2024-01-01T00:00:00Z",
        }

        request = MockRequest(method="GET", path="/api/connectors/test-1")
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 200
        body = result["body"]
        assert body["id"] == "test-1"
        assert body["type"] == "github"
        assert body["type_name"] == "GitHub Enterprise"

    @pytest.mark.asyncio
    async def test_get_nonexistent_connector(self, connectors_handler):
        """Test 404 for nonexistent connector."""
        request = MockRequest(method="GET", path="/api/connectors/nonexistent")
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 404


# =============================================================================
# Create Connector Tests
# =============================================================================


class TestCreateConnector:
    """Tests for POST /api/connectors endpoint."""

    @pytest.mark.asyncio
    async def test_create_connector(self, connectors_handler):
        """Test creating a new connector."""
        request = MockRequest(
            method="POST",
            path="/api/connectors",
            _body={
                "type": "github",
                "name": "My GitHub",
                "config": {"repo": "test/repo", "token": "secret"},
            },
        )
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 201
        body = result["body"]
        assert body["type"] == "github"
        assert body["name"] == "My GitHub"
        assert body["status"] == "configured"
        assert "id" in body

    @pytest.mark.asyncio
    async def test_create_connector_missing_type(self, connectors_handler):
        """Test rejection when type is missing."""
        request = MockRequest(
            method="POST",
            path="/api/connectors",
            _body={"name": "My Connector"},
        )
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 400
        assert "type is required" in result["body"]["error"]

    @pytest.mark.asyncio
    async def test_create_connector_unknown_type(self, connectors_handler):
        """Test rejection for unknown connector type."""
        request = MockRequest(
            method="POST",
            path="/api/connectors",
            _body={"type": "unknown_type"},
        )
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 400
        assert "Unknown connector type" in result["body"]["error"]

    @pytest.mark.asyncio
    async def test_create_connector_coming_soon_type(self, connectors_handler):
        """Test rejection for coming soon connector type."""
        request = MockRequest(
            method="POST",
            path="/api/connectors",
            _body={"type": "gdrive"},
        )
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 400
        assert "coming soon" in result["body"]["error"]


# =============================================================================
# Update Connector Tests
# =============================================================================


class TestUpdateConnector:
    """Tests for PUT /api/connectors/{id} endpoint."""

    @pytest.mark.asyncio
    async def test_update_connector(self, connectors_handler):
        """Test updating a connector."""
        _connectors["test-1"] = {
            "id": "test-1",
            "type": "github",
            "name": "Original Name",
            "status": "configured",
            "config": {"repo": "old/repo"},
        }

        request = MockRequest(
            method="PUT",
            path="/api/connectors/test-1",
            _body={"name": "Updated Name"},
        )
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 200
        body = result["body"]
        assert body["name"] == "Updated Name"

    @pytest.mark.asyncio
    async def test_update_nonexistent_connector(self, connectors_handler):
        """Test 404 for updating nonexistent connector."""
        request = MockRequest(
            method="PUT",
            path="/api/connectors/nonexistent",
            _body={"name": "New Name"},
        )
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_update_config_changes_status(self, connectors_handler):
        """Test that updating config of connected connector changes status."""
        _connectors["test-1"] = {
            "id": "test-1",
            "type": "github",
            "name": "Test",
            "status": "connected",
            "config": {"repo": "old/repo"},
        }

        request = MockRequest(
            method="PUT",
            path="/api/connectors/test-1",
            _body={"config": {"repo": "new/repo"}},
        )
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 200
        assert result["body"]["status"] == "configuring"


# =============================================================================
# Delete Connector Tests
# =============================================================================


class TestDeleteConnector:
    """Tests for DELETE /api/connectors/{id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_connector(self, connectors_handler):
        """Test deleting a connector."""
        _connectors["test-1"] = {
            "id": "test-1",
            "type": "github",
            "name": "Test",
            "status": "connected",
            "config": {},
        }

        request = MockRequest(method="DELETE", path="/api/connectors/test-1")
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 200
        assert "test-1" not in _connectors

    @pytest.mark.asyncio
    async def test_delete_nonexistent_connector(self, connectors_handler):
        """Test 404 for deleting nonexistent connector."""
        request = MockRequest(method="DELETE", path="/api/connectors/nonexistent")
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_delete_cancels_active_syncs(self, connectors_handler):
        """Test that deleting a connector cancels active syncs."""
        _connectors["test-1"] = {
            "id": "test-1",
            "type": "github",
            "name": "Test",
            "status": "syncing",
            "config": {},
        }
        _sync_jobs["sync-1"] = {
            "id": "sync-1",
            "connector_id": "test-1",
            "status": "running",
        }

        request = MockRequest(method="DELETE", path="/api/connectors/test-1")
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 200
        assert _sync_jobs["sync-1"]["status"] == "cancelled"


# =============================================================================
# Sync Operations Tests
# =============================================================================


class TestSyncOperations:
    """Tests for sync operation endpoints."""

    @pytest.mark.asyncio
    async def test_start_sync(self, connectors_handler):
        """Test starting a sync operation."""
        _connectors["test-1"] = {
            "id": "test-1",
            "type": "github",
            "name": "Test",
            "status": "configured",
            "config": {},
        }

        request = MockRequest(
            method="POST",
            path="/api/connectors/test-1/sync",
        )
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 202
        assert "sync_id" in result["body"]
        assert result["body"]["connector_id"] == "test-1"

    @pytest.mark.asyncio
    async def test_start_sync_nonexistent_connector(self, connectors_handler):
        """Test 404 for syncing nonexistent connector."""
        request = MockRequest(
            method="POST",
            path="/api/connectors/nonexistent/sync",
        )
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_start_sync_already_running(self, connectors_handler):
        """Test 409 when sync already running."""
        _connectors["test-1"] = {
            "id": "test-1",
            "type": "github",
            "name": "Test",
            "status": "syncing",
            "config": {},
        }
        _sync_jobs["sync-1"] = {
            "id": "sync-1",
            "connector_id": "test-1",
            "status": "running",
        }

        request = MockRequest(
            method="POST",
            path="/api/connectors/test-1/sync",
        )
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 409
        assert "already in progress" in result["body"]["error"]

    @pytest.mark.asyncio
    async def test_cancel_sync(self, connectors_handler):
        """Test cancelling a running sync."""
        _connectors["test-1"] = {
            "id": "test-1",
            "type": "github",
            "name": "Test",
            "status": "syncing",
            "config": {},
        }
        _sync_jobs["sync-1"] = {
            "id": "sync-1",
            "connector_id": "test-1",
            "status": "running",
        }

        request = MockRequest(
            method="POST",
            path="/api/connectors/sync/sync-1/cancel",
        )
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 200
        assert _sync_jobs["sync-1"]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_sync(self, connectors_handler):
        """Test 404 for cancelling nonexistent sync."""
        request = MockRequest(
            method="POST",
            path="/api/connectors/sync/nonexistent/cancel",
        )
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_cancel_non_running_sync(self, connectors_handler):
        """Test 400 for cancelling completed sync."""
        _sync_jobs["sync-1"] = {
            "id": "sync-1",
            "connector_id": "test-1",
            "status": "completed",
        }

        request = MockRequest(
            method="POST",
            path="/api/connectors/sync/sync-1/cancel",
        )
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 400


# =============================================================================
# Test Connection Tests
# =============================================================================


class TestTestConnection:
    """Tests for POST /api/connectors/test endpoint."""

    @pytest.mark.asyncio
    async def test_test_connection_success(self, connectors_handler):
        """Test successful connection test."""
        request = MockRequest(
            method="POST",
            path="/api/connectors/test",
            _body={"config": {"token": "test_token"}},
        )
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 200
        assert result["body"]["success"] is True

    @pytest.mark.asyncio
    async def test_test_connection_failure(self, connectors_handler):
        """Test failed connection test (empty config)."""
        request = MockRequest(
            method="POST",
            path="/api/connectors/test",
            _body={"config": {}},
        )
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 200
        assert result["body"]["success"] is False


# =============================================================================
# Sync History Tests
# =============================================================================


class TestSyncHistory:
    """Tests for GET /api/connectors/sync-history endpoint."""

    @pytest.mark.asyncio
    async def test_get_empty_history(self, connectors_handler):
        """Test getting empty sync history."""
        request = MockRequest(method="GET", path="/api/connectors/sync-history")
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 200
        assert result["body"]["total"] == 0
        assert result["body"]["history"] == []

    @pytest.mark.asyncio
    async def test_get_sync_history(self, connectors_handler):
        """Test getting sync history."""
        _sync_history.append(
            {
                "id": "sync-1",
                "connector_id": "test-1",
                "status": "completed",
                "started_at": "2024-01-01T00:00:00Z",
                "items_processed": 100,
            }
        )

        request = MockRequest(method="GET", path="/api/connectors/sync-history")
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 200
        assert result["body"]["total"] == 1

    @pytest.mark.asyncio
    async def test_get_sync_history_filtered(self, connectors_handler):
        """Test filtering sync history by connector."""
        _sync_history.append(
            {
                "id": "sync-1",
                "connector_id": "test-1",
                "status": "completed",
                "started_at": "2024-01-01T00:00:00Z",
            }
        )
        _sync_history.append(
            {
                "id": "sync-2",
                "connector_id": "test-2",
                "status": "completed",
                "started_at": "2024-01-02T00:00:00Z",
            }
        )

        request = MockRequest(
            method="GET",
            path="/api/connectors/sync-history",
            query={"connector_id": "test-1"},
        )
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 200
        assert result["body"]["total"] == 1


# =============================================================================
# Stats Tests
# =============================================================================


class TestStats:
    """Tests for GET /api/connectors/stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_empty_stats(self, connectors_handler):
        """Test getting stats with no connectors."""
        request = MockRequest(method="GET", path="/api/connectors/stats")
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 200
        body = result["body"]
        assert body["total_connectors"] == 0
        assert body["connected"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_with_data(self, connectors_handler):
        """Test getting stats with connectors."""
        _connectors["test-1"] = {
            "id": "test-1",
            "type": "github",
            "name": "Test",
            "status": "connected",
            "config": {},
            "items_synced": 100,
        }
        _connectors["test-2"] = {
            "id": "test-2",
            "type": "s3",
            "name": "Test S3",
            "status": "error",
            "config": {},
            "items_synced": 50,
        }

        request = MockRequest(method="GET", path="/api/connectors/stats")
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 200
        body = result["body"]
        assert body["total_connectors"] == 2
        assert body["connected"] == 1
        assert body["errors"] == 1
        assert body["total_items_synced"] == 150


# =============================================================================
# List Types Tests
# =============================================================================


class TestListTypes:
    """Tests for GET /api/connectors/types endpoint."""

    @pytest.mark.asyncio
    async def test_list_types(self, connectors_handler):
        """Test listing connector types."""
        request = MockRequest(method="GET", path="/api/connectors/types")
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 200
        body = result["body"]
        assert "types" in body
        assert len(body["types"]) == len(CONNECTOR_TYPES)

        # Check a specific type
        github_type = next(t for t in body["types"] if t["type"] == "github")
        assert github_type["name"] == "GitHub Enterprise"
        assert github_type["category"] == "git"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_unknown_endpoint(self, connectors_handler):
        """Test 404 for unknown endpoint."""
        request = MockRequest(method="GET", path="/api/connectors/unknown/endpoint")
        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_invalid_json_body(self, connectors_handler):
        """Test handling of invalid JSON body."""
        request = MockRequest(
            method="POST",
            path="/api/connectors",
        )
        # Make json() raise an exception
        request.json = AsyncMock(side_effect=Exception("Invalid JSON"))

        result = await connectors_handler.handle_request(request)

        assert result["status_code"] == 400
        assert "Invalid JSON" in result["body"]["error"]


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for connector workflow."""

    @pytest.mark.asyncio
    async def test_create_then_sync_workflow(self, connectors_handler):
        """Test creating a connector then starting a sync."""
        # Create connector
        create_request = MockRequest(
            method="POST",
            path="/api/connectors",
            _body={"type": "github", "name": "Test Repo"},
        )
        create_result = await connectors_handler.handle_request(create_request)
        assert create_result["status_code"] == 201
        connector_id = create_result["body"]["id"]

        # Start sync
        sync_request = MockRequest(
            method="POST",
            path=f"/api/connectors/{connector_id}/sync",
        )
        sync_result = await connectors_handler.handle_request(sync_request)
        assert sync_result["status_code"] == 202
        assert "sync_id" in sync_result["body"]

    @pytest.mark.asyncio
    async def test_full_crud_workflow(self, connectors_handler):
        """Test create, read, update, delete workflow."""
        # Create
        create_request = MockRequest(
            method="POST",
            path="/api/connectors",
            _body={"type": "s3", "name": "Test Bucket"},
        )
        create_result = await connectors_handler.handle_request(create_request)
        assert create_result["status_code"] == 201
        connector_id = create_result["body"]["id"]

        # Read
        read_request = MockRequest(
            method="GET",
            path=f"/api/connectors/{connector_id}",
        )
        read_result = await connectors_handler.handle_request(read_request)
        assert read_result["status_code"] == 200
        assert read_result["body"]["name"] == "Test Bucket"

        # Update
        update_request = MockRequest(
            method="PUT",
            path=f"/api/connectors/{connector_id}",
            _body={"name": "Updated Bucket"},
        )
        update_result = await connectors_handler.handle_request(update_request)
        assert update_result["status_code"] == 200
        assert update_result["body"]["name"] == "Updated Bucket"

        # Delete
        delete_request = MockRequest(
            method="DELETE",
            path=f"/api/connectors/{connector_id}",
        )
        delete_result = await connectors_handler.handle_request(delete_request)
        assert delete_result["status_code"] == 200

        # Verify deleted
        verify_request = MockRequest(
            method="GET",
            path=f"/api/connectors/{connector_id}",
        )
        verify_result = await connectors_handler.handle_request(verify_request)
        assert verify_result["status_code"] == 404
