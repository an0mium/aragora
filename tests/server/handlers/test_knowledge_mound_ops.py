"""Tests for Knowledge Mound operations handler endpoints.

Tests cover the new Knowledge Mound features:
- Visibility operations (set/get visibility, access grants)
- Sharing operations (share with workspace/user)
- Global knowledge operations (verified facts)
- Federation operations (region sync)
"""

import json
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.knowledge_base.mound.handler import KnowledgeMoundHandler


@pytest.fixture
def mound_handler():
    """Create a knowledge mound handler with mocked dependencies."""
    ctx = {"storage": None, "elo_system": None, "nomic_dir": None}
    handler = KnowledgeMoundHandler(ctx)
    return handler


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler for GET requests."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {"Content-Length": "0"}
    handler.command = "GET"
    return handler


def create_request_body(data: dict, method: str = "POST") -> MagicMock:
    """Create a mock handler with request body."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    body = json.dumps(data).encode("utf-8")
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = BytesIO(body)
    handler.command = method
    return handler


def mock_user(user_id: str = "test-user", is_admin: bool = False):
    """Create a mock user context."""
    user = MagicMock()
    user.id = user_id
    user.user_id = user_id
    user.roles = ["admin"] if is_admin else ["user"]
    user.permissions = ["admin"] if is_admin else ["read"]
    user.is_admin = is_admin
    return user


# =============================================================================
# Visibility Endpoints Tests
# =============================================================================


class TestVisibilityCanHandle:
    """Test can_handle for visibility endpoints."""

    def test_can_handle_visibility_get(self, mound_handler):
        """Test can_handle for GET visibility."""
        assert mound_handler.can_handle("/api/knowledge/mound/nodes/node-123/visibility")

    def test_can_handle_access_grants(self, mound_handler):
        """Test can_handle for access grants."""
        assert mound_handler.can_handle("/api/knowledge/mound/nodes/node-123/access")


class TestGetVisibility:
    """Test GET /api/knowledge/mound/nodes/:id/visibility endpoint."""

    def test_get_visibility_mound_unavailable(self, mound_handler, mock_http_handler):
        """Test returns 503 when mound not available."""
        with patch.object(mound_handler, "_get_mound", return_value=None):
            result = mound_handler.handle(
                "/api/knowledge/mound/nodes/node-123/visibility", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 503

    def test_get_visibility_node_not_found(self, mound_handler, mock_http_handler):
        """Test returns 404 when node not found."""
        mock_mound = MagicMock()
        mock_mound.get_node = AsyncMock(return_value=None)

        with patch.object(mound_handler, "_get_mound", return_value=mock_mound):
            result = mound_handler.handle(
                "/api/knowledge/mound/nodes/nonexistent/visibility", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 404

    def test_get_visibility_success(self, mound_handler, mock_http_handler):
        """Test getting visibility for existing node."""
        mock_node = MagicMock()
        mock_node.id = "node-123"
        mock_node.metadata = {
            "visibility": "workspace",
            "visibility_set_by": "admin",
            "is_discoverable": True,
        }

        mock_mound = MagicMock()
        mock_mound.get_node = AsyncMock(return_value=mock_node)

        with patch.object(mound_handler, "_get_mound", return_value=mock_mound):
            result = mound_handler.handle(
                "/api/knowledge/mound/nodes/node-123/visibility", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["visibility"] == "workspace"
        assert body["visibility_set_by"] == "admin"
        assert body["is_discoverable"] is True


class TestSetVisibility:
    """Test PUT /api/knowledge/mound/nodes/:id/visibility endpoint."""

    def test_set_visibility_requires_auth(self, mound_handler):
        """Test setting visibility requires authentication."""
        handler = create_request_body({"visibility": "private"}, "PUT")

        with patch.object(mound_handler, "require_auth_or_error") as mock_auth:
            from aragora.server.handlers.base import error_response

            mock_auth.return_value = (None, error_response("Unauthorized", 401))

            result = mound_handler.handle(
                "/api/knowledge/mound/nodes/node-123/visibility", {}, handler
            )

        assert result is not None
        assert result.status_code == 401

    def test_set_visibility_invalid_level(self, mound_handler):
        """Test setting invalid visibility level."""
        handler = create_request_body({"visibility": "invalid"}, "PUT")

        with patch.object(mound_handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (mock_user(), None)

            result = mound_handler.handle(
                "/api/knowledge/mound/nodes/node-123/visibility", {}, handler
            )

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body

    def test_set_visibility_success(self, mound_handler):
        """Test setting visibility successfully."""
        handler = create_request_body(
            {
                "visibility": "private",
                "is_discoverable": False,
            },
            "PUT",
        )

        mock_mound = MagicMock()
        mock_mound.set_visibility = AsyncMock(return_value=True)

        with patch.object(mound_handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (mock_user(), None)
            with patch.object(mound_handler, "_get_mound", return_value=mock_mound):
                result = mound_handler.handle(
                    "/api/knowledge/mound/nodes/node-123/visibility", {}, handler
                )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True
        assert body["visibility"] == "private"


class TestAccessGrants:
    """Test access grant endpoints."""

    def test_grant_access_success(self, mound_handler):
        """Test granting access to a node."""
        handler = create_request_body(
            {
                "grantee_type": "user",
                "grantee_id": "user-456",
                "permissions": ["read", "write"],
            },
            "POST",
        )

        mock_grant = MagicMock()
        mock_grant.to_dict = MagicMock(
            return_value={
                "id": "grant-1",
                "item_id": "node-123",
                "grantee_type": "user",
                "grantee_id": "user-456",
                "permissions": ["read", "write"],
            }
        )

        mock_mound = MagicMock()
        mock_mound.grant_access = AsyncMock(return_value=mock_grant)

        with patch.object(mound_handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (mock_user(), None)
            with patch.object(mound_handler, "_get_mound", return_value=mock_mound):
                result = mound_handler.handle(
                    "/api/knowledge/mound/nodes/node-123/access", {}, handler
                )

        assert result is not None
        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["success"] is True

    def test_list_access_grants(self, mound_handler, mock_http_handler):
        """Test listing access grants for a node."""
        mock_grant = MagicMock()
        mock_grant.to_dict = MagicMock(
            return_value={
                "id": "grant-1",
                "grantee_type": "user",
                "grantee_id": "user-456",
            }
        )

        mock_mound = MagicMock()
        mock_mound.get_access_grants = AsyncMock(return_value=[mock_grant])

        with patch.object(mound_handler, "_get_mound", return_value=mock_mound):
            result = mound_handler.handle(
                "/api/knowledge/mound/nodes/node-123/access", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] == 1

    def test_revoke_access(self, mound_handler):
        """Test revoking access from a node."""
        handler = create_request_body({"grantee_id": "user-456"}, "DELETE")

        mock_mound = MagicMock()
        mock_mound.revoke_access = AsyncMock(return_value=True)

        with patch.object(mound_handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (mock_user(), None)
            with patch.object(mound_handler, "_get_mound", return_value=mock_mound):
                result = mound_handler.handle(
                    "/api/knowledge/mound/nodes/node-123/access", {}, handler
                )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True


# =============================================================================
# Sharing Endpoints Tests
# =============================================================================


class TestSharingCanHandle:
    """Test can_handle for sharing endpoints."""

    def test_can_handle_share(self, mound_handler):
        """Test can_handle for share endpoint."""
        assert mound_handler.can_handle("/api/knowledge/mound/share")

    def test_can_handle_shared_with_me(self, mound_handler):
        """Test can_handle for shared-with-me endpoint."""
        assert mound_handler.can_handle("/api/knowledge/mound/shared-with-me")

    def test_can_handle_my_shares(self, mound_handler):
        """Test can_handle for my-shares endpoint."""
        assert mound_handler.can_handle("/api/knowledge/mound/my-shares")


class TestShareItem:
    """Test POST /api/knowledge/mound/share endpoint."""

    def test_share_requires_auth(self, mound_handler):
        """Test sharing requires authentication."""
        handler = create_request_body(
            {
                "item_id": "item-123",
                "target_type": "workspace",
                "target_id": "ws-456",
            }
        )

        with patch.object(mound_handler, "require_auth_or_error") as mock_auth:
            from aragora.server.handlers.base import error_response

            mock_auth.return_value = (None, error_response("Unauthorized", 401))

            result = mound_handler.handle("/api/knowledge/mound/share", {}, handler)

        assert result is not None
        assert result.status_code == 401

    def test_share_with_workspace_success(self, mound_handler):
        """Test sharing with workspace."""
        handler = create_request_body(
            {
                "item_id": "item-123",
                "target_type": "workspace",
                "target_id": "ws-456",
                "permissions": ["read"],
            }
        )

        mock_grant = MagicMock()
        mock_mound = MagicMock()
        mock_mound.share_with_workspace = AsyncMock(return_value=mock_grant)

        with patch.object(mound_handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (mock_user(), None)
            with patch.object(mound_handler, "_get_mound", return_value=mock_mound):
                result = mound_handler.handle("/api/knowledge/mound/share", {}, handler)

        assert result is not None
        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["success"] is True

    def test_share_with_user_success(self, mound_handler):
        """Test sharing with user."""
        handler = create_request_body(
            {
                "item_id": "item-123",
                "target_type": "user",
                "target_id": "user-456",
            }
        )

        mock_grant = MagicMock()
        mock_mound = MagicMock()
        mock_mound.share_with_user = AsyncMock(return_value=mock_grant)

        with patch.object(mound_handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (mock_user(), None)
            with patch.object(mound_handler, "_get_mound", return_value=mock_mound):
                result = mound_handler.handle("/api/knowledge/mound/share", {}, handler)

        assert result is not None
        assert result.status_code == 201

    def test_share_invalid_target_type(self, mound_handler):
        """Test sharing with invalid target type."""
        handler = create_request_body(
            {
                "item_id": "item-123",
                "target_type": "invalid",
                "target_id": "target-456",
            }
        )

        with patch.object(mound_handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (mock_user(), None)

            result = mound_handler.handle("/api/knowledge/mound/share", {}, handler)

        assert result is not None
        assert result.status_code == 400


class TestSharedWithMe:
    """Test GET /api/knowledge/mound/shared-with-me endpoint."""

    def test_shared_with_me_requires_auth(self, mound_handler, mock_http_handler):
        """Test shared-with-me requires authentication."""
        with patch.object(mound_handler, "require_auth_or_error") as mock_auth:
            from aragora.server.handlers.base import error_response

            mock_auth.return_value = (None, error_response("Unauthorized", 401))

            result = mound_handler.handle(
                "/api/knowledge/mound/shared-with-me", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 401

    def test_shared_with_me_success(self, mound_handler, mock_http_handler):
        """Test getting items shared with me."""
        mock_item = MagicMock()
        mock_item.id = "item-123"
        mock_item.content = "Shared content"
        mock_item.to_dict = MagicMock(return_value={"id": "item-123", "content": "Shared content"})

        mock_mound = MagicMock()
        mock_mound.get_shared_with_me = AsyncMock(return_value=[mock_item])

        with patch.object(mound_handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (mock_user(), None)
            with patch.object(mound_handler, "_get_mound", return_value=mock_mound):
                result = mound_handler.handle(
                    "/api/knowledge/mound/shared-with-me", {}, mock_http_handler
                )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] == 1


# =============================================================================
# Global Knowledge Endpoints Tests
# =============================================================================


class TestGlobalKnowledgeCanHandle:
    """Test can_handle for global knowledge endpoints."""

    def test_can_handle_global(self, mound_handler):
        """Test can_handle for global endpoint."""
        assert mound_handler.can_handle("/api/knowledge/mound/global")

    def test_can_handle_global_promote(self, mound_handler):
        """Test can_handle for global/promote endpoint."""
        assert mound_handler.can_handle("/api/knowledge/mound/global/promote")

    def test_can_handle_global_facts(self, mound_handler):
        """Test can_handle for global/facts endpoint."""
        assert mound_handler.can_handle("/api/knowledge/mound/global/facts")


class TestQueryGlobalKnowledge:
    """Test GET /api/knowledge/mound/global endpoint."""

    def test_query_global_mound_unavailable(self, mound_handler, mock_http_handler):
        """Test returns 503 when mound not available."""
        with patch.object(mound_handler, "_get_mound", return_value=None):
            result = mound_handler.handle("/api/knowledge/mound/global", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 503

    def test_query_global_success(self, mound_handler, mock_http_handler):
        """Test querying global knowledge."""
        mock_item = MagicMock()
        mock_item.id = "global-1"
        mock_item.content = "Global fact"
        mock_item.importance = 0.9
        mock_item.to_dict = MagicMock(
            return_value={
                "id": "global-1",
                "content": "Global fact",
                "importance": 0.9,
            }
        )

        mock_mound = MagicMock()
        mock_mound.query_global_knowledge = AsyncMock(return_value=[mock_item])

        query_params = {"query": "test query"}

        with patch.object(mound_handler, "_get_mound", return_value=mock_mound):
            result = mound_handler.handle(
                "/api/knowledge/mound/global", query_params, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] == 1


class TestStoreVerifiedFact:
    """Test POST /api/knowledge/mound/global endpoint."""

    def test_store_fact_requires_admin(self, mound_handler):
        """Test storing verified fact requires admin."""
        handler = create_request_body(
            {
                "content": "Verified fact",
                "source": "scientific_consensus",
            }
        )

        with patch.object(mound_handler, "require_admin_or_error") as mock_admin:
            from aragora.server.handlers.base import error_response

            mock_admin.return_value = (None, error_response("Forbidden", 403))
            with patch.object(mound_handler, "require_auth_or_error") as mock_auth:
                mock_auth.return_value = (mock_user(is_admin=False), None)

                result = mound_handler.handle("/api/knowledge/mound/global", {}, handler)

        assert result is not None
        assert result.status_code == 403

    def test_store_fact_success(self, mound_handler):
        """Test storing verified fact as admin."""
        handler = create_request_body(
            {
                "content": "Water boils at 100C",
                "source": "scientific_consensus",
                "confidence": 0.99,
            }
        )

        mock_mound = MagicMock()
        mock_mound.store_verified_fact = AsyncMock(return_value="kn_123")

        with patch.object(mound_handler, "require_admin_or_error") as mock_admin:
            mock_admin.return_value = (mock_user(is_admin=True), None)
            with patch.object(mound_handler, "_get_mound", return_value=mock_mound):
                result = mound_handler.handle("/api/knowledge/mound/global", {}, handler)

        assert result is not None
        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["success"] is True
        assert body["node_id"] == "kn_123"


class TestPromoteToGlobal:
    """Test POST /api/knowledge/mound/global/promote endpoint."""

    def test_promote_requires_auth(self, mound_handler):
        """Test promoting requires authentication."""
        handler = create_request_body(
            {
                "item_id": "item-123",
                "workspace_id": "ws-1",
                "reason": "high_consensus",
            }
        )

        with patch.object(mound_handler, "require_auth_or_error") as mock_auth:
            from aragora.server.handlers.base import error_response

            mock_auth.return_value = (None, error_response("Unauthorized", 401))

            result = mound_handler.handle("/api/knowledge/mound/global/promote", {}, handler)

        assert result is not None
        assert result.status_code == 401

    def test_promote_success(self, mound_handler):
        """Test promoting to global successfully."""
        handler = create_request_body(
            {
                "item_id": "item-123",
                "workspace_id": "ws-1",
                "reason": "high_consensus",
            }
        )

        mock_mound = MagicMock()
        mock_mound.promote_to_global = AsyncMock(return_value="global-123")

        with patch.object(mound_handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (mock_user(), None)
            with patch.object(mound_handler, "_get_mound", return_value=mock_mound):
                result = mound_handler.handle("/api/knowledge/mound/global/promote", {}, handler)

        assert result is not None
        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["global_id"] == "global-123"


# =============================================================================
# Federation Endpoints Tests
# =============================================================================


class TestFederationCanHandle:
    """Test can_handle for federation endpoints."""

    def test_can_handle_federation_regions(self, mound_handler):
        """Test can_handle for federation regions endpoint."""
        assert mound_handler.can_handle("/api/knowledge/mound/federation/regions")

    def test_can_handle_federation_sync_push(self, mound_handler):
        """Test can_handle for federation sync/push endpoint."""
        assert mound_handler.can_handle("/api/knowledge/mound/federation/sync/push")

    def test_can_handle_federation_sync_pull(self, mound_handler):
        """Test can_handle for federation sync/pull endpoint."""
        assert mound_handler.can_handle("/api/knowledge/mound/federation/sync/pull")

    def test_can_handle_federation_status(self, mound_handler):
        """Test can_handle for federation status endpoint."""
        assert mound_handler.can_handle("/api/knowledge/mound/federation/status")


class TestRegisterRegion:
    """Test POST /api/knowledge/mound/federation/regions endpoint."""

    def test_register_requires_admin(self, mound_handler):
        """Test registering region requires admin."""
        handler = create_request_body(
            {
                "region_id": "region-1",
                "endpoint_url": "https://region1.example.com",
                "api_key": "secret-key",
            }
        )

        with patch.object(mound_handler, "require_admin_or_error") as mock_admin:
            from aragora.server.handlers.base import error_response

            mock_admin.return_value = (None, error_response("Forbidden", 403))

            result = mound_handler.handle("/api/knowledge/mound/federation/regions", {}, handler)

        assert result is not None
        assert result.status_code == 403

    def test_register_success(self, mound_handler):
        """Test registering region as admin."""
        handler = create_request_body(
            {
                "region_id": "region-1",
                "endpoint_url": "https://region1.example.com",
                "api_key": "secret-key",
                "mode": "bidirectional",
                "sync_scope": "summary",
            }
        )

        mock_region = MagicMock()
        mock_region.region_id = "region-1"
        mock_region.endpoint_url = "https://region1.example.com"
        mock_region.mode = MagicMock()
        mock_region.mode.value = "bidirectional"
        mock_region.sync_scope = MagicMock()
        mock_region.sync_scope.value = "summary"
        mock_region.enabled = True

        mock_mound = MagicMock()
        mock_mound.register_federated_region = AsyncMock(return_value=mock_region)

        with patch.object(mound_handler, "require_admin_or_error") as mock_admin:
            mock_admin.return_value = (mock_user(is_admin=True), None)
            with patch.object(mound_handler, "_get_mound", return_value=mock_mound):
                result = mound_handler.handle(
                    "/api/knowledge/mound/federation/regions", {}, handler
                )

        assert result is not None
        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["success"] is True
        assert body["region"]["region_id"] == "region-1"


class TestListRegions:
    """Test GET /api/knowledge/mound/federation/regions endpoint."""

    def test_list_regions_mound_unavailable(self, mound_handler, mock_http_handler):
        """Test returns 503 when mound not available."""
        with patch.object(mound_handler, "_get_mound", return_value=None):
            result = mound_handler.handle(
                "/api/knowledge/mound/federation/regions", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 503

    def test_list_regions_success(self, mound_handler, mock_http_handler):
        """Test listing registered regions."""
        mock_status = {
            "region-1": {
                "endpoint_url": "https://region1.example.com",
                "mode": "bidirectional",
                "enabled": True,
            },
            "region-2": {
                "endpoint_url": "https://region2.example.com",
                "mode": "pull",
                "enabled": False,
            },
        }

        mock_mound = MagicMock()
        mock_mound.get_federation_status = AsyncMock(return_value=mock_status)

        with patch.object(mound_handler, "_get_mound", return_value=mock_mound):
            result = mound_handler.handle(
                "/api/knowledge/mound/federation/regions", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] == 2


class TestSyncToRegion:
    """Test POST /api/knowledge/mound/federation/sync/push endpoint."""

    def test_sync_push_requires_auth(self, mound_handler):
        """Test sync push requires authentication."""
        handler = create_request_body({"region_id": "region-1"})

        with patch.object(mound_handler, "require_auth_or_error") as mock_auth:
            from aragora.server.handlers.base import error_response

            mock_auth.return_value = (None, error_response("Unauthorized", 401))

            result = mound_handler.handle("/api/knowledge/mound/federation/sync/push", {}, handler)

        assert result is not None
        assert result.status_code == 401

    def test_sync_push_success(self, mound_handler):
        """Test sync push successfully."""
        handler = create_request_body(
            {
                "region_id": "region-1",
                "workspace_id": "ws-1",
            }
        )

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.region_id = "region-1"
        mock_result.direction = "push"
        mock_result.nodes_synced = 10
        mock_result.nodes_skipped = 2
        mock_result.nodes_failed = 0
        mock_result.duration_ms = 150.5
        mock_result.error = None

        mock_mound = MagicMock()
        mock_mound.sync_to_region = AsyncMock(return_value=mock_result)

        with patch.object(mound_handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (mock_user(), None)
            with patch.object(mound_handler, "_get_mound", return_value=mock_mound):
                result = mound_handler.handle(
                    "/api/knowledge/mound/federation/sync/push", {}, handler
                )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True
        assert body["nodes_synced"] == 10


class TestSyncFromRegion:
    """Test POST /api/knowledge/mound/federation/sync/pull endpoint."""

    def test_sync_pull_success(self, mound_handler):
        """Test sync pull successfully."""
        handler = create_request_body(
            {
                "region_id": "region-1",
            }
        )

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.region_id = "region-1"
        mock_result.direction = "pull"
        mock_result.nodes_synced = 5
        mock_result.nodes_failed = 0
        mock_result.duration_ms = 200.0
        mock_result.error = None

        mock_mound = MagicMock()
        mock_mound.pull_from_region = AsyncMock(return_value=mock_result)

        with patch.object(mound_handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (mock_user(), None)
            with patch.object(mound_handler, "_get_mound", return_value=mock_mound):
                result = mound_handler.handle(
                    "/api/knowledge/mound/federation/sync/pull", {}, handler
                )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True
        assert body["direction"] == "pull"


class TestFederationStatus:
    """Test GET /api/knowledge/mound/federation/status endpoint."""

    def test_federation_status_success(self, mound_handler, mock_http_handler):
        """Test getting federation status."""
        mock_status = {
            "region-1": {
                "endpoint_url": "https://region1.example.com",
                "mode": "bidirectional",
                "enabled": True,
                "last_sync_at": "2024-01-15T12:00:00",
            },
        }

        mock_mound = MagicMock()
        mock_mound.get_federation_status = AsyncMock(return_value=mock_status)

        with patch.object(mound_handler, "_get_mound", return_value=mock_mound):
            result = mound_handler.handle(
                "/api/knowledge/mound/federation/status", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total_regions"] == 1
        assert body["enabled_regions"] == 1
