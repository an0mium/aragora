"""
Integration tests for Knowledge Mound visibility, sharing, global knowledge, and federation.

Tests the complete API flow for:
1. Visibility level management (private, workspace, organization, public, system)
2. Access grant operations (grant, revoke, list)
3. Cross-workspace sharing
4. Global knowledge (verified facts)
5. Federation (multi-region sync)

These tests use mocked KnowledgeMound to test handler logic,
and can optionally run against a live server.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from io import BytesIO
from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mark all tests as integration tests
pytestmark = [pytest.mark.integration, pytest.mark.knowledge]


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound with all methods."""
    mound = MagicMock()

    # Visibility methods
    mound.set_visibility = AsyncMock(return_value=True)
    mound.get_node = AsyncMock(return_value=MagicMock(
        id="test_node_1",
        content="Test content",
        metadata={
            "visibility": "workspace",
            "visibility_set_by": "user_1",
            "is_discoverable": True,
        },
    ))
    mound.grant_access = AsyncMock(return_value=MagicMock(
        id="grant_1",
        item_id="test_node_1",
        grantee_type="user",
        grantee_id="user_2",
        permissions=["read"],
        granted_by="user_1",
    ))
    mound.revoke_access = AsyncMock(return_value=True)
    mound.get_access_grants = AsyncMock(return_value=[])

    # Sharing methods
    mound.share_with_workspace = AsyncMock(return_value=MagicMock(
        id="share_1",
        item_id="test_node_1",
        grantee_id="workspace_2",
    ))
    mound.share_with_user = AsyncMock(return_value=MagicMock(
        id="share_2",
        item_id="test_node_1",
        grantee_id="user_3",
    ))
    mound.get_shared_with_me = AsyncMock(return_value=[])
    mound.revoke_share = AsyncMock(return_value=True)
    mound.get_share_grants = AsyncMock(return_value=[])
    mound.update_share_permissions = AsyncMock(return_value=MagicMock(
        item_id="test_node_1",
        grantee_id="user_2",
        permissions=["read", "write"],
    ))

    # Global knowledge methods
    mound.store_verified_fact = AsyncMock(return_value="global_fact_1")
    mound.query_global_knowledge = AsyncMock(return_value=[])
    mound.get_system_facts = AsyncMock(return_value=[])
    mound.promote_to_global = AsyncMock(return_value="promoted_fact_1")
    mound.get_system_workspace_id = MagicMock(return_value="__system__")

    # Federation methods
    mound.register_federated_region = AsyncMock(return_value=MagicMock(
        region_id="us-west-2",
        endpoint_url="https://us-west-2.example.com/api",
        mode=MagicMock(value="bidirectional"),
        sync_scope=MagicMock(value="summary"),
        enabled=True,
    ))
    mound.unregister_federated_region = AsyncMock(return_value=True)
    mound.sync_to_region = AsyncMock(return_value=MagicMock(
        success=True,
        region_id="us-west-2",
        direction="push",
        nodes_synced=10,
        nodes_skipped=2,
        nodes_failed=0,
        duration_ms=150,
        error=None,
    ))
    mound.pull_from_region = AsyncMock(return_value=MagicMock(
        success=True,
        region_id="us-west-2",
        direction="pull",
        nodes_synced=5,
        nodes_failed=0,
        duration_ms=120,
        error=None,
    ))
    mound.sync_all_regions = AsyncMock(return_value=[])
    mound.get_federation_status = AsyncMock(return_value={})

    return mound


@pytest.fixture
def mock_user():
    """Create a mock authenticated user."""
    user = MagicMock()
    user.id = "test_user_1"
    user.user_id = "test_user_1"
    user.permissions = ["read", "write"]
    return user


@pytest.fixture
def mock_admin_user():
    """Create a mock admin user."""
    user = MagicMock()
    user.id = "admin_user_1"
    user.user_id = "admin_user_1"
    user.permissions = ["admin", "global_write"]
    return user


@pytest.fixture
def mock_handler(mock_user):
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.headers = {"Content-Length": "0"}
    handler.rfile = BytesIO(b"")
    return handler


def make_handler_with_body(data: dict, user: Any = None) -> MagicMock:
    """Create a mock handler with JSON body."""
    body = json.dumps(data).encode("utf-8")
    handler = MagicMock()
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = BytesIO(body)
    return handler


# =============================================================================
# Visibility Tests
# =============================================================================


class TestVisibilityOperations:
    """Test visibility level management."""

    @pytest.mark.asyncio
    async def test_set_visibility_to_private(self, mock_mound, mock_user):
        """Test setting an item to private visibility."""
        from aragora.server.handlers.knowledge_base.mound.visibility import (
            VisibilityOperationsMixin,
        )
        from aragora.knowledge.mound.types import VisibilityLevel

        class TestHandler(VisibilityOperationsMixin):
            def _get_mound(self):
                return mock_mound

            def require_auth_or_error(self, handler):
                return mock_user, None

        handler_instance = TestHandler()
        http_handler = make_handler_with_body({
            "visibility": "private",
            "is_discoverable": False,
        })

        result = handler_instance._handle_set_visibility("node_1", http_handler)

        assert result.status_code == 200
        response = json.loads(result.body.decode("utf-8"))
        assert response["success"] is True
        assert response["visibility"] == "private"
        assert response["is_discoverable"] is False

    @pytest.mark.asyncio
    async def test_set_visibility_invalid_level(self, mock_mound, mock_user):
        """Test setting an invalid visibility level."""
        from aragora.server.handlers.knowledge_base.mound.visibility import (
            VisibilityOperationsMixin,
        )

        class TestHandler(VisibilityOperationsMixin):
            def _get_mound(self):
                return mock_mound

            def require_auth_or_error(self, handler):
                return mock_user, None

        handler_instance = TestHandler()
        http_handler = make_handler_with_body({
            "visibility": "invalid_level",
        })

        result = handler_instance._handle_set_visibility("node_1", http_handler)

        assert result.status_code == 400
        response = json.loads(result.body.decode("utf-8"))
        assert "Invalid visibility level" in response["error"]

    @pytest.mark.asyncio
    async def test_get_visibility(self, mock_mound):
        """Test getting item visibility."""
        from aragora.server.handlers.knowledge_base.mound.visibility import (
            VisibilityOperationsMixin,
        )

        class TestHandler(VisibilityOperationsMixin):
            def _get_mound(self):
                return mock_mound

        handler_instance = TestHandler()
        result = handler_instance._handle_get_visibility("node_1")

        assert result.status_code == 200
        response = json.loads(result.body.decode("utf-8"))
        assert response["item_id"] == "node_1"
        assert response["visibility"] == "workspace"

    @pytest.mark.asyncio
    async def test_grant_access(self, mock_mound, mock_user):
        """Test granting access to an item."""
        from aragora.server.handlers.knowledge_base.mound.visibility import (
            VisibilityOperationsMixin,
        )

        class TestHandler(VisibilityOperationsMixin):
            def _get_mound(self):
                return mock_mound

            def require_auth_or_error(self, handler):
                return mock_user, None

        handler_instance = TestHandler()
        http_handler = make_handler_with_body({
            "grantee_type": "user",
            "grantee_id": "user_2",
            "permissions": ["read", "write"],
        })

        result = handler_instance._handle_grant_access("node_1", http_handler)

        assert result.status_code == 201
        response = json.loads(result.body.decode("utf-8"))
        assert response["success"] is True
        mock_mound.grant_access.assert_called_once()

    @pytest.mark.asyncio
    async def test_grant_access_with_expiry(self, mock_mound, mock_user):
        """Test granting access with expiration."""
        from aragora.server.handlers.knowledge_base.mound.visibility import (
            VisibilityOperationsMixin,
        )

        class TestHandler(VisibilityOperationsMixin):
            def _get_mound(self):
                return mock_mound

            def require_auth_or_error(self, handler):
                return mock_user, None

        handler_instance = TestHandler()
        expires_at = (datetime.utcnow() + timedelta(days=7)).isoformat() + "Z"
        http_handler = make_handler_with_body({
            "grantee_type": "workspace",
            "grantee_id": "workspace_2",
            "permissions": ["read"],
            "expires_at": expires_at,
        })

        result = handler_instance._handle_grant_access("node_1", http_handler)

        assert result.status_code == 201
        call_args = mock_mound.grant_access.call_args
        assert call_args.kwargs["expires_at"] is not None

    @pytest.mark.asyncio
    async def test_revoke_access(self, mock_mound, mock_user):
        """Test revoking access."""
        from aragora.server.handlers.knowledge_base.mound.visibility import (
            VisibilityOperationsMixin,
        )

        class TestHandler(VisibilityOperationsMixin):
            def _get_mound(self):
                return mock_mound

            def require_auth_or_error(self, handler):
                return mock_user, None

        handler_instance = TestHandler()
        http_handler = make_handler_with_body({
            "grantee_id": "user_2",
        })

        result = handler_instance._handle_revoke_access("node_1", http_handler)

        assert result.status_code == 200
        response = json.loads(result.body.decode("utf-8"))
        assert response["success"] is True
        mock_mound.revoke_access.assert_called_once()


# =============================================================================
# Sharing Tests
# =============================================================================


class TestSharingOperations:
    """Test cross-workspace sharing."""

    @pytest.mark.asyncio
    async def test_share_with_workspace(self, mock_mound, mock_user):
        """Test sharing an item with another workspace."""
        from aragora.server.handlers.knowledge_base.mound.sharing import (
            SharingOperationsMixin,
        )

        class TestHandler(SharingOperationsMixin):
            def _get_mound(self):
                return mock_mound

            def require_auth_or_error(self, handler):
                return mock_user, None

        handler_instance = TestHandler()
        http_handler = make_handler_with_body({
            "item_id": "node_1",
            "target_type": "workspace",
            "target_id": "workspace_2",
            "permissions": ["read"],
            "from_workspace_id": "workspace_1",
        })

        result = handler_instance._handle_share_item(http_handler)

        assert result.status_code == 201
        response = json.loads(result.body.decode("utf-8"))
        assert response["success"] is True
        assert response["share"]["target_type"] == "workspace"
        mock_mound.share_with_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_share_with_user(self, mock_mound, mock_user):
        """Test sharing an item with a specific user."""
        from aragora.server.handlers.knowledge_base.mound.sharing import (
            SharingOperationsMixin,
        )

        class TestHandler(SharingOperationsMixin):
            def _get_mound(self):
                return mock_mound

            def require_auth_or_error(self, handler):
                return mock_user, None

        handler_instance = TestHandler()
        http_handler = make_handler_with_body({
            "item_id": "node_1",
            "target_type": "user",
            "target_id": "user_3",
            "permissions": ["read", "write"],
        })

        result = handler_instance._handle_share_item(http_handler)

        assert result.status_code == 201
        response = json.loads(result.body.decode("utf-8"))
        assert response["share"]["target_type"] == "user"
        mock_mound.share_with_user.assert_called_once()

    @pytest.mark.asyncio
    async def test_share_invalid_target_type(self, mock_mound, mock_user):
        """Test sharing with invalid target type."""
        from aragora.server.handlers.knowledge_base.mound.sharing import (
            SharingOperationsMixin,
        )

        class TestHandler(SharingOperationsMixin):
            def _get_mound(self):
                return mock_mound

            def require_auth_or_error(self, handler):
                return mock_user, None

        handler_instance = TestHandler()
        http_handler = make_handler_with_body({
            "item_id": "node_1",
            "target_type": "invalid",
            "target_id": "target_1",
        })

        result = handler_instance._handle_share_item(http_handler)

        assert result.status_code == 400
        response = json.loads(result.body.decode("utf-8"))
        assert "target_type" in response["error"]

    @pytest.mark.asyncio
    async def test_get_shared_with_me(self, mock_mound, mock_user):
        """Test getting items shared with current user/workspace."""
        from aragora.server.handlers.knowledge_base.mound.sharing import (
            SharingOperationsMixin,
        )

        # Setup mock to return shared items
        mock_mound.get_shared_with_me = AsyncMock(return_value=[
            MagicMock(
                id="shared_1",
                content="Shared content 1",
                to_dict=lambda: {"id": "shared_1", "content": "Shared content 1"},
            ),
            MagicMock(
                id="shared_2",
                content="Shared content 2",
                to_dict=lambda: {"id": "shared_2", "content": "Shared content 2"},
            ),
        ])

        class TestHandler(SharingOperationsMixin):
            def _get_mound(self):
                return mock_mound

            def require_auth_or_error(self, handler):
                return mock_user, None

        handler_instance = TestHandler()
        http_handler = MagicMock()

        result = handler_instance._handle_shared_with_me(
            {"workspace_id": ["workspace_1"], "limit": ["50"]},
            http_handler,
        )

        assert result.status_code == 200
        response = json.loads(result.body.decode("utf-8"))
        assert response["count"] == 2
        assert len(response["items"]) == 2

    @pytest.mark.asyncio
    async def test_revoke_share(self, mock_mound, mock_user):
        """Test revoking a share."""
        from aragora.server.handlers.knowledge_base.mound.sharing import (
            SharingOperationsMixin,
        )

        class TestHandler(SharingOperationsMixin):
            def _get_mound(self):
                return mock_mound

            def require_auth_or_error(self, handler):
                return mock_user, None

        handler_instance = TestHandler()
        http_handler = make_handler_with_body({
            "item_id": "node_1",
            "grantee_id": "workspace_2",
        })

        result = handler_instance._handle_revoke_share(http_handler)

        assert result.status_code == 200
        response = json.loads(result.body.decode("utf-8"))
        assert response["success"] is True
        mock_mound.revoke_share.assert_called_once()


# =============================================================================
# Global Knowledge Tests
# =============================================================================


class TestGlobalKnowledgeOperations:
    """Test global/system knowledge operations."""

    @pytest.mark.asyncio
    async def test_store_verified_fact_as_admin(self, mock_mound, mock_admin_user):
        """Test storing a verified fact as admin."""
        from aragora.server.handlers.knowledge_base.mound.global_knowledge import (
            GlobalKnowledgeOperationsMixin,
        )

        class TestHandler(GlobalKnowledgeOperationsMixin):
            def _get_mound(self):
                return mock_mound

            def require_auth_or_error(self, handler):
                return mock_admin_user, None

            def require_admin_or_error(self, handler):
                return mock_admin_user, None

        handler_instance = TestHandler()
        http_handler = make_handler_with_body({
            "content": "The Earth orbits the Sun",
            "source": "scientific_consensus",
            "confidence": 0.99,
            "topics": ["astronomy", "science"],
        })

        result = handler_instance._handle_store_verified_fact(http_handler)

        assert result.status_code == 201
        response = json.loads(result.body.decode("utf-8"))
        assert response["success"] is True
        assert response["node_id"] == "global_fact_1"
        mock_mound.store_verified_fact.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_verified_fact_requires_permission(self, mock_mound, mock_user):
        """Test that storing verified facts requires admin/global_write permission."""
        from aragora.server.handlers.knowledge_base.mound.global_knowledge import (
            GlobalKnowledgeOperationsMixin,
        )
        from aragora.server.handlers.base import error_response

        class TestHandler(GlobalKnowledgeOperationsMixin):
            def _get_mound(self):
                return mock_mound

            def require_auth_or_error(self, handler):
                return mock_user, None

            def require_admin_or_error(self, handler):
                return None, error_response("Admin required", 403)

        handler_instance = TestHandler()
        http_handler = make_handler_with_body({
            "content": "Test fact",
            "source": "test",
        })

        result = handler_instance._handle_store_verified_fact(http_handler)

        # Should fail because user doesn't have admin or global_write permission
        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_query_global_knowledge(self, mock_mound):
        """Test querying global knowledge."""
        from aragora.server.handlers.knowledge_base.mound.global_knowledge import (
            GlobalKnowledgeOperationsMixin,
        )

        mock_mound.query_global_knowledge = AsyncMock(return_value=[
            MagicMock(
                id="fact_1",
                content="Important global fact",
                importance=0.9,
                to_dict=lambda: {"id": "fact_1", "content": "Important global fact"},
            ),
        ])

        class TestHandler(GlobalKnowledgeOperationsMixin):
            def _get_mound(self):
                return mock_mound

        handler_instance = TestHandler()
        result = handler_instance._handle_query_global(
            {"query": ["climate change"], "limit": ["20"]}
        )

        assert result.status_code == 200
        response = json.loads(result.body.decode("utf-8"))
        assert response["count"] == 1
        assert "climate change" in response["query"]

    @pytest.mark.asyncio
    async def test_promote_to_global(self, mock_mound, mock_user):
        """Test promoting a workspace item to global."""
        from aragora.server.handlers.knowledge_base.mound.global_knowledge import (
            GlobalKnowledgeOperationsMixin,
        )

        class TestHandler(GlobalKnowledgeOperationsMixin):
            def _get_mound(self):
                return mock_mound

            def require_auth_or_error(self, handler):
                return mock_user, None

        handler_instance = TestHandler()
        http_handler = make_handler_with_body({
            "item_id": "node_1",
            "workspace_id": "workspace_1",
            "reason": "high_consensus",
        })

        result = handler_instance._handle_promote_to_global(http_handler)

        assert result.status_code == 201
        response = json.loads(result.body.decode("utf-8"))
        assert response["success"] is True
        assert response["global_id"] == "promoted_fact_1"
        mock_mound.promote_to_global.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_system_workspace_id(self, mock_mound):
        """Test getting system workspace ID."""
        from aragora.server.handlers.knowledge_base.mound.global_knowledge import (
            GlobalKnowledgeOperationsMixin,
        )

        class TestHandler(GlobalKnowledgeOperationsMixin):
            def _get_mound(self):
                return mock_mound

        handler_instance = TestHandler()
        result = handler_instance._handle_get_system_workspace_id()

        assert result.status_code == 200
        response = json.loads(result.body.decode("utf-8"))
        assert response["system_workspace_id"] == "__system__"


# =============================================================================
# Federation Tests
# =============================================================================


class TestFederationOperations:
    """Test multi-region federation operations."""

    @pytest.mark.asyncio
    async def test_register_federated_region(self, mock_mound, mock_admin_user):
        """Test registering a new federated region."""
        from aragora.server.handlers.knowledge_base.mound.federation import (
            FederationOperationsMixin,
        )

        class TestHandler(FederationOperationsMixin):
            def _get_mound(self):
                return mock_mound

            def require_auth_or_error(self, handler):
                return mock_admin_user, None

            def require_admin_or_error(self, handler):
                return mock_admin_user, None

        handler_instance = TestHandler()
        http_handler = make_handler_with_body({
            "region_id": "us-west-2",
            "endpoint_url": "https://us-west-2.example.com/api",
            "api_key": "secret_api_key",
            "mode": "bidirectional",
            "sync_scope": "summary",
        })

        result = handler_instance._handle_register_region(http_handler)

        assert result.status_code == 201
        response = json.loads(result.body.decode("utf-8"))
        assert response["success"] is True
        assert response["region"]["region_id"] == "us-west-2"
        mock_mound.register_federated_region.assert_called_once()

    @pytest.mark.asyncio
    async def test_unregister_region(self, mock_mound, mock_admin_user):
        """Test unregistering a federated region."""
        from aragora.server.handlers.knowledge_base.mound.federation import (
            FederationOperationsMixin,
        )

        class TestHandler(FederationOperationsMixin):
            def _get_mound(self):
                return mock_mound

            def require_auth_or_error(self, handler):
                return mock_admin_user, None

            def require_admin_or_error(self, handler):
                return mock_admin_user, None

        handler_instance = TestHandler()
        http_handler = MagicMock()

        result = handler_instance._handle_unregister_region("us-west-2", http_handler)

        assert result.status_code == 200
        response = json.loads(result.body.decode("utf-8"))
        assert response["success"] is True
        mock_mound.unregister_federated_region.assert_called_once_with("us-west-2")

    @pytest.mark.asyncio
    async def test_sync_to_region(self, mock_mound, mock_user):
        """Test syncing knowledge to a federated region."""
        from aragora.server.handlers.knowledge_base.mound.federation import (
            FederationOperationsMixin,
        )

        class TestHandler(FederationOperationsMixin):
            def _get_mound(self):
                return mock_mound

            def require_auth_or_error(self, handler):
                return mock_user, None

        handler_instance = TestHandler()
        http_handler = make_handler_with_body({
            "region_id": "us-west-2",
            "workspace_id": "workspace_1",
        })

        result = handler_instance._handle_sync_to_region(http_handler)

        assert result.status_code == 200
        response = json.loads(result.body.decode("utf-8"))
        assert response["success"] is True
        assert response["nodes_synced"] == 10
        assert response["direction"] == "push"

    @pytest.mark.asyncio
    async def test_pull_from_region(self, mock_mound, mock_user):
        """Test pulling knowledge from a federated region."""
        from aragora.server.handlers.knowledge_base.mound.federation import (
            FederationOperationsMixin,
        )

        class TestHandler(FederationOperationsMixin):
            def _get_mound(self):
                return mock_mound

            def require_auth_or_error(self, handler):
                return mock_user, None

        handler_instance = TestHandler()
        http_handler = make_handler_with_body({
            "region_id": "us-west-2",
            "workspace_id": "workspace_1",
        })

        result = handler_instance._handle_pull_from_region(http_handler)

        assert result.status_code == 200
        response = json.loads(result.body.decode("utf-8"))
        assert response["success"] is True
        assert response["nodes_synced"] == 5
        assert response["direction"] == "pull"

    @pytest.mark.asyncio
    async def test_sync_with_since_filter(self, mock_mound, mock_user):
        """Test syncing with a since timestamp filter."""
        from aragora.server.handlers.knowledge_base.mound.federation import (
            FederationOperationsMixin,
        )

        class TestHandler(FederationOperationsMixin):
            def _get_mound(self):
                return mock_mound

            def require_auth_or_error(self, handler):
                return mock_user, None

        handler_instance = TestHandler()
        since_time = (datetime.utcnow() - timedelta(hours=1)).isoformat() + "Z"
        http_handler = make_handler_with_body({
            "region_id": "us-west-2",
            "since": since_time,
        })

        result = handler_instance._handle_sync_to_region(http_handler)

        assert result.status_code == 200
        call_args = mock_mound.sync_to_region.call_args
        assert call_args.kwargs["since"] is not None

    @pytest.mark.asyncio
    async def test_get_federation_status(self, mock_mound):
        """Test getting federation status."""
        from aragora.server.handlers.knowledge_base.mound.federation import (
            FederationOperationsMixin,
        )

        mock_mound.get_federation_status = AsyncMock(return_value={
            "us-west-2": {
                "enabled": True,
                "mode": "bidirectional",
                "last_sync": datetime.utcnow().isoformat(),
                "health": "healthy",
            },
            "eu-west-1": {
                "enabled": False,
                "mode": "pull",
                "last_sync": None,
                "health": "unknown",
            },
        })

        class TestHandler(FederationOperationsMixin):
            def _get_mound(self):
                return mock_mound

        handler_instance = TestHandler()
        result = handler_instance._handle_get_federation_status({})

        assert result.status_code == 200
        response = json.loads(result.body.decode("utf-8"))
        assert response["total_regions"] == 2
        assert response["enabled_regions"] == 1

    @pytest.mark.asyncio
    async def test_list_federated_regions(self, mock_mound):
        """Test listing all federated regions."""
        from aragora.server.handlers.knowledge_base.mound.federation import (
            FederationOperationsMixin,
        )

        mock_mound.get_federation_status = AsyncMock(return_value={
            "us-west-2": {"enabled": True, "mode": "bidirectional"},
            "eu-west-1": {"enabled": False, "mode": "pull"},
        })

        class TestHandler(FederationOperationsMixin):
            def _get_mound(self):
                return mock_mound

        handler_instance = TestHandler()
        result = handler_instance._handle_list_regions({})

        assert result.status_code == 200
        response = json.loads(result.body.decode("utf-8"))
        assert response["count"] == 2
        assert len(response["regions"]) == 2


# =============================================================================
# Integration Tests with Full Stack (Optional)
# =============================================================================


class TestFullStackIntegration:
    """
    Full-stack integration tests that run against a live server.

    These tests are skipped by default and require:
    - A running server (set ARAGORA_TEST_SERVER_URL)
    - Valid authentication (set ARAGORA_TEST_AUTH_TOKEN)
    """

    @pytest.fixture
    def server_url(self):
        """Get test server URL from environment."""
        import os
        url = os.environ.get("ARAGORA_TEST_SERVER_URL")
        if not url:
            pytest.skip("ARAGORA_TEST_SERVER_URL not set")
        return url

    @pytest.fixture
    def auth_token(self):
        """Get test auth token from environment."""
        import os
        token = os.environ.get("ARAGORA_TEST_AUTH_TOKEN")
        if not token:
            pytest.skip("ARAGORA_TEST_AUTH_TOKEN not set")
        return token

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires running server")
    async def test_visibility_workflow_e2e(self, server_url, auth_token):
        """Test complete visibility workflow against live server."""
        import httpx

        async with httpx.AsyncClient(base_url=server_url) as client:
            headers = {"Authorization": f"Bearer {auth_token}"}

            # Create a node first (assuming there's an endpoint)
            # Then test visibility operations

            # 1. Get initial visibility
            response = await client.get(
                "/api/knowledge/mound/nodes/test_node/visibility",
                headers=headers,
            )
            assert response.status_code in (200, 404)

            # 2. Set visibility
            response = await client.put(
                "/api/knowledge/mound/nodes/test_node/visibility",
                headers=headers,
                json={"visibility": "private", "is_discoverable": False},
            )
            # Might be 200 (success) or 404 (node doesn't exist)
            assert response.status_code in (200, 404)

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires running server")
    async def test_sharing_workflow_e2e(self, server_url, auth_token):
        """Test complete sharing workflow against live server."""
        import httpx

        async with httpx.AsyncClient(base_url=server_url) as client:
            headers = {"Authorization": f"Bearer {auth_token}"}

            # 1. Share an item
            response = await client.post(
                "/api/knowledge/mound/share",
                headers=headers,
                json={
                    "item_id": "test_node",
                    "target_type": "workspace",
                    "target_id": "workspace_2",
                },
            )
            # Might succeed or fail depending on node existence
            assert response.status_code in (201, 404, 500)

            # 2. Get shared with me
            response = await client.get(
                "/api/knowledge/mound/shared-with-me",
                headers=headers,
            )
            assert response.status_code == 200

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires running server")
    async def test_global_knowledge_workflow_e2e(self, server_url, auth_token):
        """Test global knowledge workflow against live server."""
        import httpx

        async with httpx.AsyncClient(base_url=server_url) as client:
            headers = {"Authorization": f"Bearer {auth_token}"}

            # 1. Query global knowledge
            response = await client.get(
                "/api/knowledge/mound/global",
                headers=headers,
                params={"query": "test", "limit": 10},
            )
            assert response.status_code == 200

            # 2. Get system workspace ID
            response = await client.get(
                "/api/knowledge/mound/global/workspace-id",
                headers=headers,
            )
            assert response.status_code == 200
            data = response.json()
            assert "system_workspace_id" in data

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires running server")
    async def test_federation_workflow_e2e(self, server_url, auth_token):
        """Test federation workflow against live server."""
        import httpx

        async with httpx.AsyncClient(base_url=server_url) as client:
            headers = {"Authorization": f"Bearer {auth_token}"}

            # 1. Get federation status
            response = await client.get(
                "/api/knowledge/mound/federation/status",
                headers=headers,
            )
            assert response.status_code == 200
            data = response.json()
            assert "total_regions" in data

            # 2. List federated regions
            response = await client.get(
                "/api/knowledge/mound/federation/regions",
                headers=headers,
            )
            assert response.status_code == 200
            data = response.json()
            assert "regions" in data
