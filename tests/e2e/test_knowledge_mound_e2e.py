"""
E2E tests for Knowledge Mound features.

Tests the complete flow for:
1. Visibility API - setting/getting item visibility levels
2. Sharing API - cross-workspace knowledge sharing
3. Federation API - multi-region sync
4. Global Knowledge API - system-wide verified facts
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio

# Mark all tests as e2e tests
pytestmark = [pytest.mark.e2e, pytest.mark.knowledge]


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_meta_store():
    """Create a mock metadata store with access grant support."""
    store = MagicMock()
    store._grants: Dict[str, List[Any]] = {}  # item_id -> list of grants
    store._grantee_grants: Dict[str, List[Any]] = {}  # grantee_id -> list of grants

    async def save_access_grant_async(grant):
        item_id = grant.item_id
        if item_id not in store._grants:
            store._grants[item_id] = []
        # Replace existing grant for same grantee
        store._grants[item_id] = [
            g for g in store._grants[item_id] if g.grantee_id != grant.grantee_id
        ]
        store._grants[item_id].append(grant)

        # Also index by grantee
        grantee_id = grant.grantee_id
        if grantee_id not in store._grantee_grants:
            store._grantee_grants[grantee_id] = []
        store._grantee_grants[grantee_id] = [
            g for g in store._grantee_grants[grantee_id] if g.item_id != item_id
        ]
        store._grantee_grants[grantee_id].append(grant)

    async def get_access_grants_async(item_id):
        return store._grants.get(item_id, [])

    async def get_grants_for_grantee_async(grantee_id, grantee_type):
        return [
            g for g in store._grantee_grants.get(grantee_id, []) if g.grantee_type == grantee_type
        ]

    async def delete_access_grant_async(item_id, grantee_id):
        if item_id in store._grants:
            original_len = len(store._grants[item_id])
            store._grants[item_id] = [
                g for g in store._grants[item_id] if g.grantee_id != grantee_id
            ]
            if len(store._grants[item_id]) < original_len:
                # Also remove from grantee index
                if grantee_id in store._grantee_grants:
                    store._grantee_grants[grantee_id] = [
                        g for g in store._grantee_grants[grantee_id] if g.item_id != item_id
                    ]
                return True
        return False

    store.save_access_grant_async = save_access_grant_async
    store.get_access_grants_async = get_access_grants_async
    store.get_grants_for_grantee_async = get_grants_for_grantee_async
    store.delete_access_grant_async = delete_access_grant_async

    return store


@pytest.fixture
def sample_knowledge_item():
    """Create a sample knowledge item."""
    from aragora.knowledge.mound.types import ConfidenceLevel, KnowledgeItem, KnowledgeSource

    return KnowledgeItem(
        id=f"item_{uuid4().hex[:8]}",
        content="Machine learning models require training data.",
        source=KnowledgeSource.FACT,
        source_id=f"source_{uuid4().hex[:8]}",
        confidence=ConfidenceLevel.HIGH,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        importance=0.8,
        metadata={
            "topics": ["ml", "data"],
            "visibility": "workspace",
        },
    )


@pytest.fixture
def mock_knowledge_mound(mock_meta_store, sample_knowledge_item):
    """Create a mock Knowledge Mound with necessary methods."""
    from aragora.knowledge.mound.types import IngestionResult, QueryResult

    mound = MagicMock()
    mound._meta_store = mock_meta_store
    mound._cache = None
    mound._initialized = True
    mound.workspace_id = "test_workspace"
    mound.config = MagicMock()

    # Store items in memory
    mound._items: Dict[str, Any] = {sample_knowledge_item.id: sample_knowledge_item}

    def _ensure_initialized():
        pass

    async def get(node_id: str, workspace_id: Optional[str] = None):
        return mound._items.get(node_id)

    async def store(request):
        item_id = f"item_{uuid4().hex[:8]}"
        return IngestionResult(node_id=item_id, success=True)

    async def query(query: str, workspace_id: Optional[str] = None, limit: int = 20, **kwargs):
        items = list(mound._items.values())[:limit]
        return QueryResult(items=items, total_count=len(items), query=query)

    mound._ensure_initialized = _ensure_initialized
    mound.get = AsyncMock(side_effect=get)
    mound.store = AsyncMock(side_effect=store)
    mound.query = AsyncMock(side_effect=query)

    return mound


# =============================================================================
# Visibility API Tests
# =============================================================================


class TestVisibilityAPI:
    """Test suite for visibility level management."""

    @pytest.mark.asyncio
    async def test_visibility_levels_enum(self):
        """Test that all visibility levels are defined correctly."""
        from aragora.knowledge.mound.types import VisibilityLevel

        assert VisibilityLevel.PRIVATE.value == "private"
        assert VisibilityLevel.WORKSPACE.value == "workspace"
        assert VisibilityLevel.ORGANIZATION.value == "organization"
        assert VisibilityLevel.PUBLIC.value == "public"
        assert VisibilityLevel.SYSTEM.value == "system"

    @pytest.mark.asyncio
    async def test_access_grant_creation(self):
        """Test creating an access grant."""
        from aragora.knowledge.mound.types import AccessGrant, AccessGrantType

        grant = AccessGrant(
            id="grant_001",
            item_id="item_123",
            grantee_type=AccessGrantType.USER,
            grantee_id="user_456",
            permissions=["read", "write"],
            granted_by="admin",
        )

        assert grant.id == "grant_001"
        assert grant.item_id == "item_123"
        assert grant.grantee_type == AccessGrantType.USER
        assert grant.grantee_id == "user_456"
        assert "read" in grant.permissions
        assert "write" in grant.permissions
        assert not grant.is_expired()

    @pytest.mark.asyncio
    async def test_access_grant_expiration(self):
        """Test that expired grants are detected."""
        from aragora.knowledge.mound.types import AccessGrant, AccessGrantType

        # Create expired grant
        expired_grant = AccessGrant(
            id="grant_expired",
            item_id="item_123",
            grantee_type=AccessGrantType.USER,
            grantee_id="user_456",
            expires_at=datetime.now() - timedelta(hours=1),
        )
        assert expired_grant.is_expired()

        # Create valid grant
        valid_grant = AccessGrant(
            id="grant_valid",
            item_id="item_123",
            grantee_type=AccessGrantType.USER,
            grantee_id="user_456",
            expires_at=datetime.now() + timedelta(hours=1),
        )
        assert not valid_grant.is_expired()

    @pytest.mark.asyncio
    async def test_access_grant_permissions(self):
        """Test permission checking on access grants."""
        from aragora.knowledge.mound.types import AccessGrant, AccessGrantType

        grant = AccessGrant(
            id="grant_001",
            item_id="item_123",
            grantee_type=AccessGrantType.USER,
            grantee_id="user_456",
            permissions=["read"],
        )

        assert grant.has_permission("read")
        assert not grant.has_permission("write")

        # Admin permission grants all
        admin_grant = AccessGrant(
            id="grant_admin",
            item_id="item_123",
            grantee_type=AccessGrantType.USER,
            grantee_id="admin_user",
            permissions=["admin"],
        )
        assert admin_grant.has_permission("read")
        assert admin_grant.has_permission("write")
        assert admin_grant.has_permission("delete")

    @pytest.mark.asyncio
    async def test_access_grant_serialization(self):
        """Test access grant serialization."""
        from aragora.knowledge.mound.types import AccessGrant, AccessGrantType

        grant = AccessGrant(
            id="grant_001",
            item_id="item_123",
            grantee_type=AccessGrantType.WORKSPACE,
            grantee_id="workspace_789",
            permissions=["read"],
            granted_by="admin",
        )

        data = grant.to_dict()

        assert data["id"] == "grant_001"
        assert data["grantee_type"] == "workspace"
        assert data["grantee_id"] == "workspace_789"
        assert "granted_at" in data


# =============================================================================
# Sharing API Tests
# =============================================================================


class TestSharingAPI:
    """Test suite for cross-workspace sharing."""

    @pytest.mark.asyncio
    async def test_share_with_workspace(self, mock_knowledge_mound, sample_knowledge_item):
        """Test sharing an item with another workspace."""
        from aragora.knowledge.mound.ops.sharing import KnowledgeSharingMixin
        from aragora.knowledge.mound.types import AccessGrantType

        # Create a mound instance with the mixin
        class TestMound(KnowledgeSharingMixin):
            pass

        mound = TestMound()
        mound._meta_store = mock_knowledge_mound._meta_store
        mound._cache = None
        mound._initialized = True
        mound._ensure_initialized = lambda: None
        mound.get = mock_knowledge_mound.get

        grant = await mound.share_with_workspace(
            item_id=sample_knowledge_item.id,
            from_workspace_id="workspace_a",
            to_workspace_id="workspace_b",
            shared_by="user_123",
            permissions=["read"],
        )

        assert grant.grantee_type == AccessGrantType.WORKSPACE
        assert grant.grantee_id == "workspace_b"
        assert "read" in grant.permissions

    @pytest.mark.asyncio
    async def test_share_with_same_workspace_fails(
        self, mock_knowledge_mound, sample_knowledge_item
    ):
        """Test that sharing with the same workspace is rejected."""
        from aragora.knowledge.mound.ops.sharing import KnowledgeSharingMixin

        class TestMound(KnowledgeSharingMixin):
            pass

        mound = TestMound()
        mound._meta_store = mock_knowledge_mound._meta_store
        mound._cache = None
        mound._initialized = True
        mound._ensure_initialized = lambda: None
        mound.get = mock_knowledge_mound.get

        with pytest.raises(ValueError, match="same workspace"):
            await mound.share_with_workspace(
                item_id=sample_knowledge_item.id,
                from_workspace_id="workspace_a",
                to_workspace_id="workspace_a",
                shared_by="user_123",
            )

    @pytest.mark.asyncio
    async def test_share_with_user(self, mock_knowledge_mound, sample_knowledge_item):
        """Test sharing an item with a specific user."""
        from aragora.knowledge.mound.ops.sharing import KnowledgeSharingMixin
        from aragora.knowledge.mound.types import AccessGrantType

        class TestMound(KnowledgeSharingMixin):
            pass

        mound = TestMound()
        mound._meta_store = mock_knowledge_mound._meta_store
        mound._cache = None
        mound._initialized = True
        mound._ensure_initialized = lambda: None
        mound.get = mock_knowledge_mound.get

        grant = await mound.share_with_user(
            item_id=sample_knowledge_item.id,
            from_workspace_id="workspace_a",
            user_id="user_external",
            shared_by="user_123",
            permissions=["read", "comment"],
        )

        assert grant.grantee_type == AccessGrantType.USER
        assert grant.grantee_id == "user_external"
        assert "comment" in grant.permissions

    @pytest.mark.asyncio
    async def test_share_nonexistent_item_fails(self, mock_knowledge_mound):
        """Test that sharing a nonexistent item fails."""
        from aragora.knowledge.mound.ops.sharing import KnowledgeSharingMixin

        class TestMound(KnowledgeSharingMixin):
            pass

        mound = TestMound()
        mound._meta_store = mock_knowledge_mound._meta_store
        mound._cache = None
        mound._initialized = True
        mound._ensure_initialized = lambda: None
        mound.get = AsyncMock(return_value=None)  # Item not found

        with pytest.raises(ValueError, match="not found"):
            await mound.share_with_workspace(
                item_id="nonexistent_item",
                from_workspace_id="workspace_a",
                to_workspace_id="workspace_b",
                shared_by="user_123",
            )

    @pytest.mark.asyncio
    async def test_revoke_share(self, mock_knowledge_mound, sample_knowledge_item):
        """Test revoking a share grant."""
        from aragora.knowledge.mound.ops.sharing import KnowledgeSharingMixin

        class TestMound(KnowledgeSharingMixin):
            pass

        mound = TestMound()
        mound._meta_store = mock_knowledge_mound._meta_store
        mound._cache = None
        mound._initialized = True
        mound._ensure_initialized = lambda: None
        mound.get = mock_knowledge_mound.get

        # First share
        grant = await mound.share_with_workspace(
            item_id=sample_knowledge_item.id,
            from_workspace_id="workspace_a",
            to_workspace_id="workspace_b",
            shared_by="user_123",
        )

        # Then revoke
        result = await mound.revoke_share(
            item_id=sample_knowledge_item.id,
            grantee_id="workspace_b",
            revoked_by="user_123",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_get_share_grants(self, mock_knowledge_mound, sample_knowledge_item):
        """Test getting all grants for an item."""
        from aragora.knowledge.mound.ops.sharing import KnowledgeSharingMixin

        class TestMound(KnowledgeSharingMixin):
            pass

        mound = TestMound()
        mound._meta_store = mock_knowledge_mound._meta_store
        mound._cache = None
        mound._initialized = True
        mound._ensure_initialized = lambda: None
        mound.get = mock_knowledge_mound.get

        # Create multiple shares
        await mound.share_with_workspace(
            item_id=sample_knowledge_item.id,
            from_workspace_id="workspace_a",
            to_workspace_id="workspace_b",
            shared_by="user_123",
        )
        await mound.share_with_user(
            item_id=sample_knowledge_item.id,
            from_workspace_id="workspace_a",
            user_id="user_external",
            shared_by="user_123",
        )

        grants = await mound.get_share_grants(sample_knowledge_item.id)
        assert len(grants) == 2


# =============================================================================
# Federation API Tests
# =============================================================================


class TestFederationAPI:
    """Test suite for multi-region federation."""

    @pytest.mark.asyncio
    async def test_federation_modes(self):
        """Test federation mode enum values."""
        from aragora.knowledge.mound.ops.federation import FederationMode

        assert FederationMode.PUSH.value == "push"
        assert FederationMode.PULL.value == "pull"
        assert FederationMode.BIDIRECTIONAL.value == "bidirectional"
        assert FederationMode.NONE.value == "none"

    @pytest.mark.asyncio
    async def test_sync_scope_enum(self):
        """Test sync scope enum values."""
        from aragora.knowledge.mound.ops.federation import SyncScope

        assert SyncScope.FULL.value == "full"
        assert SyncScope.METADATA.value == "metadata"
        assert SyncScope.SUMMARY.value == "summary"

    @pytest.mark.asyncio
    async def test_register_federated_region(self, mock_knowledge_mound):
        """Test registering a federated region."""
        from aragora.knowledge.mound.ops.federation import (
            FederationMode,
            KnowledgeFederationMixin,
            SyncScope,
        )

        class TestMound(KnowledgeFederationMixin):
            pass

        # Clear any existing regions
        KnowledgeFederationMixin._federated_regions = {}

        mound = TestMound()
        mound._meta_store = mock_knowledge_mound._meta_store
        mound._cache = None
        mound._initialized = True
        mound._ensure_initialized = lambda: None
        mound.workspace_id = "test_workspace"

        region = await mound.register_federated_region(
            region_id="us-west-2",
            endpoint_url="https://us-west-2.aragora.example.com/api",
            api_key="test_api_key_123",
            mode=FederationMode.BIDIRECTIONAL,
            sync_scope=SyncScope.SUMMARY,
        )

        assert region.region_id == "us-west-2"
        assert region.endpoint_url == "https://us-west-2.aragora.example.com/api"
        assert region.mode == FederationMode.BIDIRECTIONAL
        assert region.sync_scope == SyncScope.SUMMARY
        assert region.enabled is True

    @pytest.mark.asyncio
    async def test_unregister_federated_region(self, mock_knowledge_mound):
        """Test unregistering a federated region."""
        from aragora.knowledge.mound.ops.federation import KnowledgeFederationMixin

        class TestMound(KnowledgeFederationMixin):
            pass

        # Clear any existing regions
        KnowledgeFederationMixin._federated_regions = {}

        mound = TestMound()
        mound._meta_store = mock_knowledge_mound._meta_store
        mound._cache = None
        mound._initialized = True
        mound._ensure_initialized = lambda: None
        mound.workspace_id = "test_workspace"

        # Register first
        await mound.register_federated_region(
            region_id="eu-west-1",
            endpoint_url="https://eu-west-1.aragora.example.com/api",
            api_key="test_key",
        )

        # Unregister
        result = await mound.unregister_federated_region("eu-west-1")
        assert result is True

        # Try to unregister again
        result = await mound.unregister_federated_region("eu-west-1")
        assert result is False

    @pytest.mark.asyncio
    async def test_sync_to_region(self, mock_knowledge_mound, sample_knowledge_item):
        """Test pushing knowledge to a federated region."""
        from aragora.knowledge.mound.ops.federation import (
            FederationMode,
            KnowledgeFederationMixin,
        )

        class TestMound(KnowledgeFederationMixin):
            pass

        # Clear any existing regions
        KnowledgeFederationMixin._federated_regions = {}

        mound = TestMound()
        mound._meta_store = mock_knowledge_mound._meta_store
        mound._cache = None
        mound._initialized = True
        mound._ensure_initialized = lambda: None
        mound.workspace_id = "test_workspace"
        mound.query = mock_knowledge_mound.query

        # Register region
        await mound.register_federated_region(
            region_id="us-east-1",
            endpoint_url="https://us-east-1.aragora.example.com/api",
            api_key="test_key",
            mode=FederationMode.PUSH,
        )

        # Sync to region
        result = await mound.sync_to_region("us-east-1")

        assert result.region_id == "us-east-1"
        assert result.direction == "push"
        assert result.success is True

    @pytest.mark.asyncio
    async def test_sync_to_pull_only_region_fails(self, mock_knowledge_mound):
        """Test that push to pull-only region fails."""
        from aragora.knowledge.mound.ops.federation import (
            FederationMode,
            KnowledgeFederationMixin,
        )

        class TestMound(KnowledgeFederationMixin):
            pass

        # Clear any existing regions
        KnowledgeFederationMixin._federated_regions = {}

        mound = TestMound()
        mound._meta_store = mock_knowledge_mound._meta_store
        mound._cache = None
        mound._initialized = True
        mound._ensure_initialized = lambda: None
        mound.workspace_id = "test_workspace"

        # Register pull-only region
        await mound.register_federated_region(
            region_id="pull-only",
            endpoint_url="https://pull-only.example.com/api",
            api_key="test_key",
            mode=FederationMode.PULL,
        )

        # Try to push
        result = await mound.sync_to_region("pull-only")

        assert result.success is False
        assert "pull-only" in result.error

    @pytest.mark.asyncio
    async def test_pull_from_region(self, mock_knowledge_mound):
        """Test pulling knowledge from a federated region."""
        from aragora.knowledge.mound.ops.federation import (
            FederationMode,
            KnowledgeFederationMixin,
        )

        class TestMound(KnowledgeFederationMixin):
            pass

        # Clear any existing regions
        KnowledgeFederationMixin._federated_regions = {}

        mound = TestMound()
        mound._meta_store = mock_knowledge_mound._meta_store
        mound._cache = None
        mound._initialized = True
        mound._ensure_initialized = lambda: None
        mound.workspace_id = "test_workspace"
        mound.store = mock_knowledge_mound.store

        # Register region
        await mound.register_federated_region(
            region_id="us-west-2",
            endpoint_url="https://us-west-2.aragora.example.com/api",
            api_key="test_key",
            mode=FederationMode.PULL,
        )

        # Pull from region
        result = await mound.pull_from_region("us-west-2")

        assert result.region_id == "us-west-2"
        assert result.direction == "pull"
        assert result.success is True

    @pytest.mark.asyncio
    async def test_get_federation_status(self, mock_knowledge_mound):
        """Test getting federation status for all regions."""
        from aragora.knowledge.mound.ops.federation import (
            FederationMode,
            KnowledgeFederationMixin,
            SyncScope,
        )

        class TestMound(KnowledgeFederationMixin):
            pass

        # Clear any existing regions
        KnowledgeFederationMixin._federated_regions = {}

        mound = TestMound()
        mound._meta_store = mock_knowledge_mound._meta_store
        mound._cache = None
        mound._initialized = True
        mound._ensure_initialized = lambda: None
        mound.workspace_id = "test_workspace"

        # Register multiple regions
        await mound.register_federated_region(
            region_id="us-east-1",
            endpoint_url="https://us-east-1.example.com/api",
            api_key="key1",
            mode=FederationMode.PUSH,
        )
        await mound.register_federated_region(
            region_id="eu-west-1",
            endpoint_url="https://eu-west-1.example.com/api",
            api_key="key2",
            mode=FederationMode.BIDIRECTIONAL,
        )

        status = await mound.get_federation_status()

        assert "us-east-1" in status
        assert "eu-west-1" in status
        assert status["us-east-1"]["mode"] == "push"
        assert status["eu-west-1"]["mode"] == "bidirectional"

    @pytest.mark.asyncio
    async def test_sync_unregistered_region_fails(self, mock_knowledge_mound):
        """Test that syncing to unregistered region fails."""
        from aragora.knowledge.mound.ops.federation import KnowledgeFederationMixin

        class TestMound(KnowledgeFederationMixin):
            pass

        # Clear any existing regions
        KnowledgeFederationMixin._federated_regions = {}

        mound = TestMound()
        mound._meta_store = mock_knowledge_mound._meta_store
        mound._cache = None
        mound._initialized = True
        mound._ensure_initialized = lambda: None
        mound.workspace_id = "test_workspace"

        result = await mound.sync_to_region("nonexistent-region")

        assert result.success is False
        assert "not registered" in result.error


# =============================================================================
# Global Knowledge API Tests
# =============================================================================


class TestGlobalKnowledgeAPI:
    """Test suite for global/system knowledge."""

    @pytest.mark.asyncio
    async def test_system_workspace_id(self):
        """Test that system workspace ID is defined."""
        from aragora.knowledge.mound.ops.global_knowledge import SYSTEM_WORKSPACE_ID

        assert SYSTEM_WORKSPACE_ID == "__system__"

    @pytest.mark.asyncio
    async def test_store_verified_fact(self, mock_knowledge_mound):
        """Test storing a verified fact in global knowledge."""
        from aragora.knowledge.mound.ops.global_knowledge import GlobalKnowledgeMixin

        class TestMound(GlobalKnowledgeMixin):
            pass

        mound = TestMound()
        mound._meta_store = mock_knowledge_mound._meta_store
        mound._cache = None
        mound._initialized = True
        mound._ensure_initialized = lambda: None
        mound.config = mock_knowledge_mound.config
        mound.workspace_id = "test_workspace"
        mound.store = mock_knowledge_mound.store

        fact_id = await mound.store_verified_fact(
            content="The speed of light is approximately 299,792,458 m/s.",
            source="physics_textbook",
            confidence=0.99,
            verified_by="admin",
            topics=["physics", "constants"],
        )

        assert fact_id is not None
        assert fact_id.startswith("item_")

        # Verify store was called with correct workspace
        call_args = mound.store.call_args[0][0]
        assert call_args.workspace_id == "__system__"

    @pytest.mark.asyncio
    async def test_query_global_knowledge(self, mock_knowledge_mound, sample_knowledge_item):
        """Test querying global knowledge."""
        from aragora.knowledge.mound.ops.global_knowledge import GlobalKnowledgeMixin

        class TestMound(GlobalKnowledgeMixin):
            pass

        mound = TestMound()
        mound._meta_store = mock_knowledge_mound._meta_store
        mound._cache = None
        mound._initialized = True
        mound._ensure_initialized = lambda: None
        mound.config = mock_knowledge_mound.config
        mound.workspace_id = "test_workspace"
        mound.query = mock_knowledge_mound.query

        results = await mound.query_global_knowledge(
            query="machine learning",
            limit=10,
        )

        # Verify query was called with system workspace
        call_args = mound.query.call_args
        assert call_args.kwargs.get("workspace_id") == "__system__"

    @pytest.mark.asyncio
    async def test_promote_to_global(self, mock_knowledge_mound, sample_knowledge_item):
        """Test promoting workspace knowledge to global."""
        from aragora.knowledge.mound.ops.global_knowledge import GlobalKnowledgeMixin

        class TestMound(GlobalKnowledgeMixin):
            pass

        mound = TestMound()
        mound._meta_store = mock_knowledge_mound._meta_store
        mound._cache = None
        mound._initialized = True
        mound._ensure_initialized = lambda: None
        mound.config = mock_knowledge_mound.config
        mound.workspace_id = "test_workspace"
        mound.get = mock_knowledge_mound.get
        mound.store = mock_knowledge_mound.store

        global_id = await mound.promote_to_global(
            item_id=sample_knowledge_item.id,
            workspace_id="workspace_a",
            promoted_by="admin",
            reason="high_consensus",
        )

        assert global_id is not None

    @pytest.mark.asyncio
    async def test_promote_nonexistent_item_fails(self, mock_knowledge_mound):
        """Test that promoting a nonexistent item fails."""
        from aragora.knowledge.mound.ops.global_knowledge import GlobalKnowledgeMixin

        class TestMound(GlobalKnowledgeMixin):
            pass

        mound = TestMound()
        mound._meta_store = mock_knowledge_mound._meta_store
        mound._cache = None
        mound._initialized = True
        mound._ensure_initialized = lambda: None
        mound.config = mock_knowledge_mound.config
        mound.workspace_id = "test_workspace"
        mound.get = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await mound.promote_to_global(
                item_id="nonexistent",
                workspace_id="workspace_a",
                promoted_by="admin",
                reason="test",
            )

    @pytest.mark.asyncio
    async def test_get_system_facts(self, mock_knowledge_mound):
        """Test getting all system facts."""
        from aragora.knowledge.mound.ops.global_knowledge import GlobalKnowledgeMixin

        class TestMound(GlobalKnowledgeMixin):
            pass

        mound = TestMound()
        mound._meta_store = mock_knowledge_mound._meta_store
        mound._cache = None
        mound._initialized = True
        mound._ensure_initialized = lambda: None
        mound.config = mock_knowledge_mound.config
        mound.workspace_id = "test_workspace"
        mound.query = mock_knowledge_mound.query

        facts = await mound.get_system_facts(limit=50)

        # Should have called query with system workspace
        call_args = mound.query.call_args
        assert call_args.kwargs.get("workspace_id") == "__system__"

    @pytest.mark.asyncio
    async def test_merge_global_results(self, mock_knowledge_mound, sample_knowledge_item):
        """Test merging workspace results with global knowledge."""
        from aragora.knowledge.mound.ops.global_knowledge import GlobalKnowledgeMixin
        from aragora.knowledge.mound.types import KnowledgeItem

        class TestMound(GlobalKnowledgeMixin):
            pass

        mound = TestMound()
        mound._meta_store = mock_knowledge_mound._meta_store
        mound._cache = None
        mound._initialized = True
        mound._ensure_initialized = lambda: None
        mound.config = mock_knowledge_mound.config
        mound.workspace_id = "test_workspace"
        mound.query = mock_knowledge_mound.query

        workspace_results = [sample_knowledge_item]

        merged = await mound.merge_global_results(
            workspace_results=workspace_results,
            query="test query",
            global_limit=3,
        )

        # Should contain at least the workspace results
        assert len(merged) >= len(workspace_results)

    @pytest.mark.asyncio
    async def test_get_system_workspace_id(self, mock_knowledge_mound):
        """Test getting the system workspace ID."""
        from aragora.knowledge.mound.ops.global_knowledge import GlobalKnowledgeMixin

        class TestMound(GlobalKnowledgeMixin):
            pass

        mound = TestMound()
        mound._meta_store = mock_knowledge_mound._meta_store
        mound._cache = None
        mound._initialized = True
        mound._ensure_initialized = lambda: None
        mound.config = mock_knowledge_mound.config
        mound.workspace_id = "test_workspace"

        system_id = mound.get_system_workspace_id()
        assert system_id == "__system__"


# =============================================================================
# Integration Tests
# =============================================================================


class TestKnowledgeMoundIntegration:
    """Integration tests across multiple Knowledge Mound features."""

    @pytest.mark.asyncio
    async def test_share_and_revoke_flow(self, mock_knowledge_mound, sample_knowledge_item):
        """Test complete share and revoke workflow."""
        from aragora.knowledge.mound.ops.sharing import KnowledgeSharingMixin

        class TestMound(KnowledgeSharingMixin):
            pass

        mound = TestMound()
        mound._meta_store = mock_knowledge_mound._meta_store
        mound._cache = None
        mound._initialized = True
        mound._ensure_initialized = lambda: None
        mound.get = mock_knowledge_mound.get

        # 1. Share with workspace
        grant = await mound.share_with_workspace(
            item_id=sample_knowledge_item.id,
            from_workspace_id="workspace_a",
            to_workspace_id="workspace_b",
            shared_by="user_1",
        )
        assert grant is not None

        # 2. Verify grant exists
        grants = await mound.get_share_grants(sample_knowledge_item.id)
        assert len(grants) == 1

        # 3. Revoke share
        revoked = await mound.revoke_share(
            item_id=sample_knowledge_item.id,
            grantee_id="workspace_b",
            revoked_by="user_1",
        )
        assert revoked is True

        # 4. Verify grant is removed
        grants = await mound.get_share_grants(sample_knowledge_item.id)
        assert len(grants) == 0

    @pytest.mark.asyncio
    async def test_federation_bidirectional_sync(self, mock_knowledge_mound):
        """Test bidirectional federation sync."""
        from aragora.knowledge.mound.ops.federation import (
            FederationMode,
            KnowledgeFederationMixin,
        )

        class TestMound(KnowledgeFederationMixin):
            pass

        # Clear any existing regions
        KnowledgeFederationMixin._federated_regions = {}

        mound = TestMound()
        mound._meta_store = mock_knowledge_mound._meta_store
        mound._cache = None
        mound._initialized = True
        mound._ensure_initialized = lambda: None
        mound.workspace_id = "test_workspace"
        mound.query = mock_knowledge_mound.query
        mound.store = mock_knowledge_mound.store

        # Register bidirectional region
        await mound.register_federated_region(
            region_id="partner-region",
            endpoint_url="https://partner.example.com/api",
            api_key="partner_key",
            mode=FederationMode.BIDIRECTIONAL,
        )

        # Push
        push_result = await mound.sync_to_region("partner-region")
        assert push_result.success is True
        assert push_result.direction == "push"

        # Pull
        pull_result = await mound.pull_from_region("partner-region")
        assert pull_result.success is True
        assert pull_result.direction == "pull"

    @pytest.mark.asyncio
    async def test_global_knowledge_store_and_query(self, mock_knowledge_mound):
        """Test storing and querying global knowledge."""
        from aragora.knowledge.mound.ops.global_knowledge import GlobalKnowledgeMixin

        class TestMound(GlobalKnowledgeMixin):
            pass

        mound = TestMound()
        mound._meta_store = mock_knowledge_mound._meta_store
        mound._cache = None
        mound._initialized = True
        mound._ensure_initialized = lambda: None
        mound.config = mock_knowledge_mound.config
        mound.workspace_id = "test_workspace"
        mound.store = mock_knowledge_mound.store
        mound.query = mock_knowledge_mound.query

        # Store facts
        fact_id_1 = await mound.store_verified_fact(
            content="Python is a programming language.",
            source="wikipedia",
            confidence=0.95,
            topics=["programming", "python"],
        )

        fact_id_2 = await mound.store_verified_fact(
            content="E=mc² is the mass-energy equivalence formula.",
            source="physics_textbook",
            confidence=0.99,
            topics=["physics", "relativity"],
        )

        # Verify facts were stored
        assert fact_id_1 is not None
        assert fact_id_2 is not None

        # Query without filters (no topics/min_confidence)
        results = await mound.query_global_knowledge(
            query="energy",
            limit=10,
        )

        # Verify query was made to system workspace
        assert mound.query.called
        call_kwargs = mound.query.call_args.kwargs
        assert call_kwargs.get("workspace_id") == "__system__"

    @pytest.mark.asyncio
    async def test_visibility_with_sharing_interaction(
        self, mock_knowledge_mound, sample_knowledge_item
    ):
        """Test that visibility and sharing work together correctly."""
        from aragora.knowledge.mound.ops.sharing import KnowledgeSharingMixin
        from aragora.knowledge.mound.types import AccessGrantType, VisibilityLevel

        class TestMound(KnowledgeSharingMixin):
            pass

        mound = TestMound()
        mound._meta_store = mock_knowledge_mound._meta_store
        mound._cache = None
        mound._initialized = True
        mound._ensure_initialized = lambda: None
        mound.get = mock_knowledge_mound.get

        # Share a workspace item
        grant = await mound.share_with_workspace(
            item_id=sample_knowledge_item.id,
            from_workspace_id="workspace_a",
            to_workspace_id="workspace_b",
            shared_by="user_1",
            permissions=["read"],
        )

        # Verify the share was created correctly
        assert grant.grantee_type == AccessGrantType.WORKSPACE
        assert grant.has_permission("read")
        assert not grant.has_permission("write")

        # Get grants to verify persistence
        grants = await mound.get_share_grants(sample_knowledge_item.id)
        assert len(grants) == 1
        assert grants[0].grantee_id == "workspace_b"
