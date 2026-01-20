"""
Tests for Knowledge Mound Visibility System.

Tests cover:
- VisibilityLevel enum
- AccessGrant dataclass
- AccessGrantType enum
- Visibility filtering in queries
- EnhancedKnowledgeItem visibility fields
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.knowledge.mound.types import (
    VisibilityLevel,
    AccessGrantType,
    AccessGrant,
    EnhancedKnowledgeItem,
    KnowledgeSource,
    ConfidenceLevel,
)


# =============================================================================
# VisibilityLevel Tests
# =============================================================================


class TestVisibilityLevel:
    """Tests for VisibilityLevel enum."""

    def test_visibility_values(self):
        """Should have correct visibility levels."""
        assert VisibilityLevel.PRIVATE.value == "private"
        assert VisibilityLevel.WORKSPACE.value == "workspace"
        assert VisibilityLevel.ORGANIZATION.value == "organization"
        assert VisibilityLevel.PUBLIC.value == "public"
        assert VisibilityLevel.SYSTEM.value == "system"

    def test_visibility_from_string(self):
        """Should create from string value."""
        assert VisibilityLevel("private") == VisibilityLevel.PRIVATE
        assert VisibilityLevel("workspace") == VisibilityLevel.WORKSPACE
        assert VisibilityLevel("public") == VisibilityLevel.PUBLIC
        assert VisibilityLevel("system") == VisibilityLevel.SYSTEM

    def test_visibility_invalid_string(self):
        """Should raise on invalid string."""
        with pytest.raises(ValueError):
            VisibilityLevel("invalid")

    def test_all_visibility_levels(self):
        """Should have exactly 5 visibility levels."""
        assert len(VisibilityLevel) == 5


# =============================================================================
# AccessGrantType Tests
# =============================================================================


class TestAccessGrantType:
    """Tests for AccessGrantType enum."""

    def test_grant_type_values(self):
        """Should have correct grant types."""
        assert AccessGrantType.USER.value == "user"
        assert AccessGrantType.ROLE.value == "role"
        assert AccessGrantType.WORKSPACE.value == "workspace"
        assert AccessGrantType.ORGANIZATION.value == "organization"

    def test_grant_type_from_string(self):
        """Should create from string value."""
        assert AccessGrantType("user") == AccessGrantType.USER
        assert AccessGrantType("workspace") == AccessGrantType.WORKSPACE


# =============================================================================
# AccessGrant Tests
# =============================================================================


class TestAccessGrant:
    """Tests for AccessGrant dataclass."""

    def test_create_basic_grant(self):
        """Should create a basic grant with defaults."""
        grant = AccessGrant(
            id="grant_123",
            item_id="item_456",
            grantee_type=AccessGrantType.USER,
            grantee_id="user_789",
        )
        assert grant.id == "grant_123"
        assert grant.item_id == "item_456"
        assert grant.grantee_type == AccessGrantType.USER
        assert grant.grantee_id == "user_789"
        assert grant.permissions == ["read"]
        assert grant.granted_by is None
        assert grant.expires_at is None

    def test_create_full_grant(self):
        """Should create a grant with all fields."""
        expires = datetime.now() + timedelta(days=30)
        grant = AccessGrant(
            id="grant_123",
            item_id="item_456",
            grantee_type=AccessGrantType.WORKSPACE,
            grantee_id="ws_789",
            permissions=["read", "write"],
            granted_by="admin_user",
            expires_at=expires,
        )
        assert grant.permissions == ["read", "write"]
        assert grant.granted_by == "admin_user"
        assert grant.expires_at == expires

    def test_is_expired_no_expiry(self):
        """Should not be expired when no expiry set."""
        grant = AccessGrant(
            id="grant_123",
            item_id="item_456",
            grantee_type=AccessGrantType.USER,
            grantee_id="user_789",
        )
        assert grant.is_expired() is False

    def test_is_expired_future(self):
        """Should not be expired when expiry is in future."""
        grant = AccessGrant(
            id="grant_123",
            item_id="item_456",
            grantee_type=AccessGrantType.USER,
            grantee_id="user_789",
            expires_at=datetime.now() + timedelta(days=1),
        )
        assert grant.is_expired() is False

    def test_is_expired_past(self):
        """Should be expired when expiry is in past."""
        grant = AccessGrant(
            id="grant_123",
            item_id="item_456",
            grantee_type=AccessGrantType.USER,
            grantee_id="user_789",
            expires_at=datetime.now() - timedelta(days=1),
        )
        assert grant.is_expired() is True

    def test_has_permission_read(self):
        """Should check read permission."""
        grant = AccessGrant(
            id="grant_123",
            item_id="item_456",
            grantee_type=AccessGrantType.USER,
            grantee_id="user_789",
            permissions=["read"],
        )
        assert grant.has_permission("read") is True
        assert grant.has_permission("write") is False

    def test_has_permission_admin(self):
        """Admin permission should grant all."""
        grant = AccessGrant(
            id="grant_123",
            item_id="item_456",
            grantee_type=AccessGrantType.USER,
            grantee_id="user_789",
            permissions=["admin"],
        )
        assert grant.has_permission("read") is True
        assert grant.has_permission("write") is True
        assert grant.has_permission("delete") is True

    def test_to_dict(self):
        """Should serialize to dictionary."""
        grant = AccessGrant(
            id="grant_123",
            item_id="item_456",
            grantee_type=AccessGrantType.USER,
            grantee_id="user_789",
            permissions=["read", "write"],
            granted_by="admin",
        )
        d = grant.to_dict()
        assert d["id"] == "grant_123"
        assert d["item_id"] == "item_456"
        assert d["grantee_type"] == "user"
        assert d["grantee_id"] == "user_789"
        assert d["permissions"] == ["read", "write"]
        assert d["granted_by"] == "admin"
        assert d["expires_at"] is None


# =============================================================================
# EnhancedKnowledgeItem Visibility Tests
# =============================================================================


class TestEnhancedKnowledgeItemVisibility:
    """Tests for visibility fields on EnhancedKnowledgeItem."""

    def test_default_visibility(self):
        """Should default to WORKSPACE visibility."""
        now = datetime.now()
        item = EnhancedKnowledgeItem(
            id="item_123",
            content="Test content",
            source=KnowledgeSource.FACT,
            source_id="src_123",
            confidence=ConfidenceLevel.HIGH,
            created_at=now,
            updated_at=now,
        )
        assert item.visibility == VisibilityLevel.WORKSPACE
        assert item.is_discoverable is True
        assert item.access_grants == []
        assert item.visibility_set_by is None

    def test_set_visibility(self):
        """Should allow setting visibility."""
        now = datetime.now()
        item = EnhancedKnowledgeItem(
            id="item_123",
            content="Test content",
            source=KnowledgeSource.FACT,
            source_id="src_123",
            confidence=ConfidenceLevel.HIGH,
            created_at=now,
            updated_at=now,
            visibility=VisibilityLevel.PRIVATE,
            visibility_set_by="user_admin",
            is_discoverable=False,
        )
        assert item.visibility == VisibilityLevel.PRIVATE
        assert item.visibility_set_by == "user_admin"
        assert item.is_discoverable is False

    def test_to_dict_includes_visibility(self):
        """Should include visibility in to_dict."""
        now = datetime.now()
        grant = AccessGrant(
            id="grant_1",
            item_id="item_123",
            grantee_type=AccessGrantType.USER,
            grantee_id="user_1",
        )
        item = EnhancedKnowledgeItem(
            id="item_123",
            content="Test content",
            source=KnowledgeSource.FACT,
            source_id="src_123",
            confidence=ConfidenceLevel.HIGH,
            created_at=now,
            updated_at=now,
            visibility=VisibilityLevel.PRIVATE,
            visibility_set_by="admin",
            access_grants=[grant],
            is_discoverable=False,
        )
        d = item.to_dict()
        assert d["visibility"] == "private"
        assert d["visibility_set_by"] == "admin"
        assert d["is_discoverable"] is False
        assert len(d["access_grants"]) == 1
        assert d["access_grants"][0]["grantee_id"] == "user_1"


# =============================================================================
# Visibility Filtering Tests
# =============================================================================


class TestVisibilityFiltering:
    """Tests for visibility filtering in queries."""

    @pytest.fixture
    def mock_items(self):
        """Create mock items with different visibility levels."""
        now = datetime.now()
        return [
            MagicMock(
                id="public_item",
                content="Public content",
                metadata={"visibility": "public", "workspace_id": "ws_1"},
                importance=0.8,
            ),
            MagicMock(
                id="system_item",
                content="System content",
                metadata={"visibility": "system", "workspace_id": "__system__"},
                importance=0.9,
            ),
            MagicMock(
                id="workspace_item",
                content="Workspace content",
                metadata={"visibility": "workspace", "workspace_id": "ws_1"},
                importance=0.7,
            ),
            MagicMock(
                id="other_workspace_item",
                content="Other workspace content",
                metadata={"visibility": "workspace", "workspace_id": "ws_2"},
                importance=0.6,
            ),
            MagicMock(
                id="org_item",
                content="Org content",
                metadata={"visibility": "organization", "workspace_id": "ws_1", "org_id": "org_1"},
                importance=0.75,
            ),
            MagicMock(
                id="private_item",
                content="Private content",
                metadata={
                    "visibility": "private",
                    "workspace_id": "ws_1",
                    "access_grants": [
                        {"grantee_type": "user", "grantee_id": "user_1", "expires_at": None}
                    ],
                },
                importance=0.85,
            ),
            MagicMock(
                id="private_no_grant",
                content="Private no access",
                metadata={"visibility": "private", "workspace_id": "ws_1", "access_grants": []},
                importance=0.5,
            ),
        ]

    @pytest.mark.asyncio
    async def test_filter_public_visible_to_all(self, mock_items):
        """Public items should be visible to everyone."""
        from aragora.knowledge.mound.api.query import QueryOperationsMixin

        mixin = QueryOperationsMixin()
        # We need to test the _filter_by_visibility method
        # Since it's a mixin, we'll test it via the logic directly

        filtered = await mixin._filter_by_visibility(
            items=mock_items,
            actor_id="any_user",
            actor_workspace_id="any_workspace",
        )

        # Public and system items should always be visible
        visible_ids = [item.id for item in filtered]
        assert "public_item" in visible_ids
        assert "system_item" in visible_ids

    @pytest.mark.asyncio
    async def test_filter_workspace_items(self, mock_items):
        """Workspace items visible only to workspace members."""
        from aragora.knowledge.mound.api.query import QueryOperationsMixin

        mixin = QueryOperationsMixin()
        filtered = await mixin._filter_by_visibility(
            items=mock_items,
            actor_id="user_1",
            actor_workspace_id="ws_1",
        )

        visible_ids = [item.id for item in filtered]
        assert "workspace_item" in visible_ids
        assert "other_workspace_item" not in visible_ids

    @pytest.mark.asyncio
    async def test_filter_org_items(self, mock_items):
        """Org items visible to org members."""
        from aragora.knowledge.mound.api.query import QueryOperationsMixin

        mixin = QueryOperationsMixin()
        filtered = await mixin._filter_by_visibility(
            items=mock_items,
            actor_id="user_1",
            actor_workspace_id="ws_1",
            actor_org_id="org_1",
        )

        visible_ids = [item.id for item in filtered]
        assert "org_item" in visible_ids

    @pytest.mark.asyncio
    async def test_filter_private_with_grant(self, mock_items):
        """Private items visible with access grant."""
        from aragora.knowledge.mound.api.query import QueryOperationsMixin

        mixin = QueryOperationsMixin()
        filtered = await mixin._filter_by_visibility(
            items=mock_items,
            actor_id="user_1",
            actor_workspace_id="ws_1",
        )

        visible_ids = [item.id for item in filtered]
        assert "private_item" in visible_ids
        assert "private_no_grant" not in visible_ids

    @pytest.mark.asyncio
    async def test_filter_private_without_grant(self, mock_items):
        """Private items not visible without access grant."""
        from aragora.knowledge.mound.api.query import QueryOperationsMixin

        mixin = QueryOperationsMixin()
        filtered = await mixin._filter_by_visibility(
            items=mock_items,
            actor_id="user_2",  # Different user
            actor_workspace_id="ws_1",
        )

        visible_ids = [item.id for item in filtered]
        assert "private_item" not in visible_ids


# =============================================================================
# Has Access Grant Helper Tests
# =============================================================================


class TestHasAccessGrant:
    """Tests for _has_access_grant helper."""

    def test_user_grant_matches(self):
        """Should match user grant."""
        from aragora.knowledge.mound.api.query import QueryOperationsMixin

        mixin = QueryOperationsMixin()
        grants = [
            {"grantee_type": "user", "grantee_id": "user_1", "expires_at": None}
        ]
        assert mixin._has_access_grant(grants, "user_1", "ws_1", None) is True
        assert mixin._has_access_grant(grants, "user_2", "ws_1", None) is False

    def test_workspace_grant_matches(self):
        """Should match workspace grant."""
        from aragora.knowledge.mound.api.query import QueryOperationsMixin

        mixin = QueryOperationsMixin()
        grants = [
            {"grantee_type": "workspace", "grantee_id": "ws_1", "expires_at": None}
        ]
        assert mixin._has_access_grant(grants, "any_user", "ws_1", None) is True
        assert mixin._has_access_grant(grants, "any_user", "ws_2", None) is False

    def test_org_grant_matches(self):
        """Should match organization grant."""
        from aragora.knowledge.mound.api.query import QueryOperationsMixin

        mixin = QueryOperationsMixin()
        grants = [
            {"grantee_type": "organization", "grantee_id": "org_1", "expires_at": None}
        ]
        assert mixin._has_access_grant(grants, "any_user", "any_ws", "org_1") is True
        assert mixin._has_access_grant(grants, "any_user", "any_ws", "org_2") is False

    def test_expired_grant_not_matches(self):
        """Should not match expired grants."""
        from aragora.knowledge.mound.api.query import QueryOperationsMixin

        mixin = QueryOperationsMixin()
        expired_time = (datetime.now() - timedelta(days=1)).isoformat()
        grants = [
            {"grantee_type": "user", "grantee_id": "user_1", "expires_at": expired_time}
        ]
        assert mixin._has_access_grant(grants, "user_1", "ws_1", None) is False

    def test_future_expiry_matches(self):
        """Should match grants with future expiry."""
        from aragora.knowledge.mound.api.query import QueryOperationsMixin

        mixin = QueryOperationsMixin()
        future_time = (datetime.now() + timedelta(days=1)).isoformat()
        grants = [
            {"grantee_type": "user", "grantee_id": "user_1", "expires_at": future_time}
        ]
        assert mixin._has_access_grant(grants, "user_1", "ws_1", None) is True
