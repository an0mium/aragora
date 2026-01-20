"""
Tests for Knowledge Sharing Mixin.

Tests cover:
- share_with_workspace
- share_with_user
- get_shared_with_me
- revoke_share
- get_share_grants
- update_share_permissions
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.knowledge.mound.ops.sharing import KnowledgeSharingMixin
from aragora.knowledge.mound.types import (
    AccessGrant,
    AccessGrantType,
    KnowledgeSource,
)


# =============================================================================
# MockKnowledgeMound for Testing
# =============================================================================


class MockMetaStore:
    """Mock meta store with access grant support."""

    def __init__(self):
        self._grants = []

    async def save_access_grant_async(self, grant):
        # Replace existing grant if same item/grantee
        self._grants = [
            g
            for g in self._grants
            if not (g.item_id == grant.item_id and g.grantee_id == grant.grantee_id)
        ]
        self._grants.append(grant)

    async def get_access_grants_async(self, item_id):
        return [g for g in self._grants if g.item_id == item_id]

    async def get_grants_for_grantee_async(self, grantee_id, grantee_type=None):
        results = []
        for g in self._grants:
            if g.grantee_id == grantee_id:
                if grantee_type is None or g.grantee_type == grantee_type:
                    results.append(g)
        return results

    async def delete_access_grant_async(self, item_id, grantee_id):
        before = len(self._grants)
        self._grants = [
            g for g in self._grants if not (g.item_id == item_id and g.grantee_id == grantee_id)
        ]
        return len(self._grants) < before


class MockKnowledgeMound(KnowledgeSharingMixin):
    """Mock KnowledgeMound with KnowledgeSharingMixin."""

    def __init__(self):
        self.config = MagicMock()
        self.workspace_id = "test_workspace"
        self._meta_store = MockMetaStore()
        self._cache = None
        self._initialized = True
        self._items = {}

    def _ensure_initialized(self):
        if not self._initialized:
            raise RuntimeError("Not initialized")

    async def get(self, node_id, workspace_id=None):
        """Mock get method."""
        return self._items.get(node_id)

    def add_item(self, item_id, content, workspace_id):
        """Helper to add test items."""
        item = MagicMock()
        item.id = item_id
        item.content = content
        item.workspace_id = workspace_id
        self._items[item_id] = item
        return item


# =============================================================================
# share_with_workspace Tests
# =============================================================================


class TestShareWithWorkspace:
    """Tests for share_with_workspace method."""

    @pytest.mark.asyncio
    async def test_share_basic(self):
        """Should create workspace share grant."""
        mound = MockKnowledgeMound()
        mound.add_item("item_1", "Test content", "ws_source")

        grant = await mound.share_with_workspace(
            item_id="item_1",
            from_workspace_id="ws_source",
            to_workspace_id="ws_target",
            shared_by="user_admin",
        )

        assert grant.item_id == "item_1"
        assert grant.grantee_type == AccessGrantType.WORKSPACE
        assert grant.grantee_id == "ws_target"
        assert grant.permissions == ["read"]
        assert grant.granted_by == "user_admin"

    @pytest.mark.asyncio
    async def test_share_with_permissions(self):
        """Should create grant with custom permissions."""
        mound = MockKnowledgeMound()
        mound.add_item("item_1", "Test content", "ws_source")

        grant = await mound.share_with_workspace(
            item_id="item_1",
            from_workspace_id="ws_source",
            to_workspace_id="ws_target",
            shared_by="admin",
            permissions=["read", "write", "share"],
        )

        assert grant.permissions == ["read", "write", "share"]

    @pytest.mark.asyncio
    async def test_share_with_expiry(self):
        """Should create grant with expiration."""
        mound = MockKnowledgeMound()
        mound.add_item("item_1", "Test content", "ws_source")

        expires = datetime.now() + timedelta(days=30)
        grant = await mound.share_with_workspace(
            item_id="item_1",
            from_workspace_id="ws_source",
            to_workspace_id="ws_target",
            shared_by="admin",
            expires_at=expires,
        )

        assert grant.expires_at == expires

    @pytest.mark.asyncio
    async def test_share_same_workspace_fails(self):
        """Should not allow sharing with same workspace."""
        mound = MockKnowledgeMound()
        mound.add_item("item_1", "Test content", "ws_1")

        with pytest.raises(ValueError, match="same workspace"):
            await mound.share_with_workspace(
                item_id="item_1",
                from_workspace_id="ws_1",
                to_workspace_id="ws_1",
                shared_by="admin",
            )

    @pytest.mark.asyncio
    async def test_share_nonexistent_item_fails(self):
        """Should fail when item doesn't exist."""
        mound = MockKnowledgeMound()

        with pytest.raises(ValueError, match="not found"):
            await mound.share_with_workspace(
                item_id="nonexistent",
                from_workspace_id="ws_1",
                to_workspace_id="ws_2",
                shared_by="admin",
            )

    @pytest.mark.asyncio
    async def test_share_persists_grant(self):
        """Should persist grant to store."""
        mound = MockKnowledgeMound()
        mound.add_item("item_1", "Test content", "ws_source")

        await mound.share_with_workspace(
            item_id="item_1",
            from_workspace_id="ws_source",
            to_workspace_id="ws_target",
            shared_by="admin",
        )

        # Verify persisted
        grants = await mound._meta_store.get_access_grants_async("item_1")
        assert len(grants) == 1
        assert grants[0].grantee_id == "ws_target"


# =============================================================================
# share_with_user Tests
# =============================================================================


class TestShareWithUser:
    """Tests for share_with_user method."""

    @pytest.mark.asyncio
    async def test_share_with_user_basic(self):
        """Should create user share grant."""
        mound = MockKnowledgeMound()
        mound.add_item("item_1", "Test content", "ws_source")

        grant = await mound.share_with_user(
            item_id="item_1",
            from_workspace_id="ws_source",
            user_id="user_target",
            shared_by="user_owner",
        )

        assert grant.item_id == "item_1"
        assert grant.grantee_type == AccessGrantType.USER
        assert grant.grantee_id == "user_target"
        assert grant.permissions == ["read"]

    @pytest.mark.asyncio
    async def test_share_with_user_permissions(self):
        """Should create user grant with custom permissions."""
        mound = MockKnowledgeMound()
        mound.add_item("item_1", "Test content", "ws_source")

        grant = await mound.share_with_user(
            item_id="item_1",
            from_workspace_id="ws_source",
            user_id="collaborator",
            shared_by="owner",
            permissions=["read", "comment"],
        )

        assert grant.permissions == ["read", "comment"]

    @pytest.mark.asyncio
    async def test_share_with_user_nonexistent_fails(self):
        """Should fail when item doesn't exist."""
        mound = MockKnowledgeMound()

        with pytest.raises(ValueError, match="not found"):
            await mound.share_with_user(
                item_id="nonexistent",
                from_workspace_id="ws_1",
                user_id="user_1",
                shared_by="admin",
            )


# =============================================================================
# get_shared_with_me Tests
# =============================================================================


class TestGetSharedWithMe:
    """Tests for get_shared_with_me method."""

    @pytest.mark.asyncio
    async def test_get_empty_shares(self):
        """Should return empty list when no shares."""
        mound = MockKnowledgeMound()
        results = await mound.get_shared_with_me(
            workspace_id="ws_1",
            user_id="user_1",
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_get_workspace_shares(self):
        """Should return items shared with workspace."""
        mound = MockKnowledgeMound()
        mound.add_item("item_1", "Shared content", "ws_source")

        # Share with workspace
        await mound.share_with_workspace(
            item_id="item_1",
            from_workspace_id="ws_source",
            to_workspace_id="ws_target",
            shared_by="admin",
        )

        results = await mound.get_shared_with_me(
            workspace_id="ws_target",
        )

        assert len(results) == 1
        assert results[0].id == "item_1"

    @pytest.mark.asyncio
    async def test_get_user_shares(self):
        """Should return items shared with user."""
        mound = MockKnowledgeMound()
        mound.add_item("item_1", "Shared content", "ws_source")

        # Share with user
        await mound.share_with_user(
            item_id="item_1",
            from_workspace_id="ws_source",
            user_id="user_1",
            shared_by="owner",
        )

        results = await mound.get_shared_with_me(
            workspace_id="ws_other",
            user_id="user_1",
        )

        assert len(results) == 1
        assert results[0].id == "item_1"

    @pytest.mark.asyncio
    async def test_get_shares_excludes_expired(self):
        """Should not return expired shares."""
        mound = MockKnowledgeMound()
        mound.add_item("item_1", "Shared content", "ws_source")

        # Share with past expiry
        expired = datetime.now() - timedelta(days=1)
        await mound.share_with_workspace(
            item_id="item_1",
            from_workspace_id="ws_source",
            to_workspace_id="ws_target",
            shared_by="admin",
            expires_at=expired,
        )

        results = await mound.get_shared_with_me(workspace_id="ws_target")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_get_shares_respects_limit(self):
        """Should respect limit parameter."""
        mound = MockKnowledgeMound()

        # Share multiple items
        for i in range(10):
            mound.add_item(f"item_{i}", f"Content {i}", "ws_source")
            await mound.share_with_workspace(
                item_id=f"item_{i}",
                from_workspace_id="ws_source",
                to_workspace_id="ws_target",
                shared_by="admin",
            )

        results = await mound.get_shared_with_me(
            workspace_id="ws_target",
            limit=3,
        )

        assert len(results) <= 3


# =============================================================================
# revoke_share Tests
# =============================================================================


class TestRevokeShare:
    """Tests for revoke_share method."""

    @pytest.mark.asyncio
    async def test_revoke_existing_share(self):
        """Should revoke existing share."""
        mound = MockKnowledgeMound()
        mound.add_item("item_1", "Content", "ws_source")

        await mound.share_with_workspace(
            item_id="item_1",
            from_workspace_id="ws_source",
            to_workspace_id="ws_target",
            shared_by="admin",
        )

        result = await mound.revoke_share(
            item_id="item_1",
            grantee_id="ws_target",
            revoked_by="admin",
        )

        assert result is True

        # Verify revoked
        grants = await mound._meta_store.get_access_grants_async("item_1")
        assert len(grants) == 0

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_share(self):
        """Should return False for nonexistent share."""
        mound = MockKnowledgeMound()

        result = await mound.revoke_share(
            item_id="item_1",
            grantee_id="ws_target",
            revoked_by="admin",
        )

        assert result is False


# =============================================================================
# get_share_grants Tests
# =============================================================================


class TestGetShareGrants:
    """Tests for get_share_grants method."""

    @pytest.mark.asyncio
    async def test_get_grants_empty(self):
        """Should return empty list when no grants."""
        mound = MockKnowledgeMound()
        grants = await mound.get_share_grants("item_1")
        assert grants == []

    @pytest.mark.asyncio
    async def test_get_grants_multiple(self):
        """Should return all grants for item."""
        mound = MockKnowledgeMound()
        mound.add_item("item_1", "Content", "ws_source")

        # Share with multiple grantees
        await mound.share_with_workspace(
            item_id="item_1",
            from_workspace_id="ws_source",
            to_workspace_id="ws_1",
            shared_by="admin",
        )
        await mound.share_with_workspace(
            item_id="item_1",
            from_workspace_id="ws_source",
            to_workspace_id="ws_2",
            shared_by="admin",
        )
        await mound.share_with_user(
            item_id="item_1",
            from_workspace_id="ws_source",
            user_id="user_1",
            shared_by="admin",
        )

        grants = await mound.get_share_grants("item_1")
        assert len(grants) == 3


# =============================================================================
# update_share_permissions Tests
# =============================================================================


class TestUpdateSharePermissions:
    """Tests for update_share_permissions method."""

    @pytest.mark.asyncio
    async def test_update_permissions(self):
        """Should update existing grant permissions."""
        mound = MockKnowledgeMound()
        mound.add_item("item_1", "Content", "ws_source")

        await mound.share_with_workspace(
            item_id="item_1",
            from_workspace_id="ws_source",
            to_workspace_id="ws_target",
            shared_by="admin",
            permissions=["read"],
        )

        updated = await mound.update_share_permissions(
            item_id="item_1",
            grantee_id="ws_target",
            new_permissions=["read", "write", "admin"],
            updated_by="admin",
        )

        assert updated is not None
        assert updated.permissions == ["read", "write", "admin"]

    @pytest.mark.asyncio
    async def test_update_nonexistent_grant(self):
        """Should return None for nonexistent grant."""
        mound = MockKnowledgeMound()

        result = await mound.update_share_permissions(
            item_id="item_1",
            grantee_id="nonexistent",
            new_permissions=["read"],
            updated_by="admin",
        )

        assert result is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestSharingIntegration:
    """Integration tests for sharing workflow."""

    @pytest.mark.asyncio
    async def test_full_sharing_workflow(self):
        """Test complete sharing workflow."""
        mound = MockKnowledgeMound()

        # Create item
        mound.add_item("research_doc", "Important research findings", "research_team")

        # Share with collaborators workspace
        grant = await mound.share_with_workspace(
            item_id="research_doc",
            from_workspace_id="research_team",
            to_workspace_id="collaborators",
            shared_by="lead_researcher",
            permissions=["read", "comment"],
        )

        # Verify collaborators can see it
        shared = await mound.get_shared_with_me(workspace_id="collaborators")
        assert len(shared) == 1
        assert shared[0].id == "research_doc"

        # Update to add write permission
        await mound.update_share_permissions(
            item_id="research_doc",
            grantee_id="collaborators",
            new_permissions=["read", "comment", "write"],
            updated_by="lead_researcher",
        )

        grants = await mound.get_share_grants("research_doc")
        assert grants[0].permissions == ["read", "comment", "write"]

        # Revoke access
        await mound.revoke_share(
            item_id="research_doc",
            grantee_id="collaborators",
            revoked_by="lead_researcher",
        )

        # Verify no longer shared
        shared = await mound.get_shared_with_me(workspace_id="collaborators")
        assert len(shared) == 0

    @pytest.mark.asyncio
    async def test_multi_grantee_sharing(self):
        """Test sharing with multiple grantees."""
        mound = MockKnowledgeMound()
        mound.add_item("doc_1", "Shared document", "owner_ws")

        # Share with workspaces and users
        await mound.share_with_workspace(
            item_id="doc_1",
            from_workspace_id="owner_ws",
            to_workspace_id="team_a",
            shared_by="owner",
        )
        await mound.share_with_workspace(
            item_id="doc_1",
            from_workspace_id="owner_ws",
            to_workspace_id="team_b",
            shared_by="owner",
        )
        await mound.share_with_user(
            item_id="doc_1",
            from_workspace_id="owner_ws",
            user_id="external_consultant",
            shared_by="owner",
        )

        # All should have access
        grants = await mound.get_share_grants("doc_1")
        assert len(grants) == 3

        grantee_ids = {g.grantee_id for g in grants}
        assert "team_a" in grantee_ids
        assert "team_b" in grantee_ids
        assert "external_consultant" in grantee_ids
