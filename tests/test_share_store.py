"""
Tests for the SQLite-backed ShareLinkStore.

Covers:
- Basic CRUD operations
- Token-based lookups
- Expiration and cleanup
- View count tracking
- Statistics
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from aragora.storage.share_store import ShareLinkStore
from aragora.server.handlers.sharing import ShareSettings, DebateVisibility


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_share_links.db"


@pytest.fixture
def store(temp_db):
    """Create a ShareLinkStore instance."""
    return ShareLinkStore(temp_db)


class TestShareLinkStoreBasics:
    """Test basic CRUD operations."""

    def test_save_and_get(self, store):
        """Test saving and retrieving settings."""
        settings = ShareSettings(
            debate_id="debate_123",
            visibility=DebateVisibility.PUBLIC,
            owner_id="user_1",
        )

        store.save(settings)
        retrieved = store.get("debate_123")

        assert retrieved is not None
        assert retrieved.debate_id == "debate_123"
        assert retrieved.visibility == DebateVisibility.PUBLIC
        assert retrieved.owner_id == "user_1"
        # Token should be generated
        assert retrieved.share_token is not None

    def test_save_preserves_token(self, store):
        """Test that explicit token is preserved."""
        settings = ShareSettings(
            debate_id="debate_456",
            visibility=DebateVisibility.PUBLIC,
            share_token="explicit_token_123",
        )

        store.save(settings)
        retrieved = store.get("debate_456")

        assert retrieved.share_token == "explicit_token_123"

    def test_get_nonexistent(self, store):
        """Test getting non-existent debate returns None."""
        result = store.get("nonexistent_id")
        assert result is None

    def test_update_existing(self, store):
        """Test updating existing settings."""
        settings = ShareSettings(
            debate_id="debate_update",
            visibility=DebateVisibility.PRIVATE,
        )
        store.save(settings)

        # Update
        settings.visibility = DebateVisibility.PUBLIC
        settings.allow_comments = True
        store.save(settings)

        retrieved = store.get("debate_update")
        assert retrieved.visibility == DebateVisibility.PUBLIC
        assert retrieved.allow_comments is True

    def test_delete(self, store):
        """Test deleting settings."""
        settings = ShareSettings(
            debate_id="debate_delete",
            visibility=DebateVisibility.PUBLIC,
        )
        store.save(settings)

        assert store.get("debate_delete") is not None

        result = store.delete("debate_delete")

        assert result is True
        assert store.get("debate_delete") is None

    def test_delete_nonexistent(self, store):
        """Test deleting non-existent returns False."""
        result = store.delete("nonexistent_id")
        assert result is False


class TestTokenOperations:
    """Test token-based operations."""

    def test_get_by_token(self, store):
        """Test retrieving by share token."""
        settings = ShareSettings(
            debate_id="debate_token",
            visibility=DebateVisibility.PUBLIC,
            share_token="my_token_abc",
        )
        store.save(settings)

        retrieved = store.get_by_token("my_token_abc")

        assert retrieved is not None
        assert retrieved.debate_id == "debate_token"

    def test_get_by_token_nonexistent(self, store):
        """Test getting by non-existent token returns None."""
        result = store.get_by_token("nonexistent_token")
        assert result is None

    def test_revoke_token(self, store):
        """Test revoking share token."""
        settings = ShareSettings(
            debate_id="debate_revoke",
            visibility=DebateVisibility.PUBLIC,
            share_token="token_to_revoke",
        )
        store.save(settings)

        result = store.revoke_token("debate_revoke")

        assert result is True

        # Token lookup should fail
        assert store.get_by_token("token_to_revoke") is None

        # But debate should still exist
        retrieved = store.get("debate_revoke")
        assert retrieved is not None
        # Token should be None after revocation
        assert retrieved.share_token is None

    def test_revoke_token_no_token(self, store):
        """Test revoking when no token exists returns False."""
        settings = ShareSettings(
            debate_id="debate_no_token",
            visibility=DebateVisibility.PRIVATE,
        )
        store.save(settings)

        # First revoke
        store.revoke_token("debate_no_token")

        # Second revoke should return False
        result = store.revoke_token("debate_no_token")
        assert result is False


class TestExpiration:
    """Test expiration handling."""

    def test_expired_link_still_returned(self, store):
        """Test that expired links are still returned (for error messages)."""
        settings = ShareSettings(
            debate_id="debate_expired",
            visibility=DebateVisibility.PUBLIC,
            expires_at=time.time() - 3600,  # Expired 1 hour ago
        )
        store.save(settings)

        retrieved = store.get("debate_expired")

        assert retrieved is not None
        assert retrieved.is_expired is True

    def test_cleanup_expired(self, store):
        """Test cleanup removes expired links."""
        # Create expired link
        expired_settings = ShareSettings(
            debate_id="debate_cleanup_expired",
            visibility=DebateVisibility.PUBLIC,
            expires_at=time.time() - 3600,
        )
        store.save(expired_settings)

        # Create non-expired link
        valid_settings = ShareSettings(
            debate_id="debate_cleanup_valid",
            visibility=DebateVisibility.PUBLIC,
            expires_at=time.time() + 3600,
        )
        store.save(valid_settings)

        # Create link with no expiration
        permanent_settings = ShareSettings(
            debate_id="debate_cleanup_permanent",
            visibility=DebateVisibility.PUBLIC,
        )
        store.save(permanent_settings)

        removed = store.cleanup_expired()

        assert removed == 1
        assert store.get("debate_cleanup_expired") is None
        assert store.get("debate_cleanup_valid") is not None
        assert store.get("debate_cleanup_permanent") is not None


class TestViewCounts:
    """Test view count tracking."""

    def test_increment_view_count(self, store):
        """Test incrementing view count."""
        settings = ShareSettings(
            debate_id="debate_views",
            visibility=DebateVisibility.PUBLIC,
            view_count=0,
        )
        store.save(settings)

        store.increment_view_count("debate_views")
        store.increment_view_count("debate_views")
        store.increment_view_count("debate_views")

        retrieved = store.get("debate_views")
        assert retrieved.view_count == 3

    def test_increment_nonexistent(self, store):
        """Test incrementing non-existent debate doesn't error."""
        # Should not raise
        store.increment_view_count("nonexistent")


class TestStatistics:
    """Test statistics collection."""

    def test_get_stats_empty(self, store):
        """Test stats on empty store."""
        stats = store.get_stats()

        assert stats["total"] == 0
        assert stats["with_tokens"] == 0
        assert stats["total_views"] == 0

    def test_get_stats_with_data(self, store):
        """Test stats with data."""
        # Create various settings (store generates tokens for all)
        store.save(ShareSettings(
            debate_id="d1",
            visibility=DebateVisibility.PUBLIC,
            share_token="t1",
            view_count=10,
        ))
        store.save(ShareSettings(
            debate_id="d2",
            visibility=DebateVisibility.PUBLIC,
            share_token="t2",
            view_count=5,
        ))
        store.save(ShareSettings(
            debate_id="d3",
            visibility=DebateVisibility.PRIVATE,
        ))
        store.save(ShareSettings(
            debate_id="d4",
            visibility=DebateVisibility.TEAM,
            share_token="t4",
        ))

        stats = store.get_stats()

        assert stats["total"] == 4
        assert stats["by_visibility"]["public"] == 2
        assert stats["by_visibility"]["private"] == 1
        assert stats["by_visibility"]["team"] == 1
        # Store generates tokens for all entries when None
        assert stats["with_tokens"] == 4
        assert stats["total_views"] == 15


class TestVisibilityLevels:
    """Test different visibility levels."""

    @pytest.mark.parametrize("visibility", [
        DebateVisibility.PRIVATE,
        DebateVisibility.TEAM,
        DebateVisibility.PUBLIC,
    ])
    def test_visibility_preserved(self, store, visibility):
        """Test that visibility level is preserved."""
        settings = ShareSettings(
            debate_id=f"debate_{visibility.value}",
            visibility=visibility,
        )
        store.save(settings)

        retrieved = store.get(f"debate_{visibility.value}")
        assert retrieved.visibility == visibility


class TestAllFieldsPersistence:
    """Test that all fields are properly persisted."""

    def test_all_fields_preserved(self, store):
        """Test all ShareSettings fields are saved and restored."""
        now = time.time()
        settings = ShareSettings(
            debate_id="debate_full",
            visibility=DebateVisibility.PUBLIC,
            share_token="full_token",
            created_at=now,
            expires_at=now + 86400,
            allow_comments=True,
            allow_forking=True,
            view_count=42,
            owner_id="owner_123",
            org_id="org_456",
        )

        store.save(settings)
        retrieved = store.get("debate_full")

        assert retrieved.debate_id == "debate_full"
        assert retrieved.visibility == DebateVisibility.PUBLIC
        assert retrieved.share_token == "full_token"
        assert retrieved.created_at == now
        assert retrieved.expires_at == now + 86400
        assert retrieved.allow_comments is True
        assert retrieved.allow_forking is True
        assert retrieved.view_count == 42
        assert retrieved.owner_id == "owner_123"
        assert retrieved.org_id == "org_456"


class TestConcurrentAccess:
    """Test concurrent access patterns."""

    def test_unique_debate_id_constraint(self, store):
        """Test that debate_id uniqueness is enforced."""
        settings1 = ShareSettings(
            debate_id="unique_test",
            visibility=DebateVisibility.PRIVATE,
            share_token="token1",
        )
        store.save(settings1)

        # Save again with different data
        settings2 = ShareSettings(
            debate_id="unique_test",
            visibility=DebateVisibility.PUBLIC,
            share_token="token2",
        )
        store.save(settings2)

        # Should have updated, not created duplicate
        retrieved = store.get("unique_test")
        assert retrieved.visibility == DebateVisibility.PUBLIC
        # Token should be preserved (COALESCE in update)
        assert retrieved.share_token in ("token1", "token2")

        # Verify only one record
        stats = store.get_stats()
        assert stats["total"] == 1
