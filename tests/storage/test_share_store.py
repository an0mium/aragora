"""
Unit tests for ShareLinkStore.

Tests SQLite-backed share link persistence with TTL cleanup.
"""

import pytest
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from aragora.storage.share_store import ShareLinkStore


# Mock ShareSettings and DebateVisibility for isolated testing
class MockDebateVisibility:
    """Mock visibility enum."""

    PRIVATE = "private"
    TEAM = "team"
    PUBLIC = "public"

    def __init__(self, value):
        self.value = value


class MockShareSettings:
    """Mock ShareSettings for testing without circular imports."""

    def __init__(
        self,
        debate_id: str,
        visibility: str = "private",
        share_token: str = None,
        owner_id: str = None,
        org_id: str = None,
        created_at: float = None,
        expires_at: float = None,
        allow_comments: bool = False,
        allow_forking: bool = False,
        view_count: int = 0,
    ):
        self.debate_id = debate_id
        self.visibility = MockDebateVisibility(visibility)
        self.share_token = share_token
        self.owner_id = owner_id
        self.org_id = org_id
        self.created_at = created_at or time.time()
        self.expires_at = expires_at
        self.allow_comments = allow_comments
        self.allow_forking = allow_forking
        self.view_count = view_count


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield Path(f.name)
    # Cleanup
    try:
        Path(f.name).unlink()
        # Also clean up WAL and SHM files
        Path(f.name + "-wal").unlink(missing_ok=True)
        Path(f.name + "-shm").unlink(missing_ok=True)
    except FileNotFoundError:
        pass


@pytest.fixture
def store(temp_db):
    """Create a ShareLinkStore instance."""
    return ShareLinkStore(temp_db, cleanup_interval=60)


@pytest.fixture
def memory_store(tmp_path):
    """Create a ShareLinkStore with temp file (WAL doesn't work with :memory:)."""
    db_path = tmp_path / "test_share_links.db"
    return ShareLinkStore(db_path, cleanup_interval=60)


class TestShareLinkStoreBasics:
    """Test basic CRUD operations."""

    def test_save_and_get(self, memory_store):
        """Should save and retrieve share settings."""
        settings = MockShareSettings(
            debate_id="debate-123",
            visibility="public",
            share_token="token123abc456def",
            owner_id="user-1",
        )

        memory_store.save(settings)

        retrieved = memory_store.get("debate-123")
        assert retrieved is not None
        assert retrieved.debate_id == "debate-123"
        assert retrieved.visibility.value == "public"
        assert retrieved.owner_id == "user-1"

    def test_get_nonexistent(self, memory_store):
        """Should return None for nonexistent debate."""
        result = memory_store.get("nonexistent-id")
        assert result is None

    def test_get_by_token(self, memory_store):
        """Should retrieve settings by share token."""
        settings = MockShareSettings(
            debate_id="debate-456",
            visibility="public",
            share_token="unique_token_12345",
        )

        memory_store.save(settings)

        retrieved = memory_store.get_by_token("unique_token_12345")
        assert retrieved is not None
        assert retrieved.debate_id == "debate-456"

    def test_get_by_token_nonexistent(self, memory_store):
        """Should return None for nonexistent token."""
        result = memory_store.get_by_token("nonexistent-token")
        assert result is None

    def test_delete(self, memory_store):
        """Should delete share settings."""
        settings = MockShareSettings(
            debate_id="debate-to-delete",
            visibility="public",
        )

        memory_store.save(settings)
        assert memory_store.get("debate-to-delete") is not None

        deleted = memory_store.delete("debate-to-delete")
        assert deleted is True
        assert memory_store.get("debate-to-delete") is None

    def test_delete_nonexistent(self, memory_store):
        """Should return False when deleting nonexistent."""
        deleted = memory_store.delete("nonexistent-id")
        assert deleted is False


class TestShareLinkStoreUpsert:
    """Test upsert (insert or update) behavior."""

    def test_update_existing(self, memory_store):
        """Should update existing settings on conflict."""
        # Initial save
        settings1 = MockShareSettings(
            debate_id="debate-upsert",
            visibility="private",
            share_token="token1_abcdefghij",
            allow_comments=False,
        )
        memory_store.save(settings1)

        # Update with new values
        settings2 = MockShareSettings(
            debate_id="debate-upsert",
            visibility="public",
            share_token="token2_klmnopqrst",
            allow_comments=True,
        )
        memory_store.save(settings2)

        # Verify update
        retrieved = memory_store.get("debate-upsert")
        assert retrieved.visibility.value == "public"
        assert retrieved.allow_comments is True

    def test_generates_token_if_missing(self, memory_store):
        """Should generate token if not provided."""
        settings = MockShareSettings(
            debate_id="debate-no-token",
            visibility="public",
            share_token=None,
        )

        memory_store.save(settings)

        retrieved = memory_store.get("debate-no-token")
        assert retrieved.share_token is not None
        assert len(retrieved.share_token) >= 16


class TestShareLinkStoreTokenRevocation:
    """Test token revocation functionality."""

    def test_revoke_token(self, memory_store):
        """Should revoke (nullify) share token."""
        settings = MockShareSettings(
            debate_id="debate-revoke",
            visibility="public",
            share_token="token_to_revoke_123",
        )

        memory_store.save(settings)

        # Verify token works
        assert memory_store.get_by_token("token_to_revoke_123") is not None

        # Revoke
        revoked = memory_store.revoke_token("debate-revoke")
        assert revoked is True

        # Token should no longer work
        assert memory_store.get_by_token("token_to_revoke_123") is None

        # But debate settings should still exist
        retrieved = memory_store.get("debate-revoke")
        assert retrieved is not None
        assert retrieved.share_token is None

    def test_revoke_nonexistent(self, memory_store):
        """Should return False when revoking nonexistent."""
        revoked = memory_store.revoke_token("nonexistent-debate")
        assert revoked is False


class TestShareLinkStoreViewCount:
    """Test view count functionality."""

    def test_increment_view_count(self, memory_store):
        """Should increment view count atomically."""
        settings = MockShareSettings(
            debate_id="debate-views",
            visibility="public",
            view_count=0,
        )

        memory_store.save(settings)

        # Increment multiple times
        memory_store.increment_view_count("debate-views")
        memory_store.increment_view_count("debate-views")
        memory_store.increment_view_count("debate-views")

        retrieved = memory_store.get("debate-views")
        assert retrieved.view_count == 3

    def test_increment_updates_last_viewed(self, memory_store):
        """Should update last_viewed_at timestamp."""
        settings = MockShareSettings(
            debate_id="debate-timestamp",
            visibility="public",
        )

        memory_store.save(settings)

        before = time.time()
        memory_store.increment_view_count("debate-timestamp")
        after = time.time()

        # Check last_viewed_at via raw query
        row = memory_store.fetch_one(
            "SELECT last_viewed_at FROM share_links WHERE debate_id = ?", ("debate-timestamp",)
        )
        assert row is not None
        assert before <= row[0] <= after


class TestShareLinkStoreTTLCleanup:
    """Test TTL-based cleanup functionality."""

    def test_cleanup_expired(self, memory_store):
        """Should remove expired share links."""
        # Create expired link
        expired_settings = MockShareSettings(
            debate_id="debate-expired",
            visibility="public",
            expires_at=time.time() - 3600,  # Expired 1 hour ago
        )
        memory_store.save(expired_settings)

        # Create valid link
        valid_settings = MockShareSettings(
            debate_id="debate-valid",
            visibility="public",
            expires_at=time.time() + 3600,  # Expires in 1 hour
        )
        memory_store.save(valid_settings)

        # Create link with no expiration
        no_expire_settings = MockShareSettings(
            debate_id="debate-no-expire",
            visibility="public",
            expires_at=None,
        )
        memory_store.save(no_expire_settings)

        # Run cleanup
        removed = memory_store.cleanup_expired()
        assert removed == 1

        # Verify correct links remain
        assert memory_store.get("debate-expired") is None
        assert memory_store.get("debate-valid") is not None
        assert memory_store.get("debate-no-expire") is not None

    def test_maybe_cleanup_respects_interval(self, memory_store):
        """Should only cleanup if interval has passed."""
        # Set last cleanup to now
        memory_store._last_cleanup = time.time()

        # Create expired link
        expired_settings = MockShareSettings(
            debate_id="debate-maybe-expired",
            visibility="public",
            expires_at=time.time() - 3600,
        )
        memory_store.save(expired_settings)

        # _maybe_cleanup should not run since interval hasn't passed
        # The link should still exist
        assert memory_store.get("debate-maybe-expired") is not None


class TestShareLinkStoreStats:
    """Test statistics functionality."""

    def test_get_stats(self, memory_store):
        """Should return accurate statistics."""
        # Create various links
        for i in range(3):
            memory_store.save(
                MockShareSettings(
                    debate_id=f"public-{i}",
                    visibility="public",
                    share_token=f"token_public_{i}_abc",
                )
            )

        for i in range(2):
            memory_store.save(
                MockShareSettings(
                    debate_id=f"private-{i}",
                    visibility="private",
                    share_token=None,
                )
            )

        # Create one expired
        memory_store.save(
            MockShareSettings(
                debate_id="expired-1",
                visibility="public",
                expires_at=time.time() - 100,
            )
        )

        stats = memory_store.get_stats()

        assert stats["total"] == 6
        assert stats["by_visibility"]["public"] == 4
        assert stats["by_visibility"]["private"] == 2
        assert stats["with_tokens"] >= 3
        assert stats["expired"] == 1

    def test_get_stats_empty(self, memory_store):
        """Should handle empty database."""
        stats = memory_store.get_stats()

        assert stats["total"] == 0
        assert stats["by_visibility"] == {}
        assert stats["with_tokens"] == 0
        assert stats["expired"] == 0
        assert stats["total_views"] == 0


class TestShareLinkStorePersistence:
    """Test file-based persistence."""

    def test_persists_across_instances(self, temp_db):
        """Should persist data across store instances."""
        # Create and save with first instance
        store1 = ShareLinkStore(temp_db)
        store1.save(
            MockShareSettings(
                debate_id="persistent-debate",
                visibility="public",
                share_token="persistent_token_1",
                owner_id="user-persistent",
            )
        )
        del store1

        # Load with second instance
        store2 = ShareLinkStore(temp_db)
        retrieved = store2.get("persistent-debate")

        assert retrieved is not None
        assert retrieved.debate_id == "persistent-debate"
        assert retrieved.owner_id == "user-persistent"

    def test_schema_initialization(self, temp_db):
        """Should create schema on first initialization."""
        store = ShareLinkStore(temp_db)

        # Check tables exist
        row = store.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='share_links'"
        )
        assert row is not None

        # Check indexes exist
        row = store.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_share_links_debate'"
        )
        assert row is not None


class TestShareLinkStoreVisibility:
    """Test visibility-specific behavior."""

    def test_all_visibility_levels(self, memory_store):
        """Should handle all visibility levels correctly."""
        visibilities = ["private", "team", "public"]

        for vis in visibilities:
            memory_store.save(
                MockShareSettings(
                    debate_id=f"debate-{vis}",
                    visibility=vis,
                )
            )

        for vis in visibilities:
            retrieved = memory_store.get(f"debate-{vis}")
            assert retrieved is not None
            assert retrieved.visibility.value == vis


class TestShareLinkStoreEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_special_characters_in_debate_id(self, memory_store):
        """Should handle special characters in debate ID."""
        # Note: In practice, IDs should be validated before reaching the store
        settings = MockShareSettings(
            debate_id="debate-with-dashes_and_underscores",
            visibility="public",
        )

        memory_store.save(settings)
        retrieved = memory_store.get("debate-with-dashes_and_underscores")
        assert retrieved is not None

    def test_handles_long_debate_id(self, memory_store):
        """Should handle reasonably long debate IDs."""
        long_id = "debate-" + "x" * 100
        settings = MockShareSettings(
            debate_id=long_id,
            visibility="public",
        )

        memory_store.save(settings)
        retrieved = memory_store.get(long_id)
        assert retrieved is not None
        assert retrieved.debate_id == long_id

    def test_concurrent_increments(self, memory_store):
        """Should handle concurrent view count increments."""
        settings = MockShareSettings(
            debate_id="debate-concurrent",
            visibility="public",
            view_count=0,
        )
        memory_store.save(settings)

        # Simulate concurrent increments
        import threading

        def increment():
            for _ in range(10):
                memory_store.increment_view_count("debate-concurrent")

        threads = [threading.Thread(target=increment) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        retrieved = memory_store.get("debate-concurrent")
        assert retrieved.view_count == 50  # 5 threads * 10 increments


class TestShareLinkStoreIntegration:
    """Integration tests with real ShareSettings."""

    def test_with_real_share_settings(self, memory_store):
        """Should work with actual ShareSettings class."""
        try:
            from aragora.server.handlers.social.sharing import ShareSettings, DebateVisibility

            settings = ShareSettings(
                debate_id="real-debate-123",
                visibility=DebateVisibility.PUBLIC,
                share_token="real_token_abcdef",
                owner_id="real-user-1",
                org_id="real-org-1",
                allow_comments=True,
                allow_forking=True,
            )

            memory_store.save(settings)

            retrieved = memory_store.get("real-debate-123")
            assert retrieved is not None
            assert retrieved.debate_id == "real-debate-123"
            assert retrieved.visibility == DebateVisibility.PUBLIC
            assert retrieved.owner_id == "real-user-1"
            assert retrieved.allow_comments is True
            assert retrieved.allow_forking is True

        except ImportError:
            pytest.skip("ShareSettings not available")
