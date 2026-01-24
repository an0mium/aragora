"""
Tests for TeamsConversationStore - Teams conversation reference storage.

Tests cover:
- TeamsConversationReference dataclass
- Conversation reference CRUD operations
- Tenant-based queries
- Cleanup of old references
- Bot Framework format conversion
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aragora.connectors.chat.teams_conversations import (
    TeamsConversationReference,
    TeamsConversationStore,
    StoredConversation,
    get_teams_conversation_store,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield str(Path(tmpdir) / "test_teams_conversations.db")


@pytest.fixture
def conversation_store(temp_db_path):
    """Create a conversation store for testing."""
    return TeamsConversationStore(db_path=temp_db_path)


@pytest.fixture
def sample_reference():
    """Create a sample conversation reference."""
    return TeamsConversationReference(
        conversation_id="conv-id-123",
        service_url="https://smba.trafficmanager.net/teams/",
        channel_id="channel-456",
        tenant_id="tenant-789",
        bot_id="28:bot-id-000",
        activity_id="activity-111",
        user_id="29:user-id-222",
        message_id="msg-333",
        metadata={
            "conversation_name": "Test Conversation",
            "channel_name": "General",
            "team_id": "team-444",
            "team_name": "Test Team",
        },
    )


@pytest.fixture
def sample_activity():
    """Create a sample Bot Framework activity."""
    return {
        "type": "message",
        "id": "activity-123",
        "serviceUrl": "https://smba.trafficmanager.net/teams/",
        "from": {
            "id": "29:user-id-123",
            "name": "Test User",
        },
        "conversation": {
            "id": "conv-id-456",
            "tenantId": "tenant-id-789",
            "name": "Test Conversation",
        },
        "channelData": {
            "tenant": {"id": "tenant-id-789"},
            "channel": {"id": "channel-id-abc", "name": "General"},
            "team": {"id": "team-id-def", "name": "Test Team"},
        },
        "recipient": {
            "id": "28:bot-id-000",
            "name": "Aragora Bot",
        },
        "replyToId": "reply-to-msg-555",
    }


# ===========================================================================
# TeamsConversationReference Tests
# ===========================================================================


class TestTeamsConversationReference:
    """Tests for TeamsConversationReference dataclass."""

    def test_to_dict(self, sample_reference):
        """Test to_dict includes all fields."""
        result = sample_reference.to_dict()

        assert result["conversation_id"] == "conv-id-123"
        assert result["service_url"] == "https://smba.trafficmanager.net/teams/"
        assert result["channel_id"] == "channel-456"
        assert result["tenant_id"] == "tenant-789"
        assert result["bot_id"] == "28:bot-id-000"
        assert result["activity_id"] == "activity-111"
        assert result["user_id"] == "29:user-id-222"
        assert result["message_id"] == "msg-333"
        assert result["metadata"]["team_name"] == "Test Team"

    def test_to_dict_with_none_fields(self):
        """Test to_dict handles None optional fields."""
        ref = TeamsConversationReference(
            conversation_id="conv-123",
            service_url="https://example.com/",
            tenant_id="tenant-456",
            bot_id="bot-789",
        )

        result = ref.to_dict()

        assert result["conversation_id"] == "conv-123"
        assert result["channel_id"] is None
        assert result["activity_id"] is None
        assert result["metadata"] == {}

    def test_to_bot_framework_reference_channel(self, sample_reference):
        """Test conversion to Bot Framework format for channel message."""
        result = sample_reference.to_bot_framework_reference()

        assert result["conversation"]["id"] == "conv-id-123"
        assert result["conversation"]["tenantId"] == "tenant-789"
        assert result["conversation"]["conversationType"] == "channel"
        assert result["serviceUrl"] == "https://smba.trafficmanager.net/teams/"
        assert result["channelId"] == "channel-456"
        assert result["bot"]["id"] == "28:bot-id-000"
        assert result["user"]["id"] == "29:user-id-222"
        assert result["activityId"] == "activity-111"

    def test_to_bot_framework_reference_personal(self):
        """Test conversion to Bot Framework format for personal chat."""
        ref = TeamsConversationReference(
            conversation_id="conv-123",
            service_url="https://example.com/",
            tenant_id="tenant-456",
            bot_id="bot-789",
            user_id="user-111",
            channel_id=None,
        )

        result = ref.to_bot_framework_reference()

        assert result["conversation"]["conversationType"] == "personal"
        assert result["user"]["id"] == "user-111"
        assert "channelId" not in result or result["channelId"] == "msteams"

    def test_to_bot_framework_reference_group_chat(self):
        """Test conversion to Bot Framework format for group chat."""
        ref = TeamsConversationReference(
            conversation_id="conv-123",
            service_url="https://example.com/",
            tenant_id="tenant-456",
            bot_id="bot-789",
            user_id=None,
            channel_id=None,
        )

        result = ref.to_bot_framework_reference()

        assert result["conversation"]["conversationType"] == "groupChat"

    def test_from_activity(self, sample_activity):
        """Test creating reference from Bot Framework activity."""
        result = TeamsConversationReference.from_activity(sample_activity)

        assert result.conversation_id == "conv-id-456"
        assert result.service_url == "https://smba.trafficmanager.net/teams/"
        assert result.tenant_id == "tenant-id-789"
        assert result.channel_id == "channel-id-abc"
        assert result.bot_id == "28:bot-id-000"
        assert result.activity_id == "activity-123"
        assert result.user_id == "29:user-id-123"
        assert result.message_id == "reply-to-msg-555"
        assert result.metadata["conversation_name"] == "Test Conversation"
        assert result.metadata["channel_name"] == "General"
        assert result.metadata["team_id"] == "team-id-def"
        assert result.metadata["team_name"] == "Test Team"

    def test_from_activity_minimal(self):
        """Test creating reference from minimal activity."""
        activity = {
            "conversation": {"id": "conv-123"},
            "serviceUrl": "https://example.com/",
            "recipient": {"id": "bot-456"},
            "channelData": {},
        }

        result = TeamsConversationReference.from_activity(activity)

        assert result.conversation_id == "conv-123"
        assert result.service_url == "https://example.com/"
        assert result.bot_id == "bot-456"
        assert result.tenant_id == ""
        assert result.channel_id is None

    def test_from_activity_uses_channel_data_tenant(self):
        """Test extracts tenant from channelData when not in conversation."""
        activity = {
            "conversation": {"id": "conv-123"},
            "serviceUrl": "https://example.com/",
            "recipient": {},
            "channelData": {"tenant": {"id": "channel-data-tenant"}},
        }

        result = TeamsConversationReference.from_activity(activity)

        assert result.tenant_id == "channel-data-tenant"


# ===========================================================================
# StoredConversation Tests
# ===========================================================================


class TestStoredConversation:
    """Tests for StoredConversation dataclass."""

    def test_to_dict(self, sample_reference):
        """Test to_dict includes all fields."""
        now = time.time()
        stored = StoredConversation(
            debate_id="debate-123",
            reference=sample_reference,
            created_at=now,
            updated_at=now,
        )

        result = stored.to_dict()

        assert result["debate_id"] == "debate-123"
        assert "reference" in result
        assert result["reference"]["conversation_id"] == "conv-id-123"
        assert result["created_at"] == now
        assert "created_at_iso" in result
        assert "T" in result["created_at_iso"]  # ISO format


# ===========================================================================
# TeamsConversationStore Initialization Tests
# ===========================================================================


class TestTeamsConversationStoreInit:
    """Tests for TeamsConversationStore initialization."""

    def test_init_with_custom_path(self, temp_db_path):
        """Test initialization with custom database path."""
        store = TeamsConversationStore(db_path=temp_db_path)

        assert store._db_path == temp_db_path
        assert store._initialized is False

    def test_init_default_path(self):
        """Test initialization uses default path when not specified."""
        store = TeamsConversationStore()

        assert ".aragora" in store._db_path
        assert "teams_conversations.db" in store._db_path

    def test_schema_created_on_first_access(self, temp_db_path):
        """Test schema is created when database is first accessed."""
        store = TeamsConversationStore(db_path=temp_db_path)

        # Access database
        conn = store._get_connection()

        # Check schema exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='teams_conversations'"
        )
        assert cursor.fetchone() is not None

        # Check indexes exist
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_teams_conv_tenant'"
        )
        assert cursor.fetchone() is not None


# ===========================================================================
# TeamsConversationStore CRUD Tests
# ===========================================================================


class TestTeamsConversationStoreCRUD:
    """Tests for CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_reference(self, conversation_store, sample_reference):
        """Test save and retrieve a conversation reference."""
        debate_id = "debate-123"

        saved = await conversation_store.save_reference(debate_id, sample_reference)
        assert saved is True

        retrieved = await conversation_store.get_reference(debate_id)
        assert retrieved is not None
        assert retrieved.conversation_id == "conv-id-123"
        assert retrieved.service_url == "https://smba.trafficmanager.net/teams/"
        assert retrieved.tenant_id == "tenant-789"
        assert retrieved.channel_id == "channel-456"
        assert retrieved.metadata["team_name"] == "Test Team"

    @pytest.mark.asyncio
    async def test_get_reference_not_found(self, conversation_store):
        """Test get returns None for nonexistent reference."""
        result = await conversation_store.get_reference("nonexistent-debate")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_updates_existing(self, conversation_store, sample_reference):
        """Test save updates existing reference (upsert)."""
        debate_id = "debate-123"

        # Save initial
        await conversation_store.save_reference(debate_id, sample_reference)

        # Update with new data
        updated_reference = TeamsConversationReference(
            conversation_id="new-conv-id",
            service_url="https://new-service.com/",
            tenant_id="tenant-789",
            bot_id="new-bot-id",
        )
        await conversation_store.save_reference(debate_id, updated_reference)

        # Verify updated
        retrieved = await conversation_store.get_reference(debate_id)
        assert retrieved.conversation_id == "new-conv-id"
        assert retrieved.service_url == "https://new-service.com/"

    @pytest.mark.asyncio
    async def test_delete_reference(self, conversation_store, sample_reference):
        """Test delete removes reference."""
        debate_id = "debate-123"

        await conversation_store.save_reference(debate_id, sample_reference)
        assert await conversation_store.get_reference(debate_id) is not None

        deleted = await conversation_store.delete_reference(debate_id)
        assert deleted is True

        assert await conversation_store.get_reference(debate_id) is None

    @pytest.mark.asyncio
    async def test_delete_reference_not_found(self, conversation_store):
        """Test delete returns False for nonexistent reference."""
        result = await conversation_store.delete_reference("nonexistent-debate")
        assert result is False


# ===========================================================================
# TeamsConversationStore Query Tests
# ===========================================================================


class TestTeamsConversationStoreQueries:
    """Tests for query operations."""

    @pytest.mark.asyncio
    async def test_get_by_tenant(self, conversation_store):
        """Test get_by_tenant returns conversations for specific tenant."""
        # Add conversations for tenant-001
        for i in range(3):
            ref = TeamsConversationReference(
                conversation_id=f"conv-t1-{i}",
                service_url="https://example.com/",
                tenant_id="tenant-001",
                bot_id="bot-123",
            )
            await conversation_store.save_reference(f"debate-t1-{i}", ref)

        # Add conversations for tenant-002
        ref = TeamsConversationReference(
            conversation_id="conv-t2-0",
            service_url="https://example.com/",
            tenant_id="tenant-002",
            bot_id="bot-123",
        )
        await conversation_store.save_reference("debate-t2-0", ref)

        # Query by tenant
        tenant1_results = await conversation_store.get_by_tenant("tenant-001")
        tenant2_results = await conversation_store.get_by_tenant("tenant-002")

        assert len(tenant1_results) == 3
        assert len(tenant2_results) == 1
        assert all(r.reference.tenant_id == "tenant-001" for r in tenant1_results)

    @pytest.mark.asyncio
    async def test_get_by_tenant_with_limit(self, conversation_store):
        """Test get_by_tenant respects limit."""
        for i in range(10):
            ref = TeamsConversationReference(
                conversation_id=f"conv-{i}",
                service_url="https://example.com/",
                tenant_id="tenant-001",
                bot_id="bot-123",
            )
            await conversation_store.save_reference(f"debate-{i}", ref)

        results = await conversation_store.get_by_tenant("tenant-001", limit=5)

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_get_by_tenant_ordered_by_created_at(self, conversation_store):
        """Test get_by_tenant returns results ordered by created_at DESC."""
        # Add with small delay to ensure different timestamps
        for i in range(3):
            ref = TeamsConversationReference(
                conversation_id=f"conv-{i}",
                service_url="https://example.com/",
                tenant_id="tenant-001",
                bot_id="bot-123",
            )
            await conversation_store.save_reference(f"debate-{i}", ref)

        results = await conversation_store.get_by_tenant("tenant-001")

        # Most recently created should be first
        assert len(results) == 3
        # Verify ordering (later created_at should come first)
        for i in range(len(results) - 1):
            assert results[i].created_at >= results[i + 1].created_at

    @pytest.mark.asyncio
    async def test_get_by_tenant_empty(self, conversation_store):
        """Test get_by_tenant returns empty list for unknown tenant."""
        results = await conversation_store.get_by_tenant("unknown-tenant")
        assert results == []


# ===========================================================================
# TeamsConversationStore Cleanup Tests
# ===========================================================================


class TestTeamsConversationStoreCleanup:
    """Tests for cleanup operations."""

    @pytest.mark.asyncio
    async def test_cleanup_old_removes_expired(self, temp_db_path):
        """Test cleanup_old removes references older than max_age_days."""
        store = TeamsConversationStore(db_path=temp_db_path)

        # Directly insert old record
        conn = store._get_connection()
        old_time = time.time() - (40 * 86400)  # 40 days ago

        conn.execute(
            """
            INSERT INTO teams_conversations
            (debate_id, conversation_id, service_url, channel_id, tenant_id,
             bot_id, activity_id, user_id, message_id, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "old-debate",
                "old-conv",
                "https://example.com/",
                None,
                "tenant-123",
                "bot-456",
                None,
                None,
                None,
                old_time,
                old_time,
                "{}",
            ),
        )
        conn.commit()

        # Add recent record
        ref = TeamsConversationReference(
            conversation_id="recent-conv",
            service_url="https://example.com/",
            tenant_id="tenant-123",
            bot_id="bot-456",
        )
        await store.save_reference("recent-debate", ref)

        # Cleanup (30 days)
        deleted_count = await store.cleanup_old(max_age_days=30)

        assert deleted_count == 1
        assert await store.get_reference("old-debate") is None
        assert await store.get_reference("recent-debate") is not None

    @pytest.mark.asyncio
    async def test_cleanup_old_no_expired(self, conversation_store, sample_reference):
        """Test cleanup_old returns 0 when nothing to clean."""
        await conversation_store.save_reference("recent-debate", sample_reference)

        deleted_count = await conversation_store.cleanup_old(max_age_days=30)

        assert deleted_count == 0

    @pytest.mark.asyncio
    async def test_cleanup_old_custom_age(self, temp_db_path):
        """Test cleanup_old uses custom max_age_days."""
        store = TeamsConversationStore(db_path=temp_db_path)

        # Insert record from 10 days ago
        conn = store._get_connection()
        ten_days_ago = time.time() - (10 * 86400)

        conn.execute(
            """
            INSERT INTO teams_conversations
            (debate_id, conversation_id, service_url, channel_id, tenant_id,
             bot_id, activity_id, user_id, message_id, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "old-debate",
                "old-conv",
                "https://example.com/",
                None,
                "tenant-123",
                "bot-456",
                None,
                None,
                None,
                ten_days_ago,
                ten_days_ago,
                "{}",
            ),
        )
        conn.commit()

        # Cleanup with 7 days should remove it
        deleted_count = await store.cleanup_old(max_age_days=7)
        assert deleted_count == 1

        # Re-add and cleanup with 15 days should not remove it
        conn.execute(
            """
            INSERT INTO teams_conversations
            (debate_id, conversation_id, service_url, channel_id, tenant_id,
             bot_id, activity_id, user_id, message_id, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "old-debate-2",
                "old-conv",
                "https://example.com/",
                None,
                "tenant-123",
                "bot-456",
                None,
                None,
                None,
                ten_days_ago,
                ten_days_ago,
                "{}",
            ),
        )
        conn.commit()

        deleted_count = await store.cleanup_old(max_age_days=15)
        assert deleted_count == 0


# ===========================================================================
# TeamsConversationStore Error Handling Tests
# ===========================================================================


class TestTeamsConversationStoreErrors:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_save_handles_error(self, temp_db_path, sample_reference):
        """Test save handles database errors gracefully."""
        store = TeamsConversationStore(db_path=temp_db_path)

        # Replace connection with mock that raises
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = Exception("DB error")
        store._local.connection = mock_conn

        result = await store.save_reference("debate-123", sample_reference)

        assert result is False

    @pytest.mark.asyncio
    async def test_get_handles_error(self, temp_db_path):
        """Test get handles database errors gracefully."""
        store = TeamsConversationStore(db_path=temp_db_path)

        mock_conn = MagicMock()
        mock_conn.execute.side_effect = Exception("DB error")
        store._local.connection = mock_conn

        result = await store.get_reference("debate-123")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_tenant_handles_error(self, temp_db_path):
        """Test get_by_tenant handles database errors gracefully."""
        store = TeamsConversationStore(db_path=temp_db_path)

        mock_conn = MagicMock()
        mock_conn.execute.side_effect = Exception("DB error")
        store._local.connection = mock_conn

        result = await store.get_by_tenant("tenant-123")

        assert result == []

    @pytest.mark.asyncio
    async def test_cleanup_handles_error(self, temp_db_path):
        """Test cleanup_old handles database errors gracefully."""
        store = TeamsConversationStore(db_path=temp_db_path)

        mock_conn = MagicMock()
        mock_conn.execute.side_effect = Exception("DB error")
        store._local.connection = mock_conn

        result = await store.cleanup_old()

        assert result == 0


# ===========================================================================
# TeamsConversationStore Metadata Tests
# ===========================================================================


class TestTeamsConversationStoreMetadata:
    """Tests for metadata handling."""

    @pytest.mark.asyncio
    async def test_metadata_preserved(self, conversation_store):
        """Test metadata is preserved through save/get cycle."""
        ref = TeamsConversationReference(
            conversation_id="conv-123",
            service_url="https://example.com/",
            tenant_id="tenant-456",
            bot_id="bot-789",
            metadata={
                "team_id": "team-111",
                "team_name": "Engineering",
                "custom_field": "custom_value",
            },
        )

        await conversation_store.save_reference("debate-123", ref)
        retrieved = await conversation_store.get_reference("debate-123")

        assert retrieved.metadata["team_id"] == "team-111"
        assert retrieved.metadata["team_name"] == "Engineering"
        assert retrieved.metadata["custom_field"] == "custom_value"

    @pytest.mark.asyncio
    async def test_empty_metadata(self, conversation_store):
        """Test empty metadata is handled correctly."""
        ref = TeamsConversationReference(
            conversation_id="conv-123",
            service_url="https://example.com/",
            tenant_id="tenant-456",
            bot_id="bot-789",
            metadata={},
        )

        await conversation_store.save_reference("debate-123", ref)
        retrieved = await conversation_store.get_reference("debate-123")

        assert retrieved.metadata == {}

    @pytest.mark.asyncio
    async def test_invalid_metadata_json(self, temp_db_path):
        """Test handles invalid JSON in metadata gracefully."""
        store = TeamsConversationStore(db_path=temp_db_path)

        # Directly insert with invalid JSON
        conn = store._get_connection()
        conn.execute(
            """
            INSERT INTO teams_conversations
            (debate_id, conversation_id, service_url, channel_id, tenant_id,
             bot_id, activity_id, user_id, message_id, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "debate-123",
                "conv-456",
                "https://example.com/",
                None,
                "tenant-789",
                "bot-000",
                None,
                None,
                None,
                time.time(),
                time.time(),
                "invalid-json{",  # Invalid JSON
            ),
        )
        conn.commit()

        # Should not raise, just return empty metadata
        retrieved = await store.get_reference("debate-123")
        assert retrieved is not None
        assert retrieved.metadata == {}


# ===========================================================================
# Singleton Tests
# ===========================================================================


class TestTeamsConversationStoreSingleton:
    """Tests for singleton pattern."""

    def test_get_teams_conversation_store_returns_singleton(self):
        """Test get_teams_conversation_store returns the same instance."""
        import aragora.connectors.chat.teams_conversations as module

        # Reset singleton
        module._store = None

        store1 = get_teams_conversation_store()
        store2 = get_teams_conversation_store()

        assert store1 is store2

        # Cleanup
        module._store = None


# ===========================================================================
# Thread Safety Tests
# ===========================================================================


class TestTeamsConversationStoreThreadSafety:
    """Tests for thread safety."""

    def test_schema_initialization_thread_safe(self, temp_db_path):
        """Test schema initialization is thread-safe."""
        import threading

        store = TeamsConversationStore(db_path=temp_db_path)
        errors = []

        def init_connection():
            try:
                conn = store._get_connection()
                assert conn is not None
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=init_connection) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
