"""
Tests for durability and persistence.

Verifies that state survives server restarts and is properly persisted.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestDebateOriginDurability:
    """Test debate origin persistence."""

    def test_origin_saved_to_sqlite(self):
        """Verify origins are persisted to SQLite."""
        from aragora.server.debate_origin import (
            DebateOrigin,
            SQLiteOriginStore,
            register_debate_origin,
            get_debate_origin,
        )

        # Use temp database
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_origins.db")
            store = SQLiteOriginStore(db_path)

            # Create and save origin
            origin = DebateOrigin(
                debate_id="test-debate-123",
                platform="telegram",
                channel_id="chat-456",
                user_id="user-789",
                metadata={"username": "tester"},
            )
            store.save(origin)

            # Retrieve and verify
            loaded = store.get("test-debate-123")
            assert loaded is not None
            assert loaded.debate_id == origin.debate_id
            assert loaded.platform == "telegram"
            assert loaded.channel_id == "chat-456"
            assert loaded.metadata.get("username") == "tester"

    def test_origin_survives_restart(self):
        """Test that origin can be recovered after 'restart'."""
        from aragora.server.debate_origin import SQLiteOriginStore, DebateOrigin

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "origins.db")

            # First "session"
            store1 = SQLiteOriginStore(db_path)
            origin = DebateOrigin(
                debate_id="persist-test-1",
                platform="slack",
                channel_id="C123",
                user_id="U456",
            )
            store1.save(origin)

            # New "session" (simulating restart)
            store2 = SQLiteOriginStore(db_path)
            recovered = store2.get("persist-test-1")

            assert recovered is not None
            assert recovered.platform == "slack"

    def test_cleanup_expired_origins(self):
        """Test that expired origins are cleaned up."""
        from aragora.server.debate_origin import SQLiteOriginStore, DebateOrigin

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "cleanup.db")
            store = SQLiteOriginStore(db_path)

            # Create old origin
            old_origin = DebateOrigin(
                debate_id="old-debate",
                platform="test",
                channel_id="ch",
                user_id="u",
                created_at=time.time() - 100000,  # Very old
            )
            store.save(old_origin)

            # Cleanup with short TTL
            cleaned = store.cleanup_expired(ttl_seconds=1000)

            # Old origin should be cleaned
            assert cleaned >= 1


class TestEmailReplyOriginDurability:
    """Test email reply origin persistence."""

    def test_email_origin_saved_to_sqlite(self):
        """Verify email origins are persisted to SQLite."""
        from aragora.integrations.email_reply_loop import (
            EmailReplyOrigin,
            SQLiteEmailReplyStore,
        )
        from datetime import datetime

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "email_origins.db")
            store = SQLiteEmailReplyStore(db_path)

            origin = EmailReplyOrigin(
                debate_id="debate-123",
                message_id="<msg-id-123@example.com>",
                recipient_email="user@example.com",
                recipient_name="Test User",
            )
            store.save(origin)

            loaded = store.get("<msg-id-123@example.com>")
            assert loaded is not None
            assert loaded.debate_id == "debate-123"
            assert loaded.recipient_email == "user@example.com"


class TestControlPlaneSQLiteFallback:
    """Test control plane shared state SQLite fallback."""

    @pytest.mark.asyncio
    async def test_sqlite_fallback_initialization(self):
        """Test that SQLite fallback initializes correctly."""
        from aragora.control_plane.shared_state import SharedControlPlaneState

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "control_plane.db")

            state = SharedControlPlaneState(
                redis_url="redis://nonexistent:6379",  # Will fail
                sqlite_path=db_path,
            )

            # Connect should fail to Redis, fall back to SQLite
            connected = await state.connect()
            assert connected is False  # No Redis
            assert state._sqlite_initialized is True
            assert Path(db_path).exists()

    @pytest.mark.asyncio
    async def test_agent_persisted_to_sqlite(self):
        """Test that agents are persisted to SQLite fallback."""
        from aragora.control_plane.shared_state import (
            SharedControlPlaneState,
            AgentState,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "cp.db")

            state = SharedControlPlaneState(
                redis_url="redis://nonexistent:6379",
                sqlite_path=db_path,
            )
            await state.connect()

            # Register an agent
            agent = await state.register_agent({
                "id": "agent-123",
                "name": "Test Agent",
                "type": "anthropic-api",
                "model": "claude-3",
                "status": "active",
            })

            assert agent.id == "agent-123"

            # Create new state instance (simulating restart)
            state2 = SharedControlPlaneState(
                redis_url="redis://nonexistent:6379",
                sqlite_path=db_path,
            )
            await state2.connect()

            # Agent should be recovered from SQLite
            recovered = await state2.get_agent("agent-123")
            assert recovered is not None
            assert recovered["name"] == "Test Agent"


class TestHumanCheckpointDurability:
    """Test human checkpoint approval persistence."""

    def test_approval_persisted_to_governance_store(self):
        """Test that approvals are saved to GovernanceStore."""
        # This requires GovernanceStore to be available
        pass  # Placeholder - requires more mocking

    def test_pending_approvals_recovered(self):
        """Test that pending approvals can be recovered after restart."""
        from aragora.workflow.nodes.human_checkpoint import (
            get_pending_approvals,
            get_approval_request,
        )

        # Would need GovernanceStore mock
        pass  # Placeholder


class TestJobQueueDurability:
    """Test job queue persistence."""

    @pytest.mark.asyncio
    async def test_transcription_job_persisted(self):
        """Test that transcription jobs are persisted."""
        from aragora.storage.job_queue_store import QueuedJob, JobStatus

        # Would need actual job store
        pass  # Placeholder

    @pytest.mark.asyncio
    async def test_routing_job_recovery(self):
        """Test that routing jobs can be recovered."""
        from aragora.queue.workers.routing_worker import recover_interrupted_routing

        # Would need actual job store
        pass  # Placeholder


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
