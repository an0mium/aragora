"""
Tests for Sender History Service.

Tests cover:
- Database initialization
- Interaction recording
- Sender statistics
- Reputation calculation
- VIP/blocked sender management
- Feedback tracking
"""

import pytest
from pathlib import Path
import tempfile
import os

from aragora.services.sender_history import (
    SenderHistoryService,
    SenderStats,
    SenderReputation,
    create_sender_history_service,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def service():
    """Create an in-memory sender history service."""
    svc = SenderHistoryService(db_path=":memory:")
    await svc.initialize()
    yield svc
    await svc.close()


@pytest.fixture
async def persistent_service():
    """Create a persistent sender history service."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    svc = SenderHistoryService(db_path=db_path)
    await svc.initialize()
    yield svc
    await svc.close()

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestServiceInitialization:
    """Test service initialization."""

    @pytest.mark.asyncio
    async def test_in_memory_init(self):
        """Test in-memory database initialization."""
        service = SenderHistoryService(db_path=":memory:")
        await service.initialize()

        assert service._connection is not None
        await service.close()

    @pytest.mark.asyncio
    async def test_persistent_init(self):
        """Test persistent database initialization."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        service = SenderHistoryService(db_path=db_path)
        await service.initialize()

        assert service._connection is not None
        assert Path(db_path).exists()

        await service.close()
        os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_factory_function(self):
        """Test create_sender_history_service factory."""
        service = await create_sender_history_service()

        assert service._connection is not None
        await service.close()


# =============================================================================
# Interaction Recording Tests
# =============================================================================


class TestInteractionRecording:
    """Test interaction recording functionality."""

    @pytest.mark.asyncio
    async def test_record_received_email(self, service):
        """Test recording a received email."""
        await service.record_interaction(
            user_id="user1",
            sender_email="sender@example.com",
            action="received",
            email_id="email_123",
        )

        stats = await service.get_sender_stats("user1", "sender@example.com")
        assert stats is not None
        assert stats.total_emails == 1

    @pytest.mark.asyncio
    async def test_record_opened_email(self, service):
        """Test recording an opened email."""
        await service.record_interaction(
            user_id="user1",
            sender_email="sender@example.com",
            action="received",
        )
        await service.record_interaction(
            user_id="user1",
            sender_email="sender@example.com",
            action="opened",
        )

        stats = await service.get_sender_stats("user1", "sender@example.com")
        assert stats.emails_opened == 1

    @pytest.mark.asyncio
    async def test_record_replied_with_response_time(self, service):
        """Test recording a reply with response time."""
        await service.record_interaction(
            user_id="user1",
            sender_email="sender@example.com",
            action="received",
        )
        await service.record_interaction(
            user_id="user1",
            sender_email="sender@example.com",
            action="replied",
            response_time_minutes=30,
        )

        stats = await service.get_sender_stats("user1", "sender@example.com")
        assert stats.emails_replied == 1
        assert stats.avg_response_time_minutes == 30

    @pytest.mark.asyncio
    async def test_record_archived_email(self, service):
        """Test recording an archived email."""
        await service.record_interaction(
            user_id="user1",
            sender_email="sender@example.com",
            action="archived",
        )

        stats = await service.get_sender_stats("user1", "sender@example.com")
        assert stats.emails_archived == 1

    @pytest.mark.asyncio
    async def test_record_deleted_email(self, service):
        """Test recording a deleted email."""
        await service.record_interaction(
            user_id="user1",
            sender_email="sender@example.com",
            action="deleted",
        )

        stats = await service.get_sender_stats("user1", "sender@example.com")
        assert stats.emails_deleted == 1

    @pytest.mark.asyncio
    async def test_sender_email_normalized(self, service):
        """Test that sender emails are normalized to lowercase."""
        await service.record_interaction(
            user_id="user1",
            sender_email="SENDER@Example.COM",
            action="received",
        )

        stats = await service.get_sender_stats("user1", "sender@example.com")
        assert stats is not None
        assert stats.sender_email == "sender@example.com"


# =============================================================================
# Sender Stats Tests
# =============================================================================


class TestSenderStats:
    """Test sender statistics."""

    @pytest.mark.asyncio
    async def test_open_rate_calculation(self, service):
        """Test open rate calculation."""
        for i in range(10):
            await service.record_interaction(
                user_id="user1",
                sender_email="sender@example.com",
                action="received",
            )

        for i in range(7):
            await service.record_interaction(
                user_id="user1",
                sender_email="sender@example.com",
                action="opened",
            )

        stats = await service.get_sender_stats("user1", "sender@example.com")
        assert stats.open_rate == 0.7  # 7/10

    @pytest.mark.asyncio
    async def test_reply_rate_calculation(self, service):
        """Test reply rate calculation."""
        for i in range(5):
            await service.record_interaction(
                user_id="user1",
                sender_email="sender@example.com",
                action="received",
            )

        for i in range(2):
            await service.record_interaction(
                user_id="user1",
                sender_email="sender@example.com",
                action="replied",
            )

        stats = await service.get_sender_stats("user1", "sender@example.com")
        assert stats.reply_rate == 0.4  # 2/5

    @pytest.mark.asyncio
    async def test_engagement_score_calculation(self, service):
        """Test engagement score calculation."""
        # High engagement sender
        for i in range(10):
            await service.record_interaction(
                user_id="user1",
                sender_email="engaged@example.com",
                action="received",
            )
            await service.record_interaction(
                user_id="user1",
                sender_email="engaged@example.com",
                action="opened",
            )
            await service.record_interaction(
                user_id="user1",
                sender_email="engaged@example.com",
                action="replied",
            )

        stats = await service.get_sender_stats("user1", "engaged@example.com")
        assert stats.engagement_score > 0.7  # Should be high

    @pytest.mark.asyncio
    async def test_stats_not_found(self, service):
        """Test getting stats for unknown sender."""
        stats = await service.get_sender_stats("user1", "unknown@example.com")
        assert stats is None


# =============================================================================
# Reputation Calculation Tests
# =============================================================================


class TestReputationCalculation:
    """Test reputation calculation."""

    @pytest.mark.asyncio
    async def test_unknown_sender_neutral_reputation(self, service):
        """Test that unknown senders get neutral reputation."""
        rep = await service.get_sender_reputation("user1", "unknown@example.com")

        assert rep.reputation_score == 0.5
        assert rep.confidence == 0.0
        assert rep.category == "unknown"
        assert not rep.is_vip
        assert not rep.is_blocked

    @pytest.mark.asyncio
    async def test_blocked_sender_reputation(self, service):
        """Test blocked sender reputation."""
        await service.set_blocked("user1", "blocked@example.com", True)

        rep = await service.get_sender_reputation("user1", "blocked@example.com")

        assert rep.reputation_score == 0.0
        assert rep.priority_boost == -0.5
        assert rep.is_blocked
        assert rep.category == "blocked"

    @pytest.mark.asyncio
    async def test_vip_sender_reputation(self, service):
        """Test VIP sender reputation."""
        await service.set_vip("user1", "vip@example.com", True)

        rep = await service.get_sender_reputation("user1", "vip@example.com")

        assert rep.is_vip
        assert rep.reputation_score > 0.5  # VIP bonus
        assert rep.category == "vip"

    @pytest.mark.asyncio
    async def test_high_engagement_reputation(self, service):
        """Test reputation for high engagement sender."""
        # Simulate high engagement
        for i in range(20):
            await service.record_interaction(
                user_id="user1",
                sender_email="important@example.com",
                action="received",
            )
            await service.record_interaction(
                user_id="user1",
                sender_email="important@example.com",
                action="opened",
            )
            await service.record_interaction(
                user_id="user1",
                sender_email="important@example.com",
                action="replied",
                response_time_minutes=15,
            )

        rep = await service.get_sender_reputation("user1", "important@example.com")

        assert rep.reputation_score > 0.7
        assert rep.confidence == 1.0  # 20 emails = max confidence
        assert rep.category == "important"
        assert rep.priority_boost > 0

    @pytest.mark.asyncio
    async def test_low_engagement_reputation(self, service):
        """Test reputation for low engagement sender (many deletes)."""
        # Simulate low engagement - many deletes
        for i in range(10):
            await service.record_interaction(
                user_id="user1",
                sender_email="spam@example.com",
                action="received",
            )
            await service.record_interaction(
                user_id="user1",
                sender_email="spam@example.com",
                action="deleted",
            )

        rep = await service.get_sender_reputation("user1", "spam@example.com")

        assert rep.reputation_score < 0.5
        assert rep.category in ["low_priority", "normal"]
        assert rep.priority_boost < 0

    @pytest.mark.asyncio
    async def test_reputation_caching(self, service):
        """Test that reputation is cached."""
        await service.record_interaction(
            user_id="user1",
            sender_email="cached@example.com",
            action="received",
        )

        rep1 = await service.get_sender_reputation("user1", "cached@example.com")
        rep2 = await service.get_sender_reputation("user1", "cached@example.com")

        # Should be same cached object
        assert rep1 == rep2


# =============================================================================
# VIP Management Tests
# =============================================================================


class TestVIPManagement:
    """Test VIP sender management."""

    @pytest.mark.asyncio
    async def test_set_vip(self, service):
        """Test setting a sender as VIP."""
        await service.set_vip("user1", "vip@example.com", True)

        stats = await service.get_sender_stats("user1", "vip@example.com")
        assert stats.is_vip

    @pytest.mark.asyncio
    async def test_unset_vip(self, service):
        """Test unsetting a VIP sender."""
        await service.set_vip("user1", "vip@example.com", True)
        await service.set_vip("user1", "vip@example.com", False)

        stats = await service.get_sender_stats("user1", "vip@example.com")
        assert not stats.is_vip

    @pytest.mark.asyncio
    async def test_get_vip_senders(self, service):
        """Test getting list of VIP senders."""
        await service.set_vip("user1", "vip1@example.com", True)
        await service.set_vip("user1", "vip2@example.com", True)
        await service.set_vip("user1", "notvip@example.com", False)

        vips = await service.get_vip_senders("user1")

        assert len(vips) == 2
        assert "vip1@example.com" in vips
        assert "vip2@example.com" in vips
        assert "notvip@example.com" not in vips


# =============================================================================
# Blocked Sender Tests
# =============================================================================


class TestBlockedSenders:
    """Test blocked sender management."""

    @pytest.mark.asyncio
    async def test_block_sender(self, service):
        """Test blocking a sender."""
        await service.set_blocked("user1", "blocked@example.com", True)

        stats = await service.get_sender_stats("user1", "blocked@example.com")
        assert stats.is_blocked

    @pytest.mark.asyncio
    async def test_unblock_sender(self, service):
        """Test unblocking a sender."""
        await service.set_blocked("user1", "blocked@example.com", True)
        await service.set_blocked("user1", "blocked@example.com", False)

        stats = await service.get_sender_stats("user1", "blocked@example.com")
        assert not stats.is_blocked

    @pytest.mark.asyncio
    async def test_get_blocked_senders(self, service):
        """Test getting list of blocked senders."""
        await service.set_blocked("user1", "blocked1@example.com", True)
        await service.set_blocked("user1", "blocked2@example.com", True)

        blocked = await service.get_blocked_senders("user1")

        assert len(blocked) == 2
        assert "blocked1@example.com" in blocked
        assert "blocked2@example.com" in blocked


# =============================================================================
# Priority Feedback Tests
# =============================================================================


class TestPriorityFeedback:
    """Test priority feedback tracking."""

    @pytest.mark.asyncio
    async def test_record_priority_feedback(self, service):
        """Test recording priority feedback."""
        await service.record_priority_feedback(
            user_id="user1",
            email_id="email_123",
            sender_email="sender@example.com",
            predicted_priority="high",
            actual_priority="critical",
            is_correct=False,
            feedback_type="explicit",
        )

        # Should not raise an error
        # We can verify via accuracy stats
        accuracy = await service.get_prediction_accuracy("user1")
        assert accuracy["total_feedback"] == 1
        assert accuracy["incorrect"] == 1

    @pytest.mark.asyncio
    async def test_prediction_accuracy(self, service):
        """Test prediction accuracy calculation."""
        # Record correct predictions
        for i in range(7):
            await service.record_priority_feedback(
                user_id="user1",
                email_id=f"email_{i}",
                sender_email="sender@example.com",
                predicted_priority="high",
                is_correct=True,
            )

        # Record incorrect predictions
        for i in range(3):
            await service.record_priority_feedback(
                user_id="user1",
                email_id=f"wrong_{i}",
                sender_email="sender@example.com",
                predicted_priority="low",
                is_correct=False,
            )

        accuracy = await service.get_prediction_accuracy("user1")

        assert accuracy["total_feedback"] == 10
        assert accuracy["correct"] == 7
        assert accuracy["incorrect"] == 3
        assert accuracy["accuracy"] == 0.7


# =============================================================================
# Top Senders Tests
# =============================================================================


class TestTopSenders:
    """Test top senders retrieval."""

    @pytest.mark.asyncio
    async def test_get_top_senders_by_total_emails(self, service):
        """Test getting top senders by total emails."""
        # Add senders with different email counts
        for i in range(10):
            await service.record_interaction(
                user_id="user1",
                sender_email="frequent@example.com",
                action="received",
            )

        for i in range(5):
            await service.record_interaction(
                user_id="user1",
                sender_email="moderate@example.com",
                action="received",
            )

        for i in range(2):
            await service.record_interaction(
                user_id="user1",
                sender_email="rare@example.com",
                action="received",
            )

        top = await service.get_top_senders("user1", limit=3, order_by="total_emails")

        assert len(top) == 3
        assert top[0].sender_email == "frequent@example.com"
        assert top[0].total_emails == 10


# =============================================================================
# Multi-User Isolation Tests
# =============================================================================


class TestMultiUserIsolation:
    """Test that data is isolated between users."""

    @pytest.mark.asyncio
    async def test_user_data_isolation(self, service):
        """Test that different users have isolated data."""
        await service.record_interaction(
            user_id="user1",
            sender_email="sender@example.com",
            action="received",
        )
        await service.set_vip("user1", "vip@example.com", True)

        await service.record_interaction(
            user_id="user2",
            sender_email="sender@example.com",
            action="received",
        )
        await service.record_interaction(
            user_id="user2",
            sender_email="sender@example.com",
            action="received",
        )

        stats1 = await service.get_sender_stats("user1", "sender@example.com")
        stats2 = await service.get_sender_stats("user2", "sender@example.com")

        assert stats1.total_emails == 1
        assert stats2.total_emails == 2

        vips1 = await service.get_vip_senders("user1")
        vips2 = await service.get_vip_senders("user2")

        assert "vip@example.com" in vips1
        assert "vip@example.com" not in vips2


# =============================================================================
# Data Class Tests
# =============================================================================


class TestSenderStatsDataClass:
    """Test SenderStats dataclass."""

    def test_default_values(self):
        """Test default values."""
        stats = SenderStats(sender_email="test@example.com")

        assert stats.total_emails == 0
        assert stats.emails_opened == 0
        assert stats.open_rate == 0.0
        assert stats.reply_rate == 0.0
        assert not stats.is_vip
        assert not stats.is_blocked

    def test_open_rate_zero_division(self):
        """Test open rate with zero emails."""
        stats = SenderStats(sender_email="test@example.com", total_emails=0)
        assert stats.open_rate == 0.0

    def test_reply_rate_zero_division(self):
        """Test reply rate with zero emails."""
        stats = SenderStats(sender_email="test@example.com", total_emails=0)
        assert stats.reply_rate == 0.0


class TestSenderReputationDataClass:
    """Test SenderReputation dataclass."""

    def test_to_dict(self):
        """Test to_dict conversion."""
        rep = SenderReputation(
            sender_email="test@example.com",
            reputation_score=0.8,
            confidence=0.9,
            priority_boost=0.15,
            is_vip=True,
            is_blocked=False,
            category="vip",
            reasons=["VIP sender", "High engagement"],
        )

        d = rep.to_dict()

        assert d["sender_email"] == "test@example.com"
        assert d["reputation_score"] == 0.8
        assert d["confidence"] == 0.9
        assert d["priority_boost"] == 0.15
        assert d["is_vip"] is True
        assert d["is_blocked"] is False
        assert d["category"] == "vip"
        assert len(d["reasons"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
