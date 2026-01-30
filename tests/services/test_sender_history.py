"""
Tests for Sender History Persistence Service.

Tests the sender history service including:
- Database initialization
- Interaction recording
- Sender statistics and reputation
- VIP and blocked sender management
- Domain reputation
- Batch operations
- Priority feedback
"""

import pytest
from datetime import datetime, timedelta
from typing import Any, Dict

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
    """Create test service with in-memory database."""
    svc = SenderHistoryService(db_path=":memory:", cache_ttl_seconds=60)
    await svc.initialize()
    yield svc
    await svc.close()


@pytest.fixture
def user_id():
    """Standard test user ID."""
    return "user@example.com"


@pytest.fixture
def sender_email():
    """Standard test sender email."""
    return "sender@company.com"


# =============================================================================
# Initialization Tests
# =============================================================================


class TestInitialization:
    """Test service initialization."""

    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self, service):
        """Should create required tables on initialization."""
        # Service is already initialized via fixture
        assert service._connection is not None

    @pytest.mark.asyncio
    async def test_initialize_in_memory(self):
        """Should work with in-memory database."""
        svc = SenderHistoryService(db_path=":memory:")
        await svc.initialize()
        assert svc._connection is not None
        await svc.close()

    @pytest.mark.asyncio
    async def test_close_connection(self):
        """Should close connection properly."""
        svc = SenderHistoryService(db_path=":memory:")
        await svc.initialize()
        await svc.close()
        assert svc._connection is None

    @pytest.mark.asyncio
    async def test_create_service_factory(self):
        """Should create service via factory function."""
        svc = await create_sender_history_service(db_path=":memory:")
        assert svc._connection is not None
        await svc.close()


# =============================================================================
# Interaction Recording Tests
# =============================================================================


class TestRecordInteraction:
    """Test interaction recording."""

    @pytest.mark.asyncio
    async def test_record_received(self, service, user_id, sender_email):
        """Should record received email."""
        await service.record_interaction(
            user_id=user_id,
            sender_email=sender_email,
            action="received",
            email_id="email-123",
        )

        stats = await service.get_sender_stats(user_id, sender_email)
        assert stats is not None
        assert stats.total_emails == 1

    @pytest.mark.asyncio
    async def test_record_opened(self, service, user_id, sender_email):
        """Should record opened email."""
        await service.record_interaction(
            user_id=user_id, sender_email=sender_email, action="received"
        )
        await service.record_interaction(
            user_id=user_id, sender_email=sender_email, action="opened"
        )

        stats = await service.get_sender_stats(user_id, sender_email)
        assert stats.emails_opened == 1

    @pytest.mark.asyncio
    async def test_record_replied_with_response_time(self, service, user_id, sender_email):
        """Should record reply with response time."""
        await service.record_interaction(
            user_id=user_id,
            sender_email=sender_email,
            action="replied",
            response_time_minutes=30,
        )

        stats = await service.get_sender_stats(user_id, sender_email)
        assert stats.emails_replied == 1
        assert stats.avg_response_time_minutes == 30.0

    @pytest.mark.asyncio
    async def test_record_archived(self, service, user_id, sender_email):
        """Should record archived email."""
        await service.record_interaction(
            user_id=user_id, sender_email=sender_email, action="archived"
        )

        stats = await service.get_sender_stats(user_id, sender_email)
        assert stats.emails_archived == 1

    @pytest.mark.asyncio
    async def test_record_deleted(self, service, user_id, sender_email):
        """Should record deleted email."""
        await service.record_interaction(
            user_id=user_id, sender_email=sender_email, action="deleted"
        )

        stats = await service.get_sender_stats(user_id, sender_email)
        assert stats.emails_deleted == 1

    @pytest.mark.asyncio
    async def test_email_normalized_to_lowercase(self, service, user_id):
        """Should normalize email to lowercase."""
        await service.record_interaction(
            user_id=user_id,
            sender_email="SENDER@COMPANY.COM",
            action="received",
        )

        stats = await service.get_sender_stats(user_id, "sender@company.com")
        assert stats is not None
        assert stats.sender_email == "sender@company.com"

    @pytest.mark.asyncio
    async def test_record_with_metadata(self, service, user_id, sender_email):
        """Should record interaction with metadata."""
        await service.record_interaction(
            user_id=user_id,
            sender_email=sender_email,
            action="received",
            email_id="email-123",
            metadata={"subject": "Test email", "thread_id": "thread-456"},
        )

        # Metadata is stored in interaction_log
        stats = await service.get_sender_stats(user_id, sender_email)
        assert stats is not None


# =============================================================================
# Sender Stats Tests
# =============================================================================


class TestSenderStats:
    """Test sender statistics."""

    @pytest.mark.asyncio
    async def test_get_stats_not_found(self, service, user_id):
        """Should return None for unknown sender."""
        stats = await service.get_sender_stats(user_id, "unknown@example.com")
        assert stats is None

    @pytest.mark.asyncio
    async def test_open_rate_calculation(self, service, user_id, sender_email):
        """Should calculate open rate correctly."""
        # 10 emails, 5 opened = 50% open rate
        for _ in range(10):
            await service.record_interaction(
                user_id=user_id, sender_email=sender_email, action="received"
            )
        for _ in range(5):
            await service.record_interaction(
                user_id=user_id, sender_email=sender_email, action="opened"
            )

        stats = await service.get_sender_stats(user_id, sender_email)
        assert stats.open_rate == 0.5

    @pytest.mark.asyncio
    async def test_reply_rate_calculation(self, service, user_id, sender_email):
        """Should calculate reply rate correctly."""
        # 10 emails, 2 replied = 20% reply rate
        for _ in range(10):
            await service.record_interaction(
                user_id=user_id, sender_email=sender_email, action="received"
            )
        for _ in range(2):
            await service.record_interaction(
                user_id=user_id, sender_email=sender_email, action="replied"
            )

        stats = await service.get_sender_stats(user_id, sender_email)
        assert stats.reply_rate == 0.2

    @pytest.mark.asyncio
    async def test_average_response_time(self, service, user_id, sender_email):
        """Should calculate average response time."""
        await service.record_interaction(
            user_id=user_id,
            sender_email=sender_email,
            action="replied",
            response_time_minutes=10,
        )
        await service.record_interaction(
            user_id=user_id,
            sender_email=sender_email,
            action="replied",
            response_time_minutes=20,
        )
        await service.record_interaction(
            user_id=user_id,
            sender_email=sender_email,
            action="replied",
            response_time_minutes=30,
        )

        stats = await service.get_sender_stats(user_id, sender_email)
        # Average of 10, 20, 30 = 20
        assert stats.avg_response_time_minutes == 20.0

    def test_engagement_score_property(self):
        """Should calculate engagement score correctly."""
        stats = SenderStats(
            sender_email="test@example.com",
            total_emails=10,
            emails_opened=5,
            emails_replied=3,
        )
        # open_rate = 0.5, reply_rate = 0.3
        # engagement = (0.5 * 0.4) + (0.3 * 0.4) = 0.32
        # No recency boost since no last_email_date
        assert 0.30 <= stats.engagement_score <= 0.35


# =============================================================================
# Sender Reputation Tests
# =============================================================================


class TestSenderReputation:
    """Test sender reputation calculation."""

    @pytest.mark.asyncio
    async def test_unknown_sender_reputation(self, service, user_id):
        """Should return neutral reputation for unknown sender."""
        reputation = await service.get_sender_reputation(user_id, "unknown@example.com")

        assert reputation.reputation_score == 0.5
        assert reputation.confidence == 0.0
        assert reputation.category == "unknown"

    @pytest.mark.asyncio
    async def test_blocked_sender_reputation(self, service, user_id, sender_email):
        """Should return blocked reputation for blocked sender."""
        await service.set_blocked(user_id, sender_email, is_blocked=True)

        reputation = await service.get_sender_reputation(user_id, sender_email)

        assert reputation.reputation_score == 0.0
        assert reputation.is_blocked is True
        assert reputation.category == "blocked"
        assert reputation.priority_boost == -0.5

    @pytest.mark.asyncio
    async def test_vip_sender_reputation(self, service, user_id, sender_email):
        """Should include VIP bonus in reputation."""
        await service.record_interaction(
            user_id=user_id, sender_email=sender_email, action="received"
        )
        await service.set_vip(user_id, sender_email, is_vip=True)

        reputation = await service.get_sender_reputation(user_id, sender_email)

        assert reputation.is_vip is True
        assert "VIP sender" in " ".join(reputation.reasons)

    @pytest.mark.asyncio
    async def test_reputation_caching(self, service, user_id, sender_email):
        """Should cache reputation lookups."""
        await service.record_interaction(
            user_id=user_id, sender_email=sender_email, action="received"
        )

        # First call populates cache
        rep1 = await service.get_sender_reputation(user_id, sender_email)

        # Second call should return cached value
        rep2 = await service.get_sender_reputation(user_id, sender_email)

        assert rep1.reputation_score == rep2.reputation_score

    @pytest.mark.asyncio
    async def test_reputation_cache_invalidation(self, service, user_id, sender_email):
        """Should invalidate cache on new interaction."""
        await service.record_interaction(
            user_id=user_id, sender_email=sender_email, action="received"
        )

        # Populate cache
        await service.get_sender_reputation(user_id, sender_email)

        # Record more interactions - should invalidate cache
        await service.record_interaction(
            user_id=user_id, sender_email=sender_email, action="opened"
        )

        # Cache should be cleared
        cache_key = f"{user_id}:{sender_email.lower()}"
        assert cache_key not in service._reputation_cache

    def test_reputation_to_dict(self):
        """Should convert reputation to dictionary."""
        reputation = SenderReputation(
            sender_email="test@example.com",
            reputation_score=0.8,
            confidence=0.9,
            priority_boost=0.1,
            is_vip=True,
            is_blocked=False,
            category="vip",
            reasons=["High engagement"],
        )
        d = reputation.to_dict()

        assert d["sender_email"] == "test@example.com"
        assert d["reputation_score"] == 0.8
        assert d["is_vip"] is True


# =============================================================================
# VIP Management Tests
# =============================================================================


class TestVIPManagement:
    """Test VIP sender management."""

    @pytest.mark.asyncio
    async def test_set_vip(self, service, user_id, sender_email):
        """Should mark sender as VIP."""
        await service.record_interaction(
            user_id=user_id, sender_email=sender_email, action="received"
        )
        await service.set_vip(user_id, sender_email, is_vip=True)

        stats = await service.get_sender_stats(user_id, sender_email)
        assert stats.is_vip is True

    @pytest.mark.asyncio
    async def test_remove_vip(self, service, user_id, sender_email):
        """Should remove VIP status."""
        await service.record_interaction(
            user_id=user_id, sender_email=sender_email, action="received"
        )
        await service.set_vip(user_id, sender_email, is_vip=True)
        await service.set_vip(user_id, sender_email, is_vip=False)

        stats = await service.get_sender_stats(user_id, sender_email)
        assert stats.is_vip is False

    @pytest.mark.asyncio
    async def test_get_vip_senders(self, service, user_id):
        """Should get list of VIP senders."""
        # Add some senders
        for email in ["vip1@example.com", "vip2@example.com", "normal@example.com"]:
            await service.record_interaction(user_id=user_id, sender_email=email, action="received")

        # Mark some as VIP
        await service.set_vip(user_id, "vip1@example.com", is_vip=True)
        await service.set_vip(user_id, "vip2@example.com", is_vip=True)

        vips = await service.get_vip_senders(user_id)

        assert len(vips) == 2
        assert "vip1@example.com" in vips
        assert "vip2@example.com" in vips


# =============================================================================
# Blocked Sender Tests
# =============================================================================


class TestBlockedSenders:
    """Test blocked sender management."""

    @pytest.mark.asyncio
    async def test_set_blocked(self, service, user_id, sender_email):
        """Should mark sender as blocked."""
        await service.record_interaction(
            user_id=user_id, sender_email=sender_email, action="received"
        )
        await service.set_blocked(user_id, sender_email, is_blocked=True)

        stats = await service.get_sender_stats(user_id, sender_email)
        assert stats.is_blocked is True

    @pytest.mark.asyncio
    async def test_unblock_sender(self, service, user_id, sender_email):
        """Should unblock sender."""
        await service.record_interaction(
            user_id=user_id, sender_email=sender_email, action="received"
        )
        await service.set_blocked(user_id, sender_email, is_blocked=True)
        await service.set_blocked(user_id, sender_email, is_blocked=False)

        stats = await service.get_sender_stats(user_id, sender_email)
        assert stats.is_blocked is False

    @pytest.mark.asyncio
    async def test_is_blocked_quick_check(self, service, user_id, sender_email):
        """Should quickly check if sender is blocked."""
        await service.record_interaction(
            user_id=user_id, sender_email=sender_email, action="received"
        )

        assert await service.is_blocked(user_id, sender_email) is False

        await service.set_blocked(user_id, sender_email, is_blocked=True)

        assert await service.is_blocked(user_id, sender_email) is True

    @pytest.mark.asyncio
    async def test_get_blocked_senders(self, service, user_id):
        """Should get list of blocked senders."""
        for email in ["spam1@example.com", "spam2@example.com", "legit@example.com"]:
            await service.record_interaction(user_id=user_id, sender_email=email, action="received")

        await service.set_blocked(user_id, "spam1@example.com", is_blocked=True)
        await service.set_blocked(user_id, "spam2@example.com", is_blocked=True)

        blocked = await service.get_blocked_senders(user_id)

        assert len(blocked) == 2
        assert "spam1@example.com" in blocked
        assert "spam2@example.com" in blocked


# =============================================================================
# Domain Reputation Tests
# =============================================================================


class TestDomainReputation:
    """Test domain-level reputation."""

    @pytest.mark.asyncio
    async def test_get_domain_reputation(self, service, user_id):
        """Should aggregate reputation by domain."""
        # Add multiple senders from same domain
        for i in range(5):
            email = f"user{i}@company.com"
            await service.record_interaction(user_id=user_id, sender_email=email, action="received")
            await service.record_interaction(user_id=user_id, sender_email=email, action="opened")

        # Returns Tuple[float, int] - (avg_reputation, sender_count)
        result = await service.get_domain_reputation(user_id, "company.com")

        assert result is not None
        avg_reputation, sender_count = result
        assert avg_reputation > 0
        assert sender_count == 5


# =============================================================================
# Batch Operations Tests
# =============================================================================


class TestBatchOperations:
    """Test batch operations."""

    @pytest.mark.asyncio
    async def test_get_batch_reputations(self, service, user_id):
        """Should get reputations for multiple senders."""
        senders = ["sender1@example.com", "sender2@example.com", "sender3@example.com"]

        for sender in senders:
            await service.record_interaction(
                user_id=user_id, sender_email=sender, action="received"
            )

        reputations = await service.get_batch_reputations(user_id, senders)

        assert len(reputations) == 3
        for sender in senders:
            assert sender.lower() in reputations


# =============================================================================
# Priority Feedback Tests
# =============================================================================


class TestPriorityFeedback:
    """Test priority feedback recording."""

    @pytest.mark.asyncio
    async def test_record_priority_feedback(self, service, user_id, sender_email):
        """Should record priority feedback."""
        await service.record_priority_feedback(
            user_id=user_id,
            email_id="email-123",
            sender_email=sender_email,
            predicted_priority="high",
            actual_priority="medium",
            is_correct=False,
            feedback_type="explicit",
        )

        # Feedback is recorded - check via accuracy
        accuracy = await service.get_prediction_accuracy(user_id)
        # With one incorrect prediction, accuracy should be 0%
        assert accuracy["accuracy"] == 0.0

    @pytest.mark.asyncio
    async def test_get_prediction_accuracy(self, service, user_id, sender_email):
        """Should calculate prediction accuracy."""
        # Record 7 correct and 3 incorrect predictions
        for i in range(7):
            await service.record_priority_feedback(
                user_id=user_id,
                email_id=f"email-{i}",
                sender_email=sender_email,
                predicted_priority="high",
                actual_priority="high",
                is_correct=True,
                feedback_type="explicit",
            )

        for i in range(3):
            await service.record_priority_feedback(
                user_id=user_id,
                email_id=f"email-wrong-{i}",
                sender_email=sender_email,
                predicted_priority="high",
                actual_priority="low",
                is_correct=False,
                feedback_type="explicit",
            )

        accuracy = await service.get_prediction_accuracy(user_id)

        # 7 correct out of 10 = 70%
        assert accuracy["accuracy"] == 0.7
        assert accuracy["total_feedback"] == 10


# =============================================================================
# Top Senders Tests
# =============================================================================


class TestTopSenders:
    """Test top senders retrieval."""

    @pytest.mark.asyncio
    async def test_get_top_senders(self, service, user_id):
        """Should get top senders by email count."""
        # Create senders with different email counts
        senders_counts = [
            ("frequent@example.com", 10),
            ("moderate@example.com", 5),
            ("rare@example.com", 1),
        ]

        for sender, count in senders_counts:
            for _ in range(count):
                await service.record_interaction(
                    user_id=user_id, sender_email=sender, action="received"
                )

        top_senders = await service.get_top_senders(user_id, limit=3)

        assert len(top_senders) == 3
        # First should be most frequent
        assert top_senders[0].sender_email == "frequent@example.com"
        assert top_senders[0].total_emails == 10


# =============================================================================
# Data Model Tests
# =============================================================================


class TestDataModels:
    """Test data model creation."""

    def test_sender_stats_creation(self):
        """Should create SenderStats with defaults."""
        stats = SenderStats(sender_email="test@example.com")
        assert stats.total_emails == 0
        assert stats.is_vip is False
        assert stats.tags == []

    def test_sender_stats_open_rate_zero_emails(self):
        """Should handle zero emails for open rate."""
        stats = SenderStats(sender_email="test@example.com", total_emails=0)
        assert stats.open_rate == 0.0

    def test_sender_stats_reply_rate_zero_emails(self):
        """Should handle zero emails for reply rate."""
        stats = SenderStats(sender_email="test@example.com", total_emails=0)
        assert stats.reply_rate == 0.0

    def test_sender_reputation_creation(self):
        """Should create SenderReputation."""
        rep = SenderReputation(
            sender_email="test@example.com",
            reputation_score=0.8,
            confidence=0.9,
            priority_boost=0.1,
            is_vip=False,
            is_blocked=False,
            category="important",
        )
        assert rep.reputation_score == 0.8
        assert rep.reasons == []
