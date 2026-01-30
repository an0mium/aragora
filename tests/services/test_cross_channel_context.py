"""
Tests for Cross-Channel Context Service.

Tests for the cross-channel context service including:
- Signal gathering from Slack, Drive, Calendar
- Context caching and TTL
- Email-to-Slack user mapping
- Derived signal calculation
- Email context boost computation
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


class TestSlackActivitySignal:
    """Tests for SlackActivitySignal dataclass."""

    def test_signal_creation_defaults(self):
        """Test creating signal with defaults."""
        from aragora.services.cross_channel_context import SlackActivitySignal

        signal = SlackActivitySignal(user_email="user@example.com")

        assert signal.user_email == "user@example.com"
        assert signal.is_online is False
        assert signal.active_channels == []
        assert signal.recent_mentions == 0
        assert signal.urgent_threads == []
        assert signal.last_activity is None
        assert signal.activity_score == 0.0

    def test_signal_creation_full(self):
        """Test creating signal with all fields."""
        from aragora.services.cross_channel_context import SlackActivitySignal

        now = datetime.now()
        signal = SlackActivitySignal(
            user_email="user@example.com",
            is_online=True,
            active_channels=["C123", "C456"],
            recent_mentions=5,
            urgent_threads=["ts_1", "ts_2"],
            last_activity=now,
            activity_score=0.8,
        )

        assert signal.is_online is True
        assert len(signal.active_channels) == 2
        assert signal.recent_mentions == 5
        assert len(signal.urgent_threads) == 2
        assert signal.last_activity == now
        assert signal.activity_score == 0.8


class TestDriveActivitySignal:
    """Tests for DriveActivitySignal dataclass."""

    def test_drive_signal_creation(self):
        """Test creating Drive activity signal."""
        from aragora.services.cross_channel_context import DriveActivitySignal

        signal = DriveActivitySignal(
            user_email="user@example.com",
            recently_edited_files=["doc1.pdf", "doc2.xlsx"],
            recently_viewed_files=["report.docx"],
            shared_with_me_recent=["budget.xlsx"],
            activity_score=0.5,
        )

        assert signal.user_email == "user@example.com"
        assert len(signal.recently_edited_files) == 2
        assert len(signal.recently_viewed_files) == 1
        assert signal.activity_score == 0.5


class TestCalendarSignal:
    """Tests for CalendarSignal dataclass."""

    def test_calendar_signal_creation(self):
        """Test creating Calendar signal."""
        from aragora.services.cross_channel_context import CalendarSignal

        now = datetime.now()
        signal = CalendarSignal(
            user_email="user@example.com",
            upcoming_meetings=[{"title": "Standup", "start": now.isoformat()}],
            busy_periods=[(now, now + timedelta(hours=1))],
            next_free_slot=now + timedelta(hours=2),
            meeting_density_score=0.6,
        )

        assert len(signal.upcoming_meetings) == 1
        assert len(signal.busy_periods) == 1
        assert signal.meeting_density_score == 0.6


class TestChannelContext:
    """Tests for ChannelContext dataclass."""

    def test_context_creation_minimal(self):
        """Test creating context with minimal fields."""
        from aragora.services.cross_channel_context import ChannelContext

        context = ChannelContext(user_email="user@example.com")

        assert context.user_email == "user@example.com"
        assert context.slack is None
        assert context.drive is None
        assert context.calendar is None
        assert context.overall_activity_score == 0.0
        assert context.is_likely_busy is False

    def test_context_to_dict(self):
        """Test ChannelContext.to_dict serialization."""
        from aragora.services.cross_channel_context import (
            ChannelContext,
            SlackActivitySignal,
        )

        slack_signal = SlackActivitySignal(
            user_email="user@example.com",
            is_online=True,
            active_channels=["C123"],
            activity_score=0.5,
        )

        context = ChannelContext(
            user_email="user@example.com",
            slack=slack_signal,
            overall_activity_score=0.5,
            suggested_response_window="Now - user is active",
        )

        data = context.to_dict()

        assert data["user_email"] == "user@example.com"
        assert data["overall_activity_score"] == 0.5
        assert data["suggested_response_window"] == "Now - user is active"
        assert data["slack"]["is_online"] is True
        assert data["slack"]["active_channels"] == ["C123"]
        assert data["drive"] is None
        assert data["calendar"] is None

    def test_context_to_dict_all_signals(self):
        """Test to_dict with all signal types."""
        from aragora.services.cross_channel_context import (
            ChannelContext,
            SlackActivitySignal,
            DriveActivitySignal,
            CalendarSignal,
        )

        context = ChannelContext(
            user_email="user@example.com",
            slack=SlackActivitySignal(user_email="user@example.com", is_online=True),
            drive=DriveActivitySignal(
                user_email="user@example.com",
                recently_edited_files=["doc1.pdf"],
            ),
            calendar=CalendarSignal(
                user_email="user@example.com",
                upcoming_meetings=[{"title": "Meeting"}],
                meeting_density_score=0.8,
            ),
        )

        data = context.to_dict()

        assert data["slack"] is not None
        assert data["drive"] is not None
        assert data["calendar"] is not None
        assert data["drive"]["recently_edited_count"] == 1
        assert data["calendar"]["upcoming_meetings_count"] == 1
        assert data["calendar"]["meeting_density"] == 0.8


class TestEmailContextBoost:
    """Tests for EmailContextBoost dataclass."""

    def test_boost_creation_defaults(self):
        """Test creating boost with defaults."""
        from aragora.services.cross_channel_context import EmailContextBoost

        boost = EmailContextBoost(email_id="email_123")

        assert boost.email_id == "email_123"
        assert boost.slack_activity_boost == 0.0
        assert boost.drive_relevance_boost == 0.0
        assert boost.calendar_urgency_boost == 0.0
        assert boost.total_boost == 0.0

    def test_boost_total_calculation(self):
        """Test total boost calculation."""
        from aragora.services.cross_channel_context import EmailContextBoost

        boost = EmailContextBoost(
            email_id="email_123",
            slack_activity_boost=0.15,
            drive_relevance_boost=0.1,
            calendar_urgency_boost=0.05,
        )

        assert boost.total_boost == 0.3

    def test_boost_with_reasons(self):
        """Test boost with explanation reasons."""
        from aragora.services.cross_channel_context import EmailContextBoost

        boost = EmailContextBoost(
            email_id="email_123",
            slack_activity_boost=0.15,
            slack_reason="Sender is currently active on Slack",
            related_slack_channels=["#engineering", "#general"],
        )

        assert boost.slack_reason == "Sender is currently active on Slack"
        assert len(boost.related_slack_channels) == 2


class TestCrossChannelContextService:
    """Tests for CrossChannelContextService."""

    def test_service_initialization(self):
        """Test service initialization."""
        from aragora.services.cross_channel_context import CrossChannelContextService

        service = CrossChannelContextService()

        assert service.slack is None
        assert service.gmail is None
        assert service.mound is None
        assert service.cache_ttl == 300  # Default 5 minutes
        assert service._context_cache == {}
        assert service._email_to_slack_id == {}

    def test_service_with_connectors(self):
        """Test service initialization with connectors."""
        from aragora.services.cross_channel_context import CrossChannelContextService

        mock_slack = MagicMock()
        mock_gmail = MagicMock()

        service = CrossChannelContextService(
            slack_connector=mock_slack,
            gmail_connector=mock_gmail,
            cache_ttl_seconds=600,
            user_id="tenant_123",
        )

        assert service.slack is mock_slack
        assert service.gmail is mock_gmail
        assert service.cache_ttl == 600
        assert service._user_id == "tenant_123"

    def test_clear_cache_specific_user(self):
        """Test clearing cache for specific user."""
        from aragora.services.cross_channel_context import (
            CrossChannelContextService,
            ChannelContext,
        )

        service = CrossChannelContextService()

        # Add some cached contexts
        context1 = ChannelContext(user_email="user1@example.com")
        context2 = ChannelContext(user_email="user2@example.com")

        service._context_cache["user1@example.com"] = (datetime.now(), context1)
        service._context_cache["user2@example.com"] = (datetime.now(), context2)

        assert len(service._context_cache) == 2

        service.clear_cache("user1@example.com")

        assert len(service._context_cache) == 1
        assert "user1@example.com" not in service._context_cache
        assert "user2@example.com" in service._context_cache

    def test_clear_cache_all(self):
        """Test clearing all cache."""
        from aragora.services.cross_channel_context import (
            CrossChannelContextService,
            ChannelContext,
        )

        service = CrossChannelContextService()

        # Add cached contexts
        for i in range(3):
            context = ChannelContext(user_email=f"user{i}@example.com")
            service._context_cache[f"user{i}@example.com"] = (datetime.now(), context)

        assert len(service._context_cache) == 3

        service.clear_cache()

        assert len(service._context_cache) == 0

    def test_calculate_derived_signals_no_signals(self):
        """Test derived signal calculation with no signals."""
        from aragora.services.cross_channel_context import (
            CrossChannelContextService,
            ChannelContext,
        )

        service = CrossChannelContextService()
        context = ChannelContext(user_email="user@example.com")

        result = service._calculate_derived_signals(context)

        assert result.overall_activity_score == 0.0
        assert result.is_likely_busy is False
        assert result.suggested_response_window == "Within business hours"

    def test_calculate_derived_signals_with_slack(self):
        """Test derived signal calculation with Slack signal."""
        from aragora.services.cross_channel_context import (
            CrossChannelContextService,
            ChannelContext,
            SlackActivitySignal,
        )

        service = CrossChannelContextService()
        context = ChannelContext(
            user_email="user@example.com",
            slack=SlackActivitySignal(
                user_email="user@example.com",
                is_online=True,
                activity_score=0.6,
            ),
        )

        result = service._calculate_derived_signals(context)

        assert result.overall_activity_score == 0.6
        assert result.is_likely_busy is False
        assert result.suggested_response_window == "Now - user is active"

    def test_calculate_derived_signals_busy_calendar(self):
        """Test derived signal calculation with busy calendar."""
        from aragora.services.cross_channel_context import (
            CrossChannelContextService,
            ChannelContext,
            CalendarSignal,
        )

        service = CrossChannelContextService()
        context = ChannelContext(
            user_email="user@example.com",
            calendar=CalendarSignal(
                user_email="user@example.com",
                meeting_density_score=0.85,  # Back-to-back meetings
            ),
        )

        result = service._calculate_derived_signals(context)

        assert result.is_likely_busy is True
        assert result.suggested_response_window == "Evening or next morning"

    def test_calculate_derived_signals_high_activity(self):
        """Test derived signal calculation with high overall activity."""
        from aragora.services.cross_channel_context import (
            CrossChannelContextService,
            ChannelContext,
            SlackActivitySignal,
            DriveActivitySignal,
        )

        service = CrossChannelContextService()
        context = ChannelContext(
            user_email="user@example.com",
            slack=SlackActivitySignal(
                user_email="user@example.com",
                is_online=True,
                activity_score=0.9,
            ),
            drive=DriveActivitySignal(
                user_email="user@example.com",
                activity_score=0.9,
            ),
        )

        result = service._calculate_derived_signals(context)

        # Average of 0.9 and 0.9 = 0.9
        assert result.overall_activity_score == 0.9
        assert result.is_likely_busy is True  # > 0.8

    @pytest.mark.asyncio
    async def test_get_user_context_cached(self):
        """Test getting cached user context."""
        from aragora.services.cross_channel_context import (
            CrossChannelContextService,
            ChannelContext,
        )

        service = CrossChannelContextService()

        # Pre-populate cache
        cached_context = ChannelContext(
            user_email="user@example.com",
            overall_activity_score=0.5,
        )
        service._context_cache["user@example.com"] = (datetime.now(), cached_context)

        # Get context (should return cached)
        result = await service.get_user_context("user@example.com")

        assert result.user_email == "user@example.com"
        assert result.overall_activity_score == 0.5

    @pytest.mark.asyncio
    async def test_get_user_context_expired_cache(self):
        """Test getting context with expired cache."""
        from aragora.services.cross_channel_context import (
            CrossChannelContextService,
            ChannelContext,
        )

        service = CrossChannelContextService(cache_ttl_seconds=60)

        # Pre-populate cache with old timestamp
        cached_context = ChannelContext(
            user_email="user@example.com",
            overall_activity_score=0.5,
        )
        old_time = datetime.now() - timedelta(seconds=120)  # 2 minutes old
        service._context_cache["user@example.com"] = (old_time, cached_context)

        # Get context (should refresh due to expiry)
        result = await service.get_user_context("user@example.com")

        # New context created (no connectors, so minimal context)
        assert result.user_email == "user@example.com"

    @pytest.mark.asyncio
    async def test_get_user_context_force_refresh(self):
        """Test forcing context refresh."""
        from aragora.services.cross_channel_context import (
            CrossChannelContextService,
            ChannelContext,
        )

        service = CrossChannelContextService()

        # Pre-populate cache
        cached_context = ChannelContext(
            user_email="user@example.com",
            overall_activity_score=0.5,
        )
        service._context_cache["user@example.com"] = (datetime.now(), cached_context)

        # Force refresh (should bypass cache)
        result = await service.get_user_context("user@example.com", force_refresh=True)

        # New context created
        assert result.user_email == "user@example.com"

    @pytest.mark.asyncio
    async def test_resolve_slack_user_from_memory_cache(self):
        """Test resolving Slack user from in-memory cache."""
        from aragora.services.cross_channel_context import CrossChannelContextService

        service = CrossChannelContextService()

        # Pre-populate memory cache
        service._email_to_slack_id["user@example.com"] = "U12345"

        result = await service._resolve_slack_user("user@example.com")

        assert result == "U12345"

    @pytest.mark.asyncio
    async def test_resolve_slack_user_from_store(self):
        """Test resolving Slack user from persistent store."""
        from aragora.services.cross_channel_context import CrossChannelContextService

        mock_store = AsyncMock()
        mock_mapping = MagicMock()
        mock_mapping.platform_user_id = "U67890"
        mock_store.get_user_mapping.return_value = mock_mapping

        service = CrossChannelContextService(integration_store=mock_store)

        result = await service._resolve_slack_user("user@example.com")

        assert result == "U67890"
        assert service._email_to_slack_id["user@example.com"] == "U67890"
        mock_store.get_user_mapping.assert_called_once()

    @pytest.mark.asyncio
    async def test_resolve_slack_user_from_api(self):
        """Test resolving Slack user from Slack API."""
        from aragora.services.cross_channel_context import CrossChannelContextService

        mock_slack = AsyncMock()
        mock_slack.get_user_by_email.return_value = {
            "id": "U11111",
            "real_name": "Test User",
            "name": "testuser",
        }

        service = CrossChannelContextService(slack_connector=mock_slack)

        result = await service._resolve_slack_user("user@example.com")

        assert result == "U11111"
        assert service._email_to_slack_id["user@example.com"] == "U11111"
        mock_slack.get_user_by_email.assert_called_once_with("user@example.com")

    @pytest.mark.asyncio
    async def test_resolve_slack_user_not_found(self):
        """Test resolving Slack user when not found."""
        from aragora.services.cross_channel_context import CrossChannelContextService

        mock_slack = AsyncMock()
        mock_slack.get_user_by_email.return_value = None

        service = CrossChannelContextService(slack_connector=mock_slack)

        result = await service._resolve_slack_user("unknown@example.com")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_slack_signal_no_connector(self):
        """Test getting Slack signal without connector."""
        from aragora.services.cross_channel_context import CrossChannelContextService

        service = CrossChannelContextService()

        result = await service._get_slack_signal("user@example.com")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_slack_signal_with_presence(self):
        """Test getting Slack signal with presence data."""
        from aragora.services.cross_channel_context import CrossChannelContextService

        mock_slack = AsyncMock()
        mock_slack.get_user_by_email.return_value = {"id": "U12345", "name": "user"}
        mock_slack.get_user_presence.return_value = {"presence": "active"}
        mock_slack.search_messages.return_value = []

        service = CrossChannelContextService(slack_connector=mock_slack)

        result = await service._get_slack_signal("user@example.com")

        assert result is not None
        assert result.is_online is True
        assert result.activity_score >= 0.2  # Online bonus

    @pytest.mark.asyncio
    async def test_get_slack_signal_with_messages(self):
        """Test getting Slack signal with message data."""
        from aragora.services.cross_channel_context import CrossChannelContextService

        mock_slack = AsyncMock()
        mock_slack.get_user_by_email.return_value = {"id": "U12345", "name": "user"}
        mock_slack.get_user_presence.return_value = {"presence": "away"}
        mock_slack.search_messages.side_effect = [
            # First call: user's messages
            [
                {"channel": "C123", "text": "Hello", "ts": "1234567890.123456"},
                {"channel": "C456", "text": "Urgent: need help!", "ts": "1234567891.123456"},
            ],
            # Second call: mentions
            [{"text": "Hey <@U12345>"}],
        ]

        service = CrossChannelContextService(slack_connector=mock_slack)

        result = await service._get_slack_signal("user@example.com")

        assert result is not None
        assert len(result.active_channels) == 2
        assert "C123" in result.active_channels
        assert result.recent_mentions == 1
        assert len(result.urgent_threads) >= 1  # Found urgent keyword

    @pytest.mark.asyncio
    async def test_load_mappings_from_store(self):
        """Test loading user mappings from store."""
        from aragora.services.cross_channel_context import CrossChannelContextService

        mock_store = AsyncMock()
        mock_mapping1 = MagicMock()
        mock_mapping1.email = "user1@example.com"
        mock_mapping1.platform_user_id = "U11111"

        mock_mapping2 = MagicMock()
        mock_mapping2.email = "user2@example.com"
        mock_mapping2.platform_user_id = "U22222"

        mock_store.list_user_mappings.return_value = [mock_mapping1, mock_mapping2]

        service = CrossChannelContextService(integration_store=mock_store)

        count = await service.load_mappings_from_store()

        assert count == 2
        assert service._email_to_slack_id["user1@example.com"] == "U11111"
        assert service._email_to_slack_id["user2@example.com"] == "U22222"

    @pytest.mark.asyncio
    async def test_load_mappings_no_store(self):
        """Test loading mappings without store."""
        from aragora.services.cross_channel_context import CrossChannelContextService

        service = CrossChannelContextService()

        count = await service.load_mappings_from_store()

        assert count == 0


class TestEmailContextBoostComputation:
    """Tests for email context boost computation."""

    @pytest.mark.asyncio
    async def test_get_email_context_basic(self):
        """Test basic email context computation."""
        from aragora.services.cross_channel_context import CrossChannelContextService

        # Create mock email
        mock_email = MagicMock()
        mock_email.id = "email_123"
        mock_email.from_address = "sender@example.com"
        mock_email.subject = "Test Subject"
        mock_email.body_text = "Test body content"

        service = CrossChannelContextService()

        boost = await service.get_email_context(mock_email)

        assert boost.email_id == "email_123"
        # No Slack connector, so no boost
        assert boost.slack_activity_boost == 0.0

    @pytest.mark.asyncio
    async def test_get_email_context_with_active_sender(self):
        """Test email context with active Slack sender."""
        from aragora.services.cross_channel_context import (
            CrossChannelContextService,
            ChannelContext,
            SlackActivitySignal,
        )

        mock_email = MagicMock()
        mock_email.id = "email_456"
        mock_email.from_address = "sender@example.com"
        mock_email.subject = "Important"
        mock_email.body_text = "Please review"

        service = CrossChannelContextService()

        # Pre-populate context cache with active sender
        cached_context = ChannelContext(
            user_email="sender@example.com",
            slack=SlackActivitySignal(
                user_email="sender@example.com",
                is_online=True,
                active_channels=["#general", "#engineering"],
            ),
        )
        service._context_cache["sender@example.com"] = (datetime.now(), cached_context)

        boost = await service.get_email_context(mock_email)

        assert boost.slack_activity_boost == 0.15
        assert "active on Slack" in boost.slack_reason
        assert len(boost.related_slack_channels) <= 3

    @pytest.mark.asyncio
    async def test_get_email_context_with_urgent_threads(self):
        """Test email context with sender having urgent threads."""
        from aragora.services.cross_channel_context import (
            CrossChannelContextService,
            ChannelContext,
            SlackActivitySignal,
        )

        mock_email = MagicMock()
        mock_email.id = "email_789"
        mock_email.from_address = "sender@example.com"
        mock_email.subject = "Follow up"
        mock_email.body_text = "Regarding our discussion"

        service = CrossChannelContextService()

        # Pre-populate with sender having urgent threads
        cached_context = ChannelContext(
            user_email="sender@example.com",
            slack=SlackActivitySignal(
                user_email="sender@example.com",
                is_online=True,
                urgent_threads=["ts_1", "ts_2", "ts_3"],
            ),
        )
        service._context_cache["sender@example.com"] = (datetime.now(), cached_context)

        boost = await service.get_email_context(mock_email)

        # 0.15 (online) + 0.1 (urgent threads)
        assert boost.slack_activity_boost == 0.25
        assert "urgent threads" in boost.slack_reason


class TestCreateContextService:
    """Tests for create_context_service factory function."""

    @pytest.mark.asyncio
    async def test_create_service_minimal(self):
        """Test creating service with minimal config."""
        from aragora.services.cross_channel_context import create_context_service

        service = await create_context_service()

        assert service.slack is None
        assert service._user_id == "default"

    @pytest.mark.asyncio
    async def test_create_service_with_knowledge_mound(self):
        """Test creating service with knowledge mound."""
        from aragora.services.cross_channel_context import create_context_service

        mock_mound = MagicMock()

        service = await create_context_service(
            knowledge_mound=mock_mound,
            user_id="tenant_456",
            load_mappings=False,
        )

        assert service.mound is mock_mound
        assert service._user_id == "tenant_456"
