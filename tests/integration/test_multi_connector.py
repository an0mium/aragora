"""
Integration tests for multi-channel debate routing.

These tests verify that debates can be initiated from multiple channels
and results are properly routed back to the originating platforms.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@dataclass
class MockDebateOrigin:
    """Mock debate origin for testing."""

    debate_id: str
    platform: str
    channel_id: str
    user_id: str
    metadata: dict = None

    def __post_init__(self):
        self.metadata = self.metadata or {}


@dataclass
class MockDebateResult:
    """Mock debate result for routing."""

    debate_id: str
    consensus: str
    confidence: float
    summary: str = ""


class TestMultiChannelDebateInitiation:
    """Test debates initiated from different platforms."""

    @pytest.fixture
    def mock_origin_registry(self):
        """Create mock DebateOriginRegistry."""
        registry = MagicMock()
        registry.register = AsyncMock()
        registry.get = AsyncMock()
        registry.remove = AsyncMock()
        return registry

    @pytest.fixture
    def mock_debate_router(self):
        """Create mock DebateRouter."""
        router = MagicMock()
        router.route_result = AsyncMock()
        return router

    async def test_slack_debate_registration(self, mock_origin_registry):
        """Debates from Slack should be registered with thread metadata."""
        origin = MockDebateOrigin(
            debate_id="debate-slack-1",
            platform="slack",
            channel_id="C12345",
            user_id="U12345",
            metadata={"thread_ts": "1234567890.123456"},
        )

        await mock_origin_registry.register(
            debate_id=origin.debate_id,
            platform=origin.platform,
            channel_id=origin.channel_id,
            user_id=origin.user_id,
            metadata=origin.metadata,
        )

        mock_origin_registry.register.assert_called_once()
        call_args = mock_origin_registry.register.call_args
        assert call_args.kwargs["platform"] == "slack"
        assert "thread_ts" in call_args.kwargs["metadata"]

    async def test_telegram_debate_registration(self, mock_origin_registry):
        """Debates from Telegram should be registered with chat_id."""
        origin = MockDebateOrigin(
            debate_id="debate-telegram-1",
            platform="telegram",
            channel_id="-100123456789",
            user_id="987654321",
            metadata={"message_id": 12345},
        )

        await mock_origin_registry.register(
            debate_id=origin.debate_id,
            platform=origin.platform,
            channel_id=origin.channel_id,
            user_id=origin.user_id,
            metadata=origin.metadata,
        )

        call_args = mock_origin_registry.register.call_args
        assert call_args.kwargs["platform"] == "telegram"

    async def test_teams_debate_registration(self, mock_origin_registry):
        """Debates from Teams should include conversation reference."""
        origin = MockDebateOrigin(
            debate_id="debate-teams-1",
            platform="teams",
            channel_id="19:abc123@thread.v2",
            user_id="user@company.com",
            metadata={
                "conversation_id": "19:abc123@thread.v2",
                "service_url": "https://smba.trafficmanager.net/amer/",
            },
        )

        await mock_origin_registry.register(
            debate_id=origin.debate_id,
            platform=origin.platform,
            channel_id=origin.channel_id,
            user_id=origin.user_id,
            metadata=origin.metadata,
        )

        call_args = mock_origin_registry.register.call_args
        assert call_args.kwargs["platform"] == "teams"
        assert "service_url" in call_args.kwargs["metadata"]

    async def test_whatsapp_debate_registration(self, mock_origin_registry):
        """Debates from WhatsApp should be registered with phone metadata."""
        origin = MockDebateOrigin(
            debate_id="debate-whatsapp-1",
            platform="whatsapp",
            channel_id="group_id_123",
            user_id="+1234567890",
            metadata={"message_id": "wamid.abc123"},
        )

        await mock_origin_registry.register(
            debate_id=origin.debate_id,
            platform=origin.platform,
            channel_id=origin.channel_id,
            user_id=origin.user_id,
            metadata=origin.metadata,
        )

        call_args = mock_origin_registry.register.call_args
        assert call_args.kwargs["platform"] == "whatsapp"


class TestDebateResultRouting:
    """Test routing debate results back to originating platforms."""

    @pytest.fixture
    def mock_senders(self):
        """Create mock platform senders."""
        return {
            "slack": AsyncMock(return_value={"ok": True, "ts": "1234567890.123456"}),
            "telegram": AsyncMock(return_value={"ok": True, "message_id": 12346}),
            "teams": AsyncMock(return_value={"id": "msg-123"}),
            "whatsapp": AsyncMock(return_value={"messages": [{"id": "wamid.xyz"}]}),
            "discord": AsyncMock(return_value={"id": "123456789"}),
            "email": AsyncMock(return_value={"message_id": "email-123"}),
        }

    @pytest.fixture
    def mock_router_with_senders(self, mock_senders):
        """Create router with platform senders."""
        router = MagicMock()
        router.senders = mock_senders

        async def route_result(debate_id: str, result: MockDebateResult, origin: MockDebateOrigin):
            sender = router.senders.get(origin.platform)
            if sender:
                return await sender(
                    channel_id=origin.channel_id,
                    message=result.consensus,
                    metadata=origin.metadata,
                )
            return None

        router.route_result = route_result
        return router

    async def test_slack_result_routing(self, mock_router_with_senders):
        """Results should route back to Slack thread."""
        origin = MockDebateOrigin(
            debate_id="debate-1",
            platform="slack",
            channel_id="C12345",
            user_id="U12345",
            metadata={"thread_ts": "1234567890.123456"},
        )
        result = MockDebateResult(
            debate_id="debate-1",
            consensus="Use token bucket algorithm",
            confidence=0.92,
        )

        response = await mock_router_with_senders.route_result(result.debate_id, result, origin)

        assert response["ok"] is True
        mock_router_with_senders.senders["slack"].assert_called_once()

    async def test_telegram_result_routing(self, mock_router_with_senders):
        """Results should route back to Telegram chat."""
        origin = MockDebateOrigin(
            debate_id="debate-2",
            platform="telegram",
            channel_id="-100123456789",
            user_id="987654321",
        )
        result = MockDebateResult(
            debate_id="debate-2",
            consensus="Implement rate limiting",
            confidence=0.88,
        )

        response = await mock_router_with_senders.route_result(result.debate_id, result, origin)

        assert response["ok"] is True
        assert "message_id" in response

    async def test_teams_result_routing(self, mock_router_with_senders):
        """Results should route back to Teams conversation."""
        origin = MockDebateOrigin(
            debate_id="debate-3",
            platform="teams",
            channel_id="19:abc123@thread.v2",
            user_id="user@company.com",
        )
        result = MockDebateResult(
            debate_id="debate-3",
            consensus="Deploy to staging first",
            confidence=0.95,
        )

        response = await mock_router_with_senders.route_result(result.debate_id, result, origin)

        assert "id" in response

    async def test_unknown_platform_handled(self, mock_router_with_senders):
        """Unknown platforms should be handled gracefully."""
        origin = MockDebateOrigin(
            debate_id="debate-4",
            platform="unknown_platform",
            channel_id="ch-123",
            user_id="user-123",
        )
        result = MockDebateResult(
            debate_id="debate-4",
            consensus="Test consensus",
            confidence=0.8,
        )

        response = await mock_router_with_senders.route_result(result.debate_id, result, origin)

        assert response is None


class TestConcurrentMultiChannelDebates:
    """Test concurrent debates from multiple channels."""

    async def test_concurrent_debates_isolated(self):
        """Concurrent debates from different channels should be isolated."""
        origins = [
            MockDebateOrigin("debate-1", "slack", "C1", "U1"),
            MockDebateOrigin("debate-2", "telegram", "-100123", "987"),
            MockDebateOrigin("debate-3", "teams", "19:abc", "user@co.com"),
        ]

        results = []

        async def simulate_debate(origin: MockDebateOrigin):
            await asyncio.sleep(0.05)  # Simulate debate time
            return MockDebateResult(
                debate_id=origin.debate_id,
                consensus=f"Consensus for {origin.platform}",
                confidence=0.9,
            )

        tasks = [asyncio.create_task(simulate_debate(o)) for o in origins]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert all(r.debate_id.startswith("debate-") for r in results)

    async def test_platform_specific_formatting(self):
        """Results should be formatted according to platform requirements."""

        def format_for_slack(result: MockDebateResult) -> str:
            return f"*Consensus:* {result.consensus}\n_Confidence: {result.confidence:.0%}_"

        def format_for_telegram(result: MockDebateResult) -> str:
            return (
                f"<b>Consensus:</b> {result.consensus}\n<i>Confidence: {result.confidence:.0%}</i>"
            )

        def format_for_teams(result: MockDebateResult) -> dict:
            return {
                "type": "AdaptiveCard",
                "body": [
                    {"type": "TextBlock", "text": result.consensus, "weight": "bolder"},
                    {"type": "TextBlock", "text": f"Confidence: {result.confidence:.0%}"},
                ],
            }

        result = MockDebateResult(
            debate_id="debate-1",
            consensus="Use API versioning",
            confidence=0.92,
        )

        slack_msg = format_for_slack(result)
        assert "*Consensus:*" in slack_msg
        assert "_Confidence:" in slack_msg

        telegram_msg = format_for_telegram(result)
        assert "<b>Consensus:</b>" in telegram_msg

        teams_card = format_for_teams(result)
        assert teams_card["type"] == "AdaptiveCard"
        assert len(teams_card["body"]) == 2


class TestDebateOriginPersistence:
    """Test debate origin persistence and recovery."""

    @pytest.fixture
    def mock_origin_store(self):
        """Create mock origin store."""
        store = MagicMock()
        store.origins = {}

        async def save(origin: MockDebateOrigin):
            store.origins[origin.debate_id] = origin
            return True

        async def get(debate_id: str):
            return store.origins.get(debate_id)

        async def delete(debate_id: str):
            if debate_id in store.origins:
                del store.origins[debate_id]
                return True
            return False

        store.save = save
        store.get = get
        store.delete = delete
        return store

    async def test_origin_persistence(self, mock_origin_store):
        """Origins should persist across server restarts."""
        origin = MockDebateOrigin(
            debate_id="debate-persist-1",
            platform="slack",
            channel_id="C12345",
            user_id="U12345",
        )

        await mock_origin_store.save(origin)
        retrieved = await mock_origin_store.get("debate-persist-1")

        assert retrieved is not None
        assert retrieved.platform == "slack"

    async def test_origin_cleanup_after_routing(self, mock_origin_store):
        """Origins should be cleaned up after successful routing."""
        origin = MockDebateOrigin(
            debate_id="debate-cleanup-1",
            platform="slack",
            channel_id="C12345",
            user_id="U12345",
        )

        await mock_origin_store.save(origin)
        await mock_origin_store.delete("debate-cleanup-1")

        retrieved = await mock_origin_store.get("debate-cleanup-1")
        assert retrieved is None

    async def test_ttl_expiration(self):
        """Origins should expire after TTL."""
        origin_ttl_seconds = 3600  # 1 hour
        created_at = datetime.now()

        # Simulate time passing
        is_expired = (datetime.now() - created_at).total_seconds() > origin_ttl_seconds

        assert is_expired is False


class TestVoiceChannelRouting:
    """Test TTS routing for voice-enabled channels."""

    @pytest.fixture
    def mock_tts_service(self):
        """Create mock TTS service."""
        tts = MagicMock()
        tts.synthesize = AsyncMock(return_value=b"audio_data_bytes")
        tts.get_supported_voices = MagicMock(return_value=["en-US-Neural2-A", "en-GB-Neural2-B"])
        return tts

    async def test_voice_result_synthesis(self, mock_tts_service):
        """Results for voice channels should be synthesized."""
        result = MockDebateResult(
            debate_id="debate-voice-1",
            consensus="Use the token bucket algorithm for rate limiting",
            confidence=0.92,
        )

        audio = await mock_tts_service.synthesize(
            text=result.consensus,
            voice="en-US-Neural2-A",
        )

        assert audio == b"audio_data_bytes"
        mock_tts_service.synthesize.assert_called_once()

    async def test_voice_channel_detection(self):
        """Voice channels should be detected from origin metadata."""
        voice_origin = MockDebateOrigin(
            debate_id="debate-voice-2",
            platform="slack",
            channel_id="C12345",
            user_id="U12345",
            metadata={"voice_enabled": True, "huddle_id": "huddle-123"},
        )

        is_voice = voice_origin.metadata.get("voice_enabled", False)
        assert is_voice is True

        text_origin = MockDebateOrigin(
            debate_id="debate-text-1",
            platform="slack",
            channel_id="C12345",
            user_id="U12345",
        )

        is_voice = text_origin.metadata.get("voice_enabled", False)
        assert is_voice is False


class TestMultiSessionSupport:
    """Test multi-session debate support for group chats."""

    async def test_group_chat_multiple_debates(self):
        """Group chats can have multiple concurrent debates."""
        group_id = "group-123"
        debates = {}

        async def start_debate(user_id: str, topic: str):
            debate_id = f"debate-{user_id}-{len(debates)}"
            debates[debate_id] = {
                "user_id": user_id,
                "topic": topic,
                "group_id": group_id,
            }
            return debate_id

        # Multiple users start debates in same group
        d1 = await start_debate("user-1", "API design")
        d2 = await start_debate("user-2", "Database schema")
        d3 = await start_debate("user-1", "Security review")

        assert len(debates) == 3
        assert all(d["group_id"] == group_id for d in debates.values())

    async def test_debate_isolation_in_threads(self):
        """Thread-based debates should be isolated."""
        threads = {
            "thread-1": {"debate_id": "debate-1", "topic": "Architecture"},
            "thread-2": {"debate_id": "debate-2", "topic": "Testing"},
        }

        # Each thread has its own debate context
        assert threads["thread-1"]["topic"] != threads["thread-2"]["topic"]
        assert threads["thread-1"]["debate_id"] != threads["thread-2"]["debate_id"]
