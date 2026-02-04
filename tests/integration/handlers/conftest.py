"""
Shared fixtures for bot handler integration tests.

Provides mock handlers, platform contexts, and debate state management
for cross-platform integration testing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


@dataclass
class PlatformConfig:
    """Configuration for a test platform."""

    name: str
    signing_secret: str = "test_secret"
    bot_token: str = "xoxb-test-token"
    app_id: str = "A123456"
    team_id: str = "T123456"


@dataclass
class MockMessage:
    """Mock message for testing."""

    platform: str
    channel_id: str
    user_id: str
    text: str
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())
    thread_id: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class MockDebateContext:
    """Context for tracking a debate across platforms."""

    debate_id: str
    task: str
    origin_platform: str
    origin_channel: str
    origin_user: str
    participants: list = field(default_factory=list)
    votes: dict = field(default_factory=dict)
    status: str = "pending"


@pytest.fixture
def slack_config() -> PlatformConfig:
    """Slack platform configuration."""
    return PlatformConfig(
        name="slack",
        signing_secret="slack_signing_secret_test",
        bot_token="xoxb-slack-bot-token",
        app_id="A_SLACK_APP",
        team_id="T_SLACK_TEAM",
    )


@pytest.fixture
def teams_config() -> PlatformConfig:
    """Teams platform configuration."""
    return PlatformConfig(
        name="teams",
        signing_secret="teams_app_secret",
        bot_token="teams_bot_token",
        app_id="teams_app_id",
        team_id="teams_tenant_id",
    )


@pytest.fixture
def telegram_config() -> PlatformConfig:
    """Telegram platform configuration."""
    return PlatformConfig(
        name="telegram",
        signing_secret="telegram_bot_token",
        bot_token="123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
        app_id="telegram_bot",
    )


@pytest.fixture
def whatsapp_config() -> PlatformConfig:
    """WhatsApp platform configuration."""
    return PlatformConfig(
        name="whatsapp",
        signing_secret="whatsapp_app_secret",
        bot_token="whatsapp_access_token",
        app_id="whatsapp_phone_number_id",
    )


@pytest.fixture
def mock_slack_handler():
    """Create mock Slack handler."""
    handler = MagicMock()
    handler.name = "slack"
    handler.send_message = AsyncMock(return_value={"ok": True, "ts": "1234567890.123456"})
    handler.update_message = AsyncMock(return_value={"ok": True})
    handler.add_reaction = AsyncMock(return_value={"ok": True})
    handler.verify_signature = MagicMock(return_value=True)
    return handler


@pytest.fixture
def mock_teams_handler():
    """Create mock Teams handler."""
    handler = MagicMock()
    handler.name = "teams"
    handler.send_message = AsyncMock(return_value={"id": "msg-teams-123"})
    handler.update_message = AsyncMock(return_value={"id": "msg-teams-123"})
    handler.send_card = AsyncMock(return_value={"id": "card-teams-123"})
    handler.verify_token = MagicMock(return_value=True)
    return handler


@pytest.fixture
def mock_telegram_handler():
    """Create mock Telegram handler."""
    handler = MagicMock()
    handler.name = "telegram"
    handler.send_message = AsyncMock(return_value={"ok": True, "result": {"message_id": 100}})
    handler.edit_message = AsyncMock(return_value={"ok": True})
    handler.answer_callback = AsyncMock(return_value={"ok": True})
    handler.verify_webhook = MagicMock(return_value=True)
    return handler


@pytest.fixture
def mock_whatsapp_handler():
    """Create mock WhatsApp handler."""
    handler = MagicMock()
    handler.name = "whatsapp"
    handler.send_message = AsyncMock(return_value={"messages": [{"id": "wamid.HBgLMTIz"}]})
    handler.send_template = AsyncMock(return_value={"messages": [{"id": "wamid.template123"}]})
    handler.mark_read = AsyncMock(return_value={"success": True})
    handler.verify_signature = MagicMock(return_value=True)
    return handler


@pytest.fixture
def all_platform_handlers(
    mock_slack_handler,
    mock_teams_handler,
    mock_telegram_handler,
    mock_whatsapp_handler,
):
    """Dictionary of all platform handlers."""
    return {
        "slack": mock_slack_handler,
        "teams": mock_teams_handler,
        "telegram": mock_telegram_handler,
        "whatsapp": mock_whatsapp_handler,
    }


@pytest.fixture
def mock_debate_state_store():
    """Create mock debate state store."""
    store = MagicMock()
    store._debates: dict[str, MockDebateContext] = {}

    async def create_debate(debate_id: str, context: MockDebateContext):
        store._debates[debate_id] = context
        return context

    async def get_debate(debate_id: str):
        return store._debates.get(debate_id)

    async def update_debate(debate_id: str, updates: dict):
        if debate_id in store._debates:
            debate = store._debates[debate_id]
            for key, value in updates.items():
                setattr(debate, key, value)
            return True
        return False

    async def add_vote(debate_id: str, voter: str, choice: str):
        if debate_id in store._debates:
            store._debates[debate_id].votes[voter] = choice
            return True
        return False

    store.create_debate = AsyncMock(side_effect=create_debate)
    store.get_debate = AsyncMock(side_effect=get_debate)
    store.update_debate = AsyncMock(side_effect=update_debate)
    store.add_vote = AsyncMock(side_effect=add_vote)

    return store


@pytest.fixture
def mock_debate_router(mock_debate_state_store, all_platform_handlers):
    """Create mock debate router for cross-platform routing."""
    router = MagicMock()
    router.state_store = mock_debate_state_store
    router.handlers = all_platform_handlers

    async def route_to_origin(debate_id: str, result: dict):
        debate = await mock_debate_state_store.get_debate(debate_id)
        if debate:
            handler = all_platform_handlers.get(debate.origin_platform)
            if handler:
                await handler.send_message(
                    channel=debate.origin_channel,
                    text=f"Debate result: {result.get('consensus', 'No consensus')}",
                )
                return {"delivered": True, "platform": debate.origin_platform}
        return {"delivered": False, "error": "Debate not found"}

    router.route_to_origin = AsyncMock(side_effect=route_to_origin)
    return router


@pytest.fixture
def mock_rate_limiter():
    """Create mock rate limiter with per-platform tracking."""
    limiter = MagicMock()
    limiter._counts: dict[str, int] = {}
    limiter._limits = {
        "slack": 100,
        "teams": 100,
        "telegram": 30,
        "whatsapp": 80,
    }

    def check_limit(platform: str, user_id: str) -> tuple[bool, str | None]:
        key = f"{platform}:{user_id}"
        current = limiter._counts.get(key, 0)
        limit = limiter._limits.get(platform, 100)
        if current >= limit:
            return False, "Rate limit exceeded"
        return True, None

    def record_request(platform: str, user_id: str):
        key = f"{platform}:{user_id}"
        limiter._counts[key] = limiter._counts.get(key, 0) + 1

    def reset_limits():
        limiter._counts.clear()

    limiter.check_limit = MagicMock(side_effect=check_limit)
    limiter.record_request = MagicMock(side_effect=record_request)
    limiter.reset_limits = MagicMock(side_effect=reset_limits)

    return limiter


@pytest.fixture
def sample_debate_context() -> MockDebateContext:
    """Create sample debate context for testing."""
    return MockDebateContext(
        debate_id="test-debate-123",
        task="Should we adopt microservices architecture?",
        origin_platform="slack",
        origin_channel="C_TECH_DECISIONS",
        origin_user="U_LEAD_ARCHITECT",
        participants=[
            "slack:U_LEAD_ARCHITECT",
            "slack:U_SENIOR_DEV",
            "teams:engineer@company.com",
        ],
    )


@pytest.fixture
def sample_messages() -> dict[str, MockMessage]:
    """Create sample messages from different platforms."""
    return {
        "slack": MockMessage(
            platform="slack",
            channel_id="C123456",
            user_id="U789012",
            text="<@BOTID> Should we use Redis for caching?",
            thread_id="1234567890.123456",
        ),
        "teams": MockMessage(
            platform="teams",
            channel_id="19:abc123@thread.tacv2",
            user_id="user@contoso.com",
            text="@bot Should we adopt Kubernetes?",
        ),
        "telegram": MockMessage(
            platform="telegram",
            channel_id="-100123456789",
            user_id="987654321",
            text="/ask What database should we use?",
        ),
        "whatsapp": MockMessage(
            platform="whatsapp",
            channel_id="+1234567890",
            user_id="+0987654321",
            text="Should we migrate to the cloud?",
        ),
    }
