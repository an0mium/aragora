"""
Performance Benchmarks for Chat Platform Connectors.

Measures throughput and latency for:
- Message serialization/parsing
- Webhook event parsing
- Block formatting
- Evidence relevance scoring
"""

import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# Message Parsing Benchmarks
# =============================================================================


class TestWebhookParsingBenchmarks:
    """Benchmarks for webhook event parsing performance."""

    @pytest.fixture
    def discord_connector(self):
        """Create Discord connector."""
        from aragora.connectors.chat.discord import DiscordConnector

        return DiscordConnector(bot_token="test")

    @pytest.fixture
    def teams_connector(self):
        """Create Teams connector."""
        from aragora.connectors.chat.teams import TeamsConnector

        return TeamsConnector(app_id="test", app_password="test")

    @pytest.fixture
    def slack_connector(self):
        """Create Slack connector."""
        from aragora.connectors.chat.slack import SlackConnector

        return SlackConnector(bot_token="xoxb-test")

    @pytest.fixture
    def discord_message_payload(self):
        """Sample Discord message payload."""
        return json.dumps(
            {
                "type": 0,
                "id": "msg-123456789",
                "channel_id": "chan-987654321",
                "guild_id": "guild-111222333",
                "author": {
                    "id": "user-444555666",
                    "username": "testuser",
                    "global_name": "Test User",
                    "avatar": "abc123",
                },
                "content": "This is a test message for benchmarking",
                "timestamp": "2024-01-15T10:30:00.000000+00:00",
                "edited_timestamp": None,
                "attachments": [],
                "embeds": [],
            }
        ).encode("utf-8")

    @pytest.fixture
    def teams_activity_payload(self):
        """Sample Teams Bot Framework activity payload."""
        return json.dumps(
            {
                "type": "message",
                "id": "act-123456789",
                "text": "This is a test message for benchmarking",
                "from": {
                    "id": "user-444555666",
                    "name": "Test User",
                },
                "conversation": {
                    "id": "conv-987654321",
                    "conversationType": "channel",
                },
                "channelId": "msteams",
                "serviceUrl": "https://smba.trafficmanager.net/emea/",
                "timestamp": "2024-01-15T10:30:00.000Z",
            }
        ).encode("utf-8")

    @pytest.fixture
    def slack_event_payload(self):
        """Sample Slack event payload."""
        return json.dumps(
            {
                "type": "event_callback",
                "event": {
                    "type": "message",
                    "channel": "C123456",
                    "user": "U789012",
                    "text": "This is a test message for benchmarking",
                    "ts": "1705315800.123456",
                    "team": "T111222",
                },
                "team_id": "T111222",
                "api_app_id": "A333444",
            }
        ).encode("utf-8")

    def test_discord_webhook_parsing(self, benchmark, discord_connector, discord_message_payload):
        """Benchmark Discord webhook parsing."""
        result = benchmark(
            discord_connector.parse_webhook_event,
            headers={"Content-Type": "application/json"},
            body=discord_message_payload,
        )
        assert result is not None

    def test_teams_webhook_parsing(self, benchmark, teams_connector, teams_activity_payload):
        """Benchmark Teams webhook parsing."""
        result = benchmark(
            teams_connector.parse_webhook_event,
            headers={"Content-Type": "application/json"},
            body=teams_activity_payload,
        )
        assert result is not None

    def test_slack_webhook_parsing(self, benchmark, slack_connector, slack_event_payload):
        """Benchmark Slack webhook parsing."""
        result = benchmark(
            slack_connector.parse_webhook_event,
            headers={"Content-Type": "application/json"},
            body=slack_event_payload,
        )
        assert result is not None


# =============================================================================
# Block Formatting Benchmarks
# =============================================================================


class TestBlockFormattingBenchmarks:
    """Benchmarks for platform-specific block formatting."""

    @pytest.fixture
    def discord_connector(self):
        from aragora.connectors.chat.discord import DiscordConnector

        return DiscordConnector(bot_token="test")

    @pytest.fixture
    def teams_connector(self):
        from aragora.connectors.chat.teams import TeamsConnector

        return TeamsConnector(app_id="test", app_password="test")

    @pytest.fixture
    def slack_connector(self):
        from aragora.connectors.chat.slack import SlackConnector

        return SlackConnector(bot_token="xoxb-test")

    @pytest.fixture
    def rich_content(self):
        """Rich content for block formatting."""
        return {
            "title": "Debate Results",
            "body": "After 5 rounds of deliberation, the consensus reached is that GraphQL provides better flexibility for complex queries while REST remains simpler for basic CRUD operations.",
            "fields": [
                ("Consensus", "Partial Agreement"),
                ("Confidence", "87%"),
                ("Rounds", "5"),
                ("Participants", "4"),
            ],
            "footer": "Generated by Aragora AI",
        }

    def test_discord_format_blocks(self, benchmark, discord_connector, rich_content):
        """Benchmark Discord embed formatting."""
        result = benchmark(
            discord_connector.format_blocks,
            title=rich_content["title"],
            body=rich_content["body"],
            fields=rich_content["fields"],
            footer=rich_content["footer"],
        )
        assert isinstance(result, list)

    def test_teams_format_blocks(self, benchmark, teams_connector, rich_content):
        """Benchmark Teams Adaptive Card formatting."""
        result = benchmark(
            teams_connector.format_blocks,
            title=rich_content["title"],
            body=rich_content["body"],
            fields=rich_content["fields"],
            footer=rich_content["footer"],
        )
        assert isinstance(result, list)

    def test_slack_format_blocks(self, benchmark, slack_connector, rich_content):
        """Benchmark Slack Block Kit formatting."""
        result = benchmark(
            slack_connector.format_blocks,
            title=rich_content["title"],
            body=rich_content["body"],
            fields=rich_content["fields"],
            footer=rich_content["footer"],
        )
        assert isinstance(result, list)


# =============================================================================
# Button Formatting Benchmarks
# =============================================================================


class TestButtonFormattingBenchmarks:
    """Benchmarks for button formatting across platforms."""

    @pytest.fixture
    def connectors(self):
        from aragora.connectors.chat.discord import DiscordConnector
        from aragora.connectors.chat.teams import TeamsConnector
        from aragora.connectors.chat.slack import SlackConnector

        return {
            "discord": DiscordConnector(bot_token="test"),
            "teams": TeamsConnector(app_id="test", app_password="test"),
            "slack": SlackConnector(bot_token="xoxb-test"),
        }

    def test_discord_format_button(self, benchmark, connectors):
        """Benchmark Discord button formatting."""
        result = benchmark(
            connectors["discord"].format_button,
            text="Vote Yes",
            action_id="vote_yes",
            value="yes",
            style="primary",
        )
        assert isinstance(result, dict)

    def test_teams_format_button(self, benchmark, connectors):
        """Benchmark Teams button formatting."""
        result = benchmark(
            connectors["teams"].format_button,
            text="Vote Yes",
            action_id="vote_yes",
            value="yes",
            style="primary",
        )
        assert isinstance(result, dict)

    def test_slack_format_button(self, benchmark, connectors):
        """Benchmark Slack button formatting."""
        result = benchmark(
            connectors["slack"].format_button,
            text="Vote Yes",
            action_id="vote_yes",
            value="yes",
            style="primary",
        )
        assert isinstance(result, dict)


# =============================================================================
# Evidence Relevance Scoring Benchmarks
# =============================================================================


class TestEvidenceRelevanceBenchmarks:
    """Benchmarks for evidence relevance scoring."""

    @pytest.fixture
    def base_connector(self):
        """Create connector with relevance scoring capability."""
        from aragora.connectors.chat.teams import TeamsConnector

        return TeamsConnector(app_id="test", app_password="test")

    @pytest.fixture
    def sample_messages(self):
        """Sample messages for relevance scoring."""
        from aragora.connectors.chat.models import ChatMessage, ChatChannel, ChatUser

        channel = ChatChannel(id="chan-123", platform="teams")
        user = ChatUser(id="user-456", platform="teams", display_name="Test User")

        messages = []
        contents = [
            "We should definitely consider using GraphQL for the new API",
            "REST APIs have been working great for us",
            "What time is the meeting tomorrow?",
            "The GraphQL schema needs more work",
            "Anyone want coffee?",
            "I think REST is more appropriate for simple endpoints",
            "Happy Friday everyone!",
            "GraphQL queries are more flexible",
            "Don't forget to submit your timesheets",
            "The REST vs GraphQL debate continues",
        ]

        for i, content in enumerate(contents):
            messages.append(
                ChatMessage(
                    id=f"msg-{i}",
                    platform="teams",
                    channel=channel,
                    author=user,
                    content=content,
                    timestamp=datetime.utcnow(),
                )
            )

        return messages

    def test_relevance_scoring_single(self, benchmark, base_connector, sample_messages):
        """Benchmark single message relevance scoring."""
        message = sample_messages[0]
        query = "GraphQL API"

        result = benchmark(
            base_connector._compute_message_relevance,
            message,
            query,
        )
        assert 0.0 <= result <= 1.0

    def test_relevance_scoring_batch(self, benchmark, base_connector, sample_messages):
        """Benchmark batch message relevance scoring."""
        query = "GraphQL API"

        def score_batch():
            return [
                base_connector._compute_message_relevance(msg, query) for msg in sample_messages
            ]

        results = benchmark(score_batch)
        assert len(results) == len(sample_messages)
        assert all(0.0 <= r <= 1.0 for r in results)


# =============================================================================
# Serialization Benchmarks
# =============================================================================


class TestSerializationBenchmarks:
    """Benchmarks for message serialization."""

    @pytest.fixture
    def chat_message(self):
        """Create a sample ChatMessage for serialization."""
        from aragora.connectors.chat.models import ChatMessage, ChatChannel, ChatUser

        return ChatMessage(
            id="msg-123",
            platform="discord",
            channel=ChatChannel(
                id="chan-456",
                platform="discord",
                name="general",
                team_id="guild-789",
            ),
            author=ChatUser(
                id="user-111",
                platform="discord",
                username="testuser",
                display_name="Test User",
            ),
            content="This is a test message with some content for benchmarking serialization performance",
            timestamp=datetime.utcnow(),
            metadata={"key1": "value1", "key2": "value2"},
        )

    def test_message_to_dict(self, benchmark, chat_message):
        """Benchmark ChatMessage.to_dict() serialization."""
        result = benchmark(chat_message.to_dict)
        assert isinstance(result, dict)
        assert "id" in result

    def test_message_to_json(self, benchmark, chat_message):
        """Benchmark ChatMessage to JSON serialization."""

        def to_json():
            return json.dumps(chat_message.to_dict())

        result = benchmark(to_json)
        assert isinstance(result, str)


# =============================================================================
# Comparative Benchmarks
# =============================================================================


class TestCrossplatformComparison:
    """Comparative benchmarks across all platforms."""

    @pytest.fixture
    def all_connectors(self):
        """Create all connectors for comparison."""
        from aragora.connectors.chat.discord import DiscordConnector
        from aragora.connectors.chat.teams import TeamsConnector
        from aragora.connectors.chat.slack import SlackConnector
        from aragora.connectors.chat.telegram import TelegramConnector

        return {
            "discord": DiscordConnector(bot_token="test"),
            "teams": TeamsConnector(app_id="test", app_password="test"),
            "slack": SlackConnector(bot_token="xoxb-test"),
            "telegram": TelegramConnector(bot_token="test"),
        }

    @pytest.mark.parametrize("platform", ["discord", "teams", "slack", "telegram"])
    def test_format_blocks_comparison(self, benchmark, all_connectors, platform):
        """Compare format_blocks performance across platforms."""
        connector = all_connectors[platform]

        result = benchmark(
            connector.format_blocks,
            title="Test Title",
            body="Test body content",
        )
        assert isinstance(result, list)

    @pytest.mark.parametrize("platform", ["discord", "teams", "slack", "telegram"])
    def test_format_button_comparison(self, benchmark, all_connectors, platform):
        """Compare format_button performance across platforms."""
        connector = all_connectors[platform]

        result = benchmark(
            connector.format_button,
            text="Click",
            action_id="action",
            value="value",
        )
        assert isinstance(result, dict)
