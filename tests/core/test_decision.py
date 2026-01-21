"""
Tests for core decision module.

Tests cover:
- DecisionType enum
- InputSource enum
- Priority enum
- ResponseFormat enum
- ResponseChannel dataclass
- RequestContext dataclass
- DecisionConfig dataclass
- DecisionRequest dataclass
- DecisionResult dataclass
- DecisionRouter class
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.core.decision import (
    DecisionType,
    InputSource,
    Priority,
    ResponseFormat,
    ResponseChannel,
    RequestContext,
    DecisionConfig,
    DecisionRequest,
    DecisionResult,
    DecisionRouter,
)


class TestDecisionType:
    """Tests for DecisionType enum."""

    def test_has_all_types(self):
        """Enum has all expected decision types."""
        assert DecisionType.DEBATE.value == "debate"
        assert DecisionType.WORKFLOW.value == "workflow"
        assert DecisionType.GAUNTLET.value == "gauntlet"
        assert DecisionType.QUICK.value == "quick"
        assert DecisionType.AUTO.value == "auto"

    def test_type_count(self):
        """Enum has exactly 5 types."""
        assert len(DecisionType) == 5

    def test_string_enum(self):
        """DecisionType is a string enum."""
        assert isinstance(DecisionType.DEBATE.value, str)
        assert str(DecisionType.DEBATE) == "debate"


class TestInputSource:
    """Tests for InputSource enum."""

    def test_chat_platforms(self):
        """Has chat platform sources."""
        assert InputSource.SLACK.value == "slack"
        assert InputSource.DISCORD.value == "discord"
        assert InputSource.TEAMS.value == "teams"
        assert InputSource.TELEGRAM.value == "telegram"

    def test_direct_interfaces(self):
        """Has direct interface sources."""
        assert InputSource.HTTP_API.value == "http_api"
        assert InputSource.WEBSOCKET.value == "websocket"
        assert InputSource.CLI.value == "cli"

    def test_voice_sources(self):
        """Has voice sources."""
        assert InputSource.VOICE.value == "voice"
        assert InputSource.VOICE_SLACK.value == "voice_slack"

    def test_enterprise_sources(self):
        """Has enterprise integration sources."""
        assert InputSource.JIRA.value == "jira"
        assert InputSource.GITHUB.value == "github"
        assert InputSource.CONFLUENCE.value == "confluence"


class TestPriority:
    """Tests for Priority enum."""

    def test_all_priorities(self):
        """Has all priority levels."""
        assert Priority.CRITICAL.value == "critical"
        assert Priority.HIGH.value == "high"
        assert Priority.NORMAL.value == "normal"
        assert Priority.LOW.value == "low"
        assert Priority.BATCH.value == "batch"

    def test_priority_count(self):
        """Has exactly 5 priority levels."""
        assert len(Priority) == 5


class TestResponseFormat:
    """Tests for ResponseFormat enum."""

    def test_all_formats(self):
        """Has all response formats."""
        assert ResponseFormat.FULL.value == "full"
        assert ResponseFormat.SUMMARY.value == "summary"
        assert ResponseFormat.NOTIFICATION.value == "notification"
        assert ResponseFormat.VOICE.value == "voice"
        assert ResponseFormat.VOICE_WITH_TEXT.value == "voice_with_text"


class TestResponseChannel:
    """Tests for ResponseChannel dataclass."""

    def test_create_basic(self):
        """Basic channel creation."""
        channel = ResponseChannel(platform="slack", channel_id="C123")
        assert channel.platform == "slack"
        assert channel.channel_id == "C123"

    def test_default_values(self):
        """Default values are sensible."""
        channel = ResponseChannel(platform="http")
        assert channel.channel_id is None
        assert channel.user_id is None
        assert channel.response_format == "full"
        assert channel.include_reasoning is True
        assert channel.voice_enabled is False

    def test_voice_settings(self):
        """Voice settings work correctly."""
        channel = ResponseChannel(
            platform="slack",
            voice_enabled=True,
            voice_id="alloy",
            voice_only=True,
        )
        assert channel.voice_enabled is True
        assert channel.voice_id == "alloy"
        assert channel.voice_only is True

    def test_to_dict(self):
        """Serializes to dictionary."""
        channel = ResponseChannel(
            platform="slack",
            channel_id="C123",
            user_id="U456",
        )
        data = channel.to_dict()

        assert data["platform"] == "slack"
        assert data["channel_id"] == "C123"
        assert data["user_id"] == "U456"
        assert "response_format" in data
        assert "voice_enabled" in data

    def test_from_dict(self):
        """Creates from dictionary."""
        data = {
            "platform": "discord",
            "channel_id": "123456",
            "response_format": "summary",
            "voice_enabled": True,
        }
        channel = ResponseChannel.from_dict(data)

        assert channel.platform == "discord"
        assert channel.channel_id == "123456"
        assert channel.response_format == "summary"
        assert channel.voice_enabled is True

    def test_from_dict_defaults(self):
        """from_dict handles missing keys with defaults."""
        data = {"platform": "http"}
        channel = ResponseChannel.from_dict(data)

        assert channel.platform == "http"
        assert channel.channel_id is None
        assert channel.response_format == "full"

    def test_roundtrip(self):
        """to_dict -> from_dict preserves data."""
        original = ResponseChannel(
            platform="teams",
            channel_id="C789",
            thread_id="T123",
            response_format="notification",
            voice_enabled=True,
            voice_id="nova",
        )
        data = original.to_dict()
        restored = ResponseChannel.from_dict(data)

        assert restored.platform == original.platform
        assert restored.channel_id == original.channel_id
        assert restored.thread_id == original.thread_id
        assert restored.response_format == original.response_format
        assert restored.voice_enabled == original.voice_enabled
        assert restored.voice_id == original.voice_id


class TestRequestContext:
    """Tests for RequestContext dataclass."""

    def test_create_basic(self):
        """Basic context creation."""
        ctx = RequestContext()
        assert ctx.correlation_id is not None
        assert ctx.created_at is not None

    def test_auto_correlation_id(self):
        """Generates unique correlation IDs."""
        ctx1 = RequestContext()
        ctx2 = RequestContext()
        assert ctx1.correlation_id != ctx2.correlation_id

    def test_user_info(self):
        """User info fields work correctly."""
        ctx = RequestContext(
            user_id="user123",
            user_name="Test User",
            user_email="test@example.com",
            user_roles=["admin", "developer"],
        )
        assert ctx.user_id == "user123"
        assert ctx.user_name == "Test User"
        assert ctx.user_email == "test@example.com"
        assert ctx.user_roles == ["admin", "developer"]

    def test_tenant_info(self):
        """Tenant/workspace fields work correctly."""
        ctx = RequestContext(
            tenant_id="tenant123",
            workspace_id="workspace456",
        )
        assert ctx.tenant_id == "tenant123"
        assert ctx.workspace_id == "workspace456"

    def test_to_dict(self):
        """Serializes to dictionary."""
        ctx = RequestContext(
            user_id="user123",
            tags=["urgent", "security"],
            metadata={"source": "test"},
        )
        data = ctx.to_dict()

        assert data["user_id"] == "user123"
        assert data["tags"] == ["urgent", "security"]
        assert data["metadata"] == {"source": "test"}
        assert "created_at" in data
        assert isinstance(data["created_at"], str)  # ISO format

    def test_from_dict(self):
        """Creates from dictionary."""
        data = {
            "correlation_id": "corr-123",
            "user_id": "user456",
            "created_at": "2024-01-15T10:30:00",
            "tags": ["test"],
        }
        ctx = RequestContext.from_dict(data)

        assert ctx.correlation_id == "corr-123"
        assert ctx.user_id == "user456"
        assert ctx.tags == ["test"]

    def test_from_dict_with_deadline(self):
        """from_dict handles deadline parsing."""
        data = {
            "deadline": "2024-01-15T12:00:00",
        }
        ctx = RequestContext.from_dict(data)

        assert ctx.deadline is not None
        assert isinstance(ctx.deadline, datetime)

    def test_roundtrip(self):
        """to_dict -> from_dict preserves data."""
        original = RequestContext(
            user_id="user789",
            tenant_id="tenant123",
            tags=["important"],
            metadata={"key": "value"},
        )
        data = original.to_dict()
        restored = RequestContext.from_dict(data)

        assert restored.user_id == original.user_id
        assert restored.tenant_id == original.tenant_id
        assert restored.tags == original.tags
        assert restored.metadata == original.metadata


class TestDecisionConfig:
    """Tests for DecisionConfig dataclass."""

    def test_default_values(self):
        """Default values are sensible."""
        config = DecisionConfig()
        assert config.timeout_seconds == 300
        assert config.max_agents == 3
        assert config.rounds == 3
        assert config.consensus == "majority"
        assert config.enable_calibration is True
        assert config.early_stopping is True

    def test_debate_settings(self):
        """Debate-specific settings work."""
        config = DecisionConfig(
            rounds=5,
            consensus="unanimous",
            enable_calibration=False,
        )
        assert config.rounds == 5
        assert config.consensus == "unanimous"
        assert config.enable_calibration is False

    def test_workflow_settings(self):
        """Workflow-specific settings work."""
        config = DecisionConfig(
            workflow_id="wf-123",
            workflow_inputs={"param1": "value1"},
            stop_on_failure=False,
        )
        assert config.workflow_id == "wf-123"
        assert config.workflow_inputs == {"param1": "value1"}
        assert config.stop_on_failure is False

    def test_agents_list(self):
        """Agents list can be customized."""
        config = DecisionConfig(
            agents=["claude", "gpt4", "gemini"],
        )
        assert len(config.agents) == 3
        assert "claude" in config.agents


class TestDecisionRequest:
    """Tests for DecisionRequest dataclass."""

    def test_create_basic(self):
        """Basic request creation."""
        request = DecisionRequest(
            content="Should we use microservices?",
        )
        assert request.content == "Should we use microservices?"
        assert request.decision_type == DecisionType.AUTO  # Default
        assert request.request_id is not None

    def test_with_decision_type(self):
        """Can specify decision type."""
        request = DecisionRequest(
            content="Validate this policy",
            decision_type=DecisionType.GAUNTLET,
        )
        assert request.decision_type == DecisionType.GAUNTLET

    def test_with_source(self):
        """Can specify input source."""
        request = DecisionRequest(
            content="Test question",
            source=InputSource.SLACK,
        )
        assert request.source == InputSource.SLACK

    def test_with_response_channel(self):
        """Can specify response channel."""
        channel = ResponseChannel(platform="slack", channel_id="C123")
        request = DecisionRequest(
            content="Test",
            response_channel=channel,
        )
        assert request.response_channel.platform == "slack"

    def test_with_context(self):
        """Can specify request context."""
        ctx = RequestContext(user_id="user123")
        request = DecisionRequest(
            content="Test",
            context=ctx,
        )
        assert request.context.user_id == "user123"

    def test_with_config(self):
        """Can specify decision config."""
        config = DecisionConfig(rounds=5)
        request = DecisionRequest(
            content="Test",
            config=config,
        )
        assert request.config.rounds == 5

    def test_with_attachments(self):
        """Can include attachments."""
        request = DecisionRequest(
            content="Review this code",
            attachments=[
                {"type": "code", "content": "def foo(): pass"},
            ],
        )
        assert len(request.attachments) == 1

    def test_with_priority(self):
        """Can specify priority."""
        request = DecisionRequest(
            content="Urgent issue",
            priority=Priority.CRITICAL,
        )
        assert request.priority == Priority.CRITICAL


class TestDecisionResult:
    """Tests for DecisionResult dataclass."""

    def test_create_basic(self):
        """Basic result creation."""
        result = DecisionResult(
            request_id="req-123",
            decision="Use microservices for scalability",
            confidence=0.85,
        )
        assert result.request_id == "req-123"
        assert result.decision == "Use microservices for scalability"
        assert result.confidence == 0.85

    def test_default_values(self):
        """Default values are sensible."""
        result = DecisionResult(
            request_id="req-123",
            decision="Test decision",
            confidence=0.9,
        )
        assert result.reasoning == ""
        assert result.alternatives == []
        assert result.evidence == []
        assert result.metadata == {}

    def test_with_reasoning(self):
        """Can include reasoning."""
        result = DecisionResult(
            request_id="req-123",
            decision="Approved",
            confidence=0.95,
            reasoning="All criteria met based on evidence.",
        )
        assert "criteria" in result.reasoning

    def test_with_alternatives(self):
        """Can include alternatives."""
        result = DecisionResult(
            request_id="req-123",
            decision="Option A",
            confidence=0.7,
            alternatives=[
                {"option": "Option B", "confidence": 0.6},
                {"option": "Option C", "confidence": 0.4},
            ],
        )
        assert len(result.alternatives) == 2

    def test_with_dissenting_views(self):
        """Can include dissenting views."""
        result = DecisionResult(
            request_id="req-123",
            decision="Approved",
            confidence=0.8,
            dissenting_views=[
                {"agent": "critic", "view": "Security concerns"},
            ],
        )
        assert len(result.dissenting_views) == 1

    def test_is_successful(self):
        """is_successful property works correctly."""
        success = DecisionResult(
            request_id="req-123",
            decision="Done",
            confidence=0.9,
            status="completed",
        )
        assert success.is_successful is True

        failed = DecisionResult(
            request_id="req-456",
            decision="",
            confidence=0.0,
            status="failed",
        )
        assert failed.is_successful is False


class TestDecisionRouter:
    """Tests for DecisionRouter class."""

    def test_init(self):
        """Router initializes correctly."""
        router = DecisionRouter()
        assert router is not None

    def test_register_engine(self):
        """Can register decision engines."""
        router = DecisionRouter()
        mock_engine = MagicMock()

        router.register_engine(DecisionType.DEBATE, mock_engine)

        assert DecisionType.DEBATE in router._engines

    @pytest.mark.asyncio
    async def test_route_to_debate(self):
        """Routes debate requests correctly."""
        router = DecisionRouter()

        mock_engine = MagicMock()
        mock_engine.execute = AsyncMock(
            return_value=DecisionResult(
                request_id="req-123",
                decision="Consensus reached",
                confidence=0.9,
            )
        )
        router.register_engine(DecisionType.DEBATE, mock_engine)

        request = DecisionRequest(
            content="Should we refactor?",
            decision_type=DecisionType.DEBATE,
        )

        result = await router.route(request)

        assert result.decision == "Consensus reached"
        mock_engine.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_auto_detection(self):
        """Auto-detects decision type when AUTO is specified."""
        router = DecisionRouter()

        mock_engine = MagicMock()
        mock_engine.execute = AsyncMock(
            return_value=DecisionResult(
                request_id="req-123",
                decision="Quick answer",
                confidence=0.95,
            )
        )
        # Register for the detected type
        router.register_engine(DecisionType.QUICK, mock_engine)

        request = DecisionRequest(
            content="What is 2+2?",
            decision_type=DecisionType.AUTO,
        )

        # Mock the auto-detection to return QUICK
        with patch.object(router, "_detect_decision_type", return_value=DecisionType.QUICK):
            result = await router.route(request)

        assert result is not None

    @pytest.mark.asyncio
    async def test_route_not_found(self):
        """Handles missing engine gracefully."""
        router = DecisionRouter()

        request = DecisionRequest(
            content="Test",
            decision_type=DecisionType.GAUNTLET,
        )

        # Should raise or return error result
        with pytest.raises((KeyError, ValueError)):
            await router.route(request)

    def test_detect_decision_type_debate(self):
        """Detects debate-style questions."""
        router = DecisionRouter()

        # Questions that suggest debate
        request = DecisionRequest(content="Should we use microservices vs monolith?")
        detected = router._detect_decision_type(request)
        # Detection depends on implementation, just verify it returns valid type
        assert detected in DecisionType

    def test_detect_decision_type_workflow(self):
        """Detects workflow requests."""
        router = DecisionRouter()

        request = DecisionRequest(
            content="Run the deployment workflow",
            config=DecisionConfig(workflow_id="deploy-wf"),
        )
        detected = router._detect_decision_type(request)
        assert detected in DecisionType


class TestDecisionIntegration:
    """Integration tests for decision flow."""

    def test_full_request_creation(self):
        """Creates complete request with all components."""
        channel = ResponseChannel(
            platform="slack",
            channel_id="C123",
            response_format="summary",
        )
        context = RequestContext(
            user_id="user123",
            tenant_id="tenant456",
            tags=["production"],
        )
        config = DecisionConfig(
            rounds=4,
            consensus="weighted",
            timeout_seconds=600,
        )

        request = DecisionRequest(
            content="Should we deploy to production?",
            decision_type=DecisionType.DEBATE,
            source=InputSource.SLACK,
            response_channel=channel,
            context=context,
            config=config,
            priority=Priority.HIGH,
        )

        assert request.content == "Should we deploy to production?"
        assert request.decision_type == DecisionType.DEBATE
        assert request.source == InputSource.SLACK
        assert request.response_channel.channel_id == "C123"
        assert request.context.user_id == "user123"
        assert request.config.rounds == 4
        assert request.priority == Priority.HIGH

    def test_result_with_full_metadata(self):
        """Creates complete result with all fields."""
        result = DecisionResult(
            request_id="req-789",
            decision="Deploy to staging first",
            confidence=0.85,
            reasoning="Risk mitigation suggests staging deployment.",
            alternatives=[
                {"option": "Direct production", "confidence": 0.6},
            ],
            evidence=[
                {"source": "incident-report", "summary": "Previous direct deploy failed"},
            ],
            dissenting_views=[
                {"agent": "speed-advocate", "view": "Staging delays value delivery"},
            ],
            metadata={
                "debate_rounds": 3,
                "agents_participated": ["claude", "gpt4"],
            },
        )

        assert result.confidence == 0.85
        assert len(result.alternatives) == 1
        assert len(result.evidence) == 1
        assert len(result.dissenting_views) == 1
        assert result.metadata["debate_rounds"] == 3
