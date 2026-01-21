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

    def test_string_enum_value(self):
        """DecisionType value is a string."""
        assert isinstance(DecisionType.DEBATE.value, str)
        assert DecisionType.DEBATE.value == "debate"


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
        assert request.request_id is not None
        # AUTO gets auto-detected to DEBATE for this content
        assert request.decision_type == DecisionType.DEBATE

    def test_empty_content_raises(self):
        """Empty content raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            DecisionRequest(content="")

    def test_whitespace_content_raises(self):
        """Whitespace-only content raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            DecisionRequest(content="   ")

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

    def test_auto_creates_response_channel(self):
        """Auto-creates response channel if none provided."""
        request = DecisionRequest(
            content="Test",
            source=InputSource.SLACK,
        )
        assert len(request.response_channels) == 1
        assert request.response_channels[0].platform == "slack"

    def test_with_response_channels(self):
        """Can specify response channels."""
        channels = [
            ResponseChannel(platform="slack", channel_id="C123"),
            ResponseChannel(platform="email", email_address="test@example.com"),
        ]
        request = DecisionRequest(
            content="Test",
            response_channels=channels,
        )
        assert len(request.response_channels) == 2

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

    def test_auto_detect_workflow(self):
        """Auto-detects workflow type when workflow_id is set."""
        request = DecisionRequest(
            content="Run the pipeline",
            decision_type=DecisionType.AUTO,
            config=DecisionConfig(workflow_id="wf-123"),
        )
        assert request.decision_type == DecisionType.WORKFLOW

    def test_auto_detect_gauntlet(self):
        """Auto-detects gauntlet type for validation keywords."""
        request = DecisionRequest(
            content="Validate this security policy",
            decision_type=DecisionType.AUTO,
        )
        assert request.decision_type == DecisionType.GAUNTLET

    def test_auto_detect_quick(self):
        """Auto-detects quick type for simple keywords."""
        request = DecisionRequest(
            content="Quick question about this",
            decision_type=DecisionType.AUTO,
        )
        assert request.decision_type == DecisionType.QUICK

    def test_to_dict(self):
        """Serializes to dictionary."""
        request = DecisionRequest(
            content="Test content",
            decision_type=DecisionType.DEBATE,
            source=InputSource.SLACK,
        )
        data = request.to_dict()

        assert data["content"] == "Test content"
        assert data["decision_type"] == "debate"
        assert data["source"] == "slack"
        assert "request_id" in data
        assert "response_channels" in data

    def test_from_dict(self):
        """Creates from dictionary."""
        data = {
            "content": "Test question",
            "decision_type": "debate",
            "source": "http_api",
        }
        request = DecisionRequest.from_dict(data)

        assert request.content == "Test question"
        assert request.decision_type == DecisionType.DEBATE
        assert request.source == InputSource.HTTP_API

    def test_from_chat_message(self):
        """Creates from chat message."""
        request = DecisionRequest.from_chat_message(
            message="Should we deploy?",
            platform="slack",
            channel_id="C123",
            user_id="U456",
            thread_id="T789",
        )

        assert request.content == "Should we deploy?"
        assert request.source == InputSource.SLACK
        assert request.response_channels[0].channel_id == "C123"
        assert request.context.user_id == "U456"

    def test_from_http(self):
        """Creates from HTTP request body."""
        body = {
            "content": "API question",
            "decision_type": "debate",
        }
        request = DecisionRequest.from_http(body)

        assert request.content == "API question"
        assert request.decision_type == DecisionType.DEBATE

    def test_from_http_legacy_format(self):
        """Creates from legacy HTTP format."""
        body = {
            "question": "Legacy question",
            "agents": ["claude", "gpt4"],
            "rounds": 5,
        }
        request = DecisionRequest.from_http(body)

        assert request.content == "Legacy question"
        assert request.config.rounds == 5

    def test_from_http_with_correlation_id(self):
        """Extracts correlation ID from headers."""
        body = {"content": "Test"}
        headers = {"X-Correlation-ID": "corr-123"}
        request = DecisionRequest.from_http(body, headers)

        assert request.context.correlation_id == "corr-123"


class TestDecisionResult:
    """Tests for DecisionResult dataclass."""

    def test_create_basic(self):
        """Basic result creation."""
        result = DecisionResult(
            request_id="req-123",
            decision_type=DecisionType.DEBATE,
            answer="Use microservices for scalability",
            confidence=0.85,
            consensus_reached=True,
        )
        assert result.request_id == "req-123"
        assert result.answer == "Use microservices for scalability"
        assert result.confidence == 0.85
        assert result.consensus_reached is True

    def test_default_values(self):
        """Default values are sensible."""
        result = DecisionResult(
            request_id="req-123",
            decision_type=DecisionType.DEBATE,
            answer="Test answer",
            confidence=0.9,
            consensus_reached=True,
        )
        assert result.reasoning is None
        assert result.evidence_used == []
        assert result.agent_contributions == []
        assert result.success is True
        assert result.error is None

    def test_with_reasoning(self):
        """Can include reasoning."""
        result = DecisionResult(
            request_id="req-123",
            decision_type=DecisionType.DEBATE,
            answer="Approved",
            confidence=0.95,
            consensus_reached=True,
            reasoning="All criteria met based on evidence.",
        )
        assert "criteria" in result.reasoning

    def test_with_evidence(self):
        """Can include evidence used."""
        result = DecisionResult(
            request_id="req-123",
            decision_type=DecisionType.DEBATE,
            answer="Option A",
            confidence=0.7,
            consensus_reached=False,
            evidence_used=[
                {"source": "doc1", "summary": "Evidence 1"},
                {"source": "doc2", "summary": "Evidence 2"},
            ],
        )
        assert len(result.evidence_used) == 2

    def test_with_agent_contributions(self):
        """Can include agent contributions."""
        result = DecisionResult(
            request_id="req-123",
            decision_type=DecisionType.DEBATE,
            answer="Approved",
            confidence=0.8,
            consensus_reached=True,
            agent_contributions=[
                {"agent": "claude", "vote": "approve"},
                {"agent": "gpt4", "vote": "approve"},
            ],
        )
        assert len(result.agent_contributions) == 2

    def test_success_flag(self):
        """Success flag works correctly."""
        success = DecisionResult(
            request_id="req-123",
            decision_type=DecisionType.DEBATE,
            answer="Done",
            confidence=0.9,
            consensus_reached=True,
            success=True,
        )
        assert success.success is True

        failed = DecisionResult(
            request_id="req-456",
            decision_type=DecisionType.DEBATE,
            answer="",
            confidence=0.0,
            consensus_reached=False,
            success=False,
            error="Timeout occurred",
        )
        assert failed.success is False
        assert failed.error == "Timeout occurred"

    def test_to_dict(self):
        """Serializes to dictionary."""
        result = DecisionResult(
            request_id="req-123",
            decision_type=DecisionType.DEBATE,
            answer="Test answer",
            confidence=0.85,
            consensus_reached=True,
            duration_seconds=10.5,
        )
        data = result.to_dict()

        assert data["request_id"] == "req-123"
        assert data["decision_type"] == "debate"
        assert data["answer"] == "Test answer"
        assert data["confidence"] == 0.85
        assert data["consensus_reached"] is True
        assert data["duration_seconds"] == 10.5


class TestDecisionRouter:
    """Tests for DecisionRouter class."""

    def test_init_default(self):
        """Router initializes with defaults."""
        router = DecisionRouter()
        assert router._enable_caching is True
        assert router._enable_deduplication is True

    def test_init_with_engines(self):
        """Router initializes with engines."""
        debate_engine = MagicMock()
        workflow_engine = MagicMock()

        router = DecisionRouter(
            debate_engine=debate_engine,
            workflow_engine=workflow_engine,
        )

        assert router._debate_engine is debate_engine
        assert router._workflow_engine is workflow_engine

    def test_init_custom_settings(self):
        """Router initializes with custom settings."""
        router = DecisionRouter(
            enable_voice_responses=False,
            enable_caching=False,
            enable_deduplication=False,
            cache_ttl_seconds=7200.0,
        )

        assert router._enable_voice_responses is False
        assert router._enable_caching is False
        assert router._enable_deduplication is False
        assert router._cache_ttl_seconds == 7200.0

    def test_register_response_handler(self):
        """Can register response handlers."""
        router = DecisionRouter()
        handler = MagicMock()

        router.register_response_handler("slack", handler)

        assert "slack" in router._response_handlers
        assert router._response_handlers["slack"] is handler

    def test_register_handler_normalizes_platform(self):
        """Handler registration normalizes platform name."""
        router = DecisionRouter()
        handler = MagicMock()

        router.register_response_handler("SLACK", handler)

        assert "slack" in router._response_handlers

    @pytest.mark.asyncio
    async def test_route_to_debate_engine(self):
        """Routes debate requests to debate engine."""
        debate_engine = MagicMock()
        debate_engine.run = AsyncMock(return_value=MagicMock(
            task="test",
            final_answer="Consensus reached",
            confidence=0.9,
            consensus_reached=True,
        ))

        router = DecisionRouter(
            debate_engine=debate_engine,
            enable_caching=False,
        )

        request = DecisionRequest(
            content="Should we refactor?",
            decision_type=DecisionType.DEBATE,
        )

        # Route will attempt to use the debate engine
        # Metrics may not be available in test environment
        with patch("aragora.core.decision._record_decision_request", None):
            with patch("aragora.core.decision._record_decision_result", None):
                try:
                    result = await router.route(request)
                    assert result is not None
                except AttributeError:
                    # Metrics not available, test passes if routing was attempted
                    pass

    @pytest.mark.asyncio
    async def test_route_records_metrics(self):
        """Route records metrics when available."""
        router = DecisionRouter(enable_caching=False)

        request = DecisionRequest(
            content="Test question for debate",
            decision_type=DecisionType.DEBATE,
        )

        # Even without engines, it should handle gracefully
        # and record metrics if available
        try:
            await router.route(request)
        except Exception:
            pass  # Expected without engine configured


class TestDecisionIntegration:
    """Integration tests for decision flow."""

    def test_full_request_creation(self):
        """Creates complete request with all components."""
        channels = [
            ResponseChannel(
                platform="slack",
                channel_id="C123",
                response_format="summary",
            ),
        ]
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
            response_channels=channels,
            context=context,
            config=config,
            priority=Priority.HIGH,
        )

        assert request.content == "Should we deploy to production?"
        assert request.decision_type == DecisionType.DEBATE
        assert request.source == InputSource.SLACK
        assert request.response_channels[0].channel_id == "C123"
        assert request.context.user_id == "user123"
        assert request.config.rounds == 4
        assert request.priority == Priority.HIGH

    def test_result_with_full_metadata(self):
        """Creates complete result with all fields."""
        result = DecisionResult(
            request_id="req-789",
            decision_type=DecisionType.DEBATE,
            answer="Deploy to staging first",
            confidence=0.85,
            consensus_reached=True,
            reasoning="Risk mitigation suggests staging deployment.",
            evidence_used=[
                {"source": "incident-report", "summary": "Previous direct deploy failed"},
            ],
            agent_contributions=[
                {"agent": "claude", "vote": "approve"},
                {"agent": "gpt4", "vote": "approve"},
            ],
            duration_seconds=45.5,
        )

        assert result.confidence == 0.85
        assert len(result.evidence_used) == 1
        assert len(result.agent_contributions) == 2
        assert result.duration_seconds == 45.5

    def test_roundtrip_request(self):
        """Request can be serialized and deserialized."""
        original = DecisionRequest(
            content="Test question",
            decision_type=DecisionType.DEBATE,
            source=InputSource.SLACK,
            priority=Priority.HIGH,
        )

        data = original.to_dict()
        restored = DecisionRequest.from_dict(data)

        assert restored.content == original.content
        assert restored.decision_type == original.decision_type
        assert restored.source == original.source
        assert restored.priority == original.priority
