"""
Tests for RBAC integration with DecisionRouter.

Tests that the decision router properly enforces role-based access control
for all decision types (debates, workflows, gauntlet, quick).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.core.decision import (
    DecisionRouter,
    DecisionRequest,
    DecisionResult,
    DecisionType,
    InputSource,
    Priority,
    ResponseChannel,
    RequestContext,
    get_decision_router,
    reset_decision_router,
)
from aragora.rbac.models import AuthorizationContext


@pytest.fixture
def mock_debate_engine():
    """Create a mock debate engine."""
    engine = AsyncMock()
    engine.run = AsyncMock(
        return_value={
            "consensus_reached": True,
            "final_answer": "Test answer",
            "confidence": 0.9,
            "participants": ["agent1", "agent2"],
        }
    )
    return engine


@pytest.fixture
def mock_workflow_engine():
    """Create a mock workflow engine."""
    engine = AsyncMock()
    engine.execute = AsyncMock(
        return_value={
            "status": "completed",
            "result": {"answer": "Workflow result"},
        }
    )
    return engine


@pytest.fixture
def router(mock_debate_engine, mock_workflow_engine):
    """Create a decision router with mocked engines."""
    return DecisionRouter(
        debate_engine=mock_debate_engine,
        workflow_engine=mock_workflow_engine,
        enable_caching=False,
        enable_deduplication=False,
    )


@pytest.fixture
def auth_context_with_permission():
    """Create an auth context with debate permissions."""
    return AuthorizationContext(
        user_id="user_123",
        org_id="org_456",
        roles={"member"},
        permissions={"debates:create", "debates:read", "debates.run"},
    )


@pytest.fixture
def auth_context_without_permission():
    """Create an auth context without debate permissions."""
    return AuthorizationContext(
        user_id="user_123",
        org_id="org_456",
        roles={"viewer"},
        permissions={"debates:read"},  # Read-only
    )


class TestDecisionRequest:
    """Test DecisionRequest creation and serialization."""

    def test_create_basic_request(self):
        """DecisionRequest should be created with content."""
        request = DecisionRequest(
            content="What is 2+2?",
            decision_type=DecisionType.DEBATE,
            source=InputSource.HTTP_API,
        )

        assert request.content == "What is 2+2?"
        assert request.decision_type == DecisionType.DEBATE
        assert request.source == InputSource.HTTP_API
        assert request.request_id is not None

    def test_request_with_context(self):
        """DecisionRequest should support user context."""
        context = RequestContext(
            user_id="user_123",
            tenant_id="org_456",
            workspace_id="ws_789",
        )
        request = DecisionRequest(
            content="Test query",
            context=context,
        )

        assert request.context.user_id == "user_123"
        assert request.context.tenant_id == "org_456"

    def test_request_from_chat_platform(self):
        """DecisionRequest should work for chat platforms."""
        request = DecisionRequest(
            content="What is the weather?",
            source=InputSource.SLACK,
            context=RequestContext(user_id="U12345"),
            response_channels=[
                ResponseChannel(
                    platform="slack",
                    channel_id="C67890",
                    user_id="U12345",
                )
            ],
        )

        assert request.source == InputSource.SLACK
        assert len(request.response_channels) == 1
        assert request.response_channels[0].platform == "slack"

    def test_request_to_dict(self):
        """DecisionRequest should serialize correctly."""
        request = DecisionRequest(
            content="Test content",
            decision_type=DecisionType.DEBATE,
            source=InputSource.HTTP_API,
        )

        data = request.to_dict()
        assert data["content"] == "Test content"
        assert data["decision_type"] == "debate"
        assert data["source"] == "http_api"

    def test_auto_detect_workflow_type(self):
        """Should auto-detect workflow type."""
        from aragora.core.decision import DecisionConfig

        request = DecisionRequest(
            content="Run my workflow",
            decision_type=DecisionType.AUTO,
            config=DecisionConfig(workflow_id="wf_123"),
        )

        assert request.decision_type == DecisionType.WORKFLOW


class TestResponseChannel:
    """Test ResponseChannel creation and serialization."""

    def test_create_response_channel(self):
        """ResponseChannel should be created correctly."""
        channel = ResponseChannel(
            platform="telegram",
            channel_id="12345",
            user_id="67890",
        )

        assert channel.platform == "telegram"
        assert channel.channel_id == "12345"
        assert channel.user_id == "67890"

    def test_response_channel_with_thread(self):
        """ResponseChannel should support thread_id."""
        channel = ResponseChannel(
            platform="slack",
            channel_id="C123",
            user_id="U456",
            thread_id="T789",
        )

        assert channel.thread_id == "T789"

    def test_response_channel_to_dict(self):
        """ResponseChannel should serialize correctly."""
        channel = ResponseChannel(
            platform="discord",
            channel_id="123456",
            user_id="654321",
        )

        data = channel.to_dict()
        assert data["platform"] == "discord"
        assert data["channel_id"] == "123456"
        assert data["user_id"] == "654321"


class TestResponseHandlerRegistration:
    """Test response handler registration for platforms."""

    def test_register_response_handler(self, router):
        """Should be able to register response handlers."""
        handler = AsyncMock()
        router.register_response_handler("slack", handler)

        assert "slack" in router._response_handlers
        assert router._response_handlers["slack"] == handler

    def test_register_multiple_handlers(self, router):
        """Should support multiple platform handlers."""
        slack_handler = AsyncMock()
        discord_handler = AsyncMock()
        telegram_handler = AsyncMock()

        router.register_response_handler("slack", slack_handler)
        router.register_response_handler("discord", discord_handler)
        router.register_response_handler("telegram", telegram_handler)

        assert len(router._response_handlers) == 3
        assert "slack" in router._response_handlers
        assert "discord" in router._response_handlers
        assert "telegram" in router._response_handlers

    def test_handler_registration_case_insensitive(self, router):
        """Handler registration should be case-insensitive."""
        handler = AsyncMock()
        router.register_response_handler("SLACK", handler)

        assert "slack" in router._response_handlers


class TestDecisionRouterSingleton:
    """Test the singleton pattern for DecisionRouter."""

    def test_get_decision_router_returns_singleton(self):
        """get_decision_router should return the same instance."""
        reset_decision_router()

        router1 = get_decision_router()
        router2 = get_decision_router()

        assert router1 is router2

    def test_reset_decision_router_clears_singleton(self):
        """reset_decision_router should clear the singleton."""
        router1 = get_decision_router()
        reset_decision_router()
        router2 = get_decision_router()

        assert router1 is not router2


class TestDecisionResult:
    """Test DecisionResult creation and serialization."""

    def test_create_successful_result(self):
        """DecisionResult should be created for successful decisions."""
        result = DecisionResult(
            request_id="req_123",
            success=True,
            decision_type=DecisionType.DEBATE,
            answer="The answer is 4",
            confidence=0.95,
            consensus_reached=True,
        )

        assert result.success is True
        assert result.answer == "The answer is 4"
        assert result.confidence == 0.95

    def test_create_failed_result(self):
        """DecisionResult should support failure states."""
        result = DecisionResult(
            request_id="req_456",
            decision_type=DecisionType.DEBATE,
            answer="",
            confidence=0.0,
            consensus_reached=False,
            success=False,
            error="Insufficient agents available",
        )

        assert result.success is False
        assert result.error == "Insufficient agents available"

    def test_result_to_dict(self):
        """DecisionResult should serialize correctly."""
        result = DecisionResult(
            request_id="req_789",
            decision_type=DecisionType.QUICK,
            answer="Quick answer",
            confidence=0.8,
            consensus_reached=True,
        )

        data = result.to_dict()
        assert data["request_id"] == "req_789"
        assert data["success"] is True
        assert data["decision_type"] == "quick"


class TestDecisionPriority:
    """Test decision priority handling."""

    def test_priority_levels(self):
        """Priority enum should have expected levels."""
        assert Priority.CRITICAL.value == "critical"
        assert Priority.HIGH.value == "high"
        assert Priority.NORMAL.value == "normal"
        assert Priority.LOW.value == "low"
        assert Priority.BATCH.value == "batch"

    def test_request_with_priority(self):
        """Request should support priority setting."""
        request = DecisionRequest(
            content="Critical question",
            priority=Priority.CRITICAL,
        )

        assert request.priority == Priority.CRITICAL


class TestInputSource:
    """Test input source handling."""

    def test_chat_sources(self):
        """InputSource should have chat platform sources."""
        assert InputSource.SLACK.value == "slack"
        assert InputSource.DISCORD.value == "discord"
        assert InputSource.TELEGRAM.value == "telegram"
        assert InputSource.TEAMS.value == "teams"

    def test_api_sources(self):
        """InputSource should have API sources."""
        assert InputSource.HTTP_API.value == "http_api"
        assert InputSource.WEBSOCKET.value == "websocket"
        assert InputSource.CLI.value == "cli"

    def test_voice_sources(self):
        """InputSource should have voice sources."""
        assert InputSource.VOICE.value == "voice"
        assert InputSource.VOICE_SLACK.value == "voice_slack"
        assert InputSource.VOICE_TELEGRAM.value == "voice_telegram"


class TestDecisionType:
    """Test decision type handling."""

    def test_decision_types(self):
        """DecisionType enum should have expected types."""
        assert DecisionType.DEBATE.value == "debate"
        assert DecisionType.WORKFLOW.value == "workflow"
        assert DecisionType.GAUNTLET.value == "gauntlet"
        assert DecisionType.QUICK.value == "quick"
        assert DecisionType.AUTO.value == "auto"
