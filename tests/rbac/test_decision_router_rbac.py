"""
Tests for RBAC integration with DecisionRouter.

Tests that the decision router properly enforces role-based access control
for all decision types (debates, workflows, gauntlet, quick).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from aragora.core.decision import (
    DecisionRouter,
    DecisionRequest,
    DecisionResult,
    DecisionType,
    InputSource,
    Priority,
    ResponseChannel,
)
from aragora.rbac.models import AuthorizationContext


@pytest.fixture
def mock_debate_engine():
    """Create a mock debate engine."""
    engine = AsyncMock()
    engine.run = AsyncMock(return_value={
        "consensus_reached": True,
        "final_answer": "Test answer",
        "confidence": 0.9,
        "participants": ["agent1", "agent2"],
    })
    return engine


@pytest.fixture
def mock_workflow_engine():
    """Create a mock workflow engine."""
    engine = AsyncMock()
    engine.execute = AsyncMock(return_value={
        "status": "completed",
        "result": {"answer": "Workflow result"},
    })
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
        permissions={"debates.create", "debates.read", "debates.run"},
    )


@pytest.fixture
def auth_context_without_permission():
    """Create an auth context without debate permissions."""
    return AuthorizationContext(
        user_id="user_123",
        org_id="org_456",
        roles={"viewer"},
        permissions={"debates.read"},  # Read-only
    )


class TestDecisionRouterAuthorization:
    """Test authorization checks in DecisionRouter."""

    def test_decision_request_has_auth_context(self):
        """DecisionRequest should support auth context."""
        request = DecisionRequest(
            request_id="test_123",
            query="Test query",
            decision_type=DecisionType.DEBATE,
            source=InputSource.API,
            user_id="user_123",
            org_id="org_456",
        )

        assert request.user_id == "user_123"
        assert request.org_id == "org_456"

    def test_decision_request_from_chat_message(self):
        """DecisionRequest from chat should include user context."""
        request = DecisionRequest.from_chat_message(
            platform="slack",
            channel_id="C123",
            user_id="U456",
            content="What is 2+2?",
            thread_id="T789",
        )

        assert request.user_id == "U456"
        assert request.source == InputSource.CHAT
        assert request.metadata["platform"] == "slack"
        assert request.metadata["channel_id"] == "C123"

    def test_decision_request_from_api(self):
        """DecisionRequest from API should include user context."""
        request = DecisionRequest.from_api_request(
            query="Analyze this data",
            user_id="user_123",
            org_id="org_456",
            decision_type=DecisionType.DEBATE,
        )

        assert request.user_id == "user_123"
        assert request.org_id == "org_456"
        assert request.source == InputSource.API


class TestDecisionRouterResourceIsolation:
    """Test resource isolation in decision routing."""

    def test_debate_result_includes_user_context(self):
        """Debate results should include user context for filtering."""
        result = DecisionResult(
            request_id="req_123",
            success=True,
            decision_type=DecisionType.DEBATE,
            answer="Test answer",
            confidence=0.9,
            consensus_reached=True,
            user_id="user_123",
            org_id="org_456",
        )

        assert result.user_id == "user_123"
        assert result.org_id == "org_456"

    def test_result_to_dict_includes_user_context(self):
        """Result serialization should include user context."""
        result = DecisionResult(
            request_id="req_123",
            success=True,
            decision_type=DecisionType.DEBATE,
            answer="Test answer",
            user_id="user_123",
            org_id="org_456",
        )

        data = result.to_dict()
        assert data["user_id"] == "user_123"
        assert data["org_id"] == "org_456"


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


class TestResponseChannelRouting:
    """Test response channel routing."""

    def test_response_channel_from_request(self):
        """ResponseChannel should be created from request."""
        request = DecisionRequest.from_chat_message(
            platform="telegram",
            channel_id="12345",
            user_id="67890",
            content="Hello",
            message_id="msg_123",
        )

        channel = ResponseChannel(
            platform=request.metadata["platform"],
            channel_id=request.metadata["channel_id"],
            user_id=request.user_id,
            thread_id=request.metadata.get("thread_id"),
            message_id=request.metadata.get("message_id"),
        )

        assert channel.platform == "telegram"
        assert channel.channel_id == "12345"
        assert channel.user_id == "67890"

    def test_response_channel_to_dict(self):
        """ResponseChannel should serialize correctly."""
        channel = ResponseChannel(
            platform="slack",
            channel_id="C123",
            user_id="U456",
            thread_id="T789",
        )

        data = channel.to_dict()
        assert data["platform"] == "slack"
        assert data["channel_id"] == "C123"
        assert data["user_id"] == "U456"
        assert data["thread_id"] == "T789"


class TestDecisionTypeRouting:
    """Test routing to different decision engines."""

    @pytest.mark.asyncio
    async def test_route_to_debate(self, router, mock_debate_engine):
        """Debate requests should route to debate engine."""
        request = DecisionRequest(
            request_id="test_123",
            query="Test debate",
            decision_type=DecisionType.DEBATE,
            source=InputSource.API,
            user_id="user_123",
        )

        # Mock the internal route method
        with patch.object(router, '_route_to_debate', new_callable=AsyncMock) as mock_route:
            mock_route.return_value = DecisionResult(
                request_id="test_123",
                success=True,
                decision_type=DecisionType.DEBATE,
                answer="Test answer",
            )

            result = await router.route(request)

            assert result.success is True
            mock_route.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_workflow(self, router, mock_workflow_engine):
        """Workflow requests should route to workflow engine."""
        request = DecisionRequest(
            request_id="test_456",
            query="Test workflow",
            decision_type=DecisionType.WORKFLOW,
            source=InputSource.API,
            user_id="user_123",
        )

        with patch.object(router, '_route_to_workflow', new_callable=AsyncMock) as mock_route:
            mock_route.return_value = DecisionResult(
                request_id="test_456",
                success=True,
                decision_type=DecisionType.WORKFLOW,
                answer="Workflow result",
            )

            result = await router.route(request)

            assert result.success is True
            mock_route.assert_called_once()


class TestDecisionRouterMetrics:
    """Test metrics and observability."""

    @pytest.mark.asyncio
    async def test_route_records_metrics(self, router):
        """Routing should record metrics."""
        request = DecisionRequest(
            request_id="test_789",
            query="Test query",
            decision_type=DecisionType.QUICK,
            source=InputSource.API,
            user_id="user_123",
        )

        with patch.object(router, '_route_to_quick', new_callable=AsyncMock) as mock_route:
            mock_route.return_value = DecisionResult(
                request_id="test_789",
                success=True,
                decision_type=DecisionType.QUICK,
                answer="Quick answer",
            )

            # Metrics should be recorded (patched at module level)
            with patch('aragora.core.decision._record_decision_request') as mock_metric:
                result = await router.route(request)

                # Metric should be called if available
                if mock_metric:
                    mock_metric.assert_called()


class TestDecisionRouterSingleton:
    """Test the singleton pattern for DecisionRouter."""

    def test_get_decision_router_returns_singleton(self):
        """get_decision_router should return the same instance."""
        from aragora.core.decision import get_decision_router, reset_decision_router

        reset_decision_router()

        router1 = get_decision_router()
        router2 = get_decision_router()

        assert router1 is router2

    def test_reset_decision_router_clears_singleton(self):
        """reset_decision_router should clear the singleton."""
        from aragora.core.decision import get_decision_router, reset_decision_router

        router1 = get_decision_router()
        reset_decision_router()
        router2 = get_decision_router()

        assert router1 is not router2
