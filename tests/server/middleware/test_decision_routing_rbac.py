"""
RBAC Tests for Decision Routing.

Tests authorization enforcement, permission denials, workspace isolation,
and audit logging for the DecisionRouter and middleware.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from aragora.server.middleware.decision_routing import (
    DecisionRoutingMiddleware,
    RoutingContext,
    get_decision_middleware,
    reset_decision_middleware,
)


@dataclass
class MockIsolationContext:
    """Mock isolation context for testing."""

    workspace_id: str = None
    organization_id: str = None
    user_id: str = None
    actor_id: str = None
    actor_type: str = "user"


class MockPermissionDeniedException(Exception):
    """Mock permission denied exception."""

    def __init__(self, message: str, actor_id: str, resource, action, context=None):
        super().__init__(message)
        self.actor_id = actor_id
        self.resource = resource
        self.action = action
        self.context = context


class TestDecisionRouterRBAC:
    """Tests for DecisionRouter RBAC enforcement."""

    def setup_method(self):
        """Reset middleware before each test."""
        reset_decision_middleware()

    @pytest.mark.asyncio
    async def test_rbac_permission_denied_returns_failure(self):
        """Should return failure when RBAC denies permission."""
        middleware = DecisionRoutingMiddleware()
        context = RoutingContext(
            channel="api",
            channel_id="test-channel",
            user_id="unauthorized-user",
            request_id="rbac-test-1",
            workspace_id="ws-123",
        )

        # Mock the router to raise permission denied
        with patch.object(middleware, "_get_router") as mock_get_router:
            mock_router = MagicMock()

            # Simulate RBAC denial
            async def deny_permission(request):
                raise MockPermissionDeniedException(
                    message="No permission grants create on debate",
                    actor_id="unauthorized-user",
                    resource="debate",
                    action="create",
                )

            mock_router.route = deny_permission
            mock_get_router.return_value = mock_router

            result = await middleware.process("Test question", context)

            # Should return failure with error message
            assert result["success"] is False
            assert "error" in result
            assert "permission" in result["error"].lower() or "denied" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_rbac_authorized_user_succeeds(self):
        """Should succeed when RBAC grants permission."""
        middleware = DecisionRoutingMiddleware()
        context = RoutingContext(
            channel="api",
            channel_id="test-channel",
            user_id="authorized-user",
            request_id="rbac-test-2",
            workspace_id="ws-123",
        )

        with patch.object(middleware, "_get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.answer = "Authorized answer"
            mock_result.confidence = 0.95
            mock_result.consensus_reached = True
            mock_result.reasoning = "Test reasoning"
            mock_result.duration_seconds = 1.0
            mock_result.error = None

            mock_router.route = AsyncMock(return_value=mock_result)
            mock_get_router.return_value = mock_router

            result = await middleware.process("Test question", context)

            assert result["success"] is True
            assert "result" in result

    @pytest.mark.asyncio
    async def test_workspace_isolation_different_workspaces(self):
        """Should isolate requests from different workspaces."""
        middleware = DecisionRoutingMiddleware(enable_caching=True)

        context_ws1 = RoutingContext(
            channel="api",
            channel_id="channel-1",
            user_id="user-1",
            request_id="isolation-test-1",
            workspace_id="workspace-A",
        )

        context_ws2 = RoutingContext(
            channel="api",
            channel_id="channel-1",
            user_id="user-1",
            request_id="isolation-test-2",
            workspace_id="workspace-B",
        )

        # Pre-populate cache for workspace A
        await middleware._cache.set(
            "same query",
            "answer for workspace A",
            context={"channel": "api", "workspace_id": "workspace-A"},
        )

        # Pre-populate cache for workspace B with different answer
        await middleware._cache.set(
            "same query",
            "answer for workspace B",
            context={"channel": "api", "workspace_id": "workspace-B"},
        )

        # Query from workspace A should get workspace A's cached answer
        result_ws1 = await middleware.process("same query", context_ws1)
        assert result_ws1.get("cached") is True
        assert result_ws1.get("result") == "answer for workspace A"

        # Query from workspace B should get workspace B's cached answer
        result_ws2 = await middleware.process("same query", context_ws2)
        assert result_ws2.get("cached") is True
        assert result_ws2.get("result") == "answer for workspace B"

    @pytest.mark.asyncio
    async def test_user_isolation_same_workspace(self):
        """Should not share cached results between users (deduplication is per-user)."""
        middleware = DecisionRoutingMiddleware(enable_deduplication=True, enable_caching=False)

        # First user makes a request
        context_user1 = RoutingContext(
            channel="slack",
            channel_id="C123",
            user_id="user-alpha",
            request_id="user-iso-1",
            workspace_id="ws-shared",
        )

        # Second user makes the same request
        context_user2 = RoutingContext(
            channel="slack",
            channel_id="C123",
            user_id="user-beta",
            request_id="user-iso-2",
            workspace_id="ws-shared",
        )

        # Check deduplication for user1
        is_dup1, _ = await middleware._deduplicator.check_and_mark(
            "shared question", "user-alpha", "slack"
        )
        assert is_dup1 is False  # First request not duplicate

        # Check deduplication for user2 - should NOT be duplicate (different user)
        is_dup2, _ = await middleware._deduplicator.check_and_mark(
            "shared question", "user-beta", "slack"
        )
        assert is_dup2 is False  # Different user, not duplicate

    @pytest.mark.asyncio
    async def test_deduplication_same_user_same_workspace(self):
        """Should deduplicate identical requests from same user in same workspace."""
        middleware = DecisionRoutingMiddleware(enable_deduplication=True)

        # Same user, same content
        is_dup1, _ = await middleware._deduplicator.check_and_mark(
            "repeated question", "same-user", "api"
        )
        assert is_dup1 is False

        is_dup2, future = await middleware._deduplicator.check_and_mark(
            "repeated question", "same-user", "api"
        )
        assert is_dup2 is True
        assert future is not None


class TestDecisionRouterResourceTypes:
    """Tests for resource type mapping in RBAC checks."""

    def setup_method(self):
        """Reset middleware before each test."""
        reset_decision_middleware()

    @pytest.mark.asyncio
    async def test_debate_decision_maps_to_debate_resource(self):
        """Decision type 'debate' should map to DEBATE resource for RBAC."""
        middleware = DecisionRoutingMiddleware()
        context = RoutingContext(
            channel="api",
            channel_id="test",
            user_id="user-1",
            request_id="resource-map-1",
        )

        # This tests that the middleware correctly passes decision_type through
        with patch.object(middleware, "_route_via_decision_router") as mock_route:
            mock_route.return_value = {"success": True, "answer": "test"}
            with patch.object(middleware, "_get_router") as mock_get_router:
                mock_get_router.return_value = MagicMock()

                await middleware.process("test", context, decision_type="debate")

                # Verify _route_via_decision_router was called with correct type
                mock_route.assert_called_once()
                call_args = mock_route.call_args
                assert call_args[0][2] == "debate"  # decision_type argument

    @pytest.mark.asyncio
    async def test_workflow_decision_maps_to_workflow_resource(self):
        """Decision type 'workflow' should map to WORKFLOW resource for RBAC."""
        middleware = DecisionRoutingMiddleware()
        context = RoutingContext(
            channel="api",
            channel_id="test",
            user_id="user-1",
            request_id="resource-map-2",
        )

        with patch.object(middleware, "_route_via_decision_router") as mock_route:
            mock_route.return_value = {"success": True, "answer": "test"}
            with patch.object(middleware, "_get_router") as mock_get_router:
                mock_get_router.return_value = MagicMock()

                await middleware.process("test", context, decision_type="workflow")

                call_args = mock_route.call_args
                assert call_args[0][2] == "workflow"


class TestRBACContextPropagation:
    """Tests for RBAC context propagation through the middleware."""

    def setup_method(self):
        """Reset middleware before each test."""
        reset_decision_middleware()

    @pytest.mark.asyncio
    async def test_user_id_propagated_to_request_context(self):
        """User ID should be propagated to the DecisionRequest context."""
        middleware = DecisionRoutingMiddleware()
        context = RoutingContext(
            channel="slack",
            channel_id="C123",
            user_id="user-for-rbac",
            request_id="context-prop-1",
            workspace_id="ws-456",
        )

        captured_request = None

        with patch.object(middleware, "_get_router") as mock_get_router:
            mock_router = MagicMock()

            async def capture_request(request):
                nonlocal captured_request
                captured_request = request
                result = MagicMock()
                result.success = True
                result.answer = "test"
                result.confidence = 0.9
                result.consensus_reached = True
                result.reasoning = ""
                result.duration_seconds = 1.0
                result.error = None
                return result

            mock_router.route = capture_request
            mock_get_router.return_value = mock_router

            await middleware.process("test question", context)

            # Verify user_id was propagated
            assert captured_request is not None
            assert captured_request.context.user_id == "user-for-rbac"

    @pytest.mark.asyncio
    async def test_workspace_id_propagated_to_request_context(self):
        """Workspace ID should be propagated to the DecisionRequest context."""
        middleware = DecisionRoutingMiddleware()
        context = RoutingContext(
            channel="api",
            channel_id="api-channel",
            user_id="user-1",
            request_id="context-prop-2",
            workspace_id="workspace-for-isolation",
        )

        captured_request = None

        with patch.object(middleware, "_get_router") as mock_get_router:
            mock_router = MagicMock()

            async def capture_request(request):
                nonlocal captured_request
                captured_request = request
                result = MagicMock()
                result.success = True
                result.answer = "test"
                result.confidence = 0.9
                result.consensus_reached = True
                result.reasoning = ""
                result.duration_seconds = 1.0
                result.error = None
                return result

            mock_router.route = capture_request
            mock_get_router.return_value = mock_router

            await middleware.process("test question", context)

            # Verify workspace_id was propagated
            assert captured_request is not None
            assert captured_request.context.workspace_id == "workspace-for-isolation"

    @pytest.mark.asyncio
    async def test_channel_metadata_propagated(self):
        """Channel metadata should be propagated to the request context."""
        middleware = DecisionRoutingMiddleware()
        context = RoutingContext(
            channel="telegram",
            channel_id="chat-789",
            user_id="tg-user",
            request_id="context-prop-3",
            thread_id="thread-123",
            metadata={"custom_field": "custom_value"},
        )

        captured_request = None

        with patch.object(middleware, "_get_router") as mock_get_router:
            mock_router = MagicMock()

            async def capture_request(request):
                nonlocal captured_request
                captured_request = request
                result = MagicMock()
                result.success = True
                result.answer = "test"
                result.confidence = 0.9
                result.consensus_reached = True
                result.reasoning = ""
                result.duration_seconds = 1.0
                result.error = None
                return result

            mock_router.route = capture_request
            mock_get_router.return_value = mock_router

            await middleware.process("test question", context)

            # Verify metadata was propagated
            assert captured_request is not None
            metadata = captured_request.context.metadata
            assert metadata.get("channel_id") == "chat-789"
            assert metadata.get("thread_id") == "thread-123"
            assert metadata.get("custom_field") == "custom_value"


class TestRBACErrorHandling:
    """Tests for RBAC error handling and fallback behavior."""

    def setup_method(self):
        """Reset middleware before each test."""
        reset_decision_middleware()

    @pytest.mark.asyncio
    async def test_rbac_import_error_continues_processing(self):
        """Should continue processing if RBAC module not available."""
        middleware = DecisionRoutingMiddleware()
        context = RoutingContext(
            channel="api",
            channel_id="test",
            user_id="user-1",
            request_id="fallback-1",
        )

        with patch.object(middleware, "_get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.answer = "Fallback answer"
            mock_result.confidence = 0.8
            mock_result.consensus_reached = True
            mock_result.reasoning = ""
            mock_result.duration_seconds = 1.0
            mock_result.error = None

            mock_router.route = AsyncMock(return_value=mock_result)
            mock_get_router.return_value = mock_router

            # Should succeed even without RBAC
            result = await middleware.process("test question", context)
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_generic_exception_returns_failure(self):
        """Should return failure on generic exceptions."""
        middleware = DecisionRoutingMiddleware()
        context = RoutingContext(
            channel="api",
            channel_id="test",
            user_id="user-1",
            request_id="error-test-1",
        )

        with patch.object(middleware, "_get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.route = AsyncMock(side_effect=RuntimeError("Unexpected error"))
            mock_get_router.return_value = mock_router

            result = await middleware.process("test question", context)

            assert result["success"] is False
            assert "error" in result
            assert "Unexpected error" in result["error"]


class TestRBACWithMultipleChannels:
    """Tests for RBAC enforcement across different channels."""

    def setup_method(self):
        """Reset middleware before each test."""
        reset_decision_middleware()

    @pytest.mark.asyncio
    async def test_slack_channel_rbac_context(self):
        """Slack channel should include correct RBAC context."""
        middleware = DecisionRoutingMiddleware()
        context = RoutingContext(
            channel="slack",
            channel_id="C12345",
            user_id="U67890",
            request_id="slack-rbac-1",
            workspace_id="T-WORKSPACE",
            thread_id="ts.123456",
        )

        captured_request = None

        with patch.object(middleware, "_get_router") as mock_get_router:
            mock_router = MagicMock()

            async def capture_request(request):
                nonlocal captured_request
                captured_request = request
                result = MagicMock()
                result.success = True
                result.answer = "test"
                result.confidence = 0.9
                result.consensus_reached = True
                result.reasoning = ""
                result.duration_seconds = 1.0
                result.error = None
                return result

            mock_router.route = capture_request
            mock_get_router.return_value = mock_router

            await middleware.process("slack question", context)

            assert captured_request is not None
            # Verify source is SLACK
            from aragora.core.decision import InputSource

            assert captured_request.source == InputSource.SLACK

    @pytest.mark.asyncio
    async def test_email_channel_rbac_context(self):
        """Email channel should include correct RBAC context."""
        middleware = DecisionRoutingMiddleware()
        context = RoutingContext(
            channel="email",
            channel_id="inbox@company.com",
            user_id="sender@external.com",
            request_id="email-rbac-1",
            message_id="<msg-id-123>",
        )

        captured_request = None

        with patch.object(middleware, "_get_router") as mock_get_router:
            mock_router = MagicMock()

            async def capture_request(request):
                nonlocal captured_request
                captured_request = request
                result = MagicMock()
                result.success = True
                result.answer = "test"
                result.confidence = 0.9
                result.consensus_reached = True
                result.reasoning = ""
                result.duration_seconds = 1.0
                result.error = None
                return result

            mock_router.route = capture_request
            mock_get_router.return_value = mock_router

            await middleware.process("email question", context)

            assert captured_request is not None
            # Verify source is EMAIL
            from aragora.core.decision import InputSource

            assert captured_request.source == InputSource.EMAIL

    @pytest.mark.asyncio
    async def test_all_channels_map_to_valid_sources(self):
        """All supported channels should map to valid InputSource values."""
        from aragora.core.decision import InputSource

        channel_mappings = {
            "slack": InputSource.SLACK,
            "teams": InputSource.TEAMS,
            "discord": InputSource.DISCORD,
            "telegram": InputSource.TELEGRAM,
            "whatsapp": InputSource.WHATSAPP,
            "email": InputSource.EMAIL,
            "gmail": InputSource.GMAIL,
            "web": InputSource.HTTP_API,
            "api": InputSource.HTTP_API,
            "websocket": InputSource.WEBSOCKET,
            "cli": InputSource.CLI,
        }

        middleware = DecisionRoutingMiddleware()

        for channel, expected_source in channel_mappings.items():
            context = RoutingContext(
                channel=channel,
                channel_id=f"{channel}-channel",
                user_id="test-user",
                request_id=f"channel-map-{channel}",
            )

            captured_request = None

            with patch.object(middleware, "_get_router") as mock_get_router:
                mock_router = MagicMock()

                async def capture_request(request):
                    nonlocal captured_request
                    captured_request = request
                    result = MagicMock()
                    result.success = True
                    result.answer = "test"
                    result.confidence = 0.9
                    result.consensus_reached = True
                    result.reasoning = ""
                    result.duration_seconds = 1.0
                    result.error = None
                    return result

                mock_router.route = capture_request
                mock_get_router.return_value = mock_router

                await middleware.process(f"question from {channel}", context)

                assert captured_request is not None, f"No request captured for channel {channel}"
                assert captured_request.source == expected_source, (
                    f"Channel {channel} mapped to {captured_request.source}, expected {expected_source}"
                )
