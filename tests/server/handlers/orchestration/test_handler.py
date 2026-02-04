"""
Tests for orchestration handler.

Tests cover:
- Authentication and RBAC permission enforcement
- GET endpoint handling (templates, status)
- POST endpoint handling (deliberate, deliberate/sync)
- Knowledge source and output channel validation
- Error handling
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.orchestration import (
    OrchestrationHandler,
    OrchestrationRequest,
    OrchestrationResult,
    KnowledgeContextSource,
    OutputChannel,
    TeamStrategy,
    OutputFormat,
    TEMPLATES,
    _orchestration_requests,
    _orchestration_results,
)
from aragora.server.handlers.orchestration.validation import (
    PERM_ORCH_DELIBERATE,
    PERM_KNOWLEDGE_SLACK,
    PERM_CHANNEL_SLACK,
)
from aragora.server.handlers.secure import ForbiddenError, UnauthorizedError


class MockAuthContext:
    """Mock authorization context."""

    def __init__(self, user_id: str = "test_user", permissions: set[str] | None = None):
        self.user_id = user_id
        self.permissions = permissions or set()


@pytest.fixture
def handler():
    """Create an OrchestrationHandler instance."""
    return OrchestrationHandler({})


@pytest.fixture
def mock_auth_context():
    """Create a mock auth context with full permissions."""
    return MockAuthContext(
        user_id="test_user",
        permissions={
            "orchestration:read",
            "orchestration:execute",
            PERM_ORCH_DELIBERATE,
            PERM_KNOWLEDGE_SLACK,
            PERM_CHANNEL_SLACK,
            "orchestration:knowledge:read",
            "orchestration:channels:write",
        },
    )


@pytest.fixture(autouse=True)
def clear_state():
    """Clear in-memory state before each test."""
    _orchestration_requests.clear()
    _orchestration_results.clear()
    yield
    _orchestration_requests.clear()
    _orchestration_results.clear()


class TestOrchestrationHandlerRouting:
    """Tests for request routing."""

    def test_can_handle_orchestration_paths(self, handler):
        """Test can_handle for orchestration paths."""
        assert handler.can_handle("/api/v1/orchestration/templates")
        assert handler.can_handle("/api/v1/orchestration/deliberate")
        assert handler.can_handle("/api/v1/orchestration/deliberate/sync")
        assert handler.can_handle("/api/v1/orchestration/status/abc123")

    def test_cannot_handle_other_paths(self, handler):
        """Test can_handle rejects non-orchestration paths."""
        assert not handler.can_handle("/api/v1/payments/charge")
        assert not handler.can_handle("/api/v1/auth/login")
        assert not handler.can_handle("/api/orchestration/templates")  # Wrong version

    def test_routes_list(self, handler):
        """Test ROUTES list contains expected endpoints."""
        assert "/api/v1/orchestration/deliberate" in handler.ROUTES
        assert "/api/v1/orchestration/deliberate/sync" in handler.ROUTES
        assert "/api/v1/orchestration/status/*" in handler.ROUTES
        assert "/api/v1/orchestration/templates" in handler.ROUTES


class TestAuthentication:
    """Tests for authentication requirements."""

    @pytest.mark.asyncio
    async def test_handle_requires_auth(self, handler):
        """Test GET endpoints require authentication."""
        mock_handler = MagicMock()

        with patch.object(handler, "get_auth_context", side_effect=UnauthorizedError()):
            result = await handler.handle(
                "/api/v1/orchestration/templates",
                {},
                mock_handler,
            )

        assert result is not None
        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_handle_post_requires_auth(self, handler):
        """Test POST endpoints require authentication."""
        with patch.object(handler, "get_auth_context", side_effect=UnauthorizedError()):
            result = await handler.handle_post(
                "/api/v1/orchestration/deliberate",
                {"question": "Test?"},
                {},
                MagicMock(),
            )

        assert result is not None
        assert result.status_code == 401


class TestPermissions:
    """Tests for RBAC permission enforcement."""

    @pytest.mark.asyncio
    async def test_handle_requires_read_permission(self, handler, mock_auth_context):
        """Test GET endpoints require orchestration:read permission."""
        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", side_effect=ForbiddenError("denied")):
                result = await handler.handle(
                    "/api/v1/orchestration/templates",
                    {},
                    MagicMock(),
                )

        assert result is not None
        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_handle_post_requires_execute_permission(self, handler, mock_auth_context):
        """Test POST endpoints require orchestration:execute permission."""
        call_count = 0

        def mock_check_permission(ctx, perm, *args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # First call is execute permission
                raise ForbiddenError("denied")
            return None

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", side_effect=mock_check_permission):
                result = await handler.handle_post(
                    "/api/v1/orchestration/deliberate",
                    {"question": "Test?"},
                    {},
                    MagicMock(),
                )

        assert result is not None
        assert result.status_code == 403


class TestGetTemplates:
    """Tests for GET /templates endpoint."""

    @pytest.mark.asyncio
    async def test_get_templates_success(self, handler, mock_auth_context):
        """Test successful templates retrieval."""
        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission"):
                result = await handler.handle(
                    "/api/v1/orchestration/templates",
                    {},
                    MagicMock(),
                )

        assert result is not None
        assert result.status_code == 200

        body = json.loads(result.body)
        assert "templates" in body
        assert "count" in body
        assert body["count"] == len(TEMPLATES)

    @pytest.mark.asyncio
    async def test_get_templates_returns_template_data(self, handler, mock_auth_context):
        """Test templates endpoint returns template details."""
        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission"):
                result = await handler.handle(
                    "/api/v1/orchestration/templates",
                    {},
                    MagicMock(),
                )

        body = json.loads(result.body)
        if body["templates"]:
            template = body["templates"][0]
            assert "name" in template or "id" in template


class TestGetStatus:
    """Tests for GET /status/:id endpoint."""

    @pytest.mark.asyncio
    async def test_get_status_completed(self, handler, mock_auth_context):
        """Test status for completed request."""
        # Add a completed result
        result = OrchestrationResult(
            request_id="test-123",
            success=True,
            final_answer="Test answer",
        )
        _orchestration_results["test-123"] = result

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission"):
                response = await handler.handle(
                    "/api/v1/orchestration/status/test-123",
                    {},
                    MagicMock(),
                )

        assert response is not None
        assert response.status_code == 200

        body = json.loads(response.body)
        assert body["request_id"] == "test-123"
        assert body["status"] == "completed"
        assert body["result"] is not None

    @pytest.mark.asyncio
    async def test_get_status_in_progress(self, handler, mock_auth_context):
        """Test status for in-progress request."""
        # Add an in-progress request
        request = OrchestrationRequest(
            request_id="test-456",
            question="Test question",
        )
        _orchestration_requests["test-456"] = request

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission"):
                response = await handler.handle(
                    "/api/v1/orchestration/status/test-456",
                    {},
                    MagicMock(),
                )

        assert response is not None
        assert response.status_code == 200

        body = json.loads(response.body)
        assert body["request_id"] == "test-456"
        assert body["status"] == "in_progress"
        assert body["result"] is None

    @pytest.mark.asyncio
    async def test_get_status_not_found(self, handler, mock_auth_context):
        """Test status for non-existent request."""
        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission"):
                response = await handler.handle(
                    "/api/v1/orchestration/status/nonexistent",
                    {},
                    MagicMock(),
                )

        assert response is not None
        assert response.status_code == 404


class TestDeliberate:
    """Tests for POST /deliberate endpoint."""

    @pytest.mark.asyncio
    async def test_deliberate_missing_question(self, handler, mock_auth_context):
        """Test deliberate fails without question."""
        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission"):
                result = await handler.handle_post(
                    "/api/v1/orchestration/deliberate",
                    {},  # No question
                    {},
                    MagicMock(),
                )

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_deliberate_async_returns_202(self, handler, mock_auth_context):
        """Test async deliberate returns 202 Accepted."""
        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission"):
                with patch.object(handler, "_validate_knowledge_source", return_value=None):
                    with patch.object(handler, "_validate_output_channel", return_value=None):
                        result = await handler.handle_post(
                            "/api/v1/orchestration/deliberate",
                            {"question": "What is the best approach?"},
                            {},
                            MagicMock(),
                        )

        assert result is not None
        assert result.status_code == 202

        body = json.loads(result.body)
        assert "request_id" in body
        assert body["status"] == "queued"


class TestKnowledgeSourceValidation:
    """Tests for knowledge source validation."""

    def test_validate_knowledge_source_path_traversal(self, handler, mock_auth_context):
        """Test knowledge source with path traversal is rejected."""
        source = KnowledgeContextSource(
            source_type="slack",
            source_id="../etc/passwd",
        )

        result = handler._validate_knowledge_source(source, mock_auth_context)

        assert result is not None
        assert result.status_code == 400
        assert "path traversal" in json.loads(result.body)["error"].lower()

    def test_validate_knowledge_source_valid(self, handler, mock_auth_context):
        """Test valid knowledge source passes validation."""
        source = KnowledgeContextSource(
            source_type="slack",
            source_id="C12345678",
        )

        with patch.object(handler, "_check_permission", return_value=None):
            result = handler._validate_knowledge_source(source, mock_auth_context)

        assert result is None  # No error

    def test_validate_knowledge_source_permission_denied(self, handler, mock_auth_context):
        """Test knowledge source validation fails on permission denied."""
        source = KnowledgeContextSource(
            source_type="slack",
            source_id="C12345678",
        )

        from aragora.server.handlers.base import error_response

        with patch.object(
            handler, "_check_permission", return_value=error_response("Permission denied", 403)
        ):
            result = handler._validate_knowledge_source(source, mock_auth_context)

        assert result is not None
        assert result.status_code == 403


class TestOutputChannelValidation:
    """Tests for output channel validation."""

    def test_validate_output_channel_path_traversal(self, handler, mock_auth_context):
        """Test output channel with path traversal is rejected."""
        channel = OutputChannel(
            channel_type="slack",
            channel_id="../malicious",
        )

        result = handler._validate_output_channel(channel, mock_auth_context)

        assert result is not None
        assert result.status_code == 400

    def test_validate_output_channel_valid(self, handler, mock_auth_context):
        """Test valid output channel passes validation."""
        channel = OutputChannel(
            channel_type="slack",
            channel_id="C12345678",
        )

        with patch.object(handler, "_check_permission", return_value=None):
            result = handler._validate_output_channel(channel, mock_auth_context)

        assert result is None  # No error

    def test_validate_webhook_channel_invalid_url(self, handler, mock_auth_context):
        """Test webhook channel with invalid URL is rejected."""
        channel = OutputChannel(
            channel_type="webhook",
            channel_id="not-a-url",
        )

        result = handler._validate_output_channel(channel, mock_auth_context)

        assert result is not None
        assert result.status_code == 400


class TestTeamSelection:
    """Tests for agent team selection."""

    @pytest.mark.asyncio
    async def test_select_team_specified(self, handler):
        """Test SPECIFIED strategy uses provided agents."""
        request = OrchestrationRequest(
            request_id="test",
            question="Test",
            agents=["agent1", "agent2"],
            team_strategy=TeamStrategy.SPECIFIED,
        )

        agents = await handler._select_agent_team(request)

        assert agents == ["agent1", "agent2"]

    @pytest.mark.asyncio
    async def test_select_team_fast(self, handler):
        """Test FAST strategy returns 2 agents."""
        request = OrchestrationRequest(
            request_id="test",
            question="Test",
            team_strategy=TeamStrategy.FAST,
        )

        agents = await handler._select_agent_team(request)

        assert len(agents) == 2

    @pytest.mark.asyncio
    async def test_select_team_diverse(self, handler):
        """Test DIVERSE strategy returns more agents."""
        request = OrchestrationRequest(
            request_id="test",
            question="Test",
            team_strategy=TeamStrategy.DIVERSE,
        )

        agents = await handler._select_agent_team(request)

        assert len(agents) >= 3


class TestResultFormatting:
    """Tests for result formatting."""

    def test_format_result_summary(self, handler):
        """Test SUMMARY output format."""
        result = OrchestrationResult(
            request_id="test",
            success=True,
            final_answer="The answer is 42.",
        )
        request = OrchestrationRequest(
            request_id="test",
            question="What is the answer?",
            output_format=OutputFormat.SUMMARY,
        )

        message = handler._format_result_for_channel(result, request)

        assert "Deliberation Complete" in message
        assert "42" in message

    def test_format_result_standard(self, handler):
        """Test STANDARD output format."""
        result = OrchestrationResult(
            request_id="test",
            success=True,
            consensus_reached=True,
            final_answer="The answer is 42.",
            confidence=0.85,
            agents_participated=["agent1", "agent2"],
            duration_seconds=5.5,
        )
        request = OrchestrationRequest(
            request_id="test",
            question="What is the answer?",
            output_format=OutputFormat.STANDARD,
        )

        message = handler._format_result_for_channel(result, request)

        assert "Deliberation Result" in message
        assert "Consensus reached" in message
        assert "85%" in message
        assert "agent1" in message
        assert "5.5" in message
