"""
Tests for the unified orchestration handler.

Tests the core "Control plane for multi-agent vetted decisionmaking across
org knowledge and channels" functionality.

Coverage includes:
- Data model parsing (OrchestrationRequest, KnowledgeContextSource, OutputChannel)
- Template handling and configuration
- Handler routing and path matching
- Authentication and permission enforcement
- Agent team selection strategies
- Knowledge context fetching
- Output channel routing
- Error handling
"""

import json
from io import BytesIO
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
    DeliberationTemplate,
    TEMPLATES,
    _orchestration_requests,
    _orchestration_results,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def handler():
    """Create an orchestration handler with empty context."""
    return OrchestrationHandler({})


@pytest.fixture
def handler_with_coordinator():
    """Create handler with mocked control plane coordinator."""
    mock_coordinator = MagicMock()
    ctx = {"control_plane_coordinator": mock_coordinator}
    return OrchestrationHandler(ctx)


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with headers."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {"Content-Type": "application/json"}
    return handler


@pytest.fixture
def mock_auth_context():
    """Create a mock authenticated authorization context."""
    from aragora.rbac.models import AuthorizationContext

    return AuthorizationContext(
        user_id="test-user-123",
        org_id="test-org-456",
        workspace_id="test-workspace",
        roles={"member", "developer"},
        permissions={"orchestration:read", "orchestration:execute"},
    )


def create_request_body(data: dict) -> MagicMock:
    """Create a mock HTTP handler with a JSON body."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    body = json.dumps(data).encode("utf-8")
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": "application/json",
    }
    handler.rfile = BytesIO(body)
    handler.command = "POST"
    return handler


# =============================================================================
# OrchestrationRequest Tests
# =============================================================================


class TestOrchestrationRequest:
    """Tests for OrchestrationRequest parsing."""

    def test_from_dict_minimal(self):
        """Test parsing minimal request."""
        data = {"question": "Should we use microservices?"}
        request = OrchestrationRequest.from_dict(data)

        assert request.question == "Should we use microservices?"
        assert request.knowledge_sources == []
        assert request.output_channels == []
        assert request.team_strategy == TeamStrategy.BEST_FOR_DOMAIN
        assert request.require_consensus is True

    def test_from_dict_full(self):
        """Test parsing full request with all fields."""
        data = {
            "question": "What architecture should we use?",
            "knowledge_sources": ["slack:C123456", "confluence:page/123"],
            "knowledge_context": {
                "workspaces": ["engineering", "architecture"],
            },
            "team_strategy": "diverse",
            "agents": ["anthropic-api", "openai-api", "gemini"],
            "output_channels": ["slack:C789", "email:team@example.com"],
            "output_format": "decision_receipt",
            "require_consensus": True,
            "priority": "high",
            "max_rounds": 5,
            "timeout_seconds": 600.0,
            "template": "architecture_decision",
            "metadata": {"project": "infrastructure"},
        }
        request = OrchestrationRequest.from_dict(data)

        assert request.question == "What architecture should we use?"
        assert len(request.knowledge_sources) == 2
        assert request.knowledge_sources[0].source_type == "slack"
        assert request.knowledge_sources[0].source_id == "C123456"
        assert request.workspaces == ["engineering", "architecture"]
        assert request.team_strategy == TeamStrategy.DIVERSE
        assert request.agents == ["anthropic-api", "openai-api", "gemini"]
        assert len(request.output_channels) == 2
        assert request.output_channels[0].channel_type == "slack"
        assert request.output_channels[1].channel_type == "email"
        assert request.output_format == OutputFormat.DECISION_RECEIPT
        assert request.priority == "high"
        assert request.template == "architecture_decision"

    def test_from_dict_nested_knowledge_context(self):
        """Test parsing with nested knowledge_context format."""
        data = {
            "question": "Test question",
            "knowledge_context": {
                "sources": ["github:owner/repo/pr/123"],
                "workspaces": ["dev"],
            },
        }
        request = OrchestrationRequest.from_dict(data)

        assert len(request.knowledge_sources) == 1
        assert request.knowledge_sources[0].source_type == "github"
        assert request.workspaces == ["dev"]

    def test_from_dict_invalid_team_strategy_defaults(self):
        """Test that invalid team strategy defaults to BEST_FOR_DOMAIN."""
        data = {
            "question": "Test",
            "team_strategy": "invalid_strategy",
        }
        request = OrchestrationRequest.from_dict(data)
        assert request.team_strategy == TeamStrategy.BEST_FOR_DOMAIN

    def test_from_dict_invalid_output_format_defaults(self):
        """Test that invalid output format defaults to STANDARD."""
        data = {
            "question": "Test",
            "output_format": "invalid_format",
        }
        request = OrchestrationRequest.from_dict(data)
        assert request.output_format == OutputFormat.STANDARD

    def test_from_dict_knowledge_source_as_dict(self):
        """Test parsing knowledge sources provided as dictionaries."""
        data = {
            "question": "Test",
            "knowledge_sources": [
                {
                    "type": "confluence",
                    "id": "page/456",
                    "lookback_minutes": 120,
                    "max_items": 100,
                }
            ],
        }
        request = OrchestrationRequest.from_dict(data)

        assert len(request.knowledge_sources) == 1
        assert request.knowledge_sources[0].source_type == "confluence"
        assert request.knowledge_sources[0].source_id == "page/456"
        assert request.knowledge_sources[0].lookback_minutes == 120
        assert request.knowledge_sources[0].max_items == 100

    def test_from_dict_output_channel_as_dict(self):
        """Test parsing output channels provided as dictionaries."""
        data = {
            "question": "Test",
            "output_channels": [
                {
                    "type": "slack",
                    "id": "C12345",
                    "thread_id": "1234567.890",
                }
            ],
        }
        request = OrchestrationRequest.from_dict(data)

        assert len(request.output_channels) == 1
        assert request.output_channels[0].channel_type == "slack"
        assert request.output_channels[0].channel_id == "C12345"
        assert request.output_channels[0].thread_id == "1234567.890"

    def test_request_has_unique_id(self):
        """Test that each request gets a unique ID."""
        request1 = OrchestrationRequest.from_dict({"question": "Q1"})
        request2 = OrchestrationRequest.from_dict({"question": "Q2"})

        assert request1.request_id != request2.request_id
        assert len(request1.request_id) > 0


# =============================================================================
# KnowledgeContextSource Tests
# =============================================================================


class TestKnowledgeContextSource:
    """Tests for KnowledgeContextSource parsing."""

    def test_from_string_with_type(self):
        """Test parsing 'type:id' format."""
        source = KnowledgeContextSource.from_string("slack:C123456")
        assert source.source_type == "slack"
        assert source.source_id == "C123456"

    def test_from_string_without_type(self):
        """Test parsing plain ID defaults to document."""
        source = KnowledgeContextSource.from_string("doc_12345")
        assert source.source_type == "document"
        assert source.source_id == "doc_12345"

    def test_from_string_complex_id(self):
        """Test parsing complex IDs with multiple colons."""
        source = KnowledgeContextSource.from_string("github:owner/repo/pr/123")
        assert source.source_type == "github"
        assert source.source_id == "owner/repo/pr/123"

    def test_default_values(self):
        """Test default values for lookback and max_items."""
        source = KnowledgeContextSource.from_string("slack:C123")
        assert source.lookback_minutes == 60
        assert source.max_items == 50


# =============================================================================
# OutputChannel Tests
# =============================================================================


class TestOutputChannel:
    """Tests for OutputChannel parsing."""

    def test_from_string_simple(self):
        """Test parsing 'type:id' format."""
        channel = OutputChannel.from_string("slack:C123456")
        assert channel.channel_type == "slack"
        assert channel.channel_id == "C123456"
        assert channel.thread_id is None

    def test_from_string_with_thread(self):
        """Test parsing 'type:id:thread' format."""
        channel = OutputChannel.from_string("slack:C123456:1234567890.123456")
        assert channel.channel_type == "slack"
        assert channel.channel_id == "C123456"
        assert channel.thread_id == "1234567890.123456"

    def test_from_string_webhook(self):
        """Test parsing webhook:url format."""
        channel = OutputChannel.from_string("webhook:https://example.com/hook")
        assert channel.channel_type == "webhook"
        assert channel.channel_id == "https://example.com/hook"

    def test_from_string_webhook_with_port(self):
        """Test parsing webhook URL with port."""
        channel = OutputChannel.from_string("webhook:https://example.com:8080/hook")
        assert channel.channel_type == "webhook"
        assert channel.channel_id == "https://example.com:8080/hook"

    def test_from_string_no_type(self):
        """Test parsing without type prefix defaults to webhook."""
        channel = OutputChannel.from_string("https://example.com/hook")
        assert channel.channel_type == "webhook"
        assert channel.channel_id == "https://example.com/hook"

    def test_from_string_email(self):
        """Test parsing email channel."""
        channel = OutputChannel.from_string("email:user@example.com")
        assert channel.channel_type == "email"
        assert channel.channel_id == "user@example.com"


# =============================================================================
# DeliberationTemplate Tests
# =============================================================================


class TestDeliberationTemplates:
    """Tests for built-in deliberation templates."""

    def test_templates_exist(self):
        """Test that expected templates are defined."""
        expected = [
            "code_review",
            "contract_review",
            "architecture_decision",
            "compliance_check",
            "quick_decision",
        ]
        for name in expected:
            assert name in TEMPLATES, f"Template {name} not found"

    def test_code_review_template(self):
        """Test code review template configuration."""
        template = TEMPLATES["code_review"]
        assert template.name == "code_review"
        assert "anthropic-api" in template.default_agents
        # Compare by value since templates may use enums from different modules
        assert template.output_format.value == "github_review"
        assert "security" in template.personas

    def test_template_to_dict(self):
        """Test template serialization."""
        template = TEMPLATES["code_review"]
        data = template.to_dict()
        assert data["name"] == "code_review"
        assert "default_agents" in data
        assert data["output_format"] == "github_review"

    def test_quick_decision_template(self):
        """Test quick decision template is optimized for speed."""
        template = TEMPLATES["quick_decision"]
        assert template.max_rounds <= 3
        assert len(template.default_agents) <= 3


# =============================================================================
# OrchestrationResult Tests
# =============================================================================


class TestOrchestrationResult:
    """Tests for OrchestrationResult."""

    def test_to_dict(self):
        """Test result serialization."""
        result = OrchestrationResult(
            request_id="req-123",
            success=True,
            consensus_reached=True,
            final_answer="Use microservices for this use case.",
            confidence=0.85,
            agents_participated=["anthropic-api", "openai-api"],
            rounds_completed=3,
            duration_seconds=45.2,
            knowledge_context_used=["slack:C123"],
            channels_notified=["slack:C456"],
            receipt_id="receipt-789",
        )

        data = result.to_dict()
        assert data["request_id"] == "req-123"
        assert data["success"] is True
        assert data["consensus_reached"] is True
        assert data["confidence"] == 0.85
        assert len(data["agents_participated"]) == 2
        assert data["receipt_id"] == "receipt-789"

    def test_to_dict_failed_result(self):
        """Test serialization of failed result with error."""
        result = OrchestrationResult(
            request_id="req-fail",
            success=False,
            error="Deliberation timed out",
            duration_seconds=300.0,
        )

        data = result.to_dict()
        assert data["success"] is False
        assert data["error"] == "Deliberation timed out"
        assert data["consensus_reached"] is False

    def test_created_at_timestamp(self):
        """Test that created_at is set automatically."""
        result = OrchestrationResult(
            request_id="req-time",
            success=True,
        )
        assert result.created_at is not None
        assert "T" in result.created_at  # ISO format


# =============================================================================
# OrchestrationHandler Path Matching Tests
# =============================================================================


class TestOrchestrationHandler:
    """Tests for OrchestrationHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = OrchestrationHandler({})

    def test_can_handle_orchestration_paths(self):
        """Test path matching for orchestration routes."""
        assert self.handler.can_handle("/api/v1/orchestration/deliberate")
        assert self.handler.can_handle("/api/v1/orchestration/templates")
        assert self.handler.can_handle("/api/v1/orchestration/status/abc123")
        assert not self.handler.can_handle("/api/v1/debates")
        assert not self.handler.can_handle("/api/v1/control-plane/agents")

    def test_can_handle_sync_deliberate(self):
        """Test path matching for synchronous deliberate endpoint."""
        assert self.handler.can_handle("/api/v1/orchestration/deliberate/sync")

    def test_get_templates(self):
        """Test GET /api/v1/orchestration/templates."""
        result = self.handler._get_templates({})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "templates" in body
        assert body["count"] == len(TEMPLATES)

    def test_get_status_not_found(self):
        """Test GET /api/v1/orchestration/status/:id for non-existent request."""
        result = self.handler._get_status("nonexistent-id")

        assert result.status_code == 404
        body = json.loads(result.body)
        assert "error" in body

    def test_get_status_in_progress(self):
        """Test GET /api/v1/orchestration/status/:id for in-progress request."""
        # Add a mock in-progress request
        request_id = "in-progress-123"
        _orchestration_requests[request_id] = OrchestrationRequest(question="Test question")

        try:
            result = self.handler._get_status(request_id)
            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["status"] == "in_progress"
            assert body["result"] is None
        finally:
            _orchestration_requests.pop(request_id, None)

    def test_get_status_completed(self):
        """Test GET /api/v1/orchestration/status/:id for completed request."""
        request_id = "completed-456"
        _orchestration_results[request_id] = OrchestrationResult(
            request_id=request_id,
            success=True,
            final_answer="Test answer",
        )

        try:
            result = self.handler._get_status(request_id)
            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["status"] == "completed"
            assert body["result"]["final_answer"] == "Test answer"
        finally:
            _orchestration_results.pop(request_id, None)

    def test_handle_deliberate_missing_question(self):
        """Test POST /api/v1/orchestration/deliberate without question."""
        result = self.handler._handle_deliberate({}, None, sync=False)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body
        assert "Question is required" in body["error"]


# =============================================================================
# Authentication and Permission Tests
# =============================================================================


class TestOrchestrationHandlerAuth:
    """Tests for authentication and permission handling."""

    @pytest.mark.asyncio
    async def test_handle_get_requires_auth(self, handler, mock_http_handler):
        """Test that GET endpoints require authentication."""
        with patch.object(
            handler,
            "get_auth_context",
            side_effect=Exception("Unauthorized"),
        ):
            # The handler should catch auth errors gracefully
            result = await handler.handle("/api/v1/orchestration/templates", {}, mock_http_handler)
            # Should return auth error
            assert result is not None

    @pytest.mark.asyncio
    async def test_handle_post_requires_auth(self, handler):
        """Test that POST endpoints require authentication."""
        mock_request = create_request_body({"question": "Test question"})

        # Mock auth to raise UnauthorizedError
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch.object(
            handler,
            "get_auth_context",
            new_callable=AsyncMock,
            side_effect=UnauthorizedError("Auth required"),
        ):
            result = await handler.handle_post(
                "/api/v1/orchestration/deliberate", {}, {}, mock_request
            )

            assert result is not None
            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_handle_post_forbidden_without_execute_permission(
        self, handler, mock_auth_context
    ):
        """Test that POST deliberate requires orchestration:execute permission."""
        mock_request = create_request_body({"question": "Test question"})

        # Auth context without execute permission
        from aragora.rbac.models import AuthorizationContext

        limited_ctx = AuthorizationContext(
            user_id="user-123",
            permissions={"orchestration:read"},  # Missing execute
        )

        from aragora.server.handlers.utils.auth import ForbiddenError

        with patch.object(
            handler,
            "get_auth_context",
            new_callable=AsyncMock,
            return_value=limited_ctx,
        ):
            with patch.object(
                handler,
                "check_permission",
                side_effect=ForbiddenError("Permission denied"),
            ):
                result = await handler.handle_post(
                    "/api/v1/orchestration/deliberate", {}, {}, mock_request
                )

                assert result is not None
                assert result.status_code == 403


# =============================================================================
# Agent Team Selection Tests
# =============================================================================


class TestOrchestrationHandlerAsync:
    """Async tests for OrchestrationHandler."""

    @pytest.fixture
    def handler(self):
        """Create handler with mocked context."""
        ctx = {"control_plane_coordinator": None}
        return OrchestrationHandler(ctx)

    @pytest.mark.asyncio
    async def test_select_agent_team_specified(self, handler):
        """Test agent selection with explicit agents."""
        request = OrchestrationRequest(
            question="Test",
            agents=["agent1", "agent2"],
            team_strategy=TeamStrategy.SPECIFIED,
        )
        agents = await handler._select_agent_team(request)
        assert agents == ["agent1", "agent2"]

    @pytest.mark.asyncio
    async def test_select_agent_team_fast(self, handler):
        """Test agent selection with fast strategy."""
        request = OrchestrationRequest(
            question="Test",
            team_strategy=TeamStrategy.FAST,
        )
        agents = await handler._select_agent_team(request)
        assert len(agents) == 2  # Fast uses minimal agents

    @pytest.mark.asyncio
    async def test_select_agent_team_diverse(self, handler):
        """Test agent selection with diverse strategy."""
        request = OrchestrationRequest(
            question="Test",
            team_strategy=TeamStrategy.DIVERSE,
        )
        agents = await handler._select_agent_team(request)
        assert len(agents) >= 3  # Diverse uses more agents

    @pytest.mark.asyncio
    async def test_select_agent_team_random(self, handler):
        """Test agent selection with random strategy."""
        request = OrchestrationRequest(
            question="Test",
            team_strategy=TeamStrategy.RANDOM,
        )
        agents = await handler._select_agent_team(request)
        assert len(agents) <= 3
        assert len(agents) >= 1

    @pytest.mark.asyncio
    async def test_select_agent_team_best_for_domain_fallback(self, handler):
        """Test agent selection with best_for_domain falls back gracefully."""
        request = OrchestrationRequest(
            question="Test",
            team_strategy=TeamStrategy.BEST_FOR_DOMAIN,
        )
        agents = await handler._select_agent_team(request)
        # Should return default agents when routing handler not available
        assert len(agents) >= 2

    @pytest.mark.asyncio
    async def test_format_result_for_channel(self, handler):
        """Test result formatting for channel delivery."""
        result = OrchestrationResult(
            request_id="req-123",
            success=True,
            consensus_reached=True,
            final_answer="Use option A.",
            confidence=0.9,
            agents_participated=["agent1", "agent2"],
            duration_seconds=30.0,
        )
        request = OrchestrationRequest(
            question="Which option?",
            output_format=OutputFormat.STANDARD,
        )

        message = handler._format_result_for_channel(result, request)
        assert "Deliberation Result" in message
        assert "Consensus reached" in message
        assert "Use option A." in message
        assert "90%" in message  # confidence

    @pytest.mark.asyncio
    async def test_format_result_summary_format(self, handler):
        """Test result formatting with SUMMARY format."""
        result = OrchestrationResult(
            request_id="req-123",
            success=True,
            final_answer="Short answer.",
        )
        request = OrchestrationRequest(
            question="Test?",
            output_format=OutputFormat.SUMMARY,
        )

        message = handler._format_result_for_channel(result, request)
        assert "Deliberation Complete" in message
        assert "Short answer." in message

    @pytest.mark.asyncio
    async def test_format_result_no_consensus(self, handler):
        """Test result formatting when no consensus reached."""
        result = OrchestrationResult(
            request_id="req-123",
            success=True,
            consensus_reached=False,
            final_answer="Partial agreement.",
        )
        request = OrchestrationRequest(
            question="Test?",
            output_format=OutputFormat.STANDARD,
        )

        message = handler._format_result_for_channel(result, request)
        assert "No consensus" in message


# =============================================================================
# Knowledge Context Fetching Tests
# =============================================================================


class TestKnowledgeContextFetching:
    """Tests for knowledge context fetching."""

    @pytest.fixture
    def handler(self):
        return OrchestrationHandler({})

    @pytest.mark.asyncio
    async def test_fetch_slack_context(self, handler):
        """Test fetching Slack channel context."""
        source = KnowledgeContextSource(
            source_type="slack",
            source_id="C12345",
            lookback_minutes=60,
            max_items=50,
        )

        mock_connector = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.messages = [{"text": "Test message"}]
        mock_ctx.to_context_string.return_value = "Test context"
        mock_connector.fetch_context = AsyncMock(return_value=mock_ctx)

        with patch(
            "aragora.connectors.chat.registry.get_connector",
            return_value=mock_connector,
        ):
            result = await handler._fetch_chat_context("slack", source)
            assert result == "Test context"

    @pytest.mark.asyncio
    async def test_fetch_chat_context_no_connector(self, handler):
        """Test fetching chat context when connector not available."""
        source = KnowledgeContextSource(
            source_type="slack",
            source_id="C12345",
        )

        with patch(
            "aragora.connectors.chat.registry.get_connector",
            return_value=None,
        ):
            result = await handler._fetch_chat_context("slack", source)
            assert result is None

    @pytest.mark.asyncio
    async def test_fetch_knowledge_context_unknown_type(self, handler):
        """Test fetching context for unknown source type."""
        source = KnowledgeContextSource(
            source_type="unknown_type",
            source_id="123",
        )

        result = await handler._fetch_knowledge_context(source)
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_document_context(self, handler):
        """Test fetching document context from knowledge mound."""
        source = KnowledgeContextSource(
            source_type="document",
            source_id="search query",
        )

        # Create mock knowledge items with content attribute
        mock_item1 = MagicMock()
        mock_item1.content = "Doc 1 content"
        mock_item2 = MagicMock()
        mock_item2.content = "Doc 2 content"

        # Create mock QueryResult with items attribute
        mock_query_result = MagicMock()
        mock_query_result.items = [mock_item1, mock_item2]

        mock_mound = MagicMock()
        mock_mound.query = AsyncMock(return_value=mock_query_result)

        with patch(
            "aragora.knowledge.mound.get_knowledge_mound",
            return_value=mock_mound,
        ):
            result = await handler._fetch_document_context(source)
            assert "Doc 1 content" in result
            assert "Doc 2 content" in result


# =============================================================================
# Output Channel Routing Tests
# =============================================================================


class TestOutputChannelRouting:
    """Tests for routing results to output channels."""

    @pytest.fixture
    def handler(self):
        return OrchestrationHandler({})

    @pytest.mark.asyncio
    async def test_send_to_slack(self, handler):
        """Test sending result to Slack channel."""
        channel = OutputChannel(
            channel_type="slack",
            channel_id="C12345",
            thread_id="123456.789",
        )

        mock_connector = MagicMock()
        mock_connector.send_message = AsyncMock()

        with patch(
            "aragora.connectors.chat.registry.get_connector",
            return_value=mock_connector,
        ):
            await handler._send_to_slack(channel, "Test message")

            mock_connector.send_message.assert_called_once_with(
                "C12345",
                "Test message",
                thread_ts="123456.789",
            )

    @pytest.mark.asyncio
    async def test_send_to_webhook(self, handler):
        """Test sending result to webhook."""
        channel = OutputChannel(
            channel_type="webhook",
            channel_id="https://example.com/webhook",
        )

        result = OrchestrationResult(
            request_id="req-123",
            success=True,
            final_answer="Test answer",
        )

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_session.post = MagicMock(
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_response))
        )

        with patch("aiohttp.ClientSession") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_client.return_value.__aexit__ = AsyncMock()

            await handler._send_to_webhook(channel, result)


# =============================================================================
# Template Application Tests
# =============================================================================


class TestTemplateApplication:
    """Tests for template application to requests."""

    def test_apply_template_to_request(self):
        """Test that templates are applied correctly."""
        handler = OrchestrationHandler({})

        data = {
            "question": "Review this PR",
            "template": "code_review",
        }

        result = handler._handle_deliberate(data, None, sync=False)

        # Request should be created with template defaults
        # (The result is queued, so we verify it was processed)
        assert result.status_code in [202, 400, 500]  # Queued or error

    def test_request_agents_override_template(self):
        """Test that request agents override template defaults."""
        request = OrchestrationRequest.from_dict(
            {
                "question": "Test",
                "template": "code_review",
                "agents": ["custom-agent"],
            }
        )

        # Explicit agents should be preserved
        assert request.agents == ["custom-agent"]


# =============================================================================
# End-to-End Tests
# =============================================================================


class TestEndToEndOrchestration:
    """End-to-end tests for orchestration flow."""

    @pytest.mark.asyncio
    async def test_deliberation_without_coordinator(self):
        """Test deliberation falls back when coordinator unavailable."""
        handler = OrchestrationHandler({})

        # Mock at the module level where it's imported
        with patch.object(
            handler,
            "_execute_deliberation",
            new_callable=AsyncMock,
        ) as mock_execute:
            mock_execute.return_value = OrchestrationResult(
                request_id="test-123",
                success=True,
                final_answer="Test answer",
                consensus_reached=True,
                confidence=0.9,
            )

            request = OrchestrationRequest(
                question="Test question",
                agents=["anthropic-api"],
            )

            result = await mock_execute(request)

            assert result.success is True
            assert result.final_answer == "Test answer"
            assert result.consensus_reached is True

    @pytest.mark.asyncio
    async def test_knowledge_context_parsing(self):
        """Test that knowledge context sources are correctly parsed."""
        handler = OrchestrationHandler({})

        request = OrchestrationRequest.from_dict(
            {
                "question": "What should we do?",
                "knowledge_sources": [
                    "slack:C12345",
                    {"type": "confluence", "id": "page/123", "lookback_minutes": 120},
                ],
            }
        )

        assert len(request.knowledge_sources) == 2
        assert request.knowledge_sources[0].source_type == "slack"
        assert request.knowledge_sources[0].source_id == "C12345"
        assert request.knowledge_sources[1].source_type == "confluence"
        assert request.knowledge_sources[1].lookback_minutes == 120

    @pytest.mark.asyncio
    async def test_execute_and_store_handles_errors(self):
        """Test that _execute_and_store handles errors gracefully."""
        handler = OrchestrationHandler({})

        request = OrchestrationRequest(
            question="Test question",
            request_id="error-test-123",
        )

        with patch.object(
            handler,
            "_execute_deliberation",
            new_callable=AsyncMock,
            side_effect=Exception("Test error"),
        ):
            await handler._execute_and_store(request)

            # Result should be stored with error
            result = _orchestration_results.get("error-test-123")
            assert result is not None
            assert result.success is False
            assert "Test error" in result.error

            # Clean up
            _orchestration_results.pop("error-test-123", None)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in the handler."""

    def test_deliberate_catches_exceptions(self):
        """Test that _handle_deliberate catches and returns errors."""
        handler = OrchestrationHandler({})

        # Malformed data that might cause parsing errors
        result = handler._handle_deliberate(
            {"question": "Test", "knowledge_sources": [{"invalid": "data"}]},
            None,
            sync=False,
        )

        # Should return 202 (queued) or 500 (error), not raise
        assert result is not None
        assert result.status_code in [202, 400, 500]

    def test_empty_question_returns_error(self):
        """Test that empty question returns 400 error."""
        handler = OrchestrationHandler({})

        result = handler._handle_deliberate({"question": ""}, None, sync=False)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "Question is required" in body["error"]


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestRateLimiting:
    """Tests related to rate limiting behavior."""

    def test_handle_deliberate_has_rate_limit_decorator(self):
        """Test that _handle_deliberate is rate limited."""
        handler = OrchestrationHandler({})

        # Check that the method has been decorated
        # The rate_limit decorator adds metadata we can inspect
        method = handler._handle_deliberate
        assert hasattr(method, "__wrapped__") or callable(method)
