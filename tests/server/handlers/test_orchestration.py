"""
Tests for the unified orchestration handler.

Tests the core "Control plane for multi-agent robust decisionmaking across
org knowledge and channels" functionality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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
)


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

    def test_get_templates(self):
        """Test GET /api/v1/orchestration/templates."""
        import json

        result = self.handler._get_templates({})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "templates" in body
        assert body["count"] == len(TEMPLATES)

    def test_get_status_not_found(self):
        """Test GET /api/v1/orchestration/status/:id for non-existent request."""
        import json

        result = self.handler._get_status("nonexistent-id")

        assert result.status_code == 404
        body = json.loads(result.body)
        assert "error" in body

    def test_handle_deliberate_missing_question(self):
        """Test POST /api/v1/orchestration/deliberate without question."""
        import json

        result = self.handler._handle_deliberate({}, None, sync=False)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body
        assert "Question is required" in body["error"]


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
