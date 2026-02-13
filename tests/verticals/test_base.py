"""Tests for VerticalSpecialistAgent base class.

Tests the abstract base class functionality for vertical specialists.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from aragora.verticals.base import VerticalSpecialistAgent
from aragora.verticals.config import (
    VerticalConfig,
    ToolConfig,
    ComplianceConfig,
    ComplianceLevel,
    ModelConfig,
)
from aragora.core import Message, Critique


# =============================================================================
# Concrete Test Implementation
# =============================================================================


class ConcreteVerticalAgent(VerticalSpecialistAgent):
    """Concrete implementation for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._execute_tool_mock = AsyncMock(return_value={"result": "ok"})
        self._check_framework_mock = AsyncMock(return_value=[])
        self._generate_response_mock = AsyncMock()

    async def _execute_tool(self, tool, parameters):
        return await self._execute_tool_mock(tool, parameters)

    async def _check_framework_compliance(self, content, framework):
        return await self._check_framework_mock(content, framework)

    async def _generate_response(self, task, system_prompt, context=None, **kwargs):
        return await self._generate_response_mock(task, system_prompt, context, **kwargs)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def model_config():
    """Create a basic model config."""
    return ModelConfig(
        primary_model="test-model",
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,
    )


@pytest.fixture
def tool_configs():
    """Create test tool configurations."""
    return [
        ToolConfig(
            name="code_review",
            description="Review code for issues",
            enabled=True,
            parameters={"language": "python"},
        ),
        ToolConfig(
            name="security_scan",
            description="Scan for security vulnerabilities",
            enabled=True,
            parameters={},
        ),
        ToolConfig(
            name="disabled_tool",
            description="This tool is disabled",
            enabled=False,
            parameters={},
        ),
    ]


@pytest.fixture
def compliance_configs():
    """Create test compliance configurations."""
    return [
        ComplianceConfig(
            framework="SOC2",
            level=ComplianceLevel.ENFORCED,
            rules=["encrypt_at_rest", "audit_logging"],
        ),
        ComplianceConfig(
            framework="GDPR",
            level=ComplianceLevel.ADVISORY,
            rules=["data_minimization", "consent"],
        ),
    ]


@pytest.fixture
def vertical_config(model_config, tool_configs, compliance_configs):
    """Create a vertical config."""
    return VerticalConfig(
        vertical_id="software",
        display_name="Software Engineering",
        description="Software development specialist",
        system_prompt_template="You are a {{ display_name }} specialist. Tools: {{ tools }}",
        expertise_areas=["Code Review", "Security Analysis", "Architecture"],
        tools=tool_configs,
        compliance_frameworks=compliance_configs,
        model_config=model_config,
    )


@pytest.fixture
def agent(vertical_config):
    """Create a test agent."""
    return ConcreteVerticalAgent(
        name="test-agent",
        model="gpt-4",
        config=vertical_config,
        role="analyst",
        api_key="test-key",
        timeout=60,
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestVerticalSpecialistAgentInit:
    """Test agent initialization."""

    def test_name_set(self, agent):
        """Test name is set."""
        assert agent.name == "test-agent"

    def test_model_set(self, agent):
        """Test model is set."""
        assert agent.model == "gpt-4"

    def test_role_set(self, agent):
        """Test role is set."""
        assert agent.role == "analyst"

    def test_config_stored(self, agent, vertical_config):
        """Test config is stored."""
        assert agent._config == vertical_config

    def test_tools_dict_created(self, agent):
        """Test tools dict is created from list."""
        assert "code_review" in agent._tools
        assert "security_scan" in agent._tools
        assert "disabled_tool" in agent._tools

    def test_tool_history_empty(self, agent):
        """Test tool call history starts empty."""
        assert agent._tool_call_history == []


# =============================================================================
# Property Tests
# =============================================================================


class TestVerticalSpecialistAgentProperties:
    """Test agent properties."""

    def test_vertical_id(self, agent):
        """Test vertical_id property."""
        assert agent.vertical_id == "software"

    def test_config_property(self, agent, vertical_config):
        """Test config property."""
        assert agent.config == vertical_config

    def test_expertise_areas(self, agent):
        """Test expertise_areas property."""
        areas = agent.expertise_areas
        assert "Code Review" in areas
        assert "Security Analysis" in areas
        assert "Architecture" in areas
        assert len(areas) == 3


# =============================================================================
# System Prompt Tests
# =============================================================================


class TestVerticalSpecialistAgentSystemPrompt:
    """Test system prompt building."""

    def test_build_system_prompt_basic(self, agent):
        """Test basic system prompt building."""
        prompt = agent.build_system_prompt()
        assert "Software Engineering specialist" in prompt

    def test_build_system_prompt_includes_tools(self, agent):
        """Test system prompt includes tools."""
        prompt = agent.build_system_prompt()
        # Template uses {{ tools }} which is list of enabled tool names
        assert "code_review" in prompt or "Tools:" in prompt

    def test_build_system_prompt_with_context(self, agent):
        """Test system prompt with additional context."""
        prompt = agent.build_system_prompt(context={"extra": "value"})
        # Should not raise, context is merged


# =============================================================================
# Tool Management Tests
# =============================================================================


class TestVerticalSpecialistAgentTools:
    """Test tool management."""

    def test_get_tool_found(self, agent):
        """Test getting existing tool."""
        tool = agent.get_tool("code_review")
        assert tool is not None
        assert tool.name == "code_review"

    def test_get_tool_not_found(self, agent):
        """Test getting non-existent tool."""
        tool = agent.get_tool("nonexistent")
        assert tool is None

    def test_get_enabled_tools(self, agent):
        """Test getting only enabled tools."""
        enabled = agent.get_enabled_tools()
        names = [t.name for t in enabled]
        assert "code_review" in names
        assert "security_scan" in names
        assert "disabled_tool" not in names

    def test_get_enabled_tools_count(self, agent):
        """Test enabled tools count."""
        enabled = agent.get_enabled_tools()
        assert len(enabled) == 2


# =============================================================================
# Tool Invocation Tests
# =============================================================================


class TestVerticalSpecialistAgentInvokeTool:
    """Test tool invocation."""

    @pytest.mark.asyncio
    async def test_invoke_tool_success(self, agent):
        """Test successful tool invocation."""
        result = await agent.invoke_tool("code_review", {"file": "test.py"})
        assert result == {"result": "ok"}
        agent._execute_tool_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_tool_records_history(self, agent):
        """Test tool invocation records history."""
        await agent.invoke_tool("code_review", {"file": "test.py"})
        assert len(agent._tool_call_history) == 1
        history = agent._tool_call_history[0]
        assert history["tool"] == "code_review"
        assert history["parameters"] == {"file": "test.py"}
        assert "timestamp" in history

    @pytest.mark.asyncio
    async def test_invoke_tool_not_found(self, agent):
        """Test invoking non-existent tool raises."""
        with pytest.raises(ValueError, match="Tool not found"):
            await agent.invoke_tool("nonexistent", {})

    @pytest.mark.asyncio
    async def test_invoke_tool_disabled(self, agent):
        """Test invoking disabled tool raises."""
        with pytest.raises(ValueError, match="Tool not enabled"):
            await agent.invoke_tool("disabled_tool", {})


# =============================================================================
# Compliance Tests
# =============================================================================


class TestVerticalSpecialistAgentCompliance:
    """Test compliance checking."""

    @pytest.mark.asyncio
    async def test_check_compliance_all_frameworks(self, agent):
        """Test checking all compliance frameworks."""
        await agent.check_compliance("test content")
        # Should call for each framework
        assert agent._check_framework_mock.call_count == 2

    @pytest.mark.asyncio
    async def test_check_compliance_specific_framework(self, agent):
        """Test checking specific framework."""
        await agent.check_compliance("test content", framework="SOC2")
        # Should only call for SOC2
        assert agent._check_framework_mock.call_count == 1

    @pytest.mark.asyncio
    async def test_check_compliance_unknown_framework(self, agent):
        """Test checking unknown framework."""
        result = await agent.check_compliance("test content", framework="UNKNOWN")
        # Should return empty since no matching framework
        assert result == []
        assert agent._check_framework_mock.call_count == 0


class TestVerticalSpecialistAgentShouldBlock:
    """Test compliance blocking logic."""

    def test_should_block_enforced_violation(self, agent):
        """Test blocking on enforced violation."""
        violations = [{"framework": "SOC2", "message": "Missing encryption", "severity": "high"}]
        assert agent.should_block_on_compliance(violations) is True

    def test_should_not_block_advisory_violation(self, agent):
        """Test not blocking on advisory violation."""
        violations = [
            {"framework": "GDPR", "message": "Consider data minimization", "severity": "medium"}
        ]
        assert agent.should_block_on_compliance(violations) is False

    def test_should_not_block_empty_violations(self, agent):
        """Test not blocking with no violations."""
        assert agent.should_block_on_compliance([]) is False

    def test_should_block_mixed_violations(self, agent):
        """Test blocking when any enforced violation exists."""
        violations = [
            {"framework": "GDPR", "message": "Advisory", "severity": "low"},
            {"framework": "SOC2", "message": "Enforced", "severity": "high"},
        ]
        assert agent.should_block_on_compliance(violations) is True


# =============================================================================
# Tool History Tests
# =============================================================================


class TestVerticalSpecialistAgentToolHistory:
    """Test tool call history management."""

    @pytest.mark.asyncio
    async def test_get_tool_call_history(self, agent):
        """Test getting tool call history."""
        await agent.invoke_tool("code_review", {"a": 1})
        await agent.invoke_tool("security_scan", {"b": 2})

        history = agent.get_tool_call_history()
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_get_tool_call_history_returns_copy(self, agent):
        """Test history returns a copy."""
        await agent.invoke_tool("code_review", {})

        history1 = agent.get_tool_call_history()
        history2 = agent.get_tool_call_history()
        assert history1 is not history2

    @pytest.mark.asyncio
    async def test_clear_tool_call_history(self, agent):
        """Test clearing tool call history."""
        await agent.invoke_tool("code_review", {})
        assert len(agent.get_tool_call_history()) == 1

        agent.clear_tool_call_history()
        assert len(agent.get_tool_call_history()) == 0


# =============================================================================
# Serialization Tests
# =============================================================================


class TestVerticalSpecialistAgentSerialization:
    """Test agent serialization."""

    def test_to_dict_structure(self, agent):
        """Test to_dict returns correct structure."""
        d = agent.to_dict()
        assert "name" in d
        assert "model" in d
        assert "role" in d
        assert "vertical_id" in d
        assert "expertise_areas" in d
        assert "tools" in d
        assert "compliance_frameworks" in d

    def test_to_dict_values(self, agent):
        """Test to_dict has correct values."""
        d = agent.to_dict()
        assert d["name"] == "test-agent"
        assert d["model"] == "gpt-4"
        assert d["role"] == "analyst"
        assert d["vertical_id"] == "software"

    def test_to_dict_tools_are_names(self, agent):
        """Test tools in dict are just names (enabled only)."""
        d = agent.to_dict()
        assert "code_review" in d["tools"]
        assert "security_scan" in d["tools"]
        assert "disabled_tool" not in d["tools"]

    def test_to_dict_compliance_are_names(self, agent):
        """Test compliance frameworks are just names."""
        d = agent.to_dict()
        assert "SOC2" in d["compliance_frameworks"]
        assert "GDPR" in d["compliance_frameworks"]


# =============================================================================
# Critique Tests
# =============================================================================


class TestVerticalSpecialistAgentCritique:
    """Test critique generation."""

    @pytest.mark.asyncio
    async def test_critique_no_violations(self, agent):
        """Test critique with no violations."""
        agent._check_framework_mock.return_value = []

        critique = await agent.critique(
            proposal="Some proposal",
            task="Review this code",
        )

        assert isinstance(critique, Critique)
        assert critique.agent == "test-agent"
        assert critique.severity == 0.0  # No violations

    @pytest.mark.asyncio
    async def test_critique_with_violations(self, agent):
        """Test critique with violations."""
        agent._check_framework_mock.return_value = [
            {
                "framework": "SOC2",
                "message": "Missing audit",
                "rule": "audit_logging",
                "severity": "high",
            }
        ]

        critique = await agent.critique(
            proposal="Unsafe proposal",
            task="Review security",
        )

        assert critique.severity > 0
        assert any("SOC2" in issue for issue in critique.issues)

    @pytest.mark.asyncio
    async def test_critique_severity_mapping(self, agent):
        """Test severity mapping in critique."""
        agent._check_framework_mock.return_value = [
            {"framework": "SOC2", "message": "Critical issue", "rule": "r1", "severity": "critical"}
        ]

        critique = await agent.critique(proposal="p", task="t")
        assert critique.severity == 10.0  # Critical maps to 10

    @pytest.mark.asyncio
    async def test_critique_target_agent(self, agent):
        """Test critique target agent."""
        agent._check_framework_mock.return_value = []

        critique = await agent.critique(
            proposal="p",
            task="t",
            target_agent="other-agent",
        )

        assert critique.target_agent == "other-agent"

    @pytest.mark.asyncio
    async def test_critique_target_content_truncated(self, agent):
        """Test target content is truncated."""
        agent._check_framework_mock.return_value = []
        long_proposal = "x" * 500

        critique = await agent.critique(proposal=long_proposal, task="t")

        assert len(critique.target_content) <= 200


# =============================================================================
# Generate Tests
# =============================================================================


class TestVerticalSpecialistAgentGenerate:
    """Test generate method."""

    @pytest.mark.asyncio
    async def test_generate_calls_internal_method(self, agent):
        """Test generate calls _generate_response."""
        mock_msg = Message(role="assistant", content="Response", agent="test")
        agent._generate_response_mock.return_value = mock_msg

        result = await agent.generate("Test prompt")

        assert result == "Response"
        agent._generate_response_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_builds_system_prompt(self, agent):
        """Test generate builds system prompt."""
        mock_msg = Message(role="assistant", content="Response", agent="test")
        agent._generate_response_mock.return_value = mock_msg

        await agent.generate("Test prompt")

        # Check system prompt was built and passed
        call_args = agent._generate_response_mock.call_args
        system_prompt = call_args[0][1]  # Second positional arg
        assert "specialist" in system_prompt.lower()
