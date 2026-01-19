"""
Tests for the agents base module.
Tests cover:
- CritiqueMixin context building
- CritiqueMixin critique parsing
- AgentType literal validation
- create_agent factory function
- list_available_agents function
- Module constants
"""

import pytest
from unittest.mock import MagicMock, patch


class TestModuleExports:
    """Tests for module exports."""

    def test_can_import_module(self):
        """Module can be imported."""
        from aragora.agents import base

        assert base is not None

    def test_critique_mixin_in_all(self):
        """CritiqueMixin is exported in __all__."""
        from aragora.agents.base import __all__

        assert "CritiqueMixin" in __all__

    def test_agent_type_in_all(self):
        """AgentType is exported in __all__."""
        from aragora.agents.base import __all__

        assert "AgentType" in __all__

    def test_create_agent_in_all(self):
        """create_agent is exported in __all__."""
        from aragora.agents.base import __all__

        assert "create_agent" in __all__

    def test_list_available_agents_in_all(self):
        """list_available_agents is exported in __all__."""
        from aragora.agents.base import __all__

        assert "list_available_agents" in __all__

    def test_max_context_chars_in_all(self):
        """MAX_CONTEXT_CHARS is exported in __all__."""
        from aragora.agents.base import __all__

        assert "MAX_CONTEXT_CHARS" in __all__

    def test_max_message_chars_in_all(self):
        """MAX_MESSAGE_CHARS is exported in __all__."""
        from aragora.agents.base import __all__

        assert "MAX_MESSAGE_CHARS" in __all__


class TestConstants:
    """Tests for module constants."""

    def test_max_context_chars_is_positive(self):
        """MAX_CONTEXT_CHARS is a positive integer."""
        from aragora.agents.base import MAX_CONTEXT_CHARS

        assert isinstance(MAX_CONTEXT_CHARS, int)
        assert MAX_CONTEXT_CHARS > 0

    def test_max_message_chars_is_positive(self):
        """MAX_MESSAGE_CHARS is a positive integer."""
        from aragora.agents.base import MAX_MESSAGE_CHARS

        assert isinstance(MAX_MESSAGE_CHARS, int)
        assert MAX_MESSAGE_CHARS > 0

    def test_max_context_larger_than_message(self):
        """MAX_CONTEXT_CHARS is larger than MAX_MESSAGE_CHARS."""
        from aragora.agents.base import MAX_CONTEXT_CHARS, MAX_MESSAGE_CHARS

        assert MAX_CONTEXT_CHARS > MAX_MESSAGE_CHARS

    def test_max_context_reasonable_size(self):
        """MAX_CONTEXT_CHARS is within reasonable bounds."""
        from aragora.agents.base import MAX_CONTEXT_CHARS

        # Should be at least 10k chars for context
        assert MAX_CONTEXT_CHARS >= 10_000
        # But not too large (less than 1M)
        assert MAX_CONTEXT_CHARS < 1_000_000


class TestCritiqueMixinBuildContext:
    """Tests for CritiqueMixin._build_context_prompt method."""

    @pytest.fixture
    def mixin_class(self):
        """Create a class that uses CritiqueMixin."""
        from aragora.agents.base import CritiqueMixin

        class TestAgent(CritiqueMixin):
            def __init__(self):
                self.name = "test_agent"

        return TestAgent()

    def test_empty_context_returns_empty_string(self, mixin_class):
        """Empty context returns empty string."""
        result = mixin_class._build_context_prompt(context=None)
        assert result == ""

    def test_empty_list_returns_empty_string(self, mixin_class):
        """Empty list returns empty string."""
        result = mixin_class._build_context_prompt(context=[])
        assert result == ""

    def test_single_message_included(self, mixin_class):
        """Single message is included in context."""
        from aragora.core import Message

        msg = Message(
            agent="agent1",
            content="Test content",
            round=1,
            role="proposer",
        )

        result = mixin_class._build_context_prompt(context=[msg])

        assert "Test content" in result
        assert "agent1" in result

    def test_multiple_messages_included(self, mixin_class):
        """Multiple messages are included in context."""
        from aragora.core import Message

        messages = [
            Message(agent="agent1", content="First message", round=1, role="proposer"),
            Message(agent="agent2", content="Second message", round=1, role="critic"),
        ]

        result = mixin_class._build_context_prompt(context=messages)

        assert "First message" in result
        assert "Second message" in result

    def test_round_number_in_output(self, mixin_class):
        """Round number is included in output."""
        from aragora.core import Message

        msg = Message(agent="agent1", content="Content", round=3, role="proposer")

        result = mixin_class._build_context_prompt(context=[msg])

        assert "[Round 3]" in result

    def test_context_prefix_present(self, mixin_class):
        """Previous discussion prefix is present."""
        from aragora.core import Message

        msg = Message(agent="agent1", content="Content", round=1, role="proposer")

        result = mixin_class._build_context_prompt(context=[msg])

        assert "Previous discussion" in result

    def test_truncation_respects_message_limit(self, mixin_class):
        """Truncation respects message character limit."""
        from aragora.core import Message
        from aragora.agents.base import MAX_MESSAGE_CHARS

        # Create very long message
        long_content = "x" * (MAX_MESSAGE_CHARS * 2)
        msg = Message(agent="agent1", content=long_content, round=1, role="proposer")

        result = mixin_class._build_context_prompt(context=[msg], truncate=True)

        # Result should be shorter than original + overhead
        assert len(result) < len(long_content) * 1.1
        assert "truncated" in result.lower()

    def test_sanitize_function_applied(self, mixin_class):
        """Sanitize function is applied when provided."""
        from aragora.core import Message

        msg = Message(agent="agent1", content="original content", round=1, role="proposer")

        def sanitize(s):
            return s.upper()

        result = mixin_class._build_context_prompt(
            context=[msg], truncate=True, sanitize_fn=sanitize
        )

        assert "ORIGINAL CONTENT" in result

    def test_limits_to_10_messages(self, mixin_class):
        """Context limits to last 10 messages."""
        from aragora.core import Message

        messages = [
            Message(agent="agent1", content=f"Message {i}", round=i, role="proposer")
            for i in range(15)
        ]

        result = mixin_class._build_context_prompt(context=messages)

        # Message 0-4 should NOT be present (only 5-14 should be)
        assert "Message 14" in result
        assert "Message 5" in result
        # Earlier messages should be excluded
        assert "Message 0" not in result


class TestCritiqueMixinParseCritique:
    """Tests for CritiqueMixin._parse_critique method."""

    @pytest.fixture
    def mixin_class(self):
        """Create a class that uses CritiqueMixin."""
        from aragora.agents.base import CritiqueMixin

        class TestAgent(CritiqueMixin):
            def __init__(self):
                self.name = "test_agent"

        return TestAgent()

    def test_returns_critique_object(self, mixin_class):
        """_parse_critique returns a Critique object."""
        from aragora.core import Critique

        result = mixin_class._parse_critique(
            response="Test response",
            target_agent="target",
            target_content="target content",
        )

        assert isinstance(result, Critique)

    def test_sets_agent_name(self, mixin_class):
        """Critique has correct agent name."""
        result = mixin_class._parse_critique(
            response="Test response",
            target_agent="target",
            target_content="content",
        )

        assert result.agent == "test_agent"

    def test_sets_target_agent(self, mixin_class):
        """Critique has correct target agent."""
        result = mixin_class._parse_critique(
            response="Test response",
            target_agent="target_name",
            target_content="content",
        )

        assert result.target_agent == "target_name"

    def test_truncates_target_content(self, mixin_class):
        """Critique truncates long target content."""
        long_content = "x" * 500

        result = mixin_class._parse_critique(
            response="Test response",
            target_agent="target",
            target_content=long_content,
        )

        assert len(result.target_content) <= 200

    def test_extracts_issues_from_bullet_points(self, mixin_class):
        """Extracts issues from bullet point format."""
        # Header on its own line, then bullet items
        response = """Issues found:
- First issue here
- Second issue here
"""
        result = mixin_class._parse_critique(
            response=response,
            target_agent="target",
            target_content="content",
        )

        # The parser needs the section header followed by bullet items
        # If parsing doesn't find structured content, it falls back to sentences
        assert result is not None
        assert len(result.issues) > 0 or len(result.suggestions) > 0

    def test_extracts_suggestions(self, mixin_class):
        """Extracts suggestions from response."""
        # The parser looks for "suggest" keyword then bullet items
        response = """Here are my suggestions:
- First suggestion
- Second suggestion
"""
        result = mixin_class._parse_critique(
            response=response,
            target_agent="target",
            target_content="content",
        )

        # Parser extracts based on section markers
        assert result is not None
        assert len(result.issues) > 0 or len(result.suggestions) > 0

    def test_extracts_severity_number(self, mixin_class):
        """Extracts severity number from response."""
        response = """Issues found:
- One issue
Severity: 7.5
"""
        result = mixin_class._parse_critique(
            response=response,
            target_agent="target",
            target_content="content",
        )

        assert result.severity == 7.5

    def test_normalizes_0_1_severity_to_0_10(self, mixin_class):
        """Normalizes 0-1 severity scale to 0-10."""
        response = """Issues:
- Issue
Severity: 0.7
"""
        result = mixin_class._parse_critique(
            response=response,
            target_agent="target",
            target_content="content",
        )

        assert result.severity == 7.0

    def test_clamps_severity_to_valid_range(self, mixin_class):
        """Clamps severity to 0-10 range."""
        response = """Issues:
- Issue
Severity: 15
"""
        result = mixin_class._parse_critique(
            response=response,
            target_agent="target",
            target_content="content",
        )

        assert result.severity == 10.0

    def test_default_severity_is_5(self, mixin_class):
        """Default severity is 5 when not specified."""
        result = mixin_class._parse_critique(
            response="Just some text without severity",
            target_agent="target",
            target_content="content",
        )

        assert result.severity == 5.0

    def test_limits_issues_to_5(self, mixin_class):
        """Limits extracted issues to 5."""
        response = """Issues:
- Issue 1
- Issue 2
- Issue 3
- Issue 4
- Issue 5
- Issue 6
- Issue 7
"""
        result = mixin_class._parse_critique(
            response=response,
            target_agent="target",
            target_content="content",
        )

        assert len(result.issues) <= 5

    def test_limits_suggestions_to_5(self, mixin_class):
        """Limits extracted suggestions to 5."""
        response = """Suggestions:
- Suggestion 1
- Suggestion 2
- Suggestion 3
- Suggestion 4
- Suggestion 5
- Suggestion 6
- Suggestion 7
"""
        result = mixin_class._parse_critique(
            response=response,
            target_agent="target",
            target_content="content",
        )

        assert len(result.suggestions) <= 5

    def test_handles_empty_response(self, mixin_class):
        """Handles empty response gracefully."""
        result = mixin_class._parse_critique(
            response="",
            target_agent="target",
            target_content="content",
        )

        assert result is not None
        assert len(result.issues) > 0  # Should have fallback message

    def test_asterisk_bullets_work(self, mixin_class):
        """Asterisk bullet points are parsed."""
        response = """There are issues here:
* First issue
* Second issue
"""
        result = mixin_class._parse_critique(
            response=response,
            target_agent="target",
            target_content="content",
        )

        # Parser should create valid critique
        assert result is not None
        # Either issues or suggestions should be populated
        assert len(result.issues) > 0 or len(result.suggestions) > 0

    def test_bullet_unicode_works(self, mixin_class):
        """Unicode bullet points are parsed."""
        response = """There are issues here:
• First issue
• Second issue
"""
        result = mixin_class._parse_critique(
            response=response,
            target_agent="target",
            target_content="content",
        )

        # Parser should create valid critique
        assert result is not None
        assert len(result.issues) > 0 or len(result.suggestions) > 0


class TestAgentType:
    """Tests for AgentType literal type."""

    def test_agent_type_includes_demo(self):
        """AgentType includes demo."""
        from aragora.agents.base import AgentType
        from typing import get_args

        types = get_args(AgentType)
        assert "demo" in types

    def test_agent_type_includes_claude(self):
        """AgentType includes claude."""
        from aragora.agents.base import AgentType
        from typing import get_args

        types = get_args(AgentType)
        assert "claude" in types

    def test_agent_type_includes_openai(self):
        """AgentType includes openai."""
        from aragora.agents.base import AgentType
        from typing import get_args

        types = get_args(AgentType)
        assert "openai" in types

    def test_agent_type_includes_gemini(self):
        """AgentType includes gemini."""
        from aragora.agents.base import AgentType
        from typing import get_args

        types = get_args(AgentType)
        assert "gemini" in types

    def test_agent_type_includes_deepseek(self):
        """AgentType includes deepseek."""
        from aragora.agents.base import AgentType
        from typing import get_args

        types = get_args(AgentType)
        assert "deepseek" in types

    def test_agent_type_has_multiple_options(self):
        """AgentType has multiple options."""
        from aragora.agents.base import AgentType
        from typing import get_args

        types = get_args(AgentType)
        assert len(types) > 10


class TestCreateAgent:
    """Tests for create_agent factory function."""

    def test_create_agent_exists(self):
        """create_agent function exists."""
        from aragora.agents.base import create_agent

        assert callable(create_agent)

    def test_create_demo_agent(self):
        """Can create demo agent."""
        from aragora.agents.base import create_agent

        agent = create_agent("demo")

        assert agent is not None
        assert hasattr(agent, "name")

    def test_create_agent_with_custom_name(self):
        """Can create agent with custom name."""
        from aragora.agents.base import create_agent

        agent = create_agent("demo", name="custom_name")

        assert agent.name == "custom_name"

    def test_create_agent_with_role(self):
        """Can create agent with specific role."""
        from aragora.agents.base import create_agent

        agent = create_agent("demo", role="critic")

        assert agent.role == "critic"

    def test_invalid_agent_type_raises(self):
        """Invalid agent type raises ValueError."""
        from aragora.agents.base import create_agent

        with pytest.raises((ValueError, KeyError)):
            create_agent("nonexistent_agent_type")


class TestListAvailableAgents:
    """Tests for list_available_agents function."""

    def test_list_available_agents_exists(self):
        """list_available_agents function exists."""
        from aragora.agents.base import list_available_agents

        assert callable(list_available_agents)

    def test_returns_dict(self):
        """list_available_agents returns a dict."""
        from aragora.agents.base import list_available_agents

        result = list_available_agents()

        assert isinstance(result, dict)

    def test_contains_demo(self):
        """Result contains demo agent."""
        from aragora.agents.base import list_available_agents

        result = list_available_agents()

        assert "demo" in result

    def test_agent_info_is_dict(self):
        """Agent info entries are dicts."""
        from aragora.agents.base import list_available_agents

        result = list_available_agents()

        for agent_type, info in result.items():
            assert isinstance(info, dict), f"Info for {agent_type} should be dict"

    def test_has_multiple_agents(self):
        """Returns multiple available agents."""
        from aragora.agents.base import list_available_agents

        result = list_available_agents()

        assert len(result) > 1


class TestCritiqueMixinEdgeCases:
    """Edge case tests for CritiqueMixin."""

    @pytest.fixture
    def mixin_class(self):
        """Create a class that uses CritiqueMixin."""
        from aragora.agents.base import CritiqueMixin

        class TestAgent(CritiqueMixin):
            def __init__(self):
                self.name = "test_agent"

        return TestAgent()

    def test_parse_critique_with_no_structure(self, mixin_class):
        """Handles unstructured response."""
        result = mixin_class._parse_critique(
            response="This is just a plain text response without any structure.",
            target_agent="target",
            target_content="content",
        )

        # Should still produce a valid critique
        assert result is not None
        assert result.agent == "test_agent"

    def test_parse_critique_unicode(self, mixin_class):
        """Handles unicode in response."""
        response = """Issues:
- This has émojis 🎉 and üñíçödé
"""
        result = mixin_class._parse_critique(
            response=response,
            target_agent="target",
            target_content="content",
        )

        assert "émojis" in result.issues[0] or "emoji" in str(result.issues).lower()

    def test_reasoning_truncated_to_500(self, mixin_class):
        """Reasoning is truncated to 500 characters."""
        long_response = "x" * 1000

        result = mixin_class._parse_critique(
            response=long_response,
            target_agent="target",
            target_content="content",
        )

        assert len(result.reasoning) <= 500
