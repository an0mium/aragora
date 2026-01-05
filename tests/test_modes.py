"""
Tests for the Mode System.

Tests cover:
- ToolGroup permissions and combinations
- Mode base class functionality
- ModeRegistry operations
- Built-in modes (Architect, Coder, Reviewer, Debugger, Orchestrator)
- Handoff context and mode transitions
- Custom mode loading
"""

import pytest
from datetime import datetime

from aragora.modes.tool_groups import (
    ToolGroup,
    get_required_group,
    can_use_tool,
    TOOL_GROUP_MAP,
)
from aragora.modes.base import Mode, ModeRegistry
from aragora.modes.handoff import HandoffContext, ModeHandoff


# ============================================================================
# ToolGroup Tests
# ============================================================================

class TestToolGroup:
    """Tests for ToolGroup enum and permissions."""

    def test_basic_flags(self):
        """Test basic ToolGroup flags exist."""
        assert ToolGroup.NONE.value == 0
        assert ToolGroup.READ
        assert ToolGroup.EDIT
        assert ToolGroup.COMMAND
        assert ToolGroup.BROWSER
        assert ToolGroup.MCP
        assert ToolGroup.DEBATE

    def test_flag_combination(self):
        """Test combining ToolGroup flags."""
        combined = ToolGroup.READ | ToolGroup.EDIT

        assert ToolGroup.READ in combined
        assert ToolGroup.EDIT in combined
        assert ToolGroup.COMMAND not in combined

    def test_developer_composite(self):
        """Test DEVELOPER composite group."""
        developer = ToolGroup.DEVELOPER()

        assert ToolGroup.READ in developer
        assert ToolGroup.EDIT in developer
        assert ToolGroup.COMMAND in developer
        assert ToolGroup.BROWSER not in developer

    def test_readonly_composite(self):
        """Test READONLY composite group."""
        readonly = ToolGroup.READONLY()

        assert ToolGroup.READ in readonly
        assert ToolGroup.BROWSER in readonly
        assert ToolGroup.EDIT not in readonly
        assert ToolGroup.COMMAND not in readonly

    def test_full_composite(self):
        """Test FULL composite group includes all."""
        full = ToolGroup.FULL()

        assert ToolGroup.READ in full
        assert ToolGroup.EDIT in full
        assert ToolGroup.COMMAND in full
        assert ToolGroup.BROWSER in full
        assert ToolGroup.MCP in full
        assert ToolGroup.DEBATE in full


class TestGetRequiredGroup:
    """Tests for get_required_group function."""

    def test_read_tools(self):
        """Test read tools require READ group."""
        assert get_required_group("read") == ToolGroup.READ
        assert get_required_group("glob") == ToolGroup.READ
        assert get_required_group("grep") == ToolGroup.READ

    def test_edit_tools(self):
        """Test edit tools require EDIT group."""
        assert get_required_group("edit") == ToolGroup.EDIT
        assert get_required_group("write") == ToolGroup.EDIT
        assert get_required_group("notebook_edit") == ToolGroup.EDIT

    def test_command_tools(self):
        """Test command tools require COMMAND group."""
        assert get_required_group("bash") == ToolGroup.COMMAND
        assert get_required_group("kill_shell") == ToolGroup.COMMAND

    def test_browser_tools(self):
        """Test browser tools require BROWSER group."""
        assert get_required_group("web_fetch") == ToolGroup.BROWSER
        assert get_required_group("web_search") == ToolGroup.BROWSER

    def test_debate_tools(self):
        """Test debate tools require DEBATE group."""
        assert get_required_group("debate") == ToolGroup.DEBATE
        assert get_required_group("arena") == ToolGroup.DEBATE

    def test_unknown_tool_returns_none(self):
        """Test unknown tools return NONE (allowed by default)."""
        assert get_required_group("unknown_tool") == ToolGroup.NONE
        assert get_required_group("custom_tool") == ToolGroup.NONE

    def test_case_insensitive(self):
        """Test tool names are case-insensitive."""
        assert get_required_group("READ") == ToolGroup.READ
        assert get_required_group("Read") == ToolGroup.READ
        assert get_required_group("BASH") == ToolGroup.COMMAND

    def test_hyphen_underscore_normalization(self):
        """Test hyphens are normalized to underscores."""
        assert get_required_group("web-fetch") == ToolGroup.BROWSER
        assert get_required_group("kill-shell") == ToolGroup.COMMAND


class TestCanUseTool:
    """Tests for can_use_tool function."""

    def test_allowed_tool(self):
        """Test allowed tool returns True."""
        assert can_use_tool(ToolGroup.READ, "read") is True
        assert can_use_tool(ToolGroup.EDIT, "edit") is True
        assert can_use_tool(ToolGroup.DEVELOPER(), "bash") is True

    def test_disallowed_tool(self):
        """Test disallowed tool returns False."""
        assert can_use_tool(ToolGroup.READ, "edit") is False
        assert can_use_tool(ToolGroup.EDIT, "bash") is False
        assert can_use_tool(ToolGroup.READONLY(), "edit") is False

    def test_unknown_tool_allowed(self):
        """Test unknown tools are allowed by default."""
        assert can_use_tool(ToolGroup.READ, "unknown") is True
        assert can_use_tool(ToolGroup.NONE, "custom_tool") is True

    def test_combined_groups(self):
        """Test combined groups allow multiple tools."""
        combined = ToolGroup.READ | ToolGroup.EDIT

        assert can_use_tool(combined, "read") is True
        assert can_use_tool(combined, "edit") is True
        assert can_use_tool(combined, "bash") is False


# ============================================================================
# Mode Base Class Tests
# ============================================================================

class ConcreteMode(Mode):
    """Concrete implementation for testing that auto-registers."""

    def get_system_prompt(self) -> str:
        return f"You are in {self.name} mode. {self.system_prompt_additions}"


class NonRegisteringMode(Mode):
    """Concrete implementation that does NOT auto-register."""

    _auto_register = False

    def get_system_prompt(self) -> str:
        return f"You are in {self.name} mode. {self.system_prompt_additions}"


class TestModeBase:
    """Tests for Mode base class."""

    def setup_method(self):
        """Clear registry before each test."""
        ModeRegistry.clear()

    def test_mode_creation(self):
        """Test basic mode creation."""
        mode = NonRegisteringMode(
            name="test",
            description="Test mode",
            tool_groups=ToolGroup.READ,
        )

        assert mode.name == "test"
        assert mode.description == "Test mode"
        assert mode.tool_groups == ToolGroup.READ

    def test_mode_auto_registers(self):
        """Test mode auto-registers on creation."""
        mode = ConcreteMode(
            name="auto_test",
            description="Auto-register test",
            tool_groups=ToolGroup.READ,
        )

        assert ModeRegistry.get("auto_test") is mode

    def test_can_access_tool(self):
        """Test can_access_tool method."""
        mode = NonRegisteringMode(
            name="reader",
            description="Read-only mode",
            tool_groups=ToolGroup.READ,
        )

        assert mode.can_access_tool("read") is True
        assert mode.can_access_tool("glob") is True
        assert mode.can_access_tool("edit") is False
        assert mode.can_access_tool("bash") is False

    def test_can_access_file_no_patterns(self):
        """Test can_access_file with no patterns allows all."""
        mode = NonRegisteringMode(
            name="no_patterns",
            description="No file patterns",
            tool_groups=ToolGroup.READ,
        )

        assert mode.can_access_file("any/file.py") is True
        assert mode.can_access_file("another/path.txt") is True

    def test_can_access_file_with_patterns(self):
        """Test can_access_file with specific patterns."""
        mode = NonRegisteringMode(
            name="restricted",
            description="Restricted patterns",
            tool_groups=ToolGroup.READ,
            file_patterns=["*.py", "tests/*"],
        )

        assert mode.can_access_file("script.py") is True
        assert mode.can_access_file("tests/test_example.py") is True
        assert mode.can_access_file("config.json") is False
        assert mode.can_access_file("src/main.js") is False

    def test_get_system_prompt(self):
        """Test get_system_prompt method."""
        mode = NonRegisteringMode(
            name="prompter",
            description="Test prompt",
            tool_groups=ToolGroup.READ,
            system_prompt_additions="Be helpful.",
        )

        prompt = mode.get_system_prompt()

        assert "prompter" in prompt
        assert "Be helpful" in prompt

    def test_get_restrictions_summary(self):
        """Test get_restrictions_summary method."""
        mode = NonRegisteringMode(
            name="summary_test",
            description="Test summary generation",
            tool_groups=ToolGroup.READ | ToolGroup.EDIT,
            file_patterns=["*.py"],
        )

        summary = mode.get_restrictions_summary()

        assert "summary_test" in summary
        assert "Test summary generation" in summary
        assert "Read files" in summary
        assert "Edit files" in summary
        assert "*.py" in summary


# ============================================================================
# ModeRegistry Tests
# ============================================================================

class TestModeRegistry:
    """Tests for ModeRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        ModeRegistry.clear()

    def test_register_and_get(self):
        """Test registering and getting a mode."""
        mode = NonRegisteringMode(
            name="test_mode",
            description="Test",
            tool_groups=ToolGroup.READ,
        )
        ModeRegistry.register(mode)

        retrieved = ModeRegistry.get("test_mode")
        assert retrieved is mode

    def test_get_case_insensitive(self):
        """Test get is case-insensitive."""
        mode = NonRegisteringMode(
            name="CaseSensitive",
            description="Test",
            tool_groups=ToolGroup.READ,
        )
        ModeRegistry.register(mode)

        assert ModeRegistry.get("casesensitive") is mode
        assert ModeRegistry.get("CASESENSITIVE") is mode
        assert ModeRegistry.get("CaseSensitive") is mode

    def test_get_nonexistent_returns_none(self):
        """Test get returns None for nonexistent mode."""
        assert ModeRegistry.get("nonexistent") is None

    def test_get_or_raise_success(self):
        """Test get_or_raise returns mode when found."""
        mode = ConcreteMode(
            name="exists",
            description="Test",
            tool_groups=ToolGroup.READ,
        )

        retrieved = ModeRegistry.get_or_raise("exists")
        assert retrieved is mode

    def test_get_or_raise_error(self):
        """Test get_or_raise raises KeyError when not found."""
        with pytest.raises(KeyError) as exc_info:
            ModeRegistry.get_or_raise("nonexistent")

        assert "nonexistent" in str(exc_info.value)

    def test_unregister(self):
        """Test unregistering a mode."""
        ConcreteMode(
            name="to_remove",
            description="Test",
            tool_groups=ToolGroup.READ,
        )

        assert ModeRegistry.get("to_remove") is not None

        result = ModeRegistry.unregister("to_remove")

        assert result is True
        assert ModeRegistry.get("to_remove") is None

    def test_unregister_nonexistent_returns_false(self):
        """Test unregistering nonexistent mode returns False."""
        result = ModeRegistry.unregister("nonexistent")
        assert result is False

    def test_list_all(self):
        """Test listing all registered modes."""
        ConcreteMode(name="mode1", description="", tool_groups=ToolGroup.READ)
        ConcreteMode(name="mode2", description="", tool_groups=ToolGroup.EDIT)
        ConcreteMode(name="mode3", description="", tool_groups=ToolGroup.COMMAND)

        names = ModeRegistry.list_all()

        assert "mode1" in names
        assert "mode2" in names
        assert "mode3" in names
        assert len(names) == 3

    def test_get_all(self):
        """Test getting all registered modes."""
        m1 = ConcreteMode(name="mode_a", description="", tool_groups=ToolGroup.READ)
        m2 = ConcreteMode(name="mode_b", description="", tool_groups=ToolGroup.EDIT)

        modes = ModeRegistry.get_all()

        assert m1 in modes
        assert m2 in modes
        assert len(modes) == 2

    def test_clear(self):
        """Test clearing registry."""
        ConcreteMode(name="to_clear1", description="", tool_groups=ToolGroup.READ)
        ConcreteMode(name="to_clear2", description="", tool_groups=ToolGroup.READ)

        assert len(ModeRegistry.list_all()) == 2

        ModeRegistry.clear()

        assert len(ModeRegistry.list_all()) == 0


# ============================================================================
# Built-in Modes Tests
# ============================================================================

class TestBuiltinModes:
    """Tests for built-in operational modes."""

    def setup_method(self):
        """Clear and re-register builtins before each test."""
        ModeRegistry.clear()
        from aragora.modes.builtin import register_all_builtins
        register_all_builtins()

    def test_architect_mode_exists(self):
        """Test Architect mode is registered."""
        mode = ModeRegistry.get("architect")

        assert mode is not None
        assert ToolGroup.READ in mode.tool_groups
        assert ToolGroup.BROWSER in mode.tool_groups
        # Architect should NOT have edit
        assert ToolGroup.EDIT not in mode.tool_groups

    def test_coder_mode_exists(self):
        """Test Coder mode is registered."""
        mode = ModeRegistry.get("coder")

        assert mode is not None
        assert ToolGroup.READ in mode.tool_groups
        assert ToolGroup.EDIT in mode.tool_groups
        assert ToolGroup.COMMAND in mode.tool_groups

    def test_reviewer_mode_exists(self):
        """Test Reviewer mode is registered."""
        mode = ModeRegistry.get("reviewer")

        assert mode is not None
        assert ToolGroup.READ in mode.tool_groups
        # Reviewer should NOT have edit by default
        assert ToolGroup.EDIT not in mode.tool_groups

    def test_debugger_mode_exists(self):
        """Test Debugger mode is registered."""
        mode = ModeRegistry.get("debugger")

        assert mode is not None
        assert ToolGroup.READ in mode.tool_groups
        assert ToolGroup.COMMAND in mode.tool_groups

    def test_orchestrator_mode_exists(self):
        """Test Orchestrator mode is registered."""
        mode = ModeRegistry.get("orchestrator")

        assert mode is not None
        # Orchestrator typically has full access
        assert ToolGroup.READ in mode.tool_groups

    def test_all_builtins_have_system_prompts(self):
        """Test all built-in modes have system prompts."""
        builtin_names = ["architect", "coder", "reviewer", "debugger", "orchestrator"]

        for name in builtin_names:
            mode = ModeRegistry.get(name)
            assert mode is not None, f"Mode {name} not found"

            prompt = mode.get_system_prompt()
            assert prompt, f"Mode {name} has empty system prompt"
            assert len(prompt) > 10, f"Mode {name} has very short system prompt"


# ============================================================================
# Handoff Context Tests
# ============================================================================

class TestHandoffContext:
    """Tests for HandoffContext."""

    def test_basic_creation(self):
        """Test basic handoff context creation."""
        ctx = HandoffContext(
            from_mode="architect",
            to_mode="coder",
            task_summary="Design is complete, ready to implement.",
        )

        assert ctx.from_mode == "architect"
        assert ctx.to_mode == "coder"
        assert "Design is complete" in ctx.task_summary
        assert ctx.key_findings == []
        assert ctx.files_touched == []
        assert isinstance(ctx.timestamp, datetime)

    def test_with_all_fields(self):
        """Test handoff context with all fields populated."""
        ctx = HandoffContext(
            from_mode="coder",
            to_mode="reviewer",
            task_summary="Implementation complete.",
            key_findings=["Added new API endpoint", "Updated tests"],
            files_touched=["src/api.py", "tests/test_api.py"],
            open_questions=["Should we add rate limiting?"],
            artifacts={"coverage": 85.5},
        )

        assert len(ctx.key_findings) == 2
        assert len(ctx.files_touched) == 2
        assert len(ctx.open_questions) == 1
        assert ctx.artifacts["coverage"] == 85.5

    def test_to_prompt(self):
        """Test generating prompt from context."""
        ctx = HandoffContext(
            from_mode="architect",
            to_mode="coder",
            task_summary="Design the user authentication system.",
            key_findings=["Use JWT tokens", "Add refresh token support"],
            files_touched=["docs/auth_design.md"],
            open_questions=["Which library for JWT?"],
        )

        prompt = ctx.to_prompt()

        assert "Handoff from architect to coder" in prompt
        assert "Design the user authentication" in prompt
        assert "JWT tokens" in prompt
        assert "docs/auth_design.md" in prompt
        assert "Which library for JWT?" in prompt

    def test_to_prompt_minimal(self):
        """Test to_prompt with minimal context."""
        ctx = HandoffContext(
            from_mode="a",
            to_mode="b",
            task_summary="Quick task.",
        )

        prompt = ctx.to_prompt()

        assert "Handoff from a to b" in prompt
        assert "Quick task" in prompt


# ============================================================================
# ModeHandoff Tests
# ============================================================================

class TestModeHandoff:
    """Tests for ModeHandoff manager."""

    def test_create_context(self):
        """Test creating handoff context."""
        handoff = ModeHandoff()

        ctx = handoff.create_context(
            from_mode="architect",
            to_mode="coder",
            task_summary="Ready to implement.",
        )

        assert ctx.from_mode == "architect"
        assert ctx.to_mode == "coder"
        assert len(handoff.history) == 1

    def test_create_context_with_details(self):
        """Test creating context with all details."""
        handoff = ModeHandoff()

        ctx = handoff.create_context(
            from_mode="coder",
            to_mode="reviewer",
            task_summary="Implementation done.",
            key_findings=["Added API"],
            files_touched=["api.py"],
            open_questions=["Performance?"],
            artifacts={"lines_changed": 150},
        )

        assert ctx.key_findings == ["Added API"]
        assert ctx.files_touched == ["api.py"]
        assert ctx.artifacts["lines_changed"] == 150

    def test_generate_transition_prompt(self):
        """Test generating full transition prompt."""
        handoff = ModeHandoff()

        ctx = handoff.create_context(
            from_mode="architect",
            to_mode="coder",
            task_summary="Design complete.",
        )

        prompt = handoff.generate_transition_prompt(
            ctx,
            "You are a coder. Write clean code.",
        )

        assert "You are a coder" in prompt
        assert "Handoff from architect to coder" in prompt
        assert "Design complete" in prompt
        assert "Continue from where the previous mode left off" in prompt

    def test_get_history(self):
        """Test getting handoff history."""
        handoff = ModeHandoff()

        handoff.create_context("a", "b", "First transition")
        handoff.create_context("b", "c", "Second transition")
        handoff.create_context("c", "d", "Third transition")

        history = handoff.get_history()

        assert len(history) == 3
        assert history[0].from_mode == "a"
        assert history[2].to_mode == "d"

    def test_history_is_copy(self):
        """Test that get_history returns a copy."""
        handoff = ModeHandoff()
        handoff.create_context("a", "b", "Test")

        history = handoff.get_history()
        history.clear()

        assert len(handoff.history) == 1  # Original unaffected

    def test_summarize_session_empty(self):
        """Test summarizing empty session."""
        handoff = ModeHandoff()

        summary = handoff.summarize_session()

        assert "No mode transitions" in summary

    def test_summarize_session_with_history(self):
        """Test summarizing session with transitions."""
        handoff = ModeHandoff()

        handoff.create_context("architect", "coder", "Design complete, implementing feature X.")
        handoff.create_context("coder", "reviewer", "Implementation done, ready for review.")

        summary = handoff.summarize_session()

        assert "Session Mode History" in summary
        assert "architect -> coder" in summary
        assert "coder -> reviewer" in summary
        assert "Design complete" in summary


# ============================================================================
# Integration Tests
# ============================================================================

class TestModeSystemIntegration:
    """Integration tests for the mode system."""

    def setup_method(self):
        """Set up clean registry."""
        ModeRegistry.clear()
        from aragora.modes.builtin import register_all_builtins
        register_all_builtins()

    def test_full_workflow_transitions(self):
        """Test a full workflow with mode transitions."""
        handoff = ModeHandoff()

        # Architect designs
        architect = ModeRegistry.get("architect")
        assert architect is not None
        assert architect.can_access_tool("read") is True
        assert architect.can_access_tool("edit") is False

        ctx1 = handoff.create_context(
            from_mode="architect",
            to_mode="coder",
            task_summary="Designed new authentication module.",
            key_findings=["Use OAuth2", "Add refresh tokens"],
            files_touched=["docs/design.md"],
        )

        # Coder implements
        coder = ModeRegistry.get("coder")
        assert coder is not None
        assert coder.can_access_tool("read") is True
        assert coder.can_access_tool("edit") is True
        assert coder.can_access_tool("bash") is True

        ctx2 = handoff.create_context(
            from_mode="coder",
            to_mode="reviewer",
            task_summary="Implemented OAuth2 authentication.",
            key_findings=["Added auth module", "100% test coverage"],
            files_touched=["src/auth.py", "tests/test_auth.py"],
        )

        # Reviewer reviews
        reviewer = ModeRegistry.get("reviewer")
        assert reviewer is not None
        assert reviewer.can_access_tool("read") is True
        assert reviewer.can_access_tool("edit") is False

        # Check history
        assert len(handoff.history) == 2

        summary = handoff.summarize_session()
        assert "architect -> coder" in summary
        assert "coder -> reviewer" in summary
