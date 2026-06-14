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


# ============================================================================
# CustomMode Tests
# ============================================================================

from aragora.modes.custom import CustomMode, CustomModeLoader


class TestCustomMode:
    """Tests for CustomMode class."""

    def setup_method(self):
        """Clear registry before each test."""
        ModeRegistry.clear()
        from aragora.modes.builtin import register_all_builtins

        register_all_builtins()

    def test_basic_creation(self):
        """Test basic CustomMode creation."""
        mode = CustomMode(
            name="security-auditor",
            description="Security-focused code auditor",
        )

        assert mode.name == "security-auditor"
        assert mode.description == "Security-focused code auditor"
        assert mode.tool_groups == ToolGroup.READ  # Default
        assert mode.file_patterns == []
        assert mode.base_mode is None

    def test_creation_with_tool_groups(self):
        """Test CustomMode with specific tool groups."""
        mode = CustomMode(
            name="developer",
            description="Full developer access",
            tool_groups=ToolGroup.READ | ToolGroup.EDIT | ToolGroup.COMMAND,
        )

        assert ToolGroup.READ in mode.tool_groups
        assert ToolGroup.EDIT in mode.tool_groups
        assert ToolGroup.COMMAND in mode.tool_groups
        assert ToolGroup.BROWSER not in mode.tool_groups

    def test_creation_with_file_patterns(self):
        """Test CustomMode with file patterns."""
        mode = CustomMode(
            name="python-only",
            description="Python files only",
            file_patterns=["**/*.py", "pyproject.toml"],
        )

        assert len(mode.file_patterns) == 2
        assert "**/*.py" in mode.file_patterns

    def test_get_system_prompt_no_base(self):
        """Test system prompt generation without base mode."""
        mode = CustomMode(
            name="test-mode",
            description="A test mode for testing",
            system_prompt_additions="Focus on unit tests.",
        )

        prompt = mode.get_system_prompt()

        assert "Custom Mode: test-mode" in prompt
        assert "A test mode for testing" in prompt
        assert "Focus on unit tests." in prompt

    def test_get_system_prompt_with_base_mode(self):
        """Test system prompt inherits from base mode."""
        mode = CustomMode(
            name="enhanced-reviewer",
            description="Enhanced code reviewer",
            base_mode="reviewer",
            system_prompt_additions="Also check for security issues.",
        )

        prompt = mode.get_system_prompt()

        # Should include reviewer's system prompt
        assert "Custom Mode: enhanced-reviewer" in prompt
        assert "Also check for security issues." in prompt
        # The separator between base and custom
        assert "---" in prompt

    def test_get_system_prompt_with_invalid_base(self):
        """Test system prompt with non-existent base mode."""
        ModeRegistry.clear()  # Clear so base doesn't exist

        mode = CustomMode(
            name="orphan-mode",
            description="Mode with invalid base",
            base_mode="nonexistent",
        )

        prompt = mode.get_system_prompt()

        # Should still work, just without base prompt
        assert "Custom Mode: orphan-mode" in prompt
        assert "Mode with invalid base" in prompt


class TestCustomModeLoader:
    """Tests for CustomModeLoader class."""

    def setup_method(self):
        """Clear registry before each test."""
        ModeRegistry.clear()

    def test_default_search_paths(self):
        """Test default search paths are set."""
        loader = CustomModeLoader()

        assert ".aragora/modes" in loader.search_paths
        assert any("config/aragora/modes" in p for p in loader.search_paths)

    def test_custom_search_paths(self):
        """Test custom search paths override defaults."""
        loader = CustomModeLoader(search_paths=["/custom/path"])

        assert loader.search_paths == ["/custom/path"]
        assert ".aragora/modes" not in loader.search_paths

    def test_tool_group_map_has_all_groups(self):
        """Test TOOL_GROUP_MAP has expected entries."""
        assert "read" in CustomModeLoader.TOOL_GROUP_MAP
        assert "edit" in CustomModeLoader.TOOL_GROUP_MAP
        assert "command" in CustomModeLoader.TOOL_GROUP_MAP
        assert "browser" in CustomModeLoader.TOOL_GROUP_MAP
        assert "mcp" in CustomModeLoader.TOOL_GROUP_MAP
        assert "debate" in CustomModeLoader.TOOL_GROUP_MAP
        # Composite groups
        assert "readonly" in CustomModeLoader.TOOL_GROUP_MAP
        assert "developer" in CustomModeLoader.TOOL_GROUP_MAP
        assert "full" in CustomModeLoader.TOOL_GROUP_MAP

    def test_readonly_composite(self):
        """Test readonly composite group mapping."""
        readonly = CustomModeLoader.TOOL_GROUP_MAP["readonly"]

        assert ToolGroup.READ in readonly
        assert ToolGroup.BROWSER in readonly
        assert ToolGroup.EDIT not in readonly

    def test_developer_composite(self):
        """Test developer composite group mapping."""
        developer = CustomModeLoader.TOOL_GROUP_MAP["developer"]

        assert ToolGroup.READ in developer
        assert ToolGroup.EDIT in developer
        assert ToolGroup.COMMAND in developer
        assert ToolGroup.BROWSER not in developer

    def test_full_composite(self):
        """Test full composite group mapping."""
        full = CustomModeLoader.TOOL_GROUP_MAP["full"]

        assert ToolGroup.READ in full
        assert ToolGroup.EDIT in full
        assert ToolGroup.COMMAND in full
        assert ToolGroup.BROWSER in full
        assert ToolGroup.MCP in full
        assert ToolGroup.DEBATE in full

    def test_parse_config_minimal(self):
        """Test parsing minimal config."""
        loader = CustomModeLoader()

        config = {"name": "minimal"}
        mode = loader._parse_config(config)

        assert mode.name == "minimal"
        assert mode.description == ""
        assert mode.tool_groups == ToolGroup.READ  # Default is ["read"]
        assert mode.file_patterns == []

    def test_parse_config_full(self):
        """Test parsing full config."""
        loader = CustomModeLoader()

        config = {
            "name": "full-mode",
            "description": "A fully configured mode",
            "tool_groups": ["read", "edit", "browser"],
            "file_patterns": ["*.py", "*.js"],
            "system_prompt_additions": "Be thorough.",
            "base_mode": "coder",
        }
        mode = loader._parse_config(config)

        assert mode.name == "full-mode"
        assert mode.description == "A fully configured mode"
        assert ToolGroup.READ in mode.tool_groups
        assert ToolGroup.EDIT in mode.tool_groups
        assert ToolGroup.BROWSER in mode.tool_groups
        assert ToolGroup.COMMAND not in mode.tool_groups
        assert mode.file_patterns == ["*.py", "*.js"]
        assert mode.system_prompt_additions == "Be thorough."
        assert mode.base_mode == "coder"

    def test_parse_config_with_composite_groups(self):
        """Test parsing config with composite tool groups."""
        loader = CustomModeLoader()

        config = {
            "name": "dev-mode",
            "tool_groups": ["developer"],  # Should expand to read+edit+command
        }
        mode = loader._parse_config(config)

        assert ToolGroup.READ in mode.tool_groups
        assert ToolGroup.EDIT in mode.tool_groups
        assert ToolGroup.COMMAND in mode.tool_groups

    def test_parse_config_mixed_groups(self):
        """Test parsing config with mixed basic and composite groups."""
        loader = CustomModeLoader()

        config = {
            "name": "mixed",
            "tool_groups": ["developer", "browser"],  # developer + browser
        }
        mode = loader._parse_config(config)

        assert ToolGroup.READ in mode.tool_groups
        assert ToolGroup.EDIT in mode.tool_groups
        assert ToolGroup.COMMAND in mode.tool_groups
        assert ToolGroup.BROWSER in mode.tool_groups

    def test_parse_config_case_insensitive_groups(self):
        """Test tool group names are case-insensitive."""
        loader = CustomModeLoader()

        config = {
            "name": "case-test",
            "tool_groups": ["READ", "Edit", "BROWSER"],
        }
        mode = loader._parse_config(config)

        assert ToolGroup.READ in mode.tool_groups
        assert ToolGroup.EDIT in mode.tool_groups
        assert ToolGroup.BROWSER in mode.tool_groups

    def test_parse_config_unknown_group_ignored(self):
        """Test unknown tool groups are silently ignored."""
        loader = CustomModeLoader()

        config = {
            "name": "unknown-groups",
            "tool_groups": ["read", "unknown_group", "also_unknown"],
        }
        mode = loader._parse_config(config)

        assert ToolGroup.READ in mode.tool_groups
        # Unknown groups should not cause errors

    def test_load_from_yaml_path_security(self, tmp_path):
        """Test path security check for YAML loading."""
        # Create a loader with specific allowed paths
        loader = CustomModeLoader(search_paths=[str(tmp_path)])

        # Try to load from outside allowed paths
        with pytest.raises(PermissionError) as exc_info:
            loader.load_from_yaml("/etc/passwd")

        assert "outside allowed mode directories" in str(exc_info.value)

    def test_load_from_yaml_success(self, tmp_path):
        """Test successful YAML loading."""
        loader = CustomModeLoader(search_paths=[str(tmp_path)])

        # Create a valid YAML file
        yaml_content = """
name: test-yaml-mode
description: Loaded from YAML
tool_groups:
  - read
  - browser
file_patterns:
  - "*.py"
system_prompt_additions: Be helpful.
"""
        yaml_file = tmp_path / "test_mode.yaml"
        yaml_file.write_text(yaml_content)

        mode = loader.load_from_yaml(yaml_file)

        assert mode.name == "test-yaml-mode"
        assert mode.description == "Loaded from YAML"
        assert ToolGroup.READ in mode.tool_groups
        assert ToolGroup.BROWSER in mode.tool_groups
        assert "*.py" in mode.file_patterns
        assert "Be helpful." in mode.system_prompt_additions

    def test_load_all_from_directory(self, tmp_path):
        """Test loading all modes from a directory."""
        loader = CustomModeLoader(search_paths=[str(tmp_path)])

        # Create multiple YAML files
        (tmp_path / "mode1.yaml").write_text("name: mode1\n")
        (tmp_path / "mode2.yml").write_text("name: mode2\n")
        (tmp_path / "not_a_mode.txt").write_text("ignored")

        modes = loader.load_all(tmp_path)

        names = [m.name for m in modes]
        assert "mode1" in names
        assert "mode2" in names
        assert len(modes) == 2

    def test_load_all_handles_errors(self, tmp_path):
        """Test load_all continues on individual file errors."""
        loader = CustomModeLoader(search_paths=[str(tmp_path)])

        # Create one valid and one invalid YAML (missing required structure)
        (tmp_path / "valid.yaml").write_text("name: valid_mode\n")
        # This will fail during load_from_yaml due to path security check
        # since tmp_path changes between calls. Use a syntax error instead.
        (tmp_path / "invalid.yaml").write_text("{invalid: [unclosed\n")

        modes = loader.load_all(tmp_path)

        # Should load at least the valid one (may also load invalid if it parses)
        names = [m.name for m in modes]
        assert "valid_mode" in names

    def test_load_all_no_directory(self):
        """Test load_all with non-existent directory."""
        loader = CustomModeLoader(search_paths=["/nonexistent/path"])

        modes = loader.load_all()

        assert modes == []

    def test_load_and_register_all(self, tmp_path):
        """Test loading and registering modes."""
        loader = CustomModeLoader(search_paths=[str(tmp_path)])

        # Create modes to register
        (tmp_path / "reg1.yaml").write_text("name: registered1\n")
        (tmp_path / "reg2.yaml").write_text("name: registered2\n")

        count = loader.load_and_register_all(tmp_path)

        assert count == 2
        assert ModeRegistry.get("registered1") is not None
        assert ModeRegistry.get("registered2") is not None


# ============================================================================
# DeepAudit Tests
# ============================================================================

from aragora.modes.deep_audit import (
    DeepAuditConfig,
    AuditFinding,
    DeepAuditVerdict,
    DeepAuditOrchestrator,
    run_deep_audit,
    STRATEGY_AUDIT,
    CONTRACT_AUDIT,
    CODE_ARCHITECTURE_AUDIT,
)
from aragora.debate.roles import CognitiveRole


class TestDeepAuditConfig:
    """Tests for DeepAuditConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DeepAuditConfig()

        assert config.rounds == 6
        assert config.enable_research is True
        assert len(config.roles) == 4
        assert CognitiveRole.ANALYST in config.roles
        assert CognitiveRole.SKEPTIC in config.roles
        assert config.synthesizer_final_round is True
        assert config.cross_examination_depth == 3
        assert config.require_citations is True
        assert config.risk_threshold == 0.7

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DeepAuditConfig(
            rounds=4,
            enable_research=False,
            roles=[CognitiveRole.ANALYST, CognitiveRole.SKEPTIC],
            synthesizer_final_round=False,
            cross_examination_depth=5,
            require_citations=False,
            risk_threshold=0.5,
        )

        assert config.rounds == 4
        assert config.enable_research is False
        assert len(config.roles) == 2
        assert config.synthesizer_final_round is False
        assert config.cross_examination_depth == 5
        assert config.require_citations is False
        assert config.risk_threshold == 0.5


class TestAuditFinding:
    """Tests for AuditFinding dataclass."""

    def test_default_values(self):
        """Test default finding values."""
        finding = AuditFinding(
            category="risk",
            summary="Security vulnerability found",
            details="SQL injection in user input handler",
        )

        assert finding.category == "risk"
        assert finding.summary == "Security vulnerability found"
        assert finding.agents_agree == []
        assert finding.agents_disagree == []
        assert finding.confidence == 0.0
        assert finding.citations == []
        assert finding.severity == 0.0

    def test_full_finding(self):
        """Test finding with all fields."""
        finding = AuditFinding(
            category="unanimous",
            summary="All agree on security fix",
            details="Need to sanitize input",
            agents_agree=["claude", "gemini"],
            agents_disagree=[],
            confidence=0.95,
            citations=["OWASP A03:2021"],
            severity=0.9,
        )

        assert len(finding.agents_agree) == 2
        assert finding.confidence == 0.95
        assert finding.severity == 0.9
        assert "OWASP A03:2021" in finding.citations


class TestDeepAuditVerdict:
    """Tests for DeepAuditVerdict dataclass."""

    def test_default_values(self):
        """Test default verdict values."""
        verdict = DeepAuditVerdict(
            recommendation="Proceed with caution",
            confidence=0.75,
        )

        assert verdict.recommendation == "Proceed with caution"
        assert verdict.confidence == 0.75
        assert verdict.findings == []
        assert verdict.unanimous_issues == []
        assert verdict.split_opinions == []
        assert verdict.risk_areas == []
        assert verdict.citations == []
        assert verdict.cross_examination_notes == ""

    def test_full_verdict(self):
        """Test verdict with all fields populated."""
        verdict = DeepAuditVerdict(
            recommendation="Do not proceed",
            confidence=0.3,
            findings=[AuditFinding("risk", "Security issue", "Details")],
            unanimous_issues=["Input validation missing"],
            split_opinions=["API design"],
            risk_areas=["Authentication bypass"],
            citations=["CVE-2024-1234"],
            cross_examination_notes="Further review needed",
        )

        assert len(verdict.findings) == 1
        assert len(verdict.unanimous_issues) == 1
        assert len(verdict.split_opinions) == 1
        assert len(verdict.risk_areas) == 1

    def test_summary_basic(self):
        """Test basic summary generation."""
        verdict = DeepAuditVerdict(
            recommendation="Approved",
            confidence=0.9,
        )

        summary = verdict.summary()

        assert "DEEP AUDIT VERDICT" in summary
        assert "Approved" in summary
        assert "90%" in summary

    def test_summary_with_issues(self):
        """Test summary with unanimous issues."""
        verdict = DeepAuditVerdict(
            recommendation="Needs work",
            confidence=0.5,
            unanimous_issues=["Missing tests", "No error handling"],
        )

        summary = verdict.summary()

        assert "2 UNANIMOUS ISSUES" in summary
        assert "Missing tests" in summary
        assert "No error handling" in summary

    def test_summary_with_split_opinions(self):
        """Test summary with split opinions."""
        verdict = DeepAuditVerdict(
            recommendation="Review needed",
            confidence=0.6,
            split_opinions=["API versioning approach"],
        )

        summary = verdict.summary()

        assert "1 SPLIT OPINIONS" in summary
        assert "API versioning" in summary

    def test_summary_with_risks(self):
        """Test summary with risk areas."""
        verdict = DeepAuditVerdict(
            recommendation="Proceed with monitoring",
            confidence=0.7,
            risk_areas=["Performance under load", "Memory leaks"],
        )

        summary = verdict.summary()

        assert "2 RISK AREAS" in summary
        assert "Performance under load" in summary

    def test_summary_with_citations(self):
        """Test summary with citations."""
        verdict = DeepAuditVerdict(
            recommendation="Follow best practices",
            confidence=0.8,
            citations=["RFC 7231", "OWASP Guidelines"],
        )

        summary = verdict.summary()

        assert "Citations (2)" in summary
        assert "RFC 7231" in summary

    def test_summary_truncation(self):
        """Test summary handles long content gracefully."""
        long_issue = "x" * 1000
        verdict = DeepAuditVerdict(
            recommendation="a" * 1000,  # Long recommendation
            confidence=0.5,
            unanimous_issues=[long_issue],
        )

        summary = verdict.summary()

        # Should be truncated
        assert len(summary) < 5000


class TestPreConfiguredAudits:
    """Tests for pre-configured audit configs."""

    def test_strategy_audit(self):
        """Test STRATEGY_AUDIT configuration."""
        assert STRATEGY_AUDIT.rounds == 6
        assert STRATEGY_AUDIT.enable_research is True
        assert CognitiveRole.DEVIL_ADVOCATE in STRATEGY_AUDIT.roles
        assert STRATEGY_AUDIT.cross_examination_depth == 4

    def test_contract_audit(self):
        """Test CONTRACT_AUDIT configuration."""
        assert CONTRACT_AUDIT.rounds == 4
        assert CONTRACT_AUDIT.enable_research is False
        assert CONTRACT_AUDIT.cross_examination_depth == 5
        assert CONTRACT_AUDIT.risk_threshold == 0.5

    def test_code_architecture_audit(self):
        """Test CODE_ARCHITECTURE_AUDIT configuration."""
        assert CODE_ARCHITECTURE_AUDIT.rounds == 5
        assert CODE_ARCHITECTURE_AUDIT.enable_research is True
        assert CODE_ARCHITECTURE_AUDIT.require_citations is False


class TestDeepAuditOrchestrator:
    """Tests for DeepAuditOrchestrator class."""

    def test_initialization_default_config(self):
        """Test orchestrator with default config."""
        from aragora.core import Agent, Critique

        # Create mock agents
        class MockAgent(Agent):
            def __init__(self, name):
                super().__init__(name=name, model="mock")

            async def generate(self, prompt, history):
                return "Mock response"

            async def critique(self, proposal, context):
                return Critique(agent=self.name, issues=[], suggestions=[])

        agents = [MockAgent("agent1"), MockAgent("agent2")]
        orchestrator = DeepAuditOrchestrator(agents)

        assert len(orchestrator.agents) == 2
        assert orchestrator.config.rounds == 6
        assert orchestrator.research_fn is None
        assert orchestrator.findings == []
        assert orchestrator.round_summaries == []
        assert orchestrator.citations == []

    def test_initialization_custom_config(self):
        """Test orchestrator with custom config."""
        from aragora.core import Agent, Critique

        class MockAgent(Agent):
            def __init__(self, name):
                super().__init__(name=name, model="mock")

            async def generate(self, prompt, history):
                return "Mock response"

            async def critique(self, proposal, context):
                return Critique(agent=self.name, issues=[], suggestions=[])

        agents = [MockAgent("agent1")]
        config = DeepAuditConfig(rounds=4, enable_research=False)

        async def mock_research(query):
            return "Research results"

        orchestrator = DeepAuditOrchestrator(
            agents=agents,
            config=config,
            research_fn=mock_research,
        )

        assert orchestrator.config.rounds == 4
        assert orchestrator.config.enable_research is False
        assert orchestrator.research_fn is not None

    def test_role_rotator_initialized(self):
        """Test role rotator is properly initialized."""
        from aragora.core import Agent, Critique

        class MockAgent(Agent):
            def __init__(self, name):
                super().__init__(name=name, model="mock")

            async def generate(self, prompt, history):
                return "Mock response"

            async def critique(self, proposal, context):
                return Critique(agent=self.name, issues=[], suggestions=[])

        agents = [MockAgent("agent1")]
        config = DeepAuditConfig(
            roles=[CognitiveRole.ANALYST, CognitiveRole.SKEPTIC],
            synthesizer_final_round=True,
        )

        orchestrator = DeepAuditOrchestrator(agents, config)

        assert orchestrator.role_rotator is not None
        assert orchestrator.role_rotator.config.enabled is True


class TestBuildVerdict:
    """Tests for verdict building logic."""

    def test_build_verdict_basic(self):
        """Test building basic verdict from result."""
        from aragora.core import Agent, Critique, DebateResult

        class MockAgent(Agent):
            def __init__(self, name):
                super().__init__(name=name, model="mock")

            async def generate(self, prompt, history):
                return "Mock response"

            async def critique(self, proposal, context):
                return Critique(agent=self.name, issues=[], suggestions=[])

        agents = [MockAgent("agent1")]
        orchestrator = DeepAuditOrchestrator(agents)

        # Create a minimal debate result
        result = DebateResult(
            final_answer="The recommendation is to proceed",
            confidence=0.85,
            critiques=[],
        )

        verdict = orchestrator._build_verdict(result, "Cross exam notes")

        assert verdict.recommendation == "The recommendation is to proceed"
        assert verdict.confidence == 0.85
        assert verdict.cross_examination_notes == "Cross exam notes"

    def test_build_verdict_no_answer(self):
        """Test verdict when no final answer."""
        from aragora.core import Agent, Critique, DebateResult

        class MockAgent(Agent):
            def __init__(self, name):
                super().__init__(name=name, model="mock")

            async def generate(self, prompt, history):
                return "Mock response"

            async def critique(self, proposal, context):
                return Critique(agent=self.name, issues=[], suggestions=[])

        agents = [MockAgent("agent1")]
        orchestrator = DeepAuditOrchestrator(agents)

        result = DebateResult(
            final_answer=None,
            confidence=0.0,
            critiques=[],
        )

        verdict = orchestrator._build_verdict(result, "")

        assert verdict.recommendation == "No recommendation reached"


class TestRunDeepAuditFunction:
    """Tests for run_deep_audit convenience function."""

    def test_run_deep_audit_signature(self):
        """Test run_deep_audit has proper signature."""
        # We can't actually run the debate without mocking Arena
        # but we can verify the function signature works
        import inspect

        sig = inspect.signature(run_deep_audit)

        assert "task" in sig.parameters
        assert "agents" in sig.parameters
        assert "context" in sig.parameters
        assert "config" in sig.parameters

    def test_function_is_async(self):
        """Test run_deep_audit is an async function."""
        import asyncio

        assert asyncio.iscoroutinefunction(run_deep_audit)
