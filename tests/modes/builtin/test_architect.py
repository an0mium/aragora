"""Tests for the built-in architect mode."""

import pytest

from aragora.modes.base import ModeRegistry
from aragora.modes.builtin.architect import ArchitectMode
from aragora.modes.tool_groups import ToolGroup


@pytest.fixture(autouse=True)
def clear_mode_registry():
    """Keep the global mode registry isolated across tests."""
    ModeRegistry.clear()
    yield
    ModeRegistry.clear()


def test_default_values():
    """Architect mode exposes the expected defaults."""
    mode = ArchitectMode()

    assert mode.name == "architect"
    assert mode.description == "High-level design and planning without implementation"
    assert mode.tool_groups == ToolGroup.READ | ToolGroup.BROWSER
    assert mode.file_patterns == []
    assert mode.system_prompt_additions == ""


def test_mode_auto_registers_in_registry():
    """Architect mode instances auto-register with the mode registry."""
    mode = ArchitectMode()

    assert ModeRegistry.get("architect") is mode
    assert ModeRegistry.get("ARCHITECT") is mode


def test_allows_read_tools_by_default():
    """Architect mode can use read-only local inspection tools."""
    mode = ArchitectMode()

    assert mode.can_access_tool("read")
    assert mode.can_access_tool("grep")


def test_allows_browser_tools_by_default():
    """Architect mode can use browser tools for research."""
    mode = ArchitectMode()

    assert mode.can_access_tool("web_search")
    assert mode.can_access_tool("web_fetch")


def test_disallows_edit_and_command_tools_by_default():
    """Architect mode cannot modify files or execute commands."""
    mode = ArchitectMode()

    assert not mode.can_access_tool("edit")
    assert not mode.can_access_tool("write")
    assert not mode.can_access_tool("bash")
    assert not mode.can_access_tool("kill_shell")


def test_unknown_tools_remain_allowed_for_backward_compatibility():
    """Unknown tool names continue to be treated as allowed."""
    mode = ArchitectMode()

    assert mode.can_access_tool("internal-custom-tool")


def test_all_files_are_accessible_when_no_patterns_are_defined():
    """Architect mode defaults to unrestricted file reads."""
    mode = ArchitectMode()

    assert mode.can_access_file("aragora/modes/builtin/architect.py")
    assert mode.can_access_file("docs/architecture/decision-record.md")


def test_custom_file_patterns_can_limit_file_access():
    """Explicit file patterns are enforced by inherited file access checks."""
    mode = ArchitectMode(file_patterns=["**/*.py", "docs/**/*.md"])

    assert mode.can_access_file("aragora/modes/builtin/architect.py")
    assert mode.can_access_file("docs/architecture/decision-record.md")
    assert not mode.can_access_file("README.rst")


def test_system_prompt_contains_allowed_actions_and_restrictions():
    """The architect prompt documents its permitted and forbidden actions."""
    prompt = ArchitectMode().get_system_prompt()

    assert "## Architect Mode" in prompt
    assert "Read and search code to understand the codebase" in prompt
    assert "Browse the web for documentation and best practices" in prompt
    assert "DO NOT edit or write any files" in prompt
    assert "DO NOT execute any commands" in prompt
    assert "DO NOT implement changes - only plan them" in prompt


def test_system_prompt_contains_focus_areas_and_output_guidance():
    """The prompt captures the architect workflow and output style."""
    prompt = ArchitectMode().get_system_prompt()

    assert "1. **Understand First**: Thoroughly explore before proposing" in prompt
    assert "2. **Big Picture**: Focus on overall architecture and patterns" in prompt
    assert "3. **Trade-offs**: Consider multiple approaches with pros/cons" in prompt
    assert "Provide clear architectural diagrams (ASCII or described)" in prompt
    assert "List specific files and functions affected by proposals" in prompt
    assert "Flag risks and technical debt implications" in prompt


def test_system_prompt_appends_trimmed_additions():
    """Custom prompt additions are appended once and trimmed."""
    prompt = ArchitectMode(
        system_prompt_additions="  Prefer incremental migration plans.  \n"
    ).get_system_prompt()

    assert "\n\nPrefer incremental migration plans.\n" in prompt
    assert prompt.endswith("Prefer incremental migration plans.\n")
