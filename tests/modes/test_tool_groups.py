"""Tests for ToolGroup permission flags and access control."""

import pytest

from aragora.modes.tool_groups import (
    TOOL_GROUP_MAP,
    ToolGroup,
    can_use_tool,
    get_required_group,
)


class TestToolGroupFlags:
    """Tests for ToolGroup flag operations."""

    def test_none_flag(self):
        """NONE flag has value 0."""
        assert ToolGroup.NONE.value == 0
        assert not ToolGroup.NONE

    def test_individual_flags(self):
        """Individual flags are non-zero and distinct."""
        flags = [ToolGroup.READ, ToolGroup.EDIT, ToolGroup.COMMAND, ToolGroup.BROWSER, ToolGroup.MCP, ToolGroup.DEBATE]
        values = [f.value for f in flags]
        assert len(values) == len(set(values)), "All flags should have unique values"
        assert all(v != 0 for v in values), "All flags should be non-zero"

    def test_combining_flags(self):
        """Flags can be combined with bitwise OR."""
        combined = ToolGroup.READ | ToolGroup.EDIT
        assert ToolGroup.READ in combined
        assert ToolGroup.EDIT in combined
        assert ToolGroup.COMMAND not in combined

    def test_readonly_composite(self):
        """READONLY returns READ | BROWSER."""
        readonly = ToolGroup.READONLY()
        assert ToolGroup.READ in readonly
        assert ToolGroup.BROWSER in readonly
        assert ToolGroup.EDIT not in readonly
        assert ToolGroup.COMMAND not in readonly

    def test_developer_composite(self):
        """DEVELOPER returns READ | EDIT | COMMAND."""
        developer = ToolGroup.DEVELOPER()
        assert ToolGroup.READ in developer
        assert ToolGroup.EDIT in developer
        assert ToolGroup.COMMAND in developer
        assert ToolGroup.BROWSER not in developer
        assert ToolGroup.MCP not in developer

    def test_full_composite(self):
        """FULL returns all flags combined."""
        full = ToolGroup.FULL()
        assert ToolGroup.READ in full
        assert ToolGroup.EDIT in full
        assert ToolGroup.COMMAND in full
        assert ToolGroup.BROWSER in full
        assert ToolGroup.MCP in full
        assert ToolGroup.DEBATE in full


class TestToolGroupMap:
    """Tests for the TOOL_GROUP_MAP mapping."""

    def test_read_tools_mapped(self):
        """Read tools are mapped to READ group."""
        assert TOOL_GROUP_MAP["read"] == ToolGroup.READ
        assert TOOL_GROUP_MAP["glob"] == ToolGroup.READ
        assert TOOL_GROUP_MAP["grep"] == ToolGroup.READ

    def test_edit_tools_mapped(self):
        """Edit tools are mapped to EDIT group."""
        assert TOOL_GROUP_MAP["edit"] == ToolGroup.EDIT
        assert TOOL_GROUP_MAP["write"] == ToolGroup.EDIT
        assert TOOL_GROUP_MAP["notebook_edit"] == ToolGroup.EDIT

    def test_command_tools_mapped(self):
        """Command tools are mapped to COMMAND group."""
        assert TOOL_GROUP_MAP["bash"] == ToolGroup.COMMAND
        assert TOOL_GROUP_MAP["kill_shell"] == ToolGroup.COMMAND

    def test_browser_tools_mapped(self):
        """Browser tools are mapped to BROWSER group."""
        assert TOOL_GROUP_MAP["web_fetch"] == ToolGroup.BROWSER
        assert TOOL_GROUP_MAP["web_search"] == ToolGroup.BROWSER

    def test_debate_tools_mapped(self):
        """Debate tools are mapped to DEBATE group."""
        assert TOOL_GROUP_MAP["debate"] == ToolGroup.DEBATE
        assert TOOL_GROUP_MAP["arena"] == ToolGroup.DEBATE


class TestGetRequiredGroup:
    """Tests for get_required_group function."""

    def test_known_tool(self):
        """Known tools return their required group."""
        assert get_required_group("read") == ToolGroup.READ
        assert get_required_group("bash") == ToolGroup.COMMAND
        assert get_required_group("web_fetch") == ToolGroup.BROWSER

    def test_case_insensitive(self):
        """Tool names are case insensitive."""
        assert get_required_group("READ") == ToolGroup.READ
        assert get_required_group("Bash") == ToolGroup.COMMAND
        assert get_required_group("WEB_FETCH") == ToolGroup.BROWSER

    def test_hyphen_to_underscore(self):
        """Hyphens are converted to underscores."""
        assert get_required_group("web-fetch") == ToolGroup.BROWSER
        assert get_required_group("web-search") == ToolGroup.BROWSER
        assert get_required_group("kill-shell") == ToolGroup.COMMAND

    def test_unknown_tool_returns_none(self):
        """Unknown tools return NONE."""
        assert get_required_group("unknown_tool") == ToolGroup.NONE
        assert get_required_group("my_custom_tool") == ToolGroup.NONE


class TestCanUseTool:
    """Tests for can_use_tool function."""

    def test_allowed_tool(self):
        """Allowed groups permit matching tools."""
        assert can_use_tool(ToolGroup.READ, "read")
        assert can_use_tool(ToolGroup.EDIT, "write")
        assert can_use_tool(ToolGroup.COMMAND, "bash")
        assert can_use_tool(ToolGroup.BROWSER, "web_fetch")

    def test_disallowed_tool(self):
        """Missing groups deny tools."""
        assert not can_use_tool(ToolGroup.READ, "edit")
        assert not can_use_tool(ToolGroup.READ, "bash")
        assert not can_use_tool(ToolGroup.EDIT, "bash")
        assert not can_use_tool(ToolGroup.COMMAND, "web_fetch")

    def test_combined_groups(self):
        """Combined groups allow multiple tool types."""
        groups = ToolGroup.READ | ToolGroup.EDIT
        assert can_use_tool(groups, "read")
        assert can_use_tool(groups, "edit")
        assert not can_use_tool(groups, "bash")

    def test_developer_composite(self):
        """DEVELOPER composite allows read, edit, command."""
        developer = ToolGroup.DEVELOPER()
        assert can_use_tool(developer, "read")
        assert can_use_tool(developer, "edit")
        assert can_use_tool(developer, "bash")
        assert not can_use_tool(developer, "web_fetch")

    def test_full_composite(self):
        """FULL composite allows all tools."""
        full = ToolGroup.FULL()
        assert can_use_tool(full, "read")
        assert can_use_tool(full, "edit")
        assert can_use_tool(full, "bash")
        assert can_use_tool(full, "web_fetch")
        assert can_use_tool(full, "debate")

    def test_unknown_tool_allowed_by_default(self):
        """Unknown tools are allowed by default."""
        assert can_use_tool(ToolGroup.READ, "unknown_tool")
        assert can_use_tool(ToolGroup.NONE, "unknown_tool")

    def test_none_groups_denies_known_tools(self):
        """NONE groups deny all known tools."""
        assert not can_use_tool(ToolGroup.NONE, "read")
        assert not can_use_tool(ToolGroup.NONE, "bash")
        assert not can_use_tool(ToolGroup.NONE, "web_fetch")
