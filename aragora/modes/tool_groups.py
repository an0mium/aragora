"""
Tool Groups for Mode-Based Access Control.

Inspired by Kilocode's granular permission system, this module defines
tool access groups that can be combined to create operational modes.
"""

from enum import Flag, auto


class ToolGroup(Flag):
    """
    Granular permission flags for tool access.

    Can be combined using bitwise operators:
        DEVELOPER = READ | EDIT | COMMAND
        READONLY = READ | BROWSER
    """

    NONE = 0

    # Core file operations
    READ = auto()  # Read files, glob, grep
    EDIT = auto()  # Edit, write files

    # Execution
    COMMAND = auto()  # Execute shell commands

    # External access
    BROWSER = auto()  # Web fetch, search

    # Advanced capabilities
    MCP = auto()  # MCP server tools
    DEBATE = auto()  # Participate in debates

    # Composite groups for convenience
    @classmethod
    def READONLY(cls) -> "ToolGroup":
        """Read-only access with web browsing."""
        return cls.READ | cls.BROWSER

    @classmethod
    def DEVELOPER(cls) -> "ToolGroup":
        """Standard development access: read, edit, run commands."""
        return cls.READ | cls.EDIT | cls.COMMAND

    @classmethod
    def FULL(cls) -> "ToolGroup":
        """Full access to all tools."""
        return cls.READ | cls.EDIT | cls.COMMAND | cls.BROWSER | cls.MCP | cls.DEBATE


# Tool name to required group mapping
TOOL_GROUP_MAP: dict[str, ToolGroup] = {
    # Read tools
    "read": ToolGroup.READ,
    "glob": ToolGroup.READ,
    "grep": ToolGroup.READ,
    # Edit tools
    "edit": ToolGroup.EDIT,
    "write": ToolGroup.EDIT,
    "notebook_edit": ToolGroup.EDIT,
    # Command tools
    "bash": ToolGroup.COMMAND,
    "kill_shell": ToolGroup.COMMAND,
    # Browser tools
    "web_fetch": ToolGroup.BROWSER,
    "web_search": ToolGroup.BROWSER,
    # Debate tools
    "debate": ToolGroup.DEBATE,
    "arena": ToolGroup.DEBATE,
}


def get_required_group(tool_name: str) -> ToolGroup:
    """Get the required tool group for a given tool name."""
    normalized = tool_name.lower().replace("-", "_")
    return TOOL_GROUP_MAP.get(normalized, ToolGroup.NONE)


def can_use_tool(allowed_groups: ToolGroup, tool_name: str) -> bool:
    """Check if the allowed groups permit using a specific tool."""
    required = get_required_group(tool_name)
    if required == ToolGroup.NONE:
        return True  # Unknown tools allowed by default
    return bool(allowed_groups & required)
