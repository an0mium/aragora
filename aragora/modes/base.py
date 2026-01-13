"""
Mode Base Class and Registry.

Provides the foundation for operational modes inspired by Kilocode's
multi-mode architecture. Each mode defines:
- Tool access permissions (via ToolGroup)
- File access patterns
- Mode-specific system prompts
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import ClassVar

from aragora.modes.tool_groups import ToolGroup, can_use_tool


@dataclass
class Mode(ABC):
    """
    Abstract base class for operational modes.

    A mode defines how an agent operates: what tools it can use,
    what files it can access, and how it should behave.
    """

    name: str
    description: str
    tool_groups: ToolGroup
    file_patterns: list[str] = field(default_factory=list)
    system_prompt_additions: str = ""

    # Auto-register when instantiated
    _auto_register: ClassVar[bool] = True

    def __post_init__(self):
        if self._auto_register and self.name:
            ModeRegistry.register(self)

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Return the mode-specific system prompt.

        This should include behavioral guidelines specific to this mode.
        """
        raise NotImplementedError("Subclasses must implement get_system_prompt method")

    def can_access_tool(self, tool_name: str) -> bool:
        """Check if this mode allows using a given tool."""
        return can_use_tool(self.tool_groups, tool_name)

    def can_access_file(self, file_path: str) -> bool:
        """
        Check if this mode can access a given file.

        If no file patterns are specified, all files are accessible.
        """
        if not self.file_patterns:
            return True

        for pattern in self.file_patterns:
            if fnmatch(file_path, pattern):
                return True
        return False

    def get_restrictions_summary(self) -> str:
        """Generate a human-readable summary of mode restrictions."""
        lines = [f"## Mode: {self.name}", f"{self.description}", ""]

        # Tool groups
        allowed_tools = []
        if ToolGroup.READ in self.tool_groups:
            allowed_tools.append("Read files")
        if ToolGroup.EDIT in self.tool_groups:
            allowed_tools.append("Edit files")
        if ToolGroup.COMMAND in self.tool_groups:
            allowed_tools.append("Run commands")
        if ToolGroup.BROWSER in self.tool_groups:
            allowed_tools.append("Browse web")
        if ToolGroup.MCP in self.tool_groups:
            allowed_tools.append("MCP tools")
        if ToolGroup.DEBATE in self.tool_groups:
            allowed_tools.append("Debate")

        lines.append(
            "**Allowed:** " + ", ".join(allowed_tools) if allowed_tools else "**Allowed:** None"
        )

        # File patterns
        if self.file_patterns:
            lines.append("**File access:** " + ", ".join(self.file_patterns))
        else:
            lines.append("**File access:** All files")

        return "\n".join(lines)


class ModeRegistry:
    """
    Global registry of available modes.

    Modes auto-register when instantiated, or can be manually registered.
    """

    _modes: ClassVar[dict[str, Mode]] = {}

    @classmethod
    def register(cls, mode: Mode) -> None:
        """Register a mode in the global registry."""
        cls._modes[mode.name.lower()] = mode

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Remove a mode from the registry. Returns True if found."""
        name = name.lower()
        if name in cls._modes:
            del cls._modes[name]
            return True
        return False

    @classmethod
    def get(cls, name: str) -> Mode | None:
        """Get a mode by name (case-insensitive)."""
        return cls._modes.get(name.lower())

    @classmethod
    def get_or_raise(cls, name: str) -> Mode:
        """Get a mode by name, raising KeyError if not found."""
        mode = cls.get(name)
        if mode is None:
            available = ", ".join(cls.list_all())
            raise KeyError(f"Mode '{name}' not found. Available: {available}")
        return mode

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered mode names."""
        return list(cls._modes.keys())

    @classmethod
    def get_all(cls) -> list[Mode]:
        """Get all registered modes."""
        return list(cls._modes.values())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered modes. Mainly for testing."""
        cls._modes.clear()
