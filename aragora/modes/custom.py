"""
Custom Mode Loader.

Loads user-defined modes from YAML configuration files,
enabling extensible mode creation without code changes.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aragora.modes.base import Mode, ModeRegistry
from aragora.modes.tool_groups import ToolGroup

logger = logging.getLogger(__name__)


@dataclass
class CustomMode(Mode):
    """
    A mode defined via YAML configuration.

    Supports inheritance from built-in modes via base_mode.
    """

    name: str = "custom"
    description: str = "Custom mode"
    tool_groups: ToolGroup = field(default_factory=lambda: ToolGroup.READ)
    file_patterns: list[str] = field(default_factory=list)
    system_prompt_additions: str = ""
    base_mode: str | None = None

    def get_system_prompt(self) -> str:
        """Get system prompt, inheriting from base mode if specified."""
        base_prompt = ""

        if self.base_mode:
            base = ModeRegistry.get(self.base_mode)
            if base:
                base_prompt = base.get_system_prompt() + "\n\n---\n\n"

        return (
            base_prompt
            + f"""## Custom Mode: {self.name}

{self.description}

{self.system_prompt_additions}
"""
        )


class CustomModeLoader:
    """
    Loads custom modes from YAML configuration files.

    Default search paths:
    - .aragora/modes/
    - ~/.config/aragora/modes/
    """

    DEFAULT_PATHS = [
        ".aragora/modes",
        os.path.expanduser("~/.config/aragora/modes"),
    ]

    # Mapping of string names to ToolGroup flags
    TOOL_GROUP_MAP = {
        "read": ToolGroup.READ,
        "edit": ToolGroup.EDIT,
        "command": ToolGroup.COMMAND,
        "browser": ToolGroup.BROWSER,
        "mcp": ToolGroup.MCP,
        "debate": ToolGroup.DEBATE,
        "readonly": ToolGroup.READ | ToolGroup.BROWSER,
        "developer": ToolGroup.READ | ToolGroup.EDIT | ToolGroup.COMMAND,
        "full": ToolGroup.READ
        | ToolGroup.EDIT
        | ToolGroup.COMMAND
        | ToolGroup.BROWSER
        | ToolGroup.MCP
        | ToolGroup.DEBATE,
    }

    def __init__(self, search_paths: list[str] | None = None):
        self.search_paths = search_paths or self.DEFAULT_PATHS

    def load_from_yaml(self, path: str | Path) -> CustomMode:
        """
        Load a custom mode from a YAML file.

        Example YAML:
        ```yaml
        name: security-auditor
        description: Security-focused code auditor
        base_mode: reviewer
        tool_groups:
          - read
          - browser
        file_patterns:
          - "**/*.py"
          - "**/*.js"
        system_prompt_additions: |
          Focus on OWASP Top 10 vulnerabilities.
        ```
        """
        # Security: Validate path is within allowed directories
        resolved_path = Path(path).resolve()
        allowed = False
        for search_path in self.search_paths:
            try:
                allowed_dir = Path(search_path).resolve()
                resolved_path.relative_to(allowed_dir)
                allowed = True
                break
            except ValueError:
                continue

        if not allowed:
            raise PermissionError(
                f"Access denied: '{path}' is outside allowed mode directories. "
                f"Allowed: {self.search_paths}"
            )

        with open(resolved_path) as f:
            config = yaml.safe_load(f)

        return self._parse_config(config)

    def _parse_config(self, config: dict[str, Any]) -> CustomMode:
        """Parse a configuration dictionary into a CustomMode."""
        name = config.get("name", "custom")
        description = config.get("description", "")

        # Parse tool groups
        tool_groups = ToolGroup.NONE
        for group in config.get("tool_groups", ["read"]):
            group_lower = group.lower()
            if group_lower in self.TOOL_GROUP_MAP:
                tool_groups |= self.TOOL_GROUP_MAP[group_lower]

        return CustomMode(
            name=name,
            description=description,
            tool_groups=tool_groups,
            file_patterns=config.get("file_patterns", []),
            system_prompt_additions=config.get("system_prompt_additions", ""),
            base_mode=config.get("base_mode"),
        )

    def load_all(self, directory: str | Path | None = None) -> list[CustomMode]:
        """
        Load all custom modes from a directory.

        If no directory specified, searches all default paths.
        """
        modes = []

        directories = [directory] if directory else self.search_paths

        for dir_path in directories:
            path = Path(dir_path)
            if not path.exists():
                continue

            for yaml_file in path.glob("*.yaml"):
                try:
                    mode = self.load_from_yaml(yaml_file)
                    modes.append(mode)
                except Exception as e:
                    # Log but don't fail on individual file errors
                    logger.warning(f"Failed to load {yaml_file}: {e}")

            for yml_file in path.glob("*.yml"):
                try:
                    mode = self.load_from_yaml(yml_file)
                    modes.append(mode)
                except Exception as e:
                    logger.warning(f"Failed to load {yml_file}: {e}")

        return modes

    def load_and_register_all(self, directory: str | Path | None = None) -> int:
        """
        Load all custom modes and register them.

        Returns the number of modes registered.
        """
        modes = self.load_all(directory)
        for mode in modes:
            ModeRegistry.register(mode)
        return len(modes)
