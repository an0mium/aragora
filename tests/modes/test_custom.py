"""
Tests for custom mode loader.

Tests cover:
- CustomMode dataclass
- CustomModeLoader class
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from aragora.modes.custom import CustomMode, CustomModeLoader
from aragora.modes.tool_groups import ToolGroup
from aragora.modes.base import ModeRegistry


class TestCustomMode:
    """Tests for CustomMode dataclass."""

    def test_default_values(self):
        """Default values are sensible."""
        mode = CustomMode()

        assert mode.name == "custom"
        assert mode.description == "Custom mode"
        assert mode.tool_groups == ToolGroup.READ
        assert mode.file_patterns == []
        assert mode.system_prompt_additions == ""
        assert mode.base_mode is None

    def test_custom_values(self):
        """Custom values can be set."""
        mode = CustomMode(
            name="security-auditor",
            description="Security focused auditing",
            tool_groups=ToolGroup.READ | ToolGroup.BROWSER,
            file_patterns=["**/*.py"],
            system_prompt_additions="Focus on OWASP Top 10",
            base_mode="reviewer",
        )

        assert mode.name == "security-auditor"
        assert mode.description == "Security focused auditing"
        assert ToolGroup.READ in mode.tool_groups
        assert ToolGroup.BROWSER in mode.tool_groups
        assert mode.file_patterns == ["**/*.py"]
        assert mode.base_mode == "reviewer"

    def test_get_system_prompt_no_base(self):
        """System prompt without base mode."""
        mode = CustomMode(
            name="test-mode",
            description="A test mode",
            system_prompt_additions="Custom instructions here",
        )

        prompt = mode.get_system_prompt()

        assert "## Custom Mode: test-mode" in prompt
        assert "A test mode" in prompt
        assert "Custom instructions here" in prompt

    def test_get_system_prompt_with_base(self):
        """System prompt inherits from base mode."""
        # Register a mock base mode
        base_mode = CustomMode(
            name="base-mode",
            description="Base mode description",
            system_prompt_additions="Base instructions",
        )
        ModeRegistry.register(base_mode)

        try:
            mode = CustomMode(
                name="derived-mode",
                description="Derived mode",
                system_prompt_additions="Derived instructions",
                base_mode="base-mode",
            )

            prompt = mode.get_system_prompt()

            # Should contain base prompt
            assert "Base mode description" in prompt
            # Should contain derived prompt
            assert "Derived mode" in prompt
            assert "Derived instructions" in prompt
            # Should have separator
            assert "---" in prompt
        finally:
            ModeRegistry.unregister("base-mode")

    def test_get_system_prompt_invalid_base(self):
        """System prompt with invalid base mode just uses own prompt."""
        mode = CustomMode(
            name="orphan-mode",
            description="Mode with missing base",
            base_mode="nonexistent-base",
        )

        prompt = mode.get_system_prompt()

        # Should still work, just without base
        assert "orphan-mode" in prompt
        assert "---" not in prompt  # No separator since no base


class TestCustomModeLoader:
    """Tests for CustomModeLoader class."""

    def test_init_default_paths(self):
        """Initializes with default search paths."""
        loader = CustomModeLoader()

        assert ".aragora/modes" in loader.search_paths
        assert any("config/aragora/modes" in p for p in loader.search_paths)

    def test_init_custom_paths(self):
        """Initializes with custom search paths."""
        loader = CustomModeLoader(search_paths=["/custom/path"])

        assert loader.search_paths == ["/custom/path"]

    def test_tool_group_map_has_all_groups(self):
        """TOOL_GROUP_MAP has all expected groups."""
        expected = ["read", "edit", "command", "browser", "mcp", "debate", "readonly", "developer", "full"]

        for group in expected:
            assert group in CustomModeLoader.TOOL_GROUP_MAP

    def test_load_from_yaml(self, tmp_path):
        """Loads mode from YAML file."""
        # Create YAML file
        yaml_content = """
name: test-mode
description: A test mode for unit testing
tool_groups:
  - read
  - edit
file_patterns:
  - "**/*.py"
  - "**/*.js"
system_prompt_additions: |
  This is a test mode.
  Follow these instructions.
"""
        yaml_file = tmp_path / "test-mode.yaml"
        yaml_file.write_text(yaml_content)

        loader = CustomModeLoader(search_paths=[str(tmp_path)])
        mode = loader.load_from_yaml(yaml_file)

        assert mode.name == "test-mode"
        assert mode.description == "A test mode for unit testing"
        assert ToolGroup.READ in mode.tool_groups
        assert ToolGroup.EDIT in mode.tool_groups
        assert "**/*.py" in mode.file_patterns
        assert "This is a test mode" in mode.system_prompt_additions

    def test_load_from_yaml_with_base_mode(self, tmp_path):
        """Loads mode with base_mode reference."""
        yaml_content = """
name: derived-mode
description: Derived from reviewer
base_mode: reviewer
tool_groups:
  - read
"""
        yaml_file = tmp_path / "derived.yaml"
        yaml_file.write_text(yaml_content)

        loader = CustomModeLoader(search_paths=[str(tmp_path)])
        mode = loader.load_from_yaml(yaml_file)

        assert mode.base_mode == "reviewer"

    def test_load_from_yaml_minimal(self, tmp_path):
        """Loads mode with minimal config."""
        yaml_content = """
name: minimal
"""
        yaml_file = tmp_path / "minimal.yaml"
        yaml_file.write_text(yaml_content)

        loader = CustomModeLoader(search_paths=[str(tmp_path)])
        mode = loader.load_from_yaml(yaml_file)

        assert mode.name == "minimal"
        assert mode.description == ""
        # Default to read
        assert ToolGroup.READ in mode.tool_groups

    def test_load_from_yaml_outside_allowed_paths(self, tmp_path):
        """Raises PermissionError for paths outside allowed directories."""
        # Create file outside allowed paths
        yaml_file = tmp_path / "evil.yaml"
        yaml_file.write_text("name: evil")

        # Loader with different allowed path
        other_path = tmp_path / "other"
        other_path.mkdir()
        loader = CustomModeLoader(search_paths=[str(other_path)])

        with pytest.raises(PermissionError, match="outside allowed mode directories"):
            loader.load_from_yaml(yaml_file)

    def test_parse_config_tool_groups_combined(self):
        """Parses multiple tool groups into combined flags."""
        loader = CustomModeLoader()

        config = {
            "name": "multi-tool",
            "tool_groups": ["read", "edit", "command"],
        }

        mode = loader._parse_config(config)

        assert ToolGroup.READ in mode.tool_groups
        assert ToolGroup.EDIT in mode.tool_groups
        assert ToolGroup.COMMAND in mode.tool_groups
        assert ToolGroup.BROWSER not in mode.tool_groups

    def test_parse_config_developer_composite(self):
        """Parses developer composite group."""
        loader = CustomModeLoader()

        config = {
            "name": "dev-mode",
            "tool_groups": ["developer"],
        }

        mode = loader._parse_config(config)

        # Developer = read + edit + command
        assert ToolGroup.READ in mode.tool_groups
        assert ToolGroup.EDIT in mode.tool_groups
        assert ToolGroup.COMMAND in mode.tool_groups

    def test_parse_config_unknown_group_ignored(self):
        """Unknown tool groups are ignored."""
        loader = CustomModeLoader()

        config = {
            "name": "unknown-tools",
            "tool_groups": ["read", "unknown_tool", "invalid"],
        }

        mode = loader._parse_config(config)

        # Only read should be present
        assert ToolGroup.READ in mode.tool_groups
        # No error raised for unknown

    def test_load_all_from_directory(self, tmp_path):
        """Loads all modes from a directory."""
        # Create multiple YAML files
        (tmp_path / "mode1.yaml").write_text("name: mode1\ndescription: First mode")
        (tmp_path / "mode2.yaml").write_text("name: mode2\ndescription: Second mode")
        (tmp_path / "mode3.yml").write_text("name: mode3\ndescription: Third mode")

        loader = CustomModeLoader(search_paths=[str(tmp_path)])
        modes = loader.load_all(tmp_path)

        assert len(modes) == 3
        names = [m.name for m in modes]
        assert "mode1" in names
        assert "mode2" in names
        assert "mode3" in names

    def test_load_all_ignores_invalid_files(self, tmp_path):
        """Ignores invalid YAML files without failing."""
        (tmp_path / "valid.yaml").write_text("name: valid")
        (tmp_path / "invalid.yaml").write_text("name: [invalid yaml syntax")

        loader = CustomModeLoader(search_paths=[str(tmp_path)])
        modes = loader.load_all(tmp_path)

        # Should only load valid file
        assert len(modes) == 1
        assert modes[0].name == "valid"

    def test_load_all_from_default_paths(self, tmp_path):
        """Loads from all search paths when no directory specified."""
        # Create two directories
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        (dir1 / "mode1.yaml").write_text("name: mode1")
        (dir2 / "mode2.yaml").write_text("name: mode2")

        loader = CustomModeLoader(search_paths=[str(dir1), str(dir2)])
        modes = loader.load_all()

        assert len(modes) == 2

    def test_load_all_nonexistent_directory(self, tmp_path):
        """Handles nonexistent directories gracefully."""
        loader = CustomModeLoader(search_paths=[str(tmp_path / "nonexistent")])
        modes = loader.load_all()

        assert modes == []

    def test_load_and_register_all(self, tmp_path):
        """Loads and registers all modes."""
        (tmp_path / "custom1.yaml").write_text("name: custom1")
        (tmp_path / "custom2.yaml").write_text("name: custom2")

        loader = CustomModeLoader(search_paths=[str(tmp_path)])

        try:
            count = loader.load_and_register_all(tmp_path)

            assert count == 2
            assert ModeRegistry.get("custom1") is not None
            assert ModeRegistry.get("custom2") is not None
        finally:
            ModeRegistry.unregister("custom1")
            ModeRegistry.unregister("custom2")


class TestCustomModeLoaderSecurity:
    """Security tests for CustomModeLoader."""

    def test_path_traversal_blocked(self, tmp_path):
        """Path traversal attacks are blocked."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()

        secret = tmp_path / "secret"
        secret.mkdir()
        secret_file = secret / "secret.yaml"
        secret_file.write_text("name: secret")

        loader = CustomModeLoader(search_paths=[str(allowed)])

        # Try to access file outside allowed path via traversal
        with pytest.raises(PermissionError):
            loader.load_from_yaml(allowed / ".." / "secret" / "secret.yaml")

    def test_symlink_outside_allowed(self, tmp_path):
        """Symlinks pointing outside allowed paths are blocked."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()

        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "target.yaml").write_text("name: target")

        # Create symlink from allowed to outside
        symlink = allowed / "link.yaml"
        try:
            symlink.symlink_to(outside / "target.yaml")
        except OSError:
            pytest.skip("Symlink creation not supported")

        loader = CustomModeLoader(search_paths=[str(allowed)])

        # The resolved path is outside allowed
        with pytest.raises(PermissionError):
            loader.load_from_yaml(symlink)
