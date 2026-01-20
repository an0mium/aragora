"""Tests for Mode base class and ModeRegistry."""

import pytest
from dataclasses import dataclass

from aragora.modes.base import Mode, ModeRegistry
from aragora.modes.tool_groups import ToolGroup


@dataclass
class TestMode(Mode):
    """Concrete Mode implementation for testing."""

    _auto_register: bool = False  # Disable auto-register for tests

    def get_system_prompt(self) -> str:
        """Return test system prompt."""
        return f"Test mode: {self.name}"


class TestModeClass:
    """Tests for Mode base class."""

    def setup_method(self):
        """Clear registry before each test."""
        ModeRegistry.clear()

    def test_mode_creation(self):
        """Mode can be created with required fields."""
        mode = TestMode(
            name="test",
            description="A test mode",
            tool_groups=ToolGroup.READ,
        )
        assert mode.name == "test"
        assert mode.description == "A test mode"
        assert mode.tool_groups == ToolGroup.READ

    def test_default_file_patterns(self):
        """File patterns default to empty list."""
        mode = TestMode(
            name="test",
            description="Test",
            tool_groups=ToolGroup.READ,
        )
        assert mode.file_patterns == []

    def test_default_system_prompt_additions(self):
        """System prompt additions default to empty string."""
        mode = TestMode(
            name="test",
            description="Test",
            tool_groups=ToolGroup.READ,
        )
        assert mode.system_prompt_additions == ""

    def test_can_access_tool_allowed(self):
        """can_access_tool returns True for allowed tools."""
        mode = TestMode(
            name="test",
            description="Test",
            tool_groups=ToolGroup.READ | ToolGroup.EDIT,
        )
        assert mode.can_access_tool("read")
        assert mode.can_access_tool("edit")
        assert mode.can_access_tool("glob")

    def test_can_access_tool_denied(self):
        """can_access_tool returns False for denied tools."""
        mode = TestMode(
            name="test",
            description="Test",
            tool_groups=ToolGroup.READ,
        )
        assert not mode.can_access_tool("edit")
        assert not mode.can_access_tool("bash")
        assert not mode.can_access_tool("web_fetch")

    def test_can_access_file_no_patterns(self):
        """No patterns means all files accessible."""
        mode = TestMode(
            name="test",
            description="Test",
            tool_groups=ToolGroup.READ,
            file_patterns=[],
        )
        assert mode.can_access_file("any/path/file.py")
        assert mode.can_access_file("/etc/passwd")
        assert mode.can_access_file("README.md")

    def test_can_access_file_with_patterns(self):
        """Patterns restrict file access."""
        mode = TestMode(
            name="test",
            description="Test",
            tool_groups=ToolGroup.READ,
            file_patterns=["*.py", "*.txt"],
        )
        assert mode.can_access_file("test.py")
        assert mode.can_access_file("readme.txt")
        assert not mode.can_access_file("config.json")
        assert not mode.can_access_file("data.csv")

    def test_can_access_file_with_glob_patterns(self):
        """Glob patterns work for file access (fnmatch patterns)."""
        # Note: fnmatch uses shell patterns, not full glob. ** is not supported.
        mode = TestMode(
            name="test",
            description="Test",
            tool_groups=ToolGroup.READ,
            file_patterns=["src/*.py", "tests/*.py", "*.md"],
        )
        assert mode.can_access_file("src/main.py")
        assert mode.can_access_file("tests/test_main.py")
        assert mode.can_access_file("README.md")
        assert not mode.can_access_file("config.json")
        assert not mode.can_access_file("docs/readme.txt")

    def test_get_restrictions_summary(self):
        """get_restrictions_summary returns human-readable text."""
        mode = TestMode(
            name="developer",
            description="Full development access",
            tool_groups=ToolGroup.DEVELOPER(),
            file_patterns=["src/**/*.py"],
        )
        summary = mode.get_restrictions_summary()
        assert "Mode: developer" in summary
        assert "Full development access" in summary
        assert "Read files" in summary
        assert "Edit files" in summary
        assert "Run commands" in summary
        assert "src/**/*.py" in summary

    def test_get_restrictions_summary_all_files(self):
        """Summary shows 'All files' when no patterns."""
        mode = TestMode(
            name="test",
            description="Test",
            tool_groups=ToolGroup.READ,
        )
        summary = mode.get_restrictions_summary()
        assert "All files" in summary

    def test_get_restrictions_summary_no_tools(self):
        """Summary shows 'None' when no tools allowed."""
        mode = TestMode(
            name="restricted",
            description="No tools",
            tool_groups=ToolGroup.NONE,
        )
        summary = mode.get_restrictions_summary()
        assert "Allowed:** None" in summary


class TestModeRegistry:
    """Tests for ModeRegistry class."""

    def setup_method(self):
        """Clear registry before each test."""
        ModeRegistry.clear()

    def test_register_mode(self):
        """Modes can be registered."""
        mode = TestMode(
            name="test",
            description="Test",
            tool_groups=ToolGroup.READ,
        )
        ModeRegistry.register(mode)
        assert "test" in ModeRegistry.list_all()

    def test_get_registered_mode(self):
        """Registered modes can be retrieved."""
        mode = TestMode(
            name="test",
            description="Test",
            tool_groups=ToolGroup.READ,
        )
        ModeRegistry.register(mode)
        retrieved = ModeRegistry.get("test")
        assert retrieved is mode

    def test_get_case_insensitive(self):
        """Mode retrieval is case insensitive."""
        mode = TestMode(
            name="TestMode",
            description="Test",
            tool_groups=ToolGroup.READ,
        )
        ModeRegistry.register(mode)
        assert ModeRegistry.get("testmode") is mode
        assert ModeRegistry.get("TESTMODE") is mode
        assert ModeRegistry.get("TestMode") is mode

    def test_get_nonexistent_returns_none(self):
        """get() returns None for unregistered modes."""
        assert ModeRegistry.get("nonexistent") is None

    def test_get_or_raise_success(self):
        """get_or_raise returns mode when found."""
        mode = TestMode(
            name="test",
            description="Test",
            tool_groups=ToolGroup.READ,
        )
        ModeRegistry.register(mode)
        retrieved = ModeRegistry.get_or_raise("test")
        assert retrieved is mode

    def test_get_or_raise_failure(self):
        """get_or_raise raises KeyError when not found."""
        with pytest.raises(KeyError) as exc_info:
            ModeRegistry.get_or_raise("nonexistent")
        assert "nonexistent" in str(exc_info.value)

    def test_list_all(self):
        """list_all returns all registered mode names."""
        mode1 = TestMode(name="mode1", description="M1", tool_groups=ToolGroup.READ)
        mode2 = TestMode(name="mode2", description="M2", tool_groups=ToolGroup.EDIT)
        ModeRegistry.register(mode1)
        ModeRegistry.register(mode2)

        names = ModeRegistry.list_all()
        assert "mode1" in names
        assert "mode2" in names
        assert len(names) == 2

    def test_get_all(self):
        """get_all returns all registered Mode objects."""
        mode1 = TestMode(name="mode1", description="M1", tool_groups=ToolGroup.READ)
        mode2 = TestMode(name="mode2", description="M2", tool_groups=ToolGroup.EDIT)
        ModeRegistry.register(mode1)
        ModeRegistry.register(mode2)

        modes = ModeRegistry.get_all()
        assert mode1 in modes
        assert mode2 in modes
        assert len(modes) == 2

    def test_unregister_existing(self):
        """unregister removes mode and returns True."""
        mode = TestMode(name="test", description="Test", tool_groups=ToolGroup.READ)
        ModeRegistry.register(mode)

        result = ModeRegistry.unregister("test")
        assert result is True
        assert ModeRegistry.get("test") is None

    def test_unregister_nonexistent(self):
        """unregister returns False for nonexistent mode."""
        result = ModeRegistry.unregister("nonexistent")
        assert result is False

    def test_clear(self):
        """clear removes all registered modes."""
        mode1 = TestMode(name="mode1", description="M1", tool_groups=ToolGroup.READ)
        mode2 = TestMode(name="mode2", description="M2", tool_groups=ToolGroup.EDIT)
        ModeRegistry.register(mode1)
        ModeRegistry.register(mode2)

        ModeRegistry.clear()

        assert len(ModeRegistry.list_all()) == 0
        assert ModeRegistry.get("mode1") is None
        assert ModeRegistry.get("mode2") is None

    def test_auto_register_disabled(self):
        """Auto-register can be disabled."""
        # TestMode has _auto_register=False
        mode = TestMode(
            name="noauto",
            description="Test",
            tool_groups=ToolGroup.READ,
        )
        # Should not be auto-registered
        assert ModeRegistry.get("noauto") is None

    def test_overwrite_existing(self):
        """Registering same name overwrites previous."""
        mode1 = TestMode(name="test", description="First", tool_groups=ToolGroup.READ)
        mode2 = TestMode(name="test", description="Second", tool_groups=ToolGroup.EDIT)

        ModeRegistry.register(mode1)
        ModeRegistry.register(mode2)

        retrieved = ModeRegistry.get("test")
        assert retrieved.description == "Second"
        assert len(ModeRegistry.list_all()) == 1
