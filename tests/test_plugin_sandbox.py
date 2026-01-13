"""
Plugin Sandbox Security Tests.

Tests for the sandboxed plugin execution environment to ensure:
- Restricted builtins prevent dangerous operations
- Path traversal attacks are blocked
- Timeout enforcement works
- Memory limits are respected
- Capability-based permissions work correctly
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.plugins.runner import (
    PluginRunner,
    PluginContext,
    PluginResult,
)
from aragora.plugins.manifest import (
    PluginManifest,
    PluginCapability,
    PluginRequirement,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_manifest():
    """Create a basic plugin manifest for testing."""
    return PluginManifest(
        name="test-plugin",
        version="1.0.0",
        description="Test plugin",
        entry_point="test_module:run",
        capabilities=[PluginCapability.CODE_ANALYSIS],
        timeout_seconds=5,
    )


@pytest.fixture
def file_read_manifest():
    """Manifest with file read requirement."""
    return PluginManifest(
        name="file-reader",
        version="1.0.0",
        description="File reader plugin",
        entry_point="test_module:run",
        capabilities=[PluginCapability.CODE_ANALYSIS],
        requirements=[PluginRequirement.READ_FILES],
        timeout_seconds=5,
    )


@pytest.fixture
def file_write_manifest():
    """Manifest with file write requirement."""
    return PluginManifest(
        name="file-writer",
        version="1.0.0",
        description="File writer plugin",
        entry_point="test_module:run",
        capabilities=[PluginCapability.FORMATTER],
        requirements=[PluginRequirement.WRITE_FILES],
        timeout_seconds=5,
    )


@pytest.fixture
def basic_context(tmp_path):
    """Create a basic plugin context."""
    return PluginContext(
        input_data={"test": "data"},
        config={"key": "value"},
        working_dir=str(tmp_path),
        debate_id="test-debate-123",
    )


# =============================================================================
# Restricted Builtins Tests
# =============================================================================


class TestRestrictedBuiltins:
    """Test that dangerous builtins are not available."""

    def test_restricted_builtins_list(self, basic_manifest):
        """Verify restricted builtins list contains safe functions."""
        runner = PluginRunner(basic_manifest)

        # Safe builtins should be present
        safe_funcs = ["len", "str", "int", "list", "dict", "print", "range"]
        for func in safe_funcs:
            assert func in runner.RESTRICTED_BUILTINS

    def test_dangerous_builtins_excluded(self, basic_manifest):
        """Verify dangerous builtins are excluded."""
        runner = PluginRunner(basic_manifest)

        # Dangerous functions should NOT be in restricted list
        dangerous = ["eval", "exec", "compile", "__import__", "open"]
        for func in dangerous:
            assert func not in runner.RESTRICTED_BUILTINS

    def test_create_restricted_globals_excludes_dangerous(self, basic_manifest, basic_context):
        """Test that restricted globals don't include dangerous functions."""
        runner = PluginRunner(basic_manifest)
        globals_dict = runner._create_restricted_globals(basic_context)

        builtins = globals_dict.get("__builtins__", {})

        # Should not have dangerous functions
        assert "eval" not in builtins
        assert "exec" not in builtins
        assert "compile" not in builtins
        assert "__import__" not in builtins

    def test_restricted_globals_has_safe_functions(self, basic_manifest, basic_context):
        """Test that restricted globals have safe functions."""
        runner = PluginRunner(basic_manifest)
        globals_dict = runner._create_restricted_globals(basic_context)

        builtins = globals_dict.get("__builtins__", {})

        # Should have safe functions
        assert "len" in builtins
        assert "str" in builtins
        assert "print" in builtins
        assert "range" in builtins

    def test_context_available_in_globals(self, basic_manifest, basic_context):
        """Test that context is available in globals."""
        runner = PluginRunner(basic_manifest)
        globals_dict = runner._create_restricted_globals(basic_context)

        assert "context" in globals_dict
        assert globals_dict["context"] is basic_context


# =============================================================================
# Path Traversal Security Tests
# =============================================================================


class TestPathTraversalPrevention:
    """Test that path traversal attacks are blocked."""

    def test_safe_open_blocks_parent_traversal(self, file_read_manifest, tmp_path):
        """Test that ../.. path traversal is blocked."""
        context = PluginContext(
            working_dir=str(tmp_path),
            allowed_operations={"read_files"},
        )

        runner = PluginRunner(file_read_manifest)
        safe_open = runner._make_safe_open(context, read_only=True)

        # Attempt to read outside working directory
        with pytest.raises(PermissionError, match="outside working directory"):
            safe_open("../../../etc/passwd")

    def test_safe_open_blocks_absolute_path_escape(self, file_read_manifest, tmp_path):
        """Test that absolute paths outside working dir are blocked."""
        context = PluginContext(
            working_dir=str(tmp_path),
            allowed_operations={"read_files"},
        )

        runner = PluginRunner(file_read_manifest)
        safe_open = runner._make_safe_open(context, read_only=True)

        # Attempt to read from absolute path outside working dir
        with pytest.raises(PermissionError, match="outside working directory"):
            safe_open("/etc/passwd")

    def test_safe_open_blocks_symlink_escape(self, file_read_manifest, tmp_path):
        """Test that symlink-based escapes are blocked."""
        # Create a symlink pointing outside working dir
        secret_file = Path("/etc/hostname")  # Common readable file
        if not secret_file.exists():
            pytest.skip("Test requires /etc/hostname")

        link_path = tmp_path / "sneaky_link"
        try:
            link_path.symlink_to("/etc")
        except OSError:
            pytest.skip("Cannot create symlinks")

        context = PluginContext(
            working_dir=str(tmp_path),
            allowed_operations={"read_files"},
        )

        runner = PluginRunner(file_read_manifest)
        safe_open = runner._make_safe_open(context, read_only=True)

        # Attempt to read via symlink
        with pytest.raises(PermissionError, match="outside working directory"):
            safe_open("sneaky_link/hostname")

    def test_safe_open_allows_valid_subpath(self, file_read_manifest, tmp_path):
        """Test that valid paths within working dir are allowed."""
        # Create a test file
        test_file = tmp_path / "subdir" / "test.txt"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("test content")

        context = PluginContext(
            working_dir=str(tmp_path),
            allowed_operations={"read_files"},
        )

        runner = PluginRunner(file_read_manifest)
        safe_open = runner._make_safe_open(context, read_only=True)

        # Should be able to read file in working directory (use absolute path)
        with safe_open(str(test_file)) as f:
            content = f.read()

        assert content == "test content"

    def test_safe_open_read_only_blocks_write(self, file_read_manifest, tmp_path):
        """Test that read-only mode blocks write operations."""
        # Create a test file first
        test_file = tmp_path / "test.txt"
        test_file.write_text("existing content")

        context = PluginContext(
            working_dir=str(tmp_path),
            allowed_operations={"read_files"},
        )

        runner = PluginRunner(file_read_manifest)
        safe_open = runner._make_safe_open(context, read_only=True)

        # Attempt to open for writing (use absolute path)
        with pytest.raises(PermissionError, match="Write operations not permitted"):
            safe_open(str(test_file), "w")

        with pytest.raises(PermissionError, match="Write operations not permitted"):
            safe_open(str(test_file), "a")

        new_file = tmp_path / "new.txt"
        with pytest.raises(PermissionError, match="Write operations not permitted"):
            safe_open(str(new_file), "x")

    def test_safe_open_write_mode_allows_write(self, file_write_manifest, tmp_path):
        """Test that write mode allows write operations."""
        context = PluginContext(
            working_dir=str(tmp_path),
            allowed_operations={"write_files"},
        )

        runner = PluginRunner(file_write_manifest)
        safe_open = runner._make_safe_open(context, read_only=False)

        # Should be able to write (use absolute path)
        output_file = tmp_path / "output.txt"
        with safe_open(str(output_file), "w") as f:
            f.write("test output")

        # Verify content
        assert output_file.read_text() == "test output"


# =============================================================================
# Capability Tests
# =============================================================================


class TestCapabilities:
    """Test capability-based permission system."""

    def test_context_can_check_allowed_operation(self, tmp_path):
        """Test that context.can() correctly checks operations."""
        context = PluginContext(
            working_dir=str(tmp_path),
            allowed_operations={"read_files", "read_debate"},
        )

        assert context.can("read_files") is True
        assert context.can("read_debate") is True
        assert context.can("write_files") is False
        assert context.can("execute_code") is False

    def test_restricted_globals_no_open_without_capability(self, basic_manifest, basic_context):
        """Test that open is not available without file capabilities."""
        # basic_manifest has no file capabilities
        runner = PluginRunner(basic_manifest)
        globals_dict = runner._create_restricted_globals(basic_context)

        builtins = globals_dict.get("__builtins__", {})
        assert "open" not in builtins

    def test_restricted_globals_has_open_with_read_capability(self, file_read_manifest, tmp_path):
        """Test that open is available with read_files capability."""
        context = PluginContext(
            working_dir=str(tmp_path),
            allowed_operations={"read_files"},
        )

        runner = PluginRunner(file_read_manifest)
        globals_dict = runner._create_restricted_globals(context)

        builtins = globals_dict.get("__builtins__", {})
        assert "open" in builtins

    def test_capability_enum_values(self):
        """Test that all expected capabilities exist."""
        expected_capabilities = [
            "CODE_ANALYSIS",
            "LINT",
            "SECURITY_SCAN",
            "TEST_RUNNER",
            "FORMATTER",
        ]

        for cap in expected_capabilities:
            assert hasattr(PluginCapability, cap), f"Missing capability: {cap}"

        expected_requirements = [
            "READ_FILES",
            "WRITE_FILES",
            "RUN_COMMANDS",
            "NETWORK",
        ]

        for req in expected_requirements:
            assert hasattr(PluginRequirement, req), f"Missing requirement: {req}"


# =============================================================================
# Plugin Context Tests
# =============================================================================


class TestPluginContext:
    """Test PluginContext functionality."""

    def test_context_log_adds_timestamp(self, basic_context):
        """Test that log() adds timestamped messages."""
        basic_context.log("Test message")

        assert len(basic_context.logs) == 1
        assert "Test message" in basic_context.logs[0]
        # Should have ISO timestamp
        assert "T" in basic_context.logs[0]  # ISO format has T separator

    def test_context_error_adds_to_errors(self, basic_context):
        """Test that error() adds to errors list."""
        basic_context.error("Something went wrong")

        assert len(basic_context.errors) == 1
        assert basic_context.errors[0] == "Something went wrong"

    def test_context_set_output(self, basic_context):
        """Test that set_output() sets values."""
        basic_context.set_output("result", {"key": "value"})

        assert basic_context.output["result"] == {"key": "value"}

    def test_context_multiple_outputs(self, basic_context):
        """Test multiple output values."""
        basic_context.set_output("score", 0.95)
        basic_context.set_output("category", "high")
        basic_context.set_output("details", ["a", "b"])

        assert basic_context.output["score"] == 0.95
        assert basic_context.output["category"] == "high"
        assert basic_context.output["details"] == ["a", "b"]


# =============================================================================
# Plugin Result Tests
# =============================================================================


class TestPluginResult:
    """Test PluginResult functionality."""

    def test_result_to_dict(self):
        """Test result serialization to dict."""
        result = PluginResult(
            success=True,
            output={"key": "value"},
            logs=["log1", "log2"],
            errors=[],
            duration_seconds=1.5,
            memory_used_mb=50.0,
            plugin_name="test-plugin",
            plugin_version="1.0.0",
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["output"] == {"key": "value"}
        assert data["logs"] == ["log1", "log2"]
        assert data["errors"] == []
        assert data["duration_seconds"] == 1.5
        assert data["memory_used_mb"] == 50.0
        assert data["plugin_name"] == "test-plugin"
        assert data["plugin_version"] == "1.0.0"
        assert "executed_at" in data

    def test_result_default_values(self):
        """Test default values for result."""
        result = PluginResult(success=False)

        assert result.success is False
        assert result.output == {}
        assert result.logs == []
        assert result.errors == []
        assert result.duration_seconds == 0.0
        assert result.memory_used_mb == 0.0


# =============================================================================
# Requirement Validation Tests
# =============================================================================


class TestRequirementValidation:
    """Test plugin requirement validation."""

    def test_validate_python_package_exists(self, basic_manifest):
        """Test validation of existing Python package."""
        basic_manifest.python_packages = ["json"]  # Built-in

        runner = PluginRunner(basic_manifest)
        valid, missing = runner._validate_requirements()

        assert valid is True
        assert len(missing) == 0

    def test_validate_python_package_missing(self, basic_manifest):
        """Test validation of missing Python package."""
        basic_manifest.python_packages = ["nonexistent_package_xyz"]

        runner = PluginRunner(basic_manifest)
        valid, missing = runner._validate_requirements()

        assert valid is False
        assert len(missing) == 1
        assert "nonexistent_package_xyz" in missing[0]

    def test_validate_system_tool_exists(self, basic_manifest):
        """Test validation of existing system tool."""
        basic_manifest.system_tools = ["python"]  # Should exist

        runner = PluginRunner(basic_manifest)
        valid, missing = runner._validate_requirements()

        assert valid is True
        assert len(missing) == 0

    def test_validate_system_tool_missing(self, basic_manifest):
        """Test validation of missing system tool."""
        basic_manifest.system_tools = ["nonexistent_tool_xyz"]

        runner = PluginRunner(basic_manifest)
        valid, missing = runner._validate_requirements()

        assert valid is False
        assert len(missing) == 1
        assert "nonexistent_tool_xyz" in missing[0]


# =============================================================================
# Entry Point Loading Tests
# =============================================================================


class TestEntryPointLoading:
    """Test plugin entry point loading."""

    def test_invalid_entry_point_format(self, basic_manifest):
        """Test that invalid entry point format raises error."""
        basic_manifest.entry_point = "invalid_format_no_colon"

        runner = PluginRunner(basic_manifest)

        with pytest.raises(ValueError, match="Invalid entry point"):
            runner._load_entry_point()

    def test_missing_module_raises_import_error(self, basic_manifest):
        """Test that missing module raises ImportError."""
        basic_manifest.entry_point = "nonexistent_module_xyz:run"

        runner = PluginRunner(basic_manifest)

        with pytest.raises(ImportError, match="Failed to load"):
            runner._load_entry_point()

    def test_missing_function_raises_import_error(self, basic_manifest):
        """Test that missing function raises ImportError."""
        basic_manifest.entry_point = "json:nonexistent_function"

        runner = PluginRunner(basic_manifest)

        with pytest.raises(ImportError, match="Failed to load"):
            runner._load_entry_point()


# =============================================================================
# Sandbox Level Tests
# =============================================================================


class TestSandboxLevels:
    """Test different sandbox levels."""

    def test_sandbox_level_strict(self, basic_manifest):
        """Test strict sandbox level."""
        runner = PluginRunner(basic_manifest, sandbox_level="strict")
        assert runner.sandbox_level == "strict"

    def test_sandbox_level_standard(self, basic_manifest):
        """Test standard sandbox level."""
        runner = PluginRunner(basic_manifest, sandbox_level="standard")
        assert runner.sandbox_level == "standard"

    def test_sandbox_level_permissive(self, basic_manifest):
        """Test permissive sandbox level."""
        runner = PluginRunner(basic_manifest, sandbox_level="permissive")
        assert runner.sandbox_level == "permissive"


# =============================================================================
# Resource Limit Tests
# =============================================================================


class TestResourceLimits:
    """Test resource limit enforcement."""

    @pytest.mark.skipif(sys.platform == "win32", reason="Resource limits not supported on Windows")
    def test_set_resource_limits_called(self, basic_manifest):
        """Test that resource limits are set on Unix."""
        runner = PluginRunner(basic_manifest)

        with patch("resource.setrlimit") as mock_setrlimit:
            runner._set_resource_limits()

            # Should have been called for memory limit
            mock_setrlimit.assert_called()

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
    def test_set_resource_limits_skipped_windows(self, basic_manifest):
        """Test that resource limits are skipped on Windows."""
        runner = PluginRunner(basic_manifest)

        # Should not raise on Windows
        runner._set_resource_limits()


# =============================================================================
# Plugin Manifest Validation Tests
# =============================================================================


class TestPluginManifest:
    """Test plugin manifest validation."""

    def test_manifest_required_fields(self):
        """Test that required fields are validated."""
        manifest = PluginManifest(
            name="test",
            version="1.0.0",
            description="Test",
            entry_point="module:func",
        )

        assert manifest.name == "test"
        assert manifest.version == "1.0.0"
        assert manifest.entry_point == "module:func"

    def test_manifest_default_values(self):
        """Test manifest default values."""
        manifest = PluginManifest(
            name="test",
            version="1.0.0",
            description="Test",
            entry_point="module:func",
        )

        assert manifest.timeout_seconds == 60.0  # Default from manifest.py
        assert manifest.capabilities == []
        assert manifest.requirements == []


# =============================================================================
# Integration Tests
# =============================================================================


class TestPluginRunnerIntegration:
    """Integration tests for the full plugin runner flow."""

    @pytest.mark.asyncio
    async def test_run_with_mocked_entry_point(self, basic_manifest, basic_context):
        """Test running a plugin with mocked entry point."""
        runner = PluginRunner(basic_manifest)

        # Mock the entry point function
        async def mock_entry(ctx):
            ctx.set_output("result", "success")
            ctx.log("Plugin executed")
            return True

        with patch.object(runner, "_load_entry_point", return_value=mock_entry):
            with patch.object(runner, "_validate_requirements", return_value=(True, [])):
                result = await runner.run(basic_context)

        # Plugin should have executed
        assert result.plugin_name == "test-plugin"
        assert result.plugin_version == "1.0.0"

    @pytest.mark.asyncio
    async def test_run_with_missing_requirements(self, basic_manifest, basic_context):
        """Test running a plugin with missing requirements."""
        basic_manifest.python_packages = ["nonexistent_xyz"]

        runner = PluginRunner(basic_manifest)
        result = await runner.run(basic_context)

        # Should fail due to missing requirements
        assert result.success is False
        assert len(result.errors) > 0


# =============================================================================
# Security Edge Cases
# =============================================================================


class TestSecurityEdgeCases:
    """Test security edge cases and attack vectors."""

    def test_null_byte_injection_blocked(self, file_read_manifest, tmp_path):
        """Test that null byte injection is handled."""
        context = PluginContext(
            working_dir=str(tmp_path),
            allowed_operations={"read_files"},
        )

        runner = PluginRunner(file_read_manifest)
        safe_open = runner._make_safe_open(context, read_only=True)

        # Null byte injection attempt
        with pytest.raises((PermissionError, ValueError, OSError)):
            safe_open("file.txt\x00/etc/passwd")

    def test_unicode_path_normalization(self, file_read_manifest, tmp_path):
        """Test that unicode paths are properly normalized."""
        context = PluginContext(
            working_dir=str(tmp_path),
            allowed_operations={"read_files"},
        )

        runner = PluginRunner(file_read_manifest)
        safe_open = runner._make_safe_open(context, read_only=True)

        # Unicode normalization attack (fullwidth characters)
        with pytest.raises((PermissionError, FileNotFoundError, OSError)):
            safe_open("..／..／etc／passwd")  # Fullwidth solidus

    def test_double_encoding_blocked(self, file_read_manifest, tmp_path):
        """Test that double-encoded paths are blocked."""
        context = PluginContext(
            working_dir=str(tmp_path),
            allowed_operations={"read_files"},
        )

        runner = PluginRunner(file_read_manifest)
        safe_open = runner._make_safe_open(context, read_only=True)

        # Should not resolve %2e%2e as ..
        with pytest.raises((PermissionError, FileNotFoundError)):
            safe_open("%2e%2e/%2e%2e/etc/passwd")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
