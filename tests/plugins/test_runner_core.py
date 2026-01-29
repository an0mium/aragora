"""Tests for plugin runner core functionality.

Tests the plugin execution environment including:
- PluginContext lifecycle and methods
- PluginResult creation and serialization
- PluginRunner initialization and validation
- PluginRegistry operations
- Sandboxed execution
"""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.plugins.manifest import (
    PluginCapability,
    PluginManifest,
    PluginRequirement,
)
from aragora.plugins.runner import (
    PluginContext,
    PluginRegistry,
    PluginResult,
    PluginRunner,
    get_registry,
    reset_registry,
    run_plugin,
)


# =============================================================================
# PluginContext Tests
# =============================================================================


class TestPluginContext:
    """Tests for plugin execution context."""

    def test_default_context(self):
        """Default context has empty values."""
        ctx = PluginContext()
        assert ctx.input_data == {}
        assert ctx.config == {}
        assert ctx.output == {}
        assert ctx.logs == []
        assert ctx.errors == []

    def test_context_with_input(self):
        """Context can be initialized with input data."""
        ctx = PluginContext(
            input_data={"files": ["test.py"]},
            config={"strict": True},
            working_dir="/tmp/test",
        )
        assert ctx.input_data == {"files": ["test.py"]}
        assert ctx.config == {"strict": True}
        assert ctx.working_dir == "/tmp/test"

    def test_log_method(self):
        """log() adds timestamped messages."""
        ctx = PluginContext()
        ctx.log("Test message")

        assert len(ctx.logs) == 1
        assert "Test message" in ctx.logs[0]
        # Check timestamp format
        assert "[" in ctx.logs[0] and "]" in ctx.logs[0]

    def test_log_multiple_messages(self):
        """Multiple log messages are preserved in order."""
        ctx = PluginContext()
        ctx.log("First message")
        ctx.log("Second message")
        ctx.log("Third message")

        assert len(ctx.logs) == 3
        assert "First message" in ctx.logs[0]
        assert "Second message" in ctx.logs[1]
        assert "Third message" in ctx.logs[2]

    def test_error_method(self):
        """error() adds error messages."""
        ctx = PluginContext()
        ctx.error("Something went wrong")

        assert len(ctx.errors) == 1
        assert "Something went wrong" in ctx.errors[0]

    def test_error_multiple(self):
        """Multiple errors are preserved."""
        ctx = PluginContext()
        ctx.error("Error 1")
        ctx.error("Error 2")

        assert len(ctx.errors) == 2

    def test_set_output_method(self):
        """set_output() stores key-value pairs."""
        ctx = PluginContext()
        ctx.set_output("result", {"score": 95})
        ctx.set_output("status", "success")

        assert ctx.output["result"] == {"score": 95}
        assert ctx.output["status"] == "success"

    def test_set_output_overwrites(self):
        """set_output() overwrites existing keys."""
        ctx = PluginContext()
        ctx.set_output("key", "value1")
        ctx.set_output("key", "value2")

        assert ctx.output["key"] == "value2"

    def test_can_method_allowed(self):
        """can() returns True for allowed operations."""
        ctx = PluginContext()
        ctx.allowed_operations = {"read_files", "network"}

        assert ctx.can("read_files") is True
        assert ctx.can("network") is True

    def test_can_method_denied(self):
        """can() returns False for disallowed operations."""
        ctx = PluginContext()
        ctx.allowed_operations = {"read_files"}

        assert ctx.can("write_files") is False
        assert ctx.can("run_commands") is False

    def test_context_with_debate_id(self):
        """Context can store debate context."""
        ctx = PluginContext(
            debate_id="debate-123",
            claim_id="claim-456",
        )
        assert ctx.debate_id == "debate-123"
        assert ctx.claim_id == "claim-456"

    def test_context_timeout(self):
        """Context can specify timeout."""
        ctx = PluginContext(timeout_seconds=30.0)
        assert ctx.timeout_seconds == 30.0

    def test_context_resource_limits(self):
        """Context can specify resource limits."""
        ctx = PluginContext(resource_limits={"max_memory_mb": 512})
        assert ctx.resource_limits["max_memory_mb"] == 512


# =============================================================================
# PluginResult Tests
# =============================================================================


class TestPluginResult:
    """Tests for plugin execution results."""

    def test_success_result(self):
        """Success result has correct fields."""
        result = PluginResult(
            success=True,
            output={"findings": []},
            logs=["Completed analysis"],
        )
        assert result.success is True
        assert result.output == {"findings": []}
        assert result.logs == ["Completed analysis"]
        assert result.errors == []

    def test_failure_result(self):
        """Failure result captures errors."""
        result = PluginResult(
            success=False,
            errors=["Import failed", "Missing dependency"],
        )
        assert result.success is False
        assert len(result.errors) == 2

    def test_result_metrics(self):
        """Result tracks execution metrics."""
        result = PluginResult(
            success=True,
            duration_seconds=1.5,
            memory_used_mb=128.0,
        )
        assert result.duration_seconds == 1.5
        assert result.memory_used_mb == 128.0

    def test_result_plugin_info(self):
        """Result includes plugin metadata."""
        result = PluginResult(
            success=True,
            plugin_name="security-scan",
            plugin_version="1.2.0",
        )
        assert result.plugin_name == "security-scan"
        assert result.plugin_version == "1.2.0"

    def test_result_executed_at(self):
        """Result has execution timestamp."""
        result = PluginResult(success=True)
        assert result.executed_at is not None
        # Should be ISO format
        datetime.fromisoformat(result.executed_at)

    def test_result_to_dict(self):
        """Result converts to dictionary."""
        result = PluginResult(
            success=True,
            output={"count": 5},
            logs=["Log 1"],
            errors=[],
            duration_seconds=0.5,
            memory_used_mb=64.0,
            plugin_name="lint",
            plugin_version="1.0.0",
        )
        data = result.to_dict()

        assert data["success"] is True
        assert data["output"] == {"count": 5}
        assert data["logs"] == ["Log 1"]
        assert data["errors"] == []
        assert data["duration_seconds"] == 0.5
        assert data["memory_used_mb"] == 64.0
        assert data["plugin_name"] == "lint"
        assert data["plugin_version"] == "1.0.0"
        assert "executed_at" in data

    def test_result_default_values(self):
        """Result has sensible defaults."""
        result = PluginResult(success=False)
        assert result.output == {}
        assert result.logs == []
        assert result.errors == []
        assert result.duration_seconds == 0.0
        assert result.memory_used_mb == 0.0
        assert result.plugin_name == ""
        assert result.plugin_version == ""


# =============================================================================
# PluginRunner Initialization Tests
# =============================================================================


class TestPluginRunnerInit:
    """Tests for plugin runner initialization."""

    def test_runner_with_manifest(self):
        """Runner initializes with manifest."""
        manifest = PluginManifest(
            name="test-plugin",
            entry_point="test:run",
        )
        runner = PluginRunner(manifest)

        assert runner.manifest == manifest
        assert runner.sandbox_level == "standard"

    def test_runner_sandbox_levels(self):
        """Runner accepts different sandbox levels."""
        manifest = PluginManifest(name="test", entry_point="test:run")

        strict = PluginRunner(manifest, sandbox_level="strict")
        assert strict.sandbox_level == "strict"

        permissive = PluginRunner(manifest, sandbox_level="permissive")
        assert permissive.sandbox_level == "permissive"

    def test_runner_restricted_builtins(self):
        """Runner has restricted builtins list."""
        manifest = PluginManifest(name="test", entry_point="test:run")
        runner = PluginRunner(manifest)

        # Safe builtins should be allowed
        assert "len" in runner.RESTRICTED_BUILTINS
        assert "str" in runner.RESTRICTED_BUILTINS
        assert "dict" in runner.RESTRICTED_BUILTINS
        assert "list" in runner.RESTRICTED_BUILTINS

        # Dangerous builtins should NOT be allowed
        assert "exec" not in runner.RESTRICTED_BUILTINS
        assert "eval" not in runner.RESTRICTED_BUILTINS
        assert "open" not in runner.RESTRICTED_BUILTINS
        assert "__import__" not in runner.RESTRICTED_BUILTINS


# =============================================================================
# PluginRunner Validation Tests
# =============================================================================


class TestPluginRunnerValidation:
    """Tests for plugin requirement validation."""

    def test_validate_no_requirements(self):
        """Plugin with no requirements passes validation."""
        manifest = PluginManifest(
            name="simple",
            entry_point="test:run",
            requirements=[],
        )
        runner = PluginRunner(manifest)
        valid, missing = runner._validate_requirements()

        assert valid is True
        assert len(missing) == 0

    def test_validate_missing_package(self):
        """Missing Python package is detected."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            python_packages=["nonexistent_package_xyz_123"],
        )
        runner = PluginRunner(manifest)
        valid, missing = runner._validate_requirements()

        assert valid is False
        assert any("nonexistent_package_xyz_123" in m for m in missing)

    def test_validate_available_package(self):
        """Available Python package passes."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            python_packages=["pytest"],  # Should be installed
        )
        runner = PluginRunner(manifest)
        valid, missing = runner._validate_requirements()

        assert valid is True

    def test_validate_missing_tool(self):
        """Missing system tool is detected."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            system_tools=["nonexistent_tool_xyz"],
        )
        runner = PluginRunner(manifest)
        valid, missing = runner._validate_requirements()

        assert valid is False
        assert any("nonexistent_tool_xyz" in m for m in missing)


# =============================================================================
# PluginRunner Entry Point Tests
# =============================================================================


class TestPluginRunnerEntryPoint:
    """Tests for entry point loading."""

    def test_invalid_entry_point_format(self):
        """Invalid entry point format raises error."""
        manifest = PluginManifest(
            name="test",
            entry_point="no_colon_here",
        )
        runner = PluginRunner(manifest)

        with pytest.raises(ValueError, match="Invalid entry point"):
            runner._load_entry_point()

    def test_missing_module(self):
        """Missing module raises ImportError."""
        manifest = PluginManifest(
            name="test",
            entry_point="nonexistent.module:run",
        )
        runner = PluginRunner(manifest)

        with pytest.raises(ImportError):
            runner._load_entry_point()


# =============================================================================
# PluginRunner Execution Tests
# =============================================================================


class TestPluginRunnerExecution:
    """Tests for plugin execution."""

    @pytest.mark.asyncio
    async def test_run_missing_requirements(self):
        """Run fails gracefully with missing requirements."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            python_packages=["nonexistent_xyz_123"],
        )
        runner = PluginRunner(manifest)
        context = PluginContext()

        result = await runner.run(context)

        assert result.success is False
        assert any("Missing requirements" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_run_missing_entry_point(self):
        """Run fails gracefully with missing entry point."""
        manifest = PluginManifest(
            name="test",
            entry_point="nonexistent.module:func",
        )
        runner = PluginRunner(manifest)
        context = PluginContext()

        result = await runner.run(context)

        assert result.success is False
        assert any("Failed to load" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_run_sets_allowed_operations(self):
        """Run sets allowed operations from manifest."""
        manifest = PluginManifest(
            name="test",
            entry_point="nonexistent:run",  # Will fail, but operations are set first
            requirements=[
                PluginRequirement.READ_FILES,
                PluginRequirement.NETWORK,
            ],
        )
        runner = PluginRunner(manifest)
        context = PluginContext()

        # Run will fail, but operations should be set
        await runner.run(context)

        assert "read_files" in context.allowed_operations
        assert "network" in context.allowed_operations

    @pytest.mark.asyncio
    async def test_run_records_duration(self):
        """Run records execution duration."""
        manifest = PluginManifest(
            name="test",
            entry_point="nonexistent:run",
        )
        runner = PluginRunner(manifest)
        context = PluginContext()

        result = await runner.run(context)

        # Duration should be recorded even on failure
        assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_run_with_timeout_override(self):
        """Run respects timeout override."""
        manifest = PluginManifest(
            name="test",
            entry_point="nonexistent:run",
            timeout_seconds=60.0,
        )
        runner = PluginRunner(manifest)
        context = PluginContext()

        # Should use override
        result = await runner.run(context, timeout_override=5.0)

        # At minimum, plugin name should be set
        assert result.plugin_name == "test"


# =============================================================================
# PluginRegistry Tests
# =============================================================================


class TestPluginRegistry:
    """Tests for plugin registry."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_registry()

    def test_registry_loads_builtins(self):
        """Registry loads built-in plugins on init."""
        registry = PluginRegistry()

        assert "lint" in registry.manifests
        assert "security-scan" in registry.manifests
        assert "test-runner" in registry.manifests

    def test_registry_get_plugin(self):
        """Can get plugin manifest by name."""
        registry = PluginRegistry()
        lint = registry.get("lint")

        assert lint is not None
        assert lint.name == "lint"

    def test_registry_get_unknown(self):
        """Unknown plugin returns None."""
        registry = PluginRegistry()
        result = registry.get("nonexistent")

        assert result is None

    def test_registry_list_plugins(self):
        """Can list all plugins."""
        registry = PluginRegistry()
        plugins = registry.list_plugins()

        assert len(plugins) >= 3
        names = [p.name for p in plugins]
        assert "lint" in names

    def test_registry_list_by_capability(self):
        """Can list plugins by capability."""
        registry = PluginRegistry()

        lint_plugins = registry.list_by_capability(PluginCapability.LINT)
        assert len(lint_plugins) >= 1
        assert all(p.has_capability(PluginCapability.LINT) for p in lint_plugins)

        security_plugins = registry.list_by_capability(PluginCapability.SECURITY_SCAN)
        assert len(security_plugins) >= 1

    def test_registry_get_runner(self):
        """Can get runner for plugin."""
        registry = PluginRegistry()
        runner = registry.get_runner("lint")

        assert runner is not None
        assert runner.manifest.name == "lint"

    def test_registry_get_runner_cached(self):
        """Runner is cached on second call."""
        registry = PluginRegistry()
        runner1 = registry.get_runner("lint")
        runner2 = registry.get_runner("lint")

        assert runner1 is runner2  # Same instance

    def test_registry_get_runner_unknown(self):
        """Unknown plugin returns None for runner."""
        registry = PluginRegistry()
        runner = registry.get_runner("nonexistent")

        assert runner is None

    @pytest.mark.asyncio
    async def test_registry_run_plugin(self):
        """Can run plugin through registry."""
        registry = PluginRegistry()

        # Will fail because lint isn't actually runnable in tests,
        # but should return a PluginResult
        result = await registry.run_plugin(
            "lint",
            input_data={"files": ["test.py"]},
        )

        assert isinstance(result, PluginResult)

    @pytest.mark.asyncio
    async def test_registry_run_unknown_plugin(self):
        """Running unknown plugin returns error result."""
        registry = PluginRegistry()

        result = await registry.run_plugin(
            "nonexistent",
            input_data={},
        )

        assert result.success is False
        assert any("not found" in e.lower() for e in result.errors)


# =============================================================================
# Plugin Discovery Tests
# =============================================================================


class TestPluginDiscovery:
    """Tests for plugin directory discovery."""

    def test_discover_empty_dirs(self, tmp_path):
        """Discover handles empty directories."""
        registry = PluginRegistry(plugin_dirs=[tmp_path / "plugins"])
        registry.discover()

        # Should still have built-ins
        assert len(registry.manifests) >= 3

    def test_discover_nonexistent_dir(self, tmp_path):
        """Discover handles nonexistent directories."""
        registry = PluginRegistry(plugin_dirs=[tmp_path / "nonexistent"])
        registry.discover()  # Should not raise

        # Should still have built-ins
        assert len(registry.manifests) >= 3

    def test_discover_valid_plugin(self, tmp_path):
        """Discover loads valid plugin manifests."""
        # Create plugin directory structure
        plugin_dir = tmp_path / "plugins"
        my_plugin_dir = plugin_dir / "my-plugin"
        my_plugin_dir.mkdir(parents=True)

        # Write manifest
        manifest_path = my_plugin_dir / "manifest.json"
        manifest_path.write_text(
            """{
            "name": "my-plugin",
            "version": "1.0.0",
            "entry_point": "my_plugin:run",
            "capabilities": ["custom"]
        }"""
        )

        registry = PluginRegistry(plugin_dirs=[plugin_dir])
        registry.discover()

        assert "my-plugin" in registry.manifests

    def test_discover_invalid_manifest_skipped(self, tmp_path):
        """Discover skips invalid manifests."""
        plugin_dir = tmp_path / "plugins"
        bad_plugin_dir = plugin_dir / "bad-plugin"
        bad_plugin_dir.mkdir(parents=True)

        # Write invalid JSON
        manifest_path = bad_plugin_dir / "manifest.json"
        manifest_path.write_text("not valid json")

        registry = PluginRegistry(plugin_dirs=[plugin_dir])
        registry.discover()  # Should not raise

        # Bad plugin should not be loaded
        assert "bad-plugin" not in registry.manifests


# =============================================================================
# Global Registry Functions Tests
# =============================================================================


class TestGlobalRegistryFunctions:
    """Tests for global registry functions."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_registry()

    def test_get_registry(self):
        """get_registry returns a PluginRegistry."""
        registry = get_registry()
        assert isinstance(registry, PluginRegistry)

    def test_get_registry_singleton(self):
        """get_registry returns same instance."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2

    def test_reset_registry(self):
        """reset_registry clears the global registry."""
        registry1 = get_registry()
        reset_registry()
        registry2 = get_registry()

        # Should be different instances after reset
        assert registry1 is not registry2

    @pytest.mark.asyncio
    async def test_run_plugin_function(self):
        """run_plugin function uses global registry."""
        reset_registry()

        # Will fail but should return result
        result = await run_plugin("lint", {"files": []})
        assert isinstance(result, PluginResult)


# =============================================================================
# Safe Open Tests
# =============================================================================


class TestSafeOpen:
    """Tests for safe file open functionality."""

    def test_safe_open_within_working_dir(self, tmp_path):
        """Safe open allows files within working directory."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            requirements=[PluginRequirement.READ_FILES],
        )
        runner = PluginRunner(manifest)
        context = PluginContext(working_dir=str(tmp_path))
        context.allowed_operations = {"read_files"}

        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        safe_open = runner._make_safe_open(context, read_only=True)

        with safe_open(str(test_file), "r") as f:
            content = f.read()
            assert content == "test content"

    def test_safe_open_blocks_write_in_read_only(self, tmp_path):
        """Safe open blocks write operations in read-only mode."""
        manifest = PluginManifest(name="test", entry_point="test:run")
        runner = PluginRunner(manifest)
        context = PluginContext(working_dir=str(tmp_path))

        safe_open = runner._make_safe_open(context, read_only=True)

        with pytest.raises(PermissionError, match="Write operations"):
            safe_open(str(tmp_path / "test.txt"), "w")

    def test_safe_open_blocks_outside_working_dir(self, tmp_path):
        """Safe open blocks access outside working directory."""
        manifest = PluginManifest(name="test", entry_point="test:run")
        runner = PluginRunner(manifest)
        context = PluginContext(working_dir=str(tmp_path / "subdir"))

        safe_open = runner._make_safe_open(context, read_only=True)

        with pytest.raises(PermissionError, match="outside working directory"):
            safe_open("/etc/passwd", "r")
