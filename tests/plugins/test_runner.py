"""Comprehensive tests for plugin runner module.

Tests the plugin execution environment including:
- PluginContext lifecycle and methods
- PluginResult creation and serialization
- PluginRunner initialization, validation, and execution
- PluginRegistry operations and discovery
- Sandboxed execution and safety
- Global registry functions
- Safe file operations
"""

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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
    RESOURCE_AVAILABLE,
)


# =============================================================================
# PluginContext Tests - Extended Coverage
# =============================================================================


class TestPluginContextExtended:
    """Extended tests for plugin execution context."""

    def test_context_with_all_fields(self):
        """Context can be initialized with all fields."""
        ctx = PluginContext(
            input_data={"key": "value"},
            config={"setting": True},
            working_dir="/tmp/work",
            debate_id="debate-001",
            claim_id="claim-001",
            timeout_seconds=120.0,
            cleanup_on_error=True,
            resource_limits={"max_memory_mb": 1024},
            allowed_operations={"read_files", "network"},
            output={"existing": "data"},
            logs=["previous log"],
            errors=["previous error"],
        )

        assert ctx.input_data == {"key": "value"}
        assert ctx.config == {"setting": True}
        assert ctx.working_dir == "/tmp/work"
        assert ctx.debate_id == "debate-001"
        assert ctx.claim_id == "claim-001"
        assert ctx.timeout_seconds == 120.0
        assert ctx.cleanup_on_error is True
        assert ctx.resource_limits == {"max_memory_mb": 1024}
        assert ctx.allowed_operations == {"read_files", "network"}
        assert ctx.output == {"existing": "data"}
        assert len(ctx.logs) == 1
        assert len(ctx.errors) == 1

    def test_log_timestamp_format(self):
        """log() includes ISO format timestamp."""
        ctx = PluginContext()
        before = datetime.now()
        ctx.log("Test message")
        after = datetime.now()

        # Extract timestamp from log
        log_entry = ctx.logs[0]
        assert log_entry.startswith("[")
        timestamp_str = log_entry.split("]")[0][1:]
        log_time = datetime.fromisoformat(timestamp_str)

        assert before <= log_time <= after

    def test_can_with_empty_operations(self):
        """can() returns False when no operations are allowed."""
        ctx = PluginContext()
        ctx.allowed_operations = set()

        assert ctx.can("read_files") is False
        assert ctx.can("write_files") is False
        assert ctx.can("anything") is False

    def test_set_output_with_complex_values(self):
        """set_output() handles complex values."""
        ctx = PluginContext()

        # Nested dict
        ctx.set_output("nested", {"a": {"b": {"c": 1}}})
        assert ctx.output["nested"]["a"]["b"]["c"] == 1

        # List of dicts
        ctx.set_output("items", [{"id": 1}, {"id": 2}])
        assert len(ctx.output["items"]) == 2

        # None value
        ctx.set_output("null", None)
        assert ctx.output["null"] is None


# =============================================================================
# PluginResult Tests - Extended Coverage
# =============================================================================


class TestPluginResultExtended:
    """Extended tests for plugin execution results."""

    def test_result_with_all_fields(self):
        """Result can be created with all fields."""
        result = PluginResult(
            success=True,
            output={"data": [1, 2, 3]},
            logs=["Log 1", "Log 2"],
            errors=["Warning 1"],
            duration_seconds=2.5,
            memory_used_mb=256.0,
            plugin_name="test-plugin",
            plugin_version="2.0.0",
            executed_at="2024-01-15T12:00:00",
        )

        assert result.success is True
        assert result.output == {"data": [1, 2, 3]}
        assert len(result.logs) == 2
        assert len(result.errors) == 1
        assert result.duration_seconds == 2.5
        assert result.memory_used_mb == 256.0
        assert result.plugin_name == "test-plugin"
        assert result.plugin_version == "2.0.0"
        assert result.executed_at == "2024-01-15T12:00:00"

    def test_result_to_dict_complete(self):
        """to_dict() includes all fields."""
        result = PluginResult(
            success=True,
            output={"result": "ok"},
            logs=["Done"],
            errors=[],
            duration_seconds=1.0,
            memory_used_mb=100.0,
            plugin_name="complete-plugin",
            plugin_version="1.0.0",
        )

        data = result.to_dict()

        # Verify all expected fields
        expected_keys = {
            "success",
            "output",
            "logs",
            "errors",
            "duration_seconds",
            "memory_used_mb",
            "plugin_name",
            "plugin_version",
            "executed_at",
        }
        assert set(data.keys()) == expected_keys

        # Verify serializable
        json_str = json.dumps(data)
        assert "complete-plugin" in json_str

    def test_result_executed_at_auto_generated(self):
        """executed_at is auto-generated if not provided."""
        before = datetime.now()
        result = PluginResult(success=True)
        after = datetime.now()

        executed = datetime.fromisoformat(result.executed_at)
        assert before <= executed <= after


# =============================================================================
# PluginRunner Tests - Extended Coverage
# =============================================================================


class TestPluginRunnerExtended:
    """Extended tests for plugin runner."""

    def test_runner_all_restricted_builtins(self):
        """Runner has comprehensive restricted builtins."""
        manifest = PluginManifest(name="test", entry_point="test:run")
        runner = PluginRunner(manifest)

        # Verify safe math/comparison functions
        assert "abs" in runner.RESTRICTED_BUILTINS
        assert "min" in runner.RESTRICTED_BUILTINS
        assert "max" in runner.RESTRICTED_BUILTINS
        assert "sum" in runner.RESTRICTED_BUILTINS
        assert "pow" in runner.RESTRICTED_BUILTINS
        assert "round" in runner.RESTRICTED_BUILTINS

        # Verify safe type functions
        assert "bool" in runner.RESTRICTED_BUILTINS
        assert "int" in runner.RESTRICTED_BUILTINS
        assert "float" in runner.RESTRICTED_BUILTINS
        assert "complex" in runner.RESTRICTED_BUILTINS

        # Verify safe collection functions
        assert "list" in runner.RESTRICTED_BUILTINS
        assert "tuple" in runner.RESTRICTED_BUILTINS
        assert "dict" in runner.RESTRICTED_BUILTINS
        assert "set" in runner.RESTRICTED_BUILTINS
        assert "frozenset" in runner.RESTRICTED_BUILTINS

        # Verify iteration functions
        assert "iter" in runner.RESTRICTED_BUILTINS
        assert "next" in runner.RESTRICTED_BUILTINS
        assert "range" in runner.RESTRICTED_BUILTINS
        assert "enumerate" in runner.RESTRICTED_BUILTINS
        assert "zip" in runner.RESTRICTED_BUILTINS
        assert "map" in runner.RESTRICTED_BUILTINS
        assert "filter" in runner.RESTRICTED_BUILTINS
        assert "reversed" in runner.RESTRICTED_BUILTINS
        assert "sorted" in runner.RESTRICTED_BUILTINS

    def test_create_restricted_globals(self):
        """_create_restricted_globals creates proper namespace."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            requirements=[PluginRequirement.READ_FILES],
        )
        runner = PluginRunner(manifest)
        context = PluginContext(working_dir="/tmp")
        context.allowed_operations = {"read_files"}

        globals_dict = runner._create_restricted_globals(context)

        assert "__builtins__" in globals_dict
        assert "__name__" in globals_dict
        assert "context" in globals_dict
        assert globals_dict["__name__"] == "test"
        assert globals_dict["context"] is context

        # Should have open because read_files is allowed
        assert "open" in globals_dict["__builtins__"]

    def test_create_restricted_globals_with_write(self):
        """_create_restricted_globals includes open for write_files."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            requirements=[PluginRequirement.WRITE_FILES],
        )
        runner = PluginRunner(manifest)
        context = PluginContext(working_dir="/tmp")
        context.allowed_operations = {"write_files"}

        globals_dict = runner._create_restricted_globals(context)

        # Should have open because write_files is allowed
        assert "open" in globals_dict["__builtins__"]

    def test_create_restricted_globals_no_file_access(self):
        """_create_restricted_globals excludes open without file permissions."""
        manifest = PluginManifest(name="test", entry_point="test:run")
        runner = PluginRunner(manifest)
        context = PluginContext(working_dir="/tmp")
        context.allowed_operations = set()  # No file operations

        globals_dict = runner._create_restricted_globals(context)

        # Should NOT have open
        assert "open" not in globals_dict["__builtins__"]

    def test_validate_package_with_extras(self):
        """Validation handles packages with extras notation."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            python_packages=["pytest[cov]"],  # pytest exists, extras should be stripped
        )
        runner = PluginRunner(manifest)
        valid, missing = runner._validate_requirements()

        assert valid is True  # pytest should be found

    @pytest.mark.asyncio
    async def test_run_with_sync_plugin(self):
        """Run handles synchronous plugin functions."""

        def sync_plugin(context):
            context.set_output("sync", True)
            return {"from_return": "value"}

        manifest = PluginManifest(name="sync-test", entry_point="test:sync_plugin")
        runner = PluginRunner(manifest)
        context = PluginContext()

        with patch.object(runner, "_load_entry_point", return_value=sync_plugin):
            result = await runner.run(context)

        assert result.success is True
        assert result.output.get("sync") is True
        assert result.output.get("from_return") == "value"

    @pytest.mark.asyncio
    async def test_run_with_async_plugin(self):
        """Run handles asynchronous plugin functions."""

        async def async_plugin(context):
            await asyncio.sleep(0.01)
            context.set_output("async", True)
            return {"from_async": "value"}

        manifest = PluginManifest(name="async-test", entry_point="test:async_plugin")
        runner = PluginRunner(manifest)
        context = PluginContext()

        with patch.object(runner, "_load_entry_point", return_value=async_plugin):
            result = await runner.run(context)

        assert result.success is True
        assert result.output.get("async") is True
        assert result.output.get("from_async") == "value"

    @pytest.mark.asyncio
    async def test_run_captures_context_logs_and_errors(self):
        """Run captures logs and errors from context."""

        def logging_plugin(context):
            context.log("Info message")
            context.log("Another info")
            context.error("Warning message")
            return {}

        manifest = PluginManifest(name="log-test", entry_point="test:logging_plugin")
        runner = PluginRunner(manifest)
        context = PluginContext()

        with patch.object(runner, "_load_entry_point", return_value=logging_plugin):
            result = await runner.run(context)

        assert result.success is True
        assert len(result.logs) == 2
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_run_timeout_error(self):
        """Run handles timeout correctly."""

        async def slow_plugin(context):
            await asyncio.sleep(10)  # Longer than timeout
            return {}

        manifest = PluginManifest(
            name="slow-test",
            entry_point="test:slow_plugin",
            timeout_seconds=0.1,
        )
        runner = PluginRunner(manifest)
        context = PluginContext()

        with patch.object(runner, "_load_entry_point", return_value=slow_plugin):
            result = await runner.run(context)

        assert result.success is False
        assert any("timed out" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_run_permission_error(self):
        """Run handles permission errors."""

        def permission_plugin(context):
            raise PermissionError("Access denied to /etc/passwd")

        manifest = PluginManifest(name="perm-test", entry_point="test:permission_plugin")
        runner = PluginRunner(manifest)
        context = PluginContext()

        with patch.object(runner, "_load_entry_point", return_value=permission_plugin):
            result = await runner.run(context)

        assert result.success is False
        assert any("Permission denied" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_run_runtime_error(self):
        """Run handles runtime errors gracefully."""

        def error_plugin(context):
            raise RuntimeError("Something went wrong")

        manifest = PluginManifest(name="error-test", entry_point="test:error_plugin")
        runner = PluginRunner(manifest)
        context = PluginContext()

        with patch.object(runner, "_load_entry_point", return_value=error_plugin):
            result = await runner.run(context)

        assert result.success is False
        assert any("RuntimeError" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_run_value_error(self):
        """Run handles value errors gracefully."""

        def value_error_plugin(context):
            raise ValueError("Invalid input")

        manifest = PluginManifest(name="value-test", entry_point="test:value_error_plugin")
        runner = PluginRunner(manifest)
        context = PluginContext()

        with patch.object(runner, "_load_entry_point", return_value=value_error_plugin):
            result = await runner.run(context)

        assert result.success is False
        assert any("ValueError" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_run_type_error(self):
        """Run handles type errors gracefully."""

        def type_error_plugin(context):
            raise TypeError("Type mismatch")

        manifest = PluginManifest(name="type-test", entry_point="test:type_error_plugin")
        runner = PluginRunner(manifest)
        context = PluginContext()

        with patch.object(runner, "_load_entry_point", return_value=type_error_plugin):
            result = await runner.run(context)

        assert result.success is False
        assert any("TypeError" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_run_sets_all_requirement_operations(self):
        """Run sets all operation types from requirements."""
        manifest = PluginManifest(
            name="test",
            entry_point="nonexistent:run",
            requirements=[
                PluginRequirement.READ_FILES,
                PluginRequirement.WRITE_FILES,
                PluginRequirement.RUN_COMMANDS,
                PluginRequirement.NETWORK,
            ],
        )
        runner = PluginRunner(manifest)
        context = PluginContext()

        await runner.run(context)

        assert "read_files" in context.allowed_operations
        assert "write_files" in context.allowed_operations
        assert "run_commands" in context.allowed_operations
        assert "network" in context.allowed_operations

    @pytest.mark.asyncio
    async def test_run_with_non_dict_output(self):
        """Run handles plugin returning non-dict output."""

        def non_dict_plugin(context):
            context.set_output("result", "from_context")
            return "string_output"  # Not a dict

        manifest = PluginManifest(name="non-dict-test", entry_point="test:non_dict_plugin")
        runner = PluginRunner(manifest)
        context = PluginContext()

        with patch.object(runner, "_load_entry_point", return_value=non_dict_plugin):
            result = await runner.run(context)

        assert result.success is True
        # Only context output should be in result, not the string
        assert result.output.get("result") == "from_context"


# =============================================================================
# Safe Open Tests - Extended Coverage
# =============================================================================


class TestSafeOpenExtended:
    """Extended tests for safe file open functionality."""

    def test_safe_open_append_mode_blocked_in_read_only(self, tmp_path):
        """Safe open blocks append mode in read-only."""
        manifest = PluginManifest(name="test", entry_point="test:run")
        runner = PluginRunner(manifest)
        context = PluginContext(working_dir=str(tmp_path))

        safe_open = runner._make_safe_open(context, read_only=True)

        with pytest.raises(PermissionError, match="Write operations"):
            safe_open(str(tmp_path / "test.txt"), "a")

    def test_safe_open_create_mode_blocked_in_read_only(self, tmp_path):
        """Safe open blocks create mode in read-only."""
        manifest = PluginManifest(name="test", entry_point="test:run")
        runner = PluginRunner(manifest)
        context = PluginContext(working_dir=str(tmp_path))

        safe_open = runner._make_safe_open(context, read_only=True)

        with pytest.raises(PermissionError, match="Write operations"):
            safe_open(str(tmp_path / "test.txt"), "x")

    def test_safe_open_allows_write_when_permitted(self, tmp_path):
        """Safe open allows write when read_only=False."""
        manifest = PluginManifest(name="test", entry_point="test:run")
        runner = PluginRunner(manifest)
        context = PluginContext(working_dir=str(tmp_path))

        safe_open = runner._make_safe_open(context, read_only=False)

        test_file = tmp_path / "test.txt"
        with safe_open(str(test_file), "w") as f:
            f.write("test content")

        assert test_file.read_text() == "test content"

    def test_safe_open_blocks_parent_traversal(self, tmp_path):
        """Safe open blocks accessing parent directory."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        # Create a file in parent
        parent_file = tmp_path / "secret.txt"
        parent_file.write_text("secret")

        manifest = PluginManifest(name="test", entry_point="test:run")
        runner = PluginRunner(manifest)
        context = PluginContext(working_dir=str(subdir))

        safe_open = runner._make_safe_open(context, read_only=True)

        with pytest.raises(PermissionError, match="outside working directory"):
            safe_open(str(tmp_path / "secret.txt"), "r")

    def test_safe_open_blocks_absolute_path_outside(self, tmp_path):
        """Safe open blocks absolute paths outside working dir."""
        manifest = PluginManifest(name="test", entry_point="test:run")
        runner = PluginRunner(manifest)
        context = PluginContext(working_dir=str(tmp_path))

        safe_open = runner._make_safe_open(context, read_only=True)

        with pytest.raises(PermissionError, match="outside working directory"):
            safe_open("/etc/passwd", "r")

    def test_safe_open_allows_subdirectory(self, tmp_path):
        """Safe open allows files in subdirectories."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        test_file = subdir / "test.txt"
        test_file.write_text("nested content")

        manifest = PluginManifest(name="test", entry_point="test:run")
        runner = PluginRunner(manifest)
        context = PluginContext(working_dir=str(tmp_path))

        safe_open = runner._make_safe_open(context, read_only=True)

        with safe_open(str(test_file), "r") as f:
            assert f.read() == "nested content"


# =============================================================================
# PluginRegistry Tests - Extended Coverage
# =============================================================================


class TestPluginRegistryExtended:
    """Extended tests for plugin registry."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_registry()

    def test_registry_with_custom_plugin_dirs(self, tmp_path):
        """Registry accepts custom plugin directories."""
        plugin_dir1 = tmp_path / "plugins1"
        plugin_dir2 = tmp_path / "plugins2"
        plugin_dir1.mkdir()
        plugin_dir2.mkdir()

        registry = PluginRegistry(plugin_dirs=[plugin_dir1, plugin_dir2])

        assert plugin_dir1 in registry.plugin_dirs
        assert plugin_dir2 in registry.plugin_dirs

    def test_registry_discover_multiple_plugins(self, tmp_path):
        """Registry discovers multiple plugins from directory."""
        plugin_dir = tmp_path / "plugins"

        # Create multiple plugins
        for i in range(3):
            p_dir = plugin_dir / f"plugin-{i}"
            p_dir.mkdir(parents=True)
            manifest = {
                "name": f"plugin-{i}",
                "version": "1.0.0",
                "entry_point": f"plugin{i}:run",
            }
            (p_dir / "manifest.json").write_text(json.dumps(manifest))

        registry = PluginRegistry(plugin_dirs=[plugin_dir])
        initial_count = len(registry.manifests)
        registry.discover()

        assert len(registry.manifests) == initial_count + 3
        assert "plugin-0" in registry.manifests
        assert "plugin-1" in registry.manifests
        assert "plugin-2" in registry.manifests

    def test_registry_discover_handles_malformed_json(self, tmp_path):
        """Registry handles malformed JSON in manifest."""
        plugin_dir = tmp_path / "plugins"
        bad_plugin = plugin_dir / "bad-json"
        bad_plugin.mkdir(parents=True)
        (bad_plugin / "manifest.json").write_text("{ invalid json }")

        registry = PluginRegistry(plugin_dirs=[plugin_dir])
        registry.discover()  # Should not raise

        assert "bad-json" not in registry.manifests

    def test_registry_discover_handles_missing_fields(self, tmp_path):
        """Registry handles manifest with missing required fields."""
        plugin_dir = tmp_path / "plugins"
        incomplete_plugin = plugin_dir / "incomplete"
        incomplete_plugin.mkdir(parents=True)
        # Missing entry_point
        manifest = {"name": "incomplete", "version": "1.0.0"}
        (incomplete_plugin / "manifest.json").write_text(json.dumps(manifest))

        registry = PluginRegistry(plugin_dirs=[plugin_dir])
        registry.discover()

        # Plugin without entry_point should fail validation
        assert "incomplete" not in registry.manifests

    def test_registry_list_by_capability_multiple(self):
        """list_by_capability returns all matching plugins."""
        registry = PluginRegistry()

        # Both lint and security-scan have CODE_ANALYSIS
        code_analysis_plugins = registry.list_by_capability(PluginCapability.CODE_ANALYSIS)

        assert len(code_analysis_plugins) >= 2
        names = [p.name for p in code_analysis_plugins]
        assert "lint" in names
        assert "security-scan" in names

    def test_registry_runner_caching(self):
        """Runners are cached and reused."""
        registry = PluginRegistry()

        runner1 = registry.get_runner("lint")
        runner2 = registry.get_runner("lint")
        runner3 = registry.get_runner("security-scan")

        assert runner1 is runner2  # Same instance
        assert runner1 is not runner3  # Different plugins

    @pytest.mark.asyncio
    async def test_registry_run_plugin_with_config(self):
        """run_plugin passes config to context."""
        registry = PluginRegistry()

        # Create a manifest that we can control
        test_manifest = PluginManifest(
            name="config-test",
            entry_point="test:run",
            default_config={"default_key": "default_value"},
        )
        registry.manifests["config-test"] = test_manifest

        # Mock the runner
        mock_runner = MagicMock()

        async def mock_run(context, timeout_override=None):
            return PluginResult(
                success=True,
                output={"config": context.config},
            )

        mock_runner.run = mock_run
        mock_runner.manifest = test_manifest
        registry.runners["config-test"] = mock_runner

        result = await registry.run_plugin(
            "config-test",
            input_data={"test": True},
            config={"custom": "value"},
        )

        assert result.success is True
        assert result.output["config"]["custom"] == "value"

    @pytest.mark.asyncio
    async def test_registry_run_plugin_uses_default_config(self):
        """run_plugin uses manifest default_config if no config provided."""
        registry = PluginRegistry()

        test_manifest = PluginManifest(
            name="default-config-test",
            entry_point="test:run",
            default_config={"default_setting": True},
        )
        registry.manifests["default-config-test"] = test_manifest

        mock_runner = MagicMock()

        async def mock_run(context, timeout_override=None):
            return PluginResult(
                success=True,
                output={"config": context.config},
            )

        mock_runner.run = mock_run
        mock_runner.manifest = test_manifest
        registry.runners["default-config-test"] = mock_runner

        result = await registry.run_plugin(
            "default-config-test",
            input_data={},
        )

        assert result.success is True
        assert result.output["config"]["default_setting"] is True


# =============================================================================
# Global Registry Functions Tests - Extended
# =============================================================================


class TestGlobalRegistryFunctionsExtended:
    """Extended tests for global registry functions."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_registry()

    def test_get_registry_has_builtins(self):
        """get_registry returns registry with built-in plugins."""
        registry = get_registry()

        assert "lint" in registry.manifests
        assert "security-scan" in registry.manifests
        assert "test-runner" in registry.manifests

    @pytest.mark.asyncio
    async def test_run_plugin_unknown_returns_error(self):
        """run_plugin returns error for unknown plugin."""
        reset_registry()

        result = await run_plugin("completely-unknown-plugin", {})

        assert result.success is False
        assert any("not found" in e.lower() for e in result.errors)


# =============================================================================
# Resource Limits Tests
# =============================================================================


class TestResourceLimits:
    """Tests for resource limit functionality."""

    def test_resource_available_constant(self):
        """RESOURCE_AVAILABLE is defined."""
        # Just verifying the constant is imported and is a bool
        assert isinstance(RESOURCE_AVAILABLE, bool)

    def test_set_resource_limits_no_crash(self):
        """_set_resource_limits doesn't crash."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            max_memory_mb=256,
        )
        runner = PluginRunner(manifest)

        # Should not raise on any platform
        runner._set_resource_limits()


# =============================================================================
# Entry Point Loading Tests - Extended
# =============================================================================


class TestEntryPointLoadingExtended:
    """Extended tests for entry point loading."""

    def test_load_entry_point_missing_function(self):
        """Missing function in valid module raises ImportError."""
        manifest = PluginManifest(
            name="test",
            entry_point="json:nonexistent_function_xyz",  # json exists, function doesn't
        )
        runner = PluginRunner(manifest)

        with pytest.raises(ImportError, match="Failed to load"):
            runner._load_entry_point()

    def test_load_entry_point_valid(self):
        """Valid entry point loads successfully."""
        manifest = PluginManifest(
            name="test",
            entry_point="json:dumps",  # json.dumps exists
        )
        runner = PluginRunner(manifest)

        func = runner._load_entry_point()
        assert func is json.dumps

    def test_load_entry_point_nested_module(self):
        """Entry point can load from nested modules."""
        manifest = PluginManifest(
            name="test",
            entry_point="os.path:join",  # os.path.join exists
        )
        runner = PluginRunner(manifest)

        func = runner._load_entry_point()
        from os.path import join

        assert func is join
