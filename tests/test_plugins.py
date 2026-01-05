"""
Tests for plugin system security and functionality.

Tests:
- Plugin manifest validation
- Plugin sandbox restrictions
- Path traversal prevention
- Timeout enforcement
- Capability-based permissions
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

import pytest

from aragora.plugins.manifest import (
    PluginManifest,
    PluginCapability,
    PluginRequirement,
    BUILTIN_MANIFESTS,
    list_builtin_plugins,
    get_builtin_plugin,
)
from aragora.plugins.runner import (
    PluginContext,
    PluginResult,
    PluginRunner,
    PluginRegistry,
    run_plugin,
    get_registry,
)


class TestPluginManifest:
    """Test plugin manifest validation."""

    def test_valid_manifest(self):
        """Valid manifest should pass validation."""
        manifest = PluginManifest(
            name="test-plugin",
            version="1.0.0",
            description="Test plugin",
            entry_point="test_module:run",
            capabilities=[PluginCapability.CODE_ANALYSIS],
            requirements=[PluginRequirement.READ_FILES],
        )
        valid, errors = manifest.validate()
        assert valid is True
        assert errors == []

    def test_missing_name_fails(self):
        """Manifest without name should fail validation."""
        manifest = PluginManifest(
            name="",
            entry_point="test:run",
        )
        valid, errors = manifest.validate()
        assert valid is False
        assert any("name" in e.lower() for e in errors)

    def test_missing_entry_point_fails(self):
        """Manifest without entry point should fail validation."""
        manifest = PluginManifest(
            name="test",
            entry_point="",
        )
        valid, errors = manifest.validate()
        assert valid is False
        assert any("entry_point" in e.lower() for e in errors)

    def test_invalid_entry_point_format_fails(self):
        """Entry point must be in module:function format."""
        manifest = PluginManifest(
            name="test",
            entry_point="invalid_no_colon",
        )
        valid, errors = manifest.validate()
        assert valid is False
        assert any("module:function" in e for e in errors)

    def test_invalid_name_chars_fails(self):
        """Name with invalid characters should fail."""
        manifest = PluginManifest(
            name="test@plugin!",
            entry_point="test:run",
        )
        valid, errors = manifest.validate()
        assert valid is False
        assert any("alphanumeric" in e for e in errors)

    def test_valid_name_with_dash_underscore(self):
        """Name with dashes and underscores should be valid."""
        manifest = PluginManifest(
            name="test-plugin_v2",
            entry_point="test:run",
        )
        valid, errors = manifest.validate()
        assert valid is True

    def test_invalid_timeout_fails(self):
        """Invalid timeout should fail validation."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            timeout_seconds=0,
        )
        valid, errors = manifest.validate()
        assert valid is False
        assert any("timeout" in e.lower() for e in errors)

        manifest2 = PluginManifest(
            name="test",
            entry_point="test:run",
            timeout_seconds=5000,  # > 3600
        )
        valid2, errors2 = manifest2.validate()
        assert valid2 is False

    def test_to_dict_roundtrip(self):
        """Manifest should survive to_dict/from_dict roundtrip."""
        manifest = PluginManifest(
            name="roundtrip-test",
            version="2.0.0",
            description="Test roundtrip",
            entry_point="test:run",
            capabilities=[PluginCapability.LINT, PluginCapability.SECURITY_SCAN],
            requirements=[PluginRequirement.READ_FILES],
            timeout_seconds=45.0,
            tags=["test", "roundtrip"],
        )

        data = manifest.to_dict()
        restored = PluginManifest.from_dict(data)

        assert restored.name == manifest.name
        assert restored.version == manifest.version
        assert restored.timeout_seconds == manifest.timeout_seconds
        assert len(restored.capabilities) == len(manifest.capabilities)

    def test_has_capability(self):
        """Test capability checking."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            capabilities=[PluginCapability.LINT],
        )

        assert manifest.has_capability(PluginCapability.LINT) is True
        assert manifest.has_capability(PluginCapability.SECURITY_SCAN) is False

    def test_requires(self):
        """Test requirement checking."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            requirements=[PluginRequirement.READ_FILES],
        )

        assert manifest.requires(PluginRequirement.READ_FILES) is True
        assert manifest.requires(PluginRequirement.NETWORK) is False


class TestBuiltinManifests:
    """Test built-in plugin manifests."""

    def test_builtin_manifests_exist(self):
        """Built-in manifests should be defined."""
        assert len(BUILTIN_MANIFESTS) > 0
        assert "lint" in BUILTIN_MANIFESTS
        assert "security-scan" in BUILTIN_MANIFESTS

    def test_builtin_manifests_valid(self):
        """All built-in manifests should be valid."""
        for name, manifest in BUILTIN_MANIFESTS.items():
            valid, errors = manifest.validate()
            assert valid is True, f"Manifest '{name}' failed validation: {errors}"

    def test_list_builtin_plugins(self):
        """list_builtin_plugins should return all manifests."""
        plugins = list_builtin_plugins()
        assert len(plugins) == len(BUILTIN_MANIFESTS)

    def test_get_builtin_plugin(self):
        """get_builtin_plugin should retrieve by name."""
        lint = get_builtin_plugin("lint")
        assert lint is not None
        assert lint.name == "lint"

        missing = get_builtin_plugin("nonexistent")
        assert missing is None


class TestPluginContext:
    """Test plugin execution context."""

    def test_context_defaults(self):
        """Context should have sensible defaults."""
        ctx = PluginContext()
        assert ctx.input_data == {}
        assert ctx.output == {}
        assert ctx.logs == []
        assert ctx.errors == []

    def test_context_log(self):
        """Context should support logging."""
        ctx = PluginContext()
        ctx.log("test message")
        assert len(ctx.logs) == 1
        assert "test message" in ctx.logs[0]

    def test_context_error(self):
        """Context should support error logging."""
        ctx = PluginContext()
        ctx.error("test error")
        assert len(ctx.errors) == 1
        assert "test error" in ctx.errors[0]

    def test_context_set_output(self):
        """Context should support output setting."""
        ctx = PluginContext()
        ctx.set_output("result", 42)
        assert ctx.output["result"] == 42

    def test_context_can_check(self):
        """Context should check allowed operations."""
        ctx = PluginContext()
        ctx.allowed_operations = {"read_files", "network"}

        assert ctx.can("read_files") is True
        assert ctx.can("network") is True
        assert ctx.can("write_files") is False


class TestPluginRunner:
    """Test plugin runner sandbox."""

    def test_restricted_builtins_no_dangerous(self):
        """Restricted builtins should exclude dangerous functions."""
        dangerous = ["exec", "eval", "compile", "__import__", "open"]
        for name in dangerous:
            assert name not in PluginRunner.RESTRICTED_BUILTINS

    def test_restricted_builtins_has_safe(self):
        """Restricted builtins should include safe functions."""
        safe = ["len", "str", "int", "list", "dict", "sum", "max", "min"]
        for name in safe:
            assert name in PluginRunner.RESTRICTED_BUILTINS

    def test_safe_open_path_traversal_blocked(self):
        """Safe open should block path traversal."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            requirements=[PluginRequirement.READ_FILES],
        )
        runner = PluginRunner(manifest)

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = PluginContext(working_dir=tmpdir)
            ctx.allowed_operations = {"read_files"}

            safe_open = runner._make_safe_open(ctx, read_only=True)

            # Attempting to access parent directory should fail
            with pytest.raises(PermissionError):
                safe_open("../etc/passwd")

            with pytest.raises(PermissionError):
                safe_open("/etc/passwd")

    def test_safe_open_read_only_blocks_write(self):
        """Read-only safe open should block write operations."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            requirements=[PluginRequirement.READ_FILES],
        )
        runner = PluginRunner(manifest)

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = PluginContext(working_dir=tmpdir)
            ctx.allowed_operations = {"read_files"}

            safe_open = runner._make_safe_open(ctx, read_only=True)

            # Write mode should be blocked
            with pytest.raises(PermissionError):
                safe_open(f"{tmpdir}/test.txt", "w")

            with pytest.raises(PermissionError):
                safe_open(f"{tmpdir}/test.txt", "a")

    def test_validate_requirements_missing_package(self):
        """Should detect missing Python packages."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            python_packages=["nonexistent_package_12345"],
        )
        runner = PluginRunner(manifest)

        valid, missing = runner._validate_requirements()
        assert valid is False
        assert len(missing) > 0
        assert any("nonexistent_package" in m for m in missing)

    def test_validate_requirements_existing_package(self):
        """Should detect existing Python packages."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            python_packages=["json"],  # stdlib always available
        )
        runner = PluginRunner(manifest)

        valid, missing = runner._validate_requirements()
        assert valid is True


class TestPluginRegistry:
    """Test plugin registry."""

    def test_registry_loads_builtins(self):
        """Registry should load built-in plugins."""
        registry = PluginRegistry()
        plugins = registry.list_plugins()

        assert len(plugins) >= len(BUILTIN_MANIFESTS)

    def test_registry_get_by_name(self):
        """Registry should retrieve plugins by name."""
        registry = PluginRegistry()

        lint = registry.get("lint")
        assert lint is not None
        assert lint.name == "lint"

        missing = registry.get("nonexistent")
        assert missing is None

    def test_registry_list_by_capability(self):
        """Registry should filter by capability."""
        registry = PluginRegistry()

        lint_plugins = registry.list_by_capability(PluginCapability.LINT)
        assert len(lint_plugins) > 0
        assert all(p.has_capability(PluginCapability.LINT) for p in lint_plugins)

    def test_registry_get_runner(self):
        """Registry should create runners for plugins."""
        registry = PluginRegistry()

        runner = registry.get_runner("lint")
        assert runner is not None
        assert isinstance(runner, PluginRunner)

        # Same runner should be cached
        runner2 = registry.get_runner("lint")
        assert runner is runner2


class TestPluginResult:
    """Test plugin result handling."""

    def test_result_to_dict(self):
        """Result should serialize to dict."""
        result = PluginResult(
            success=True,
            output={"key": "value"},
            logs=["log1", "log2"],
            errors=[],
            plugin_name="test",
            plugin_version="1.0.0",
        )

        data = result.to_dict()
        assert data["success"] is True
        assert data["output"] == {"key": "value"}
        assert data["plugin_name"] == "test"

    def test_failed_result(self):
        """Failed result should indicate failure."""
        result = PluginResult(
            success=False,
            errors=["Something went wrong"],
        )

        assert result.success is False
        assert len(result.errors) > 0


class TestGlobalRegistry:
    """Test global registry singleton."""

    def test_get_registry_singleton(self):
        """get_registry should return same instance."""
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2


@pytest.mark.asyncio
class TestPluginExecution:
    """Test async plugin execution."""

    async def test_run_plugin_not_found(self):
        """Running nonexistent plugin should return error."""
        result = await run_plugin("nonexistent_plugin_12345", {})

        assert result.success is False
        assert any("not found" in e.lower() for e in result.errors)

    async def test_plugin_timeout(self):
        """Plugin should timeout if execution exceeds limit."""
        # Create a manifest with a very short timeout
        manifest = PluginManifest(
            name="timeout-test",
            entry_point="test:run",
            timeout_seconds=0.1,
        )

        runner = PluginRunner(manifest)

        # Mock a slow function
        async def slow_func(ctx):
            await asyncio.sleep(10)
            return {}

        with patch.object(runner, '_load_entry_point', return_value=slow_func):
            ctx = PluginContext()
            result = await runner.run(ctx, timeout_override=0.1)

            assert result.success is False
            assert any("timed out" in e.lower() or "timeout" in e.lower() for e in result.errors)
