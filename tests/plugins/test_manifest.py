"""Comprehensive tests for plugin manifest module.

Tests the plugin manifest schema including:
- Path validation and security
- Capability and requirement enums
- Pricing configuration
- Manifest validation rules
- Serialization round-trips
- File operations
- Built-in plugins
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from aragora.plugins.manifest import (
    ALLOWED_PLUGIN_DIRS,
    BUILTIN_MANIFESTS,
    PluginCapability,
    PluginManifest,
    PluginPricing,
    PluginPricingModel,
    PluginRequirement,
    _validate_manifest_path,
    get_builtin_plugin,
    list_builtin_plugins,
)


# =============================================================================
# Path Validation Tests - Extended Coverage
# =============================================================================


class TestPathValidationExtended:
    """Extended tests for manifest path validation security."""

    def test_simple_relative_path(self):
        """Simple relative paths pass validation."""
        path = Path("manifest.json")
        is_valid, error = _validate_manifest_path(path)
        assert is_valid is True
        assert error == ""

    def test_nested_relative_path(self):
        """Nested relative paths pass validation."""
        path = Path("plugins/category/my-plugin/manifest.json")
        is_valid, error = _validate_manifest_path(path)
        assert is_valid is True
        assert error == ""

    def test_path_with_dot_dot_anywhere(self):
        """Path traversal anywhere in path is blocked."""
        paths = [
            Path("../manifest.json"),
            Path("plugins/../manifest.json"),
            Path("plugins/foo/../bar/manifest.json"),
            Path("plugins/.."),
        ]
        for path in paths:
            is_valid, error = _validate_manifest_path(path)
            assert is_valid is False, f"Path {path} should be invalid"
            assert "traversal" in error.lower()

    def test_directory_path_allowed(self):
        """Directory paths without extension are allowed."""
        path = Path("plugins/my-plugin")
        is_valid, _ = _validate_manifest_path(path)
        assert is_valid is True

    def test_executable_extension_blocked(self):
        """Executable extensions are blocked."""
        blocked_extensions = [".py", ".sh", ".bat", ".exe", ".php"]
        for ext in blocked_extensions:
            path = Path(f"plugins/test/manifest{ext}")
            is_valid, error = _validate_manifest_path(path)
            assert is_valid is False, f"Extension {ext} should be blocked"
            assert ".json" in error.lower()

    def test_hidden_file_allowed(self):
        """Hidden files with .json extension are allowed."""
        path = Path("plugins/.hidden/manifest.json")
        is_valid, _ = _validate_manifest_path(path)
        assert is_valid is True

    def test_unicode_path_components(self):
        """Unicode in path components is handled."""
        path = Path("plugins/plugin-\u00e9/manifest.json")
        is_valid, _ = _validate_manifest_path(path)
        # Should be valid as long as no traversal
        assert is_valid is True

    def test_path_with_spaces(self):
        """Paths with spaces are handled."""
        path = Path("plugins/my plugin/manifest.json")
        is_valid, _ = _validate_manifest_path(path)
        assert is_valid is True


# =============================================================================
# PluginCapability Tests - Complete Coverage
# =============================================================================


class TestPluginCapabilityComplete:
    """Complete tests for plugin capability enum."""

    def test_all_analysis_capabilities(self):
        """All analysis capabilities have correct values."""
        assert PluginCapability.CODE_ANALYSIS.value == "code_analysis"
        assert PluginCapability.LINT.value == "lint"
        assert PluginCapability.SECURITY_SCAN.value == "security_scan"
        assert PluginCapability.TYPE_CHECK.value == "type_check"

    def test_all_execution_capabilities(self):
        """All execution capabilities have correct values."""
        assert PluginCapability.TEST_RUNNER.value == "test_runner"
        assert PluginCapability.BENCHMARK.value == "benchmark"
        assert PluginCapability.FORMATTER.value == "formatter"

    def test_all_evidence_capabilities(self):
        """All evidence capabilities have correct values."""
        assert PluginCapability.EVIDENCE_FETCH.value == "evidence_fetch"
        assert PluginCapability.DOCUMENTATION.value == "documentation"

    def test_all_verification_capabilities(self):
        """All verification capabilities have correct values."""
        assert PluginCapability.FORMAL_VERIFY.value == "formal_verify"
        assert PluginCapability.PROPERTY_CHECK.value == "property_check"

    def test_capabilities_are_unique(self):
        """All capability values are unique."""
        values = [cap.value for cap in PluginCapability]
        assert len(values) == len(set(values))

    def test_capability_from_value(self):
        """Can create capability from value string."""
        cap = PluginCapability("lint")
        assert cap == PluginCapability.LINT

    def test_invalid_capability_raises(self):
        """Invalid capability value raises ValueError."""
        with pytest.raises(ValueError):
            PluginCapability("invalid_capability")


# =============================================================================
# PluginRequirement Tests - Complete Coverage
# =============================================================================


class TestPluginRequirementComplete:
    """Complete tests for plugin requirement enum."""

    def test_all_filesystem_requirements(self):
        """All filesystem requirements have correct values."""
        assert PluginRequirement.READ_FILES.value == "read_files"
        assert PluginRequirement.WRITE_FILES.value == "write_files"

    def test_all_execution_requirements(self):
        """All execution requirements have correct values."""
        assert PluginRequirement.RUN_COMMANDS.value == "run_commands"
        assert PluginRequirement.NETWORK.value == "network"

    def test_all_resource_requirements(self):
        """All resource requirements have correct values."""
        assert PluginRequirement.HIGH_MEMORY.value == "high_memory"
        assert PluginRequirement.LONG_RUNNING.value == "long_running"

    def test_all_dependency_requirements(self):
        """All dependency requirements have correct values."""
        assert PluginRequirement.PYTHON_PACKAGES.value == "python_packages"
        assert PluginRequirement.SYSTEM_TOOLS.value == "system_tools"

    def test_requirements_are_unique(self):
        """All requirement values are unique."""
        values = [req.value for req in PluginRequirement]
        assert len(values) == len(set(values))

    def test_requirement_from_value(self):
        """Can create requirement from value string."""
        req = PluginRequirement("read_files")
        assert req == PluginRequirement.READ_FILES


# =============================================================================
# PluginPricingModel Tests - Complete Coverage
# =============================================================================


class TestPluginPricingModelComplete:
    """Complete tests for pricing model enum."""

    def test_all_pricing_models(self):
        """All pricing models have correct values."""
        assert PluginPricingModel.FREE.value == "free"
        assert PluginPricingModel.ONE_TIME.value == "one_time"
        assert PluginPricingModel.SUBSCRIPTION.value == "subscription"
        assert PluginPricingModel.USAGE_BASED.value == "usage_based"

    def test_pricing_models_count(self):
        """There are exactly 4 pricing models."""
        assert len(PluginPricingModel) == 4


# =============================================================================
# PluginPricing Tests - Extended Coverage
# =============================================================================


class TestPluginPricingExtended:
    """Extended tests for plugin pricing configuration."""

    def test_one_time_pricing(self):
        """Can create one-time pricing."""
        pricing = PluginPricing(
            model=PluginPricingModel.ONE_TIME,
            price_cents=2999,
        )
        assert pricing.model == PluginPricingModel.ONE_TIME
        assert pricing.price_cents == 2999

    def test_pricing_with_custom_currency(self):
        """Can create pricing with custom currency."""
        pricing = PluginPricing(
            model=PluginPricingModel.SUBSCRIPTION,
            price_cents=1999,
            currency="GBP",
        )
        assert pricing.currency == "GBP"

    def test_pricing_with_developer_share(self):
        """Can create pricing with custom developer share."""
        pricing = PluginPricing(
            model=PluginPricingModel.SUBSCRIPTION,
            price_cents=999,
            developer_share_percent=85,
        )
        assert pricing.developer_share_percent == 85

    def test_pricing_from_dict_defaults(self):
        """from_dict uses defaults for missing fields."""
        data = {}
        pricing = PluginPricing.from_dict(data)

        assert pricing.model == PluginPricingModel.FREE
        assert pricing.price_cents == 0
        assert pricing.usage_price_cents == 0
        assert pricing.trial_days == 0
        assert pricing.currency == "USD"
        assert pricing.developer_share_percent == 70

    def test_pricing_to_dict_all_fields(self):
        """to_dict includes all fields."""
        pricing = PluginPricing(
            model=PluginPricingModel.USAGE_BASED,
            price_cents=0,
            usage_price_cents=10,
            trial_days=30,
            currency="EUR",
            developer_share_percent=60,
        )
        data = pricing.to_dict()

        assert data["model"] == "usage_based"
        assert data["price_cents"] == 0
        assert data["usage_price_cents"] == 10
        assert data["trial_days"] == 30
        assert data["currency"] == "EUR"
        assert data["developer_share_percent"] == 60


# =============================================================================
# PluginManifest Validation Tests - Extended Coverage
# =============================================================================


class TestManifestValidationExtended:
    """Extended tests for manifest validation rules."""

    def test_valid_minimal_manifest(self):
        """Minimal valid manifest passes validation."""
        manifest = PluginManifest(
            name="min",
            entry_point="m:r",
            version="1.0",
        )
        is_valid, errors = manifest.validate()
        assert is_valid is True
        assert len(errors) == 0

    def test_invalid_name_with_spaces(self):
        """Name with spaces fails validation."""
        manifest = PluginManifest(
            name="my plugin",
            entry_point="test:run",
        )
        is_valid, errors = manifest.validate()
        assert is_valid is False
        assert any("alphanumeric" in e for e in errors)

    def test_invalid_name_with_dots(self):
        """Name with dots fails validation."""
        manifest = PluginManifest(
            name="my.plugin",
            entry_point="test:run",
        )
        is_valid, errors = manifest.validate()
        assert is_valid is False
        assert any("alphanumeric" in e for e in errors)

    def test_valid_name_numeric(self):
        """Numeric name passes validation."""
        manifest = PluginManifest(
            name="plugin123",
            entry_point="test:run",
        )
        is_valid, _ = manifest.validate()
        assert is_valid is True

    def test_valid_name_mixed(self):
        """Mixed alphanumeric with hyphens and underscores passes."""
        manifest = PluginManifest(
            name="my-cool_plugin-2",
            entry_point="test:run",
        )
        is_valid, _ = manifest.validate()
        assert is_valid is True

    def test_timeout_boundary_low(self):
        """Timeout at 0 fails validation."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            timeout_seconds=0,
        )
        is_valid, errors = manifest.validate()
        assert is_valid is False
        assert any("timeout" in e for e in errors)

    def test_timeout_boundary_high(self):
        """Timeout at 3600 passes validation."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            timeout_seconds=3600,
        )
        is_valid, _ = manifest.validate()
        assert is_valid is True

    def test_timeout_above_max(self):
        """Timeout above 3600 fails validation."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            timeout_seconds=3601,
        )
        is_valid, errors = manifest.validate()
        assert is_valid is False
        assert any("timeout" in e for e in errors)

    def test_valid_semver_two_parts(self):
        """Two-part semver passes validation."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            version="1.0",
        )
        is_valid, _ = manifest.validate()
        assert is_valid is True

    def test_valid_semver_four_parts(self):
        """Four-part version passes validation."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            version="1.2.3.4",
        )
        is_valid, _ = manifest.validate()
        assert is_valid is True

    def test_valid_semver_with_prerelease(self):
        """Semver with prerelease passes validation."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            version="1.0.0-alpha",
        )
        is_valid, _ = manifest.validate()
        assert is_valid is True

    def test_multiple_validation_errors(self):
        """Multiple validation errors are all reported."""
        manifest = PluginManifest(
            name="",  # Invalid: empty
            entry_point="invalid",  # Invalid: no colon
            version="1",  # Invalid: single part
            timeout_seconds=0,  # Invalid: zero
        )
        is_valid, errors = manifest.validate()
        assert is_valid is False
        assert len(errors) >= 4


# =============================================================================
# PluginManifest Serialization Tests - Extended Coverage
# =============================================================================


class TestManifestSerializationExtended:
    """Extended tests for manifest serialization."""

    def test_to_dict_with_pricing(self):
        """to_dict includes pricing correctly."""
        manifest = PluginManifest(
            name="paid-plugin",
            entry_point="paid:run",
            pricing=PluginPricing(
                model=PluginPricingModel.SUBSCRIPTION,
                price_cents=999,
            ),
        )
        data = manifest.to_dict()

        assert data["pricing"]["model"] == "subscription"
        assert data["pricing"]["price_cents"] == 999

    def test_to_json_pretty_print(self):
        """to_json creates formatted JSON."""
        manifest = PluginManifest(name="test", entry_point="test:run")
        json_str = manifest.to_json(indent=4)

        # Should have newlines from indentation
        assert "\n" in json_str
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["name"] == "test"

    def test_from_dict_with_nested_capabilities(self):
        """from_dict handles capability list correctly."""
        data = {
            "name": "multi-cap",
            "entry_point": "multi:run",
            "capabilities": ["lint", "security_scan", "code_analysis"],
        }
        manifest = PluginManifest.from_dict(data)

        assert len(manifest.capabilities) == 3
        assert PluginCapability.LINT in manifest.capabilities
        assert PluginCapability.SECURITY_SCAN in manifest.capabilities
        assert PluginCapability.CODE_ANALYSIS in manifest.capabilities

    def test_from_dict_mixed_valid_invalid_requirements(self):
        """from_dict keeps only valid requirements."""
        data = {
            "name": "test",
            "entry_point": "test:run",
            "requirements": [
                "read_files",
                "invalid1",
                "network",
                "invalid2",
                "write_files",
            ],
        }
        manifest = PluginManifest.from_dict(data)

        assert len(manifest.requirements) == 3
        assert PluginRequirement.READ_FILES in manifest.requirements
        assert PluginRequirement.NETWORK in manifest.requirements
        assert PluginRequirement.WRITE_FILES in manifest.requirements

    def test_from_dict_with_all_fields(self):
        """from_dict handles all fields correctly."""
        data = {
            "name": "complete-plugin",
            "version": "2.5.0",
            "description": "A complete plugin",
            "author": "Test Author",
            "author_id": "author-123",
            "capabilities": ["lint"],
            "requirements": ["read_files"],
            "entry_point": "complete:run",
            "timeout_seconds": 120,
            "max_memory_mb": 1024,
            "config_schema": {"type": "object"},
            "default_config": {"enabled": True},
            "python_packages": ["pytest", "ruff"],
            "system_tools": ["git"],
            "pricing": {"model": "subscription", "price_cents": 999},
            "license": "Apache-2.0",
            "homepage": "https://example.com",
            "tags": ["testing", "quality"],
            "created_at": "2024-01-01T00:00:00",
        }
        manifest = PluginManifest.from_dict(data)

        assert manifest.name == "complete-plugin"
        assert manifest.version == "2.5.0"
        assert manifest.description == "A complete plugin"
        assert manifest.author == "Test Author"
        assert manifest.author_id == "author-123"
        assert len(manifest.capabilities) == 1
        assert len(manifest.requirements) == 1
        assert manifest.entry_point == "complete:run"
        assert manifest.timeout_seconds == 120
        assert manifest.max_memory_mb == 1024
        assert manifest.config_schema == {"type": "object"}
        assert manifest.default_config == {"enabled": True}
        assert manifest.python_packages == ["pytest", "ruff"]
        assert manifest.system_tools == ["git"]
        assert manifest.pricing.model == PluginPricingModel.SUBSCRIPTION
        assert manifest.license == "Apache-2.0"
        assert manifest.homepage == "https://example.com"
        assert manifest.tags == ["testing", "quality"]
        assert manifest.created_at == "2024-01-01T00:00:00"

    def test_roundtrip_json(self):
        """Manifest survives JSON round-trip."""
        original = PluginManifest(
            name="json-roundtrip",
            version="1.2.3",
            entry_point="roundtrip:run",
            capabilities=[PluginCapability.LINT],
            requirements=[PluginRequirement.READ_FILES],
        )

        json_str = original.to_json()
        restored = PluginManifest.from_json(json_str)

        assert restored.name == original.name
        assert restored.version == original.version
        assert restored.entry_point == original.entry_point
        assert restored.capabilities == original.capabilities
        assert restored.requirements == original.requirements


# =============================================================================
# PluginManifest File Operations Tests - Extended Coverage
# =============================================================================


class TestManifestFileOperationsExtended:
    """Extended tests for manifest save/load operations."""

    def test_save_creates_parent_directories(self, tmp_path):
        """save() creates parent directories if needed."""
        deep_path = tmp_path / "a" / "b" / "c" / "manifest.json"
        manifest = PluginManifest(name="deep", entry_point="deep:run")

        manifest.save(deep_path)

        assert deep_path.exists()
        assert deep_path.parent.exists()

    def test_save_overwrites_existing(self, tmp_path):
        """save() overwrites existing file."""
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text('{"name": "old"}')

        manifest = PluginManifest(name="new", entry_point="new:run")
        manifest.save(manifest_path)

        loaded = PluginManifest.load(manifest_path)
        assert loaded.name == "new"

    def test_load_invalid_json_content(self, tmp_path):
        """load() raises ValueError for invalid JSON content."""
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("not json at all")

        with pytest.raises(ValueError, match="Invalid plugin manifest JSON"):
            PluginManifest.load(manifest_path)

    def test_load_empty_file(self, tmp_path):
        """load() handles empty file."""
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("")

        with pytest.raises(ValueError):
            PluginManifest.load(manifest_path)


# =============================================================================
# PluginManifest Capability Checks Tests - Extended
# =============================================================================


class TestManifestCapabilityChecksExtended:
    """Extended tests for capability and requirement checking."""

    def test_has_capability_multiple(self):
        """has_capability works with multiple capabilities."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            capabilities=[
                PluginCapability.LINT,
                PluginCapability.CODE_ANALYSIS,
                PluginCapability.FORMATTER,
            ],
        )

        assert manifest.has_capability(PluginCapability.LINT) is True
        assert manifest.has_capability(PluginCapability.CODE_ANALYSIS) is True
        assert manifest.has_capability(PluginCapability.FORMATTER) is True
        assert manifest.has_capability(PluginCapability.SECURITY_SCAN) is False

    def test_requires_multiple(self):
        """requires works with multiple requirements."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            requirements=[
                PluginRequirement.READ_FILES,
                PluginRequirement.WRITE_FILES,
                PluginRequirement.NETWORK,
            ],
        )

        assert manifest.requires(PluginRequirement.READ_FILES) is True
        assert manifest.requires(PluginRequirement.WRITE_FILES) is True
        assert manifest.requires(PluginRequirement.NETWORK) is True
        assert manifest.requires(PluginRequirement.RUN_COMMANDS) is False

    def test_has_capability_empty(self):
        """has_capability returns False with empty capabilities."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            capabilities=[],
        )

        for cap in PluginCapability:
            assert manifest.has_capability(cap) is False

    def test_requires_empty(self):
        """requires returns False with empty requirements."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            requirements=[],
        )

        for req in PluginRequirement:
            assert manifest.requires(req) is False


# =============================================================================
# Built-in Plugins Tests - Extended Coverage
# =============================================================================


class TestBuiltinPluginsExtended:
    """Extended tests for built-in plugin manifests."""

    def test_lint_plugin_complete(self):
        """Lint plugin has all expected fields."""
        lint = get_builtin_plugin("lint")

        assert lint is not None
        assert lint.name == "lint"
        assert lint.version == "1.0.0"
        assert lint.author == "aragora"
        assert lint.entry_point == "aragora.plugins.builtin.lint:run"
        assert lint.timeout_seconds == 30
        assert PluginCapability.LINT in lint.capabilities
        assert PluginCapability.CODE_ANALYSIS in lint.capabilities
        assert PluginRequirement.READ_FILES in lint.requirements
        assert PluginRequirement.RUN_COMMANDS in lint.requirements
        assert "ruff" in lint.system_tools or "flake8" in lint.system_tools
        assert "code-quality" in lint.tags or "lint" in lint.tags

    def test_security_scan_plugin_complete(self):
        """Security scan plugin has all expected fields."""
        security = get_builtin_plugin("security-scan")

        assert security is not None
        assert security.name == "security-scan"
        assert security.version == "1.0.0"
        assert security.author == "aragora"
        assert security.entry_point == "aragora.plugins.builtin.security:run"
        assert security.timeout_seconds == 60
        assert PluginCapability.SECURITY_SCAN in security.capabilities
        assert PluginCapability.CODE_ANALYSIS in security.capabilities
        assert PluginRequirement.READ_FILES in security.requirements
        assert "bandit" in security.python_packages
        assert "security" in security.tags or "analysis" in security.tags

    def test_test_runner_plugin_complete(self):
        """Test runner plugin has all expected fields."""
        runner = get_builtin_plugin("test-runner")

        assert runner is not None
        assert runner.name == "test-runner"
        assert runner.version == "1.0.0"
        assert runner.author == "aragora"
        assert runner.entry_point == "aragora.plugins.builtin.tests:run"
        assert runner.timeout_seconds == 300  # Longer for tests
        assert PluginCapability.TEST_RUNNER in runner.capabilities
        assert PluginRequirement.READ_FILES in runner.requirements
        assert PluginRequirement.RUN_COMMANDS in runner.requirements
        assert PluginRequirement.LONG_RUNNING in runner.requirements
        assert "pytest" in runner.python_packages
        assert "testing" in runner.tags or "verification" in runner.tags

    def test_builtin_manifests_dict(self):
        """BUILTIN_MANIFESTS is a proper dict with expected entries."""
        assert isinstance(BUILTIN_MANIFESTS, dict)
        assert "lint" in BUILTIN_MANIFESTS
        assert "security-scan" in BUILTIN_MANIFESTS
        assert "test-runner" in BUILTIN_MANIFESTS

        for name, manifest in BUILTIN_MANIFESTS.items():
            assert isinstance(manifest, PluginManifest)
            assert manifest.name == name

    def test_list_builtin_plugins_returns_manifests(self):
        """list_builtin_plugins returns PluginManifest instances."""
        plugins = list_builtin_plugins()

        assert len(plugins) >= 3
        for plugin in plugins:
            assert isinstance(plugin, PluginManifest)


# =============================================================================
# ALLOWED_PLUGIN_DIRS Tests
# =============================================================================


class TestAllowedPluginDirs:
    """Tests for ALLOWED_PLUGIN_DIRS constant."""

    def test_allowed_dirs_is_frozenset(self):
        """ALLOWED_PLUGIN_DIRS is a frozenset."""
        assert isinstance(ALLOWED_PLUGIN_DIRS, frozenset)

    def test_allowed_dirs_contains_expected(self):
        """ALLOWED_PLUGIN_DIRS contains expected directories."""
        assert "plugins" in ALLOWED_PLUGIN_DIRS
        assert ".aragora/plugins" in ALLOWED_PLUGIN_DIRS
        assert "aragora/plugins" in ALLOWED_PLUGIN_DIRS
