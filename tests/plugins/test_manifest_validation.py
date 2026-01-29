"""Tests for plugin manifest validation and serialization.

Tests the plugin manifest schema including:
- Path validation and security
- Capability and requirement enums
- Pricing configuration
- Manifest validation rules
- Serialization round-trips
"""

import json
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
# Path Validation Tests
# =============================================================================


class TestPathValidation:
    """Tests for manifest path validation security."""

    def test_valid_path(self):
        """Valid paths pass validation."""
        path = Path("plugins/my-plugin/manifest.json")
        is_valid, error = _validate_manifest_path(path)
        assert is_valid is True
        assert error == ""

    def test_path_traversal_blocked(self):
        """Path traversal attempts are blocked."""
        path = Path("plugins/../../../etc/passwd")
        is_valid, error = _validate_manifest_path(path)
        assert is_valid is False
        assert "traversal" in error.lower()

    def test_path_traversal_middle(self):
        """Path traversal in middle is blocked."""
        path = Path("plugins/foo/../bar/manifest.json")
        is_valid, error = _validate_manifest_path(path)
        # .. in path should be blocked
        assert is_valid is False

    def test_json_extension_allowed(self):
        """JSON extension is allowed."""
        path = Path("plugins/test/manifest.json")
        is_valid, _ = _validate_manifest_path(path)
        assert is_valid is True

    def test_no_extension_allowed(self):
        """No extension is allowed (for directories)."""
        path = Path("plugins/test")
        is_valid, _ = _validate_manifest_path(path)
        assert is_valid is True

    def test_invalid_extension_blocked(self):
        """Non-JSON extensions are blocked."""
        path = Path("plugins/test/manifest.py")
        is_valid, error = _validate_manifest_path(path)
        assert is_valid is False
        assert ".json" in error.lower()

    def test_absolute_path_valid(self, tmp_path):
        """Absolute paths can be valid."""
        manifest_path = tmp_path / "manifest.json"
        is_valid, _ = _validate_manifest_path(manifest_path)
        assert is_valid is True


# =============================================================================
# PluginCapability Tests
# =============================================================================


class TestPluginCapability:
    """Tests for plugin capability enum."""

    def test_code_analysis_capability(self):
        """Code analysis capability exists."""
        cap = PluginCapability.CODE_ANALYSIS
        assert cap.value == "code_analysis"

    def test_lint_capability(self):
        """Lint capability exists."""
        cap = PluginCapability.LINT
        assert cap.value == "lint"

    def test_security_scan_capability(self):
        """Security scan capability exists."""
        cap = PluginCapability.SECURITY_SCAN
        assert cap.value == "security_scan"

    def test_test_runner_capability(self):
        """Test runner capability exists."""
        cap = PluginCapability.TEST_RUNNER
        assert cap.value == "test_runner"

    def test_custom_capability(self):
        """Custom capability exists for extensibility."""
        cap = PluginCapability.CUSTOM
        assert cap.value == "custom"

    def test_all_capabilities_have_values(self):
        """All capabilities have string values."""
        for cap in PluginCapability:
            assert isinstance(cap.value, str)
            assert len(cap.value) > 0


# =============================================================================
# PluginRequirement Tests
# =============================================================================


class TestPluginRequirement:
    """Tests for plugin requirement enum."""

    def test_read_files_requirement(self):
        """Read files requirement exists."""
        req = PluginRequirement.READ_FILES
        assert req.value == "read_files"

    def test_write_files_requirement(self):
        """Write files requirement exists."""
        req = PluginRequirement.WRITE_FILES
        assert req.value == "write_files"

    def test_run_commands_requirement(self):
        """Run commands requirement exists."""
        req = PluginRequirement.RUN_COMMANDS
        assert req.value == "run_commands"

    def test_network_requirement(self):
        """Network requirement exists."""
        req = PluginRequirement.NETWORK
        assert req.value == "network"

    def test_high_memory_requirement(self):
        """High memory requirement exists."""
        req = PluginRequirement.HIGH_MEMORY
        assert req.value == "high_memory"

    def test_long_running_requirement(self):
        """Long running requirement exists."""
        req = PluginRequirement.LONG_RUNNING
        assert req.value == "long_running"


# =============================================================================
# PluginPricingModel Tests
# =============================================================================


class TestPluginPricingModel:
    """Tests for pricing model enum."""

    def test_free_model(self):
        """Free pricing model exists."""
        assert PluginPricingModel.FREE.value == "free"

    def test_one_time_model(self):
        """One-time pricing model exists."""
        assert PluginPricingModel.ONE_TIME.value == "one_time"

    def test_subscription_model(self):
        """Subscription pricing model exists."""
        assert PluginPricingModel.SUBSCRIPTION.value == "subscription"

    def test_usage_based_model(self):
        """Usage-based pricing model exists."""
        assert PluginPricingModel.USAGE_BASED.value == "usage_based"


# =============================================================================
# PluginPricing Tests
# =============================================================================


class TestPluginPricing:
    """Tests for plugin pricing configuration."""

    def test_default_pricing(self):
        """Default pricing is free."""
        pricing = PluginPricing()
        assert pricing.model == PluginPricingModel.FREE
        assert pricing.price_cents == 0
        assert pricing.developer_share_percent == 70

    def test_subscription_pricing(self):
        """Can create subscription pricing."""
        pricing = PluginPricing(
            model=PluginPricingModel.SUBSCRIPTION,
            price_cents=999,
            trial_days=14,
        )
        assert pricing.model == PluginPricingModel.SUBSCRIPTION
        assert pricing.price_cents == 999
        assert pricing.trial_days == 14

    def test_usage_based_pricing(self):
        """Can create usage-based pricing."""
        pricing = PluginPricing(
            model=PluginPricingModel.USAGE_BASED,
            usage_price_cents=5,
        )
        assert pricing.model == PluginPricingModel.USAGE_BASED
        assert pricing.usage_price_cents == 5

    def test_pricing_to_dict(self):
        """Pricing converts to dictionary."""
        pricing = PluginPricing(
            model=PluginPricingModel.SUBSCRIPTION,
            price_cents=1999,
            trial_days=7,
            currency="EUR",
        )
        data = pricing.to_dict()

        assert data["model"] == "subscription"
        assert data["price_cents"] == 1999
        assert data["trial_days"] == 7
        assert data["currency"] == "EUR"

    def test_pricing_from_dict(self):
        """Pricing loads from dictionary."""
        data = {
            "model": "subscription",
            "price_cents": 999,
            "trial_days": 14,
            "developer_share_percent": 80,
        }
        pricing = PluginPricing.from_dict(data)

        assert pricing.model == PluginPricingModel.SUBSCRIPTION
        assert pricing.price_cents == 999
        assert pricing.trial_days == 14
        assert pricing.developer_share_percent == 80

    def test_pricing_from_dict_invalid_model(self):
        """Invalid pricing model defaults to free."""
        data = {"model": "invalid_model"}
        pricing = PluginPricing.from_dict(data)
        assert pricing.model == PluginPricingModel.FREE

    def test_pricing_roundtrip(self):
        """Pricing survives dict round-trip."""
        original = PluginPricing(
            model=PluginPricingModel.ONE_TIME,
            price_cents=4999,
            developer_share_percent=60,
        )
        restored = PluginPricing.from_dict(original.to_dict())

        assert restored.model == original.model
        assert restored.price_cents == original.price_cents
        assert restored.developer_share_percent == original.developer_share_percent


# =============================================================================
# PluginManifest Validation Tests
# =============================================================================


class TestManifestValidation:
    """Tests for manifest validation rules."""

    def test_valid_manifest(self):
        """Valid manifest passes validation."""
        manifest = PluginManifest(
            name="test-plugin",
            entry_point="test.module:run",
        )
        is_valid, errors = manifest.validate()
        assert is_valid is True
        assert len(errors) == 0

    def test_missing_name(self):
        """Manifest without name fails validation."""
        manifest = PluginManifest(name="", entry_point="test:run")
        is_valid, errors = manifest.validate()
        assert is_valid is False
        assert any("name" in e for e in errors)

    def test_missing_entry_point(self):
        """Manifest without entry point fails validation."""
        manifest = PluginManifest(name="test", entry_point="")
        is_valid, errors = manifest.validate()
        assert is_valid is False
        assert any("entry_point" in e for e in errors)

    def test_invalid_entry_point_format(self):
        """Entry point must be module:function format."""
        manifest = PluginManifest(name="test", entry_point="invalid_format")
        is_valid, errors = manifest.validate()
        assert is_valid is False
        assert any("module:function" in e for e in errors)

    def test_invalid_name_characters(self):
        """Name must be alphanumeric with - or _."""
        manifest = PluginManifest(name="test@plugin!", entry_point="test:run")
        is_valid, errors = manifest.validate()
        assert is_valid is False
        assert any("alphanumeric" in e for e in errors)

    def test_valid_name_with_hyphens(self):
        """Names with hyphens are valid."""
        manifest = PluginManifest(name="my-cool-plugin", entry_point="test:run")
        is_valid, errors = manifest.validate()
        assert is_valid is True

    def test_valid_name_with_underscores(self):
        """Names with underscores are valid."""
        manifest = PluginManifest(name="my_cool_plugin", entry_point="test:run")
        is_valid, errors = manifest.validate()
        assert is_valid is True

    def test_invalid_version_format(self):
        """Version should be semver format."""
        manifest = PluginManifest(name="test", entry_point="test:run", version="1")
        is_valid, errors = manifest.validate()
        assert is_valid is False
        assert any("semver" in e for e in errors)

    def test_valid_semver_version(self):
        """Valid semver versions pass."""
        manifest = PluginManifest(name="test", entry_point="test:run", version="1.2.3")
        is_valid, errors = manifest.validate()
        assert is_valid is True

    def test_timeout_too_low(self):
        """Timeout must be positive."""
        manifest = PluginManifest(name="test", entry_point="test:run", timeout_seconds=0)
        is_valid, errors = manifest.validate()
        assert is_valid is False
        assert any("timeout" in e for e in errors)

    def test_timeout_too_high(self):
        """Timeout must be at most 3600."""
        manifest = PluginManifest(name="test", entry_point="test:run", timeout_seconds=4000)
        is_valid, errors = manifest.validate()
        assert is_valid is False
        assert any("timeout" in e for e in errors)


# =============================================================================
# PluginManifest Serialization Tests
# =============================================================================


class TestManifestSerialization:
    """Tests for manifest serialization."""

    def test_to_dict(self):
        """Manifest converts to dictionary."""
        manifest = PluginManifest(
            name="test-plugin",
            version="2.0.0",
            description="Test description",
            entry_point="test:run",
            capabilities=[PluginCapability.LINT],
            requirements=[PluginRequirement.READ_FILES],
        )
        data = manifest.to_dict()

        assert data["name"] == "test-plugin"
        assert data["version"] == "2.0.0"
        assert data["description"] == "Test description"
        assert data["entry_point"] == "test:run"
        assert "lint" in data["capabilities"]
        assert "read_files" in data["requirements"]

    def test_to_json(self):
        """Manifest converts to JSON string."""
        manifest = PluginManifest(name="test", entry_point="test:run")
        json_str = manifest.to_json()

        parsed = json.loads(json_str)
        assert parsed["name"] == "test"
        assert parsed["entry_point"] == "test:run"

    def test_from_dict(self):
        """Manifest loads from dictionary."""
        data = {
            "name": "loaded-plugin",
            "version": "1.5.0",
            "entry_point": "loaded:run",
            "capabilities": ["security_scan", "code_analysis"],
            "requirements": ["read_files", "network"],
        }
        manifest = PluginManifest.from_dict(data)

        assert manifest.name == "loaded-plugin"
        assert manifest.version == "1.5.0"
        assert PluginCapability.SECURITY_SCAN in manifest.capabilities
        assert PluginCapability.CODE_ANALYSIS in manifest.capabilities
        assert PluginRequirement.READ_FILES in manifest.requirements
        assert PluginRequirement.NETWORK in manifest.requirements

    def test_from_dict_invalid_capability(self):
        """Invalid capability converts to CUSTOM."""
        data = {
            "name": "test",
            "entry_point": "test:run",
            "capabilities": ["unknown_capability"],
        }
        manifest = PluginManifest.from_dict(data)
        assert PluginCapability.CUSTOM in manifest.capabilities

    def test_from_dict_invalid_requirement_skipped(self):
        """Invalid requirements are skipped."""
        data = {
            "name": "test",
            "entry_point": "test:run",
            "requirements": ["unknown_requirement", "read_files"],
        }
        manifest = PluginManifest.from_dict(data)
        # Only valid requirement is kept
        assert len(manifest.requirements) == 1
        assert PluginRequirement.READ_FILES in manifest.requirements

    def test_from_json(self):
        """Manifest loads from JSON string."""
        json_str = '{"name": "json-plugin", "entry_point": "json:run", "version": "1.0.0"}'
        manifest = PluginManifest.from_json(json_str)

        assert manifest.name == "json-plugin"
        assert manifest.entry_point == "json:run"

    def test_from_json_invalid(self):
        """Invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid plugin manifest JSON"):
            PluginManifest.from_json("not valid json")

    def test_roundtrip_dict(self):
        """Manifest survives dictionary round-trip."""
        original = PluginManifest(
            name="roundtrip-plugin",
            version="3.0.0",
            description="Roundtrip test",
            author="tester",
            entry_point="roundtrip:run",
            capabilities=[PluginCapability.LINT, PluginCapability.SECURITY_SCAN],
            requirements=[PluginRequirement.READ_FILES],
            timeout_seconds=120,
            max_memory_mb=1024,
            tags=["test", "validation"],
        )

        restored = PluginManifest.from_dict(original.to_dict())

        assert restored.name == original.name
        assert restored.version == original.version
        assert restored.description == original.description
        assert restored.capabilities == original.capabilities
        assert restored.requirements == original.requirements
        assert restored.timeout_seconds == original.timeout_seconds
        assert restored.max_memory_mb == original.max_memory_mb
        assert restored.tags == original.tags


# =============================================================================
# PluginManifest File Operations Tests
# =============================================================================


class TestManifestFileOperations:
    """Tests for manifest save/load operations."""

    def test_save_and_load(self, tmp_path):
        """Manifest can be saved and loaded."""
        manifest_path = tmp_path / "plugins" / "test" / "manifest.json"
        manifest = PluginManifest(
            name="file-test",
            entry_point="file:run",
            description="File test plugin",
        )

        manifest.save(manifest_path)
        assert manifest_path.exists()

        loaded = PluginManifest.load(manifest_path)
        assert loaded.name == manifest.name
        assert loaded.entry_point == manifest.entry_point

    def test_load_nonexistent_file(self, tmp_path):
        """Loading nonexistent file raises FileNotFoundError."""
        path = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            PluginManifest.load(path)

    def test_save_path_traversal_blocked(self, tmp_path):
        """Save blocks path traversal."""
        manifest = PluginManifest(name="test", entry_point="test:run")
        bad_path = tmp_path / ".." / ".." / "etc" / "manifest.json"

        with pytest.raises(ValueError, match="traversal"):
            manifest.save(bad_path)

    def test_load_path_traversal_blocked(self, tmp_path):
        """Load blocks path traversal."""
        bad_path = tmp_path / ".." / ".." / "etc" / "passwd"

        with pytest.raises(ValueError, match="traversal"):
            PluginManifest.load(bad_path)


# =============================================================================
# PluginManifest Capability Checks Tests
# =============================================================================


class TestManifestCapabilityChecks:
    """Tests for capability and requirement checking."""

    def test_has_capability_true(self):
        """has_capability returns True when capability present."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            capabilities=[PluginCapability.LINT],
        )
        assert manifest.has_capability(PluginCapability.LINT) is True

    def test_has_capability_false(self):
        """has_capability returns False when capability absent."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            capabilities=[PluginCapability.LINT],
        )
        assert manifest.has_capability(PluginCapability.SECURITY_SCAN) is False

    def test_requires_true(self):
        """requires returns True when requirement present."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            requirements=[PluginRequirement.READ_FILES],
        )
        assert manifest.requires(PluginRequirement.READ_FILES) is True

    def test_requires_false(self):
        """requires returns False when requirement absent."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            requirements=[PluginRequirement.READ_FILES],
        )
        assert manifest.requires(PluginRequirement.NETWORK) is False


# =============================================================================
# Built-in Plugins Tests
# =============================================================================


class TestBuiltinPlugins:
    """Tests for built-in plugin manifests."""

    def test_lint_builtin_exists(self):
        """Lint plugin is built-in."""
        assert "lint" in BUILTIN_MANIFESTS

    def test_security_scan_builtin_exists(self):
        """Security scan plugin is built-in."""
        assert "security-scan" in BUILTIN_MANIFESTS

    def test_test_runner_builtin_exists(self):
        """Test runner plugin is built-in."""
        assert "test-runner" in BUILTIN_MANIFESTS

    def test_list_builtin_plugins(self):
        """list_builtin_plugins returns all built-ins."""
        plugins = list_builtin_plugins()
        assert len(plugins) >= 3
        names = [p.name for p in plugins]
        assert "lint" in names
        assert "security-scan" in names

    def test_get_builtin_plugin(self):
        """get_builtin_plugin returns plugin by name."""
        plugin = get_builtin_plugin("lint")
        assert plugin is not None
        assert plugin.name == "lint"

    def test_get_builtin_plugin_nonexistent(self):
        """get_builtin_plugin returns None for unknown."""
        plugin = get_builtin_plugin("nonexistent")
        assert plugin is None

    def test_builtin_lint_has_capabilities(self):
        """Lint plugin has correct capabilities."""
        lint = get_builtin_plugin("lint")
        assert lint.has_capability(PluginCapability.LINT)
        assert lint.has_capability(PluginCapability.CODE_ANALYSIS)

    def test_builtin_security_has_capabilities(self):
        """Security scan plugin has correct capabilities."""
        security = get_builtin_plugin("security-scan")
        assert security.has_capability(PluginCapability.SECURITY_SCAN)
        assert security.has_capability(PluginCapability.CODE_ANALYSIS)

    def test_builtins_are_valid(self):
        """All built-in manifests pass validation."""
        for name, manifest in BUILTIN_MANIFESTS.items():
            is_valid, errors = manifest.validate()
            assert is_valid, f"{name} failed: {errors}"
