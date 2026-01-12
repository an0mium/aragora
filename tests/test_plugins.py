"""
Tests for the Aragora Plugin System.

Tests cover:
- Plugin manifest validation
- Manifest loading and saving
- Plugin capabilities and requirements
- Built-in plugin manifests
- Security validations (path traversal)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from aragora.plugins.manifest import (
    PluginManifest,
    PluginCapability,
    PluginRequirement,
    BUILTIN_MANIFESTS,
    _validate_manifest_path,
)


class TestPluginCapabilities:
    """Test PluginCapability enum."""

    def test_all_capabilities_exist(self):
        """Test that expected capabilities exist."""
        expected = [
            "CODE_ANALYSIS", "LINT", "SECURITY_SCAN", "TYPE_CHECK",
            "TEST_RUNNER", "BENCHMARK", "FORMATTER",
            "EVIDENCE_FETCH", "DOCUMENTATION",
            "FORMAL_VERIFY", "PROPERTY_CHECK",
            "CUSTOM"
        ]
        actual = [c.name for c in PluginCapability]
        for cap in expected:
            assert cap in actual

    def test_capability_values(self):
        """Test capability values are lowercase."""
        for cap in PluginCapability:
            assert cap.value == cap.name.lower()


class TestPluginRequirements:
    """Test PluginRequirement enum."""

    def test_all_requirements_exist(self):
        """Test that expected requirements exist."""
        expected = [
            "READ_FILES", "WRITE_FILES",
            "RUN_COMMANDS", "NETWORK",
            "HIGH_MEMORY", "LONG_RUNNING",
            "PYTHON_PACKAGES", "SYSTEM_TOOLS"
        ]
        actual = [r.name for r in PluginRequirement]
        for req in expected:
            assert req in actual


class TestPluginManifestCreation:
    """Test PluginManifest creation."""

    def test_minimal_manifest(self):
        """Test creating manifest with minimal fields."""
        manifest = PluginManifest(
            name="test-plugin",
            entry_point="test:run",
        )
        assert manifest.name == "test-plugin"
        assert manifest.entry_point == "test:run"
        assert manifest.version == "1.0.0"
        assert manifest.capabilities == []
        assert manifest.requirements == []

    def test_full_manifest(self):
        """Test creating manifest with all fields."""
        manifest = PluginManifest(
            name="full-plugin",
            version="2.1.3",
            description="A full test plugin",
            author="test-author",
            capabilities=[PluginCapability.LINT, PluginCapability.CODE_ANALYSIS],
            requirements=[PluginRequirement.READ_FILES],
            entry_point="full_plugin:run",
            timeout_seconds=120,
            max_memory_mb=1024,
            config_schema={"type": "object"},
            default_config={"verbose": True},
            python_packages=["pytest"],
            system_tools=["git"],
            license="Apache-2.0",
            homepage="https://example.com",
            tags=["test", "example"],
        )

        assert manifest.name == "full-plugin"
        assert manifest.version == "2.1.3"
        assert manifest.description == "A full test plugin"
        assert manifest.author == "test-author"
        assert len(manifest.capabilities) == 2
        assert len(manifest.requirements) == 1
        assert manifest.timeout_seconds == 120
        assert manifest.max_memory_mb == 1024
        assert manifest.license == "Apache-2.0"


class TestPluginManifestValidation:
    """Test PluginManifest validation."""

    def test_valid_manifest(self):
        """Test validation passes for valid manifest."""
        manifest = PluginManifest(
            name="valid-plugin",
            entry_point="valid:run",
        )
        is_valid, errors = manifest.validate()
        assert is_valid is True
        assert errors == []

    def test_missing_name(self):
        """Test validation fails for missing name."""
        manifest = PluginManifest(
            name="",
            entry_point="test:run",
        )
        is_valid, errors = manifest.validate()
        assert is_valid is False
        assert "name is required" in errors

    def test_missing_entry_point(self):
        """Test validation fails for missing entry_point."""
        manifest = PluginManifest(
            name="test",
            entry_point="",
        )
        is_valid, errors = manifest.validate()
        assert is_valid is False
        assert "entry_point is required" in errors

    def test_invalid_name_format(self):
        """Test validation fails for invalid name format."""
        manifest = PluginManifest(
            name="invalid name!",
            entry_point="test:run",
        )
        is_valid, errors = manifest.validate()
        assert is_valid is False
        assert any("alphanumeric" in e for e in errors)

    def test_valid_name_with_dashes(self):
        """Test validation passes for name with dashes."""
        manifest = PluginManifest(
            name="my-test-plugin",
            entry_point="test:run",
        )
        is_valid, errors = manifest.validate()
        assert is_valid is True

    def test_invalid_entry_point_format(self):
        """Test validation fails for invalid entry_point format."""
        manifest = PluginManifest(
            name="test",
            entry_point="no_colon_here",
        )
        is_valid, errors = manifest.validate()
        assert is_valid is False
        assert any("module:function" in e for e in errors)

    def test_invalid_timeout(self):
        """Test validation fails for invalid timeout."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            timeout_seconds=0,
        )
        is_valid, errors = manifest.validate()
        assert is_valid is False
        assert any("timeout" in e for e in errors)


class TestPluginManifestSerialization:
    """Test PluginManifest serialization."""

    def test_to_dict(self):
        """Test converting manifest to dictionary."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            capabilities=[PluginCapability.LINT],
            requirements=[PluginRequirement.READ_FILES],
        )

        data = manifest.to_dict()

        assert data["name"] == "test"
        assert data["entry_point"] == "test:run"
        assert data["capabilities"] == ["lint"]
        assert data["requirements"] == ["read_files"]

    def test_to_json(self):
        """Test converting manifest to JSON."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
        )

        json_str = manifest.to_json()
        parsed = json.loads(json_str)

        assert parsed["name"] == "test"
        assert parsed["entry_point"] == "test:run"

    def test_from_dict(self):
        """Test creating manifest from dictionary."""
        data = {
            "name": "from-dict",
            "version": "2.0.0",
            "entry_point": "dict:run",
            "capabilities": ["lint", "code_analysis"],
            "requirements": ["read_files"],
        }

        manifest = PluginManifest.from_dict(data)

        assert manifest.name == "from-dict"
        assert manifest.version == "2.0.0"
        assert PluginCapability.LINT in manifest.capabilities
        assert PluginCapability.CODE_ANALYSIS in manifest.capabilities
        assert PluginRequirement.READ_FILES in manifest.requirements

    def test_from_dict_unknown_capability(self):
        """Test from_dict handles unknown capabilities gracefully."""
        data = {
            "name": "test",
            "entry_point": "test:run",
            "capabilities": ["unknown_cap", "lint"],
        }

        manifest = PluginManifest.from_dict(data)

        # Unknown should become CUSTOM
        assert PluginCapability.CUSTOM in manifest.capabilities
        assert PluginCapability.LINT in manifest.capabilities

    def test_from_json(self):
        """Test creating manifest from JSON string."""
        json_str = '{"name": "json-plugin", "entry_point": "json:run", "version": "1.2.3"}'

        manifest = PluginManifest.from_json(json_str)

        assert manifest.name == "json-plugin"
        assert manifest.version == "1.2.3"

    def test_from_json_invalid(self):
        """Test from_json raises on invalid JSON."""
        with pytest.raises(ValueError, match="Invalid plugin manifest JSON"):
            PluginManifest.from_json("not valid json {")

    def test_roundtrip(self):
        """Test manifest survives dict roundtrip."""
        original = PluginManifest(
            name="roundtrip",
            version="1.5.0",
            description="Test roundtrip",
            entry_point="round:trip",
            capabilities=[PluginCapability.LINT, PluginCapability.TEST_RUNNER],
            requirements=[PluginRequirement.READ_FILES, PluginRequirement.RUN_COMMANDS],
            timeout_seconds=90,
            tags=["test", "roundtrip"],
        )

        data = original.to_dict()
        restored = PluginManifest.from_dict(data)

        assert restored.name == original.name
        assert restored.version == original.version
        assert restored.description == original.description
        assert restored.entry_point == original.entry_point
        assert set(restored.capabilities) == set(original.capabilities)
        assert set(restored.requirements) == set(original.requirements)


class TestPluginManifestFileOps:
    """Test PluginManifest file operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_save_and_load(self, temp_dir):
        """Test saving and loading manifest from file."""
        manifest = PluginManifest(
            name="file-test",
            entry_point="file:test",
            version="1.0.0",
        )

        manifest_path = temp_dir / "manifest.json"
        manifest.save(manifest_path)

        loaded = PluginManifest.load(manifest_path)

        assert loaded.name == manifest.name
        assert loaded.entry_point == manifest.entry_point
        assert loaded.version == manifest.version

    def test_load_nonexistent(self, temp_dir):
        """Test loading nonexistent manifest raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PluginManifest.load(temp_dir / "nonexistent.json")


class TestPluginManifestCapabilityMethods:
    """Test capability and requirement helper methods."""

    def test_has_capability(self):
        """Test has_capability method."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            capabilities=[PluginCapability.LINT, PluginCapability.CODE_ANALYSIS],
        )

        assert manifest.has_capability(PluginCapability.LINT) is True
        assert manifest.has_capability(PluginCapability.CODE_ANALYSIS) is True
        assert manifest.has_capability(PluginCapability.SECURITY_SCAN) is False

    def test_requires(self):
        """Test requires method."""
        manifest = PluginManifest(
            name="test",
            entry_point="test:run",
            requirements=[PluginRequirement.READ_FILES, PluginRequirement.NETWORK],
        )

        assert manifest.requires(PluginRequirement.READ_FILES) is True
        assert manifest.requires(PluginRequirement.NETWORK) is True
        assert manifest.requires(PluginRequirement.RUN_COMMANDS) is False


class TestPathValidation:
    """Test manifest path validation security."""

    def test_valid_path(self):
        """Test valid path passes validation."""
        is_valid, error = _validate_manifest_path(Path("/safe/path/manifest.json"))
        assert is_valid is True
        assert error == ""

    def test_path_traversal_blocked(self):
        """Test path traversal is blocked."""
        is_valid, error = _validate_manifest_path(Path("/safe/../etc/passwd"))
        assert is_valid is False
        assert "traversal" in error.lower()

    def test_invalid_extension(self):
        """Test non-json extension is rejected."""
        is_valid, error = _validate_manifest_path(Path("/safe/manifest.py"))
        assert is_valid is False
        assert ".json" in error


class TestBuiltinManifests:
    """Test built-in plugin manifests."""

    def test_lint_manifest_exists(self):
        """Test lint manifest exists and is valid."""
        assert "lint" in BUILTIN_MANIFESTS
        manifest = BUILTIN_MANIFESTS["lint"]
        is_valid, errors = manifest.validate()
        assert is_valid is True, f"Lint manifest invalid: {errors}"

    def test_security_scan_manifest_exists(self):
        """Test security-scan manifest exists and is valid."""
        assert "security-scan" in BUILTIN_MANIFESTS
        manifest = BUILTIN_MANIFESTS["security-scan"]
        is_valid, errors = manifest.validate()
        assert is_valid is True, f"Security-scan manifest invalid: {errors}"

    def test_all_builtin_manifests_valid(self):
        """Test all built-in manifests are valid."""
        for name, manifest in BUILTIN_MANIFESTS.items():
            is_valid, errors = manifest.validate()
            assert is_valid is True, f"Built-in manifest '{name}' is invalid: {errors}"

    def test_lint_manifest_has_correct_capabilities(self):
        """Test lint manifest has expected capabilities."""
        manifest = BUILTIN_MANIFESTS["lint"]
        assert manifest.has_capability(PluginCapability.LINT)
        assert manifest.has_capability(PluginCapability.CODE_ANALYSIS)


class TestPluginManifestDefaults:
    """Test default values for PluginManifest."""

    def test_default_version(self):
        """Test default version is 1.0.0."""
        manifest = PluginManifest(name="test", entry_point="test:run")
        assert manifest.version == "1.0.0"

    def test_default_timeout(self):
        """Test default timeout is 60 seconds."""
        manifest = PluginManifest(name="test", entry_point="test:run")
        assert manifest.timeout_seconds == 60.0

    def test_default_max_memory(self):
        """Test default max memory is 512 MB."""
        manifest = PluginManifest(name="test", entry_point="test:run")
        assert manifest.max_memory_mb == 512

    def test_default_license(self):
        """Test default license is MIT."""
        manifest = PluginManifest(name="test", entry_point="test:run")
        assert manifest.license == "MIT"

    def test_created_at_auto_set(self):
        """Test created_at is automatically set."""
        manifest = PluginManifest(name="test", entry_point="test:run")
        assert manifest.created_at is not None
        assert len(manifest.created_at) > 0
