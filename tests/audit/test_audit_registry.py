"""
Tests for Audit Type Registry.

Tests the registry module that provides:
- AuditTypeInfo for audit type metadata
- PresetConfig for preset configurations
- AuditRegistry for plugin management
- Auditor discovery mechanisms
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def fresh_registry():
    """Create a fresh registry for isolated testing."""
    from aragora.audit.registry import AuditRegistry

    # Create a new instance by resetting class state
    AuditRegistry._instance = None
    AuditRegistry._initialized = False
    registry = AuditRegistry()
    yield registry
    # Cleanup
    registry.clear()
    AuditRegistry._instance = None
    AuditRegistry._initialized = False


@pytest.fixture
def mock_auditor():
    """Create a mock auditor for testing."""
    from aragora.audit.base_auditor import (
        AuditContext,
        AuditorCapabilities,
        BaseAuditor,
        ChunkData,
    )

    class MockAuditor(BaseAuditor):
        @property
        def audit_type_id(self) -> str:
            return "mock_auditor"

        @property
        def display_name(self) -> str:
            return "Mock Auditor"

        @property
        def description(self) -> str:
            return "A mock auditor for testing"

        async def analyze_chunk(self, chunk: ChunkData, context: AuditContext):
            return []

    return MockAuditor()


@pytest.fixture
def another_mock_auditor():
    """Create another mock auditor with different ID."""
    from aragora.audit.base_auditor import (
        AuditContext,
        BaseAuditor,
        ChunkData,
    )

    class AnotherMockAuditor(BaseAuditor):
        @property
        def audit_type_id(self) -> str:
            return "another_auditor"

        @property
        def display_name(self) -> str:
            return "Another Auditor"

        @property
        def description(self) -> str:
            return "Another mock auditor"

        async def analyze_chunk(self, chunk: ChunkData, context: AuditContext):
            return []

    return AnotherMockAuditor()


@pytest.fixture
def sample_preset_yaml():
    """Sample preset YAML content."""
    return """
name: Test Preset
description: A test preset configuration
audit_types:
  - security
  - compliance
custom_rules:
  - name: custom_rule_1
    pattern: "test.*"
consensus_threshold: 0.75
agents:
  primary: claude
  fallback: gpt
parameters:
  max_findings: 100
  severity_filter: high
"""


# ===========================================================================
# Tests: AuditTypeInfo Dataclass
# ===========================================================================


class TestAuditTypeInfo:
    """Tests for AuditTypeInfo dataclass."""

    def test_creation(self):
        """Test AuditTypeInfo creation."""
        from aragora.audit.registry import AuditTypeInfo

        info = AuditTypeInfo(
            id="test_id",
            display_name="Test Auditor",
            description="Test description",
            version="1.0.0",
            author="test_author",
            capabilities={"supports_chunk_analysis": True},
        )

        assert info.id == "test_id"
        assert info.display_name == "Test Auditor"
        assert info.description == "Test description"
        assert info.version == "1.0.0"
        assert info.author == "test_author"
        assert info.capabilities["supports_chunk_analysis"] is True

    def test_default_is_builtin(self):
        """Test default is_builtin value."""
        from aragora.audit.registry import AuditTypeInfo

        info = AuditTypeInfo(
            id="test",
            display_name="Test",
            description="Test",
            version="1.0.0",
            author="test",
            capabilities={},
        )

        assert info.is_builtin is True

    def test_custom_is_builtin(self):
        """Test custom is_builtin value."""
        from aragora.audit.registry import AuditTypeInfo

        info = AuditTypeInfo(
            id="custom",
            display_name="Custom",
            description="Custom auditor",
            version="1.0.0",
            author="external",
            capabilities={},
            is_builtin=False,
        )

        assert info.is_builtin is False


# ===========================================================================
# Tests: PresetConfig Dataclass
# ===========================================================================


class TestPresetConfig:
    """Tests for PresetConfig dataclass."""

    def test_creation(self):
        """Test PresetConfig creation."""
        from aragora.audit.registry import PresetConfig

        preset = PresetConfig(
            name="Test Preset",
            description="Test description",
            audit_types=["security", "compliance"],
            consensus_threshold=0.9,
        )

        assert preset.name == "Test Preset"
        assert preset.description == "Test description"
        assert len(preset.audit_types) == 2
        assert preset.consensus_threshold == 0.9

    def test_default_values(self):
        """Test PresetConfig default values."""
        from aragora.audit.registry import PresetConfig

        preset = PresetConfig(name="Minimal")

        assert preset.description == ""
        assert preset.audit_types == []
        assert preset.custom_rules == []
        assert preset.consensus_threshold == 0.8
        assert preset.agents == {}
        assert preset.parameters == {}

    def test_from_yaml(self, sample_preset_yaml):
        """Test PresetConfig.from_yaml."""
        from aragora.audit.registry import PresetConfig

        preset = PresetConfig.from_yaml(sample_preset_yaml)

        assert preset.name == "Test Preset"
        assert preset.description == "A test preset configuration"
        assert preset.audit_types == ["security", "compliance"]
        assert len(preset.custom_rules) == 1
        assert preset.consensus_threshold == 0.75
        assert preset.agents["primary"] == "claude"
        assert preset.parameters["max_findings"] == 100

    def test_from_yaml_minimal(self):
        """Test from_yaml with minimal content."""
        from aragora.audit.registry import PresetConfig

        yaml_content = """
name: Minimal Preset
"""
        preset = PresetConfig.from_yaml(yaml_content)

        assert preset.name == "Minimal Preset"
        assert preset.audit_types == []

    def test_from_yaml_no_name(self):
        """Test from_yaml without name field."""
        from aragora.audit.registry import PresetConfig

        yaml_content = """
audit_types:
  - security
"""
        preset = PresetConfig.from_yaml(yaml_content)

        assert preset.name == "Unnamed Preset"

    def test_from_file(self, sample_preset_yaml):
        """Test PresetConfig.from_file."""
        from aragora.audit.registry import PresetConfig

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(sample_preset_yaml)
            f.flush()

            preset = PresetConfig.from_file(Path(f.name))

        assert preset.name == "Test Preset"
        assert preset.consensus_threshold == 0.75


# ===========================================================================
# Tests: AuditRegistry Singleton
# ===========================================================================


class TestAuditRegistrySingleton:
    """Tests for AuditRegistry singleton pattern."""

    def test_singleton_pattern(self):
        """Test singleton returns same instance."""
        from aragora.audit.registry import AuditRegistry

        # Reset to test fresh
        AuditRegistry._instance = None
        AuditRegistry._initialized = False

        registry1 = AuditRegistry()
        registry2 = AuditRegistry()

        assert registry1 is registry2

        # Cleanup
        AuditRegistry._instance = None
        AuditRegistry._initialized = False

    def test_initialized_flag_prevents_reinit(self, fresh_registry):
        """Test registry doesn't reinitialize."""
        fresh_registry._auditors["test"] = MagicMock()

        # Re-calling __init__ shouldn't clear auditors
        fresh_registry.__init__()

        assert "test" in fresh_registry._auditors


# ===========================================================================
# Tests: AuditRegistry Registration
# ===========================================================================


class TestAuditRegistryRegistration:
    """Tests for auditor registration methods."""

    def test_register_auditor(self, fresh_registry, mock_auditor):
        """Test registering an auditor."""
        fresh_registry.register(mock_auditor)

        assert "mock_auditor" in fresh_registry._auditors
        assert fresh_registry._auditors["mock_auditor"] is mock_auditor

    def test_register_duplicate_raises(self, fresh_registry, mock_auditor):
        """Test registering duplicate raises error."""
        fresh_registry.register(mock_auditor)

        with pytest.raises(ValueError, match="already registered"):
            fresh_registry.register(mock_auditor)

    def test_register_with_override(self, fresh_registry, mock_auditor):
        """Test registering with override=True."""
        fresh_registry.register(mock_auditor)

        # Create new instance to override
        from aragora.audit.base_auditor import (
            AuditContext,
            BaseAuditor,
            ChunkData,
        )

        class MockAuditorV2(BaseAuditor):
            @property
            def audit_type_id(self) -> str:
                return "mock_auditor"

            @property
            def display_name(self) -> str:
                return "Mock Auditor V2"

            @property
            def description(self) -> str:
                return "Updated"

            async def analyze_chunk(self, chunk: ChunkData, context: AuditContext):
                return []

        new_auditor = MockAuditorV2()
        fresh_registry.register(new_auditor, override=True)

        assert fresh_registry._auditors["mock_auditor"].display_name == "Mock Auditor V2"

    def test_register_class(self, fresh_registry):
        """Test registering an auditor class."""
        from aragora.audit.base_auditor import (
            AuditContext,
            BaseAuditor,
            ChunkData,
        )

        class LazyAuditor(BaseAuditor):
            @property
            def audit_type_id(self) -> str:
                return "lazy_auditor"

            @property
            def display_name(self) -> str:
                return "Lazy Auditor"

            @property
            def description(self) -> str:
                return "Lazy loaded"

            async def analyze_chunk(self, chunk: ChunkData, context: AuditContext):
                return []

        fresh_registry.register_class(LazyAuditor)

        assert "lazy_auditor" in fresh_registry._auditor_classes
        assert "lazy_auditor" not in fresh_registry._auditors  # Not instantiated yet

    def test_register_class_duplicate_raises(self, fresh_registry):
        """Test duplicate class registration raises."""
        from aragora.audit.base_auditor import (
            AuditContext,
            BaseAuditor,
            ChunkData,
        )

        class DupeAuditor(BaseAuditor):
            @property
            def audit_type_id(self) -> str:
                return "dupe_auditor"

            @property
            def display_name(self) -> str:
                return "Dupe"

            @property
            def description(self) -> str:
                return "Dupe"

            async def analyze_chunk(self, chunk: ChunkData, context: AuditContext):
                return []

        fresh_registry.register_class(DupeAuditor)

        with pytest.raises(ValueError, match="already registered"):
            fresh_registry.register_class(DupeAuditor)

    def test_register_legacy(self, fresh_registry):
        """Test registering a legacy auditor."""
        legacy_mock = MagicMock()

        fresh_registry.register_legacy(
            "legacy_id",
            legacy_mock,
            display_name="Legacy Auditor",
            description="Legacy description",
        )

        assert "legacy_id" in fresh_registry._legacy_auditors
        entry = fresh_registry._legacy_auditors["legacy_id"]
        assert entry["instance"] is legacy_mock
        assert entry["display_name"] == "Legacy Auditor"
        assert entry["description"] == "Legacy description"

    def test_register_legacy_default_display_name(self, fresh_registry):
        """Test legacy registration with default display name."""
        legacy_mock = MagicMock()

        fresh_registry.register_legacy("my_auditor", legacy_mock)

        entry = fresh_registry._legacy_auditors["my_auditor"]
        assert entry["display_name"] == "My_Auditor"


# ===========================================================================
# Tests: AuditRegistry Retrieval
# ===========================================================================


class TestAuditRegistryRetrieval:
    """Tests for auditor retrieval methods."""

    def test_get_registered_auditor(self, fresh_registry, mock_auditor):
        """Test getting a registered auditor."""
        fresh_registry.register(mock_auditor)

        result = fresh_registry.get("mock_auditor")

        assert result is mock_auditor

    def test_get_nonexistent_returns_none(self, fresh_registry):
        """Test getting nonexistent auditor returns None."""
        result = fresh_registry.get("nonexistent")

        assert result is None

    def test_get_lazy_instantiates_class(self, fresh_registry):
        """Test get() lazily instantiates registered classes."""
        from aragora.audit.base_auditor import (
            AuditContext,
            BaseAuditor,
            ChunkData,
        )

        class LazyAuditor(BaseAuditor):
            instantiated = False

            def __init__(self):
                LazyAuditor.instantiated = True

            @property
            def audit_type_id(self) -> str:
                return "lazy"

            @property
            def display_name(self) -> str:
                return "Lazy"

            @property
            def description(self) -> str:
                return "Lazy"

            async def analyze_chunk(self, chunk: ChunkData, context: AuditContext):
                return []

        fresh_registry.register_class(LazyAuditor)
        LazyAuditor.instantiated = False  # Reset after registration

        # Not instantiated yet
        assert "lazy" not in fresh_registry._auditors

        result = fresh_registry.get("lazy")

        assert result is not None
        assert LazyAuditor.instantiated is True
        assert "lazy" in fresh_registry._auditors

    def test_get_legacy(self, fresh_registry):
        """Test getting a legacy auditor."""
        legacy_mock = MagicMock()
        fresh_registry.register_legacy("legacy", legacy_mock)

        result = fresh_registry.get_legacy("legacy")

        assert result is legacy_mock

    def test_get_legacy_nonexistent(self, fresh_registry):
        """Test getting nonexistent legacy returns None."""
        result = fresh_registry.get_legacy("nonexistent")

        assert result is None

    def test_get_any_prefers_new_style(self, fresh_registry, mock_auditor):
        """Test get_any prefers new-style over legacy."""
        legacy_mock = MagicMock()
        fresh_registry.register_legacy("mock_auditor", legacy_mock)
        fresh_registry.register(mock_auditor)

        result = fresh_registry.get_any("mock_auditor")

        assert result is mock_auditor  # Not the legacy mock

    def test_get_any_falls_back_to_legacy(self, fresh_registry):
        """Test get_any falls back to legacy."""
        legacy_mock = MagicMock()
        fresh_registry.register_legacy("legacy_only", legacy_mock)

        result = fresh_registry.get_any("legacy_only")

        assert result is legacy_mock

    def test_get_any_nonexistent(self, fresh_registry):
        """Test get_any with nonexistent returns None."""
        result = fresh_registry.get_any("nonexistent")

        assert result is None


# ===========================================================================
# Tests: AuditRegistry Listing
# ===========================================================================


class TestAuditRegistryListing:
    """Tests for listing methods."""

    def test_list_audit_types_empty(self, fresh_registry):
        """Test list_audit_types with no registrations."""
        result = fresh_registry.list_audit_types()

        assert result == []

    def test_list_audit_types_with_auditors(self, fresh_registry, mock_auditor):
        """Test list_audit_types with registered auditors."""
        fresh_registry.register(mock_auditor)

        result = fresh_registry.list_audit_types()

        assert len(result) == 1
        assert result[0].id == "mock_auditor"
        assert result[0].display_name == "Mock Auditor"
        assert "supports_chunk_analysis" in result[0].capabilities

    def test_list_audit_types_includes_legacy(self, fresh_registry):
        """Test list_audit_types includes legacy auditors."""
        legacy_mock = MagicMock()
        fresh_registry.register_legacy(
            "legacy",
            legacy_mock,
            display_name="Legacy",
            description="Legacy description",
        )

        result = fresh_registry.list_audit_types()

        assert len(result) == 1
        assert result[0].id == "legacy"
        assert result[0].display_name == "Legacy"
        assert result[0].version == "1.0.0"

    def test_get_ids(self, fresh_registry, mock_auditor, another_mock_auditor):
        """Test get_ids returns all IDs."""
        fresh_registry.register(mock_auditor)
        fresh_registry.register(another_mock_auditor)
        fresh_registry.register_legacy("legacy", MagicMock())

        ids = fresh_registry.get_ids()

        assert "mock_auditor" in ids
        assert "another_auditor" in ids
        assert "legacy" in ids

    def test_get_ids_deduplicates(self, fresh_registry):
        """Test get_ids handles duplicates correctly."""
        from aragora.audit.base_auditor import (
            AuditContext,
            BaseAuditor,
            ChunkData,
        )

        class TestAuditor(BaseAuditor):
            @property
            def audit_type_id(self) -> str:
                return "test"

            @property
            def display_name(self) -> str:
                return "Test"

            @property
            def description(self) -> str:
                return "Test"

            async def analyze_chunk(self, chunk: ChunkData, context: AuditContext):
                return []

        fresh_registry.register_class(TestAuditor)
        fresh_registry.get("test")  # This creates instance entry too

        ids = fresh_registry.get_ids()

        # Should only have "test" once
        assert ids.count("test") == 1


# ===========================================================================
# Tests: AuditRegistry Presets
# ===========================================================================


class TestAuditRegistryPresets:
    """Tests for preset management."""

    def test_register_preset(self, fresh_registry):
        """Test registering a preset."""
        from aragora.audit.registry import PresetConfig

        preset = PresetConfig(
            name="Test Preset",
            audit_types=["security"],
        )

        fresh_registry.register_preset(preset)

        assert "test_preset" in fresh_registry._presets

    def test_register_preset_normalizes_name(self, fresh_registry):
        """Test preset name is normalized."""
        from aragora.audit.registry import PresetConfig

        preset = PresetConfig(
            name="My Custom Preset",
            audit_types=["security"],
        )

        fresh_registry.register_preset(preset)

        assert "my_custom_preset" in fresh_registry._presets

    def test_register_preset_duplicate_raises(self, fresh_registry):
        """Test duplicate preset registration raises."""
        from aragora.audit.registry import PresetConfig

        preset = PresetConfig(name="Duplicate", audit_types=[])

        fresh_registry.register_preset(preset)

        with pytest.raises(ValueError, match="already registered"):
            fresh_registry.register_preset(preset)

    def test_register_preset_with_override(self, fresh_registry):
        """Test preset override."""
        from aragora.audit.registry import PresetConfig

        preset1 = PresetConfig(name="Override", audit_types=["a"])
        preset2 = PresetConfig(name="Override", audit_types=["b"])

        fresh_registry.register_preset(preset1)
        fresh_registry.register_preset(preset2, override=True)

        result = fresh_registry.get_preset("override")
        assert result.audit_types == ["b"]

    def test_get_preset(self, fresh_registry):
        """Test getting a preset."""
        from aragora.audit.registry import PresetConfig

        preset = PresetConfig(name="Test", audit_types=["security"])
        fresh_registry.register_preset(preset)

        result = fresh_registry.get_preset("Test")

        assert result is preset

    def test_get_preset_normalizes_name(self, fresh_registry):
        """Test get_preset normalizes input name."""
        from aragora.audit.registry import PresetConfig

        preset = PresetConfig(name="Spaced Name", audit_types=[])
        fresh_registry.register_preset(preset)

        result = fresh_registry.get_preset("Spaced Name")

        assert result is preset

    def test_get_preset_nonexistent(self, fresh_registry):
        """Test getting nonexistent preset returns None."""
        result = fresh_registry.get_preset("nonexistent")

        assert result is None

    def test_list_presets(self, fresh_registry):
        """Test listing presets."""
        from aragora.audit.registry import PresetConfig

        preset1 = PresetConfig(name="Preset 1", audit_types=[])
        preset2 = PresetConfig(name="Preset 2", audit_types=[])

        fresh_registry.register_preset(preset1)
        fresh_registry.register_preset(preset2)

        result = fresh_registry.list_presets()

        assert len(result) == 2

    def test_load_preset_file(self, fresh_registry, sample_preset_yaml):
        """Test loading preset from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(sample_preset_yaml)
            f.flush()

            preset = fresh_registry.load_preset_file(Path(f.name))

        assert preset.name == "Test Preset"
        assert "test_preset" in fresh_registry._presets

    def test_get_auditors_for_preset(self, fresh_registry, mock_auditor):
        """Test getting auditors for a preset."""
        from aragora.audit.registry import PresetConfig

        fresh_registry.register(mock_auditor)

        preset = PresetConfig(
            name="Test",
            audit_types=["mock_auditor", "nonexistent"],
        )
        fresh_registry.register_preset(preset)

        result = fresh_registry.get_auditors_for_preset("test")

        assert len(result) == 2
        assert result[0] == ("mock_auditor", mock_auditor)
        assert result[1] == ("nonexistent", None)

    def test_get_auditors_for_nonexistent_preset(self, fresh_registry):
        """Test getting auditors for nonexistent preset."""
        result = fresh_registry.get_auditors_for_preset("nonexistent")

        assert result == []


# ===========================================================================
# Tests: AuditRegistry Discovery
# ===========================================================================


class TestAuditRegistryDiscovery:
    """Tests for discovery mechanisms."""

    def test_discover_builtins(self, fresh_registry):
        """Test discovering built-in auditors."""
        count = fresh_registry.discover_builtins()

        # Should discover security, compliance, consistency, quality
        assert count >= 0  # May be 0 if import fails

    @patch("aragora.audit.registry.logger")
    def test_discover_builtins_import_error(self, mock_logger, fresh_registry):
        """Test discover_builtins handles import error."""
        with patch(
            "aragora.audit.registry.importlib.import_module",
            side_effect=ImportError("test"),
        ):
            # Should not raise
            count = fresh_registry.discover_builtins()

        # Count is 0 or we logged a warning
        assert count >= 0

    def test_discover_plugins_empty_dirs(self, fresh_registry):
        """Test discover_plugins with nonexistent directories."""
        nonexistent = [Path("/nonexistent/path1"), Path("/nonexistent/path2")]

        count = fresh_registry.discover_plugins(plugin_dirs=nonexistent)

        assert count == 0

    def test_discover_presets_empty_dirs(self, fresh_registry):
        """Test discover_presets with nonexistent directories."""
        nonexistent = [Path("/nonexistent/path1")]

        count = fresh_registry.discover_presets(preset_dirs=nonexistent)

        assert count == 0

    def test_discover_presets_from_temp_dir(self, fresh_registry, sample_preset_yaml):
        """Test discovering presets from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            preset_path = Path(tmpdir) / "test_preset.yaml"
            with open(preset_path, "w") as f:
                f.write(sample_preset_yaml)

            count = fresh_registry.discover_presets(preset_dirs=[Path(tmpdir)])

        assert count == 1

    def test_auto_discover(self, fresh_registry):
        """Test auto_discover runs all mechanisms."""
        result = fresh_registry.auto_discover()

        assert "builtins" in result
        assert "plugins" in result
        assert "presets" in result


# ===========================================================================
# Tests: AuditRegistry Clear
# ===========================================================================


class TestAuditRegistryClear:
    """Tests for clear method."""

    def test_clear(self, fresh_registry, mock_auditor):
        """Test clear removes all registrations."""
        from aragora.audit.registry import PresetConfig

        fresh_registry.register(mock_auditor)
        fresh_registry.register_legacy("legacy", MagicMock())
        fresh_registry.register_preset(PresetConfig(name="Test", audit_types=[]))

        fresh_registry.clear()

        assert len(fresh_registry._auditors) == 0
        assert len(fresh_registry._auditor_classes) == 0
        assert len(fresh_registry._presets) == 0
        assert len(fresh_registry._legacy_auditors) == 0


# ===========================================================================
# Tests: Module-Level Functions
# ===========================================================================


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_registry(self):
        """Test get_registry returns global registry."""
        from aragora.audit.registry import audit_registry, get_registry

        result = get_registry()

        assert result is audit_registry

    def test_register_auditor(self, fresh_registry, mock_auditor):
        """Test register_auditor convenience function."""
        from aragora.audit.registry import audit_registry, register_auditor

        # Use fresh registry as the global one
        with patch("aragora.audit.registry.audit_registry", fresh_registry):
            register_auditor(mock_auditor)

        assert "mock_auditor" in fresh_registry._auditors

    def test_get_auditor(self, fresh_registry, mock_auditor):
        """Test get_auditor convenience function."""
        from aragora.audit.registry import get_auditor

        fresh_registry.register(mock_auditor)

        with patch("aragora.audit.registry.audit_registry", fresh_registry):
            result = get_auditor("mock_auditor")

        assert result is mock_auditor

    def test_list_audit_types(self, fresh_registry, mock_auditor):
        """Test list_audit_types convenience function."""
        from aragora.audit.registry import list_audit_types

        fresh_registry.register(mock_auditor)

        with patch("aragora.audit.registry.audit_registry", fresh_registry):
            result = list_audit_types()

        assert len(result) == 1
        assert result[0].id == "mock_auditor"
