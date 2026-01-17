"""
Audit Type Registry.

Central registry for discovering and managing audit types.
Supports both built-in auditors and custom plugins.

Usage:
    from aragora.audit.registry import audit_registry

    # List available audit types
    audit_types = audit_registry.list_audit_types()

    # Get specific auditor
    security_auditor = audit_registry.get("security")

    # Register custom auditor
    audit_registry.register(MyCustomAuditor())

    # Load preset configuration
    auditors = audit_registry.load_preset("legal_due_diligence")
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Type

import yaml

if TYPE_CHECKING:
    from aragora.audit.base_auditor import BaseAuditor

logger = logging.getLogger(__name__)


@dataclass
class AuditTypeInfo:
    """Information about a registered audit type."""

    id: str
    display_name: str
    description: str
    version: str
    author: str
    capabilities: dict[str, Any]
    is_builtin: bool = True


@dataclass
class PresetConfig:
    """Configuration for an audit preset."""

    name: str
    description: str = ""
    audit_types: list[str] = field(default_factory=list)
    custom_rules: list[dict[str, Any]] = field(default_factory=list)
    consensus_threshold: float = 0.8
    agents: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_content: str) -> "PresetConfig":
        """Load preset from YAML content."""
        data = yaml.safe_load(yaml_content)
        return cls(
            name=data.get("name", "Unnamed Preset"),
            description=data.get("description", ""),
            audit_types=data.get("audit_types", []),
            custom_rules=data.get("custom_rules", []),
            consensus_threshold=data.get("consensus_threshold", 0.8),
            agents=data.get("agents", {}),
            parameters=data.get("parameters", {}),
        )

    @classmethod
    def from_file(cls, path: Path) -> "PresetConfig":
        """Load preset from YAML file."""
        with open(path, "r") as f:
            return cls.from_yaml(f.read())


class AuditRegistry:
    """
    Registry for audit type plugins.

    Manages registration, discovery, and instantiation of
    audit types, both built-in and custom.

    Thread-safe singleton pattern.
    """

    _instance: Optional["AuditRegistry"] = None
    _initialized: bool = False

    def __new__(cls) -> "AuditRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._auditors: dict[str, "BaseAuditor"] = {}
        self._auditor_classes: dict[str, Type["BaseAuditor"]] = {}
        self._presets: dict[str, PresetConfig] = {}
        self._legacy_auditors: dict[str, Any] = {}
        self._initialized = True

    def register(
        self,
        auditor: "BaseAuditor",
        *,
        override: bool = False,
    ) -> None:
        """
        Register an auditor instance.

        Args:
            auditor: The auditor instance to register
            override: If True, replace existing auditor with same ID

        Raises:
            ValueError: If auditor with same ID exists and override=False
        """
        audit_type_id = auditor.audit_type_id

        if audit_type_id in self._auditors and not override:
            raise ValueError(
                f"Auditor '{audit_type_id}' is already registered. "
                f"Use override=True to replace."
            )

        self._auditors[audit_type_id] = auditor
        logger.info(
            f"Registered auditor: {audit_type_id} (v{auditor.version}) " f"by {auditor.author}"
        )

    def register_class(
        self,
        auditor_class: Type["BaseAuditor"],
        *,
        override: bool = False,
    ) -> None:
        """
        Register an auditor class (lazy instantiation).

        Args:
            auditor_class: The auditor class to register
            override: If True, replace existing registration
        """
        # Create temporary instance to get metadata
        temp_instance = auditor_class()
        audit_type_id = temp_instance.audit_type_id

        if audit_type_id in self._auditor_classes and not override:
            raise ValueError(f"Auditor class '{audit_type_id}' is already registered.")

        self._auditor_classes[audit_type_id] = auditor_class
        logger.debug(f"Registered auditor class: {audit_type_id}")

    def register_legacy(
        self,
        audit_type_id: str,
        auditor: Any,
        *,
        display_name: str = "",
        description: str = "",
    ) -> None:
        """
        Register a legacy auditor that doesn't inherit from BaseAuditor.

        This maintains backward compatibility with existing auditors.

        Args:
            audit_type_id: Unique identifier for the audit type
            auditor: The legacy auditor instance
            display_name: Human-readable name
            description: Description of what it does
        """
        self._legacy_auditors[audit_type_id] = {
            "instance": auditor,
            "display_name": display_name or audit_type_id.title(),
            "description": description,
        }
        logger.debug(f"Registered legacy auditor: {audit_type_id}")

    def get(self, audit_type_id: str) -> Optional["BaseAuditor"]:
        """
        Get an auditor by ID.

        Args:
            audit_type_id: The audit type identifier

        Returns:
            The auditor instance, or None if not found
        """
        # Check direct registrations first
        if audit_type_id in self._auditors:
            return self._auditors[audit_type_id]

        # Check class registrations (lazy instantiation)
        if audit_type_id in self._auditor_classes:
            instance = self._auditor_classes[audit_type_id]()
            self._auditors[audit_type_id] = instance
            return instance

        return None

    def get_legacy(self, audit_type_id: str) -> Optional[Any]:
        """Get a legacy auditor by ID."""
        entry = self._legacy_auditors.get(audit_type_id)
        return entry["instance"] if entry else None

    def get_any(self, audit_type_id: str) -> Optional[Any]:
        """
        Get either a new-style or legacy auditor.

        Checks new-style first, falls back to legacy.
        """
        auditor = self.get(audit_type_id)
        if auditor:
            return auditor
        return self.get_legacy(audit_type_id)

    def list_audit_types(self) -> list[AuditTypeInfo]:
        """
        List all registered audit types.

        Returns:
            List of audit type information
        """
        result = []

        # New-style auditors
        for audit_type_id in set(list(self._auditors.keys()) + list(self._auditor_classes.keys())):
            auditor = self.get(audit_type_id)
            if auditor:
                caps = auditor.capabilities
                result.append(
                    AuditTypeInfo(
                        id=auditor.audit_type_id,
                        display_name=auditor.display_name,
                        description=auditor.description,
                        version=auditor.version,
                        author=auditor.author,
                        capabilities={
                            "supports_chunk_analysis": caps.supports_chunk_analysis,
                            "supports_cross_document": caps.supports_cross_document,
                            "supports_streaming": caps.supports_streaming,
                            "requires_llm": caps.requires_llm,
                            "finding_categories": caps.finding_categories,
                        },
                        is_builtin=True,
                    )
                )

        # Legacy auditors
        for audit_type_id, entry in self._legacy_auditors.items():
            result.append(
                AuditTypeInfo(
                    id=audit_type_id,
                    display_name=entry["display_name"],
                    description=entry["description"],
                    version="1.0.0",
                    author="aragora",
                    capabilities={
                        "supports_chunk_analysis": True,
                        "supports_cross_document": False,
                        "requires_llm": True,
                    },
                    is_builtin=True,
                )
            )

        return result

    def get_ids(self) -> list[str]:
        """Get list of all registered audit type IDs."""
        return list(
            set(
                list(self._auditors.keys())
                + list(self._auditor_classes.keys())
                + list(self._legacy_auditors.keys())
            )
        )

    # Preset management

    def register_preset(
        self,
        preset: PresetConfig,
        *,
        override: bool = False,
    ) -> None:
        """Register an audit preset."""
        name = preset.name.lower().replace(" ", "_")
        if name in self._presets and not override:
            raise ValueError(f"Preset '{name}' is already registered.")
        self._presets[name] = preset
        logger.debug(f"Registered preset: {name}")

    def load_preset_file(self, path: Path) -> PresetConfig:
        """Load and register a preset from file."""
        preset = PresetConfig.from_file(path)
        self.register_preset(preset)
        return preset

    def get_preset(self, name: str) -> Optional[PresetConfig]:
        """Get a registered preset by name."""
        normalized = name.lower().replace(" ", "_")
        return self._presets.get(normalized)

    def list_presets(self) -> list[PresetConfig]:
        """List all registered presets."""
        return list(self._presets.values())

    def get_auditors_for_preset(
        self, preset_name: str
    ) -> list[tuple[str, Optional["BaseAuditor"]]]:
        """
        Get all auditors specified by a preset.

        Returns:
            List of (audit_type_id, auditor) tuples
        """
        preset = self.get_preset(preset_name)
        if not preset:
            return []

        return [
            (audit_type_id, self.get_any(audit_type_id)) for audit_type_id in preset.audit_types
        ]

    # Auto-discovery

    def discover_builtins(self) -> int:
        """
        Discover and register built-in auditors.

        Returns:
            Number of auditors discovered
        """
        count = 0

        # Import built-in audit types
        try:
            from aragora.audit.audit_types import (
                ComplianceAuditor,
                ConsistencyAuditor,
                QualityAuditor,
                SecurityAuditor,
            )

            # Register as legacy (they don't inherit from BaseAuditor yet)
            builtins = [
                (
                    "security",
                    SecurityAuditor(),
                    "Security Analysis",
                    "Detects credentials, injection vulnerabilities, and security risks",
                ),
                (
                    "compliance",
                    ComplianceAuditor(),
                    "Compliance Check",
                    "Checks GDPR, HIPAA, SOC2, and contractual compliance",
                ),
                (
                    "consistency",
                    ConsistencyAuditor(),
                    "Consistency Analysis",
                    "Finds cross-document contradictions and inconsistencies",
                ),
                (
                    "quality",
                    QualityAuditor(),
                    "Quality Assessment",
                    "Evaluates ambiguity, completeness, and documentation quality",
                ),
            ]

            for audit_type_id, instance, display_name, description in builtins:
                self.register_legacy(
                    audit_type_id,
                    instance,
                    display_name=display_name,
                    description=description,
                )
                count += 1

        except ImportError as e:
            logger.warning(f"Could not import built-in auditors: {e}")

        return count

    def discover_plugins(self, plugin_dirs: Optional[list[Path]] = None) -> int:
        """
        Discover auditor plugins from directories.

        Looks for Python modules that export a class inheriting from BaseAuditor.

        Args:
            plugin_dirs: Directories to search. Defaults to standard locations.

        Returns:
            Number of plugins discovered
        """
        if plugin_dirs is None:
            # Default plugin locations
            plugin_dirs = [
                Path.home() / ".aragora" / "plugins",
                Path("/etc/aragora/plugins"),
            ]

        count = 0

        for plugin_dir in plugin_dirs:
            if not plugin_dir.exists():
                continue

            for module_info in pkgutil.iter_modules([str(plugin_dir)]):
                try:
                    spec = importlib.util.spec_from_file_location(
                        module_info.name,
                        plugin_dir / f"{module_info.name}.py",
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        # Look for BaseAuditor subclasses
                        from aragora.audit.base_auditor import BaseAuditor

                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (
                                isinstance(attr, type)
                                and issubclass(attr, BaseAuditor)
                                and attr is not BaseAuditor
                            ):
                                self.register(attr())
                                count += 1
                                logger.info(
                                    f"Discovered plugin auditor: {attr_name} "
                                    f"from {plugin_dir / module_info.name}.py"
                                )

                except Exception as e:
                    logger.warning(f"Failed to load plugin {module_info.name}: {e}")

        return count

    def discover_presets(self, preset_dirs: Optional[list[Path]] = None) -> int:
        """
        Discover preset configurations from directories.

        Looks for YAML files with preset configurations.

        Args:
            preset_dirs: Directories to search. Defaults to standard locations.

        Returns:
            Number of presets discovered
        """
        if preset_dirs is None:
            # Default preset locations
            preset_dirs = [
                Path(__file__).parent / "presets",
                Path.home() / ".aragora" / "presets",
                Path("/etc/aragora/presets"),
            ]

        count = 0

        for preset_dir in preset_dirs:
            if not preset_dir.exists():
                continue

            for yaml_file in preset_dir.glob("*.yaml"):
                try:
                    preset = PresetConfig.from_file(yaml_file)
                    self.register_preset(preset)
                    count += 1
                    logger.debug(f"Discovered preset: {preset.name} from {yaml_file}")
                except Exception as e:
                    logger.warning(f"Failed to load preset {yaml_file}: {e}")

            for yaml_file in preset_dir.glob("*.yml"):
                try:
                    preset = PresetConfig.from_file(yaml_file)
                    self.register_preset(preset)
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to load preset {yaml_file}: {e}")

        return count

    def auto_discover(self) -> dict[str, int]:
        """
        Run all discovery mechanisms.

        Returns:
            Dictionary with counts of discovered items by type
        """
        return {
            "builtins": self.discover_builtins(),
            "plugins": self.discover_plugins(),
            "presets": self.discover_presets(),
        }

    def clear(self) -> None:
        """Clear all registrations. Useful for testing."""
        self._auditors.clear()
        self._auditor_classes.clear()
        self._presets.clear()
        self._legacy_auditors.clear()


# Global registry instance
audit_registry = AuditRegistry()


def get_registry() -> AuditRegistry:
    """Get the global audit registry."""
    return audit_registry


# Convenience functions


def register_auditor(auditor: "BaseAuditor", override: bool = False) -> None:
    """Register an auditor with the global registry."""
    audit_registry.register(auditor, override=override)


def get_auditor(audit_type_id: str) -> Optional["BaseAuditor"]:
    """Get an auditor from the global registry."""
    return audit_registry.get(audit_type_id)


def list_audit_types() -> list[AuditTypeInfo]:
    """List all audit types in the global registry."""
    return audit_registry.list_audit_types()
