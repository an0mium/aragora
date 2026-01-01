"""
Plugin Manifest - Schema for declaring plugin capabilities.

Plugins must declare:
- What they do (capabilities)
- What they need (requirements)
- How to invoke them (entry point)

Manifests are validated before plugins are loaded.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import json
from pathlib import Path


class PluginCapability(Enum):
    """What a plugin can do."""

    # Analysis capabilities
    CODE_ANALYSIS = "code_analysis"      # Analyze code structure
    LINT = "lint"                        # Check code style
    SECURITY_SCAN = "security_scan"      # Security vulnerability check
    TYPE_CHECK = "type_check"            # Static type checking

    # Execution capabilities
    TEST_RUNNER = "test_runner"          # Run tests
    BENCHMARK = "benchmark"              # Performance benchmarks
    FORMATTER = "formatter"              # Code formatting

    # Evidence capabilities
    EVIDENCE_FETCH = "evidence_fetch"    # Fetch external evidence
    DOCUMENTATION = "documentation"      # Generate/check docs

    # Verification capabilities
    FORMAL_VERIFY = "formal_verify"      # Formal verification
    PROPERTY_CHECK = "property_check"    # Property-based testing

    # Utility
    CUSTOM = "custom"                    # Custom capability


class PluginRequirement(Enum):
    """What a plugin needs to run."""

    # File system
    READ_FILES = "read_files"            # Read local files
    WRITE_FILES = "write_files"          # Write local files

    # Execution
    RUN_COMMANDS = "run_commands"        # Execute shell commands
    NETWORK = "network"                  # Make network requests

    # Resources
    HIGH_MEMORY = "high_memory"          # > 1GB RAM
    LONG_RUNNING = "long_running"        # > 60s execution

    # Dependencies
    PYTHON_PACKAGES = "python_packages"  # External Python packages
    SYSTEM_TOOLS = "system_tools"        # External system tools


@dataclass
class PluginManifest:
    """
    Manifest declaring plugin metadata and requirements.

    Example manifest.json:
    {
        "name": "security-scan",
        "version": "1.0.0",
        "description": "Scan code for security vulnerabilities",
        "author": "aagora",
        "capabilities": ["security_scan", "code_analysis"],
        "requirements": ["read_files"],
        "entry_point": "security_scan:run",
        "config_schema": {...}
    }
    """

    # Identity
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = "unknown"

    # Capabilities and requirements
    capabilities: list[PluginCapability] = field(default_factory=list)
    requirements: list[PluginRequirement] = field(default_factory=list)

    # Execution
    entry_point: str = ""  # module:function format
    timeout_seconds: float = 60.0
    max_memory_mb: int = 512

    # Configuration
    config_schema: dict = field(default_factory=dict)  # JSON Schema
    default_config: dict = field(default_factory=dict)

    # Dependencies
    python_packages: list[str] = field(default_factory=list)  # pip packages
    system_tools: list[str] = field(default_factory=list)     # External tools

    # Metadata
    license: str = "MIT"
    homepage: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def validate(self) -> tuple[bool, list[str]]:
        """Validate manifest structure."""
        errors = []

        # Required fields
        if not self.name:
            errors.append("name is required")
        if not self.entry_point:
            errors.append("entry_point is required")

        # Name format
        if self.name and not self.name.replace("-", "").replace("_", "").isalnum():
            errors.append("name must be alphanumeric with - or _")

        # Entry point format
        if self.entry_point and ":" not in self.entry_point:
            errors.append("entry_point must be in module:function format")

        # Version format (semver-ish)
        if self.version:
            parts = self.version.split(".")
            if len(parts) < 2:
                errors.append("version should be semver format (x.y.z)")

        # Capabilities
        for cap in self.capabilities:
            if not isinstance(cap, PluginCapability):
                errors.append(f"Unknown capability: {cap}")

        # Requirements
        for req in self.requirements:
            if not isinstance(req, PluginRequirement):
                errors.append(f"Unknown requirement: {req}")

        # Timeout
        if self.timeout_seconds <= 0 or self.timeout_seconds > 3600:
            errors.append("timeout_seconds must be between 0 and 3600")

        return len(errors) == 0, errors

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "capabilities": [c.value for c in self.capabilities],
            "requirements": [r.value for r in self.requirements],
            "entry_point": self.entry_point,
            "timeout_seconds": self.timeout_seconds,
            "max_memory_mb": self.max_memory_mb,
            "config_schema": self.config_schema,
            "default_config": self.default_config,
            "python_packages": self.python_packages,
            "system_tools": self.system_tools,
            "license": self.license,
            "homepage": self.homepage,
            "tags": self.tags,
            "created_at": self.created_at,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Path):
        """Save manifest to file."""
        path.write_text(self.to_json())

    @classmethod
    def from_dict(cls, data: dict) -> "PluginManifest":
        """Create manifest from dictionary."""
        capabilities = []
        for c in data.get("capabilities", []):
            try:
                capabilities.append(PluginCapability(c))
            except ValueError:
                capabilities.append(PluginCapability.CUSTOM)

        requirements = []
        for r in data.get("requirements", []):
            try:
                requirements.append(PluginRequirement(r))
            except ValueError:
                pass

        return cls(
            name=data.get("name", ""),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            author=data.get("author", "unknown"),
            capabilities=capabilities,
            requirements=requirements,
            entry_point=data.get("entry_point", ""),
            timeout_seconds=data.get("timeout_seconds", 60.0),
            max_memory_mb=data.get("max_memory_mb", 512),
            config_schema=data.get("config_schema", {}),
            default_config=data.get("default_config", {}),
            python_packages=data.get("python_packages", []),
            system_tools=data.get("system_tools", []),
            license=data.get("license", "MIT"),
            homepage=data.get("homepage", ""),
            tags=data.get("tags", []),
            created_at=data.get("created_at", ""),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "PluginManifest":
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, path: Path) -> "PluginManifest":
        """Load manifest from file."""
        return cls.from_json(path.read_text())

    def has_capability(self, capability: PluginCapability) -> bool:
        """Check if plugin has a capability."""
        return capability in self.capabilities

    def requires(self, requirement: PluginRequirement) -> bool:
        """Check if plugin has a requirement."""
        return requirement in self.requirements


# Built-in plugin manifests
BUILTIN_MANIFESTS = {
    "lint": PluginManifest(
        name="lint",
        version="1.0.0",
        description="Check code for style issues using ruff/flake8",
        author="aagora",
        capabilities=[PluginCapability.LINT, PluginCapability.CODE_ANALYSIS],
        requirements=[PluginRequirement.READ_FILES, PluginRequirement.RUN_COMMANDS],
        entry_point="aagora.plugins.builtin.lint:run",
        timeout_seconds=30,
        system_tools=["ruff", "flake8"],
        tags=["code-quality", "lint"],
    ),

    "security-scan": PluginManifest(
        name="security-scan",
        version="1.0.0",
        description="Scan code for security vulnerabilities using bandit",
        author="aagora",
        capabilities=[PluginCapability.SECURITY_SCAN, PluginCapability.CODE_ANALYSIS],
        requirements=[PluginRequirement.READ_FILES, PluginRequirement.RUN_COMMANDS],
        entry_point="aagora.plugins.builtin.security:run",
        timeout_seconds=60,
        python_packages=["bandit"],
        tags=["security", "analysis"],
    ),

    "test-runner": PluginManifest(
        name="test-runner",
        version="1.0.0",
        description="Run pytest test suites",
        author="aagora",
        capabilities=[PluginCapability.TEST_RUNNER],
        requirements=[
            PluginRequirement.READ_FILES,
            PluginRequirement.RUN_COMMANDS,
            PluginRequirement.LONG_RUNNING,
        ],
        entry_point="aagora.plugins.builtin.tests:run",
        timeout_seconds=300,
        python_packages=["pytest"],
        tags=["testing", "verification"],
    ),
}


def list_builtin_plugins() -> list[PluginManifest]:
    """Get list of built-in plugin manifests."""
    return list(BUILTIN_MANIFESTS.values())


def get_builtin_plugin(name: str) -> Optional[PluginManifest]:
    """Get a built-in plugin by name."""
    return BUILTIN_MANIFESTS.get(name)
