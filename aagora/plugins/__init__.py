"""
Plugin System - Extensible tools for aagora debates.

Provides a manifest-based plugin architecture with sandboxed execution:
- PluginManifest: Schema for declaring plugin capabilities
- PluginRunner: Sandboxed execution environment
- Built-in plugins: lint, security-scan, test-runner

Key design decisions:
- Python-first (not WASM): Uses ProofExecutor's restricted namespace
- Manifest validation: Plugins must declare their requirements
- Curated set: Ship with core distribution, user plugins later
"""

from aagora.plugins.manifest import PluginManifest, PluginCapability, PluginRequirement
from aagora.plugins.runner import PluginRunner, PluginResult, PluginContext

__all__ = [
    "PluginManifest",
    "PluginCapability",
    "PluginRequirement",
    "PluginRunner",
    "PluginResult",
    "PluginContext",
]
