"""
Plugin System - Extensible tools for aragora debates.

Provides a manifest-based plugin architecture with sandboxed execution:
- PluginManifest: Schema for declaring plugin capabilities
- PluginRunner: Sandboxed execution environment
- Built-in plugins: lint, security-scan, test-runner

Key design decisions:
- Python-first (not WASM): Uses ProofExecutor's restricted namespace
- Manifest validation: Plugins must declare their requirements
- Curated set: Ship with core distribution, user plugins later
"""

from aragora.plugins.manifest import PluginCapability, PluginManifest, PluginRequirement
from aragora.plugins.runner import PluginContext, PluginResult, PluginRunner

__all__ = [
    "PluginManifest",
    "PluginCapability",
    "PluginRequirement",
    "PluginRunner",
    "PluginResult",
    "PluginContext",
]
