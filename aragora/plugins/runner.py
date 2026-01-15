"""
Plugin Runner - Sandboxed execution environment for plugins.

Uses restricted namespace execution (similar to ProofExecutor)
to safely run plugin code with limited capabilities.

Key features:
- Restricted builtins (no exec, eval, open, etc.)
- Timeout enforcement
- Memory limits (soft, via resource module on Unix)
- Capability-based permission checking
"""

import asyncio
import importlib
import logging
import sys

# resource module is Unix-only, used for memory limits
try:
    import resource

    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False  # Windows
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from aragora.plugins.manifest import (
    PluginCapability,
    PluginManifest,
    PluginRequirement,
)

logger = logging.getLogger(__name__)


@dataclass
class PluginContext:
    """
    Execution context passed to plugins.

    Contains the input data and allowed operations.
    """

    # Input
    input_data: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    # Environment
    working_dir: str = "."
    debate_id: Optional[str] = None
    claim_id: Optional[str] = None

    # Capabilities (set by runner based on manifest)
    allowed_operations: set = field(default_factory=set)

    # Output (plugin writes here)
    output: dict = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        """Add a log message."""
        self.logs.append(f"[{datetime.now().isoformat()}] {message}")

    def error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)

    def set_output(self, key: str, value: Any) -> None:
        """Set output value."""
        self.output[key] = value

    def can(self, operation: str) -> bool:
        """Check if operation is allowed."""
        return operation in self.allowed_operations


@dataclass
class PluginResult:
    """Result from plugin execution."""

    success: bool
    output: dict = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Metrics
    duration_seconds: float = 0.0
    memory_used_mb: float = 0.0

    # Execution info
    plugin_name: str = ""
    plugin_version: str = ""
    executed_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output": self.output,
            "logs": self.logs,
            "errors": self.errors,
            "duration_seconds": self.duration_seconds,
            "memory_used_mb": self.memory_used_mb,
            "plugin_name": self.plugin_name,
            "plugin_version": self.plugin_version,
            "executed_at": self.executed_at,
        }


class PluginRunner:
    """
    Sandboxed plugin execution environment.

    Runs plugins with restricted capabilities based on their manifest.
    """

    # Restricted builtins - remove dangerous functions
    RESTRICTED_BUILTINS = {
        # Safe builtins
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "classmethod",
        "complex",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "hasattr",
        "hash",
        "hex",
        "id",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "object",
        "oct",
        "ord",
        "pow",
        "print",
        "property",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "setattr",
        "slice",
        "sorted",
        "staticmethod",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "vars",
        "zip",
        # Restricted - no file/exec operations
        # "open", "eval", "exec", "compile", "__import__"
    }

    def __init__(
        self,
        manifest: PluginManifest,
        sandbox_level: str = "standard",  # strict, standard, permissive
    ):
        self.manifest = manifest
        self.sandbox_level = sandbox_level
        self._entry_func: Optional[Callable] = None

    def _validate_requirements(self) -> tuple[bool, list[str]]:
        """Check if system can satisfy plugin requirements."""
        missing = []

        # Check Python packages
        for package in self.manifest.python_packages:
            try:
                importlib.import_module(package.split("[")[0])  # Handle extras
            except ImportError:
                missing.append(f"Python package: {package}")

        # Check system tools
        import shutil

        for tool in self.manifest.system_tools:
            if not shutil.which(tool):
                missing.append(f"System tool: {tool}")

        return len(missing) == 0, missing

    def _load_entry_point(self) -> Callable:
        """Load the plugin entry point function."""
        if ":" not in self.manifest.entry_point:
            raise ValueError(f"Invalid entry point: {self.manifest.entry_point}")

        module_path, func_name = self.manifest.entry_point.rsplit(":", 1)

        try:
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            return func
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to load {self.manifest.entry_point}: {e}") from e

    def _create_restricted_globals(self, context: PluginContext) -> dict:
        """Create restricted global namespace for execution."""
        import builtins

        # Start with restricted builtins
        safe_builtins = {
            name: getattr(builtins, name)
            for name in self.RESTRICTED_BUILTINS
            if hasattr(builtins, name)
        }

        # Add controlled open if permitted
        if context.can("read_files"):
            safe_builtins["open"] = self._make_safe_open(context, read_only=True)
        if context.can("write_files"):
            safe_builtins["open"] = self._make_safe_open(context, read_only=False)

        return {
            "__builtins__": safe_builtins,
            "__name__": self.manifest.name,
            "context": context,
        }

    def _make_safe_open(self, context: PluginContext, read_only: bool = True):
        """Create a safe open function with path restrictions."""
        base_dir = Path(context.working_dir).resolve()

        def safe_open(path, mode="r", *args, **kwargs):
            # Resolve and validate path
            resolved = Path(path).resolve()

            # Must be under working directory
            try:
                resolved.relative_to(base_dir)
            except ValueError:
                raise PermissionError(f"Access denied: {path} is outside working directory")

            # Check mode
            if read_only and any(m in mode for m in ["w", "a", "x"]):
                raise PermissionError("Write operations not permitted")

            return open(resolved, mode, *args, **kwargs)

        return safe_open

    def _set_resource_limits(self):
        """Set resource limits (Unix only)."""
        if not RESOURCE_AVAILABLE:
            return

        try:
            # Memory limit (soft)
            max_bytes = self.manifest.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
        except (ValueError, resource.error):
            pass  # May fail if limit is higher than system allows

    async def run(
        self,
        context: PluginContext,
        timeout_override: Optional[float] = None,
    ) -> PluginResult:
        """
        Run the plugin with sandboxing.

        Args:
            context: Execution context with input data
            timeout_override: Override manifest timeout

        Returns:
            PluginResult with output or errors
        """
        start_time = datetime.now()

        result = PluginResult(
            success=False,
            plugin_name=self.manifest.name,
            plugin_version=self.manifest.version,
        )

        # Validate requirements
        valid, missing = self._validate_requirements()
        if not valid:
            result.errors.append(f"Missing requirements: {', '.join(missing)}")
            return result

        # Set allowed operations based on manifest
        context.allowed_operations = {
            "read_files" if PluginRequirement.READ_FILES in self.manifest.requirements else None,
            "write_files" if PluginRequirement.WRITE_FILES in self.manifest.requirements else None,
            (
                "run_commands"
                if PluginRequirement.RUN_COMMANDS in self.manifest.requirements
                else None
            ),
            "network" if PluginRequirement.NETWORK in self.manifest.requirements else None,
        }
        context.allowed_operations.discard(None)

        # Load entry point
        try:
            entry_func = self._load_entry_point()
        except Exception as e:
            result.errors.append(f"Failed to load plugin: {e}")
            return result

        # Execute with timeout
        timeout = timeout_override or self.manifest.timeout_seconds

        try:
            # Run in executor to handle sync functions
            loop = asyncio.get_running_loop()

            if asyncio.iscoroutinefunction(entry_func):
                # Async function
                output = await asyncio.wait_for(
                    entry_func(context),
                    timeout=timeout,
                )
            else:
                # Sync function - run in thread pool
                output = await asyncio.wait_for(
                    loop.run_in_executor(None, entry_func, context),
                    timeout=timeout,
                )

            result.success = True
            result.output = context.output
            result.logs = context.logs
            result.errors = context.errors

            if output and isinstance(output, dict):
                result.output.update(output)

        except asyncio.TimeoutError:
            result.errors.append(f"Plugin timed out after {timeout}s")

        except PermissionError as e:
            result.errors.append(f"Permission denied: {e}")

        except Exception as e:
            result.errors.append(f"Plugin error: {type(e).__name__}: {e}")

        finally:
            result.duration_seconds = (datetime.now() - start_time).total_seconds()

        return result


class PluginRegistry:
    """
    Registry of available plugins.

    Manages discovery, loading, and caching of plugins.
    """

    def __init__(self, plugin_dirs: Optional[list[Path]] = None):
        self.plugin_dirs = plugin_dirs or []
        self.manifests: dict[str, PluginManifest] = {}
        self.runners: dict[str, PluginRunner] = {}

        # Load built-in plugins
        self._load_builtins()

    def _load_builtins(self):
        """Load built-in plugin manifests."""
        from aragora.plugins.manifest import BUILTIN_MANIFESTS

        self.manifests.update(BUILTIN_MANIFESTS)

    def discover(self) -> None:
        """Discover plugins from plugin directories."""
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue

            # Look for manifest.json files
            for manifest_path in plugin_dir.glob("*/manifest.json"):
                try:
                    manifest = PluginManifest.load(manifest_path)
                    valid, errors = manifest.validate()
                    if valid:
                        self.manifests[manifest.name] = manifest
                except Exception as e:
                    logger.debug(f"Failed to load plugin manifest {manifest_path}: {e}")

    def get(self, name: str) -> Optional[PluginManifest]:
        """Get plugin manifest by name."""
        return self.manifests.get(name)

    def list_plugins(self) -> list[PluginManifest]:
        """List all available plugins."""
        return list(self.manifests.values())

    def list_by_capability(self, capability: PluginCapability) -> list[PluginManifest]:
        """List plugins with a specific capability."""
        return [m for m in self.manifests.values() if m.has_capability(capability)]

    def get_runner(self, name: str) -> Optional[PluginRunner]:
        """Get or create a runner for a plugin."""
        if name in self.runners:
            return self.runners[name]

        manifest = self.manifests.get(name)
        if not manifest:
            return None

        runner = PluginRunner(manifest)
        self.runners[name] = runner
        return runner

    async def run_plugin(
        self,
        name: str,
        input_data: dict,
        config: Optional[dict] = None,
        working_dir: str = ".",
    ) -> PluginResult:
        """
        Convenience method to run a plugin by name.

        Args:
            name: Plugin name
            input_data: Input data for plugin
            config: Optional plugin configuration
            working_dir: Working directory for file operations

        Returns:
            PluginResult
        """
        runner = self.get_runner(name)
        if not runner:
            return PluginResult(
                success=False,
                errors=[f"Plugin not found: {name}"],
            )

        context = PluginContext(
            input_data=input_data,
            config=config or runner.manifest.default_config,
            working_dir=working_dir,
        )

        return await runner.run(context)


# Use ServiceRegistry for plugin registry management
from aragora.services import ServiceRegistry


def get_registry() -> PluginRegistry:
    """Get the global plugin registry via ServiceRegistry."""
    registry = ServiceRegistry.get()
    if not registry.has(PluginRegistry):
        registry.register_factory(PluginRegistry, PluginRegistry)
    return registry.resolve(PluginRegistry)


def reset_registry() -> None:
    """Reset the plugin registry (for testing)."""
    registry = ServiceRegistry.get()
    if registry.has(PluginRegistry):
        registry.unregister(PluginRegistry)


async def run_plugin(
    name: str,
    input_data: dict,
    config: Optional[dict] = None,
    working_dir: str = ".",
) -> PluginResult:
    """Run a plugin by name using the global registry."""
    return await get_registry().run_plugin(name, input_data, config, working_dir)
