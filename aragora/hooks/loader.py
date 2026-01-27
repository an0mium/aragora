"""
Hook Configuration Loader.

Loads hook definitions from YAML files with auto-discovery support.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

import yaml

from aragora.hooks.config import HookConfig
from aragora.hooks.conditions import ConditionEvaluator

if TYPE_CHECKING:
    from aragora.debate.hooks import HookManager

__all__ = [
    "HookConfigLoader",
    "get_hook_loader",
    "setup_arena_hooks",
    "setup_arena_hooks_from_config",
]

logger = logging.getLogger(__name__)


class HookConfigLoader:
    """
    Loads and manages hook configurations from YAML files.

    Features:
    - Auto-discovery of YAML files in directories
    - Validation of hook configurations
    - Integration with HookManager
    - Handler resolution and caching
    """

    # Default hooks directory
    DEFAULT_HOOKS_DIR = "hooks"

    def __init__(self):
        """Initialize the loader."""
        self._configs: dict[str, HookConfig] = {}
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._condition_evaluator = ConditionEvaluator()
        self._registered_hooks: list[tuple[str, Callable[[], None]]] = []

    @property
    def configs(self) -> dict[str, HookConfig]:
        """Get all loaded hook configurations."""
        return self._configs.copy()

    def load_file(self, file_path: str | Path) -> list[HookConfig]:
        """
        Load hooks from a single YAML file.

        Args:
            file_path: Path to the YAML file

        Returns:
            List of loaded HookConfig objects
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"Hook file not found: {file_path}")
            return []

        try:
            with open(file_path) as f:
                data = yaml.safe_load(f)

            if not data:
                return []

            # Handle both single hook and hook list formats
            hooks_data = data.get("hooks", [data] if "name" in data else [])

            loaded = []
            for hook_data in hooks_data:
                try:
                    config = HookConfig.from_dict(hook_data, source_file=str(file_path))
                    self._configs[config.name] = config
                    loaded.append(config)
                    logger.debug(f"Loaded hook '{config.name}' from {file_path}")
                except Exception as e:
                    logger.error(f"Failed to parse hook in {file_path}: {e}")

            return loaded

        except yaml.YAMLError as e:
            logger.error(f"YAML parse error in {file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to load hooks from {file_path}: {e}")
            return []

    def discover_and_load(
        self,
        directory: str | Path,
        recursive: bool = True,
        pattern: str = "*.yaml",
    ) -> list[HookConfig]:
        """
        Discover and load all hook files in a directory.

        Args:
            directory: Directory to search
            recursive: Whether to search subdirectories
            pattern: Glob pattern for hook files

        Returns:
            List of all loaded HookConfig objects
        """
        directory = Path(directory)
        if not directory.exists():
            logger.debug(f"Hooks directory not found: {directory}")
            return []

        loaded = []

        # Find all matching files
        if recursive:
            files = directory.rglob(pattern)
        else:
            files = directory.glob(pattern)

        for file_path in sorted(files):
            configs = self.load_file(file_path)
            loaded.extend(configs)

        # Also check for .yml extension
        yml_pattern = pattern.replace(".yaml", ".yml")
        if yml_pattern != pattern:
            if recursive:
                yml_files = directory.rglob(yml_pattern)
            else:
                yml_files = directory.glob(yml_pattern)

            for file_path in sorted(yml_files):
                configs = self.load_file(file_path)
                loaded.extend(configs)

        logger.info(f"Discovered {len(loaded)} hooks from {directory}")
        return loaded

    def load_from_string(self, yaml_content: str, source: str = "inline") -> list[HookConfig]:
        """
        Load hooks from a YAML string.

        Args:
            yaml_content: YAML content as string
            source: Source identifier for error messages

        Returns:
            List of loaded HookConfig objects
        """
        try:
            data = yaml.safe_load(yaml_content)
            if not data:
                return []

            hooks_data = data.get("hooks", [data] if "name" in data else [])

            loaded = []
            for hook_data in hooks_data:
                try:
                    config = HookConfig.from_dict(hook_data, source_file=source)
                    self._configs[config.name] = config
                    loaded.append(config)
                except Exception as e:
                    logger.error(f"Failed to parse hook from {source}: {e}")

            return loaded

        except yaml.YAMLError as e:
            logger.error(f"YAML parse error from {source}: {e}")
            return []

    def get_config(self, name: str) -> Optional[HookConfig]:
        """Get a hook configuration by name."""
        return self._configs.get(name)

    def get_configs_by_trigger(self, trigger: str) -> list[HookConfig]:
        """Get all hooks that trigger on a specific event."""
        return [c for c in self._configs.values() if c.trigger == trigger and c.enabled]

    def get_configs_by_tag(self, tag: str) -> list[HookConfig]:
        """Get all hooks with a specific tag."""
        return [c for c in self._configs.values() if tag in c.tags and c.enabled]

    def resolve_handler(self, handler_path: str) -> Optional[Callable[..., Any]]:
        """
        Resolve a handler path to a callable.

        Args:
            handler_path: Fully qualified path (e.g., 'aragora.hooks.builtin.notify')

        Returns:
            The resolved callable, or None if not found
        """
        if handler_path in self._handlers:
            return self._handlers[handler_path]

        try:
            # Split module and attribute
            parts = handler_path.rsplit(".", 1)
            if len(parts) != 2:
                logger.error(f"Invalid handler path: {handler_path}")
                return None

            module_path, attr_name = parts

            # Import module
            module = importlib.import_module(module_path)

            # Get attribute
            handler = getattr(module, attr_name, None)
            if handler is None:
                logger.error(f"Handler not found: {handler_path}")
                return None

            # Cache and return
            self._handlers[handler_path] = handler
            return handler

        except ImportError as e:
            logger.error(f"Failed to import handler module: {handler_path} - {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to resolve handler: {handler_path} - {e}")
            return None

    def apply_to_manager(
        self,
        manager: "HookManager",
        configs: Optional[list[HookConfig]] = None,
    ) -> int:
        """
        Apply hook configurations to a HookManager.

        Creates wrapper functions that:
        1. Evaluate conditions before executing
        2. Call the resolved handler with configured args
        3. Handle one-shot behavior

        Args:
            manager: The HookManager to register hooks with
            configs: Specific configs to apply, or None for all loaded configs

        Returns:
            Number of hooks successfully registered
        """
        from aragora.debate.hooks import HookPriority

        configs = configs or list(self._configs.values())
        registered = 0

        # Map priority strings to enum values
        priority_map = {
            "critical": HookPriority.CRITICAL,
            "high": HookPriority.HIGH,
            "normal": HookPriority.NORMAL,
            "low": HookPriority.LOW,
            "cleanup": HookPriority.CLEANUP,
        }

        for config in configs:
            if not config.enabled:
                continue

            # Resolve handler
            handler = self.resolve_handler(config.action.handler)
            if handler is None:
                logger.warning(f"Skipping hook '{config.name}': handler not found")
                continue

            # Get priority
            priority = priority_map.get(config.priority.lower(), HookPriority.NORMAL)

            # Create wrapper that evaluates conditions
            wrapper = self._create_hook_wrapper(config, handler)

            # Register with manager
            try:
                unregister = manager.register(
                    config.trigger,
                    wrapper,
                    priority=priority,
                    name=config.name,
                    once=config.one_shot,
                )
                self._registered_hooks.append((config.name, unregister))
                registered += 1
                logger.debug(f"Registered hook '{config.name}' for trigger '{config.trigger}'")
            except Exception as e:
                logger.error(f"Failed to register hook '{config.name}': {e}")

        logger.info(f"Applied {registered}/{len(configs)} hooks to manager")
        return registered

    def _create_hook_wrapper(
        self,
        config: HookConfig,
        handler: Callable[..., Any],
    ) -> Callable[..., Any]:
        """
        Create a wrapper function that evaluates conditions before calling handler.

        Args:
            config: The hook configuration
            handler: The resolved handler function

        Returns:
            Wrapper function
        """
        import asyncio

        async def async_wrapper(**kwargs: Any) -> Any:
            # Evaluate conditions
            if config.conditions:
                if not self._condition_evaluator.evaluate_all(config.conditions, kwargs):
                    logger.debug(f"Hook '{config.name}' conditions not met, skipping")
                    return None

            # Merge configured args with trigger kwargs
            call_args = {**config.action.args, **kwargs}

            # Call handler
            try:
                if asyncio.iscoroutinefunction(handler):
                    return await handler(**call_args)
                else:
                    return handler(**call_args)
            except Exception as e:
                logger.error(f"Hook '{config.name}' handler error: {e}")
                raise

        def sync_wrapper(**kwargs: Any) -> Any:
            # Evaluate conditions
            if config.conditions:
                if not self._condition_evaluator.evaluate_all(config.conditions, kwargs):
                    logger.debug(f"Hook '{config.name}' conditions not met, skipping")
                    return None

            # Merge configured args with trigger kwargs
            call_args = {**config.action.args, **kwargs}

            # Call handler
            try:
                return handler(**call_args)
            except Exception as e:
                logger.error(f"Hook '{config.name}' handler error: {e}")
                raise

        # Return appropriate wrapper based on handler type
        if asyncio.iscoroutinefunction(handler) or config.action.async_execution:
            return async_wrapper
        return sync_wrapper

    def unregister_all(self) -> int:
        """
        Unregister all hooks that were registered through this loader.

        Returns:
            Number of hooks unregistered
        """
        count = 0
        for name, unregister_fn in self._registered_hooks:
            try:
                unregister_fn()
                count += 1
                logger.debug(f"Unregistered hook '{name}'")
            except Exception as e:
                logger.warning(f"Failed to unregister hook '{name}': {e}")

        self._registered_hooks.clear()
        return count

    def clear(self) -> None:
        """Clear all loaded configurations and handlers."""
        self.unregister_all()
        self._configs.clear()
        self._handlers.clear()

    def validate_config(self, config: HookConfig) -> list[str]:
        """
        Validate a hook configuration.

        Args:
            config: The configuration to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check required fields
        if not config.name:
            errors.append("Hook name is required")

        if not config.trigger:
            errors.append("Hook trigger is required")

        if not config.action or not config.action.handler:
            errors.append("Hook action handler is required")

        # Validate handler path format
        if config.action and config.action.handler:
            parts = config.action.handler.rsplit(".", 1)
            if len(parts) != 2:
                errors.append(f"Invalid handler path format: {config.action.handler}")

        # Validate priority
        valid_priorities = {"critical", "high", "normal", "low", "cleanup"}
        if config.priority.lower() not in valid_priorities:
            errors.append(f"Invalid priority: {config.priority}")

        # Validate condition operators
        valid_operators = {
            "eq",
            "ne",
            "gt",
            "gte",
            "lt",
            "lte",
            "contains",
            "not_contains",
            "starts_with",
            "ends_with",
            "matches",
            "is_null",
            "is_not_null",
            "is_empty",
            "is_not_empty",
            "in",
            "not_in",
            "has_key",
            "is_true",
            "is_false",
        }
        for cond in config.conditions:
            if cond.operator.lower() not in valid_operators:
                errors.append(f"Invalid condition operator: {cond.operator}")

        return errors


# Global loader singleton
_hook_loader: Optional[HookConfigLoader] = None


def get_hook_loader() -> HookConfigLoader:
    """
    Get the global hook loader singleton.

    Returns:
        HookConfigLoader instance
    """
    global _hook_loader
    if _hook_loader is None:
        _hook_loader = HookConfigLoader()
    return _hook_loader


def setup_arena_hooks(
    hook_manager: "HookManager",
    hooks_dir: str = "hooks",
    recursive: bool = True,
    validate: bool = True,
) -> int:
    """
    Discover and apply YAML hooks to an Arena's HookManager.

    Convenience function for Arena integration that:
    1. Discovers YAML hook files in the specified directory
    2. Validates all hook configurations
    3. Applies valid hooks to the HookManager

    Args:
        hook_manager: Arena's HookManager to apply hooks to
        hooks_dir: Directory to search for YAML hook definitions
        recursive: Whether to search subdirectories
        validate: Whether to validate configurations before applying

    Returns:
        Number of hooks successfully registered

    Example:
        # In Arena initialization or startup
        from aragora.hooks.loader import setup_arena_hooks
        from aragora.debate.hooks import HookManager

        hook_manager = HookManager()
        count = setup_arena_hooks(hook_manager, "hooks/debate")
        logger.info(f"Loaded {count} declarative hooks")
    """
    loader = get_hook_loader()

    # Discover and load hooks
    configs = loader.discover_and_load(
        directory=hooks_dir,
        recursive=recursive,
    )

    if not configs:
        logger.debug(f"No hook definitions found in {hooks_dir}")
        return 0

    # Validate if requested
    if validate:
        valid_configs = []
        for config in configs:
            errors = loader.validate_config(config)
            if errors:
                logger.warning(f"Hook '{config.name}' validation failed: {'; '.join(errors)}")
            else:
                valid_configs.append(config)
        configs = valid_configs

    # Apply to manager
    registered = loader.apply_to_manager(hook_manager, configs)

    logger.info(
        f"setup_arena_hooks dir={hooks_dir} discovered={len(configs)} registered={registered}"
    )

    return registered


def setup_arena_hooks_from_config(
    hook_manager: "HookManager",
    yaml_hooks_dir: str = "hooks",
    enable_yaml_hooks: bool = True,
    yaml_hooks_recursive: bool = True,
) -> int:
    """
    Setup hooks from ArenaConfig settings.

    Integrates with ArenaConfig to provide seamless YAML hook loading
    based on configuration parameters.

    Args:
        hook_manager: Arena's HookManager to apply hooks to
        yaml_hooks_dir: ArenaConfig.yaml_hooks_dir value
        enable_yaml_hooks: ArenaConfig.enable_yaml_hooks value
        yaml_hooks_recursive: ArenaConfig.yaml_hooks_recursive value

    Returns:
        Number of hooks successfully registered (0 if disabled)

    Example:
        # In Arena __init__ or from_config
        if config.enable_yaml_hooks and self.hook_manager:
            setup_arena_hooks_from_config(
                self.hook_manager,
                yaml_hooks_dir=config.yaml_hooks_dir,
                enable_yaml_hooks=config.enable_yaml_hooks,
                yaml_hooks_recursive=config.yaml_hooks_recursive,
            )
    """
    if not enable_yaml_hooks:
        logger.debug("YAML hooks disabled via configuration")
        return 0

    if hook_manager is None:
        logger.debug("No HookManager provided, skipping YAML hooks")
        return 0

    return setup_arena_hooks(
        hook_manager=hook_manager,
        hooks_dir=yaml_hooks_dir,
        recursive=yaml_hooks_recursive,
        validate=True,
    )
