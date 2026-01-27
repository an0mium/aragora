"""
Declarative Event Hooks System.

Provides YAML-based configuration for debate event hooks with:
- Declarative condition evaluation
- Auto-discovery of hook definitions
- Priority-based execution ordering
- Integration with the existing HookManager

Example YAML hook definition:
    hooks:
      - name: save_checkpoint_on_consensus
        trigger: post_debate
        priority: high
        conditions:
          - field: consensus_reached
            operator: eq
            value: true
        action:
          handler: aragora.hooks.builtin.save_checkpoint
          args:
            path: /data/checkpoints

Usage:
    from aragora.hooks import HookConfigLoader, get_hook_loader

    # Load hooks from YAML files
    loader = get_hook_loader()
    configs = loader.discover_and_load("hooks/")

    # Apply to HookManager
    loader.apply_to_manager(hook_manager)
"""

from aragora.hooks.config import (
    HookConfig,
    ActionConfig,
    ConditionConfig,
)
from aragora.hooks.conditions import (
    ConditionEvaluator,
    Operator,
)
from aragora.hooks.loader import (
    HookConfigLoader,
    get_hook_loader,
    setup_arena_hooks,
    setup_arena_hooks_from_config,
)

__all__ = [
    # Config types
    "HookConfig",
    "ActionConfig",
    "ConditionConfig",
    # Conditions
    "ConditionEvaluator",
    "Operator",
    # Loader
    "HookConfigLoader",
    "get_hook_loader",
    # Arena integration
    "setup_arena_hooks",
    "setup_arena_hooks_from_config",
]
