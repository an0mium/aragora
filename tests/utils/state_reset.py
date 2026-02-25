"""Shared test-state reset helpers for flake-resistant fixtures."""

from __future__ import annotations

import sys
from typing import Any


def unset_env_vars(monkeypatch: Any, env_vars: list[str]) -> None:
    """Unset environment variables if present."""
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)


def invalidate_legacy_config_module(monkeypatch: Any) -> object | None:
    """Invalidate cached aragora.config legacy module and return prior module."""
    legacy_key = "aragora.config.legacy"
    saved = sys.modules.pop(legacy_key, None)
    try:
        import aragora.config as cfg_pkg

        if hasattr(cfg_pkg, "_legacy_mod"):
            monkeypatch.setattr(cfg_pkg, "_legacy_mod", None)
    except (ImportError, AttributeError):
        pass
    return saved


def restore_legacy_config_module(saved: object | None) -> None:
    """Restore previously cached aragora.config legacy module."""
    if saved is not None:
        sys.modules["aragora.config.legacy"] = saved


def clear_all_auth_rate_limiters() -> None:
    """Clear global auth/runtime rate limiter buckets if module is available."""
    try:
        from aragora.server.handlers.utils.rate_limit import clear_all_limiters

        clear_all_limiters()
    except (ImportError, AttributeError):
        pass


def reset_permission_checker_override() -> None:
    """Drop any instance-level check_permission override from singleton checker."""
    try:
        from aragora.rbac.checker import get_permission_checker

        checker = get_permission_checker()
        checker.__dict__.pop("check_permission", None)
    except (ImportError, AttributeError):
        pass


def restore_rbac_context_extractor(original_get_context_from_args: object) -> None:
    """Restore rbac.decorators._get_context_from_args to its original implementation."""
    if original_get_context_from_args is None:
        return
    try:
        from aragora.rbac import decorators as rbac_decorators

        rbac_decorators._get_context_from_args = original_get_context_from_args
    except (ImportError, AttributeError):
        pass

