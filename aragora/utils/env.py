"""
Environment helpers.

Centralizes environment-variable parsing to avoid scattered ad-hoc checks.
"""

from __future__ import annotations

import os

_TRUTHY = {"1", "true", "yes", "y", "on"}
_FALSY = {"0", "false", "no", "n", "off"}


def is_truthy_env(name: str, *, default: bool = False) -> bool:
    """Return True/False for common env var boolean encodings."""
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    return default


def is_offline_mode() -> bool:
    """Offline mode disables network-backed features by convention."""
    return is_truthy_env("ARAGORA_OFFLINE", default=False)


__all__ = ["is_truthy_env", "is_offline_mode"]
