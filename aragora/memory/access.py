"""Access control helpers for ContinuumMemory entries.

Provides lightweight, metadata-based filtering for memory entries using
AuthorizationContext and existing RBAC permissions. This enables
workspace/org isolation and personal-memory visibility without requiring
schema changes.
"""

from __future__ import annotations

import os
from typing import Any, Iterable

try:  # Optional import for typing only
    from aragora.rbac.models import AuthorizationContext
except Exception:  # pragma: no cover - optional dependency
    AuthorizationContext = Any  # type: ignore


def _env_flag(name: str, default: str = "1") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value not in {"0", "false", "no", "off"}


def resolve_tenant_id(auth_context: AuthorizationContext | None) -> str | None:
    """Resolve tenant/workspace id for memory isolation."""
    if auth_context is None or not _env_flag("ARAGORA_MEMORY_TENANT_ENFORCE", "1"):
        return None
    workspace_id = getattr(auth_context, "workspace_id", None)
    org_id = getattr(auth_context, "org_id", None)
    return workspace_id or org_id


def tenant_enforcement_enabled() -> bool:
    """Return True if tenant isolation enforcement is enabled."""
    return _env_flag("ARAGORA_MEMORY_TENANT_ENFORCE", "1")


def _has_permission(auth_context: AuthorizationContext | None, permission: str) -> bool:
    if auth_context is None:
        return False
    check = getattr(auth_context, "has_permission", None)
    if callable(check):
        try:
            return bool(check(permission))
        except Exception:
            return False
    return False


def _has_role(auth_context: AuthorizationContext | None, *roles: str) -> bool:
    if auth_context is None:
        return False
    check = getattr(auth_context, "has_any_role", None)
    if callable(check):
        try:
            return bool(check(*roles))
        except Exception:
            return False
    # Fallback: check roles attribute
    current_roles = getattr(auth_context, "roles", set()) or set()
    return bool(set(roles) & set(current_roles))


def is_admin_context(auth_context: AuthorizationContext | None) -> bool:
    """Return True if context has admin-level memory access."""
    if auth_context is None:
        return False
    return _has_permission(auth_context, "memory:manage") or _has_role(
        auth_context, "owner", "admin", "superadmin"
    )


def _get_metadata(entry: Any) -> dict[str, Any]:
    if isinstance(entry, dict):
        meta = entry.get("metadata") or {}
    else:
        meta = getattr(entry, "metadata", None) or {}
    return meta if isinstance(meta, dict) else {}


def _get_value(entry: Any, key: str) -> Any:
    if isinstance(entry, dict):
        return entry.get(key)
    return getattr(entry, key, None)


def can_view_entry(entry: Any, auth_context: AuthorizationContext | None) -> bool:
    """Return True if auth_context can view the memory entry."""
    if auth_context is None:
        return True

    meta = _get_metadata(entry)
    admin = is_admin_context(auth_context)

    # Tenant/workspace isolation (if tenant_id present)
    tenant_id = meta.get("tenant_id") or meta.get("workspace_id")
    if tenant_id:
        expected = resolve_tenant_id(auth_context)
        if expected and tenant_id != expected and not admin:
            return False

    # Org isolation (if org_id present)
    org_id = meta.get("org_id")
    ctx_org = getattr(auth_context, "org_id", None)
    if org_id and ctx_org and org_id != ctx_org and not admin:
        return False

    # Personal memory scope
    owner_id = meta.get("owner_id") or meta.get("user_id") or _get_value(entry, "owner_id")
    scope = meta.get("scope") or meta.get("visibility")
    if scope in {"user", "private"} or owner_id:
        if owner_id and owner_id != getattr(auth_context, "user_id", None) and not admin:
            return False

    return True


def filter_entries(
    entries: Iterable[Any],
    auth_context: AuthorizationContext | None,
) -> list[Any]:
    """Filter memory entries based on auth_context."""
    if auth_context is None:
        return list(entries)
    return [entry for entry in entries if can_view_entry(entry, auth_context)]


__all__ = [
    "resolve_tenant_id",
    "tenant_enforcement_enabled",
    "is_admin_context",
    "can_view_entry",
    "filter_entries",
]
