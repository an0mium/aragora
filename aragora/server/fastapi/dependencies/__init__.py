"""FastAPI dependency injection modules for auth, RBAC, and shared resources."""

from .auth import get_auth_context, require_authenticated, require_permission

__all__ = [
    "get_auth_context",
    "require_authenticated",
    "require_permission",
]
