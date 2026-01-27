"""Tests for RBAC permission compatibility between colon and dot formats."""

from aragora.rbac.models import APIKeyScope, AuthorizationContext


def test_permission_equivalence_in_context() -> None:
    ctx = AuthorizationContext(user_id="user-1", permissions={"analytics.read"})
    assert ctx.has_permission("analytics:read") is True

    ctx = AuthorizationContext(user_id="user-1", permissions={"analytics:read"})
    assert ctx.has_permission("analytics.read") is True

    ctx = AuthorizationContext(user_id="user-1", permissions={"knowledge.analytics.read"})
    assert ctx.has_permission("knowledge:analytics:read") is True

    ctx = AuthorizationContext(user_id="user-1", permissions={"knowledge:analytics:read"})
    assert ctx.has_permission("knowledge.analytics.read") is True


def test_permission_wildcards_in_context() -> None:
    ctx = AuthorizationContext(user_id="user-1", permissions={"analytics.*"})
    assert ctx.has_permission("analytics:read") is True

    ctx = AuthorizationContext(user_id="user-1", permissions={"analytics:*"})
    assert ctx.has_permission("analytics.read") is True


def test_permission_equivalence_in_api_key_scope() -> None:
    scope = APIKeyScope(permissions={"analytics.read"})
    assert scope.allows_permission("analytics:read") is True

    scope = APIKeyScope(permissions={"analytics:read"})
    assert scope.allows_permission("analytics.read") is True

    scope = APIKeyScope(permissions={"analytics.*"})
    assert scope.allows_permission("analytics:read") is True

    scope = APIKeyScope(permissions={"analytics:*"})
    assert scope.allows_permission("analytics.read") is True
