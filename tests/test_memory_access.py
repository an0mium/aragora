import os

from aragora.memory.access import can_view_entry, filter_entries, resolve_tenant_id
from aragora.rbac.models import AuthorizationContext


def _ctx(
    *,
    user_id: str = "user-1",
    workspace_id: str | None = "ws-1",
    org_id: str | None = "org-1",
    roles: set[str] | None = None,
    permissions: set[str] | None = None,
) -> AuthorizationContext:
    return AuthorizationContext(
        user_id=user_id,
        workspace_id=workspace_id,
        org_id=org_id,
        roles=roles or set(),
        permissions=permissions or set(),
    )


def test_resolve_tenant_id_respects_env(monkeypatch):
    ctx = _ctx()
    monkeypatch.setenv("ARAGORA_MEMORY_TENANT_ENFORCE", "0")
    assert resolve_tenant_id(ctx) is None
    monkeypatch.setenv("ARAGORA_MEMORY_TENANT_ENFORCE", "1")
    assert resolve_tenant_id(ctx) == "ws-1"


def test_filter_entries_tenant_isolation(monkeypatch):
    monkeypatch.setenv("ARAGORA_MEMORY_TENANT_ENFORCE", "1")
    ctx = _ctx(workspace_id="ws-1")
    entries = [
        {"id": "a", "metadata": {"tenant_id": "ws-1"}},
        {"id": "b", "metadata": {"tenant_id": "ws-2"}},
        {"id": "c", "metadata": {}},
    ]
    filtered = filter_entries(entries, ctx)
    ids = {e["id"] for e in filtered}
    assert ids == {"a", "c"}


def test_filter_entries_owner_scope(monkeypatch):
    monkeypatch.setenv("ARAGORA_MEMORY_TENANT_ENFORCE", "0")
    ctx = _ctx(user_id="user-1")
    entries = [
        {"id": "own", "metadata": {"owner_id": "user-1", "scope": "user"}},
        {"id": "other", "metadata": {"owner_id": "user-2", "scope": "user"}},
        {"id": "team", "metadata": {"scope": "workspace"}},
    ]
    filtered = filter_entries(entries, ctx)
    ids = {e["id"] for e in filtered}
    assert ids == {"own", "team"}


def test_admin_context_bypasses_filters(monkeypatch):
    monkeypatch.setenv("ARAGORA_MEMORY_TENANT_ENFORCE", "1")
    ctx = _ctx(user_id="admin", workspace_id="ws-1", permissions={"memory:manage"})
    entry = {"id": "x", "metadata": {"tenant_id": "ws-2", "owner_id": "user-2"}}
    assert can_view_entry(entry, ctx) is True


def test_can_view_entry_without_context():
    entry = {"id": "x", "metadata": {"tenant_id": "ws-2", "owner_id": "user-2"}}
    assert can_view_entry(entry, None) is True
