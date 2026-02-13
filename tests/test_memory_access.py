from aragora.memory.access import (
    build_access_envelope,
    can_view_entry,
    filter_entries,
    has_memory_read_access,
    resolve_tenant_id,
)
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


def test_has_memory_read_access():
    with_read = _ctx(permissions={"memory:read"})
    without_read = _ctx(permissions={"debates:read"})
    assert has_memory_read_access(with_read) is True
    assert has_memory_read_access(without_read) is False


def test_persona_scope_denied_without_persona_access():
    entry = {"id": "x", "metadata": {"persona_id": "persona-security"}}
    ctx = _ctx(permissions={"memory:read"})
    assert can_view_entry(entry, ctx) is False


def test_persona_scope_allowed_with_persona_permission():
    entry = {"id": "x", "metadata": {"persona_id": "persona-security"}}
    ctx = _ctx(permissions={"memory:read", "personas.read"})
    assert can_view_entry(entry, ctx) is True


def test_filter_entries_logs_denial_telemetry(monkeypatch, caplog):
    monkeypatch.setenv("ARAGORA_MEMORY_TENANT_ENFORCE", "1")
    ctx = _ctx(workspace_id="ws-1", permissions={"memory:read"})
    entries = [
        {"id": "allow", "metadata": {"tenant_id": "ws-1"}},
        {"id": "deny", "metadata": {"tenant_id": "ws-2"}},
    ]

    with caplog.at_level("WARNING"):
        filtered = filter_entries(entries, ctx, source="tests.memory_access")

    assert [e["id"] for e in filtered] == ["allow"]
    assert any("memory_access_denied" in rec.message for rec in caplog.records)


def test_build_access_envelope_contains_scope():
    ctx = _ctx(
        user_id="u-1",
        workspace_id="ws-1",
        org_id="org-1",
        roles={"member"},
        permissions={"memory:read", "debates:read"},
    )
    envelope = build_access_envelope(ctx, source="test.source")

    assert envelope["source"] == "test.source"
    assert envelope["user_id"] == "u-1"
    assert envelope["workspace_id"] == "ws-1"
    assert envelope["tenant_id"] == "ws-1"
    assert envelope["org_id"] == "org-1"
    assert envelope["roles"] == ["member"]
    assert envelope["permission_count"] == 2
