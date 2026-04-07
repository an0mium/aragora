"""FastAPI tests for v2 notification routes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from aragora.server.fastapi.routes import notifications as notifications_routes


def _email_integration(*, recipients: list[SimpleNamespace] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            smtp_host="smtp.example.com",
            notify_on_consensus=True,
            notify_on_debate_end=True,
            notify_on_error=True,
            enable_digest=True,
            digest_frequency="daily",
        ),
        recipients=recipients or [SimpleNamespace(email="ops@example.com", name="Ops")],
        _send_email=AsyncMock(return_value=True),
        add_recipient=lambda _recipient: None,
        remove_recipient=lambda _email: True,
    )


def _telegram_integration() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            chat_id="1234567890",
            notify_on_consensus=True,
            notify_on_debate_end=True,
            notify_on_error=True,
        ),
        _send_message=AsyncMock(return_value=True),
    )


def test_notification_status_requires_auth(client) -> None:
    response = client.get("/api/v2/notifications/status")

    assert response.status_code == 401


def test_notification_status_scopes_org_context(client, override_auth, monkeypatch) -> None:
    email_calls: list[str | None] = []
    telegram_calls: list[str | None] = []

    async def fake_get_email_integration(org_id: str | None = None):
        email_calls.append(org_id)
        return _email_integration()

    async def fake_get_telegram_integration(org_id: str | None = None):
        telegram_calls.append(org_id)
        return _telegram_integration()

    monkeypatch.setattr(
        notifications_routes,
        "_get_email_integration",
        fake_get_email_integration,
    )
    monkeypatch.setattr(
        notifications_routes,
        "_get_telegram_integration",
        fake_get_telegram_integration,
    )

    override_auth(client, permissions={"notifications:read"}, org_id="org-tenant")
    response = client.get("/api/v2/notifications/status")

    assert response.status_code == 200
    assert email_calls == ["org-tenant"]
    assert telegram_calls == ["org-tenant"]
    assert response.json()["data"]["email"]["configured"] is True


def test_get_email_recipients_reads_org_store(client, override_auth, monkeypatch) -> None:
    calls: list[str] = []

    async def fake_list_email_recipients_for_org(org_id: str):
        calls.append(org_id)
        return [SimpleNamespace(email="alice@example.com", name="Alice")]

    async def unexpected_get_email_integration(org_id: str | None = None):
        raise AssertionError(
            f"fallback email integration should not be used for org-scoped reads: {org_id}"
        )

    monkeypatch.setattr(
        notifications_routes,
        "_list_email_recipients_for_org",
        fake_list_email_recipients_for_org,
    )
    monkeypatch.setattr(
        notifications_routes,
        "_get_email_integration",
        unexpected_get_email_integration,
    )

    override_auth(client, permissions={"notifications:read"}, org_id="org-tenant")
    response = client.get("/api/v2/notifications/email/recipients")

    assert response.status_code == 200
    assert calls == ["org-tenant"]
    assert response.json() == {
        "data": {
            "recipients": [{"email": "alice@example.com", "name": "Alice"}],
            "count": 1,
            "org_id": "org-tenant",
        }
    }


def test_configure_email_persists_to_authenticated_org(client, override_auth, monkeypatch) -> None:
    save_mock = AsyncMock()
    monkeypatch.setattr(notifications_routes, "_save_email_config_for_org", save_mock)

    override_auth(client, permissions={"notifications:write"}, org_id="org-tenant")
    response = client.post(
        "/api/v2/notifications/email/config",
        json={"smtp_host": "smtp.partner.test", "from_email": "alerts@partner.test"},
    )

    assert response.status_code == 200
    assert save_mock.await_count == 1
    saved_body, saved_org_id = save_mock.await_args.args
    assert saved_org_id == "org-tenant"
    assert saved_body.smtp_host == "smtp.partner.test"
    assert response.json()["data"]["org_id"] == "org-tenant"


def test_configure_email_accepts_legacy_write_permission(
    client, override_auth, monkeypatch
) -> None:
    save_mock = AsyncMock()
    monkeypatch.setattr(notifications_routes, "_save_email_config_for_org", save_mock)

    override_auth(client, permissions={"write"}, org_id="org-tenant")
    response = client.post(
        "/api/v2/notifications/email/config",
        json={"smtp_host": "smtp.partner.test"},
    )

    assert response.status_code == 200
    assert save_mock.await_count == 1


def test_add_email_recipient_persists_to_authenticated_org(
    client, override_auth, monkeypatch
) -> None:
    add_mock = AsyncMock(return_value=3)
    monkeypatch.setattr(notifications_routes, "_add_email_recipient_for_org", add_mock)

    override_auth(client, permissions={"notifications:write"}, org_id="org-tenant")
    response = client.post(
        "/api/v2/notifications/email/recipient",
        json={"email": "new@example.com", "name": "New User"},
    )

    assert response.status_code == 200
    assert add_mock.await_count == 1
    saved_body, saved_org_id = add_mock.await_args.args
    assert saved_org_id == "org-tenant"
    assert saved_body.email == "new@example.com"
    assert response.json()["data"]["recipients_count"] == 3


def test_send_test_notification_scopes_org_context(client, override_auth, monkeypatch) -> None:
    email_calls: list[str | None] = []
    telegram_calls: list[str | None] = []
    email = _email_integration()
    telegram = _telegram_integration()

    async def fake_get_email_integration(org_id: str | None = None):
        email_calls.append(org_id)
        return email

    async def fake_get_telegram_integration(org_id: str | None = None):
        telegram_calls.append(org_id)
        return telegram

    monkeypatch.setattr(
        notifications_routes,
        "_get_email_integration",
        fake_get_email_integration,
    )
    monkeypatch.setattr(
        notifications_routes,
        "_get_telegram_integration",
        fake_get_telegram_integration,
    )

    override_auth(client, permissions={"notifications:write"}, org_id="org-tenant")
    response = client.post("/api/v2/notifications/test", json={"type": "all"})

    assert response.status_code == 200
    assert email_calls == ["org-tenant"]
    assert telegram_calls == ["org-tenant"]
    assert email._send_email.await_count == 1
    assert telegram._send_message.await_count == 1
    assert response.json()["data"]["success"] is True
