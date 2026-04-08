from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aragora.rbac.models import AuthorizationContext
from aragora.server.fastapi import create_app
from aragora.server.fastapi.dependencies.auth import require_authenticated
from aragora.server.fastapi.routes import notifications as notifications_routes
from aragora.storage.notification_config_store import StoredEmailConfig, StoredTelegramConfig


@pytest.fixture
def app():
    app = create_app()
    app.state.context = {}
    return app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


def _override_auth(client: TestClient, permissions: set[str]) -> None:
    auth_ctx = AuthorizationContext(
        user_id="user-1",
        org_id="org-notify",
        workspace_id="ws-notify",
        roles={"member"},
        permissions=permissions,
    )
    client.app.dependency_overrides[require_authenticated] = lambda: auth_ctx


def test_notification_status_requires_auth(client) -> None:
    response = client.get("/api/v2/notifications/status")

    assert response.status_code == 401


def test_notification_status_scopes_integrations_to_auth_org(client) -> None:
    _override_auth(client, {"notifications:read"})
    email = MagicMock()
    email.config = SimpleNamespace(
        smtp_host="smtp.example.com",
        notify_on_consensus=True,
        notify_on_debate_end=True,
        notify_on_error=True,
        enable_digest=True,
        digest_frequency="daily",
    )
    email.recipients = [SimpleNamespace(email="alerts@example.com", name="Alerts")]
    telegram = MagicMock()
    telegram.config = SimpleNamespace(
        chat_id="123456789",
        notify_on_consensus=True,
        notify_on_debate_end=True,
        notify_on_error=True,
    )

    with (
        patch.object(
            notifications_routes,
            "_get_email_integration",
            new=AsyncMock(return_value=email),
        ) as get_email,
        patch.object(
            notifications_routes,
            "_get_telegram_integration",
            new=AsyncMock(return_value=telegram),
        ) as get_telegram,
    ):
        response = client.get("/api/v2/notifications/status")

    client.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["data"]["email"]["configured"] is True
    get_email.assert_awaited_once_with("org-notify")
    get_telegram.assert_awaited_once_with("org-notify")


def test_notification_recipients_scope_to_auth_org(client) -> None:
    _override_auth(client, {"notifications:read"})
    email = MagicMock()
    email.recipients = [SimpleNamespace(email="alerts@example.com", name="Alerts")]

    with patch.object(
        notifications_routes,
        "_get_email_integration",
        new=AsyncMock(return_value=email),
    ) as get_email:
        response = client.get("/api/v2/notifications/email/recipients")

    client.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["data"]["count"] == 1
    get_email.assert_awaited_once_with("org-notify")


def test_configure_email_saves_org_scoped_config(client) -> None:
    _override_auth(client, {"notifications:write"})
    store = MagicMock()
    store.save_email_config = AsyncMock()

    with (
        patch.object(notifications_routes, "get_notification_config_store", return_value=store),
        patch(
            "aragora.server.handlers.social.notifications.invalidate_org_integration_cache"
        ) as invalidate_cache,
        patch(
            "aragora.server.handlers.social.notifications.configure_email_integration"
        ) as configure_system,
    ):
        response = client.post(
            "/api/v2/notifications/email/config",
            json={
                "smtp_host": "smtp.example.com",
                "smtp_port": 2525,
                "from_email": "alerts@example.com",
            },
        )

    client.app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["org_id"] == "org-notify"
    saved = store.save_email_config.await_args.args[0]
    assert isinstance(saved, StoredEmailConfig)
    assert saved.org_id == "org-notify"
    assert saved.smtp_host == "smtp.example.com"
    invalidate_cache.assert_called_once_with("org-notify")
    configure_system.assert_not_called()


def test_configure_email_accepts_legacy_write_permission(client) -> None:
    _override_auth(client, {"write"})
    store = MagicMock()
    store.save_email_config = AsyncMock()

    with (
        patch.object(notifications_routes, "get_notification_config_store", return_value=store),
        patch("aragora.server.handlers.social.notifications.invalidate_org_integration_cache"),
    ):
        response = client.post(
            "/api/v2/notifications/email/config",
            json={"smtp_host": "smtp.example.com"},
        )

    client.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert store.save_email_config.await_args.args[0].org_id == "org-notify"


def test_configure_telegram_saves_org_scoped_config(client) -> None:
    _override_auth(client, {"notifications:write"})
    store = MagicMock()
    store.save_telegram_config = AsyncMock()

    with (
        patch.object(notifications_routes, "get_notification_config_store", return_value=store),
        patch(
            "aragora.server.handlers.social.notifications.invalidate_org_integration_cache"
        ) as invalidate_cache,
        patch(
            "aragora.server.handlers.social.notifications.configure_telegram_integration"
        ) as configure_system,
    ):
        response = client.post(
            "/api/v2/notifications/telegram/config",
            json={"bot_token": "bot-token", "chat_id": "chat-123"},
        )

    client.app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["org_id"] == "org-notify"
    saved = store.save_telegram_config.await_args.args[0]
    assert isinstance(saved, StoredTelegramConfig)
    assert saved.org_id == "org-notify"
    assert saved.chat_id == "chat-123"
    invalidate_cache.assert_called_once_with("org-notify")
    configure_system.assert_not_called()


def test_add_email_recipient_saves_to_org_store(client) -> None:
    _override_auth(client, {"notifications:write"})
    store = MagicMock()
    store.add_recipient = AsyncMock()
    store.get_recipients = AsyncMock(
        return_value=[SimpleNamespace(email="alerts@example.com", name="Alerts")]
    )

    with patch.object(notifications_routes, "get_notification_config_store", return_value=store):
        response = client.post(
            "/api/v2/notifications/email/recipient",
            json={"email": "alerts@example.com", "name": "Alerts"},
        )

    client.app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["org_id"] == "org-notify"
    saved = store.add_recipient.await_args.args[0]
    assert saved.org_id == "org-notify"
    assert saved.email == "alerts@example.com"


def test_remove_email_recipient_uses_org_store(client) -> None:
    _override_auth(client, {"notifications:delete"})
    store = MagicMock()
    store.remove_recipient = AsyncMock(return_value=True)
    store.get_recipients = AsyncMock(return_value=[])

    with patch.object(notifications_routes, "get_notification_config_store", return_value=store):
        response = client.delete(
            "/api/v2/notifications/email/recipient",
            params={"email": "alerts@example.com"},
        )

    client.app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["org_id"] == "org-notify"
    store.remove_recipient.assert_awaited_once_with("org-notify", "alerts@example.com")


def test_send_test_notification_scopes_integrations_to_auth_org(client) -> None:
    _override_auth(client, {"notifications:write"})
    email = MagicMock()
    email.recipients = [SimpleNamespace(email="alerts@example.com", name="Alerts")]
    email._send_email = AsyncMock(return_value=True)
    telegram = MagicMock()
    telegram._send_message = AsyncMock(return_value=True)

    with (
        patch.object(
            notifications_routes,
            "_get_email_integration",
            new=AsyncMock(return_value=email),
        ) as get_email,
        patch.object(
            notifications_routes,
            "_get_telegram_integration",
            new=AsyncMock(return_value=telegram),
        ) as get_telegram,
    ):
        response = client.post("/api/v2/notifications/test", json={"type": "all"})

    client.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["data"]["success"] is True
    get_email.assert_awaited_once_with("org-notify")
    get_telegram.assert_awaited_once_with("org-notify")


def test_send_notification_scopes_integrations_to_auth_org(client) -> None:
    _override_auth(client, {"notifications:write"})
    email = MagicMock()
    email.recipients = [SimpleNamespace(email="alerts@example.com", name="Alerts")]
    email._send_email = AsyncMock(return_value=True)
    telegram = MagicMock()
    telegram._send_message = AsyncMock(return_value=True)

    with (
        patch.object(
            notifications_routes,
            "_get_email_integration",
            new=AsyncMock(return_value=email),
        ) as get_email,
        patch.object(
            notifications_routes,
            "_get_telegram_integration",
            new=AsyncMock(return_value=telegram),
        ) as get_telegram,
    ):
        response = client.post(
            "/api/v2/notifications/send",
            json={"type": "all", "subject": "Notice", "message": "Scoped"},
        )

    client.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["data"]["success"] is True
    get_email.assert_awaited_once_with("org-notify")
    get_telegram.assert_awaited_once_with("org-notify")
