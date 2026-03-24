from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aragora.server.fastapi.routes import auth as auth_routes


class _TokenPair:
    def __init__(self) -> None:
        self._payload = {
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "token_type": "bearer",
            "expires_in": 3600,
        }

    def to_dict(self) -> dict[str, object]:
        return dict(self._payload)


def _build_app(store: object) -> FastAPI:
    app = FastAPI()
    app.include_router(auth_routes.router)
    app.dependency_overrides[auth_routes.get_user_store] = lambda: store
    return app


def test_login_prefers_async_user_store_methods() -> None:
    store = MagicMock()
    user = MagicMock()
    user.id = "user-1"
    user.email = "user@example.com"
    user.org_id = "org-1"
    user.role = "member"
    user.is_active = True
    user.mfa_enabled = False
    user.mfa_secret = None
    user.created_at = None
    user.verify_password.return_value = True
    user.to_dict.return_value = {"id": "user-1", "email": "user@example.com"}

    org = MagicMock()
    org.id = "org-1"
    org.to_dict.return_value = {"id": "org-1", "name": "Acme"}

    store.get_user_by_email.side_effect = AssertionError("sync email lookup should not be used")
    store.get_user_by_email_async = AsyncMock(return_value=user)
    store.get_organization_by_id.side_effect = AssertionError("sync org lookup should not be used")
    store.get_organization_by_id_async = AsyncMock(return_value=org)

    lockout_tracker = MagicMock()
    lockout_tracker.is_locked.return_value = False

    with (
        patch("aragora.auth.lockout.get_lockout_tracker", return_value=lockout_tracker),
        patch("aragora.billing.jwt_auth.create_token_pair", return_value=_TokenPair()),
        patch("aragora.billing.jwt_auth.create_mfa_pending_token", return_value="pending"),
        TestClient(_build_app(store)) as client,
    ):
        response = client.post(
            "/api/v2/auth/login",
            json={"email": "User@Example.com", "password": "secret"},
        )

    assert response.status_code == 200
    assert response.json()["user"]["email"] == "user@example.com"
    store.get_user_by_email_async.assert_awaited_once_with("user@example.com")
    store.get_organization_by_id_async.assert_awaited_once_with("org-1")


def test_refresh_prefers_async_user_store_methods() -> None:
    store = MagicMock()
    user = MagicMock()
    user.id = "user-1"
    user.email = "user@example.com"
    user.org_id = "org-1"
    user.role = "member"
    user.is_active = True

    store.get_user_by_id.side_effect = AssertionError("sync user lookup should not be used")
    store.get_user_by_id_async = AsyncMock(return_value=user)

    blacklist = MagicMock()

    with (
        patch(
            "aragora.billing.jwt_auth.validate_refresh_token",
            return_value=SimpleNamespace(user_id="user-1"),
        ),
        patch("aragora.billing.jwt_auth.create_token_pair", return_value=_TokenPair()),
        patch("aragora.billing.jwt_auth.get_token_blacklist", return_value=blacklist),
        patch("aragora.billing.jwt_auth.revoke_token_persistent"),
        TestClient(_build_app(store)) as client,
    ):
        response = client.post("/api/v2/auth/refresh", json={"refresh_token": "refresh-1"})

    assert response.status_code == 200
    assert response.json()["tokens"]["access_token"] == "access-token"
    store.get_user_by_id_async.assert_awaited_once_with("user-1")
    blacklist.revoke_token.assert_called_once_with("refresh-1")
