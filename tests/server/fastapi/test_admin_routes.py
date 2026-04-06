from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.fastapi.routes import admin as admin_routes


def test_list_users_prefers_async_store_methods(route_request_factory, route_auth_factory) -> None:
    user = MagicMock()
    user.to_dict.return_value = {
        "id": "user-1",
        "email": "user@example.com",
        "name": "User One",
        "role": "member",
        "org_id": "org-1",
        "is_active": True,
    }

    store = MagicMock()
    store.list_all_users.side_effect = AssertionError("sync list_all_users should not be used")
    store.list_all_users_async = AsyncMock(return_value=([user], 1))

    response = asyncio.run(
        admin_routes.list_users(
            request=route_request_factory(context={}),
            limit=50,
            offset=0,
            org_id=None,
            role=None,
            active_only=False,
            auth=route_auth_factory(user_id="admin-1"),
            user_store=store,
        )
    )

    assert response.total == 1
    assert response.users[0]["email"] == "user@example.com"
    store.list_all_users_async.assert_awaited_once_with(
        limit=50,
        offset=0,
        org_id_filter=None,
        role_filter=None,
        active_only=False,
    )


def test_create_user_hashes_password_and_prefers_async_store_methods(
    route_request_factory, route_auth_factory
) -> None:
    store = MagicMock()
    store.get_user_by_email.side_effect = AssertionError(
        "sync get_user_by_email should not be used"
    )
    store.get_user_by_email_async = AsyncMock(return_value=None)
    store.create_user.side_effect = AssertionError("sync create_user should not be used")
    store.create_user_async = AsyncMock(return_value=SimpleNamespace(id="user-1"))

    with (
        patch("aragora.billing.models.hash_password", return_value=("hashed-password", "salt-1")),
        patch("aragora.audit.unified.audit_admin"),
    ):
        response = asyncio.run(
            admin_routes.create_user(
                body=admin_routes.CreateUserRequest(
                    email="user@example.com",
                    name="User One",
                    role="member",
                    password="super-secret-1",
                ),
                request=route_request_factory(context={}),
                auth=route_auth_factory(user_id="admin-1"),
                user_store=store,
            )
        )

    assert response.success is True
    assert response.user_id == "user-1"
    store.get_user_by_email_async.assert_awaited_once_with("user@example.com")
    store.create_user_async.assert_awaited_once_with(
        email="user@example.com",
        name="User One",
        role="member",
        password_hash="hashed-password",
        password_salt="salt-1",
    )


def test_deactivate_user_prefers_async_store_methods(
    route_request_factory, route_auth_factory
) -> None:
    target_user = SimpleNamespace(email="user@example.com")
    store = MagicMock()
    store.get_user_by_id.side_effect = AssertionError("sync get_user_by_id should not be used")
    store.get_user_by_id_async = AsyncMock(return_value=target_user)
    store.update_user.side_effect = AssertionError("sync update_user should not be used")
    store.update_user_async = AsyncMock(return_value=True)

    with patch("aragora.audit.unified.audit_admin"):
        response = asyncio.run(
            admin_routes.deactivate_user(
                user_id="user-1",
                request=route_request_factory(context={}),
                auth=route_auth_factory(user_id="admin-1"),
                user_store=store,
            )
        )

    assert response.success is True
    store.get_user_by_id_async.assert_awaited_once_with("user-1")
    store.update_user_async.assert_awaited_once_with("user-1", is_active=False)


def test_get_revenue_stats_prefers_async_store_methods(
    route_request_factory, route_auth_factory
) -> None:
    store = MagicMock()
    store.get_admin_stats.side_effect = AssertionError("sync get_admin_stats should not be used")
    store.get_admin_stats_async = AsyncMock(
        return_value={
            "tier_distribution": {"free": 2, "professional": 1},
            "total_organizations": 3,
        }
    )

    response = asyncio.run(
        admin_routes.get_revenue_stats(
            request=route_request_factory(context={}),
            auth=route_auth_factory(user_id="admin-1"),
            user_store=store,
        )
    )

    assert response.revenue["total_organizations"] == 3
    store.get_admin_stats_async.assert_awaited_once_with()


def test_get_user_store_falls_back_to_storage_singleton(route_request_factory) -> None:
    store = object()
    request = route_request_factory(context={})

    with patch("aragora.storage.user_store.get_user_store", return_value=store):
        result = admin_routes._get_user_store(request)

    assert result is store
