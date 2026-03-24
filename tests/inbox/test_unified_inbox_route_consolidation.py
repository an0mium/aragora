from __future__ import annotations

from unittest.mock import MagicMock, patch

from aragora.server.handlers.features.unified_inbox.handler import UnifiedInboxHandler
from aragora.server.handlers.shared_inbox.handler import SharedInboxHandler
from aragora.server.router import RequestRouter


def _build_unified_handler() -> UnifiedInboxHandler:
    with patch(
        "aragora.server.handlers.features.unified_inbox.handler.get_canonical_gateway_stores"
    ) as mock_gateway_stores:
        gateway_stores = MagicMock()
        gateway_stores.inbox_store.return_value = MagicMock()
        mock_gateway_stores.return_value = gateway_stores
        return UnifiedInboxHandler({})


def test_unified_inbox_only_advertises_real_dispatch_routes():
    handler = _build_unified_handler()

    advertised_routes = set(handler.ROUTES)

    assert "/api/v1/inbox/oauth/gmail" in advertised_routes
    assert "/api/v1/inbox/connect" in advertised_routes
    assert "/api/v1/inbox/messages" in advertised_routes
    assert "/api/v1/inbox/trends" in advertised_routes

    dead_routes = {
        "/api/v1/inbox/actions",
        "/api/v1/inbox/bulk-actions",
        "/api/v1/inbox/command",
        "/api/v1/inbox/daily-digest",
        "/api/v1/inbox/mentions",
        "/api/v1/inbox/reprioritize",
        "/api/v1/inbox/sender-profile",
        "/inbox/accounts",
        "/inbox/connect",
        "/inbox/messages",
        "/inbox/messages/send",
        "/inbox/oauth/gmail",
        "/inbox/oauth/outlook",
        "/inbox/stats",
        "/inbox/trends",
        "/inbox/triage",
    }

    assert advertised_routes.isdisjoint(dead_routes)


def test_unified_inbox_can_handle_only_supported_paths():
    handler = _build_unified_handler()

    assert handler.can_handle("/api/v1/inbox/oauth/gmail") is True
    assert handler.can_handle("/api/v1/inbox/accounts/acct-123") is True
    assert handler.can_handle("/api/v1/inbox/messages/msg-123") is True
    assert handler.can_handle("/api/v1/inbox/messages/msg-123/debate") is True

    assert handler.can_handle("/api/v1/inbox/shared") is False
    assert handler.can_handle("/api/v1/inbox/shared/inbox-123/messages") is False
    assert handler.can_handle("/api/v1/inbox/routing/rules") is False
    assert handler.can_handle("/api/v1/inbox/messages/send") is False
    assert handler.can_handle("/api/v1/inbox/wedge/receipts") is False
    assert handler.can_handle("/inbox/connect") is False


def test_shared_inbox_routes_dispatch_to_shared_handler_not_unified_handler():
    unified_handler = _build_unified_handler()
    shared_handler = SharedInboxHandler({})
    router = RequestRouter()

    router.register(unified_handler)
    router.register(shared_handler)

    handler = router.get_handler_for_path("/api/v1/inbox/shared/inbox-123/messages")

    assert handler is shared_handler
