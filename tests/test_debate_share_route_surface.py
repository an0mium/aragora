from __future__ import annotations

from aragora.server.handlers.debates.public_viewer import PublicDebateViewerHandler
from aragora.server.handlers.debates.share import DebateShareHandler
from aragora.server.handlers.social.sharing import SharingHandler
from aragora.server.router import RequestRouter


def _build_router() -> RequestRouter:
    router = RequestRouter()
    router.register(DebateShareHandler({}))
    router.register(PublicDebateViewerHandler({}))
    router.register(SharingHandler({}))
    return router


def test_live_router_resolves_versioned_debate_share_surface() -> None:
    router = _build_router()

    expected = {
        ("POST", "/api/v1/debates/debate-1/share"): "DebateShareHandler",
        ("POST", "/api/v1/debates/debate-1/share/revoke"): "SharingHandler",
        ("GET", "/api/v1/debates/debate-1/spectate/public"): "DebateShareHandler",
        ("GET", "/api/v1/debates/public/debate-1"): "PublicDebateViewerHandler",
        ("GET", "/api/v1/debates/public/debate-1/og"): "PublicDebateViewerHandler",
    }

    for (method, path), handler_name in expected.items():
        handler, _ = router._find_handler(method, path)
        assert handler is not None, f"{method} {path} did not resolve"
        assert handler.__class__.__name__ == handler_name


def test_live_router_rejects_legacy_debate_share_surface() -> None:
    router = _build_router()

    legacy_paths = [
        ("POST", "/api/debates/debate-1/share"),
        ("POST", "/api/debates/debate-1/share/revoke"),
        ("GET", "/api/debates/debate-1/spectate/public"),
        ("GET", "/api/debates/public/debate-1"),
        ("GET", "/api/debates/public/debate-1/og"),
    ]

    for method, path in legacy_paths:
        handler, _ = router._find_handler(method, path)
        assert handler is None, f"{method} {path} unexpectedly resolved"
