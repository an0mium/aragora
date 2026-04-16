from __future__ import annotations

from aragora.server.handler_registry import RouteIndex
from aragora.server.handlers.self_improve import SelfImproveHandler
from aragora.server.handlers.self_improve_details import SelfImproveDetailsHandler


class _SelfImproveRegistry:
    def __init__(self) -> None:
        self._self_improve_handler = SelfImproveHandler({})
        self._self_improve_details_handler = SelfImproveDetailsHandler({})


def test_queue_item_routes_resolve_to_details_handler_before_broad_fallback() -> None:
    registry = _SelfImproveRegistry()
    route_index = RouteIndex()
    route_index.build(
        registry,
        [
            ("_self_improve_handler", SelfImproveHandler),
            ("_self_improve_details_handler", SelfImproveDetailsHandler),
        ],
    )

    for path in (
        "/api/self-improve/improvement-queue/user-123",
        "/api/self-improve/improvement-queue/user-123/priority",
        "/api/v1/self-improve/improvement-queue/user-123",
        "/api/v1/self-improve/improvement-queue/user-123/priority",
    ):
        route_match = route_index.get_handler(path)

        assert route_match is not None
        attr_name, handler = route_match
        assert attr_name == "_self_improve_details_handler"
        assert isinstance(handler, SelfImproveDetailsHandler)
