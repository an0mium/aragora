from __future__ import annotations

from aragora.server.handlers import ALL_HANDLERS
from aragora.server.handlers.api_docs import ApiDocsHandler
from aragora.server.openapi import generate_openapi_schema


def test_api_docs_handler_participates_in_all_handlers() -> None:
    assert ApiDocsHandler in set(ALL_HANDLERS)


def test_api_docs_routes_appear_in_generated_openapi() -> None:
    paths = generate_openapi_schema()["paths"]

    for route in ApiDocsHandler.ROUTES:
        assert route in paths
        assert set(paths[route]) == {"get"}
