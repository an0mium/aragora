from __future__ import annotations

from unittest.mock import MagicMock


def test_memory_search_requires_auth(fastapi_client, fastapi_context):
    fastapi_context["continuum_memory"] = MagicMock()

    response = fastapi_client.get("/api/v2/memory/search?q=alpha")

    assert response.status_code == 401


def test_memory_search_applies_listing_filters(fastapi_client, fastapi_context, override_auth):
    continuum_memory = MagicMock()
    continuum_memory.retrieve.return_value = []
    fastapi_context["continuum_memory"] = continuum_memory

    override_auth(fastapi_client)
    response = fastapi_client.get(
        "/api/v2/memory/search?q=alpha&tier=fast,slow&min_importance=0.6&limit=5"
    )

    assert response.status_code == 200
    continuum_memory.retrieve.assert_called_once()
    kwargs = continuum_memory.retrieve.call_args.kwargs
    assert kwargs["query"] == "alpha"
    assert kwargs["limit"] == 5
    assert kwargs["min_importance"] == 0.6
    assert [tier.name.lower() for tier in kwargs["tiers"]] == ["fast", "slow"]


def test_memory_search_rejects_malformed_pagination(fastapi_client, fastapi_context, override_auth):
    fastapi_context["continuum_memory"] = MagicMock()

    override_auth(fastapi_client)
    response = fastapi_client.get("/api/v2/memory/search?q=alpha&limit=oops")

    assert response.status_code == 422


def test_memory_search_returns_empty_list_for_missing_results(
    fastapi_client, fastapi_context, override_auth
):
    continuum_memory = MagicMock()
    continuum_memory.retrieve.return_value = []
    fastapi_context["continuum_memory"] = continuum_memory

    override_auth(fastapi_client)
    response = fastapi_client.get("/api/v2/memory/search?q=missing")

    assert response.status_code == 200
    assert response.json()["results"] == []
    assert response.json()["total"] == 0


def test_memory_recall_returns_empty_list_for_missing_results(
    fastapi_client, fastapi_context, override_auth
):
    continuum_memory = MagicMock()
    continuum_memory.retrieve.return_value = []
    fastapi_context["continuum_memory"] = continuum_memory

    override_auth(fastapi_client)
    response = fastapi_client.get("/api/v2/memory/recall?q=missing")

    assert response.status_code == 200
    assert response.json()["memories"] == []
    assert response.json()["total"] == 0
