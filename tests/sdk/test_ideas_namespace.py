"""
Tests for the Aragora Python SDK Ideas namespace.

Covers: IdeasAPI (sync) and AsyncIdeasAPI (async) methods for
idea canvas CRUD, node/edge management, export, and promotion.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("aragora_sdk", reason="aragora-sdk not installed")

from aragora_sdk.namespaces.ideas import (  # noqa: E402
    AsyncIdeasAPI,
    IdeasAPI,
)

# ---------------------------------------------------------------------------
# Method list shared across tests
# ---------------------------------------------------------------------------

ALL_METHODS = [
    "list_canvases",
    "create_canvas",
    "get_canvas",
    "update_canvas",
    "delete_canvas",
    "add_node",
    "update_node",
    "delete_node",
    "add_edge",
    "delete_edge",
    "export_canvas",
    "promote_nodes",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sync_client() -> MagicMock:
    """Create a mock synchronous client."""
    client = MagicMock()
    client._request = MagicMock(return_value={"ok": True})
    return client


@pytest.fixture
def async_client() -> MagicMock:
    """Create a mock asynchronous client."""
    client = MagicMock()
    client._request = AsyncMock(return_value={"ok": True})
    return client


@pytest.fixture
def sync_api(sync_client: MagicMock) -> IdeasAPI:
    return IdeasAPI(sync_client)


@pytest.fixture
def async_api(async_client: MagicMock) -> AsyncIdeasAPI:
    return AsyncIdeasAPI(async_client)


# ===========================================================================
# Structural / existence tests
# ===========================================================================


class TestIdeasNamespaceStructure:
    """Verify that both sync and async classes expose all 12 methods."""

    def test_sync_has_all_methods(self) -> None:
        for method_name in ALL_METHODS:
            assert hasattr(IdeasAPI, method_name), f"IdeasAPI missing method: {method_name}"

    def test_async_has_all_methods(self) -> None:
        for method_name in ALL_METHODS:
            assert hasattr(AsyncIdeasAPI, method_name), (
                f"AsyncIdeasAPI missing method: {method_name}"
            )

    def test_sync_method_count(self) -> None:
        public = [
            m for m in dir(IdeasAPI) if not m.startswith("_") and callable(getattr(IdeasAPI, m))
        ]
        assert len(public) == 12

    def test_async_method_count(self) -> None:
        public = [
            m
            for m in dir(AsyncIdeasAPI)
            if not m.startswith("_") and callable(getattr(AsyncIdeasAPI, m))
        ]
        assert len(public) == 12

    def test_async_methods_are_coroutines(self) -> None:
        api = AsyncIdeasAPI(MagicMock())
        for method_name in ALL_METHODS:
            method = getattr(api, method_name)
            assert inspect.iscoroutinefunction(method), (
                f"AsyncIdeasAPI.{method_name} should be a coroutine function"
            )


# ===========================================================================
# Client registration
# ===========================================================================


class TestClientRegistration:
    """Verify that the SDK clients expose an ``ideas`` attribute."""

    def test_sync_client_has_ideas(self) -> None:
        from aragora_sdk.client import AragoraClient

        assert hasattr(AragoraClient, "_init_namespaces")
        # Instantiate with minimal config to trigger _init_namespaces
        client = AragoraClient.__new__(AragoraClient)
        client.config = MagicMock()
        client._session = MagicMock()
        client._init_namespaces()
        assert hasattr(client, "ideas")
        assert isinstance(client.ideas, IdeasAPI)

    def test_async_client_has_ideas(self) -> None:
        from aragora_sdk.client import AragoraAsyncClient

        client = AragoraAsyncClient.__new__(AragoraAsyncClient)
        client.config = MagicMock()
        client._session = MagicMock()
        client._init_namespaces()
        assert hasattr(client, "ideas")
        assert isinstance(client.ideas, AsyncIdeasAPI)


# ===========================================================================
# Sync method signatures & request dispatch
# ===========================================================================


class TestIdeasAPISyncListCanvases:
    def test_default_params(self, sync_api: IdeasAPI, sync_client: MagicMock) -> None:
        sync_api.list_canvases()
        sync_client._request.assert_called_once_with(
            "GET", "/api/v1/ideas", params={"limit": 100, "offset": 0}
        )

    def test_with_filters(self, sync_api: IdeasAPI, sync_client: MagicMock) -> None:
        sync_api.list_canvases(workspace_id="ws1", owner_id="u1", limit=10, offset=5)
        sync_client._request.assert_called_once_with(
            "GET",
            "/api/v1/ideas",
            params={"limit": 10, "offset": 5, "workspace_id": "ws1", "owner_id": "u1"},
        )


class TestIdeasAPISyncCreateCanvas:
    def test_minimal(self, sync_api: IdeasAPI, sync_client: MagicMock) -> None:
        sync_api.create_canvas("My Canvas")
        sync_client._request.assert_called_once_with(
            "POST",
            "/api/v1/ideas",
            json={"name": "My Canvas", "description": ""},
        )

    def test_with_metadata(self, sync_api: IdeasAPI, sync_client: MagicMock) -> None:
        sync_api.create_canvas("C", description="desc", metadata={"k": "v"})
        sync_client._request.assert_called_once_with(
            "POST",
            "/api/v1/ideas",
            json={"name": "C", "description": "desc", "metadata": {"k": "v"}},
        )


class TestIdeasAPISyncGetCanvas:
    def test_calls_correct_endpoint(self, sync_api: IdeasAPI, sync_client: MagicMock) -> None:
        sync_api.get_canvas("abc-123")
        sync_client._request.assert_called_once_with("GET", "/api/v1/ideas/abc-123")


class TestIdeasAPISyncUpdateCanvas:
    def test_partial_update(self, sync_api: IdeasAPI, sync_client: MagicMock) -> None:
        sync_api.update_canvas("c1", name="New Name")
        sync_client._request.assert_called_once_with(
            "PUT", "/api/v1/ideas/c1", json={"name": "New Name"}
        )


class TestIdeasAPISyncDeleteCanvas:
    def test_calls_delete(self, sync_api: IdeasAPI, sync_client: MagicMock) -> None:
        sync_api.delete_canvas("c1")
        sync_client._request.assert_called_once_with("DELETE", "/api/v1/ideas/c1")


class TestIdeasAPISyncAddNode:
    def test_minimal(self, sync_api: IdeasAPI, sync_client: MagicMock) -> None:
        sync_api.add_node("c1", "My Idea")
        sync_client._request.assert_called_once_with(
            "POST",
            "/api/v1/ideas/c1/nodes",
            json={"label": "My Idea", "idea_type": "concept"},
        )

    def test_with_position_and_data(self, sync_api: IdeasAPI, sync_client: MagicMock) -> None:
        sync_api.add_node(
            "c1", "Idea", idea_type="question", position={"x": 10, "y": 20}, data={"key": 1}
        )
        sync_client._request.assert_called_once_with(
            "POST",
            "/api/v1/ideas/c1/nodes",
            json={
                "label": "Idea",
                "idea_type": "question",
                "position": {"x": 10, "y": 20},
                "data": {"key": 1},
            },
        )


class TestIdeasAPISyncUpdateNode:
    def test_partial(self, sync_api: IdeasAPI, sync_client: MagicMock) -> None:
        sync_api.update_node("c1", "n1", label="Updated")
        sync_client._request.assert_called_once_with(
            "PUT", "/api/v1/ideas/c1/nodes/n1", json={"label": "Updated"}
        )


class TestIdeasAPISyncDeleteNode:
    def test_calls_delete(self, sync_api: IdeasAPI, sync_client: MagicMock) -> None:
        sync_api.delete_node("c1", "n1")
        sync_client._request.assert_called_once_with("DELETE", "/api/v1/ideas/c1/nodes/n1")


class TestIdeasAPISyncAddEdge:
    def test_minimal(self, sync_api: IdeasAPI, sync_client: MagicMock) -> None:
        sync_api.add_edge("c1", "n1", "n2")
        sync_client._request.assert_called_once_with(
            "POST",
            "/api/v1/ideas/c1/edges",
            json={
                "source_id": "n1",
                "target_id": "n2",
                "edge_type": "default",
                "label": "",
            },
        )

    def test_with_all_params(self, sync_api: IdeasAPI, sync_client: MagicMock) -> None:
        sync_api.add_edge("c1", "n1", "n2", edge_type="supports", label="because", data={"w": 0.9})
        sync_client._request.assert_called_once_with(
            "POST",
            "/api/v1/ideas/c1/edges",
            json={
                "source_id": "n1",
                "target_id": "n2",
                "edge_type": "supports",
                "label": "because",
                "data": {"w": 0.9},
            },
        )


class TestIdeasAPISyncDeleteEdge:
    def test_calls_delete(self, sync_api: IdeasAPI, sync_client: MagicMock) -> None:
        sync_api.delete_edge("c1", "e1")
        sync_client._request.assert_called_once_with("DELETE", "/api/v1/ideas/c1/edges/e1")


class TestIdeasAPISyncExportCanvas:
    def test_calls_get(self, sync_api: IdeasAPI, sync_client: MagicMock) -> None:
        sync_api.export_canvas("c1")
        sync_client._request.assert_called_once_with("GET", "/api/v1/ideas/c1/export")


class TestIdeasAPISyncPromoteNodes:
    def test_sends_node_ids(self, sync_api: IdeasAPI, sync_client: MagicMock) -> None:
        sync_api.promote_nodes("c1", ["n1", "n2"])
        sync_client._request.assert_called_once_with(
            "POST",
            "/api/v1/ideas/c1/promote",
            json={"node_ids": ["n1", "n2"]},
        )


# ===========================================================================
# Async method signatures & request dispatch
# ===========================================================================


class TestIdeasAPIAsyncListCanvases:
    @pytest.mark.asyncio
    async def test_default_params(self, async_api: AsyncIdeasAPI, async_client: MagicMock) -> None:
        await async_api.list_canvases()
        async_client._request.assert_awaited_once_with(
            "GET", "/api/v1/ideas", params={"limit": 100, "offset": 0}
        )


class TestIdeasAPIAsyncCreateCanvas:
    @pytest.mark.asyncio
    async def test_minimal(self, async_api: AsyncIdeasAPI, async_client: MagicMock) -> None:
        await async_api.create_canvas("My Canvas")
        async_client._request.assert_awaited_once_with(
            "POST",
            "/api/v1/ideas",
            json={"name": "My Canvas", "description": ""},
        )


class TestIdeasAPIAsyncGetCanvas:
    @pytest.mark.asyncio
    async def test_calls_correct_endpoint(
        self, async_api: AsyncIdeasAPI, async_client: MagicMock
    ) -> None:
        await async_api.get_canvas("abc-123")
        async_client._request.assert_awaited_once_with("GET", "/api/v1/ideas/abc-123")


class TestIdeasAPIAsyncUpdateCanvas:
    @pytest.mark.asyncio
    async def test_partial_update(self, async_api: AsyncIdeasAPI, async_client: MagicMock) -> None:
        await async_api.update_canvas("c1", name="New Name")
        async_client._request.assert_awaited_once_with(
            "PUT", "/api/v1/ideas/c1", json={"name": "New Name"}
        )


class TestIdeasAPIAsyncDeleteCanvas:
    @pytest.mark.asyncio
    async def test_calls_delete(self, async_api: AsyncIdeasAPI, async_client: MagicMock) -> None:
        await async_api.delete_canvas("c1")
        async_client._request.assert_awaited_once_with("DELETE", "/api/v1/ideas/c1")


class TestIdeasAPIAsyncAddNode:
    @pytest.mark.asyncio
    async def test_minimal(self, async_api: AsyncIdeasAPI, async_client: MagicMock) -> None:
        await async_api.add_node("c1", "My Idea")
        async_client._request.assert_awaited_once_with(
            "POST",
            "/api/v1/ideas/c1/nodes",
            json={"label": "My Idea", "idea_type": "concept"},
        )


class TestIdeasAPIAsyncUpdateNode:
    @pytest.mark.asyncio
    async def test_partial(self, async_api: AsyncIdeasAPI, async_client: MagicMock) -> None:
        await async_api.update_node("c1", "n1", label="Updated")
        async_client._request.assert_awaited_once_with(
            "PUT", "/api/v1/ideas/c1/nodes/n1", json={"label": "Updated"}
        )


class TestIdeasAPIAsyncDeleteNode:
    @pytest.mark.asyncio
    async def test_calls_delete(self, async_api: AsyncIdeasAPI, async_client: MagicMock) -> None:
        await async_api.delete_node("c1", "n1")
        async_client._request.assert_awaited_once_with("DELETE", "/api/v1/ideas/c1/nodes/n1")


class TestIdeasAPIAsyncAddEdge:
    @pytest.mark.asyncio
    async def test_minimal(self, async_api: AsyncIdeasAPI, async_client: MagicMock) -> None:
        await async_api.add_edge("c1", "n1", "n2")
        async_client._request.assert_awaited_once_with(
            "POST",
            "/api/v1/ideas/c1/edges",
            json={
                "source_id": "n1",
                "target_id": "n2",
                "edge_type": "default",
                "label": "",
            },
        )


class TestIdeasAPIAsyncDeleteEdge:
    @pytest.mark.asyncio
    async def test_calls_delete(self, async_api: AsyncIdeasAPI, async_client: MagicMock) -> None:
        await async_api.delete_edge("c1", "e1")
        async_client._request.assert_awaited_once_with("DELETE", "/api/v1/ideas/c1/edges/e1")


class TestIdeasAPIAsyncExportCanvas:
    @pytest.mark.asyncio
    async def test_calls_get(self, async_api: AsyncIdeasAPI, async_client: MagicMock) -> None:
        await async_api.export_canvas("c1")
        async_client._request.assert_awaited_once_with("GET", "/api/v1/ideas/c1/export")


class TestIdeasAPIAsyncPromoteNodes:
    @pytest.mark.asyncio
    async def test_sends_node_ids(self, async_api: AsyncIdeasAPI, async_client: MagicMock) -> None:
        await async_api.promote_nodes("c1", ["n1", "n2"])
        async_client._request.assert_awaited_once_with(
            "POST",
            "/api/v1/ideas/c1/promote",
            json={"node_ids": ["n1", "n2"]},
        )
