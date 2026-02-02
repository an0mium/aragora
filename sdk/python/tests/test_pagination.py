"""Tests for pagination helpers: SyncPaginator and AsyncPaginator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora_sdk.pagination import AsyncPaginator, SyncPaginator


class TestSyncPaginatorInitialization:
    """Tests for SyncPaginator initialization."""

    def test_init_with_defaults(self) -> None:
        """SyncPaginator initializes with default page size of 20."""
        mock_client = MagicMock()
        paginator = SyncPaginator(mock_client, "/api/v1/items")

        assert paginator._client is mock_client
        assert paginator._path == "/api/v1/items"
        assert paginator._params == {}
        assert paginator._page_size == 20
        assert paginator._offset == 0
        assert paginator._buffer == []
        assert paginator._exhausted is False
        assert paginator._total is None

    def test_init_with_custom_params(self) -> None:
        """SyncPaginator accepts custom params and page_size."""
        mock_client = MagicMock()
        params = {"status": "active", "workspace_id": "ws_123"}
        paginator = SyncPaginator(mock_client, "/api/v1/items", params=params, page_size=50)

        assert paginator._params == params
        assert paginator._page_size == 50

    def test_init_with_none_params_defaults_to_empty_dict(self) -> None:
        """SyncPaginator converts None params to empty dict."""
        mock_client = MagicMock()
        paginator = SyncPaginator(mock_client, "/api/v1/items", params=None)

        assert paginator._params == {}


class TestSyncPaginatorIteration:
    """Tests for SyncPaginator iteration behavior."""

    def test_iter_returns_self(self) -> None:
        """SyncPaginator __iter__ returns self."""
        mock_client = MagicMock()
        paginator = SyncPaginator(mock_client, "/api/v1/items")

        assert iter(paginator) is paginator

    def test_iteration_through_single_page(self) -> None:
        """SyncPaginator iterates through a single page of results."""
        mock_client = MagicMock()
        mock_client.request.return_value = {
            "items": [{"id": "1"}, {"id": "2"}, {"id": "3"}],
            "total": 3,
        }

        paginator = SyncPaginator(mock_client, "/api/v1/items", page_size=10)
        results = list(paginator)

        assert len(results) == 3
        assert results[0]["id"] == "1"
        assert results[1]["id"] == "2"
        assert results[2]["id"] == "3"
        mock_client.request.assert_called_once_with(
            "GET", "/api/v1/items", params={"limit": 10, "offset": 0}
        )

    def test_iteration_through_multiple_pages(self) -> None:
        """SyncPaginator fetches multiple pages automatically."""
        mock_client = MagicMock()
        mock_client.request.side_effect = [
            {"items": [{"id": "1"}, {"id": "2"}], "total": 5},
            {"items": [{"id": "3"}, {"id": "4"}], "total": 5},
            {"items": [{"id": "5"}], "total": 5},
        ]

        paginator = SyncPaginator(mock_client, "/api/v1/items", page_size=2)
        results = list(paginator)

        assert len(results) == 5
        assert [r["id"] for r in results] == ["1", "2", "3", "4", "5"]
        assert mock_client.request.call_count == 3

        # Verify correct offset progression
        calls = mock_client.request.call_args_list
        assert calls[0][1]["params"]["offset"] == 0
        assert calls[1][1]["params"]["offset"] == 2
        assert calls[2][1]["params"]["offset"] == 4

    def test_iteration_stops_when_fewer_items_than_page_size(self) -> None:
        """SyncPaginator stops fetching when a page has fewer items than page_size."""
        mock_client = MagicMock()
        mock_client.request.side_effect = [
            {"items": [{"id": "1"}, {"id": "2"}]},
            {"items": [{"id": "3"}]},  # Less than page_size, marks exhausted
        ]

        paginator = SyncPaginator(mock_client, "/api/v1/items", page_size=2)
        results = list(paginator)

        assert len(results) == 3
        assert mock_client.request.call_count == 2

    def test_iteration_stops_when_offset_reaches_total(self) -> None:
        """SyncPaginator stops when offset equals or exceeds total."""
        mock_client = MagicMock()
        mock_client.request.side_effect = [
            {"items": [{"id": "1"}, {"id": "2"}], "total": 4},
            {"items": [{"id": "3"}, {"id": "4"}], "total": 4},
        ]

        paginator = SyncPaginator(mock_client, "/api/v1/items", page_size=2)
        results = list(paginator)

        assert len(results) == 4
        assert mock_client.request.call_count == 2


class TestSyncPaginatorEmptyResults:
    """Tests for SyncPaginator handling of empty results."""

    def test_empty_first_page(self) -> None:
        """SyncPaginator handles empty first page gracefully."""
        mock_client = MagicMock()
        mock_client.request.return_value = {"items": [], "total": 0}

        paginator = SyncPaginator(mock_client, "/api/v1/items")
        results = list(paginator)

        assert results == []
        mock_client.request.assert_called_once()

    def test_empty_response_dict_without_items(self) -> None:
        """SyncPaginator handles response dict without items key."""
        mock_client = MagicMock()
        mock_client.request.return_value = {"total": 0}

        paginator = SyncPaginator(mock_client, "/api/v1/items")
        results = list(paginator)

        assert results == []


class TestSyncPaginatorResponseFormats:
    """Tests for SyncPaginator handling different response formats."""

    def test_response_with_items_key(self) -> None:
        """SyncPaginator handles response with 'items' key."""
        mock_client = MagicMock()
        mock_client.request.return_value = {"items": [{"id": "1"}], "total": 1}

        paginator = SyncPaginator(mock_client, "/api/v1/items", page_size=10)
        results = list(paginator)

        assert len(results) == 1
        assert results[0]["id"] == "1"

    def test_response_with_data_key(self) -> None:
        """SyncPaginator handles response with 'data' key instead of 'items'."""
        mock_client = MagicMock()
        mock_client.request.return_value = {"data": [{"id": "1"}, {"id": "2"}], "total": 2}

        paginator = SyncPaginator(mock_client, "/api/v1/items", page_size=10)
        results = list(paginator)

        assert len(results) == 2
        assert results[0]["id"] == "1"
        assert results[1]["id"] == "2"

    def test_response_as_list(self) -> None:
        """SyncPaginator handles response that is a raw list."""
        mock_client = MagicMock()
        mock_client.request.return_value = [{"id": "1"}, {"id": "2"}]

        paginator = SyncPaginator(mock_client, "/api/v1/items", page_size=10)
        results = list(paginator)

        assert len(results) == 2

    def test_response_neither_dict_nor_list(self) -> None:
        """SyncPaginator handles non-dict non-list response gracefully."""
        mock_client = MagicMock()
        mock_client.request.return_value = "unexpected"

        paginator = SyncPaginator(mock_client, "/api/v1/items", page_size=10)
        results = list(paginator)

        assert results == []


class TestSyncPaginatorTotalProperty:
    """Tests for SyncPaginator total property."""

    def test_total_is_none_before_first_fetch(self) -> None:
        """SyncPaginator total is None before first fetch."""
        mock_client = MagicMock()
        paginator = SyncPaginator(mock_client, "/api/v1/items")

        assert paginator.total is None

    def test_total_set_from_response(self) -> None:
        """SyncPaginator total is set from API response."""
        mock_client = MagicMock()
        mock_client.request.return_value = {"items": [{"id": "1"}], "total": 42}

        paginator = SyncPaginator(mock_client, "/api/v1/items", page_size=10)
        # Trigger fetch by getting first item
        next(paginator)

        assert paginator.total == 42

    def test_total_none_when_not_in_response(self) -> None:
        """SyncPaginator total remains None when response lacks total."""
        mock_client = MagicMock()
        mock_client.request.return_value = {"items": [{"id": "1"}]}

        paginator = SyncPaginator(mock_client, "/api/v1/items", page_size=10)
        next(paginator)

        assert paginator.total is None


class TestSyncPaginatorParams:
    """Tests for SyncPaginator parameter handling."""

    def test_params_merged_with_pagination(self) -> None:
        """SyncPaginator merges user params with pagination params."""
        mock_client = MagicMock()
        mock_client.request.return_value = {"items": []}

        paginator = SyncPaginator(
            mock_client,
            "/api/v1/items",
            params={"status": "active", "workspace_id": "ws_123"},
            page_size=25,
        )
        list(paginator)

        mock_client.request.assert_called_once_with(
            "GET",
            "/api/v1/items",
            params={"status": "active", "workspace_id": "ws_123", "limit": 25, "offset": 0},
        )


class TestAsyncPaginatorInitialization:
    """Tests for AsyncPaginator initialization."""

    def test_init_with_defaults(self) -> None:
        """AsyncPaginator initializes with default page size of 20."""
        mock_client = MagicMock()
        paginator = AsyncPaginator(mock_client, "/api/v1/items")

        assert paginator._client is mock_client
        assert paginator._path == "/api/v1/items"
        assert paginator._params == {}
        assert paginator._page_size == 20
        assert paginator._offset == 0
        assert paginator._buffer == []
        assert paginator._exhausted is False
        assert paginator._total is None

    def test_init_with_custom_params(self) -> None:
        """AsyncPaginator accepts custom params and page_size."""
        mock_client = MagicMock()
        params = {"status": "active"}
        paginator = AsyncPaginator(mock_client, "/api/v1/items", params=params, page_size=100)

        assert paginator._params == params
        assert paginator._page_size == 100


class TestAsyncPaginatorIteration:
    """Tests for AsyncPaginator iteration behavior."""

    def test_aiter_returns_self(self) -> None:
        """AsyncPaginator __aiter__ returns self."""
        mock_client = MagicMock()
        paginator = AsyncPaginator(mock_client, "/api/v1/items")

        assert paginator.__aiter__() is paginator

    @pytest.mark.asyncio
    async def test_async_iteration_through_single_page(self) -> None:
        """AsyncPaginator iterates through a single page of results."""
        mock_client = MagicMock()
        mock_client.request = AsyncMock(
            return_value={"items": [{"id": "1"}, {"id": "2"}, {"id": "3"}], "total": 3}
        )

        paginator = AsyncPaginator(mock_client, "/api/v1/items", page_size=10)
        results = []
        async for item in paginator:
            results.append(item)

        assert len(results) == 3
        assert results[0]["id"] == "1"
        assert results[1]["id"] == "2"
        assert results[2]["id"] == "3"
        mock_client.request.assert_called_once_with(
            "GET", "/api/v1/items", params={"limit": 10, "offset": 0}
        )

    @pytest.mark.asyncio
    async def test_async_iteration_through_multiple_pages(self) -> None:
        """AsyncPaginator fetches multiple pages automatically."""
        mock_client = MagicMock()
        mock_client.request = AsyncMock(
            side_effect=[
                {"items": [{"id": "1"}, {"id": "2"}], "total": 5},
                {"items": [{"id": "3"}, {"id": "4"}], "total": 5},
                {"items": [{"id": "5"}], "total": 5},
            ]
        )

        paginator = AsyncPaginator(mock_client, "/api/v1/items", page_size=2)
        results = []
        async for item in paginator:
            results.append(item)

        assert len(results) == 5
        assert [r["id"] for r in results] == ["1", "2", "3", "4", "5"]
        assert mock_client.request.call_count == 3

    @pytest.mark.asyncio
    async def test_async_iteration_stops_when_fewer_items(self) -> None:
        """AsyncPaginator stops fetching when a page has fewer items than page_size."""
        mock_client = MagicMock()
        mock_client.request = AsyncMock(
            side_effect=[
                {"items": [{"id": "1"}, {"id": "2"}]},
                {"items": [{"id": "3"}]},
            ]
        )

        paginator = AsyncPaginator(mock_client, "/api/v1/items", page_size=2)
        results = []
        async for item in paginator:
            results.append(item)

        assert len(results) == 3
        assert mock_client.request.call_count == 2


class TestAsyncPaginatorEmptyResults:
    """Tests for AsyncPaginator handling of empty results."""

    @pytest.mark.asyncio
    async def test_async_empty_first_page(self) -> None:
        """AsyncPaginator handles empty first page gracefully."""
        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value={"items": [], "total": 0})

        paginator = AsyncPaginator(mock_client, "/api/v1/items")
        results = []
        async for item in paginator:
            results.append(item)

        assert results == []
        mock_client.request.assert_called_once()


class TestAsyncPaginatorResponseFormats:
    """Tests for AsyncPaginator handling different response formats."""

    @pytest.mark.asyncio
    async def test_async_response_with_data_key(self) -> None:
        """AsyncPaginator handles response with 'data' key."""
        mock_client = MagicMock()
        mock_client.request = AsyncMock(
            return_value={"data": [{"id": "1"}, {"id": "2"}], "total": 2}
        )

        paginator = AsyncPaginator(mock_client, "/api/v1/items", page_size=10)
        results = []
        async for item in paginator:
            results.append(item)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_async_response_as_list(self) -> None:
        """AsyncPaginator handles response that is a raw list."""
        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=[{"id": "1"}])

        paginator = AsyncPaginator(mock_client, "/api/v1/items", page_size=10)
        results = []
        async for item in paginator:
            results.append(item)

        assert len(results) == 1


class TestAsyncPaginatorTotalProperty:
    """Tests for AsyncPaginator total property."""

    def test_async_total_is_none_before_first_fetch(self) -> None:
        """AsyncPaginator total is None before first fetch."""
        mock_client = MagicMock()
        paginator = AsyncPaginator(mock_client, "/api/v1/items")

        assert paginator.total is None

    @pytest.mark.asyncio
    async def test_async_total_set_from_response(self) -> None:
        """AsyncPaginator total is set from API response."""
        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value={"items": [{"id": "1"}], "total": 100})

        paginator = AsyncPaginator(mock_client, "/api/v1/items", page_size=10)
        await paginator.__anext__()

        assert paginator.total == 100
