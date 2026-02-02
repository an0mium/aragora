"""
Aragora Pagination Helpers

Provides auto-paginating iterators for list endpoints, allowing users to
iterate through all results without manually handling pagination.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import AragoraAsyncClient, AragoraClient


class SyncPaginator(Iterator[dict[str, Any]]):
    """Auto-paginating synchronous iterator for list endpoints.

    Automatically fetches additional pages as needed while iterating.

    Example::

        for debate in client.debates.list_all(status="active"):
            print(debate["id"])
    """

    def __init__(
        self,
        client: AragoraClient,
        path: str,
        params: dict[str, Any] | None = None,
        page_size: int = 20,
    ) -> None:
        """Initialize the paginator.

        Args:
            client: The AragoraClient instance to use for requests.
            path: The API endpoint path.
            params: Additional query parameters to include in requests.
            page_size: Number of items to fetch per page.
        """
        self._client = client
        self._path = path
        self._params = params or {}
        self._page_size = page_size
        self._offset = 0
        self._buffer: list[dict[str, Any]] = []
        self._exhausted = False
        self._total: int | None = None

    def __iter__(self) -> SyncPaginator:
        return self

    def __next__(self) -> dict[str, Any]:
        if not self._buffer:
            if self._exhausted:
                raise StopIteration
            self._fetch_page()
        if not self._buffer:
            raise StopIteration
        return self._buffer.pop(0)

    def _fetch_page(self) -> None:
        """Fetch the next page of results."""
        params = {
            **self._params,
            "limit": self._page_size,
            "offset": self._offset,
        }
        response = self._client.request("GET", self._path, params=params)

        # Handle different response formats
        if isinstance(response, dict):
            raw_items = response.get("items", response.get("data", []))
            items: list[dict[str, Any]] = raw_items if raw_items is not None else []
            self._total = response.get("total")
        else:
            items = response if isinstance(response, list) else []

        if items:
            self._buffer.extend(items)
            self._offset += len(items)

            # Check if we've exhausted all results
            if len(items) < self._page_size:
                self._exhausted = True
            elif self._total is not None and self._offset >= self._total:
                self._exhausted = True
        else:
            self._exhausted = True

    @property
    def total(self) -> int | None:
        """Return the total number of items, if known from the API response."""
        return self._total


class AsyncPaginator(AsyncIterator[dict[str, Any]]):
    """Auto-paginating asynchronous iterator for list endpoints.

    Automatically fetches additional pages as needed while iterating.

    Example::

        async for debate in client.debates.list_all(status="active"):
            print(debate["id"])
    """

    def __init__(
        self,
        client: AragoraAsyncClient,
        path: str,
        params: dict[str, Any] | None = None,
        page_size: int = 20,
    ) -> None:
        """Initialize the paginator.

        Args:
            client: The AragoraAsyncClient instance to use for requests.
            path: The API endpoint path.
            params: Additional query parameters to include in requests.
            page_size: Number of items to fetch per page.
        """
        self._client = client
        self._path = path
        self._params = params or {}
        self._page_size = page_size
        self._offset = 0
        self._buffer: list[dict[str, Any]] = []
        self._exhausted = False
        self._total: int | None = None

    def __aiter__(self) -> AsyncPaginator:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if not self._buffer:
            if self._exhausted:
                raise StopAsyncIteration
            await self._fetch_page()
        if not self._buffer:
            raise StopAsyncIteration
        return self._buffer.pop(0)

    async def _fetch_page(self) -> None:
        """Fetch the next page of results."""
        params = {
            **self._params,
            "limit": self._page_size,
            "offset": self._offset,
        }
        response = await self._client.request("GET", self._path, params=params)

        # Handle different response formats
        if isinstance(response, dict):
            raw_items = response.get("items", response.get("data", []))
            items: list[dict[str, Any]] = raw_items if raw_items is not None else []
            self._total = response.get("total")
        else:
            items = response if isinstance(response, list) else []

        if items:
            self._buffer.extend(items)
            self._offset += len(items)

            # Check if we've exhausted all results
            if len(items) < self._page_size:
                self._exhausted = True
            elif self._total is not None and self._offset >= self._total:
                self._exhausted = True
        else:
            self._exhausted = True

    @property
    def total(self) -> int | None:
        """Return the total number of items, if known from the API response."""
        return self._total
