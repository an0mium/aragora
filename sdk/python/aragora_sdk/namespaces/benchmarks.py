"""
Benchmarks Namespace API

Provides methods for running and comparing benchmarks.
"""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class BenchmarksAPI:
    """Synchronous Benchmarks API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def list(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List benchmark results.

        Args:
            limit: Maximum results to return.
            offset: Pagination offset.

        Returns:
            Dict with benchmark results and pagination info.
        """
        return self._client.request(
            "GET", "/api/v1/benchmarks", params={"limit": limit, "offset": offset}
        )

    def categories(self) -> dict[str, Any]:
        """List available benchmark categories."""
        return self._client.request("GET", "/api/v1/benchmarks/categories")

    def compare(
        self,
        benchmark_ids: builtins.list[str] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """Compare benchmark results.

        Args:
            benchmark_ids: IDs of benchmarks to compare.
            **params: Additional comparison parameters.

        Returns:
            Dict with comparison results.
        """
        query: dict[str, Any] = {**params}
        if benchmark_ids:
            query["ids"] = ",".join(benchmark_ids)
        return self._client.request("GET", "/api/v1/benchmarks/compare", params=query)


class AsyncBenchmarksAPI:
    """Asynchronous Benchmarks API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List benchmark results."""
        return await self._client.request(
            "GET", "/api/v1/benchmarks", params={"limit": limit, "offset": offset}
        )

    async def compare(
        self,
        benchmark_ids: builtins.list[str] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """Compare benchmark results."""
        query: dict[str, Any] = {**params}
        if benchmark_ids:
            query["ids"] = ",".join(benchmark_ids)
        return await self._client.request("GET", "/api/v1/benchmarks/compare", params=query)

    async def categories(self) -> dict[str, Any]:
        """List available benchmark categories."""
        return await self._client.request("GET", "/api/v1/benchmarks/categories")
