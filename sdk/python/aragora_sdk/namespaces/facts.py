"""
Facts Namespace API.

Provides REST APIs for fact management:
- Listing and creating facts
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class FactsAPI:
    """Synchronous Facts API."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    def create_fact(
        self,
        content: str,
        source: str | None = None,
        confidence: float | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new fact."""
        data: dict[str, Any] = {"content": content}
        if source:
            data["source"] = source
        if confidence is not None:
            data["confidence"] = confidence
        if tags:
            data["tags"] = tags
        if metadata:
            data["metadata"] = metadata
        return self._client.request("POST", "/api/v1/facts", json=data)

    def list_facts(
        self,
        limit: int | None = None,
        offset: int | None = None,
        tag: str | None = None,
        source: str | None = None,
        min_confidence: float | None = None,
    ) -> dict[str, Any]:
        """List facts with optional filters."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        if tag:
            params["tag"] = tag
        if source:
            params["source"] = source
        if min_confidence is not None:
            params["min_confidence"] = min_confidence
        return self._client.request("GET", "/api/v1/facts", params=params if params else None)


class AsyncFactsAPI:
    """Asynchronous Facts API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def create_fact(
        self,
        content: str,
        source: str | None = None,
        confidence: float | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new fact."""
        data: dict[str, Any] = {"content": content}
        if source:
            data["source"] = source
        if confidence is not None:
            data["confidence"] = confidence
        if tags:
            data["tags"] = tags
        if metadata:
            data["metadata"] = metadata
        return await self._client.request("POST", "/api/v1/facts", json=data)

    async def list_facts(
        self,
        limit: int | None = None,
        offset: int | None = None,
        tag: str | None = None,
        source: str | None = None,
        min_confidence: float | None = None,
    ) -> dict[str, Any]:
        """List facts with optional filters."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        if tag:
            params["tag"] = tag
        if source:
            params["source"] = source
        if min_confidence is not None:
            params["min_confidence"] = min_confidence
        return await self._client.request("GET", "/api/v1/facts", params=params if params else None)
