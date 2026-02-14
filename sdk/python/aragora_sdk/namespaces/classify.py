"""
Classify Namespace API

Provides access to content classification operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

class ClassifyAPI:
    """Synchronous Classify API for content classification."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def classify(
        self,
        content: str,
        categories: list[str] | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Classify content.

        Args:
            content: Content to classify.
            categories: Optional category filter.
            threshold: Minimum confidence threshold.

        Returns:
            Classification results with categories and scores.
        """
        body: dict[str, Any] = {"content": content}
        if categories:
            body["categories"] = categories
        if threshold is not None:
            body["threshold"] = threshold
        return self._client.request("POST", "/api/v1/classify", json=body)


class AsyncClassifyAPI:
    """Asynchronous Classify API for content classification."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def classify(
        self,
        content: str,
        categories: list[str] | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Classify content.

        Args:
            content: Content to classify.
            categories: Optional category filter.
            threshold: Minimum confidence threshold.

        Returns:
            Classification results with categories and scores.
        """
        body: dict[str, Any] = {"content": content}
        if categories:
            body["categories"] = categories
        if threshold is not None:
            body["threshold"] = threshold
        return await self._client.request("POST", "/api/v1/classify", json=body)
