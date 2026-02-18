"""
Classify Namespace API

Provides access to content classification and sensitivity operations:
- Classify content by sensitivity level
- Get classification policies for sensitivity levels
- Batch classification support
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class ClassifyAPI:
    """
    Synchronous Classify API for content classification.

    Classifies content by sensitivity level and provides recommended
    handling policies for each level.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> result = client.classify.classify("Patient medical records...")
        >>> print(f"Level: {result['classification']['level']}")
        >>> policy = client.classify.get_policy("confidential")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def classify(
        self,
        content: str,
        categories: list[str] | None = None,
        threshold: float | None = None,
        document_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Classify content by sensitivity level.

        Args:
            content: Content to classify.
            categories: Optional category filter.
            threshold: Minimum confidence threshold.
            document_id: Optional document ID for audit logging.
            metadata: Additional metadata for classification context.

        Returns:
            Dict with classification results including level,
            confidence, and categories.
        """
        body: dict[str, Any] = {"content": content}
        if categories:
            body["categories"] = categories
        if threshold is not None:
            body["threshold"] = threshold
        if document_id:
            body["document_id"] = document_id
        if metadata:
            body["metadata"] = metadata
        return self._client.request("POST", "/api/v1/classify", json=body)

    def get_policy(self, level: str) -> dict[str, Any]:
        """
        Get recommended handling policy for a sensitivity level.

        Args:
            level: Sensitivity level (e.g., 'public', 'internal',
                'confidential', 'restricted').

        Returns:
            Dict with recommended policy including access controls,
            retention rules, and handling guidelines.
        """
        return self._client.request("GET", f"/api/v1/classify/policy/{level}")


class AsyncClassifyAPI:
    """
    Asynchronous Classify API for content classification.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     result = await client.classify.classify("Patient records...")
        ...     policy = await client.classify.get_policy("confidential")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def classify(
        self,
        content: str,
        categories: list[str] | None = None,
        threshold: float | None = None,
        document_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Classify content by sensitivity level.

        Args:
            content: Content to classify.
            categories: Optional category filter.
            threshold: Minimum confidence threshold.
            document_id: Optional document ID for audit logging.
            metadata: Additional metadata for classification context.

        Returns:
            Dict with classification results.
        """
        body: dict[str, Any] = {"content": content}
        if categories:
            body["categories"] = categories
        if threshold is not None:
            body["threshold"] = threshold
        if document_id:
            body["document_id"] = document_id
        if metadata:
            body["metadata"] = metadata
        return await self._client.request("POST", "/api/v1/classify", json=body)

    async def get_policy(self, level: str) -> dict[str, Any]:
        """Get recommended handling policy for a sensitivity level."""
        return await self._client.request("GET", f"/api/v1/classify/policy/{level}")
