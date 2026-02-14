"""
Matrix Debates Namespace API

Provides methods for matrix debate management:
- Create and list matrix debates
- Get individual matrix debate details
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class MatrixDebatesAPI:
    """Synchronous Matrix Debates API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def list(self, **kwargs: Any) -> dict[str, Any]:
        """
        List matrix debates.

        GET /api/v1/matrix-debates

        Returns:
            Dict with matrix debates list
        """
        return self._client.request("GET", "/api/v1/matrix-debates", params=kwargs or None)

    def get(self, debate_id: str) -> dict[str, Any]:
        """
        Get a matrix debate by ID.

        GET /api/v1/matrix-debates/:id

        Args:
            debate_id: Matrix debate identifier

        Returns:
            Dict with matrix debate details
        """
        return self._client.request("GET", f"/api/v1/matrix-debates/{debate_id}")

    def create(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Create a new matrix debate.

        POST /api/v1/matrix-debates

        Args:
            data: Matrix debate configuration

        Returns:
            Dict with created matrix debate
        """
        return self._client.request("POST", "/api/v1/matrix-debates", json=data)


class AsyncMatrixDebatesAPI:
    """Asynchronous Matrix Debates API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list(self, **kwargs: Any) -> dict[str, Any]:
        """List matrix debates. GET /api/v1/matrix-debates"""
        return await self._client.request("GET", "/api/v1/matrix-debates", params=kwargs or None)

    async def get(self, debate_id: str) -> dict[str, Any]:
        """Get a matrix debate by ID. GET /api/v1/matrix-debates/:id"""
        return await self._client.request("GET", f"/api/v1/matrix-debates/{debate_id}")

    async def create(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create a new matrix debate. POST /api/v1/matrix-debates"""
        return await self._client.request("POST", "/api/v1/matrix-debates", json=data)
