"""
Quotas Namespace API

Provides methods for quota management:
- View usage limits
- Request quota increases
- Monitor quota consumption
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

class QuotasAPI:
    """Synchronous Quotas API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def list(self) -> dict[str, Any]:
        """List all quotas."""
        return self._client.request("GET", "/api/v1/quotas")

class AsyncQuotasAPI:
    """Asynchronous Quotas API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list(self) -> dict[str, Any]:
        """List all quotas."""
        return await self._client.request("GET", "/api/v1/quotas")

