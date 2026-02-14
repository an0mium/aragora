"""
CRM Namespace API

Provides methods for CRM integration:
- Contact management
- Deal tracking
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class CRMAPI:
    """Synchronous CRM API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def list_contacts(self, search: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List CRM contacts."""
        params: dict[str, Any] = {"limit": limit}
        if search:
            params["search"] = search
        return self._client.request("GET", "/api/v1/crm/contacts", params=params)

    def create_contact(self, email: str, name: str | None = None, **fields: Any) -> dict[str, Any]:
        """Create a contact."""
        data: dict[str, Any] = {"email": email, **fields}
        if name:
            data["name"] = name
        return self._client.request("POST", "/api/v1/crm/contacts", json=data)

    def list_deals(self, stage: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List deals."""
        params: dict[str, Any] = {"limit": limit}
        if stage:
            params["stage"] = stage
        return self._client.request("GET", "/api/v1/crm/deals", params=params)

    def create_deal(self, name: str, contact_id: str, value: float | None = None) -> dict[str, Any]:
        """Create a deal."""
        data: dict[str, Any] = {"name": name, "contact_id": contact_id}
        if value is not None:
            data["value"] = value
        return self._client.request("POST", "/api/v1/crm/deals", json=data)

    def get_status(self) -> dict[str, Any]:
        """
        Get CRM circuit breaker status.

        GET /api/v1/crm/status

        Returns:
            Dict with CRM status information
        """
        return self._client.request("GET", "/api/v1/crm/status")


class AsyncCRMAPI:
    """Asynchronous CRM API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_contacts(self, search: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List CRM contacts."""
        params: dict[str, Any] = {"limit": limit}
        if search:
            params["search"] = search
        return await self._client.request("GET", "/api/v1/crm/contacts", params=params)

    async def create_contact(
        self, email: str, name: str | None = None, **fields: Any
    ) -> dict[str, Any]:
        """Create a contact."""
        data: dict[str, Any] = {"email": email, **fields}
        if name:
            data["name"] = name
        return await self._client.request("POST", "/api/v1/crm/contacts", json=data)

    async def list_deals(self, stage: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List deals."""
        params: dict[str, Any] = {"limit": limit}
        if stage:
            params["stage"] = stage
        return await self._client.request("GET", "/api/v1/crm/deals", params=params)

    async def create_deal(
        self, name: str, contact_id: str, value: float | None = None
    ) -> dict[str, Any]:
        """Create a deal."""
        data: dict[str, Any] = {"name": name, "contact_id": contact_id}
        if value is not None:
            data["value"] = value
        return await self._client.request("POST", "/api/v1/crm/deals", json=data)

    async def get_status(self) -> dict[str, Any]:
        """Get CRM circuit breaker status. GET /api/v1/crm/status"""
        return await self._client.request("GET", "/api/v1/crm/status")
