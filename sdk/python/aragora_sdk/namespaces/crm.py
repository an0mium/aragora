"""
CRM Namespace API

Provides methods for CRM integration:
- Contact management
- Deal tracking
- Activity logging
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

    def get_contact(self, contact_id: str) -> dict[str, Any]:
        """Get contact by ID."""
        return self._client.request("GET", f"/api/v1/crm/contacts/{contact_id}")

    def create_contact(self, email: str, name: str | None = None, **fields: Any) -> dict[str, Any]:
        """Create a contact."""
        data: dict[str, Any] = {"email": email, **fields}
        if name:
            data["name"] = name
        return self._client.request("POST", "/api/v1/crm/contacts", json=data)

    def update_contact(self, contact_id: str, **fields: Any) -> dict[str, Any]:
        """Update a contact."""
        return self._client.request("PATCH", f"/api/v1/crm/contacts/{contact_id}", json=fields)

    def list_deals(self, stage: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List deals."""
        params: dict[str, Any] = {"limit": limit}
        if stage:
            params["stage"] = stage
        return self._client.request("GET", "/api/v1/crm/deals", params=params)

    def get_deal(self, deal_id: str) -> dict[str, Any]:
        """Get deal by ID."""
        return self._client.request("GET", f"/api/v1/crm/deals/{deal_id}")

    def create_deal(self, name: str, contact_id: str, value: float | None = None) -> dict[str, Any]:
        """Create a deal."""
        data: dict[str, Any] = {"name": name, "contact_id": contact_id}
        if value is not None:
            data["value"] = value
        return self._client.request("POST", "/api/v1/crm/deals", json=data)

    def update_deal(self, deal_id: str, **fields: Any) -> dict[str, Any]:
        """Update a deal."""
        return self._client.request("PATCH", f"/api/v1/crm/deals/{deal_id}", json=fields)

    def log_activity(self, contact_id: str, activity_type: str, content: str) -> dict[str, Any]:
        """Log an activity."""
        return self._client.request(
            "POST",
            f"/api/v1/crm/contacts/{contact_id}/activities",
            json={
                "type": activity_type,
                "content": content,
            },
        )

    def sync(self, provider: str) -> dict[str, Any]:
        """Sync with external CRM provider."""
        return self._client.request("POST", f"/api/v1/crm/sync/{provider}")


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

    async def get_contact(self, contact_id: str) -> dict[str, Any]:
        """Get contact by ID."""
        return await self._client.request("GET", f"/api/v1/crm/contacts/{contact_id}")

    async def create_contact(
        self, email: str, name: str | None = None, **fields: Any
    ) -> dict[str, Any]:
        """Create a contact."""
        data: dict[str, Any] = {"email": email, **fields}
        if name:
            data["name"] = name
        return await self._client.request("POST", "/api/v1/crm/contacts", json=data)

    async def update_contact(self, contact_id: str, **fields: Any) -> dict[str, Any]:
        """Update a contact."""
        return await self._client.request(
            "PATCH", f"/api/v1/crm/contacts/{contact_id}", json=fields
        )

    async def list_deals(self, stage: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List deals."""
        params: dict[str, Any] = {"limit": limit}
        if stage:
            params["stage"] = stage
        return await self._client.request("GET", "/api/v1/crm/deals", params=params)

    async def get_deal(self, deal_id: str) -> dict[str, Any]:
        """Get deal by ID."""
        return await self._client.request("GET", f"/api/v1/crm/deals/{deal_id}")

    async def create_deal(
        self, name: str, contact_id: str, value: float | None = None
    ) -> dict[str, Any]:
        """Create a deal."""
        data: dict[str, Any] = {"name": name, "contact_id": contact_id}
        if value is not None:
            data["value"] = value
        return await self._client.request("POST", "/api/v1/crm/deals", json=data)

    async def update_deal(self, deal_id: str, **fields: Any) -> dict[str, Any]:
        """Update a deal."""
        return await self._client.request("PATCH", f"/api/v1/crm/deals/{deal_id}", json=fields)

    async def log_activity(
        self, contact_id: str, activity_type: str, content: str
    ) -> dict[str, Any]:
        """Log an activity."""
        return await self._client.request(
            "POST",
            f"/api/v1/crm/contacts/{contact_id}/activities",
            json={
                "type": activity_type,
                "content": content,
            },
        )

    async def sync(self, provider: str) -> dict[str, Any]:
        """Sync with external CRM provider."""
        return await self._client.request("POST", f"/api/v1/crm/sync/{provider}")
