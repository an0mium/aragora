"""
CRM Namespace API

Provides methods for unified CRM platform integration:
- Platform connections (Salesforce, HubSpot, Pipedrive)
- Cross-platform contact management
- Company and deal tracking
- Pipeline management
- Lead enrichment and sync
- CRM search
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class CRMAPI:
    """
    Synchronous CRM API.

    Unified interface for CRM platforms including Salesforce, HubSpot,
    and Pipedrive with cross-platform contact, company, and deal management.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> platforms = client.crm.list_platforms()
        >>> contacts = client.crm.list_contacts(search="Acme")
        >>> deals = client.crm.list_deals(stage="negotiation")
        >>> pipeline = client.crm.get_pipeline()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # =========================================================================
    # Platform Management
    # =========================================================================

    def list_platforms(self) -> dict[str, Any]:
        """
        List connected CRM platforms.

        Returns:
            Dict with connected CRM platforms and their configurations.
        """
        return self._client.request("GET", "/api/v1/crm/platforms")

    def connect(self, **kwargs: Any) -> dict[str, Any]:
        """
        Connect a CRM platform.

        Args:
            **kwargs: Connection parameters including:
                - platform: Platform type (salesforce, hubspot, pipedrive)
                - api_key: Platform API key or OAuth token
                - domain: Platform domain

        Returns:
            Dict with connection status and platform details.
        """
        return self._client.request("POST", "/api/v1/crm/connect", json=kwargs)

    def disconnect(self, platform: str) -> dict[str, Any]:
        """
        Disconnect a CRM platform.

        Args:
            platform: Platform identifier to disconnect.

        Returns:
            Dict with disconnection confirmation.
        """
        return self._client.request("DELETE", f"/api/v1/crm/{platform}")

    def get_status(self) -> dict[str, Any]:
        """
        Get CRM integration status including circuit breaker state.

        Returns:
            Dict with CRM status information and circuit breaker health.
        """
        return self._client.request("GET", "/api/v1/crm/status")

    # =========================================================================
    # Contacts
    # =========================================================================

    def list_contacts(
        self, search: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        """
        List CRM contacts across all connected platforms.

        Args:
            search: Optional search query to filter contacts.
            limit: Maximum number of contacts to return.

        Returns:
            Dict with contacts list.
        """
        params: dict[str, Any] = {"limit": limit}
        if search:
            params["search"] = search
        return self._client.request("GET", "/api/v1/crm/contacts", params=params)

    def list_platform_contacts(
        self, platform: str, limit: int = 20
    ) -> dict[str, Any]:
        """
        List contacts for a specific CRM platform.

        Args:
            platform: Platform identifier.
            limit: Maximum number of contacts to return.

        Returns:
            Dict with platform-specific contacts.
        """
        params: dict[str, Any] = {"limit": limit}
        return self._client.request(
            "GET", f"/api/v1/crm/{platform}/contacts", params=params
        )

    def get_contact(self, platform: str, contact_id: str) -> dict[str, Any]:
        """
        Get contact details from a specific platform.

        Args:
            platform: Platform identifier.
            contact_id: Contact identifier.

        Returns:
            Dict with contact details.
        """
        return self._client.request(
            "GET", f"/api/v1/crm/{platform}/contacts/{contact_id}"
        )

    def create_contact(
        self, email: str, name: str | None = None, **fields: Any
    ) -> dict[str, Any]:
        """
        Create a contact (cross-platform).

        Args:
            email: Contact email address.
            name: Contact name.
            **fields: Additional contact fields.

        Returns:
            Dict with created contact details.
        """
        data: dict[str, Any] = {"email": email, **fields}
        if name:
            data["name"] = name
        return self._client.request("POST", "/api/v1/crm/contacts", json=data)

    # =========================================================================
    # Companies
    # =========================================================================

    def list_companies(self, limit: int = 20) -> dict[str, Any]:
        """
        List companies across all connected platforms.

        Args:
            limit: Maximum number of companies to return.

        Returns:
            Dict with companies list.
        """
        params: dict[str, Any] = {"limit": limit}
        return self._client.request("GET", "/api/v1/crm/companies", params=params)

    def list_platform_companies(
        self, platform: str, limit: int = 20
    ) -> dict[str, Any]:
        """
        List companies for a specific CRM platform.

        Args:
            platform: Platform identifier.
            limit: Maximum number of companies to return.

        Returns:
            Dict with platform-specific companies.
        """
        params: dict[str, Any] = {"limit": limit}
        return self._client.request(
            "GET", f"/api/v1/crm/{platform}/companies", params=params
        )

    def get_company(self, platform: str, company_id: str) -> dict[str, Any]:
        """
        Get company details from a specific platform.

        Args:
            platform: Platform identifier.
            company_id: Company identifier.

        Returns:
            Dict with company details.
        """
        return self._client.request(
            "GET", f"/api/v1/crm/{platform}/companies/{company_id}"
        )

    # =========================================================================
    # Deals
    # =========================================================================

    def list_deals(
        self, stage: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        """
        List deals across all connected platforms.

        Args:
            stage: Filter by deal stage (e.g., 'qualification', 'negotiation').
            limit: Maximum number of deals to return.

        Returns:
            Dict with deals list.
        """
        params: dict[str, Any] = {"limit": limit}
        if stage:
            params["stage"] = stage
        return self._client.request("GET", "/api/v1/crm/deals", params=params)

    def list_platform_deals(
        self, platform: str, limit: int = 20
    ) -> dict[str, Any]:
        """
        List deals for a specific CRM platform.

        Args:
            platform: Platform identifier.
            limit: Maximum number of deals to return.

        Returns:
            Dict with platform-specific deals.
        """
        params: dict[str, Any] = {"limit": limit}
        return self._client.request(
            "GET", f"/api/v1/crm/{platform}/deals", params=params
        )

    def get_deal(self, platform: str, deal_id: str) -> dict[str, Any]:
        """
        Get deal details from a specific platform.

        Args:
            platform: Platform identifier.
            deal_id: Deal identifier.

        Returns:
            Dict with deal details.
        """
        return self._client.request(
            "GET", f"/api/v1/crm/{platform}/deals/{deal_id}"
        )

    def create_deal(
        self, name: str, contact_id: str, value: float | None = None
    ) -> dict[str, Any]:
        """
        Create a deal (cross-platform).

        Args:
            name: Deal name.
            contact_id: Associated contact identifier.
            value: Deal value.

        Returns:
            Dict with created deal details.
        """
        data: dict[str, Any] = {"name": name, "contact_id": contact_id}
        if value is not None:
            data["value"] = value
        return self._client.request("POST", "/api/v1/crm/deals", json=data)

    # =========================================================================
    # Pipeline & Intelligence
    # =========================================================================

    def get_pipeline(self, **kwargs: Any) -> dict[str, Any]:
        """
        Get sales pipeline overview.

        Returns:
            Dict with pipeline stages, deal counts, and total values.
        """
        return self._client.request("GET", "/api/v1/crm/pipeline", params=kwargs or None)

    def enrich(self, **kwargs: Any) -> dict[str, Any]:
        """
        Enrich contact or company data with external sources.

        Args:
            **kwargs: Enrichment parameters including:
                - contact_id: Contact to enrich
                - company_id: Company to enrich
                - sources: Data sources to use

        Returns:
            Dict with enriched data from external sources.
        """
        return self._client.request("POST", "/api/v1/crm/enrich", json=kwargs)

    def sync_lead(self, **kwargs: Any) -> dict[str, Any]:
        """
        Sync a lead across connected CRM platforms.

        Args:
            **kwargs: Lead sync parameters including:
                - lead_data: Lead information to sync
                - target_platforms: Platforms to sync to

        Returns:
            Dict with sync results per platform.
        """
        return self._client.request("POST", "/api/v1/crm/sync-lead", json=kwargs)

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """
        Search across CRM data (contacts, companies, deals).

        Args:
            query: Search query string.
            **kwargs: Additional search filters.

        Returns:
            Dict with search results across all CRM entities.
        """
        params: dict[str, Any] = {"query": query, **kwargs}
        return self._client.request("GET", "/api/v1/crm/search", params=params)


class AsyncCRMAPI:
    """
    Asynchronous CRM API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     platforms = await client.crm.list_platforms()
        ...     contacts = await client.crm.list_contacts(search="Acme")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # =========================================================================
    # Platform Management
    # =========================================================================

    async def list_platforms(self) -> dict[str, Any]:
        """List connected CRM platforms."""
        return await self._client.request("GET", "/api/v1/crm/platforms")

    async def connect(self, **kwargs: Any) -> dict[str, Any]:
        """Connect a CRM platform."""
        return await self._client.request("POST", "/api/v1/crm/connect", json=kwargs)

    async def disconnect(self, platform: str) -> dict[str, Any]:
        """Disconnect a CRM platform."""
        return await self._client.request("DELETE", f"/api/v1/crm/{platform}")

    async def get_status(self) -> dict[str, Any]:
        """Get CRM integration status including circuit breaker state."""
        return await self._client.request("GET", "/api/v1/crm/status")

    # =========================================================================
    # Contacts
    # =========================================================================

    async def list_contacts(
        self, search: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        """List CRM contacts across all connected platforms."""
        params: dict[str, Any] = {"limit": limit}
        if search:
            params["search"] = search
        return await self._client.request("GET", "/api/v1/crm/contacts", params=params)

    async def list_platform_contacts(
        self, platform: str, limit: int = 20
    ) -> dict[str, Any]:
        """List contacts for a specific CRM platform."""
        params: dict[str, Any] = {"limit": limit}
        return await self._client.request(
            "GET", f"/api/v1/crm/{platform}/contacts", params=params
        )

    async def get_contact(self, platform: str, contact_id: str) -> dict[str, Any]:
        """Get contact details from a specific platform."""
        return await self._client.request(
            "GET", f"/api/v1/crm/{platform}/contacts/{contact_id}"
        )

    async def create_contact(
        self, email: str, name: str | None = None, **fields: Any
    ) -> dict[str, Any]:
        """Create a contact (cross-platform)."""
        data: dict[str, Any] = {"email": email, **fields}
        if name:
            data["name"] = name
        return await self._client.request("POST", "/api/v1/crm/contacts", json=data)

    # =========================================================================
    # Companies
    # =========================================================================

    async def list_companies(self, limit: int = 20) -> dict[str, Any]:
        """List companies across all connected platforms."""
        params: dict[str, Any] = {"limit": limit}
        return await self._client.request("GET", "/api/v1/crm/companies", params=params)

    async def list_platform_companies(
        self, platform: str, limit: int = 20
    ) -> dict[str, Any]:
        """List companies for a specific CRM platform."""
        params: dict[str, Any] = {"limit": limit}
        return await self._client.request(
            "GET", f"/api/v1/crm/{platform}/companies", params=params
        )

    async def get_company(self, platform: str, company_id: str) -> dict[str, Any]:
        """Get company details from a specific platform."""
        return await self._client.request(
            "GET", f"/api/v1/crm/{platform}/companies/{company_id}"
        )

    # =========================================================================
    # Deals
    # =========================================================================

    async def list_deals(
        self, stage: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        """List deals across all connected platforms."""
        params: dict[str, Any] = {"limit": limit}
        if stage:
            params["stage"] = stage
        return await self._client.request("GET", "/api/v1/crm/deals", params=params)

    async def list_platform_deals(
        self, platform: str, limit: int = 20
    ) -> dict[str, Any]:
        """List deals for a specific CRM platform."""
        params: dict[str, Any] = {"limit": limit}
        return await self._client.request(
            "GET", f"/api/v1/crm/{platform}/deals", params=params
        )

    async def get_deal(self, platform: str, deal_id: str) -> dict[str, Any]:
        """Get deal details from a specific platform."""
        return await self._client.request(
            "GET", f"/api/v1/crm/{platform}/deals/{deal_id}"
        )

    async def create_deal(
        self, name: str, contact_id: str, value: float | None = None
    ) -> dict[str, Any]:
        """Create a deal (cross-platform)."""
        data: dict[str, Any] = {"name": name, "contact_id": contact_id}
        if value is not None:
            data["value"] = value
        return await self._client.request("POST", "/api/v1/crm/deals", json=data)

    # =========================================================================
    # Pipeline & Intelligence
    # =========================================================================

    async def get_pipeline(self, **kwargs: Any) -> dict[str, Any]:
        """Get sales pipeline overview."""
        return await self._client.request(
            "GET", "/api/v1/crm/pipeline", params=kwargs or None
        )

    async def enrich(self, **kwargs: Any) -> dict[str, Any]:
        """Enrich contact or company data with external sources."""
        return await self._client.request("POST", "/api/v1/crm/enrich", json=kwargs)

    async def sync_lead(self, **kwargs: Any) -> dict[str, Any]:
        """Sync a lead across connected CRM platforms."""
        return await self._client.request("POST", "/api/v1/crm/sync-lead", json=kwargs)

    async def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Search across CRM data (contacts, companies, deals)."""
        params: dict[str, Any] = {"query": query, **kwargs}
        return await self._client.request("GET", "/api/v1/crm/search", params=params)
