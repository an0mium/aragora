"""
n8n Integration Namespace API

Provides methods for n8n workflow automation integration:
- Credential management
- Node definitions
- Trigger/webhook management
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class N8nAPI:
    """Synchronous n8n Integration API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    # =========================================================================
    # Credentials
    # =========================================================================

    def list_credentials(self) -> dict[str, Any]:
        """List n8n credentials.

        Returns:
            Dict with credentials array.
        """
        return self._client.request("GET", "/api/v1/n8n/credentials")

    def create_credential(self, **kwargs: Any) -> dict[str, Any]:
        """Create a new n8n credential.

        Args:
            **kwargs: Credential configuration (name, type, data, etc.).

        Returns:
            Dict with created credential details.
        """
        return self._client.request("POST", "/api/v1/n8n/credentials", json=kwargs)

    def delete_credential(self, credential_id: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Delete an n8n credential.

        Args:
            credential_id: Optional credential ID (can also pass in kwargs).
            **kwargs: Additional parameters.

        Returns:
            Dict with deletion confirmation.
        """
        data: dict[str, Any] = {**kwargs}
        if credential_id:
            data["credential_id"] = credential_id
        return self._client.request("DELETE", "/api/v1/n8n/credentials", json=data)

    # =========================================================================
    # Node
    # =========================================================================

    def get_node(self) -> dict[str, Any]:
        """Get n8n node definition.

        Returns:
            Dict with node definition for the Aragora n8n node.
        """
        return self._client.request("GET", "/api/v1/n8n/node")

    def create_node(self, **kwargs: Any) -> dict[str, Any]:
        """Register a custom n8n node.

        Args:
            **kwargs: Node configuration.

        Returns:
            Dict with created node details.
        """
        return self._client.request("POST", "/api/v1/n8n/node", json=kwargs)

    def delete_node(self, **kwargs: Any) -> dict[str, Any]:
        """Delete a custom n8n node.

        Args:
            **kwargs: Node identification parameters.

        Returns:
            Dict with deletion confirmation.
        """
        return self._client.request("DELETE", "/api/v1/n8n/node", json=kwargs)

    # =========================================================================
    # Trigger
    # =========================================================================

    def get_trigger(self) -> dict[str, Any]:
        """Get n8n trigger definitions.

        Returns:
            Dict with available trigger types.
        """
        return self._client.request("GET", "/api/v1/n8n/trigger")

    def create_trigger(self, **kwargs: Any) -> dict[str, Any]:
        """Create an n8n trigger subscription.

        Args:
            **kwargs: Trigger configuration (events, webhook_url, etc.).

        Returns:
            Dict with created trigger details.
        """
        return self._client.request("POST", "/api/v1/n8n/trigger", json=kwargs)

    def delete_trigger(self, **kwargs: Any) -> dict[str, Any]:
        """Delete an n8n trigger subscription.

        Args:
            **kwargs: Trigger identification parameters.

        Returns:
            Dict with deletion confirmation.
        """
        return self._client.request("DELETE", "/api/v1/n8n/trigger", json=kwargs)


class AsyncN8nAPI:
    """Asynchronous n8n Integration API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_credentials(self) -> dict[str, Any]:
        """List n8n credentials."""
        return await self._client.request("GET", "/api/v1/n8n/credentials")

    async def create_credential(self, **kwargs: Any) -> dict[str, Any]:
        """Create a new n8n credential."""
        return await self._client.request("POST", "/api/v1/n8n/credentials", json=kwargs)

    async def delete_credential(
        self, credential_id: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Delete an n8n credential."""
        data: dict[str, Any] = {**kwargs}
        if credential_id:
            data["credential_id"] = credential_id
        return await self._client.request("DELETE", "/api/v1/n8n/credentials", json=data)

    async def get_node(self) -> dict[str, Any]:
        """Get n8n node definition."""
        return await self._client.request("GET", "/api/v1/n8n/node")

    async def create_node(self, **kwargs: Any) -> dict[str, Any]:
        """Register a custom n8n node."""
        return await self._client.request("POST", "/api/v1/n8n/node", json=kwargs)

    async def delete_node(self, **kwargs: Any) -> dict[str, Any]:
        """Delete a custom n8n node."""
        return await self._client.request("DELETE", "/api/v1/n8n/node", json=kwargs)

    async def get_trigger(self) -> dict[str, Any]:
        """Get n8n trigger definitions."""
        return await self._client.request("GET", "/api/v1/n8n/trigger")

    async def create_trigger(self, **kwargs: Any) -> dict[str, Any]:
        """Create an n8n trigger subscription."""
        return await self._client.request("POST", "/api/v1/n8n/trigger", json=kwargs)

    async def delete_trigger(self, **kwargs: Any) -> dict[str, Any]:
        """Delete an n8n trigger subscription."""
        return await self._client.request("DELETE", "/api/v1/n8n/trigger", json=kwargs)
