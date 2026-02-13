"""
Audit Namespace API

Provides methods for audit trail management:
- List and query audit entries
- Session lifecycle management
- Finding management (assign, status, priority, comments)
- Compliance reporting and verification
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class AuditAPI:
    """
    Synchronous Audit API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> entries = client.audit.list_entries()
        >>> report = client.audit.get_report()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # =========================================================================
    # Entries, Report & Verification
    # =========================================================================

    def list_entries(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List audit entries."""
        return self._client.request(
            "GET", "/api/v1/audit/entries", params={"limit": limit, "offset": offset}
        )

    def get_report(self) -> dict[str, Any]:
        """Get audit report."""
        return self._client.request("GET", "/api/v1/audit/report")

    def verify(self) -> dict[str, Any]:
        """Verify audit integrity."""
        return self._client.request("GET", "/api/v1/audit/verify")

    # =========================================================================
    # Session Management
    # =========================================================================

    def list_sessions(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List audit sessions."""
        return self._client.request(
            "GET", "/api/v1/audit/sessions", params={"limit": limit, "offset": offset}
        )

    def create_session(self, name: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Create an audit session."""
        data: dict[str, Any] = {"name": name}
        if config:
            data["config"] = config
        return self._client.request("POST", "/api/v1/audit/sessions", json=data)


class AsyncAuditAPI:
    """
    Asynchronous Audit API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     entries = await client.audit.list_entries()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # =========================================================================
    # Entries, Report & Verification
    # =========================================================================

    async def list_entries(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List audit entries."""
        return await self._client.request(
            "GET", "/api/v1/audit/entries", params={"limit": limit, "offset": offset}
        )

    async def get_report(self) -> dict[str, Any]:
        """Get audit report."""
        return await self._client.request("GET", "/api/v1/audit/report")

    async def verify(self) -> dict[str, Any]:
        """Verify audit integrity."""
        return await self._client.request("GET", "/api/v1/audit/verify")

    # =========================================================================
    # Session Management
    # =========================================================================

    async def list_sessions(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List audit sessions."""
        return await self._client.request(
            "GET", "/api/v1/audit/sessions", params={"limit": limit, "offset": offset}
        )

    async def create_session(
        self, name: str, config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create an audit session."""
        data: dict[str, Any] = {"name": name}
        if config:
            data["config"] = config
        return await self._client.request("POST", "/api/v1/audit/sessions", json=data)
