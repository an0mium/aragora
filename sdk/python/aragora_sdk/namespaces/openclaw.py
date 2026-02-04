"""
OpenClaw Namespace API

Provides methods for the OpenClaw legal gateway:
- Case research
- Legal document analysis
- Citation verification
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class OpenclawAPI:
    """Synchronous OpenClaw API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def search_cases(
        self, query: str, jurisdiction: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        """Search legal cases."""
        params: dict[str, Any] = {"query": query, "limit": limit}
        if jurisdiction:
            params["jurisdiction"] = jurisdiction
        return self._client.request("GET", "/api/v1/openclaw/cases/search", params=params)

    def get_case(self, case_id: str) -> dict[str, Any]:
        """Get case by ID."""
        return self._client.request("GET", f"/api/v1/openclaw/cases/{case_id}")

    def get_case_citations(self, case_id: str) -> dict[str, Any]:
        """Get citations for a case."""
        return self._client.request("GET", f"/api/v1/openclaw/cases/{case_id}/citations")

    def analyze_document(self, content: str, analysis_type: str = "legal") -> dict[str, Any]:
        """Analyze a legal document."""
        return self._client.request(
            "POST",
            "/api/v1/openclaw/analyze",
            json={
                "content": content,
                "analysis_type": analysis_type,
            },
        )

    def verify_citation(self, citation: str) -> dict[str, Any]:
        """Verify a legal citation."""
        return self._client.request(
            "POST", "/api/v1/openclaw/citations/verify", json={"citation": citation}
        )

    def list_jurisdictions(self) -> dict[str, Any]:
        """List available jurisdictions."""
        return self._client.request("GET", "/api/v1/openclaw/jurisdictions")

    def get_statute(self, statute_id: str) -> dict[str, Any]:
        """Get statute by ID."""
        return self._client.request("GET", f"/api/v1/openclaw/statutes/{statute_id}")

    def search_statutes(self, query: str, jurisdiction: str | None = None) -> dict[str, Any]:
        """Search statutes."""
        params: dict[str, Any] = {"query": query}
        if jurisdiction:
            params["jurisdiction"] = jurisdiction
        return self._client.request("GET", "/api/v1/openclaw/statutes/search", params=params)


class AsyncOpenclawAPI:
    """Asynchronous OpenClaw API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def search_cases(
        self, query: str, jurisdiction: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        """Search legal cases."""
        params: dict[str, Any] = {"query": query, "limit": limit}
        if jurisdiction:
            params["jurisdiction"] = jurisdiction
        return await self._client.request("GET", "/api/v1/openclaw/cases/search", params=params)

    async def get_case(self, case_id: str) -> dict[str, Any]:
        """Get case by ID."""
        return await self._client.request("GET", f"/api/v1/openclaw/cases/{case_id}")

    async def get_case_citations(self, case_id: str) -> dict[str, Any]:
        """Get citations for a case."""
        return await self._client.request("GET", f"/api/v1/openclaw/cases/{case_id}/citations")

    async def analyze_document(self, content: str, analysis_type: str = "legal") -> dict[str, Any]:
        """Analyze a legal document."""
        return await self._client.request(
            "POST",
            "/api/v1/openclaw/analyze",
            json={
                "content": content,
                "analysis_type": analysis_type,
            },
        )

    async def verify_citation(self, citation: str) -> dict[str, Any]:
        """Verify a legal citation."""
        return await self._client.request(
            "POST", "/api/v1/openclaw/citations/verify", json={"citation": citation}
        )

    async def list_jurisdictions(self) -> dict[str, Any]:
        """List available jurisdictions."""
        return await self._client.request("GET", "/api/v1/openclaw/jurisdictions")

    async def get_statute(self, statute_id: str) -> dict[str, Any]:
        """Get statute by ID."""
        return await self._client.request("GET", f"/api/v1/openclaw/statutes/{statute_id}")

    async def search_statutes(self, query: str, jurisdiction: str | None = None) -> dict[str, Any]:
        """Search statutes."""
        params: dict[str, Any] = {"query": query}
        if jurisdiction:
            params["jurisdiction"] = jurisdiction
        return await self._client.request("GET", "/api/v1/openclaw/statutes/search", params=params)
