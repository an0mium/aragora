"""
Codebase Analysis Namespace API

Provides methods for codebase analysis, security scanning, and code intelligence:
- Dependency analysis and license checking
- SBOM generation
- Vulnerability scanning
- Cache management
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class CodebaseAPI:
    """Synchronous Codebase Analysis API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    # =========================================================================
    # Dependencies & SBOM
    # =========================================================================

    def analyze_dependencies(
        self, manifest_content: str, manifest_type: str = "auto"
    ) -> dict[str, Any]:
        """
        Analyze dependencies from manifest.

        Args:
            manifest_content: Content of package manifest
            manifest_type: Type of manifest (auto, npm, pip, cargo, etc.)

        Returns:
            Dependency analysis
        """
        return self._client.request(
            "POST",
            "/api/v1/codebase/analyze-dependencies",
            json={"manifest": manifest_content, "type": manifest_type},
        )

    def generate_sbom(self, repo: str, format: str = "spdx") -> dict[str, Any]:
        """
        Generate Software Bill of Materials.

        Args:
            repo: Repository identifier
            format: SBOM format (spdx, cyclonedx)

        Returns:
            SBOM document
        """
        return self._client.request(
            "POST", "/api/v1/codebase/sbom", json={"repo": repo, "format": format}
        )

    def check_licenses(self, repo: str) -> dict[str, Any]:
        """
        Check dependency licenses.

        Args:
            repo: Repository identifier

        Returns:
            License compliance results
        """
        return self._client.request("POST", "/api/v1/codebase/check-licenses", json={"repo": repo})

    def scan_vulnerabilities(self, repo: str) -> dict[str, Any]:
        """
        Scan for known vulnerabilities.

        Args:
            repo: Repository identifier

        Returns:
            Vulnerability findings
        """
        return self._client.request(
            "POST", "/api/v1/codebase/scan-vulnerabilities", json={"repo": repo}
        )

    def clear_cache(self, repo: str | None = None) -> dict[str, Any]:
        """
        Clear analysis cache.

        Args:
            repo: Optional repository to clear (all if not specified)

        Returns:
            Cache clear confirmation
        """
        data: dict[str, Any] = {}
        if repo:
            data["repo"] = repo
        return self._client.request("POST", "/api/v1/codebase/clear-cache", json=data)


class AsyncCodebaseAPI:
    """Asynchronous Codebase Analysis API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def analyze_dependencies(
        self, manifest_content: str, manifest_type: str = "auto"
    ) -> dict[str, Any]:
        """Analyze dependencies from manifest."""
        return await self._client.request(
            "POST",
            "/api/v1/codebase/analyze-dependencies",
            json={"manifest": manifest_content, "type": manifest_type},
        )

    async def generate_sbom(self, repo: str, format: str = "spdx") -> dict[str, Any]:
        """Generate Software Bill of Materials."""
        return await self._client.request(
            "POST", "/api/v1/codebase/sbom", json={"repo": repo, "format": format}
        )

    async def check_licenses(self, repo: str) -> dict[str, Any]:
        """Check dependency licenses."""
        return await self._client.request(
            "POST", "/api/v1/codebase/check-licenses", json={"repo": repo}
        )

    async def scan_vulnerabilities(self, repo: str) -> dict[str, Any]:
        """Scan for known vulnerabilities."""
        return await self._client.request(
            "POST", "/api/v1/codebase/scan-vulnerabilities", json={"repo": repo}
        )

    async def clear_cache(self, repo: str | None = None) -> dict[str, Any]:
        """Clear analysis cache."""
        data: dict[str, Any] = {}
        if repo:
            data["repo"] = repo
        return await self._client.request("POST", "/api/v1/codebase/clear-cache", json=data)
