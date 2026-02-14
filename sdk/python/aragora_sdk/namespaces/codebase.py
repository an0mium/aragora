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

    def analyze_repo(self, repo: str) -> dict[str, Any]:
        """"""
        return self._client.request("POST", f"/api/v1/codebase/{repo}/analyze")

    def audit_repo(self, repo: str) -> dict[str, Any]:
        """"""
        return self._client.request("POST", f"/api/v1/codebase/{repo}/audit")

    def get_audit_result(self, repo: str, audit_id: str) -> dict[str, Any]:
        """"""
        return self._client.request("GET", f"/api/v1/codebase/{repo}/audit/{audit_id}")

    def get_callgraph(self, repo: str) -> dict[str, Any]:
        """"""
        return self._client.request("GET", f"/api/v1/codebase/{repo}/callgraph")

    def get_deadcode(self, repo: str) -> dict[str, Any]:
        """"""
        return self._client.request("GET", f"/api/v1/codebase/{repo}/deadcode")

    def get_duplicates(self, repo: str) -> dict[str, Any]:
        """"""
        return self._client.request("GET", f"/api/v1/codebase/{repo}/duplicates")

    def get_hotspots(self, repo: str) -> dict[str, Any]:
        """"""
        return self._client.request("GET", f"/api/v1/codebase/{repo}/hotspots")

    def analyze_impact(self, repo: str) -> dict[str, Any]:
        """"""
        return self._client.request("POST", f"/api/v1/codebase/{repo}/impact")

    def get_metrics(self, repo: str) -> dict[str, Any]:
        """"""
        return self._client.request("GET", f"/api/v1/codebase/{repo}/metrics")

    def analyze_metrics(self, repo: str) -> dict[str, Any]:
        """"""
        return self._client.request("POST", f"/api/v1/codebase/{repo}/metrics/analyze")

    def get_file_metrics(self, repo: str, file_path: str) -> dict[str, Any]:
        """"""
        return self._client.request("GET", f"/api/v1/codebase/{repo}/metrics/file/{file_path}")

    def get_metrics_history(self, repo: str) -> dict[str, Any]:
        """"""
        return self._client.request("GET", f"/api/v1/codebase/{repo}/metrics/history")

    def get_metric_by_id(self, repo: str, metric_id: str) -> dict[str, Any]:
        """"""
        return self._client.request("GET", f"/api/v1/codebase/{repo}/metrics/{metric_id}")

    def get_sast_findings(self, repo: str) -> dict[str, Any]:
        """"""
        return self._client.request("GET", f"/api/v1/codebase/{repo}/sast/findings")

    def get_owasp_summary(self, repo: str) -> dict[str, Any]:
        """"""
        return self._client.request("GET", f"/api/v1/codebase/{repo}/sast/owasp-summary")

    def scan_repo(self, repo: str) -> dict[str, Any]:
        """"""
        return self._client.request("POST", f"/api/v1/codebase/{repo}/scan")

    def get_latest_scan(self, repo: str) -> dict[str, Any]:
        """"""
        return self._client.request("GET", f"/api/v1/codebase/{repo}/scan/latest")

    def scan_sast(self, repo: str) -> dict[str, Any]:
        """"""
        return self._client.request("POST", f"/api/v1/codebase/{repo}/scan/sast")

    def get_sast_scan(self, repo: str, scan_id: str) -> dict[str, Any]:
        """"""
        return self._client.request("GET", f"/api/v1/codebase/{repo}/scan/sast/{scan_id}")

    def scan_secrets(self, repo: str) -> dict[str, Any]:
        """"""
        return self._client.request("POST", f"/api/v1/codebase/{repo}/scan/secrets")

    def get_latest_secrets_scan(self, repo: str) -> dict[str, Any]:
        """"""
        return self._client.request("GET", f"/api/v1/codebase/{repo}/scan/secrets/latest")

    def get_secrets_scan(self, repo: str, scan_id: str) -> dict[str, Any]:
        """"""
        return self._client.request("GET", f"/api/v1/codebase/{repo}/scan/secrets/{scan_id}")

    def get_scan_result(self, repo: str, scan_id: str) -> dict[str, Any]:
        """"""
        return self._client.request("GET", f"/api/v1/codebase/{repo}/scan/{scan_id}")

    def list_scans(self, repo: str) -> dict[str, Any]:
        """"""
        return self._client.request("GET", f"/api/v1/codebase/{repo}/scans")

    def list_secrets_scans(self, repo: str) -> dict[str, Any]:
        """"""
        return self._client.request("GET", f"/api/v1/codebase/{repo}/scans/secrets")

    def get_secrets(self, repo: str) -> dict[str, Any]:
        """"""
        return self._client.request("GET", f"/api/v1/codebase/{repo}/secrets")

    def get_symbols(self, repo: str) -> dict[str, Any]:
        """"""
        return self._client.request("GET", f"/api/v1/codebase/{repo}/symbols")

    def understand_code(self, repo: str) -> dict[str, Any]:
        """"""
        return self._client.request("POST", f"/api/v1/codebase/{repo}/understand")

    def get_vulnerabilities(self, repo: str) -> dict[str, Any]:
        """"""
        return self._client.request("GET", f"/api/v1/codebase/{repo}/vulnerabilities")


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

    async def analyze_repo(self, repo: str) -> dict[str, Any]:
        """"""
        return await self._client.request("POST", f"/api/v1/codebase/{repo}/analyze")

    async def audit_repo(self, repo: str) -> dict[str, Any]:
        """"""
        return await self._client.request("POST", f"/api/v1/codebase/{repo}/audit")

    async def get_audit_result(self, repo: str, audit_id: str) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/audit/{audit_id}")

    async def get_callgraph(self, repo: str) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/callgraph")

    async def get_deadcode(self, repo: str) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/deadcode")

    async def get_duplicates(self, repo: str) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/duplicates")

    async def get_hotspots(self, repo: str) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/hotspots")

    async def analyze_impact(self, repo: str) -> dict[str, Any]:
        """"""
        return await self._client.request("POST", f"/api/v1/codebase/{repo}/impact")

    async def get_metrics(self, repo: str) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/metrics")

    async def analyze_metrics(self, repo: str) -> dict[str, Any]:
        """"""
        return await self._client.request("POST", f"/api/v1/codebase/{repo}/metrics/analyze")

    async def get_file_metrics(self, repo: str, file_path: str) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/metrics/file/{file_path}")

    async def get_metrics_history(self, repo: str) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/metrics/history")

    async def get_metric_by_id(self, repo: str, metric_id: str) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/metrics/{metric_id}")

    async def get_sast_findings(self, repo: str) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/sast/findings")

    async def get_owasp_summary(self, repo: str) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/sast/owasp-summary")

    async def scan_repo(self, repo: str) -> dict[str, Any]:
        """"""
        return await self._client.request("POST", f"/api/v1/codebase/{repo}/scan")

    async def get_latest_scan(self, repo: str) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/scan/latest")

    async def scan_sast(self, repo: str) -> dict[str, Any]:
        """"""
        return await self._client.request("POST", f"/api/v1/codebase/{repo}/scan/sast")

    async def get_sast_scan(self, repo: str, scan_id: str) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/scan/sast/{scan_id}")

    async def scan_secrets(self, repo: str) -> dict[str, Any]:
        """"""
        return await self._client.request("POST", f"/api/v1/codebase/{repo}/scan/secrets")

    async def get_latest_secrets_scan(self, repo: str) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/scan/secrets/latest")

    async def get_secrets_scan(self, repo: str, scan_id: str) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/scan/secrets/{scan_id}")

    async def get_scan_result(self, repo: str, scan_id: str) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/scan/{scan_id}")

    async def list_scans(self, repo: str) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/scans")

    async def list_secrets_scans(self, repo: str) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/scans/secrets")

    async def get_secrets(self, repo: str) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/secrets")

    async def get_symbols(self, repo: str) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/symbols")

    async def understand_code(self, repo: str) -> dict[str, Any]:
        """"""
        return await self._client.request("POST", f"/api/v1/codebase/{repo}/understand")

    async def get_vulnerabilities(self, repo: str) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/vulnerabilities")
