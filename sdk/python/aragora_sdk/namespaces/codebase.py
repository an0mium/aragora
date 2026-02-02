"""
Codebase Analysis Namespace API

Provides methods for codebase analysis, security scanning, and code intelligence:
- Static analysis and code metrics
- Security scanning (SAST, secrets detection)
- SBOM and dependency analysis
- Code understanding and impact analysis
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
    # Analysis
    # =========================================================================

    def analyze(self, repo: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Analyze a repository.

        Args:
            repo: Repository identifier
            options: Analysis options

        Returns:
            Analysis results
        """
        return self._client.request("POST", f"/api/v1/codebase/{repo}/analyze", json=options or {})

    def understand(self, repo: str, query: str) -> dict[str, Any]:
        """
        Query codebase understanding.

        Args:
            repo: Repository identifier
            query: Natural language query about the code

        Returns:
            Understanding results
        """
        return self._client.request(
            "POST", f"/api/v1/codebase/{repo}/understand", json={"query": query}
        )

    def get_symbols(self, repo: str, file_path: str | None = None) -> dict[str, Any]:
        """
        Get code symbols from repository.

        Args:
            repo: Repository identifier
            file_path: Optional file to filter symbols

        Returns:
            Symbol information
        """
        params: dict[str, Any] = {}
        if file_path:
            params["file_path"] = file_path
        return self._client.request("GET", f"/api/v1/codebase/{repo}/symbols", params=params)

    def get_callgraph(self, repo: str, function: str | None = None) -> dict[str, Any]:
        """
        Get call graph for repository.

        Args:
            repo: Repository identifier
            function: Optional function to center graph on

        Returns:
            Call graph data
        """
        params: dict[str, Any] = {}
        if function:
            params["function"] = function
        return self._client.request("GET", f"/api/v1/codebase/{repo}/callgraph", params=params)

    def analyze_impact(self, repo: str, files: list[str]) -> dict[str, Any]:
        """
        Analyze impact of file changes.

        Args:
            repo: Repository identifier
            files: List of changed files

        Returns:
            Impact analysis results
        """
        return self._client.request(
            "POST", f"/api/v1/codebase/{repo}/impact", json={"files": files}
        )

    # =========================================================================
    # Code Quality
    # =========================================================================

    def get_deadcode(self, repo: str) -> dict[str, Any]:
        """
        Find dead (unreachable) code.

        Args:
            repo: Repository identifier

        Returns:
            Dead code findings
        """
        return self._client.request("GET", f"/api/v1/codebase/{repo}/deadcode")

    def get_duplicates(self, repo: str, min_lines: int = 10) -> dict[str, Any]:
        """
        Find code duplicates.

        Args:
            repo: Repository identifier
            min_lines: Minimum lines for duplicate detection

        Returns:
            Duplicate code findings
        """
        return self._client.request(
            "GET", f"/api/v1/codebase/{repo}/duplicates", params={"min_lines": min_lines}
        )

    def get_hotspots(self, repo: str) -> dict[str, Any]:
        """
        Find code hotspots (frequently changed files).

        Args:
            repo: Repository identifier

        Returns:
            Hotspot analysis
        """
        return self._client.request("GET", f"/api/v1/codebase/{repo}/hotspots")

    # =========================================================================
    # Metrics
    # =========================================================================

    def get_metrics(self, repo: str) -> dict[str, Any]:
        """
        Get code metrics for repository.

        Args:
            repo: Repository identifier

        Returns:
            Code metrics (complexity, coverage, etc.)
        """
        return self._client.request("GET", f"/api/v1/codebase/{repo}/metrics")

    def analyze_metrics(self, repo: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Run metrics analysis.

        Args:
            repo: Repository identifier
            options: Analysis options

        Returns:
            Metrics analysis results
        """
        return self._client.request(
            "POST", f"/api/v1/codebase/{repo}/metrics/analyze", json=options or {}
        )

    def get_metrics_analysis(self, repo: str, analysis_id: str) -> dict[str, Any]:
        """
        Get metrics analysis results.

        Args:
            repo: Repository identifier
            analysis_id: Analysis identifier

        Returns:
            Analysis results
        """
        return self._client.request("GET", f"/api/v1/codebase/{repo}/metrics/{analysis_id}")

    def get_file_metrics(self, repo: str, file_path: str) -> dict[str, Any]:
        """
        Get metrics for a specific file.

        Args:
            repo: Repository identifier
            file_path: File path

        Returns:
            File metrics
        """
        return self._client.request("GET", f"/api/v1/codebase/{repo}/metrics/file/{file_path}")

    def get_metrics_history(self, repo: str, days: int = 30) -> dict[str, Any]:
        """
        Get metrics history.

        Args:
            repo: Repository identifier
            days: Number of days of history

        Returns:
            Metrics over time
        """
        return self._client.request(
            "GET", f"/api/v1/codebase/{repo}/metrics/history", params={"days": days}
        )

    # =========================================================================
    # Security Scanning
    # =========================================================================

    def scan(self, repo: str, scan_type: str = "full") -> dict[str, Any]:
        """
        Run a security scan.

        Args:
            repo: Repository identifier
            scan_type: Type of scan (full, quick, sast, secrets)

        Returns:
            Scan initiation result
        """
        return self._client.request(
            "POST", f"/api/v1/codebase/{repo}/scan", json={"scan_type": scan_type}
        )

    def get_scan(self, repo: str, scan_id: str) -> dict[str, Any]:
        """
        Get scan results.

        Args:
            repo: Repository identifier
            scan_id: Scan identifier

        Returns:
            Scan results
        """
        return self._client.request("GET", f"/api/v1/codebase/{repo}/scan/{scan_id}")

    def get_latest_scan(self, repo: str) -> dict[str, Any]:
        """
        Get latest scan results.

        Args:
            repo: Repository identifier

        Returns:
            Latest scan results
        """
        return self._client.request("GET", f"/api/v1/codebase/{repo}/scan/latest")

    def list_scans(self, repo: str, limit: int = 20) -> dict[str, Any]:
        """
        List all scans for repository.

        Args:
            repo: Repository identifier
            limit: Maximum scans to return

        Returns:
            List of scans
        """
        return self._client.request(
            "GET", f"/api/v1/codebase/{repo}/scans", params={"limit": limit}
        )

    # =========================================================================
    # SAST (Static Application Security Testing)
    # =========================================================================

    def scan_sast(self, repo: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Run SAST scan.

        Args:
            repo: Repository identifier
            options: SAST options

        Returns:
            SAST scan results
        """
        return self._client.request(
            "POST", f"/api/v1/codebase/{repo}/scan/sast", json=options or {}
        )

    def get_sast_scan(self, repo: str, scan_id: str) -> dict[str, Any]:
        """
        Get SAST scan results.

        Args:
            repo: Repository identifier
            scan_id: Scan identifier

        Returns:
            SAST findings
        """
        return self._client.request("GET", f"/api/v1/codebase/{repo}/scan/sast/{scan_id}")

    def get_sast_findings(self, repo: str, severity: str | None = None) -> dict[str, Any]:
        """
        Get SAST findings.

        Args:
            repo: Repository identifier
            severity: Filter by severity (critical, high, medium, low)

        Returns:
            SAST findings
        """
        params: dict[str, Any] = {}
        if severity:
            params["severity"] = severity
        return self._client.request("GET", f"/api/v1/codebase/{repo}/sast/findings", params=params)

    def get_owasp_summary(self, repo: str) -> dict[str, Any]:
        """
        Get OWASP Top 10 summary.

        Args:
            repo: Repository identifier

        Returns:
            OWASP findings summary
        """
        return self._client.request("GET", f"/api/v1/codebase/{repo}/sast/owasp-summary")

    # =========================================================================
    # Secrets Detection
    # =========================================================================

    def scan_secrets(self, repo: str) -> dict[str, Any]:
        """
        Scan for secrets in code.

        Args:
            repo: Repository identifier

        Returns:
            Secrets scan results
        """
        return self._client.request("POST", f"/api/v1/codebase/{repo}/scan/secrets")

    def get_secrets_scan(self, repo: str, scan_id: str) -> dict[str, Any]:
        """
        Get secrets scan results.

        Args:
            repo: Repository identifier
            scan_id: Scan identifier

        Returns:
            Secrets findings
        """
        return self._client.request("GET", f"/api/v1/codebase/{repo}/scan/secrets/{scan_id}")

    def get_latest_secrets_scan(self, repo: str) -> dict[str, Any]:
        """
        Get latest secrets scan.

        Args:
            repo: Repository identifier

        Returns:
            Latest secrets findings
        """
        return self._client.request("GET", f"/api/v1/codebase/{repo}/scan/secrets/latest")

    def list_secrets_scans(self, repo: str) -> dict[str, Any]:
        """
        List secrets scans.

        Args:
            repo: Repository identifier

        Returns:
            List of secrets scans
        """
        return self._client.request("GET", f"/api/v1/codebase/{repo}/scans/secrets")

    def get_secrets(self, repo: str) -> dict[str, Any]:
        """
        Get all detected secrets.

        Args:
            repo: Repository identifier

        Returns:
            All secrets findings
        """
        return self._client.request("GET", f"/api/v1/codebase/{repo}/secrets")

    # =========================================================================
    # Audits
    # =========================================================================

    def audit(self, repo: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Run a code audit.

        Args:
            repo: Repository identifier
            options: Audit options

        Returns:
            Audit results
        """
        return self._client.request("POST", f"/api/v1/codebase/{repo}/audit", json=options or {})

    def get_audit(self, repo: str, audit_id: str) -> dict[str, Any]:
        """
        Get audit results.

        Args:
            repo: Repository identifier
            audit_id: Audit identifier

        Returns:
            Audit results
        """
        return self._client.request("GET", f"/api/v1/codebase/{repo}/audit/{audit_id}")

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

    async def analyze(self, repo: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Analyze a repository."""
        return await self._client.request(
            "POST", f"/api/v1/codebase/{repo}/analyze", json=options or {}
        )

    async def understand(self, repo: str, query: str) -> dict[str, Any]:
        """Query codebase understanding."""
        return await self._client.request(
            "POST", f"/api/v1/codebase/{repo}/understand", json={"query": query}
        )

    async def get_symbols(self, repo: str, file_path: str | None = None) -> dict[str, Any]:
        """Get code symbols from repository."""
        params: dict[str, Any] = {}
        if file_path:
            params["file_path"] = file_path
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/symbols", params=params)

    async def get_callgraph(self, repo: str, function: str | None = None) -> dict[str, Any]:
        """Get call graph for repository."""
        params: dict[str, Any] = {}
        if function:
            params["function"] = function
        return await self._client.request(
            "GET", f"/api/v1/codebase/{repo}/callgraph", params=params
        )

    async def analyze_impact(self, repo: str, files: list[str]) -> dict[str, Any]:
        """Analyze impact of file changes."""
        return await self._client.request(
            "POST", f"/api/v1/codebase/{repo}/impact", json={"files": files}
        )

    async def get_deadcode(self, repo: str) -> dict[str, Any]:
        """Find dead (unreachable) code."""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/deadcode")

    async def get_duplicates(self, repo: str, min_lines: int = 10) -> dict[str, Any]:
        """Find code duplicates."""
        return await self._client.request(
            "GET", f"/api/v1/codebase/{repo}/duplicates", params={"min_lines": min_lines}
        )

    async def get_hotspots(self, repo: str) -> dict[str, Any]:
        """Find code hotspots."""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/hotspots")

    async def get_metrics(self, repo: str) -> dict[str, Any]:
        """Get code metrics for repository."""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/metrics")

    async def analyze_metrics(
        self, repo: str, options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Run metrics analysis."""
        return await self._client.request(
            "POST", f"/api/v1/codebase/{repo}/metrics/analyze", json=options or {}
        )

    async def get_metrics_analysis(self, repo: str, analysis_id: str) -> dict[str, Any]:
        """Get metrics analysis results."""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/metrics/{analysis_id}")

    async def get_file_metrics(self, repo: str, file_path: str) -> dict[str, Any]:
        """Get metrics for a specific file."""
        return await self._client.request(
            "GET", f"/api/v1/codebase/{repo}/metrics/file/{file_path}"
        )

    async def get_metrics_history(self, repo: str, days: int = 30) -> dict[str, Any]:
        """Get metrics history."""
        return await self._client.request(
            "GET", f"/api/v1/codebase/{repo}/metrics/history", params={"days": days}
        )

    async def scan(self, repo: str, scan_type: str = "full") -> dict[str, Any]:
        """Run a security scan."""
        return await self._client.request(
            "POST", f"/api/v1/codebase/{repo}/scan", json={"scan_type": scan_type}
        )

    async def get_scan(self, repo: str, scan_id: str) -> dict[str, Any]:
        """Get scan results."""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/scan/{scan_id}")

    async def get_latest_scan(self, repo: str) -> dict[str, Any]:
        """Get latest scan results."""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/scan/latest")

    async def list_scans(self, repo: str, limit: int = 20) -> dict[str, Any]:
        """List all scans for repository."""
        return await self._client.request(
            "GET", f"/api/v1/codebase/{repo}/scans", params={"limit": limit}
        )

    async def scan_sast(self, repo: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run SAST scan."""
        return await self._client.request(
            "POST", f"/api/v1/codebase/{repo}/scan/sast", json=options or {}
        )

    async def get_sast_scan(self, repo: str, scan_id: str) -> dict[str, Any]:
        """Get SAST scan results."""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/scan/sast/{scan_id}")

    async def get_sast_findings(self, repo: str, severity: str | None = None) -> dict[str, Any]:
        """Get SAST findings."""
        params: dict[str, Any] = {}
        if severity:
            params["severity"] = severity
        return await self._client.request(
            "GET", f"/api/v1/codebase/{repo}/sast/findings", params=params
        )

    async def get_owasp_summary(self, repo: str) -> dict[str, Any]:
        """Get OWASP Top 10 summary."""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/sast/owasp-summary")

    async def scan_secrets(self, repo: str) -> dict[str, Any]:
        """Scan for secrets in code."""
        return await self._client.request("POST", f"/api/v1/codebase/{repo}/scan/secrets")

    async def get_secrets_scan(self, repo: str, scan_id: str) -> dict[str, Any]:
        """Get secrets scan results."""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/scan/secrets/{scan_id}")

    async def get_latest_secrets_scan(self, repo: str) -> dict[str, Any]:
        """Get latest secrets scan."""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/scan/secrets/latest")

    async def list_secrets_scans(self, repo: str) -> dict[str, Any]:
        """List secrets scans."""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/scans/secrets")

    async def get_secrets(self, repo: str) -> dict[str, Any]:
        """Get all detected secrets."""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/secrets")

    async def audit(self, repo: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run a code audit."""
        return await self._client.request(
            "POST", f"/api/v1/codebase/{repo}/audit", json=options or {}
        )

    async def get_audit(self, repo: str, audit_id: str) -> dict[str, Any]:
        """Get audit results."""
        return await self._client.request("GET", f"/api/v1/codebase/{repo}/audit/{audit_id}")

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
