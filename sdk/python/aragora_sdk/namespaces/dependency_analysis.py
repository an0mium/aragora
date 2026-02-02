"""
Dependency Analysis Namespace API

Provides a namespaced interface for codebase dependency analysis operations:
- Analyzing project dependencies
- Generating SBOM (Software Bill of Materials)
- Scanning for vulnerabilities (CVEs)
- Checking license compliance
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class DependencyAnalysisAPI:
    """
    Synchronous Dependency Analysis API.

    Provides methods for codebase dependency analysis:
    - Analyzing project dependencies
    - Generating SBOM (Software Bill of Materials)
    - Scanning for vulnerabilities (CVEs)
    - Checking license compliance

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> deps = client.dependency_analysis.analyze(
        ...     repository_url="https://github.com/org/repo",
        ...     include_transitive=True,
        ... )
        >>> sbom = client.dependency_analysis.generate_sbom(
        ...     repository_url="https://github.com/org/repo",
        ...     format="cyclonedx",
        ... )
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Dependency Analysis
    # ===========================================================================

    def analyze(
        self,
        repository_url: str | None = None,
        local_path: str | None = None,
        package_managers: list[Literal["npm", "pip", "cargo", "go", "maven", "gradle"]]
        | None = None,
        include_dev: bool = False,
        include_transitive: bool = True,
        max_depth: int | None = None,
    ) -> dict[str, Any]:
        """
        Analyze project dependencies.

        Args:
            repository_url: URL of the repository to analyze
            local_path: Local filesystem path to analyze
            package_managers: List of package managers to detect
                (npm, pip, cargo, go, maven, gradle)
            include_dev: Whether to include development dependencies
            include_transitive: Whether to include transitive dependencies
            max_depth: Maximum depth for transitive dependency resolution

        Returns:
            Dict with dependency analysis results including:
            - total_dependencies: Total number of dependencies
            - direct_dependencies: Number of direct dependencies
            - transitive_dependencies: Number of transitive dependencies
            - dev_dependencies: Number of dev dependencies
            - dependencies: List of dependency details
            - dependency_tree: Dependency hierarchy
            - analysis_time_ms: Time taken for analysis
        """
        data: dict[str, Any] = {
            "include_dev_dependencies": include_dev,
            "include_transitive": include_transitive,
        }
        if repository_url is not None:
            data["repository_url"] = repository_url
        if local_path is not None:
            data["local_path"] = local_path
        if package_managers is not None:
            data["package_managers"] = package_managers
        if max_depth is not None:
            data["max_depth"] = max_depth
        return self._client.request("POST", "/api/v1/codebase/analyze-dependencies", json=data)

    # ===========================================================================
    # SBOM Generation
    # ===========================================================================

    def generate_sbom(
        self,
        repository_url: str | None = None,
        local_path: str | None = None,
        format: Literal["spdx", "cyclonedx", "json"] = "cyclonedx",
        include_checksums: bool = False,
        include_licenses: bool = True,
    ) -> dict[str, Any]:
        """
        Generate Software Bill of Materials (SBOM).

        Args:
            repository_url: URL of the repository to analyze
            local_path: Local filesystem path to analyze
            format: SBOM output format (spdx, cyclonedx, json)
            include_checksums: Whether to include file checksums
            include_licenses: Whether to include license information

        Returns:
            Dict with SBOM data including:
            - format: SBOM format used
            - version: SBOM specification version
            - created_at: Generation timestamp
            - tool: Tool used for generation
            - components: List of software components
            - relationships: Dependency relationships
        """
        data: dict[str, Any] = {
            "format": format,
            "include_checksums": include_checksums,
            "include_licenses": include_licenses,
        }
        if repository_url is not None:
            data["repository_url"] = repository_url
        if local_path is not None:
            data["local_path"] = local_path
        return self._client.request("POST", "/api/v1/codebase/sbom", json=data)

    # ===========================================================================
    # Vulnerability Scanning
    # ===========================================================================

    def scan_vulnerabilities(
        self,
        repository_url: str | None = None,
        local_path: str | None = None,
        sbom: dict[str, Any] | None = None,
        severity_threshold: Literal["low", "medium", "high", "critical"] = "low",
        ignore_cves: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Scan for known vulnerabilities (CVEs).

        Args:
            repository_url: URL of the repository to scan
            local_path: Local filesystem path to scan
            sbom: Pre-generated SBOM to use for scanning
            severity_threshold: Minimum severity to report
            ignore_cves: List of CVE IDs to ignore

        Returns:
            Dict with vulnerability scan results including:
            - total_vulnerabilities: Total vulnerabilities found
            - critical: Number of critical vulnerabilities
            - high: Number of high vulnerabilities
            - medium: Number of medium vulnerabilities
            - low: Number of low vulnerabilities
            - vulnerabilities: List of vulnerability details
            - scan_time_ms: Time taken for scan
            - databases_checked: Security databases consulted
        """
        data: dict[str, Any] = {
            "severity_threshold": severity_threshold,
        }
        if repository_url is not None:
            data["repository_url"] = repository_url
        if local_path is not None:
            data["local_path"] = local_path
        if sbom is not None:
            data["sbom"] = sbom
        if ignore_cves is not None:
            data["ignore_cves"] = ignore_cves
        return self._client.request("POST", "/api/v1/codebase/scan-vulnerabilities", json=data)

    # ===========================================================================
    # License Compliance
    # ===========================================================================

    def check_licenses(
        self,
        repository_url: str | None = None,
        local_path: str | None = None,
        allowed_licenses: list[str] | None = None,
        denied_licenses: list[str] | None = None,
        policy: Literal["permissive", "copyleft", "commercial", "custom"] = "permissive",
    ) -> dict[str, Any]:
        """
        Check license compliance.

        Args:
            repository_url: URL of the repository to check
            local_path: Local filesystem path to check
            allowed_licenses: List of explicitly allowed licenses
            denied_licenses: List of explicitly denied licenses
            policy: License policy preset to apply

        Returns:
            Dict with license check results including:
            - compliant: Whether the project is compliant
            - total_packages: Total packages checked
            - compatible: Number of compatible licenses
            - incompatible: Number of incompatible licenses
            - unknown: Number of unknown licenses
            - licenses: List of license details per package
            - policy_used: Policy that was applied
            - issues: List of compliance issues
        """
        data: dict[str, Any] = {
            "policy": policy,
        }
        if repository_url is not None:
            data["repository_url"] = repository_url
        if local_path is not None:
            data["local_path"] = local_path
        if allowed_licenses is not None:
            data["allowed_licenses"] = allowed_licenses
        if denied_licenses is not None:
            data["denied_licenses"] = denied_licenses
        return self._client.request("POST", "/api/v1/codebase/check-licenses", json=data)

    # ===========================================================================
    # Cache Management
    # ===========================================================================

    def clear_cache(
        self,
        repository_url: str | None = None,
    ) -> dict[str, Any]:
        """
        Clear analysis cache.

        Args:
            repository_url: URL of the repository to clear cache for.
                If not provided, clears all cached analysis data.

        Returns:
            Dict with cache clearing status:
            - cleared: Whether cache was successfully cleared
        """
        data: dict[str, Any] = {}
        if repository_url is not None:
            data["repository_url"] = repository_url
        return self._client.request("POST", "/api/v1/codebase/clear-cache", json=data)

    # ===========================================================================
    # Convenience Methods
    # ===========================================================================

    def full_audit(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run a full security audit (dependencies + vulnerabilities + licenses).

        Convenience method that combines multiple checks into a single audit.

        Args:
            request: Audit request containing:
                - repository_url: URL of the repository to audit
                - local_path: Local filesystem path to audit
                - package_managers: List of package managers to detect
                - include_dev_dependencies: Include dev dependencies
                - include_transitive: Include transitive dependencies
                - max_depth: Maximum dependency resolution depth
                - license_policy: License policy to apply (permissive, copyleft, commercial)

        Returns:
            Dict with full audit results including:
            - dependencies: Dependency analysis results
            - sbom: Generated SBOM
            - vulnerabilities: Vulnerability scan results
            - licenses: License compliance results
            - summary: Overall audit summary with:
                - total_dependencies: Total number of dependencies
                - total_vulnerabilities: Total vulnerabilities found
                - license_compliant: Whether licenses are compliant
                - risk_level: Overall risk level (low, medium, high, critical)
        """
        # Run dependency analysis first
        dependencies = self.analyze(
            repository_url=request.get("repository_url"),
            local_path=request.get("local_path"),
            package_managers=request.get("package_managers"),
            include_dev=request.get("include_dev_dependencies", False),
            include_transitive=request.get("include_transitive", True),
            max_depth=request.get("max_depth"),
        )

        # Generate SBOM
        sbom = self.generate_sbom(
            repository_url=request.get("repository_url"),
            local_path=request.get("local_path"),
            format="cyclonedx",
            include_licenses=True,
        )

        # Scan for vulnerabilities using the SBOM
        vulnerabilities = self.scan_vulnerabilities(
            sbom=sbom,
            severity_threshold="low",
        )

        # Check licenses
        licenses = self.check_licenses(
            repository_url=request.get("repository_url"),
            local_path=request.get("local_path"),
            policy=request.get("license_policy", "permissive"),
        )

        # Determine overall risk level
        risk_level: str = "low"
        if vulnerabilities.get("critical", 0) > 0:
            risk_level = "critical"
        elif vulnerabilities.get("high", 0) > 0 or not licenses.get("compliant", True):
            risk_level = "high"
        elif vulnerabilities.get("medium", 0) > 0:
            risk_level = "medium"

        return {
            "dependencies": dependencies,
            "sbom": sbom,
            "vulnerabilities": vulnerabilities,
            "licenses": licenses,
            "summary": {
                "total_dependencies": dependencies.get("total_dependencies", 0),
                "total_vulnerabilities": vulnerabilities.get("total_vulnerabilities", 0),
                "license_compliant": licenses.get("compliant", True),
                "risk_level": risk_level,
            },
        }


class AsyncDependencyAnalysisAPI:
    """
    Asynchronous Dependency Analysis API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     deps = await client.dependency_analysis.analyze(
        ...         repository_url="https://github.com/org/repo",
        ...         include_transitive=True,
        ...     )
        ...     sbom = await client.dependency_analysis.generate_sbom(
        ...         repository_url="https://github.com/org/repo",
        ...         format="cyclonedx",
        ...     )
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Dependency Analysis
    # ===========================================================================

    async def analyze(
        self,
        repository_url: str | None = None,
        local_path: str | None = None,
        package_managers: list[Literal["npm", "pip", "cargo", "go", "maven", "gradle"]]
        | None = None,
        include_dev: bool = False,
        include_transitive: bool = True,
        max_depth: int | None = None,
    ) -> dict[str, Any]:
        """Analyze project dependencies."""
        data: dict[str, Any] = {
            "include_dev_dependencies": include_dev,
            "include_transitive": include_transitive,
        }
        if repository_url is not None:
            data["repository_url"] = repository_url
        if local_path is not None:
            data["local_path"] = local_path
        if package_managers is not None:
            data["package_managers"] = package_managers
        if max_depth is not None:
            data["max_depth"] = max_depth
        return await self._client.request(
            "POST", "/api/v1/codebase/analyze-dependencies", json=data
        )

    # ===========================================================================
    # SBOM Generation
    # ===========================================================================

    async def generate_sbom(
        self,
        repository_url: str | None = None,
        local_path: str | None = None,
        format: Literal["spdx", "cyclonedx", "json"] = "cyclonedx",
        include_checksums: bool = False,
        include_licenses: bool = True,
    ) -> dict[str, Any]:
        """Generate Software Bill of Materials (SBOM)."""
        data: dict[str, Any] = {
            "format": format,
            "include_checksums": include_checksums,
            "include_licenses": include_licenses,
        }
        if repository_url is not None:
            data["repository_url"] = repository_url
        if local_path is not None:
            data["local_path"] = local_path
        return await self._client.request("POST", "/api/v1/codebase/sbom", json=data)

    # ===========================================================================
    # Vulnerability Scanning
    # ===========================================================================

    async def scan_vulnerabilities(
        self,
        repository_url: str | None = None,
        local_path: str | None = None,
        sbom: dict[str, Any] | None = None,
        severity_threshold: Literal["low", "medium", "high", "critical"] = "low",
        ignore_cves: list[str] | None = None,
    ) -> dict[str, Any]:
        """Scan for known vulnerabilities (CVEs)."""
        data: dict[str, Any] = {
            "severity_threshold": severity_threshold,
        }
        if repository_url is not None:
            data["repository_url"] = repository_url
        if local_path is not None:
            data["local_path"] = local_path
        if sbom is not None:
            data["sbom"] = sbom
        if ignore_cves is not None:
            data["ignore_cves"] = ignore_cves
        return await self._client.request(
            "POST", "/api/v1/codebase/scan-vulnerabilities", json=data
        )

    # ===========================================================================
    # License Compliance
    # ===========================================================================

    async def check_licenses(
        self,
        repository_url: str | None = None,
        local_path: str | None = None,
        allowed_licenses: list[str] | None = None,
        denied_licenses: list[str] | None = None,
        policy: Literal["permissive", "copyleft", "commercial", "custom"] = "permissive",
    ) -> dict[str, Any]:
        """Check license compliance."""
        data: dict[str, Any] = {
            "policy": policy,
        }
        if repository_url is not None:
            data["repository_url"] = repository_url
        if local_path is not None:
            data["local_path"] = local_path
        if allowed_licenses is not None:
            data["allowed_licenses"] = allowed_licenses
        if denied_licenses is not None:
            data["denied_licenses"] = denied_licenses
        return await self._client.request("POST", "/api/v1/codebase/check-licenses", json=data)

    # ===========================================================================
    # Cache Management
    # ===========================================================================

    async def clear_cache(
        self,
        repository_url: str | None = None,
    ) -> dict[str, Any]:
        """Clear analysis cache."""
        data: dict[str, Any] = {}
        if repository_url is not None:
            data["repository_url"] = repository_url
        return await self._client.request("POST", "/api/v1/codebase/clear-cache", json=data)

    # ===========================================================================
    # Convenience Methods
    # ===========================================================================

    async def full_audit(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run a full security audit (dependencies + vulnerabilities + licenses).

        Convenience method that combines multiple checks into a single audit.
        """
        # Run dependency analysis first
        dependencies = await self.analyze(
            repository_url=request.get("repository_url"),
            local_path=request.get("local_path"),
            package_managers=request.get("package_managers"),
            include_dev=request.get("include_dev_dependencies", False),
            include_transitive=request.get("include_transitive", True),
            max_depth=request.get("max_depth"),
        )

        # Generate SBOM
        sbom = await self.generate_sbom(
            repository_url=request.get("repository_url"),
            local_path=request.get("local_path"),
            format="cyclonedx",
            include_licenses=True,
        )

        # Scan for vulnerabilities using the SBOM
        vulnerabilities = await self.scan_vulnerabilities(
            sbom=sbom,
            severity_threshold="low",
        )

        # Check licenses
        licenses = await self.check_licenses(
            repository_url=request.get("repository_url"),
            local_path=request.get("local_path"),
            policy=request.get("license_policy", "permissive"),
        )

        # Determine overall risk level
        risk_level: str = "low"
        if vulnerabilities.get("critical", 0) > 0:
            risk_level = "critical"
        elif vulnerabilities.get("high", 0) > 0 or not licenses.get("compliant", True):
            risk_level = "high"
        elif vulnerabilities.get("medium", 0) > 0:
            risk_level = "medium"

        return {
            "dependencies": dependencies,
            "sbom": sbom,
            "vulnerabilities": vulnerabilities,
            "licenses": licenses,
            "summary": {
                "total_dependencies": dependencies.get("total_dependencies", 0),
                "total_vulnerabilities": vulnerabilities.get("total_vulnerabilities", 0),
                "license_compliant": licenses.get("compliant", True),
                "risk_level": risk_level,
            },
        }
