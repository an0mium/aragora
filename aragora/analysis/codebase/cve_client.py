"""
CVE Client for Vulnerability Database Integration.

Provides integration with multiple vulnerability databases:
- NVD (NIST National Vulnerability Database)
- OSV (Open Source Vulnerabilities)
- GitHub Security Advisories

Features:
- Async HTTP client with circuit breaker
- Response caching
- Rate limit handling
- Batch queries
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .models import (
    VulnerabilityFinding,
    VulnerabilityReference,
    VulnerabilitySeverity,
    VulnerabilitySource,
)

logger = logging.getLogger(__name__)


class CVEClient:
    """
    Client for querying CVE/vulnerability databases.

    Supports multiple sources:
    - NVD API (requires API key for higher rate limits)
    - OSV API (free, focused on open source)
    - GitHub Advisory Database (requires token)

    Example:
        client = CVEClient(nvd_api_key="your-key")

        # Query by CVE ID
        vuln = await client.get_cve("CVE-2023-12345")

        # Query by package
        vulns = await client.query_package("lodash", "npm", "4.17.0")

        # Batch query
        results = await client.batch_query_packages([
            ("requests", "pypi", "2.28.0"),
            ("express", "npm", "4.18.0"),
        ])
    """

    # API endpoints
    NVD_API_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    OSV_API_BASE = "https://api.osv.dev/v1"
    GITHUB_API_BASE = "https://api.github.com"

    # Ecosystem mappings for different APIs
    ECOSYSTEM_MAP = {
        "npm": {"osv": "npm", "github": "NPM"},
        "pypi": {"osv": "PyPI", "github": "PIP"},
        "maven": {"osv": "Maven", "github": "MAVEN"},
        "cargo": {"osv": "crates.io", "github": "RUST"},
        "go": {"osv": "Go", "github": "GO"},
        "nuget": {"osv": "NuGet", "github": "NUGET"},
        "rubygems": {"osv": "RubyGems", "github": "RUBYGEMS"},
        "composer": {"osv": "Packagist", "github": "COMPOSER"},
    }

    def __init__(
        self,
        nvd_api_key: Optional[str] = None,
        github_token: Optional[str] = None,
        cache_ttl_seconds: int = 3600,
        enable_circuit_breaker: bool = True,
    ):
        """
        Initialize CVE client.

        Args:
            nvd_api_key: Optional NVD API key for higher rate limits
            github_token: Optional GitHub token for advisory API
            cache_ttl_seconds: Cache TTL in seconds
            enable_circuit_breaker: Enable circuit breaker protection
        """
        self.nvd_api_key = nvd_api_key or os.environ.get("NVD_API_KEY")
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")
        self.cache_ttl = cache_ttl_seconds

        # Simple in-memory cache
        self._cache: Dict[str, tuple[datetime, Any]] = {}

        # Circuit breaker state
        self._enable_circuit_breaker = enable_circuit_breaker
        self._circuit_breaker_failures: Dict[str, int] = {}
        self._circuit_breaker_open_until: Dict[str, datetime] = {}
        self._failure_threshold = 5
        self._cooldown_seconds = 60.0

    def _get_cache_key(self, *args) -> str:
        """Generate cache key from arguments."""
        return hashlib.md5(str(args).encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self._cache:
            cached_at, value = self._cache[key]
            if datetime.now(timezone.utc) - cached_at < timedelta(seconds=self.cache_ttl):
                return value
            del self._cache[key]
        return None

    def _set_cached(self, key: str, value: Any) -> None:
        """Set value in cache."""
        self._cache[key] = (datetime.now(timezone.utc), value)

    def _check_circuit_breaker(self, source: str) -> bool:
        """Check if requests to source are allowed."""
        if not self._enable_circuit_breaker:
            return True

        if source in self._circuit_breaker_open_until:
            if datetime.now(timezone.utc) < self._circuit_breaker_open_until[source]:
                return False
            del self._circuit_breaker_open_until[source]
            self._circuit_breaker_failures[source] = 0

        return True

    def _record_success(self, source: str) -> None:
        """Record successful request."""
        self._circuit_breaker_failures[source] = 0

    def _record_failure(self, source: str) -> None:
        """Record failed request."""
        if not self._enable_circuit_breaker:
            return

        self._circuit_breaker_failures[source] = self._circuit_breaker_failures.get(source, 0) + 1

        if self._circuit_breaker_failures[source] >= self._failure_threshold:
            self._circuit_breaker_open_until[source] = datetime.now(timezone.utc) + timedelta(
                seconds=self._cooldown_seconds
            )
            logger.warning(f"[CVEClient] Circuit breaker opened for {source}")

    async def get_cve(self, cve_id: str) -> Optional[VulnerabilityFinding]:
        """
        Get vulnerability details by CVE ID.

        Args:
            cve_id: CVE identifier (e.g., "CVE-2023-12345")

        Returns:
            VulnerabilityFinding or None if not found
        """
        # Check cache
        cache_key = self._get_cache_key("cve", cve_id)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # Try NVD first
        vuln = await self._query_nvd_cve(cve_id)
        if vuln:
            self._set_cached(cache_key, vuln)
            return vuln

        # Fall back to OSV
        vuln = await self._query_osv_cve(cve_id)
        if vuln:
            self._set_cached(cache_key, vuln)
            return vuln

        return None

    async def query_package(
        self,
        package_name: str,
        ecosystem: str,
        version: Optional[str] = None,
    ) -> List[VulnerabilityFinding]:
        """
        Query vulnerabilities for a package.

        Args:
            package_name: Package name
            ecosystem: Package ecosystem (npm, pypi, etc.)
            version: Specific version to check

        Returns:
            List of vulnerability findings
        """
        # Check cache
        cache_key = self._get_cache_key("package", package_name, ecosystem, version)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        findings = []

        # Query OSV (primary for open source)
        osv_findings = await self._query_osv_package(package_name, ecosystem, version)
        findings.extend(osv_findings)

        # Query GitHub Advisory
        gh_findings = await self._query_github_advisory(package_name, ecosystem)
        findings.extend(gh_findings)

        # Deduplicate by CVE ID
        seen_ids = set()
        unique_findings = []
        for finding in findings:
            if finding.id not in seen_ids:
                seen_ids.add(finding.id)
                unique_findings.append(finding)

        self._set_cached(cache_key, unique_findings)
        return unique_findings

    async def batch_query_packages(
        self,
        packages: List[tuple[str, str, str]],  # (name, ecosystem, version)
        concurrency: int = 10,
    ) -> Dict[str, List[VulnerabilityFinding]]:
        """
        Query vulnerabilities for multiple packages concurrently.

        Args:
            packages: List of (package_name, ecosystem, version) tuples
            concurrency: Max concurrent requests

        Returns:
            Dict mapping "package_name@version" to findings
        """
        results: Dict[str, List[VulnerabilityFinding]] = {}
        semaphore = asyncio.Semaphore(concurrency)

        async def query_one(name: str, ecosystem: str, version: str):
            async with semaphore:
                findings = await self.query_package(name, ecosystem, version)
                key = f"{name}@{version}"
                results[key] = findings

        await asyncio.gather(
            *[query_one(name, eco, ver) for name, eco, ver in packages],
            return_exceptions=True,
        )

        return results

    async def _query_nvd_cve(self, cve_id: str) -> Optional[VulnerabilityFinding]:
        """Query NVD API for CVE details."""
        if not self._check_circuit_breaker("nvd"):
            return None

        import httpx

        try:
            headers = {}
            if self.nvd_api_key:
                headers["apiKey"] = self.nvd_api_key

            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    self.NVD_API_BASE,
                    params={"cveId": cve_id},
                    headers=headers,
                )

                if response.status_code == 429:
                    self._record_failure("nvd")
                    logger.warning("[CVEClient] NVD rate limited")
                    return None

                if response.status_code != 200:
                    return None

                data = response.json()
                self._record_success("nvd")

            vulnerabilities = data.get("vulnerabilities", [])
            if not vulnerabilities:
                return None

            cve_data = vulnerabilities[0].get("cve", {})
            return self._parse_nvd_cve(cve_data)

        except Exception as e:
            self._record_failure("nvd")
            logger.error(f"[CVEClient] NVD query failed: {e}")
            return None

    def _parse_nvd_cve(self, cve_data: Dict[str, Any]) -> VulnerabilityFinding:
        """Parse NVD CVE response into VulnerabilityFinding."""
        cve_id = cve_data.get("id", "")

        # Get description
        descriptions = cve_data.get("descriptions", [])
        description = ""
        for desc in descriptions:
            if desc.get("lang") == "en":
                description = desc.get("value", "")
                break

        # Get CVSS score
        cvss_score = None
        cvss_vector = None
        severity = VulnerabilitySeverity.UNKNOWN

        metrics = cve_data.get("metrics", {})
        for version in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
            if version in metrics and metrics[version]:
                metric = metrics[version][0]
                cvss_data = metric.get("cvssData", {})
                cvss_score = cvss_data.get("baseScore")
                cvss_vector = cvss_data.get("vectorString")
                severity = VulnerabilitySeverity.from_cvss(cvss_score or 0)
                break

        # Get CWE IDs
        cwe_ids = []
        weaknesses = cve_data.get("weaknesses", [])
        for weakness in weaknesses:
            for desc in weakness.get("description", []):
                if desc.get("value", "").startswith("CWE-"):
                    cwe_ids.append(desc["value"])

        # Get references
        references = []
        for ref in cve_data.get("references", []):
            references.append(
                VulnerabilityReference(
                    url=ref.get("url", ""),
                    source=ref.get("source", ""),
                    tags=ref.get("tags", []),
                )
            )

        # Parse dates
        published_at = None
        updated_at = None
        if cve_data.get("published"):
            try:
                published_at = datetime.fromisoformat(cve_data["published"].replace("Z", "+00:00"))
            except ValueError:
                pass
        if cve_data.get("lastModified"):
            try:
                updated_at = datetime.fromisoformat(cve_data["lastModified"].replace("Z", "+00:00"))
            except ValueError:
                pass

        return VulnerabilityFinding(
            id=cve_id,
            title=cve_id,
            description=description,
            severity=severity,
            cvss_score=cvss_score,
            cvss_vector=cvss_vector,
            source=VulnerabilitySource.NVD,
            published_at=published_at,
            updated_at=updated_at,
            references=references,
            cwe_ids=cwe_ids,
        )

    async def _query_osv_cve(self, cve_id: str) -> Optional[VulnerabilityFinding]:
        """Query OSV API for CVE details."""
        if not self._check_circuit_breaker("osv"):
            return None

        import httpx

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{self.OSV_API_BASE}/vulns/{cve_id}",
                )

                if response.status_code == 404:
                    return None

                if response.status_code != 200:
                    self._record_failure("osv")
                    return None

                data = response.json()
                self._record_success("osv")

            return self._parse_osv_vuln(data)

        except Exception as e:
            self._record_failure("osv")
            logger.error(f"[CVEClient] OSV query failed: {e}")
            return None

    async def _query_osv_package(
        self,
        package_name: str,
        ecosystem: str,
        version: Optional[str] = None,
    ) -> List[VulnerabilityFinding]:
        """Query OSV API for package vulnerabilities."""
        if not self._check_circuit_breaker("osv"):
            return []

        import httpx

        osv_ecosystem = self.ECOSYSTEM_MAP.get(ecosystem, {}).get("osv", ecosystem)

        try:
            payload: Dict[str, Any] = {
                "package": {
                    "name": package_name,
                    "ecosystem": osv_ecosystem,
                }
            }
            if version:
                payload["version"] = version

            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{self.OSV_API_BASE}/query",
                    json=payload,
                )

                if response.status_code != 200:
                    self._record_failure("osv")
                    return []

                data = response.json()
                self._record_success("osv")

            findings = []
            for vuln in data.get("vulns", []):
                finding = self._parse_osv_vuln(vuln)
                finding.package_name = package_name
                finding.package_ecosystem = ecosystem
                findings.append(finding)

            return findings

        except Exception as e:
            self._record_failure("osv")
            logger.error(f"[CVEClient] OSV package query failed: {e}")
            return []

    def _parse_osv_vuln(self, data: Dict[str, Any]) -> VulnerabilityFinding:
        """Parse OSV vulnerability response."""
        vuln_id = data.get("id", "")

        # Get description
        description = data.get("summary", "") or data.get("details", "")

        # Get severity from CVSS
        cvss_score = None
        cvss_vector = None
        severity = VulnerabilitySeverity.UNKNOWN

        for severity_info in data.get("severity", []):
            if severity_info.get("type") == "CVSS_V3":
                cvss_vector = severity_info.get("score")
                # Parse score from vector if needed
                if cvss_vector and "CVSS:3" in cvss_vector:
                    # Extract base score from vector or use heuristics
                    pass

        # Check for severity in database_specific
        db_specific = data.get("database_specific", {})
        if db_specific.get("severity"):
            severity_str = db_specific["severity"].lower()
            if severity_str == "critical":
                severity = VulnerabilitySeverity.CRITICAL
            elif severity_str == "high":
                severity = VulnerabilitySeverity.HIGH
            elif severity_str == "medium" or severity_str == "moderate":
                severity = VulnerabilitySeverity.MEDIUM
            elif severity_str == "low":
                severity = VulnerabilitySeverity.LOW

        # Get affected versions
        vulnerable_versions = []
        patched_versions = []
        for affected in data.get("affected", []):
            for range_info in affected.get("ranges", []):
                for event in range_info.get("events", []):
                    if "introduced" in event:
                        vulnerable_versions.append(f">= {event['introduced']}")
                    if "fixed" in event:
                        patched_versions.append(event["fixed"])

        # Get references
        references = []
        for ref in data.get("references", []):
            references.append(
                VulnerabilityReference(
                    url=ref.get("url", ""),
                    source="osv",
                    tags=[ref.get("type", "")],
                )
            )

        # Get CWE IDs from aliases
        cwe_ids = []
        for alias in data.get("aliases", []):
            if alias.startswith("CWE-"):
                cwe_ids.append(alias)

        # Parse dates
        published_at = None
        if data.get("published"):
            try:
                published_at = datetime.fromisoformat(data["published"].replace("Z", "+00:00"))
            except ValueError:
                pass

        return VulnerabilityFinding(
            id=vuln_id,
            title=vuln_id,
            description=description,
            severity=severity,
            cvss_score=cvss_score,
            cvss_vector=cvss_vector,
            vulnerable_versions=vulnerable_versions,
            patched_versions=patched_versions,
            source=VulnerabilitySource.OSV,
            published_at=published_at,
            references=references,
            cwe_ids=cwe_ids,
            fix_available=len(patched_versions) > 0,
            recommended_version=patched_versions[0] if patched_versions else None,
        )

    async def _query_github_advisory(
        self,
        package_name: str,
        ecosystem: str,
    ) -> List[VulnerabilityFinding]:
        """Query GitHub Security Advisory Database."""
        if not self.github_token:
            return []

        if not self._check_circuit_breaker("github"):
            return []

        import httpx

        gh_ecosystem = self.ECOSYSTEM_MAP.get(ecosystem, {}).get("github", ecosystem.upper())

        try:
            # Use GraphQL API for advisories
            query = """
            query($ecosystem: SecurityAdvisoryEcosystem!, $package: String!) {
              securityVulnerabilities(
                first: 50,
                ecosystem: $ecosystem,
                package: $package
              ) {
                nodes {
                  advisory {
                    ghsaId
                    summary
                    description
                    severity
                    cvss {
                      score
                      vectorString
                    }
                    references {
                      url
                    }
                    cwes(first: 10) {
                      nodes {
                        cweId
                      }
                    }
                    publishedAt
                    updatedAt
                  }
                  vulnerableVersionRange
                  firstPatchedVersion {
                    identifier
                  }
                }
              }
            }
            """

            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{self.GITHUB_API_BASE}/graphql",
                    headers={
                        "Authorization": f"Bearer {self.github_token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "query": query,
                        "variables": {
                            "ecosystem": gh_ecosystem,
                            "package": package_name,
                        },
                    },
                )

                if response.status_code != 200:
                    self._record_failure("github")
                    return []

                data = response.json()
                self._record_success("github")

            findings = []
            vulns = data.get("data", {}).get("securityVulnerabilities", {}).get("nodes", [])

            for vuln in vulns:
                advisory = vuln.get("advisory", {})
                finding = self._parse_github_advisory(advisory, vuln)
                finding.package_name = package_name
                finding.package_ecosystem = ecosystem
                findings.append(finding)

            return findings

        except Exception as e:
            self._record_failure("github")
            logger.error(f"[CVEClient] GitHub advisory query failed: {e}")
            return []

    def _parse_github_advisory(
        self,
        advisory: Dict[str, Any],
        vuln_data: Dict[str, Any],
    ) -> VulnerabilityFinding:
        """Parse GitHub advisory response."""
        ghsa_id = advisory.get("ghsaId", "")

        # Get severity
        severity_str = advisory.get("severity", "").lower()
        severity = VulnerabilitySeverity.UNKNOWN
        if severity_str == "critical":
            severity = VulnerabilitySeverity.CRITICAL
        elif severity_str == "high":
            severity = VulnerabilitySeverity.HIGH
        elif severity_str == "moderate":
            severity = VulnerabilitySeverity.MEDIUM
        elif severity_str == "low":
            severity = VulnerabilitySeverity.LOW

        # Get CVSS
        cvss = advisory.get("cvss", {})
        cvss_score = cvss.get("score")
        cvss_vector = cvss.get("vectorString")

        # Get CWE IDs
        cwe_ids = []
        for cwe_node in advisory.get("cwes", {}).get("nodes", []):
            cwe_ids.append(cwe_node.get("cweId", ""))

        # Get references
        references = []
        for ref in advisory.get("references", []):
            references.append(
                VulnerabilityReference(
                    url=ref.get("url", ""),
                    source="github",
                    tags=[],
                )
            )

        # Get version info
        vulnerable_versions = []
        if vuln_data.get("vulnerableVersionRange"):
            vulnerable_versions.append(vuln_data["vulnerableVersionRange"])

        patched_versions = []
        if vuln_data.get("firstPatchedVersion", {}).get("identifier"):
            patched_versions.append(vuln_data["firstPatchedVersion"]["identifier"])

        # Parse dates
        published_at = None
        updated_at = None
        if advisory.get("publishedAt"):
            try:
                published_at = datetime.fromisoformat(
                    advisory["publishedAt"].replace("Z", "+00:00")
                )
            except ValueError:
                pass
        if advisory.get("updatedAt"):
            try:
                updated_at = datetime.fromisoformat(advisory["updatedAt"].replace("Z", "+00:00"))
            except ValueError:
                pass

        return VulnerabilityFinding(
            id=ghsa_id,
            title=advisory.get("summary", ghsa_id),
            description=advisory.get("description", ""),
            severity=severity,
            cvss_score=cvss_score,
            cvss_vector=cvss_vector,
            vulnerable_versions=vulnerable_versions,
            patched_versions=patched_versions,
            source=VulnerabilitySource.GITHUB,
            published_at=published_at,
            updated_at=updated_at,
            references=references,
            cwe_ids=cwe_ids,
            fix_available=len(patched_versions) > 0,
            recommended_version=patched_versions[0] if patched_versions else None,
        )


__all__ = ["CVEClient"]
