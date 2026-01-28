"""
Threat Intelligence Namespace API.

Provides a namespaced interface for threat intelligence operations:
- URL scanning against VirusTotal and PhishTank
- IP reputation checking via AbuseIPDB
- File hash lookup in VirusTotal
- Email content analysis
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

ThreatSeverity = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]
ThreatType = Literal[
    "malware", "phishing", "spam", "suspicious", "ransomware", "trojan", "botnet", "none"
]
HashType = Literal["md5", "sha1", "sha256"]


class URLCheckResult(TypedDict, total=False):
    """URL check result."""

    target: str
    is_malicious: bool
    threat_type: str
    severity: str
    confidence: float
    virustotal: dict[str, Any] | None
    phishtank: dict[str, Any] | None
    cached: bool


class IPReputationResult(TypedDict, total=False):
    """IP reputation result."""

    ip_address: str
    is_malicious: bool
    abuse_score: int
    total_reports: int
    country_code: str | None
    isp: str | None
    domain: str | None
    usage_type: str | None
    last_reported: str | None
    categories: list[str]
    cached: bool


class HashCheckResult(TypedDict, total=False):
    """File hash check result."""

    hash_value: str
    hash_type: str
    is_malware: bool
    threat_type: str
    detection_ratio: str
    positives: int
    total_scanners: int
    scan_date: str | None
    file_name: str | None
    file_size: int | None
    file_type: str | None
    cached: bool


class EmailScanResult(TypedDict, total=False):
    """Email scan result."""

    urls: list[URLCheckResult]
    ips: list[IPReputationResult]
    overall_threat_score: float
    is_suspicious: bool
    threat_summary: list[str]


class ThreatIntelStatus(TypedDict, total=False):
    """Threat intel service status."""

    virustotal: dict[str, Any]
    abuseipdb: dict[str, Any]
    phishtank: dict[str, Any]
    caching: bool
    cache_ttl_hours: int


class ThreatIntelAPI:
    """
    Synchronous Threat Intelligence API.

    Provides comprehensive threat detection:
    - URL scanning against VirusTotal and PhishTank
    - IP reputation checking via AbuseIPDB
    - File hash lookup in VirusTotal
    - Email content analysis

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> # Check a URL for threats
        >>> result = client.threat_intel.check_url("https://suspicious-site.com")
        >>> if result["is_malicious"]:
        ...     print(f"Threat detected: {result['threat_type']}")
        >>> # Check IP reputation
        >>> ip_result = client.threat_intel.check_ip("192.168.1.1")
        >>> print(f"Abuse score: {ip_result['abuse_score']}")
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # URL Scanning
    # =========================================================================

    def check_url(
        self,
        url: str,
        check_virustotal: bool = True,
        check_phishtank: bool = True,
    ) -> URLCheckResult:
        """
        Check a URL for threats.

        Scans against VirusTotal and PhishTank for malware and phishing detection.

        Args:
            url: URL to check.
            check_virustotal: Include VirusTotal scan.
            check_phishtank: Include PhishTank scan.

        Returns:
            Threat analysis result with severity and confidence.
        """
        data: dict[str, Any] = {
            "url": url,
            "check_virustotal": check_virustotal,
            "check_phishtank": check_phishtank,
        }
        response = self._client.request("POST", "/api/v1/threat/url", json=data)
        return response.get("data", response)

    def check_urls_batch(
        self,
        urls: list[str],
        max_concurrent: int | None = None,
    ) -> dict[str, Any]:
        """
        Batch check multiple URLs for threats.

        Args:
            urls: List of URLs to check.
            max_concurrent: Maximum concurrent requests.

        Returns:
            Dict with 'results' list and 'summary' statistics.
        """
        data: dict[str, Any] = {"urls": urls}
        if max_concurrent is not None:
            data["max_concurrent"] = max_concurrent
        response = self._client.request("POST", "/api/v1/threat/urls", json=data)
        return response.get("data", response)

    # =========================================================================
    # IP Reputation
    # =========================================================================

    def check_ip(self, ip_address: str) -> IPReputationResult:
        """
        Check IP address reputation.

        Gets reputation data from AbuseIPDB including abuse score and reports.

        Args:
            ip_address: IP address to check.

        Returns:
            Reputation result with abuse score and categories.
        """
        response = self._client.request("GET", f"/api/v1/threat/ip/{ip_address}")
        return response.get("data", response)

    def check_ips_batch(self, ips: list[str]) -> dict[str, Any]:
        """
        Batch check multiple IP addresses.

        Args:
            ips: List of IP addresses to check.

        Returns:
            Dict with 'results' list and 'summary' statistics.
        """
        response = self._client.request("POST", "/api/v1/threat/ips", json={"ips": ips})
        return response.get("data", response)

    # =========================================================================
    # File Hash Lookup
    # =========================================================================

    def check_hash(self, hash_value: str) -> HashCheckResult:
        """
        Check file hash for malware.

        Looks up MD5, SHA1, or SHA256 hash in VirusTotal.

        Args:
            hash_value: File hash to check.

        Returns:
            Hash analysis result with detection ratio.
        """
        response = self._client.request("GET", f"/api/v1/threat/hash/{hash_value}")
        return response.get("data", response)

    def check_hashes_batch(self, hashes: list[str]) -> dict[str, Any]:
        """
        Batch check multiple file hashes.

        Args:
            hashes: List of file hashes to check.

        Returns:
            Dict with 'results' list and 'summary' statistics.
        """
        response = self._client.request("POST", "/api/v1/threat/hashes", json={"hashes": hashes})
        return response.get("data", response)

    # =========================================================================
    # Email Scanning
    # =========================================================================

    def scan_email(
        self,
        body: str,
        headers: dict[str, str] | None = None,
    ) -> EmailScanResult:
        """
        Scan email content for threats.

        Extracts and checks URLs and IPs from email body and headers.

        Args:
            body: Email body content.
            headers: Optional email headers.

        Returns:
            Comprehensive email analysis with threat score.
        """
        data: dict[str, Any] = {"body": body}
        if headers:
            data["headers"] = headers
        response = self._client.request("POST", "/api/v1/threat/email", json=data)
        return response.get("data", response)

    # =========================================================================
    # Service Status
    # =========================================================================

    def get_status(self) -> ThreatIntelStatus:
        """
        Get threat intelligence service status.

        Shows which providers are configured and available.

        Returns:
            Service status for all providers.
        """
        response = self._client.request("GET", "/api/v1/threat/status")
        return response.get("data", response)


class AsyncThreatIntelAPI:
    """Asynchronous Threat Intelligence API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def check_url(
        self,
        url: str,
        check_virustotal: bool = True,
        check_phishtank: bool = True,
    ) -> URLCheckResult:
        """Check a URL for threats."""
        data: dict[str, Any] = {
            "url": url,
            "check_virustotal": check_virustotal,
            "check_phishtank": check_phishtank,
        }
        response = await self._client.request("POST", "/api/v1/threat/url", json=data)
        return response.get("data", response)

    async def check_urls_batch(
        self,
        urls: list[str],
        max_concurrent: int | None = None,
    ) -> dict[str, Any]:
        """Batch check multiple URLs for threats."""
        data: dict[str, Any] = {"urls": urls}
        if max_concurrent is not None:
            data["max_concurrent"] = max_concurrent
        response = await self._client.request("POST", "/api/v1/threat/urls", json=data)
        return response.get("data", response)

    async def check_ip(self, ip_address: str) -> IPReputationResult:
        """Check IP address reputation."""
        response = await self._client.request("GET", f"/api/v1/threat/ip/{ip_address}")
        return response.get("data", response)

    async def check_ips_batch(self, ips: list[str]) -> dict[str, Any]:
        """Batch check multiple IP addresses."""
        response = await self._client.request("POST", "/api/v1/threat/ips", json={"ips": ips})
        return response.get("data", response)

    async def check_hash(self, hash_value: str) -> HashCheckResult:
        """Check file hash for malware."""
        response = await self._client.request("GET", f"/api/v1/threat/hash/{hash_value}")
        return response.get("data", response)

    async def check_hashes_batch(self, hashes: list[str]) -> dict[str, Any]:
        """Batch check multiple file hashes."""
        response = await self._client.request(
            "POST", "/api/v1/threat/hashes", json={"hashes": hashes}
        )
        return response.get("data", response)

    async def scan_email(
        self,
        body: str,
        headers: dict[str, str] | None = None,
    ) -> EmailScanResult:
        """Scan email content for threats."""
        data: dict[str, Any] = {"body": body}
        if headers:
            data["headers"] = headers
        response = await self._client.request("POST", "/api/v1/threat/email", json=data)
        return response.get("data", response)

    async def get_status(self) -> ThreatIntelStatus:
        """Get threat intelligence service status."""
        response = await self._client.request("GET", "/api/v1/threat/status")
        return response.get("data", response)
