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
        >>> result = client.threat_intel.check_url("https://suspicious-site.com")
        >>> if result["is_malicious"]:
        ...     print(f"Threat detected: {result['threat_type']}")
        >>> ip_result = client.threat_intel.check_ip("192.168.1.1")
        >>> print(f"Abuse score: {ip_result['abuse_score']}")
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # URL Scanning
    # =========================================================================

    def check_url(self, url: str) -> dict[str, Any]:
        """Scan a URL for threats."""
        return self._client.request("POST", "/api/v1/threat/url", json={"url": url})

    def check_urls(self, urls: list[str]) -> dict[str, Any]:
        """Scan multiple URLs for threats."""
        return self._client.request("POST", "/api/v1/threat/urls", json={"urls": urls})

    # =========================================================================
    # IP Reputation
    # =========================================================================

    def check_ip(self, ip_address: str) -> dict[str, Any]:
        """Check IP reputation."""
        return self._client.request("GET", f"/api/v1/threat/ip/{ip_address}")

    def check_ips(self, ip_addresses: list[str]) -> dict[str, Any]:
        """Check reputation of multiple IPs."""
        return self._client.request("POST", "/api/v1/threat/ips", json={"ips": ip_addresses})

    # =========================================================================
    # Hash Lookup
    # =========================================================================

    def check_hash(self, hash_value: str) -> dict[str, Any]:
        """Look up a file hash in threat databases."""
        return self._client.request("GET", f"/api/v1/threat/hash/{hash_value}")

    def check_hashes(self, hashes: list[str]) -> dict[str, Any]:
        """Look up multiple file hashes."""
        return self._client.request("POST", "/api/v1/threat/hashes", json={"hashes": hashes})

    # =========================================================================
    # Email Scanning
    # =========================================================================

    def scan_email(self, content: str, **kwargs: Any) -> dict[str, Any]:
        """Scan email content for threats."""
        body: dict[str, Any] = {"content": content, **kwargs}
        return self._client.request("POST", "/api/v1/threat/email", json=body)

    # =========================================================================
    # Status
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """Get threat intelligence service status."""
        return self._client.request("GET", "/api/v1/threat/status")


class AsyncThreatIntelAPI:
    """Asynchronous Threat Intelligence API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def check_url(self, url: str) -> dict[str, Any]:
        """Scan a URL for threats."""
        return await self._client.request("POST", "/api/v1/threat/url", json={"url": url})

    async def check_urls(self, urls: list[str]) -> dict[str, Any]:
        """Scan multiple URLs for threats."""
        return await self._client.request("POST", "/api/v1/threat/urls", json={"urls": urls})

    async def check_ip(self, ip_address: str) -> dict[str, Any]:
        """Check IP reputation."""
        return await self._client.request("GET", f"/api/v1/threat/ip/{ip_address}")

    async def check_ips(self, ip_addresses: list[str]) -> dict[str, Any]:
        """Check reputation of multiple IPs."""
        return await self._client.request("POST", "/api/v1/threat/ips", json={"ips": ip_addresses})

    async def check_hash(self, hash_value: str) -> dict[str, Any]:
        """Look up a file hash in threat databases."""
        return await self._client.request("GET", f"/api/v1/threat/hash/{hash_value}")

    async def check_hashes(self, hashes: list[str]) -> dict[str, Any]:
        """Look up multiple file hashes."""
        return await self._client.request("POST", "/api/v1/threat/hashes", json={"hashes": hashes})

    async def scan_email(self, content: str, **kwargs: Any) -> dict[str, Any]:
        """Scan email content for threats."""
        body: dict[str, Any] = {"content": content, **kwargs}
        return await self._client.request("POST", "/api/v1/threat/email", json=body)

    async def get_status(self) -> dict[str, Any]:
        """Get threat intelligence service status."""
        return await self._client.request("GET", "/api/v1/threat/status")
