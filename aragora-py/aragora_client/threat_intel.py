"""Threat Intelligence API for the Aragora SDK.

Provides security scanning and threat detection capabilities:
- URL threat checking
- IP reputation lookup
- File hash verification
- Email content scanning
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora_client.client import AragoraClient


@dataclass
class ThreatResult:
    """Result of a threat intelligence check."""

    is_threat: bool
    threat_level: str  # none, low, medium, high, critical
    threat_type: str | None  # malware, phishing, spam, etc.
    confidence: float
    details: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThreatResult:
        return cls(
            is_threat=data.get("is_threat", False),
            threat_level=data.get("threat_level", "none"),
            threat_type=data.get("threat_type"),
            confidence=data.get("confidence", 0.0),
            details=data.get("details", {}),
        )


@dataclass
class IPReputation:
    """IP address reputation information."""

    ip_address: str
    is_malicious: bool
    reputation_score: float  # 0-100, higher is better
    categories: list[str]  # vpn, tor, proxy, botnet, etc.
    country: str | None
    asn: str | None
    details: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IPReputation:
        return cls(
            ip_address=data.get("ip_address", ""),
            is_malicious=data.get("is_malicious", False),
            reputation_score=data.get("reputation_score", 0.0),
            categories=data.get("categories", []),
            country=data.get("country"),
            asn=data.get("asn"),
            details=data.get("details", {}),
        )


@dataclass
class HashReputation:
    """File hash reputation information."""

    hash_value: str
    hash_type: str  # md5, sha1, sha256
    is_malicious: bool
    malware_family: str | None
    detection_count: int
    first_seen: str | None
    last_seen: str | None
    details: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HashReputation:
        return cls(
            hash_value=data.get("hash_value", ""),
            hash_type=data.get("hash_type", "sha256"),
            is_malicious=data.get("is_malicious", False),
            malware_family=data.get("malware_family"),
            detection_count=data.get("detection_count", 0),
            first_seen=data.get("first_seen"),
            last_seen=data.get("last_seen"),
            details=data.get("details", {}),
        )


@dataclass
class EmailScanResult:
    """Result of email content scanning."""

    is_threat: bool
    threat_types: list[str]  # phishing, spam, malware, etc.
    confidence: float
    suspicious_links: list[str]
    suspicious_attachments: list[str]
    indicators: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmailScanResult:
        return cls(
            is_threat=data.get("is_threat", False),
            threat_types=data.get("threat_types", []),
            confidence=data.get("confidence", 0.0),
            suspicious_links=data.get("suspicious_links", []),
            suspicious_attachments=data.get("suspicious_attachments", []),
            indicators=data.get("indicators", {}),
        )


class ThreatIntelAPI:
    """API for threat intelligence operations.

    Provides security scanning capabilities for URLs, IPs, file hashes,
    and email content. Uses multiple threat intelligence sources for
    comprehensive coverage.
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # URL Checking
    # =========================================================================

    async def check_url(self, url: str) -> ThreatResult:
        """Check a URL for threats.

        Analyzes the URL against threat intelligence databases for
        malware, phishing, and other malicious content.

        Args:
            url: URL to check

        Returns:
            ThreatResult with threat assessment

        Example:
            result = await client.threat_intel.check_url("https://example.com")
            if result.is_threat:
                print(f"Threat detected: {result.threat_type}")
        """
        data = await self._client._post("/api/v1/threat/url", {"url": url})
        return ThreatResult.from_dict(data)

    async def check_urls(self, urls: list[str]) -> list[ThreatResult]:
        """Check multiple URLs for threats.

        Batch operation for checking many URLs efficiently.

        Args:
            urls: List of URLs to check

        Returns:
            List of ThreatResult for each URL

        Example:
            results = await client.threat_intel.check_urls([
                "https://example.com",
                "https://suspicious-site.com"
            ])
            threats = [r for r in results if r.is_threat]
        """
        data = await self._client._post("/api/v1/threat/urls", {"urls": urls})
        return [ThreatResult.from_dict(r) for r in data.get("results", [])]

    # =========================================================================
    # IP Reputation
    # =========================================================================

    async def check_ip(self, ip_address: str) -> IPReputation:
        """Check IP address reputation.

        Looks up reputation data for an IP address including
        abuse history, geographic location, and threat categories.

        Args:
            ip_address: IP address to check (IPv4 or IPv6)

        Returns:
            IPReputation with reputation data

        Example:
            rep = await client.threat_intel.check_ip("192.168.1.1")
            if rep.is_malicious:
                print(f"Malicious IP in categories: {rep.categories}")
        """
        data = await self._client._get(f"/api/v1/threat/ip/{ip_address}")
        return IPReputation.from_dict(data)

    async def check_ips(self, ip_addresses: list[str]) -> list[IPReputation]:
        """Check multiple IP addresses.

        Batch operation for checking many IPs efficiently.

        Args:
            ip_addresses: List of IP addresses to check

        Returns:
            List of IPReputation for each IP
        """
        data = await self._client._post(
            "/api/v1/threat/ips", {"ip_addresses": ip_addresses}
        )
        return [IPReputation.from_dict(r) for r in data.get("results", [])]

    # =========================================================================
    # File Hash Reputation
    # =========================================================================

    async def check_hash(self, hash_value: str) -> HashReputation:
        """Check file hash reputation.

        Looks up a file hash (MD5, SHA1, or SHA256) against malware
        databases to determine if the file is known to be malicious.

        Args:
            hash_value: File hash to check

        Returns:
            HashReputation with reputation data

        Example:
            rep = await client.threat_intel.check_hash(
                "e99a18c428cb38d5f260853678922e03"
            )
            if rep.is_malicious:
                print(f"Known malware: {rep.malware_family}")
        """
        data = await self._client._get(f"/api/v1/threat/hash/{hash_value}")
        return HashReputation.from_dict(data)

    async def check_hashes(self, hash_values: list[str]) -> list[HashReputation]:
        """Check multiple file hashes.

        Batch operation for checking many hashes efficiently.

        Args:
            hash_values: List of file hashes to check

        Returns:
            List of HashReputation for each hash
        """
        data = await self._client._post(
            "/api/v1/threat/hashes", {"hash_values": hash_values}
        )
        return [HashReputation.from_dict(r) for r in data.get("results", [])]

    # =========================================================================
    # Email Scanning
    # =========================================================================

    async def scan_email(
        self,
        *,
        subject: str | None = None,
        body: str | None = None,
        sender: str | None = None,
        links: list[str] | None = None,
        attachment_hashes: list[str] | None = None,
    ) -> EmailScanResult:
        """Scan email content for threats.

        Analyzes email components for phishing, malware, and spam
        indicators.

        Args:
            subject: Email subject line
            body: Email body content
            sender: Sender email address
            links: URLs found in the email
            attachment_hashes: Hashes of email attachments

        Returns:
            EmailScanResult with threat assessment

        Example:
            result = await client.threat_intel.scan_email(
                subject="Urgent: Verify your account",
                body="Click here to verify...",
                sender="support@ph1shing-site.com",
                links=["http://ph1shing-site.com/verify"]
            )
            if result.is_threat:
                print(f"Threats: {result.threat_types}")
        """
        data: dict[str, Any] = {}
        if subject:
            data["subject"] = subject
        if body:
            data["body"] = body
        if sender:
            data["sender"] = sender
        if links:
            data["links"] = links
        if attachment_hashes:
            data["attachment_hashes"] = attachment_hashes

        response = await self._client._post("/api/v1/threat/email", data)
        return EmailScanResult.from_dict(response)

    # =========================================================================
    # Service Status
    # =========================================================================

    async def get_status(self) -> dict[str, Any]:
        """Get threat intelligence service status.

        Returns information about the service health, connected
        threat feeds, and rate limit status.

        Returns:
            Dictionary with service status information
        """
        return await self._client._get("/api/v1/threat/status")
