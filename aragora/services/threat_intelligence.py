"""
Threat Intelligence Integration Service.

Provides unified access to external threat intelligence feeds for:
- URL/attachment scanning (VirusTotal)
- IP reputation checking (AbuseIPDB)
- Phishing URL detection (PhishTank)

Features:
- Async API clients with rate limiting
- Local caching to reduce API calls
- Confidence scoring and threat classification
- Batch scanning capabilities
- Fallback handling when APIs are unavailable

Usage:
    from aragora.services.threat_intelligence import ThreatIntelligenceService

    service = ThreatIntelligenceService(
        virustotal_api_key="your-key",
        abuseipdb_api_key="your-key",
    )

    # Check a URL
    result = await service.check_url("https://suspicious-site.com")
    if result.is_malicious:
        print(f"Threat detected: {result.threat_type}")

    # Check an IP
    ip_result = await service.check_ip("1.2.3.4")
    print(f"Abuse score: {ip_result.abuse_score}")

    # Check a file hash
    hash_result = await service.check_file_hash("abc123...")
    print(f"Malware: {hash_result.is_malware}")
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ThreatType(Enum):
    """Types of detected threats."""

    NONE = "none"
    MALWARE = "malware"
    PHISHING = "phishing"
    SPAM = "spam"
    SUSPICIOUS = "suspicious"
    MALICIOUS_IP = "malicious_ip"
    COMMAND_AND_CONTROL = "c2"
    BOTNET = "botnet"
    CRYPTO_MINER = "crypto_miner"
    RANSOMWARE = "ransomware"
    TROJAN = "trojan"
    UNKNOWN = "unknown"


class ThreatSeverity(Enum):
    """Severity levels for threats."""

    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ThreatSource(Enum):
    """Source of threat intelligence."""

    VIRUSTOTAL = "virustotal"
    ABUSEIPDB = "abuseipdb"
    PHISHTANK = "phishtank"
    LOCAL_RULES = "local_rules"
    CACHED = "cached"


@dataclass
class ThreatResult:
    """Result of a threat intelligence lookup."""

    target: str  # URL, IP, or hash that was checked
    target_type: str  # "url", "ip", "hash", "email"
    is_malicious: bool
    threat_type: ThreatType
    severity: ThreatSeverity
    confidence: float  # 0.0 to 1.0
    sources: List[ThreatSource]
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=datetime.now)
    cached: bool = False

    # Source-specific scores
    virustotal_positives: int = 0
    virustotal_total: int = 0
    abuseipdb_score: int = 0
    phishtank_verified: bool = False

    @property
    def threat_score(self) -> float:
        """Combined threat score (0-100)."""
        scores = []

        if self.virustotal_total > 0:
            vt_score = (self.virustotal_positives / self.virustotal_total) * 100
            scores.append(vt_score)

        if self.abuseipdb_score > 0:
            scores.append(self.abuseipdb_score)

        if self.phishtank_verified:
            scores.append(100)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "target": self.target,
            "target_type": self.target_type,
            "is_malicious": self.is_malicious,
            "threat_type": self.threat_type.value,
            "severity": self.severity.name,
            "severity_value": self.severity.value,
            "confidence": self.confidence,
            "threat_score": self.threat_score,
            "sources": [s.value for s in self.sources],
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
            "cached": self.cached,
            "virustotal": {
                "positives": self.virustotal_positives,
                "total": self.virustotal_total,
            },
            "abuseipdb_score": self.abuseipdb_score,
            "phishtank_verified": self.phishtank_verified,
        }


@dataclass
class IPReputationResult:
    """Result of IP reputation check."""

    ip_address: str
    is_malicious: bool
    abuse_score: int  # 0-100
    total_reports: int
    last_reported: Optional[datetime]
    country_code: Optional[str]
    isp: Optional[str]
    domain: Optional[str]
    usage_type: Optional[str]
    categories: List[str] = field(default_factory=list)
    is_tor: bool = False
    is_vpn: bool = False
    is_proxy: bool = False
    is_datacenter: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ip_address": self.ip_address,
            "is_malicious": self.is_malicious,
            "abuse_score": self.abuse_score,
            "total_reports": self.total_reports,
            "last_reported": self.last_reported.isoformat() if self.last_reported else None,
            "country_code": self.country_code,
            "isp": self.isp,
            "domain": self.domain,
            "usage_type": self.usage_type,
            "categories": self.categories,
            "is_tor": self.is_tor,
            "is_vpn": self.is_vpn,
            "is_proxy": self.is_proxy,
            "is_datacenter": self.is_datacenter,
        }


@dataclass
class FileHashResult:
    """Result of file hash lookup."""

    hash_value: str
    hash_type: str  # "md5", "sha1", "sha256"
    is_malware: bool
    malware_names: List[str] = field(default_factory=list)
    detection_ratio: str = "0/0"
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hash_value": self.hash_value,
            "hash_type": self.hash_type,
            "is_malware": self.is_malware,
            "malware_names": self.malware_names,
            "detection_ratio": self.detection_ratio,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "tags": self.tags,
        }


@dataclass
class ThreatIntelConfig:
    """Configuration for threat intelligence service."""

    # API Keys
    virustotal_api_key: Optional[str] = None
    abuseipdb_api_key: Optional[str] = None
    phishtank_api_key: Optional[str] = None

    # Rate limiting (requests per minute)
    virustotal_rate_limit: int = 4  # Free tier: 4/min
    abuseipdb_rate_limit: int = 60  # Free tier: 1000/day
    phishtank_rate_limit: int = 30

    # Cache settings
    cache_ttl_hours: int = 24
    cache_db_path: str = "threat_intel_cache.db"

    # Thresholds
    virustotal_malicious_threshold: int = 3  # Positives to consider malicious
    abuseipdb_malicious_threshold: int = 50  # Abuse score to consider malicious

    # Timeouts
    request_timeout_seconds: float = 10.0

    # Feature flags
    enable_virustotal: bool = True
    enable_abuseipdb: bool = True
    enable_phishtank: bool = True
    enable_caching: bool = True


# Known malicious patterns for local detection
MALICIOUS_URL_PATTERNS = [
    r"(?i)paypal.*\.(?!paypal\.com)",  # PayPal phishing
    r"(?i)google.*login.*\.(?!google\.com)",  # Google phishing
    r"(?i)microsoft.*verify.*\.(?!microsoft\.com)",  # Microsoft phishing
    r"(?i)apple.*id.*\.(?!apple\.com)",  # Apple phishing
    r"(?i)bank.*verify",  # Banking phishing
    r"(?i)account.*suspend",  # Account suspension scam
    r"(?i)\.tk$|\.ml$|\.ga$|\.cf$",  # Free TLD abuse
    r"(?i)bit\.ly.*[a-z0-9]{6,}",  # Suspicious shortened URLs
]

SUSPICIOUS_TLDS = {
    ".tk",
    ".ml",
    ".ga",
    ".cf",
    ".gq",  # Free TLDs
    ".xyz",
    ".top",
    ".work",
    ".click",  # Often abused
    ".zip",
    ".mov",  # Confusing file extensions
}


class ThreatIntelligenceService:
    """
    Unified threat intelligence service.

    Integrates with multiple threat feeds to provide comprehensive
    threat assessment for URLs, IPs, and file hashes.
    """

    def __init__(
        self,
        config: Optional[ThreatIntelConfig] = None,
        virustotal_api_key: Optional[str] = None,
        abuseipdb_api_key: Optional[str] = None,
        phishtank_api_key: Optional[str] = None,
    ):
        """
        Initialize threat intelligence service.

        Args:
            config: Full configuration object
            virustotal_api_key: VirusTotal API key (overrides config)
            abuseipdb_api_key: AbuseIPDB API key (overrides config)
            phishtank_api_key: PhishTank API key (overrides config)
        """
        self.config = config or ThreatIntelConfig()

        # Override with explicit keys
        if virustotal_api_key:
            self.config.virustotal_api_key = virustotal_api_key
        if abuseipdb_api_key:
            self.config.abuseipdb_api_key = abuseipdb_api_key
        if phishtank_api_key:
            self.config.phishtank_api_key = phishtank_api_key

        # Load from environment if not set
        if not self.config.virustotal_api_key:
            self.config.virustotal_api_key = os.getenv("VIRUSTOTAL_API_KEY")
        if not self.config.abuseipdb_api_key:
            self.config.abuseipdb_api_key = os.getenv("ABUSEIPDB_API_KEY")
        if not self.config.phishtank_api_key:
            self.config.phishtank_api_key = os.getenv("PHISHTANK_API_KEY")

        # Rate limiting state
        self._rate_limits: Dict[str, List[float]] = {
            "virustotal": [],
            "abuseipdb": [],
            "phishtank": [],
        }

        # Compile patterns
        self._malicious_patterns = [re.compile(p) for p in MALICIOUS_URL_PATTERNS]

        # Cache connection
        self._cache_conn: Optional[sqlite3.Connection] = None

        # HTTP session (lazy initialized)
        self._http_session = None

        logger.info(
            f"ThreatIntelligenceService initialized "
            f"(VT: {bool(self.config.virustotal_api_key)}, "
            f"AIPDB: {bool(self.config.abuseipdb_api_key)}, "
            f"PT: {bool(self.config.phishtank_api_key)})"
        )

    async def initialize(self) -> None:
        """Initialize the service (cache, etc.)."""
        if self.config.enable_caching:
            await self._init_cache()

    async def _init_cache(self) -> None:
        """Initialize SQLite cache."""
        try:
            self._cache_conn = sqlite3.connect(
                self.config.cache_db_path,
                check_same_thread=False,
            )
            cursor = self._cache_conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS threat_cache (
                    target_hash TEXT PRIMARY KEY,
                    target TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_threat_cache_expires
                ON threat_cache(expires_at)
            """)

            self._cache_conn.commit()
            logger.info(f"Threat intel cache initialized: {self.config.cache_db_path}")

        except Exception as e:
            logger.warning(f"Failed to initialize threat cache: {e}")

    def _get_cache_key(self, target: str, target_type: str) -> str:
        """Generate cache key for a target."""
        key = f"{target_type}:{target.lower()}"
        return hashlib.sha256(key.encode()).hexdigest()

    async def _get_cached(self, target: str, target_type: str) -> Optional[ThreatResult]:
        """Get cached result if available and not expired."""
        if not self._cache_conn or not self.config.enable_caching:
            return None

        try:
            cache_key = self._get_cache_key(target, target_type)
            cursor = self._cache_conn.cursor()

            cursor.execute(
                """
                SELECT result_json FROM threat_cache
                WHERE target_hash = ? AND expires_at > ?
                """,
                (cache_key, datetime.now().isoformat()),
            )

            row = cursor.fetchone()
            if row:
                data = json.loads(row[0])
                return ThreatResult(
                    target=data["target"],
                    target_type=data["target_type"],
                    is_malicious=data["is_malicious"],
                    threat_type=ThreatType(data["threat_type"]),
                    severity=ThreatSeverity[data["severity"]],
                    confidence=data["confidence"],
                    sources=[ThreatSource.CACHED],
                    details=data.get("details", {}),
                    checked_at=datetime.fromisoformat(data["checked_at"]),
                    cached=True,
                    virustotal_positives=data.get("virustotal_positives", 0),
                    virustotal_total=data.get("virustotal_total", 0),
                    abuseipdb_score=data.get("abuseipdb_score", 0),
                    phishtank_verified=data.get("phishtank_verified", False),
                )

        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")

        return None

    async def _set_cached(self, result: ThreatResult) -> None:
        """Cache a result."""
        if not self._cache_conn or not self.config.enable_caching:
            return

        try:
            cache_key = self._get_cache_key(result.target, result.target_type)
            expires_at = datetime.now() + timedelta(hours=self.config.cache_ttl_hours)

            result_json = json.dumps(
                {
                    "target": result.target,
                    "target_type": result.target_type,
                    "is_malicious": result.is_malicious,
                    "threat_type": result.threat_type.value,
                    "severity": result.severity.name,
                    "confidence": result.confidence,
                    "details": result.details,
                    "checked_at": result.checked_at.isoformat(),
                    "virustotal_positives": result.virustotal_positives,
                    "virustotal_total": result.virustotal_total,
                    "abuseipdb_score": result.abuseipdb_score,
                    "phishtank_verified": result.phishtank_verified,
                }
            )

            cursor = self._cache_conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO threat_cache
                (target_hash, target, target_type, result_json, expires_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    result.target,
                    result.target_type,
                    result_json,
                    expires_at.isoformat(),
                ),
            )
            self._cache_conn.commit()

        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")

    async def _check_rate_limit(self, service: str) -> bool:
        """Check if we're within rate limits for a service."""
        limits = {
            "virustotal": self.config.virustotal_rate_limit,
            "abuseipdb": self.config.abuseipdb_rate_limit,
            "phishtank": self.config.phishtank_rate_limit,
        }

        limit = limits.get(service, 60)
        now = time.time()
        minute_ago = now - 60

        # Clean old entries
        self._rate_limits[service] = [t for t in self._rate_limits[service] if t > minute_ago]

        # Check limit
        if len(self._rate_limits[service]) >= limit:
            return False

        # Record this request
        self._rate_limits[service].append(now)
        return True

    async def _get_http_session(self):
        """Get or create HTTP session."""
        if self._http_session is None:
            try:
                import aiohttp

                self._http_session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.config.request_timeout_seconds)
                )
            except ImportError:
                logger.warning("aiohttp not available, HTTP requests will fail")
                return None
        return self._http_session

    # =========================================================================
    # URL Checking
    # =========================================================================

    async def check_url(
        self,
        url: str,
        check_virustotal: bool = True,
        check_phishtank: bool = True,
    ) -> ThreatResult:
        """
        Check a URL against threat intelligence sources.

        Args:
            url: URL to check
            check_virustotal: Query VirusTotal
            check_phishtank: Query PhishTank

        Returns:
            ThreatResult with findings
        """
        # Check cache first
        cached = await self._get_cached(url, "url")
        if cached:
            return cached

        sources: List[ThreatSource] = []
        is_malicious = False
        threat_type = ThreatType.NONE
        severity = ThreatSeverity.NONE
        confidence = 0.0
        details: Dict[str, Any] = {}
        vt_positives = 0
        vt_total = 0
        pt_verified = False

        # Local pattern check first (fast)
        local_result = self._check_url_patterns(url)
        if local_result:
            sources.append(ThreatSource.LOCAL_RULES)
            is_malicious = True
            threat_type = local_result["threat_type"]
            severity = ThreatSeverity.MEDIUM
            confidence = 0.6
            details["local_match"] = local_result["pattern"]

        # VirusTotal check
        if check_virustotal and self.config.enable_virustotal and self.config.virustotal_api_key:
            vt_result = await self._check_url_virustotal(url)
            if vt_result:
                sources.append(ThreatSource.VIRUSTOTAL)
                vt_positives = vt_result.get("positives", 0)
                vt_total = vt_result.get("total", 0)
                details["virustotal"] = vt_result

                if vt_positives >= self.config.virustotal_malicious_threshold:
                    is_malicious = True
                    threat_type = self._classify_vt_threat(vt_result)
                    severity = self._vt_severity(vt_positives, vt_total)
                    confidence = max(confidence, vt_positives / max(vt_total, 1))

        # PhishTank check
        if check_phishtank and self.config.enable_phishtank:
            pt_result = await self._check_url_phishtank(url)
            if pt_result:
                sources.append(ThreatSource.PHISHTANK)
                pt_verified = pt_result.get("verified", False)
                details["phishtank"] = pt_result

                if pt_verified:
                    is_malicious = True
                    threat_type = ThreatType.PHISHING
                    severity = ThreatSeverity.HIGH
                    confidence = max(confidence, 0.95)

        # Build result
        result = ThreatResult(
            target=url,
            target_type="url",
            is_malicious=is_malicious,
            threat_type=threat_type,
            severity=severity,
            confidence=confidence,
            sources=sources if sources else [ThreatSource.LOCAL_RULES],
            details=details,
            virustotal_positives=vt_positives,
            virustotal_total=vt_total,
            phishtank_verified=pt_verified,
        )

        # Cache result
        await self._set_cached(result)

        return result

    def _check_url_patterns(self, url: str) -> Optional[Dict[str, Any]]:
        """Check URL against local malicious patterns."""
        for pattern in self._malicious_patterns:
            if pattern.search(url):
                return {
                    "threat_type": ThreatType.PHISHING,
                    "pattern": pattern.pattern,
                }

        # Check suspicious TLDs
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            for tld in SUSPICIOUS_TLDS:
                if domain.endswith(tld):
                    return {
                        "threat_type": ThreatType.SUSPICIOUS,
                        "pattern": f"suspicious_tld:{tld}",
                    }
        except Exception:
            pass

        return None

    async def _check_url_virustotal(self, url: str) -> Optional[Dict[str, Any]]:
        """Check URL with VirusTotal API."""
        if not await self._check_rate_limit("virustotal"):
            logger.warning("VirusTotal rate limit reached")
            return None

        session = await self._get_http_session()
        if not session:
            return None

        try:
            # VirusTotal uses base64 URL encoding
            url_id = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")

            headers = {"x-apikey": self.config.virustotal_api_key}

            async with session.get(
                f"https://www.virustotal.com/api/v3/urls/{url_id}",
                headers=headers,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    attrs = data.get("data", {}).get("attributes", {})
                    stats = attrs.get("last_analysis_stats", {})

                    return {
                        "positives": stats.get("malicious", 0) + stats.get("suspicious", 0),
                        "total": sum(stats.values()),
                        "categories": attrs.get("categories", {}),
                        "reputation": attrs.get("reputation", 0),
                        "last_analysis_date": attrs.get("last_analysis_date"),
                    }
                elif response.status == 404:
                    # URL not in database, submit for scanning
                    return await self._submit_url_virustotal(url)

        except Exception as e:
            logger.warning(f"VirusTotal URL check failed: {e}")

        return None

    async def _submit_url_virustotal(self, url: str) -> Optional[Dict[str, Any]]:
        """Submit a URL to VirusTotal for scanning."""
        if not await self._check_rate_limit("virustotal"):
            return None

        session = await self._get_http_session()
        if not session:
            return None

        try:
            headers = {
                "x-apikey": self.config.virustotal_api_key,
                "Content-Type": "application/x-www-form-urlencoded",
            }

            async with session.post(
                "https://www.virustotal.com/api/v3/urls",
                headers=headers,
                data={"url": url},
            ) as response:
                if response.status == 200:
                    # Scan submitted, return minimal result
                    return {
                        "positives": 0,
                        "total": 0,
                        "pending": True,
                    }

        except Exception as e:
            logger.warning(f"VirusTotal URL submit failed: {e}")

        return None

    async def _check_url_phishtank(self, url: str) -> Optional[Dict[str, Any]]:
        """Check URL with PhishTank API."""
        if not await self._check_rate_limit("phishtank"):
            logger.warning("PhishTank rate limit reached")
            return None

        session = await self._get_http_session()
        if not session:
            return None

        try:
            # PhishTank uses POST with form data
            data = {
                "url": url,
                "format": "json",
            }
            if self.config.phishtank_api_key:
                data["app_key"] = self.config.phishtank_api_key

            async with session.post(
                "https://checkurl.phishtank.com/checkurl/",
                data=data,
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    results = result.get("results", {})

                    return {
                        "in_database": results.get("in_database", False),
                        "verified": results.get("verified", False),
                        "phish_id": results.get("phish_id"),
                        "phish_detail_url": results.get("phish_detail_page"),
                    }

        except Exception as e:
            logger.warning(f"PhishTank check failed: {e}")

        return None

    def _classify_vt_threat(self, vt_result: Dict[str, Any]) -> ThreatType:
        """Classify threat type from VirusTotal result."""
        categories = vt_result.get("categories", {})

        # Check category values
        for vendor, category in categories.items():
            category_lower = category.lower() if category else ""
            if "phish" in category_lower:
                return ThreatType.PHISHING
            if "malware" in category_lower:
                return ThreatType.MALWARE
            if "spam" in category_lower:
                return ThreatType.SPAM

        # Default based on positives
        if vt_result.get("positives", 0) > 0:
            return ThreatType.SUSPICIOUS

        return ThreatType.NONE

    def _vt_severity(self, positives: int, total: int) -> ThreatSeverity:
        """Calculate severity from VirusTotal results."""
        if total == 0:
            return ThreatSeverity.NONE

        ratio = positives / total

        if ratio >= 0.5:
            return ThreatSeverity.CRITICAL
        elif ratio >= 0.25:
            return ThreatSeverity.HIGH
        elif ratio >= 0.1:
            return ThreatSeverity.MEDIUM
        elif positives > 0:
            return ThreatSeverity.LOW

        return ThreatSeverity.NONE

    # =========================================================================
    # IP Checking
    # =========================================================================

    async def check_ip(self, ip_address: str) -> IPReputationResult:
        """
        Check an IP address against threat intelligence sources.

        Args:
            ip_address: IP address to check

        Returns:
            IPReputationResult with reputation data
        """
        # Validate IP
        if not self._is_valid_ip(ip_address):
            return IPReputationResult(
                ip_address=ip_address,
                is_malicious=False,
                abuse_score=0,
                total_reports=0,
                last_reported=None,
                country_code=None,
                isp=None,
                domain=None,
                usage_type=None,
            )

        # Check AbuseIPDB
        if self.config.enable_abuseipdb and self.config.abuseipdb_api_key:
            return await self._check_ip_abuseipdb(ip_address)

        # Fallback to empty result
        return IPReputationResult(
            ip_address=ip_address,
            is_malicious=False,
            abuse_score=0,
            total_reports=0,
            last_reported=None,
            country_code=None,
            isp=None,
            domain=None,
            usage_type=None,
        )

    def _is_valid_ip(self, ip: str) -> bool:
        """Check if string is a valid IP address."""
        # IPv4
        ipv4_pattern = r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
        if re.match(ipv4_pattern, ip):
            return True

        # IPv6 (simplified)
        if ":" in ip and all(c in "0123456789abcdefABCDEF:" for c in ip):
            return True

        return False

    async def _check_ip_abuseipdb(self, ip_address: str) -> IPReputationResult:
        """Check IP with AbuseIPDB API."""
        if not await self._check_rate_limit("abuseipdb"):
            logger.warning("AbuseIPDB rate limit reached")
            return IPReputationResult(
                ip_address=ip_address,
                is_malicious=False,
                abuse_score=0,
                total_reports=0,
                last_reported=None,
                country_code=None,
                isp=None,
                domain=None,
                usage_type=None,
            )

        session = await self._get_http_session()
        if not session:
            return IPReputationResult(
                ip_address=ip_address,
                is_malicious=False,
                abuse_score=0,
                total_reports=0,
                last_reported=None,
                country_code=None,
                isp=None,
                domain=None,
                usage_type=None,
            )

        try:
            headers = {
                "Key": self.config.abuseipdb_api_key,
                "Accept": "application/json",
            }

            params = {
                "ipAddress": ip_address,
                "maxAgeInDays": 90,
                "verbose": "",
            }

            async with session.get(
                "https://api.abuseipdb.com/api/v2/check",
                headers=headers,
                params=params,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get("data", {})

                    abuse_score = result.get("abuseConfidenceScore", 0)
                    last_reported = None
                    if result.get("lastReportedAt"):
                        try:
                            last_reported = datetime.fromisoformat(
                                result["lastReportedAt"].replace("Z", "+00:00")
                            )
                        except ValueError:
                            pass

                    return IPReputationResult(
                        ip_address=ip_address,
                        is_malicious=abuse_score >= self.config.abuseipdb_malicious_threshold,
                        abuse_score=abuse_score,
                        total_reports=result.get("totalReports", 0),
                        last_reported=last_reported,
                        country_code=result.get("countryCode"),
                        isp=result.get("isp"),
                        domain=result.get("domain"),
                        usage_type=result.get("usageType"),
                        is_tor=result.get("isTor", False),
                    )

        except Exception as e:
            logger.warning(f"AbuseIPDB check failed: {e}")

        return IPReputationResult(
            ip_address=ip_address,
            is_malicious=False,
            abuse_score=0,
            total_reports=0,
            last_reported=None,
            country_code=None,
            isp=None,
            domain=None,
            usage_type=None,
        )

    # =========================================================================
    # File Hash Checking
    # =========================================================================

    async def check_file_hash(
        self,
        hash_value: str,
        hash_type: Optional[str] = None,
    ) -> FileHashResult:
        """
        Check a file hash against VirusTotal.

        Args:
            hash_value: MD5, SHA1, or SHA256 hash
            hash_type: Optional hash type hint

        Returns:
            FileHashResult with malware information
        """
        # Auto-detect hash type
        if not hash_type:
            hash_type = self._detect_hash_type(hash_value)

        if not hash_type:
            return FileHashResult(
                hash_value=hash_value,
                hash_type="unknown",
                is_malware=False,
            )

        # Check VirusTotal
        if self.config.enable_virustotal and self.config.virustotal_api_key:
            return await self._check_hash_virustotal(hash_value, hash_type)

        return FileHashResult(
            hash_value=hash_value,
            hash_type=hash_type,
            is_malware=False,
        )

    def _detect_hash_type(self, hash_value: str) -> Optional[str]:
        """Detect hash type from length."""
        hash_value = hash_value.strip().lower()

        if not all(c in "0123456789abcdef" for c in hash_value):
            return None

        if len(hash_value) == 32:
            return "md5"
        elif len(hash_value) == 40:
            return "sha1"
        elif len(hash_value) == 64:
            return "sha256"

        return None

    async def _check_hash_virustotal(self, hash_value: str, hash_type: str) -> FileHashResult:
        """Check file hash with VirusTotal API."""
        if not await self._check_rate_limit("virustotal"):
            logger.warning("VirusTotal rate limit reached")
            return FileHashResult(
                hash_value=hash_value,
                hash_type=hash_type,
                is_malware=False,
            )

        session = await self._get_http_session()
        if not session:
            return FileHashResult(
                hash_value=hash_value,
                hash_type=hash_type,
                is_malware=False,
            )

        try:
            headers = {"x-apikey": self.config.virustotal_api_key}

            async with session.get(
                f"https://www.virustotal.com/api/v3/files/{hash_value}",
                headers=headers,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    attrs = data.get("data", {}).get("attributes", {})
                    stats = attrs.get("last_analysis_stats", {})

                    malicious = stats.get("malicious", 0)
                    suspicious = stats.get("suspicious", 0)
                    total = sum(stats.values())

                    # Get malware names from detections
                    malware_names = []
                    results = attrs.get("last_analysis_results", {})
                    for vendor, result in results.items():
                        if result.get("category") == "malicious" and result.get("result"):
                            malware_names.append(result["result"])

                    # Parse dates
                    first_seen = None
                    last_seen = None
                    if attrs.get("first_submission_date"):
                        first_seen = datetime.fromtimestamp(attrs["first_submission_date"])
                    if attrs.get("last_analysis_date"):
                        last_seen = datetime.fromtimestamp(attrs["last_analysis_date"])

                    return FileHashResult(
                        hash_value=hash_value,
                        hash_type=hash_type,
                        is_malware=malicious >= self.config.virustotal_malicious_threshold,
                        malware_names=malware_names[:10],  # Limit names
                        detection_ratio=f"{malicious + suspicious}/{total}",
                        first_seen=first_seen,
                        last_seen=last_seen,
                        file_type=attrs.get("type_description"),
                        file_size=attrs.get("size"),
                        tags=attrs.get("tags", [])[:10],
                    )

                elif response.status == 404:
                    # Hash not found
                    return FileHashResult(
                        hash_value=hash_value,
                        hash_type=hash_type,
                        is_malware=False,
                    )

        except Exception as e:
            logger.warning(f"VirusTotal hash check failed: {e}")

        return FileHashResult(
            hash_value=hash_value,
            hash_type=hash_type,
            is_malware=False,
        )

    # =========================================================================
    # Batch Operations
    # =========================================================================

    async def check_urls_batch(
        self,
        urls: List[str],
        max_concurrent: int = 5,
    ) -> List[ThreatResult]:
        """
        Check multiple URLs concurrently.

        Args:
            urls: List of URLs to check
            max_concurrent: Maximum concurrent checks

        Returns:
            List of ThreatResults
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def check_with_semaphore(url: str) -> ThreatResult:
            async with semaphore:
                return await self.check_url(url)

        tasks = [check_with_semaphore(url) for url in urls]
        return await asyncio.gather(*tasks)

    async def check_email_content(
        self,
        email_body: str,
        email_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Check email content for threats.

        Extracts and checks:
        - URLs in body
        - Sender IP (from headers)
        - Attachment hashes (if provided)

        Args:
            email_body: Email body content
            email_headers: Optional email headers

        Returns:
            Dictionary with threat findings
        """
        results: Dict[str, Any] = {
            "urls": [],
            "ips": [],
            "overall_threat_score": 0,
            "is_suspicious": False,
            "threat_summary": [],
        }

        # Extract URLs from body
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = list(set(re.findall(url_pattern, email_body)))

        if urls:
            url_results = await self.check_urls_batch(urls[:10])  # Limit to 10
            results["urls"] = [r.to_dict() for r in url_results]

            malicious_urls = [r for r in url_results if r.is_malicious]
            if malicious_urls:
                results["is_suspicious"] = True
                results["threat_summary"].append(f"Found {len(malicious_urls)} malicious URLs")

        # Check sender IP from headers
        if email_headers:
            # Extract IP from Received headers
            received = email_headers.get("Received", "")
            ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
            ips = re.findall(ip_pattern, received)

            for ip in ips[:3]:  # Check first 3 IPs
                # Skip private IPs
                if ip.startswith(("10.", "192.168.", "172.")):
                    continue
                ip_result = await self.check_ip(ip)
                results["ips"].append(ip_result.to_dict())

                if ip_result.is_malicious:
                    results["is_suspicious"] = True
                    results["threat_summary"].append(
                        f"Sender IP {ip} has abuse score {ip_result.abuse_score}"
                    )

        # Calculate overall threat score
        scores = []
        for url_result in results["urls"]:
            scores.append(url_result.get("threat_score", 0))
        for ip_result in results["ips"]:
            scores.append(ip_result.get("abuse_score", 0))

        if scores:
            results["overall_threat_score"] = sum(scores) / len(scores)

        return results

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def cleanup_cache(self, older_than_hours: int = 168) -> int:
        """
        Clean up expired cache entries.

        Args:
            older_than_hours: Delete entries older than this

        Returns:
            Number of entries deleted
        """
        if not self._cache_conn:
            return 0

        try:
            cutoff = datetime.now() - timedelta(hours=older_than_hours)
            cursor = self._cache_conn.cursor()

            cursor.execute(
                "DELETE FROM threat_cache WHERE created_at < ?",
                (cutoff.isoformat(),),
            )
            deleted = cursor.rowcount
            self._cache_conn.commit()

            logger.info(f"Cleaned up {deleted} expired cache entries")
            return deleted

        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")
            return 0

    async def close(self) -> None:
        """Close service connections."""
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

        if self._cache_conn:
            self._cache_conn.close()
            self._cache_conn = None


# Convenience function for quick checks
async def check_threat(
    target: str,
    target_type: str = "auto",
    virustotal_api_key: Optional[str] = None,
    abuseipdb_api_key: Optional[str] = None,
) -> ThreatResult:
    """
    Quick convenience function for threat checking.

    Args:
        target: URL, IP, or hash to check
        target_type: "url", "ip", "hash", or "auto" to detect
        virustotal_api_key: Optional API key
        abuseipdb_api_key: Optional API key

    Returns:
        ThreatResult
    """
    service = ThreatIntelligenceService(
        virustotal_api_key=virustotal_api_key,
        abuseipdb_api_key=abuseipdb_api_key,
    )

    try:
        # Auto-detect target type
        if target_type == "auto":
            if target.startswith(("http://", "https://")):
                target_type = "url"
            elif service._is_valid_ip(target):
                target_type = "ip"
            elif service._detect_hash_type(target):
                target_type = "hash"
            else:
                target_type = "url"  # Default

        if target_type == "url":
            return await service.check_url(target)
        elif target_type == "ip":
            ip_result = await service.check_ip(target)
            return ThreatResult(
                target=target,
                target_type="ip",
                is_malicious=ip_result.is_malicious,
                threat_type=ThreatType.MALICIOUS_IP if ip_result.is_malicious else ThreatType.NONE,
                severity=ThreatSeverity.HIGH if ip_result.is_malicious else ThreatSeverity.NONE,
                confidence=ip_result.abuse_score / 100,
                sources=[ThreatSource.ABUSEIPDB],
                abuseipdb_score=ip_result.abuse_score,
            )
        elif target_type == "hash":
            hash_result = await service.check_file_hash(target)
            return ThreatResult(
                target=target,
                target_type="hash",
                is_malicious=hash_result.is_malware,
                threat_type=ThreatType.MALWARE if hash_result.is_malware else ThreatType.NONE,
                severity=ThreatSeverity.CRITICAL if hash_result.is_malware else ThreatSeverity.NONE,
                confidence=0.9 if hash_result.is_malware else 0.0,
                sources=[ThreatSource.VIRUSTOTAL],
                details={"malware_names": hash_result.malware_names},
            )

        # Fallback
        return ThreatResult(
            target=target,
            target_type=target_type,
            is_malicious=False,
            threat_type=ThreatType.NONE,
            severity=ThreatSeverity.NONE,
            confidence=0.0,
            sources=[],
        )

    finally:
        await service.close()
