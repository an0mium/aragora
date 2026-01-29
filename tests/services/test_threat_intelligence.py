"""
Tests for Threat Intelligence Service.

Tests cover:
- Enum values (ThreatType, ThreatSeverity, ThreatSource)
- Dataclasses (ThreatResult, SourceResult, ThreatAssessment, etc.)
- ThreatIntelligenceService initialization
- Local URL pattern detection
- IP address validation
- Hash type detection
- Rate limiting
- Severity calculation
- Aggregate risk calculation
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Enum Tests
# =============================================================================


class TestThreatType:
    """Tests for ThreatType enum."""

    def test_threat_type_values(self):
        """Test ThreatType enum values."""
        from aragora.services.threat_intelligence import ThreatType

        assert ThreatType.NONE.value == "none"
        assert ThreatType.MALWARE.value == "malware"
        assert ThreatType.PHISHING.value == "phishing"
        assert ThreatType.SPAM.value == "spam"
        assert ThreatType.SUSPICIOUS.value == "suspicious"
        assert ThreatType.MALICIOUS_IP.value == "malicious_ip"
        assert ThreatType.COMMAND_AND_CONTROL.value == "c2"
        assert ThreatType.BOTNET.value == "botnet"
        assert ThreatType.CRYPTO_MINER.value == "crypto_miner"
        assert ThreatType.RANSOMWARE.value == "ransomware"
        assert ThreatType.TROJAN.value == "trojan"


class TestThreatSeverity:
    """Tests for ThreatSeverity enum."""

    def test_severity_values(self):
        """Test ThreatSeverity enum values."""
        from aragora.services.threat_intelligence import ThreatSeverity

        assert ThreatSeverity.NONE.value == 0
        assert ThreatSeverity.LOW.value == 1
        assert ThreatSeverity.MEDIUM.value == 2
        assert ThreatSeverity.HIGH.value == 3
        assert ThreatSeverity.CRITICAL.value == 4

    def test_severity_ordering(self):
        """Test severity values are properly ordered."""
        from aragora.services.threat_intelligence import ThreatSeverity

        assert ThreatSeverity.NONE.value < ThreatSeverity.LOW.value
        assert ThreatSeverity.LOW.value < ThreatSeverity.MEDIUM.value
        assert ThreatSeverity.MEDIUM.value < ThreatSeverity.HIGH.value
        assert ThreatSeverity.HIGH.value < ThreatSeverity.CRITICAL.value


class TestThreatSource:
    """Tests for ThreatSource enum."""

    def test_source_values(self):
        """Test ThreatSource enum values."""
        from aragora.services.threat_intelligence import ThreatSource

        assert ThreatSource.VIRUSTOTAL.value == "virustotal"
        assert ThreatSource.ABUSEIPDB.value == "abuseipdb"
        assert ThreatSource.PHISHTANK.value == "phishtank"
        assert ThreatSource.URLHAUS.value == "urlhaus"
        assert ThreatSource.LOCAL_RULES.value == "local_rules"
        assert ThreatSource.CACHED.value == "cached"


# =============================================================================
# ThreatResult Tests
# =============================================================================


class TestThreatResult:
    """Tests for ThreatResult dataclass."""

    def test_threat_result_creation(self):
        """Test ThreatResult creation."""
        from aragora.services.threat_intelligence import (
            ThreatResult,
            ThreatSeverity,
            ThreatSource,
            ThreatType,
        )

        result = ThreatResult(
            target="http://malicious.com",
            target_type="url",
            is_malicious=True,
            threat_type=ThreatType.PHISHING,
            severity=ThreatSeverity.HIGH,
            confidence=0.9,
            sources=[ThreatSource.VIRUSTOTAL],
        )

        assert result.target == "http://malicious.com"
        assert result.is_malicious is True
        assert result.threat_type == ThreatType.PHISHING
        assert result.severity == ThreatSeverity.HIGH
        assert result.confidence == 0.9

    def test_threat_score_with_virustotal(self):
        """Test threat_score calculation with VirusTotal data."""
        from aragora.services.threat_intelligence import (
            ThreatResult,
            ThreatSeverity,
            ThreatSource,
            ThreatType,
        )

        result = ThreatResult(
            target="http://test.com",
            target_type="url",
            is_malicious=True,
            threat_type=ThreatType.MALWARE,
            severity=ThreatSeverity.HIGH,
            confidence=0.8,
            sources=[ThreatSource.VIRUSTOTAL],
            virustotal_positives=20,
            virustotal_total=70,
        )

        expected_score = (20 / 70) * 100
        assert abs(result.threat_score - expected_score) < 0.1

    def test_threat_score_with_multiple_sources(self):
        """Test threat_score with multiple sources."""
        from aragora.services.threat_intelligence import (
            ThreatResult,
            ThreatSeverity,
            ThreatSource,
            ThreatType,
        )

        result = ThreatResult(
            target="http://test.com",
            target_type="url",
            is_malicious=True,
            threat_type=ThreatType.PHISHING,
            severity=ThreatSeverity.CRITICAL,
            confidence=0.95,
            sources=[ThreatSource.VIRUSTOTAL, ThreatSource.PHISHTANK],
            virustotal_positives=10,
            virustotal_total=50,
            phishtank_verified=True,
        )

        # Score should be average of VT (20%) and PhishTank (100%)
        assert result.threat_score == 60.0

    def test_threat_score_empty(self):
        """Test threat_score with no data."""
        from aragora.services.threat_intelligence import (
            ThreatResult,
            ThreatSeverity,
            ThreatSource,
            ThreatType,
        )

        result = ThreatResult(
            target="http://test.com",
            target_type="url",
            is_malicious=False,
            threat_type=ThreatType.NONE,
            severity=ThreatSeverity.NONE,
            confidence=0.0,
            sources=[],
        )

        assert result.threat_score == 0.0

    def test_threat_result_to_dict(self):
        """Test ThreatResult.to_dict()."""
        from aragora.services.threat_intelligence import (
            ThreatResult,
            ThreatSeverity,
            ThreatSource,
            ThreatType,
        )

        result = ThreatResult(
            target="http://test.com",
            target_type="url",
            is_malicious=True,
            threat_type=ThreatType.MALWARE,
            severity=ThreatSeverity.HIGH,
            confidence=0.8,
            sources=[ThreatSource.VIRUSTOTAL],
        )

        d = result.to_dict()
        assert d["target"] == "http://test.com"
        assert d["is_malicious"] is True
        assert d["threat_type"] == "malware"
        assert d["severity"] == "HIGH"
        assert d["severity_value"] == 3
        assert "checked_at" in d


# =============================================================================
# SourceResult Tests
# =============================================================================


class TestSourceResult:
    """Tests for SourceResult dataclass."""

    def test_source_result_creation(self):
        """Test SourceResult creation."""
        from aragora.services.threat_intelligence import SourceResult, ThreatSource

        result = SourceResult(
            source=ThreatSource.VIRUSTOTAL,
            is_malicious=True,
            confidence=0.85,
            threat_types=["malware"],
            raw_score=15.0,
        )

        assert result.source == ThreatSource.VIRUSTOTAL
        assert result.is_malicious is True
        assert result.confidence == 0.85
        assert "malware" in result.threat_types

    def test_source_result_to_dict(self):
        """Test SourceResult.to_dict()."""
        from aragora.services.threat_intelligence import SourceResult, ThreatSource

        result = SourceResult(
            source=ThreatSource.PHISHTANK,
            is_malicious=True,
            confidence=0.95,
            error=None,
        )

        d = result.to_dict()
        assert d["source"] == "phishtank"
        assert d["is_malicious"] is True
        assert d["confidence"] == 0.95
        assert d["error"] is None


# =============================================================================
# ThreatAssessment Tests
# =============================================================================


class TestThreatAssessment:
    """Tests for ThreatAssessment dataclass."""

    def test_assessment_creation(self):
        """Test ThreatAssessment creation."""
        from aragora.services.threat_intelligence import ThreatAssessment

        assessment = ThreatAssessment(
            target="http://test.com",
            target_type="url",
            overall_risk=0.75,
            is_malicious=True,
            threat_types=["phishing", "malware"],
        )

        assert assessment.target == "http://test.com"
        assert assessment.overall_risk == 0.75
        assert assessment.is_malicious is True
        assert len(assessment.threat_types) == 2

    def test_assessment_to_dict(self):
        """Test ThreatAssessment.to_dict()."""
        from aragora.services.threat_intelligence import ThreatAssessment

        assessment = ThreatAssessment(
            target="192.168.1.1",
            target_type="ip",
            overall_risk=0.6,
            is_malicious=True,
            max_source_risk=0.8,
            avg_source_risk=0.5,
            source_agreement=0.75,
            sources_checked=4,
            sources_responding=3,
        )

        d = assessment.to_dict()
        assert d["overall_risk"] == 0.6
        assert d["risk_breakdown"]["max_source_risk"] == 0.8
        assert d["metadata"]["sources_checked"] == 4
        assert d["metadata"]["sources_responding"] == 3


# =============================================================================
# IPReputationResult Tests
# =============================================================================


class TestIPReputationResult:
    """Tests for IPReputationResult dataclass."""

    def test_ip_result_creation(self):
        """Test IPReputationResult creation."""
        from aragora.services.threat_intelligence import IPReputationResult

        result = IPReputationResult(
            ip_address="1.2.3.4",
            is_malicious=True,
            abuse_score=85,
            total_reports=100,
            last_reported=datetime.now(),
            country_code="US",
            isp="Test ISP",
            domain="test.com",
            usage_type="Data Center",
            is_tor=False,
            is_vpn=True,
        )

        assert result.ip_address == "1.2.3.4"
        assert result.is_malicious is True
        assert result.abuse_score == 85
        assert result.is_vpn is True

    def test_ip_result_to_dict(self):
        """Test IPReputationResult.to_dict()."""
        from aragora.services.threat_intelligence import IPReputationResult

        result = IPReputationResult(
            ip_address="8.8.8.8",
            is_malicious=False,
            abuse_score=0,
            total_reports=0,
            last_reported=None,
            country_code="US",
            isp="Google",
            domain="google.com",
            usage_type="Data Center",
        )

        d = result.to_dict()
        assert d["ip_address"] == "8.8.8.8"
        assert d["is_malicious"] is False
        assert d["country_code"] == "US"


# =============================================================================
# FileHashResult Tests
# =============================================================================


class TestFileHashResult:
    """Tests for FileHashResult dataclass."""

    def test_hash_result_creation(self):
        """Test FileHashResult creation."""
        from aragora.services.threat_intelligence import FileHashResult

        result = FileHashResult(
            hash_value="abc123" * 10 + "abcd",  # 64 chars
            hash_type="sha256",
            is_malware=True,
            malware_names=["Trojan.Generic", "Win32.Malware"],
            detection_ratio="45/70",
        )

        assert result.is_malware is True
        assert len(result.malware_names) == 2
        assert result.detection_ratio == "45/70"


# =============================================================================
# ThreatIntelConfig Tests
# =============================================================================


class TestThreatIntelConfig:
    """Tests for ThreatIntelConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from aragora.services.threat_intelligence import ThreatIntelConfig

        config = ThreatIntelConfig()

        assert config.virustotal_rate_limit == 4
        assert config.abuseipdb_rate_limit == 60
        assert config.cache_ttl_hours == 24
        assert config.cache_ip_ttl_hours == 1
        assert config.virustotal_malicious_threshold == 3
        assert config.abuseipdb_malicious_threshold == 50
        assert config.high_risk_threshold == 0.7
        assert config.enable_caching is True

    def test_custom_config(self):
        """Test custom configuration."""
        from aragora.services.threat_intelligence import ThreatIntelConfig

        config = ThreatIntelConfig(
            virustotal_api_key="test-key",
            virustotal_malicious_threshold=5,
            cache_ttl_hours=48,
            enable_caching=False,
        )

        assert config.virustotal_api_key == "test-key"
        assert config.virustotal_malicious_threshold == 5
        assert config.cache_ttl_hours == 48
        assert config.enable_caching is False

    def test_source_weights(self):
        """Test default source weights."""
        from aragora.services.threat_intelligence import ThreatIntelConfig

        config = ThreatIntelConfig()

        assert config.source_weights["virustotal"] == 0.9
        assert config.source_weights["abuseipdb"] == 0.8
        assert config.source_weights["phishtank"] == 0.85
        assert config.source_weights["urlhaus"] == 0.85
        assert config.source_weights["local_rules"] == 0.5


# =============================================================================
# ThreatIntelligenceService Tests
# =============================================================================


class TestThreatIntelligenceServiceInit:
    """Tests for ThreatIntelligenceService initialization."""

    def test_init_with_defaults(self):
        """Test initialization with defaults."""
        from aragora.services.threat_intelligence import ThreatIntelligenceService

        with patch.dict("os.environ", {}, clear=True):
            service = ThreatIntelligenceService()

        assert service.config is not None
        assert service.config.enable_caching is True

    def test_init_with_api_keys(self):
        """Test initialization with API keys."""
        from aragora.services.threat_intelligence import ThreatIntelligenceService

        service = ThreatIntelligenceService(
            virustotal_api_key="vt-key",
            abuseipdb_api_key="aipdb-key",
        )

        assert service.config.virustotal_api_key == "vt-key"
        assert service.config.abuseipdb_api_key == "aipdb-key"

    def test_init_with_config(self):
        """Test initialization with config object."""
        from aragora.services.threat_intelligence import (
            ThreatIntelConfig,
            ThreatIntelligenceService,
        )

        config = ThreatIntelConfig(
            virustotal_api_key="custom-key",
            cache_ttl_hours=12,
        )

        service = ThreatIntelligenceService(config=config)

        assert service.config.virustotal_api_key == "custom-key"
        assert service.config.cache_ttl_hours == 12

    def test_init_loads_from_env(self):
        """Test initialization loads API keys from environment."""
        from aragora.services.threat_intelligence import ThreatIntelligenceService

        with patch.dict(
            "os.environ",
            {
                "VIRUSTOTAL_API_KEY": "env-vt-key",
                "ABUSEIPDB_API_KEY": "env-aipdb-key",
            },
        ):
            service = ThreatIntelligenceService()

        assert service.config.virustotal_api_key == "env-vt-key"
        assert service.config.abuseipdb_api_key == "env-aipdb-key"


class TestURLPatternDetection:
    """Tests for local URL pattern detection."""

    @pytest.fixture
    def service(self):
        """Create service for pattern tests."""
        from aragora.services.threat_intelligence import ThreatIntelligenceService

        return ThreatIntelligenceService()

    def test_detect_paypal_phishing(self, service):
        """Test detection of PayPal phishing URL."""
        result = service._check_url_patterns("http://paypal-verify.malicious.com/login")

        assert result is not None
        assert result["threat_type"].value == "phishing"

    def test_detect_google_phishing(self, service):
        """Test detection of Google phishing URL."""
        result = service._check_url_patterns("http://google-login.fake.com/signin")

        assert result is not None
        assert result["threat_type"].value == "phishing"

    def test_detect_suspicious_tld(self, service):
        """Test detection of suspicious TLD."""
        result = service._check_url_patterns("http://free-stuff.tk/download")

        assert result is not None
        assert result["threat_type"].value == "suspicious"

    def test_detect_suspicious_xyz_tld(self, service):
        """Test detection of .xyz TLD."""
        result = service._check_url_patterns("http://scam-site.xyz/offer")

        assert result is not None
        assert result["threat_type"].value == "suspicious"

    def test_legitimate_url_no_match(self, service):
        """Test legitimate URL doesn't match."""
        result = service._check_url_patterns("https://www.google.com/search")

        assert result is None

    def test_pattern_matches_subdomain_abuse(self, service):
        """Test pattern catches subdomain abuse (paypal.fake.com)."""
        # Pattern is designed to catch paypal.fake-domain.com style URLs
        result = service._check_url_patterns("https://paypal.login-verify.com/signin")

        assert result is not None
        assert result["threat_type"].value == "phishing"


class TestIPValidation:
    """Tests for IP address validation."""

    @pytest.fixture
    def service(self):
        """Create service for IP validation tests."""
        from aragora.services.threat_intelligence import ThreatIntelligenceService

        return ThreatIntelligenceService()

    def test_valid_ipv4(self, service):
        """Test valid IPv4 addresses."""
        assert service._is_valid_ip("192.168.1.1") is True
        assert service._is_valid_ip("10.0.0.1") is True
        assert service._is_valid_ip("8.8.8.8") is True
        assert service._is_valid_ip("255.255.255.255") is True
        assert service._is_valid_ip("0.0.0.0") is True

    def test_invalid_ipv4(self, service):
        """Test invalid IPv4 addresses."""
        assert service._is_valid_ip("256.1.1.1") is False
        assert service._is_valid_ip("192.168.1") is False
        assert service._is_valid_ip("192.168.1.1.1") is False
        assert service._is_valid_ip("not.an.ip.address") is False
        assert service._is_valid_ip("") is False

    def test_valid_ipv6_simplified(self, service):
        """Test valid IPv6 addresses (simplified check)."""
        # The service uses simplified IPv6 validation
        assert service._is_valid_ip("::1") is True
        assert service._is_valid_ip("2001:db8::1") is True


class TestHashTypeDetection:
    """Tests for hash type detection."""

    @pytest.fixture
    def service(self):
        """Create service for hash detection tests."""
        from aragora.services.threat_intelligence import ThreatIntelligenceService

        return ThreatIntelligenceService()

    def test_detect_md5(self, service):
        """Test MD5 hash detection (32 chars)."""
        md5_hash = "d41d8cd98f00b204e9800998ecf8427e"
        assert service._detect_hash_type(md5_hash) == "md5"

    def test_detect_sha1(self, service):
        """Test SHA1 hash detection (40 chars)."""
        sha1_hash = "da39a3ee5e6b4b0d3255bfef95601890afd80709"
        assert service._detect_hash_type(sha1_hash) == "sha1"

    def test_detect_sha256(self, service):
        """Test SHA256 hash detection (64 chars)."""
        sha256_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert service._detect_hash_type(sha256_hash) == "sha256"

    def test_invalid_hash(self, service):
        """Test invalid hash detection."""
        assert service._detect_hash_type("not-a-hash") is None
        assert service._detect_hash_type("") is None
        assert service._detect_hash_type("12345") is None

    def test_hash_with_uppercase(self, service):
        """Test hash with uppercase characters."""
        sha256_upper = "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855"
        assert service._detect_hash_type(sha256_upper) == "sha256"


class TestVirusTotalSeverity:
    """Tests for VirusTotal severity calculation."""

    @pytest.fixture
    def service(self):
        """Create service for severity tests."""
        from aragora.services.threat_intelligence import ThreatIntelligenceService

        return ThreatIntelligenceService()

    def test_severity_critical(self, service):
        """Test CRITICAL severity (>=50% detection)."""
        from aragora.services.threat_intelligence import ThreatSeverity

        assert service._vt_severity(50, 100) == ThreatSeverity.CRITICAL
        assert service._vt_severity(35, 70) == ThreatSeverity.CRITICAL

    def test_severity_high(self, service):
        """Test HIGH severity (>=25% detection)."""
        from aragora.services.threat_intelligence import ThreatSeverity

        assert service._vt_severity(25, 100) == ThreatSeverity.HIGH
        # 18/70 = 25.7%, should be HIGH
        assert service._vt_severity(18, 70) == ThreatSeverity.HIGH

    def test_severity_medium(self, service):
        """Test MEDIUM severity (>=10% detection)."""
        from aragora.services.threat_intelligence import ThreatSeverity

        assert service._vt_severity(10, 100) == ThreatSeverity.MEDIUM
        assert service._vt_severity(7, 70) == ThreatSeverity.MEDIUM

    def test_severity_low(self, service):
        """Test LOW severity (>0 positives but <10%)."""
        from aragora.services.threat_intelligence import ThreatSeverity

        assert service._vt_severity(5, 100) == ThreatSeverity.LOW
        assert service._vt_severity(1, 70) == ThreatSeverity.LOW

    def test_severity_none(self, service):
        """Test NONE severity (no positives)."""
        from aragora.services.threat_intelligence import ThreatSeverity

        assert service._vt_severity(0, 100) == ThreatSeverity.NONE
        assert service._vt_severity(0, 0) == ThreatSeverity.NONE


class TestThreatClassification:
    """Tests for threat classification from VirusTotal results."""

    @pytest.fixture
    def service(self):
        """Create service for classification tests."""
        from aragora.services.threat_intelligence import ThreatIntelligenceService

        return ThreatIntelligenceService()

    def test_classify_phishing(self, service):
        """Test phishing classification."""
        from aragora.services.threat_intelligence import ThreatType

        vt_result = {"categories": {"Vendor1": "phishing"}, "positives": 10}
        assert service._classify_vt_threat(vt_result) == ThreatType.PHISHING

    def test_classify_malware(self, service):
        """Test malware classification."""
        from aragora.services.threat_intelligence import ThreatType

        vt_result = {"categories": {"Vendor1": "malware"}, "positives": 10}
        assert service._classify_vt_threat(vt_result) == ThreatType.MALWARE

    def test_classify_spam(self, service):
        """Test spam classification."""
        from aragora.services.threat_intelligence import ThreatType

        vt_result = {"categories": {"Vendor1": "spam"}, "positives": 10}
        assert service._classify_vt_threat(vt_result) == ThreatType.SPAM

    def test_classify_suspicious_default(self, service):
        """Test suspicious classification when category unknown."""
        from aragora.services.threat_intelligence import ThreatType

        vt_result = {"categories": {}, "positives": 10}
        assert service._classify_vt_threat(vt_result) == ThreatType.SUSPICIOUS

    def test_classify_none_no_positives(self, service):
        """Test NONE when no positives."""
        from aragora.services.threat_intelligence import ThreatType

        vt_result = {"categories": {}, "positives": 0}
        assert service._classify_vt_threat(vt_result) == ThreatType.NONE


class TestAggregateRiskCalculation:
    """Tests for aggregate risk calculation."""

    @pytest.fixture
    def service(self):
        """Create service for risk calculation tests."""
        from aragora.services.threat_intelligence import ThreatIntelligenceService

        return ThreatIntelligenceService()

    def test_empty_results(self, service):
        """Test aggregate risk with empty results."""
        overall, weighted, is_mal = service._calculate_aggregate_risk({})
        assert overall == 0.0
        assert weighted == 0.0
        assert is_mal is False

    def test_single_source_high_confidence(self, service):
        """Test aggregate risk with single high-confidence source."""
        from aragora.services.threat_intelligence import SourceResult, ThreatSource

        source_results = {
            "virustotal": SourceResult(
                source=ThreatSource.VIRUSTOTAL,
                is_malicious=True,
                confidence=0.9,
            )
        }

        overall, weighted, is_mal = service._calculate_aggregate_risk(source_results)
        assert overall > 0.5  # Should exceed malicious threshold
        assert is_mal is True

    def test_multiple_sources_agreement(self, service):
        """Test aggregate risk with multiple agreeing sources."""
        from aragora.services.threat_intelligence import SourceResult, ThreatSource

        source_results = {
            "virustotal": SourceResult(
                source=ThreatSource.VIRUSTOTAL,
                is_malicious=True,
                confidence=0.8,
            ),
            "phishtank": SourceResult(
                source=ThreatSource.PHISHTANK,
                is_malicious=True,
                confidence=0.95,
            ),
        }

        overall, weighted, is_mal = service._calculate_aggregate_risk(source_results)
        assert overall > 0.7  # High agreement
        assert is_mal is True

    def test_sources_disagree(self, service):
        """Test aggregate risk when sources disagree."""
        from aragora.services.threat_intelligence import SourceResult, ThreatSource

        source_results = {
            "virustotal": SourceResult(
                source=ThreatSource.VIRUSTOTAL,
                is_malicious=True,
                confidence=0.5,
            ),
            "phishtank": SourceResult(
                source=ThreatSource.PHISHTANK,
                is_malicious=False,
                confidence=0.0,
            ),
        }

        overall, weighted, is_mal = service._calculate_aggregate_risk(source_results)
        # Lower overall due to disagreement
        assert overall < 0.7

    def test_sources_with_error_skipped(self, service):
        """Test that sources with errors are skipped."""
        from aragora.services.threat_intelligence import SourceResult, ThreatSource

        source_results = {
            "virustotal": SourceResult(
                source=ThreatSource.VIRUSTOTAL,
                is_malicious=True,
                confidence=0.9,
            ),
            "phishtank": SourceResult(
                source=ThreatSource.PHISHTANK,
                is_malicious=False,
                confidence=0.0,
                error="API timeout",
            ),
        }

        overall, weighted, is_mal = service._calculate_aggregate_risk(source_results)
        # Error source should be skipped, only virustotal counted
        assert is_mal is True


class TestRateLimiting:
    """Tests for rate limiting functionality."""

    @pytest.fixture
    def service(self):
        """Create service for rate limiting tests."""
        from aragora.services.threat_intelligence import ThreatIntelligenceService

        return ThreatIntelligenceService()

    @pytest.mark.asyncio
    async def test_rate_limit_allows_under_limit(self, service):
        """Test rate limit allows requests under limit."""
        # First request should be allowed
        assert await service._check_rate_limit("virustotal") is True
        assert await service._check_rate_limit("virustotal") is True

    @pytest.mark.asyncio
    async def test_rate_limit_blocks_over_limit(self, service):
        """Test rate limit blocks requests over limit."""
        # VirusTotal has limit of 4 per minute
        for _ in range(4):
            await service._check_rate_limit("virustotal")

        # 5th request should be blocked
        assert await service._check_rate_limit("virustotal") is False


class TestEventHandlers:
    """Tests for event handler functionality."""

    @pytest.fixture
    def service(self):
        """Create service for event tests."""
        from aragora.services.threat_intelligence import ThreatIntelligenceService

        return ThreatIntelligenceService()

    def test_add_event_handler(self, service):
        """Test adding event handler."""
        handler = MagicMock()
        service.add_event_handler(handler)
        assert handler in service._event_handlers

    def test_remove_event_handler(self, service):
        """Test removing event handler."""
        handler = MagicMock()
        service.add_event_handler(handler)
        result = service.remove_event_handler(handler)
        assert result is True
        assert handler not in service._event_handlers

    def test_remove_nonexistent_handler(self, service):
        """Test removing handler that wasn't added."""
        handler = MagicMock()
        result = service.remove_event_handler(handler)
        assert result is False

    def test_emit_event(self, service):
        """Test event emission."""
        handler = MagicMock()
        service.add_event_handler(handler)

        service._emit_event("test_event", {"key": "value"})

        handler.assert_called_once_with("test_event", {"key": "value"})

    def test_emit_event_handler_error(self, service):
        """Test event emission handles handler errors."""
        handler = MagicMock(side_effect=Exception("Handler error"))
        service.add_event_handler(handler)

        # Should not raise
        service._emit_event("test_event", {})


class TestCacheKeyGeneration:
    """Tests for cache key generation."""

    @pytest.fixture
    def service(self):
        """Create service for cache tests."""
        from aragora.services.threat_intelligence import ThreatIntelligenceService

        return ThreatIntelligenceService()

    def test_cache_key_deterministic(self, service):
        """Test cache key is deterministic."""
        key1 = service._get_cache_key("http://test.com", "url")
        key2 = service._get_cache_key("http://test.com", "url")
        assert key1 == key2

    def test_cache_key_different_targets(self, service):
        """Test different targets produce different keys."""
        key1 = service._get_cache_key("http://test1.com", "url")
        key2 = service._get_cache_key("http://test2.com", "url")
        assert key1 != key2

    def test_cache_key_case_insensitive(self, service):
        """Test cache key is case insensitive."""
        key1 = service._get_cache_key("http://TEST.com", "url")
        key2 = service._get_cache_key("http://test.com", "url")
        assert key1 == key2

    def test_ttl_for_url(self, service):
        """Test TTL for URL type."""
        ttl = service._get_ttl_for_type("url")
        # Default URL TTL is 24 hours = 86400 seconds
        assert ttl == 24 * 3600

    def test_ttl_for_ip(self, service):
        """Test TTL for IP type (shorter)."""
        ttl = service._get_ttl_for_type("ip")
        # Default IP TTL is 1 hour = 3600 seconds
        assert ttl == 1 * 3600

    def test_ttl_for_hash(self, service):
        """Test TTL for hash type (longer)."""
        ttl = service._get_ttl_for_type("hash")
        # Default hash TTL is 168 hours (7 days)
        assert ttl == 168 * 3600


class TestCircuitBreakerStatus:
    """Tests for circuit breaker status."""

    @pytest.fixture
    def service(self):
        """Create service for circuit breaker tests."""
        from aragora.services.threat_intelligence import ThreatIntelligenceService

        return ThreatIntelligenceService()

    def test_get_circuit_breaker_status(self, service):
        """Test getting circuit breaker status."""
        status = service.get_circuit_breaker_status()

        # Should have status for all services
        assert "virustotal" in status
        assert "abuseipdb" in status
        assert "phishtank" in status
        assert "urlhaus" in status

        # Each should have expected fields
        for service_status in status.values():
            assert "status" in service_status
            assert "failures" in service_status
            assert "threshold" in service_status

    def test_circuit_breaker_closed_initially(self, service):
        """Test circuit breakers are closed initially."""
        assert service._is_circuit_open("virustotal") is False
        assert service._is_circuit_open("abuseipdb") is False
        assert service._is_circuit_open("phishtank") is False
