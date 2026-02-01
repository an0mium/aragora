"""
Tests for CVE Client module.

Tests CVE/vulnerability database integration: NVD, OSV, and GitHub advisory queries,
response parsing, severity calculation, caching, circuit breaker, and batch lookups.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.analysis.codebase.cve_client import CVEClient
from aragora.analysis.codebase.models import (
    VulnerabilityFinding,
    VulnerabilitySeverity,
    VulnerabilitySource,
)


# ============================================================
# CVEClient Initialization
# ============================================================


class TestCVEClientInit:
    """Tests for CVEClient initialization."""

    def test_default_initialization(self):
        """Client initializes with default settings."""
        client = CVEClient()
        assert client.nvd_api_key is None or isinstance(client.nvd_api_key, str)
        assert client.cache_ttl == 3600
        assert client._enable_circuit_breaker is True
        assert client._failure_threshold == 5
        assert client._cooldown_seconds == 60.0

    def test_with_nvd_api_key(self):
        """Client accepts NVD API key."""
        client = CVEClient(nvd_api_key="test-key-123")
        assert client.nvd_api_key == "test-key-123"

    def test_with_github_token(self):
        """Client accepts GitHub token."""
        client = CVEClient(github_token="ghp_test123")
        assert client.github_token == "ghp_test123"

    def test_custom_cache_ttl(self):
        """Client accepts custom cache TTL."""
        client = CVEClient(cache_ttl_seconds=7200)
        assert client.cache_ttl == 7200

    def test_disable_circuit_breaker(self):
        """Client can disable circuit breaker."""
        client = CVEClient(enable_circuit_breaker=False)
        assert client._enable_circuit_breaker is False


# ============================================================
# Severity Parsing (CVSS Scores)
# ============================================================


class TestSeverityParsing:
    """Tests for CVSS score to severity conversion."""

    def test_critical_severity(self):
        """CVSS 9.0+ is CRITICAL."""
        assert VulnerabilitySeverity.from_cvss(9.0) == VulnerabilitySeverity.CRITICAL
        assert VulnerabilitySeverity.from_cvss(9.5) == VulnerabilitySeverity.CRITICAL
        assert VulnerabilitySeverity.from_cvss(10.0) == VulnerabilitySeverity.CRITICAL

    def test_high_severity(self):
        """CVSS 7.0-8.9 is HIGH."""
        assert VulnerabilitySeverity.from_cvss(7.0) == VulnerabilitySeverity.HIGH
        assert VulnerabilitySeverity.from_cvss(8.0) == VulnerabilitySeverity.HIGH
        assert VulnerabilitySeverity.from_cvss(8.9) == VulnerabilitySeverity.HIGH

    def test_medium_severity(self):
        """CVSS 4.0-6.9 is MEDIUM."""
        assert VulnerabilitySeverity.from_cvss(4.0) == VulnerabilitySeverity.MEDIUM
        assert VulnerabilitySeverity.from_cvss(5.5) == VulnerabilitySeverity.MEDIUM
        assert VulnerabilitySeverity.from_cvss(6.9) == VulnerabilitySeverity.MEDIUM

    def test_low_severity(self):
        """CVSS 0.1-3.9 is LOW."""
        assert VulnerabilitySeverity.from_cvss(0.1) == VulnerabilitySeverity.LOW
        assert VulnerabilitySeverity.from_cvss(2.0) == VulnerabilitySeverity.LOW
        assert VulnerabilitySeverity.from_cvss(3.9) == VulnerabilitySeverity.LOW

    def test_unknown_severity(self):
        """CVSS 0 is UNKNOWN."""
        assert VulnerabilitySeverity.from_cvss(0) == VulnerabilitySeverity.UNKNOWN
        assert VulnerabilitySeverity.from_cvss(0.0) == VulnerabilitySeverity.UNKNOWN


# ============================================================
# Cache Behavior
# ============================================================


class TestCaching:
    """Tests for CVE client caching behavior."""

    def test_cache_key_generation(self):
        """Cache key is generated consistently."""
        client = CVEClient()
        key1 = client._get_cache_key("cve", "CVE-2023-12345")
        key2 = client._get_cache_key("cve", "CVE-2023-12345")
        key3 = client._get_cache_key("cve", "CVE-2023-12346")

        assert key1 == key2
        assert key1 != key3
        assert isinstance(key1, str)

    def test_cache_set_and_get(self):
        """Values can be cached and retrieved."""
        client = CVEClient()
        test_data = {"id": "CVE-2023-12345", "severity": "critical"}

        key = client._get_cache_key("test", "value")
        client._set_cached(key, test_data)

        cached = client._get_cached(key)
        assert cached == test_data

    def test_cache_miss(self):
        """Cache returns None for missing keys."""
        client = CVEClient()
        cached = client._get_cached("nonexistent-key")
        assert cached is None

    def test_cache_expiration(self):
        """Expired cache entries return None."""
        client = CVEClient(cache_ttl_seconds=1)
        key = client._get_cache_key("test", "value")
        client._set_cached(key, {"test": "data"})

        # Manually expire the cache entry
        cached_at = datetime.now(timezone.utc) - timedelta(seconds=2)
        client._cache[key] = (cached_at, {"test": "data"})

        cached = client._get_cached(key)
        assert cached is None
        assert key not in client._cache  # Entry should be removed


# ============================================================
# Circuit Breaker
# ============================================================


class TestCircuitBreaker:
    """Tests for circuit breaker behavior."""

    def test_circuit_breaker_allows_requests_initially(self):
        """Circuit breaker allows requests when no failures."""
        client = CVEClient()
        assert client._check_circuit_breaker("nvd") is True
        assert client._check_circuit_breaker("osv") is True

    def test_circuit_breaker_opens_after_threshold(self):
        """Circuit breaker opens after failure threshold."""
        client = CVEClient()
        client._failure_threshold = 3

        for _ in range(3):
            client._record_failure("nvd")

        assert client._check_circuit_breaker("nvd") is False

    def test_circuit_breaker_resets_on_success(self):
        """Circuit breaker failure count resets on success."""
        client = CVEClient()
        client._record_failure("nvd")
        client._record_failure("nvd")
        assert client._circuit_breaker_failures.get("nvd", 0) == 2

        client._record_success("nvd")
        assert client._circuit_breaker_failures.get("nvd", 0) == 0

    def test_circuit_breaker_cooldown(self):
        """Circuit breaker reopens after cooldown period."""
        client = CVEClient()
        client._failure_threshold = 3
        client._cooldown_seconds = 0.1  # Short cooldown for test

        for _ in range(3):
            client._record_failure("nvd")

        assert client._check_circuit_breaker("nvd") is False

        # Wait for cooldown
        import time

        time.sleep(0.15)

        assert client._check_circuit_breaker("nvd") is True

    def test_circuit_breaker_disabled(self):
        """Circuit breaker can be disabled."""
        client = CVEClient(enable_circuit_breaker=False)

        for _ in range(10):
            client._record_failure("nvd")

        assert client._check_circuit_breaker("nvd") is True


# ============================================================
# NVD CVE Response Parsing
# ============================================================


class TestNVDParsing:
    """Tests for NVD API response parsing."""

    def test_parse_nvd_cve_basic(self):
        """Parse basic NVD CVE response."""
        client = CVEClient()
        cve_data = {
            "id": "CVE-2023-12345",
            "descriptions": [{"lang": "en", "value": "A test vulnerability description."}],
            "metrics": {},
            "weaknesses": [],
            "references": [],
        }

        finding = client._parse_nvd_cve(cve_data)

        assert finding.id == "CVE-2023-12345"
        assert finding.description == "A test vulnerability description."
        assert finding.source == VulnerabilitySource.NVD

    def test_parse_nvd_cve_with_cvss_v31(self):
        """Parse NVD CVE with CVSS v3.1 score."""
        client = CVEClient()
        cve_data = {
            "id": "CVE-2023-12345",
            "descriptions": [{"lang": "en", "value": "Test"}],
            "metrics": {
                "cvssMetricV31": [
                    {
                        "cvssData": {
                            "baseScore": 9.8,
                            "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                        }
                    }
                ]
            },
            "weaknesses": [],
            "references": [],
        }

        finding = client._parse_nvd_cve(cve_data)

        assert finding.cvss_score == 9.8
        assert finding.severity == VulnerabilitySeverity.CRITICAL
        assert "CVSS:3.1" in finding.cvss_vector

    def test_parse_nvd_cve_with_cwe(self):
        """Parse NVD CVE with CWE identifiers."""
        client = CVEClient()
        cve_data = {
            "id": "CVE-2023-12345",
            "descriptions": [{"lang": "en", "value": "Test"}],
            "metrics": {},
            "weaknesses": [
                {"description": [{"value": "CWE-79"}]},
                {"description": [{"value": "CWE-89"}]},
            ],
            "references": [],
        }

        finding = client._parse_nvd_cve(cve_data)

        assert "CWE-79" in finding.cwe_ids
        assert "CWE-89" in finding.cwe_ids

    def test_parse_nvd_cve_with_references(self):
        """Parse NVD CVE with references."""
        client = CVEClient()
        cve_data = {
            "id": "CVE-2023-12345",
            "descriptions": [{"lang": "en", "value": "Test"}],
            "metrics": {},
            "weaknesses": [],
            "references": [
                {"url": "https://example.com/advisory", "source": "vendor", "tags": ["Patch"]},
            ],
        }

        finding = client._parse_nvd_cve(cve_data)

        assert len(finding.references) == 1
        assert finding.references[0].url == "https://example.com/advisory"
        assert "Patch" in finding.references[0].tags

    def test_parse_nvd_cve_with_dates(self):
        """Parse NVD CVE with publication dates."""
        client = CVEClient()
        cve_data = {
            "id": "CVE-2023-12345",
            "descriptions": [{"lang": "en", "value": "Test"}],
            "metrics": {},
            "weaknesses": [],
            "references": [],
            "published": "2023-06-15T14:30:00Z",
            "lastModified": "2023-07-01T10:00:00Z",
        }

        finding = client._parse_nvd_cve(cve_data)

        assert finding.published_at is not None
        assert finding.published_at.year == 2023
        assert finding.published_at.month == 6
        assert finding.updated_at is not None


# ============================================================
# OSV Vulnerability Response Parsing
# ============================================================


class TestOSVParsing:
    """Tests for OSV API response parsing."""

    def test_parse_osv_vuln_basic(self):
        """Parse basic OSV vulnerability response."""
        client = CVEClient()
        osv_data = {
            "id": "GHSA-abc123",
            "summary": "Test vulnerability summary",
            "details": "Detailed description",
        }

        finding = client._parse_osv_vuln(osv_data)

        assert finding.id == "GHSA-abc123"
        assert finding.description == "Test vulnerability summary"
        assert finding.source == VulnerabilitySource.OSV

    def test_parse_osv_vuln_with_severity(self):
        """Parse OSV vulnerability with severity."""
        client = CVEClient()
        osv_data = {
            "id": "GHSA-abc123",
            "summary": "Test",
            "database_specific": {"severity": "HIGH"},
        }

        finding = client._parse_osv_vuln(osv_data)
        assert finding.severity == VulnerabilitySeverity.HIGH

    def test_parse_osv_vuln_affected_versions(self):
        """Parse OSV vulnerability with affected versions."""
        client = CVEClient()
        osv_data = {
            "id": "GHSA-abc123",
            "summary": "Test",
            "affected": [
                {
                    "ranges": [
                        {
                            "events": [
                                {"introduced": "0"},
                                {"fixed": "1.0.1"},
                            ]
                        }
                    ]
                }
            ],
        }

        finding = client._parse_osv_vuln(osv_data)

        assert ">= 0" in finding.vulnerable_versions
        assert "1.0.1" in finding.patched_versions
        assert finding.fix_available is True
        assert finding.recommended_version == "1.0.1"


# ============================================================
# CVE Lookup by ID
# ============================================================


class TestCVELookup:
    """Tests for CVE lookup by ID."""

    @pytest.mark.asyncio
    async def test_get_cve_returns_cached(self):
        """get_cve returns cached result if available."""
        client = CVEClient()

        # Pre-populate cache
        cache_key = client._get_cache_key("cve", "CVE-2023-12345")
        cached_finding = VulnerabilityFinding(
            id="CVE-2023-12345",
            title="CVE-2023-12345",
            description="Cached vulnerability",
            severity=VulnerabilitySeverity.HIGH,
            source=VulnerabilitySource.NVD,
        )
        client._set_cached(cache_key, cached_finding)

        result = await client.get_cve("CVE-2023-12345")

        assert result is not None
        assert result.id == "CVE-2023-12345"
        assert result.description == "Cached vulnerability"

    @pytest.mark.asyncio
    async def test_get_cve_queries_nvd(self):
        """get_cve queries NVD API and parses response."""
        client = CVEClient()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "id": "CVE-2023-12345",
                        "descriptions": [{"lang": "en", "value": "Test CVE"}],
                        "metrics": {},
                        "weaknesses": [],
                        "references": [],
                    }
                }
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client.get_cve("CVE-2023-12345")

        assert result is not None
        assert result.id == "CVE-2023-12345"

    @pytest.mark.asyncio
    async def test_get_cve_returns_none_when_not_found(self):
        """get_cve returns None when CVE not found from both NVD and OSV."""
        client = CVEClient()

        # Mock both NVD (returns empty list) and OSV (returns 404)
        with (
            patch.object(client, "_query_nvd_cve", return_value=None),
            patch.object(client, "_query_osv_cve", return_value=None),
        ):
            result = await client.get_cve("CVE-9999-99999")

        assert result is None


# ============================================================
# API Error Handling
# ============================================================


class TestAPIErrorHandling:
    """Tests for API error handling."""

    @pytest.mark.asyncio
    async def test_nvd_rate_limit_triggers_circuit_breaker(self):
        """NVD 429 response triggers circuit breaker."""
        client = CVEClient()

        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client._query_nvd_cve("CVE-2023-12345")

        assert result is None
        assert client._circuit_breaker_failures.get("nvd", 0) == 1

    @pytest.mark.asyncio
    async def test_nvd_network_error_triggers_circuit_breaker(self):
        """Network errors trigger circuit breaker."""
        client = CVEClient()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get = AsyncMock(side_effect=TimeoutError("Connection timeout"))
            mock_client_class.return_value = mock_client

            result = await client._query_nvd_cve("CVE-2023-12345")

        assert result is None
        assert client._circuit_breaker_failures.get("nvd", 0) == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_requests(self):
        """Open circuit breaker prevents requests."""
        client = CVEClient()

        # Force circuit breaker open
        client._circuit_breaker_open_until["nvd"] = datetime.now(timezone.utc) + timedelta(
            seconds=60
        )

        result = await client._query_nvd_cve("CVE-2023-12345")
        assert result is None


# ============================================================
# Package Query
# ============================================================


class TestPackageQuery:
    """Tests for package vulnerability queries."""

    @pytest.mark.asyncio
    async def test_query_package_returns_cached(self):
        """query_package returns cached results."""
        client = CVEClient()

        cache_key = client._get_cache_key("package", "lodash", "npm", "4.17.0")
        cached_findings = [
            VulnerabilityFinding(
                id="CVE-2020-8203",
                title="Prototype Pollution",
                description="Test",
                severity=VulnerabilitySeverity.HIGH,
            )
        ]
        client._set_cached(cache_key, cached_findings)

        results = await client.query_package("lodash", "npm", "4.17.0")

        assert len(results) == 1
        assert results[0].id == "CVE-2020-8203"

    @pytest.mark.asyncio
    async def test_query_package_deduplicates_findings(self):
        """query_package deduplicates findings by ID."""
        client = CVEClient()

        # Mock OSV and GitHub to return same CVE
        osv_finding = VulnerabilityFinding(
            id="CVE-2020-8203",
            title="From OSV",
            description="Test",
            severity=VulnerabilitySeverity.HIGH,
        )
        github_finding = VulnerabilityFinding(
            id="CVE-2020-8203",
            title="From GitHub",
            description="Test",
            severity=VulnerabilitySeverity.HIGH,
        )

        with (
            patch.object(client, "_query_osv_package", return_value=[osv_finding]),
            patch.object(client, "_query_github_advisory", return_value=[github_finding]),
        ):
            results = await client.query_package("lodash", "npm", "4.17.0")

        assert len(results) == 1
        assert results[0].id == "CVE-2020-8203"


# ============================================================
# Batch Lookup
# ============================================================


class TestBatchLookup:
    """Tests for batch package queries."""

    @pytest.mark.asyncio
    async def test_batch_query_packages(self):
        """batch_query_packages queries multiple packages concurrently."""
        client = CVEClient()

        finding1 = VulnerabilityFinding(
            id="CVE-2020-8203",
            title="Lodash vuln",
            description="Test",
            severity=VulnerabilitySeverity.HIGH,
        )
        finding2 = VulnerabilityFinding(
            id="CVE-2021-12345",
            title="Express vuln",
            description="Test",
            severity=VulnerabilitySeverity.MEDIUM,
        )

        async def mock_query_package(name, ecosystem, version):
            if name == "lodash":
                return [finding1]
            elif name == "express":
                return [finding2]
            return []

        with patch.object(client, "query_package", side_effect=mock_query_package):
            results = await client.batch_query_packages(
                [
                    ("lodash", "npm", "4.17.0"),
                    ("express", "npm", "4.18.0"),
                ],
                concurrency=2,
            )

        assert "lodash@4.17.0" in results
        assert "express@4.18.0" in results
        assert len(results["lodash@4.17.0"]) == 1
        assert len(results["express@4.18.0"]) == 1

    @pytest.mark.asyncio
    async def test_batch_query_respects_concurrency(self):
        """batch_query_packages respects concurrency limit."""
        client = CVEClient()

        call_count = 0
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def mock_query_package(name, ecosystem, version):
            nonlocal call_count, max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
                call_count += 1
            await asyncio.sleep(0.01)
            async with lock:
                current_concurrent -= 1
            return []

        with patch.object(client, "query_package", side_effect=mock_query_package):
            await client.batch_query_packages(
                [
                    ("pkg1", "npm", "1.0.0"),
                    ("pkg2", "npm", "1.0.0"),
                    ("pkg3", "npm", "1.0.0"),
                    ("pkg4", "npm", "1.0.0"),
                    ("pkg5", "npm", "1.0.0"),
                ],
                concurrency=2,
            )

        assert call_count == 5
        assert max_concurrent <= 2


# ============================================================
# Ecosystem Mapping
# ============================================================


class TestEcosystemMapping:
    """Tests for ecosystem name mapping."""

    def test_ecosystem_map_npm(self):
        """NPM ecosystem is mapped correctly."""
        assert CVEClient.ECOSYSTEM_MAP["npm"]["osv"] == "npm"
        assert CVEClient.ECOSYSTEM_MAP["npm"]["github"] == "NPM"

    def test_ecosystem_map_pypi(self):
        """PyPI ecosystem is mapped correctly."""
        assert CVEClient.ECOSYSTEM_MAP["pypi"]["osv"] == "PyPI"
        assert CVEClient.ECOSYSTEM_MAP["pypi"]["github"] == "PIP"

    def test_ecosystem_map_cargo(self):
        """Cargo ecosystem is mapped correctly."""
        assert CVEClient.ECOSYSTEM_MAP["cargo"]["osv"] == "crates.io"
        assert CVEClient.ECOSYSTEM_MAP["cargo"]["github"] == "RUST"

    def test_ecosystem_map_go(self):
        """Go ecosystem is mapped correctly."""
        assert CVEClient.ECOSYSTEM_MAP["go"]["osv"] == "Go"
        assert CVEClient.ECOSYSTEM_MAP["go"]["github"] == "GO"
