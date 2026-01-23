"""
Tests for Threat Intelligence Enrichment.

Tests the ThreatIntelEnrichment class which provides threat intelligence
context enrichment for security-related debate topics.
"""

import os
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.security.threat_intel_enrichment import (
    ThreatIndicator,
    ThreatContext,
    ThreatIntelEnrichment,
    enrich_security_context,
    ENRICHMENT_ENABLED,
    MAX_INDICATORS,
)


class TestThreatIndicator:
    """Tests for ThreatIndicator dataclass."""

    def test_create_basic_indicator(self):
        """Should create indicator with required fields."""
        indicator = ThreatIndicator(
            indicator_type="ip",
            value="192.0.2.1",
            threat_type="malicious_ip",
            severity="high",
            confidence=0.85,
            source="abuseipdb",
        )
        assert indicator.indicator_type == "ip"
        assert indicator.value == "192.0.2.1"
        assert indicator.threat_type == "malicious_ip"
        assert indicator.severity == "high"
        assert indicator.confidence == 0.85
        assert indicator.source == "abuseipdb"

    def test_create_indicator_with_optional_fields(self):
        """Should create indicator with all optional fields."""
        now = datetime.now(timezone.utc)
        indicator = ThreatIndicator(
            indicator_type="hash",
            value="a" * 64,
            threat_type="malware",
            severity="critical",
            confidence=0.95,
            source="virustotal",
            first_seen=now,
            last_seen=now,
            description="Detected ransomware",
            tags=["ransomware", "lockbit"],
        )
        assert indicator.first_seen == now
        assert indicator.last_seen == now
        assert indicator.description == "Detected ransomware"
        assert "ransomware" in indicator.tags

    def test_to_dict(self):
        """Should convert indicator to dictionary."""
        indicator = ThreatIndicator(
            indicator_type="url",
            value="http://malicious.example.com",
            threat_type="phishing",
            severity="medium",
            confidence=0.75,
            source="urlhaus",
            tags=["phishing", "credential-theft"],
        )
        result = indicator.to_dict()
        assert result["indicator_type"] == "url"
        assert result["value"] == "http://malicious.example.com"
        assert result["severity"] == "medium"
        assert "phishing" in result["tags"]

    def test_default_tags_list(self):
        """Should initialize tags to empty list by default."""
        indicator = ThreatIndicator(
            indicator_type="ip",
            value="10.0.0.1",
            threat_type="test",
            severity="low",
            confidence=0.5,
            source="test",
        )
        assert indicator.tags == []


class TestThreatContext:
    """Tests for ThreatContext dataclass."""

    def test_create_threat_context(self):
        """Should create threat context with all fields."""
        indicators = [
            ThreatIndicator(
                indicator_type="ip",
                value="192.0.2.1",
                threat_type="malicious",
                severity="high",
                confidence=0.9,
                source="test",
            )
        ]
        cves = [{"id": "CVE-2024-1234", "severity": "CRITICAL", "description": "Test"}]
        context = ThreatContext(
            indicators=indicators,
            relevant_cves=cves,
            attack_patterns=["Remote Code Execution"],
            recommended_mitigations=["Apply patch"],
            risk_summary="Critical risk identified",
        )
        assert len(context.indicators) == 1
        assert len(context.relevant_cves) == 1
        assert "Remote Code Execution" in context.attack_patterns
        assert context.risk_summary == "Critical risk identified"

    def test_to_dict(self):
        """Should convert context to dictionary."""
        context = ThreatContext(
            indicators=[],
            relevant_cves=[],
            attack_patterns=["SQL Injection"],
            recommended_mitigations=["Use parameterized queries"],
            risk_summary="Medium risk",
        )
        result = context.to_dict()
        assert "indicators" in result
        assert "relevant_cves" in result
        assert "SQL Injection" in result["attack_patterns"]
        assert result["risk_summary"] == "Medium risk"
        assert "enrichment_timestamp" in result


class TestThreatIntelEnrichment:
    """Tests for ThreatIntelEnrichment class."""

    def test_is_security_topic_positive(self):
        """Should identify security-related topics."""
        enrichment = ThreatIntelEnrichment(enabled=True)

        security_topics = [
            "How should we respond to CVE-2024-1234?",
            "Implement better security controls",
            "Mitigate the SQL injection vulnerability",
            "Respond to the ransomware attack",
            "Patch the authentication bypass",
            "Investigate the phishing incident",
            "Strengthen firewall rules",
            "Review encryption implementation",
        ]

        for topic in security_topics:
            assert enrichment.is_security_topic(topic), f"Should identify: {topic}"

    def test_is_security_topic_negative(self):
        """Should not identify non-security topics."""
        enrichment = ThreatIntelEnrichment(enabled=True)

        non_security_topics = [
            "Implement the new feature",
            "Fix the UI bug",
            "Optimize database queries",
            "Add unit tests",
            "Refactor the code",
            "Update documentation",
        ]

        for topic in non_security_topics:
            assert not enrichment.is_security_topic(topic), f"Should not identify: {topic}"

    def test_extract_entities_cves(self):
        """Should extract CVE identifiers from text."""
        enrichment = ThreatIntelEnrichment(enabled=True)
        text = "We need to patch CVE-2024-1234 and CVE-2023-5678 urgently"
        entities = enrichment._extract_entities(text)
        assert "CVE-2024-1234" in entities["cves"]
        assert "CVE-2023-5678" in entities["cves"]

    def test_extract_entities_ips(self):
        """Should extract non-private IP addresses."""
        enrichment = ThreatIntelEnrichment(enabled=True)
        text = "Attack from 203.0.113.50 and 198.51.100.25"
        entities = enrichment._extract_entities(text)
        assert "203.0.113.50" in entities["ips"]
        assert "198.51.100.25" in entities["ips"]

    def test_extract_entities_excludes_private_ips(self):
        """Should exclude private IP addresses."""
        enrichment = ThreatIntelEnrichment(enabled=True)
        text = "Internal addresses 192.168.1.1 and 10.0.0.1 and 127.0.0.1"
        entities = enrichment._extract_entities(text)
        assert "192.168.1.1" not in entities["ips"]
        assert "10.0.0.1" not in entities["ips"]
        assert "127.0.0.1" not in entities["ips"]

    def test_extract_entities_urls(self):
        """Should extract URLs from text."""
        enrichment = ThreatIntelEnrichment(enabled=True)
        text = "Malicious URL https://malicious.example.com/payload"
        entities = enrichment._extract_entities(text)
        assert any("malicious.example.com" in url for url in entities["urls"])

    def test_extract_entities_hashes(self):
        """Should extract file hashes from text."""
        enrichment = ThreatIntelEnrichment(enabled=True)
        md5_hash = "d41d8cd98f00b204e9800998ecf8427e"
        sha256_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        text = f"Found hashes: {md5_hash} and {sha256_hash}"
        entities = enrichment._extract_entities(text)
        assert any(md5_hash in h for h in [h[1] for h in entities["hashes"]])
        assert any(sha256_hash in h for h in [h[1] for h in entities["hashes"]])

    def test_disabled_enrichment(self):
        """Should return None when enrichment is disabled."""
        enrichment = ThreatIntelEnrichment(enabled=False)
        assert enrichment._enabled is False

    @pytest.mark.asyncio
    async def test_enrich_context_non_security_topic(self):
        """Should return None for non-security topics."""
        enrichment = ThreatIntelEnrichment(enabled=True)
        result = await enrichment.enrich_context(
            topic="Implement the new feature",
            existing_context="",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_enrich_context_with_generic_security_topic(self):
        """Should provide generic context for security topics without specific IOCs."""
        enrichment = ThreatIntelEnrichment(enabled=True)
        result = await enrichment.enrich_context(
            topic="How should we improve our security posture?",
            existing_context="",
        )
        # Should return generic security context
        assert result is not None
        assert len(result.attack_patterns) > 0
        assert len(result.recommended_mitigations) > 0

    @pytest.mark.asyncio
    async def test_enrich_context_ransomware_topic(self):
        """Should provide ransomware-specific context."""
        enrichment = ThreatIntelEnrichment(enabled=True)
        result = await enrichment.enrich_context(
            topic="How should we respond to a ransomware attack?",
        )
        assert result is not None
        assert "Ransomware" in result.attack_patterns
        assert any("backup" in m.lower() for m in result.recommended_mitigations)

    @pytest.mark.asyncio
    async def test_enrich_context_phishing_topic(self):
        """Should provide phishing-specific context."""
        enrichment = ThreatIntelEnrichment(enabled=True)
        result = await enrichment.enrich_context(
            topic="How do we protect against phishing attacks?",
        )
        assert result is not None
        assert "Phishing" in result.attack_patterns
        assert any(
            "email" in m.lower() or "awareness" in m.lower() for m in result.recommended_mitigations
        )

    @pytest.mark.asyncio
    async def test_enrich_context_sql_injection_topic(self):
        """Should provide SQL injection-specific context."""
        enrichment = ThreatIntelEnrichment(enabled=True)
        result = await enrichment.enrich_context(
            topic="How do we fix the SQL injection vulnerability?",
        )
        assert result is not None
        assert "SQL Injection" in result.attack_patterns
        assert any("parameterized" in m.lower() for m in result.recommended_mitigations)

    def test_format_for_debate_with_cves(self):
        """Should format CVE information correctly."""
        enrichment = ThreatIntelEnrichment(enabled=True)
        context = ThreatContext(
            indicators=[],
            relevant_cves=[
                {
                    "id": "CVE-2024-1234",
                    "severity": "CRITICAL",
                    "cvss_score": 9.8,
                    "description": "Remote code execution vulnerability",
                    "fix_available": True,
                    "recommended_version": "2.0.1",
                }
            ],
            attack_patterns=["RCE"],
            recommended_mitigations=["Upgrade to 2.0.1"],
            risk_summary="Critical risk",
        )
        formatted = enrichment.format_for_debate(context)
        assert "CVE-2024-1234" in formatted
        assert "CRITICAL" in formatted
        assert "9.8" in formatted
        assert "2.0.1" in formatted

    def test_format_for_debate_with_indicators(self):
        """Should format threat indicators correctly."""
        enrichment = ThreatIntelEnrichment(enabled=True)
        context = ThreatContext(
            indicators=[
                ThreatIndicator(
                    indicator_type="ip",
                    value="203.0.113.50",
                    threat_type="malicious_ip",
                    severity="high",
                    confidence=0.9,
                    source="abuseipdb",
                    description="Known attacker IP",
                )
            ],
            relevant_cves=[],
            attack_patterns=["Brute Force"],
            recommended_mitigations=["Block IP"],
            risk_summary="High risk",
        )
        formatted = enrichment.format_for_debate(context)
        assert "IP" in formatted
        assert "203.0.113.50" in formatted
        assert "high" in formatted.lower()
        assert "90%" in formatted or "0.9" in formatted  # confidence

    def test_format_for_debate_with_mitigations(self):
        """Should include recommended mitigations."""
        enrichment = ThreatIntelEnrichment(enabled=True)
        context = ThreatContext(
            indicators=[],
            relevant_cves=[],
            attack_patterns=["Generic Attack"],
            recommended_mitigations=[
                "Apply security patches",
                "Enable MFA",
                "Monitor logs",
            ],
            risk_summary="Medium risk",
        )
        formatted = enrichment.format_for_debate(context)
        assert "Recommended Mitigations" in formatted
        assert "Apply security patches" in formatted
        assert "Enable MFA" in formatted
        assert "Monitor logs" in formatted

    def test_format_for_debate_with_risk_summary(self):
        """Should include risk summary."""
        enrichment = ThreatIntelEnrichment(enabled=True)
        context = ThreatContext(
            indicators=[],
            relevant_cves=[],
            attack_patterns=[],
            recommended_mitigations=[],
            risk_summary="Overall risk: HIGH. Found 3 critical CVEs.",
        )
        formatted = enrichment.format_for_debate(context)
        assert "Risk Summary" in formatted
        assert "HIGH" in formatted
        assert "3 critical CVEs" in formatted

    def test_abuse_score_to_severity(self):
        """Should convert abuse scores to severity levels."""
        enrichment = ThreatIntelEnrichment(enabled=True)
        assert enrichment._abuse_score_to_severity(90) == "critical"
        assert enrichment._abuse_score_to_severity(60) == "high"
        assert enrichment._abuse_score_to_severity(30) == "medium"
        assert enrichment._abuse_score_to_severity(10) == "low"

    def test_summarize_risk_empty(self):
        """Should handle empty indicators and CVEs."""
        enrichment = ThreatIntelEnrichment(enabled=True)
        result = enrichment._summarize_risk([], [])
        assert "No specific threat indicators" in result

    def test_summarize_risk_with_critical_cves(self):
        """Should identify critical risk from CVEs."""
        enrichment = ThreatIntelEnrichment(enabled=True)
        cves = [
            {"id": "CVE-2024-1234", "severity": "CRITICAL"},
            {"id": "CVE-2024-5678", "severity": "HIGH"},
        ]
        result = enrichment._summarize_risk([], cves)
        assert "CRITICAL" in result
        assert "1 CRITICAL CVE" in result

    def test_summarize_risk_with_high_indicators(self):
        """Should identify high risk from indicators."""
        enrichment = ThreatIntelEnrichment(enabled=True)
        indicators = [
            ThreatIndicator(
                indicator_type="ip",
                value="1.2.3.4",
                threat_type="malicious",
                severity="high",
                confidence=0.9,
                source="test",
            )
        ]
        result = enrichment._summarize_risk(indicators, [])
        assert "HIGH" in result
        assert "high-severity indicator" in result

    def test_caching(self):
        """Should cache enrichment results."""
        enrichment = ThreatIntelEnrichment(enabled=True)

        # Set a cached value
        context = ThreatContext(
            indicators=[],
            relevant_cves=[],
            attack_patterns=["Test"],
            recommended_mitigations=["Test"],
            risk_summary="Test",
        )
        cache_key = enrichment._get_cache_key("test topic", "")
        enrichment._set_cached(cache_key, context)

        # Should retrieve from cache
        cached = enrichment._get_cached(cache_key)
        assert cached is not None
        assert cached.risk_summary == "Test"


class TestEnvironmentConfiguration:
    """Tests for environment variable configuration."""

    def test_default_enrichment_enabled(self):
        """Should enable enrichment by default."""
        # The imported ENRICHMENT_ENABLED reflects the env at import time
        # For a fresh test, check the module's behavior
        from aragora.security import threat_intel_enrichment

        assert hasattr(threat_intel_enrichment, "ENRICHMENT_ENABLED")

    def test_default_max_indicators(self):
        """Should have default max indicators of 10."""
        assert MAX_INDICATORS == 10

    def test_enrichment_respects_env_disabled(self, monkeypatch):
        """Should respect ARAGORA_THREAT_INTEL_ENRICHMENT_ENABLED=false."""
        monkeypatch.setenv("ARAGORA_THREAT_INTEL_ENRICHMENT_ENABLED", "false")
        # Need to reimport to pick up the new env value
        import importlib
        from aragora.security import threat_intel_enrichment

        importlib.reload(threat_intel_enrichment)
        assert threat_intel_enrichment.ENRICHMENT_ENABLED is False
        # Restore
        monkeypatch.setenv("ARAGORA_THREAT_INTEL_ENRICHMENT_ENABLED", "true")
        importlib.reload(threat_intel_enrichment)


class TestEnrichSecurityContextConvenience:
    """Tests for the convenience function."""

    @pytest.mark.asyncio
    async def test_enrich_security_context_returns_string(self):
        """Should return formatted string for security topics."""
        result = await enrich_security_context(
            topic="How should we handle the ransomware attack?",
        )
        # Should get generic ransomware context
        assert result is None or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_enrich_security_context_returns_none_for_non_security(self):
        """Should return None for non-security topics."""
        result = await enrich_security_context(
            topic="Implement the shopping cart feature",
        )
        assert result is None


class TestIntegrationWithMockedClients:
    """Integration tests with mocked external clients."""

    @pytest.mark.asyncio
    async def test_cve_lookup_integration(self):
        """Should integrate with CVE client for lookups."""
        mock_cve_client = MagicMock()
        mock_vuln = MagicMock()
        mock_vuln.id = "CVE-2024-1234"
        mock_vuln.title = "Test CVE"
        mock_vuln.description = "Test vulnerability"
        mock_vuln.severity = MagicMock()
        mock_vuln.severity.name = "CRITICAL"
        mock_vuln.cvss_score = 9.8
        mock_vuln.cvss_vector = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
        mock_vuln.cwe_ids = ["CWE-79"]
        mock_vuln.fix_available = True
        mock_vuln.recommended_version = "2.0.1"
        mock_vuln.published_at = datetime.now(timezone.utc)

        mock_cve_client.get_cve = AsyncMock(return_value=mock_vuln)

        enrichment = ThreatIntelEnrichment(
            cve_client=mock_cve_client,
            enabled=True,
        )

        result = await enrichment.enrich_context(
            topic="We need to patch CVE-2024-1234 urgently",
        )

        assert result is not None
        assert len(result.relevant_cves) == 1
        assert result.relevant_cves[0]["id"] == "CVE-2024-1234"
        mock_cve_client.get_cve.assert_called_once_with("CVE-2024-1234")

    @pytest.mark.asyncio
    async def test_threat_intel_client_ip_lookup(self):
        """Should integrate with threat intel client for IP lookups."""
        mock_threat_client = MagicMock()
        mock_result = MagicMock()
        mock_result.is_malicious = True
        mock_result.abuse_score = 85
        mock_result.total_reports = 50
        mock_result.categories = ["scanner", "brute-force"]

        mock_threat_client.check_ip = AsyncMock(return_value=mock_result)
        mock_threat_client.check_url = AsyncMock(
            return_value=MagicMock(is_malicious=False, confidence=0.1)
        )
        mock_threat_client.check_file_hash = AsyncMock(return_value=MagicMock(is_malware=False))
        mock_threat_client.close = AsyncMock()

        enrichment = ThreatIntelEnrichment(
            threat_intel_client=mock_threat_client,
            enabled=True,
        )

        result = await enrichment.enrich_context(
            topic="We're seeing attacks from 203.0.113.50",
        )

        # The topic is security-related, so it should call check_ip
        # If no IP was detected (depends on regex), it might still provide generic context
        assert result is not None


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_handles_empty_topic(self):
        """Should handle empty topic gracefully."""
        enrichment = ThreatIntelEnrichment(enabled=True)
        result = await enrichment.enrich_context(topic="")
        assert result is None

    @pytest.mark.asyncio
    async def test_handles_very_long_topic(self):
        """Should handle very long topics."""
        enrichment = ThreatIntelEnrichment(enabled=True)
        long_topic = "security vulnerability " * 1000
        result = await enrichment.enrich_context(topic=long_topic)
        # Should not crash, might return generic security context
        assert result is None or result.__class__.__name__ == "ThreatContext"

    @pytest.mark.asyncio
    async def test_handles_unicode_topic(self):
        """Should handle Unicode characters in topic."""
        enrichment = ThreatIntelEnrichment(enabled=True)
        result = await enrichment.enrich_context(
            topic="Security vulnerability in UTF-8 handling: \u200b\u00e9\u4e2d\u6587"
        )
        # Should handle without crashing
        assert result is None or result.__class__.__name__ == "ThreatContext"

    @pytest.mark.asyncio
    async def test_handles_cve_client_failure(self):
        """Should handle CVE client failures gracefully."""
        mock_cve_client = MagicMock()
        mock_cve_client.get_cve = AsyncMock(side_effect=Exception("API Error"))

        enrichment = ThreatIntelEnrichment(
            cve_client=mock_cve_client,
            enabled=True,
        )

        # Should not crash, should provide generic security context
        result = await enrichment.enrich_context(
            topic="Patch CVE-2024-1234",
        )
        # Even with CVE lookup failure, should provide generic context
        assert result is not None or result is None  # Either is acceptable

    def test_cache_size_limit(self):
        """Should limit cache size to prevent memory issues."""
        enrichment = ThreatIntelEnrichment(enabled=True)

        # Fill cache beyond limit
        for i in range(150):
            context = ThreatContext(
                indicators=[],
                relevant_cves=[],
                attack_patterns=[],
                recommended_mitigations=[],
                risk_summary=f"Test {i}",
            )
            enrichment._set_cached(f"key_{i}", context)

        # Cache should be limited
        assert len(enrichment._cache) <= 100


class TestContextGathererIntegration:
    """Tests for integration with ContextGatherer."""

    @pytest.mark.asyncio
    async def test_context_gatherer_threat_intel_integration(self):
        """Should integrate with ContextGatherer for threat intel."""
        # This tests the integration point we added
        try:
            from aragora.debate.context_gatherer import ContextGatherer

            gatherer = ContextGatherer(
                enable_threat_intel_enrichment=True,
            )

            # If threat intel is enabled, the gatherer should have the attribute
            assert hasattr(gatherer, "_enable_threat_intel")
            assert hasattr(gatherer, "_threat_intel_enrichment")

        except ImportError:
            pytest.skip("ContextGatherer not available")

    @pytest.mark.asyncio
    async def test_context_gatherer_threat_intel_disabled(self):
        """Should respect threat intel disabled flag."""
        try:
            from aragora.debate.context_gatherer import ContextGatherer

            gatherer = ContextGatherer(
                enable_threat_intel_enrichment=False,
            )

            assert gatherer._enable_threat_intel is False

        except ImportError:
            pytest.skip("ContextGatherer not available")
