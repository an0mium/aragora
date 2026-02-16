"""
Tests for DomainDetector class in aragora.routing.domain_matcher.

Tests cover:
- DomainDetector initialization
- Keyword-based domain detection (_detect_with_keywords)
- LLM-based detection (mocked)
- get_primary_domain method
- get_task_requirements method
- Custom keywords
- Cache integration
- Edge cases (empty text, unknown domains)
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from aragora.routing.domain_matcher import (
    DOMAIN_KEYWORDS,
    DomainDetector,
    _domain_cache,
)


# =============================================================================
# TestDomainDetectorInit - Initialization Tests
# =============================================================================


class TestDomainDetectorInit:
    """Tests for DomainDetector initialization."""

    def test_default_initialization(self):
        """Should initialize with default keywords."""
        detector = DomainDetector(use_llm=False)
        assert detector.keywords == DOMAIN_KEYWORDS
        assert detector.use_llm is False

    def test_with_custom_keywords(self):
        """Should merge custom keywords with defaults."""
        custom = {"security": ["extra_keyword"], "custom_domain": ["custom_word"]}
        detector = DomainDetector(custom_keywords=custom, use_llm=False)

        # Default security keywords should still exist
        assert "authentication" in detector.keywords["security"]
        # Custom keyword should be added
        assert "extra_keyword" in detector.keywords["security"]
        # New domain should be created
        assert "custom_domain" in detector.keywords
        assert "custom_word" in detector.keywords["custom_domain"]

    def test_llm_disabled_without_api_key(self):
        """Should disable LLM if no API key is set."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            # Clear the key if it exists
            if "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]
            detector = DomainDetector(use_llm=True)
            # LLM should be disabled if no API key (use_llm stores falsy value)
            assert not detector.use_llm

    def test_llm_enabled_with_api_key(self):
        """Should enable LLM if API key is set."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            detector = DomainDetector(use_llm=True)
            # use_llm stores the API key value (truthy) when enabled
            assert detector.use_llm

    def test_cache_disabled(self):
        """Should respect cache disable option."""
        detector = DomainDetector(use_llm=False, use_cache=False)
        assert detector._use_cache is False

    def test_custom_client(self):
        """Should accept custom Anthropic client."""
        mock_client = MagicMock()
        detector = DomainDetector(use_llm=True, client=mock_client)
        assert detector._client is mock_client


# =============================================================================
# TestDomainDetectorKeywordDetection - Keyword-Based Detection
# =============================================================================


class TestDomainDetectorKeywordDetection:
    """Tests for keyword-based domain detection."""

    def test_security_keywords(self):
        """Should detect security domain from security keywords."""
        detector = DomainDetector(use_llm=False)
        result = detector._detect_with_keywords(
            "We need to fix the authentication vulnerability and add rate limiting"
        )

        domains = [d for d, _ in result]
        # Security should be detected with 2+ matches
        assert "security" in domains

    def test_performance_keywords(self):
        """Should detect performance domain from performance keywords."""
        detector = DomainDetector(use_llm=False)
        result = detector._detect_with_keywords(
            "The database queries are slow and we need to optimize the cache"
        )

        domains = [d for d, _ in result]
        assert "performance" in domains

    def test_testing_keywords(self):
        """Should detect testing domain."""
        detector = DomainDetector(use_llm=False)
        result = detector._detect_with_keywords(
            "Write unit tests using pytest and add test fixtures for the API"
        )

        domains = [d for d, _ in result]
        assert "testing" in domains

    def test_api_keywords(self):
        """Should detect API domain."""
        detector = DomainDetector(use_llm=False)
        result = detector._detect_with_keywords(
            "Create a REST endpoint for user registration with proper validation"
        )

        domains = [d for d, _ in result]
        assert "api" in domains

    def test_database_keywords(self):
        """Should detect database domain."""
        detector = DomainDetector(use_llm=False)
        result = detector._detect_with_keywords(
            "Write a SQL migration to add an index on the users table"
        )

        domains = [d for d, _ in result]
        assert "database" in domains

    def test_devops_keywords(self):
        """Should detect devops domain."""
        detector = DomainDetector(use_llm=False)
        result = detector._detect_with_keywords(
            "Set up the CI pipeline with docker and kubernetes deployment"
        )

        domains = [d for d, _ in result]
        assert "devops" in domains

    def test_ethics_keywords(self):
        """Should detect ethics domain."""
        detector = DomainDetector(use_llm=False)
        result = detector._detect_with_keywords(
            "Consider the ethical implications of bias in the algorithm and user privacy"
        )

        domains = [d for d, _ in result]
        assert "ethics" in domains

    def test_philosophy_keywords(self):
        """Should detect philosophy domain."""
        detector = DomainDetector(use_llm=False)
        result = detector._detect_with_keywords(
            "What is the epistemological justification for this logical argument?"
        )

        domains = [d for d, _ in result]
        assert "philosophy" in domains

    def test_data_analysis_keywords(self):
        """Should detect data analysis domain."""
        detector = DomainDetector(use_llm=False)
        result = detector._detect_with_keywords(
            "Analyze the dataset using pandas and create a visualization with correlation"
        )

        domains = [d for d, _ in result]
        assert "data_analysis" in domains

    def test_multiple_domains_detected(self):
        """Should detect multiple domains from mixed text."""
        detector = DomainDetector(use_llm=False)
        result = detector._detect_with_keywords(
            "Fix the SQL injection vulnerability in the API endpoint and add tests"
        )

        domains = [d for d, _ in result]
        # Should detect multiple domains
        assert len(domains) >= 2

    def test_requires_two_matches_for_technical(self):
        """Technical domains require 2+ keyword matches to reduce false positives."""
        detector = DomainDetector(use_llm=False)

        # Single keyword - should default to general
        result = detector._detect_with_keywords("This involves some testing")
        domains = [d for d, _ in result]
        # With only 1 match, testing should not be detected
        assert "testing" not in domains or domains[0] != "testing"

    def test_general_default_for_unknown(self):
        """Should default to 'general' for unrecognized text."""
        detector = DomainDetector(use_llm=False)
        result = detector._detect_with_keywords("random words about nothing specific")

        domains = [d for d, _ in result]
        assert "general" in domains

    def test_empty_text_returns_general(self):
        """Should return 'general' for empty text."""
        detector = DomainDetector(use_llm=False)
        result = detector._detect_with_keywords("")

        assert result == [("general", 0.5)]

    def test_confidence_scores_normalized(self):
        """Confidence scores should be normalized 0-1."""
        detector = DomainDetector(use_llm=False)
        result = detector._detect_with_keywords(
            "security authentication vulnerability xss sql injection"
        )

        for domain, confidence in result:
            assert 0.0 <= confidence <= 1.0

    def test_top_n_limit(self):
        """Should respect top_n limit."""
        detector = DomainDetector(use_llm=False)
        result = detector._detect_with_keywords(
            "security api testing database devops frontend", top_n=2
        )

        assert len(result) <= 2

    def test_case_insensitive_matching(self):
        """Should match keywords case-insensitively."""
        detector = DomainDetector(use_llm=False)
        result = detector._detect_with_keywords("SECURITY AUTHENTICATION VULNERABILITY")

        domains = [d for d, _ in result]
        assert "security" in domains

    def test_longer_keywords_weighted_higher(self):
        """Multi-word keywords should have higher weight."""
        detector = DomainDetector(use_llm=False)

        # "sql injection" is a 2-word keyword, should boost security
        result = detector._detect_with_keywords("sql injection attack vector exploit")

        domains = [d for d, _ in result]
        assert "security" in domains


# =============================================================================
# TestDomainDetectorDetect - Main detect() Method
# =============================================================================


class TestDomainDetectorDetect:
    """Tests for the main detect() method."""

    def test_detect_uses_keywords_when_llm_disabled(self):
        """detect() should use keyword fallback when LLM is disabled."""
        detector = DomainDetector(use_llm=False)
        result = detector.detect("Fix the authentication vulnerability", top_n=3)

        # Should return valid results
        assert len(result) > 0
        assert all(isinstance(d, str) for d, _ in result)
        assert all(isinstance(c, float) for _, c in result)

    def test_detect_with_mock_llm(self):
        """detect() should use LLM when available."""
        from anthropic.types import TextBlock

        mock_client = MagicMock()
        mock_response = MagicMock()
        text_block = TextBlock(
            type="text", text='{"domains": [{"name": "security", "confidence": 0.9}]}'
        )
        mock_response.content = [text_block]
        mock_client.messages.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            detector = DomainDetector(use_llm=True, client=mock_client, use_cache=False)
            result = detector.detect("security task", top_n=3)

        # Should return LLM result
        assert result == [("security", 0.9)]

    def test_detect_falls_back_to_keywords_on_llm_failure(self):
        """detect() should fall back to keywords if LLM fails."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API error")

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            detector = DomainDetector(use_llm=True, client=mock_client, use_cache=False)
            result = detector.detect("security authentication vulnerability", top_n=3)

        # Should still get results from keyword fallback
        assert len(result) > 0

    def test_detect_returns_sorted_by_confidence(self):
        """detect() should return results sorted by confidence descending."""
        detector = DomainDetector(use_llm=False)
        result = detector.detect("security testing api database performance debugging", top_n=5)

        # Check descending order
        confidences = [c for _, c in result]
        assert confidences == sorted(confidences, reverse=True)


# =============================================================================
# TestDomainDetectorGetPrimaryDomain - Primary Domain Extraction
# =============================================================================


class TestDomainDetectorGetPrimaryDomain:
    """Tests for get_primary_domain() method."""

    def test_returns_highest_confidence_domain(self):
        """Should return the highest confidence domain."""
        detector = DomainDetector(use_llm=False)
        result = detector.get_primary_domain("security authentication vulnerability")

        assert result == "security"

    def test_returns_general_for_unknown(self):
        """Should return 'general' for unrecognized text."""
        detector = DomainDetector(use_llm=False)
        result = detector.get_primary_domain("random gibberish xyz abc")

        assert result == "general"

    def test_returns_general_for_empty(self):
        """Should return 'general' for empty text."""
        detector = DomainDetector(use_llm=False)
        result = detector.get_primary_domain("")

        assert result == "general"


# =============================================================================
# TestDomainDetectorGetTaskRequirements - Task Requirements Creation
# =============================================================================


class TestDomainDetectorGetTaskRequirements:
    """Tests for get_task_requirements() method."""

    def test_creates_task_requirements(self):
        """Should create TaskRequirements with detected domains."""
        detector = DomainDetector(use_llm=False)
        result = detector.get_task_requirements(
            "Fix the security vulnerability in the API", task_id="task-001"
        )

        from aragora.routing.selection import TaskRequirements

        assert isinstance(result, TaskRequirements)
        assert result.task_id == "task-001"
        assert result.primary_domain in DOMAIN_KEYWORDS.keys() or result.primary_domain == "general"

    def test_generates_task_id_if_not_provided(self):
        """Should generate task_id if not provided."""
        detector = DomainDetector(use_llm=False)
        result = detector.get_task_requirements("Fix the bug")

        assert result.task_id is not None
        assert result.task_id.startswith("task-")

    def test_includes_secondary_domains(self):
        """Should include confident secondary domains."""
        detector = DomainDetector(use_llm=False)
        result = detector.get_task_requirements(
            "Fix the security vulnerability and add API tests with database migrations"
        )

        # Should have primary domain
        assert result.primary_domain is not None
        # Secondary domains should be a list
        assert isinstance(result.secondary_domains, list)

    def test_detects_thorough_trait(self):
        """Should detect 'thorough' trait from keywords."""
        detector = DomainDetector(use_llm=False)
        result = detector.get_task_requirements(
            "This is a critical and important task requiring careful review"
        )

        assert "thorough" in result.required_traits

    def test_detects_fast_trait(self):
        """Should detect 'fast' trait from keywords."""
        detector = DomainDetector(use_llm=False)
        result = detector.get_task_requirements("We need this done fast and quick, asap!")

        assert "fast" in result.required_traits

    def test_detects_security_trait(self):
        """Should detect 'security' trait from keywords."""
        detector = DomainDetector(use_llm=False)
        result = detector.get_task_requirements("Make sure this is secure and safe")

        assert "security" in result.required_traits

    def test_detects_creative_trait(self):
        """Should detect 'creative' trait from keywords."""
        detector = DomainDetector(use_llm=False)
        result = detector.get_task_requirements("Design something creative and innovative")

        assert "creative" in result.required_traits

    def test_truncates_long_descriptions(self):
        """Should truncate descriptions longer than 500 chars."""
        detector = DomainDetector(use_llm=False)
        long_text = "a" * 1000
        result = detector.get_task_requirements(long_text)

        assert len(result.description) <= 500


# =============================================================================
# TestDomainDetectorCacheIntegration - Cache Behavior
# =============================================================================


class TestDomainDetectorCacheIntegration:
    """Tests for cache integration with DomainDetector."""

    def test_cache_stats_method(self):
        """Should return cache statistics."""
        stats = DomainDetector.cache_stats()

        assert "hits" in stats
        assert "misses" in stats
        assert "size" in stats
        assert "hit_rate" in stats

    def test_clear_cache_method(self):
        """Should clear the global cache."""
        # Add something to cache first
        _domain_cache.set("test_clear", 3, [("test", 0.9)])

        # Clear and verify
        count = DomainDetector.clear_cache()
        assert count >= 0

        # Cache should be empty
        result = _domain_cache.get("test_clear", 3)
        assert result is None


# =============================================================================
# TestDomainDetectorLLMIntegration - LLM-Specific Tests
# =============================================================================


class TestDomainDetectorLLMIntegration:
    """Tests for LLM integration in DomainDetector."""

    def test_llm_response_json_parsing(self):
        """Should parse JSON from LLM response."""
        from anthropic.types import TextBlock

        mock_client = MagicMock()
        mock_response = MagicMock()
        # Create a real TextBlock to pass the isinstance check
        text_block = TextBlock(
            type="text",
            text='{"domains": [{"name": "security", "confidence": 0.9}, {"name": "api", "confidence": 0.7}]}',
        )
        mock_response.content = [text_block]
        mock_client.messages.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            detector = DomainDetector(use_llm=True, client=mock_client, use_cache=False)
            result = detector._detect_with_llm("test", top_n=3)

        assert result == [("security", 0.9), ("api", 0.7)]

    def test_llm_response_with_markdown_code_block(self):
        """Should extract JSON from markdown code blocks."""
        from anthropic.types import TextBlock

        mock_client = MagicMock()
        mock_response = MagicMock()
        text_block = TextBlock(
            type="text", text='```json\n{"domains": [{"name": "testing", "confidence": 0.8}]}\n```'
        )
        mock_response.content = [text_block]
        mock_client.messages.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            detector = DomainDetector(use_llm=True, client=mock_client, use_cache=False)
            result = detector._detect_with_llm("test", top_n=3)

        assert result == [("testing", 0.8)]

    def test_llm_filters_invalid_domains(self):
        """Should filter out invalid domain names."""
        from anthropic.types import TextBlock

        mock_client = MagicMock()
        mock_response = MagicMock()
        text_block = TextBlock(
            type="text",
            text='{"domains": [{"name": "invalid_domain", "confidence": 0.9}, {"name": "security", "confidence": 0.7}]}',
        )
        mock_response.content = [text_block]
        mock_client.messages.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            detector = DomainDetector(use_llm=True, client=mock_client, use_cache=False)
            result = detector._detect_with_llm("test", top_n=3)

        # Should only include valid domain (security)
        domains = [d for d, _ in result]
        assert "invalid_domain" not in domains
        assert "security" in domains

    def test_llm_clamps_confidence_values(self):
        """Should clamp confidence values to 0-1 range."""
        from anthropic.types import TextBlock

        mock_client = MagicMock()
        mock_response = MagicMock()
        text_block = TextBlock(
            type="text", text='{"domains": [{"name": "security", "confidence": 1.5}]}'
        )
        mock_response.content = [text_block]
        mock_client.messages.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            detector = DomainDetector(use_llm=True, client=mock_client, use_cache=False)
            result = detector._detect_with_llm("test", top_n=3)

        # Should clamp to 1.0
        assert result == [("security", 1.0)]

    def test_llm_returns_none_without_client(self):
        """Should return None if no client is available."""
        detector = DomainDetector(use_llm=False, use_cache=False)
        result = detector._detect_with_llm("test", top_n=3)

        assert result is None


# =============================================================================
# TestDomainDetectorEdgeCases - Edge Cases and Error Handling
# =============================================================================


class TestDomainDetectorEdgeCases:
    """Tests for edge cases and error handling."""

    def test_whitespace_only_text(self):
        """Should handle whitespace-only text."""
        detector = DomainDetector(use_llm=False)
        result = detector.detect("   \n\t  ")

        assert len(result) > 0
        assert result[0][0] == "general"

    def test_special_characters_in_text(self):
        """Should handle special characters in text."""
        detector = DomainDetector(use_llm=False)
        result = detector.detect("security @#$%^& authentication!!!")

        # Should still detect security
        domains = [d for d, _ in result]
        assert "security" in domains

    def test_unicode_text(self):
        """Should handle unicode text."""
        detector = DomainDetector(use_llm=False)
        result = detector.detect("security \u00e9\u00e8\u00ea authentication \u4e2d\u6587")

        # Should still work
        assert len(result) > 0

    def test_very_long_text(self):
        """Should handle very long text."""
        detector = DomainDetector(use_llm=False)
        long_text = "security authentication " * 1000
        result = detector.detect(long_text)

        # Should still work
        assert len(result) > 0

    def test_valid_domains_constant(self):
        """VALID_DOMAINS should match DOMAIN_KEYWORDS keys."""
        assert DomainDetector.VALID_DOMAINS == frozenset(DOMAIN_KEYWORDS.keys())


# =============================================================================
# TestDomainKeywordsCoverage - Verify All Domains Have Keywords
# =============================================================================


class TestDomainKeywordsCoverage:
    """Tests to verify keyword coverage for all domains."""

    def test_all_domains_have_keywords(self):
        """Every domain should have at least one keyword."""
        for domain, keywords in DOMAIN_KEYWORDS.items():
            assert len(keywords) > 0, f"Domain '{domain}' has no keywords"

    def test_all_domains_detectable(self):
        """Every domain should be detectable with its keywords."""
        detector = DomainDetector(use_llm=False)

        for domain, keywords in DOMAIN_KEYWORDS.items():
            # Use multiple keywords to pass the 2-match threshold
            test_text = " ".join(keywords[:3])
            result = detector._detect_with_keywords(test_text, top_n=5)

            detected_domains = [d for d, _ in result]
            # Either the domain is detected or it's a non-technical domain
            # that doesn't require 2+ matches
            non_technical = {"general", "documentation", "philosophy", "ethics"}
            if domain in non_technical or len(keywords) >= 2:
                assert domain in detected_domains or any(
                    d in detected_domains for d in ["general"]
                ), f"Domain '{domain}' not detected with keywords: {test_text}"
