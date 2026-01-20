"""
Tests for the domain_matcher module.

Tests domain detection via keywords and caching behavior.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from aragora.routing.domain_matcher import (
    DOMAIN_KEYWORDS,
    DomainDetector,
    _DomainCache,
)


# =============================================================================
# _DomainCache Tests
# =============================================================================


class TestDomainCache:
    """Tests for the _DomainCache class."""

    def test_cache_creation_defaults(self):
        """Test cache creation with default values."""
        cache = _DomainCache()
        stats = cache.stats()
        assert stats["size"] == 0
        assert stats["max_size"] == 500
        assert stats["ttl_seconds"] == 3600
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

    def test_cache_creation_custom(self):
        """Test cache creation with custom values."""
        cache = _DomainCache(max_size=100, ttl_seconds=60)
        stats = cache.stats()
        assert stats["max_size"] == 100
        assert stats["ttl_seconds"] == 60

    def test_cache_set_and_get(self):
        """Test basic set and get operations."""
        cache = _DomainCache()
        result = [("security", 0.9), ("api", 0.7)]

        cache.set("test query", 3, result)

        retrieved = cache.get("test query", 3)
        assert retrieved == result

        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 0

    def test_cache_miss(self):
        """Test cache miss for non-existent key."""
        cache = _DomainCache()

        result = cache.get("nonexistent", 3)
        assert result is None

        stats = cache.stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

    def test_cache_different_top_n(self):
        """Test that different top_n values create different cache entries."""
        cache = _DomainCache()
        result1 = [("security", 0.9)]
        result2 = [("security", 0.9), ("api", 0.7)]

        cache.set("test query", 1, result1)
        cache.set("test query", 2, result2)

        assert cache.get("test query", 1) == result1
        assert cache.get("test query", 2) == result2
        assert cache.stats()["size"] == 2

    def test_cache_ttl_expiration(self):
        """Test that cache entries expire after TTL."""
        cache = _DomainCache(ttl_seconds=1)
        result = [("security", 0.9)]

        cache.set("test query", 3, result)
        assert cache.get("test query", 3) == result

        # Wait for TTL to expire
        time.sleep(1.1)

        assert cache.get("test query", 3) is None
        stats = cache.stats()
        assert stats["size"] == 0  # Entry removed on expired access

    def test_cache_eviction_at_max_size(self):
        """Test that oldest entries are evicted when cache reaches max size."""
        cache = _DomainCache(max_size=5)

        # Fill cache
        for i in range(5):
            cache.set(f"query{i}", 3, [(f"domain{i}", 0.9)])
            time.sleep(0.01)  # Small delay to ensure different timestamps

        assert cache.stats()["size"] == 5

        # Add one more - should trigger eviction
        cache.set("query_new", 3, [("new_domain", 0.9)])

        # Should have evicted oldest entry (10% = 1 entry)
        assert cache.stats()["size"] <= 5

    def test_cache_clear(self):
        """Test clearing the cache."""
        cache = _DomainCache()

        cache.set("query1", 3, [("domain1", 0.9)])
        cache.set("query2", 3, [("domain2", 0.8)])
        assert cache.stats()["size"] == 2

        cleared = cache.clear()
        assert cleared == 2
        assert cache.stats()["size"] == 0

    def test_cache_hit_rate(self):
        """Test hit rate calculation."""
        cache = _DomainCache()
        cache.set("existing", 3, [("security", 0.9)])

        # 2 hits
        cache.get("existing", 3)
        cache.get("existing", 3)

        # 1 miss
        cache.get("nonexistent", 3)

        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2 / 3)

    def test_cache_key_normalization(self):
        """Test that cache keys are normalized (case-insensitive, trimmed)."""
        cache = _DomainCache()
        result = [("security", 0.9)]

        cache.set("Test Query", 3, result)

        # Should match regardless of case/whitespace
        assert cache.get("test query", 3) == result
        assert cache.get("  TEST QUERY  ", 3) == result


# =============================================================================
# DomainDetector Tests - Keyword Detection
# =============================================================================


class TestDomainDetectorKeywords:
    """Tests for DomainDetector keyword-based detection."""

    def test_detector_creation_default(self):
        """Test detector creation with defaults."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            detector = DomainDetector(use_llm=False)
        assert detector.keywords == DOMAIN_KEYWORDS
        assert detector.use_llm is False

    def test_detector_custom_keywords(self):
        """Test detector with custom keywords."""
        custom = {"security": ["custom_sec"], "new_domain": ["new_word"]}
        detector = DomainDetector(custom_keywords=custom, use_llm=False)

        assert "custom_sec" in detector.keywords["security"]
        assert "new_domain" in detector.keywords
        assert "new_word" in detector.keywords["new_domain"]

    def test_detect_security_domain(self):
        """Test detection of security-related tasks."""
        detector = DomainDetector(use_llm=False)

        result = detector.detect("Fix SQL injection vulnerability and add authentication")

        domains = [d for d, _ in result]
        assert "security" in domains

    def test_detect_performance_domain(self):
        """Test detection of performance-related tasks."""
        detector = DomainDetector(use_llm=False)

        result = detector.detect("Optimize database query performance and add caching")

        domains = [d for d, _ in result]
        assert "performance" in domains or "database" in domains

    def test_detect_api_domain(self):
        """Test detection of API-related tasks."""
        detector = DomainDetector(use_llm=False)

        result = detector.detect("Create REST API endpoint for user registration")

        domains = [d for d, _ in result]
        assert "api" in domains

    def test_detect_testing_domain(self):
        """Test detection of testing-related tasks."""
        detector = DomainDetector(use_llm=False)

        result = detector.detect("Write unit tests using pytest with mock fixtures")

        domains = [d for d, _ in result]
        assert "testing" in domains

    def test_detect_philosophy_domain(self):
        """Test detection of philosophical topics."""
        detector = DomainDetector(use_llm=False)

        result = detector.detect("Epistemological analysis of knowledge and belief justification")

        domains = [d for d, _ in result]
        assert "philosophy" in domains

    def test_detect_ethics_domain(self):
        """Test detection of ethics-related tasks."""
        detector = DomainDetector(use_llm=False)

        result = detector.detect("Evaluate AI fairness and bias in recommendation systems")

        domains = [d for d, _ in result]
        assert "ethics" in domains

    def test_detect_multiple_domains(self):
        """Test detection of multiple relevant domains."""
        detector = DomainDetector(use_llm=False)

        # Use query with multiple keyword matches per domain to trigger detection
        result = detector.detect(
            "Implement secure authentication API endpoint with database SQL query "
            "and REST request handler for user authorization"
        )

        domains = [d for d, _ in result]
        # Should detect multiple domains (security, api, database all have 2+ matches)
        assert len(result) >= 2

    def test_detect_general_domain(self):
        """Test that non-technical tasks get 'general' domain."""
        detector = DomainDetector(use_llm=False)

        result = detector.detect("Create a function to add two numbers")

        domains = [d for d, _ in result]
        assert "general" in domains

    def test_detect_top_n_limit(self):
        """Test that results are limited to top_n."""
        detector = DomainDetector(use_llm=False)

        result = detector.detect(
            "Security authentication API database testing performance optimization", top_n=2
        )

        assert len(result) <= 2

    def test_detect_confidence_ordering(self):
        """Test that results are ordered by confidence."""
        detector = DomainDetector(use_llm=False)

        result = detector.detect("Security vulnerability XSS attack authentication authorization")

        if len(result) > 1:
            confidences = [c for _, c in result]
            assert confidences == sorted(confidences, reverse=True)

    def test_detect_requires_multiple_matches(self):
        """Test that technical domains require 2+ keyword matches."""
        detector = DomainDetector(use_llm=False)

        # Single mention should not trigger technical domain
        result = detector.detect("The performance was okay")

        domains = [d for d, _ in result]
        # Should not be classified as performance with just one mention
        # (unless there are other signals)
        assert "general" in domains or len(result) == 0


# =============================================================================
# DomainDetector Tests - LLM Detection (Mocked)
# =============================================================================


class TestDomainDetectorLLM:
    """Tests for DomainDetector LLM-based detection."""

    def test_llm_detection_success(self):
        """Test successful LLM domain detection."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='{"domains": [{"name": "security", "confidence": 0.95}]}')
        ]
        mock_client.messages.create.return_value = mock_response

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"}):
            detector = DomainDetector(use_llm=True, client=mock_client, use_cache=False)
            result = detector.detect("Fix SQL injection vulnerability")

        assert result == [("security", 0.95)]
        mock_client.messages.create.assert_called_once()

    def test_llm_detection_with_json_block(self):
        """Test LLM detection with markdown JSON block."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='```json\n{"domains": [{"name": "api", "confidence": 0.9}]}\n```')
        ]
        mock_client.messages.create.return_value = mock_response

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"}):
            detector = DomainDetector(use_llm=True, client=mock_client, use_cache=False)
            result = detector.detect("Create REST endpoint")

        assert result == [("api", 0.9)]

    def test_llm_detection_invalid_domain(self):
        """Test LLM detection with invalid domain falls back to keywords."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='{"domains": [{"name": "invalid_domain", "confidence": 0.9}]}')
        ]
        mock_client.messages.create.return_value = mock_response

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"}):
            detector = DomainDetector(use_llm=True, client=mock_client, use_cache=False)
            result = detector.detect("Fix SQL injection vulnerability authentication")

        # Should fall back to keywords
        domains = [d for d, _ in result]
        assert "security" in domains

    def test_llm_detection_failure_fallback(self):
        """Test that LLM failure falls back to keywords."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API Error")

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"}):
            detector = DomainDetector(use_llm=True, client=mock_client, use_cache=False)
            result = detector.detect("Fix SQL injection vulnerability authentication")

        # Should fall back to keywords
        domains = [d for d, _ in result]
        assert "security" in domains

    def test_llm_detection_caching(self):
        """Test that LLM results are cached."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='{"domains": [{"name": "security", "confidence": 0.9}]}')
        ]
        mock_client.messages.create.return_value = mock_response

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"}):
            detector = DomainDetector(use_llm=True, client=mock_client, use_cache=True)
            DomainDetector.clear_cache()

            # First call - should hit API
            result1 = detector.detect("Fix SQL injection")
            call_count_1 = mock_client.messages.create.call_count

            # Second call - should hit cache
            result2 = detector.detect("Fix SQL injection")
            call_count_2 = mock_client.messages.create.call_count

        assert result1 == result2
        assert call_count_1 == 1
        assert call_count_2 == 1  # No additional calls

    def test_cache_stats_static_method(self):
        """Test the static cache_stats method."""
        DomainDetector.clear_cache()
        stats = DomainDetector.cache_stats()

        assert "size" in stats
        assert "hits" in stats
        assert "misses" in stats

    def test_clear_cache_static_method(self):
        """Test the static clear_cache method."""
        DomainDetector.clear_cache()

        # The actual cache is module-level, so clearing always works
        cleared = DomainDetector.clear_cache()
        assert cleared == 0  # Second clear should return 0


# =============================================================================
# DomainDetector Tests - Edge Cases
# =============================================================================


class TestDomainDetectorEdgeCases:
    """Edge case tests for DomainDetector."""

    def test_empty_text(self):
        """Test detection with empty text."""
        detector = DomainDetector(use_llm=False)
        result = detector.detect("")
        # Empty text may return general domain or empty list
        if result:
            assert result[0][0] == "general" or len(result) == 0

    def test_whitespace_only(self):
        """Test detection with whitespace-only text."""
        detector = DomainDetector(use_llm=False)
        result = detector.detect("   \n\t  ")
        # Whitespace-only text may return general domain or empty list
        if result:
            assert result[0][0] == "general" or len(result) == 0

    def test_very_long_text(self):
        """Test detection with very long text."""
        detector = DomainDetector(use_llm=False)
        long_text = "security " * 1000  # 10000+ characters
        result = detector.detect(long_text)
        assert len(result) > 0

    def test_special_characters(self):
        """Test detection with special characters."""
        detector = DomainDetector(use_llm=False)
        # Need 2+ security keyword matches for technical domain
        result = detector.detect("Fix @#$% authentication!! & encryption??? password security")

        domains = [d for d, _ in result]
        assert "security" in domains

    def test_case_insensitive(self):
        """Test that keyword matching is case insensitive."""
        detector = DomainDetector(use_llm=False)

        result_lower = detector.detect("sql injection vulnerability security")
        result_upper = detector.detect("SQL INJECTION VULNERABILITY SECURITY")
        result_mixed = detector.detect("SQL Injection Vulnerability Security")

        # All should detect security
        assert all(
            "security" in [d for d, _ in r] for r in [result_lower, result_upper, result_mixed]
        )

    def test_multi_word_keywords(self):
        """Test detection of multi-word keywords."""
        detector = DomainDetector(use_llm=False)

        result = detector.detect("Implement SQL injection prevention with access control")

        domains = [d for d, _ in result]
        assert "security" in domains  # "sql injection" and "access control" are multi-word

    def test_confidence_bounds(self):
        """Test that confidence values are bounded 0-1."""
        detector = DomainDetector(use_llm=False)

        result = detector.detect("security authentication encryption authorization")

        for domain, confidence in result:
            assert 0.0 <= confidence <= 1.0
