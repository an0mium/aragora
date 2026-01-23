"""
Tests for Spam Moderation Integration.

Tests the spam classifier integration with the content moderation pipeline,
including configuration, caching, thresholds, and error handling.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.moderation.spam_integration import (
    ContentModerationError,
    SpamCheckResult,
    SpamModerationConfig,
    SpamModerationIntegration,
    SpamVerdict,
    check_debate_content,
    get_spam_moderation,
    set_spam_moderation,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockClassificationResult:
    """Mock SpamClassificationResult for testing."""

    email_id: str = "test_email"
    is_spam: bool = False
    spam_score: float = 0.0
    confidence: float = 0.9
    reasons: List[str] = None
    content_score: float = 0.0
    sender_score: float = 0.0
    pattern_score: float = 0.0
    url_score: float = 0.0

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


@pytest.fixture
def mock_classifier():
    """Create a mock SpamClassifier."""
    classifier = MagicMock()
    classifier.initialize = AsyncMock()
    classifier.close = AsyncMock()
    classifier.classify_email = AsyncMock(return_value=MockClassificationResult())
    return classifier


@pytest.fixture
def config():
    """Create default test configuration."""
    return SpamModerationConfig(
        enabled=True,
        block_threshold=0.9,
        review_threshold=0.7,
        cache_enabled=True,
        cache_ttl_seconds=60,
        cache_max_size=100,
        fail_open=True,
        log_all_checks=False,
    )


@pytest.fixture
def moderation(mock_classifier, config):
    """Create a SpamModerationIntegration instance with mocked classifier."""
    return SpamModerationIntegration(classifier=mock_classifier, config=config)


@pytest.fixture
def clean_global_state():
    """Reset global moderation state before and after tests."""
    set_spam_moderation(None)
    yield
    set_spam_moderation(None)


# =============================================================================
# SpamVerdict Tests
# =============================================================================


class TestSpamVerdict:
    """Tests for SpamVerdict enum."""

    def test_verdict_values(self):
        """Test that all verdict values are strings."""
        assert SpamVerdict.CLEAN.value == "clean"
        assert SpamVerdict.SUSPICIOUS.value == "suspicious"
        assert SpamVerdict.SPAM.value == "spam"

    def test_verdict_is_string_enum(self):
        """Test that SpamVerdict is a string enum."""
        assert isinstance(SpamVerdict.CLEAN, str)
        assert SpamVerdict.CLEAN == "clean"


# =============================================================================
# SpamCheckResult Tests
# =============================================================================


class TestSpamCheckResult:
    """Tests for SpamCheckResult dataclass."""

    def test_default_values(self):
        """Test default values for SpamCheckResult."""
        result = SpamCheckResult(
            verdict=SpamVerdict.CLEAN,
            confidence=0.95,
        )
        assert result.verdict == SpamVerdict.CLEAN
        assert result.confidence == 0.95
        assert result.reasons == []
        assert result.should_block is False
        assert result.should_flag_for_review is False
        assert result.spam_score == 0.0

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = SpamCheckResult(
            verdict=SpamVerdict.SPAM,
            confidence=0.95,
            reasons=["Suspicious URL", "Spam keywords"],
            should_block=True,
            should_flag_for_review=True,
            spam_score=0.92,
            content_hash="abc123",
        )
        d = result.to_dict()

        assert d["verdict"] == "spam"
        assert d["confidence"] == 0.95
        assert d["reasons"] == ["Suspicious URL", "Spam keywords"]
        assert d["should_block"] is True
        assert d["should_flag_for_review"] is True
        assert d["spam_score"] == 0.92
        assert "checked_at" in d
        assert "scores" in d

    def test_checked_at_timestamp(self):
        """Test that checked_at is set automatically."""
        result = SpamCheckResult(verdict=SpamVerdict.CLEAN, confidence=0.5)
        assert result.checked_at is not None
        assert result.checked_at.tzinfo == timezone.utc


# =============================================================================
# SpamModerationConfig Tests
# =============================================================================


class TestSpamModerationConfig:
    """Tests for SpamModerationConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SpamModerationConfig()
        assert config.enabled is True
        assert config.block_threshold == 0.9
        assert config.review_threshold == 0.7
        assert config.cache_enabled is True
        assert config.fail_open is True

    def test_from_env_defaults(self):
        """Test loading from environment with defaults."""
        with patch.dict(os.environ, {}, clear=True):
            config = SpamModerationConfig.from_env()
            assert config.enabled is True
            assert config.block_threshold == 0.9

    def test_from_env_custom_values(self):
        """Test loading custom values from environment."""
        env_vars = {
            "ARAGORA_SPAM_CHECK_ENABLED": "false",
            "ARAGORA_SPAM_BLOCK_THRESHOLD": "0.85",
            "ARAGORA_SPAM_REVIEW_THRESHOLD": "0.6",
            "ARAGORA_SPAM_CACHE_ENABLED": "false",
            "ARAGORA_SPAM_CACHE_TTL": "600",
            "ARAGORA_SPAM_CACHE_SIZE": "500",
            "ARAGORA_SPAM_FAIL_OPEN": "false",
            "ARAGORA_SPAM_LOG_ALL": "true",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = SpamModerationConfig.from_env()
            assert config.enabled is False
            assert config.block_threshold == 0.85
            assert config.review_threshold == 0.6
            assert config.cache_enabled is False
            assert config.cache_ttl_seconds == 600
            assert config.cache_max_size == 500
            assert config.fail_open is False
            assert config.log_all_checks is True


# =============================================================================
# SpamModerationIntegration Tests
# =============================================================================


class TestSpamModerationIntegration:
    """Tests for SpamModerationIntegration class."""

    @pytest.mark.asyncio
    async def test_initialize(self, moderation, mock_classifier):
        """Test initialization."""
        assert not moderation._initialized
        await moderation.initialize()
        assert moderation._initialized
        # Classifier was provided, so its initialize should not be called
        # (it's already assumed to be initialized when passed in)

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, moderation):
        """Test that initialize is idempotent."""
        await moderation.initialize()
        await moderation.initialize()
        await moderation.initialize()
        # Should not raise

    @pytest.mark.asyncio
    async def test_check_content_clean(self, moderation, mock_classifier):
        """Test checking clean content."""
        mock_classifier.classify_email.return_value = MockClassificationResult(
            spam_score=0.1,
            confidence=0.95,
        )

        result = await moderation.check_content("Hello, this is a clean message.")

        assert result.verdict == SpamVerdict.CLEAN
        assert result.should_block is False
        assert result.should_flag_for_review is False
        assert moderation.statistics["passed"] == 1

    @pytest.mark.asyncio
    async def test_check_content_suspicious(self, moderation, mock_classifier):
        """Test checking suspicious content."""
        mock_classifier.classify_email.return_value = MockClassificationResult(
            spam_score=0.75,
            confidence=0.85,
            reasons=["Suspicious patterns detected"],
        )

        result = await moderation.check_content("Click here for free money!")

        assert result.verdict == SpamVerdict.SUSPICIOUS
        assert result.should_block is False
        assert result.should_flag_for_review is True
        assert moderation.statistics["flagged"] == 1

    @pytest.mark.asyncio
    async def test_check_content_spam(self, moderation, mock_classifier):
        """Test checking spam content."""
        mock_classifier.classify_email.return_value = MockClassificationResult(
            spam_score=0.95,
            confidence=0.99,
            reasons=["Known spam domain", "Phishing URL detected"],
        )

        result = await moderation.check_content("You won a million dollars!")

        assert result.verdict == SpamVerdict.SPAM
        assert result.should_block is True
        assert result.should_flag_for_review is True
        assert moderation.statistics["blocked"] == 1

    @pytest.mark.asyncio
    async def test_check_content_disabled(self, mock_classifier):
        """Test that checks are skipped when disabled."""
        config = SpamModerationConfig(enabled=False)
        moderation = SpamModerationIntegration(classifier=mock_classifier, config=config)

        result = await moderation.check_content("Any content")

        assert result.verdict == SpamVerdict.CLEAN
        assert result.should_block is False
        mock_classifier.classify_email.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_debate_input(self, moderation, mock_classifier):
        """Test checking debate input."""
        mock_classifier.classify_email.return_value = MockClassificationResult(
            spam_score=0.1,
            confidence=0.9,
        )

        result = await moderation.check_debate_input(
            proposal="Should we implement caching?",
            context="For the authentication system",
        )

        assert result.verdict == SpamVerdict.CLEAN
        # Verify the classifier was called with combined content
        call_args = mock_classifier.classify_email.call_args
        assert "Should we implement caching?" in call_args.kwargs.get("body", "")

    @pytest.mark.asyncio
    async def test_caching(self, moderation, mock_classifier):
        """Test that results are cached."""
        mock_classifier.classify_email.return_value = MockClassificationResult(
            spam_score=0.1,
        )

        # First call
        result1 = await moderation.check_content("Same content")
        assert mock_classifier.classify_email.call_count == 1

        # Second call with same content
        result2 = await moderation.check_content("Same content")
        assert mock_classifier.classify_email.call_count == 1  # Still 1 - cached!

        # Results should be equal
        assert result1.verdict == result2.verdict
        assert moderation.statistics["cache_hits"] == 1

    @pytest.mark.asyncio
    async def test_cache_disabled(self, mock_classifier):
        """Test behavior when cache is disabled."""
        config = SpamModerationConfig(cache_enabled=False)
        moderation = SpamModerationIntegration(classifier=mock_classifier, config=config)
        mock_classifier.classify_email.return_value = MockClassificationResult(
            spam_score=0.1,
        )

        await moderation.check_content("Content")
        await moderation.check_content("Content")

        assert mock_classifier.classify_email.call_count == 2

    @pytest.mark.asyncio
    async def test_fail_open_on_error(self, moderation, mock_classifier):
        """Test fail-open behavior on classifier error."""
        mock_classifier.classify_email.side_effect = RuntimeError("Classifier error")

        result = await moderation.check_content("Some content")

        assert result.verdict == SpamVerdict.CLEAN
        assert result.should_block is False
        assert result.should_flag_for_review is True  # Flag for manual review
        assert moderation.statistics["errors"] == 1

    @pytest.mark.asyncio
    async def test_fail_closed_on_error(self, mock_classifier):
        """Test fail-closed behavior on classifier error."""
        config = SpamModerationConfig(fail_open=False)
        moderation = SpamModerationIntegration(classifier=mock_classifier, config=config)
        mock_classifier.classify_email.side_effect = RuntimeError("Classifier error")

        with pytest.raises(ContentModerationError):
            await moderation.check_content("Some content")

    @pytest.mark.asyncio
    async def test_statistics(self, moderation, mock_classifier):
        """Test statistics tracking."""
        # Clean content
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.1)
        await moderation.check_content("Clean 1")
        await moderation.check_content("Clean 2")

        # Suspicious content
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.75)
        await moderation.check_content("Suspicious")

        # Spam content
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.95)
        await moderation.check_content("Spam")

        stats = moderation.statistics
        assert stats["checks"] == 4
        assert stats["passed"] == 2
        assert stats["flagged"] == 1
        assert stats["blocked"] == 1

    @pytest.mark.asyncio
    async def test_reset_statistics(self, moderation, mock_classifier):
        """Test statistics reset."""
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.1)
        await moderation.check_content("Content")

        old_stats = moderation.reset_statistics()
        assert old_stats["checks"] == 1
        assert moderation.statistics["checks"] == 0

    def test_clear_cache(self, moderation):
        """Test cache clearing."""
        # Add some entries to cache
        moderation._cache["hash1"] = (
            SpamCheckResult(verdict=SpamVerdict.CLEAN, confidence=0.9),
            1000,
        )
        moderation._cache["hash2"] = (
            SpamCheckResult(verdict=SpamVerdict.SPAM, confidence=0.9),
            1000,
        )

        count = moderation.clear_cache()
        assert count == 2
        assert len(moderation._cache) == 0

    @pytest.mark.asyncio
    async def test_close(self, moderation, mock_classifier):
        """Test closing resources."""
        await moderation.initialize()
        await moderation.close()

        mock_classifier.close.assert_called_once()
        assert not moderation._initialized


# =============================================================================
# Content Moderation Error Tests
# =============================================================================


class TestContentModerationError:
    """Tests for ContentModerationError exception."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = ContentModerationError("Content blocked")
        assert str(error) == "Content blocked"
        assert error.verdict == SpamVerdict.SPAM
        assert error.confidence == 0.0
        assert error.reasons == []

    def test_error_with_details(self):
        """Test error with full details."""
        error = ContentModerationError(
            "Spam detected",
            verdict=SpamVerdict.SUSPICIOUS,
            confidence=0.85,
            reasons=["Suspicious URL", "Spam keywords"],
        )
        assert error.verdict == SpamVerdict.SUSPICIOUS
        assert error.confidence == 0.85
        assert len(error.reasons) == 2


# =============================================================================
# Global Instance Tests
# =============================================================================


class TestGlobalInstance:
    """Tests for global instance management."""

    def test_get_spam_moderation_creates_instance(self, clean_global_state):
        """Test that get_spam_moderation creates an instance."""
        moderation = get_spam_moderation()
        assert moderation is not None
        assert isinstance(moderation, SpamModerationIntegration)

    def test_get_spam_moderation_returns_same_instance(self, clean_global_state):
        """Test that get_spam_moderation returns the same instance."""
        mod1 = get_spam_moderation()
        mod2 = get_spam_moderation()
        assert mod1 is mod2

    def test_set_spam_moderation(self, clean_global_state, moderation):
        """Test setting custom moderation instance."""
        set_spam_moderation(moderation)
        assert get_spam_moderation() is moderation


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestCheckDebateContent:
    """Tests for the check_debate_content convenience function."""

    @pytest.mark.asyncio
    async def test_check_debate_content_basic(self, clean_global_state, mock_classifier):
        """Test basic debate content checking."""
        mock_classifier.classify_email.return_value = MockClassificationResult(
            spam_score=0.1,
            confidence=0.9,
        )

        moderation = SpamModerationIntegration(classifier=mock_classifier)
        set_spam_moderation(moderation)

        result = await check_debate_content("Design a rate limiter")

        assert result.verdict == SpamVerdict.CLEAN
        assert result.should_block is False

    @pytest.mark.asyncio
    async def test_check_debate_content_with_context(self, clean_global_state, mock_classifier):
        """Test debate content checking with context."""
        mock_classifier.classify_email.return_value = MockClassificationResult(
            spam_score=0.1,
        )

        moderation = SpamModerationIntegration(classifier=mock_classifier)
        set_spam_moderation(moderation)

        result = await check_debate_content(
            proposal="Implement caching",
            context="For the authentication system",
            metadata={"user_id": "user_123"},
        )

        assert result.verdict == SpamVerdict.CLEAN


# =============================================================================
# Threshold Boundary Tests
# =============================================================================


class TestThresholds:
    """Tests for threshold boundary conditions."""

    @pytest.mark.asyncio
    async def test_exactly_at_review_threshold(self, mock_classifier, config):
        """Test score exactly at review threshold."""
        config.review_threshold = 0.7
        moderation = SpamModerationIntegration(classifier=mock_classifier, config=config)
        mock_classifier.classify_email.return_value = MockClassificationResult(
            spam_score=0.7,  # Exactly at threshold
        )

        result = await moderation.check_content("Test")

        assert result.verdict == SpamVerdict.SUSPICIOUS
        assert result.should_flag_for_review is True
        assert result.should_block is False

    @pytest.mark.asyncio
    async def test_exactly_at_block_threshold(self, mock_classifier, config):
        """Test score exactly at block threshold."""
        config.block_threshold = 0.9
        moderation = SpamModerationIntegration(classifier=mock_classifier, config=config)
        mock_classifier.classify_email.return_value = MockClassificationResult(
            spam_score=0.9,  # Exactly at threshold
        )

        result = await moderation.check_content("Test")

        assert result.verdict == SpamVerdict.SPAM
        assert result.should_block is True

    @pytest.mark.asyncio
    async def test_just_below_review_threshold(self, mock_classifier, config):
        """Test score just below review threshold."""
        config.review_threshold = 0.7
        moderation = SpamModerationIntegration(classifier=mock_classifier, config=config)
        mock_classifier.classify_email.return_value = MockClassificationResult(
            spam_score=0.69,
        )

        result = await moderation.check_content("Test")

        assert result.verdict == SpamVerdict.CLEAN
        assert result.should_flag_for_review is False

    @pytest.mark.asyncio
    async def test_custom_thresholds(self, mock_classifier):
        """Test with custom threshold values."""
        config = SpamModerationConfig(
            block_threshold=0.5,  # Lower block threshold
            review_threshold=0.3,  # Lower review threshold
        )
        moderation = SpamModerationIntegration(classifier=mock_classifier, config=config)

        # Score that would be clean with default thresholds
        mock_classifier.classify_email.return_value = MockClassificationResult(
            spam_score=0.55,
        )

        result = await moderation.check_content("Test")

        assert result.should_block is True  # Blocked with custom threshold


# =============================================================================
# Cache Eviction Tests
# =============================================================================


class TestCacheEviction:
    """Tests for cache eviction behavior."""

    @pytest.mark.asyncio
    async def test_cache_eviction_on_full(self, mock_classifier):
        """Test cache eviction when full."""
        config = SpamModerationConfig(
            cache_enabled=True,
            cache_max_size=5,  # Small cache for testing
        )
        moderation = SpamModerationIntegration(classifier=mock_classifier, config=config)
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.1)

        # Fill the cache beyond max size
        for i in range(10):
            await moderation.check_content(f"Content {i}")

        # Cache should be at most max_size after eviction
        # With max_size=5, eviction removes max(1, 5//10)=1 entry when full
        # So cache stays bounded around max_size
        assert len(moderation._cache) <= config.cache_max_size


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for spam moderation."""

    @pytest.mark.asyncio
    async def test_full_moderation_flow(self, moderation, mock_classifier):
        """Test complete moderation flow."""
        # Initialize
        await moderation.initialize()
        assert moderation._initialized

        # Check clean content
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.1)
        clean_result = await moderation.check_debate_input("Design an API")
        assert clean_result.verdict == SpamVerdict.CLEAN

        # Check spam content
        mock_classifier.classify_email.return_value = MockClassificationResult(
            spam_score=0.95,
            reasons=["Phishing detected"],
        )
        spam_result = await moderation.check_debate_input("Click here to win money!")
        assert spam_result.verdict == SpamVerdict.SPAM
        assert spam_result.should_block is True

        # Verify statistics
        stats = moderation.statistics
        assert stats["checks"] == 2
        assert stats["passed"] == 1
        assert stats["blocked"] == 1

        # Close
        await moderation.close()
        assert not moderation._initialized
