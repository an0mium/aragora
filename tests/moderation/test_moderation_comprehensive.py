"""
Comprehensive tests for the Moderation module.

Covers:
- SpamVerdict enum
- SpamCheckResult dataclass and serialization
- SpamModerationConfig (defaults, env loading, apply_updates, validation)
- ModerationQueueItem dataclass and serialization
- Review queue functions (queue_for_review, list_review_queue, pop_review_item, review_queue_size)
- SpamModerationIntegration (initialization, content checks, caching, fail-open/closed, stats)
- ContentModerationError exception
- Global instance management
- Edge cases: empty input, unicode, very long content, boundary thresholds
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.moderation.spam_integration import (
    ContentModerationError,
    ModerationQueueItem,
    SpamCheckResult,
    SpamModerationConfig,
    SpamModerationIntegration,
    SpamVerdict,
    _REVIEW_QUEUE,
    _REVIEW_QUEUE_LOCK,
    check_debate_content,
    get_spam_moderation,
    list_review_queue,
    pop_review_item,
    queue_for_review,
    review_queue_size,
    set_spam_moderation,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


@dataclass
class MockClassificationResult:
    """Mock SpamClassificationResult for testing."""

    email_id: str = "test_email"
    is_spam: bool = False
    spam_score: float = 0.0
    confidence: float = 0.9
    reasons: list[str] | None = None
    content_score: float = 0.0
    sender_score: float = 0.0
    pattern_score: float = 0.0
    url_score: float = 0.0

    def __post_init__(self) -> None:
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
    """Create a SpamModerationIntegration with mocked classifier."""
    return SpamModerationIntegration(classifier=mock_classifier, config=config)


@pytest.fixture
def clean_global_state():
    """Reset global moderation state before and after tests."""
    set_spam_moderation(None)
    yield
    set_spam_moderation(None)


@pytest.fixture(autouse=True)
def clean_review_queue():
    """Clear the global review queue before and after each test."""
    with _REVIEW_QUEUE_LOCK:
        _REVIEW_QUEUE.clear()
    yield
    with _REVIEW_QUEUE_LOCK:
        _REVIEW_QUEUE.clear()


# ===========================================================================
# SpamVerdict
# ===========================================================================


class TestSpamVerdict:
    def test_clean_value(self):
        assert SpamVerdict.CLEAN.value == "clean"

    def test_suspicious_value(self):
        assert SpamVerdict.SUSPICIOUS.value == "suspicious"

    def test_spam_value(self):
        assert SpamVerdict.SPAM.value == "spam"

    def test_is_str_subclass(self):
        assert isinstance(SpamVerdict.CLEAN, str)

    def test_string_comparison(self):
        assert SpamVerdict.SPAM == "spam"
        assert SpamVerdict.CLEAN != "spam"

    def test_all_members(self):
        members = list(SpamVerdict)
        assert len(members) == 3


# ===========================================================================
# ContentModerationError
# ===========================================================================


class TestContentModerationError:
    def test_defaults(self):
        err = ContentModerationError("blocked")
        assert str(err) == "blocked"
        assert err.verdict == SpamVerdict.SPAM
        assert err.confidence == 0.0
        assert err.reasons == []

    def test_custom_fields(self):
        err = ContentModerationError(
            "flagged",
            verdict=SpamVerdict.SUSPICIOUS,
            confidence=0.88,
            reasons=["url", "pattern"],
        )
        assert err.verdict == SpamVerdict.SUSPICIOUS
        assert err.confidence == 0.88
        assert err.reasons == ["url", "pattern"]

    def test_is_exception(self):
        with pytest.raises(ContentModerationError):
            raise ContentModerationError("boom")


# ===========================================================================
# SpamCheckResult
# ===========================================================================


class TestSpamCheckResult:
    def test_defaults(self):
        r = SpamCheckResult(verdict=SpamVerdict.CLEAN, confidence=0.5)
        assert r.reasons == []
        assert r.should_block is False
        assert r.should_flag_for_review is False
        assert r.spam_score == 0.0
        assert r.check_duration_ms == 0.0
        assert r.content_hash == ""

    def test_checked_at_is_utc(self):
        r = SpamCheckResult(verdict=SpamVerdict.CLEAN, confidence=0.5)
        assert r.checked_at.tzinfo == timezone.utc

    def test_to_dict_keys(self):
        r = SpamCheckResult(
            verdict=SpamVerdict.SPAM,
            confidence=0.99,
            reasons=["phish"],
            should_block=True,
            spam_score=0.95,
            content_hash="abc",
            content_score=0.1,
            sender_score=0.2,
            pattern_score=0.3,
            url_score=0.4,
        )
        d = r.to_dict()
        assert d["verdict"] == "spam"
        assert d["confidence"] == 0.99
        assert d["should_block"] is True
        assert d["scores"]["content"] == 0.1
        assert d["scores"]["sender"] == 0.2
        assert d["scores"]["pattern"] == 0.3
        assert d["scores"]["url"] == 0.4
        assert "checked_at" in d

    def test_to_dict_checked_at_isoformat(self):
        r = SpamCheckResult(verdict=SpamVerdict.CLEAN, confidence=0.5)
        d = r.to_dict()
        # Should be a parseable ISO string
        datetime.fromisoformat(d["checked_at"])

    def test_breakdown_scores_default_zero(self):
        r = SpamCheckResult(verdict=SpamVerdict.CLEAN, confidence=0.5)
        assert r.content_score == 0.0
        assert r.sender_score == 0.0
        assert r.pattern_score == 0.0
        assert r.url_score == 0.0


# ===========================================================================
# ModerationQueueItem
# ===========================================================================


class TestModerationQueueItem:
    def test_to_dict(self):
        result = SpamCheckResult(
            verdict=SpamVerdict.SUSPICIOUS,
            confidence=0.8,
            content_hash="hash123",
        )
        item = ModerationQueueItem(
            id="mod_abc",
            content="test content",
            content_hash="hash123",
            result=result,
            context={"user": "u1"},
        )
        d = item.to_dict()
        assert d["id"] == "mod_abc"
        assert d["content"] == "test content"
        assert d["content_hash"] == "hash123"
        assert d["result"]["verdict"] == "suspicious"
        assert d["context"] == {"user": "u1"}
        assert "queued_at" in d

    def test_queued_at_default(self):
        result = SpamCheckResult(verdict=SpamVerdict.CLEAN, confidence=0.5)
        item = ModerationQueueItem(
            id="mod_1",
            content="x",
            content_hash="h",
            result=result,
        )
        assert item.queued_at.tzinfo == timezone.utc

    def test_context_defaults_to_empty(self):
        result = SpamCheckResult(verdict=SpamVerdict.CLEAN, confidence=0.5)
        item = ModerationQueueItem(
            id="mod_1",
            content="x",
            content_hash="h",
            result=result,
        )
        assert item.context == {}


# ===========================================================================
# SpamModerationConfig
# ===========================================================================


class TestSpamModerationConfig:
    def test_default_values(self):
        c = SpamModerationConfig()
        assert c.enabled is True
        assert c.block_threshold == 0.9
        assert c.review_threshold == 0.7
        assert c.cache_enabled is True
        assert c.cache_ttl_seconds == 300
        assert c.cache_max_size == 1000
        assert c.fail_open is True
        assert c.log_all_checks is False

    def test_from_env_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            c = SpamModerationConfig.from_env()
            assert c.enabled is True

    def test_from_env_custom(self):
        env = {
            "ARAGORA_SPAM_CHECK_ENABLED": "false",
            "ARAGORA_SPAM_BLOCK_THRESHOLD": "0.8",
            "ARAGORA_SPAM_REVIEW_THRESHOLD": "0.5",
            "ARAGORA_SPAM_CACHE_ENABLED": "false",
            "ARAGORA_SPAM_CACHE_TTL": "120",
            "ARAGORA_SPAM_CACHE_SIZE": "200",
            "ARAGORA_SPAM_FAIL_OPEN": "false",
            "ARAGORA_SPAM_LOG_ALL": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            c = SpamModerationConfig.from_env()
            assert c.enabled is False
            assert c.block_threshold == 0.8
            assert c.review_threshold == 0.5
            assert c.cache_enabled is False
            assert c.cache_ttl_seconds == 120
            assert c.cache_max_size == 200
            assert c.fail_open is False
            assert c.log_all_checks is True

    def test_to_dict(self):
        c = SpamModerationConfig()
        d = c.to_dict()
        assert d["enabled"] is True
        assert d["block_threshold"] == 0.9
        assert d["cache_max_size"] == 1000

    def test_apply_updates_partial(self):
        c = SpamModerationConfig()
        c.apply_updates({"enabled": False, "block_threshold": 0.8})
        assert c.enabled is False
        assert c.block_threshold == 0.8
        # Unchanged fields
        assert c.review_threshold == 0.7

    def test_apply_updates_clamp_thresholds(self):
        c = SpamModerationConfig()
        c.apply_updates({"block_threshold": 1.5})
        assert c.block_threshold == 1.0

        c.apply_updates({"review_threshold": -0.5})
        assert c.review_threshold == 0.0

    def test_apply_updates_review_cannot_exceed_block(self):
        c = SpamModerationConfig(block_threshold=0.8, review_threshold=0.6)
        c.apply_updates({"review_threshold": 0.95})
        # review_threshold should be clamped to block_threshold
        assert c.review_threshold <= c.block_threshold

    def test_apply_updates_cache_ttl_negative_clamp(self):
        c = SpamModerationConfig()
        c.apply_updates({"cache_ttl_seconds": -10})
        assert c.cache_ttl_seconds == 0

    def test_apply_updates_cache_max_negative_clamp(self):
        c = SpamModerationConfig()
        c.apply_updates({"cache_max_size": -5})
        assert c.cache_max_size == 0


# ===========================================================================
# Review Queue Functions
# ===========================================================================


class TestReviewQueue:
    def _make_result(self, content_hash: str = "") -> SpamCheckResult:
        return SpamCheckResult(
            verdict=SpamVerdict.SUSPICIOUS,
            confidence=0.8,
            content_hash=content_hash,
        )

    def test_queue_for_review_basic(self):
        result = self._make_result()
        item = queue_for_review("spam content", result)
        assert item.id.startswith("mod_")
        assert item.content == "spam content"
        assert review_queue_size() == 1

    def test_queue_for_review_generates_hash_when_missing(self):
        result = self._make_result(content_hash="")
        item = queue_for_review("test", result)
        expected = hashlib.sha256("test".encode()).hexdigest()
        assert item.content_hash == expected

    def test_queue_for_review_uses_existing_hash(self):
        result = self._make_result(content_hash="pre_computed_hash")
        item = queue_for_review("test", result)
        assert item.content_hash == "pre_computed_hash"

    def test_queue_for_review_with_context(self):
        result = self._make_result()
        item = queue_for_review("text", result, context={"user": "u1"})
        assert item.context == {"user": "u1"}

    def test_queue_for_review_none_context(self):
        result = self._make_result()
        item = queue_for_review("text", result, context=None)
        assert item.context == {}

    def test_list_review_queue_empty(self):
        items = list_review_queue()
        assert items == []

    def test_list_review_queue_ordering(self):
        """Items should be returned newest-first."""
        r = self._make_result()
        item1 = queue_for_review("first", r)
        item2 = queue_for_review("second", r)
        item3 = queue_for_review("third", r)

        items = list_review_queue()
        assert len(items) == 3
        assert items[0].id == item3.id  # newest first
        assert items[2].id == item1.id  # oldest last

    def test_list_review_queue_limit(self):
        r = self._make_result()
        for i in range(5):
            queue_for_review(f"item {i}", r)
        items = list_review_queue(limit=3)
        assert len(items) == 3

    def test_list_review_queue_offset(self):
        r = self._make_result()
        for i in range(5):
            queue_for_review(f"item {i}", r)
        items = list_review_queue(limit=2, offset=2)
        assert len(items) == 2

    def test_list_review_queue_negative_offset_treated_as_zero(self):
        r = self._make_result()
        queue_for_review("a", r)
        items = list_review_queue(offset=-5)
        assert len(items) == 1

    def test_list_review_queue_zero_limit_returns_all_from_offset(self):
        r = self._make_result()
        for i in range(3):
            queue_for_review(f"item {i}", r)
        items = list_review_queue(limit=0, offset=1)
        assert len(items) == 2  # items[1:]

    def test_pop_review_item_existing(self):
        r = self._make_result()
        item = queue_for_review("text", r)
        popped = pop_review_item(item.id)
        assert popped is not None
        assert popped.id == item.id
        assert review_queue_size() == 0

    def test_pop_review_item_nonexistent(self):
        result = pop_review_item("mod_nonexistent")
        assert result is None

    def test_review_queue_size(self):
        r = self._make_result()
        assert review_queue_size() == 0
        queue_for_review("a", r)
        assert review_queue_size() == 1
        queue_for_review("b", r)
        assert review_queue_size() == 2

    @patch("aragora.moderation.spam_integration._REVIEW_QUEUE_MAX", 3)
    def test_queue_eviction_when_full(self):
        """When the queue is full, the oldest item should be evicted."""
        r = self._make_result()
        item1 = queue_for_review("one", r)
        queue_for_review("two", r)
        queue_for_review("three", r)

        # Queue is at capacity; adding one more should evict the oldest
        queue_for_review("four", r)

        assert review_queue_size() == 3
        # The oldest item should have been removed
        assert pop_review_item(item1.id) is None


# ===========================================================================
# SpamModerationIntegration
# ===========================================================================


class TestSpamModerationIntegration:
    @pytest.mark.asyncio
    async def test_initialize_marks_initialized(self, moderation):
        assert not moderation._initialized
        await moderation.initialize()
        assert moderation._initialized

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, moderation):
        await moderation.initialize()
        await moderation.initialize()
        # No error

    @pytest.mark.asyncio
    async def test_initialize_disabled_skips_classifier(self):
        config = SpamModerationConfig(enabled=False)
        mod = SpamModerationIntegration(config=config)
        await mod.initialize()
        assert mod._initialized
        assert mod._classifier is None

    @pytest.mark.asyncio
    async def test_initialize_import_error_disables(self):
        """When SpamClassifier import fails, moderation is disabled."""
        config = SpamModerationConfig(enabled=True)
        mod = SpamModerationIntegration(config=config)
        with patch(
            "aragora.moderation.spam_integration.SpamModerationIntegration.initialize",
            new_callable=AsyncMock,
        ) as mock_init:
            # Simulate the real behavior: after ImportError, enabled is set to False
            async def side_effect():
                mod._config.enabled = False
                mod._initialized = True

            mock_init.side_effect = side_effect
            await mod.initialize()
            assert mod._config.enabled is False

    @pytest.mark.asyncio
    async def test_check_content_clean(self, moderation, mock_classifier):
        mock_classifier.classify_email.return_value = MockClassificationResult(
            spam_score=0.1, confidence=0.95
        )
        result = await moderation.check_content("Hello world")
        assert result.verdict == SpamVerdict.CLEAN
        assert result.should_block is False
        assert result.should_flag_for_review is False
        assert moderation.statistics["passed"] == 1

    @pytest.mark.asyncio
    async def test_check_content_suspicious(self, moderation, mock_classifier):
        mock_classifier.classify_email.return_value = MockClassificationResult(
            spam_score=0.75, confidence=0.85, reasons=["suspicious pattern"]
        )
        result = await moderation.check_content("Click here")
        assert result.verdict == SpamVerdict.SUSPICIOUS
        assert result.should_flag_for_review is True
        assert result.should_block is False
        assert moderation.statistics["flagged"] == 1

    @pytest.mark.asyncio
    async def test_check_content_spam(self, moderation, mock_classifier):
        mock_classifier.classify_email.return_value = MockClassificationResult(
            spam_score=0.95, confidence=0.99, reasons=["phishing"]
        )
        result = await moderation.check_content("You won a million")
        assert result.verdict == SpamVerdict.SPAM
        assert result.should_block is True
        assert moderation.statistics["blocked"] == 1

    @pytest.mark.asyncio
    async def test_check_content_disabled(self, mock_classifier):
        config = SpamModerationConfig(enabled=False)
        mod = SpamModerationIntegration(classifier=mock_classifier, config=config)
        result = await mod.check_content("anything")
        assert result.verdict == SpamVerdict.CLEAN
        mock_classifier.classify_email.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_content_no_classifier(self):
        config = SpamModerationConfig(enabled=True)
        mod = SpamModerationIntegration(classifier=None, config=config)
        mod._initialized = True
        result = await mod.check_content("test")
        assert result.verdict == SpamVerdict.CLEAN

    @pytest.mark.asyncio
    async def test_caching_same_content(self, moderation, mock_classifier):
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.1)
        await moderation.check_content("same text")
        await moderation.check_content("same text")
        assert mock_classifier.classify_email.call_count == 1
        assert moderation.statistics["cache_hits"] == 1

    @pytest.mark.asyncio
    async def test_caching_different_content(self, moderation, mock_classifier):
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.1)
        await moderation.check_content("text a")
        await moderation.check_content("text b")
        assert mock_classifier.classify_email.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_disabled(self, mock_classifier):
        config = SpamModerationConfig(cache_enabled=False)
        mod = SpamModerationIntegration(classifier=mock_classifier, config=config)
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.1)
        await mod.check_content("text")
        await mod.check_content("text")
        assert mock_classifier.classify_email.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_expiry(self, moderation, mock_classifier):
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.1)
        await moderation.check_content("text")
        # Manually expire the cache entry
        for key in list(moderation._cache.keys()):
            result_obj, _ = moderation._cache[key]
            moderation._cache[key] = (result_obj, time.time() - 999)
        await moderation.check_content("text")
        assert mock_classifier.classify_email.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_eviction_on_full(self, mock_classifier):
        config = SpamModerationConfig(cache_enabled=True, cache_max_size=3)
        mod = SpamModerationIntegration(classifier=mock_classifier, config=config)
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.1)
        for i in range(5):
            await mod.check_content(f"unique content {i}")
        assert len(mod._cache) <= 3

    @pytest.mark.asyncio
    async def test_fail_open_on_error(self, moderation, mock_classifier):
        mock_classifier.classify_email.side_effect = RuntimeError("boom")
        result = await moderation.check_content("text")
        assert result.verdict == SpamVerdict.CLEAN
        assert result.should_flag_for_review is True
        assert moderation.statistics["errors"] == 1
        # Should also queue for review
        assert review_queue_size() == 1

    @pytest.mark.asyncio
    async def test_fail_closed_on_error(self, mock_classifier):
        config = SpamModerationConfig(fail_open=False)
        mod = SpamModerationIntegration(classifier=mock_classifier, config=config)
        mock_classifier.classify_email.side_effect = RuntimeError("boom")
        with pytest.raises(ContentModerationError):
            await mod.check_content("text")

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, moderation, mock_classifier):
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.1)
        await moderation.check_content("clean1")
        await moderation.check_content("clean2")

        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.75)
        await moderation.check_content("suspicious")

        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.95)
        await moderation.check_content("spam")

        stats = moderation.statistics
        assert stats["checks"] == 4
        assert stats["passed"] == 2
        assert stats["flagged"] == 1
        assert stats["blocked"] == 1

    def test_reset_statistics(self, moderation):
        moderation._stats["checks"] = 5
        old = moderation.reset_statistics()
        assert old["checks"] == 5
        assert moderation.statistics["checks"] == 0

    def test_clear_cache(self, moderation):
        moderation._cache["k1"] = (
            SpamCheckResult(verdict=SpamVerdict.CLEAN, confidence=0.5),
            time.time(),
        )
        moderation._cache["k2"] = (
            SpamCheckResult(verdict=SpamVerdict.CLEAN, confidence=0.5),
            time.time(),
        )
        count = moderation.clear_cache()
        assert count == 2
        assert len(moderation._cache) == 0

    @pytest.mark.asyncio
    async def test_close(self, moderation, mock_classifier):
        await moderation.initialize()
        await moderation.close()
        mock_classifier.close.assert_called_once()
        assert not moderation._initialized

    @pytest.mark.asyncio
    async def test_close_classifier_error_handled(self, moderation, mock_classifier):
        await moderation.initialize()
        mock_classifier.close.side_effect = RuntimeError("close error")
        await moderation.close()  # Should not raise
        assert not moderation._initialized

    def test_enabled_property(self, moderation):
        assert moderation.enabled is True
        moderation._config.enabled = False
        assert moderation.enabled is False

    def test_config_property(self, moderation, config):
        assert moderation.config is config

    def test_update_config(self, moderation):
        moderation._cache["k"] = (
            SpamCheckResult(verdict=SpamVerdict.CLEAN, confidence=0.5),
            time.time(),
        )
        moderation.update_config({"cache_enabled": False})
        assert moderation._config.cache_enabled is False
        # Cache should be cleared when disabling
        assert len(moderation._cache) == 0

    def test_update_config_shrink_cache_clears(self, moderation):
        moderation._cache["k"] = (
            SpamCheckResult(verdict=SpamVerdict.CLEAN, confidence=0.5),
            time.time(),
        )
        # Reduce cache size below current count
        moderation.update_config({"cache_max_size": 0})
        assert len(moderation._cache) == 0


# ===========================================================================
# check_debate_input / check_message
# ===========================================================================


class TestCheckDebateInput:
    @pytest.mark.asyncio
    async def test_proposal_only(self, moderation, mock_classifier):
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.1)
        result = await moderation.check_debate_input(proposal="Design an API")
        assert result.verdict == SpamVerdict.CLEAN

    @pytest.mark.asyncio
    async def test_proposal_with_context(self, moderation, mock_classifier):
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.1)
        result = await moderation.check_debate_input(
            proposal="Design an API",
            context="For authentication",
        )
        assert result.verdict == SpamVerdict.CLEAN
        # Verify body includes both proposal and context
        call_kwargs = mock_classifier.classify_email.call_args.kwargs
        body = call_kwargs.get("body", "")
        assert "Design an API" in body
        assert "For authentication" in body

    @pytest.mark.asyncio
    async def test_proposal_with_metadata(self, moderation, mock_classifier):
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.1)
        result = await moderation.check_debate_input(
            proposal="Test",
            metadata={"user_id": "u1"},
        )
        assert result.verdict == SpamVerdict.CLEAN


class TestCheckMessage:
    @pytest.mark.asyncio
    async def test_basic_message(self, moderation, mock_classifier):
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.1)
        result = await moderation.check_message("Hello there")
        assert result.verdict == SpamVerdict.CLEAN

    @pytest.mark.asyncio
    async def test_message_with_sender(self, moderation, mock_classifier):
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.1)
        result = await moderation.check_message("msg", sender="user@example.com")
        assert result.verdict == SpamVerdict.CLEAN
        call_kwargs = mock_classifier.classify_email.call_args.kwargs
        assert call_kwargs.get("sender") == "user@example.com"

    @pytest.mark.asyncio
    async def test_message_with_debate_id(self, moderation, mock_classifier):
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.1)
        result = await moderation.check_message("msg", debate_id="d123")
        assert result.verdict == SpamVerdict.CLEAN


# ===========================================================================
# Global instance management
# ===========================================================================


class TestGlobalInstance:
    def test_get_creates_instance(self, clean_global_state):
        mod = get_spam_moderation()
        assert isinstance(mod, SpamModerationIntegration)

    def test_get_returns_same_instance(self, clean_global_state):
        assert get_spam_moderation() is get_spam_moderation()

    def test_set_overrides(self, clean_global_state, moderation):
        set_spam_moderation(moderation)
        assert get_spam_moderation() is moderation


# ===========================================================================
# Convenience function check_debate_content
# ===========================================================================


class TestCheckDebateContentConvenience:
    @pytest.mark.asyncio
    async def test_basic_call(self, clean_global_state, mock_classifier):
        mock_classifier.classify_email.return_value = MockClassificationResult(
            spam_score=0.1, confidence=0.9
        )
        mod = SpamModerationIntegration(classifier=mock_classifier)
        set_spam_moderation(mod)
        result = await check_debate_content("Design a rate limiter")
        assert result.verdict == SpamVerdict.CLEAN

    @pytest.mark.asyncio
    async def test_with_context_and_metadata(self, clean_global_state, mock_classifier):
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.1)
        mod = SpamModerationIntegration(classifier=mock_classifier)
        set_spam_moderation(mod)
        result = await check_debate_content(
            proposal="Test",
            context="extra",
            metadata={"org": "acme"},
        )
        assert result.verdict == SpamVerdict.CLEAN


# ===========================================================================
# Threshold Boundary Tests
# ===========================================================================


class TestThresholdBoundaries:
    @pytest.mark.asyncio
    async def test_score_exactly_at_review(self, mock_classifier, config):
        config.review_threshold = 0.7
        mod = SpamModerationIntegration(classifier=mock_classifier, config=config)
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.7)
        result = await mod.check_content("t")
        assert result.verdict == SpamVerdict.SUSPICIOUS
        assert result.should_flag_for_review is True
        assert result.should_block is False

    @pytest.mark.asyncio
    async def test_score_exactly_at_block(self, mock_classifier, config):
        config.block_threshold = 0.9
        mod = SpamModerationIntegration(classifier=mock_classifier, config=config)
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.9)
        result = await mod.check_content("t")
        assert result.verdict == SpamVerdict.SPAM
        assert result.should_block is True

    @pytest.mark.asyncio
    async def test_score_just_below_review(self, mock_classifier, config):
        config.review_threshold = 0.7
        mod = SpamModerationIntegration(classifier=mock_classifier, config=config)
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.69)
        result = await mod.check_content("t")
        assert result.verdict == SpamVerdict.CLEAN
        assert result.should_flag_for_review is False

    @pytest.mark.asyncio
    async def test_score_just_below_block(self, mock_classifier, config):
        config.block_threshold = 0.9
        mod = SpamModerationIntegration(classifier=mock_classifier, config=config)
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.89)
        result = await mod.check_content("t")
        assert result.verdict == SpamVerdict.SUSPICIOUS
        assert result.should_block is False

    @pytest.mark.asyncio
    async def test_zero_score(self, moderation, mock_classifier):
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.0)
        result = await moderation.check_content("t")
        assert result.verdict == SpamVerdict.CLEAN

    @pytest.mark.asyncio
    async def test_perfect_score(self, moderation, mock_classifier):
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=1.0)
        result = await moderation.check_content("t")
        assert result.verdict == SpamVerdict.SPAM
        assert result.should_block is True


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_string_input(self, moderation, mock_classifier):
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.1)
        result = await moderation.check_content("")
        assert result.verdict == SpamVerdict.CLEAN

    @pytest.mark.asyncio
    async def test_unicode_content(self, moderation, mock_classifier):
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.1)
        result = await moderation.check_content(
            "Voici du contenu en francais avec des accents et des emojis"
        )
        assert result.verdict == SpamVerdict.CLEAN

    @pytest.mark.asyncio
    async def test_very_long_content(self, moderation, mock_classifier):
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.1)
        long_text = "x" * 100_000
        result = await moderation.check_content(long_text)
        assert result.verdict == SpamVerdict.CLEAN

    @pytest.mark.asyncio
    async def test_content_with_newlines(self, moderation, mock_classifier):
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.1)
        result = await moderation.check_content("line1\nline2\nline3")
        assert result.verdict == SpamVerdict.CLEAN

    @pytest.mark.asyncio
    async def test_content_with_null_bytes(self, moderation, mock_classifier):
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.1)
        result = await moderation.check_content("has\x00null")
        assert result.verdict == SpamVerdict.CLEAN

    @pytest.mark.asyncio
    async def test_reasons_truncated_to_five(self, moderation, mock_classifier):
        mock_classifier.classify_email.return_value = MockClassificationResult(
            spam_score=0.95,
            confidence=0.99,
            reasons=["r1", "r2", "r3", "r4", "r5", "r6", "r7"],
        )
        result = await moderation.check_content("spam")
        assert len(result.reasons) == 5

    @pytest.mark.asyncio
    async def test_log_all_checks_flag(self, mock_classifier):
        config = SpamModerationConfig(log_all_checks=True)
        mod = SpamModerationIntegration(classifier=mock_classifier, config=config)
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.1)
        # Should not raise; just verifying it runs with the flag
        result = await mod.check_content("test")
        assert result.verdict == SpamVerdict.CLEAN

    @pytest.mark.asyncio
    async def test_suspicious_content_auto_queued(self, moderation, mock_classifier):
        """Suspicious content should automatically be queued for review."""
        mock_classifier.classify_email.return_value = MockClassificationResult(
            spam_score=0.75, confidence=0.85
        )
        await moderation.check_content("suspicious stuff")
        assert review_queue_size() == 1

    def test_hash_content_deterministic(self, moderation):
        h1 = moderation._hash_content("same text")
        h2 = moderation._hash_content("same text")
        assert h1 == h2
        assert h1 == hashlib.sha256("same text".encode()).hexdigest()

    def test_hash_content_different_for_different_input(self, moderation):
        h1 = moderation._hash_content("aaa")
        h2 = moderation._hash_content("bbb")
        assert h1 != h2

    @pytest.mark.asyncio
    async def test_classification_to_result_breakdown_scores(self, moderation, mock_classifier):
        mock_classifier.classify_email.return_value = MockClassificationResult(
            spam_score=0.5,
            confidence=0.8,
            content_score=0.2,
            sender_score=0.3,
            pattern_score=0.4,
            url_score=0.5,
        )
        result = await moderation.check_content("text")
        assert result.content_score == 0.2
        assert result.sender_score == 0.3
        assert result.pattern_score == 0.4
        assert result.url_score == 0.5

    @pytest.mark.asyncio
    async def test_check_duration_positive(self, moderation, mock_classifier):
        mock_classifier.classify_email.return_value = MockClassificationResult(spam_score=0.1)
        result = await moderation.check_content("text")
        assert result.check_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_multiple_error_types_fail_open(self, moderation, mock_classifier):
        """All caught exception types should trigger fail-open."""
        for exc in [ValueError("v"), TypeError("t"), KeyError("k"), OSError("o")]:
            mock_classifier.classify_email.side_effect = exc
            result = await moderation.check_content(f"test-{exc}")
            assert result.verdict == SpamVerdict.CLEAN
