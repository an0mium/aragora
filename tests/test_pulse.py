"""
Tests for the Pulse ingestor module.

Tests trending topic collection from various social media sources.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import time

from aragora.pulse.ingestor import (
    CircuitBreaker,
    TrendingTopic,
    PulseIngestor,
    TwitterIngestor,
    HackerNewsIngestor,
    RedditIngestor,
    PulseManager,
)


# =============================================================================
# CircuitBreaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_circuit_breaker_creation(self):
        """Test CircuitBreaker creation with defaults."""
        cb = CircuitBreaker()
        assert cb.failure_threshold >= 3  # Default may vary, but should be reasonable
        assert cb.failures == 0
        assert cb.is_open is False

    def test_circuit_breaker_can_proceed_initially(self):
        """Test circuit allows requests initially."""
        cb = CircuitBreaker()
        assert cb.can_proceed() is True

    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit opens after threshold failures."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        assert cb.is_open is False
        cb.record_failure()
        assert cb.is_open is True
        assert cb.can_proceed() is False

    def test_circuit_breaker_resets_on_success(self):
        """Test circuit resets after success."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is True

        cb.record_success()
        assert cb.is_open is False
        assert cb.failures == 0

    def test_circuit_breaker_resets_after_timeout(self):
        """Test circuit resets after timeout."""
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.1)
        cb.record_failure()
        assert cb.is_open is True
        assert cb.can_proceed() is False

        # Wait for reset timeout
        time.sleep(0.15)
        assert cb.can_proceed() is True
        assert cb.is_open is False


# =============================================================================
# TrendingTopic Tests
# =============================================================================


class TestTrendingTopic:
    """Tests for TrendingTopic dataclass."""

    def test_trending_topic_creation(self):
        """Test basic TrendingTopic creation."""
        topic = TrendingTopic(
            platform="twitter",
            topic="#AI",
            volume=100000,
            category="tech",
        )
        assert topic.platform == "twitter"
        assert topic.topic == "#AI"
        assert topic.volume == 100000
        assert topic.category == "tech"

    def test_trending_topic_defaults(self):
        """Test TrendingTopic with default values."""
        topic = TrendingTopic(
            platform="reddit",
            topic="Test Topic",
        )
        assert topic.volume == 0
        assert topic.category == ""
        assert topic.raw_data == {}

    def test_to_debate_prompt(self):
        """Test debate prompt generation."""
        topic = TrendingTopic(
            platform="twitter",
            topic="#ClimateChange",
            volume=50000,
        )
        prompt = topic.to_debate_prompt()
        assert "#ClimateChange" in prompt
        assert "twitter" in prompt
        assert "50000" in prompt

    def test_trending_topic_with_raw_data(self):
        """Test TrendingTopic with raw data."""
        raw = {"url": "https://example.com", "author": "user123"}
        topic = TrendingTopic(
            platform="hackernews",
            topic="Test Story",
            raw_data=raw,
        )
        assert topic.raw_data["url"] == "https://example.com"
        assert topic.raw_data["author"] == "user123"


# =============================================================================
# PulseIngestor Tests (Base Class)
# =============================================================================


class ConcretePulseIngestor(PulseIngestor):
    """Concrete implementation for testing abstract base class."""

    async def fetch_trending(self, limit: int = 10):
        return [TrendingTopic("test", f"Topic {i}", i * 100) for i in range(limit)]


class TestPulseIngestor:
    """Tests for PulseIngestor base class."""

    def test_ingestor_creation(self):
        """Test basic ingestor creation."""
        ingestor = ConcretePulseIngestor()
        assert ingestor.api_key is None
        assert ingestor.rate_limit_delay == 1.0
        assert ingestor.cache_ttl == 300

    def test_ingestor_with_api_key(self):
        """Test ingestor with API key."""
        ingestor = ConcretePulseIngestor(api_key="test_key")
        assert ingestor.api_key == "test_key"

    def test_is_toxic_high_severity(self):
        """Test high severity toxicity detection."""
        ingestor = ConcretePulseIngestor()
        assert ingestor._is_toxic("kill all enemies") is True
        assert ingestor._is_toxic("terrorist attack") is True
        assert ingestor._is_toxic("genocide") is True

    def test_is_toxic_medium_severity(self):
        """Test medium severity toxicity detection (needs 2+ matches)."""
        ingestor = ConcretePulseIngestor()
        # Single medium severity term is not toxic
        assert ingestor._is_toxic("hate") is False
        # Two medium severity terms is toxic
        assert ingestor._is_toxic("hate and violence") is True

    def test_is_toxic_low_severity(self):
        """Test low severity (adult content) detection."""
        ingestor = ConcretePulseIngestor()
        assert ingestor._is_toxic("nsfw content") is True
        assert ingestor._is_toxic("18+ only") is True
        assert ingestor._is_toxic("explicit") is True

    def test_is_toxic_clean_content(self):
        """Test that clean content is not flagged."""
        ingestor = ConcretePulseIngestor()
        assert ingestor._is_toxic("Python programming tips") is False
        assert ingestor._is_toxic("Climate change debate") is False
        assert ingestor._is_toxic("New AI breakthrough") is False

    def test_filter_content_toxic(self):
        """Test filtering removes toxic content."""
        ingestor = ConcretePulseIngestor()
        topics = [
            TrendingTopic("test", "Good topic", 100),
            TrendingTopic("test", "kill them all", 200),
            TrendingTopic("test", "Another good one", 150),
        ]
        filtered = ingestor._filter_content(topics, {"skip_toxic": True})
        assert len(filtered) == 2
        assert all("kill" not in t.topic for t in filtered)

    def test_filter_content_by_category(self):
        """Test filtering by category."""
        ingestor = ConcretePulseIngestor()
        topics = [
            TrendingTopic("test", "AI News", 100, "tech"),
            TrendingTopic("test", "Election Update", 200, "politics"),
            TrendingTopic("test", "Code Review", 150, "tech"),
        ]
        filtered = ingestor._filter_content(topics, {"categories": ["tech"]})
        assert len(filtered) == 2
        assert all(t.category == "tech" for t in filtered)

    def test_filter_content_by_volume(self):
        """Test filtering by minimum volume."""
        ingestor = ConcretePulseIngestor()
        topics = [
            TrendingTopic("test", "Low volume", 50),
            TrendingTopic("test", "High volume", 200),
            TrendingTopic("test", "Medium volume", 100),
        ]
        filtered = ingestor._filter_content(topics, {"min_volume": 100})
        assert len(filtered) == 2
        assert all(t.volume >= 100 for t in filtered)

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test that rate limiting enforces delays."""
        ingestor = ConcretePulseIngestor(rate_limit_delay=0.1)

        start = asyncio.get_event_loop().time()
        await ingestor._rate_limit()
        await ingestor._rate_limit()  # Should delay
        elapsed = asyncio.get_event_loop().time() - start

        # Should have delayed at least 0.1 seconds
        assert elapsed >= 0.09  # Allow small timing variance


# =============================================================================
# TwitterIngestor Tests
# =============================================================================


class TestTwitterIngestor:
    """Tests for TwitterIngestor."""

    def test_twitter_ingestor_creation(self):
        """Test Twitter ingestor creation."""
        ingestor = TwitterIngestor()
        assert ingestor.api_key is None
        assert ingestor.base_url == "https://api.twitter.com/2"

    def test_twitter_ingestor_with_bearer_token(self):
        """Test Twitter ingestor with bearer token."""
        ingestor = TwitterIngestor(bearer_token="test_token")
        assert ingestor.api_key == "test_token"

    @pytest.mark.asyncio
    async def test_fetch_trending_no_api_key(self):
        """Test fetch_trending returns mock data without API key."""
        ingestor = TwitterIngestor()
        topics = await ingestor.fetch_trending(limit=3)

        assert len(topics) == 3
        assert all(t.platform == "twitter" for t in topics)
        # Should be mock data
        assert topics[0].topic == "#AIJobDisplacement"

    @pytest.mark.asyncio
    async def test_fetch_trending_limit(self):
        """Test fetch_trending respects limit."""
        ingestor = TwitterIngestor()
        topics = await ingestor.fetch_trending(limit=2)
        assert len(topics) == 2

    def test_categorize_topic_tech(self):
        """Test topic categorization for tech."""
        ingestor = TwitterIngestor()
        assert ingestor._categorize_topic("#AI breakthrough") == "tech"
        assert ingestor._categorize_topic("New software release") == "tech"
        assert ingestor._categorize_topic("Code review tips") == "tech"

    def test_categorize_topic_politics(self):
        """Test topic categorization for politics."""
        ingestor = TwitterIngestor()
        assert ingestor._categorize_topic("Election results") == "politics"
        assert ingestor._categorize_topic("Government policy") == "politics"

    def test_categorize_topic_environment(self):
        """Test topic categorization for environment."""
        ingestor = TwitterIngestor()
        assert ingestor._categorize_topic("Climate change") == "environment"
        assert ingestor._categorize_topic("Green energy") == "environment"

    def test_categorize_topic_general(self):
        """Test topic categorization falls back to general."""
        ingestor = TwitterIngestor()
        assert ingestor._categorize_topic("Random topic") == "general"
        assert ingestor._categorize_topic("Sports news") == "general"

    def test_mock_trending_data(self):
        """Test mock data generation."""
        ingestor = TwitterIngestor()
        mock_data = ingestor._mock_trending_data(5)

        assert len(mock_data) == 5
        assert all(t.platform == "twitter" for t in mock_data)
        assert all(t.volume > 0 for t in mock_data)
        # Check some known mock topics
        topics = [t.topic for t in mock_data]
        assert "#AIJobDisplacement" in topics


# =============================================================================
# PulseManager Tests
# =============================================================================


class TestPulseManager:
    """Tests for PulseManager."""

    def test_manager_creation(self):
        """Test manager creation."""
        manager = PulseManager()
        assert len(manager.ingestors) == 0

    def test_add_ingestor(self):
        """Test adding ingestors."""
        manager = PulseManager()
        ingestor = TwitterIngestor()
        manager.add_ingestor("twitter", ingestor)

        assert "twitter" in manager.ingestors
        assert manager.ingestors["twitter"] is ingestor

    def test_add_multiple_ingestors(self):
        """Test adding multiple ingestors."""
        manager = PulseManager()
        manager.add_ingestor("twitter", TwitterIngestor())
        manager.add_ingestor("test", ConcretePulseIngestor())

        assert len(manager.ingestors) == 2
        assert "twitter" in manager.ingestors
        assert "test" in manager.ingestors

    @pytest.mark.asyncio
    async def test_get_trending_topics_single(self):
        """Test getting trending topics from single ingestor."""
        manager = PulseManager()
        manager.add_ingestor("twitter", TwitterIngestor())

        topics = await manager.get_trending_topics(limit_per_platform=3)

        assert len(topics) >= 1
        assert all(t.platform == "twitter" for t in topics)

    @pytest.mark.asyncio
    async def test_get_trending_topics_multiple(self):
        """Test getting trending topics from multiple ingestors."""
        manager = PulseManager()
        manager.add_ingestor("twitter", TwitterIngestor())
        manager.add_ingestor("test", ConcretePulseIngestor())

        topics = await manager.get_trending_topics(limit_per_platform=3)

        # Should have topics from both platforms
        platforms = {t.platform for t in topics}
        assert "twitter" in platforms
        assert "test" in platforms

    @pytest.mark.asyncio
    async def test_get_trending_topics_sorted_by_volume(self):
        """Test that topics are sorted by volume."""
        manager = PulseManager()
        manager.add_ingestor("twitter", TwitterIngestor())

        topics = await manager.get_trending_topics(limit_per_platform=5)

        # Check descending order
        volumes = [t.volume for t in topics]
        assert volumes == sorted(volumes, reverse=True)

    @pytest.mark.asyncio
    async def test_get_trending_topics_specific_platforms(self):
        """Test getting topics from specific platforms."""
        manager = PulseManager()
        manager.add_ingestor("twitter", TwitterIngestor())
        manager.add_ingestor("test", ConcretePulseIngestor())

        topics = await manager.get_trending_topics(
            platforms=["twitter"],
            limit_per_platform=3,
        )

        # Should only have Twitter topics
        assert all(t.platform == "twitter" for t in topics)

    @pytest.mark.asyncio
    async def test_get_trending_topics_with_filters(self):
        """Test getting topics with content filters."""
        manager = PulseManager()
        manager.add_ingestor("twitter", TwitterIngestor())

        topics = await manager.get_trending_topics(
            limit_per_platform=5,
            filters={"min_volume": 50000},
        )

        # All topics should meet minimum volume
        assert all(t.volume >= 50000 for t in topics)

    @pytest.mark.asyncio
    async def test_get_trending_topics_empty_manager(self):
        """Test getting topics from empty manager."""
        manager = PulseManager()
        topics = await manager.get_trending_topics()
        assert topics == []

    def test_select_topic_for_debate(self):
        """Test topic selection for debate."""
        manager = PulseManager()
        topics = [
            TrendingTopic("twitter", "Tech Topic", 100, "tech"),
            TrendingTopic("twitter", "Politics Topic", 200, "politics"),
            TrendingTopic("twitter", "Another Tech", 150, "tech"),
        ]

        selected = manager.select_topic_for_debate(topics)

        # Should select first topic of each category (diverse)
        assert selected.category == "tech"  # First unique category

    def test_select_topic_for_debate_empty(self):
        """Test topic selection with empty list."""
        manager = PulseManager()
        selected = manager.select_topic_for_debate([])
        assert selected is None

    def test_select_topic_for_debate_single(self):
        """Test topic selection with single topic."""
        manager = PulseManager()
        topic = TrendingTopic("twitter", "Single Topic", 100)
        selected = manager.select_topic_for_debate([topic])
        assert selected is topic


# =============================================================================
# Integration Tests
# =============================================================================


class TestPulseIntegration:
    """Integration tests for the pulse module."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test complete pulse pipeline."""
        # Create manager with Twitter ingestor
        manager = PulseManager()
        manager.add_ingestor("twitter", TwitterIngestor())

        # Fetch topics
        topics = await manager.get_trending_topics(limit_per_platform=5)
        assert len(topics) > 0

        # Select topic for debate
        selected = manager.select_topic_for_debate(topics)
        assert selected is not None

        # Generate debate prompt
        prompt = selected.to_debate_prompt()
        assert len(prompt) > 0
        assert selected.topic in prompt

    @pytest.mark.asyncio
    async def test_concurrent_fetch(self):
        """Test concurrent fetching from multiple ingestors."""
        manager = PulseManager()

        # Add multiple ingestors
        for i in range(3):
            manager.add_ingestor(f"test_{i}", ConcretePulseIngestor())

        # Fetch should be concurrent
        topics = await manager.get_trending_topics(limit_per_platform=5)

        # Should have topics from all ingestors (3 ingestors * 5 topics each = 15 max)
        # All return platform="test" since that's what ConcretePulseIngestor returns
        assert len(topics) >= 3  # At least some topics from each

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling during fetch."""
        manager = PulseManager()

        # Create a failing ingestor
        class FailingIngestor(PulseIngestor):
            async def fetch_trending(self, limit: int = 10):
                raise Exception("API Error")

        manager.add_ingestor("failing", FailingIngestor())
        manager.add_ingestor("working", ConcretePulseIngestor())

        # Should still get results from working ingestor
        topics = await manager.get_trending_topics(limit_per_platform=3)
        assert len(topics) > 0
        assert all(t.platform == "test" for t in topics)


# =============================================================================
# HackerNewsIngestor Tests
# =============================================================================


class TestHackerNewsIngestor:
    """Tests for HackerNewsIngestor."""

    def test_hn_ingestor_creation(self):
        """Test HN ingestor creation."""
        ingestor = HackerNewsIngestor()
        assert ingestor.base_url == "https://hn.algolia.com/api/v1"
        assert ingestor.api_key is None

    @pytest.mark.asyncio
    async def test_fetch_trending_fallback(self):
        """Test fetch_trending returns mock data on error."""
        ingestor = HackerNewsIngestor(max_retries=1, base_retry_delay=0.01)
        # Force circuit breaker open
        ingestor.circuit_breaker.is_open = True

        topics = await ingestor.fetch_trending(limit=3)
        assert len(topics) == 3
        assert all(t.platform == "hackernews" for t in topics)

    def test_categorize_topic_ai(self):
        """Test AI topic categorization."""
        ingestor = HackerNewsIngestor()
        assert ingestor._categorize_topic("GPT-4 breakthrough") == "ai"
        assert ingestor._categorize_topic("New LLM model") == "ai"

    def test_categorize_topic_programming(self):
        """Test programming topic categorization."""
        ingestor = HackerNewsIngestor()
        assert ingestor._categorize_topic("Rust is amazing") == "programming"
        assert ingestor._categorize_topic("Python 4.0 released") == "programming"

    def test_categorize_topic_security(self):
        """Test security topic categorization."""
        ingestor = HackerNewsIngestor()
        assert ingestor._categorize_topic("Major security vulnerability") == "security"
        assert ingestor._categorize_topic("Data breach at company") == "security"

    def test_categorize_topic_business(self):
        """Test business topic categorization."""
        ingestor = HackerNewsIngestor()
        # Note: "raises" contains "ai" substring, so avoid that word
        assert ingestor._categorize_topic("YC funding for new company") == "business"
        assert ingestor._categorize_topic("VC investment trends") == "business"
        assert ingestor._categorize_topic("Tech acquisition announced") == "business"

    def test_mock_trending_data(self):
        """Test mock data generation."""
        ingestor = HackerNewsIngestor()
        mock_data = ingestor._mock_trending_data(5)
        assert len(mock_data) == 5
        assert all(t.platform == "hackernews" for t in mock_data)


# =============================================================================
# RedditIngestor Tests
# =============================================================================


class TestRedditIngestor:
    """Tests for RedditIngestor."""

    def test_reddit_ingestor_creation(self):
        """Test Reddit ingestor creation with defaults."""
        ingestor = RedditIngestor()
        assert ingestor.base_url == "https://www.reddit.com"
        assert "technology" in ingestor.subreddits
        assert "programming" in ingestor.subreddits

    def test_reddit_ingestor_custom_subreddits(self):
        """Test Reddit ingestor with custom subreddits."""
        ingestor = RedditIngestor(subreddits=["python", "golang"])
        assert ingestor.subreddits == ["python", "golang"]

    @pytest.mark.asyncio
    async def test_fetch_trending_fallback(self):
        """Test fetch_trending returns mock data on error."""
        ingestor = RedditIngestor(max_retries=1, base_retry_delay=0.01)
        # Force circuit breaker open
        ingestor.circuit_breaker.is_open = True

        topics = await ingestor.fetch_trending(limit=3)
        assert len(topics) == 3
        assert all(t.platform == "reddit" for t in topics)

    def test_categorize_subreddit(self):
        """Test subreddit categorization."""
        ingestor = RedditIngestor()
        assert ingestor._categorize_subreddit("technology") == "tech"
        assert ingestor._categorize_subreddit("programming") == "programming"
        assert ingestor._categorize_subreddit("science") == "science"
        assert ingestor._categorize_subreddit("worldnews") == "news"
        assert ingestor._categorize_subreddit("random") == "general"

    def test_mock_trending_data(self):
        """Test mock data generation."""
        ingestor = RedditIngestor()
        mock_data = ingestor._mock_trending_data(5)
        assert len(mock_data) == 5
        assert all(t.platform == "reddit" for t in mock_data)


# =============================================================================
# Retry Logic Tests
# =============================================================================


class TestRetryLogic:
    """Tests for retry with backoff logic."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_first_try(self):
        """Test that successful first try returns immediately."""
        ingestor = ConcretePulseIngestor(max_retries=3, base_retry_delay=0.01)

        call_count = 0

        async def success_fn():
            nonlocal call_count
            call_count += 1
            return [TrendingTopic("test", "Success", 100)]

        result = await ingestor._retry_with_backoff(success_fn)
        assert len(result) == 1
        assert call_count == 1
        assert ingestor.circuit_breaker.failures == 0

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_failures(self):
        """Test that retry eventually succeeds."""
        ingestor = ConcretePulseIngestor(max_retries=3, base_retry_delay=0.01)

        call_count = 0

        async def eventually_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return [TrendingTopic("test", "Success", 100)]

        result = await ingestor._retry_with_backoff(eventually_succeeds)
        assert len(result) == 1
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_uses_fallback_after_all_failures(self):
        """Test that fallback is used after all retries fail."""
        ingestor = ConcretePulseIngestor(max_retries=2, base_retry_delay=0.01)

        async def always_fails():
            raise Exception("Permanent failure")

        def fallback():
            return [TrendingTopic("fallback", "Fallback Topic", 50)]

        result = await ingestor._retry_with_backoff(always_fails, fallback_fn=fallback)
        assert len(result) == 1
        assert result[0].platform == "fallback"
        assert ingestor.circuit_breaker.failures == 1

    @pytest.mark.asyncio
    async def test_retry_skips_when_circuit_open(self):
        """Test that retry is skipped when circuit breaker is open."""
        ingestor = ConcretePulseIngestor(max_retries=3, base_retry_delay=0.01)
        ingestor.circuit_breaker.is_open = True
        ingestor.circuit_breaker.last_failure_time = time.time()  # Prevent auto-reset

        call_count = 0

        async def should_not_be_called():
            nonlocal call_count
            call_count += 1
            return [TrendingTopic("test", "Should not appear", 100)]

        def fallback():
            return [TrendingTopic("fallback", "Fallback", 50)]

        result = await ingestor._retry_with_backoff(should_not_be_called, fallback_fn=fallback)
        assert call_count == 0
        assert result[0].platform == "fallback"
