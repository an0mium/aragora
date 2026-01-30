"""
Tests for Email Cost Optimizer Service.

Tests for the cost optimization layer including:
- Result caching with TTL
- Budget enforcement (daily/monthly)
- Rate limiting
- Tier selection logic
- Batch processing
- Usage statistics
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


class TestCostConfig:
    """Tests for CostConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from aragora.services.email_cost_optimizer import CostConfig

        config = CostConfig()

        assert config.daily_budget_usd == 10.0
        assert config.monthly_budget_usd == 200.0
        assert config.per_request_limit_usd == 0.50
        assert config.cache_ttl_seconds == 3600
        assert config.cache_max_size == 10000
        assert config.cache_score_threshold == 0.9
        assert config.batch_size == 10
        assert config.tier_1_cost == 0.0
        assert config.tier_2_cost == 0.001
        assert config.tier_3_cost == 0.01
        assert config.max_requests_per_minute == 60
        assert config.max_tier_3_per_hour == 100
        assert config.skip_tier_2_confidence == 0.85
        assert config.skip_tier_3_confidence == 0.75

    def test_config_custom_values(self):
        """Test custom configuration values."""
        from aragora.services.email_cost_optimizer import CostConfig

        config = CostConfig(
            daily_budget_usd=5.0,
            monthly_budget_usd=100.0,
            cache_ttl_seconds=1800,
            max_requests_per_minute=30,
        )

        assert config.daily_budget_usd == 5.0
        assert config.monthly_budget_usd == 100.0
        assert config.cache_ttl_seconds == 1800
        assert config.max_requests_per_minute == 30


class TestUsageStats:
    """Tests for UsageStats dataclass."""

    def test_stats_defaults(self):
        """Test default usage stats."""
        from aragora.services.email_cost_optimizer import UsageStats

        stats = UsageStats()

        assert stats.total_requests == 0
        assert stats.tier_1_requests == 0
        assert stats.tier_2_requests == 0
        assert stats.tier_3_requests == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.estimated_cost_usd == 0.0
        assert stats.daily_cost_usd == 0.0
        assert stats.monthly_cost_usd == 0.0

    def test_stats_to_dict(self):
        """Test UsageStats.to_dict serialization."""
        from aragora.services.email_cost_optimizer import UsageStats

        stats = UsageStats(
            total_requests=100,
            tier_1_requests=70,
            tier_2_requests=25,
            tier_3_requests=5,
            cache_hits=40,
            cache_misses=60,
            estimated_cost_usd=0.075,
            daily_cost_usd=0.075,
            monthly_cost_usd=1.5,
            avg_processing_time_ms=150.5,
        )

        data = stats.to_dict()

        assert data["total_requests"] == 100
        assert data["tier_1_requests"] == 70
        assert data["tier_2_requests"] == 25
        assert data["tier_3_requests"] == 5
        assert data["cache_hits"] == 40
        assert data["cache_misses"] == 60
        assert data["cache_hit_rate"] == 0.4  # 40/(40+60)
        assert data["estimated_cost_usd"] == 0.075
        assert data["avg_processing_time_ms"] == 150.5

    def test_stats_cache_hit_rate_zero_total(self):
        """Test cache hit rate with zero requests."""
        from aragora.services.email_cost_optimizer import UsageStats

        stats = UsageStats()  # No requests yet

        data = stats.to_dict()

        assert data["cache_hit_rate"] == 0.0


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        from aragora.services.email_cost_optimizer import CacheEntry

        mock_result = MagicMock()
        mock_result.confidence = 0.95

        entry = CacheEntry(
            result=mock_result,
            timestamp=datetime.now(),
            email_hash="abc123",
            confidence=0.95,
        )

        assert entry.result is mock_result
        assert entry.email_hash == "abc123"
        assert entry.confidence == 0.95


class TestCostOptimizedPrioritizer:
    """Tests for CostOptimizedPrioritizer."""

    @pytest.fixture
    def mock_prioritizer(self):
        """Create a mock email prioritizer."""
        mock = AsyncMock()

        # Default high-confidence result
        mock_result = MagicMock()
        mock_result.confidence = 0.9
        mock_result.priority = MagicMock()
        mock_result.rationale = "Test rationale"

        mock.score_email = AsyncMock(return_value=mock_result)
        return mock

    @pytest.fixture
    def mock_email(self):
        """Create a mock email message."""
        email = MagicMock()
        email.from_address = "sender@example.com"
        email.subject = "Test Subject"
        email.snippet = "Test snippet"
        email.labels = ["INBOX"]
        return email

    def test_optimizer_initialization(self, mock_prioritizer):
        """Test optimizer initialization."""
        from aragora.services.email_cost_optimizer import (
            CostOptimizedPrioritizer,
            CostConfig,
        )

        config = CostConfig(daily_budget_usd=5.0)
        optimizer = CostOptimizedPrioritizer(
            prioritizer=mock_prioritizer,
            config=config,
        )

        assert optimizer.prioritizer is mock_prioritizer
        assert optimizer.config.daily_budget_usd == 5.0
        assert optimizer._cache == {}
        assert optimizer._stats.total_requests == 0

    def test_optimizer_default_config(self, mock_prioritizer):
        """Test optimizer with default config."""
        from aragora.services.email_cost_optimizer import CostOptimizedPrioritizer

        optimizer = CostOptimizedPrioritizer(prioritizer=mock_prioritizer)

        assert optimizer.config.daily_budget_usd == 10.0

    def test_compute_email_hash(self, mock_prioritizer, mock_email):
        """Test email hash computation."""
        from aragora.services.email_cost_optimizer import CostOptimizedPrioritizer

        optimizer = CostOptimizedPrioritizer(prioritizer=mock_prioritizer)

        hash1 = optimizer._compute_email_hash(mock_email)
        hash2 = optimizer._compute_email_hash(mock_email)

        # Same email should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 16  # SHA256 truncated to 16 chars

    def test_compute_email_hash_different_emails(self, mock_prioritizer):
        """Test different emails produce different hashes."""
        from aragora.services.email_cost_optimizer import CostOptimizedPrioritizer

        optimizer = CostOptimizedPrioritizer(prioritizer=mock_prioritizer)

        email1 = MagicMock()
        email1.from_address = "user1@example.com"
        email1.subject = "Subject 1"
        email1.snippet = ""
        email1.labels = []

        email2 = MagicMock()
        email2.from_address = "user2@example.com"
        email2.subject = "Subject 2"
        email2.snippet = ""
        email2.labels = []

        hash1 = optimizer._compute_email_hash(email1)
        hash2 = optimizer._compute_email_hash(email2)

        assert hash1 != hash2

    def test_get_usage_stats(self, mock_prioritizer):
        """Test getting usage statistics."""
        from aragora.services.email_cost_optimizer import CostOptimizedPrioritizer

        optimizer = CostOptimizedPrioritizer(prioritizer=mock_prioritizer)

        stats = optimizer.get_usage_stats()

        assert stats.total_requests == 0
        assert stats.cache_hits == 0

    def test_get_cache_stats(self, mock_prioritizer):
        """Test getting cache statistics."""
        from aragora.services.email_cost_optimizer import CostOptimizedPrioritizer

        optimizer = CostOptimizedPrioritizer(prioritizer=mock_prioritizer)

        cache_stats = optimizer.get_cache_stats()

        assert cache_stats["size"] == 0
        assert cache_stats["hit_rate"] == 0.0

    def test_reset_stats(self, mock_prioritizer):
        """Test resetting usage statistics."""
        from aragora.services.email_cost_optimizer import CostOptimizedPrioritizer

        optimizer = CostOptimizedPrioritizer(prioritizer=mock_prioritizer)

        # Manually modify stats
        optimizer._stats.total_requests = 100
        optimizer._stats.cache_hits = 50

        optimizer.reset_stats()

        assert optimizer._stats.total_requests == 0
        assert optimizer._stats.cache_hits == 0

    @pytest.mark.asyncio
    async def test_clear_cache(self, mock_prioritizer):
        """Test clearing cache."""
        from aragora.services.email_cost_optimizer import (
            CostOptimizedPrioritizer,
            CacheEntry,
        )

        optimizer = CostOptimizedPrioritizer(prioritizer=mock_prioritizer)

        # Add some cache entries
        for i in range(5):
            optimizer._cache[f"hash_{i}"] = CacheEntry(
                result=MagicMock(),
                timestamp=datetime.now(),
                email_hash=f"hash_{i}",
                confidence=0.95,
            )

        assert len(optimizer._cache) == 5

        count = await optimizer.clear_cache()

        assert count == 5
        assert len(optimizer._cache) == 0

    @pytest.mark.asyncio
    async def test_get_cached_result_hit(self, mock_prioritizer, mock_email):
        """Test cache hit."""
        from aragora.services.email_cost_optimizer import (
            CostOptimizedPrioritizer,
            CacheEntry,
        )

        optimizer = CostOptimizedPrioritizer(prioritizer=mock_prioritizer)

        # Pre-populate cache
        mock_result = MagicMock()
        mock_result.confidence = 0.95

        email_hash = optimizer._compute_email_hash(mock_email)
        optimizer._cache[email_hash] = CacheEntry(
            result=mock_result,
            timestamp=datetime.now(),
            email_hash=email_hash,
            confidence=0.95,
        )

        cached = await optimizer._get_cached_result(mock_email)

        assert cached is mock_result
        assert optimizer._stats.cache_hits == 1

    @pytest.mark.asyncio
    async def test_get_cached_result_miss(self, mock_prioritizer, mock_email):
        """Test cache miss."""
        from aragora.services.email_cost_optimizer import CostOptimizedPrioritizer

        optimizer = CostOptimizedPrioritizer(prioritizer=mock_prioritizer)

        cached = await optimizer._get_cached_result(mock_email)

        assert cached is None
        assert optimizer._stats.cache_misses == 1

    @pytest.mark.asyncio
    async def test_get_cached_result_expired(self, mock_prioritizer, mock_email):
        """Test cache entry expired."""
        from aragora.services.email_cost_optimizer import (
            CostOptimizedPrioritizer,
            CacheEntry,
            CostConfig,
        )

        config = CostConfig(cache_ttl_seconds=60)  # 1 minute TTL
        optimizer = CostOptimizedPrioritizer(
            prioritizer=mock_prioritizer,
            config=config,
        )

        # Pre-populate cache with old entry
        mock_result = MagicMock()
        email_hash = optimizer._compute_email_hash(mock_email)
        optimizer._cache[email_hash] = CacheEntry(
            result=mock_result,
            timestamp=datetime.now() - timedelta(seconds=120),  # 2 minutes old
            email_hash=email_hash,
            confidence=0.95,
        )

        cached = await optimizer._get_cached_result(mock_email)

        assert cached is None  # Expired
        assert email_hash not in optimizer._cache  # Removed
        assert optimizer._stats.cache_misses == 1

    @pytest.mark.asyncio
    async def test_cache_result_high_confidence(self, mock_prioritizer, mock_email):
        """Test caching high-confidence result."""
        from aragora.services.email_cost_optimizer import CostOptimizedPrioritizer

        optimizer = CostOptimizedPrioritizer(prioritizer=mock_prioritizer)

        mock_result = MagicMock()
        mock_result.confidence = 0.95  # Above threshold (0.9)

        await optimizer._cache_result(mock_email, mock_result)

        email_hash = optimizer._compute_email_hash(mock_email)
        assert email_hash in optimizer._cache

    @pytest.mark.asyncio
    async def test_cache_result_low_confidence_not_cached(self, mock_prioritizer, mock_email):
        """Test low-confidence result not cached."""
        from aragora.services.email_cost_optimizer import CostOptimizedPrioritizer

        optimizer = CostOptimizedPrioritizer(prioritizer=mock_prioritizer)

        mock_result = MagicMock()
        mock_result.confidence = 0.7  # Below threshold (0.9)

        await optimizer._cache_result(mock_email, mock_result)

        email_hash = optimizer._compute_email_hash(mock_email)
        assert email_hash not in optimizer._cache

    @pytest.mark.asyncio
    async def test_cache_size_limit_eviction(self, mock_prioritizer):
        """Test cache size limit enforces eviction."""
        from aragora.services.email_cost_optimizer import (
            CostOptimizedPrioritizer,
            CostConfig,
            CacheEntry,
        )

        config = CostConfig(cache_max_size=10)
        optimizer = CostOptimizedPrioritizer(
            prioritizer=mock_prioritizer,
            config=config,
        )

        # Fill cache to max
        for i in range(10):
            optimizer._cache[f"hash_{i}"] = CacheEntry(
                result=MagicMock(confidence=0.95),
                timestamp=datetime.now() - timedelta(seconds=i),
                email_hash=f"hash_{i}",
                confidence=0.95,
            )

        assert len(optimizer._cache) == 10

        # Add one more (should trigger eviction)
        new_email = MagicMock()
        new_email.from_address = "new@example.com"
        new_email.subject = "New email"
        new_email.snippet = ""
        new_email.labels = []

        mock_result = MagicMock()
        mock_result.confidence = 0.95

        await optimizer._cache_result(new_email, mock_result)

        # Should have evicted ~25% of entries
        assert len(optimizer._cache) < 10

    @pytest.mark.asyncio
    async def test_check_rate_limits_under_limit(self, mock_prioritizer):
        """Test rate limit check when under limit."""
        from aragora.services.email_cost_optimizer import CostOptimizedPrioritizer

        optimizer = CostOptimizedPrioritizer(prioritizer=mock_prioritizer)

        result = await optimizer._check_rate_limits()

        assert result is True

    @pytest.mark.asyncio
    async def test_check_rate_limits_over_limit(self, mock_prioritizer):
        """Test rate limit check when over limit."""
        from aragora.services.email_cost_optimizer import (
            CostOptimizedPrioritizer,
            CostConfig,
        )

        config = CostConfig(max_requests_per_minute=5)
        optimizer = CostOptimizedPrioritizer(
            prioritizer=mock_prioritizer,
            config=config,
        )

        # Add timestamps exceeding limit
        now = datetime.now()
        optimizer._request_timestamps = [now - timedelta(seconds=i) for i in range(6)]

        result = await optimizer._check_rate_limits()

        assert result is False

    @pytest.mark.asyncio
    async def test_record_request_tier_1(self, mock_prioritizer):
        """Test recording tier 1 request."""
        from aragora.services.email_cost_optimizer import CostOptimizedPrioritizer

        optimizer = CostOptimizedPrioritizer(prioritizer=mock_prioritizer)

        await optimizer._record_request(tier=1, processing_time_ms=50.0)

        assert optimizer._stats.total_requests == 1
        assert optimizer._stats.tier_1_requests == 1
        assert optimizer._stats.tier_2_requests == 0
        assert optimizer._stats.estimated_cost_usd == 0.0  # Tier 1 is free
        assert optimizer._stats.total_processing_time_ms == 50.0
        assert optimizer._stats.avg_processing_time_ms == 50.0

    @pytest.mark.asyncio
    async def test_record_request_tier_2(self, mock_prioritizer):
        """Test recording tier 2 request."""
        from aragora.services.email_cost_optimizer import CostOptimizedPrioritizer

        optimizer = CostOptimizedPrioritizer(prioritizer=mock_prioritizer)

        await optimizer._record_request(tier=2, processing_time_ms=100.0)

        assert optimizer._stats.tier_2_requests == 1
        assert optimizer._stats.estimated_cost_usd == 0.001

    @pytest.mark.asyncio
    async def test_record_request_tier_3(self, mock_prioritizer):
        """Test recording tier 3 request."""
        from aragora.services.email_cost_optimizer import CostOptimizedPrioritizer

        optimizer = CostOptimizedPrioritizer(prioritizer=mock_prioritizer)

        await optimizer._record_request(tier=3, processing_time_ms=500.0)

        assert optimizer._stats.tier_3_requests == 1
        assert optimizer._stats.estimated_cost_usd == 0.01
        assert len(optimizer._tier_3_timestamps) == 1

    @pytest.mark.asyncio
    async def test_check_budget_under_limit(self, mock_prioritizer):
        """Test budget check when under limit."""
        from aragora.services.email_cost_optimizer import CostOptimizedPrioritizer

        optimizer = CostOptimizedPrioritizer(prioritizer=mock_prioritizer)

        result = await optimizer._check_budget()

        assert result is True

    @pytest.mark.asyncio
    async def test_check_budget_daily_exceeded(self, mock_prioritizer):
        """Test budget check when daily limit exceeded."""
        from aragora.services.email_cost_optimizer import (
            CostOptimizedPrioritizer,
            CostConfig,
        )

        config = CostConfig(daily_budget_usd=1.0)
        optimizer = CostOptimizedPrioritizer(
            prioritizer=mock_prioritizer,
            config=config,
        )

        # Set daily cost over budget
        optimizer._stats.daily_cost_usd = 1.5

        result = await optimizer._check_budget()

        assert result is False

    @pytest.mark.asyncio
    async def test_check_budget_monthly_exceeded(self, mock_prioritizer):
        """Test budget check when monthly limit exceeded."""
        from aragora.services.email_cost_optimizer import (
            CostOptimizedPrioritizer,
            CostConfig,
        )

        config = CostConfig(monthly_budget_usd=50.0)
        optimizer = CostOptimizedPrioritizer(
            prioritizer=mock_prioritizer,
            config=config,
        )

        # Set monthly cost over budget
        optimizer._stats.monthly_cost_usd = 55.0

        result = await optimizer._check_budget()

        assert result is False

    @pytest.mark.asyncio
    async def test_score_email_uses_cache(self, mock_prioritizer, mock_email):
        """Test scoring email uses cache when available."""
        from aragora.services.email_cost_optimizer import (
            CostOptimizedPrioritizer,
            CacheEntry,
        )

        optimizer = CostOptimizedPrioritizer(prioritizer=mock_prioritizer)

        # Pre-populate cache
        cached_result = MagicMock()
        cached_result.confidence = 0.95

        email_hash = optimizer._compute_email_hash(mock_email)
        optimizer._cache[email_hash] = CacheEntry(
            result=cached_result,
            timestamp=datetime.now(),
            email_hash=email_hash,
            confidence=0.95,
        )

        result = await optimizer.score_email(mock_email)

        assert result is cached_result
        mock_prioritizer.score_email.assert_not_called()

    @pytest.mark.asyncio
    async def test_score_email_skip_cache(self, mock_prioritizer, mock_email):
        """Test scoring email with skip_cache."""
        from aragora.services.email_cost_optimizer import (
            CostOptimizedPrioritizer,
            CacheEntry,
        )

        optimizer = CostOptimizedPrioritizer(prioritizer=mock_prioritizer)

        # Pre-populate cache
        cached_result = MagicMock()
        cached_result.confidence = 0.95

        email_hash = optimizer._compute_email_hash(mock_email)
        optimizer._cache[email_hash] = CacheEntry(
            result=cached_result,
            timestamp=datetime.now(),
            email_hash=email_hash,
            confidence=0.95,
        )

        result = await optimizer.score_email(mock_email, skip_cache=True)

        # Should use prioritizer, not cache
        mock_prioritizer.score_email.assert_called()

    @pytest.mark.asyncio
    async def test_score_email_force_tier(self, mock_prioritizer, mock_email):
        """Test scoring email with forced tier."""
        from aragora.services.email_cost_optimizer import CostOptimizedPrioritizer

        optimizer = CostOptimizedPrioritizer(prioritizer=mock_prioritizer)

        result = await optimizer.score_email(mock_email, force_tier=2)

        # Should call with specified tier
        assert mock_prioritizer.score_email.called
        call_kwargs = mock_prioritizer.score_email.call_args[1]
        assert "force_tier" in call_kwargs

    @pytest.mark.asyncio
    async def test_score_email_rate_limited(self, mock_prioritizer, mock_email):
        """Test scoring email when rate limited."""
        from aragora.services.email_cost_optimizer import (
            CostOptimizedPrioritizer,
            CostConfig,
        )

        config = CostConfig(max_requests_per_minute=0)  # Immediate rate limit
        optimizer = CostOptimizedPrioritizer(
            prioritizer=mock_prioritizer,
            config=config,
        )

        # Add a timestamp to trigger rate limit
        optimizer._request_timestamps = [datetime.now()]

        result = await optimizer.score_email(mock_email)

        # Should get tier 1 result with rate limit note
        assert "rate limited" in result.rationale

    @pytest.mark.asyncio
    async def test_batch_score_emails_all_cached(self, mock_prioritizer):
        """Test batch scoring when all emails are cached."""
        from aragora.services.email_cost_optimizer import (
            CostOptimizedPrioritizer,
            CacheEntry,
        )

        optimizer = CostOptimizedPrioritizer(prioritizer=mock_prioritizer)

        # Create emails and cache them
        emails = []
        for i in range(3):
            email = MagicMock()
            email.from_address = f"user{i}@example.com"
            email.subject = f"Subject {i}"
            email.snippet = ""
            email.labels = []
            emails.append(email)

            # Cache result
            cached_result = MagicMock()
            cached_result.confidence = 0.95

            email_hash = optimizer._compute_email_hash(email)
            optimizer._cache[email_hash] = CacheEntry(
                result=cached_result,
                timestamp=datetime.now(),
                email_hash=email_hash,
                confidence=0.95,
            )

        results = await optimizer.batch_score_emails(emails)

        assert len(results) == 3
        assert optimizer._stats.cache_hits == 3
        mock_prioritizer.score_email.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_score_emails_mixed_cache(self, mock_prioritizer):
        """Test batch scoring with mixed cache hits."""
        from aragora.services.email_cost_optimizer import (
            CostOptimizedPrioritizer,
            CacheEntry,
        )

        optimizer = CostOptimizedPrioritizer(prioritizer=mock_prioritizer)

        # Create emails - cache only first one
        emails = []
        for i in range(3):
            email = MagicMock()
            email.from_address = f"user{i}@example.com"
            email.subject = f"Subject {i}"
            email.snippet = ""
            email.labels = []
            emails.append(email)

        # Cache only first email
        cached_result = MagicMock()
        cached_result.confidence = 0.95

        email_hash = optimizer._compute_email_hash(emails[0])
        optimizer._cache[email_hash] = CacheEntry(
            result=cached_result,
            timestamp=datetime.now(),
            email_hash=email_hash,
            confidence=0.95,
        )

        results = await optimizer.batch_score_emails(emails)

        assert len(results) == 3
        assert optimizer._stats.cache_hits == 1
        assert mock_prioritizer.score_email.call_count >= 2

    @pytest.mark.asyncio
    async def test_batch_score_emails_concurrency_limit(self, mock_prioritizer):
        """Test batch scoring respects concurrency limit."""
        from aragora.services.email_cost_optimizer import CostOptimizedPrioritizer

        # Track concurrent calls
        concurrent_count = 0
        max_concurrent = 0

        original_score = mock_prioritizer.score_email

        async def tracked_score(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.01)  # Small delay to allow overlap
            concurrent_count -= 1
            return await original_score(*args, **kwargs)

        mock_prioritizer.score_email = tracked_score

        optimizer = CostOptimizedPrioritizer(prioritizer=mock_prioritizer)

        # Create many emails
        emails = []
        for i in range(10):
            email = MagicMock()
            email.from_address = f"user{i}@example.com"
            email.subject = f"Subject {i}"
            email.snippet = ""
            email.labels = []
            emails.append(email)

        await optimizer.batch_score_emails(emails, max_concurrent=3)

        # Max concurrent should be limited
        assert max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_get_cost_projection_default_distribution(self, mock_prioritizer):
        """Test cost projection with default tier distribution."""
        from aragora.services.email_cost_optimizer import CostOptimizedPrioritizer

        optimizer = CostOptimizedPrioritizer(prioritizer=mock_prioritizer)

        projection = await optimizer.get_cost_projection(emails_per_day=1000)

        assert projection["emails_per_day"] == 1000
        assert "projected_daily_cost_usd" in projection
        assert "projected_monthly_cost_usd" in projection
        assert projection["tier_distribution"] == {1: 0.70, 2: 0.25, 3: 0.05}
        assert projection["assumed_cache_hit_rate"] == 0.4

    @pytest.mark.asyncio
    async def test_get_cost_projection_custom_distribution(self, mock_prioritizer):
        """Test cost projection with custom tier distribution."""
        from aragora.services.email_cost_optimizer import CostOptimizedPrioritizer

        optimizer = CostOptimizedPrioritizer(prioritizer=mock_prioritizer)

        projection = await optimizer.get_cost_projection(
            emails_per_day=500,
            tier_distribution={1: 0.9, 2: 0.08, 3: 0.02},
        )

        assert projection["emails_per_day"] == 500
        assert projection["tier_distribution"] == {1: 0.9, 2: 0.08, 3: 0.02}

    @pytest.mark.asyncio
    async def test_get_cost_projection_budget_check(self, mock_prioritizer):
        """Test cost projection includes budget check."""
        from aragora.services.email_cost_optimizer import (
            CostOptimizedPrioritizer,
            CostConfig,
        )

        config = CostConfig(daily_budget_usd=1.0, monthly_budget_usd=20.0)
        optimizer = CostOptimizedPrioritizer(
            prioritizer=mock_prioritizer,
            config=config,
        )

        projection = await optimizer.get_cost_projection(emails_per_day=100)

        assert "within_daily_budget" in projection
        assert "within_monthly_budget" in projection


class TestCreateCostOptimizedPrioritizer:
    """Tests for create_cost_optimized_prioritizer factory."""

    @pytest.mark.asyncio
    async def test_create_optimizer_minimal(self):
        """Test creating optimizer with minimal config."""
        with patch(
            "aragora.services.email_prioritization.EmailPrioritizer"
        ) as mock_prioritizer_class:
            mock_prioritizer = MagicMock()
            mock_prioritizer_class.return_value = mock_prioritizer

            from aragora.services.email_cost_optimizer import (
                create_cost_optimized_prioritizer,
            )

            optimizer = await create_cost_optimized_prioritizer()

            assert optimizer.prioritizer is mock_prioritizer
            mock_prioritizer_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_optimizer_with_config(self):
        """Test creating optimizer with custom config."""
        with patch(
            "aragora.services.email_prioritization.EmailPrioritizer"
        ) as mock_prioritizer_class:
            mock_prioritizer = MagicMock()
            mock_prioritizer_class.return_value = mock_prioritizer

            from aragora.services.email_cost_optimizer import (
                create_cost_optimized_prioritizer,
                CostConfig,
            )

            config = CostConfig(daily_budget_usd=5.0)
            optimizer = await create_cost_optimized_prioritizer(config=config)

            assert optimizer.config.daily_budget_usd == 5.0

    @pytest.mark.asyncio
    async def test_create_optimizer_with_connectors(self):
        """Test creating optimizer with connectors."""
        with patch(
            "aragora.services.email_prioritization.EmailPrioritizer"
        ) as mock_prioritizer_class:
            mock_prioritizer = MagicMock()
            mock_prioritizer_class.return_value = mock_prioritizer

            from aragora.services.email_cost_optimizer import (
                create_cost_optimized_prioritizer,
            )

            mock_gmail = MagicMock()
            mock_mound = MagicMock()

            optimizer = await create_cost_optimized_prioritizer(
                gmail_connector=mock_gmail,
                knowledge_mound=mock_mound,
            )

            assert optimizer is not None
            mock_prioritizer_class.assert_called_once_with(
                gmail_connector=mock_gmail,
                knowledge_mound=mock_mound,
            )
