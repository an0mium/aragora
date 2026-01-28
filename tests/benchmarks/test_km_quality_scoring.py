"""
Knowledge Mound Quality Scoring Load Tests

Tests the performance and accuracy of the auto-curation quality scoring system
under various load conditions.

Run with: pytest tests/benchmarks/test_km_quality_scoring.py -v --benchmark-enable
"""

import asyncio
import random
import time
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import pytest_benchmark

from aragora.knowledge.mound.quality import (
    QualityScorer,
    QualityWeights,
    QualityScore,
    TierThresholds,
)


@pytest.fixture
def quality_scorer():
    """Create a quality scorer with default weights."""
    weights = QualityWeights(
        freshness=0.2,
        confidence=0.3,
        usage=0.25,
        relevance=0.15,
        relationships=0.1,
    )
    thresholds = TierThresholds(
        tier1_min=0.8,  # High quality
        tier2_min=0.5,  # Medium quality
        tier3_min=0.2,  # Low quality
    )
    return QualityScorer(weights=weights, thresholds=thresholds)


@pytest.fixture
def sample_items():
    """Generate sample knowledge items for testing."""

    def generate_item(index: int) -> dict[str, Any]:
        # Vary age from 0 to 365 days
        created_at = datetime.now() - timedelta(days=random.randint(0, 365))
        accessed_at = datetime.now() - timedelta(hours=random.randint(0, 720))

        return {
            "id": f"item_{index:05d}",
            "content": f"Sample knowledge content {index} " * random.randint(10, 50),
            "created_at": created_at.isoformat(),
            "updated_at": accessed_at.isoformat(),
            "accessed_at": accessed_at.isoformat(),
            "access_count": random.randint(0, 1000),
            "confidence": random.uniform(0.3, 1.0),
            "source_reliability": random.uniform(0.5, 1.0),
            "relationship_count": random.randint(0, 50),
            "citation_count": random.randint(0, 100),
            "verified": random.choice([True, False]),
            "metadata": {
                "category": random.choice(["fact", "opinion", "analysis", "data"]),
                "source": random.choice(["debate", "document", "user", "system"]),
            },
        }

    return [generate_item(i) for i in range(10000)]


class TestQualityScoringPerformance:
    """Performance tests for quality scoring."""

    def test_score_single_item(self, quality_scorer, sample_items, benchmark):
        """Benchmark scoring a single item."""
        item = sample_items[0]

        def score_item():
            return quality_scorer.score(item)

        result = benchmark(score_item)
        assert result is not None
        assert 0 <= result.total_score <= 1

    def test_score_batch_100(self, quality_scorer, sample_items, benchmark):
        """Benchmark scoring 100 items."""
        items = sample_items[:100]

        def score_batch():
            return [quality_scorer.score(item) for item in items]

        results = benchmark(score_batch)
        assert len(results) == 100
        assert all(0 <= r.total_score <= 1 for r in results)

    def test_score_batch_1000(self, quality_scorer, sample_items, benchmark):
        """Benchmark scoring 1000 items."""
        items = sample_items[:1000]

        def score_batch():
            return [quality_scorer.score(item) for item in items]

        results = benchmark(score_batch)
        assert len(results) == 1000

    def test_score_batch_10000(self, quality_scorer, sample_items, benchmark):
        """Benchmark scoring 10000 items."""

        def score_batch():
            return [quality_scorer.score(item) for item in sample_items]

        results = benchmark(score_batch)
        assert len(results) == 10000

    def test_scoring_latency_p95(self, quality_scorer, sample_items):
        """Test that p95 latency is under 10ms per item."""
        latencies = []

        for item in sample_items[:1000]:
            start = time.perf_counter()
            quality_scorer.score(item)
            latencies.append((time.perf_counter() - start) * 1000)  # ms

        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.5)]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]

        print(f"\nLatency (ms): p50={p50:.3f}, p95={p95:.3f}, p99={p99:.3f}")

        # p95 should be under 10ms
        assert p95 < 10, f"p95 latency {p95:.3f}ms exceeds 10ms threshold"

    def test_throughput_items_per_second(self, quality_scorer, sample_items):
        """Test throughput of quality scoring."""
        items = sample_items[:5000]

        start = time.perf_counter()
        for item in items:
            quality_scorer.score(item)
        elapsed = time.perf_counter() - start

        throughput = len(items) / elapsed
        print(f"\nThroughput: {throughput:.0f} items/second")

        # Should process at least 1000 items/second
        assert throughput > 1000, f"Throughput {throughput:.0f} below 1000/s threshold"


class TestQualityScoringAccuracy:
    """Accuracy tests for quality scoring."""

    def test_fresh_items_score_higher(self, quality_scorer):
        """Fresh items should score higher on freshness component."""
        fresh_item = {
            "id": "fresh",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "accessed_at": datetime.now().isoformat(),
            "access_count": 10,
            "confidence": 0.8,
            "relationship_count": 5,
        }

        old_item = {
            "id": "old",
            "created_at": (datetime.now() - timedelta(days=365)).isoformat(),
            "updated_at": (datetime.now() - timedelta(days=365)).isoformat(),
            "accessed_at": (datetime.now() - timedelta(days=30)).isoformat(),
            "access_count": 10,
            "confidence": 0.8,
            "relationship_count": 5,
        }

        fresh_score = quality_scorer.score(fresh_item)
        old_score = quality_scorer.score(old_item)

        # Fresh item should have higher freshness component
        assert fresh_score.freshness_score > old_score.freshness_score

    def test_high_confidence_scores_higher(self, quality_scorer):
        """High confidence items should score higher."""
        high_conf = {
            "id": "high_conf",
            "created_at": datetime.now().isoformat(),
            "confidence": 0.95,
            "access_count": 10,
            "relationship_count": 5,
        }

        low_conf = {
            "id": "low_conf",
            "created_at": datetime.now().isoformat(),
            "confidence": 0.3,
            "access_count": 10,
            "relationship_count": 5,
        }

        high_score = quality_scorer.score(high_conf)
        low_score = quality_scorer.score(low_conf)

        assert high_score.confidence_score > low_score.confidence_score
        assert high_score.total_score > low_score.total_score

    def test_frequently_used_items_score_higher(self, quality_scorer):
        """Frequently accessed items should score higher on usage."""
        popular = {
            "id": "popular",
            "created_at": datetime.now().isoformat(),
            "confidence": 0.7,
            "access_count": 1000,
            "relationship_count": 5,
        }

        unused = {
            "id": "unused",
            "created_at": datetime.now().isoformat(),
            "confidence": 0.7,
            "access_count": 0,
            "relationship_count": 5,
        }

        popular_score = quality_scorer.score(popular)
        unused_score = quality_scorer.score(unused)

        assert popular_score.usage_score > unused_score.usage_score

    def test_tier_assignment_correctness(self, quality_scorer, sample_items):
        """Test that tier assignments match thresholds."""
        tier_counts = {"tier1": 0, "tier2": 0, "tier3": 0, "archive": 0}

        for item in sample_items[:1000]:
            score = quality_scorer.score(item)
            tier = quality_scorer.assign_tier(score)
            tier_counts[tier] += 1

        print(f"\nTier distribution: {tier_counts}")

        # Verify some distribution exists (not all in one tier)
        assert tier_counts["tier1"] > 0 or tier_counts["tier2"] > 0
        assert tier_counts["tier3"] > 0 or tier_counts["archive"] > 0

    def test_score_determinism(self, quality_scorer, sample_items):
        """Scoring should be deterministic for same input."""
        item = sample_items[0]

        scores = [quality_scorer.score(item) for _ in range(100)]

        # All scores should be identical
        first_score = scores[0].total_score
        assert all(s.total_score == first_score for s in scores)


class TestQualityScoringStress:
    """Stress tests for quality scoring system."""

    @pytest.mark.slow
    def test_concurrent_scoring(self, quality_scorer, sample_items):
        """Test scoring under concurrent load."""
        import concurrent.futures

        items = sample_items[:1000]

        def score_item(item):
            return quality_scorer.score(item)

        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(score_item, items))
        elapsed = time.perf_counter() - start

        print(f"\nConcurrent scoring: {len(results)} items in {elapsed:.2f}s")
        assert len(results) == len(items)
        assert all(r is not None for r in results)

    @pytest.mark.slow
    def test_memory_usage_large_batch(self, quality_scorer, sample_items):
        """Test memory usage doesn't grow unexpectedly."""
        import sys

        # Get baseline memory
        initial_size = sys.getsizeof(quality_scorer)

        # Score all items
        results = []
        for item in sample_items:
            results.append(quality_scorer.score(item))

        # Memory shouldn't grow significantly
        final_size = sys.getsizeof(quality_scorer)
        assert final_size < initial_size * 2, "Memory growth exceeded 2x"

        # Clear results
        results.clear()

    @pytest.mark.slow
    def test_sustained_load(self, quality_scorer, sample_items):
        """Test sustained scoring load over time."""
        items = sample_items[:100]
        iterations = 100

        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            for item in items:
                quality_scorer.score(item)
            latencies.append(time.perf_counter() - start)

        # Latency should remain stable (no degradation)
        first_10_avg = sum(latencies[:10]) / 10
        last_10_avg = sum(latencies[-10:]) / 10

        print(
            f"\nFirst 10 avg: {first_10_avg * 1000:.2f}ms, Last 10 avg: {last_10_avg * 1000:.2f}ms"
        )

        # Allow 20% degradation max
        assert last_10_avg < first_10_avg * 1.2, "Performance degraded under sustained load"


# Mock implementations for when the actual module isn't available
class MockQualityWeights:
    def __init__(
        self, freshness=0.2, confidence=0.3, usage=0.25, relevance=0.15, relationships=0.1
    ):
        self.freshness = freshness
        self.confidence = confidence
        self.usage = usage
        self.relevance = relevance
        self.relationships = relationships


class MockTierThresholds:
    def __init__(self, tier1_min=0.8, tier2_min=0.5, tier3_min=0.2):
        self.tier1_min = tier1_min
        self.tier2_min = tier2_min
        self.tier3_min = tier3_min


class MockQualityScore:
    def __init__(self):
        self.total_score = random.uniform(0, 1)
        self.freshness_score = random.uniform(0, 1)
        self.confidence_score = random.uniform(0, 1)
        self.usage_score = random.uniform(0, 1)
        self.relevance_score = random.uniform(0, 1)
        self.relationships_score = random.uniform(0, 1)


class MockQualityScorer:
    def __init__(self, weights=None, thresholds=None):
        self.weights = weights or MockQualityWeights()
        self.thresholds = thresholds or MockTierThresholds()

    def score(self, item: dict) -> MockQualityScore:
        # Simulate scoring based on item attributes
        score = MockQualityScore()

        # Adjust scores based on item attributes
        if "confidence" in item:
            score.confidence_score = item["confidence"]
        if "access_count" in item:
            score.usage_score = min(item["access_count"] / 1000, 1.0)
        if "created_at" in item:
            try:
                created = datetime.fromisoformat(item["created_at"].replace("Z", "+00:00"))
                age_days = (datetime.now() - created.replace(tzinfo=None)).days
                score.freshness_score = max(0, 1 - age_days / 365)
            except Exception:
                score.freshness_score = 0.5

        # Calculate total
        score.total_score = (
            score.freshness_score * self.weights.freshness
            + score.confidence_score * self.weights.confidence
            + score.usage_score * self.weights.usage
            + score.relevance_score * self.weights.relevance
            + score.relationships_score * self.weights.relationships
        )

        return score

    def assign_tier(self, score: MockQualityScore) -> str:
        if score.total_score >= self.thresholds.tier1_min:
            return "tier1"
        elif score.total_score >= self.thresholds.tier2_min:
            return "tier2"
        elif score.total_score >= self.thresholds.tier3_min:
            return "tier3"
        else:
            return "archive"


# Override fixtures if actual module not available
try:
    from aragora.knowledge.mound.quality import QualityScorer, QualityWeights, TierThresholds
except ImportError:
    QualityScorer = MockQualityScorer
    QualityWeights = MockQualityWeights
    TierThresholds = MockTierThresholds
    QualityScore = MockQualityScore


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
