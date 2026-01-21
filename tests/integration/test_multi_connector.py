"""
Multi-Connector Integration Tests.

Tests scenarios where multiple connectors work together:
- Cross-connector data aggregation
- Connector failover and resilience
- Concurrent connector operations
- Connector registry and discovery
- Credential management across connectors
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Mock Evidence for Testing (simplified)
# =============================================================================


@dataclass
class MockEvidence:
    """Simplified evidence for testing."""

    source: str
    content: str
    url: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Mock Connectors for Testing
# =============================================================================


class MockConnector:
    """Mock connector for testing multi-connector scenarios."""

    def __init__(
        self,
        name: str = "mock",
        delay: float = 0.0,
        fail_on_call: int = -1,
        evidence_count: int = 3,
    ):
        self.name = name
        self._delay = delay
        self._fail_on_call = fail_on_call
        self._evidence_count = evidence_count
        self._call_count = 0
        self._collected: List[MockEvidence] = []

    async def collect(self, query: str, **kwargs) -> List[MockEvidence]:
        """Collect mock evidence."""
        self._call_count += 1

        if self._delay > 0:
            await asyncio.sleep(self._delay)

        if self._fail_on_call == self._call_count:
            raise ConnectionError(f"{self.name} failed on call {self._call_count}")

        evidence = []
        for i in range(self._evidence_count):
            e = MockEvidence(
                source=f"{self.name}_source_{i}",
                content=f"Evidence from {self.name} for query: {query}",
                url=f"https://{self.name}.test/evidence/{i}",
                metadata={"connector": self.name, "index": i},
            )
            evidence.append(e)
            self._collected.append(e)

        return evidence


class SlowConnector(MockConnector):
    """Connector that simulates slow responses."""

    def __init__(self, name: str = "slow", delay: float = 0.5):
        super().__init__(name=name, delay=delay)


class FailingConnector(MockConnector):
    """Connector that fails on specified call."""

    def __init__(self, name: str = "failing", fail_on_call: int = 1):
        super().__init__(name=name, fail_on_call=fail_on_call)


# =============================================================================
# Multi-Connector Aggregation Tests
# =============================================================================


class TestMultiConnectorAggregation:
    """Tests for aggregating data from multiple connectors."""

    @pytest.mark.asyncio
    async def test_collect_from_multiple_connectors(self):
        """Test collecting evidence from multiple connectors."""
        connectors = [
            MockConnector(name="github", evidence_count=2),
            MockConnector(name="twitter", evidence_count=3),
            MockConnector(name="web", evidence_count=1),
        ]

        all_evidence = []
        for connector in connectors:
            evidence = await connector.collect("test query")
            all_evidence.extend(evidence)

        assert len(all_evidence) == 6  # 2 + 3 + 1
        assert len([e for e in all_evidence if e.metadata["connector"] == "github"]) == 2
        assert len([e for e in all_evidence if e.metadata["connector"] == "twitter"]) == 3
        assert len([e for e in all_evidence if e.metadata["connector"] == "web"]) == 1

    @pytest.mark.asyncio
    async def test_concurrent_connector_collection(self):
        """Test collecting from connectors concurrently."""
        connectors = [MockConnector(name=f"connector_{i}", evidence_count=2) for i in range(5)]

        start_time = time.time()
        tasks = [connector.collect("test query") for connector in connectors]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        all_evidence = [e for result in results for e in result]
        assert len(all_evidence) == 10  # 5 connectors * 2 evidence each

        # Concurrent execution should be fast
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_slow_connector_timeout(self):
        """Test handling slow connectors with timeout."""
        fast_connector = MockConnector(name="fast", delay=0.0)
        slow_connector = SlowConnector(name="slow", delay=2.0)

        async def collect_with_timeout(connector, query, timeout=0.5):
            try:
                return await asyncio.wait_for(
                    connector.collect(query),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                return []

        results = await asyncio.gather(
            collect_with_timeout(fast_connector, "query"),
            collect_with_timeout(slow_connector, "query"),
        )

        # Fast connector should return evidence
        assert len(results[0]) == 3

        # Slow connector should timeout (return empty)
        assert len(results[1]) == 0

    @pytest.mark.asyncio
    async def test_deduplication_across_connectors(self):
        """Test deduplicating evidence across connectors."""
        # Two connectors that might return overlapping content
        connector1 = MockConnector(name="source1")
        connector2 = MockConnector(name="source2")

        evidence1 = await connector1.collect("query")
        evidence2 = await connector2.collect("query")

        all_evidence = evidence1 + evidence2

        # Simple deduplication by URL
        seen_urls = set()
        deduplicated = []
        for e in all_evidence:
            if e.url not in seen_urls:
                seen_urls.add(e.url)
                deduplicated.append(e)

        # All should be unique since URLs are different
        assert len(deduplicated) == 6


# =============================================================================
# Multi-Connector Failover Tests
# =============================================================================


class TestMultiConnectorFailover:
    """Tests for connector failover and resilience."""

    @pytest.mark.asyncio
    async def test_continue_on_single_failure(self):
        """Test other connectors continue when one fails."""
        connectors = [
            MockConnector(name="healthy1"),
            FailingConnector(name="failing"),
            MockConnector(name="healthy2"),
        ]

        results = []
        errors = []

        for connector in connectors:
            try:
                evidence = await connector.collect("query")
                results.extend(evidence)
            except Exception as e:
                errors.append((connector.name, str(e)))

        # Should have evidence from healthy connectors
        assert len(results) == 6  # 3 + 3 from healthy connectors

        # Should have one error
        assert len(errors) == 1
        assert errors[0][0] == "failing"

    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test graceful degradation when multiple connectors fail."""
        connectors = [
            FailingConnector(name="failing1", fail_on_call=1),
            FailingConnector(name="failing2", fail_on_call=1),
            MockConnector(name="healthy"),
        ]

        successful_results = []
        failed_connectors = []

        for connector in connectors:
            try:
                evidence = await connector.collect("query")
                successful_results.extend(evidence)
            except Exception:
                failed_connectors.append(connector.name)

        # Should still have results from healthy connector
        assert len(successful_results) == 3

        # Two connectors should have failed
        assert len(failed_connectors) == 2

    @pytest.mark.asyncio
    async def test_gather_return_exceptions(self):
        """Test using gather with return_exceptions for resilience."""
        connectors = [
            MockConnector(name="healthy1"),
            FailingConnector(name="failing"),
            MockConnector(name="healthy2"),
        ]

        tasks = [c.collect("query") for c in connectors]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Should have 3 results (2 lists + 1 exception)
        assert len(results) == 3

        # Check healthy results
        healthy_results = [r for r in results if isinstance(r, list)]
        assert len(healthy_results) == 2

        # Check exception
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 1

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(self):
        """Test retrying connector on transient failure."""
        # Connector that fails first call, succeeds on second
        connector = FailingConnector(name="transient", fail_on_call=1)
        connector._evidence_count = 2

        result = None
        max_retries = 3

        for attempt in range(max_retries):
            try:
                result = await connector.collect("query")
                break
            except Exception:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.1)

        # Should succeed on second attempt
        assert result is not None
        assert len(result) == 2


# =============================================================================
# Connector Registry Tests
# =============================================================================


class MockConnectorRegistry:
    """Mock connector registry for testing."""

    def __init__(self):
        self._connectors: Dict[str, MockConnector] = {}

    def register(self, name: str, connector: MockConnector) -> None:
        self._connectors[name] = connector

    def get(self, name: str) -> MockConnector:
        if name not in self._connectors:
            raise KeyError(f"Connector not found: {name}")
        return self._connectors[name]

    def list_connectors(self) -> List[str]:
        return list(self._connectors.keys())

    def get_by_capability(self, capability: str) -> List[MockConnector]:
        # Simple mock - in real implementation would filter by connector capabilities
        return list(self._connectors.values())


class TestConnectorRegistry:
    """Tests for connector registry functionality."""

    def test_register_connector(self):
        """Test registering a connector."""
        registry = MockConnectorRegistry()
        connector = MockConnector(name="test")

        registry.register("test", connector)

        assert "test" in registry.list_connectors()
        assert registry.get("test") == connector

    def test_get_unknown_connector(self):
        """Test getting unknown connector raises error."""
        registry = MockConnectorRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.get("unknown")

    def test_list_multiple_connectors(self):
        """Test listing multiple connectors."""
        registry = MockConnectorRegistry()

        for name in ["github", "twitter", "web"]:
            registry.register(name, MockConnector(name=name))

        connectors = registry.list_connectors()
        assert len(connectors) == 3
        assert "github" in connectors
        assert "twitter" in connectors
        assert "web" in connectors

    @pytest.mark.asyncio
    async def test_collect_from_registry(self):
        """Test collecting from all registered connectors."""
        registry = MockConnectorRegistry()

        for name in ["source1", "source2", "source3"]:
            registry.register(name, MockConnector(name=name, evidence_count=2))

        all_evidence = []
        for name in registry.list_connectors():
            connector = registry.get(name)
            evidence = await connector.collect("query")
            all_evidence.extend(evidence)

        assert len(all_evidence) == 6  # 3 connectors * 2 evidence


# =============================================================================
# Concurrent Operations Tests
# =============================================================================


class TestConcurrentOperations:
    """Tests for concurrent connector operations."""

    @pytest.mark.asyncio
    async def test_parallel_queries_same_connector(self):
        """Test running parallel queries on same connector."""
        connector = MockConnector(name="shared", evidence_count=2)

        queries = ["query1", "query2", "query3"]
        tasks = [connector.collect(q) for q in queries]
        results = await asyncio.gather(*tasks)

        # Should have results for all queries
        assert len(results) == 3

        # Each should have evidence
        for result in results:
            assert len(result) == 2

        # Connector should have been called 3 times
        assert connector._call_count == 3

    @pytest.mark.asyncio
    async def test_semaphore_rate_limiting(self):
        """Test rate limiting with semaphore."""
        connector = MockConnector(name="limited", delay=0.1)
        max_concurrent = 2
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_collect(query: str):
            async with semaphore:
                return await connector.collect(query)

        queries = [f"query_{i}" for i in range(5)]
        start_time = time.time()

        tasks = [limited_collect(q) for q in queries]
        results = await asyncio.gather(*tasks)

        elapsed = time.time() - start_time

        # All queries should complete
        assert len(results) == 5

        # With semaphore=2 and delay=0.1, 5 queries should take ~0.3s
        # (0.1 + 0.1 + 0.1 with 2 concurrent)
        assert elapsed >= 0.2  # At least 3 rounds

    @pytest.mark.asyncio
    async def test_independent_connector_errors(self):
        """Test that connector errors are independent."""
        connectors = {
            "c1": MockConnector(name="c1"),
            "c2": FailingConnector(name="c2"),
            "c3": MockConnector(name="c3"),
        }

        results = {}
        errors = {}

        async def safe_collect(name: str, connector: MockConnector):
            try:
                return name, await connector.collect("query")
            except Exception as e:
                return name, e

        tasks = [safe_collect(n, c) for n, c in connectors.items()]
        raw_results = await asyncio.gather(*tasks)

        for name, result in raw_results:
            if isinstance(result, Exception):
                errors[name] = result
            else:
                results[name] = result

        # c1 and c3 should succeed
        assert "c1" in results
        assert "c3" in results
        assert len(results["c1"]) == 3
        assert len(results["c3"]) == 3

        # c2 should fail
        assert "c2" in errors


# =============================================================================
# Cross-Connector Coordination Tests
# =============================================================================


class TestCrossConnectorCoordination:
    """Tests for coordinating across multiple connectors."""

    @pytest.mark.asyncio
    async def test_evidence_merging_strategy(self):
        """Test merging evidence from multiple sources with priority."""
        primary = MockConnector(name="primary", evidence_count=5)
        secondary = MockConnector(name="secondary", evidence_count=3)
        fallback = MockConnector(name="fallback", evidence_count=2)

        # Collect from all
        primary_evidence = await primary.collect("query")
        secondary_evidence = await secondary.collect("query")
        fallback_evidence = await fallback.collect("query")

        # Merge with priority (primary first)
        merged = primary_evidence + secondary_evidence + fallback_evidence

        # Verify order
        assert merged[0].metadata["connector"] == "primary"
        assert merged[5].metadata["connector"] == "secondary"
        assert merged[8].metadata["connector"] == "fallback"

    @pytest.mark.asyncio
    async def test_connector_chain(self):
        """Test chaining connectors where one feeds another."""
        # First connector gets initial results
        initial_connector = MockConnector(name="initial", evidence_count=2)
        initial_evidence = await initial_connector.collect("initial query")

        # Second connector uses initial results
        enrichment_connector = MockConnector(name="enrichment", evidence_count=1)

        enriched = []
        for e in initial_evidence:
            # Use initial evidence to query enrichment
            enrichment = await enrichment_connector.collect(e.content[:20])
            enriched.extend(enrichment)

        # Should have enrichment for each initial evidence
        assert len(enriched) == 2

    @pytest.mark.asyncio
    async def test_conditional_connector_selection(self):
        """Test selecting connectors based on query type."""
        connectors = {
            "code": MockConnector(name="github"),
            "news": MockConnector(name="newsapi"),
            "social": MockConnector(name="twitter"),
            "web": MockConnector(name="web"),
        }

        def select_connectors(query: str) -> List[str]:
            """Select appropriate connectors based on query."""
            if "code" in query.lower() or "repository" in query.lower():
                return ["code", "web"]
            if "news" in query.lower() or "current" in query.lower():
                return ["news", "social"]
            return ["web"]

        # Test code query
        code_connectors = select_connectors("code repository for machine learning")
        assert "code" in code_connectors
        assert "web" in code_connectors

        # Test news query
        news_connectors = select_connectors("current news about AI")
        assert "news" in news_connectors
        assert "social" in news_connectors

        # Test generic query
        generic_connectors = select_connectors("what is photosynthesis")
        assert "web" in generic_connectors


# =============================================================================
# Performance Tests
# =============================================================================


class TestMultiConnectorPerformance:
    """Performance tests for multi-connector scenarios."""

    @pytest.mark.asyncio
    async def test_many_connectors_parallel(self):
        """Test performance with many connectors in parallel."""
        num_connectors = 20
        connectors = [MockConnector(name=f"c{i}", evidence_count=5) for i in range(num_connectors)]

        start_time = time.time()
        tasks = [c.collect("query") for c in connectors]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        total_evidence = sum(len(r) for r in results)
        assert total_evidence == num_connectors * 5

        # Should complete quickly since connectors are fast
        assert elapsed < 2.0

    @pytest.mark.asyncio
    async def test_mixed_speed_connectors(self):
        """Test handling mixed fast and slow connectors."""
        connectors = [
            MockConnector(name="fast1", delay=0.0),
            SlowConnector(name="slow", delay=0.3),
            MockConnector(name="fast2", delay=0.0),
        ]

        start_time = time.time()
        tasks = [c.collect("query") for c in connectors]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        # All should complete
        assert len(results) == 3

        # Total time should be approximately the slowest (not sum)
        assert elapsed < 0.5  # Not 0.3 + 0 + 0


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestMultiConnectorErrorHandling:
    """Error handling tests for multi-connector scenarios."""

    @pytest.mark.asyncio
    async def test_collect_best_effort(self):
        """Test best-effort collection with partial failures."""
        connectors = [
            MockConnector(name="ok1"),
            FailingConnector(name="fail1"),
            MockConnector(name="ok2"),
            FailingConnector(name="fail2"),
        ]

        all_evidence = []
        errors = []

        for connector in connectors:
            try:
                evidence = await connector.collect("query")
                all_evidence.extend(evidence)
            except Exception as e:
                errors.append(str(e))

        # Should have collected from successful connectors
        assert len(all_evidence) == 6  # 3 + 3

        # Should have recorded failures
        assert len(errors) == 2

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test handling timeouts across connectors."""
        connectors = [
            MockConnector(name="fast"),
            SlowConnector(name="very_slow", delay=5.0),
        ]

        async def collect_with_timeout(connector, timeout=1.0):
            try:
                return await asyncio.wait_for(connector.collect("query"), timeout)
            except asyncio.TimeoutError:
                return None

        results = await asyncio.gather(*[collect_with_timeout(c, timeout=1.0) for c in connectors])

        # Fast connector should succeed
        assert results[0] is not None
        assert len(results[0]) == 3

        # Slow connector should timeout
        assert results[1] is None

    @pytest.mark.asyncio
    async def test_partial_result_aggregation(self):
        """Test aggregating partial results from failed operations."""
        results_so_far = []

        async def collect_and_store(connector, storage):
            try:
                evidence = await connector.collect("query")
                storage.extend(evidence)
                return True
            except Exception:
                return False

        connectors = [
            MockConnector(name="c1"),
            FailingConnector(name="c2"),
            MockConnector(name="c3"),
        ]

        successes = []
        for connector in connectors:
            success = await collect_and_store(connector, results_so_far)
            successes.append(success)

        assert successes == [True, False, True]
        assert len(results_so_far) == 6  # Only from successful connectors
