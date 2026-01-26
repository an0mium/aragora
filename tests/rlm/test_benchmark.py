"""
Performance benchmarks for RLM compression and query operations.

Uses pytest-benchmark for reproducible benchmarking.

Run with:
    pytest tests/rlm/test_benchmark.py -v --benchmark-only
    pytest tests/rlm/test_benchmark.py -v --benchmark-compare
    pytest tests/rlm/test_benchmark.py -v --benchmark-autosave
"""

import asyncio
from typing import Any
import pytest

from aragora.rlm.bridge import AragoraRLM, DebateContextAdapter
from aragora.rlm.compressor import (
    HierarchicalCompressor,
    LRUCompressionCache,
    clear_compression_cache,
)
from aragora.rlm.types import (
    AbstractionLevel,
    AbstractionNode,
    RLMConfig,
    RLMContext,
    RLMResult,
)
from aragora.core import DebateResult, Message


# Test data fixtures
@pytest.fixture
def short_content() -> str:
    """Short content (~100 tokens)."""
    return """
    Feature flags are essential for safe deployments.
    They allow you to control rollout percentages, A/B test features,
    and quickly disable problematic code without full rollbacks.
    This is critical for maintaining production stability.
    """


@pytest.fixture
def medium_content() -> str:
    """Medium content (~1000 tokens)."""
    paragraphs = [
        """
        Software deployment practices have evolved significantly over the past decade.
        Modern teams employ continuous integration and continuous deployment (CI/CD)
        pipelines to automate the build, test, and deployment processes. This automation
        reduces human error and enables faster iteration cycles.
        """,
        """
        Feature flags represent a key advancement in deployment strategies. By wrapping
        new features in conditional logic, teams can deploy code to production without
        immediately exposing it to all users. This decoupling of deployment from release
        provides unprecedented control over the user experience.
        """,
        """
        The benefits of feature flags extend beyond simple on/off toggles. Advanced
        implementations support percentage-based rollouts, user segment targeting, and
        A/B testing. This granularity enables data-driven decisions about feature launches
        and quick rollback capabilities when issues arise.
        """,
        """
        However, feature flags introduce their own complexity. Without proper management,
        technical debt can accumulate as unused flags linger in the codebase. Teams must
        establish lifecycle policies that mandate regular cleanup and documentation of
        all active flags.
        """,
        """
        Best practices for feature flag management include: establishing naming conventions,
        maintaining documentation, setting expiration dates, integrating with monitoring
        systems, and conducting regular audits. These practices ensure that the benefits
        of feature flags outweigh their maintenance costs.
        """,
    ] * 4  # Repeat to get ~1000 tokens

    return "\n\n".join(paragraphs)


@pytest.fixture
def long_content() -> str:
    """Long content (~10000 tokens)."""
    paragraphs = [
        """
        The evolution of software engineering practices has been marked by continuous
        innovation in how teams build, test, and deploy applications. From waterfall
        methodologies to agile practices, the industry has consistently moved toward
        faster feedback loops and more iterative development processes.
        """,
        """
        Continuous integration emerged as a fundamental practice in the early 2000s,
        enabling teams to merge code changes frequently and catch integration issues
        early. This practice laid the groundwork for continuous deployment, where
        code changes automatically flow to production after passing automated tests.
        """,
        """
        The microservices architecture pattern has further accelerated deployment
        practices. By decomposing monolithic applications into smaller, independently
        deployable services, teams can release updates to specific components without
        affecting the entire system. This granularity enables faster iteration and
        reduced risk per deployment.
        """,
        """
        Container technologies like Docker and orchestration platforms like Kubernetes
        have revolutionized how applications are packaged and deployed. Containers
        provide consistent environments across development, testing, and production,
        while Kubernetes manages the complexity of running applications at scale.
        """,
        """
        Infrastructure as Code (IaC) tools such as Terraform and CloudFormation
        enable teams to define and provision infrastructure through code. This
        approach brings the same version control and review processes used for
        application code to infrastructure management, improving reliability and
        reproducibility.
        """,
    ] * 20  # Repeat to get ~10000 tokens

    return "\n\n".join(paragraphs)


@pytest.fixture
def sample_context() -> RLMContext:
    """Pre-built RLM context for query benchmarks."""
    context = RLMContext(
        original_content="Test content about feature flags and deployments.",
        original_tokens=1000,
        source_type="text",
    )

    # Add summary level
    summary_nodes = [
        AbstractionNode(
            id=f"summary_{i}",
            level=AbstractionLevel.SUMMARY,
            content=f"Summary {i}: Key points about topic {i}.",
            token_count=20,
            key_topics=[f"topic_{i}"],
        )
        for i in range(10)
    ]
    context.levels[AbstractionLevel.SUMMARY] = summary_nodes
    for node in summary_nodes:
        context.nodes_by_id[node.id] = node

    # Add detailed level
    detailed_nodes = [
        AbstractionNode(
            id=f"detailed_{i}",
            level=AbstractionLevel.DETAILED,
            content=f"Detailed {i}: Comprehensive explanation of topic {i} with examples.",
            token_count=50,
            parent_id=f"summary_{i // 2}",
        )
        for i in range(20)
    ]
    context.levels[AbstractionLevel.DETAILED] = detailed_nodes
    for node in detailed_nodes:
        context.nodes_by_id[node.id] = node

    return context


@pytest.fixture
def sample_debate_result() -> DebateResult:
    """Sample debate result for compression benchmarks."""
    return DebateResult(
        debate_id="bench-debate",
        task="Should we use feature flags?",
        status="completed",
        participants=["alice", "bob", "charlie"],
        messages=[
            Message(
                role="proposer",
                agent="alice",
                content=f"Proposal {i}: Feature flags provide deployment safety.",
                round=i // 3 + 1,
            )
            for i in range(12)  # 4 rounds of 3 messages each
        ],
        consensus_reached=True,
        final_answer="Yes, with proper lifecycle management.",
        rounds_completed=4,
        confidence=0.85,
    )


class TestCompressionBenchmarks:
    """Benchmarks for compression operations."""

    def test_benchmark_cache_operations(self, benchmark: Any):
        """Benchmark LRU cache get/set operations."""
        cache = LRUCompressionCache(max_size=100)
        context = RLMContext(
            original_content="Test",
            original_tokens=10,
        )

        def cache_ops():
            for i in range(100):
                key = f"key_{i % 10}"
                cache.set(key, context)
                cache.get(key)

        benchmark(cache_ops)

    def test_benchmark_token_counting(self, benchmark: Any, medium_content: str):
        """Benchmark token counting operations."""
        compressor = HierarchicalCompressor()

        benchmark(compressor._count_tokens, medium_content)

    def test_benchmark_chunk_content_small(self, benchmark: Any, short_content: str):
        """Benchmark chunking small content."""
        compressor = HierarchicalCompressor()

        benchmark(compressor._chunk_content, short_content)

    def test_benchmark_chunk_content_large(self, benchmark: Any, long_content: str):
        """Benchmark chunking large content."""
        compressor = HierarchicalCompressor()

        benchmark(compressor._chunk_content, long_content)


class TestCompressionAsyncBenchmarks:
    """Async benchmarks for compression operations."""

    def test_benchmark_compress_short(self, benchmark: Any, short_content: str):
        """Benchmark compressing short content."""
        clear_compression_cache()
        compressor = HierarchicalCompressor(config=RLMConfig(cache_compressions=False))

        def run_compress():
            return asyncio.run(compressor.compress(short_content, source_type="text"))

        benchmark.pedantic(run_compress, rounds=3, iterations=1)

    def test_benchmark_compress_medium(self, benchmark: Any, medium_content: str):
        """Benchmark compressing medium content."""
        clear_compression_cache()
        compressor = HierarchicalCompressor(config=RLMConfig(cache_compressions=False))

        def run_compress():
            return asyncio.run(compressor.compress(medium_content, source_type="text"))

        benchmark.pedantic(run_compress, rounds=3, iterations=1)

    def test_benchmark_compress_with_cache(self, benchmark: Any, medium_content: str):
        """Benchmark compression with cache enabled (cache hits)."""
        clear_compression_cache()
        compressor = HierarchicalCompressor(config=RLMConfig(cache_compressions=True))

        # Warm up cache
        asyncio.run(compressor.compress(medium_content, source_type="text"))

        def run_compress():
            return asyncio.run(compressor.compress(medium_content, source_type="text"))

        benchmark.pedantic(run_compress, rounds=5, iterations=1)


class TestQueryBenchmarks:
    """Benchmarks for query operations."""

    def test_benchmark_context_level_access(self, benchmark: Any, sample_context: RLMContext):
        """Benchmark accessing content at different levels."""

        def access_levels():
            for level in [AbstractionLevel.SUMMARY, AbstractionLevel.DETAILED]:
                sample_context.get_at_level(level)

        benchmark(access_levels)

    def test_benchmark_node_lookup(self, benchmark: Any, sample_context: RLMContext):
        """Benchmark node lookup by ID."""

        def lookup_nodes():
            for i in range(20):
                sample_context.get_node(f"summary_{i % 10}")
                sample_context.get_node(f"detailed_{i}")

        benchmark(lookup_nodes)

    def test_benchmark_drill_down(self, benchmark: Any, sample_context: RLMContext):
        """Benchmark drill-down operations."""
        # Set up parent-child relationships
        for i, node in enumerate(sample_context.levels[AbstractionLevel.SUMMARY]):
            node.child_ids = [f"detailed_{i * 2}", f"detailed_{i * 2 + 1}"]

        def drill_down_ops():
            for i in range(10):
                sample_context.drill_down(f"summary_{i}")

        benchmark(drill_down_ops)

    def test_benchmark_total_tokens_at_level(self, benchmark: Any, sample_context: RLMContext):
        """Benchmark token counting at levels."""

        def count_tokens():
            for level in AbstractionLevel:
                sample_context.total_tokens_at_level(level)

        benchmark(count_tokens)


class TestDebateCompressionBenchmarks:
    """Benchmarks for debate-specific compression."""

    def test_benchmark_debate_formatting(self, benchmark: Any, sample_debate_result: DebateResult):
        """Benchmark debate formatting for RLM."""
        adapter = DebateContextAdapter()

        benchmark(adapter.format_for_rlm, sample_debate_result)

    def test_benchmark_debate_to_text(self, benchmark: Any, sample_debate_result: DebateResult):
        """Benchmark converting debate to text."""
        adapter = DebateContextAdapter()

        benchmark(adapter.to_text, sample_debate_result)

    def test_benchmark_debate_compression(self, benchmark: Any, sample_debate_result: DebateResult):
        """Benchmark full debate compression."""
        clear_compression_cache()
        adapter = DebateContextAdapter()

        def run_compress():
            return asyncio.run(adapter.compress_debate(sample_debate_result))

        benchmark.pedantic(run_compress, rounds=3, iterations=1)


class TestStreamingBenchmarks:
    """Benchmarks for streaming operations."""

    def test_benchmark_stream_event_creation(self, benchmark: Any):
        """Benchmark stream event creation."""
        from aragora.rlm.types import RLMStreamEvent, RLMStreamEventType

        def create_events():
            for _ in range(100):
                RLMStreamEvent(
                    event_type=RLMStreamEventType.QUERY_START,
                    query="test",
                )

        benchmark(create_events)

    def test_benchmark_stream_event_to_dict(self, benchmark: Any):
        """Benchmark stream event serialization."""
        from aragora.rlm.types import RLMStreamEvent, RLMStreamEventType

        events = [
            RLMStreamEvent(
                event_type=RLMStreamEventType.NODE_EXAMINED,
                query="test",
                level=AbstractionLevel.SUMMARY,
                node_id=f"node_{i}",
                tokens_processed=100 * i,
            )
            for i in range(100)
        ]

        def serialize_events():
            for event in events:
                event.to_dict()

        benchmark(serialize_events)


class TestMetricsBenchmarks:
    """Benchmarks for metrics operations."""

    def test_benchmark_record_compression(self, benchmark: Any):
        """Benchmark recording compression metrics."""
        from aragora.server.prometheus_rlm import record_rlm_compression

        def record_metrics():
            for _ in range(100):
                record_rlm_compression(
                    source_type="debate",
                    original_tokens=10000,
                    compressed_tokens=3000,
                    levels=4,
                    duration_seconds=0.5,
                )

        benchmark(record_metrics)

    def test_benchmark_record_refinement(self, benchmark: Any):
        """Benchmark recording refinement metrics."""
        from aragora.server.prometheus_rlm import record_rlm_refinement

        def record_metrics():
            for _ in range(100):
                record_rlm_refinement(
                    strategy="grep",
                    iterations=3,
                    success=True,
                    duration_seconds=1.5,
                )

        benchmark(record_metrics)


# Additional utility benchmarks


class TestCacheBenchmarks:
    """Detailed cache benchmarks."""

    def test_benchmark_cache_hit_rate(self, benchmark: Any):
        """Benchmark cache with mixed hit/miss pattern."""
        cache = LRUCompressionCache(max_size=100)
        contexts = [
            RLMContext(original_content=f"Content {i}", original_tokens=i * 10) for i in range(50)
        ]

        # Populate half the cache
        for i, ctx in enumerate(contexts[:25]):
            cache.set(f"key_{i}", ctx)

        def mixed_operations():
            hits = 0
            misses = 0
            for i in range(100):
                key = f"key_{i % 50}"
                result = cache.get(key)
                if result:
                    hits += 1
                else:
                    misses += 1
                    cache.set(key, contexts[i % 50])

        benchmark(mixed_operations)

    def test_benchmark_cache_eviction(self, benchmark: Any):
        """Benchmark cache eviction under pressure."""
        cache = LRUCompressionCache(max_size=50)

        def eviction_operations():
            for i in range(200):
                ctx = RLMContext(
                    original_content=f"Content {i}",
                    original_tokens=i,
                )
                cache.set(f"key_{i}", ctx)

        benchmark(eviction_operations)
