#!/usr/bin/env python3
"""
Debate Engine Performance Profiler.

This script profiles the core debate engine operations to identify performance
bottlenecks. It measures and reports on:

1. Debate initialization (Arena setup, phase creation)
2. Proposal generation (parallel agent calls)
3. Critique and revision phases
4. Convergence detection (semantic similarity)
5. Consensus calculation (vote collection, weighting)
6. Memory operations (cache lookups, embeddings)

Usage:
    # Run full profiling suite
    python scripts/profile_debate.py

    # Run with cProfile output
    python scripts/profile_debate.py --cprofile

    # Profile specific component
    python scripts/profile_debate.py --component convergence

    # Run memory profiling
    python scripts/profile_debate.py --memory

    # Generate flame graph (requires py-spy)
    python scripts/profile_debate.py --flamegraph

Performance Bottlenecks Identified
==================================

After analyzing the debate engine codebase, the following bottlenecks are most likely:

1. CONVERGENCE DETECTION (aragora/debate/convergence.py)
   - Semantic similarity computation is O(n^2) for within-round convergence
   - Embedding generation is expensive (SentenceTransformer)
   - OPTIMIZATION: Already has ANN-based fast path and embedding caching

2. PARALLEL AGENT CALLS (proposal_phase.py, debate_rounds.py)
   - Bounded concurrency via semaphores (MAX_CONCURRENT_PROPOSALS/CRITIQUES)
   - Timeout protection can cause stragglers to block phases
   - OPTIMIZATION: Consider async gather with return_exceptions

3. VOTE COLLECTION (vote_collector.py)
   - Serial processing in as_completed loop
   - RLM early termination optimization exists but requires majority
   - Position shuffling multiplies vote collection time

4. WEIGHT CALCULATION (weight_calculator.py)
   - Computed per-agent for each consensus
   - OPTIMIZATION: Cache weights within a debate

5. MEMORY LOOKUPS (memory_manager.py, continuum.py)
   - ContinuumMemory retrieval involves embedding search
   - Multiple tiers (fast/medium/slow/glacial) are checked sequentially

6. PROMPT BUILDING (prompt_builder.py)
   - String formatting for each agent call
   - Historical context fetching for each prompt

Recommendations
===============

1. CACHE VOTE WEIGHTS: Pre-compute vote weights once per consensus phase
2. BATCH EMBEDDING LOOKUPS: Use batch methods for SentenceTransformer
3. ASYNC CONTEXT GATHERING: Parallelize historical/knowledge context fetching
4. CONNECTION POOLING: Ensure database connections are pooled
5. LAZY PHASE INITIALIZATION: Don't initialize unused phases
6. PROFILE INDIVIDUAL DEBATES: Use DebatePerformanceMonitor for production
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import gc
import io
import logging
import pstats
import statistics
import sys
import tempfile
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING
from collections.abc import Callable

if TYPE_CHECKING:
    from aragora.core import Critique, Message, Vote

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# =============================================================================
# Mock Agent for Profiling
# =============================================================================


class MockAgent:
    """
    Mock agent implementation for profiling purposes.

    This class provides a concrete implementation of the Agent interface
    without requiring actual API calls. Used for benchmarking Arena
    initialization, weight calculation, and other non-generation operations.
    """

    def __init__(
        self,
        name: str,
        model_type: str = "mock",
        provider: str = "mock",
        role: str = "proposer",
    ):
        """
        Initialize a mock agent.

        Args:
            name: Unique identifier for the agent
            model_type: Model type identifier (default: "mock")
            provider: Provider identifier (default: "mock")
            role: Agent role (default: "proposer")
        """
        self.name = name
        self.model = f"{provider}/{model_type}"
        self.model_type = model_type
        self.provider = provider
        self.role = role
        self.system_prompt = ""
        self.agent_type = "mock"
        self.stance = "neutral"
        self.tool_manifest = None

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a mock response."""
        return f"Mock response from {self.name} for prompt: {prompt[:50]}..."

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list[Message] | None = None,
        target_agent: str | None = None,
    ) -> Critique:
        """Generate a mock critique."""
        from aragora.core import Critique

        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content=proposal[:100],
            issues=["Mock issue 1", "Mock issue 2"],
            suggestions=["Mock suggestion 1"],
            severity=5.0,
            reasoning="Mock reasoning for critique",
        )

    async def vote(self, proposals: dict[str, str], task: str) -> Vote:
        """Generate a mock vote."""
        from aragora.core import Vote

        # Vote for first proposal
        choice = list(proposals.keys())[0] if proposals else "none"
        return Vote(
            agent=self.name,
            choice=choice,
            confidence=0.8,
            reasoning="Mock voting reasoning",
            continue_debate=True,
        )

    def set_system_prompt(self, prompt: str) -> None:
        """Update the agent's system prompt."""
        self.system_prompt = prompt

    def has_tool_permission(self, tool_name: str) -> bool:
        """Check if agent has permission to use a specific tool."""
        return True

    def __repr__(self) -> str:
        return f"MockAgent(name={self.name}, model={self.model}, role={self.role})"


@dataclass
class ProfileResult:
    """Result of profiling a single operation."""

    name: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_dev: float
    memory_peak_mb: float = 0.0
    memory_increase_mb: float = 0.0
    gc_collected: int = 0

    def __str__(self) -> str:
        lines = [
            f"{self.name}:",
            f"  Iterations: {self.iterations}",
            f"  Total: {self.total_time * 1000:.2f}ms",
            f"  Avg: {self.avg_time * 1000:.3f}ms",
            f"  Min: {self.min_time * 1000:.3f}ms",
            f"  Max: {self.max_time * 1000:.3f}ms",
            f"  StdDev: {self.std_dev * 1000:.3f}ms",
        ]
        if self.memory_peak_mb > 0:
            lines.append(f"  Memory Peak: {self.memory_peak_mb:.2f}MB")
            lines.append(f"  Memory Increase: {self.memory_increase_mb:.2f}MB")
        if self.gc_collected > 0:
            lines.append(f"  GC Collected: {self.gc_collected}")
        return "\n".join(lines)


@dataclass
class AsyncProfileResult:
    """Result of profiling an async operation."""

    name: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_dev: float
    concurrent_tasks: int = 0

    def __str__(self) -> str:
        lines = [
            f"{self.name}:",
            f"  Iterations: {self.iterations}",
            f"  Total: {self.total_time * 1000:.2f}ms",
            f"  Avg: {self.avg_time * 1000:.3f}ms",
            f"  Min: {self.min_time * 1000:.3f}ms",
            f"  Max: {self.max_time * 1000:.3f}ms",
            f"  StdDev: {self.std_dev * 1000:.3f}ms",
        ]
        if self.concurrent_tasks > 0:
            lines.append(f"  Concurrent Tasks: {self.concurrent_tasks}")
        return "\n".join(lines)


@contextmanager
def timer():
    """Context manager for timing operations."""
    start = time.perf_counter()

    def elapsed():
        return time.perf_counter() - start

    yield elapsed


def profile_function(
    func: Callable,
    iterations: int = 100,
    name: str | None = None,
    track_memory: bool = False,
) -> ProfileResult:
    """Profile a synchronous function over multiple iterations."""
    times: list[float] = []
    memory_start = 0
    memory_peak = 0
    gc_total = 0

    if track_memory:
        tracemalloc.start()
        memory_start = tracemalloc.get_traced_memory()[0]

    for i in range(iterations):
        gc.collect()
        gc_before = gc.get_count()

        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        gc_after = gc.get_count()
        gc_total += sum(gc_after) - sum(gc_before)

    if track_memory:
        memory_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

    return ProfileResult(
        name=name or func.__name__,
        iterations=iterations,
        total_time=sum(times),
        avg_time=statistics.mean(times),
        min_time=min(times),
        max_time=max(times),
        std_dev=statistics.stdev(times) if len(times) > 1 else 0.0,
        memory_peak_mb=memory_peak / (1024 * 1024) if track_memory else 0.0,
        memory_increase_mb=(memory_peak - memory_start) / (1024 * 1024) if track_memory else 0.0,
        gc_collected=gc_total,
    )


async def profile_async_function(
    func: Callable,
    iterations: int = 10,
    name: str | None = None,
    concurrent: bool = False,
) -> AsyncProfileResult:
    """Profile an async function over multiple iterations."""
    times: list[float] = []
    concurrent_count = 0

    if concurrent:
        # Run all iterations concurrently
        async def timed_call():
            start = time.perf_counter()
            await func()
            return time.perf_counter() - start

        start_total = time.perf_counter()
        times = await asyncio.gather(*[timed_call() for _ in range(iterations)])
        concurrent_count = iterations
    else:
        # Run iterations sequentially
        for _ in range(iterations):
            start = time.perf_counter()
            await func()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

    return AsyncProfileResult(
        name=name or func.__name__,
        iterations=iterations,
        total_time=sum(times),
        avg_time=statistics.mean(times),
        min_time=min(times),
        max_time=max(times),
        std_dev=statistics.stdev(times) if len(times) > 1 else 0.0,
        concurrent_tasks=concurrent_count,
    )


# =============================================================================
# Profile Targets: Convergence Detection
# =============================================================================


def profile_convergence_jaccard():
    """Profile Jaccard similarity (always available fallback)."""
    from aragora.debate.similarity.backends import JaccardBackend

    backend = JaccardBackend()
    text1 = "The system should use microservices for better scalability and maintainability." * 5
    text2 = "Microservices architecture provides scalability but adds operational complexity." * 5

    def compute():
        return backend.compute_similarity(text1, text2)

    result = profile_function(compute, iterations=1000, name="Jaccard Similarity")
    print(result)
    return result


def profile_convergence_tfidf():
    """Profile TF-IDF similarity (requires scikit-learn)."""
    try:
        from aragora.debate.similarity.backends import TFIDFBackend

        backend = TFIDFBackend()
    except ImportError:
        print("TF-IDF backend not available (missing scikit-learn)")
        return None

    text1 = "The system should use microservices for better scalability and maintainability." * 5
    text2 = "Microservices architecture provides scalability but adds operational complexity." * 5

    def compute():
        return backend.compute_similarity(text1, text2)

    result = profile_function(compute, iterations=500, name="TF-IDF Similarity", track_memory=True)
    print(result)
    return result


def profile_convergence_transformer():
    """Profile SentenceTransformer similarity (best accuracy, slowest)."""
    try:
        from aragora.debate.similarity.backends import SentenceTransformerBackend

        backend = SentenceTransformerBackend()
    except ImportError:
        print("SentenceTransformer backend not available")
        return None

    text1 = "The system should use microservices for better scalability and maintainability." * 5
    text2 = "Microservices architecture provides scalability but adds operational complexity." * 5

    def compute():
        return backend.compute_similarity(text1, text2)

    result = profile_function(
        compute, iterations=50, name="SentenceTransformer Similarity", track_memory=True
    )
    print(result)
    return result


def profile_convergence_batch():
    """Profile batch similarity computation for multiple agents."""
    try:
        from aragora.debate.similarity.backends import get_similarity_backend

        backend = get_similarity_backend("auto")
    except Exception as e:
        print(f"Could not get similarity backend: {e}")
        return None

    # Simulate 5 agents with proposals
    agent_texts = [
        "We should implement rate limiting using token buckets for scalability.",
        "A sliding window algorithm would provide better burst handling.",
        "Distributed rate limiting requires coordination via Redis.",
        "Token buckets with refill rates work well for API throttling.",
        "Consider leaky bucket for smoother traffic shaping.",
    ]

    def compute_all_pairs():
        """Compute similarity between all pairs (like within-round convergence)."""
        results = []
        for i, t1 in enumerate(agent_texts):
            for j, t2 in enumerate(agent_texts[i + 1 :], i + 1):
                results.append(backend.compute_similarity(t1, t2))
        return results

    result = profile_function(
        compute_all_pairs,
        iterations=100,
        name="Batch Convergence (5 agents, 10 pairs)",
        track_memory=True,
    )
    print(result)
    return result


# =============================================================================
# Profile Targets: Weight Calculation
# =============================================================================


def profile_weight_calculation():
    """Profile vote weight calculation."""
    try:
        from aragora.debate.phases.weight_calculator import (
            WeightCalculator,
            WeightCalculatorConfig,
        )
    except ImportError as e:
        print(f"Could not import weight calculator: {e}")
        return None

    # Create mock agents
    agents = [MockAgent(name=f"agent-{i}", model_type="mock", provider="mock") for i in range(5)]

    config = WeightCalculatorConfig(
        enable_self_vote_mitigation=True,
        enable_verbosity_normalization=True,
    )

    calculator = WeightCalculator(
        memory=None,
        elo_system=None,
        flip_detector=None,
        agent_weights={a.name: 1.0 for a in agents},
        calibration_tracker=None,
        get_calibration_weight=None,
        config=config,
    )

    def compute_weights():
        return calculator.compute_weights(agents)

    result = profile_function(
        compute_weights, iterations=1000, name="Vote Weight Calculation (5 agents)"
    )
    print(result)
    return result


# =============================================================================
# Profile Targets: Memory Operations
# =============================================================================


def profile_critique_store():
    """Profile CritiqueStore operations."""
    from aragora.memory.store import CritiqueStore
    from aragora.core import Critique, DebateResult, Message

    with tempfile.TemporaryDirectory() as tmpdir:
        store = CritiqueStore(db_path=f"{tmpdir}/critique.db")

        # Seed with test data
        for i in range(20):
            messages = [
                Message(round=j, role="proposer", agent=f"agent-{j % 3}", content=f"Content {j}")
                for j in range(5)
            ]
            critiques = [
                Critique(
                    agent=f"agent-{j % 3}",
                    target_agent=f"agent-{(j + 1) % 3}",
                    target_content=f"Target {j}",
                    issues=[f"Issue {k}" for k in range(2)],
                    suggestions=[f"Suggestion {k}" for k in range(2)],
                    severity=0.5,
                    reasoning=f"Reasoning for critique {j}",
                )
                for j in range(3)
            ]
            result = DebateResult(
                id=f"debate-{i}",
                task=f"Task {i} about system design and architecture",
                final_answer=f"Answer {i}",
                confidence=0.8,
                consensus_reached=True,
                messages=messages,
                critiques=critiques,
                votes=[],
            )
            store.store_debate(result)

        def retrieve_patterns():
            return store.retrieve_patterns(limit=10)

        result1 = profile_function(
            retrieve_patterns, iterations=100, name="CritiqueStore.retrieve_patterns"
        )
        print(result1)

        def get_stats():
            return store.get_stats()

        result2 = profile_function(get_stats, iterations=100, name="CritiqueStore.get_stats")
        print(result2)

        return result1, result2


def profile_embedding_cache():
    """Profile embedding cache operations."""
    try:
        from aragora.debate.cache.embeddings_lru import (
            EmbeddingCache,
            get_scoped_embedding_cache,
            cleanup_embedding_cache,
        )
    except ImportError as e:
        print(f"Embedding cache not available: {e}")
        return None

    cache = get_scoped_embedding_cache("test-debate")

    # Pre-populate cache
    test_texts = [f"Test text {i} for embedding cache profiling" for i in range(100)]
    mock_embeddings = [[0.1] * 384 for _ in test_texts]  # Mock 384-dim embeddings

    for text, emb in zip(test_texts, mock_embeddings):
        cache.put(text, emb)

    def cache_hit():
        return cache.get("Test text 50 for embedding cache profiling")

    result1 = profile_function(cache_hit, iterations=10000, name="Embedding Cache Hit")
    print(result1)

    def cache_miss():
        return cache.get("This text is not in the cache")

    result2 = profile_function(cache_miss, iterations=10000, name="Embedding Cache Miss")
    print(result2)

    cleanup_embedding_cache("test-debate")
    return result1, result2


# =============================================================================
# Profile Targets: Arena Initialization
# =============================================================================


def profile_arena_init():
    """Profile Arena initialization time."""
    try:
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol
        from aragora.core import Environment
    except ImportError as e:
        print(f"Could not import Arena: {e}")
        return None

    agents = [MockAgent(name=f"agent-{i}", model_type="mock", provider="mock") for i in range(3)]

    protocol = DebateProtocol(
        rounds=3,
        consensus="majority",
        convergence_detection=False,  # Disable for faster init
    )

    env = Environment(task="Test task for profiling arena initialization")

    def init_arena():
        arena = Arena(
            environment=env,
            agents=agents,
            protocol=protocol,
            auto_create_knowledge_mound=False,
            enable_knowledge_retrieval=False,
            enable_knowledge_ingestion=False,
            enable_checkpointing=False,
            enable_performance_monitor=False,
            enable_telemetry=False,
        )
        return arena

    result = profile_function(
        init_arena,
        iterations=20,
        name="Arena Initialization (3 agents, 3 rounds)",
        track_memory=True,
    )
    print(result)
    return result


# =============================================================================
# Profile Targets: Vote Collection Simulation
# =============================================================================


def profile_vote_counting():
    """Profile vote counting and aggregation."""
    from collections import Counter
    import random

    # Simulate votes from 10 agents
    agents = [f"agent-{i}" for i in range(10)]
    proposals = [f"proposal-{i}" for i in range(3)]

    def generate_votes():
        return [
            {
                "agent": agent,
                "choice": random.choice(proposals),
                "confidence": random.uniform(0.5, 1.0),
            }
            for agent in agents
        ]

    votes = generate_votes()

    # Simulate weighted vote counting
    weights = {agent: random.uniform(0.8, 1.2) for agent in agents}

    def count_weighted_votes():
        vote_counts: dict[str, float] = {}
        total_weighted = 0.0

        for v in votes:
            choice = v["choice"]
            weight = weights.get(v["agent"], 1.0)
            vote_counts[choice] = vote_counts.get(choice, 0.0) + weight
            total_weighted += weight

        # Determine winner
        if vote_counts:
            winner = max(vote_counts.items(), key=lambda x: x[1])
            confidence = winner[1] / total_weighted if total_weighted > 0 else 0
            return winner[0], confidence
        return None, 0

    result = profile_function(
        count_weighted_votes, iterations=10000, name="Weighted Vote Counting (10 agents)"
    )
    print(result)
    return result


# =============================================================================
# Profile with cProfile
# =============================================================================


def run_with_cprofile(func: Callable, name: str):
    """Run a function with cProfile and print stats."""
    profiler = cProfile.Profile()
    profiler.enable()

    func()

    profiler.disable()

    # Print stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(30)

    print(f"\n=== cProfile Stats for {name} ===")
    print(stream.getvalue())


# =============================================================================
# Memory Profiling
# =============================================================================


def run_memory_profile():
    """Run memory profiling for debate operations."""
    print("=" * 60)
    print("MEMORY PROFILING")
    print("=" * 60)
    print()

    tracemalloc.start()

    # Profile Arena initialization
    try:
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol
        from aragora.core import Environment

        snapshot1 = tracemalloc.take_snapshot()

        agents = [
            MockAgent(name=f"agent-{i}", model_type="mock", provider="mock") for i in range(5)
        ]
        protocol = DebateProtocol(rounds=5, consensus="majority")
        env = Environment(task="Memory profiling task")

        arena = Arena(
            environment=env,
            agents=agents,
            protocol=protocol,
            auto_create_knowledge_mound=False,
            enable_knowledge_retrieval=False,
            enable_knowledge_ingestion=False,
            enable_checkpointing=False,
        )

        snapshot2 = tracemalloc.take_snapshot()

        print("Arena Memory Allocation (top 10):")
        print("-" * 50)
        top_stats = snapshot2.compare_to(snapshot1, "lineno")
        for stat in top_stats[:10]:
            print(stat)
        print()

        # Cleanup
        del arena
        gc.collect()

    except ImportError as e:
        print(f"Could not import for memory profiling: {e}")

    tracemalloc.stop()


# =============================================================================
# Main Runner
# =============================================================================


def run_all_profiles(use_cprofile: bool = False, track_memory: bool = False):
    """Run all profiling benchmarks."""
    print("=" * 70)
    print("ARAGORA DEBATE ENGINE PERFORMANCE PROFILING")
    print("=" * 70)
    print()

    results: dict[str, Any] = {}

    # 1. Convergence Detection
    print("1. CONVERGENCE DETECTION")
    print("-" * 50)

    results["jaccard"] = profile_convergence_jaccard()
    print()

    results["tfidf"] = profile_convergence_tfidf()
    print()

    results["transformer"] = profile_convergence_transformer()
    print()

    results["batch_convergence"] = profile_convergence_batch()
    print()

    # 2. Weight Calculation
    print("2. VOTE WEIGHT CALCULATION")
    print("-" * 50)

    results["weights"] = profile_weight_calculation()
    print()

    # 3. Memory Operations
    print("3. MEMORY OPERATIONS")
    print("-" * 50)

    results["critique_store"] = profile_critique_store()
    print()

    results["embedding_cache"] = profile_embedding_cache()
    print()

    # 4. Vote Counting
    print("4. VOTE PROCESSING")
    print("-" * 50)

    results["vote_counting"] = profile_vote_counting()
    print()

    # 5. Arena Initialization
    print("5. ARENA INITIALIZATION")
    print("-" * 50)

    results["arena_init"] = profile_arena_init()
    print()

    # Summary
    print_summary(results)

    if track_memory:
        run_memory_profile()

    return results


def print_summary(results: dict[str, Any]):
    """Print a summary table of results."""
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Operation':<50} {'Avg (ms)':>10} {'Max (ms)':>10}")
    print("-" * 70)

    flat_results = []
    for key, value in results.items():
        if value is None:
            continue
        if isinstance(value, tuple):
            flat_results.extend([v for v in value if v is not None])
        elif isinstance(value, (ProfileResult, AsyncProfileResult)):
            flat_results.append(value)

    # Sort by avg time descending
    flat_results.sort(key=lambda x: x.avg_time, reverse=True)

    for r in flat_results:
        print(f"{r.name[:48]:<50} {r.avg_time * 1000:>10.3f} {r.max_time * 1000:>10.3f}")

    print()
    print("=" * 70)
    print("PERFORMANCE RECOMMENDATIONS")
    print("=" * 70)
    print()

    # Identify bottlenecks
    if flat_results:
        slowest = flat_results[0]
        print(f"1. HIGHEST LATENCY: '{slowest.name}'")
        print(f"   Avg: {slowest.avg_time * 1000:.2f}ms, Max: {slowest.max_time * 1000:.2f}ms")
        print()

        if len(flat_results) > 1:
            second = flat_results[1]
            print(f"2. SECOND HIGHEST: '{second.name}'")
            print(f"   Avg: {second.avg_time * 1000:.2f}ms, Max: {second.max_time * 1000:.2f}ms")
            print()

    # General recommendations
    print("GENERAL RECOMMENDATIONS:")
    print("-" * 50)
    print("""
1. EMBEDDING CACHING: Ensure EmbeddingCache is used for repeated texts.
   The debate engine already uses scoped caches per debate_id.

2. BATCH OPERATIONS: Use batch similarity methods when available.
   SentenceTransformerBackend.compute_pairwise_similarities() batches GPU calls.

3. CONVERGENCE BACKEND: Use TF-IDF for faster convergence (80% accuracy)
   or SentenceTransformer for accuracy-critical scenarios.

4. VOTE WEIGHT CACHING: Cache computed weights within a consensus phase.
   WeightCalculator.compute_weights() is called per consensus round.

5. DATABASE POOLING: Use connection pooling for CritiqueStore and ELO lookups.
   DebateLoaders provides DataLoader pattern for batching.

6. ASYNC PARALLELIZATION: Agent calls are already parallelized with semaphores.
   Consider increasing MAX_CONCURRENT_PROPOSALS if API limits allow.

7. LAZY INITIALIZATION: Phases are initialized on first use. Consider
   disabling unused features (calibration, position_ledger) for benchmarks.

8. PROFILING IN PRODUCTION: Use DebatePerformanceMonitor for round-level
   tracking in production debates.
""")


def main():
    parser = argparse.ArgumentParser(
        description="Profile the Aragora debate engine for performance bottlenecks."
    )
    parser.add_argument(
        "--cprofile",
        action="store_true",
        help="Include cProfile output for detailed function-level profiling",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Include memory profiling with tracemalloc",
    )
    parser.add_argument(
        "--component",
        type=str,
        choices=["convergence", "weights", "memory", "votes", "arena", "all"],
        default="all",
        help="Profile specific component only",
    )
    parser.add_argument(
        "--flamegraph",
        action="store_true",
        help="Generate flamegraph (requires py-spy: pip install py-spy)",
    )

    args = parser.parse_args()

    if args.flamegraph:
        print("Flamegraph generation requires running:")
        print("  py-spy record -o profile.svg -- python scripts/profile_debate.py")
        print()
        print("Running standard profiling instead...")
        print()

    if args.component == "all":
        run_all_profiles(use_cprofile=args.cprofile, track_memory=args.memory)
    elif args.component == "convergence":
        print("Profiling Convergence Detection...")
        profile_convergence_jaccard()
        profile_convergence_tfidf()
        profile_convergence_transformer()
        profile_convergence_batch()
    elif args.component == "weights":
        print("Profiling Weight Calculation...")
        profile_weight_calculation()
    elif args.component == "memory":
        print("Profiling Memory Operations...")
        profile_critique_store()
        profile_embedding_cache()
    elif args.component == "votes":
        print("Profiling Vote Processing...")
        profile_vote_counting()
    elif args.component == "arena":
        print("Profiling Arena Initialization...")
        profile_arena_init()


if __name__ == "__main__":
    main()
