"""Memory loop tests for multi-debate Supermemory cycles.

Tests the complete memory cycle across multiple debates:
- Debate 1 concludes and syncs to Supermemory
- Debate 2 starts and injects context from Debate 1
- Debate 3 benefits from accumulated knowledge

These tests verify:
- Knowledge accumulation over time
- Proper filtering and relevance matching
- Memory persistence and retrieval consistency
- Deduplication of synced outcomes
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.core_types import DebateResult, Environment
from aragora.debate.arena_config import ArenaConfig
from aragora.debate.knowledge_manager import ArenaKnowledgeManager
from aragora.debate.orchestrator import Arena, DebateProtocol
from aragora.debate.prompt_builder import PromptBuilder
from aragora.knowledge.mound.adapters.supermemory_adapter import (
    ContextInjectionResult,
    SupermemoryAdapter,
    SyncOutcomeResult,
)


# =============================================================================
# In-Memory Supermemory Simulator
# =============================================================================


@dataclass
class SimulatedMemory:
    """Simulated memory entry for testing."""

    memory_id: str
    content: str
    container_tag: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


class InMemorySupermemorySimulator:
    """Simulates Supermemory storage for multi-debate loop testing.

    Provides realistic behavior for testing memory accumulation:
    - Semantic similarity simulation based on keyword matching
    - Container-based filtering
    - Chronological ordering
    """

    def __init__(self):
        self.memories: list[SimulatedMemory] = []
        self._next_id = 1

    async def add_memory(
        self,
        content: str,
        container_tag: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Add a memory to the simulated store."""
        memory = SimulatedMemory(
            memory_id=f"sim_mem_{self._next_id:04d}",
            content=content,
            container_tag=container_tag or "default",
            metadata=metadata or {},
        )
        self._next_id += 1
        self.memories.append(memory)

        result = MagicMock()
        result.success = True
        result.memory_id = memory.memory_id
        return result

    async def search(
        self,
        query: str,
        limit: int = 10,
        container_tag: str | None = None,
    ):
        """Search memories with simple keyword-based similarity."""
        query_keywords = set(query.lower().split())
        scored_memories = []

        for mem in self.memories:
            # Filter by container if specified
            if container_tag and mem.container_tag != container_tag:
                continue

            # Calculate simple similarity based on keyword overlap
            content_keywords = set(mem.content.lower().split())
            overlap = len(query_keywords & content_keywords)
            similarity = overlap / max(len(query_keywords), 1) if query_keywords else 0.5

            if similarity > 0.1:  # Minimum threshold
                scored_memories.append((similarity, mem))

        # Sort by similarity descending
        scored_memories.sort(key=lambda x: x[0], reverse=True)

        response = MagicMock()
        response.results = [
            MagicMock(
                content=mem.content,
                similarity=sim,
                memory_id=mem.memory_id,
                container_tag=mem.container_tag,
                metadata=mem.metadata,
            )
            for sim, mem in scored_memories[:limit]
        ]
        return response

    async def health_check(self):
        return {"healthy": True, "memory_count": len(self.memories)}


# =============================================================================
# Memory Loop Test Fixtures
# =============================================================================


@pytest.fixture
def memory_simulator():
    """Create a fresh in-memory Supermemory simulator."""
    return InMemorySupermemorySimulator()


@pytest.fixture
def simulator_adapter(memory_simulator):
    """Create an adapter with the in-memory simulator."""
    config = MagicMock()
    config.get_container_tag = MagicMock(return_value="aragora_debates")

    adapter = SupermemoryAdapter(
        client=memory_simulator,
        config=config,
        min_importance_threshold=0.5,
        enable_privacy_filter=False,
        enable_resilience=False,
    )
    return adapter


# =============================================================================
# Multi-Debate Memory Loop Tests
# =============================================================================


@pytest.mark.asyncio
async def test_three_debate_memory_accumulation(memory_simulator, simulator_adapter):
    """Test knowledge accumulation across three sequential debates."""
    # Debate 1: Establish initial knowledge
    result1 = DebateResult(
        task="Design a rate limiter",
        debate_id="debate-001",
        conclusion="Token bucket algorithm is optimal for rate limiting with burst support",
        consensus_type="majority",
        confidence=0.9,
        key_claims=["token bucket", "burst handling", "efficient"],
    )
    sync1 = await simulator_adapter.sync_debate_outcome(result1)
    assert sync1.success is True

    # Debate 2: Build on Debate 1's knowledge
    inject2 = await simulator_adapter.inject_context(
        debate_topic="API gateway design with rate limiting",
        debate_id="debate-002",
    )
    assert inject2.memories_injected >= 1
    assert "token bucket" in inject2.context_content[0].lower()

    result2 = DebateResult(
        task="API gateway design",
        debate_id="debate-002",
        conclusion="Use Kong with token bucket rate limiting at edge",
        consensus_type="unanimous",
        confidence=0.95,
        key_claims=["Kong gateway", "edge rate limiting", "token bucket"],
    )
    sync2 = await simulator_adapter.sync_debate_outcome(result2)
    assert sync2.success is True

    # Debate 3: Should see accumulated knowledge
    inject3 = await simulator_adapter.inject_context(
        debate_topic="microservices rate limiting and API gateway",
        debate_id="debate-003",
    )
    assert inject3.memories_injected == 2  # Both previous debates

    # Verify both concepts are present
    all_content = " ".join(inject3.context_content).lower()
    assert "token bucket" in all_content
    assert "kong" in all_content or "gateway" in all_content


@pytest.mark.asyncio
async def test_topic_relevance_filtering(memory_simulator, simulator_adapter):
    """Test that context injection filters by topic relevance."""
    # Store diverse memories
    memories = [
        DebateResult(
            task="Rate limiting",
            debate_id="mem-rate",
            conclusion="Token bucket for rate limiting",
            confidence=0.9,
        ),
        DebateResult(
            task="Database optimization",
            debate_id="mem-db",
            conclusion="Use PostgreSQL with proper indexing",
            confidence=0.9,
        ),
        DebateResult(
            task="Authentication",
            debate_id="mem-auth",
            conclusion="OAuth 2.0 with JWT tokens for auth",
            confidence=0.9,
        ),
    ]

    for mem in memories:
        await simulator_adapter.sync_debate_outcome(mem)

    # Search for rate limiting - should find relevant memory
    inject = await simulator_adapter.inject_context(
        debate_topic="How to implement rate limiting for APIs",
    )
    assert inject.memories_injected >= 1
    assert (
        "token" in inject.context_content[0].lower() or "rate" in inject.context_content[0].lower()
    )


@pytest.mark.asyncio
async def test_container_isolation(memory_simulator, simulator_adapter):
    """Test that different containers isolate memories."""
    # Store memories in different containers
    await memory_simulator.add_memory(
        content="Production config: Use Redis cluster",
        container_tag="production",
    )
    await memory_simulator.add_memory(
        content="Test config: Use in-memory store",
        container_tag="testing",
    )

    # Search in production container only
    inject_prod = await simulator_adapter.inject_context(
        debate_topic="configuration",
        container_tag="production",
    )

    # Search in testing container only
    inject_test = await simulator_adapter.inject_context(
        debate_topic="configuration",
        container_tag="testing",
    )

    # Verify isolation
    if inject_prod.memories_injected > 0:
        assert "Redis" in inject_prod.context_content[0]
    if inject_test.memories_injected > 0:
        assert "in-memory" in inject_test.context_content[0]


@pytest.mark.asyncio
async def test_deduplication_across_debates(memory_simulator, simulator_adapter):
    """Test that same debate outcome is not synced twice."""
    result = DebateResult(
        task="Test dedup",
        debate_id="debate-dedup-001",
        conclusion="Unique conclusion",
        confidence=0.9,
    )

    # Create manager to test deduplication
    manager = ArenaKnowledgeManager(
        enable_supermemory=True,
        supermemory_adapter=simulator_adapter,
        supermemory_sync_on_conclusion=True,
    )

    env = Environment(task="Test dedup")

    # Sync same result multiple times
    await manager.ingest_outcome(result, env)
    await manager.ingest_outcome(result, env)
    await manager.ingest_outcome(result, env)

    # Should only have one memory
    assert len(memory_simulator.memories) == 1
    assert "debate-dedup-001" in manager._supermemory_synced_debate_ids


@pytest.mark.asyncio
async def test_confidence_threshold_filtering(memory_simulator):
    """Test that low-confidence debates are not synced."""
    config = MagicMock()
    config.get_container_tag = MagicMock(return_value="debates")

    # High threshold adapter
    high_threshold_adapter = SupermemoryAdapter(
        client=memory_simulator,
        config=config,
        min_importance_threshold=0.8,  # High threshold
        enable_privacy_filter=False,
        enable_resilience=False,
    )

    # Low confidence result
    low_conf = DebateResult(
        task="Uncertain conclusion",
        debate_id="debate-low",
        conclusion="Maybe this works?",
        confidence=0.5,  # Below threshold
    )
    result_low = await high_threshold_adapter.sync_debate_outcome(low_conf)
    assert result_low.success is True
    assert "Below importance threshold" in result_low.error
    assert len(memory_simulator.memories) == 0

    # High confidence result
    high_conf = DebateResult(
        task="Certain conclusion",
        debate_id="debate-high",
        conclusion="Definitely use this approach",
        confidence=0.9,  # Above threshold
    )
    result_high = await high_threshold_adapter.sync_debate_outcome(high_conf)
    assert result_high.success is True
    assert result_high.error is None
    assert len(memory_simulator.memories) == 1


# =============================================================================
# Knowledge Manager Memory Loop Tests
# =============================================================================


@pytest.mark.asyncio
async def test_knowledge_manager_full_loop():
    """Test ArenaKnowledgeManager orchestrating complete memory loop."""
    simulator = InMemorySupermemorySimulator()

    config = MagicMock()
    config.get_container_tag = MagicMock(return_value="debates")

    adapter = SupermemoryAdapter(
        client=simulator,
        config=config,
        min_importance_threshold=0.5,
        enable_privacy_filter=False,
        enable_resilience=False,
    )

    manager = ArenaKnowledgeManager(
        enable_supermemory=True,
        supermemory_adapter=adapter,
        supermemory_sync_on_conclusion=True,
    )

    # Simulate multiple debate outcomes
    debates = [
        DebateResult(
            task="Caching strategy",
            debate_id="loop-001",
            conclusion="Use Redis for distributed caching with TTL",
            confidence=0.88,
        ),
        DebateResult(
            task="Session management",
            debate_id="loop-002",
            conclusion="Store sessions in Redis with automatic expiration",
            confidence=0.92,
        ),
        DebateResult(
            task="Rate limiting",
            debate_id="loop-003",
            conclusion="Implement sliding window rate limiting with Redis counters",
            confidence=0.85,
        ),
    ]

    for debate in debates:
        env = Environment(task=debate.task)
        await manager.ingest_outcome(debate, env)

    # Verify all debates were synced
    assert len(simulator.memories) == 3
    assert len(manager._supermemory_synced_debate_ids) == 3

    # New debate should get all relevant context
    inject_result = await adapter.inject_context(
        debate_topic="Redis architecture for distributed systems",
    )
    assert inject_result.memories_injected == 3


# =============================================================================
# PromptBuilder Memory Loop Tests
# =============================================================================


@pytest.mark.asyncio
async def test_prompt_builder_uses_injected_context():
    """Test that PromptBuilder incorporates injected context into prompts."""
    adapter = MagicMock()
    adapter.inject_context = AsyncMock(
        return_value=ContextInjectionResult(
            memories_injected=2,
            context_content=[
                "Previous insight: Token bucket handles bursts well",
                "Historical pattern: Redis is reliable for rate limits",
            ],
            total_tokens_estimate=100,
            search_time_ms=15,
        )
    )

    env = Environment(task="Design rate limiter for microservices")
    protocol = DebateProtocol()

    builder = PromptBuilder(protocol=protocol, env=env, supermemory_adapter=adapter)

    # Inject context
    context = await builder.inject_supermemory_context()

    assert "External Memory Context" in context
    assert "Token bucket" in context
    assert "Redis" in context

    # Verify context is available for prompt building
    cached = builder.get_supermemory_context()
    assert cached == context


@pytest.mark.asyncio
async def test_prompt_builder_handles_no_memories():
    """Test PromptBuilder handles case with no relevant memories."""
    adapter = MagicMock()
    adapter.inject_context = AsyncMock(
        return_value=ContextInjectionResult(
            memories_injected=0,
            context_content=[],
            total_tokens_estimate=0,
            search_time_ms=5,
        )
    )

    env = Environment(task="Novel topic with no history")
    protocol = DebateProtocol()

    builder = PromptBuilder(protocol=protocol, env=env, supermemory_adapter=adapter)

    context = await builder.inject_supermemory_context()

    # Should handle gracefully without error
    assert context == "" or context is not None


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.asyncio
async def test_memory_loop_with_intermittent_failures():
    """Test memory loop resilience with intermittent failures."""
    call_count = 0

    async def flaky_add_memory(content, container_tag, metadata):
        nonlocal call_count
        call_count += 1
        if call_count == 2:  # Fail on second call
            raise Exception("Intermittent failure")
        result = MagicMock()
        result.success = True
        result.memory_id = f"mem_{call_count}"
        return result

    async def always_search(query, limit, container_tag=None):
        response = MagicMock()
        response.results = []
        return response

    client = MagicMock()
    client.add_memory = AsyncMock(side_effect=flaky_add_memory)
    client.search = AsyncMock(side_effect=always_search)

    config = MagicMock()
    config.get_container_tag = MagicMock(return_value="debates")

    adapter = SupermemoryAdapter(
        client=client,
        config=config,
        min_importance_threshold=0.5,
        enable_resilience=False,
    )

    results = []
    for i in range(3):
        result = DebateResult(
            task=f"Task {i}",
            debate_id=f"debate-{i}",
            conclusion=f"Conclusion {i}",
            confidence=0.9,
        )
        sync_result = await adapter.sync_debate_outcome(result)
        results.append(sync_result)

    # First and third should succeed, second should fail
    assert results[0].success is True
    assert results[1].success is False
    assert results[2].success is True


@pytest.mark.asyncio
async def test_memory_loop_empty_conclusion():
    """Test handling of debates with empty conclusions."""
    simulator = InMemorySupermemorySimulator()

    config = MagicMock()
    config.get_container_tag = MagicMock(return_value="debates")

    adapter = SupermemoryAdapter(
        client=simulator,
        config=config,
        min_importance_threshold=0.5,
        enable_privacy_filter=False,
        enable_resilience=False,
    )

    # Result without conclusion attributes
    result = DebateResult(
        task="Empty debate",
        debate_id="debate-empty",
        confidence=0.9,
    )

    sync_result = await adapter.sync_debate_outcome(result)

    # Should still sync (converts result to string)
    assert sync_result.success is True
    assert len(simulator.memories) == 1


@pytest.mark.asyncio
async def test_large_memory_accumulation():
    """Test handling of large number of accumulated memories."""
    simulator = InMemorySupermemorySimulator()

    config = MagicMock()
    config.get_container_tag = MagicMock(return_value="debates")

    adapter = SupermemoryAdapter(
        client=simulator,
        config=config,
        min_importance_threshold=0.1,  # Low threshold to sync all
        max_context_items=5,  # Limit returned items
        enable_privacy_filter=False,
        enable_resilience=False,
    )

    # Add many memories
    for i in range(20):
        result = DebateResult(
            task=f"Rate limiting pattern {i}",
            debate_id=f"debate-bulk-{i:03d}",
            conclusion=f"Rate limiting approach number {i} uses token bucket",
            confidence=0.9,
        )
        await adapter.sync_debate_outcome(result)

    assert len(simulator.memories) == 20

    # Inject should respect limit
    inject = await adapter.inject_context(
        debate_topic="rate limiting",
        limit=5,
    )

    assert inject.memories_injected <= 5
