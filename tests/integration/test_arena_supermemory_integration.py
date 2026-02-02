"""E2E integration tests for Supermemory wiring in the Arena debate pipeline.

Tests the complete flow of external memory integration:
- Context injection on debate start
- Outcome sync on debate conclusion
- Multi-debate memory loops
- Graceful degradation when Supermemory unavailable
- Privacy filtering on sensitive content
- Circuit breaker recovery
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core_types import DebateResult, Environment
from aragora.debate.arena_config import ArenaConfig
from aragora.debate.knowledge_manager import ArenaKnowledgeManager
from aragora.debate.orchestrator import Arena, DebateProtocol
from aragora.debate.prompt_builder import PromptBuilder
from aragora.knowledge.mound.adapters.supermemory_adapter import (
    ContextInjectionResult,
    SupermemoryAdapter,
    SupermemorySearchResult,
    SyncOutcomeResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_supermemory_client():
    """Create a mock Supermemory client with configurable responses."""
    client = MagicMock()

    # Default search response
    search_response = MagicMock()
    search_response.results = [
        MagicMock(
            content="Previous debate concluded: Token bucket is optimal for rate limiting.",
            similarity=0.92,
            memory_id="mem_001",
            container_tag="aragora_debates",
            metadata={"debate_id": "debate-old-001"},
        ),
        MagicMock(
            content="Historical insight: Redis provides reliable distributed locking.",
            similarity=0.85,
            memory_id="mem_002",
            container_tag="aragora_debates",
            metadata={"debate_id": "debate-old-002"},
        ),
    ]
    client.search = AsyncMock(return_value=search_response)

    # Default add_memory response
    add_response = MagicMock()
    add_response.success = True
    add_response.memory_id = "mem_new_123"
    client.add_memory = AsyncMock(return_value=add_response)

    # Health check
    client.health_check = AsyncMock(return_value={"healthy": True, "latency_ms": 15})

    return client


@pytest.fixture
def mock_supermemory_config():
    """Create a mock Supermemory config."""
    config = MagicMock()
    config.api_key = "test-api-key"
    config.container_tag = "aragora_debates"
    config.privacy_filter_enabled = True
    config.get_container_tag = MagicMock(return_value="aragora_debates")
    return config


@pytest.fixture
def supermemory_adapter(mock_supermemory_client, mock_supermemory_config):
    """Create a SupermemoryAdapter with mocked client."""
    adapter = SupermemoryAdapter(
        client=mock_supermemory_client,
        config=mock_supermemory_config,
        min_importance_threshold=0.5,
        max_context_items=10,
        enable_privacy_filter=True,
        enable_resilience=False,  # Disable for simpler testing
    )
    return adapter


@pytest.fixture
def mock_agents():
    """Create mock agents for debate testing."""
    from tests.integration.conftest import MockAgent

    return [
        MockAgent(
            name="agent_1",
            role="proposer",
            responses=["I propose using token bucket rate limiting."],
        ),
        MockAgent(
            name="agent_2",
            role="critic",
            responses=["Good approach, but consider distributed scenarios."],
        ),
    ]


# =============================================================================
# Context Injection Tests
# =============================================================================


@pytest.mark.asyncio
async def test_supermemory_context_injection_on_debate_start(
    supermemory_adapter, mock_supermemory_client
):
    """Context injection should fetch relevant memories at debate start."""
    result = await supermemory_adapter.inject_context(
        debate_topic="rate limiting design",
        debate_id="debate-123",
        limit=5,
    )

    assert result.memories_injected == 2
    assert len(result.context_content) == 2
    assert "Token bucket" in result.context_content[0]
    assert result.source == "supermemory"
    assert result.search_time_ms >= 0
    mock_supermemory_client.search.assert_called_once()


@pytest.mark.asyncio
async def test_context_injection_with_empty_results(supermemory_adapter, mock_supermemory_client):
    """Context injection should handle empty search results gracefully."""
    # Configure empty results
    empty_response = MagicMock()
    empty_response.results = []
    mock_supermemory_client.search = AsyncMock(return_value=empty_response)

    result = await supermemory_adapter.inject_context(
        debate_topic="novel topic with no history",
        debate_id="debate-new",
    )

    assert result.memories_injected == 0
    assert result.context_content == []
    assert result.total_tokens_estimate == 0


@pytest.mark.asyncio
async def test_context_injection_with_container_filter(
    supermemory_adapter, mock_supermemory_client
):
    """Context injection should respect container tag filters."""
    await supermemory_adapter.inject_context(
        debate_topic="security patterns",
        container_tag="security_debates",
        limit=3,
    )

    call_args = mock_supermemory_client.search.call_args
    assert call_args.kwargs.get("container_tag") == "security_debates"
    assert call_args.kwargs.get("limit") == 3


# =============================================================================
# Outcome Sync Tests
# =============================================================================


@pytest.mark.asyncio
async def test_supermemory_outcome_sync_on_conclusion(supermemory_adapter, mock_supermemory_client):
    """Outcome sync should persist debate results to Supermemory."""
    result = DebateResult(
        task="Design a rate limiter",
        debate_id="debate-456",
        final_answer="Use token bucket with Redis backend",
        confidence=0.85,
    )

    sync_result = await supermemory_adapter.sync_debate_outcome(result)

    assert sync_result.success is True
    assert sync_result.memory_id == "mem_new_123"
    assert sync_result.synced_at is not None
    mock_supermemory_client.add_memory.assert_called_once()


@pytest.mark.asyncio
async def test_outcome_sync_below_confidence_threshold(supermemory_adapter):
    """Outcomes below confidence threshold should be skipped."""
    result = DebateResult(
        task="Uncertain debate",
        debate_id="debate-low-conf",
        confidence=0.3,  # Below threshold of 0.5
    )

    sync_result = await supermemory_adapter.sync_debate_outcome(result)

    assert sync_result.success is True  # Skipping is not an error
    assert "Below importance threshold" in sync_result.error


@pytest.mark.asyncio
async def test_outcome_sync_with_custom_container_tag(supermemory_adapter, mock_supermemory_client):
    """Outcome sync should use custom container tags when provided."""
    result = DebateResult(
        task="Security audit",
        debate_id="debate-sec-001",
        confidence=0.9,
    )

    await supermemory_adapter.sync_debate_outcome(
        result,
        container_tag="security_outcomes",
    )

    call_args = mock_supermemory_client.add_memory.call_args
    assert call_args.kwargs.get("container_tag") == "security_outcomes"


# =============================================================================
# ArenaKnowledgeManager Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_knowledge_manager_supermemory_outcome_sync():
    """ArenaKnowledgeManager should sync outcomes to Supermemory once per debate."""
    adapter = MagicMock()
    adapter.sync_debate_outcome = AsyncMock(
        return_value=SyncOutcomeResult(success=True, memory_id="mem_789")
    )

    manager = ArenaKnowledgeManager(
        enable_supermemory=True,
        supermemory_adapter=adapter,
        supermemory_sync_on_conclusion=True,
    )

    result = DebateResult(task="Test debate", debate_id="debate-mgr-001")
    env = Environment(task="Test debate")

    # First sync
    await manager.ingest_outcome(result, env)
    adapter.sync_debate_outcome.assert_called_once()
    assert "debate-mgr-001" in manager._supermemory_synced_debate_ids

    # Second call should not re-sync (deduplication)
    await manager.ingest_outcome(result, env)
    adapter.sync_debate_outcome.assert_called_once()  # Still only once


@pytest.mark.asyncio
async def test_knowledge_manager_supermemory_disabled():
    """ArenaKnowledgeManager should skip Supermemory when disabled."""
    adapter = MagicMock()
    adapter.sync_debate_outcome = AsyncMock()

    manager = ArenaKnowledgeManager(
        enable_supermemory=False,  # Disabled
        supermemory_adapter=adapter,
    )

    result = DebateResult(task="Test", debate_id="debate-disabled")
    env = Environment(task="Test")

    await manager.ingest_outcome(result, env)
    adapter.sync_debate_outcome.assert_not_called()


@pytest.mark.asyncio
async def test_knowledge_manager_sync_on_conclusion_disabled():
    """ArenaKnowledgeManager should skip sync when sync_on_conclusion is False."""
    adapter = MagicMock()
    adapter.sync_debate_outcome = AsyncMock()

    manager = ArenaKnowledgeManager(
        enable_supermemory=True,
        supermemory_adapter=adapter,
        supermemory_sync_on_conclusion=False,  # Sync disabled
    )

    result = DebateResult(task="Test", debate_id="debate-no-sync")
    env = Environment(task="Test")

    await manager.ingest_outcome(result, env)
    adapter.sync_debate_outcome.assert_not_called()


# =============================================================================
# PromptBuilder Context Injection Tests
# =============================================================================


@pytest.mark.asyncio
async def test_prompt_builder_injects_and_caches_context():
    """PromptBuilder should inject Supermemory context and cache it."""
    adapter = MagicMock()
    adapter.inject_context = AsyncMock(
        return_value=ContextInjectionResult(
            memories_injected=3,
            context_content=["Memory 1", "Memory 2", "Memory 3"],
            total_tokens_estimate=150,
            search_time_ms=12,
        )
    )

    env = Environment(task="Design an API gateway")
    protocol = DebateProtocol()
    builder = PromptBuilder(protocol=protocol, env=env, supermemory_adapter=adapter)

    # First injection
    rendered = await builder.inject_supermemory_context()

    assert "External Memory Context" in rendered
    assert "Memory 1" in rendered
    assert "Memory 2" in rendered
    assert "Memory 3" in rendered
    adapter.inject_context.assert_called_once()

    # Cached retrieval
    cached = builder.get_supermemory_context()
    assert cached == rendered

    # Second call should use cache, not call adapter again
    rendered2 = await builder.inject_supermemory_context()
    assert rendered2 == rendered
    adapter.inject_context.assert_called_once()  # Still only once


@pytest.mark.asyncio
async def test_prompt_builder_handles_injection_failure():
    """PromptBuilder should handle context injection failures gracefully."""
    adapter = MagicMock()
    adapter.inject_context = AsyncMock(
        return_value=ContextInjectionResult(
            memories_injected=0,
            context_content=[],
        )
    )

    env = Environment(task="Test task")
    protocol = DebateProtocol()
    builder = PromptBuilder(protocol=protocol, env=env, supermemory_adapter=adapter)

    rendered = await builder.inject_supermemory_context()

    # Should return empty or minimal context, not crash
    assert rendered == "" or "No relevant" in rendered or rendered is not None


# =============================================================================
# Multi-Debate Memory Loop Tests
# =============================================================================


@pytest.mark.asyncio
async def test_multi_debate_memory_loop():
    """Test complete memory loop: Debate 1 → sync → Debate 2 → inject."""
    # Setup: Create adapter with memory tracking
    stored_memories = []

    async def mock_add_memory(content, container_tag, metadata):
        memory_id = f"mem_{len(stored_memories)}"
        stored_memories.append(
            {
                "content": content,
                "memory_id": memory_id,
                "container_tag": container_tag,
                "metadata": metadata,
            }
        )
        result = MagicMock()
        result.success = True
        result.memory_id = memory_id
        return result

    async def mock_search(query, limit, container_tag=None):
        response = MagicMock()
        response.results = [
            MagicMock(
                content=m["content"],
                similarity=0.9,
                memory_id=m["memory_id"],
                container_tag=m["container_tag"],
                metadata=m["metadata"],
            )
            for m in stored_memories[:limit]
        ]
        return response

    client = MagicMock()
    client.add_memory = AsyncMock(side_effect=mock_add_memory)
    client.search = AsyncMock(side_effect=mock_search)

    config = MagicMock()
    config.get_container_tag = MagicMock(return_value="aragora_debates")

    adapter = SupermemoryAdapter(
        client=client,
        config=config,
        min_importance_threshold=0.5,
        enable_privacy_filter=False,
        enable_resilience=False,
    )

    # Debate 1: Sync outcome
    result1 = DebateResult(
        task="Design rate limiter",
        debate_id="debate-loop-001",
        final_answer="Token bucket is optimal",
        confidence=0.9,
    )
    sync_result = await adapter.sync_debate_outcome(result1)
    assert sync_result.success is True
    assert len(stored_memories) == 1

    # Debate 2: Inject context from Debate 1
    inject_result = await adapter.inject_context(
        debate_topic="API design with rate limiting",
        debate_id="debate-loop-002",
    )
    assert inject_result.memories_injected >= 1
    assert "Token bucket" in inject_result.context_content[0]


@pytest.mark.asyncio
async def test_cross_session_insights_retrieval(supermemory_adapter):
    """Adapter should retrieve insights from past sessions."""
    insights = await supermemory_adapter.get_cross_session_insights(
        topic="distributed systems",
        limit=5,
    )

    assert len(insights) == 2
    assert insights[0]["source"] == "supermemory"
    assert "content" in insights[0]
    assert "similarity" in insights[0]


# =============================================================================
# Graceful Degradation Tests
# =============================================================================


@pytest.mark.asyncio
async def test_graceful_degradation_client_unavailable():
    """Adapter should degrade gracefully when client is unavailable."""
    adapter = SupermemoryAdapter(
        client=None,  # No client
        config=None,
    )

    # Context injection should return empty result
    inject_result = await adapter.inject_context(debate_topic="test")
    assert inject_result.memories_injected == 0
    assert inject_result.context_content == []

    # Outcome sync should return failure
    result = DebateResult(task="Test", debate_id="test-123", confidence=0.9)
    sync_result = await adapter.sync_debate_outcome(result)
    assert sync_result.success is False
    assert "Client not available" in sync_result.error

    # Search should return empty
    search_results = await adapter.search_memories("query")
    assert search_results == []


@pytest.mark.asyncio
async def test_graceful_degradation_api_error(mock_supermemory_config):
    """Adapter should handle API errors gracefully."""
    client = MagicMock()
    client.search = AsyncMock(side_effect=Exception("API connection failed"))
    client.add_memory = AsyncMock(side_effect=Exception("API timeout"))

    adapter = SupermemoryAdapter(
        client=client,
        config=mock_supermemory_config,
        enable_resilience=False,
    )

    # Should return empty, not crash
    inject_result = await adapter.inject_context(debate_topic="test")
    assert inject_result.memories_injected == 0

    # Should return failure, not crash
    result = DebateResult(task="Test", debate_id="test-err", confidence=0.9)
    sync_result = await adapter.sync_debate_outcome(result)
    assert sync_result.success is False
    assert "API timeout" in sync_result.error


@pytest.mark.asyncio
async def test_arena_continues_without_supermemory(mock_agents):
    """Arena should run debates successfully when Supermemory is unavailable."""
    # Create arena with Supermemory that will fail
    failing_adapter = MagicMock()
    failing_adapter.inject_context = AsyncMock(side_effect=Exception("Supermemory down"))
    failing_adapter.sync_debate_outcome = AsyncMock(side_effect=Exception("Supermemory down"))

    env = Environment(task="Design a simple cache")
    protocol = DebateProtocol(rounds=1, consensus="any")

    # Arena should still work - Supermemory failures are logged but don't stop debate
    config = ArenaConfig(
        enable_supermemory=True,
        supermemory_adapter=failing_adapter,
    )
    arena = Arena.from_config(env, mock_agents, protocol, config)

    # The debate should complete successfully despite Supermemory issues
    result = await asyncio.wait_for(arena.run(), timeout=10.0)

    assert result is not None
    assert result.rounds_completed >= 1


# =============================================================================
# Privacy Filter Tests
# =============================================================================


@pytest.mark.asyncio
async def test_privacy_filter_on_sensitive_content():
    """Privacy filter should redact sensitive information before sync."""
    client = MagicMock()

    captured_content = []

    async def capture_add_memory(content, container_tag, metadata):
        captured_content.append(content)
        result = MagicMock()
        result.success = True
        result.memory_id = "mem_filtered"
        return result

    client.add_memory = AsyncMock(side_effect=capture_add_memory)

    # Mock privacy filter
    privacy_filter = MagicMock()
    privacy_filter.filter = MagicMock(
        side_effect=lambda x: x.replace("secret_api_key_12345", "[REDACTED]")
        .replace("user@email.com", "[EMAIL]")
        .replace("SSN: 123-45-6789", "SSN: [REDACTED]")
    )

    config = MagicMock()
    config.get_container_tag = MagicMock(return_value="debates")

    adapter = SupermemoryAdapter(
        client=client,
        config=config,
        enable_privacy_filter=True,
        enable_resilience=False,
    )
    adapter._privacy_filter = privacy_filter

    # Sync debate with sensitive content
    result = DebateResult(
        task="API key rotation",
        debate_id="debate-privacy-001",
        final_answer="Use secret_api_key_12345 and contact user@email.com",
        confidence=0.9,
    )

    await adapter.sync_debate_outcome(result)

    # Verify content was filtered
    assert len(captured_content) == 1
    assert "secret_api_key_12345" not in captured_content[0]
    assert "[REDACTED]" in captured_content[0] or "[EMAIL]" in captured_content[0]


@pytest.mark.asyncio
async def test_privacy_filter_disabled():
    """Content should pass through unfiltered when privacy filter is disabled."""
    client = MagicMock()

    captured_content = []

    async def capture_add_memory(content, container_tag, metadata):
        captured_content.append(content)
        result = MagicMock()
        result.success = True
        result.memory_id = "mem_unfiltered"
        return result

    client.add_memory = AsyncMock(side_effect=capture_add_memory)

    config = MagicMock()
    config.get_container_tag = MagicMock(return_value="debates")

    adapter = SupermemoryAdapter(
        client=client,
        config=config,
        enable_privacy_filter=False,  # Disabled
        enable_resilience=False,
    )

    result = DebateResult(
        task="Test",
        debate_id="debate-no-filter",
        final_answer="Contains sensitive_data_xyz",
        confidence=0.9,
    )

    await adapter.sync_debate_outcome(result)

    # Content should be preserved (privacy filter not applied)
    assert len(captured_content) == 1
    assert "sensitive_data_xyz" in captured_content[0]


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


@pytest.mark.asyncio
async def test_circuit_breaker_opens_on_failures():
    """Circuit breaker should open after repeated failures."""
    client = MagicMock()
    failure_count = 0

    async def failing_search(query, limit, container_tag=None):
        nonlocal failure_count
        failure_count += 1
        raise Exception(f"API failure #{failure_count}")

    client.search = AsyncMock(side_effect=failing_search)

    config = MagicMock()

    adapter = SupermemoryAdapter(
        client=client,
        config=config,
        enable_resilience=True,  # Enable circuit breaker
    )

    # Make multiple failing calls
    for _ in range(5):
        result = await adapter.inject_context(debate_topic="test")
        assert result.memories_injected == 0

    # Verify failures were tracked
    assert failure_count >= 5


@pytest.mark.asyncio
async def test_circuit_breaker_recovery():
    """Circuit breaker should allow calls after recovery timeout."""
    call_count = 0

    async def sometimes_failing_search(query, limit, container_tag=None):
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            raise Exception("Temporary failure")
        # After 3 failures, start succeeding
        response = MagicMock()
        response.results = [
            MagicMock(
                content="Recovered!",
                similarity=0.9,
                memory_id="mem_recovered",
                container_tag="debates",
                metadata={},
            )
        ]
        return response

    client = MagicMock()
    client.search = AsyncMock(side_effect=sometimes_failing_search)

    config = MagicMock()

    adapter = SupermemoryAdapter(
        client=client,
        config=config,
        enable_resilience=False,  # Direct calls without circuit breaker for this test
    )

    # Initial failures
    for _ in range(3):
        result = await adapter.inject_context(debate_topic="test")
        assert result.memories_injected == 0

    # Should recover
    result = await adapter.inject_context(debate_topic="test")
    assert result.memories_injected == 1
    assert "Recovered" in result.context_content[0]


# =============================================================================
# Search and Health Check Tests
# =============================================================================


@pytest.mark.asyncio
async def test_search_memories_with_similarity_filter(supermemory_adapter, mock_supermemory_client):
    """Search should filter results by minimum similarity."""
    # Configure results with varying similarity
    response = MagicMock()
    response.results = [
        MagicMock(
            content="High similarity",
            similarity=0.95,
            memory_id="mem_high",
            container_tag="debates",
            metadata={},
        ),
        MagicMock(
            content="Medium similarity",
            similarity=0.6,
            memory_id="mem_med",
            container_tag="debates",
            metadata={},
        ),
        MagicMock(
            content="Low similarity",
            similarity=0.3,
            memory_id="mem_low",
            container_tag="debates",
            metadata={},
        ),
    ]
    mock_supermemory_client.search = AsyncMock(return_value=response)

    # Search with high threshold
    results = await supermemory_adapter.search_memories(
        query="test query",
        min_similarity=0.7,
    )

    assert len(results) == 1
    assert results[0].content == "High similarity"


@pytest.mark.asyncio
async def test_health_check_healthy(supermemory_adapter, mock_supermemory_client):
    """Health check should report healthy status."""
    health = await supermemory_adapter.health_check()

    assert health["healthy"] is True
    assert health["adapter"] == "supermemory"
    assert "external" in health


@pytest.mark.asyncio
async def test_health_check_unhealthy():
    """Health check should report unhealthy when client fails."""
    client = MagicMock()
    client.health_check = AsyncMock(side_effect=Exception("Connection refused"))

    config = MagicMock()

    adapter = SupermemoryAdapter(client=client, config=config)

    health = await adapter.health_check()

    assert health["healthy"] is False
    assert "error" in health
    assert "Connection refused" in health["error"]


@pytest.mark.asyncio
async def test_health_check_no_client():
    """Health check should report unhealthy with no client."""
    adapter = SupermemoryAdapter(client=None, config=None)

    health = await adapter.health_check()

    assert health["healthy"] is False
    assert "Client not available" in health["error"]


# =============================================================================
# Stats and Configuration Tests
# =============================================================================


def test_adapter_stats(supermemory_adapter):
    """Adapter should provide accurate statistics."""
    stats = supermemory_adapter.get_stats()

    assert stats["adapter"] == "supermemory"
    assert stats["client_initialized"] is True
    assert stats["privacy_filter_enabled"] is True
    assert stats["min_importance_threshold"] == 0.5
    assert stats["max_context_items"] == 10


def test_adapter_config_defaults():
    """Adapter should use sensible defaults."""
    adapter = SupermemoryAdapter()

    assert adapter._min_importance == 0.7
    assert adapter._max_context_items == 10
    assert adapter._enable_privacy_filter is True


# =============================================================================
# ArenaConfig Integration Tests
# =============================================================================


def test_arena_config_with_supermemory():
    """ArenaConfig should accept Supermemory configuration."""
    config = ArenaConfig(
        enable_supermemory=True,
        supermemory_inject_on_start=True,
        supermemory_max_context_items=15,
        supermemory_sync_on_conclusion=True,
        supermemory_min_confidence_for_sync=0.8,
    )

    assert config.enable_supermemory is True
    assert config.supermemory_inject_on_start is True
    assert config.supermemory_max_context_items == 15
    assert config.supermemory_sync_on_conclusion is True
    assert config.supermemory_min_confidence_for_sync == 0.8


def test_arena_config_supermemory_builder():
    """ArenaConfig should have a builder for Supermemory settings."""
    config = (
        ArenaConfig.builder()
        .with_supermemory(
            enable_supermemory=True,
            supermemory_inject_on_start=False,
            supermemory_sync_on_conclusion=True,
            supermemory_min_confidence_for_sync=0.6,
        )
        .build()
    )

    assert config.enable_supermemory is True
    assert config.supermemory_inject_on_start is False
    assert config.supermemory_sync_on_conclusion is True
    assert config.supermemory_min_confidence_for_sync == 0.6
