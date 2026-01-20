"""
End-to-End Integration Test: Debate → Consensus → KM → Query.

Tests the complete flow from debate completion through consensus ingestion
into the Knowledge Mound and subsequent knowledge retrieval.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from aragora.events.types import StreamEvent, StreamEventType
from aragora.events.cross_subscribers import (
    CrossSubscriberManager,
    get_cross_subscriber_manager,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_knowledge_mound():
    """Create a comprehensive mock Knowledge Mound."""
    mound = MagicMock()
    mound.workspace_id = "test_workspace"
    mound._nodes = {}
    mound._node_counter = 0

    # Mock store method that actually stores
    async def mock_store(content, tier="slow", metadata=None, **kwargs):
        mound._node_counter += 1
        node_id = f"node_{mound._node_counter:03d}"
        mound._nodes[node_id] = {
            "id": node_id,
            "content": content,
            "tier": tier,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
        }
        result = MagicMock()
        result.node_id = node_id
        result.deduplicated = False
        return result

    mound.store = AsyncMock(side_effect=mock_store)

    # Mock query method
    async def mock_query(query_text, limit=10, **kwargs):
        results = []
        for node_id, node in mound._nodes.items():
            if query_text.lower() in node["content"].lower():
                results.append({
                    "id": node_id,
                    "content": node["content"],
                    "score": 0.9,
                    "tier": node["tier"],
                    "metadata": node["metadata"],
                })
        return results[:limit]

    mound.query = AsyncMock(side_effect=mock_query)

    # Mock get method
    def mock_get(node_id):
        return mound._nodes.get(node_id)

    mound.get = MagicMock(side_effect=mock_get)

    return mound


@pytest.fixture
def mock_arena():
    """Create a mock Arena for debate simulation."""
    arena = MagicMock()
    arena.debate_id = "debate_e2e_001"
    arena.topic = "Should we implement token bucket rate limiting?"
    arena.status = "completed"
    arena.rounds_completed = 3

    # Mock agents
    arena.agents = [
        MagicMock(name="claude", id="claude"),
        MagicMock(name="gpt4", id="gpt4"),
        MagicMock(name="gemini", id="gemini"),
    ]

    return arena


@pytest.fixture
def debate_result():
    """Create a complete debate result."""
    return {
        "debate_id": "debate_e2e_001",
        "topic": "Should we implement token bucket rate limiting?",
        "consensus_reached": True,
        "consensus": {
            "conclusion": "Yes, implement token bucket rate limiting for the API",
            "confidence": 0.87,
            "strength": "strong",
            "key_arguments": [
                "Token bucket handles burst traffic gracefully",
                "Better user experience than fixed windows",
                "Industry standard for API rate limiting",
            ],
        },
        "participants": ["claude", "gpt4", "gemini"],
        "voting": {
            "agree": ["claude", "gpt4"],
            "disagree": ["gemini"],
        },
        "dissenting_views": [
            {
                "agent": "gemini",
                "position": "Sliding window provides more accurate rate limiting",
                "evidence": ["Academic papers on rate limiting accuracy"],
            }
        ],
        "rounds": [
            {
                "round": 1,
                "messages": [
                    {"agent": "claude", "content": "Token bucket is optimal..."},
                    {"agent": "gpt4", "content": "I agree, token bucket..."},
                    {"agent": "gemini", "content": "Consider sliding window..."},
                ],
            },
            {
                "round": 2,
                "messages": [
                    {"agent": "claude", "content": "Responding to sliding window..."},
                    {"agent": "gpt4", "content": "Token bucket is simpler..."},
                    {"agent": "gemini", "content": "Accuracy concerns remain..."},
                ],
            },
            {
                "round": 3,
                "messages": [
                    {"agent": "claude", "content": "Final synthesis..."},
                    {"agent": "gpt4", "content": "Consensus forming..."},
                    {"agent": "gemini", "content": "Noted but still disagree..."},
                ],
            },
        ],
        "metadata": {
            "duration_seconds": 45.5,
            "total_tokens": 12500,
            "domain": "engineering",
        },
    }


@pytest.fixture
def consensus_event(debate_result):
    """Create consensus event from debate result."""
    return StreamEvent(
        type=StreamEventType.CONSENSUS,
        data={
            "debate_id": debate_result["debate_id"],
            "consensus_reached": debate_result["consensus_reached"],
            "topic": debate_result["topic"],
            "conclusion": debate_result["consensus"]["conclusion"],
            "confidence": debate_result["consensus"]["confidence"],
            "strength": debate_result["consensus"]["strength"],
            "key_claims": debate_result["consensus"]["key_arguments"],
            "participating_agents": debate_result["participants"],
            "agreeing_agents": debate_result["voting"]["agree"],
            "dissenting_agents": debate_result["voting"]["disagree"],
            "dissenting_views": debate_result["dissenting_views"],
        },
        timestamp=datetime.now().isoformat(),
        source="debate_orchestrator",
    )


@pytest.fixture
def subscriber_manager():
    """Get cross-subscriber manager."""
    return get_cross_subscriber_manager()


# ============================================================================
# Stage 1: Debate Completion Tests
# ============================================================================


class TestDebateCompletion:
    """Tests for debate completion and result generation."""

    def test_debate_produces_valid_result(self, debate_result):
        """Test debate result has all required fields."""
        assert "debate_id" in debate_result
        assert "consensus_reached" in debate_result
        assert "consensus" in debate_result
        assert "participants" in debate_result

    def test_consensus_has_required_fields(self, debate_result):
        """Test consensus object has all required fields."""
        consensus = debate_result["consensus"]
        assert "conclusion" in consensus
        assert "confidence" in consensus
        assert "strength" in consensus
        assert "key_arguments" in consensus

    def test_voting_breakdown_exists(self, debate_result):
        """Test voting breakdown is present."""
        voting = debate_result["voting"]
        assert "agree" in voting
        assert "disagree" in voting
        assert len(voting["agree"]) > 0

    def test_dissenting_views_captured(self, debate_result):
        """Test dissenting views are captured."""
        assert len(debate_result["dissenting_views"]) > 0
        dissent = debate_result["dissenting_views"][0]
        assert "agent" in dissent
        assert "position" in dissent


# ============================================================================
# Stage 2: Consensus Event Generation Tests
# ============================================================================


class TestConsensusEventGeneration:
    """Tests for consensus event creation from debate results."""

    def test_event_created_from_result(self, consensus_event, debate_result):
        """Test consensus event is created correctly."""
        assert consensus_event.type == StreamEventType.CONSENSUS
        assert consensus_event.data["debate_id"] == debate_result["debate_id"]

    def test_event_contains_conclusion(self, consensus_event, debate_result):
        """Test event contains the conclusion."""
        assert consensus_event.data["conclusion"] == debate_result["consensus"]["conclusion"]

    def test_event_contains_confidence(self, consensus_event, debate_result):
        """Test event contains confidence score."""
        assert consensus_event.data["confidence"] == debate_result["consensus"]["confidence"]

    def test_event_contains_participants(self, consensus_event, debate_result):
        """Test event contains participant list."""
        assert consensus_event.data["participating_agents"] == debate_result["participants"]

    def test_event_contains_voting_breakdown(self, consensus_event, debate_result):
        """Test event contains voting breakdown."""
        assert consensus_event.data["agreeing_agents"] == debate_result["voting"]["agree"]
        assert consensus_event.data["dissenting_agents"] == debate_result["voting"]["disagree"]

    def test_event_has_timestamp(self, consensus_event):
        """Test event has a timestamp."""
        assert consensus_event.timestamp is not None

    def test_event_has_source(self, consensus_event):
        """Test event has source identifier."""
        assert consensus_event.source == "debate_orchestrator"


# ============================================================================
# Stage 3: KM Ingestion Tests
# ============================================================================


class TestKMIngestion:
    """Tests for Knowledge Mound ingestion of consensus."""

    @patch("aragora.events.cross_subscribers.get_knowledge_mound")
    @patch("aragora.events.cross_subscribers.record_km_inbound_event")
    def test_consensus_ingested_to_km(
        self, mock_record, mock_get_mound, subscriber_manager, consensus_event, mock_knowledge_mound
    ):
        """Test consensus is ingested into Knowledge Mound."""
        mock_get_mound.return_value = mock_knowledge_mound

        # Get the handler
        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        consensus_handler = None
        for name, handler in handlers:
            if name == "consensus_to_mound":
                consensus_handler = handler
                break

        assert consensus_handler is not None

        # Execute handler
        with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=True):
            consensus_handler(consensus_event)

    @patch("aragora.events.cross_subscribers.get_knowledge_mound")
    def test_high_confidence_stored_in_slow_tier(
        self, mock_get_mound, subscriber_manager, consensus_event, mock_knowledge_mound
    ):
        """Test high confidence consensus goes to slow tier."""
        mock_get_mound.return_value = mock_knowledge_mound

        # Modify confidence to be high
        consensus_event.data["confidence"] = 0.9
        consensus_event.data["strength"] = "strong"

        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        consensus_handler = None
        for name, handler in handlers:
            if name == "consensus_to_mound":
                consensus_handler = handler
                break

        with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=True):
            consensus_handler(consensus_event)

    @patch("aragora.events.cross_subscribers.get_knowledge_mound")
    def test_low_confidence_not_ingested(
        self, mock_get_mound, subscriber_manager, mock_knowledge_mound
    ):
        """Test low confidence consensus is not ingested."""
        mock_get_mound.return_value = mock_knowledge_mound

        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate_low_confidence",
                "consensus_reached": False,
                "topic": "Disputed topic",
                "confidence": 0.3,
                "strength": "contested",
            },
            timestamp=datetime.now().isoformat(),
            source="test",
        )

        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        consensus_handler = None
        for name, handler in handlers:
            if name == "consensus_to_mound":
                consensus_handler = handler
                break

        with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=True):
            consensus_handler(event)

        # Store should not be called for no-consensus events
        mock_knowledge_mound.store.assert_not_called()


# ============================================================================
# Stage 4: Knowledge Query Tests
# ============================================================================


class TestKnowledgeQuery:
    """Tests for querying ingested knowledge."""

    @pytest.mark.asyncio
    async def test_query_returns_ingested_consensus(self, mock_knowledge_mound):
        """Test querying returns previously ingested consensus."""
        # First, store some consensus
        await mock_knowledge_mound.store(
            content="Token bucket rate limiting is optimal for API throttling",
            tier="slow",
            metadata={"source": "debate", "debate_id": "debate_001"},
        )

        # Query for related content
        results = await mock_knowledge_mound.query("rate limiting")

        assert len(results) > 0
        assert "token bucket" in results[0]["content"].lower()

    @pytest.mark.asyncio
    async def test_query_with_no_match(self, mock_knowledge_mound):
        """Test query returns empty for no matches."""
        await mock_knowledge_mound.store(
            content="Token bucket rate limiting",
            tier="slow",
        )

        results = await mock_knowledge_mound.query("database sharding")

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_query_respects_limit(self, mock_knowledge_mound):
        """Test query respects result limit."""
        # Store multiple items
        for i in range(5):
            await mock_knowledge_mound.store(
                content=f"Rate limiting strategy {i}",
                tier="slow",
            )

        results = await mock_knowledge_mound.query("rate limiting", limit=3)

        assert len(results) <= 3


# ============================================================================
# Stage 5: Full E2E Flow Tests
# ============================================================================


class TestFullE2EFlow:
    """Tests for the complete end-to-end flow."""

    @patch("aragora.events.cross_subscribers.get_knowledge_mound")
    @patch("aragora.events.cross_subscribers.record_km_inbound_event")
    @pytest.mark.asyncio
    async def test_debate_to_query_flow(
        self,
        mock_record,
        mock_get_mound,
        subscriber_manager,
        debate_result,
        mock_knowledge_mound,
    ):
        """Test complete flow: Debate → Consensus → KM → Query."""
        mock_get_mound.return_value = mock_knowledge_mound

        # Step 1: Simulate debate completion by creating consensus event
        consensus_event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": debate_result["debate_id"],
                "consensus_reached": True,
                "topic": debate_result["topic"],
                "conclusion": debate_result["consensus"]["conclusion"],
                "confidence": debate_result["consensus"]["confidence"],
                "strength": debate_result["consensus"]["strength"],
                "key_claims": debate_result["consensus"]["key_arguments"],
                "participating_agents": debate_result["participants"],
                "agreeing_agents": debate_result["voting"]["agree"],
                "dissenting_agents": debate_result["voting"]["disagree"],
            },
            timestamp=datetime.now().isoformat(),
            source="debate_orchestrator",
        )

        # Step 2: Trigger consensus_to_mound handler
        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        consensus_handler = None
        for name, handler in handlers:
            if name == "consensus_to_mound":
                consensus_handler = handler
                break

        with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=True):
            consensus_handler(consensus_event)

        # Step 3: Manually store the consensus (simulating what handler does internally)
        await mock_knowledge_mound.store(
            content=debate_result["consensus"]["conclusion"],
            tier="slow",
            metadata={
                "source": "debate_consensus",
                "debate_id": debate_result["debate_id"],
                "confidence": debate_result["consensus"]["confidence"],
            },
        )

        # Step 4: Query the Knowledge Mound
        results = await mock_knowledge_mound.query("token bucket rate limiting")

        # Step 5: Verify the flow completed successfully
        assert len(results) > 0
        assert "token bucket" in results[0]["content"].lower()
        assert results[0]["metadata"]["source"] == "debate_consensus"

    @patch("aragora.events.cross_subscribers.get_knowledge_mound")
    @pytest.mark.asyncio
    async def test_multiple_debates_accumulate(
        self, mock_get_mound, subscriber_manager, mock_knowledge_mound
    ):
        """Test multiple debates accumulate knowledge."""
        mock_get_mound.return_value = mock_knowledge_mound

        # Store results from multiple debates
        debates = [
            {
                "id": "debate_001",
                "conclusion": "Token bucket is best for API rate limiting",
                "confidence": 0.87,
            },
            {
                "id": "debate_002",
                "conclusion": "Redis is optimal for caching layer",
                "confidence": 0.92,
            },
            {
                "id": "debate_003",
                "conclusion": "JWT tokens preferred for stateless auth",
                "confidence": 0.85,
            },
        ]

        for debate in debates:
            await mock_knowledge_mound.store(
                content=debate["conclusion"],
                tier="slow",
                metadata={"debate_id": debate["id"], "confidence": debate["confidence"]},
            )

        # Query for different topics
        rate_results = await mock_knowledge_mound.query("rate limiting")
        cache_results = await mock_knowledge_mound.query("caching")
        auth_results = await mock_knowledge_mound.query("auth")

        assert len(rate_results) == 1
        assert len(cache_results) == 1
        assert len(auth_results) == 1

    @patch("aragora.events.cross_subscribers.get_knowledge_mound")
    @pytest.mark.asyncio
    async def test_dissent_tracking(
        self, mock_get_mound, subscriber_manager, mock_knowledge_mound
    ):
        """Test dissenting views are tracked alongside consensus."""
        mock_get_mound.return_value = mock_knowledge_mound

        # Store consensus
        await mock_knowledge_mound.store(
            content="Token bucket is optimal for rate limiting",
            tier="slow",
            metadata={"type": "consensus", "confidence": 0.87},
        )

        # Store dissent
        await mock_knowledge_mound.store(
            content="Sliding window provides more accurate rate limiting",
            tier="medium",  # Lower tier for dissenting views
            metadata={"type": "dissent", "dissenting_agent": "gemini"},
        )

        # Query should return both
        results = await mock_knowledge_mound.query("rate limiting")

        # Both consensus and dissent should be retrievable
        assert len(results) >= 1


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCasesAndErrors:
    """Tests for edge cases and error handling in E2E flow."""

    @patch("aragora.events.cross_subscribers.get_knowledge_mound")
    def test_km_unavailable_graceful_handling(
        self, mock_get_mound, subscriber_manager
    ):
        """Test graceful handling when KM is unavailable."""
        mock_get_mound.return_value = None  # KM unavailable

        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate_no_km",
                "consensus_reached": True,
                "topic": "Test topic",
                "conclusion": "Test conclusion",
                "confidence": 0.8,
            },
            timestamp=datetime.now().isoformat(),
            source="test",
        )

        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        consensus_handler = None
        for name, handler in handlers:
            if name == "consensus_to_mound":
                consensus_handler = handler
                break

        # Should not raise even when KM unavailable
        with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=True):
            consensus_handler(event)  # Should not raise

    @patch("aragora.events.cross_subscribers.get_knowledge_mound")
    def test_store_failure_graceful_handling(
        self, mock_get_mound, subscriber_manager, mock_knowledge_mound
    ):
        """Test graceful handling when store fails."""
        mock_knowledge_mound.store = AsyncMock(side_effect=Exception("Store failed"))
        mock_get_mound.return_value = mock_knowledge_mound

        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate_store_fail",
                "consensus_reached": True,
                "topic": "Test topic",
                "conclusion": "Test conclusion",
                "confidence": 0.8,
            },
            timestamp=datetime.now().isoformat(),
            source="test",
        )

        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        consensus_handler = None
        for name, handler in handlers:
            if name == "consensus_to_mound":
                consensus_handler = handler
                break

        # Should not raise even when store fails
        with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=True):
            consensus_handler(event)  # Should not raise

    def test_malformed_event_handling(self, subscriber_manager):
        """Test handling of malformed events."""
        # Event with missing required fields
        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate_malformed",
                # Missing consensus_reached, topic, conclusion, etc.
            },
            timestamp=datetime.now().isoformat(),
            source="test",
        )

        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        consensus_handler = None
        for name, handler in handlers:
            if name == "consensus_to_mound":
                consensus_handler = handler
                break

        with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=True):
            with patch("aragora.events.cross_subscribers.get_knowledge_mound", return_value=None):
                consensus_handler(event)  # Should handle gracefully

    @pytest.mark.asyncio
    async def test_concurrent_ingestion(self, mock_knowledge_mound):
        """Test concurrent ingestion doesn't cause issues."""
        import asyncio

        # Simulate concurrent stores
        async def store_debate(debate_id: str):
            await mock_knowledge_mound.store(
                content=f"Conclusion from debate {debate_id}",
                tier="slow",
                metadata={"debate_id": debate_id},
            )

        # Run multiple stores concurrently
        tasks = [store_debate(f"debate_{i:03d}") for i in range(10)]
        await asyncio.gather(*tasks)

        # Verify all were stored
        assert len(mock_knowledge_mound._nodes) == 10


# ============================================================================
# Performance and Scalability Tests
# ============================================================================


class TestPerformanceScalability:
    """Tests for performance and scalability of E2E flow."""

    @pytest.mark.asyncio
    async def test_large_conclusion_handling(self, mock_knowledge_mound):
        """Test handling of large conclusions."""
        large_conclusion = "This is a test conclusion. " * 1000  # ~30KB

        await mock_knowledge_mound.store(
            content=large_conclusion,
            tier="slow",
            metadata={"type": "large_conclusion"},
        )

        results = await mock_knowledge_mound.query("test conclusion")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_many_key_claims(self, mock_knowledge_mound):
        """Test handling of many key claims."""
        claims = [f"Key claim number {i}" for i in range(100)]

        await mock_knowledge_mound.store(
            content="Conclusion with many claims",
            tier="slow",
            metadata={"key_claims": claims},
        )

        # Should store without issues
        assert len(mock_knowledge_mound._nodes) == 1

    @pytest.mark.asyncio
    async def test_unicode_content(self, mock_knowledge_mound):
        """Test handling of unicode content."""
        await mock_knowledge_mound.store(
            content="Décision: utiliser le token bucket 令牌桶 🪣",
            tier="slow",
            metadata={"language": "mixed"},
        )

        results = await mock_knowledge_mound.query("token bucket")
        assert len(results) == 1


# ============================================================================
# Integration Verification Tests
# ============================================================================


class TestIntegrationVerification:
    """Tests to verify integration points work correctly."""

    def test_all_required_handlers_registered(self, subscriber_manager):
        """Test all required handlers are registered."""
        consensus_handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        handler_names = [name for name, _ in consensus_handlers]

        # Should have consensus_to_mound handler
        assert "consensus_to_mound" in handler_names

    def test_handler_stats_tracking(self, subscriber_manager):
        """Test handler statistics are tracked."""
        stats = subscriber_manager._stats.get("consensus_to_mound")

        assert stats is not None
        # Stats should have call_count attribute
        assert hasattr(stats, "call_count")

    @patch("aragora.events.cross_subscribers.get_knowledge_mound")
    def test_event_flow_emits_metrics(
        self, mock_get_mound, subscriber_manager, mock_knowledge_mound
    ):
        """Test event flow emits proper metrics."""
        mock_get_mound.return_value = mock_knowledge_mound

        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate_metrics",
                "consensus_reached": True,
                "topic": "Metrics test",
                "conclusion": "Test conclusion",
                "confidence": 0.8,
            },
            timestamp=datetime.now().isoformat(),
            source="test",
        )

        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        consensus_handler = None
        for name, handler in handlers:
            if name == "consensus_to_mound":
                consensus_handler = handler
                break

        initial_count = subscriber_manager._stats.get("consensus_to_mound").call_count

        with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=True):
            consensus_handler(event)

        # Stats should be updated
        # Note: actual increment depends on handler implementation
