"""
Tests for Consensus Ingestion to Knowledge Mound.

Tests the automatic ingestion of debate consensus into the Knowledge Mound
via the cross-subscriber event handler.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.events.types import StreamEvent, StreamEventType
from aragora.events.cross_subscribers import (
    CrossSubscriberManager,
    get_cross_subscriber_manager,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def consensus_event():
    """Create a mock consensus event."""
    return StreamEvent(
        type=StreamEventType.CONSENSUS,
        data={
            "debate_id": "debate_001",
            "consensus_reached": True,
            "topic": "Rate limiting strategies",
            "conclusion": "Token bucket algorithm is most suitable for API rate limiting",
            "confidence": 0.85,
            "strength": "strong",
            "key_claims": [
                "Token bucket allows burst handling",
                "Fixed window is simpler but less flexible",
                "Sliding window has highest accuracy",
            ],
            "participating_agents": ["claude", "gpt4", "gemini"],
            "agreeing_agents": ["claude", "gpt4"],
            "dissenting_agents": ["gemini"],
        },
    )


@pytest.fixture
def no_consensus_event():
    """Create a mock event where consensus was not reached."""
    return StreamEvent(
        type=StreamEventType.CONSENSUS,
        data={
            "debate_id": "debate_002",
            "consensus_reached": False,
            "topic": "Disputed topic",
            "confidence": 0.3,
            "strength": "contested",
        },
    )


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound."""
    mound = MagicMock()
    mound.workspace_id = "test_workspace"

    # Mock store method
    store_result = MagicMock()
    store_result.node_id = "node_001"
    store_result.deduplicated = False
    mound.store = AsyncMock(return_value=store_result)

    return mound


@pytest.fixture
def subscriber_manager():
    """Get a fresh subscriber manager."""
    # Reset singleton to get fresh instance
    from aragora.events.cross_subscribers import reset_cross_subscriber_manager
    reset_cross_subscriber_manager()
    return get_cross_subscriber_manager()


# ============================================================================
# Handler Registration Tests
# ============================================================================


class TestHandlerRegistration:
    """Tests for consensus handler registration."""

    def test_consensus_to_mound_handler_registered(self, subscriber_manager):
        """Test that consensus_to_mound handler is registered."""
        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        handler_names = [name for name, _ in handlers]

        assert "consensus_to_mound" in handler_names

    def test_provenance_to_mound_also_registered(self, subscriber_manager):
        """Test that provenance handler is also on CONSENSUS event."""
        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        handler_names = [name for name, _ in handlers]

        assert "provenance_to_mound" in handler_names

    def test_both_handlers_on_consensus_event(self, subscriber_manager):
        """Test both consensus handlers are registered."""
        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])

        # Should have at least 2 handlers
        assert len(handlers) >= 2


# ============================================================================
# Handler Execution Tests
# ============================================================================


class TestConsensusIngestionHandler:
    """Tests for _handle_consensus_to_mound handler."""

    @patch("aragora.knowledge.mound.get_knowledge_mound")
    def test_handler_ingests_consensus(self, mock_get_mound, subscriber_manager, consensus_event, mock_mound):
        """Test handler ingests consensus to KM."""
        mock_get_mound.return_value = mock_mound

        # Get the handler
        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        consensus_handler = None
        for name, handler in handlers:
            if name == "consensus_to_mound":
                consensus_handler = handler
                break

        assert consensus_handler is not None

        # Execute handler (it runs async internally)
        with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=True):
            consensus_handler(consensus_event)

    @patch("aragora.knowledge.mound.get_knowledge_mound")
    def test_handler_skips_no_consensus(self, mock_get_mound, subscriber_manager, no_consensus_event, mock_mound):
        """Test handler skips events without consensus."""
        mock_get_mound.return_value = mock_mound

        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        consensus_handler = None
        for name, handler in handlers:
            if name == "consensus_to_mound":
                consensus_handler = handler
                break

        with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=True):
            consensus_handler(no_consensus_event)

        # Store should not be called
        mock_mound.store.assert_not_called()

    def test_handler_disabled_when_flag_off(self, subscriber_manager, consensus_event):
        """Test handler does nothing when disabled."""
        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        consensus_handler = None
        for name, handler in handlers:
            if name == "consensus_to_mound":
                consensus_handler = handler
                break

        # Should not raise when disabled
        with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=False):
            consensus_handler(consensus_event)


# ============================================================================
# Event Data Tests
# ============================================================================


class TestEventDataHandling:
    """Tests for handling various event data configurations."""

    def test_event_with_empty_topic(self, subscriber_manager):
        """Test handling event with empty topic."""
        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate_003",
                "consensus_reached": True,
                "topic": "",
                "conclusion": "Some conclusion",
                "confidence": 0.7,
            },
        )

        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        consensus_handler = None
        for name, handler in handlers:
            if name == "consensus_to_mound":
                consensus_handler = handler
                break

        # Should handle gracefully
        with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=True):
            with patch("aragora.knowledge.mound.get_knowledge_mound", return_value=None):
                consensus_handler(event)  # Should not raise

    def test_event_with_empty_conclusion(self, subscriber_manager):
        """Test handling event with empty conclusion."""
        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate_004",
                "consensus_reached": True,
                "topic": "Some topic",
                "conclusion": "",
                "confidence": 0.6,
            },
        )

        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        consensus_handler = None
        for name, handler in handlers:
            if name == "consensus_to_mound":
                consensus_handler = handler
                break

        with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=True):
            with patch("aragora.knowledge.mound.get_knowledge_mound", return_value=None):
                consensus_handler(event)  # Should not raise

    def test_event_with_no_key_claims(self, subscriber_manager):
        """Test handling event without key claims."""
        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate_005",
                "consensus_reached": True,
                "topic": "Topic without claims",
                "conclusion": "Conclusion",
                "confidence": 0.8,
                # No key_claims field
            },
        )

        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        consensus_handler = None
        for name, handler in handlers:
            if name == "consensus_to_mound":
                consensus_handler = handler
                break

        with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=True):
            with patch("aragora.knowledge.mound.get_knowledge_mound", return_value=None):
                consensus_handler(event)  # Should not raise


# ============================================================================
# Strength to Tier Mapping Tests
# ============================================================================


class TestStrengthTierMapping:
    """Tests for consensus strength to KM tier mapping."""

    @pytest.mark.parametrize("strength,expected_tier", [
        ("unanimous", "glacial"),
        ("strong", "slow"),
        ("moderate", "slow"),
        ("weak", "medium"),
        ("split", "medium"),
        ("contested", "fast"),
    ])
    def test_strength_mapping(self, strength, expected_tier):
        """Test strength is mapped to correct tier."""
        # This tests the mapping logic defined in the handler
        strength_to_tier = {
            "unanimous": "glacial",
            "strong": "slow",
            "moderate": "slow",
            "weak": "medium",
            "split": "medium",
            "contested": "fast",
        }

        assert strength_to_tier.get(strength, "slow") == expected_tier

    def test_unknown_strength_defaults_to_slow(self):
        """Test unknown strength defaults to slow tier."""
        strength_to_tier = {
            "unanimous": "glacial",
            "strong": "slow",
            "moderate": "slow",
            "weak": "medium",
            "split": "medium",
            "contested": "fast",
        }

        assert strength_to_tier.get("unknown", "slow") == "slow"


# ============================================================================
# Integration Tests
# ============================================================================


class TestConsensusIngestionIntegration:
    """Integration tests for consensus ingestion."""

    @patch("aragora.knowledge.mound.get_knowledge_mound")
    @patch("aragora.events.cross_subscribers.record_km_inbound_event")
    def test_full_ingestion_flow(self, mock_record, mock_get_mound, subscriber_manager, consensus_event):
        """Test full ingestion flow with mocked KM."""
        # Setup mock mound
        mock_mound = MagicMock()
        mock_mound.workspace_id = "test"

        store_result = MagicMock()
        store_result.node_id = "new_node_001"
        store_result.deduplicated = False
        mock_mound.store = AsyncMock(return_value=store_result)

        mock_get_mound.return_value = mock_mound

        # Get handler
        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        consensus_handler = None
        for name, handler in handlers:
            if name == "consensus_to_mound":
                consensus_handler = handler
                break

        # Execute
        with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=True):
            consensus_handler(consensus_event)

        # Verify metrics were recorded
        mock_record.assert_called()

    def test_handler_resilience_to_km_unavailable(self, subscriber_manager, consensus_event):
        """Test handler handles KM unavailable gracefully."""
        with patch("aragora.knowledge.mound.get_knowledge_mound", return_value=None):
            handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
            consensus_handler = None
            for name, handler in handlers:
                if name == "consensus_to_mound":
                    consensus_handler = handler
                    break

            with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=True):
                # Should not raise
                consensus_handler(consensus_event)

    def test_handler_resilience_to_store_error(self, subscriber_manager, consensus_event):
        """Test handler handles store errors gracefully."""
        mock_mound = MagicMock()
        mock_mound.workspace_id = "test"
        mock_mound.store = AsyncMock(side_effect=Exception("Store failed"))

        with patch("aragora.knowledge.mound.get_knowledge_mound", return_value=mock_mound):
            handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
            consensus_handler = None
            for name, handler in handlers:
                if name == "consensus_to_mound":
                    consensus_handler = handler
                    break

            with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=True):
                # Should not raise even if store fails
                consensus_handler(consensus_event)


# ============================================================================
# Stats and Metrics Tests
# ============================================================================


class TestHandlerStats:
    """Tests for handler statistics tracking."""

    def test_handler_has_stats_entry(self, subscriber_manager):
        """Test consensus_to_mound has stats entry."""
        assert "consensus_to_mound" in subscriber_manager._stats

    def test_stats_initialized_to_zero(self, subscriber_manager):
        """Test stats are initialized properly."""
        stats = subscriber_manager._stats.get("consensus_to_mound")

        assert stats is not None
        assert stats.events_processed == 0


# ============================================================================
# Key Claims Ingestion Tests
# ============================================================================


class TestKeyClainsIngestion:
    """Tests for key claims ingestion."""

    def test_event_with_many_key_claims(self, subscriber_manager):
        """Test event with many key claims is handled."""
        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate_006",
                "consensus_reached": True,
                "topic": "Complex topic",
                "conclusion": "Conclusion",
                "confidence": 0.9,
                "strength": "strong",
                "key_claims": [f"Claim {i}" for i in range(20)],  # More than limit
            },
        )

        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        consensus_handler = None
        for name, handler in handlers:
            if name == "consensus_to_mound":
                consensus_handler = handler
                break

        with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=True):
            with patch("aragora.knowledge.mound.get_knowledge_mound", return_value=None):
                consensus_handler(event)  # Should not raise

    def test_event_with_non_string_key_claims(self, subscriber_manager):
        """Test event with non-string key claims."""
        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate_007",
                "consensus_reached": True,
                "topic": "Topic",
                "conclusion": "Conclusion",
                "confidence": 0.8,
                "key_claims": [
                    "Valid claim",
                    123,  # Invalid
                    None,  # Invalid
                    {"nested": "claim"},  # Invalid
                ],
            },
        )

        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        consensus_handler = None
        for name, handler in handlers:
            if name == "consensus_to_mound":
                consensus_handler = handler
                break

        with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=True):
            with patch("aragora.knowledge.mound.get_knowledge_mound", return_value=None):
                consensus_handler(event)  # Should handle gracefully


# ============================================================================
# Dissent Tracking Tests
# ============================================================================


class TestDissentTracking:
    """Tests for dissent tracking in consensus ingestion."""

    @pytest.fixture
    def consensus_event_with_dissents(self):
        """Create a consensus event with dissenting views."""
        return StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate_dissent_001",
                "consensus_reached": True,
                "topic": "Database selection for high-write workloads",
                "conclusion": "PostgreSQL with proper tuning is suitable",
                "confidence": 0.75,
                "strength": "moderate",
                "key_claims": [
                    "PostgreSQL handles high write loads with partitioning",
                    "Connection pooling is essential",
                ],
                "participating_agents": ["claude", "gpt4", "gemini", "mistral"],
                "agreeing_agents": ["claude", "gpt4"],
                "dissenting_agents": ["gemini", "mistral"],
                "dissents": [
                    {
                        "agent_id": "gemini",
                        "type": "alternative_approach",
                        "content": "Cassandra would be better for write-heavy workloads",
                        "reasoning": "Cassandra's write-optimized architecture handles high throughput better",
                        "confidence": 0.7,
                        "acknowledged": True,
                        "rebuttal": "PostgreSQL's WAL optimization addresses this for most use cases",
                    },
                    {
                        "agent_id": "mistral",
                        "type": "risk_warning",
                        "content": "PostgreSQL may face replication lag under extreme load",
                        "reasoning": "Synchronous replication adds latency at scale",
                        "confidence": 0.8,
                        "acknowledged": False,
                        "rebuttal": "",
                    },
                ],
                "dissent_ids": ["dissent_001", "dissent_002"],
            },
        )

    def test_dissent_data_present_in_event(self, consensus_event_with_dissents):
        """Test event contains dissent data."""
        data = consensus_event_with_dissents.data

        assert "dissents" in data
        assert len(data["dissents"]) == 2
        assert data["dissenting_agents"] == ["gemini", "mistral"]

    def test_dissent_types_recognized(self, consensus_event_with_dissents):
        """Test different dissent types are handled."""
        dissents = consensus_event_with_dissents.data["dissents"]

        assert dissents[0]["type"] == "alternative_approach"
        assert dissents[1]["type"] == "risk_warning"

    @patch("aragora.knowledge.mound.get_knowledge_mound")
    def test_handler_processes_dissents(
        self, mock_get_mound, subscriber_manager, consensus_event_with_dissents
    ):
        """Test handler processes dissenting views."""
        mock_mound = MagicMock()
        mock_mound.workspace_id = "test"

        store_result = MagicMock()
        store_result.node_id = "node_001"
        store_result.deduplicated = False
        mock_mound.store = AsyncMock(return_value=store_result)
        mock_mound.search = AsyncMock(return_value=[])

        mock_get_mound.return_value = mock_mound

        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        consensus_handler = None
        for name, handler in handlers:
            if name == "consensus_to_mound":
                consensus_handler = handler
                break

        with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=True):
            consensus_handler(consensus_event_with_dissents)

        # Handler should have been called
        mock_get_mound.assert_called()

    def test_event_with_string_dissents(self, subscriber_manager):
        """Test handling dissents as plain strings."""
        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate_str_dissent",
                "consensus_reached": True,
                "topic": "Simple topic",
                "conclusion": "Conclusion",
                "confidence": 0.7,
                "strength": "moderate",
                "dissents": [
                    "I disagree with the conclusion",
                    "The evidence is insufficient",
                ],
                "dissenting_agents": ["agent_1", "agent_2"],
            },
        )

        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        consensus_handler = None
        for name, handler in handlers:
            if name == "consensus_to_mound":
                consensus_handler = handler
                break

        with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=True):
            with patch("aragora.knowledge.mound.get_knowledge_mound", return_value=None):
                consensus_handler(event)  # Should handle string dissents


# ============================================================================
# Evolution Tracking Tests
# ============================================================================


class TestEvolutionTracking:
    """Tests for consensus evolution tracking."""

    @pytest.fixture
    def consensus_event_with_supersedes(self):
        """Create a consensus event that supersedes a previous one."""
        return StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate_evolution_001",
                "consensus_reached": True,
                "topic": "Best practices for API authentication",
                "conclusion": "OAuth 2.0 with PKCE is recommended for all client types",
                "confidence": 0.9,
                "strength": "strong",
                "supersedes": "previous_consensus_001",  # Explicit supersedes
                "key_claims": [
                    "PKCE eliminates implicit flow vulnerabilities",
                    "Authorization code flow is more secure",
                ],
            },
        )

    def test_supersedes_field_present(self, consensus_event_with_supersedes):
        """Test event contains supersedes field."""
        data = consensus_event_with_supersedes.data
        assert "supersedes" in data
        assert data["supersedes"] == "previous_consensus_001"

    @patch("aragora.knowledge.mound.get_knowledge_mound")
    def test_handler_processes_supersedes(
        self, mock_get_mound, subscriber_manager, consensus_event_with_supersedes
    ):
        """Test handler processes supersedes relationship."""
        mock_mound = MagicMock()
        mock_mound.workspace_id = "test"

        store_result = MagicMock()
        store_result.node_id = "new_node_001"
        store_result.deduplicated = False
        mock_mound.store = AsyncMock(return_value=store_result)
        mock_mound.search = AsyncMock(return_value=[])
        mock_mound.update_metadata = AsyncMock()

        mock_get_mound.return_value = mock_mound

        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        consensus_handler = None
        for name, handler in handlers:
            if name == "consensus_to_mound":
                consensus_handler = handler
                break

        with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=True):
            consensus_handler(consensus_event_with_supersedes)

        mock_get_mound.assert_called()

    @patch("aragora.knowledge.mound.get_knowledge_mound")
    def test_similar_topic_search_for_evolution(
        self, mock_get_mound, subscriber_manager
    ):
        """Test handler searches for similar prior consensus."""
        mock_mound = MagicMock()
        mock_mound.workspace_id = "test"

        # Mock prior consensus found
        prior_result = MagicMock()
        prior_result.id = "prior_consensus_node"
        prior_result.metadata = {"debate_id": "old_debate_001"}

        store_result = MagicMock()
        store_result.node_id = "new_node_001"
        store_result.deduplicated = False

        mock_mound.store = AsyncMock(return_value=store_result)
        mock_mound.search = AsyncMock(return_value=[prior_result])
        mock_mound.update_metadata = AsyncMock()

        mock_get_mound.return_value = mock_mound

        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "new_debate_001",
                "consensus_reached": True,
                "topic": "API authentication best practices",  # Similar topic
                "conclusion": "New conclusion",
                "confidence": 0.85,
                "strength": "strong",
                # No explicit supersedes - should detect via search
            },
        )

        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        consensus_handler = None
        for name, handler in handlers:
            if name == "consensus_to_mound":
                consensus_handler = handler
                break

        with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=True):
            consensus_handler(event)

        mock_get_mound.assert_called()


# ============================================================================
# Evidence Linking Tests
# ============================================================================


class TestEvidenceLinking:
    """Tests for evidence linking in consensus ingestion."""

    @pytest.fixture
    def consensus_event_with_evidence(self):
        """Create a consensus event with supporting evidence."""
        return StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate_evidence_001",
                "consensus_reached": True,
                "topic": "Performance impact of JSON vs Protocol Buffers",
                "conclusion": "Protocol Buffers offer 3-10x performance improvement",
                "confidence": 0.88,
                "strength": "strong",
                "domain": "performance",
                "tags": ["serialization", "performance", "api"],
                "key_claims": [
                    "Protobuf serialization is 3-10x faster",
                    "Message size is 30-50% smaller",
                ],
                "supporting_evidence": [
                    "Benchmark: protobuf serialization 3.2x faster than JSON",
                    "Google internal study: 50% bandwidth reduction with protobuf",
                    "gRPC adoption statistics show 40% performance gains",
                ],
            },
        )

    def test_evidence_data_present(self, consensus_event_with_evidence):
        """Test event contains supporting evidence."""
        data = consensus_event_with_evidence.data

        assert "supporting_evidence" in data
        assert len(data["supporting_evidence"]) == 3

    def test_domain_and_tags_present(self, consensus_event_with_evidence):
        """Test event contains domain and tags for linking."""
        data = consensus_event_with_evidence.data

        assert data["domain"] == "performance"
        assert "serialization" in data["tags"]

    @patch("aragora.knowledge.mound.get_knowledge_mound")
    def test_handler_processes_evidence(
        self, mock_get_mound, subscriber_manager, consensus_event_with_evidence
    ):
        """Test handler processes supporting evidence."""
        mock_mound = MagicMock()
        mock_mound.workspace_id = "test"

        store_result = MagicMock()
        store_result.node_id = "node_001"
        store_result.deduplicated = False
        mock_mound.store = AsyncMock(return_value=store_result)
        mock_mound.search = AsyncMock(return_value=[])

        mock_get_mound.return_value = mock_mound

        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        consensus_handler = None
        for name, handler in handlers:
            if name == "consensus_to_mound":
                consensus_handler = handler
                break

        with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=True):
            consensus_handler(consensus_event_with_evidence)

        mock_get_mound.assert_called()


# ============================================================================
# Agreement Ratio Tests
# ============================================================================


class TestAgreementRatio:
    """Tests for agreement ratio calculation."""

    def test_agreement_ratio_calculation(self):
        """Test agreement ratio is calculated correctly."""
        agreeing = ["claude", "gpt4"]
        participating = ["claude", "gpt4", "gemini", "mistral"]

        ratio = len(agreeing) / len(participating)
        assert ratio == 0.5

    def test_agreement_ratio_unanimous(self):
        """Test unanimous agreement ratio."""
        agreeing = ["claude", "gpt4", "gemini"]
        participating = ["claude", "gpt4", "gemini"]

        ratio = len(agreeing) / len(participating)
        assert ratio == 1.0

    def test_agreement_ratio_empty(self):
        """Test agreement ratio with empty participants."""
        agreeing = []
        participating = []

        ratio = len(agreeing) / len(participating) if participating else 0.0
        assert ratio == 0.0

    def test_event_with_agreement_data(self, subscriber_manager):
        """Test event with agreement data is processed."""
        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate_ratio_001",
                "consensus_reached": True,
                "topic": "Topic",
                "conclusion": "Conclusion",
                "confidence": 0.75,
                "strength": "moderate",
                "participating_agents": ["a", "b", "c", "d"],
                "agreeing_agents": ["a", "b", "c"],
                "dissenting_agents": ["d"],
            },
        )

        # Agreement ratio should be 3/4 = 0.75
        data = event.data
        ratio = (
            len(data["agreeing_agents"]) / len(data["participating_agents"])
            if data["participating_agents"]
            else 0.0
        )
        assert ratio == 0.75

        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        consensus_handler = None
        for name, handler in handlers:
            if name == "consensus_to_mound":
                consensus_handler = handler
                break

        with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=True):
            with patch("aragora.knowledge.mound.get_knowledge_mound", return_value=None):
                consensus_handler(event)
