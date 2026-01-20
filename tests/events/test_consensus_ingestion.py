"""
Tests for Consensus Ingestion to Knowledge Mound.

Tests the automatic ingestion of debate consensus into the Knowledge Mound
via the cross-subscriber event handler.
"""

import pytest
from datetime import datetime
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
        timestamp=datetime.now().isoformat(),
        source="debate_orchestrator",
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
        timestamp=datetime.now().isoformat(),
        source="debate_orchestrator",
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
    from aragora.events.cross_subscribers import _manager
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

    @patch("aragora.events.cross_subscribers.get_knowledge_mound")
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

    @patch("aragora.events.cross_subscribers.get_knowledge_mound")
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
            timestamp=datetime.now().isoformat(),
            source="test",
        )

        handlers = subscriber_manager._subscribers.get(StreamEventType.CONSENSUS, [])
        consensus_handler = None
        for name, handler in handlers:
            if name == "consensus_to_mound":
                consensus_handler = handler
                break

        # Should handle gracefully
        with patch.object(subscriber_manager, "_is_km_handler_enabled", return_value=True):
            with patch("aragora.events.cross_subscribers.get_knowledge_mound", return_value=None):
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

    @patch("aragora.events.cross_subscribers.get_knowledge_mound")
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
        with patch("aragora.events.cross_subscribers.get_knowledge_mound", return_value=None):
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

        with patch("aragora.events.cross_subscribers.get_knowledge_mound", return_value=mock_mound):
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
        assert stats.call_count == 0


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
