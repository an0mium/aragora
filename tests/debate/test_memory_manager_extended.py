"""
Extended tests for MemoryManager - focusing on gaps in coverage.

Tests cover:
- Tier calculation (slow tier for low-quality, importance with consensus bonus)
- Evidence filtering (snippets < 50 chars, deduplication)
- Outcome updates (prediction_error based on success, partial failures)
- Event emission (spectator events, WebSocket events, notification failures)
"""

import logging
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import aragora.debate.memory_manager as memory_manager_module
from aragora.debate.memory_manager import MemoryManager


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_continuum_memory():
    """Create a mock ContinuumMemory."""
    mock = MagicMock()
    mock.add = MagicMock()
    mock.update_outcome = MagicMock()
    return mock


@pytest.fixture
def mock_critique_store():
    """Create a mock CritiqueStore."""
    mock = MagicMock()
    mock.retrieve_patterns = MagicMock(return_value=[])
    return mock


@pytest.fixture
def mock_debate_embeddings():
    """Create a mock DebateEmbeddings."""
    mock = MagicMock()
    mock.find_similar_debates = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def mock_spectator():
    """Create a mock spectator stream."""
    mock = MagicMock()
    mock.emit = MagicMock()
    return mock


@pytest.fixture
def mock_event_emitter():
    """Create a mock event emitter."""
    mock = MagicMock()
    mock.emit = MagicMock()
    return mock


@pytest.fixture
def manager(
    mock_continuum_memory,
    mock_critique_store,
    mock_debate_embeddings,
    mock_spectator,
    mock_event_emitter,
):
    """Create a MemoryManager with all mock dependencies."""
    return MemoryManager(
        continuum_memory=mock_continuum_memory,
        critique_store=mock_critique_store,
        debate_embeddings=mock_debate_embeddings,
        spectator=mock_spectator,
        event_emitter=mock_event_emitter,
        loop_id="test_loop",
    )


@pytest.fixture
def mock_debate_result():
    """Create a mock DebateResult."""
    result = MagicMock()
    result.id = "test_debate_123"
    result.final_answer = "This is the winning approach"
    result.confidence = 0.8
    result.rounds_used = 2
    result.consensus_reached = True
    result.winner = "claude"
    return result


# =============================================================================
# Tier Calculation Tests
# =============================================================================


class TestTierCalculation:
    """Tests for tier calculation in store_debate_outcome."""

    def test_slow_tier_for_low_quality_rounds_0(
        self, manager, mock_debate_result, mock_continuum_memory
    ):
        """Test that low-quality debates (rounds=0, low confidence) go to slow tier."""
        mock_debate_result.rounds_used = 0
        mock_debate_result.confidence = 0.3
        mock_debate_result.consensus_reached = False

        manager.store_debate_outcome(mock_debate_result, "test task")

        # Verify add was called with slow tier
        mock_continuum_memory.add.assert_called_once()
        call_args = mock_continuum_memory.add.call_args
        tier = call_args.kwargs["tier"]
        assert getattr(tier, "value", tier) == "slow"

    def test_slow_tier_for_low_confidence(self, manager, mock_debate_result, mock_continuum_memory):
        """Test that low confidence debates go to slow tier."""
        mock_debate_result.rounds_used = 1
        mock_debate_result.confidence = 0.4

        manager.store_debate_outcome(mock_debate_result, "test task")

        mock_continuum_memory.add.assert_called_once()
        call_args = mock_continuum_memory.add.call_args
        tier = call_args.kwargs["tier"]
        assert getattr(tier, "value", tier) == "slow"

    def test_medium_tier_for_medium_quality(
        self, manager, mock_debate_result, mock_continuum_memory
    ):
        """Test that medium quality debates go to medium tier."""
        mock_debate_result.rounds_used = 1
        mock_debate_result.confidence = 0.6

        manager.store_debate_outcome(mock_debate_result, "test task")

        mock_continuum_memory.add.assert_called_once()
        call_args = mock_continuum_memory.add.call_args
        tier = call_args.kwargs["tier"]
        assert getattr(tier, "value", tier) == "medium"

    def test_fast_tier_for_high_quality(self, manager, mock_debate_result, mock_continuum_memory):
        """Test that high quality debates go to fast tier."""
        mock_debate_result.rounds_used = 3
        mock_debate_result.confidence = 0.85

        manager.store_debate_outcome(mock_debate_result, "test task")

        mock_continuum_memory.add.assert_called_once()
        call_args = mock_continuum_memory.add.call_args
        tier = call_args.kwargs["tier"]
        assert getattr(tier, "value", tier) == "fast"

    def test_importance_with_consensus_bonus(
        self, manager, mock_debate_result, mock_continuum_memory
    ):
        """Test that consensus adds 0.1 to importance."""
        mock_debate_result.confidence = 0.6
        mock_debate_result.consensus_reached = True

        manager.store_debate_outcome(mock_debate_result, "test task")

        mock_continuum_memory.add.assert_called_once()
        call_args = mock_continuum_memory.add.call_args
        importance = call_args.kwargs["importance"]

        # Without consensus: (0.6 + 0.5) / 1.5 = 0.733
        # With consensus: min(1.0, 0.733 + 0.1) = 0.833
        assert importance > 0.7
        assert importance <= 1.0

    def test_long_content_truncation(self, manager, mock_debate_result, mock_continuum_memory):
        """Test that long content is truncated to 300 chars."""
        mock_debate_result.final_answer = "x" * 500

        manager.store_debate_outcome(mock_debate_result, "test task")

        mock_continuum_memory.add.assert_called_once()
        call_args = mock_continuum_memory.add.call_args
        content = call_args.kwargs["content"]

        # Should contain truncated content (300 chars + "...")
        assert "..." in content
        # The first 300 chars should be present
        assert len(content) < 450


# =============================================================================
# Evidence Filtering Tests
# =============================================================================


class TestEvidenceFiltering:
    """Tests for evidence filtering in store_evidence."""

    def test_filters_snippets_less_than_50_chars(self, manager, mock_continuum_memory):
        """Test that snippets < 50 chars are filtered out."""
        short_snippets = [
            MagicMock(content="Short", source="test", relevance=0.5),
            MagicMock(content="A" * 60, source="test", relevance=0.6),  # Long enough
        ]

        manager.store_evidence(short_snippets, "test task")

        # Only the long snippet should be stored
        assert mock_continuum_memory.add.call_count == 1

    def test_deduplication_via_sha256(self, manager, mock_continuum_memory):
        """Test that evidence IDs use SHA256 for deduplication."""
        snippet = MagicMock(content="A" * 100, source="test", relevance=0.5)

        manager.store_evidence([snippet], "test task")

        mock_continuum_memory.add.assert_called_once()
        call_args = mock_continuum_memory.add.call_args

        # ID should start with "evidence_" followed by hex hash
        evidence_id = call_args.kwargs["id"]
        assert evidence_id.startswith("evidence_")
        # Hash portion should be 10 hex characters
        hash_part = evidence_id.replace("evidence_", "")
        assert len(hash_part) == 10
        assert all(c in "0123456789abcdef" for c in hash_part)


# =============================================================================
# Outcome Updates Tests
# =============================================================================


class TestOutcomeUpdates:
    """Tests for outcome updates in update_memory_outcomes."""

    def test_correct_prediction_error_based_on_success(
        self, manager, mock_debate_result, mock_continuum_memory
    ):
        """Test that prediction_error is calculated correctly based on success."""
        manager._retrieved_ids = ["mem_1", "mem_2"]
        mock_debate_result.consensus_reached = True
        mock_debate_result.confidence = 0.8

        manager.update_memory_outcomes(mock_debate_result)

        # Both memories should be updated
        assert mock_continuum_memory.update_outcome.call_count == 2

        # Check prediction_error: for success, it's 1.0 - confidence = 0.2
        call_args = mock_continuum_memory.update_outcome.call_args_list[0]
        assert call_args.kwargs["success"] is True
        assert call_args.kwargs["agent_prediction_error"] == pytest.approx(0.2)

    def test_prediction_error_for_failure(self, manager, mock_debate_result, mock_continuum_memory):
        """Test prediction_error for failed debates."""
        manager._retrieved_ids = ["mem_1"]
        mock_debate_result.consensus_reached = False
        mock_debate_result.confidence = 0.7

        manager.update_memory_outcomes(mock_debate_result)

        mock_continuum_memory.update_outcome.assert_called_once()
        call_args = mock_continuum_memory.update_outcome.call_args

        # For failure, prediction_error is the confidence itself
        assert call_args.kwargs["success"] is False
        assert call_args.kwargs["agent_prediction_error"] == 0.7

    def test_partial_update_failures_handled(
        self, manager, mock_debate_result, mock_continuum_memory
    ):
        """Test that partial update failures are handled gracefully."""
        manager._retrieved_ids = ["mem_1", "mem_2", "mem_3"]

        # Make second update fail
        mock_continuum_memory.update_outcome.side_effect = [
            None,  # First succeeds
            RuntimeError("Database error"),  # Second fails
            None,  # Third succeeds
        ]

        # Should not raise
        manager.update_memory_outcomes(mock_debate_result)

        # All three should have been attempted
        assert mock_continuum_memory.update_outcome.call_count == 3
        assert manager._retrieved_ids == []
        assert [update.memory_id for update in manager._pending_outcome_updates] == ["mem_2"]

    def test_sqlite_outcome_failure_is_retried_without_reapplying_successes(
        self, manager, mock_debate_result, mock_continuum_memory
    ):
        """SQLite write failures are transient and do not replay already-applied updates."""
        manager._retrieved_ids = ["mem_ok", "mem_locked"]
        mock_continuum_memory.update_outcome.side_effect = [
            None,
            sqlite3.OperationalError("database is locked"),
        ]

        manager.update_memory_outcomes(mock_debate_result)

        assert [update.memory_id for update in manager._pending_outcome_updates] == ["mem_locked"]
        assert [
            call.kwargs["id"] for call in mock_continuum_memory.update_outcome.call_args_list
        ] == ["mem_ok", "mem_locked"]

        mock_continuum_memory.update_outcome.reset_mock()
        mock_continuum_memory.update_outcome.side_effect = None

        manager.update_memory_outcomes(mock_debate_result)

        mock_continuum_memory.update_outcome.assert_called_once()
        assert mock_continuum_memory.update_outcome.call_args.kwargs["id"] == "mem_locked"
        assert manager._pending_outcome_updates == []

    def test_failed_outcome_updates_keep_tier_state_for_retry(
        self, manager, mock_debate_result, mock_continuum_memory
    ):
        """A failed outcome write must not discard retry state for that memory."""
        manager._retrieved_ids = ["mem_1", "mem_2", "mem_3"]
        manager._retrieved_tiers = {
            "mem_1": "fast",
            "mem_2": "slow",
            "mem_3": "fast",
        }
        mock_continuum_memory.update_outcome.side_effect = [
            None,
            RuntimeError("transient write failure"),
            None,
        ]

        manager.update_memory_outcomes(mock_debate_result)

        assert manager._retrieved_ids == []
        assert manager._retrieved_tiers == {}
        assert len(manager._pending_outcome_updates) == 1
        pending = manager._pending_outcome_updates[0]
        assert pending.memory_id == "mem_2"
        assert pending.tier == "slow"
        assert pending.success is True
        assert pending.confidence == pytest.approx(0.8)

    @pytest.mark.parametrize(
        "exception",
        [
            AttributeError("missing update method"),
            TypeError("bad adapter signature"),
            ValueError("bad memory payload"),
            KeyError("missing memory row"),
        ],
    )
    def test_permanent_outcome_update_failures_are_not_queued(
        self, manager, mock_debate_result, mock_continuum_memory, exception
    ):
        """Permanent adapter/data errors are logged and dropped, not retried forever."""
        manager._retrieved_ids = ["mem_1"]
        mock_continuum_memory.update_outcome.side_effect = exception

        manager.update_memory_outcomes(mock_debate_result)

        assert manager._retrieved_ids == []
        assert manager._retrieved_tiers == {}
        assert manager._pending_outcome_updates == []

    def test_outcome_retry_uses_original_debate_payload(self, manager, mock_continuum_memory):
        """Pending outcome retries use their source debate payload, not the next debate's."""
        debate_a = MagicMock()
        debate_a.id = "debate-a"
        debate_a.consensus_reached = True
        debate_a.confidence = 0.9

        debate_b = MagicMock()
        debate_b.id = "debate-b"
        debate_b.consensus_reached = False
        debate_b.confidence = 0.2

        manager.track_retrieved_ids(["mem_a"], tiers={"mem_a": "fast"})
        mock_continuum_memory.update_outcome.side_effect = [RuntimeError("transient")]

        manager.update_memory_outcomes(debate_a)

        assert manager._retrieved_ids == []
        assert [update.memory_id for update in manager._pending_outcome_updates] == ["mem_a"]

        manager.track_retrieved_ids(["mem_b"], tiers={"mem_b": "slow"})
        mock_continuum_memory.update_outcome.side_effect = [None, None]

        manager.update_memory_outcomes(debate_b)

        assert mock_continuum_memory.update_outcome.call_count == 3
        retry_call = mock_continuum_memory.update_outcome.call_args_list[1]
        current_call = mock_continuum_memory.update_outcome.call_args_list[2]
        assert retry_call.kwargs["id"] == "mem_a"
        assert retry_call.kwargs["success"] is True
        assert retry_call.kwargs["agent_prediction_error"] == pytest.approx(0.1)
        assert current_call.kwargs["id"] == "mem_b"
        assert current_call.kwargs["success"] is False
        assert current_call.kwargs["agent_prediction_error"] == pytest.approx(0.2)
        assert manager._pending_outcome_updates == []
        assert manager._retrieved_ids == []

    def test_pending_outcome_retries_are_coalesced_and_capped(
        self, manager, mock_continuum_memory, monkeypatch
    ):
        """Repeated transient failures keep the bounded latest pending outcome per memory."""
        monkeypatch.setattr(memory_manager_module, "_MAX_PENDING_OUTCOME_UPDATES", 2)
        mock_continuum_memory.update_outcome.side_effect = RuntimeError("transient")

        for debate_id, confidence, memory_id in [
            ("debate-a", 0.7, "mem_a"),
            ("debate-b", 0.8, "mem_a"),
            ("debate-c", 0.9, "mem_b"),
            ("debate-d", 0.6, "mem_c"),
        ]:
            result = MagicMock()
            result.id = debate_id
            result.consensus_reached = True
            result.confidence = confidence
            manager.track_retrieved_ids([memory_id])
            manager.update_memory_outcomes(result)

        pending = manager._pending_outcome_updates
        assert [update.memory_id for update in pending] == ["mem_b", "mem_c"]
        assert [update.debate_id for update in pending] == ["debate-c", "debate-d"]
        assert len(pending) == 2

    def test_pending_outcome_cap_logs_dropped_memory_ids(
        self, manager, mock_continuum_memory, monkeypatch, caplog
    ):
        """Overflow policy keeps latest unique IDs and logs the dropped memory IDs."""
        monkeypatch.setattr(memory_manager_module, "_MAX_PENDING_OUTCOME_UPDATES", 2)
        mock_continuum_memory.update_outcome.side_effect = RuntimeError("transient")
        manager.track_retrieved_ids(["mem_a", "mem_b", "mem_c"])

        with caplog.at_level(logging.WARNING):
            manager.update_memory_outcomes(
                MagicMock(id="debate", consensus_reached=True, confidence=0.8)
            )

        assert [update.memory_id for update in manager._pending_outcome_updates] == [
            "mem_b",
            "mem_c",
        ]
        assert "policy=latest_per_memory_id" in caplog.text
        assert "dropped_memory_ids=mem_a" in caplog.text

    def test_tier_analytics_failures_do_not_abort_batch_or_leave_tracked_ids(
        self, manager, mock_debate_result, mock_continuum_memory
    ):
        """Tier analytics failures stay per-memory and tracking state is cleared."""
        manager._retrieved_ids = ["mem_1", "mem_2", "mem_3"]
        manager._retrieved_tiers = {
            "mem_1": "fast",
            "mem_2": "slow",
            "mem_3": "fast",
        }
        tracker = MagicMock()
        tracker.record_usage.side_effect = [
            StopIteration("missing tier row"),
            ImportError("analytics unavailable"),
            None,
        ]
        manager.tier_analytics_tracker = tracker

        manager.update_memory_outcomes(mock_debate_result)

        assert mock_continuum_memory.update_outcome.call_count == 3
        assert tracker.record_usage.call_count == 3
        assert [call.kwargs["memory_id"] for call in tracker.record_usage.call_args_list] == [
            "mem_1",
            "mem_2",
            "mem_3",
        ]
        assert manager._retrieved_ids == []
        assert manager._retrieved_tiers == {}

    def test_sqlite_tier_analytics_failure_does_not_abort_outcome_drain(
        self, manager, mock_debate_result, mock_continuum_memory, caplog
    ):
        """SQLite analytics failures should not skip later updates or cleanup."""
        manager._retrieved_ids = ["mem_1", "mem_2"]
        manager._retrieved_tiers = {
            "mem_1": "fast",
            "mem_2": "slow",
        }
        tracker = MagicMock()
        tracker.record_usage.side_effect = [
            sqlite3.OperationalError("database is locked"),
            None,
        ]
        manager.tier_analytics_tracker = tracker

        with caplog.at_level(logging.WARNING):
            manager.update_memory_outcomes(mock_debate_result)

        assert mock_continuum_memory.update_outcome.call_count == 2
        assert tracker.record_usage.call_count == 2
        assert manager._retrieved_ids == []
        assert manager._retrieved_tiers == {}
        assert manager._pending_outcome_updates == []
        assert "Unexpected error recording usage for mem_1" in caplog.text

    def test_current_success_clears_stale_pending_retry_for_same_memory(
        self, manager, mock_debate_result, mock_continuum_memory
    ):
        """A newer successful current update must retire stale same-memory retries."""
        manager._pending_outcome_updates = [
            memory_manager_module._PendingMemoryOutcomeUpdate(
                memory_id="mem_same",
                success=True,
                confidence=0.9,
                debate_id="old-debate",
                tier="fast",
            )
        ]
        manager.track_retrieved_ids(["mem_same"], tiers={"mem_same": "slow"})
        mock_continuum_memory.update_outcome.side_effect = [
            RuntimeError("transient retry failure"),
            None,
        ]

        manager.update_memory_outcomes(mock_debate_result)

        assert [
            call.kwargs["id"] for call in mock_continuum_memory.update_outcome.call_args_list
        ] == ["mem_same", "mem_same"]
        assert manager._pending_outcome_updates == []
        assert manager._retrieved_ids == []
        assert manager._retrieved_tiers == {}

        mock_continuum_memory.update_outcome.reset_mock()
        mock_continuum_memory.update_outcome.side_effect = None

        manager.update_memory_outcomes(mock_debate_result)

        mock_continuum_memory.update_outcome.assert_not_called()


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestEventEmission:
    """Tests for event emission in fetch_historical_context."""

    @pytest.mark.asyncio
    async def test_spectator_memory_recall_event_structure(
        self, manager, mock_debate_embeddings, mock_spectator
    ):
        """Test spectator memory_recall event has correct structure."""
        mock_debate_embeddings.find_similar_debates.return_value = [
            ("debate_1", "Topic 1", 0.85),
            ("debate_2", "Topic 2", 0.72),
        ]

        await manager.fetch_historical_context("test task")

        mock_spectator.emit.assert_called_once()
        call_args = mock_spectator.emit.call_args
        assert call_args[0][0] == "memory_recall"
        assert "details" in call_args.kwargs
        assert "metric" in call_args.kwargs
        assert call_args.kwargs["metric"] == 0.85  # Top similarity

    @pytest.mark.asyncio
    async def test_websocket_stream_event_emission(
        self, manager, mock_debate_embeddings, mock_event_emitter
    ):
        """Test WebSocket StreamEvent emission."""
        mock_debate_embeddings.find_similar_debates.return_value = [
            ("debate_1", "Topic 1", 0.85),
        ]

        # StreamEvent is imported inside the method from aragora.server.stream
        with patch("aragora.server.stream.StreamEvent") as MockStreamEvent:
            with patch("aragora.server.stream.StreamEventType") as MockStreamEventType:
                MockStreamEventType.MEMORY_RECALL = "memory_recall"
                MockStreamEvent.return_value = MagicMock()

                await manager.fetch_historical_context("test task")

                # Event emitter should have been called
                mock_event_emitter.emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_notification_failures_gracefully(
        self, manager, mock_debate_embeddings, mock_spectator
    ):
        """Test that notification failures don't crash the method."""
        mock_debate_embeddings.find_similar_debates.return_value = [
            ("debate_1", "Topic 1", 0.85),
        ]
        mock_spectator.emit.side_effect = RuntimeError("Notification failed")

        # Should not raise
        result = await manager.fetch_historical_context("test task")

        # Should still return context
        assert "HISTORICAL CONTEXT" in result

    def test_expected_spectator_adapter_failures_log_at_debug(
        self, manager, mock_spectator, caplog
    ):
        """Expected adapter-shape failures stay quiet while still being contained."""
        mock_spectator.emit.side_effect = AttributeError("missing emit shape")

        with caplog.at_level(logging.DEBUG, logger="aragora.debate.memory_manager"):
            manager._notify_spectator("memory_recall", "details", metric=0.5)

        assert "Spectator notification error" in caplog.text
        assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


# =============================================================================
# Integration Tests
# =============================================================================


class TestMemoryManagerIntegration:
    """Integration tests for MemoryManager."""

    def test_domain_extractor_used_for_domain(self, mock_continuum_memory):
        """Test that domain extractor is used when provided."""
        manager = MemoryManager(
            continuum_memory=mock_continuum_memory,
            domain_extractor=lambda: "technology",
        )

        assert manager._get_domain() == "technology"

    def test_default_domain_when_no_extractor(self, mock_continuum_memory):
        """Test default domain when no extractor provided."""
        manager = MemoryManager(continuum_memory=mock_continuum_memory)

        assert manager._get_domain() == "general"

    def test_track_and_clear_retrieved_ids(self, manager):
        """Test tracking and clearing of retrieved IDs."""
        manager.track_retrieved_ids(["id1", "id2", "id3"])

        assert manager._retrieved_ids == ["id1", "id2", "id3"]

        manager.clear_retrieved_ids()

        assert manager._retrieved_ids == []

    def test_clear_retrieved_ids_preserves_pending_outcome_retries(self, manager):
        """Clearing per-debate retrieval state must not abandon pending writes."""
        manager._pending_outcome_updates = [
            memory_manager_module._PendingMemoryOutcomeUpdate(
                memory_id="mem_1",
                success=True,
                confidence=0.8,
                debate_id="debate-1",
            )
        ]
        manager.track_retrieved_ids(["id1"], tiers={"id1": "fast"})

        manager.clear_retrieved_ids()

        assert manager._retrieved_ids == []
        assert manager._retrieved_tiers == {}
        assert [update.memory_id for update in manager._pending_outcome_updates] == ["mem_1"]

    def test_track_retrieved_ids_filters_empty(self, manager):
        """Test that empty IDs are filtered out."""
        manager.track_retrieved_ids(["id1", "", None, "id2"])

        assert manager._retrieved_ids == ["id1", "id2"]

    def test_store_outcome_no_continuum_memory(self, mock_debate_result):
        """Test that store_debate_outcome handles missing continuum_memory."""
        manager = MemoryManager()  # No continuum_memory

        # Should not raise
        manager.store_debate_outcome(mock_debate_result, "test task")

    def test_store_outcome_no_final_answer(self, mock_continuum_memory, mock_debate_result):
        """Test that store_debate_outcome handles missing final_answer."""
        manager = MemoryManager(continuum_memory=mock_continuum_memory)
        mock_debate_result.final_answer = None

        manager.store_debate_outcome(mock_debate_result, "test task")

        # Should not have stored anything
        mock_continuum_memory.add.assert_not_called()

    def test_format_patterns_for_prompt_empty(self, manager):
        """Test formatting empty patterns returns empty string."""
        result = manager._format_patterns_for_prompt([])
        assert result == ""

    def test_format_patterns_for_prompt_with_patterns(self, manager):
        """Test formatting patterns includes severity labels."""
        patterns = [
            {
                "category": "logic",
                "pattern": "Circular reasoning",
                "occurrences": 5,
                "avg_severity": 0.8,
            },
            {
                "category": "evidence",
                "pattern": "Missing sources",
                "occurrences": 3,
                "avg_severity": 0.5,
            },
        ]

        result = manager._format_patterns_for_prompt(patterns)

        assert "LEARNED PATTERNS" in result
        assert "LOGIC" in result
        assert "HIGH SEVERITY" in result
        assert "MEDIUM" in result
        assert "Circular reasoning" in result
