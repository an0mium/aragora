"""
Integration tests for new feature integrations.

Tests end-to-end functionality of:
- TTS Integration for live voice responses
- Convergence Tracker for debate round tracking
- Vote Bonus Calculator for evidence and process bonuses
- RLM Streaming Mixin for streaming query execution
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional
from unittest.mock import AsyncMock, Mock, MagicMock, patch

import pytest

from aragora.core import Vote


# =============================================================================
# TTS Integration Tests
# =============================================================================


class TestTTSIntegration:
    """Tests for TTSIntegration feature."""

    @pytest.fixture
    def mock_voice_handler(self):
        """Create a mock voice handler."""
        handler = Mock()
        handler.is_tts_available = True
        handler.has_voice_session = Mock(return_value=True)
        handler.synthesize_agent_message = AsyncMock(return_value=1)
        return handler

    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock event bus."""
        bus = Mock()
        bus.subscribe = Mock()
        return bus

    @pytest.mark.asyncio
    async def test_tts_integration_initialization(self, mock_voice_handler):
        """Test TTSIntegration initializes correctly."""
        from aragora.server.stream.tts_integration import TTSIntegration

        integration = TTSIntegration(voice_handler=mock_voice_handler)

        assert integration.is_available
        assert integration._enabled

    @pytest.mark.asyncio
    async def test_tts_integration_registers_with_event_bus(
        self, mock_voice_handler, mock_event_bus
    ):
        """Test TTSIntegration registers for agent_message events."""
        from aragora.server.stream.tts_integration import TTSIntegration

        integration = TTSIntegration(voice_handler=mock_voice_handler)
        integration.register(mock_event_bus)

        mock_event_bus.subscribe.assert_called_once_with(
            "agent_message", integration._handle_agent_message
        )

    @pytest.mark.asyncio
    async def test_tts_integration_handles_agent_message(self, mock_voice_handler):
        """Test TTSIntegration handles agent messages correctly."""
        from aragora.server.stream.tts_integration import TTSIntegration

        integration = TTSIntegration(voice_handler=mock_voice_handler)

        # Create a mock event
        event = Mock()
        event.debate_id = "test-debate-123"
        event.data = {
            "agent": "claude",
            "content": "This is a test message for TTS synthesis.",
            "enable_tts": True,
        }

        await integration._handle_agent_message(event)

        mock_voice_handler.has_voice_session.assert_called_once_with("test-debate-123")
        mock_voice_handler.synthesize_agent_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_tts_integration_respects_disable_flag(self, mock_voice_handler):
        """Test TTSIntegration respects enable_tts=False."""
        from aragora.server.stream.tts_integration import TTSIntegration

        integration = TTSIntegration(voice_handler=mock_voice_handler)

        event = Mock()
        event.debate_id = "test-debate"
        event.data = {"enable_tts": False}

        await integration._handle_agent_message(event)

        mock_voice_handler.synthesize_agent_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_tts_integration_rate_limiting(self, mock_voice_handler):
        """Test TTSIntegration rate limits rapid messages."""
        from aragora.server.stream.tts_integration import TTSIntegration

        integration = TTSIntegration(
            voice_handler=mock_voice_handler,
            min_interval_seconds=0.5,
        )

        event = Mock()
        event.debate_id = "test-debate"
        event.data = {
            "agent": "claude",
            "content": "Message 1",
            "enable_tts": True,
        }

        # First message should process
        await integration._handle_agent_message(event)

        # Immediate second message should be rate limited
        event.data["content"] = "Message 2"
        await integration._handle_agent_message(event)

        # Only one synthesis call expected
        assert mock_voice_handler.synthesize_agent_message.call_count == 1


# =============================================================================
# Convergence Tracker Tests
# =============================================================================


class TestConvergenceTracker:
    """Tests for DebateConvergenceTracker."""

    @pytest.fixture
    def mock_convergence_detector(self):
        """Create a mock convergence detector."""
        detector = Mock()
        result = Mock()
        result.converged = False
        result.status = "checking"
        result.avg_similarity = 0.65
        result.per_agent_similarity = {"agent_1": 0.7, "agent_2": 0.6}
        detector.check_convergence = Mock(return_value=result)
        return detector

    @pytest.fixture
    def mock_novelty_tracker(self):
        """Create a mock novelty tracker."""
        tracker = Mock()
        result = Mock()
        result.avg_novelty = 0.8
        result.min_novelty = 0.6
        result.per_agent_novelty = {"agent_1": 0.85, "agent_2": 0.75}
        result.low_novelty_agents = []
        result.has_low_novelty = Mock(return_value=False)
        tracker.compute_novelty = Mock(return_value=result)
        tracker.add_to_history = Mock()
        return tracker

    @pytest.fixture
    def mock_debate_context(self):
        """Create a mock debate context."""
        ctx = Mock()
        ctx.proposals = {"agent_1": "Proposal A", "agent_2": "Proposal B"}
        ctx.result = Mock()
        ctx.result.convergence_status = None
        ctx.result.convergence_similarity = 0.0
        ctx.result.per_agent_similarity = {}
        ctx.per_agent_novelty = {}
        ctx.avg_novelty = 0.0
        ctx.low_novelty_agents = []
        ctx.debate_id = "test-debate-456"
        return ctx

    def test_convergence_tracker_initialization(
        self, mock_convergence_detector, mock_novelty_tracker
    ):
        """Test DebateConvergenceTracker initializes correctly."""
        from aragora.debate.phases.convergence_tracker import DebateConvergenceTracker

        tracker = DebateConvergenceTracker(
            convergence_detector=mock_convergence_detector,
            novelty_tracker=mock_novelty_tracker,
        )

        assert tracker.convergence_detector is mock_convergence_detector
        assert tracker.novelty_tracker is mock_novelty_tracker

    def test_convergence_tracker_reset(
        self, mock_convergence_detector, mock_novelty_tracker
    ):
        """Test convergence tracker state reset."""
        from aragora.debate.phases.convergence_tracker import DebateConvergenceTracker

        tracker = DebateConvergenceTracker(
            convergence_detector=mock_convergence_detector,
            novelty_tracker=mock_novelty_tracker,
        )

        # Simulate some state
        tracker._previous_round_responses = {"agent_1": "Old response"}

        tracker.reset()

        assert tracker._previous_round_responses == {}

    def test_convergence_check(
        self, mock_convergence_detector, mock_debate_context
    ):
        """Test convergence checking logic."""
        from aragora.debate.phases.convergence_tracker import DebateConvergenceTracker

        tracker = DebateConvergenceTracker(
            convergence_detector=mock_convergence_detector,
        )

        # First call should store responses
        result1 = tracker.check_convergence(mock_debate_context, round_num=1)
        assert not result1.converged

        # Second call should perform comparison
        result2 = tracker.check_convergence(mock_debate_context, round_num=2)
        mock_convergence_detector.check_convergence.assert_called()

    def test_novelty_tracking(
        self, mock_novelty_tracker, mock_debate_context
    ):
        """Test novelty tracking functionality."""
        from aragora.debate.phases.convergence_tracker import DebateConvergenceTracker

        tracker = DebateConvergenceTracker(
            novelty_tracker=mock_novelty_tracker,
        )

        tracker.track_novelty(mock_debate_context, round_num=1)

        mock_novelty_tracker.compute_novelty.assert_called_once()
        mock_novelty_tracker.add_to_history.assert_called_once()
        assert mock_debate_context.avg_novelty == 0.8


# =============================================================================
# Vote Bonus Calculator Tests
# =============================================================================


class TestVoteBonusCalculator:
    """Tests for VoteBonusCalculator."""

    @pytest.fixture
    def mock_protocol(self):
        """Create a mock protocol with evidence weighting enabled."""
        protocol = Mock()
        protocol.enable_evidence_weighting = True
        protocol.evidence_citation_bonus = 0.15
        protocol.enable_evidence_quality_weighting = True
        protocol.enable_process_evaluation = False
        return protocol

    @pytest.fixture
    def mock_evidence_pack(self):
        """Create a mock evidence pack."""
        snippet1 = Mock()
        snippet1.id = "abc123"
        snippet1.quality_scores = {
            "semantic_relevance": 0.9,
            "authority": 0.8,
            "freshness": 0.7,
            "completeness": 0.85,
        }

        snippet2 = Mock()
        snippet2.id = "def456"
        snippet2.quality_scores = {
            "semantic_relevance": 0.6,
            "authority": 0.5,
            "freshness": 0.4,
            "completeness": 0.55,
        }

        pack = Mock()
        pack.snippets = [snippet1, snippet2]
        return pack

    @pytest.fixture
    def mock_context_with_evidence(self, mock_evidence_pack):
        """Create a mock debate context with evidence."""
        ctx = Mock()
        ctx.evidence_pack = mock_evidence_pack
        ctx.result = Mock()
        ctx.result.verification_results = None
        return ctx

    def test_vote_bonus_calculator_initialization(self, mock_protocol):
        """Test VoteBonusCalculator initializes correctly."""
        from aragora.debate.phases.vote_bonus_calculator import VoteBonusCalculator

        calculator = VoteBonusCalculator(protocol=mock_protocol)
        assert calculator.protocol is mock_protocol

    def test_evidence_citation_bonus_applied(
        self, mock_protocol, mock_context_with_evidence
    ):
        """Test evidence citation bonus is correctly applied."""
        from aragora.debate.phases.vote_bonus_calculator import VoteBonusCalculator

        calculator = VoteBonusCalculator(protocol=mock_protocol)

        # Create vote with evidence citation
        vote = Mock()
        vote.agent = "agent_1"
        vote.choice = "agent_1"
        vote.reasoning = "I cite EVID-abc123 as strong evidence for this position."

        votes = [vote]
        vote_counts = {"agent_1": 1.0, "agent_2": 0.0}
        choice_mapping = {"agent_1": "agent_1", "agent_2": "agent_2"}

        result = calculator.apply_evidence_citation_bonuses(
            ctx=mock_context_with_evidence,
            votes=votes,
            vote_counts=vote_counts.copy(),
            choice_mapping=choice_mapping,
        )

        # Should have bonus applied
        assert result["agent_1"] > 1.0

    def test_no_bonus_without_evidence_weighting(self, mock_context_with_evidence):
        """Test no bonus when evidence weighting disabled."""
        from aragora.debate.phases.vote_bonus_calculator import VoteBonusCalculator

        # Protocol with evidence weighting disabled
        protocol = Mock()
        protocol.enable_evidence_weighting = False

        calculator = VoteBonusCalculator(protocol=protocol)

        vote = Mock()
        vote.agent = "agent_1"
        vote.choice = "agent_1"
        vote.reasoning = "I cite EVID-abc123."

        votes = [vote]
        vote_counts = {"agent_1": 1.0}
        choice_mapping = {"agent_1": "agent_1"}

        result = calculator.apply_evidence_citation_bonuses(
            ctx=mock_context_with_evidence,
            votes=votes,
            vote_counts=vote_counts.copy(),
            choice_mapping=choice_mapping,
        )

        # No change expected
        assert result["agent_1"] == 1.0

    def test_invalid_evidence_citation_ignored(
        self, mock_protocol, mock_context_with_evidence
    ):
        """Test invalid evidence citations are ignored."""
        from aragora.debate.phases.vote_bonus_calculator import VoteBonusCalculator

        calculator = VoteBonusCalculator(protocol=mock_protocol)

        vote = Mock()
        vote.agent = "agent_1"
        vote.choice = "agent_1"
        vote.reasoning = "I cite EVID-invalid999 which doesn't exist."

        votes = [vote]
        vote_counts = {"agent_1": 1.0}
        choice_mapping = {"agent_1": "agent_1"}

        result = calculator.apply_evidence_citation_bonuses(
            ctx=mock_context_with_evidence,
            votes=votes,
            vote_counts=vote_counts.copy(),
            choice_mapping=choice_mapping,
        )

        # No change expected (invalid citation)
        assert result["agent_1"] == 1.0


# =============================================================================
# RLM Streaming Mixin Tests
# =============================================================================


class TestRLMStreamingMixin:
    """Tests for RLMStreamingMixin functionality."""

    @pytest.fixture
    def mock_rlm_context(self):
        """Create a mock RLM context."""
        from aragora.rlm.types import AbstractionLevel

        node = Mock()
        node.id = "node-1"
        node.content = "Test content for node"
        node.token_count = 100

        context = Mock()
        context.levels = {
            AbstractionLevel.ABSTRACT: [node],
            AbstractionLevel.SUMMARY: [node],
            AbstractionLevel.DETAILED: [node, node],
        }
        context.total_tokens_at_level = Mock(return_value=200)
        return context

    @pytest.fixture
    def mock_rlm_result(self):
        """Create a mock RLM result."""
        from aragora.rlm.types import RLMResult

        return RLMResult(
            answer="This is the answer from RLM.",
            ready=True,
            confidence=0.92,
            tokens_processed=500,
            sub_calls_made=3,
            iteration=0,
        )

    @pytest.mark.asyncio
    async def test_streaming_mixin_query_stream_events(
        self, mock_rlm_context, mock_rlm_result
    ):
        """Test query_stream yields correct event sequence."""
        from aragora.rlm.streaming_mixin import RLMStreamingMixin
        from aragora.rlm.types import RLMStreamEventType

        # Create a test class that uses the mixin
        class TestRLM(RLMStreamingMixin):
            async def query(self, query, context, strategy):
                return mock_rlm_result

        rlm = TestRLM()

        events = []
        async for event in rlm.query_stream(
            query="What is the answer?",
            context=mock_rlm_context,
            strategy="auto",
        ):
            events.append(event)

        # Verify event sequence
        event_types = [e.event_type for e in events]

        assert RLMStreamEventType.QUERY_START in event_types
        assert RLMStreamEventType.QUERY_COMPLETE in event_types

        # Should have level entered events
        level_events = [e for e in events if e.event_type == RLMStreamEventType.LEVEL_ENTERED]
        assert len(level_events) == 3  # ABSTRACT, SUMMARY, DETAILED

    @pytest.mark.asyncio
    async def test_streaming_mixin_handles_errors(self, mock_rlm_context):
        """Test query_stream handles errors gracefully."""
        from aragora.rlm.streaming_mixin import RLMStreamingMixin
        from aragora.rlm.types import RLMStreamEventType

        class FailingRLM(RLMStreamingMixin):
            async def query(self, query, context, strategy):
                raise ValueError("Query failed")

        rlm = FailingRLM()

        events = []
        with pytest.raises(ValueError):
            async for event in rlm.query_stream(
                query="This will fail",
                context=mock_rlm_context,
                strategy="auto",
            ):
                events.append(event)

        # Should have error event before raising
        event_types = [e.event_type for e in events]
        assert RLMStreamEventType.ERROR in event_types

    @pytest.mark.asyncio
    async def test_streaming_mixin_node_examination(self, mock_rlm_context, mock_rlm_result):
        """Test query_stream emits node examination events."""
        from aragora.rlm.streaming_mixin import RLMStreamingMixin
        from aragora.rlm.types import RLMStreamEventType

        class TestRLM(RLMStreamingMixin):
            async def query(self, query, context, strategy):
                return mock_rlm_result

        rlm = TestRLM()

        events = []
        async for event in rlm.query_stream(
            query="Examine nodes",
            context=mock_rlm_context,
            strategy="auto",
        ):
            events.append(event)

        # Should have node examined events
        node_events = [e for e in events if e.event_type == RLMStreamEventType.NODE_EXAMINED]
        assert len(node_events) > 0

        # Verify node content is truncated for long content
        for event in node_events:
            if event.content:
                assert len(event.content) <= 203  # 200 + "..."


# =============================================================================
# Cross-Feature Integration Tests
# =============================================================================


class TestCrossFeatureIntegration:
    """Tests for integration between multiple features."""

    @pytest.mark.asyncio
    async def test_debate_with_convergence_and_bonuses(self):
        """Test debate flow with convergence tracking and vote bonuses."""
        from aragora.debate.phases.convergence_tracker import DebateConvergenceTracker
        from aragora.debate.phases.vote_bonus_calculator import VoteBonusCalculator

        # Setup tracker
        tracker = DebateConvergenceTracker()

        # Setup calculator (no protocol = no bonuses)
        calculator = VoteBonusCalculator()

        # Verify they can be instantiated together without conflicts
        assert tracker is not None
        assert calculator is not None

        # Reset tracker works
        tracker.reset()
        assert tracker._previous_round_responses == {}

    @pytest.mark.asyncio
    async def test_tts_integration_singleton_pattern(self):
        """Test TTS integration singleton management."""
        from aragora.server.stream.tts_integration import (
            get_tts_integration,
            set_tts_integration,
            init_tts_integration,
            TTSIntegration,
        )

        # Clear any existing singleton
        set_tts_integration(None)

        # Get returns None initially
        assert get_tts_integration() is None

        # Init creates singleton
        integration = init_tts_integration()
        assert get_tts_integration() is integration

        # Subsequent calls return same instance
        integration2 = init_tts_integration()
        assert integration2 is integration

        # Clean up
        set_tts_integration(None)

    @pytest.mark.asyncio
    async def test_metrics_recorded_during_operations(self):
        """Test that Prometheus metrics are recorded during operations."""
        from unittest.mock import patch

        # Test TTS metrics
        with patch("aragora.server.stream.tts_integration.record_tts_synthesis") as mock_tts:
            with patch("aragora.server.stream.tts_integration.record_tts_latency") as mock_latency:
                from aragora.server.stream.tts_integration import TTSIntegration

                handler = Mock()
                handler.is_tts_available = True
                handler.has_voice_session = Mock(return_value=True)
                handler.synthesize_agent_message = AsyncMock(return_value=1)

                integration = TTSIntegration(voice_handler=handler)

                event = Mock()
                event.debate_id = "test-123"
                event.data = {
                    "agent": "claude",
                    "content": "Test message",
                    "enable_tts": True,
                }

                await integration._handle_agent_message(event)

                mock_tts.assert_called_once()
                mock_latency.assert_called_once()

        # Test convergence metrics
        with patch("aragora.debate.phases.convergence_tracker.record_convergence_check") as mock_conv:
            from aragora.debate.phases.convergence_tracker import DebateConvergenceTracker

            detector = Mock()
            result = Mock()
            result.converged = False
            result.status = "checking"
            result.avg_similarity = 0.5
            result.per_agent_similarity = {}
            detector.check_convergence = Mock(return_value=result)

            tracker = DebateConvergenceTracker(convergence_detector=detector)

            ctx = Mock()
            ctx.proposals = {"a": "P1"}
            ctx.result = Mock()
            ctx.result.convergence_status = None
            ctx.result.convergence_similarity = 0.0
            ctx.result.per_agent_similarity = {}
            ctx.debate_id = "test"

            # First call stores
            tracker.check_convergence(ctx, 1)
            # Second call compares
            tracker.check_convergence(ctx, 2)

            mock_conv.assert_called()
