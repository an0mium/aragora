"""
Tests for aragora.debate.protocol - Protocol Messages and Store.

Tests cover:
- Protocol message creation and serialization
- Message store CRUD operations
- Query filters
- Handler registry
- Factory functions
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.debate.protocol_messages import (
    ProtocolMessage,
    ProtocolMessageType,
    ProtocolMessageStore,
    ProtocolHandler,
    ProtocolHandlerRegistry,
    ProposalPayload,
    CritiquePayload,
    VotePayload,
    ConsensusPayload,
    RoundPayload,
    AgentEventPayload,
)
from aragora.debate.protocol_messages.messages import (
    proposal_message,
    critique_message,
    vote_message,
    consensus_message,
    round_message,
    agent_event_message,
)
from aragora.debate.protocol_messages.store import QueryFilters


# ===========================================================================
# Message Creation Tests
# ===========================================================================


class TestProtocolMessage:
    """Tests for ProtocolMessage dataclass."""

    def test_create_basic_message(self):
        """Test creating a basic protocol message."""
        msg = ProtocolMessage(
            message_type=ProtocolMessageType.PROPOSAL_SUBMITTED,
            debate_id="debate-123",
            agent_id="claude-opus",
            round_number=1,
        )

        assert msg.message_type == ProtocolMessageType.PROPOSAL_SUBMITTED
        assert msg.debate_id == "debate-123"
        assert msg.agent_id == "claude-opus"
        assert msg.round_number == 1
        assert msg.message_id is not None
        assert msg.timestamp is not None

    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        msg = ProtocolMessage(
            message_type=ProtocolMessageType.VOTE_CAST,
            debate_id="debate-456",
            agent_id="gpt-4",
            correlation_id="trace-123",
        )

        d = msg.to_dict()

        assert d["message_type"] == "vote_cast"
        assert d["debate_id"] == "debate-456"
        assert d["agent_id"] == "gpt-4"
        assert d["correlation_id"] == "trace-123"
        assert "timestamp" in d

    def test_message_to_json(self):
        """Test converting message to JSON."""
        msg = ProtocolMessage(
            message_type=ProtocolMessageType.CONSENSUS_REACHED,
            debate_id="debate-789",
        )

        json_str = msg.to_json()
        parsed = json.loads(json_str)

        assert parsed["message_type"] == "consensus_reached"
        assert parsed["debate_id"] == "debate-789"

    def test_message_from_dict(self):
        """Test creating message from dictionary."""
        data = {
            "message_id": "msg-123",
            "message_type": "round_started",
            "debate_id": "debate-abc",
            "round_number": 2,
            "timestamp": "2024-01-15T10:30:00+00:00",
        }

        msg = ProtocolMessage.from_dict(data)

        assert msg.message_id == "msg-123"
        assert msg.message_type == ProtocolMessageType.ROUND_STARTED
        assert msg.debate_id == "debate-abc"
        assert msg.round_number == 2

    def test_message_with_payload(self):
        """Test message with typed payload."""
        payload = ProposalPayload(
            proposal_id="prop-1",
            content="My proposal content",
            model="claude-opus-4",
            round_number=1,
            token_count=150,
        )

        msg = ProtocolMessage(
            message_type=ProtocolMessageType.PROPOSAL_SUBMITTED,
            debate_id="debate-123",
            agent_id="proposer-1",
            payload=payload,
        )

        d = msg.to_dict()
        assert d["payload"]["proposal_id"] == "prop-1"
        assert d["payload"]["content"] == "My proposal content"
        assert d["payload"]["token_count"] == 150


# ===========================================================================
# Payload Tests
# ===========================================================================


class TestProtocolPayloads:
    """Tests for typed payload dataclasses."""

    def test_proposal_payload(self):
        """Test ProposalPayload creation and serialization."""
        payload = ProposalPayload(
            proposal_id="prop-123",
            content="Design a distributed cache",
            model="claude-opus-4",
            round_number=1,
            latency_ms=1234.5,
            metadata={"source": "agent"},
        )

        d = payload.to_dict()
        assert d["proposal_id"] == "prop-123"
        assert d["latency_ms"] == 1234.5
        assert d["metadata"]["source"] == "agent"

    def test_critique_payload(self):
        """Test CritiquePayload creation."""
        payload = CritiquePayload(
            critique_id="crit-456",
            proposal_id="prop-123",
            content="This proposal lacks error handling",
            model="gpt-4",
            round_number=1,
            severity="major",
            addressed_issues=["error_handling", "edge_cases"],
        )

        d = payload.to_dict()
        assert d["critique_id"] == "crit-456"
        assert d["severity"] == "major"
        assert len(d["addressed_issues"]) == 2

    def test_vote_payload(self):
        """Test VotePayload creation."""
        payload = VotePayload(
            vote_id="vote-789",
            proposal_id="prop-123",
            vote_type="support",
            confidence=0.85,
            rationale="Well-reasoned approach",
        )

        d = payload.to_dict()
        assert d["vote_type"] == "support"
        assert d["confidence"] == 0.85

    def test_consensus_payload(self):
        """Test ConsensusPayload creation."""
        payload = ConsensusPayload(
            consensus_id="cons-001",
            winning_proposal_id="prop-123",
            final_answer="Implement Redis with write-through",
            confidence=0.92,
            vote_distribution={"support": 3, "oppose": 1},
            rounds_taken=3,
        )

        d = payload.to_dict()
        assert d["confidence"] == 0.92
        assert d["rounds_taken"] == 3
        assert d["vote_distribution"]["support"] == 3


# ===========================================================================
# Factory Function Tests
# ===========================================================================


class TestFactoryFunctions:
    """Tests for message factory functions."""

    def test_proposal_message_factory(self):
        """Test proposal_message factory."""
        msg = proposal_message(
            debate_id="debate-123",
            agent_id="proposer-1",
            proposal_id="prop-1",
            content="My proposal",
            model="claude-opus-4",
            round_number=1,
            token_count=100,
        )

        assert msg.message_type == ProtocolMessageType.PROPOSAL_SUBMITTED
        assert msg.debate_id == "debate-123"
        assert msg.payload.proposal_id == "prop-1"
        assert msg.payload.token_count == 100

    def test_critique_message_factory(self):
        """Test critique_message factory."""
        msg = critique_message(
            debate_id="debate-123",
            agent_id="critic-1",
            critique_id="crit-1",
            proposal_id="prop-1",
            content="Missing error handling",
            model="gpt-4",
            round_number=1,
        )

        assert msg.message_type == ProtocolMessageType.CRITIQUE_SUBMITTED
        assert msg.parent_message_id == "prop-1"  # Links to proposal
        assert msg.payload.critique_id == "crit-1"

    def test_vote_message_factory(self):
        """Test vote_message factory."""
        msg = vote_message(
            debate_id="debate-123",
            agent_id="voter-1",
            vote_id="vote-1",
            proposal_id="prop-1",
            vote_type="support",
            confidence=0.9,
        )

        assert msg.message_type == ProtocolMessageType.VOTE_CAST
        assert msg.payload.vote_type == "support"
        assert msg.payload.confidence == 0.9

    def test_consensus_message_factory(self):
        """Test consensus_message factory."""
        msg = consensus_message(
            debate_id="debate-123",
            consensus_id="cons-1",
            winning_proposal_id="prop-1",
            final_answer="Use Redis cluster",
            confidence=0.95,
            rounds_taken=2,
        )

        assert msg.message_type == ProtocolMessageType.CONSENSUS_REACHED
        assert msg.payload.final_answer == "Use Redis cluster"
        assert msg.payload.rounds_taken == 2

    def test_round_message_factory_start(self):
        """Test round_message factory for round start."""
        msg = round_message(
            debate_id="debate-123",
            round_number=2,
            phase="proposal",
            started=True,
        )

        assert msg.message_type == ProtocolMessageType.ROUND_STARTED
        assert msg.round_number == 2
        assert msg.payload.phase == "proposal"

    def test_round_message_factory_complete(self):
        """Test round_message factory for round completion."""
        msg = round_message(
            debate_id="debate-123",
            round_number=2,
            phase="voting",
            started=False,
            proposal_count=3,
            critique_count=6,
        )

        assert msg.message_type == ProtocolMessageType.ROUND_COMPLETED
        assert msg.payload.proposal_count == 3
        assert msg.payload.critique_count == 6

    def test_agent_event_message_factory(self):
        """Test agent_event_message factory."""
        msg = agent_event_message(
            debate_id="debate-123",
            agent_id="agent-1",
            agent_name="Claude Opus",
            model="claude-opus-4",
            role="proposer",
            event_type=ProtocolMessageType.AGENT_JOINED,
        )

        assert msg.message_type == ProtocolMessageType.AGENT_JOINED
        assert msg.payload.agent_name == "Claude Opus"
        assert msg.payload.role == "proposer"


# ===========================================================================
# Message Store Tests
# ===========================================================================


class TestProtocolMessageStore:
    """Tests for ProtocolMessageStore."""

    @pytest.fixture
    def store(self):
        """Create an in-memory message store."""
        return ProtocolMessageStore()  # Uses :memory: by default

    @pytest.fixture
    def file_store(self, tmp_path):
        """Create a file-backed message store."""
        db_path = str(tmp_path / "protocol.db")
        return ProtocolMessageStore(db_path)

    @pytest.mark.asyncio
    async def test_record_and_get(self, store):
        """Test recording and retrieving a message."""
        msg = ProtocolMessage(
            message_type=ProtocolMessageType.DEBATE_STARTED,
            debate_id="debate-123",
        )

        msg_id = await store.record(msg)
        assert msg_id == msg.message_id

        retrieved = await store.get(msg_id)
        assert retrieved is not None
        assert retrieved.debate_id == "debate-123"
        assert retrieved.message_type == ProtocolMessageType.DEBATE_STARTED

    @pytest.mark.asyncio
    async def test_query_by_debate_id(self, store):
        """Test querying messages by debate ID."""
        # Record messages for two debates
        await store.record(
            ProtocolMessage(
                message_type=ProtocolMessageType.PROPOSAL_SUBMITTED,
                debate_id="debate-1",
            )
        )
        await store.record(
            ProtocolMessage(
                message_type=ProtocolMessageType.PROPOSAL_SUBMITTED,
                debate_id="debate-2",
            )
        )
        await store.record(
            ProtocolMessage(
                message_type=ProtocolMessageType.CRITIQUE_SUBMITTED,
                debate_id="debate-1",
            )
        )

        # Query for debate-1
        filters = QueryFilters(debate_id="debate-1")
        results = await store.query(filters)

        assert len(results) == 2
        assert all(m.debate_id == "debate-1" for m in results)

    @pytest.mark.asyncio
    async def test_query_by_message_type(self, store):
        """Test querying messages by type."""
        await store.record(
            ProtocolMessage(
                message_type=ProtocolMessageType.PROPOSAL_SUBMITTED,
                debate_id="debate-1",
            )
        )
        await store.record(
            ProtocolMessage(
                message_type=ProtocolMessageType.VOTE_CAST,
                debate_id="debate-1",
            )
        )
        await store.record(
            ProtocolMessage(
                message_type=ProtocolMessageType.PROPOSAL_SUBMITTED,
                debate_id="debate-1",
            )
        )

        filters = QueryFilters(
            debate_id="debate-1",
            message_type=ProtocolMessageType.PROPOSAL_SUBMITTED,
        )
        results = await store.query(filters)

        assert len(results) == 2
        assert all(m.message_type == ProtocolMessageType.PROPOSAL_SUBMITTED for m in results)

    @pytest.mark.asyncio
    async def test_query_by_round(self, store):
        """Test querying messages by round number."""
        await store.record(
            ProtocolMessage(
                message_type=ProtocolMessageType.PROPOSAL_SUBMITTED,
                debate_id="debate-1",
                round_number=1,
            )
        )
        await store.record(
            ProtocolMessage(
                message_type=ProtocolMessageType.PROPOSAL_SUBMITTED,
                debate_id="debate-1",
                round_number=2,
            )
        )
        await store.record(
            ProtocolMessage(
                message_type=ProtocolMessageType.CRITIQUE_SUBMITTED,
                debate_id="debate-1",
                round_number=1,
            )
        )

        filters = QueryFilters(debate_id="debate-1", round_number=1)
        results = await store.query(filters)

        assert len(results) == 2
        assert all(m.round_number == 1 for m in results)

    @pytest.mark.asyncio
    async def test_get_debate_timeline(self, store):
        """Test getting full debate timeline."""
        debate_id = "timeline-test"

        # Add messages in random order
        await store.record(
            ProtocolMessage(
                message_type=ProtocolMessageType.VOTE_CAST,
                debate_id=debate_id,
                round_number=1,
            )
        )
        await store.record(
            ProtocolMessage(
                message_type=ProtocolMessageType.DEBATE_STARTED,
                debate_id=debate_id,
            )
        )
        await store.record(
            ProtocolMessage(
                message_type=ProtocolMessageType.PROPOSAL_SUBMITTED,
                debate_id=debate_id,
                round_number=1,
            )
        )

        timeline = await store.get_debate_timeline(debate_id)

        assert len(timeline) == 3
        # Should be in chronological order
        for i in range(len(timeline) - 1):
            assert timeline[i].timestamp <= timeline[i + 1].timestamp

    @pytest.mark.asyncio
    async def test_count(self, store):
        """Test counting messages."""
        for i in range(5):
            await store.record(
                ProtocolMessage(
                    message_type=ProtocolMessageType.PROPOSAL_SUBMITTED,
                    debate_id="debate-count",
                )
            )

        count = await store.count(QueryFilters(debate_id="debate-count"))
        assert count == 5

    @pytest.mark.asyncio
    async def test_delete_debate(self, store):
        """Test deleting all messages for a debate."""
        debate_id = "to-delete"

        for i in range(3):
            await store.record(
                ProtocolMessage(
                    message_type=ProtocolMessageType.PROPOSAL_SUBMITTED,
                    debate_id=debate_id,
                )
            )

        deleted = await store.delete_debate(debate_id)
        assert deleted == 3

        count = await store.count(QueryFilters(debate_id=debate_id))
        assert count == 0

    @pytest.mark.asyncio
    async def test_file_persistence(self, file_store, tmp_path):
        """Test that file store persists data."""
        db_path = str(tmp_path / "protocol.db")

        # Record a message
        await file_store.record(
            ProtocolMessage(
                message_type=ProtocolMessageType.DEBATE_STARTED,
                debate_id="persist-test",
            )
        )
        file_store.close()

        # Create new store and verify data
        new_store = ProtocolMessageStore(db_path)
        count = await new_store.count(QueryFilters(debate_id="persist-test"))
        assert count == 1
        new_store.close()


# ===========================================================================
# Handler Registry Tests
# ===========================================================================


class TestProtocolHandlerRegistry:
    """Tests for ProtocolHandlerRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a fresh handler registry."""
        return ProtocolHandlerRegistry()

    @pytest.mark.asyncio
    async def test_register_and_dispatch(self, registry):
        """Test registering and dispatching to a handler."""
        received = []

        async def handler(msg: ProtocolMessage):
            received.append(msg)

        registry.register(ProtocolMessageType.PROPOSAL_SUBMITTED, handler)

        msg = ProtocolMessage(
            message_type=ProtocolMessageType.PROPOSAL_SUBMITTED,
            debate_id="test-123",
        )
        count = await registry.dispatch(msg)

        assert count == 1
        assert len(received) == 1
        assert received[0].debate_id == "test-123"

    @pytest.mark.asyncio
    async def test_multiple_handlers(self, registry):
        """Test dispatching to multiple handlers."""
        handler1_called = []
        handler2_called = []

        async def handler1(msg: ProtocolMessage):
            handler1_called.append(msg)

        async def handler2(msg: ProtocolMessage):
            handler2_called.append(msg)

        registry.register(ProtocolMessageType.VOTE_CAST, handler1)
        registry.register(ProtocolMessageType.VOTE_CAST, handler2)

        msg = ProtocolMessage(
            message_type=ProtocolMessageType.VOTE_CAST,
            debate_id="test-456",
        )
        count = await registry.dispatch(msg)

        assert count == 2
        assert len(handler1_called) == 1
        assert len(handler2_called) == 1

    @pytest.mark.asyncio
    async def test_handler_priority(self, registry):
        """Test that handlers run in priority order."""
        order = []

        async def high_priority(msg: ProtocolMessage):
            order.append("high")

        async def low_priority(msg: ProtocolMessage):
            order.append("low")

        registry.register(ProtocolMessageType.CONSENSUS_REACHED, low_priority, priority=200)
        registry.register(ProtocolMessageType.CONSENSUS_REACHED, high_priority, priority=50)

        await registry.dispatch(
            ProtocolMessage(
                message_type=ProtocolMessageType.CONSENSUS_REACHED,
                debate_id="priority-test",
            )
        )

        assert order == ["high", "low"]

    @pytest.mark.asyncio
    async def test_global_handler(self, registry):
        """Test global handler receives all messages."""
        received = []

        async def global_handler(msg: ProtocolMessage):
            received.append(msg.message_type)

        registry.register_global(global_handler)

        await registry.dispatch(
            ProtocolMessage(
                message_type=ProtocolMessageType.PROPOSAL_SUBMITTED,
                debate_id="test",
            )
        )
        await registry.dispatch(
            ProtocolMessage(
                message_type=ProtocolMessageType.VOTE_CAST,
                debate_id="test",
            )
        )

        assert len(received) == 2
        assert ProtocolMessageType.PROPOSAL_SUBMITTED in received
        assert ProtocolMessageType.VOTE_CAST in received

    @pytest.mark.asyncio
    async def test_handler_error_isolation(self, registry):
        """Test that one handler's error doesn't affect others."""
        successful_calls = []

        async def failing_handler(msg: ProtocolMessage):
            raise ValueError("Handler failed")

        async def working_handler(msg: ProtocolMessage):
            successful_calls.append(msg)

        registry.register(ProtocolMessageType.ROUND_STARTED, failing_handler)
        registry.register(ProtocolMessageType.ROUND_STARTED, working_handler)

        count = await registry.dispatch(
            ProtocolMessage(
                message_type=ProtocolMessageType.ROUND_STARTED,
                debate_id="test",
            )
        )

        # working_handler should still be called
        assert len(successful_calls) == 1
        assert count == 1  # Only successful handlers counted

    @pytest.mark.asyncio
    async def test_class_handler(self, registry):
        """Test class-based handler registration."""
        received = []

        class TestHandler(ProtocolHandler):
            @property
            def message_types(self):
                return [
                    ProtocolMessageType.AGENT_JOINED,
                    ProtocolMessageType.AGENT_LEFT,
                ]

            async def handle(self, message: ProtocolMessage):
                received.append(message.message_type)

        registry.register_handler(TestHandler())

        await registry.dispatch(
            ProtocolMessage(
                message_type=ProtocolMessageType.AGENT_JOINED,
                debate_id="test",
            )
        )
        await registry.dispatch(
            ProtocolMessage(
                message_type=ProtocolMessageType.AGENT_LEFT,
                debate_id="test",
            )
        )
        await registry.dispatch(
            ProtocolMessage(
                message_type=ProtocolMessageType.VOTE_CAST,
                debate_id="test",
            )
        )

        assert len(received) == 2
        assert ProtocolMessageType.AGENT_JOINED in received
        assert ProtocolMessageType.AGENT_LEFT in received

    @pytest.mark.asyncio
    async def test_concurrent_dispatch(self, registry):
        """Test concurrent dispatch to handlers."""
        import asyncio

        call_times = []

        async def slow_handler(msg: ProtocolMessage):
            await asyncio.sleep(0.1)
            call_times.append(asyncio.get_event_loop().time())

        async def fast_handler(msg: ProtocolMessage):
            call_times.append(asyncio.get_event_loop().time())

        registry.register(ProtocolMessageType.DEBATE_STARTED, slow_handler)
        registry.register(ProtocolMessageType.DEBATE_STARTED, fast_handler)

        start = asyncio.get_event_loop().time()
        await registry.dispatch_concurrent(
            ProtocolMessage(
                message_type=ProtocolMessageType.DEBATE_STARTED,
                debate_id="test",
            )
        )
        elapsed = asyncio.get_event_loop().time() - start

        # With concurrent dispatch, both should complete in ~0.1s, not 0.2s
        assert elapsed < 0.15

    def test_unregister(self, registry):
        """Test unregistering a handler."""

        async def handler(msg: ProtocolMessage):
            pass

        registry.register(ProtocolMessageType.VOTE_CAST, handler)

        removed = registry.unregister(ProtocolMessageType.VOTE_CAST, handler)
        assert removed is True

        handlers = registry.get_handlers(ProtocolMessageType.VOTE_CAST)
        assert len(handlers) == 0

    def test_clear(self, registry):
        """Test clearing all handlers."""

        async def handler(msg: ProtocolMessage):
            pass

        registry.register(ProtocolMessageType.VOTE_CAST, handler)
        registry.register_global(handler)

        registry.clear()

        handlers = registry.get_handlers(ProtocolMessageType.VOTE_CAST)
        assert len(handlers) == 0


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestProtocolIntegration:
    """Integration tests for protocol message system."""

    @pytest.mark.asyncio
    async def test_full_debate_flow(self):
        """Test recording and querying a full debate flow."""
        store = ProtocolMessageStore()
        debate_id = "full-flow-test"

        # Simulate a debate
        await store.record(
            ProtocolMessage(
                message_type=ProtocolMessageType.DEBATE_STARTED,
                debate_id=debate_id,
            )
        )

        for agent_id in ["claude", "gpt-4", "gemini"]:
            await store.record(
                agent_event_message(
                    debate_id=debate_id,
                    agent_id=agent_id,
                    agent_name=agent_id.upper(),
                    model=agent_id,
                    role="proposer",
                    event_type=ProtocolMessageType.AGENT_JOINED,
                )
            )

        await store.record(round_message(debate_id=debate_id, round_number=1, phase="proposal"))

        for i, agent_id in enumerate(["claude", "gpt-4", "gemini"]):
            await store.record(
                proposal_message(
                    debate_id=debate_id,
                    agent_id=agent_id,
                    proposal_id=f"prop-{i}",
                    content=f"Proposal from {agent_id}",
                    model=agent_id,
                    round_number=1,
                )
            )

        await store.record(
            consensus_message(
                debate_id=debate_id,
                consensus_id="cons-1",
                winning_proposal_id="prop-0",
                final_answer="Winning answer",
                confidence=0.9,
                rounds_taken=1,
            )
        )

        await store.record(
            ProtocolMessage(
                message_type=ProtocolMessageType.DEBATE_COMPLETED,
                debate_id=debate_id,
            )
        )

        # Query and verify
        timeline = await store.get_debate_timeline(debate_id)
        assert (
            len(timeline) == 10
        )  # 1 start + 3 joins + 1 round + 3 proposals + 1 consensus + 1 complete

        proposals = await store.query(
            QueryFilters(
                debate_id=debate_id,
                message_type=ProtocolMessageType.PROPOSAL_SUBMITTED,
            )
        )
        assert len(proposals) == 3

        consensus = await store.query(
            QueryFilters(
                debate_id=debate_id,
                message_type=ProtocolMessageType.CONSENSUS_REACHED,
            )
        )
        assert len(consensus) == 1
        assert consensus[0].payload["confidence"] == 0.9
