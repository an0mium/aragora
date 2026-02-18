"""Tests for aragora.debate.protocol_messages.messages — Protocol Message Types."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest

from aragora.debate.protocol_messages.messages import (
    AgentEventPayload,
    ConsensusPayload,
    CritiquePayload,
    PayloadType,
    ProposalPayload,
    ProtocolMessage,
    ProtocolMessageType,
    ProtocolPayload,
    RoundPayload,
    VotePayload,
    agent_event_message,
    consensus_message,
    critique_message,
    proposal_message,
    round_message,
    vote_message,
)


# ---------------------------------------------------------------------------
# ProtocolMessageType enum
# ---------------------------------------------------------------------------


class TestProtocolMessageType:
    def test_all_members(self):
        expected = {
            "PROPOSAL_SUBMITTED", "PROPOSAL_REVISED",
            "CRITIQUE_SUBMITTED", "REBUTTAL_SUBMITTED",
            "REVISION_SUBMITTED",
            "VOTE_CAST", "VOTE_CHANGED",
            "CONSENSUS_REACHED", "CONSENSUS_FAILED",
            "ROUND_STARTED", "ROUND_COMPLETED",
            "AGENT_JOINED", "AGENT_LEFT", "AGENT_FAILED", "AGENT_REPLACED",
            "DEBATE_STARTED", "DEBATE_COMPLETED", "DEBATE_CANCELLED",
            "DEADLOCK_DETECTED", "RECOVERY_INITIATED", "RECOVERY_COMPLETED",
        }
        assert {m.name for m in ProtocolMessageType} == expected

    def test_values_are_snake_case(self):
        for m in ProtocolMessageType:
            assert m.value == m.name.lower()


# ---------------------------------------------------------------------------
# ProtocolPayload base
# ---------------------------------------------------------------------------


class TestProtocolPayload:
    def test_to_dict(self):
        p = ProtocolPayload()
        assert p.to_dict() == {}

    def test_from_dict(self):
        p = ProtocolPayload.from_dict({"unknown_field": 42})
        assert isinstance(p, ProtocolPayload)


# ---------------------------------------------------------------------------
# ProposalPayload
# ---------------------------------------------------------------------------


class TestProposalPayload:
    def test_fields(self):
        p = ProposalPayload(
            proposal_id="p1",
            content="test",
            model="claude",
            round_number=1,
            token_count=100,
            latency_ms=500.0,
        )
        assert p.proposal_id == "p1"
        assert p.content == "test"
        assert p.model == "claude"
        assert p.round_number == 1
        assert p.token_count == 100
        assert p.latency_ms == 500.0
        assert p.metadata == {}

    def test_to_dict_omits_none(self):
        p = ProposalPayload(proposal_id="p1", content="test", model="claude", round_number=1)
        d = p.to_dict()
        assert "token_count" not in d
        assert "latency_ms" not in d
        assert d["proposal_id"] == "p1"

    def test_from_dict(self):
        data = {"proposal_id": "p1", "content": "test", "model": "claude", "round_number": 1}
        p = ProposalPayload.from_dict(data)
        assert p.proposal_id == "p1"


# ---------------------------------------------------------------------------
# CritiquePayload
# ---------------------------------------------------------------------------


class TestCritiquePayload:
    def test_defaults(self):
        c = CritiquePayload(
            critique_id="c1", proposal_id="p1", content="weak argument",
            model="gpt", round_number=1,
        )
        assert c.critique_type == "standard"
        assert c.severity is None
        assert c.addressed_issues == []

    def test_custom(self):
        c = CritiquePayload(
            critique_id="c1", proposal_id="p1", content="rebuttal",
            model="gpt", round_number=2, critique_type="rebuttal",
            severity="major", addressed_issues=["issue1"],
        )
        assert c.critique_type == "rebuttal"
        assert c.severity == "major"
        assert c.addressed_issues == ["issue1"]


# ---------------------------------------------------------------------------
# VotePayload
# ---------------------------------------------------------------------------


class TestVotePayload:
    def test_fields(self):
        v = VotePayload(
            vote_id="v1", proposal_id="p1",
            vote_type="support", confidence=0.9,
        )
        assert v.vote_type == "support"
        assert v.confidence == 0.9
        assert v.rationale is None
        assert v.is_human is False

    def test_human_vote(self):
        v = VotePayload(
            vote_id="v2", proposal_id="p1",
            vote_type="oppose", confidence=0.8,
            is_human=True, rationale="disagree",
        )
        assert v.is_human is True
        assert v.rationale == "disagree"


# ---------------------------------------------------------------------------
# ConsensusPayload
# ---------------------------------------------------------------------------


class TestConsensusPayload:
    def test_fields(self):
        c = ConsensusPayload(
            consensus_id="con1",
            winning_proposal_id="p1",
            final_answer="use AES-256",
            confidence=0.95,
            rounds_taken=3,
        )
        assert c.final_answer == "use AES-256"
        assert c.confidence == 0.95
        assert c.vote_distribution == {}

    def test_no_winner(self):
        c = ConsensusPayload(
            consensus_id="con2",
            winning_proposal_id=None,
            final_answer="no consensus",
            confidence=0.3,
        )
        assert c.winning_proposal_id is None


# ---------------------------------------------------------------------------
# RoundPayload
# ---------------------------------------------------------------------------


class TestRoundPayload:
    def test_fields(self):
        r = RoundPayload(round_number=2, phase="critique", critique_count=5)
        assert r.round_number == 2
        assert r.phase == "critique"
        assert r.critique_count == 5
        assert r.proposal_count == 0
        assert r.duration_ms is None


# ---------------------------------------------------------------------------
# AgentEventPayload
# ---------------------------------------------------------------------------


class TestAgentEventPayload:
    def test_fields(self):
        a = AgentEventPayload(
            agent_id="a1", agent_name="claude",
            model="opus", role="proposer",
        )
        assert a.agent_name == "claude"
        assert a.role == "proposer"
        assert a.reason is None

    def test_replacement(self):
        a = AgentEventPayload(
            agent_id="a1", agent_name="claude",
            model="opus", role="proposer",
            reason="timeout", replacement_id="a2",
        )
        assert a.reason == "timeout"
        assert a.replacement_id == "a2"


# ---------------------------------------------------------------------------
# ProtocolMessage
# ---------------------------------------------------------------------------


class TestProtocolMessage:
    def test_create_minimal(self):
        m = ProtocolMessage(
            message_type=ProtocolMessageType.DEBATE_STARTED,
            debate_id="d1",
        )
        assert m.message_type == ProtocolMessageType.DEBATE_STARTED
        assert m.debate_id == "d1"
        assert m.agent_id is None
        assert m.round_number is None
        assert isinstance(m.message_id, str)
        assert isinstance(m.timestamp, datetime)

    def test_to_dict_minimal(self):
        m = ProtocolMessage(
            message_type=ProtocolMessageType.DEBATE_STARTED,
            debate_id="d1",
        )
        d = m.to_dict()
        assert d["message_type"] == "debate_started"
        assert d["debate_id"] == "d1"
        assert "message_id" in d
        assert "timestamp" in d
        assert "agent_id" not in d  # None fields omitted

    def test_to_dict_full(self):
        m = ProtocolMessage(
            message_type=ProtocolMessageType.PROPOSAL_SUBMITTED,
            debate_id="d1",
            agent_id="a1",
            round_number=1,
            correlation_id="corr-1",
            parent_message_id="parent-1",
            metadata={"key": "val"},
            payload=ProposalPayload(
                proposal_id="p1", content="test", model="claude", round_number=1,
            ),
        )
        d = m.to_dict()
        assert d["agent_id"] == "a1"
        assert d["round_number"] == 1
        assert d["correlation_id"] == "corr-1"
        assert d["parent_message_id"] == "parent-1"
        assert d["metadata"] == {"key": "val"}
        assert d["payload"]["proposal_id"] == "p1"

    def test_to_json(self):
        m = ProtocolMessage(
            message_type=ProtocolMessageType.DEBATE_STARTED,
            debate_id="d1",
        )
        j = m.to_json()
        parsed = json.loads(j)
        assert parsed["debate_id"] == "d1"

    def test_from_dict(self):
        data = {
            "message_type": "debate_started",
            "debate_id": "d1",
            "agent_id": "a1",
            "round_number": 2,
            "correlation_id": "corr-1",
            "metadata": {"k": "v"},
        }
        m = ProtocolMessage.from_dict(data)
        assert m.message_type == ProtocolMessageType.DEBATE_STARTED
        assert m.debate_id == "d1"
        assert m.agent_id == "a1"
        assert m.round_number == 2
        assert m.correlation_id == "corr-1"

    def test_from_dict_with_timestamp_str(self):
        ts = "2026-02-17T20:00:00+00:00"
        data = {
            "message_type": "debate_started",
            "debate_id": "d1",
            "timestamp": ts,
        }
        m = ProtocolMessage.from_dict(data)
        assert m.timestamp.year == 2026

    def test_from_dict_with_z_timestamp(self):
        data = {
            "message_type": "debate_started",
            "debate_id": "d1",
            "timestamp": "2026-02-17T20:00:00Z",
        }
        m = ProtocolMessage.from_dict(data)
        assert m.timestamp.tzinfo is not None

    def test_from_dict_missing_timestamp(self):
        data = {"message_type": "debate_started", "debate_id": "d1"}
        m = ProtocolMessage.from_dict(data)
        assert isinstance(m.timestamp, datetime)

    def test_repr(self):
        m = ProtocolMessage(
            message_type=ProtocolMessageType.DEBATE_STARTED,
            debate_id="d1234567890",
            agent_id="claude",
            round_number=1,
        )
        r = repr(m)
        assert "debate_started" in r
        assert "d1234567" in r
        assert "claude" in r

    def test_roundtrip(self):
        original = ProtocolMessage(
            message_type=ProtocolMessageType.VOTE_CAST,
            debate_id="d1",
            agent_id="claude",
            round_number=3,
            metadata={"extra": True},
        )
        d = original.to_dict()
        restored = ProtocolMessage.from_dict(d)
        assert restored.message_type == original.message_type
        assert restored.debate_id == original.debate_id
        assert restored.agent_id == original.agent_id
        assert restored.round_number == original.round_number


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


class TestProposalMessage:
    def test_creates_correct_type(self):
        m = proposal_message(
            debate_id="d1", agent_id="a1", proposal_id="p1",
            content="my proposal", model="claude", round_number=1,
        )
        assert m.message_type == ProtocolMessageType.PROPOSAL_SUBMITTED
        assert m.debate_id == "d1"
        assert m.agent_id == "a1"
        assert m.round_number == 1
        assert isinstance(m.payload, ProposalPayload)
        assert m.payload.content == "my proposal"

    def test_with_kwargs(self):
        m = proposal_message(
            debate_id="d1", agent_id="a1", proposal_id="p1",
            content="test", model="claude", round_number=1,
            token_count=500, latency_ms=1200.0,
        )
        assert m.payload.token_count == 500
        assert m.payload.latency_ms == 1200.0


class TestCritiqueMessage:
    def test_creates_correct_type(self):
        m = critique_message(
            debate_id="d1", agent_id="a2", critique_id="c1",
            proposal_id="p1", content="weak point",
            model="gpt", round_number=1,
        )
        assert m.message_type == ProtocolMessageType.CRITIQUE_SUBMITTED
        assert m.parent_message_id == "p1"
        assert isinstance(m.payload, CritiquePayload)
        assert m.payload.content == "weak point"


class TestVoteMessage:
    def test_creates_correct_type(self):
        m = vote_message(
            debate_id="d1", agent_id="a1", vote_id="v1",
            proposal_id="p1", vote_type="support", confidence=0.9,
        )
        assert m.message_type == ProtocolMessageType.VOTE_CAST
        assert m.parent_message_id == "p1"
        assert isinstance(m.payload, VotePayload)
        assert m.payload.vote_type == "support"
        assert m.payload.confidence == 0.9


class TestConsensusMessage:
    def test_creates_correct_type(self):
        m = consensus_message(
            debate_id="d1", consensus_id="con1",
            winning_proposal_id="p1",
            final_answer="use AES-256",
            confidence=0.95, rounds_taken=3,
        )
        assert m.message_type == ProtocolMessageType.CONSENSUS_REACHED
        assert isinstance(m.payload, ConsensusPayload)
        assert m.payload.final_answer == "use AES-256"
        assert m.payload.rounds_taken == 3

    def test_no_winner(self):
        m = consensus_message(
            debate_id="d1", consensus_id="con2",
            winning_proposal_id=None,
            final_answer="no consensus",
            confidence=0.3, rounds_taken=5,
        )
        assert m.payload.winning_proposal_id is None


class TestRoundMessage:
    def test_started(self):
        m = round_message(debate_id="d1", round_number=1, phase="proposal", started=True)
        assert m.message_type == ProtocolMessageType.ROUND_STARTED
        assert isinstance(m.payload, RoundPayload)
        assert m.payload.round_number == 1
        assert m.payload.phase == "proposal"

    def test_completed(self):
        m = round_message(
            debate_id="d1", round_number=2, phase="voting",
            started=False, duration_ms=5000.0,
        )
        assert m.message_type == ProtocolMessageType.ROUND_COMPLETED
        assert m.payload.duration_ms == 5000.0


class TestAgentEventMessage:
    def test_joined(self):
        m = agent_event_message(
            debate_id="d1", agent_id="a1", agent_name="claude",
            model="opus", role="proposer",
            event_type=ProtocolMessageType.AGENT_JOINED,
        )
        assert m.message_type == ProtocolMessageType.AGENT_JOINED
        assert isinstance(m.payload, AgentEventPayload)
        assert m.payload.agent_name == "claude"

    def test_failed(self):
        m = agent_event_message(
            debate_id="d1", agent_id="a1", agent_name="gpt",
            model="gpt-4", role="critic",
            event_type=ProtocolMessageType.AGENT_FAILED,
            reason="timeout",
        )
        assert m.message_type == ProtocolMessageType.AGENT_FAILED
        assert m.payload.reason == "timeout"

    def test_replaced(self):
        m = agent_event_message(
            debate_id="d1", agent_id="a1", agent_name="claude",
            model="opus", role="proposer",
            event_type=ProtocolMessageType.AGENT_REPLACED,
            replacement_id="a2",
        )
        assert m.payload.replacement_id == "a2"
