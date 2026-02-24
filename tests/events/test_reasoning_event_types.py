"""Tests for reasoning visibility event types (agent_thinking, agent_evidence, agent_confidence).

Verifies that the new reasoning event types are properly defined in the
StreamEventType enum and can be used to construct StreamEvent instances.
"""

import pytest

from aragora.events.types import StreamEvent, StreamEventType


class TestReasoningEventTypes:
    """Tests for the three new reasoning visibility event types."""

    def test_agent_thinking_enum_exists(self):
        assert StreamEventType.AGENT_THINKING.value == "agent_thinking"

    def test_agent_evidence_enum_exists(self):
        assert StreamEventType.AGENT_EVIDENCE.value == "agent_evidence"

    def test_agent_confidence_enum_exists(self):
        assert StreamEventType.AGENT_CONFIDENCE.value == "agent_confidence"

    def test_agent_thinking_creates_event(self):
        event = StreamEvent(
            type=StreamEventType.AGENT_THINKING,
            data={
                "agent": "claude",
                "thinking": "Analyzing the trade-offs between approaches",
                "step": 1,
                "phase": "analysis",
            },
        )
        assert event.type == StreamEventType.AGENT_THINKING
        assert event.data["agent"] == "claude"
        assert event.data["thinking"] == "Analyzing the trade-offs between approaches"
        assert event.data["step"] == 1

    def test_agent_evidence_creates_event(self):
        event = StreamEvent(
            type=StreamEventType.AGENT_EVIDENCE,
            data={
                "agent": "gpt-4",
                "sources": [
                    {
                        "title": "Research Paper A",
                        "url": "https://example.com/a",
                        "relevance": 0.95,
                    },
                    {"title": "Research Paper B", "relevance": 0.72},
                ],
                "query": "rate limiter design patterns",
            },
        )
        assert event.type == StreamEventType.AGENT_EVIDENCE
        assert len(event.data["sources"]) == 2
        assert event.data["sources"][0]["title"] == "Research Paper A"
        assert event.data["sources"][0]["relevance"] == 0.95

    def test_agent_confidence_creates_event(self):
        event = StreamEvent(
            type=StreamEventType.AGENT_CONFIDENCE,
            data={
                "agent": "claude",
                "confidence": 0.87,
                "previous": 0.72,
                "reason": "New evidence supports initial hypothesis",
            },
        )
        assert event.type == StreamEventType.AGENT_CONFIDENCE
        assert event.data["confidence"] == 0.87
        assert event.data["previous"] == 0.72

    def test_reasoning_events_serialize_to_dict(self):
        """Reasoning events should be serializable via to_dict()."""
        event = StreamEvent(
            type=StreamEventType.AGENT_THINKING,
            data={"agent": "claude", "thinking": "Step 1"},
        )
        d = event.to_dict()
        assert d["type"] == "agent_thinking"
        assert d["data"]["agent"] == "claude"
        assert "timestamp" in d

    def test_all_reasoning_types_in_enum(self):
        """All three reasoning types should be accessible members."""
        reasoning_types = [
            StreamEventType.AGENT_THINKING,
            StreamEventType.AGENT_EVIDENCE,
            StreamEventType.AGENT_CONFIDENCE,
        ]
        for t in reasoning_types:
            assert isinstance(t, StreamEventType)
            assert t.value.startswith("agent_")

    def test_reasoning_types_distinct_from_token_types(self):
        """Reasoning types should be distinct from token streaming types."""
        token_types = {
            StreamEventType.TOKEN_START,
            StreamEventType.TOKEN_DELTA,
            StreamEventType.TOKEN_END,
        }
        reasoning_types = {
            StreamEventType.AGENT_THINKING,
            StreamEventType.AGENT_EVIDENCE,
            StreamEventType.AGENT_CONFIDENCE,
        }
        assert token_types.isdisjoint(reasoning_types)
