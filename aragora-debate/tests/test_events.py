"""Tests for event/callback system."""

from __future__ import annotations

import asyncio

import pytest

from aragora_debate.events import DebateEvent, EventEmitter, EventType


class TestEventType:
    def test_all_types_exist(self):
        expected = {
            "debate_start", "debate_end", "round_start", "round_end",
            "proposal", "critique", "vote", "consensus_check",
            "trickster_intervention", "convergence_detected",
        }
        actual = {e.value for e in EventType}
        assert actual == expected

    def test_string_enum(self):
        assert EventType.PROPOSAL == "proposal"
        assert str(EventType.DEBATE_START) == "EventType.DEBATE_START"


class TestDebateEvent:
    def test_creation(self):
        event = DebateEvent(event_type=EventType.PROPOSAL, round_num=2, agent="claude")
        assert event.event_type == EventType.PROPOSAL
        assert event.round_num == 2
        assert event.agent == "claude"
        assert event.data == {}
        assert event.timestamp is not None

    def test_with_data(self):
        event = DebateEvent(
            event_type=EventType.VOTE,
            data={"choice": "agent1", "confidence": 0.9},
        )
        assert event.data["choice"] == "agent1"


class TestEventEmitter:
    def setup_method(self):
        self.emitter = EventEmitter()

    @pytest.mark.asyncio
    async def test_sync_callback(self):
        received = []

        @self.emitter.on(EventType.PROPOSAL)
        def handler(event):
            received.append(event)

        await self.emitter.emit(EventType.PROPOSAL, agent="claude", round_num=1)
        assert len(received) == 1
        assert received[0].agent == "claude"

    @pytest.mark.asyncio
    async def test_async_callback(self):
        received = []

        @self.emitter.on(EventType.DEBATE_START)
        async def handler(event):
            received.append(event)

        await self.emitter.emit(EventType.DEBATE_START, data={"topic": "test"})
        assert len(received) == 1
        assert received[0].data["topic"] == "test"

    @pytest.mark.asyncio
    async def test_multiple_callbacks(self):
        count = [0]

        @self.emitter.on(EventType.ROUND_START)
        def handler1(event):
            count[0] += 1

        @self.emitter.on(EventType.ROUND_START)
        def handler2(event):
            count[0] += 10

        await self.emitter.emit(EventType.ROUND_START)
        assert count[0] == 11

    @pytest.mark.asyncio
    async def test_different_event_types(self):
        proposals = []
        votes = []

        @self.emitter.on(EventType.PROPOSAL)
        def on_proposal(event):
            proposals.append(event)

        @self.emitter.on(EventType.VOTE)
        def on_vote(event):
            votes.append(event)

        await self.emitter.emit(EventType.PROPOSAL, agent="a")
        await self.emitter.emit(EventType.VOTE, agent="b")
        await self.emitter.emit(EventType.PROPOSAL, agent="c")

        assert len(proposals) == 2
        assert len(votes) == 1

    @pytest.mark.asyncio
    async def test_off_removes_callback(self):
        count = [0]

        def handler(event):
            count[0] += 1

        self.emitter.on(EventType.PROPOSAL)(handler)
        await self.emitter.emit(EventType.PROPOSAL)
        assert count[0] == 1

        self.emitter.off(EventType.PROPOSAL, handler)
        await self.emitter.emit(EventType.PROPOSAL)
        assert count[0] == 1  # Not incremented

    @pytest.mark.asyncio
    async def test_emit_returns_event(self):
        event = await self.emitter.emit(EventType.DEBATE_END, round_num=3)
        assert isinstance(event, DebateEvent)
        assert event.round_num == 3
        assert event.event_type == EventType.DEBATE_END

    @pytest.mark.asyncio
    async def test_no_listeners(self):
        # Should not raise
        event = await self.emitter.emit(EventType.CRITIQUE)
        assert event.event_type == EventType.CRITIQUE

    @pytest.mark.asyncio
    async def test_callback_error_doesnt_break_others(self):
        received = []

        @self.emitter.on(EventType.PROPOSAL)
        def bad_handler(event):
            raise ValueError("oops")

        @self.emitter.on(EventType.PROPOSAL)
        def good_handler(event):
            received.append(event)

        await self.emitter.emit(EventType.PROPOSAL)
        assert len(received) == 1

    def test_listener_count(self):
        assert self.emitter.listener_count(EventType.PROPOSAL) == 0

        @self.emitter.on(EventType.PROPOSAL)
        def handler(event):
            pass

        assert self.emitter.listener_count(EventType.PROPOSAL) == 1

    def test_clear(self):
        @self.emitter.on(EventType.PROPOSAL)
        def handler(event):
            pass

        self.emitter.clear()
        assert self.emitter.listener_count(EventType.PROPOSAL) == 0

    @pytest.mark.asyncio
    async def test_emit_with_data(self):
        received = []

        @self.emitter.on(EventType.TRICKSTER_INTERVENTION)
        def handler(event):
            received.append(event)

        await self.emitter.emit(
            EventType.TRICKSTER_INTERVENTION,
            agent="trickster",
            data={"challenge": "Provide evidence"},
        )
        assert received[0].data["challenge"] == "Provide evidence"

    @pytest.mark.asyncio
    async def test_decorator_returns_original_function(self):
        def my_handler(event):
            pass

        result = self.emitter.on(EventType.PROPOSAL)(my_handler)
        assert result is my_handler

    @pytest.mark.asyncio
    async def test_convergence_event(self):
        received = []

        @self.emitter.on(EventType.CONVERGENCE_DETECTED)
        def handler(event):
            received.append(event)

        await self.emitter.emit(
            EventType.CONVERGENCE_DETECTED,
            round_num=2,
            data={"similarity": 0.92, "converged": True},
        )
        assert len(received) == 1
        assert received[0].data["similarity"] == 0.92
