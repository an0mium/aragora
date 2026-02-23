"""Tests for aragora.debate.protocol_messages.handlers — Protocol Handler Registry."""

from __future__ import annotations

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.debate.protocol_messages.handlers import (
    LoggingHandler,
    MetricsHandler,
    ProtocolHandler,
    ProtocolHandlerRegistry,
    get_handler_registry,
    handles,
)
from aragora.debate.protocol_messages.messages import (
    ProtocolMessage,
    ProtocolMessageType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_msg(
    msg_type: ProtocolMessageType = ProtocolMessageType.DEBATE_STARTED,
    debate_id: str = "d1",
    **kwargs,
) -> ProtocolMessage:
    return ProtocolMessage(message_type=msg_type, debate_id=debate_id, **kwargs)


class FakeHandler(ProtocolHandler):
    """Test handler that records calls."""

    def __init__(self, types: list[ProtocolMessageType]):
        self._types = types
        self.calls: list[ProtocolMessage] = []

    @property
    def message_types(self) -> list[ProtocolMessageType]:
        return self._types

    async def handle(self, message: ProtocolMessage) -> None:
        self.calls.append(message)


class FailingHandler(ProtocolHandler):
    """Handler that always raises."""

    def __init__(self, types: list[ProtocolMessageType]):
        self._types = types
        self.error_calls: list[tuple[ProtocolMessage, Exception]] = []

    @property
    def message_types(self) -> list[ProtocolMessageType]:
        return self._types

    async def handle(self, message: ProtocolMessage) -> None:
        raise ValueError("handler error")

    async def on_error(self, message: ProtocolMessage, error: Exception) -> None:
        self.error_calls.append((message, error))


# ---------------------------------------------------------------------------
# ProtocolHandlerRegistry — register / unregister
# ---------------------------------------------------------------------------


class TestRegistryBasics:
    def test_register_and_get_handlers(self):
        reg = ProtocolHandlerRegistry()
        handler = AsyncMock()
        reg.register(ProtocolMessageType.DEBATE_STARTED, handler)
        handlers = reg.get_handlers(ProtocolMessageType.DEBATE_STARTED)
        assert handler in handlers

    def test_register_multiple_same_type(self):
        reg = ProtocolHandlerRegistry()
        h1 = AsyncMock()
        h2 = AsyncMock()
        reg.register(ProtocolMessageType.DEBATE_STARTED, h1)
        reg.register(ProtocolMessageType.DEBATE_STARTED, h2)
        handlers = reg.get_handlers(ProtocolMessageType.DEBATE_STARTED)
        assert len(handlers) == 2

    def test_priority_ordering(self):
        reg = ProtocolHandlerRegistry()
        h_low = AsyncMock()
        h_high = AsyncMock()
        reg.register(ProtocolMessageType.DEBATE_STARTED, h_high, priority=200)
        reg.register(ProtocolMessageType.DEBATE_STARTED, h_low, priority=50)
        handlers = reg.get_handlers(ProtocolMessageType.DEBATE_STARTED)
        assert handlers[0] is h_low
        assert handlers[1] is h_high

    def test_unregister(self):
        reg = ProtocolHandlerRegistry()
        handler = AsyncMock()
        reg.register(ProtocolMessageType.DEBATE_STARTED, handler)
        assert reg.unregister(ProtocolMessageType.DEBATE_STARTED, handler) is True
        assert reg.get_handlers(ProtocolMessageType.DEBATE_STARTED) == []

    def test_unregister_nonexistent(self):
        reg = ProtocolHandlerRegistry()
        handler = AsyncMock()
        assert reg.unregister(ProtocolMessageType.DEBATE_STARTED, handler) is False

    def test_clear(self):
        reg = ProtocolHandlerRegistry()
        reg.register(ProtocolMessageType.DEBATE_STARTED, AsyncMock())
        reg.register_global(AsyncMock())
        reg.clear()
        assert reg.get_handlers(ProtocolMessageType.DEBATE_STARTED) == []


# ---------------------------------------------------------------------------
# register_handler (class-based)
# ---------------------------------------------------------------------------


class TestRegisterHandler:
    def test_class_handler(self):
        reg = ProtocolHandlerRegistry()
        handler = FakeHandler([ProtocolMessageType.DEBATE_STARTED])
        reg.register_handler(handler)
        msg = _make_msg()
        count = asyncio.run(reg.dispatch(msg))
        assert count == 1
        assert len(handler.calls) == 1

    def test_class_handler_multiple_types(self):
        reg = ProtocolHandlerRegistry()
        handler = FakeHandler(
            [
                ProtocolMessageType.DEBATE_STARTED,
                ProtocolMessageType.DEBATE_COMPLETED,
            ]
        )
        reg.register_handler(handler)
        asyncio.run(reg.dispatch(_make_msg(ProtocolMessageType.DEBATE_STARTED)))
        asyncio.run(reg.dispatch(_make_msg(ProtocolMessageType.DEBATE_COMPLETED)))
        assert len(handler.calls) == 2


# ---------------------------------------------------------------------------
# register_global
# ---------------------------------------------------------------------------


class TestGlobalHandler:
    def test_receives_all_types(self):
        reg = ProtocolHandlerRegistry()
        handler = AsyncMock()
        reg.register_global(handler)
        asyncio.run(reg.dispatch(_make_msg(ProtocolMessageType.DEBATE_STARTED)))
        asyncio.run(reg.dispatch(_make_msg(ProtocolMessageType.VOTE_CAST)))
        assert handler.call_count == 2

    def test_priority_ordering(self):
        reg = ProtocolHandlerRegistry()
        calls = []
        h1 = AsyncMock(side_effect=lambda m: calls.append("first"))
        h2 = AsyncMock(side_effect=lambda m: calls.append("second"))
        reg.register_global(h2, priority=200)
        reg.register_global(h1, priority=50)
        asyncio.run(reg.dispatch(_make_msg()))
        assert calls == ["first", "second"]


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_no_handlers(self):
        reg = ProtocolHandlerRegistry()
        count = asyncio.run(reg.dispatch(_make_msg()))
        assert count == 0

    def test_function_handler_called(self):
        reg = ProtocolHandlerRegistry()
        handler = AsyncMock()
        reg.register(ProtocolMessageType.DEBATE_STARTED, handler)
        msg = _make_msg()
        count = asyncio.run(reg.dispatch(msg))
        assert count == 1
        handler.assert_called_once_with(msg)

    def test_handler_error_isolated(self):
        reg = ProtocolHandlerRegistry()
        failing = AsyncMock(side_effect=ValueError("fail"))
        succeeding = AsyncMock()
        reg.register(ProtocolMessageType.DEBATE_STARTED, failing, priority=1)
        reg.register(ProtocolMessageType.DEBATE_STARTED, succeeding, priority=2)
        count = asyncio.run(reg.dispatch(_make_msg()))
        assert count == 1  # Only succeeding counted
        succeeding.assert_called_once()

    def test_class_handler_error_calls_on_error(self):
        reg = ProtocolHandlerRegistry()
        handler = FailingHandler([ProtocolMessageType.DEBATE_STARTED])
        reg.register_handler(handler)
        asyncio.run(reg.dispatch(_make_msg()))
        assert len(handler.error_calls) == 1
        assert isinstance(handler.error_calls[0][1], ValueError)

    def test_global_plus_specific(self):
        reg = ProtocolHandlerRegistry()
        global_h = AsyncMock()
        specific_h = AsyncMock()
        reg.register_global(global_h)
        reg.register(ProtocolMessageType.DEBATE_STARTED, specific_h)
        count = asyncio.run(reg.dispatch(_make_msg()))
        assert count == 2

    def test_wrong_type_not_called(self):
        reg = ProtocolHandlerRegistry()
        handler = AsyncMock()
        reg.register(ProtocolMessageType.VOTE_CAST, handler)
        asyncio.run(reg.dispatch(_make_msg(ProtocolMessageType.DEBATE_STARTED)))
        handler.assert_not_called()


# ---------------------------------------------------------------------------
# dispatch_concurrent
# ---------------------------------------------------------------------------


class TestDispatchConcurrent:
    def test_no_handlers(self):
        reg = ProtocolHandlerRegistry()
        count = asyncio.run(reg.dispatch_concurrent(_make_msg()))
        assert count == 0

    def test_all_succeed(self):
        reg = ProtocolHandlerRegistry()
        h1 = AsyncMock()
        h2 = AsyncMock()
        reg.register(ProtocolMessageType.DEBATE_STARTED, h1)
        reg.register(ProtocolMessageType.DEBATE_STARTED, h2)
        count = asyncio.run(reg.dispatch_concurrent(_make_msg()))
        assert count == 2

    def test_partial_failure(self):
        reg = ProtocolHandlerRegistry()
        failing = AsyncMock(side_effect=ValueError("fail"))
        succeeding = AsyncMock()
        reg.register(ProtocolMessageType.DEBATE_STARTED, failing)
        reg.register(ProtocolMessageType.DEBATE_STARTED, succeeding)
        count = asyncio.run(reg.dispatch_concurrent(_make_msg()))
        assert count == 1

    def test_global_concurrent(self):
        reg = ProtocolHandlerRegistry()
        handler = AsyncMock()
        reg.register_global(handler)
        count = asyncio.run(reg.dispatch_concurrent(_make_msg()))
        assert count == 1

    def test_class_handler_concurrent(self):
        reg = ProtocolHandlerRegistry()
        handler = FakeHandler([ProtocolMessageType.DEBATE_STARTED])
        reg.register_handler(handler)
        count = asyncio.run(reg.dispatch_concurrent(_make_msg()))
        assert count == 1


# ---------------------------------------------------------------------------
# get_handler_registry (singleton)
# ---------------------------------------------------------------------------


class TestGetHandlerRegistry:
    def test_returns_same_instance(self):
        with patch("aragora.debate.protocol_messages.handlers._default_registry", None):
            r1 = get_handler_registry()
            r2 = get_handler_registry()
            assert r1 is r2


# ---------------------------------------------------------------------------
# @handles decorator
# ---------------------------------------------------------------------------


class TestHandlesDecorator:
    def test_registers_handler(self):
        with patch("aragora.debate.protocol_messages.handlers._default_registry", None):

            @handles(ProtocolMessageType.PROPOSAL_SUBMITTED)
            async def on_proposal(message):
                pass

            registry = get_handler_registry()
            handlers = registry.get_handlers(ProtocolMessageType.PROPOSAL_SUBMITTED)
            assert on_proposal in handlers

    def test_multiple_types(self):
        with patch("aragora.debate.protocol_messages.handlers._default_registry", None):

            @handles(
                ProtocolMessageType.VOTE_CAST,
                ProtocolMessageType.VOTE_CHANGED,
            )
            async def on_vote(message):
                pass

            registry = get_handler_registry()
            assert on_vote in registry.get_handlers(ProtocolMessageType.VOTE_CAST)
            assert on_vote in registry.get_handlers(ProtocolMessageType.VOTE_CHANGED)


# ---------------------------------------------------------------------------
# LoggingHandler
# ---------------------------------------------------------------------------


class TestLoggingHandler:
    def test_handles_all_types(self):
        handler = LoggingHandler()
        assert len(handler.message_types) == len(ProtocolMessageType)

    def test_handle_logs(self):
        handler = LoggingHandler()
        msg = _make_msg(agent_id="claude", round_number=1)
        asyncio.run(handler.handle(msg))  # Should not raise


# ---------------------------------------------------------------------------
# MetricsHandler
# ---------------------------------------------------------------------------


class TestMetricsHandler:
    def test_handles_all_types(self):
        handler = MetricsHandler()
        assert len(handler.message_types) == len(ProtocolMessageType)

    def test_tracks_counts(self):
        handler = MetricsHandler()
        asyncio.run(handler.handle(_make_msg(ProtocolMessageType.VOTE_CAST)))
        asyncio.run(handler.handle(_make_msg(ProtocolMessageType.VOTE_CAST)))
        asyncio.run(handler.handle(_make_msg(ProtocolMessageType.DEBATE_STARTED)))
        counts = handler.get_counts()
        assert counts["vote_cast"] == 2
        assert counts["debate_started"] == 1

    def test_tracks_debate_counts(self):
        handler = MetricsHandler()
        asyncio.run(handler.handle(_make_msg(debate_id="d1")))
        asyncio.run(handler.handle(_make_msg(debate_id="d1")))
        asyncio.run(handler.handle(_make_msg(debate_id="d2")))
        assert handler.get_debate_count("d1") == 2
        assert handler.get_debate_count("d2") == 1
        assert handler.get_debate_count("d3") == 0
