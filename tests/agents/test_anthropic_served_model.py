"""The model the SERVER answered with is read back and recorded.

Finding C-P3 on #9989 (merge-gate, round 2). This branch turns Anthropic's
server-side refusal fallback on by default for Fable 5.1 / Opus 5, so a
request can legitimately be answered by a DIFFERENT model than the one asked
for. The agent never read the response's ``model`` field, so a decision
receipt attributed the output to the requested id even when the fallback had
fired -- the receipt was wrong about which model made the decision.

Both paths are covered: the non-streaming body's top-level ``model``, and the
streaming ``message_start`` event's ``message.model``.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

_REQUESTED = "claude-fable-5-1"
_SERVED = "claude-opus-4-8"


@pytest.fixture
def agent() -> AnthropicAPIAgent:
    return AnthropicAPIAgent(model=_REQUESTED, api_key="test-key")


def _session_patch(response: MagicMock):
    """Patch anthropic.create_client_session to yield ``response``."""
    post_cm = MagicMock()
    post_cm.__aenter__ = AsyncMock(return_value=response)
    post_cm.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.post = MagicMock(return_value=post_cm)

    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=False)

    return patch(
        "aragora.agents.api_agents.anthropic.create_client_session",
        return_value=session_cm,
    )


def _json_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.status = 200
    response.json = AsyncMock(return_value=payload)
    response.text = AsyncMock(return_value="")
    return response


def _sse_response(events: list[str]) -> MagicMock:
    response = MagicMock()
    response.status = 200

    async def iter_any():
        for event in events:
            yield f"data: {event}\n\n".encode()

    content = MagicMock()
    content.iter_any = iter_any
    response.content = content
    return response


class TestNonStreamingServedModel:
    @pytest.mark.asyncio
    async def test_fallback_model_is_recorded_in_metadata(self, agent) -> None:
        payload = {"model": _SERVED, "content": [{"type": "text", "text": "hi"}]}
        with _session_patch(_json_response(payload)):
            assert await agent.generate("prompt") == "hi"
        assert agent.get_metadata()["served_model"] == _SERVED
        assert agent.last_served_model == _SERVED
        # The requested id is NOT rewritten: the agent still asked for it.
        assert agent.model == _REQUESTED

    @pytest.mark.asyncio
    async def test_same_model_records_nothing(self, agent) -> None:
        payload = {"model": _REQUESTED, "content": [{"type": "text", "text": "hi"}]}
        with _session_patch(_json_response(payload)):
            await agent.generate("prompt")
        assert agent.get_metadata()["served_model"] is None

    @pytest.mark.asyncio
    async def test_a_later_matching_call_clears_the_stale_value(self, agent) -> None:
        """A served_model from an earlier generation must never be attributed
        to a later one."""
        with _session_patch(
            _json_response({"model": _SERVED, "content": [{"type": "text", "text": "a"}]})
        ):
            await agent.generate("prompt")
        assert agent.last_served_model == _SERVED
        with _session_patch(
            _json_response({"model": _REQUESTED, "content": [{"type": "text", "text": "b"}]})
        ):
            await agent.generate("prompt")
        assert agent.last_served_model is None

    @pytest.mark.asyncio
    async def test_logged_once_at_info(self, agent, caplog) -> None:
        AnthropicAPIAgent._SERVED_MODEL_LOGGED.discard((_REQUESTED, _SERVED))
        payload = {"model": _SERVED, "content": [{"type": "text", "text": "hi"}]}
        with caplog.at_level(logging.INFO, logger="aragora.agents.api_agents.anthropic"):
            with _session_patch(_json_response(payload)):
                await agent.generate("prompt")
            with _session_patch(_json_response(payload)):
                await agent.generate("prompt")
        records = [r for r in caplog.records if "served_model" in r.getMessage()]
        assert len(records) == 1
        assert records[0].levelno == logging.INFO
        assert _SERVED in records[0].getMessage()


class TestStreamingServedModel:
    @pytest.mark.asyncio
    async def test_message_start_model_is_recorded(self, agent) -> None:
        events = [
            '{"type": "message_start", "message": {"model": "%s"}}' % _SERVED,
            '{"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}}',
        ]
        with _session_patch(_sse_response(events)):
            chunks = [c async for c in agent.generate_stream("prompt")]
        assert chunks == ["hi"]
        assert agent.get_metadata()["served_model"] == _SERVED

    @pytest.mark.asyncio
    async def test_same_model_records_nothing(self, agent) -> None:
        events = [
            '{"type": "message_start", "message": {"model": "%s"}}' % _REQUESTED,
            '{"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}}',
        ]
        with _session_patch(_sse_response(events)):
            [c async for c in agent.generate_stream("prompt")]
        assert agent.get_metadata()["served_model"] is None

    @pytest.mark.asyncio
    async def test_stream_without_message_start_records_nothing(self, agent) -> None:
        events = [
            '{"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}}',
        ]
        with _session_patch(_sse_response(events)):
            [c async for c in agent.generate_stream("prompt")]
        assert agent.get_metadata()["served_model"] is None


class TestParserCapture:
    """Direct coverage of the shared SSE parser field."""

    def test_ignores_malformed_message_start(self) -> None:
        from aragora.agents.api_agents.common import create_anthropic_sse_parser

        parser = create_anthropic_sse_parser()
        parser._capture_message_start_model({"type": "message_start"})
        parser._capture_message_start_model({"type": "message_start", "message": "not-a-dict"})
        parser._capture_message_start_model({"type": "message_start", "message": {"model": ""}})
        parser._capture_message_start_model({"type": "message_delta", "message": {"model": "x"}})
        assert parser.served_model is None
        parser._capture_message_start_model({"type": "message_start", "message": {"model": "x"}})
        assert parser.served_model == "x"
