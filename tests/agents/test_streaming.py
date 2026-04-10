from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from aragora.agents.streaming import StreamingMixin
from aragora.exceptions import StreamingError


class _DummyContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def iter_any(self) -> AsyncIterator[bytes]:
        return _ChunkIterator(self._chunks)


class _ChunkIterator:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self) -> _ChunkIterator:
        return self

    async def __anext__(self) -> bytes:
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _DummyResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self.content = _DummyContent(chunks)


class _DummyAgent(StreamingMixin):
    pass


async def _collect_chunks(chunks: list[bytes], format_type: str = "openai") -> list[str]:
    agent = _DummyAgent()
    response = _DummyResponse(chunks)

    return [
        chunk async for chunk in agent.parse_sse_stream(response=response, format_type=format_type)
    ]


@pytest.mark.asyncio
async def test_parse_sse_stream_yields_openai_chunks() -> None:
    chunks = [
        b'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"world"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    assert await _collect_chunks(chunks) == ["Hello ", "world"]


@pytest.mark.asyncio
async def test_parse_sse_stream_yields_anthropic_delta_text() -> None:
    chunks = [
        b'data: {"type":"content_block_delta","delta":{"text":"alpha"}}\n\n',
        b'data: {"type":"content_block_delta","delta":{"text":" beta"}}\n\n',
        b"data: [DONE]\n\n",
    ]

    assert await _collect_chunks(chunks, format_type="anthropic") == ["alpha", " beta"]


@pytest.mark.asyncio
async def test_parse_sse_stream_reassembles_partial_packets() -> None:
    chunks = [
        b'data: {"choices":[{"del',
        b'ta":{"content":"split"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    assert await _collect_chunks(chunks) == ["split"]


@pytest.mark.asyncio
async def test_parse_sse_stream_ignores_comments_and_blank_lines() -> None:
    chunks = [
        b"\n",
        b": ping\n",
        b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    assert await _collect_chunks(chunks) == ["ok"]


@pytest.mark.asyncio
async def test_parse_sse_stream_ignores_malformed_json() -> None:
    chunks = [
        b"data: {not-json}\n\n",
        b'data: {"choices":[{"delta":{"content":"safe"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    assert await _collect_chunks(chunks) == ["safe"]


@pytest.mark.asyncio
async def test_parse_sse_stream_stops_at_done_marker() -> None:
    chunks = [
        b'data: {"choices":[{"delta":{"content":"before"}}]}\n\n',
        b"data: [DONE]\n\n",
        b'data: {"choices":[{"delta":{"content":"after"}}]}\n\n',
    ]

    assert await _collect_chunks(chunks) == ["before"]


@pytest.mark.asyncio
async def test_parse_sse_stream_skips_non_dict_json_events() -> None:
    chunks = [
        b"data: [1, 2, 3]\n\n",
        b'data: {"choices":[{"delta":{"content":"dict-event"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    assert await _collect_chunks(chunks) == ["dict-event"]


@pytest.mark.asyncio
async def test_parse_sse_stream_raises_when_buffer_exceeds_limit(monkeypatch) -> None:
    monkeypatch.setattr("aragora.agents.streaming.get_stream_buffer_size", lambda: 8)
    agent = _DummyAgent()
    response = _DummyResponse([b"x" * 20])

    with pytest.raises(StreamingError):
        async for _ in agent.parse_sse_stream(response=response):
            pass


@pytest.mark.asyncio
async def test_extract_content_returns_empty_for_unknown_format() -> None:
    chunks = [
        b'data: {"choices":[{"delta":{"content":"ignored"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    assert await _collect_chunks(chunks, format_type="custom") == []


@pytest.mark.asyncio
async def test_extract_content_ignores_non_dict_openai_delta() -> None:
    agent = _DummyAgent()
    event: dict[str, Any] = {"choices": [{"delta": "not-a-dict"}]}

    assert agent._extract_content_from_event(event, "openai") == ""
