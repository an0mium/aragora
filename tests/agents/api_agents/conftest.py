"""
Shared fixtures for API agent tests.
"""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class MockResponse:
    """Mock HTTP response for API tests."""

    def __init__(
        self,
        status: int = 200,
        json_data: Optional[Dict[str, Any]] = None,
        text: str = "",
        headers: Optional[Dict[str, str]] = None,
    ):
        self.status = status
        self._json_data = json_data or {}
        self._text = text
        self.headers = headers or {}

    async def json(self) -> Dict[str, Any]:
        return self._json_data

    async def text(self) -> str:
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockStreamResponse:
    """Mock streaming HTTP response for SSE tests."""

    def __init__(
        self,
        status: int = 200,
        chunks: Optional[List[bytes]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.status = status
        self._chunks = chunks or []
        self.headers = headers or {}
        self.content = self._create_content()

    def _create_content(self):
        """Create async iterator for content."""

        class AsyncContent:
            def __init__(self, chunks):
                self._chunks = chunks
                self._index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._index >= len(self._chunks):
                    raise StopAsyncIteration
                chunk = self._chunks[self._index]
                self._index += 1
                return chunk

        return AsyncContent(self._chunks)

    async def text(self) -> str:
        return b"".join(self._chunks).decode("utf-8")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockClientSession:
    """Mock aiohttp ClientSession."""

    def __init__(self, responses: Optional[List[MockResponse]] = None):
        self._responses = responses or []
        self._response_index = 0

    def _get_next_response(self):
        if self._response_index < len(self._responses):
            response = self._responses[self._response_index]
            self._response_index += 1
            return response
        return MockResponse(status=500, text="No mock response available")

    def post(self, url: str, **kwargs):
        return self._get_next_response()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_anthropic_response():
    """Standard Anthropic API response."""
    return {
        "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "This is a test response from Claude."}],
        "model": "claude-opus-4-5-20251101",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }


@pytest.fixture
def mock_anthropic_web_search_response():
    """Anthropic API response with web search results."""
    return {
        "id": "msg_02XFDUDYJgAACzvnptvVoYEL",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Based on my search, here is the information:"},
            {
                "type": "web_search_tool_result",
                "content": [
                    {
                        "type": "web_search_result",
                        "title": "Example Page",
                        "url": "https://example.com/page",
                    }
                ],
            },
        ],
        "model": "claude-opus-4-5-20251101",
        "usage": {"input_tokens": 150, "output_tokens": 75},
    }


@pytest.fixture
def mock_openai_response():
    """Standard OpenAI API response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-5.2",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "This is a test response from GPT."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


@pytest.fixture
def mock_mistral_response():
    """Standard Mistral API response."""
    return {
        "id": "cmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "mistral-large-2512",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "This is a test response from Mistral."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


@pytest.fixture
def mock_openrouter_response():
    """Standard OpenRouter API response."""
    return {
        "id": "gen-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "deepseek/deepseek-chat",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "This is a test response from DeepSeek."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


@pytest.fixture
def mock_sse_chunks():
    """SSE chunks for streaming response tests (OpenAI format)."""
    return [
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"!"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]


@pytest.fixture
def mock_anthropic_sse_chunks():
    """SSE chunks for Anthropic streaming response tests."""
    return [
        b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}\n\n',
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]


@pytest.fixture
def mock_rate_limit_response():
    """Rate limit (429) error response."""
    return MockResponse(
        status=429,
        text='{"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}',
        headers={"Retry-After": "30"},
    )


@pytest.fixture
def mock_quota_error_response():
    """Quota/billing error response."""
    return MockResponse(
        status=400,
        text='{"error": {"message": "Your credit balance is too low", "type": "billing_error"}}',
    )


@pytest.fixture
def mock_api_error_response():
    """Generic API error response."""
    return MockResponse(
        status=500,
        text='{"error": {"message": "Internal server error", "type": "server_error"}}',
    )


@pytest.fixture
def mock_empty_response():
    """Empty content response."""
    return {
        "id": "chatcmpl-empty",
        "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


@pytest.fixture
def mock_env_with_api_keys(monkeypatch):
    """Set up environment with mock API keys."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")


@pytest.fixture
def mock_env_no_api_keys(monkeypatch):
    """Clear all API keys from environment."""
    for key in [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "MISTRAL_API_KEY",
        "OPENROUTER_API_KEY",
        "XAI_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def sample_context():
    """Sample message context for testing."""
    from aragora.core import Message

    return [
        Message(
            agent="agent1",
            content="First message",
            role="proposer",
            round_num=1,
        ),
        Message(
            agent="agent2",
            content="Response to first message",
            role="critic",
            round_num=1,
        ),
    ]


@pytest.fixture
def mock_circuit_breaker():
    """Mock circuit breaker for testing."""
    breaker = MagicMock()
    breaker.can_proceed.return_value = True
    breaker.record_failure = MagicMock()
    breaker.record_success = MagicMock()
    return breaker


@pytest.fixture
def mock_openrouter_limiter():
    """Mock rate limiter for OpenRouter tests."""
    limiter = MagicMock()
    limiter.acquire = AsyncMock(return_value=True)
    limiter.update_from_headers = MagicMock()
    limiter.record_rate_limit_error = MagicMock(return_value=30.0)
    limiter.record_success = MagicMock()
    limiter.release_on_error = MagicMock()
    return limiter
