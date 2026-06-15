"""Mock API response factory fixtures (pytest plugin)."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_anthropic_response():
    """Create mock Anthropic API response.

    Returns a factory function that creates mock responses.
    Use with `unittest.mock.patch` to mock httpx or requests calls.

    Example:
        def test_anthropic_call(mock_anthropic_response):
            with patch('httpx.AsyncClient.post') as mock_post:
                mock_post.return_value = mock_anthropic_response("Hello!")
                # ... test code
    """

    def _make_response(
        content: str = "Test response",
        model: str = "claude-sonnet-4-20250514",
        stop_reason: str = "end_turn",
        input_tokens: int = 100,
        output_tokens: int = 50,
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "msg_test123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": content}],
            "model": model,
            "stop_reason": stop_reason,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    return _make_response


@pytest.fixture
def mock_openai_response():
    """Create mock OpenAI API response.

    Returns a factory function that creates mock responses.

    Example:
        def test_openai_call(mock_openai_response):
            with patch('openai.AsyncOpenAI') as mock_client:
                mock_client.return_value.chat.completions.create = AsyncMock(
                    return_value=mock_openai_response("Hello!")
                )
    """

    def _make_response(
        content: str = "Test response",
        model: str = "gpt-4o",
        finish_reason: str = "stop",
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
    ):
        mock_choice = MagicMock()
        mock_choice.message.content = content
        mock_choice.message.role = "assistant"
        mock_choice.finish_reason = finish_reason
        mock_choice.index = 0

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = prompt_tokens
        mock_usage.completion_tokens = completion_tokens
        mock_usage.total_tokens = prompt_tokens + completion_tokens

        mock_resp = MagicMock()
        mock_resp.id = "chatcmpl-test123"
        mock_resp.model = model
        mock_resp.choices = [mock_choice]
        mock_resp.usage = mock_usage
        mock_resp.created = 1700000000

        return mock_resp

    return _make_response


@pytest.fixture
def mock_openrouter_response():
    """Create mock OpenRouter API response.

    OpenRouter uses OpenAI-compatible format.
    """

    def _make_response(
        content: str = "Test response",
        model: str = "anthropic/claude-3.5-sonnet",
        finish_reason: str = "stop",
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "gen-test123",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    return _make_response


@pytest.fixture
def mock_streaming_response():
    """Create mock streaming API response (SSE format).

    Returns a factory that creates an async generator for streaming responses.
    """

    def _make_stream(chunks: list[str] | None = None):
        if chunks is None:
            chunks = ["Hello", " world", "!"]

        async def _stream():
            for i, chunk in enumerate(chunks):
                yield {
                    "id": f"chunk-{i}",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None if i < len(chunks) - 1 else "stop",
                        }
                    ],
                }

        return _stream()

    return _make_stream
