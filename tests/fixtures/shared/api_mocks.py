"""
Shared API client mocks for OpenAI, Anthropic, and HTTPX.

This module provides mock implementations of API clients that prevent
network calls during tests while returning deterministic responses.

Usage:
    from tests.fixtures.shared.api_mocks import (
        MockOpenAIClient,
        MockAnthropicClient,
        apply_api_mocks,
    )

    def test_with_mocked_apis(monkeypatch):
        apply_api_mocks(monkeypatch)
        # API calls now return mock responses
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# ============================================================================
# OpenAI Mocks
# ============================================================================


class MockOpenAIMessage:
    """Mock OpenAI message object."""

    def __init__(self, content: str, role: str = "assistant"):
        self.content = content
        self.role = role


class MockOpenAIChoice:
    """Mock OpenAI choice object."""

    def __init__(self, content: str, index: int = 0):
        self.message = MockOpenAIMessage(content)
        self.index = index
        self.finish_reason = "stop"


class MockOpenAIUsage:
    """Mock OpenAI usage object."""

    def __init__(self, prompt_tokens: int = 10, completion_tokens: int = 20):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class MockOpenAICompletion:
    """Mock OpenAI chat completion response."""

    def __init__(self, content: str, model: str = "gpt-4o"):
        self.id = "chatcmpl-mock123"
        self.model = model
        self.choices = [MockOpenAIChoice(content)]
        self.usage = MockOpenAIUsage()
        self.created = 1700000000


class MockOpenAIChatCompletions:
    """Mock OpenAI chat completions API."""

    def _generate_response(self, messages: List[Dict], **kwargs) -> str:
        """Generate deterministic response based on input."""
        last_msg = ""
        for msg in reversed(messages):
            content = (
                msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            )
            role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
            if role == "user":
                last_msg = content
                break

        seed = hash(last_msg) % 1000
        responses = [
            f"I understand your query about '{last_msg[:50]}...'. Here's my analysis.",
            "Based on the information provided, I would suggest considering multiple perspectives.",
            "This is an interesting question. Let me provide a structured response.",
            "After careful consideration, here are my thoughts on the matter.",
            "I'll address your question systematically with supporting reasoning.",
        ]
        return responses[seed % len(responses)]

    def create(self, messages: List[Dict], model: str = "gpt-4o", **kwargs):
        """Sync create method."""
        content = self._generate_response(messages, **kwargs)
        return MockOpenAICompletion(content, model)

    async def acreate(self, messages: List[Dict], model: str = "gpt-4o", **kwargs):
        """Async create method (for compatibility)."""
        content = self._generate_response(messages, **kwargs)
        return MockOpenAICompletion(content, model)


class MockOpenAIAsyncChatCompletions:
    """Mock async OpenAI chat completions API."""

    def _generate_response(self, messages: List[Dict], **kwargs) -> str:
        """Generate deterministic response based on input."""
        last_msg = ""
        for msg in reversed(messages):
            content = (
                msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            )
            role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
            if role == "user":
                last_msg = content
                break

        seed = hash(last_msg) % 1000
        responses = [
            f"I understand your query about '{last_msg[:50]}...'. Here's my analysis.",
            "Based on the information provided, I would suggest considering multiple perspectives.",
            "This is an interesting question. Let me provide a structured response.",
            "After careful consideration, here are my thoughts on the matter.",
            "I'll address your question systematically with supporting reasoning.",
        ]
        return responses[seed % len(responses)]

    async def create(self, messages: List[Dict], model: str = "gpt-4o", **kwargs):
        """Async create method."""
        content = self._generate_response(messages, **kwargs)
        return MockOpenAICompletion(content, model)


class MockOpenAIChat:
    """Mock OpenAI chat API."""

    def __init__(self, async_mode: bool = False):
        if async_mode:
            self.completions = MockOpenAIAsyncChatCompletions()
        else:
            self.completions = MockOpenAIChatCompletions()


class MockOpenAIClient:
    """Mock OpenAI sync client."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key or "mock-openai-key"
        self.chat = MockOpenAIChat(async_mode=False)


class MockAsyncOpenAIClient:
    """Mock OpenAI async client."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key or "mock-openai-key"
        self.chat = MockOpenAIChat(async_mode=True)


# ============================================================================
# Anthropic Mocks
# ============================================================================


class MockAnthropicTextBlock:
    """Mock Anthropic text block."""

    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class MockAnthropicUsage:
    """Mock Anthropic usage object."""

    def __init__(self, input_tokens: int = 10, output_tokens: int = 20):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockAnthropicMessage:
    """Mock Anthropic message response."""

    def __init__(self, content: str, model: str = "claude-sonnet-4-20250514"):
        self.id = "msg_mock123"
        self.type = "message"
        self.role = "assistant"
        self.content = [MockAnthropicTextBlock(content)]
        self.model = model
        self.stop_reason = "end_turn"
        self.usage = MockAnthropicUsage()


class MockAnthropicMessages:
    """Mock Anthropic messages API."""

    def _generate_response(self, messages: List[Dict], **kwargs) -> str:
        """Generate deterministic response based on input."""
        last_msg = ""
        for msg in reversed(messages):
            content = (
                msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            )
            role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
            if role == "user":
                last_msg = content if isinstance(content, str) else str(content)
                break

        seed = hash(last_msg) % 1000
        responses = [
            f"Thank you for your question. I'll provide a thorough analysis of '{last_msg[:40]}...'.",
            "Let me address this thoughtfully. There are several key considerations here.",
            "This is a nuanced topic that deserves careful examination.",
            "I appreciate the opportunity to discuss this. Here's my perspective.",
            "Based on my analysis, I can offer the following insights.",
        ]
        return responses[seed % len(responses)]

    def create(
        self,
        messages: List[Dict],
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        **kwargs,
    ):
        """Sync create method."""
        content = self._generate_response(messages, **kwargs)
        return MockAnthropicMessage(content, model)


class MockAnthropicAsyncMessages:
    """Mock async Anthropic messages API."""

    def _generate_response(self, messages: List[Dict], **kwargs) -> str:
        """Generate deterministic response based on input."""
        last_msg = ""
        for msg in reversed(messages):
            content = (
                msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            )
            role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
            if role == "user":
                last_msg = content if isinstance(content, str) else str(content)
                break

        seed = hash(last_msg) % 1000
        responses = [
            f"Thank you for your question. I'll provide a thorough analysis of '{last_msg[:40]}...'.",
            "Let me address this thoughtfully. There are several key considerations here.",
            "This is a nuanced topic that deserves careful examination.",
            "I appreciate the opportunity to discuss this. Here's my perspective.",
            "Based on my analysis, I can offer the following insights.",
        ]
        return responses[seed % len(responses)]

    async def create(
        self,
        messages: List[Dict],
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        **kwargs,
    ):
        """Async create method."""
        content = self._generate_response(messages, **kwargs)
        return MockAnthropicMessage(content, model)


class MockAnthropicClient:
    """Mock Anthropic sync client."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key or "mock-anthropic-key"
        self.messages = MockAnthropicMessages()


class MockAsyncAnthropicClient:
    """Mock Anthropic async client."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key or "mock-anthropic-key"
        self.messages = MockAnthropicAsyncMessages()


# ============================================================================
# HTTPX Mocks
# ============================================================================


class MockHTTPXResponse:
    """Mock httpx response object."""

    def __init__(
        self,
        status_code: int = 200,
        json_data: Optional[Dict] = None,
        text: str = "",
    ):
        self.status_code = status_code
        self._json_data = json_data or {}
        self._text = text or json.dumps(self._json_data)
        self.headers = {"content-type": "application/json"}
        self.is_success = 200 <= status_code < 300

    def json(self) -> Dict:
        return self._json_data

    @property
    def text(self) -> str:
        return self._text

    def raise_for_status(self) -> None:
        if not self.is_success:
            raise Exception(f"HTTP {self.status_code}")


class MockHTTPXClient:
    """Mock httpx sync client."""

    def __init__(self, **kwargs):
        self._base_url = kwargs.get("base_url", "")
        self._timeout = kwargs.get("timeout", 30)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def _make_response(self, url: str, **kwargs) -> MockHTTPXResponse:
        """Generate mock response based on URL."""
        seed = hash(url) % 100
        return MockHTTPXResponse(
            status_code=200,
            json_data={
                "status": "ok",
                "url": url,
                "mock": True,
                "seed": seed,
            },
        )

    def get(self, url: str, **kwargs) -> MockHTTPXResponse:
        return self._make_response(url, **kwargs)

    def post(self, url: str, **kwargs) -> MockHTTPXResponse:
        return self._make_response(url, **kwargs)

    def put(self, url: str, **kwargs) -> MockHTTPXResponse:
        return self._make_response(url, **kwargs)

    def delete(self, url: str, **kwargs) -> MockHTTPXResponse:
        return self._make_response(url, **kwargs)

    def patch(self, url: str, **kwargs) -> MockHTTPXResponse:
        return self._make_response(url, **kwargs)

    def request(self, method: str, url: str, **kwargs) -> MockHTTPXResponse:
        return self._make_response(url, **kwargs)

    def close(self) -> None:
        pass


class MockAsyncHTTPXClient:
    """Mock httpx async client."""

    def __init__(self, **kwargs):
        self._base_url = kwargs.get("base_url", "")
        self._timeout = kwargs.get("timeout", 30)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def _make_response(self, url: str, **kwargs) -> MockHTTPXResponse:
        """Generate mock response based on URL."""
        seed = hash(url) % 100
        return MockHTTPXResponse(
            status_code=200,
            json_data={
                "status": "ok",
                "url": url,
                "mock": True,
                "seed": seed,
            },
        )

    async def get(self, url: str, **kwargs) -> MockHTTPXResponse:
        return self._make_response(url, **kwargs)

    async def post(self, url: str, **kwargs) -> MockHTTPXResponse:
        return self._make_response(url, **kwargs)

    async def put(self, url: str, **kwargs) -> MockHTTPXResponse:
        return self._make_response(url, **kwargs)

    async def delete(self, url: str, **kwargs) -> MockHTTPXResponse:
        return self._make_response(url, **kwargs)

    async def patch(self, url: str, **kwargs) -> MockHTTPXResponse:
        return self._make_response(url, **kwargs)

    async def request(self, method: str, url: str, **kwargs) -> MockHTTPXResponse:
        return self._make_response(url, **kwargs)

    async def aclose(self) -> None:
        pass

    def close(self) -> None:
        pass


# ============================================================================
# Apply All Mocks
# ============================================================================


def apply_api_mocks(monkeypatch: Any, force: bool = False) -> None:
    """Apply all API client mocks to prevent network calls.

    This patches OpenAI, Anthropic, and httpx clients with mock implementations
    that return deterministic responses.

    Args:
        monkeypatch: pytest monkeypatch fixture
        force: If True, apply mocks even for network/integration tests
    """
    # Patch OpenAI
    try:
        import openai

        monkeypatch.setattr(openai, "OpenAI", MockOpenAIClient)
        monkeypatch.setattr(openai, "AsyncOpenAI", MockAsyncOpenAIClient)
    except ImportError:
        pass

    # String-based patches for OpenAI
    try:
        monkeypatch.setattr("openai.OpenAI", MockOpenAIClient)
        monkeypatch.setattr("openai.AsyncOpenAI", MockAsyncOpenAIClient)
    except (ImportError, AttributeError):
        pass

    # Patch Anthropic
    try:
        import anthropic

        monkeypatch.setattr(anthropic, "Anthropic", MockAnthropicClient)
        monkeypatch.setattr(anthropic, "AsyncAnthropic", MockAsyncAnthropicClient)
    except ImportError:
        pass

    # String-based patches for Anthropic
    try:
        monkeypatch.setattr("anthropic.Anthropic", MockAnthropicClient)
        monkeypatch.setattr("anthropic.AsyncAnthropic", MockAsyncAnthropicClient)
    except (ImportError, AttributeError):
        pass

    # Patch httpx
    try:
        import httpx

        monkeypatch.setattr(httpx, "Client", MockHTTPXClient)
        monkeypatch.setattr(httpx, "AsyncClient", MockAsyncHTTPXClient)
    except ImportError:
        pass

    # String-based patches for httpx
    try:
        monkeypatch.setattr("httpx.Client", MockHTTPXClient)
        monkeypatch.setattr("httpx.AsyncClient", MockAsyncHTTPXClient)
    except (ImportError, AttributeError):
        pass

    # Patch modules that may do lazy imports of API clients
    api_modules = [
        "aragora.agents.api_agents.anthropic",
        "aragora.agents.api_agents.openai",
        "aragora.agents.api_agents.openrouter",
        "aragora.agents.fallback",
        "aragora.rlm.bridge",
    ]

    for module_path in api_modules:
        # Patch OpenAI in module
        try:
            monkeypatch.setattr(f"{module_path}.OpenAI", MockOpenAIClient)
        except (ImportError, AttributeError):
            pass
        try:
            monkeypatch.setattr(f"{module_path}.AsyncOpenAI", MockAsyncOpenAIClient)
        except (ImportError, AttributeError):
            pass
        # Patch Anthropic in module
        try:
            monkeypatch.setattr(f"{module_path}.Anthropic", MockAnthropicClient)
        except (ImportError, AttributeError):
            pass
        try:
            monkeypatch.setattr(f"{module_path}.AsyncAnthropic", MockAsyncAnthropicClient)
        except (ImportError, AttributeError):
            pass


# ============================================================================
# Response Factories
# ============================================================================


def create_openai_response(
    content: str = "Test response",
    model: str = "gpt-4o",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> MockOpenAICompletion:
    """Create a mock OpenAI completion response.

    Args:
        content: Response content
        model: Model name
        prompt_tokens: Token count for prompt
        completion_tokens: Token count for completion

    Returns:
        MockOpenAICompletion instance
    """
    response = MockOpenAICompletion(content, model)
    response.usage = MockOpenAIUsage(prompt_tokens, completion_tokens)
    return response


def create_anthropic_response(
    content: str = "Test response",
    model: str = "claude-sonnet-4-20250514",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> MockAnthropicMessage:
    """Create a mock Anthropic message response.

    Args:
        content: Response content
        model: Model name
        input_tokens: Token count for input
        output_tokens: Token count for output

    Returns:
        MockAnthropicMessage instance
    """
    response = MockAnthropicMessage(content, model)
    response.usage = MockAnthropicUsage(input_tokens, output_tokens)
    return response


def create_httpx_response(
    status_code: int = 200,
    json_data: Optional[Dict] = None,
    text: str = "",
) -> MockHTTPXResponse:
    """Create a mock HTTPX response.

    Args:
        status_code: HTTP status code
        json_data: JSON response data
        text: Text response (if no json_data)

    Returns:
        MockHTTPXResponse instance
    """
    return MockHTTPXResponse(status_code, json_data, text)
