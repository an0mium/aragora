from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.agents.api_agents.openai import OpenAIAPIAgent
from aragora.core import Message
from tests.agents.api_agents.conftest import MockClientSession, MockResponse


@pytest.mark.asyncio
async def test_quota_error_routes_request_to_fallback_agent():
    agent = OpenAIAPIAgent(name="primary", api_key="sk-test", enable_fallback=True)
    agent._circuit_breaker = None
    fallback = MagicMock()
    fallback.generate = AsyncMock(return_value="fallback response")
    context = [Message(role="user", agent="boss", content="Earlier request", round=1)]
    session = MockClientSession([MockResponse(status=429, text='{"error":"rate limit exceeded"}')])

    with (
        patch(
            "aragora.agents.api_agents.openai_compatible.create_client_session",
            return_value=session,
        ),
        patch.object(agent, "_build_fallback_providers", return_value=[("openrouter", fallback)]),
    ):
        result = await agent.generate("Test prompt", context=context)

    assert result == "fallback response"
    fallback.generate.assert_called_once_with("Test prompt", context)
