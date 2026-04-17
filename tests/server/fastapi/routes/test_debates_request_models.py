from __future__ import annotations

import pytest
from pydantic import ValidationError

from aragora.server.fastapi.routes.debates import CreateDebateRequest


def test_create_debate_request_accepts_structured_agents() -> None:
    req = CreateDebateRequest(
        question="Should we adopt microservices?",
        agents=[
            {"provider": "anthropic-api", "model": "claude-opus-4-7"},
            {"provider": "openai-api", "model": "gpt-4.1"},
        ],
    )

    assert isinstance(req.agents, list)
    assert req.agents[0]["provider"] == "anthropic-api"


def test_create_debate_request_rejects_invalid_structured_agents() -> None:
    with pytest.raises(ValidationError):
        CreateDebateRequest(
            question="Should we adopt microservices?",
            agents=[{"model": "claude-opus-4-7"}],
        )
