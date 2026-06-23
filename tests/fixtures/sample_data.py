"""Sample data fixtures for debate tests (pytest plugin)."""

import pytest


@pytest.fixture
def sample_debate_messages() -> list[dict]:
    """Return sample debate messages for testing."""
    return [
        {
            "agent": "claude",
            "role": "proposer",
            "content": "I propose that we should implement feature X.",
            "round": 1,
        },
        {
            "agent": "gemini",
            "role": "critic",
            "content": "I have concerns about the scalability of feature X.",
            "round": 1,
        },
        {
            "agent": "claude",
            "role": "proposer",
            "content": "Addressing your concerns, we can add caching.",
            "round": 2,
        },
    ]


@pytest.fixture
def sample_critique() -> dict:
    """Return a sample critique for testing."""
    return {
        "critic": "gemini",
        "target": "claude",
        "content": "The proposed solution doesn't address edge cases.",
        "severity": "medium",
        "accepted": False,
    }
