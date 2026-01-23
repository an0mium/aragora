"""
Pytest fixtures for RLM tests.

Provides automatic mocking of agent calls to prevent timeouts from external API calls.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


def _mock_agent_call(prompt: str, model: str) -> str:
    """Mock agent call that returns a reasonable response without API calls."""
    # Return a compressed/summarized version based on prompt content
    if "summarize" in prompt.lower() or "compress" in prompt.lower():
        return "This is a compressed summary of the content discussing the main topics."
    elif "what" in prompt.lower() or "question" in prompt.lower():
        return "The answer based on the compressed context is that the content discusses the requested topic."
    elif "consensus" in prompt.lower():
        return "The consensus reached was to use a hybrid approach combining the best elements."
    elif "key point" in prompt.lower():
        return "The key point is the main finding from the analysis."
    else:
        return "Summarized response for the given prompt."


def _mock_agent_call_async(prompt: str, model: str, context: str) -> str:
    """Mock async-compatible agent call."""
    full_prompt = f"{prompt}\n\nContext:\n{context}" if context else prompt
    return _mock_agent_call(full_prompt, model)


@pytest.fixture(autouse=True)
def mock_rlm_agent_calls():
    """
    Automatically mock agent calls in all RLM tests.

    This prevents tests from making actual API calls to language models,
    which would cause timeouts and flaky tests.
    """
    with patch("aragora.rlm.bridge.AragoraRLM._agent_call", side_effect=_mock_agent_call):
        with patch(
            "aragora.rlm.bridge.AragoraRLM._agent_call_async", side_effect=_mock_agent_call_async
        ):
            yield


@pytest.fixture
def mock_rlm():
    """Fixture providing a mocked RLM instance."""
    from aragora.rlm import get_rlm, reset_singleton

    reset_singleton()
    rlm = get_rlm()
    return rlm


@pytest.fixture
def mock_compressor():
    """Fixture providing a mocked compressor instance."""
    from aragora.rlm import get_compressor, reset_singleton

    reset_singleton()
    compressor = get_compressor()
    return compressor
