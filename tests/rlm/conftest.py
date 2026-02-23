"""
Pytest fixtures for RLM tests.

Provides automatic mocking of agent calls to prevent timeouts from external API calls.

Also resets all RLM global state (singletons, metrics, caches) before each test
to prevent order-dependent failures when tests from other directories pollute
module-level globals.
"""

from __future__ import annotations

import os

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


# Environment variables that can affect RLM behavior and may be set by other tests.
_RLM_ENV_VARS = [
    "ARAGORA_RLM_MODE",
    "ARAGORA_RLM_REQUIRE_TRUE",
    "ARAGORA_RLM_MAX_CONTENT_BYTES",
    "ARAGORA_RLM_MAX_REPL_MEMORY_MB",
    "ARAGORA_RLM_TARGET_TOKENS",
    "ARAGORA_RLM_OVERLAP_TOKENS",
    "ARAGORA_RLM_WARN_FALLBACK",
    "ARAGORA_RLM_BACKEND",
    "ARAGORA_RLM_PROVIDER",
    "ARAGORA_RLM_MODEL",
    "ARAGORA_RLM_MODEL_NAME",
    "ARAGORA_RLM_FALLBACK_BACKEND",
    "ARAGORA_RLM_FALLBACK_MODEL",
    "ARAGORA_RLM_SUB_MODEL",
    "ARAGORA_RLM_SYSTEM_PROMPT",
    "ARAGORA_RLM_CONTEXT_DIR",
]


@pytest.fixture(autouse=True)
def _reset_rlm_global_state(monkeypatch):
    """Reset all RLM global state before and after each test.

    This prevents order-dependent failures caused by state pollution from
    tests in other directories. When pytest-randomly reorders tests across
    the full suite, prior tests may:

    1. Leave a stale factory singleton (``_rlm_instance``) whose internal
       ``_official_rlm`` was initialised with a now-vanished API key.
    2. Accumulate factory metrics (``_metrics``) that cause assertions on
       exact counts to fail.
    3. Populate the compression cache with entries from prior runs.
    4. Set environment variables (e.g. ``ARAGORA_RLM_MODE=true_rlm``) that
       change ``get_rlm()`` behaviour.
    5. Leave a stale metrics collector singleton (``_collector``).
    """
    # -- 1. Reset factory singleton and metrics --
    from aragora.rlm.factory import reset_singleton, reset_metrics
    reset_singleton()
    reset_metrics()

    # -- 2. Clear compression cache and reset call semaphore --
    import aragora.rlm.compressor as _compressor_mod
    _compressor_mod._compression_cache.clear()
    # Reset the global call semaphore to avoid cross-event-loop deadlocks.
    # If a previous test created it on a different event loop, ``async with
    # semaphore:`` will hang forever on the current loop.
    _compressor_mod._call_semaphore = None

    # -- 3. Reset metrics collector singleton --
    import aragora.rlm.metrics_export as _metrics_export_mod
    _metrics_export_mod._collector = None

    # -- 4. Scrub RLM-related environment variables that other tests may have set --
    for var in _RLM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    yield

    # Teardown: reset again so the next test (possibly from another directory)
    # starts clean.
    reset_singleton()
    reset_metrics()
    _compressor_mod._compression_cache.clear()
    _compressor_mod._call_semaphore = None
    _metrics_export_mod._collector = None


@pytest.fixture(autouse=True)
def mock_rlm_agent_calls():
    """
    Automatically mock agent calls in all RLM tests.

    This prevents tests from making actual API calls to language models,
    which would cause timeouts and flaky tests.

    Critically, ``HAS_OFFICIAL_RLM`` is patched to ``False`` so that
    ``AragoraRLM`` always uses the compression-fallback path.  Without
    this, when the official ``rlm`` package is installed the code takes
    the TRUE RLM path (``_true_rlm_query``), which calls the real OpenAI
    API through the ``rlm`` library -- bypassing our ``_agent_call`` mock
    and failing when no real API key is available.
    """
    with patch("aragora.rlm.bridge.HAS_OFFICIAL_RLM", False):
        with patch("aragora.rlm.bridge.AragoraRLM._agent_call", side_effect=_mock_agent_call):
            with patch(
                "aragora.rlm.bridge.AragoraRLM._agent_call_async",
                side_effect=_mock_agent_call_async,
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
