"""Shared fixtures for nomic phases tests."""

import pytest


@pytest.fixture(autouse=True)
def _disable_rlm_context(monkeypatch):
    """Disable RLM context gathering to avoid calling real LLM APIs in tests."""
    monkeypatch.setenv("ARAGORA_NOMIC_CONTEXT_RLM", "false")
