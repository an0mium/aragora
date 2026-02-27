"""Tests for thinking trace surfacing in explainability.

Verifies:
- Proposals include thinking metadata when agents provide it
- Proposals without thinking have no thinking key
- ExplanationBuilder._extract_thinking_traces extracts from result metadata
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.explainability.builder import ExplanationBuilder


class TestProposalThinkingMetadata:
    """Test thinking metadata in proposal-level agent metadata."""

    def test_proposal_includes_thinking_when_available(self):
        """Agent with get_metadata returning thinking should expose it."""
        agent = MagicMock()
        agent.name = "claude-api"
        agent.get_metadata.return_value = {
            "thinking": "Let me reason step by step about this problem...",
            "thinking_budget": 5000,
        }

        metadata = agent.get_metadata()
        assert metadata["thinking"] is not None
        assert "step by step" in metadata["thinking"]
        assert metadata["thinking_budget"] == 5000

    def test_proposal_without_thinking_has_no_key(self):
        """Agent without thinking should return None for thinking key."""
        agent = MagicMock()
        agent.name = "gpt-4"
        agent.get_metadata.return_value = {
            "thinking": None,
            "thinking_budget": None,
        }

        metadata = agent.get_metadata()
        assert metadata["thinking"] is None
        assert metadata["thinking_budget"] is None

    def test_explanation_includes_thinking_summary(self):
        """ExplanationBuilder._extract_thinking_traces extracts from metadata."""
        builder = ExplanationBuilder()

        # Create a mock result with agent_thinking in metadata
        result = MagicMock()
        result.metadata = {
            "agent_thinking": {
                "claude-1": "I considered multiple angles on the rate limiter design...",
                "claude-2": "My reasoning focused on token bucket vs sliding window...",
            }
        }

        traces = builder._extract_thinking_traces(result)

        assert isinstance(traces, dict)
        assert len(traces) == 2
        assert "claude-1" in traces
        assert "claude-2" in traces
        assert "rate limiter" in traces["claude-1"]
        assert "token bucket" in traces["claude-2"]


class TestExtractThinkingTracesEdgeCases:
    """Edge case tests for _extract_thinking_traces."""

    def test_no_metadata_attribute(self):
        """Result without metadata attribute returns empty dict."""
        builder = ExplanationBuilder()
        result = MagicMock(spec=[])  # no attributes

        traces = builder._extract_thinking_traces(result)
        assert traces == {}

    def test_metadata_without_agent_thinking(self):
        """Result metadata without agent_thinking key returns empty dict."""
        builder = ExplanationBuilder()
        result = MagicMock()
        result.metadata = {"some_other_key": "value"}

        traces = builder._extract_thinking_traces(result)
        assert traces == {}

    def test_empty_agent_thinking(self):
        """Empty agent_thinking dict returns empty dict."""
        builder = ExplanationBuilder()
        result = MagicMock()
        result.metadata = {"agent_thinking": {}}

        traces = builder._extract_thinking_traces(result)
        assert traces == {}

    def test_metadata_is_none(self):
        """Result with metadata=None returns empty dict."""
        builder = ExplanationBuilder()
        result = MagicMock()
        result.metadata = None

        traces = builder._extract_thinking_traces(result)
        assert traces == {}
