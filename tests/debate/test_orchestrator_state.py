"""Tests for orchestrator_state module.

Verifies state accessor functions: require_agents, sync_prompt_builder_state,
get_continuum_context, extract_debate_domain, select_debate_team,
filter_responses_by_quality, should_terminate_early, and select_judge.
"""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.orchestrator_state import (
    extract_debate_domain,
    filter_responses_by_quality,
    get_continuum_context,
    require_agents,
    select_debate_team,
    should_terminate_early,
    sync_prompt_builder_state,
)


class TestRequireAgents:
    def test_returns_agents_when_present(self):
        arena = MagicMock()
        arena.agents = [MagicMock(), MagicMock()]
        result = require_agents(arena)
        assert len(result) == 2

    def test_raises_when_empty(self):
        arena = MagicMock()
        arena.agents = []
        with pytest.raises(ValueError, match="No agents available"):
            require_agents(arena)


class TestSyncPromptBuilderState:
    def test_syncs_all_fields(self):
        arena = MagicMock()
        arena.current_role_assignments = {"agent-1": "critic"}
        arena._cache.historical_context = "history"
        arena.user_suggestions = deque([{"text": "suggestion"}])
        arena._context_delegator.get_continuum_context.return_value = "continuum"

        sync_prompt_builder_state(arena)

        assert arena.prompt_builder.current_role_assignments == {"agent-1": "critic"}
        assert arena.prompt_builder._historical_context_cache == "history"
        assert arena.prompt_builder._continuum_context_cache == "continuum"
        assert arena.prompt_builder.user_suggestions == [{"text": "suggestion"}]


class TestGetContinuumContext:
    def test_delegates_to_context_delegator(self):
        arena = MagicMock()
        arena._context_delegator.get_continuum_context.return_value = "ctx-data"
        result = get_continuum_context(arena)
        assert result == "ctx-data"


class TestExtractDebateDomain:
    def test_returns_cached_domain(self):
        arena = MagicMock()
        arena._cache.has_debate_domain.return_value = True
        arena._cache.debate_domain = "technology"
        result = extract_debate_domain(arena)
        assert result == "technology"

    def test_raises_on_none_cache(self):
        arena = MagicMock()
        arena._cache.has_debate_domain.return_value = True
        arena._cache.debate_domain = None
        with pytest.raises(RuntimeError, match="cache may be corrupted"):
            extract_debate_domain(arena)

    def test_computes_and_caches_domain(self):
        arena = MagicMock()
        arena._cache.has_debate_domain.return_value = False
        arena.env.task = "Design a rate limiter for the API"

        with patch(
            "aragora.debate.orchestrator_state._compute_domain_from_task",
            return_value="engineering",
        ):
            result = extract_debate_domain(arena)

        assert result == "engineering"
        assert arena._cache.debate_domain == "engineering"


class TestSelectDebateTeam:
    def test_delegates_to_agents_module(self):
        arena = MagicMock()
        arena.env = MagicMock()
        arena._cache.has_debate_domain.return_value = True
        arena._cache.debate_domain = "general"
        agents = [MagicMock(), MagicMock(), MagicMock()]

        with patch(
            "aragora.debate.orchestrator_state._agents_select_debate_team",
            return_value=agents[:2],
        ) as mock_select:
            result = select_debate_team(arena, agents)

        assert len(result) == 2
        mock_select.assert_called_once()


class TestFilterResponsesByQuality:
    def test_delegates_to_agents_module(self):
        arena = MagicMock()
        arena.env.task = "test task"
        responses = [("agent-1", "response-1"), ("agent-2", "response-2")]

        with patch(
            "aragora.debate.orchestrator_state._agents_filter_responses_by_quality",
            return_value=responses[:1],
        ):
            result = filter_responses_by_quality(arena, responses, "context")

        assert len(result) == 1


class TestShouldTerminateEarly:
    def test_delegates_to_agents_module(self):
        arena = MagicMock()
        arena.env.task = "test task"
        responses = [("agent-1", "resp")]

        with patch(
            "aragora.debate.orchestrator_state._agents_should_terminate_early",
            return_value=True,
        ):
            result = should_terminate_early(arena, responses, current_round=3)

        assert result is True
