"""Tests for NomicLoop._select_debate_team: agent scoring, track assignment, capability matching."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

import scripts.nomic_loop as nomic_loop
from aragora.config.settings import AgentSettings


@dataclass
class DummyAgent:
    name: str
    model: str | None = None


def _make_loop(tmp_path, agent_names=None):
    """Create a NomicLoop with a minimal agent pool and circuit breaker."""
    loop = nomic_loop.NomicLoop(aragora_path=str(tmp_path))

    if agent_names is None:
        agent_names = AgentSettings().default_agent_list

    loop.agent_pool = {name: DummyAgent(name=name, model=name) for name in agent_names}
    loop.agent_selector = None
    loop.elo_system = None
    loop.probe_filter = None
    loop.calibration_tracker = None
    return loop


# ===========================================================================
# force_full_team
# ===========================================================================


def test_select_debate_team_force_full_team(tmp_path, monkeypatch) -> None:
    """force_full_team should bypass AgentSelector and return the full default roster."""
    monkeypatch.setattr(nomic_loop, "SELECTOR_AVAILABLE", True)

    loop = _make_loop(tmp_path)
    loop.agent_selector = object()

    team = loop._select_debate_team("architecture review", force_full_team=True)
    team_names = [agent.name for agent in team]

    assert team_names == AgentSettings().default_agent_list


def test_force_full_team_preserves_order(tmp_path, monkeypatch) -> None:
    """force_full_team should maintain the preferred agent ordering."""
    monkeypatch.setattr(nomic_loop, "SELECTOR_AVAILABLE", True)

    loop = _make_loop(tmp_path)
    team = loop._select_debate_team("testing review", force_full_team=True)
    expected = AgentSettings().default_agent_list
    assert [a.name for a in team] == expected


# ===========================================================================
# No selector fallback
# ===========================================================================


def test_no_selector_returns_default_team(tmp_path, monkeypatch) -> None:
    """Without AgentSelector, default team should be returned."""
    monkeypatch.setattr(nomic_loop, "SELECTOR_AVAILABLE", False)

    loop = _make_loop(tmp_path)
    team = loop._select_debate_team("performance analysis")
    assert len(team) == len(AgentSettings().default_agent_list)


def test_no_selector_no_elo_returns_all_agents(tmp_path, monkeypatch) -> None:
    """Without selector and ELO, all available agents are returned."""
    monkeypatch.setattr(nomic_loop, "SELECTOR_AVAILABLE", False)
    monkeypatch.setattr(nomic_loop, "ELO_AVAILABLE", False)

    loop = _make_loop(tmp_path)
    team = loop._select_debate_team("some task")
    assert len(team) > 0
    assert all(isinstance(a, DummyAgent) for a in team)


# ===========================================================================
# Circuit breaker filtering
# ===========================================================================


def test_circuit_breaker_filters_unavailable_agents(tmp_path, monkeypatch) -> None:
    """Agents in circuit breaker cooldown should be excluded from the team."""
    monkeypatch.setattr(nomic_loop, "SELECTOR_AVAILABLE", False)

    loop = _make_loop(tmp_path, agent_names=["alpha", "beta", "gamma"])
    loop.circuit_breaker.cooldowns["beta"] = 2

    team = loop._select_debate_team("task")
    team_names = [a.name for a in team]
    assert "beta" not in team_names
    assert "alpha" in team_names
    assert "gamma" in team_names


def test_circuit_breaker_fallback_when_too_few_agents(tmp_path, monkeypatch) -> None:
    """When circuit breaker leaves < 2 agents, all agents should be included."""
    monkeypatch.setattr(nomic_loop, "SELECTOR_AVAILABLE", False)

    loop = _make_loop(tmp_path, agent_names=["only-one", "also-bad"])
    loop.circuit_breaker.cooldowns["only-one"] = 1
    loop.circuit_breaker.cooldowns["also-bad"] = 1

    team = loop._select_debate_team("task")
    assert len(team) == 2


# ===========================================================================
# Domain detection
# ===========================================================================


@pytest.mark.parametrize(
    "task,expected_domain",
    [
        ("improve security headers", "security"),
        ("optimize database performance", "performance"),
        ("review architecture patterns", "architecture"),
        ("add unit testing coverage", "testing"),
        ("fix error_handling in module", "error_handling"),
        ("add new feature for users", "general"),
    ],
)
def test_detect_domain(tmp_path, task, expected_domain) -> None:
    """_detect_domain should recognize known domains from task text."""
    loop = _make_loop(tmp_path)
    assert loop._detect_domain(task) == expected_domain


# ===========================================================================
# Empty / missing agent pool
# ===========================================================================


def test_empty_agent_pool_returns_empty(tmp_path, monkeypatch) -> None:
    """With an empty agent pool, the team should be empty."""
    monkeypatch.setattr(nomic_loop, "SELECTOR_AVAILABLE", False)

    loop = _make_loop(tmp_path, agent_names=[])
    team = loop._select_debate_team("some task")
    assert team == []


def test_none_agents_in_pool_filtered(tmp_path, monkeypatch) -> None:
    """None values in agent_pool should be skipped."""
    monkeypatch.setattr(nomic_loop, "SELECTOR_AVAILABLE", False)

    loop = _make_loop(tmp_path, agent_names=["alpha", "beta"])
    loop.agent_pool["alpha"] = None

    team = loop._select_debate_team("task")
    team_names = [a.name for a in team]
    assert "alpha" not in team_names
    assert "beta" in team_names


# ===========================================================================
# Selector error handling
# ===========================================================================


def test_selector_error_falls_back_to_default(tmp_path, monkeypatch) -> None:
    """If AgentSelector.select_team raises, default team is used."""
    monkeypatch.setattr(nomic_loop, "SELECTOR_AVAILABLE", True)

    loop = _make_loop(tmp_path)
    mock_selector = MagicMock()
    mock_selector.select_team.side_effect = RuntimeError("selector crash")
    mock_selector.register_agent = MagicMock()
    loop.agent_selector = mock_selector

    team = loop._select_debate_team("task")
    assert len(team) > 0


# ===========================================================================
# Custom agent lists
# ===========================================================================


def test_custom_agent_pool_size(tmp_path, monkeypatch) -> None:
    """A custom-sized agent pool should produce a matching team."""
    monkeypatch.setattr(nomic_loop, "SELECTOR_AVAILABLE", False)

    names = ["agent-1", "agent-2", "agent-3"]
    loop = _make_loop(tmp_path, agent_names=names)
    team = loop._select_debate_team("any task")
    assert len(team) == 3
    assert {a.name for a in team} == set(names)


def test_single_agent_pool(tmp_path, monkeypatch) -> None:
    """A pool with a single agent should still function."""
    monkeypatch.setattr(nomic_loop, "SELECTOR_AVAILABLE", False)

    loop = _make_loop(tmp_path, agent_names=["solo"])
    team = loop._select_debate_team("task")
    assert len(team) == 1
    assert team[0].name == "solo"


# ===========================================================================
# ELO-based sorting
# ===========================================================================


def test_elo_sorting_applied(tmp_path, monkeypatch) -> None:
    """When ELO is available, agents should be sorted by domain score."""
    monkeypatch.setattr(nomic_loop, "SELECTOR_AVAILABLE", False)
    monkeypatch.setattr(nomic_loop, "ELO_AVAILABLE", True)

    loop = _make_loop(tmp_path, agent_names=["weak", "strong", "mid"])

    mock_elo = MagicMock()

    def mock_best_domains(name, limit=10):
        if name == "strong":
            return [("security", 0.9)]
        if name == "mid":
            return [("security", 0.5)]
        return [("general", 0.1)]

    mock_elo.get_best_domains = mock_best_domains

    class FakeRating:
        elo = 1500

    mock_elo.get_rating = MagicMock(return_value=FakeRating())
    loop.elo_system = mock_elo

    team = loop._select_debate_team("improve security")
    assert team[0].name == "strong"
