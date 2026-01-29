from __future__ import annotations

from dataclasses import dataclass

import scripts.nomic_loop as nomic_loop
from aragora.config.settings import AgentSettings


@dataclass
class DummyAgent:
    name: str
    model: str | None = None


def test_select_debate_team_force_full_team(tmp_path, monkeypatch) -> None:
    """force_full_team should bypass AgentSelector and return the full default roster."""
    monkeypatch.setattr(nomic_loop, "SELECTOR_AVAILABLE", True)

    loop = nomic_loop.NomicLoop(aragora_path=str(tmp_path))

    agent_names = AgentSettings().default_agent_list
    loop.agent_pool = {name: DummyAgent(name=name, model=name) for name in agent_names}

    # Ensure selector presence doesn't reduce the team
    loop.agent_selector = object()
    loop.elo_system = None
    loop.probe_filter = None

    team = loop._select_debate_team("architecture review", force_full_team=True)
    team_names = [agent.name for agent in team]

    assert team_names == agent_names
