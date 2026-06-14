#!/usr/bin/env python3
"""Relocated end-to-end debate smoke check (originally repo-root ``test_debate.py``).

Requires the live ``codex`` CLI agent and real model access, so it is skipped by
default and kept for manual end-to-end exercise of the Arena pipeline.
"""

import pytest

from aragora.agents import create_agent
from aragora.core import Environment
from aragora.debate import Arena, DebateProtocol
from aragora.memory import CritiqueStore


@pytest.mark.skip(reason="requires live codex CLI agent + model access; manual end-to-end only")
async def test_simple_debate():
    """Run a simple 2-agent debate using Codex."""
    agents = [
        create_agent("codex", name="proposer", role="proposer"),
        create_agent("codex", name="critic", role="critic"),
    ]
    env = Environment(
        task=(
            "Design a simple in-memory cache in Python with TTL (time-to-live) "
            "support. Keep it under 50 lines."
        ),
        max_rounds=2,
    )
    protocol = DebateProtocol(rounds=2, consensus="majority")
    memory = CritiqueStore("/tmp/aragora_test.db")

    arena = Arena(env, agents, protocol, memory)
    result = await arena.run()

    assert result is not None
