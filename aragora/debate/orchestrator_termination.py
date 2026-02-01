"""Termination checking lifecycle helpers for Arena debates.

Extracted from orchestrator.py to reduce its size. These functions handle
TerminationChecker initialization with async closures for generation and
judge selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aragora.debate.termination_checker import TerminationChecker

if TYPE_CHECKING:
    from aragora.core import Agent, Message
    from aragora.debate.orchestrator import Arena


def init_termination_checker(arena: Arena) -> None:
    """Initialize the termination checker for early debate termination.

    Creates async closures for generation and judge selection that reference
    the arena instance, then initializes the TerminationChecker.

    Args:
        arena: Arena instance to initialize.
    """

    async def generate_fn(agent: Agent, prompt: str, ctx: list[Message]) -> str:
        return await arena.autonomic.generate(agent, prompt, ctx)

    async def select_judge_fn(proposals: dict[str, str], context: list[Message]) -> Agent:
        return await arena._select_judge(proposals, context)

    arena.termination_checker = TerminationChecker(
        protocol=arena.protocol,
        agents=arena._require_agents() if arena.agents else [],
        generate_fn=generate_fn,
        task=arena.env.task if arena.env else "",
        select_judge_fn=select_judge_fn,
        hooks=arena.hooks,
    )


__all__ = [
    "init_termination_checker",
]
