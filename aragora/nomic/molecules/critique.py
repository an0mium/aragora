"""
Critique phase step executors.

This module contains executors for steps related to the critique phase:
- DebateStepExecutor: Execute steps by routing contested decisions to Arena debate
"""

from __future__ import annotations

import logging
from typing import Any, cast

from aragora.config import DEFAULT_CONSENSUS, DEFAULT_ROUNDS
from aragora.config.settings import get_settings
from aragora.nomic.molecules.base import ConsensusType, MoleculeStep
from aragora.nomic.molecules.proposal import StepExecutor

logger = logging.getLogger(__name__)


class DebateStepExecutor(StepExecutor):
    """
    Execute steps by routing contested decisions to Arena debate.

    Used when a step requires multiple perspectives or when there's
    disagreement about the best approach.

    Config options:
        question: The question to debate
        agents: List of agent names to participate
        rounds: Number of debate rounds
        consensus: Consensus requirement (majority, unanimous, etc.)
    """

    async def execute(self, step: MoleculeStep, context: dict[str, Any]) -> Any:
        """Execute step via Arena debate."""
        question = step.config.get("question", step.name)
        agents_config = step.config.get("agents", get_settings().agent.default_agent_list)
        rounds: int = step.config.get("rounds", DEFAULT_ROUNDS)
        consensus_value: ConsensusType = cast(
            ConsensusType, step.config.get("consensus", DEFAULT_CONSENSUS)
        )

        try:
            from aragora.core import Environment, DebateProtocol
            from aragora.core_types import Agent
            from aragora.debate.orchestrator import Arena
            from aragora.agents.registry import AgentRegistry

            # Get agents
            agents: list[Agent] = []
            for agent_name in agents_config:
                if AgentRegistry.is_registered(agent_name):
                    agent = AgentRegistry.create(agent_name)
                    if agent:
                        agents.append(agent)

            if not agents:
                return {"status": "skipped", "reason": "No agents available"}

            # Create and run debate
            env = Environment(task=question)
            # DebateProtocol's consensus parameter accepts ConsensusType at runtime
            protocol = cast(Any, DebateProtocol)(rounds=rounds, consensus=consensus_value)
            arena = Arena(env, agents, protocol)
            result = await arena.run()

            return {
                "status": "debated",
                "decision": getattr(result, "decision", str(result)),
                "consensus_reached": getattr(result, "consensus_reached", False),
                "rounds_used": getattr(result, "rounds_used", rounds),
            }
        except ImportError as e:
            logger.warning("Debate modules not available: %s", e)
            return {"status": "skipped", "reason": "Debate modules not available"}
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("Debate execution failed: %s", e)
            raise
