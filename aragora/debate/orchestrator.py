"""Minimal async debate orchestrator for the standalone debate wedge."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from aragora.core import Critique, DebateResult, Environment, Message, Vote
from aragora.debate.protocol import DebateProtocol, resolve_default_protocol


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _compute_domain_from_task(task_lower: str) -> str:
    """Compatibility helper retained for legacy orchestrator tests/imports."""
    task_lower = task_lower.lower()
    if any(w in task_lower for w in ("security", "hack", "vulnerability", "auth", "encrypt")):
        return "security"
    if any(w in task_lower for w in ("performance", "speed", "optimize", "cache", "latency")):
        return "performance"
    if any(w in task_lower for w in ("test", "testing", "coverage", "regression")):
        return "testing"
    if any(w in task_lower for w in ("design", "architecture", "pattern", "structure")):
        return "architecture"
    if any(w in task_lower for w in ("bug", "error", "fix", "crash", "exception")):
        return "debugging"
    if any(w in task_lower for w in ("api", "endpoint", "rest", "graphql")):
        return "api"
    if any(w in task_lower for w in ("database", "sql", "query", "schema")):
        return "database"
    if any(w in task_lower for w in ("ui", "frontend", "react", "css", "layout")):
        return "frontend"
    return "general"


class Arena:
    """Run a bounded offline debate with mock or real agents.

    The standalone wedge intentionally supports the core path only:
    proposals, optional critiques, optional votes, and a synthesized final answer.
    """

    def __init__(
        self,
        environment: Environment,
        agents: list[Any],
        protocol: DebateProtocol | None = None,
        **_: Any,
    ) -> None:
        if not agents:
            raise ValueError("Arena requires at least one agent")
        self.env = environment
        self.agents = agents
        self.protocol = resolve_default_protocol(protocol)

    @classmethod
    def from_config(
        cls,
        environment: Environment,
        agents: list[Any],
        protocol: DebateProtocol | None = None,
        config: Any | None = None,
    ) -> "Arena":
        del config
        return cls(environment=environment, agents=agents, protocol=protocol)

    @classmethod
    def from_configs(
        cls,
        environment: Environment,
        agents: list[Any],
        protocol: DebateProtocol | None = None,
        **kwargs: Any,
    ) -> "Arena":
        del kwargs
        return cls(environment=environment, agents=agents, protocol=protocol)

    @classmethod
    def create(
        cls,
        environment: Environment,
        agents: list[Any],
        protocol: DebateProtocol | None = None,
        **kwargs: Any,
    ) -> "Arena":
        del kwargs
        return cls(environment=environment, agents=agents, protocol=protocol)

    async def run(self, correlation_id: str = "") -> DebateResult:
        del correlation_id
        timeout = max(int(self.protocol.timeout_seconds), 1)
        return await asyncio.wait_for(self._run_inner(), timeout=timeout)

    async def _run_inner(self) -> DebateResult:
        messages: list[Message] = []
        critiques: list[Critique] = []
        votes: list[Vote] = []
        proposals: dict[str, str] = {}

        for round_number in range(1, self.protocol.rounds + 1):
            for agent in self.agents:
                name = getattr(agent, "name", f"agent_{len(proposals) + 1}")
                content = await _maybe_await(agent.generate(self.env.task))
                content_text = str(content)
                proposals[name] = content_text
                messages.append(
                    Message(
                        role=getattr(agent, "role", "proposer"),
                        agent=name,
                        content=content_text,
                        round=round_number,
                    )
                )
            if self.protocol.early_stopping:
                break

        if self.protocol.critique_required and len(proposals) > 1:
            proposal_items = list(proposals.items())
            for index, agent in enumerate(self.agents):
                if not hasattr(agent, "critique"):
                    continue
                target_name, target_content = proposal_items[(index + 1) % len(proposal_items)]
                critique_value = await _maybe_await(
                    agent.critique(target_content, self.env.task, context=messages)
                )
                if isinstance(critique_value, Critique):
                    critiques.append(critique_value)
                else:
                    critiques.append(
                        Critique(
                            agent=getattr(agent, "name", f"critic_{index + 1}"),
                            target_agent=target_name,
                            target_content=target_content,
                            issues=[],
                            suggestions=[],
                            severity=0.0,
                            reasoning=str(critique_value),
                        )
                    )

        final_answer = next(iter(proposals.values()))
        consensus_reached = len(set(proposals.values())) == 1 or len(proposals) == 1
        confidence = 1.0 if consensus_reached else 0.5

        if self.protocol.consensus != "none":
            for agent in self.agents:
                name = getattr(agent, "name", "agent")
                if hasattr(agent, "vote"):
                    vote_value = await _maybe_await(agent.vote(proposals, self.env.task))
                    if isinstance(vote_value, Vote):
                        votes.append(vote_value)
                        continue
                    choice = getattr(vote_value, "choice", final_answer)
                    reasoning = getattr(vote_value, "reasoning", str(vote_value))
                else:
                    choice = final_answer
                    reasoning = "Selected the current leading proposal."
                votes.append(Vote(agent=name, choice=str(choice), reasoning=str(reasoning)))

            if votes:
                winner_counts: dict[str, int] = {}
                for vote in votes:
                    winner_counts[vote.choice] = winner_counts.get(vote.choice, 0) + 1
                final_answer = max(winner_counts, key=winner_counts.get)
                consensus_reached = winner_counts[final_answer] >= max(
                    1, int(len(votes) * self.protocol.consensus_threshold)
                )
                confidence = winner_counts[final_answer] / len(votes)

        return DebateResult(
            debate_id="standalone-debate",
            task=self.env.task,
            final_answer=final_answer,
            confidence=confidence,
            consensus_reached=consensus_reached,
            rounds_used=self.protocol.rounds,
            rounds_completed=self.protocol.rounds,
            status="completed",
            participants=[getattr(agent, "name", "agent") for agent in self.agents],
            proposals=proposals,
            messages=messages,
            critiques=critiques,
            votes=votes,
        )

    async def _gather_trending_context(self) -> None:
        """Compatibility stub for integration-test fixtures."""
        return None
