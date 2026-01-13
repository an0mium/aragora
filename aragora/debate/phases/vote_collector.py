"""
Vote collection for consensus phase.

Extracted from consensus_phase.py to improve maintainability and testability.
Handles concurrent vote collection from agents with timeout protection.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

from aragora.debate.complexity_governor import get_complexity_governor

if TYPE_CHECKING:
    from aragora.core import Agent, Vote
    from aragora.debate.context import DebateContext

logger = logging.getLogger(__name__)

# Default timeout for collecting all votes (seconds)
DEFAULT_VOTE_COLLECTION_TIMEOUT = 45.0
# Default per-agent timeout (seconds)
DEFAULT_AGENT_TIMEOUT = 30.0


@dataclass
class VoteCollectorConfig:
    """Configuration for vote collection."""

    vote_collection_timeout: float = DEFAULT_VOTE_COLLECTION_TIMEOUT
    agent_timeout: float = DEFAULT_AGENT_TIMEOUT


@dataclass
class VoteCollectorCallbacks:
    """Callbacks for vote collection events."""

    vote_with_agent: Optional[Callable[..., Awaitable[Any]]] = None
    with_timeout: Optional[Callable[..., Awaitable[Any]]] = None
    notify_spectator: Optional[Callable[..., None]] = None
    group_similar_votes: Optional[Callable[..., dict]] = None


@dataclass
class VoteCollectorDeps:
    """Optional dependencies for vote tracking."""

    hooks: dict = field(default_factory=dict)
    recorder: Optional[Any] = None
    position_tracker: Optional[Any] = None


class VoteCollector:
    """Collects votes from agents with timeout protection.

    Handles both standard and unanimous vote collection modes,
    with proper error tracking and partial result handling.

    Example:
        collector = VoteCollector(
            config=VoteCollectorConfig(),
            callbacks=VoteCollectorCallbacks(vote_with_agent=my_vote_fn),
        )
        votes = await collector.collect_votes(ctx)
    """

    def __init__(
        self,
        config: Optional[VoteCollectorConfig] = None,
        callbacks: Optional[VoteCollectorCallbacks] = None,
        deps: Optional[VoteCollectorDeps] = None,
    ) -> None:
        """Initialize the vote collector.

        Args:
            config: Timeout configuration.
            callbacks: Callbacks for voting and events.
            deps: Optional dependencies for recording.
        """
        self.config = config or VoteCollectorConfig()
        self.callbacks = callbacks or VoteCollectorCallbacks()
        self.deps = deps or VoteCollectorDeps()

    async def collect_votes(self, ctx: "DebateContext") -> list["Vote"]:
        """Collect votes from all agents with outer timeout protection.

        Uses configured timeout to prevent total vote collection time from
        exceeding reasonable bounds. If timeout is reached, returns partial
        votes collected so far.

        Args:
            ctx: The debate context with agents and proposals.

        Returns:
            List of Vote objects from agents that responded in time.
        """
        if not self.callbacks.vote_with_agent:
            logger.warning("No vote_with_agent callback, skipping votes")
            return []

        votes: list["Vote"] = []
        task = ctx.env.task if ctx.env else ""

        async def cast_vote(agent: "Agent") -> tuple["Agent", Any]:
            """Cast a vote for a single agent with timeout protection."""
            logger.debug(f"agent_voting agent={agent.name}")
            try:
                timeout = get_complexity_governor().get_scaled_timeout(
                    self.config.agent_timeout
                )
                if self.callbacks.with_timeout:
                    vote_result = await self.callbacks.with_timeout(
                        self.callbacks.vote_with_agent(agent, ctx.proposals, task),
                        agent.name,
                        timeout_seconds=timeout,
                    )
                else:
                    vote_result = await self.callbacks.vote_with_agent(
                        agent, ctx.proposals, task
                    )
                return (agent, vote_result)
            except Exception as e:
                logger.warning(
                    f"vote_exception agent={agent.name} error={type(e).__name__}: {e}"
                )
                return (agent, e)

        async def collect_all_votes() -> None:
            """Collect votes from all agents concurrently."""
            vote_tasks = [asyncio.create_task(cast_vote(agent)) for agent in ctx.agents]

            for completed_task in asyncio.as_completed(vote_tasks):
                try:
                    agent, vote_result = await completed_task
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"task_exception phase=vote error={e}")
                    continue

                if vote_result is None or isinstance(vote_result, Exception):
                    if isinstance(vote_result, Exception):
                        logger.error(f"vote_error agent={agent.name} error={vote_result}")
                    else:
                        logger.error(
                            f"vote_error agent={agent.name} error=vote returned None"
                        )
                else:
                    votes.append(vote_result)
                    self._handle_vote_success(ctx, agent, vote_result)

        try:
            await asyncio.wait_for(
                collect_all_votes(), timeout=self.config.vote_collection_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"vote_collection_timeout collected={len(votes)} "
                f"expected={len(ctx.agents)} timeout={self.config.vote_collection_timeout}s"
            )

        return votes

    async def collect_votes_with_errors(
        self, ctx: "DebateContext"
    ) -> tuple[list["Vote"], int]:
        """Collect votes with error tracking for unanimity mode.

        Uses outer timeout to prevent runaway collection time.

        Args:
            ctx: The debate context with agents and proposals.

        Returns:
            Tuple of (votes, error_count) where error_count includes
            timeouts, exceptions, and None results.
        """
        if not self.callbacks.vote_with_agent:
            return [], 0

        votes: list["Vote"] = []
        voting_errors = 0
        task = ctx.env.task if ctx.env else ""

        async def cast_vote(agent: "Agent") -> tuple["Agent", Any]:
            """Cast a vote for unanimous consensus with timeout protection."""
            logger.debug(f"agent_voting_unanimous agent={agent.name}")
            try:
                timeout = get_complexity_governor().get_scaled_timeout(
                    self.config.agent_timeout
                )
                if self.callbacks.with_timeout:
                    vote_result = await self.callbacks.with_timeout(
                        self.callbacks.vote_with_agent(agent, ctx.proposals, task),
                        agent.name,
                        timeout_seconds=timeout,
                    )
                else:
                    vote_result = await self.callbacks.vote_with_agent(
                        agent, ctx.proposals, task
                    )
                return (agent, vote_result)
            except Exception as e:
                logger.warning(
                    f"vote_exception_unanimous agent={agent.name} "
                    f"error={type(e).__name__}: {e}"
                )
                return (agent, e)

        async def collect_all_votes() -> None:
            """Collect votes with error counting for unanimity checks."""
            nonlocal voting_errors
            vote_tasks = [asyncio.create_task(cast_vote(agent)) for agent in ctx.agents]

            for completed_task in asyncio.as_completed(vote_tasks):
                try:
                    agent, vote_result = await completed_task
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"task_exception phase=unanimous_vote error={e}")
                    voting_errors += 1
                    continue

                if vote_result is None or isinstance(vote_result, Exception):
                    if isinstance(vote_result, Exception):
                        logger.error(
                            f"vote_error_unanimous agent={agent.name} error={vote_result}"
                        )
                    else:
                        logger.error(
                            f"vote_error_unanimous agent={agent.name} "
                            "error=vote returned None"
                        )
                    voting_errors += 1
                else:
                    votes.append(vote_result)
                    self._handle_vote_success(ctx, agent, vote_result, unanimous=True)

        try:
            await asyncio.wait_for(
                collect_all_votes(), timeout=self.config.vote_collection_timeout
            )
        except asyncio.TimeoutError:
            missing = len(ctx.agents) - len(votes) - voting_errors
            voting_errors += missing
            logger.warning(
                f"vote_collection_timeout_unanimous collected={len(votes)} "
                f"errors={voting_errors} expected={len(ctx.agents)} "
                f"timeout={self.config.vote_collection_timeout}s"
            )

        return votes, voting_errors

    def _handle_vote_success(
        self,
        ctx: "DebateContext",
        agent: "Agent",
        vote: "Vote",
        unanimous: bool = False,
    ) -> None:
        """Handle successful vote: notifications, hooks, recording.

        Args:
            ctx: The debate context.
            agent: The agent that voted.
            vote: The vote cast.
            unanimous: Whether this is for unanimous consensus mode.
        """
        result = ctx.result

        logger.debug(
            f"vote_cast{'_unanimous' if unanimous else ''} agent={agent.name} "
            f"choice={vote.choice} confidence={vote.confidence:.0%}"
        )

        # Notify spectator
        if self.callbacks.notify_spectator:
            self.callbacks.notify_spectator(
                "vote",
                agent=agent.name,
                details=f"Voted for {vote.choice}",
                metric=vote.confidence,
            )

        # Emit vote hook
        if "on_vote" in self.deps.hooks:
            self.deps.hooks["on_vote"](agent.name, vote.choice, vote.confidence)

        # Record vote
        if self.deps.recorder:
            try:
                self.deps.recorder.record_vote(agent.name, vote.choice, vote.reasoning)
            except Exception as e:
                logger.debug(f"Recorder error for vote: {e}")

        # Record position for truth-grounded personas
        if self.deps.position_tracker:
            try:
                debate_id = (
                    result.id
                    if hasattr(result, "id")
                    else (ctx.env.task[:50] if ctx.env else "")
                )
                self.deps.position_tracker.record_position(
                    debate_id=debate_id,
                    agent_name=agent.name,
                    position_type="vote",
                    position_text=vote.choice,
                    round_num=result.rounds_used,
                    confidence=vote.confidence,
                )
            except Exception as e:
                logger.debug(f"Position tracking error for vote: {e}")

    def compute_vote_groups(
        self, votes: list["Vote"]
    ) -> tuple[dict[str, list[str]], dict[str, str]]:
        """Group similar votes and create choice mapping.

        Args:
            votes: List of votes to group.

        Returns:
            Tuple of (vote_groups, choice_mapping) where vote_groups maps
            canonical choices to variants, and choice_mapping maps each
            variant to its canonical form.
        """
        if not self.callbacks.group_similar_votes:
            # No grouping, identity mapping
            choices = set(v.choice for v in votes if not isinstance(v, Exception))
            return {c: [c] for c in choices}, {c: c for c in choices}

        vote_groups = self.callbacks.group_similar_votes(votes)

        choice_mapping: dict[str, str] = {}
        for canonical, variants in vote_groups.items():
            for variant in variants:
                choice_mapping[variant] = canonical

        return vote_groups, choice_mapping
