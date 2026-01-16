"""
Vote collection orchestration for consensus phase.

Extracted from consensus_phase.py to reduce complexity.
Handles the mechanics of collecting votes from agents with timeout protection.

Key responsibilities:
- Parallel vote collection from all agents
- Timeout protection (per-agent and overall)
- Error tracking for unanimity mode
- Vote grouping for similar choices
- Success callbacks (hooks, recording, position tracking)

Usage:
    collector = VoteCollector(
        vote_with_agent=arena._vote_with_agent,
        with_timeout=arena._with_timeout,
        ...
    )
    votes = await collector.collect_votes(ctx)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from aragora.core import Agent, Vote
    from aragora.debate.context import DebateContext

logger = logging.getLogger(__name__)

# Timeout constants
AGENT_TIMEOUT_SECONDS = 45
VOTE_COLLECTION_TIMEOUT = 180  # Hard cap for collecting all votes


def get_complexity_governor():
    """Get the global complexity governor instance."""
    from aragora.debate.complexity_governor import get_complexity_governor as _get_governor

    return _get_governor()


@dataclass
class VoteCollectorConfig:
    """Configuration for VoteCollector."""

    # Required callback for voting
    vote_with_agent: Optional[Callable] = None

    # Timeout wrapper
    with_timeout: Optional[Callable] = None

    # Notifications
    notify_spectator: Optional[Callable] = None

    # Hooks
    hooks: dict = field(default_factory=dict)

    # Recording
    recorder: Optional[Any] = None
    position_tracker: Optional[Any] = None

    # Vote grouping
    group_similar_votes: Optional[Callable] = None

    # Timeouts
    vote_collection_timeout: float = VOTE_COLLECTION_TIMEOUT
    agent_timeout: float = AGENT_TIMEOUT_SECONDS


class VoteCollector:
    """
    Orchestrates vote collection from debate agents.

    Handles:
    - Parallel vote collection with timeout protection
    - Error tracking for unanimity mode
    - Vote success callbacks (hooks, recording, position tracking)
    - Vote grouping for similar choices
    """

    def __init__(self, config: VoteCollectorConfig):
        """Initialize vote collector with configuration.

        Args:
            config: VoteCollectorConfig with callbacks and settings
        """
        self.config = config
        self._vote_with_agent = config.vote_with_agent
        self._with_timeout = config.with_timeout
        self._notify_spectator = config.notify_spectator
        self.hooks = config.hooks
        self.recorder = config.recorder
        self.position_tracker = config.position_tracker
        self._group_similar_votes = config.group_similar_votes
        self.VOTE_COLLECTION_TIMEOUT = config.vote_collection_timeout

    async def collect_votes(self, ctx: "DebateContext") -> list["Vote"]:
        """Collect votes from all agents with outer timeout protection.

        Uses VOTE_COLLECTION_TIMEOUT to prevent total vote collection time from
        exceeding reasonable bounds (N agents * per-agent timeout could be very long).
        If timeout is reached, returns partial votes collected so far.

        Args:
            ctx: The debate context with agents and proposals

        Returns:
            List of Vote objects from agents that successfully voted
        """
        if not self._vote_with_agent:
            logger.warning("No vote_with_agent callback, skipping votes")
            return []

        votes: list["Vote"] = []
        task = ctx.env.task if ctx.env else ""

        async def cast_vote(agent: "Agent"):
            """Cast a vote for a single agent with timeout protection."""
            logger.debug(f"agent_voting agent={agent.name}")
            try:
                timeout = get_complexity_governor().get_scaled_timeout(
                    float(self.config.agent_timeout)
                )
                if self._with_timeout:
                    vote_result = await self._with_timeout(
                        self._vote_with_agent(agent, ctx.proposals, task),
                        agent.name,
                        timeout_seconds=timeout,
                    )
                else:
                    vote_result = await self._vote_with_agent(agent, ctx.proposals, task)
                return (agent, vote_result)
            except Exception as e:
                logger.warning(f"vote_exception agent={agent.name} error={type(e).__name__}: {e}")
                return (agent, e)

        async def collect_all_votes():
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
                        logger.error(f"vote_error agent={agent.name} error=vote returned None")
                else:
                    votes.append(vote_result)
                    self._handle_vote_success(ctx, agent, vote_result)

        # Apply outer timeout to prevent N*agent_timeout runaway
        try:
            await asyncio.wait_for(collect_all_votes(), timeout=self.VOTE_COLLECTION_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(
                f"vote_collection_timeout collected={len(votes)} "
                f"expected={len(ctx.agents)} timeout={self.VOTE_COLLECTION_TIMEOUT}s"
            )
            # Return partial votes - better than nothing

        return votes

    async def collect_votes_with_errors(self, ctx: "DebateContext") -> tuple[list["Vote"], int]:
        """Collect votes with error tracking for unanimity mode.

        Used for unanimity mode where we need to track errors.
        Uses VOTE_COLLECTION_TIMEOUT to prevent runaway collection time.

        Args:
            ctx: The debate context with agents and proposals

        Returns:
            Tuple of (votes list, error count)
        """
        if not self._vote_with_agent:
            return [], 0

        votes: list["Vote"] = []
        voting_errors = 0
        task = ctx.env.task if ctx.env else ""

        async def cast_vote(agent: "Agent"):
            """Cast a vote for unanimous consensus with timeout protection."""
            logger.debug(f"agent_voting_unanimous agent={agent.name}")
            try:
                timeout = get_complexity_governor().get_scaled_timeout(
                    float(self.config.agent_timeout)
                )
                if self._with_timeout:
                    vote_result = await self._with_timeout(
                        self._vote_with_agent(agent, ctx.proposals, task),
                        agent.name,
                        timeout_seconds=timeout,
                    )
                else:
                    vote_result = await self._vote_with_agent(agent, ctx.proposals, task)
                return (agent, vote_result)
            except Exception as e:
                logger.warning(
                    f"vote_exception_unanimous agent={agent.name} error={type(e).__name__}: {e}"
                )
                return (agent, e)

        async def collect_all_votes():
            """Collect votes from all agents with error counting for unanimity checks."""
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
                        logger.error(f"vote_error_unanimous agent={agent.name} error={vote_result}")
                    else:
                        logger.error(
                            f"vote_error_unanimous agent={agent.name} error=vote returned None"
                        )
                    voting_errors += 1
                else:
                    votes.append(vote_result)
                    self._handle_vote_success(ctx, agent, vote_result, unanimous=True)

        # Apply outer timeout to prevent N*agent_timeout runaway
        try:
            await asyncio.wait_for(collect_all_votes(), timeout=self.VOTE_COLLECTION_TIMEOUT)
        except asyncio.TimeoutError:
            # Treat timeout as errors for missing votes
            missing = len(ctx.agents) - len(votes) - voting_errors
            voting_errors += missing
            logger.warning(
                f"vote_collection_timeout_unanimous collected={len(votes)} "
                f"errors={voting_errors} expected={len(ctx.agents)} "
                f"timeout={self.VOTE_COLLECTION_TIMEOUT}s"
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
            ctx: The debate context
            agent: The agent that voted
            vote: The Vote object
            unanimous: Whether this is for unanimous consensus mode
        """
        result = ctx.result

        logger.debug(
            f"vote_cast{'_unanimous' if unanimous else ''} agent={agent.name} "
            f"choice={vote.choice} confidence={vote.confidence:.0%}"
        )

        # Notify spectator
        if self._notify_spectator:
            self._notify_spectator(
                "vote",
                agent=agent.name,
                details=f"Voted for {vote.choice}",
                metric=vote.confidence,
            )

        # Emit vote hook
        if "on_vote" in self.hooks:
            self.hooks["on_vote"](agent.name, vote.choice, vote.confidence)

        # Record vote
        if self.recorder:
            try:
                self.recorder.record_vote(agent.name, vote.choice, vote.reasoning)
            except Exception as e:
                logger.debug(f"Recorder error for vote: {e}")

        # Record position for truth-grounded personas
        if self.position_tracker:
            try:
                debate_id = (
                    result.id if hasattr(result, "id") else (ctx.env.task[:50] if ctx.env else "")
                )
                self.position_tracker.record_position(
                    debate_id=debate_id,
                    agent_name=agent.name,
                    position_type="vote",
                    position_text=vote.choice,
                    round_num=result.rounds_used if result else 0,
                    confidence=vote.confidence,
                )
            except Exception as e:
                logger.debug(f"Position tracking error for vote: {e}")

    def compute_vote_groups(
        self, votes: list["Vote"]
    ) -> tuple[dict[str, list[str]], dict[str, str]]:
        """Group similar votes and create choice mapping.

        Args:
            votes: List of Vote objects

        Returns:
            Tuple of (vote_groups dict, choice_mapping dict)
            - vote_groups: canonical choice -> list of variant choices
            - choice_mapping: variant choice -> canonical choice
        """
        if not self._group_similar_votes:
            # No grouping, identity mapping
            choices = set(v.choice for v in votes if not isinstance(v, Exception))
            return {c: [c] for c in choices}, {c: c for c in choices}

        vote_groups = self._group_similar_votes(votes)

        choice_mapping: dict[str, str] = {}
        for canonical, variants in vote_groups.items():
            for variant in variants:
                choice_mapping[variant] = canonical

        if vote_groups:
            logger.debug(f"vote_grouping_merged groups={vote_groups}")

        return vote_groups, choice_mapping


# =============================================================================
# Factory function
# =============================================================================


def create_vote_collector(
    vote_with_agent: Optional[Callable] = None,
    with_timeout: Optional[Callable] = None,
    notify_spectator: Optional[Callable] = None,
    hooks: Optional[dict] = None,
    recorder: Optional[Any] = None,
    position_tracker: Optional[Any] = None,
    group_similar_votes: Optional[Callable] = None,
    vote_collection_timeout: float = VOTE_COLLECTION_TIMEOUT,
    agent_timeout: float = AGENT_TIMEOUT_SECONDS,
) -> VoteCollector:
    """Create a VoteCollector with the given configuration.

    Args:
        vote_with_agent: Callback to vote with an agent
        with_timeout: Timeout wrapper function
        notify_spectator: Spectator notification callback
        hooks: Dict of phase hooks
        recorder: Vote recorder instance
        position_tracker: Position tracker instance
        group_similar_votes: Vote grouping callback
        vote_collection_timeout: Overall timeout for vote collection
        agent_timeout: Per-agent voting timeout

    Returns:
        Configured VoteCollector instance
    """
    config = VoteCollectorConfig(
        vote_with_agent=vote_with_agent,
        with_timeout=with_timeout,
        notify_spectator=notify_spectator,
        hooks=hooks or {},
        recorder=recorder,
        position_tracker=position_tracker,
        group_similar_votes=group_similar_votes,
        vote_collection_timeout=vote_collection_timeout,
        agent_timeout=agent_timeout,
    )
    return VoteCollector(config)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "VoteCollector",
    "VoteCollectorConfig",
    "create_vote_collector",
    "VOTE_COLLECTION_TIMEOUT",
    "AGENT_TIMEOUT_SECONDS",
]
