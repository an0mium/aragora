"""
Debate-specific batch utilities built on RLM batch parallelism.

Provides debate-aware wrappers around aragora.rlm.batch.llm_batch that
integrate with debate infrastructure:
- Circuit breaker filtering
- Hook notifications
- Timeout protection
- Progress tracking
- Early termination conditions

Based on Prime Intellect's RLM paper (arXiv:2512.24601) batch parallelism pattern.

Usage:
    from aragora.debate.phases.batch_utils import (
        batch_generate_critiques,
        batch_generate_revisions,
        batch_collect_votes,
        DebateBatchConfig,
    )

    # Generate critiques in parallel with early stopping
    critiques = await batch_generate_critiques(
        critics=critic_agents,
        proposals=proposals,
        generate_fn=generate_single_critique,
        config=DebateBatchConfig(
            max_concurrent=3,
            min_required=2,  # Early stop after 2 critiques
            circuit_breaker=arena.circuit_breaker,
        ),
    )
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Generic,
    Optional,
    TypeVar,
)

from aragora.rlm.batch import (
    BatchConfig,
    BatchItemResult,
    BatchItemStatus,
    BatchResult,
    llm_batch,
    llm_batch_detailed,
)

if TYPE_CHECKING:
    from aragora.core import Agent, Critique, Vote
    from aragora.resilience import CircuitBreaker

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class DebateBatchConfig:
    """Configuration for debate-specific batch operations."""

    max_concurrent: int = 3
    """Maximum concurrent LLM calls. Lower than general to respect rate limits."""

    timeout_per_item: float = 45.0
    """Timeout per agent operation in seconds."""

    min_required: Optional[int] = None
    """Minimum results before allowing early stop. None = no early stop."""

    stagger_delay: float = 0.0
    """Delay between starting each operation to avoid API burst."""

    circuit_breaker: Optional["CircuitBreaker"] = None
    """Optional circuit breaker for agent availability filtering."""

    hooks: dict[str, Callable] = field(default_factory=dict)
    """Event hooks for notifications."""

    notify_spectator: Optional[Callable] = None
    """Spectator notification callback."""

    fail_fast: bool = False
    """Stop all processing on first failure."""

    record_to_breaker: bool = True
    """Record success/failure to circuit breaker."""


@dataclass
class DebateBatchResult(Generic[R]):
    """Result of a debate batch operation with debate-specific metadata."""

    results: list[R]
    """Successful results."""

    errors: list[tuple[str, Exception]]
    """Errors with agent names."""

    early_stopped: bool = False
    """Whether early stopping was triggered."""

    total_agents: int = 0
    """Total agents attempted."""

    successful_agents: list[str] = field(default_factory=list)
    """Names of agents that succeeded."""

    failed_agents: list[str] = field(default_factory=list)
    """Names of agents that failed."""

    @property
    def success_rate(self) -> float:
        """Fraction of successful operations."""
        if self.total_agents == 0:
            return 0.0
        return len(self.successful_agents) / self.total_agents


async def batch_with_agents(
    agents: list["Agent"],
    process_fn: Callable[["Agent"], Awaitable[R]],
    config: Optional[DebateBatchConfig] = None,
    operation_name: str = "operation",
) -> DebateBatchResult[R]:
    """
    Execute an async operation on multiple agents in parallel.

    This is the core debate batch utility that integrates with debate infrastructure.

    Args:
        agents: List of agents to process
        process_fn: Async function that takes an agent and returns a result
        config: Debate batch configuration
        operation_name: Name for logging and notifications

    Returns:
        DebateBatchResult with results and metadata

    Example:
        results = await batch_with_agents(
            agents=critics,
            process_fn=lambda agent: generate_critique(agent, proposal),
            config=DebateBatchConfig(max_concurrent=3),
            operation_name="critique",
        )
    """
    cfg = config or DebateBatchConfig()

    # Filter agents through circuit breaker if available
    available_agents = agents
    if cfg.circuit_breaker:
        try:
            available_agents = cfg.circuit_breaker.filter_available_agents(agents)
            if len(available_agents) < len(agents):
                skipped = [a.name for a in agents if a not in available_agents]
                logger.info(f"batch_{operation_name}_circuit_breaker_skip agents={skipped}")
        except Exception as e:
            logger.warning(f"Circuit breaker filter error: {e}")

    if not available_agents:
        logger.warning(f"batch_{operation_name}_no_available_agents")
        return DebateBatchResult(
            results=[],
            errors=[],
            total_agents=len(agents),
        )

    # Build wrapper that tracks agent info
    agent_results: dict[str, Any] = {}

    async def process_with_tracking(agent: "Agent") -> tuple[str, R]:
        """Wrap process_fn to track agent info and apply stagger."""
        # Apply stagger delay if configured
        if cfg.stagger_delay > 0:
            idx = available_agents.index(agent)
            await asyncio.sleep(idx * cfg.stagger_delay)

        logger.debug(f"batch_{operation_name}_start agent={agent.name}")
        result = await process_fn(agent)
        logger.debug(f"batch_{operation_name}_complete agent={agent.name}")
        return (agent.name, result)

    # Early stop condition
    early_stop_fn = None
    if cfg.min_required is not None:
        def early_stop_fn(results: list) -> bool:
            return len(results) >= cfg.min_required

    # Convert config
    batch_config = BatchConfig(
        max_concurrent=cfg.max_concurrent,
        timeout_per_item=cfg.timeout_per_item,
        fail_fast=cfg.fail_fast,
        retry_on_error=False,  # Handle errors at debate level
    )

    # Execute batch
    batch_result = await llm_batch_detailed(
        items=available_agents,
        process_fn=process_with_tracking,
        early_stop=early_stop_fn,
        config=batch_config,
    )

    # Process results
    successful_results: list[R] = []
    errors: list[tuple[str, Exception]] = []
    successful_agents: list[str] = []
    failed_agents: list[str] = []

    for item_result in batch_result.items:
        agent = item_result.item
        agent_name = agent.name

        if item_result.status == BatchItemStatus.COMPLETED and item_result.result:
            _, result = item_result.result  # Unwrap (agent_name, result) tuple
            successful_results.append(result)
            successful_agents.append(agent_name)

            # Record success to circuit breaker
            if cfg.record_to_breaker and cfg.circuit_breaker:
                cfg.circuit_breaker.record_success(agent_name)

            # Notify spectator
            if cfg.notify_spectator:
                cfg.notify_spectator(
                    operation_name,
                    agent=agent_name,
                    details=f"{operation_name} completed",
                )

        else:
            error = item_result.error or Exception(f"Status: {item_result.status}")
            errors.append((agent_name, error))
            failed_agents.append(agent_name)

            # Record failure to circuit breaker
            if cfg.record_to_breaker and cfg.circuit_breaker:
                cfg.circuit_breaker.record_failure(agent_name)

            logger.warning(f"batch_{operation_name}_failed agent={agent_name} error={error}")

    # Emit hooks
    if "on_batch_complete" in cfg.hooks:
        cfg.hooks["on_batch_complete"](
            operation=operation_name,
            success_count=len(successful_agents),
            failure_count=len(failed_agents),
            early_stopped=batch_result.early_stopped,
        )

    return DebateBatchResult(
        results=successful_results,
        errors=errors,
        early_stopped=batch_result.early_stopped,
        total_agents=len(agents),
        successful_agents=successful_agents,
        failed_agents=failed_agents,
    )


async def batch_generate_critiques(
    critics: list["Agent"],
    proposals: dict[str, str],
    generate_fn: Callable[["Agent", str, str], Awaitable["Critique"]],
    config: Optional[DebateBatchConfig] = None,
) -> list["Critique"]:
    """
    Generate critiques from multiple critics in parallel.

    Args:
        critics: List of critic agents
        proposals: Dict mapping proposer name to proposal content
        generate_fn: Async function(critic, proposer_name, proposal) -> Critique
        config: Batch configuration

    Returns:
        List of generated Critique objects

    Example:
        critiques = await batch_generate_critiques(
            critics=critic_agents,
            proposals={"alice": "My proposal...", "bob": "Another proposal..."},
            generate_fn=lambda critic, proposer, proposal: generate_critique(critic, proposer, proposal),
            config=DebateBatchConfig(max_concurrent=3, min_required=2),
        )
    """
    cfg = config or DebateBatchConfig()

    # Build list of (critic, proposer, proposal) tuples
    critique_tasks: list[tuple["Agent", str, str]] = []
    for proposer_name, proposal in proposals.items():
        for critic in critics:
            if critic.name != proposer_name:  # Don't critique self
                critique_tasks.append((critic, proposer_name, proposal))

    if not critique_tasks:
        return []

    # Early stop when enough critiques collected
    early_stop_fn = None
    if cfg.min_required is not None:
        def early_stop_fn(results: list) -> bool:
            return len(results) >= cfg.min_required

    async def generate_with_tracking(task_tuple: tuple) -> "Critique":
        critic, proposer_name, proposal = task_tuple
        logger.debug(f"batch_critique_start critic={critic.name} target={proposer_name}")

        if cfg.stagger_delay > 0:
            idx = critique_tasks.index(task_tuple)
            await asyncio.sleep(idx * cfg.stagger_delay)

        critique = await generate_fn(critic, proposer_name, proposal)
        logger.debug(f"batch_critique_complete critic={critic.name} target={proposer_name}")
        return critique

    batch_config = BatchConfig(
        max_concurrent=cfg.max_concurrent,
        timeout_per_item=cfg.timeout_per_item,
        fail_fast=cfg.fail_fast,
    )

    results = await llm_batch(
        items=critique_tasks,
        process_fn=generate_with_tracking,
        early_stop=early_stop_fn,
        config=batch_config,
    )

    logger.info(f"batch_critique_results count={len(results)} tasks={len(critique_tasks)}")
    return results


async def batch_collect_votes(
    agents: list["Agent"],
    proposals: dict[str, str],
    vote_fn: Callable[["Agent", dict[str, str]], Awaitable["Vote"]],
    config: Optional[DebateBatchConfig] = None,
    majority_threshold: float = 0.5,
) -> tuple[list["Vote"], bool, Optional[str]]:
    """
    Collect votes from agents with RLM-style early termination.

    Stops early when a clear majority is reached.

    Args:
        agents: List of voting agents
        proposals: Dict mapping proposer name to proposal
        vote_fn: Async function(agent, proposals) -> Vote
        config: Batch configuration
        majority_threshold: Fraction of agents needed for majority (default 0.5)

    Returns:
        Tuple of (votes, early_stopped, winning_choice or None)

    Example:
        votes, early_stopped, winner = await batch_collect_votes(
            agents=voting_agents,
            proposals=proposals,
            vote_fn=lambda agent, props: collect_vote(agent, props),
            majority_threshold=0.51,  # Simple majority
        )
    """
    cfg = config or DebateBatchConfig()
    collected_votes: list["Vote"] = []
    total_agents = len(agents)
    early_stopped = False
    winning_choice: Optional[str] = None

    def check_majority(votes: list) -> bool:
        """Check if clear majority reached for early termination."""
        nonlocal winning_choice

        if len(votes) < total_agents * 0.5:  # Need at least half collected
            return False

        # Count votes by choice
        vote_counts: dict[str, int] = {}
        for vote in votes:
            if hasattr(vote, 'choice') and vote.choice:
                vote_counts[vote.choice] = vote_counts.get(vote.choice, 0) + 1

        if not vote_counts:
            return False

        # Check if leader has clear majority
        leader_choice, leader_count = max(vote_counts.items(), key=lambda x: x[1])
        if leader_count > total_agents * majority_threshold:
            # Check lead over second choice
            second_count = 0
            for choice, count in vote_counts.items():
                if choice != leader_choice and count > second_count:
                    second_count = count

            lead = leader_count - second_count
            min_lead = max(1, int(total_agents * 0.25))  # 25% lead

            if lead >= min_lead:
                winning_choice = leader_choice
                logger.info(
                    f"batch_vote_early_termination leader={leader_choice} "
                    f"votes={leader_count}/{len(votes)} lead={lead}"
                )
                return True

        return False

    async def cast_vote(agent: "Agent") -> "Vote":
        if cfg.stagger_delay > 0:
            idx = agents.index(agent)
            await asyncio.sleep(idx * cfg.stagger_delay)

        vote = await vote_fn(agent, proposals)
        collected_votes.append(vote)
        return vote

    batch_config = BatchConfig(
        max_concurrent=cfg.max_concurrent,
        timeout_per_item=cfg.timeout_per_item,
    )

    batch_result = await llm_batch_detailed(
        items=agents,
        process_fn=cast_vote,
        early_stop=check_majority,
        config=batch_config,
    )

    # Extract successful votes
    votes = [
        item.result
        for item in batch_result.items
        if item.status == BatchItemStatus.COMPLETED and item.result is not None
    ]

    early_stopped = batch_result.early_stopped

    # Notify about early termination
    if early_stopped and cfg.notify_spectator:
        cfg.notify_spectator(
            "rlm_early_termination",
            details=f"Majority for '{winning_choice}' ({len(votes)}/{total_agents} votes)",
            agent="system",
        )

    return votes, early_stopped, winning_choice


__all__ = [
    "DebateBatchConfig",
    "DebateBatchResult",
    "batch_with_agents",
    "batch_generate_critiques",
    "batch_collect_votes",
]
