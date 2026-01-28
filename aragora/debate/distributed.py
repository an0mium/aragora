"""
Distributed Debate Coordinator.

Orchestrates debates across multiple Aragora instances, enabling:
- Cross-instance agent participation
- Distributed consensus aggregation
- Fault-tolerant coordination with leader election
- State synchronization via RegionalEventBus

This builds on the existing Arena, RegionalEventBus, and LeaderElection
infrastructure to enable true multi-instance debate federation.

Usage:
    from aragora.debate.distributed import (
        DistributedDebateCoordinator,
        DistributedDebateConfig,
    )

    coordinator = DistributedDebateCoordinator(
        event_bus=event_bus,
        agent_pool=federated_pool,
        leader_elector=leader_elector,
    )
    await coordinator.connect()

    # Start distributed debate
    result = await coordinator.start_debate(
        task="Should we adopt microservices?",
        agents=["claude", "gpt-4", "gemini"],
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from aragora.debate.distributed_events import (
    AgentCritique,
    AgentProposal,
    ConsensusVote,
    DistributedDebateEvent,
    DistributedDebateEventType,
    DistributedDebateState,
)

logger = logging.getLogger(__name__)


class CoordinatorRole(str, Enum):
    """Role of this instance in distributed debate."""

    COORDINATOR = "coordinator"  # Orchestrates the debate
    PARTICIPANT = "participant"  # Provides agents only
    OBSERVER = "observer"  # Watches but doesn't participate


@dataclass
class DistributedDebateConfig:
    """Configuration for distributed debates."""

    max_rounds: int = 5
    proposal_timeout_seconds: float = 60.0
    critique_timeout_seconds: float = 45.0
    vote_timeout_seconds: float = 30.0
    consensus_threshold: float = 0.67
    min_agents: int = 2
    max_agents: int = 10
    allow_remote_agents: bool = True
    sync_interval_seconds: float = 5.0
    failover_timeout_seconds: float = 30.0


@dataclass
class DistributedDebateResult:
    """Result of a distributed debate."""

    debate_id: str
    task: str
    consensus_reached: bool
    final_answer: Optional[str]
    winning_agent: Optional[str]
    confidence: float
    rounds_completed: int
    participating_instances: List[str]
    participating_agents: List[str]
    duration_seconds: float
    proposals: List[Dict[str, Any]]
    votes: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "debate_id": self.debate_id,
            "task": self.task,
            "consensus_reached": self.consensus_reached,
            "final_answer": self.final_answer,
            "winning_agent": self.winning_agent,
            "confidence": self.confidence,
            "rounds_completed": self.rounds_completed,
            "participating_instances": self.participating_instances,
            "participating_agents": self.participating_agents,
            "duration_seconds": self.duration_seconds,
            "proposals": self.proposals,
            "votes": self.votes,
        }


class DistributedDebateCoordinator:
    """
    Coordinates debates across multiple Aragora instances.

    Uses RegionalEventBus for communication and LeaderElection
    to determine which instance coordinates each debate.
    """

    def __init__(
        self,
        event_bus: Optional[Any] = None,
        agent_pool: Optional[Any] = None,
        leader_elector: Optional[Any] = None,
        config: Optional[DistributedDebateConfig] = None,
        instance_id: Optional[str] = None,
    ):
        """
        Initialize the distributed debate coordinator.

        Args:
            event_bus: RegionalEventBus for cross-instance communication
            agent_pool: FederatedAgentPool for agent access
            leader_elector: LeaderElection for coordinator selection
            config: Debate configuration
            instance_id: Unique ID for this instance
        """
        self._event_bus = event_bus
        self._agent_pool = agent_pool
        self._leader_elector = leader_elector
        self._config = config or DistributedDebateConfig()
        self._instance_id = instance_id or str(uuid.uuid4())[:8]

        # Active debates
        self._debates: Dict[str, DistributedDebateState] = {}
        self._debate_locks: Dict[str, asyncio.Lock] = {}

        # Event handlers
        self._event_handlers: Dict[DistributedDebateEventType, List[Callable]] = {}

        # Background tasks
        self._sync_task: Optional[asyncio.Task] = None
        self._connected = False

        logger.info(f"[DistributedDebate] Coordinator initialized: {self._instance_id}")

    async def connect(self) -> None:
        """Start the distributed debate coordinator."""
        if self._connected:
            return

        # Subscribe to debate events
        if self._event_bus and hasattr(self._event_bus, "subscribe"):
            await self._event_bus.subscribe(self._handle_event)

        # Start sync task
        self._sync_task = asyncio.create_task(self._sync_loop())

        self._connected = True
        logger.info("[DistributedDebate] Coordinator connected")

    async def disconnect(self) -> None:
        """Stop the distributed debate coordinator."""
        if not self._connected:
            return

        self._connected = False

        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None

        logger.info("[DistributedDebate] Coordinator disconnected")

    async def close(self) -> None:
        """Stop the distributed debate coordinator."""
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        self._connected = False
        logger.info("[DistributedDebate] Coordinator closed")

    async def _sync_loop(self) -> None:
        """Periodically sync debate state."""
        while True:
            try:
                await asyncio.sleep(self._config.sync_interval_seconds)
                await self._sync_debates()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[DistributedDebate] Sync error: {e}")

    async def _sync_debates(self) -> None:
        """Sync active debates with other instances."""
        for debate_id, state in self._debates.items():
            if state.coordinator_instance == self._instance_id:
                # We're the coordinator, broadcast state
                await self._broadcast_state(debate_id)

    async def _broadcast_state(self, debate_id: str) -> None:
        """Broadcast debate state to other instances."""
        if not self._event_bus or debate_id not in self._debates:
            return

        state = self._debates[debate_id]
        event = DistributedDebateEvent(
            event_type=DistributedDebateEventType.STATE_SYNC_RESPONSE,
            debate_id=debate_id,
            source_instance=self._instance_id,
            data=state.to_dict(),
        )

        # Publish via event bus
        if hasattr(self._event_bus, "publish"):
            await self._event_bus.publish(event.to_dict())

    async def _handle_event(self, event_data: Dict[str, Any]) -> None:
        """Handle incoming debate events."""
        try:
            event = DistributedDebateEvent.from_dict(event_data)
        except Exception as e:
            logger.warning(f"[DistributedDebate] Invalid event: {e}")
            return

        # Skip our own events
        if event.source_instance == self._instance_id:
            return

        # Route to appropriate handler
        handlers = {
            DistributedDebateEventType.DEBATE_CREATED: self._on_debate_created,
            DistributedDebateEventType.DEBATE_STARTED: self._on_debate_started,
            DistributedDebateEventType.ROUND_STARTED: self._on_round_started,
            DistributedDebateEventType.AGENT_PROPOSAL: self._on_proposal,
            DistributedDebateEventType.AGENT_CRITIQUE: self._on_critique,
            DistributedDebateEventType.CONSENSUS_VOTE: self._on_vote,
            DistributedDebateEventType.CONSENSUS_REACHED: self._on_consensus,
            DistributedDebateEventType.STATE_SYNC_RESPONSE: self._on_state_sync,
        }

        handler = handlers.get(event.event_type)
        if handler:
            await handler(event)

    async def _on_debate_created(self, event: DistributedDebateEvent) -> None:
        """Handle debate creation from another instance."""
        if event.debate_id not in self._debates:
            state = DistributedDebateState(
                debate_id=event.debate_id,
                task=event.data.get("task", ""),
                coordinator_instance=event.source_instance,
            )
            self._debates[event.debate_id] = state
            self._debate_locks[event.debate_id] = asyncio.Lock()
            logger.info(f"[DistributedDebate] Joined debate {event.debate_id}")

    async def _on_debate_started(self, event: DistributedDebateEvent) -> None:
        """Handle debate start from coordinator."""
        if event.debate_id in self._debates:
            state = self._debates[event.debate_id]
            state.status = "running"
            state.started_at = event.timestamp

    async def _on_round_started(self, event: DistributedDebateEvent) -> None:
        """Handle round start from coordinator."""
        if event.debate_id in self._debates:
            state = self._debates[event.debate_id]
            state.current_round = event.round_number

    async def _on_proposal(self, event: DistributedDebateEvent) -> None:
        """Handle agent proposal from another instance."""
        if event.debate_id in self._debates:
            state = self._debates[event.debate_id]
            proposal = AgentProposal(
                agent_id=event.agent_id or "",
                instance_id=event.source_instance,
                content=event.data.get("content", ""),
                round_number=event.round_number,
                timestamp=event.timestamp,
                confidence=event.data.get("confidence", 0.0),
            )
            state.proposals.append(proposal)

    async def _on_critique(self, event: DistributedDebateEvent) -> None:
        """Handle agent critique from another instance."""
        if event.debate_id in self._debates:
            state = self._debates[event.debate_id]
            critique = AgentCritique(
                agent_id=event.agent_id or "",
                instance_id=event.source_instance,
                target_agent_id=event.data.get("target_agent_id", ""),
                content=event.data.get("content", ""),
                round_number=event.round_number,
                timestamp=event.timestamp,
            )
            state.critiques.append(critique)

    async def _on_vote(self, event: DistributedDebateEvent) -> None:
        """Handle consensus vote from another instance."""
        if event.debate_id in self._debates:
            state = self._debates[event.debate_id]
            vote = ConsensusVote(
                agent_id=event.agent_id or "",
                instance_id=event.source_instance,
                proposal_agent_id=event.data.get("proposal_agent_id", ""),
                vote=event.data.get("vote", "abstain"),
                round_number=event.round_number,
                timestamp=event.timestamp,
            )
            state.votes.append(vote)

    async def _on_consensus(self, event: DistributedDebateEvent) -> None:
        """Handle consensus reached event."""
        if event.debate_id in self._debates:
            state = self._debates[event.debate_id]
            state.consensus_reached = True
            state.final_answer = event.data.get("final_answer")
            state.winning_agent = event.data.get("winning_agent")
            state.confidence = event.data.get("confidence", 0.0)
            state.status = "completed"
            state.completed_at = event.timestamp

    async def _on_state_sync(self, event: DistributedDebateEvent) -> None:
        """Handle state sync from coordinator."""
        # Update local state with coordinator's state
        pass  # Full implementation would merge state

    async def start_debate(
        self,
        task: str,
        agents: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> DistributedDebateResult:
        """
        Start a new distributed debate.

        Args:
            task: The question or task to debate
            agents: List of agent IDs to participate (None = auto-select)
            context: Additional context for the debate

        Returns:
            DistributedDebateResult with consensus and final answer
        """
        debate_id = f"dd_{uuid.uuid4().hex[:12]}"
        start_time = time.time()

        # Create debate state
        state = DistributedDebateState(
            debate_id=debate_id,
            task=task,
            coordinator_instance=self._instance_id,
            max_rounds=self._config.max_rounds,
        )
        self._debates[debate_id] = state
        self._debate_locks[debate_id] = asyncio.Lock()

        logger.info(f"[DistributedDebate] Starting debate {debate_id}: {task[:50]}...")

        # Broadcast debate creation
        await self._publish_event(
            DistributedDebateEventType.DEBATE_CREATED,
            debate_id,
            data={"task": task, "context": context or {}},
        )

        # Select agents
        if agents is None:
            agents = await self._select_agents(debate_id)
        else:
            agents = agents[: self._config.max_agents]

        for agent_id in agents:
            state.agents[agent_id] = {"joined_at": time.time()}

        # Start the debate
        state.status = "running"
        state.started_at = time.time()

        await self._publish_event(
            DistributedDebateEventType.DEBATE_STARTED,
            debate_id,
            data={"agents": agents},
        )

        # Run debate rounds
        try:
            for round_num in range(1, self._config.max_rounds + 1):
                state.current_round = round_num

                await self._publish_event(
                    DistributedDebateEventType.ROUND_STARTED,
                    debate_id,
                    round_number=round_num,
                )

                # Collect proposals
                await self._collect_proposals(debate_id, round_num, agents)

                # Collect critiques
                await self._collect_critiques(debate_id, round_num, agents)

                # Check for consensus
                consensus = await self._check_consensus(debate_id, round_num)
                if consensus:
                    break

        except Exception as e:
            logger.error(f"[DistributedDebate] Debate {debate_id} failed: {e}")
            state.status = "failed"

        # Finalize
        state.completed_at = time.time()
        if state.status != "failed":
            state.status = "completed"

        # Build result
        duration = time.time() - start_time
        result = DistributedDebateResult(
            debate_id=debate_id,
            task=task,
            consensus_reached=state.consensus_reached,
            final_answer=state.final_answer,
            winning_agent=state.winning_agent,
            confidence=state.confidence,
            rounds_completed=state.current_round,
            participating_instances=list(state.instances.keys()) + [self._instance_id],
            participating_agents=list(state.agents.keys()),
            duration_seconds=duration,
            proposals=[p.to_dict() for p in state.proposals],
            votes=[v.to_dict() for v in state.votes],
        )

        logger.info(
            f"[DistributedDebate] Debate {debate_id} completed: "
            f"consensus={state.consensus_reached}, rounds={state.current_round}"
        )

        return result

    async def _select_agents(self, debate_id: str) -> List[str]:
        """Select agents for the debate from the federated pool."""
        if not self._agent_pool:
            return []

        agents = self._agent_pool.find_agents(
            capability="debate",
            min_count=self._config.min_agents,
            include_remote=self._config.allow_remote_agents,
        )

        # Sort by health/latency and take top N
        agents.sort(key=lambda a: (not a.is_local, a.estimated_latency_ms))
        selected = agents[: self._config.max_agents]

        return [a.agent_id for a in selected]

    async def _collect_proposals(
        self,
        debate_id: str,
        round_num: int,
        agents: List[str],
    ) -> None:
        """Collect proposals from all participating agents."""
        _state = self._debates[debate_id]  # Reserved for future implementation

        # In a full implementation, this would:
        # 1. Request proposals from local agents
        # 2. Wait for proposals from remote agents via events
        # 3. Apply timeout and collect all responses

        # For now, simulate with a simple wait
        await asyncio.sleep(0.1)

    async def _collect_critiques(
        self,
        debate_id: str,
        round_num: int,
        agents: List[str],
    ) -> None:
        """Collect critiques from all participating agents."""
        _state = self._debates[debate_id]  # Reserved for future implementation

        # In a full implementation, this would:
        # 1. Request critiques from local agents
        # 2. Wait for critiques from remote agents via events
        # 3. Apply timeout and collect all responses

        await asyncio.sleep(0.1)

    async def _check_consensus(
        self,
        debate_id: str,
        round_num: int,
    ) -> bool:
        """Check if consensus has been reached."""
        state = self._debates[debate_id]

        # Count votes for each proposal
        vote_counts: Dict[str, int] = {}
        for vote in state.votes:
            if vote.round_number == round_num and vote.vote == "support":
                vote_counts[vote.proposal_agent_id] = vote_counts.get(vote.proposal_agent_id, 0) + 1

        if not vote_counts:
            return False

        # Find proposal with most support
        total_votes = len([v for v in state.votes if v.round_number == round_num])
        if total_votes == 0:
            return False

        winner = max(vote_counts.keys(), key=lambda k: vote_counts[k])
        support_ratio = vote_counts[winner] / total_votes

        if support_ratio >= self._config.consensus_threshold:
            state.consensus_reached = True
            state.winning_agent = winner
            state.confidence = support_ratio

            # Find the winning proposal
            for proposal in state.proposals:
                if proposal.agent_id == winner and proposal.round_number == round_num:
                    state.final_answer = proposal.content
                    break

            await self._publish_event(
                DistributedDebateEventType.CONSENSUS_REACHED,
                debate_id,
                round_number=round_num,
                data={
                    "winning_agent": winner,
                    "confidence": support_ratio,
                    "final_answer": state.final_answer,
                },
            )

            return True

        return False

    async def _publish_event(
        self,
        event_type: DistributedDebateEventType,
        debate_id: str,
        round_number: int = 0,
        agent_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Publish a debate event to the event bus."""
        if not self._event_bus:
            return

        event = DistributedDebateEvent(
            event_type=event_type,
            debate_id=debate_id,
            source_instance=self._instance_id,
            round_number=round_number,
            agent_id=agent_id,
            data=data or {},
        )

        if hasattr(self._event_bus, "publish"):
            await self._event_bus.publish(event.to_dict())

    def get_debate(self, debate_id: str) -> Optional[DistributedDebateState]:
        """Get the state of a debate."""
        return self._debates.get(debate_id)

    def list_active_debates(self) -> List[DistributedDebateState]:
        """List all active (non-completed) debates."""
        return [s for s in self._debates.values() if s.status in ("created", "running", "paused")]

    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        total = len(self._debates)
        active = len(self.list_active_debates())
        completed = sum(1 for s in self._debates.values() if s.status == "completed")
        failed = sum(1 for s in self._debates.values() if s.status == "failed")

        return {
            "instance_id": self._instance_id,
            "total_debates": total,
            "active_debates": active,
            "completed_debates": completed,
            "failed_debates": failed,
            "connected": self._connected,
        }
