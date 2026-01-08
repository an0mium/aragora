"""
Debate Topology for Critic Selection.

Defines different network topologies that determine which critics
review which proposals during a debate.

Topologies:
- all-to-all: Every critic reviews every proposal (except self)
- round-robin: Each proposal gets one deterministic critic
- ring: Each agent's neighbors in a ring critique their proposals
- star: Hub agent critiques all, or all critique hub
- sparse/random-graph: Random subset based on sparsity factor

Usage:
    selector = TopologySelector.create("ring", agents)
    critics = selector.select_critics("claude", all_critics)
"""

import hashlib
import random
from abc import ABC, abstractmethod
from typing import Protocol, Sequence

from aragora.core import Agent


class TopologyConfig(Protocol):
    """Protocol for topology configuration."""

    @property
    def topology(self) -> str:
        """Topology type: all-to-all, round-robin, ring, star, sparse, random-graph."""
        ...

    @property
    def topology_hub_agent(self) -> str | None:
        """Hub agent name for star topology."""
        ...

    @property
    def topology_sparsity(self) -> float:
        """Sparsity factor (0-1) for sparse/random-graph topologies."""
        ...


class TopologySelector(ABC):
    """Base class for debate topology strategies."""

    @abstractmethod
    def select_critics(self, proposal_agent: str, all_critics: Sequence[Agent]) -> list[Agent]:
        """
        Select which critics should review a proposal.

        Args:
            proposal_agent: Name of the agent who made the proposal
            all_critics: All available critics

        Returns:
            List of critics who should review this proposal
        """
        ...

    @classmethod
    def create(
        cls,
        topology: str,
        agents: Sequence[Agent],
        hub_agent: str | None = None,
        sparsity: float = 0.5,
    ) -> "TopologySelector":
        """
        Factory method to create a topology selector.

        Args:
            topology: Topology type name
            agents: All agents in the debate
            hub_agent: Hub agent name for star topology
            sparsity: Sparsity factor for sparse topologies

        Returns:
            Appropriate TopologySelector instance
        """
        if topology == "all-to-all":
            return AllToAllSelector()
        elif topology == "round-robin":
            return RoundRobinSelector()
        elif topology == "ring":
            return RingSelector(agents)
        elif topology == "star":
            return StarSelector(agents, hub_agent)
        elif topology in ("sparse", "random-graph"):
            return SparseSelector(sparsity)
        else:
            # Default to all-to-all
            return AllToAllSelector()

    @classmethod
    def from_protocol(cls, protocol: TopologyConfig, agents: Sequence[Agent]) -> "TopologySelector":
        """
        Create a topology selector from a debate protocol.

        Args:
            protocol: Protocol with topology configuration
            agents: All agents in the debate

        Returns:
            TopologySelector matching the protocol's topology
        """
        return cls.create(
            topology=protocol.topology,
            agents=agents,
            hub_agent=protocol.topology_hub_agent,
            sparsity=protocol.topology_sparsity,
        )


class AllToAllSelector(TopologySelector):
    """All critics review all proposals (except their own)."""

    def select_critics(self, proposal_agent: str, all_critics: Sequence[Agent]) -> list[Agent]:
        """Select all critics except the proposer."""
        return [c for c in all_critics if c.name != proposal_agent]


class RoundRobinSelector(TopologySelector):
    """Each proposal gets one deterministic critic based on hash."""

    def select_critics(self, proposal_agent: str, all_critics: Sequence[Agent]) -> list[Agent]:
        """Select one critic deterministically based on proposal agent hash."""
        eligible_critics = [c for c in all_critics if c.name != proposal_agent]
        if not eligible_critics:
            return []

        # Sort for deterministic ordering
        eligible_sorted = sorted(eligible_critics, key=lambda c: c.name)

        # Use stable hash for deterministic assignment across Python sessions
        proposal_hash = int(hashlib.sha256(proposal_agent.encode()).hexdigest(), 16)
        proposal_index = proposal_hash % len(eligible_sorted)

        return [eligible_sorted[proposal_index]]


class RingSelector(TopologySelector):
    """Each agent's neighbors in a ring critique their proposals."""

    def __init__(self, agents: Sequence[Agent]):
        """
        Initialize ring selector with agent list.

        Args:
            agents: All agents in the debate (defines ring order)
        """
        self._agent_names = sorted(a.name for a in agents)

    def select_critics(self, proposal_agent: str, all_critics: Sequence[Agent]) -> list[Agent]:
        """Select left and right neighbors in the ring."""
        if proposal_agent not in self._agent_names:
            # Fallback to all critics if agent not in ring
            return [c for c in all_critics if c.name != proposal_agent]

        idx = self._agent_names.index(proposal_agent)
        n = len(self._agent_names)

        # Get left and right neighbors
        left = self._agent_names[(idx - 1) % n]
        right = self._agent_names[(idx + 1) % n]

        return [c for c in all_critics if c.name in (left, right)]


class StarSelector(TopologySelector):
    """Hub agent critiques all, or all critique hub."""

    def __init__(self, agents: Sequence[Agent], hub_agent: str | None = None):
        """
        Initialize star selector with hub agent.

        Args:
            agents: All agents in the debate
            hub_agent: Hub agent name (defaults to first agent alphabetically)
        """
        if hub_agent:
            self._hub = hub_agent
        else:
            # Default hub is first agent alphabetically
            agent_names = sorted(a.name for a in agents)
            self._hub = agent_names[0] if agent_names else ""

    def select_critics(self, proposal_agent: str, all_critics: Sequence[Agent]) -> list[Agent]:
        """
        Select critics based on star topology.

        If proposer is hub: all others critique
        If proposer is not hub: only hub critiques
        """
        if proposal_agent == self._hub:
            # Hub's proposal gets critiqued by all others
            return [c for c in all_critics if c.name != self._hub]
        else:
            # Others' proposals get critiqued only by hub
            return [c for c in all_critics if c.name == self._hub]


class SparseSelector(TopologySelector):
    """Random subset of critics based on sparsity factor."""

    def __init__(self, sparsity: float = 0.5):
        """
        Initialize sparse selector with sparsity factor.

        Args:
            sparsity: Fraction of critics to select (0-1). Higher = more critics.
        """
        self._sparsity = max(0.0, min(1.0, sparsity))

    def select_critics(self, proposal_agent: str, all_critics: Sequence[Agent]) -> list[Agent]:
        """Select a random subset of critics deterministically."""
        available = [c for c in all_critics if c.name != proposal_agent]
        if not available:
            return []

        num_to_select = max(1, int(len(available) * self._sparsity))

        # Use stable hash for deterministic random selection
        stable_seed = int(hashlib.sha256(proposal_agent.encode()).hexdigest(), 16) % (2**32)
        random.seed(stable_seed)
        selected = random.sample(available, min(num_to_select, len(available)))
        random.seed()  # Reset seed

        return selected


def select_critics_for_proposal(
    proposal_agent: str,
    all_critics: Sequence[Agent],
    all_agents: Sequence[Agent],
    topology: str = "all-to-all",
    hub_agent: str | None = None,
    sparsity: float = 0.5,
) -> list[Agent]:
    """
    Convenience function to select critics for a proposal.

    Args:
        proposal_agent: Name of the agent who made the proposal
        all_critics: All available critics
        all_agents: All agents in the debate
        topology: Topology type
        hub_agent: Hub agent name for star topology
        sparsity: Sparsity factor for sparse topologies

    Returns:
        List of critics who should review this proposal
    """
    selector = TopologySelector.create(
        topology=topology,
        agents=all_agents,
        hub_agent=hub_agent,
        sparsity=sparsity,
    )
    return selector.select_critics(proposal_agent, all_critics)
