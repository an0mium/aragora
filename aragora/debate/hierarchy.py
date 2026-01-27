"""
Agent Hierarchy Management (Gastown-inspired).

Implements role-based agent hierarchy for debate orchestration:
- Orchestrator (Mayor): Coordinates debate flow and synthesis
- Monitor (Witness): Observes for quality, stuck debates, violations
- Worker (Polecat): Executes individual debate tasks

Key concepts from Gastown:
- Role assignment based on capability tags
- Nondeterministic idempotence: any agent can resume any task
- Task affinity tracking for optimal assignment
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from aragora.routing.selection import AgentProfile

logger = logging.getLogger(__name__)


class HierarchyRole(Enum):
    """Agent roles in the debate hierarchy."""

    ORCHESTRATOR = "orchestrator"  # Mayor: coordinates debate flow
    MONITOR = "monitor"  # Witness: observes for issues
    WORKER = "worker"  # Polecat: executes tasks


# Default capability requirements for each role
ROLE_CAPABILITIES = {
    HierarchyRole.ORCHESTRATOR: {"reasoning", "synthesis", "coordination"},
    HierarchyRole.MONITOR: {"analysis", "quality_assessment"},
    HierarchyRole.WORKER: set(),  # Workers can have any capabilities
}

# Capability definitions
STANDARD_CAPABILITIES = {
    "reasoning": "Logical reasoning and argumentation",
    "synthesis": "Combining multiple viewpoints into coherent conclusions",
    "coordination": "Managing multi-agent workflows",
    "analysis": "Deep analysis and critique",
    "quality_assessment": "Evaluating argument quality",
    "coding": "Writing and reviewing code",
    "research": "Information gathering and fact-checking",
    "creativity": "Novel idea generation",
    "mathematics": "Mathematical reasoning",
    "domain_expert": "Deep expertise in specific domains",
}


@dataclass
class RoleAssignment:
    """Assignment of an agent to a role for a specific debate."""

    agent_name: str
    role: HierarchyRole
    assigned_at: str
    capabilities_matched: set[str] = field(default_factory=set)
    affinity_score: float = 0.0


@dataclass
class HierarchyConfig:
    """Configuration for agent hierarchy."""

    # Role counts
    max_orchestrators: int = 1
    max_monitors: int = 2
    min_workers: int = 2

    # Assignment weights
    capability_weight: float = 0.4
    elo_weight: float = 0.3
    affinity_weight: float = 0.3

    # Auto-promote workers to monitors if needed
    auto_promote: bool = True


class AgentHierarchy:
    """
    Manages role-based agent hierarchy for debates.

    Inspired by Gastown's Mayor/Witness/Polecat pattern:
    - Orchestrator (Mayor): Single agent coordinating the debate
    - Monitor (Witness): Agents watching for quality issues
    - Worker (Polecat): Agents executing debate tasks

    The hierarchy enables:
    - Clear responsibility separation
    - Efficient task delegation
    - Quality monitoring without blocking
    - Graceful degradation when agents fail
    """

    def __init__(self, config: Optional[HierarchyConfig] = None):
        self.config = config or HierarchyConfig()
        self._assignments: dict[
            str, dict[str, RoleAssignment]
        ] = {}  # debate_id -> agent_name -> assignment
        self._role_history: dict[
            str, list[tuple[str, HierarchyRole]]
        ] = {}  # debate_id -> [(agent, role), ...]

    def assign_roles(
        self,
        debate_id: str,
        agents: list["AgentProfile"],
        task_type: Optional[str] = None,
    ) -> dict[str, RoleAssignment]:
        """
        Assign roles to agents for a debate.

        Args:
            debate_id: Unique debate identifier
            agents: Available agent profiles
            task_type: Optional task type for affinity matching

        Returns:
            Dict mapping agent names to their role assignments
        """
        from datetime import datetime

        assignments: dict[str, RoleAssignment] = {}

        # Score agents for each role
        orchestrator_candidates = self._score_for_role(
            agents, HierarchyRole.ORCHESTRATOR, task_type
        )
        monitor_candidates = self._score_for_role(agents, HierarchyRole.MONITOR, task_type)

        # Assign orchestrator (highest scoring, single)
        if orchestrator_candidates:
            best_orchestrator = orchestrator_candidates[0]
            assignments[best_orchestrator[0].name] = RoleAssignment(
                agent_name=best_orchestrator[0].name,
                role=HierarchyRole.ORCHESTRATOR,
                assigned_at=datetime.now().isoformat(),
                capabilities_matched=best_orchestrator[2],
                affinity_score=best_orchestrator[1],
            )

        # Assign monitors (up to max)
        monitors_assigned = 0
        for agent, score, caps in monitor_candidates:
            if agent.name in assignments:
                continue
            if monitors_assigned >= self.config.max_monitors:
                break
            assignments[agent.name] = RoleAssignment(
                agent_name=agent.name,
                role=HierarchyRole.MONITOR,
                assigned_at=datetime.now().isoformat(),
                capabilities_matched=caps,
                affinity_score=score,
            )
            monitors_assigned += 1

        # Assign remaining as workers
        for agent in agents:
            if agent.name not in assignments:
                task_affinity = agent.task_affinity.get(task_type, 0.5) if task_type else 0.5
                assignments[agent.name] = RoleAssignment(
                    agent_name=agent.name,
                    role=HierarchyRole.WORKER,
                    assigned_at=datetime.now().isoformat(),
                    capabilities_matched=agent.capabilities,
                    affinity_score=task_affinity,
                )

        # Store assignments
        self._assignments[debate_id] = assignments

        # Track history
        if debate_id not in self._role_history:
            self._role_history[debate_id] = []
        for agent_name, assignment in assignments.items():
            self._role_history[debate_id].append((agent_name, assignment.role))

        logger.info(
            f"Assigned roles for debate {debate_id}: "
            f"orchestrator={self.get_orchestrator(debate_id)}, "
            f"monitors={[a for a, r in assignments.items() if r.role == HierarchyRole.MONITOR]}, "
            f"workers={[a for a, r in assignments.items() if r.role == HierarchyRole.WORKER]}"
        )

        return assignments

    def _score_for_role(
        self,
        agents: list["AgentProfile"],
        role: HierarchyRole,
        task_type: Optional[str] = None,
    ) -> list[tuple["AgentProfile", float, set[str]]]:
        """Score agents for a specific role.

        Returns list of (agent, score, matched_capabilities) sorted by score descending.
        """
        required_caps = ROLE_CAPABILITIES.get(role, set())
        scored = []

        for agent in agents:
            # Capability match score
            matched_caps = agent.capabilities & required_caps
            if required_caps:
                cap_score = len(matched_caps) / len(required_caps)
            else:
                cap_score = 1.0  # Workers don't need specific capabilities

            # ELO score (normalized to 0-1)
            elo_score = min(1.0, max(0.0, (agent.elo_rating - 1000) / 1000))

            # Affinity score
            affinity_score = agent.task_affinity.get(task_type, 0.5) if task_type else 0.5

            # Combined score
            total_score = (
                self.config.capability_weight * cap_score
                + self.config.elo_weight * elo_score
                + self.config.affinity_weight * affinity_score
            )

            scored.append((agent, total_score, matched_caps))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def get_role(self, debate_id: str, agent_name: str) -> Optional[HierarchyRole]:
        """Get the role of an agent in a debate."""
        if debate_id not in self._assignments:
            return None
        assignment = self._assignments[debate_id].get(agent_name)
        return assignment.role if assignment else None

    def get_orchestrator(self, debate_id: str) -> Optional[str]:
        """Get the orchestrator agent for a debate."""
        if debate_id not in self._assignments:
            return None
        for agent_name, assignment in self._assignments[debate_id].items():
            if assignment.role == HierarchyRole.ORCHESTRATOR:
                return agent_name
        return None

    def get_monitors(self, debate_id: str) -> list[str]:
        """Get monitor agents for a debate."""
        if debate_id not in self._assignments:
            return []
        return [
            agent_name
            for agent_name, assignment in self._assignments[debate_id].items()
            if assignment.role == HierarchyRole.MONITOR
        ]

    def get_workers(self, debate_id: str) -> list[str]:
        """Get worker agents for a debate."""
        if debate_id not in self._assignments:
            return []
        return [
            agent_name
            for agent_name, assignment in self._assignments[debate_id].items()
            if assignment.role == HierarchyRole.WORKER
        ]

    def promote_worker(self, debate_id: str, agent_name: str, to_role: HierarchyRole) -> bool:
        """Promote a worker to a higher role (e.g., when orchestrator fails)."""
        if debate_id not in self._assignments:
            return False

        assignment = self._assignments[debate_id].get(agent_name)
        if not assignment or assignment.role != HierarchyRole.WORKER:
            return False

        # Check if promotion is valid
        if to_role == HierarchyRole.ORCHESTRATOR:
            current_orchestrator = self.get_orchestrator(debate_id)
            if current_orchestrator:
                # Demote current orchestrator to worker
                self._assignments[debate_id][current_orchestrator].role = HierarchyRole.WORKER

        assignment.role = to_role
        self._role_history[debate_id].append((agent_name, to_role))
        logger.info(f"Promoted {agent_name} to {to_role.value} in debate {debate_id}")
        return True

    def get_hierarchy_status(self, debate_id: str) -> dict:
        """Get current hierarchy status for a debate."""
        if debate_id not in self._assignments:
            return {"debate_id": debate_id, "status": "not_initialized"}

        return {
            "debate_id": debate_id,
            "orchestrator": self.get_orchestrator(debate_id),
            "monitors": self.get_monitors(debate_id),
            "workers": self.get_workers(debate_id),
            "total_agents": len(self._assignments[debate_id]),
            "role_changes": len(self._role_history.get(debate_id, [])),
        }

    def clear_debate(self, debate_id: str) -> None:
        """Clear hierarchy state for a completed debate."""
        self._assignments.pop(debate_id, None)
        self._role_history.pop(debate_id, None)


# Convenience function for quick role assignment
def assign_debate_roles(
    debate_id: str,
    agents: list["AgentProfile"],
    task_type: Optional[str] = None,
) -> dict[str, RoleAssignment]:
    """Quick role assignment using default configuration."""
    hierarchy = AgentHierarchy()
    return hierarchy.assign_roles(debate_id, agents, task_type)


__all__ = [
    "AgentHierarchy",
    "HierarchyConfig",
    "HierarchyRole",
    "RoleAssignment",
    "ROLE_CAPABILITIES",
    "STANDARD_CAPABILITIES",
    "assign_debate_roles",
]
