"""
Molecule-Based Work Tracking (Gastown-inspired).

Molecules are atomic work units for debate phases:
- Capability requirements: what skills are needed
- Completion criteria: how to know when done
- Agent affinity: which agents are best suited
- Nondeterministic idempotence: any agent can resume any molecule

Key concepts from Gastown:
- Work is decomposed into molecules
- Each molecule has clear inputs/outputs
- Molecules can be reassigned on agent failure
- Progress is tracked per-molecule, not per-agent
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aragora.routing.selection import AgentProfile

logger = logging.getLogger(__name__)


class MoleculeStatus(Enum):
    """Status of a work molecule."""

    PENDING = "pending"  # Not yet started
    ASSIGNED = "assigned"  # Assigned to an agent
    IN_PROGRESS = "in_progress"  # Being worked on
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"  # Failed, needs reassignment
    BLOCKED = "blocked"  # Blocked by dependencies


class MoleculeType(Enum):
    """Types of debate work molecules."""

    PROPOSAL = "proposal"  # Generate a proposal
    CRITIQUE = "critique"  # Critique another's proposal
    REVISION = "revision"  # Revise based on critique
    SYNTHESIS = "synthesis"  # Synthesize multiple proposals
    VOTE = "vote"  # Cast a vote
    CONSENSUS_CHECK = "consensus_check"  # Check for consensus
    QUALITY_REVIEW = "quality_review"  # Review argument quality
    FACT_CHECK = "fact_check"  # Verify claims


# Capability requirements for each molecule type
MOLECULE_CAPABILITIES = {
    MoleculeType.PROPOSAL: {"reasoning", "creativity"},
    MoleculeType.CRITIQUE: {"analysis", "quality_assessment"},
    MoleculeType.REVISION: {"reasoning", "synthesis"},
    MoleculeType.SYNTHESIS: {"synthesis", "coordination"},
    MoleculeType.VOTE: {"reasoning"},
    MoleculeType.CONSENSUS_CHECK: {"analysis"},
    MoleculeType.QUALITY_REVIEW: {"quality_assessment"},
    MoleculeType.FACT_CHECK: {"research"},
}


@dataclass
class Molecule:
    """
    Atomic unit of debate work.

    A molecule represents a single, well-defined task that can be:
    - Assigned to any capable agent
    - Reassigned on failure
    - Tracked independently
    - Resumed from checkpoints
    """

    molecule_id: str
    debate_id: str
    molecule_type: MoleculeType
    round_number: int

    # Input/Output
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: Optional[dict[str, Any]] = None

    # Requirements
    required_capabilities: set[str] = field(default_factory=set)
    depends_on: list[str] = field(default_factory=list)  # molecule_ids

    # Assignment
    assigned_agent: Optional[str] = None
    assignment_history: list[str] = field(default_factory=list)

    # Status
    status: MoleculeStatus = MoleculeStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Tracking
    attempts: int = 0
    max_attempts: int = 3
    error_message: Optional[str] = None

    # Affinity (learned over time)
    agent_affinity: dict[str, float] = field(default_factory=dict)  # agent_name -> score

    def __post_init__(self):
        if not self.required_capabilities:
            self.required_capabilities = MOLECULE_CAPABILITIES.get(self.molecule_type, set())

    @classmethod
    def create(
        cls,
        debate_id: str,
        molecule_type: MoleculeType,
        round_number: int,
        input_data: Optional[dict[str, Any]] = None,
        depends_on: Optional[list[str]] = None,
    ) -> "Molecule":
        """Create a new molecule."""
        return cls(
            molecule_id=f"mol-{uuid.uuid4().hex[:8]}",
            debate_id=debate_id,
            molecule_type=molecule_type,
            round_number=round_number,
            input_data=input_data or {},
            depends_on=depends_on or [],
        )

    def assign(self, agent_name: str) -> None:
        """Assign molecule to an agent."""
        self.assigned_agent = agent_name
        self.assignment_history.append(agent_name)
        self.status = MoleculeStatus.ASSIGNED
        self.attempts += 1

    def start(self) -> None:
        """Mark molecule as in progress."""
        self.status = MoleculeStatus.IN_PROGRESS
        self.started_at = datetime.now().isoformat()

    def complete(self, output: dict[str, Any]) -> None:
        """Mark molecule as completed."""
        self.output_data = output
        self.status = MoleculeStatus.COMPLETED
        self.completed_at = datetime.now().isoformat()

        # Update affinity - successful completion increases affinity
        if self.assigned_agent:
            current = self.agent_affinity.get(self.assigned_agent, 0.5)
            self.agent_affinity[self.assigned_agent] = min(1.0, current + 0.1)

    def fail(self, error: str) -> None:
        """Mark molecule as failed."""
        self.error_message = error
        self.status = MoleculeStatus.FAILED

        # Update affinity - failure decreases affinity
        if self.assigned_agent:
            current = self.agent_affinity.get(self.assigned_agent, 0.5)
            self.agent_affinity[self.assigned_agent] = max(0.0, current - 0.2)
            self.assigned_agent = None

    def can_retry(self) -> bool:
        """Check if molecule can be retried."""
        return self.attempts < self.max_attempts

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "molecule_id": self.molecule_id,
            "debate_id": self.debate_id,
            "molecule_type": self.molecule_type.value,
            "round_number": self.round_number,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "required_capabilities": list(self.required_capabilities),
            "depends_on": self.depends_on,
            "assigned_agent": self.assigned_agent,
            "assignment_history": self.assignment_history,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "error_message": self.error_message,
            "agent_affinity": self.agent_affinity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Molecule":
        """Deserialize from dictionary."""
        return cls(
            molecule_id=data["molecule_id"],
            debate_id=data["debate_id"],
            molecule_type=MoleculeType(data["molecule_type"]),
            round_number=data["round_number"],
            input_data=data.get("input_data", {}),
            output_data=data.get("output_data"),
            required_capabilities=set(data.get("required_capabilities", [])),
            depends_on=data.get("depends_on", []),
            assigned_agent=data.get("assigned_agent"),
            assignment_history=data.get("assignment_history", []),
            status=MoleculeStatus(data.get("status", "pending")),
            created_at=data.get("created_at", datetime.now().isoformat()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            attempts=data.get("attempts", 0),
            max_attempts=data.get("max_attempts", 3),
            error_message=data.get("error_message"),
            agent_affinity=data.get("agent_affinity", {}),
        )


class MoleculeTracker:
    """
    Tracks and manages work molecules for debates.

    Provides:
    - Molecule creation and lifecycle management
    - Agent assignment with capability matching
    - Dependency resolution
    - Progress tracking
    - Failure recovery and reassignment
    """

    def __init__(self):
        self._molecules: dict[str, Molecule] = {}  # molecule_id -> Molecule
        self._debate_molecules: dict[str, list[str]] = {}  # debate_id -> [molecule_ids]
        self._agent_workload: dict[str, int] = {}  # agent_name -> active molecule count

    def create_molecule(
        self,
        debate_id: str,
        molecule_type: MoleculeType,
        round_number: int,
        input_data: Optional[dict[str, Any]] = None,
        depends_on: Optional[list[str]] = None,
    ) -> Molecule:
        """Create and track a new molecule."""
        molecule = Molecule.create(
            debate_id=debate_id,
            molecule_type=molecule_type,
            round_number=round_number,
            input_data=input_data,
            depends_on=depends_on,
        )

        self._molecules[molecule.molecule_id] = molecule

        if debate_id not in self._debate_molecules:
            self._debate_molecules[debate_id] = []
        self._debate_molecules[debate_id].append(molecule.molecule_id)

        logger.debug(
            f"Created molecule {molecule.molecule_id} "
            f"type={molecule_type.value} debate={debate_id} round={round_number}"
        )

        return molecule

    def get_molecule(self, molecule_id: str) -> Optional[Molecule]:
        """Get a molecule by ID."""
        return self._molecules.get(molecule_id)

    def get_debate_molecules(self, debate_id: str) -> list[Molecule]:
        """Get all molecules for a debate."""
        molecule_ids = self._debate_molecules.get(debate_id, [])
        return [self._molecules[mid] for mid in molecule_ids if mid in self._molecules]

    def get_pending_molecules(self, debate_id: str) -> list[Molecule]:
        """Get pending molecules that can be assigned."""
        molecules = self.get_debate_molecules(debate_id)
        pending = []

        for mol in molecules:
            if mol.status != MoleculeStatus.PENDING:
                continue

            # Check dependencies
            deps_satisfied = all(
                self._molecules.get(dep_id)
                and self._molecules[dep_id].status == MoleculeStatus.COMPLETED
                for dep_id in mol.depends_on
            )

            if deps_satisfied:
                pending.append(mol)
            else:
                mol.status = MoleculeStatus.BLOCKED

        return pending

    def assign_molecule(
        self,
        molecule_id: str,
        agent: "AgentProfile",
    ) -> bool:
        """Assign a molecule to an agent.

        Checks capability requirements and workload.
        """
        molecule = self._molecules.get(molecule_id)
        if not molecule:
            return False

        if molecule.status not in (MoleculeStatus.PENDING, MoleculeStatus.FAILED):
            return False

        # Check capabilities
        if molecule.required_capabilities and not molecule.required_capabilities.issubset(
            agent.capabilities
        ):
            missing = molecule.required_capabilities - agent.capabilities
            logger.debug(f"Agent {agent.name} missing capabilities: {missing}")
            return False

        # Check if agent can retry failed molecule
        if molecule.status == MoleculeStatus.FAILED and not molecule.can_retry():
            return False

        # Assign
        molecule.assign(agent.name)
        self._agent_workload[agent.name] = self._agent_workload.get(agent.name, 0) + 1

        logger.info(
            f"Assigned molecule {molecule_id} to {agent.name} "
            f"(attempt {molecule.attempts}/{molecule.max_attempts})"
        )

        return True

    def find_best_agent(
        self,
        molecule: Molecule,
        available_agents: list["AgentProfile"],
    ) -> Optional["AgentProfile"]:
        """Find the best agent for a molecule based on capabilities and affinity."""
        candidates = []

        for agent in available_agents:
            # Check capabilities
            if molecule.required_capabilities and not molecule.required_capabilities.issubset(
                agent.capabilities
            ):
                continue

            # Skip if agent already failed this molecule
            if (
                agent.name in molecule.assignment_history
                and molecule.status == MoleculeStatus.FAILED
            ):
                continue

            # Calculate score
            affinity = molecule.agent_affinity.get(agent.name, 0.5)
            elo_score = min(1.0, max(0.0, (agent.elo_rating - 1000) / 1000))
            workload_penalty = self._agent_workload.get(agent.name, 0) * 0.1

            score = (affinity * 0.4 + elo_score * 0.4 + agent.availability * 0.2) - workload_penalty
            candidates.append((agent, score))

        if not candidates:
            return None

        # Return highest scoring agent
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def start_molecule(self, molecule_id: str) -> bool:
        """Mark a molecule as in progress."""
        molecule = self._molecules.get(molecule_id)
        if molecule and molecule.status == MoleculeStatus.ASSIGNED:
            molecule.start()
            return True
        return False

    def complete_molecule(
        self,
        molecule_id: str,
        output: dict[str, Any],
    ) -> bool:
        """Mark a molecule as completed."""
        molecule = self._molecules.get(molecule_id)
        if not molecule:
            return False

        if molecule.status != MoleculeStatus.IN_PROGRESS:
            return False

        molecule.complete(output)

        # Update workload
        if molecule.assigned_agent:
            self._agent_workload[molecule.assigned_agent] = max(
                0, self._agent_workload.get(molecule.assigned_agent, 1) - 1
            )

        # Unblock dependent molecules
        for other_mol in self._molecules.values():
            if molecule_id in other_mol.depends_on:
                if other_mol.status == MoleculeStatus.BLOCKED:
                    # Check if all dependencies now satisfied
                    all_satisfied = all(
                        self._molecules.get(dep_id)
                        and self._molecules[dep_id].status == MoleculeStatus.COMPLETED
                        for dep_id in other_mol.depends_on
                    )
                    if all_satisfied:
                        other_mol.status = MoleculeStatus.PENDING

        logger.info(f"Completed molecule {molecule_id}")
        return True

    def fail_molecule(
        self,
        molecule_id: str,
        error: str,
    ) -> bool:
        """Mark a molecule as failed."""
        molecule = self._molecules.get(molecule_id)
        if not molecule:
            return False

        if molecule.status != MoleculeStatus.IN_PROGRESS:
            return False

        molecule.fail(error)

        # Update workload
        if molecule.assigned_agent:
            self._agent_workload[molecule.assigned_agent] = max(
                0, self._agent_workload.get(molecule.assigned_agent, 1) - 1
            )

        logger.warning(
            f"Molecule {molecule_id} failed: {error} "
            f"(attempts {molecule.attempts}/{molecule.max_attempts})"
        )

        return True

    def get_progress(self, debate_id: str) -> dict[str, Any]:
        """Get progress summary for a debate."""
        molecules = self.get_debate_molecules(debate_id)

        if not molecules:
            return {"debate_id": debate_id, "total": 0, "progress": 0.0}

        status_counts: dict[str, int] = {}
        for mol in molecules:
            status = mol.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        completed = status_counts.get("completed", 0)
        total = len(molecules)

        return {
            "debate_id": debate_id,
            "total": total,
            "completed": completed,
            "progress": completed / total if total > 0 else 0.0,
            "by_status": status_counts,
            "by_type": self._count_by_type(molecules),
        }

    def _count_by_type(self, molecules: list[Molecule]) -> dict[str, dict[str, int]]:
        """Count molecules by type and status."""
        by_type: dict[str, dict[str, int]] = {}
        for mol in molecules:
            mol_type = mol.molecule_type.value
            if mol_type not in by_type:
                by_type[mol_type] = {"total": 0, "completed": 0}
            by_type[mol_type]["total"] += 1
            if mol.status == MoleculeStatus.COMPLETED:
                by_type[mol_type]["completed"] += 1
        return by_type

    def clear_debate(self, debate_id: str) -> None:
        """Clear all molecules for a completed debate."""
        molecule_ids = self._debate_molecules.pop(debate_id, [])
        for mol_id in molecule_ids:
            mol = self._molecules.pop(mol_id, None)
            if mol and mol.assigned_agent:
                self._agent_workload[mol.assigned_agent] = max(
                    0, self._agent_workload.get(mol.assigned_agent, 1) - 1
                )


# Convenience functions
def create_round_molecules(
    tracker: MoleculeTracker,
    debate_id: str,
    round_number: int,
    agent_count: int,
    task: str,
) -> list[Molecule]:
    """Create standard molecules for a debate round."""
    molecules = []

    # Create proposal molecules for each agent
    proposal_ids = []
    for i in range(agent_count):
        mol = tracker.create_molecule(
            debate_id=debate_id,
            molecule_type=MoleculeType.PROPOSAL,
            round_number=round_number,
            input_data={"task": task, "agent_index": i},
        )
        proposal_ids.append(mol.molecule_id)
        molecules.append(mol)

    # Create critique molecules (each agent critiques others)
    critique_ids = []
    for i in range(agent_count):
        for j in range(agent_count):
            if i != j:
                mol = tracker.create_molecule(
                    debate_id=debate_id,
                    molecule_type=MoleculeType.CRITIQUE,
                    round_number=round_number,
                    input_data={"critic_index": i, "target_index": j},
                    depends_on=[proposal_ids[j]],  # Depends on target's proposal
                )
                critique_ids.append(mol.molecule_id)
                molecules.append(mol)

    # Create synthesis molecule (depends on all critiques)
    synthesis_mol = tracker.create_molecule(
        debate_id=debate_id,
        molecule_type=MoleculeType.SYNTHESIS,
        round_number=round_number,
        input_data={"task": task},
        depends_on=critique_ids,
    )
    molecules.append(synthesis_mol)

    return molecules


__all__ = [
    "Molecule",
    "MoleculeStatus",
    "MoleculeTracker",
    "MoleculeType",
    "MOLECULE_CAPABILITIES",
    "create_round_molecules",
]
