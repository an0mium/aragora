"""
Gastown Handoff Protocol: Context Transfer Between Beads and Molecules.

Provides formal context transfer mechanisms ensuring no critical information
is lost when work passes between agents, beads, or molecule steps.

Key concepts:
- BeadHandoffContext: Rich context for bead-to-bead transfers
- MoleculeHandoffContext: Step-to-step context within molecules
- HandoffProtocol: Serialize, validate, merge contexts

Inspired by Gastown's pinned handoff pattern where each role maintains
a persistent message queue for asynchronous context delivery.

Usage:
    # Create handoff context when completing work
    context = BeadHandoffContext.create(
        source_bead_id="task-123",
        findings=["Key insight 1", "Key insight 2"],
        artifacts={"report.md": "/path/to/report.md"},
        decisions=["Chose approach A over B because..."],
        next_steps=["Implement the solution", "Run tests"],
    )

    # Attach to target bead
    await handoff_protocol.transfer(context, target_bead_id="task-456")

    # Recover handoff on agent startup
    contexts = await handoff_protocol.recover_pending(agent_id="claude-001")
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HandoffStatus(str, Enum):
    """Status of a handoff context."""

    PENDING = "pending"  # Awaiting pickup by target
    DELIVERED = "delivered"  # Successfully delivered
    ACKNOWLEDGED = "acknowledged"  # Target confirmed receipt
    EXPIRED = "expired"  # TTL exceeded without pickup
    REJECTED = "rejected"  # Target rejected the handoff


class HandoffPriority(int, Enum):
    """Priority levels for handoffs."""

    LOW = 0
    NORMAL = 50
    HIGH = 75
    CRITICAL = 100


@dataclass
class BeadHandoffContext:
    """
    Rich context for transferring work between beads.

    Contains all relevant information gathered during work execution
    that should be passed to the next worker or bead in the chain.
    """

    id: str
    source_bead_id: str
    target_bead_id: Optional[str]
    source_agent_id: str
    target_agent_id: Optional[str]
    status: HandoffStatus
    priority: HandoffPriority
    created_at: datetime
    expires_at: Optional[datetime]

    # Core handoff content
    findings: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)  # name -> path
    decisions: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Context from execution
    execution_summary: str = ""
    duration_seconds: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Delivery tracking
    delivered_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None

    @classmethod
    def create(
        cls,
        source_bead_id: str,
        source_agent_id: str,
        target_bead_id: Optional[str] = None,
        target_agent_id: Optional[str] = None,
        findings: Optional[List[str]] = None,
        artifacts: Optional[Dict[str, str]] = None,
        decisions: Optional[List[str]] = None,
        next_steps: Optional[List[str]] = None,
        open_questions: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        execution_summary: str = "",
        priority: HandoffPriority = HandoffPriority.NORMAL,
        ttl_hours: float = 24.0,
    ) -> "BeadHandoffContext":
        """Create a new bead handoff context."""
        import uuid
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=ttl_hours) if ttl_hours > 0 else None

        return cls(
            id=f"handoff-{uuid.uuid4().hex[:12]}",
            source_bead_id=source_bead_id,
            target_bead_id=target_bead_id,
            source_agent_id=source_agent_id,
            target_agent_id=target_agent_id,
            status=HandoffStatus.PENDING,
            priority=priority,
            created_at=now,
            expires_at=expires_at,
            findings=findings or [],
            artifacts=artifacts or {},
            decisions=decisions or [],
            next_steps=next_steps or [],
            open_questions=open_questions or [],
            warnings=warnings or [],
            execution_summary=execution_summary,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert enums to strings
        data["status"] = self.status.value
        data["priority"] = self.priority.value
        # Convert datetimes to ISO strings
        for key in ["created_at", "expires_at", "delivered_at", "acknowledged_at"]:
            if data[key] is not None:
                data[key] = data[key].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BeadHandoffContext":
        """Create from dictionary."""
        data = data.copy()
        data["status"] = HandoffStatus(data["status"])
        data["priority"] = HandoffPriority(data["priority"])
        for key in ["created_at", "expires_at", "delivered_at", "acknowledged_at"]:
            if data[key] is not None:
                data[key] = datetime.fromisoformat(data[key])
        return cls(**data)

    def is_expired(self) -> bool:
        """Check if handoff has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def format_for_prompt(self) -> str:
        """Format handoff context for LLM prompt injection."""
        lines = [
            "## Handoff Context",
            f"From: {self.source_agent_id} (bead: {self.source_bead_id})",
            "",
        ]

        if self.execution_summary:
            lines.append(f"### Summary\n{self.execution_summary}\n")

        if self.findings:
            lines.append("### Key Findings")
            for finding in self.findings:
                lines.append(f"- {finding}")
            lines.append("")

        if self.decisions:
            lines.append("### Decisions Made")
            for decision in self.decisions:
                lines.append(f"- {decision}")
            lines.append("")

        if self.next_steps:
            lines.append("### Next Steps")
            for step in self.next_steps:
                lines.append(f"- [ ] {step}")
            lines.append("")

        if self.open_questions:
            lines.append("### Open Questions")
            for question in self.open_questions:
                lines.append(f"- {question}")
            lines.append("")

        if self.warnings:
            lines.append("### Warnings")
            for warning in self.warnings:
                lines.append(f"- {warning}")
            lines.append("")

        if self.artifacts:
            lines.append("### Artifacts")
            for name, path in self.artifacts.items():
                lines.append(f"- {name}: `{path}`")
            lines.append("")

        return "\n".join(lines)


@dataclass
class MoleculeHandoffContext:
    """
    Context for transferring state between molecule steps.

    Captures checkpoint data and intermediate results for
    step-to-step transitions within a molecule workflow.
    """

    id: str
    molecule_id: str
    source_step: str
    target_step: str
    created_at: datetime

    # Step execution results
    step_output: Any = None
    step_success: bool = True
    step_duration_seconds: float = 0.0

    # Checkpoint data
    checkpoint_data: Dict[str, Any] = field(default_factory=dict)

    # Accumulated context from prior steps
    accumulated_findings: List[str] = field(default_factory=list)
    accumulated_artifacts: Dict[str, str] = field(default_factory=dict)

    # Control flow
    should_skip_remaining: bool = False
    skip_reason: Optional[str] = None

    @classmethod
    def create(
        cls,
        molecule_id: str,
        source_step: str,
        target_step: str,
        step_output: Any = None,
        step_success: bool = True,
    ) -> "MoleculeHandoffContext":
        """Create a new molecule handoff context."""
        import uuid

        return cls(
            id=f"mol-handoff-{uuid.uuid4().hex[:12]}",
            molecule_id=molecule_id,
            source_step=source_step,
            target_step=target_step,
            created_at=datetime.now(timezone.utc),
            step_output=step_output,
            step_success=step_success,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MoleculeHandoffContext":
        """Create from dictionary."""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class HandoffStore:
    """
    JSONL-backed storage for handoff contexts.

    Provides persistent storage for handoffs with recovery on restart.
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self._handoffs: Dict[str, BeadHandoffContext] = {}
        self._molecule_handoffs: Dict[str, MoleculeHandoffContext] = {}

    async def initialize(self) -> None:
        """Initialize storage, loading existing handoffs."""
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load bead handoffs
        bead_path = self.storage_path / "bead_handoffs.jsonl"
        if bead_path.exists():
            with open(bead_path) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        handoff = BeadHandoffContext.from_dict(data)
                        self._handoffs[handoff.id] = handoff
            logger.info(f"Loaded {len(self._handoffs)} bead handoffs")

        # Load molecule handoffs
        mol_path = self.storage_path / "molecule_handoffs.jsonl"
        if mol_path.exists():
            with open(mol_path) as f:
                for line in f:
                    if line.strip():
                        mol_data = json.loads(line)
                        mol_handoff = MoleculeHandoffContext.from_dict(mol_data)
                        self._molecule_handoffs[mol_handoff.id] = mol_handoff
            logger.info(f"Loaded {len(self._molecule_handoffs)} molecule handoffs")

    async def save_bead_handoff(self, handoff: BeadHandoffContext) -> None:
        """Save a bead handoff context."""
        self._handoffs[handoff.id] = handoff
        await self._persist_bead_handoffs()

    async def save_molecule_handoff(self, handoff: MoleculeHandoffContext) -> None:
        """Save a molecule handoff context."""
        self._molecule_handoffs[handoff.id] = handoff
        await self._persist_molecule_handoffs()

    async def get_bead_handoff(self, handoff_id: str) -> Optional[BeadHandoffContext]:
        """Get a bead handoff by ID."""
        return self._handoffs.get(handoff_id)

    async def get_molecule_handoff(self, handoff_id: str) -> Optional[MoleculeHandoffContext]:
        """Get a molecule handoff by ID."""
        return self._molecule_handoffs.get(handoff_id)

    async def get_pending_for_agent(self, agent_id: str) -> List[BeadHandoffContext]:
        """Get all pending handoffs for an agent."""
        return [
            h
            for h in self._handoffs.values()
            if h.target_agent_id == agent_id
            and h.status == HandoffStatus.PENDING
            and not h.is_expired()
        ]

    async def get_pending_for_bead(self, bead_id: str) -> List[BeadHandoffContext]:
        """Get all pending handoffs for a bead."""
        return [
            h
            for h in self._handoffs.values()
            if h.target_bead_id == bead_id
            and h.status == HandoffStatus.PENDING
            and not h.is_expired()
        ]

    async def mark_delivered(self, handoff_id: str) -> None:
        """Mark a handoff as delivered."""
        if handoff_id in self._handoffs:
            self._handoffs[handoff_id].status = HandoffStatus.DELIVERED
            self._handoffs[handoff_id].delivered_at = datetime.now(timezone.utc)
            await self._persist_bead_handoffs()

    async def mark_acknowledged(self, handoff_id: str) -> None:
        """Mark a handoff as acknowledged."""
        if handoff_id in self._handoffs:
            self._handoffs[handoff_id].status = HandoffStatus.ACKNOWLEDGED
            self._handoffs[handoff_id].acknowledged_at = datetime.now(timezone.utc)
            await self._persist_bead_handoffs()

    async def cleanup_expired(self) -> int:
        """Remove expired handoffs. Returns count of removed."""
        expired = [h.id for h in self._handoffs.values() if h.is_expired()]
        for hid in expired:
            self._handoffs[hid].status = HandoffStatus.EXPIRED
        if expired:
            await self._persist_bead_handoffs()
        return len(expired)

    async def _persist_bead_handoffs(self) -> None:
        """Persist bead handoffs to JSONL."""
        path = self.storage_path / "bead_handoffs.jsonl"
        with open(path, "w") as f:
            for handoff in self._handoffs.values():
                f.write(json.dumps(handoff.to_dict()) + "\n")

    async def _persist_molecule_handoffs(self) -> None:
        """Persist molecule handoffs to JSONL."""
        path = self.storage_path / "molecule_handoffs.jsonl"
        with open(path, "w") as f:
            for handoff in self._molecule_handoffs.values():
                f.write(json.dumps(handoff.to_dict()) + "\n")


class HandoffProtocol:
    """
    Protocol for managing handoff lifecycle.

    Provides methods for creating, transferring, and recovering
    handoff contexts between agents and beads.
    """

    def __init__(self, store: HandoffStore):
        self.store = store

    async def create_bead_handoff(
        self,
        source_bead_id: str,
        source_agent_id: str,
        target_bead_id: Optional[str] = None,
        target_agent_id: Optional[str] = None,
        **kwargs,
    ) -> BeadHandoffContext:
        """Create and store a new bead handoff."""
        handoff = BeadHandoffContext.create(
            source_bead_id=source_bead_id,
            source_agent_id=source_agent_id,
            target_bead_id=target_bead_id,
            target_agent_id=target_agent_id,
            **kwargs,
        )
        await self.store.save_bead_handoff(handoff)
        logger.info(
            f"Created handoff {handoff.id}: {source_bead_id} -> {target_bead_id or target_agent_id}"
        )
        return handoff

    async def create_molecule_handoff(
        self,
        molecule_id: str,
        source_step: str,
        target_step: str,
        **kwargs,
    ) -> MoleculeHandoffContext:
        """Create and store a molecule handoff."""
        handoff = MoleculeHandoffContext.create(
            molecule_id=molecule_id,
            source_step=source_step,
            target_step=target_step,
            **kwargs,
        )
        await self.store.save_molecule_handoff(handoff)
        return handoff

    async def transfer_to_bead(
        self, handoff_id: str, target_bead_id: str
    ) -> Optional[BeadHandoffContext]:
        """Transfer a handoff to a specific bead."""
        handoff = await self.store.get_bead_handoff(handoff_id)
        if not handoff:
            return None

        handoff.target_bead_id = target_bead_id
        await self.store.save_bead_handoff(handoff)
        return handoff

    async def transfer_to_agent(
        self, handoff_id: str, target_agent_id: str
    ) -> Optional[BeadHandoffContext]:
        """Transfer a handoff to a specific agent."""
        handoff = await self.store.get_bead_handoff(handoff_id)
        if not handoff:
            return None

        handoff.target_agent_id = target_agent_id
        await self.store.save_bead_handoff(handoff)
        return handoff

    async def recover_for_agent(self, agent_id: str) -> List[BeadHandoffContext]:
        """
        Recover all pending handoffs for an agent.

        Called on agent startup to ensure GUPP compliance.
        """
        handoffs = await self.store.get_pending_for_agent(agent_id)
        logger.info(f"Recovered {len(handoffs)} pending handoffs for agent {agent_id}")
        return sorted(handoffs, key=lambda h: h.priority.value, reverse=True)

    async def recover_for_bead(self, bead_id: str) -> List[BeadHandoffContext]:
        """Recover all pending handoffs for a bead."""
        return await self.store.get_pending_for_bead(bead_id)

    async def acknowledge(self, handoff_id: str) -> None:
        """Acknowledge receipt of a handoff."""
        await self.store.mark_acknowledged(handoff_id)

    async def merge_contexts(self, handoffs: List[BeadHandoffContext]) -> BeadHandoffContext:
        """
        Merge multiple handoff contexts into one.

        Useful when a bead receives handoffs from multiple sources.
        """
        if not handoffs:
            raise ValueError("Cannot merge empty handoff list")

        if len(handoffs) == 1:
            return handoffs[0]

        # Sort by priority (highest first), then by creation time
        sorted_handoffs = sorted(
            handoffs,
            key=lambda h: (h.priority.value, h.created_at.timestamp()),
            reverse=True,
        )

        # Start with highest priority handoff as base
        base = sorted_handoffs[0]

        # Merge content from all handoffs
        merged_findings = list(base.findings)
        merged_artifacts = dict(base.artifacts)
        merged_decisions = list(base.decisions)
        merged_next_steps = list(base.next_steps)
        merged_questions = list(base.open_questions)
        merged_warnings = list(base.warnings)

        for handoff in sorted_handoffs[1:]:
            # Add unique findings
            for finding in handoff.findings:
                if finding not in merged_findings:
                    merged_findings.append(finding)

            # Merge artifacts (later ones don't override)
            for name, path in handoff.artifacts.items():
                if name not in merged_artifacts:
                    merged_artifacts[name] = path

            # Add unique decisions
            for decision in handoff.decisions:
                if decision not in merged_decisions:
                    merged_decisions.append(decision)

            # Add unique next steps
            for step in handoff.next_steps:
                if step not in merged_next_steps:
                    merged_next_steps.append(step)

            # Add unique questions
            for question in handoff.open_questions:
                if question not in merged_questions:
                    merged_questions.append(question)

            # Add unique warnings
            for warning in handoff.warnings:
                if warning not in merged_warnings:
                    merged_warnings.append(warning)

        # Create merged context
        merged = BeadHandoffContext.create(
            source_bead_id=f"merged-{len(handoffs)}-sources",
            source_agent_id="handoff-protocol",
            target_bead_id=base.target_bead_id,
            target_agent_id=base.target_agent_id,
            findings=merged_findings,
            artifacts=merged_artifacts,
            decisions=merged_decisions,
            next_steps=merged_next_steps,
            open_questions=merged_questions,
            warnings=merged_warnings,
            execution_summary=f"Merged from {len(handoffs)} handoffs",
            priority=base.priority,
        )

        # Track source handoffs in metadata
        merged.metadata["source_handoff_ids"] = [h.id for h in handoffs]

        return merged


# Convenience functions for common operations
async def create_handoff_store(path: Optional[Path] = None) -> HandoffStore:
    """Create and initialize a handoff store."""
    if path is None:
        path = Path(".handoffs")
    store = HandoffStore(path)
    await store.initialize()
    return store


async def create_handoff_protocol(path: Optional[Path] = None) -> HandoffProtocol:
    """Create a handoff protocol with initialized store."""
    store = await create_handoff_store(path)
    return HandoffProtocol(store)
