"""
Data models for Decision Playbooks.

A Playbook is a complete, repeatable decision workflow that combines:
- A deliberation template (defines debate structure)
- A vertical weight profile (domain-specific scoring)
- Required compliance artifacts (regulatory output)
- Agent selection criteria (who participates)
- Required output sections and approval gates
- Estimated duration for planning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ApprovalGate:
    """A checkpoint requiring human approval before proceeding."""

    name: str
    description: str
    required_role: str = "decision_maker"
    auto_approve_if_consensus: bool = False
    timeout_hours: float = 24.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "required_role": self.required_role,
            "auto_approve_if_consensus": self.auto_approve_if_consensus,
            "timeout_hours": self.timeout_hours,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApprovalGate:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            required_role=data.get("required_role", "decision_maker"),
            auto_approve_if_consensus=data.get("auto_approve_if_consensus", False),
            timeout_hours=data.get("timeout_hours", 24.0),
        )


@dataclass
class PlaybookStep:
    """A single step in a playbook execution."""

    name: str
    action: str  # debate, review, approve, notify, generate_artifact
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "action": self.action, "config": self.config}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlaybookStep:
        return cls(
            name=data["name"],
            action=data["action"],
            config=data.get("config", {}),
        )


@dataclass
class Playbook:
    """
    A complete, repeatable decision workflow.

    Composes template, vertical, compliance, agents, output, and approval
    into a single runnable package. Can be defined in YAML or created
    programmatically.
    """

    id: str
    name: str
    description: str
    category: str  # healthcare, finance, legal, engineering, compliance, general

    # Deliberation template to use (maps to aragora/deliberation/templates/)
    template_name: str = "general"
    # Vertical weight profile for domain scoring (maps to evaluation vertical profiles)
    vertical_profile: str | None = None
    # Required compliance artifacts to generate
    compliance_artifacts: list[str] = field(default_factory=list)

    # Agent selection criteria (structured hints for team selection)
    agent_criteria: dict[str, Any] = field(
        default_factory=lambda: {
            "min_agents": 3,
            "required_roles": [],
        }
    )

    # Required output section names
    required_sections: list[str] = field(default_factory=list)

    # Estimated duration in minutes for planning
    estimated_duration_minutes: int = 30

    # Legacy agent selection fields (kept for backwards compatibility)
    min_agents: int = 3
    max_agents: int = 7
    required_agent_types: list[str] = field(default_factory=list)
    agent_selection_strategy: str = "best_for_domain"

    # Debate configuration
    max_rounds: int = 5
    consensus_threshold: float = 0.7
    timeout_seconds: float = 300.0

    # Output configuration
    output_format: str = "decision_receipt"
    output_channels: list[str] = field(default_factory=list)

    # Approval gates
    approval_gates: list[ApprovalGate] = field(default_factory=list)

    # Steps in the playbook
    steps: list[PlaybookStep] = field(default_factory=list)

    # Tags for filtering
    tags: list[str] = field(default_factory=list)

    # Version
    version: str = "1.0.0"

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "template_name": self.template_name,
            "vertical_profile": self.vertical_profile,
            "compliance_artifacts": self.compliance_artifacts,
            "agent_criteria": self.agent_criteria,
            "required_sections": self.required_sections,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "min_agents": self.min_agents,
            "max_agents": self.max_agents,
            "required_agent_types": self.required_agent_types,
            "agent_selection_strategy": self.agent_selection_strategy,
            "max_rounds": self.max_rounds,
            "consensus_threshold": self.consensus_threshold,
            "timeout_seconds": self.timeout_seconds,
            "output_format": self.output_format,
            "output_channels": self.output_channels,
            "approval_gates": [g.to_dict() for g in self.approval_gates],
            "steps": [s.to_dict() for s in self.steps],
            "tags": self.tags,
            "version": self.version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Playbook:
        """Create from dictionary."""
        gates = [ApprovalGate.from_dict(g) for g in data.get("approval_gates", [])]
        steps = [PlaybookStep.from_dict(s) for s in data.get("steps", [])]

        # Build agent_criteria from explicit field or derive from legacy fields
        agent_criteria = data.get("agent_criteria")
        if agent_criteria is None:
            agent_criteria = {
                "min_agents": data.get("min_agents", 3),
                "required_roles": data.get("required_agent_types", []),
            }

        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            category=data.get("category", "general"),
            template_name=data.get("template_name", "general"),
            vertical_profile=data.get("vertical_profile"),
            compliance_artifacts=data.get("compliance_artifacts", []),
            agent_criteria=agent_criteria,
            required_sections=data.get("required_sections", []),
            estimated_duration_minutes=data.get("estimated_duration_minutes", 30),
            min_agents=data.get("min_agents", agent_criteria.get("min_agents", 3)),
            max_agents=data.get("max_agents", 7),
            required_agent_types=data.get(
                "required_agent_types", agent_criteria.get("required_roles", [])
            ),
            agent_selection_strategy=data.get("agent_selection_strategy", "best_for_domain"),
            max_rounds=data.get("max_rounds", 5),
            consensus_threshold=data.get("consensus_threshold", 0.7),
            timeout_seconds=data.get("timeout_seconds", 300.0),
            output_format=data.get("output_format", "decision_receipt"),
            output_channels=data.get("output_channels", []),
            approval_gates=gates,
            steps=steps,
            tags=data.get("tags", []),
            version=data.get("version", "1.0.0"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PlaybookRunConfig:
    """Configuration for running a playbook.

    Wraps the playbook ID with runtime parameters like topic, context,
    participant overrides, and template overrides.
    """

    playbook_id: str
    topic: str
    context: dict[str, Any] = field(default_factory=dict)
    participants: list[str] = field(default_factory=list)
    override_agents: list[str] | None = None
    override_template: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "playbook_id": self.playbook_id,
            "topic": self.topic,
            "context": self.context,
            "participants": self.participants,
            "override_agents": self.override_agents,
            "override_template": self.override_template,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlaybookRunConfig:
        return cls(
            playbook_id=data["playbook_id"],
            topic=data.get("topic", ""),
            context=data.get("context", {}),
            participants=data.get("participants", []),
            override_agents=data.get("override_agents"),
            override_template=data.get("override_template"),
        )


@dataclass
class PlaybookResult:
    """Result of a playbook execution.

    Wraps the debate result with compliance artifacts generated during
    execution and the playbook metadata for traceability.
    """

    run_id: str
    playbook_id: str
    playbook_name: str
    status: str  # queued, running, completed, failed
    debate_result: dict[str, Any] | None = None
    compliance_artifacts: list[dict[str, Any]] = field(default_factory=list)
    playbook_metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "playbook_id": self.playbook_id,
            "playbook_name": self.playbook_name,
            "status": self.status,
            "debate_result": self.debate_result,
            "compliance_artifacts": self.compliance_artifacts,
            "playbook_metadata": self.playbook_metadata,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlaybookResult:
        return cls(
            run_id=data["run_id"],
            playbook_id=data["playbook_id"],
            playbook_name=data.get("playbook_name", ""),
            status=data.get("status", "queued"),
            debate_result=data.get("debate_result"),
            compliance_artifacts=data.get("compliance_artifacts", []),
            playbook_metadata=data.get("playbook_metadata", {}),
            created_at=data.get("created_at", ""),
            completed_at=data.get("completed_at"),
        )


__all__ = [
    "Playbook",
    "PlaybookStep",
    "ApprovalGate",
    "PlaybookRunConfig",
    "PlaybookResult",
]
