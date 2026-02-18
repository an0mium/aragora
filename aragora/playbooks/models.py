"""
Data models for Decision Playbooks.

A Playbook is a complete, repeatable decision workflow that combines:
- A deliberation template (defines debate structure)
- A vertical weight profile (domain-specific scoring)
- Required compliance artifacts (regulatory output)
- Agent selection criteria (who participates)
- Output format and delivery configuration
- Approval gates (human checkpoints)
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
    category: str  # healthcare, finance, legal, engineering, general

    # Deliberation template to use
    template_name: str = "general"
    # Vertical weight profile for domain scoring
    vertical_profile: str | None = None
    # Required compliance artifacts to generate
    compliance_artifacts: list[str] = field(default_factory=list)

    # Agent selection criteria
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
        gates = [
            ApprovalGate.from_dict(g) for g in data.get("approval_gates", [])
        ]
        steps = [
            PlaybookStep.from_dict(s) for s in data.get("steps", [])
        ]
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            category=data.get("category", "general"),
            template_name=data.get("template_name", "general"),
            vertical_profile=data.get("vertical_profile"),
            compliance_artifacts=data.get("compliance_artifacts", []),
            min_agents=data.get("min_agents", 3),
            max_agents=data.get("max_agents", 7),
            required_agent_types=data.get("required_agent_types", []),
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


__all__ = ["Playbook", "PlaybookStep", "ApprovalGate"]
