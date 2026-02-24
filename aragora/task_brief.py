"""
Defines the TaskBrief, a structured representation of a user's request.

This module provides the core data structures and validation logic for
transforming ambiguous user inputs into well-defined, executable tasks
for the Aragora multi-agent system.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Optional

from aragora.core_types import AgentRole

class TaskSource(str, Enum):
    """Specifies the origin of a TaskBrief's definition."""
    USER_PROVIDED = "user_provided"
    INFERRED_FROM_CONTEXT = "inferred_from_context"
    CANONICAL_TEMPLATE = "canonical_template"
    USER_CLARIFICATION = "user_clarification"

class TaskStatus(str, Enum):
    """Represents the lifecycle stage of a task brief."""
    PROVISIONAL = "provisional"  # Inferred, requires user validation
    ACTIVE = "active"          # Validated and ready for execution
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TaskBrief:
    """
    A structured, validated, and executable definition of a task.

    This class serves as the central artifact for the task ingestion and
    resolution pipeline. It transforms a potentially vague user prompt
    into a machine-readable format that can be reliably executed by the
    orchestrator.

    Attributes:
        id: A unique identifier for the task.
        objective: A clear, concise statement of the primary goal (e.g., "Design a REST API").
        scope: A description of what is included in the task.
        constraints: A list of limitations or rules that must be followed.
        success_criteria: Measurable outcomes that define task completion.
        deliverables: A list of expected outputs (e.g., "OpenAPI spec", "Sequence diagram").
        status: The current lifecycle status of the task.
        provenance: Metadata about how the task brief was created.
            - source: How the brief was generated (e.g., inferred, user-provided).
            - confidence_score: The system's confidence in its interpretation of an ambiguous prompt.
            - original_prompt: The raw user input that this brief is based on.
            - generated_by: The agent/role that generated this brief (e.g., 'planner').
        schema_version: The version of the TaskBrief schema, for migrations.
    """
    id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    objective: str
    scope: str = "No scope provided."
    constraints: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    deliverables: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PROVISIONAL

    provenance: dict[str, Any] = field(default_factory=dict)
    schema_version: ClassVar[str] = "1.0"

    def __post_init__(self):
        """Ensure provenance dictionary has default keys."""
        self.provenance.setdefault("source", TaskSource.USER_PROVIDED)
        self.provenance.setdefault("confidence_score", 1.0)
        self.provenance.setdefault("original_prompt", self.objective)
        self.provenance.setdefault("generated_by", "user")
        self.provenance.setdefault("timestamp", datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serializes the TaskBrief to a dictionary."""
        return {
            "id": self.id,
            "objective": self.objective,
            "scope": self.scope,
            "constraints": self.constraints,
            "success_criteria": self.success_criteria,
            "deliverables": self.deliverables,
            "status": self.status.value,
            "provenance": self.provenance,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskBrief:
        """Deserializes a TaskBrief from a dictionary."""
        if data.get("schema_version") != cls.schema_version:
            # In a real system, a migration would happen here.
            pass

        return cls(
            id=data.get("id", f"task_{uuid.uuid4().hex[:12]}"),
            objective=data["objective"],
            scope=data.get("scope", "No scope provided."),
            constraints=data.get("constraints", []),
            success_criteria=data.get("success_criteria", []),
            deliverables=data.get("deliverables", []),
            status=TaskStatus(data.get("status", "provisional")),
            provenance=data.get("provenance", {}),
        )


class TaskBriefValidator:
    """
    Validates and parses inputs into a formal TaskBrief.
    """
    REQUIRED_FIELDS: ClassVar[list[str]] = ["objective"]

    @staticmethod
    def is_brief_required(prompt: str, context: Optional[dict[str, Any]] = None) -> bool:
        """
        Determines if a prompt is too ambiguous and requires a full TaskBrief.
        This replaces the brittle heuristic of checking string length or verb presence.
        A more sophisticated implementation would use a lightweight classifier model.
        For now, we consider any short prompt as needing resolution.
        """
        # A simple but more robust heuristic than before.
        # In a real system, this could be a call to a small, fast classification model.
        if len(prompt.split()) < 5:
            return True
        return False


    @staticmethod
    def create_provisional_brief(prompt: str, source: TaskSource, confidence: float, generated_by: AgentRole) -> TaskBrief:
        """
        Generates a provisional TaskBrief from a simple prompt string.
        """
        return TaskBrief(
            objective=prompt,
            status=TaskStatus.PROVISIONAL,
            provenance={
                "source": source,
                "confidence_score": confidence,
                "original_prompt": prompt,
                "generated_by": generated_by,
                "timestamp": datetime.now().isoformat(),
            }
        )

