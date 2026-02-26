"""SwarmSpec: structured specification from interrogation to orchestration."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class SwarmSpec:
    """Structured specification produced by interrogation, consumed by orchestration.

    This is the contract between the user-facing interrogation phase and
    the technical orchestration phase. It captures user intent in a format
    that maps directly to ``HardenedOrchestrator.execute_goal_coordinated()``.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # User intent
    raw_goal: str = ""
    refined_goal: str = ""

    # Acceptance criteria
    acceptance_criteria: list[str] = field(default_factory=list)

    # Constraints
    constraints: list[str] = field(default_factory=list)
    budget_limit_usd: float | None = 5.0

    # Hints for decomposition
    track_hints: list[str] = field(default_factory=list)
    file_scope_hints: list[str] = field(default_factory=list)

    # Risk assessment
    estimated_complexity: str = "medium"
    requires_approval: bool = False

    # Proactive suggestions from the CTO-Claude during interrogation
    proactive_suggestions: list[str] = field(default_factory=list)

    # Metadata
    interrogation_turns: int = 0
    user_expertise: str = "non-developer"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SwarmSpec:
        """Deserialize from dictionary."""
        data = dict(data)
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> SwarmSpec:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(text))

    def to_yaml(self) -> str:
        """Serialize to YAML string."""
        try:
            import yaml

            data = self.to_dict()
            return yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
        except ImportError:
            return self.to_json()

    @classmethod
    def from_yaml(cls, text: str) -> SwarmSpec:
        """Deserialize from YAML string."""
        try:
            import yaml

            data = yaml.safe_load(text)
        except ImportError:
            data = json.loads(text)
        return cls.from_dict(data)

    def summary(self) -> str:
        """Human-readable summary of the spec."""
        lines = [
            f"Goal: {self.refined_goal or self.raw_goal}",
            f"Complexity: {self.estimated_complexity}",
        ]
        if self.acceptance_criteria:
            lines.append(f"Acceptance criteria: {len(self.acceptance_criteria)} items")
        if self.constraints:
            lines.append(f"Constraints: {len(self.constraints)} items")
        if self.budget_limit_usd is not None:
            lines.append(f"Budget: ${self.budget_limit_usd:.2f}")
        if self.track_hints:
            lines.append(f"Tracks: {', '.join(self.track_hints)}")
        if self.file_scope_hints:
            lines.append(f"File scope: {', '.join(self.file_scope_hints[:5])}")
        return "\n".join(lines)
