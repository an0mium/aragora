"""
Rig - Per-repo project container with isolated agent context.

A Rig is the Gastown equivalent of a Kubernetes node: it represents a
repository/project with its own agent pool, configuration, and work queue.

Each rig manages:
- A set of assigned agents (crew)
- Git worktree hooks for those agents
- Configuration specific to the project
- Status and health reporting
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RigStatus(Enum):
    """Rig lifecycle status."""

    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    DRAINING = "draining"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class RigConfig:
    """Configuration for a Rig."""

    repo_url: str = ""
    repo_path: str = ""
    branch: str = "main"
    max_agents: int = 10
    max_concurrent_tasks: int = 5
    auto_assign_agents: bool = True
    allowed_agent_types: list[str] = field(default_factory=list)
    environment_vars: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class Rig:
    """
    A per-repo project container.

    Manages agents, work items, and configuration for a single
    repository or project.
    """

    rig_id: str
    name: str
    workspace_id: str
    config: RigConfig = field(default_factory=RigConfig)
    status: RigStatus = RigStatus.INITIALIZING
    assigned_agents: list[str] = field(default_factory=list)
    active_convoys: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    tasks_completed: int = 0
    tasks_failed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        from dataclasses import asdict

        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Rig:
        """Deserialize from dictionary."""
        data = dict(data)
        if isinstance(data.get("status"), str):
            data["status"] = RigStatus(data["status"])
        if isinstance(data.get("config"), dict):
            data["config"] = RigConfig(**data["config"])
        return cls(**data)
