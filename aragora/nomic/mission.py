"""Native mission contracts — the front door for "set a goal, walk away".

Pure data + a pure adapter, intentionally UNWIRED. This is PR #1 of the native
mission orchestrator (docs/plans/2026-06-16-native-mission-orchestrator.md):
the typed ``MissionSpec`` front-door plus the ``WorkItem`` queue shape and a
deterministic adapter from nomic's decomposition output. No execution, no I/O,
no model calls here — the long-running engine (boss_loop tick loop), the
relay-with-timeout, and the model transport land in later PRs and gate on the
``enable_native_mission`` flag (default OFF). Nothing imports this yet, so it
cannot change any behavior.

Design intent (recorded so later PRs hold the line):
- Runs on SUBSCRIPTIONS by default (``MissionTransport.CLI`` = 1 Claude Max +
  1 Codex Max, MCP-disabled). ``MissionTransport.API`` is an optional scale
  lever, not a prerequisite.
- ``auto_settle_max_tier`` bounds what the mission may settle autonomously on a
  passing model quorum; higher-tier items PARK for the operator. The
  merge-quorum gate remains the sole settlement authority — the mission never
  bypasses it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any


class MissionTransport(Enum):
    """How the mission drives models. CLI (subscriptions) is the default."""

    CLI = "cli"  # subscription CLIs: claude -p (MCP-disabled) + codex exec, 1 sub/provider
    API = "api"  # optional scale lever: api_agents via Secrets Manager


class WorkItemStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    PARKED = "parked"  # blocked/needs-human; mission continues OTHER items
    FAILED = "failed"


_RELAY_CHANNELS = frozenset({"none", "slack", "email"})


@dataclass(frozen=True)
class MissionSpec:
    """A high-level goal plus the bounds under which the mission may pursue it."""

    goal: str
    mission_id: str
    acceptance_criteria: tuple[str, ...] = ()
    budget_usd: float | None = None
    max_hours: float | None = None
    transport: MissionTransport = MissionTransport.CLI
    relay: str = "none"
    auto_settle_max_tier: int = 2

    def __post_init__(self) -> None:
        if not self.goal.strip():
            raise ValueError("goal must be non-empty")
        if not self.mission_id.strip():
            raise ValueError("mission_id must be non-empty")
        if self.budget_usd is not None and self.budget_usd < 0:
            raise ValueError("budget_usd must be non-negative or None")
        if self.max_hours is not None and self.max_hours <= 0:
            raise ValueError("max_hours must be positive or None")
        if self.relay not in _RELAY_CHANNELS:
            raise ValueError(f"relay must be one of {sorted(_RELAY_CHANNELS)}")
        if not 0 <= self.auto_settle_max_tier <= 4:
            raise ValueError("auto_settle_max_tier must be in [0, 4]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "mission_id": self.mission_id,
            "acceptance_criteria": list(self.acceptance_criteria),
            "budget_usd": self.budget_usd,
            "max_hours": self.max_hours,
            "transport": self.transport.value,
            "relay": self.relay,
            "auto_settle_max_tier": self.auto_settle_max_tier,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MissionSpec":
        raw_criteria = payload.get("acceptance_criteria", [])
        criteria = tuple(str(c) for c in raw_criteria) if isinstance(raw_criteria, list) else ()
        transport_raw = payload.get("transport", MissionTransport.CLI.value)
        return cls(
            goal=str(payload["goal"]),
            mission_id=str(payload["mission_id"]),
            acceptance_criteria=criteria,
            budget_usd=payload.get("budget_usd"),
            max_hours=payload.get("max_hours"),
            transport=MissionTransport(transport_raw),
            relay=str(payload.get("relay", "none")),
            auto_settle_max_tier=int(payload.get("auto_settle_max_tier", 2)),
        )


@dataclass(frozen=True)
class WorkItem:
    """One unit of mission work. The internal queue replaces the GitHub-issue gate."""

    item_id: str
    description: str
    status: WorkItemStatus = WorkItemStatus.PENDING
    complexity: str = "low"  # low|medium|high, from decomposition (NOT the merge tier)
    file_scope: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "description": self.description,
            "status": self.status.value,
            "complexity": self.complexity,
            "file_scope": list(self.file_scope),
            "dependencies": list(self.dependencies),
        }


def work_items_from_subtasks(subtasks: Iterable[Any]) -> tuple[WorkItem, ...]:
    """Adapt nomic decomposition output into a PENDING work-item queue.

    Accepts ``SubTask`` (``aragora.nomic.task_decomposer``) or any structurally
    compatible object exposing ``id``, ``title``/``description``,
    ``estimated_complexity``, ``file_scope``, ``dependencies``. Pure and
    deterministic — order preserved, no I/O. Items with no usable id are skipped
    rather than raising, so a malformed decomposition can't crash the front door.
    """
    items: list[WorkItem] = []
    for st in subtasks:
        item_id = str(getattr(st, "id", "") or "")
        if not item_id:
            continue
        description = str(getattr(st, "description", "") or getattr(st, "title", "") or "")
        complexity = str(getattr(st, "estimated_complexity", "low") or "low")
        file_scope = tuple(str(f) for f in (getattr(st, "file_scope", ()) or ()))
        dependencies = tuple(str(d) for d in (getattr(st, "dependencies", ()) or ()))
        items.append(
            WorkItem(
                item_id=item_id,
                description=description,
                complexity=complexity,
                file_scope=file_scope,
                dependencies=dependencies,
            )
        )
    return tuple(items)
