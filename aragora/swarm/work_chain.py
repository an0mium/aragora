"""Work chains: multi-step dependent work orders with DAG validation.

A WorkChain groups related BoundedWorkOrders into a dependency DAG. It
provides:
- Cycle detection (Kahn's algorithm)
- Topological ordering
- Execution wave planning (maximal independent sets)
- Chain-level status tracking

The chain is a planning abstraction; actual execution is handled by
ChainExecutor which feeds waves into the existing supervisor dispatch path.
"""

from __future__ import annotations

import hashlib
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

UTC = timezone.utc
_DEPENDENCY_ALIAS_KEYS = ("work_order_id", "pipeline_task_id", "task_key")


class ChainStatus(str, Enum):
    """Lifecycle status of a work chain."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    """Lifecycle status of a single chain step."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


_TERMINAL_STEP_STATUSES = frozenset({StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED})


@dataclass
class ChainStep:
    """One node in a work chain DAG.

    Each step wraps a work order ID and declares its predecessors via
    ``depends_on``.  The step itself does not carry the full work order
    — it references one by ID so the chain stays lightweight.
    """

    step_id: str
    work_order_id: str
    title: str
    depends_on: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    file_scope: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.step_id = str(self.step_id or "").strip() or str(uuid.uuid4())[:12]
        self.work_order_id = str(self.work_order_id or "").strip()
        self.depends_on = list(
            dict.fromkeys(dep for dep in self.depends_on if str(dep or "").strip())
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "work_order_id": self.work_order_id,
            "title": self.title,
            "depends_on": list(self.depends_on),
            "status": self.status.value,
            "file_scope": list(self.file_scope),
            "metadata": dict(self.metadata),
        }


class CycleDetectedError(ValueError):
    """Raised when a work chain contains a dependency cycle."""

    def __init__(self, involved_steps: list[str]) -> None:
        self.involved_steps = involved_steps
        super().__init__(f"Dependency cycle detected among steps: {', '.join(involved_steps)}")


@dataclass
class ExecutionWave:
    """A group of steps that can execute concurrently.

    All steps in a wave have their dependencies satisfied by earlier waves.
    """

    wave_index: int
    step_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"wave_index": self.wave_index, "step_ids": list(self.step_ids)}


@dataclass
class WorkChain:
    """A DAG of dependent work order steps.

    Construction validates the DAG is acyclic and computes topological order.
    """

    chain_id: str
    title: str
    steps: list[ChainStep] = field(default_factory=list)
    status: ChainStatus = ChainStatus.PENDING
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # Computed after validation
    _topo_order: list[str] = field(default_factory=list, repr=False)
    _waves: list[ExecutionWave] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.chain_id = str(self.chain_id or "").strip() or str(uuid.uuid4())[:12]
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()
        if self.steps:
            self.validate()

    def validate(self) -> None:
        """Validate DAG structure and compute topological order + waves.

        Raises CycleDetectedError if the dependency graph contains a cycle.
        Raises ValueError for dangling dependency references.
        """
        step_ids = {step.step_id for step in self.steps}

        # Check for dangling references
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    raise ValueError(f"Step '{step.step_id}' depends on unknown step '{dep}'")

        self._topo_order = _topological_sort(self.steps)
        self._waves = _compute_waves(self.steps, self._topo_order)

    @property
    def topological_order(self) -> list[str]:
        """Step IDs in valid execution order."""
        return list(self._topo_order)

    @property
    def waves(self) -> list[ExecutionWave]:
        """Execution waves: groups of steps that can run concurrently."""
        return list(self._waves)

    @property
    def num_waves(self) -> int:
        return len(self._waves)

    def step_by_id(self, step_id: str) -> ChainStep | None:
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def ready_steps(self) -> list[ChainStep]:
        """Return steps whose dependencies are all completed."""
        completed = {step.step_id for step in self.steps if step.status == StepStatus.COMPLETED}
        return [
            step
            for step in self.steps
            if step.status == StepStatus.PENDING
            and all(dep in completed for dep in step.depends_on)
        ]

    def mark_step(self, step_id: str, status: StepStatus) -> None:
        step = self.step_by_id(step_id)
        if step is None:
            raise KeyError(f"Unknown step: {step_id}")
        step.status = status
        self._update_chain_status()

    def _update_chain_status(self) -> None:
        statuses = {step.status for step in self.steps}
        if statuses and all(status in _TERMINAL_STEP_STATUSES for status in statuses):
            self.status = (
                ChainStatus.FAILED if StepStatus.FAILED in statuses else ChainStatus.COMPLETED
            )
            return
        if statuses and any(
            status
            in {
                StepStatus.RUNNING,
                StepStatus.READY,
                StepStatus.COMPLETED,
                StepStatus.FAILED,
                StepStatus.SKIPPED,
            }
            for status in statuses
        ):
            self.status = ChainStatus.RUNNING
            return
        self.status = ChainStatus.PENDING

    def fingerprint(self) -> str:
        """Stable hash over chain structure for dedup."""
        parts = sorted(
            f"{step.step_id}:{step.work_order_id}:{','.join(step.depends_on)}"
            for step in self.steps
        )
        raw = f"{self.title}|{'|'.join(parts)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "title": self.title,
            "steps": [step.to_dict() for step in self.steps],
            "status": self.status.value,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
            "topological_order": list(self._topo_order),
            "waves": [wave.to_dict() for wave in self._waves],
            "fingerprint": self.fingerprint(),
        }


def _topological_sort(steps: list[ChainStep]) -> list[str]:
    """Kahn's algorithm for topological sorting with cycle detection."""
    in_degree: dict[str, int] = {step.step_id: 0 for step in steps}
    adjacency: dict[str, list[str]] = defaultdict(list)

    for step in steps:
        for dep in step.depends_on:
            adjacency[dep].append(step.step_id)
            in_degree[step.step_id] += 1

    queue: deque[str] = deque(
        step_id for step_id, degree in sorted(in_degree.items()) if degree == 0
    )
    order: list[str] = []

    while queue:
        current = queue.popleft()
        order.append(current)
        for neighbor in sorted(adjacency[current]):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(steps):
        remaining = sorted(step_id for step_id, degree in in_degree.items() if degree > 0)
        raise CycleDetectedError(remaining)

    return order


def _compute_waves(steps: list[ChainStep], topo_order: list[str]) -> list[ExecutionWave]:
    """Assign each step to the earliest possible wave."""
    step_map = {step.step_id: step for step in steps}
    wave_assignment: dict[str, int] = {}

    for step_id in topo_order:
        step = step_map[step_id]
        if not step.depends_on:
            wave_assignment[step_id] = 0
        else:
            wave_assignment[step_id] = max(wave_assignment[dep] for dep in step.depends_on) + 1

    if not wave_assignment:
        return []

    max_wave = max(wave_assignment.values())
    waves: list[ExecutionWave] = []
    for idx in range(max_wave + 1):
        members = sorted(step_id for step_id, wave in wave_assignment.items() if wave == idx)
        if members:
            waves.append(ExecutionWave(wave_index=idx, step_ids=members))
    return waves


def build_chain_from_work_orders(
    title: str,
    work_orders: list[dict[str, Any]],
    *,
    chain_id: str = "",
) -> WorkChain:
    """Construct a WorkChain from a list of work order dicts.

    Each work order dict must have ``work_order_id`` and optionally
    ``dependency_ids``, ``title``, ``file_scope``.
    """
    steps: list[ChainStep] = []
    wo_to_step: dict[str, str] = {}
    alias_to_step: dict[str, str] = {}

    for wo in work_orders:
        wo_id = str(wo.get("work_order_id", "")).strip()
        if not wo_id:
            continue
        step_id = wo_id
        wo_to_step[wo_id] = step_id
        for key in _DEPENDENCY_ALIAS_KEYS:
            alias = str(wo.get(key, "")).strip()
            if not alias:
                continue
            existing = alias_to_step.get(alias)
            if existing is not None and existing != step_id:
                raise ValueError(f"Dependency alias '{alias}' maps to multiple work orders")
            alias_to_step[alias] = step_id

    for wo in work_orders:
        wo_id = str(wo.get("work_order_id", "")).strip()
        if wo_id not in wo_to_step:
            continue
        raw_deps = wo.get("dependency_ids", [])
        if not isinstance(raw_deps, list):
            raw_deps = []
        depends_on: list[str] = []
        for raw_dep in raw_deps:
            dependency_id = str(raw_dep or "").strip()
            if not dependency_id:
                continue
            resolved = alias_to_step.get(dependency_id)
            if resolved is None:
                raise ValueError(
                    f"Unknown dependency id '{dependency_id}' for work order '{wo_id}'"
                )
            depends_on.append(resolved)
        steps.append(
            ChainStep(
                step_id=wo_to_step[wo_id],
                work_order_id=wo_id,
                title=str(wo.get("title", wo_id)),
                depends_on=depends_on,
                file_scope=list(wo.get("file_scope", [])),
            )
        )

    return WorkChain(chain_id=chain_id or "", title=title, steps=steps)


__all__ = [
    "ChainStatus",
    "ChainStep",
    "CycleDetectedError",
    "ExecutionWave",
    "StepStatus",
    "WorkChain",
    "build_chain_from_work_orders",
]
