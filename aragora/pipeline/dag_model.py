"""View models for pipeline DAG visualization and live state."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

from aragora.canvas.stages import PipelineStage, StageEdgeType

_STAGE_SORT_ORDER: dict[PipelineStage, int] = {
    PipelineStage.IDEAS: 0,
    PipelineStage.PRINCIPLES: 1,
    PipelineStage.GOALS: 2,
    PipelineStage.ACTIONS: 3,
    PipelineStage.ORCHESTRATION: 4,
}

_STAGE_LABELS: dict[PipelineStage, str] = {
    PipelineStage.IDEAS: "Ideas",
    PipelineStage.PRINCIPLES: "Principles",
    PipelineStage.GOALS: "Goals",
    PipelineStage.ACTIONS: "Actions",
    PipelineStage.ORCHESTRATION: "Orchestration",
}

_BLOCKING_EDGE_TYPES = frozenset(
    {
        StageEdgeType.BLOCKS,
        StageEdgeType.CONSTRAINS,
        StageEdgeType.DECOMPOSES_INTO,
        StageEdgeType.DERIVED_FROM,
        StageEdgeType.EXECUTES,
        StageEdgeType.FOLLOWS,
        StageEdgeType.IMPLEMENTS,
        StageEdgeType.REQUIRES,
    }
)
_SUCCEEDED_STATUSES = frozenset({"approved", "complete", "completed", "succeeded"})
_FAILED_STATUSES = frozenset({"error", "failed", "federation_error", "rejected", "timeout"})
_ACTIVE_STATUSES = frozenset({"in_progress", "running"})
_WAITING_STATUSES = frozenset({"awaiting_human"})


def _coerce_stage(value: PipelineStage | str | None) -> PipelineStage | None:
    if value is None or isinstance(value, PipelineStage):
        return value
    try:
        return PipelineStage(value)
    except ValueError:
        return None


def _coerce_edge_type(value: StageEdgeType | str | None) -> StageEdgeType:
    if isinstance(value, StageEdgeType):
        return value
    if value is not None:
        try:
            return StageEdgeType(str(value))
        except ValueError:
            logger.debug("Unknown StageEdgeType %r, defaulting to RELATES_TO", value)
    return StageEdgeType.RELATES_TO


def _normalize_runtime_status(
    execution_status: str | None,
    node_status: str | None = None,
) -> str:
    raw = (execution_status or node_status or "pending").strip().lower()
    if raw in _SUCCEEDED_STATUSES:
        return "succeeded"
    if raw in _FAILED_STATUSES:
        return "failed"
    if raw in _WAITING_STATUSES:
        return "awaiting_human"
    if raw in {"active", "submitted"}:
        return "pending"
    if raw in _ACTIVE_STATUSES:
        return "in_progress"
    return "pending"


def _aggregate_stage_status(statuses: list[str]) -> str:
    if not statuses:
        return "pending"

    status_set = set(statuses)
    if status_set == {"succeeded"}:
        return "complete"
    if "failed" in status_set:
        return "failed"
    if "awaiting_human" in status_set:
        return "awaiting_human"
    if "in_progress" in status_set:
        return "in_progress"
    if "succeeded" in status_set:
        return "partial"
    return "pending"


def _stage_sort_key(stage: PipelineStage) -> tuple[int, str]:
    return (_STAGE_SORT_ORDER.get(stage, 999), stage.value)


def _stage_dependency_status(source_status: str, *, blocking: bool) -> str:
    if not blocking:
        return "informational"
    if source_status == "complete":
        return "satisfied"
    if source_status == "failed":
        return "blocked"
    if source_status == "awaiting_human":
        return "awaiting_human"
    if source_status == "in_progress":
        return "in_progress"
    if source_status == "partial":
        return "partial"
    return "pending"


@dataclass
class PipelineNodeRuntime:
    """Live runtime state for a pipeline node."""

    node_id: str
    stage: PipelineStage
    execution_status: str = "pending"
    approval_status: str = "pending"
    confidence: float = 0.0
    assigned_agent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)

    @classmethod
    def from_node(cls, node: Any) -> PipelineNodeRuntime:
        data = dict(getattr(node, "data", {}) or {})
        metadata = dict(getattr(node, "metadata", {}) or {})
        assigned_agent = (
            data.get("assigned_agent")
            or metadata.get("assigned_agent")
            or data.get("agent")
            or metadata.get("agent")
        )
        execution_status = _normalize_runtime_status(
            getattr(node, "execution_status", None) or metadata.get("execution_status"),
            getattr(node, "status", None),
        )
        return cls(
            node_id=str(getattr(node, "id")),
            stage=_coerce_stage(getattr(node, "stage", None)) or PipelineStage.IDEAS,
            execution_status=execution_status,
            approval_status=str(getattr(node, "approval_status", "pending")),
            confidence=float(getattr(node, "confidence", 0.0) or 0.0),
            assigned_agent=str(assigned_agent) if assigned_agent is not None else None,
            metadata=metadata,
            updated_at=float(getattr(node, "updated_at", time.time()) or time.time()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "stage": self.stage.value,
            "execution_status": self.execution_status,
            "approval_status": self.approval_status,
            "confidence": self.confidence,
            "assigned_agent": self.assigned_agent,
            "metadata": dict(self.metadata),
            "updated_at": self.updated_at,
        }


@dataclass
class PipelineDAGDependency:
    """Normalized dependency record for DAG visualization."""

    id: str
    source_id: str
    target_id: str
    edge_type: StageEdgeType
    source_stage: PipelineStage
    target_stage: PipelineStage
    label: str = ""
    cross_stage: bool = False
    blocking: bool = False
    implicit: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_edge(
        cls,
        edge: Any,
        *,
        source_stage: PipelineStage,
        target_stage: PipelineStage,
    ) -> PipelineDAGDependency:
        edge_type = _coerce_edge_type(getattr(edge, "edge_type", None))
        cross_stage = source_stage != target_stage
        return cls(
            id=str(getattr(edge, "id")),
            source_id=str(getattr(edge, "source_id")),
            target_id=str(getattr(edge, "target_id")),
            edge_type=edge_type,
            source_stage=source_stage,
            target_stage=target_stage,
            label=str(getattr(edge, "label", "") or edge_type.value),
            cross_stage=cross_stage,
            blocking=edge_type in _BLOCKING_EDGE_TYPES,
            implicit=bool(getattr(edge, "implicit", False)),
            metadata=dict(getattr(edge, "data", {}) or {}),
        )

    @classmethod
    def from_parent_link(cls, parent: Any, child: Any) -> PipelineDAGDependency:
        parent_stage = _coerce_stage(getattr(parent, "stage", None)) or PipelineStage.IDEAS
        child_stage = _coerce_stage(getattr(child, "stage", None)) or parent_stage
        edge_type = (
            StageEdgeType.DERIVED_FROM if parent_stage != child_stage else StageEdgeType.REQUIRES
        )
        return cls(
            id=f"implicit:{getattr(parent, 'id')}->{getattr(child, 'id')}",
            source_id=str(getattr(parent, "id")),
            target_id=str(getattr(child, "id")),
            edge_type=edge_type,
            source_stage=parent_stage,
            target_stage=child_stage,
            label=edge_type.value,
            cross_stage=parent_stage != child_stage,
            blocking=True,
            implicit=True,
            metadata={"source": "parent_ids"},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "source_stage": self.source_stage.value,
            "target_stage": self.target_stage.value,
            "label": self.label,
            "cross_stage": self.cross_stage,
            "blocking": self.blocking,
            "implicit": self.implicit,
            "metadata": dict(self.metadata),
        }

    def to_react_flow_edge(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source_id,
            "target": self.target_id,
            "label": self.label or self.edge_type.value,
            "type": "smoothstep" if self.cross_stage else "default",
            "animated": self.cross_stage,
            "data": {
                "edgeType": self.edge_type.value,
                "sourceStage": self.source_stage.value,
                "targetStage": self.target_stage.value,
                "blocking": self.blocking,
                "implicit": self.implicit,
                **self.metadata,
            },
        }


@dataclass
class PipelineStageDependency:
    """Aggregated dependency edge between two pipeline stages."""

    id: str
    source_stage: PipelineStage
    target_stage: PipelineStage
    source_status: str
    target_status: str
    edge_count: int = 0
    blocking_edge_count: int = 0
    status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def blocking(self) -> bool:
        return self.blocking_edge_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_stage": self.source_stage.value,
            "target_stage": self.target_stage.value,
            "source_status": self.source_status,
            "target_status": self.target_status,
            "edge_count": self.edge_count,
            "blocking_edge_count": self.blocking_edge_count,
            "blocking": self.blocking,
            "status": self.status,
            "metadata": dict(self.metadata),
        }


@dataclass
class PipelineDAGStage:
    """Visualization summary for a single pipeline stage."""

    stage: PipelineStage
    label: str
    order: int
    status: str
    node_ids: list[str] = field(default_factory=list)
    dependency_stage_ids: list[str] = field(default_factory=list)
    node_count: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    ready: bool = False
    blocked_by_stage_ids: list[str] = field(default_factory=list)
    satisfied_dependency_stage_ids: list[str] = field(default_factory=list)
    dependency_count: int = 0
    blocking_dependency_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_runtime(
        cls,
        stage: PipelineStage,
        *,
        node_ids: list[str],
        runtimes: list[PipelineNodeRuntime],
        dependency_stage_ids: list[str],
        blocked_by_stage_ids: list[str] | None = None,
        satisfied_dependency_stage_ids: list[str] | None = None,
        dependency_count: int = 0,
        blocking_dependency_count: int = 0,
    ) -> PipelineDAGStage:
        statuses = [runtime.execution_status for runtime in runtimes]
        status_counts: dict[str, int] = {}
        for status in statuses:
            status_counts[status] = status_counts.get(status, 0) + 1
        stage_status = _aggregate_stage_status(statuses)
        blocked = list(blocked_by_stage_ids or [])
        satisfied = list(satisfied_dependency_stage_ids or [])
        return cls(
            stage=stage,
            label=_STAGE_LABELS.get(stage, stage.value.title()),
            order=_STAGE_SORT_ORDER.get(stage, 999),
            status=stage_status,
            node_ids=list(node_ids),
            dependency_stage_ids=list(dependency_stage_ids),
            node_count=len(node_ids),
            status_counts=status_counts,
            ready=not blocked and stage_status in {"pending", "partial"},
            blocked_by_stage_ids=blocked,
            satisfied_dependency_stage_ids=satisfied,
            dependency_count=dependency_count,
            blocking_dependency_count=blocking_dependency_count,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "label": self.label,
            "order": self.order,
            "status": self.status,
            "node_ids": list(self.node_ids),
            "dependency_stage_ids": list(self.dependency_stage_ids),
            "node_count": self.node_count,
            "status_counts": dict(self.status_counts),
            "ready": self.ready,
            "blocked_by_stage_ids": list(self.blocked_by_stage_ids),
            "satisfied_dependency_stage_ids": list(self.satisfied_dependency_stage_ids),
            "dependency_count": self.dependency_count,
            "blocking_dependency_count": self.blocking_dependency_count,
            "metadata": dict(self.metadata),
        }


@dataclass
class PipelineLiveUpdate:
    """Real-time DAG update payload."""

    pipeline_id: str
    event_type: str
    node_id: str | None = None
    stage: PipelineStage | None = None
    status: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_status_change(
        cls,
        *,
        pipeline_id: str,
        node_id: str,
        stage: PipelineStage | str | None,
        status: str,
        event_type: str = "node_status_changed",
        **payload: Any,
    ) -> PipelineLiveUpdate:
        return cls(
            pipeline_id=pipeline_id,
            event_type=event_type,
            node_id=node_id,
            stage=_coerce_stage(stage),
            status=status,
            payload=payload,
        )

    @classmethod
    def from_node_added(
        cls,
        *,
        pipeline_id: str,
        node: Any,
        event_type: str = "node_added",
        **payload: Any,
    ) -> PipelineLiveUpdate:
        if hasattr(node, "to_dict"):
            node_dict = node.to_dict()
        else:
            node_dict = dict(node)
        stage = _coerce_stage(node_dict.get("stage"))
        combined_payload = {"node": node_dict, **payload}
        return cls(
            pipeline_id=pipeline_id,
            event_type=event_type,
            node_id=node_dict.get("id"),
            stage=stage,
            status=node_dict.get("execution_status"),
            payload=combined_payload,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "event_type": self.event_type,
            "node_id": self.node_id,
            "stage": self.stage.value if self.stage else None,
            "status": self.status,
            "payload": dict(self.payload),
            "timestamp": self.timestamp,
        }


@dataclass
class PipelineDAGSnapshot:
    """Serializable snapshot optimized for DAG rendering."""

    graph_id: str
    name: str
    nodes: list[dict[str, Any]]
    dependencies: list[PipelineDAGDependency]
    runtime: dict[str, PipelineNodeRuntime]
    metadata: dict[str, Any] = field(default_factory=dict)
    integrity_hash: str = ""
    stages: list[PipelineDAGStage] = field(default_factory=list)
    stage_dependencies: list[PipelineStageDependency] = field(default_factory=list)
    live_updates: list[PipelineLiveUpdate] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)

    @classmethod
    def from_graph(cls, graph: Any) -> PipelineDAGSnapshot:
        node_values = sorted(
            getattr(graph, "nodes", {}).values(),
            key=lambda node: (
                _STAGE_SORT_ORDER.get(
                    _coerce_stage(getattr(node, "stage", None)) or PipelineStage.IDEAS,
                    999,
                ),
                float(getattr(node, "created_at", 0.0) or 0.0),
                str(getattr(node, "id", "")),
            ),
        )
        runtime = {str(node.id): PipelineNodeRuntime.from_node(node) for node in node_values}
        nodes = [node.to_dict() for node in node_values]

        dependencies: list[PipelineDAGDependency] = []
        dependency_keys: set[tuple[str, str, str, bool]] = set()
        graph_nodes = getattr(graph, "nodes", {})

        for edge in getattr(graph, "edges", {}).values():
            source = graph_nodes.get(getattr(edge, "source_id", ""))
            target = graph_nodes.get(getattr(edge, "target_id", ""))
            if source is None or target is None:
                continue
            dependency = PipelineDAGDependency.from_edge(
                edge,
                source_stage=_coerce_stage(getattr(source, "stage", None)) or PipelineStage.IDEAS,
                target_stage=_coerce_stage(getattr(target, "stage", None)) or PipelineStage.IDEAS,
            )
            key = (
                dependency.source_id,
                dependency.target_id,
                dependency.edge_type.value,
                dependency.implicit,
            )
            if key not in dependency_keys:
                dependency_keys.add(key)
                dependencies.append(dependency)

        existing_pairs = {
            (dependency.source_id, dependency.target_id) for dependency in dependencies
        }
        for node in node_values:
            for parent_id in getattr(node, "parent_ids", []):
                parent = graph_nodes.get(parent_id)
                if parent is None or (parent_id, node.id) in existing_pairs:
                    continue
                dependency = PipelineDAGDependency.from_parent_link(parent, node)
                key = (
                    dependency.source_id,
                    dependency.target_id,
                    dependency.edge_type.value,
                    dependency.implicit,
                )
                if key not in dependency_keys:
                    dependency_keys.add(key)
                    dependencies.append(dependency)

        dependencies.sort(
            key=lambda dependency: (
                _STAGE_SORT_ORDER.get(dependency.source_stage, 999),
                _STAGE_SORT_ORDER.get(dependency.target_stage, 999),
                dependency.source_id,
                dependency.target_id,
                dependency.edge_type.value,
            )
        )

        metadata = dict(getattr(graph, "metadata", {}) or {})
        metadata.setdefault("owner_id", getattr(graph, "owner_id", None))
        metadata.setdefault("workspace_id", getattr(graph, "workspace_id", None))
        metadata.setdefault("updated_at", getattr(graph, "updated_at", None))

        snapshot = cls(
            graph_id=str(getattr(graph, "id")),
            name=str(getattr(graph, "name", "Untitled Pipeline")),
            nodes=nodes,
            dependencies=dependencies,
            runtime=runtime,
            metadata=metadata,
            integrity_hash=str(getattr(graph, "integrity_hash", lambda: "")() or ""),
            generated_at=time.time(),
        )
        snapshot.refresh_stage_summaries()
        return snapshot

    @property
    def stage_status(self) -> dict[str, str]:
        return {stage.stage.value: stage.status for stage in self.stages}

    def dependency_map(self) -> dict[str, list[str]]:
        dependency_map: dict[str, set[str]] = {}
        for dependency in self.dependencies:
            dependency_map.setdefault(dependency.target_id, set()).add(dependency.source_id)
        return {
            target_id: sorted(source_ids)
            for target_id, source_ids in sorted(dependency_map.items(), key=lambda item: item[0])
        }

    def stage_dependency_map(self) -> dict[str, list[str]]:
        dependency_map: dict[str, set[str]] = {}
        for dependency in self.stage_dependencies:
            dependency_map.setdefault(dependency.target_stage.value, set()).add(
                dependency.source_stage.value
            )
        return {
            target_stage: sorted(source_stages)
            for target_stage, source_stages in sorted(
                dependency_map.items(), key=lambda item: item[0]
            )
        }

    def refresh_stage_summaries(self) -> None:
        node_ids_by_stage: dict[PipelineStage, list[str]] = {stage: [] for stage in PipelineStage}
        dependency_stage_ids: dict[PipelineStage, set[PipelineStage]] = {
            stage: set() for stage in PipelineStage
        }
        stage_dependency_counts: dict[tuple[PipelineStage, PipelineStage], dict[str, int]] = {}

        for node in self.nodes:
            stage = _coerce_stage(node.get("stage"))
            node_id = node.get("id")
            if stage is None or not node_id:
                continue
            node_ids_by_stage[stage].append(str(node_id))

        for dependency in self.dependencies:
            if dependency.cross_stage:
                dependency_stage_ids[dependency.target_stage].add(dependency.source_stage)
                counts = stage_dependency_counts.setdefault(
                    (dependency.source_stage, dependency.target_stage),
                    {"edge_count": 0, "blocking_edge_count": 0},
                )
                counts["edge_count"] += 1
                if dependency.blocking:
                    counts["blocking_edge_count"] += 1

        stage_status_map: dict[PipelineStage, str] = {}
        for stage in sorted(PipelineStage, key=_stage_sort_key):
            runtimes = [
                self.runtime[node_id]
                for node_id in node_ids_by_stage[stage]
                if node_id in self.runtime
            ]
            stage_status_map[stage] = _aggregate_stage_status(
                [runtime.execution_status for runtime in runtimes]
            )

        stage_dependencies: list[PipelineStageDependency] = []
        for (source_stage, target_stage), counts in sorted(
            stage_dependency_counts.items(),
            key=lambda item: (
                _STAGE_SORT_ORDER.get(item[0][0], 999),
                _STAGE_SORT_ORDER.get(item[0][1], 999),
            ),
        ):
            source_status = stage_status_map.get(source_stage, "pending")
            target_status = stage_status_map.get(target_stage, "pending")
            stage_dependencies.append(
                PipelineStageDependency(
                    id=f"{source_stage.value}->{target_stage.value}",
                    source_stage=source_stage,
                    target_stage=target_stage,
                    source_status=source_status,
                    target_status=target_status,
                    edge_count=counts["edge_count"],
                    blocking_edge_count=counts["blocking_edge_count"],
                    status=_stage_dependency_status(
                        source_status,
                        blocking=counts["blocking_edge_count"] > 0,
                    ),
                    metadata={
                        "dependency_stage_ids": [source_stage.value, target_stage.value],
                    },
                )
            )

        stages: list[PipelineDAGStage] = []
        for stage in sorted(PipelineStage, key=_stage_sort_key):
            runtimes = [
                self.runtime[node_id]
                for node_id in node_ids_by_stage[stage]
                if node_id in self.runtime
            ]
            stage_dependencies_for_target = [
                dependency for dependency in stage_dependencies if dependency.target_stage == stage
            ]
            blocked_by_stage_ids = [
                dependency.source_stage.value
                for dependency in stage_dependencies_for_target
                if dependency.blocking and dependency.status != "satisfied"
            ]
            satisfied_dependency_stage_ids = [
                dependency.source_stage.value
                for dependency in stage_dependencies_for_target
                if dependency.status == "satisfied"
            ]
            stages.append(
                PipelineDAGStage.from_runtime(
                    stage,
                    node_ids=node_ids_by_stage[stage],
                    runtimes=runtimes,
                    dependency_stage_ids=[
                        dependency_stage.value
                        for dependency_stage in sorted(
                            dependency_stage_ids[stage],
                            key=_stage_sort_key,
                        )
                    ],
                    blocked_by_stage_ids=blocked_by_stage_ids,
                    satisfied_dependency_stage_ids=satisfied_dependency_stage_ids,
                    dependency_count=sum(
                        dependency.edge_count for dependency in stage_dependencies_for_target
                    ),
                    blocking_dependency_count=sum(
                        dependency.blocking_edge_count
                        for dependency in stage_dependencies_for_target
                    ),
                )
            )

        self.stages = stages
        self.stage_dependencies = stage_dependencies
        self.metadata["stage_status"] = self.stage_status
        self.metadata["stage_dependency_map"] = self.stage_dependency_map()

    def to_react_flow(self, stage_filter: PipelineStage | str | None = None) -> dict[str, Any]:
        from aragora.pipeline.universal_node import UniversalNode

        selected_stage = _coerce_stage(stage_filter)
        flow_nodes: list[dict[str, Any]] = []
        visible_node_ids: set[str] = set()

        for node_data in self.nodes:
            node_stage = _coerce_stage(node_data.get("stage"))
            if selected_stage is not None and node_stage != selected_stage:
                continue

            node = UniversalNode.from_dict(node_data)
            runtime = self.runtime.get(node.id)
            if runtime is not None:
                node.execution_status = runtime.execution_status
                node.metadata = {
                    **node.metadata,
                    **runtime.metadata,
                    "execution_status": runtime.execution_status,
                }
            flow_nodes.append(node.to_react_flow_node())
            visible_node_ids.add(node.id)

        flow_edges = [
            dependency.to_react_flow_edge()
            for dependency in self.dependencies
            if dependency.source_id in visible_node_ids and dependency.target_id in visible_node_ids
        ]
        return {"nodes": flow_nodes, "edges": flow_edges}

    def apply_live_update(self, update: PipelineLiveUpdate) -> None:
        self.live_updates.append(update)
        if len(self.live_updates) > 100:
            self.live_updates = self.live_updates[-100:]

        node_payload = update.payload.get("node")
        if isinstance(node_payload, dict):
            self._upsert_node(node_payload)

        dependency_payload = update.payload.get("dependency")
        if isinstance(dependency_payload, dict):
            self._upsert_dependency(dependency_payload)

        dependency_payloads = update.payload.get("dependencies")
        if isinstance(dependency_payloads, list):
            for payload in dependency_payloads:
                if isinstance(payload, dict):
                    self._upsert_dependency(payload)

        if update.node_id:
            stage = update.stage or self._stage_for_node(update.node_id)
            if stage is not None:
                runtime = self.runtime.get(update.node_id)
                if runtime is None:
                    runtime = PipelineNodeRuntime(node_id=update.node_id, stage=stage)
                    self.runtime[update.node_id] = runtime
                else:
                    runtime.stage = stage

                if update.status is not None:
                    runtime.execution_status = _normalize_runtime_status(update.status)
                runtime.updated_at = max(runtime.updated_at, update.timestamp)
                runtime.metadata.update(self._filtered_payload(update.payload))
                if "assigned_agent" in update.payload:
                    runtime.assigned_agent = str(update.payload["assigned_agent"])
                elif "agent" in update.payload:
                    runtime.assigned_agent = str(update.payload["agent"])

                node = self._find_node(update.node_id)
                if node is not None:
                    if update.status is not None:
                        node["execution_status"] = runtime.execution_status
                    node["updated_at"] = runtime.updated_at
                    node_metadata = dict(node.get("metadata", {}) or {})
                    node_metadata.update(self._filtered_payload(update.payload))
                    if update.status is not None:
                        node_metadata["execution_status"] = runtime.execution_status
                    node["metadata"] = node_metadata

        self.generated_at = max(self.generated_at, update.timestamp)
        self.refresh_stage_summaries()

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "name": self.name,
            "nodes": [dict(node) for node in self.nodes],
            "dependencies": [dependency.to_dict() for dependency in self.dependencies],
            "runtime": {
                node_id: runtime.to_dict() for node_id, runtime in sorted(self.runtime.items())
            },
            "stages": [stage.to_dict() for stage in self.stages],
            "stage_dependencies": [dependency.to_dict() for dependency in self.stage_dependencies],
            "stage_status": self.stage_status,
            "dependency_map": self.dependency_map(),
            "stage_dependency_map": self.stage_dependency_map(),
            "metadata": dict(self.metadata),
            "integrity_hash": self.integrity_hash,
            "live_updates": [update.to_dict() for update in self.live_updates],
            "generated_at": self.generated_at,
        }

    @staticmethod
    def _filtered_payload(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in payload.items()
            if key not in {"node", "dependency", "dependencies"}
        }

    def _stage_for_node(self, node_id: str) -> PipelineStage | None:
        node = self._find_node(node_id)
        if node is None:
            return None
        return _coerce_stage(node.get("stage"))

    def _find_node(self, node_id: str) -> dict[str, Any] | None:
        for node in self.nodes:
            if node.get("id") == node_id:
                return node
        return None

    def _upsert_node(self, node: dict[str, Any]) -> None:
        node_id = node.get("id")
        if not node_id:
            return
        existing = self._find_node(str(node_id))
        if existing is None:
            self.nodes.append(dict(node))
        else:
            existing.update(node)

    def _upsert_dependency(self, dependency: dict[str, Any]) -> None:
        source_stage = _coerce_stage(dependency.get("source_stage"))
        target_stage = _coerce_stage(dependency.get("target_stage"))
        if source_stage is None or target_stage is None:
            return

        edge_type = _coerce_edge_type(dependency.get("edge_type"))
        dependency_id = str(
            dependency.get("id")
            or f"{dependency.get('source_id')}->{dependency.get('target_id')}:{edge_type.value}"
        )
        candidate = PipelineDAGDependency(
            id=dependency_id,
            source_id=str(dependency.get("source_id")),
            target_id=str(dependency.get("target_id")),
            edge_type=edge_type,
            source_stage=source_stage,
            target_stage=target_stage,
            label=str(dependency.get("label", "") or edge_type.value),
            cross_stage=bool(dependency.get("cross_stage", source_stage != target_stage)),
            blocking=bool(dependency.get("blocking", edge_type in _BLOCKING_EDGE_TYPES)),
            implicit=bool(dependency.get("implicit", False)),
            metadata=dict(dependency.get("metadata", {}) or {}),
        )

        for index, existing in enumerate(self.dependencies):
            if existing.id == candidate.id:
                self.dependencies[index] = candidate
                break
        else:
            self.dependencies.append(candidate)
            self.dependencies.sort(
                key=lambda item: (
                    _STAGE_SORT_ORDER.get(item.source_stage, 999),
                    _STAGE_SORT_ORDER.get(item.target_stage, 999),
                    item.source_id,
                    item.target_id,
                    item.edge_type.value,
                )
            )


__all__ = [
    "PipelineDAGDependency",
    "PipelineDAGSnapshot",
    "PipelineDAGStage",
    "PipelineStageDependency",
    "PipelineLiveUpdate",
    "PipelineNodeRuntime",
]
