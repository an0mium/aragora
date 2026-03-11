"""Persistent campaign planning and execution on top of the Boss loop."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from aragora.agents.base import create_agent
from aragora.nomic.task_decomposer import SubTask, TaskDecomposer
from aragora.swarm.boss_loop import (
    _extract_deliverable,
    _classify_terminal_run_outcome,
    dispatch_bounded_spec,
)
from aragora.swarm.spec import SwarmSpec
from aragora.swarm.supervisor import SwarmSupervisor

logger = logging.getLogger(__name__)

UTC = timezone.utc
DEFAULT_CAMPAIGN_MANIFEST = ".aragora/campaign_manifest.yaml"

# Statuses that represent terminal project transitions (no further retries).
# Receipt emission fires only for these statuses.
_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {
        "completed",
        "failed",
        "blocked",
        "skipped",
    }
)


class CampaignProjectStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    ACTIVE = "active"
    DELIVERED = "delivered"
    NEEDS_REVISION = "needs_revision"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class CampaignRunOutcome(str, Enum):
    DELIVERABLE_CREATED = "deliverable_created"
    PR_ADOPTED = "pr_adopted"
    CLEAN_EXIT_NO_DELIVERABLE = "clean_exit_no_deliverable"
    NEEDS_HUMAN = "needs_human"
    TIMEOUT = "timeout"
    CRASH = "crash"
    BLOCKED = "blocked"


class CampaignStopReason(str, Enum):
    STILL_RUNNING = "still_running"
    CAMPAIGN_COMPLETE = "campaign_complete"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TIME_LIMIT_EXCEEDED = "time_limit_exceeded"
    CAMPAIGN_BLOCKED = "campaign_blocked"


class CampaignReviewStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    CHANGES_REQUESTED = "changes_requested"
    BLOCKED_NONREVIEWABLE = "blocked_nonreviewable"


@dataclass(slots=True)
class CampaignDependency:
    project_id: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"project_id": self.project_id, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignDependency:
        return cls(
            project_id=str(data.get("project_id", "")).strip(),
            reason=str(data.get("reason", "")).strip(),
        )


@dataclass(slots=True)
class CampaignReviewGate:
    required: bool = True
    review_model: str = "claude"
    status: str = CampaignReviewStatus.PENDING.value
    findings: list[str] = field(default_factory=list)
    reviewed_at: str | None = None
    raw_review: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "review_model": self.review_model,
            "status": self.status,
            "findings": list(self.findings),
            "reviewed_at": self.reviewed_at,
            "raw_review": dict(self.raw_review),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CampaignReviewGate:
        data = data or {}
        return cls(
            required=bool(data.get("required", True)),
            review_model=str(data.get("review_model", "claude")).strip() or "claude",
            status=str(data.get("status", CampaignReviewStatus.PENDING.value)).strip()
            or CampaignReviewStatus.PENDING.value,
            findings=[str(item) for item in data.get("findings", []) if str(item).strip()],
            reviewed_at=str(data.get("reviewed_at", "")).strip() or None,
            raw_review=dict(data.get("raw_review") or {}),
        )


@dataclass(slots=True)
class CampaignExecutionState:
    ready_queue: list[str] = field(default_factory=list)
    active_projects: list[str] = field(default_factory=list)
    completed_projects: list[str] = field(default_factory=list)
    failed_projects: list[str] = field(default_factory=list)
    skipped_projects: list[str] = field(default_factory=list)
    total_cost_usd: float = 0.0
    last_run_at: str | None = None
    last_result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready_queue": list(self.ready_queue),
            "active_projects": list(self.active_projects),
            "completed_projects": list(self.completed_projects),
            "failed_projects": list(self.failed_projects),
            "skipped_projects": list(self.skipped_projects),
            "total_cost_usd": self.total_cost_usd,
            "last_run_at": self.last_run_at,
            "last_result": dict(self.last_result),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CampaignExecutionState:
        data = data or {}
        return cls(
            ready_queue=[str(item) for item in data.get("ready_queue", []) if str(item).strip()],
            active_projects=[
                str(item) for item in data.get("active_projects", []) if str(item).strip()
            ],
            completed_projects=[
                str(item) for item in data.get("completed_projects", []) if str(item).strip()
            ],
            failed_projects=[
                str(item) for item in data.get("failed_projects", []) if str(item).strip()
            ],
            skipped_projects=[
                str(item) for item in data.get("skipped_projects", []) if str(item).strip()
            ],
            total_cost_usd=float(data.get("total_cost_usd", 0.0) or 0.0),
            last_run_at=str(data.get("last_run_at", "")).strip() or None,
            last_result=dict(data.get("last_result") or {}),
        )


@dataclass(slots=True)
class CampaignProject:
    project_id: str
    title: str
    source_refs: list[str] = field(default_factory=list)
    spec: SwarmSpec = field(default_factory=SwarmSpec)
    file_scope_hints: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    dependencies: list[CampaignDependency] = field(default_factory=list)
    estimated_cost_usd: float = 0.0
    status: str = CampaignProjectStatus.PENDING.value
    retry_count: int = 0
    last_run_outcome: str | None = None
    run_id: str | None = None
    receipt_id: str | None = None
    pr_url: str | None = None
    adopted_pr: str | None = None
    branch: str | None = None
    commit_shas: list[str] = field(default_factory=list)
    review: CampaignReviewGate = field(default_factory=CampaignReviewGate)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "title": self.title,
            "source_refs": list(self.source_refs),
            "spec": self.spec.to_dict(),
            "file_scope_hints": list(self.file_scope_hints),
            "acceptance_criteria": list(self.acceptance_criteria),
            "constraints": list(self.constraints),
            "dependencies": [item.to_dict() for item in self.dependencies],
            "estimated_cost_usd": self.estimated_cost_usd,
            "status": self.status,
            "retry_count": self.retry_count,
            "last_run_outcome": self.last_run_outcome,
            "run_id": self.run_id,
            "receipt_id": self.receipt_id,
            "pr_url": self.pr_url,
            "adopted_pr": self.adopted_pr,
            "branch": self.branch,
            "commit_shas": list(self.commit_shas),
            "review": self.review.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignProject:
        return cls(
            project_id=str(data.get("project_id", "")).strip(),
            title=str(data.get("title", "")).strip(),
            source_refs=[str(item) for item in data.get("source_refs", []) if str(item).strip()],
            spec=SwarmSpec.from_dict(dict(data.get("spec") or {})),
            file_scope_hints=[
                str(item) for item in data.get("file_scope_hints", []) if str(item).strip()
            ],
            acceptance_criteria=[
                str(item) for item in data.get("acceptance_criteria", []) if str(item).strip()
            ],
            constraints=[str(item) for item in data.get("constraints", []) if str(item).strip()],
            dependencies=[
                CampaignDependency.from_dict(item)
                for item in data.get("dependencies", [])
                if isinstance(item, dict)
            ],
            estimated_cost_usd=float(data.get("estimated_cost_usd", 0.0) or 0.0),
            status=str(data.get("status", CampaignProjectStatus.PENDING.value)).strip()
            or CampaignProjectStatus.PENDING.value,
            retry_count=int(data.get("retry_count", 0) or 0),
            last_run_outcome=str(data.get("last_run_outcome", "")).strip() or None,
            run_id=str(data.get("run_id", "")).strip() or None,
            receipt_id=str(data.get("receipt_id", "")).strip() or None,
            pr_url=str(data.get("pr_url", "")).strip() or None,
            adopted_pr=str(data.get("adopted_pr", "")).strip() or None,
            branch=str(data.get("branch", "")).strip() or None,
            commit_shas=[str(item) for item in data.get("commit_shas", []) if str(item).strip()],
            review=CampaignReviewGate.from_dict(data.get("review")),
        )


@dataclass(slots=True)
class CampaignManifest:
    campaign_id: str
    created_at: str
    source_kind: str
    source_ref: str
    planner_model: str = "claude"
    worker_model: str = "codex"
    review_model: str = "claude"
    max_parallel_ready_projects: int = 2
    max_retries_per_project: int = 2
    budget_limit_usd: float = 50.0
    time_limit_hours: float = 8.0
    projects: list[CampaignProject] = field(default_factory=list)
    execution_state: CampaignExecutionState = field(default_factory=CampaignExecutionState)
    planning_findings: list[str] = field(default_factory=list)
    manifest_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "created_at": self.created_at,
            "source_kind": self.source_kind,
            "source_ref": self.source_ref,
            "planner_model": self.planner_model,
            "worker_model": self.worker_model,
            "review_model": self.review_model,
            "max_parallel_ready_projects": self.max_parallel_ready_projects,
            "max_retries_per_project": self.max_retries_per_project,
            "budget_limit_usd": self.budget_limit_usd,
            "time_limit_hours": self.time_limit_hours,
            "projects": [project.to_dict() for project in self.projects],
            "execution_state": self.execution_state.to_dict(),
            "planning_findings": list(self.planning_findings),
            "manifest_version": self.manifest_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignManifest:
        return cls(
            campaign_id=str(data.get("campaign_id", "")).strip(),
            created_at=str(data.get("created_at", "")).strip(),
            source_kind=str(data.get("source_kind", "")).strip(),
            source_ref=str(data.get("source_ref", "")).strip(),
            planner_model=str(data.get("planner_model", "claude")).strip() or "claude",
            worker_model=str(data.get("worker_model", "codex")).strip() or "codex",
            review_model=str(data.get("review_model", "claude")).strip() or "claude",
            max_parallel_ready_projects=max(
                1, int(data.get("max_parallel_ready_projects", 2) or 2)
            ),
            max_retries_per_project=max(0, int(data.get("max_retries_per_project", 2) or 2)),
            budget_limit_usd=float(data.get("budget_limit_usd", 50.0) or 50.0),
            time_limit_hours=float(data.get("time_limit_hours", 8.0) or 8.0),
            projects=[
                CampaignProject.from_dict(item)
                for item in data.get("projects", [])
                if isinstance(item, dict)
            ],
            execution_state=CampaignExecutionState.from_dict(data.get("execution_state")),
            planning_findings=[
                str(item) for item in data.get("planning_findings", []) if str(item).strip()
            ],
            manifest_version=int(data.get("manifest_version", 1) or 1),
        )

    def to_yaml(self) -> str:
        data = self.to_dict()
        try:
            import yaml

            return yaml.safe_dump(data, sort_keys=True, allow_unicode=False)
        except ImportError:
            return json.dumps(data, indent=2, sort_keys=True)

    @classmethod
    def from_text(cls, text: str) -> CampaignManifest:
        try:
            import yaml

            loaded = yaml.safe_load(text) or {}
        except ImportError:
            loaded = json.loads(text)
        if not isinstance(loaded, dict):
            raise ValueError("Campaign manifest must deserialize to an object.")
        return cls.from_dict(loaded)

    def project_map(self) -> dict[str, CampaignProject]:
        return {project.project_id: project for project in self.projects}


@contextlib.contextmanager
def locked_manifest_path(path: Path):
    """Hold an exclusive lock around manifest operations."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+", encoding="utf-8")
    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except ImportError:
            pass
        handle.close()


def load_campaign_manifest(path: Path) -> CampaignManifest:
    text = path.read_text(encoding="utf-8")
    manifest = CampaignManifest.from_text(text)
    _validate_campaign_manifest(manifest)
    return manifest


def save_campaign_manifest(path: Path, manifest: CampaignManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_yaml(), encoding="utf-8")


def _validate_campaign_manifest(manifest: CampaignManifest) -> None:
    if not manifest.campaign_id:
        raise ValueError("Campaign manifest is missing campaign_id.")
    if not manifest.source_kind:
        raise ValueError("Campaign manifest is missing source_kind.")
    seen: set[str] = set()
    for project in manifest.projects:
        if not project.project_id:
            raise ValueError("Campaign manifest contains a project without project_id.")
        if project.project_id in seen:
            raise ValueError(
                f"Campaign manifest contains duplicate project_id {project.project_id}."
            )
        seen.add(project.project_id)
        if not project.spec.is_dispatch_bounded():
            raise ValueError(f"Project {project.project_id} is not dispatch-bounded.")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_review_model(worker_model: str, requested: str | None = None) -> str:
    candidate = str(requested or "").strip() or ("claude" if worker_model == "codex" else "codex")
    if candidate == worker_model:
        return "claude" if worker_model == "codex" else "codex"
    return candidate


def _complexity_cost(label: str) -> float:
    lowered = str(label or "").strip().lower()
    return {"low": 0.5, "medium": 1.0, "moderate": 1.0, "high": 2.0}.get(lowered, 1.0)


def _success_criteria_to_list(success_criteria: dict[str, Any]) -> list[str]:
    items: list[str] = []
    for key, value in sorted(success_criteria.items()):
        text = str(value).strip()
        if text:
            items.append(f"{key}: {text}")
    return items


def _normalized_scope(scope: list[str]) -> list[str]:
    return sorted({str(item).strip() for item in scope if str(item).strip()})


def _parse_source_items(text: str) -> list[str]:
    lines = [line.rstrip() for line in text.splitlines()]
    items: list[str] = []
    current: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            if current:
                items.append("\n".join(current).strip())
                current = []
            continue
        if line.startswith(("-", "*")) or line[:2].isdigit() and line[2:3] == ".":
            if current:
                items.append("\n".join(current).strip())
                current = []
            item = line.lstrip("-*0123456789. ").strip()
            if item:
                items.append(item)
            continue
        if line.startswith("#"):
            if current:
                items.append("\n".join(current).strip())
                current = []
            current.append(line.lstrip("# ").strip())
            continue
        current.append(line)
    if current:
        items.append("\n".join(current).strip())
    return [item for item in items if item]


def _gh_json(cmd: list[str]) -> Any:
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=30, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "GitHub command failed")
    return json.loads(proc.stdout)


class CampaignPlanner:
    """Build a persistent campaign manifest from documents or GitHub issues."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        planner_model: str = "claude",
        worker_model: str = "codex",
        review_model: str = "claude",
        budget_limit_usd: float = 50.0,
        max_parallel_ready_projects: int = 2,
        decomposer: TaskDecomposer | None = None,
        enable_model_crosscheck: bool = False,
    ) -> None:
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.planner_model = planner_model
        self.worker_model = worker_model
        self.review_model = _canonical_review_model(worker_model, review_model)
        self.budget_limit_usd = budget_limit_usd
        self.max_parallel_ready_projects = max(1, int(max_parallel_ready_projects))
        self.decomposer = decomposer or TaskDecomposer()
        self.enable_model_crosscheck = enable_model_crosscheck

    def plan_from_source_file(self, path: Path) -> CampaignManifest:
        text = path.read_text(encoding="utf-8")
        items = _parse_source_items(text)
        return self.plan_from_items(items, source_kind="source_file", source_ref=str(path))

    def plan_from_issue_list(
        self, issue_numbers: list[int], *, repo: str | None = None
    ) -> CampaignManifest:
        items: list[str] = []
        source_refs: list[str] = []
        for number in issue_numbers:
            cmd = [
                "gh",
                "issue",
                "view",
                str(number),
                "--json",
                "number,title,body,url",
            ]
            if repo:
                cmd.extend(["--repo", repo])
            data = _gh_json(cmd)
            title = str(data.get("title", "")).strip()
            body = str(data.get("body", "")).strip()
            url = str(data.get("url", "")).strip()
            items.append(f"[Issue #{number}] {title}\n\n{body}".strip())
            source_refs.append(url or f"issue:{number}")
        manifest = self.plan_from_items(
            items,
            source_kind="issue_list",
            source_ref=",".join(str(num) for num in issue_numbers),
        )
        for project, source_ref in zip(manifest.projects, source_refs, strict=False):
            if source_ref not in project.source_refs:
                project.source_refs.append(source_ref)
        return manifest

    def plan_from_github_query(self, query: str, *, repo: str | None = None) -> CampaignManifest:
        cmd = [
            "gh",
            "issue",
            "list",
            "--state",
            "open",
            "--limit",
            "100",
            "--search",
            query,
            "--json",
            "number,title,body,url",
        ]
        if repo:
            cmd.extend(["--repo", repo])
        data = _gh_json(cmd)
        items = []
        for item in data:
            if not isinstance(item, dict):
                continue
            items.append(
                f"[Issue #{item.get('number')}] {item.get('title', '')}\n\n{item.get('body', '')}".strip()
            )
        return self.plan_from_items(items, source_kind="github_query", source_ref=query)

    def plan_from_items(
        self, items: list[str], *, source_kind: str, source_ref: str
    ) -> CampaignManifest:
        projects: list[CampaignProject] = []
        source_findings: list[str] = []
        source_index = 0
        for item in items:
            source_index += 1
            created = self._projects_from_item(item, source_index)
            if not created:
                source_findings.append(f"Skipped under-specified candidate: {item[:80]}")
                continue
            projects.extend(created)
        self._apply_overlap_dependencies(projects, findings=source_findings)
        projects = self._topological_sort_projects(projects)
        self._renumber_projects(projects)
        crosscheck_findings = self._crosscheck_projects(projects)
        source_findings.extend(crosscheck_findings)
        manifest = CampaignManifest(
            campaign_id=f"campaign-{uuid.uuid4().hex[:12]}",
            created_at=_now_iso(),
            source_kind=source_kind,
            source_ref=source_ref,
            planner_model=self.planner_model,
            worker_model=self.worker_model,
            review_model=self.review_model,
            max_parallel_ready_projects=self.max_parallel_ready_projects,
            budget_limit_usd=self.budget_limit_usd,
            projects=projects,
            planning_findings=source_findings,
        )
        _refresh_execution_state(manifest)
        return manifest

    def _projects_from_item(self, item: str, source_index: int) -> list[CampaignProject]:
        base_spec = SwarmSpec.from_direct_goal(
            item,
            budget_limit_usd=self.budget_limit_usd,
            requires_approval=True,
            user_expertise="developer",
        )
        decomposition = self.decomposer.analyze(item)
        projects: list[CampaignProject] = []
        if decomposition.should_decompose and decomposition.subtasks:
            subtask_id_map: dict[str, str] = {}
            for sub_index, subtask in enumerate(decomposition.subtasks, start=1):
                spec = self._spec_from_subtask(subtask, base_spec)
                if not spec.is_dispatch_bounded():
                    continue
                project_id = f"proj-{source_index:03d}-{sub_index:02d}"
                subtask_id_map[subtask.id] = project_id
                projects.append(
                    CampaignProject(
                        project_id=project_id,
                        title=subtask.title or project_id,
                        source_refs=[f"{source_index}:{subtask.id}"],
                        spec=spec,
                        file_scope_hints=list(spec.file_scope_hints),
                        acceptance_criteria=list(spec.acceptance_criteria),
                        constraints=list(spec.constraints),
                        estimated_cost_usd=_complexity_cost(subtask.estimated_complexity),
                        review=CampaignReviewGate(
                            required=True,
                            review_model=self.review_model,
                        ),
                    )
                )
            for project, subtask in zip(projects, decomposition.subtasks, strict=False):
                project.dependencies = [
                    CampaignDependency(project_id=subtask_id_map[dep], reason="subtask_dependency")
                    for dep in subtask.dependencies
                    if dep in subtask_id_map
                ]
            if projects:
                return projects

        if not base_spec.is_dispatch_bounded():
            return []
        return [
            CampaignProject(
                project_id=f"proj-{source_index:03d}",
                title=item.splitlines()[0][:120],
                source_refs=[f"{source_index}"],
                spec=base_spec,
                file_scope_hints=list(base_spec.file_scope_hints),
                acceptance_criteria=list(base_spec.acceptance_criteria),
                constraints=list(base_spec.constraints),
                estimated_cost_usd=_complexity_cost(decomposition.complexity_level),
                review=CampaignReviewGate(required=True, review_model=self.review_model),
            )
        ]

    def _spec_from_subtask(self, subtask: SubTask, base_spec: SwarmSpec) -> SwarmSpec:
        criteria = _success_criteria_to_list(subtask.success_criteria)
        if not criteria:
            criteria = [f"Complete bounded task: {subtask.title}"]
        return SwarmSpec(
            raw_goal=subtask.description or base_spec.raw_goal,
            refined_goal=subtask.title or subtask.description or base_spec.refined_goal,
            acceptance_criteria=criteria,
            constraints=list(base_spec.constraints),
            budget_limit_usd=base_spec.budget_limit_usd,
            file_scope_hints=_normalized_scope(subtask.file_scope or base_spec.file_scope_hints),
            work_orders=[],
            estimated_complexity=subtask.estimated_complexity or "medium",
            requires_approval=True,
            user_expertise="developer",
        )

    def _apply_overlap_dependencies(
        self, projects: list[CampaignProject], *, findings: list[str]
    ) -> None:
        for index, project in enumerate(projects):
            current_scope = set(project.file_scope_hints)
            if not current_scope:
                continue
            for later in projects[index + 1 :]:
                overlap = sorted(current_scope & set(later.file_scope_hints))
                if not overlap:
                    continue
                existing = {dep.project_id for dep in later.dependencies}
                if project.project_id in existing:
                    continue
                later.dependencies.append(
                    CampaignDependency(
                        project_id=project.project_id,
                        reason=f"file_scope_overlap:{', '.join(overlap[:3])}",
                    )
                )
                findings.append(
                    f"Added dependency {later.project_id} -> {project.project_id} due to overlapping scope."
                )

    def _topological_sort_projects(self, projects: list[CampaignProject]) -> list[CampaignProject]:
        by_id = {project.project_id: project for project in projects}
        incoming = {
            project.project_id: {
                dep.project_id for dep in project.dependencies if dep.project_id in by_id
            }
            for project in projects
        }
        ready = [project.project_id for project in projects if not incoming[project.project_id]]
        ordered: list[CampaignProject] = []
        while ready:
            ready.sort()
            project_id = ready.pop(0)
            ordered.append(by_id[project_id])
            for other_id, deps in incoming.items():
                if project_id in deps:
                    deps.remove(project_id)
                    if not deps and by_id[other_id] not in ordered and other_id not in ready:
                        ready.append(other_id)
        leftovers = [project for project in projects if project not in ordered]
        return ordered + sorted(leftovers, key=lambda item: item.project_id)

    def _crosscheck_projects(self, projects: list[CampaignProject]) -> list[str]:
        findings: list[str] = []
        for project in projects:
            if not project.acceptance_criteria:
                findings.append(f"{project.project_id} has no acceptance criteria.")
        if self.enable_model_crosscheck:
            prompt = {
                "projects": [
                    {
                        "project_id": project.project_id,
                        "title": project.title,
                        "dependencies": [dep.to_dict() for dep in project.dependencies],
                        "file_scope_hints": project.file_scope_hints,
                        "acceptance_criteria": project.acceptance_criteria,
                    }
                    for project in projects
                ]
            }
            try:
                agent = create_agent(self.review_model, name="campaign-crosscheck", role="critic")
                response = asyncio.run(agent.generate(json.dumps(prompt, sort_keys=True)))
                for finding in _extract_json_list(response):
                    findings.append(f"crosscheck: {finding}")
            except Exception as exc:
                logger.debug("campaign_crosscheck_fallback: %s", exc)
        return findings

    @staticmethod
    def _renumber_projects(projects: list[CampaignProject]) -> None:
        id_map = {
            project.project_id: f"proj-{index:03d}"
            for index, project in enumerate(projects, start=1)
        }
        for project in projects:
            project.project_id = id_map[project.project_id]
        for project in projects:
            for dependency in project.dependencies:
                if dependency.project_id in id_map:
                    dependency.project_id = id_map[dependency.project_id]


_DIFF_MAX_CHARS = 50_000


def _fetch_diff_content(
    run_dict: dict[str, Any],
    *,
    repo_root: Path | None = None,
    target_branch: str = "main",
    max_chars: int = _DIFF_MAX_CHARS,
) -> str | None:
    """Fetch the actual git diff between the worker branch and the target branch.

    Returns the diff text (truncated to *max_chars*), or ``None`` if the diff
    cannot be obtained (missing branch, no repo_root, git error).
    """
    if repo_root is None:
        return None
    deliverable = _extract_deliverable(run_dict)
    if not deliverable or deliverable.get("type") not in ("branch", "pr"):
        return None
    branch = deliverable.get("branch")
    if not branch:
        # For PR-type deliverables, try extracting the branch from work orders.
        for wo in run_dict.get("work_orders", []):
            if isinstance(wo, dict) and str(wo.get("branch", "")).strip():
                branch = str(wo["branch"]).strip()
                break
    if not branch:
        return None
    try:
        result = subprocess.run(
            ["git", "diff", f"{target_branch}...{branch}"],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            timeout=30,
        )
        if result.returncode != 0:
            logger.debug("_fetch_diff_content git diff failed: %s", result.stderr)
            return None
        diff = result.stdout
        if len(diff) > max_chars:
            diff = diff[:max_chars] + f"\n\n... [truncated at {max_chars} chars]"
        return diff if diff.strip() else None
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("_fetch_diff_content error: %s", exc)
        return None


class CampaignReviewer:
    """Blocking heterogeneous review gate for completed campaign projects."""

    async def review(
        self,
        *,
        project: CampaignProject,
        worker_model: str,
        review_model: str,
        run_dict: dict[str, Any],
        repo_root: Path | None = None,
        target_branch: str = "main",
    ) -> CampaignReviewGate:
        chosen_review_model = _canonical_review_model(worker_model, review_model)
        diff_content = _fetch_diff_content(
            run_dict, repo_root=repo_root, target_branch=target_branch
        )
        prompt = self._build_prompt(
            project, run_dict, chosen_review_model, diff_content=diff_content
        )
        try:
            agent = create_agent(chosen_review_model, name="campaign-review", role="critic")
            raw = await agent.generate(prompt)
            parsed = _extract_first_json_object(raw)
            status = str(parsed.get("status", "")).strip().lower()
            findings = [str(item) for item in parsed.get("findings", []) if str(item).strip()]
            if status not in {
                CampaignReviewStatus.PASSED.value,
                CampaignReviewStatus.CHANGES_REQUESTED.value,
                CampaignReviewStatus.BLOCKED_NONREVIEWABLE.value,
            }:
                status = CampaignReviewStatus.BLOCKED_NONREVIEWABLE.value
                findings = findings or ["Review model returned an invalid decision payload."]
            return CampaignReviewGate(
                required=True,
                review_model=chosen_review_model,
                status=status,
                findings=findings,
                reviewed_at=_now_iso(),
                raw_review={"response": raw},
            )
        except Exception as exc:
            return CampaignReviewGate(
                required=True,
                review_model=chosen_review_model,
                status=CampaignReviewStatus.BLOCKED_NONREVIEWABLE.value,
                findings=[f"Review failed: {type(exc).__name__}: {exc}"],
                reviewed_at=_now_iso(),
                raw_review={"error": str(exc)},
            )

    @staticmethod
    def _build_prompt(
        project: CampaignProject,
        run_dict: dict[str, Any],
        review_model: str,
        *,
        diff_content: str | None = None,
    ) -> str:
        deliverable = _extract_deliverable(run_dict)
        work_orders = run_dict.get("work_orders", [])
        summary = {
            "review_model": review_model,
            "project_id": project.project_id,
            "title": project.title,
            "acceptance_criteria": project.acceptance_criteria,
            "file_scope_hints": project.file_scope_hints,
            "deliverable": deliverable,
            "work_orders": work_orders,
        }
        parts = [
            "Review this completed implementation against the project specification.\n"
            "Respond with strict JSON only: "
            '{"status":"passed|changes_requested|blocked_nonreviewable","findings":["..."]}\n'
            f"{json.dumps(summary, sort_keys=True)}",
        ]
        if diff_content:
            parts.append(
                f"\n\n--- ACTUAL DIFF (worker branch vs base) ---\n{diff_content}\n--- END DIFF ---"
            )
        return "\n".join(parts)


class CampaignExecutor:
    """Replay-safe campaign executor for one invocation."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        repo_root: Path | None = None,
        target_branch: str = "main",
        reviewer: CampaignReviewer | None = None,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.target_branch = target_branch
        self.reviewer = reviewer or CampaignReviewer()

    async def execute_once(self) -> dict[str, Any]:
        with locked_manifest_path(self.manifest_path):
            manifest = load_campaign_manifest(self.manifest_path)
            self._reconcile_active_projects(manifest)
            stop_reason = _compute_stop_reason(manifest)
            if stop_reason != CampaignStopReason.STILL_RUNNING.value:
                manifest.execution_state.last_run_at = _now_iso()
                manifest.execution_state.last_result = {
                    "stop_reason": stop_reason,
                    "dispatched_projects": [],
                }
                _refresh_execution_state(manifest)
                save_campaign_manifest(self.manifest_path, manifest)
                return manifest.execution_state.last_result

            ready = self._ready_projects(manifest)
            active_count = len(
                [
                    project
                    for project in manifest.projects
                    if project.status == CampaignProjectStatus.ACTIVE.value
                ]
            )
            capacity = max(0, manifest.max_parallel_ready_projects - active_count)
            selected_ids = [project.project_id for project in ready[:capacity]]
            if not selected_ids and not ready:
                # No ready projects: distinguish "waiting for in-flight" from "truly blocked"
                if active_count > 0:
                    stop_reason = CampaignStopReason.STILL_RUNNING.value
                else:
                    stop_reason = CampaignStopReason.CAMPAIGN_BLOCKED.value
                manifest.execution_state.last_result = {
                    "stop_reason": stop_reason,
                    "dispatched_projects": [],
                }
                _refresh_execution_state(manifest)
                save_campaign_manifest(self.manifest_path, manifest)
                return manifest.execution_state.last_result
            for project in manifest.projects:
                if project.project_id in selected_ids:
                    project.status = CampaignProjectStatus.ACTIVE.value
                    project.review.status = CampaignReviewStatus.PENDING.value
            manifest.execution_state.last_run_at = _now_iso()
            manifest.execution_state.last_result = {
                "stop_reason": CampaignStopReason.STILL_RUNNING.value,
                "dispatched_projects": list(selected_ids),
            }
            _refresh_execution_state(manifest)
            save_campaign_manifest(self.manifest_path, manifest)

        dispatched = await asyncio.gather(
            *(self._execute_project_id(project_id) for project_id in selected_ids)
        )

        with locked_manifest_path(self.manifest_path):
            manifest = load_campaign_manifest(self.manifest_path)
            manifest.execution_state.last_run_at = _now_iso()
            manifest.execution_state.last_result = {
                "stop_reason": _compute_stop_reason(manifest),
                "dispatched_projects": dispatched,
            }
            _refresh_execution_state(manifest)
            save_campaign_manifest(self.manifest_path, manifest)
            return manifest.execution_state.last_result

    def status(self) -> dict[str, Any]:
        with locked_manifest_path(self.manifest_path):
            manifest = load_campaign_manifest(self.manifest_path)
            _refresh_execution_state(manifest)
            return _manifest_summary(manifest)

    async def review_project(self, project_id: str) -> dict[str, Any]:
        with locked_manifest_path(self.manifest_path):
            manifest = load_campaign_manifest(self.manifest_path)
            project = manifest.project_map().get(project_id)
            if project is None:
                raise KeyError(f"Unknown project_id: {project_id}")
            if not project.run_id:
                raise ValueError(f"Project {project_id} has no recorded run_id.")
            run_dict = self._refresh_run_dict(project.run_id)
            if not run_dict:
                raise ValueError(f"Project {project_id} run {project.run_id} is not available.")
        gate = await self.reviewer.review(
            project=project,
            worker_model=manifest.worker_model,
            review_model=manifest.review_model,
            run_dict=run_dict,
            repo_root=self.repo_root,
            target_branch=self.target_branch,
        )
        with locked_manifest_path(self.manifest_path):
            manifest = load_campaign_manifest(self.manifest_path)
            project = manifest.project_map()[project_id]
            project.review = gate
            self._apply_review_result(manifest, project, gate, run_dict=run_dict)
            _refresh_execution_state(manifest)
            save_campaign_manifest(self.manifest_path, manifest)
            return {"project_id": project_id, "review": gate.to_dict(), "status": project.status}

    def sync_issue_plan(self) -> dict[str, Any]:
        with locked_manifest_path(self.manifest_path):
            manifest = load_campaign_manifest(self.manifest_path)
            items = []
            for project in manifest.projects:
                issue_refs = [
                    ref
                    for ref in project.source_refs
                    if "github.com" in ref or ref.startswith("issue:")
                ]
                items.append(
                    {
                        "project_id": project.project_id,
                        "title": project.title,
                        "existing_issue_refs": issue_refs,
                        "needs_issue_materialization": not bool(issue_refs),
                        "status": project.status,
                    }
                )
            return {
                "mode": "campaign-sync-issues",
                "campaign_id": manifest.campaign_id,
                "source_kind": manifest.source_kind,
                "items": items,
            }

    async def _execute_project_id(self, project_id: str) -> dict[str, Any]:
        with locked_manifest_path(self.manifest_path):
            manifest = load_campaign_manifest(self.manifest_path)
            project = manifest.project_map().get(project_id)
            if project is None:
                raise KeyError(f"Unknown project_id: {project_id}")
            retry_spec = self._spec_for_retry(project)
            worker_model = manifest.worker_model
            review_model = manifest.review_model
            budget_limit = project.spec.budget_limit_usd or manifest.budget_limit_usd

        result = await dispatch_bounded_spec(
            retry_spec,
            target_branch=self.target_branch,
            budget_limit_usd=budget_limit,
            max_ticks=360,
            repo_path=self.repo_root,
            default_target_agent=worker_model,
            default_reviewer_agent=review_model,
            use_managed_session_script=False,
        )

        with locked_manifest_path(self.manifest_path):
            manifest = load_campaign_manifest(self.manifest_path)
            project = manifest.project_map()[project_id]
            self._apply_dispatch_result(manifest, project, result)
            _refresh_execution_state(manifest)
            save_campaign_manifest(self.manifest_path, manifest)
            current_status = project.status

        if current_status == CampaignProjectStatus.DELIVERED.value and isinstance(
            result.get("run"), dict
        ):
            with locked_manifest_path(self.manifest_path):
                manifest = load_campaign_manifest(self.manifest_path)
                project = manifest.project_map()[project_id]
                worker_model = manifest.worker_model
                review_model = manifest.review_model
            gate = await self.reviewer.review(
                project=project,
                worker_model=worker_model,
                review_model=review_model,
                run_dict=dict(result["run"]),
                repo_root=self.repo_root,
                target_branch=self.target_branch,
            )
            with locked_manifest_path(self.manifest_path):
                manifest = load_campaign_manifest(self.manifest_path)
                project = manifest.project_map()[project_id]
                project.review = gate
                self._apply_review_result(
                    manifest, project, gate, run_dict=dict(result.get("run") or {})
                )
                _refresh_execution_state(manifest)
                save_campaign_manifest(self.manifest_path, manifest)

        with locked_manifest_path(self.manifest_path):
            manifest = load_campaign_manifest(self.manifest_path)
            project = manifest.project_map()[project_id]
            return {
                "project_id": project.project_id,
                "status": project.status,
                "outcome": project.last_run_outcome,
                "run_id": project.run_id,
                "pr_url": project.pr_url,
                "adopted_pr": project.adopted_pr,
            }

    def _apply_dispatch_result(
        self, manifest: CampaignManifest, project: CampaignProject, result: dict[str, Any]
    ) -> None:
        project.run_id = str(result.get("run_id", "")).strip() or project.run_id
        run_dict = dict(result.get("run") or {})
        deliverable = dict(result.get("deliverable") or {})
        outcome = str(result.get("outcome", CampaignRunOutcome.BLOCKED.value)).strip()
        project.last_run_outcome = outcome
        project.receipt_id = _first_receipt_id(run_dict) or project.receipt_id
        if deliverable.get("type") == "pr":
            project.pr_url = str(deliverable.get("pr_url", "")).strip() or project.pr_url
        elif deliverable.get("type") == "adopted_pr":
            project.adopted_pr = (
                str(deliverable.get("adopted_pr", "")).strip() or project.adopted_pr
            )
        elif deliverable.get("type") == "branch":
            project.branch = str(deliverable.get("branch", "")).strip() or project.branch
            project.commit_shas = [
                str(item) for item in deliverable.get("commit_shas", []) if str(item).strip()
            ]

        project.retry_count += 1
        manifest.execution_state.total_cost_usd += float(project.estimated_cost_usd or 0.0)
        if outcome in {
            CampaignRunOutcome.DELIVERABLE_CREATED.value,
            CampaignRunOutcome.PR_ADOPTED.value,
        }:
            project.status = CampaignProjectStatus.DELIVERED.value
        elif outcome == CampaignRunOutcome.CLEAN_EXIT_NO_DELIVERABLE.value:
            project.status = CampaignProjectStatus.NEEDS_REVISION.value
        elif outcome == CampaignRunOutcome.NEEDS_HUMAN.value:
            project.status = CampaignProjectStatus.BLOCKED.value
        elif outcome in {CampaignRunOutcome.TIMEOUT.value, CampaignRunOutcome.CRASH.value}:
            project.status = CampaignProjectStatus.FAILED.value
        else:
            project.status = CampaignProjectStatus.BLOCKED.value

        if project.retry_count > manifest.max_retries_per_project and project.status in {
            CampaignProjectStatus.NEEDS_REVISION.value,
            CampaignProjectStatus.FAILED.value,
        }:
            project.status = CampaignProjectStatus.SKIPPED.value

        if project.status in _TERMINAL_STATUSES:
            self._emit_receipt(manifest, project, run_dict)

    def _apply_review_result(
        self,
        manifest: CampaignManifest,
        project: CampaignProject,
        gate: CampaignReviewGate,
        run_dict: dict[str, Any] | None = None,
    ) -> None:
        project.review = gate

        if gate.status == CampaignReviewStatus.PASSED.value:
            project.status = CampaignProjectStatus.COMPLETED.value
        elif gate.status == CampaignReviewStatus.CHANGES_REQUESTED.value:
            project.status = CampaignProjectStatus.NEEDS_REVISION.value
        else:
            project.status = CampaignProjectStatus.BLOCKED.value

        if project.status in _TERMINAL_STATUSES:
            self._emit_receipt(manifest, project, run_dict)

    def _spec_for_retry(self, project: CampaignProject) -> SwarmSpec:
        spec = SwarmSpec.from_dict(project.spec.to_dict())
        if project.review.findings:
            extra_constraints = [
                f"Address prior review finding: {finding}" for finding in project.review.findings
            ]
            spec.constraints = list(dict.fromkeys(list(spec.constraints) + extra_constraints))
        return spec

    def _reconcile_active_projects(self, manifest: CampaignManifest) -> None:
        for project in manifest.projects:
            if project.status != CampaignProjectStatus.ACTIVE.value or not project.run_id:
                continue
            run_dict = self._refresh_run_dict(project.run_id)
            if not run_dict:
                continue
            # Only classify runs that have reached a terminal status.
            # _classify_terminal_run_outcome falls back to "blocked" for
            # unknown statuses (including "running"), which would
            # incorrectly transition in-flight projects out of ACTIVE.
            run_status = str(run_dict.get("status", "")).strip().lower()
            if run_status in {"running", "in_progress", "pending", "queued", ""}:
                continue
            outcome = _classify_terminal_run_outcome(run_dict)
            if outcome in {
                CampaignRunOutcome.DELIVERABLE_CREATED.value,
                CampaignRunOutcome.PR_ADOPTED.value,
                CampaignRunOutcome.CLEAN_EXIT_NO_DELIVERABLE.value,
                CampaignRunOutcome.NEEDS_HUMAN.value,
                CampaignRunOutcome.TIMEOUT.value,
                CampaignRunOutcome.CRASH.value,
                CampaignRunOutcome.BLOCKED.value,
            }:
                self._apply_dispatch_result(
                    manifest,
                    project,
                    {
                        "run": run_dict,
                        "run_id": run_dict.get("run_id"),
                        "outcome": outcome,
                        "deliverable": _extract_deliverable(run_dict),
                    },
                )

    def _ready_projects(self, manifest: CampaignManifest) -> list[CampaignProject]:
        completed = {
            project.project_id
            for project in manifest.projects
            if project.status == CampaignProjectStatus.COMPLETED.value
        }
        ready: list[CampaignProject] = []
        for project in manifest.projects:
            if project.status not in {
                CampaignProjectStatus.PENDING.value,
                CampaignProjectStatus.READY.value,
                CampaignProjectStatus.NEEDS_REVISION.value,
            }:
                continue
            if project.retry_count > manifest.max_retries_per_project:
                project.status = CampaignProjectStatus.SKIPPED.value
                self._emit_receipt(manifest, project, None)
                continue
            deps = {dep.project_id for dep in project.dependencies}
            if deps.issubset(completed):
                if project.status == CampaignProjectStatus.PENDING.value:
                    project.status = CampaignProjectStatus.READY.value
                ready.append(project)
        return ready

    def _refresh_run_dict(self, run_id: str) -> dict[str, Any] | None:
        supervisor = SwarmSupervisor(repo_root=self.repo_root)
        try:
            return supervisor.refresh_run(run_id).to_dict()
        except Exception:
            record = supervisor.store.get_supervisor_run(run_id)
            return dict(record) if isinstance(record, dict) else None

    def _emit_receipt(
        self,
        manifest: CampaignManifest,
        project: CampaignProject,
        run_dict: dict[str, Any] | None,
    ) -> Path:
        """Write an authoritative receipt file at terminal project transition.

        Receipts are written to docs/receipts/<campaign_id>/<project_id>.yaml
        atomically (temp file + replace). Raises RuntimeError on failure so
        the caller knows the receipt was not persisted.
        """
        receipt_dir = self.repo_root / "docs" / "receipts" / manifest.campaign_id
        receipt_path = receipt_dir / f"{project.project_id}.yaml"
        try:
            try:
                manifest_input = str(self.manifest_path.relative_to(self.repo_root))
            except ValueError:
                manifest_input = str(self.manifest_path)

            payload: dict[str, Any] = {
                "task_id": project.project_id,
                "campaign_id": manifest.campaign_id,
                "phase": _derive_phase(manifest.campaign_id),
                "manifest_input": manifest_input,
                "worker_branch": _worker_branch_from_run(project, run_dict),
                "worker_commit": _worker_commit_from_run(project, run_dict),
                "changed_files": _changed_files_from_run(run_dict),
                "review_verdict": _receipt_review_verdict(project.review.status),
                "verification_results": {
                    "pytest_exit_code": None,
                    "syntax_check": None,
                    "truth_suite": None,
                },
                "final_status": _receipt_final_status(project.status),
                "failure_classification": _failure_classification_from_outcome(
                    project.last_run_outcome
                ),
                "rescue_required": False,
                "rescue_description": None,
                "cost_usd": project.estimated_cost_usd,
                "duration_seconds": _duration_seconds_from_run(run_dict),
                "created_at": datetime.now(UTC).isoformat(),
            }

            try:
                import yaml

                content = yaml.safe_dump(payload, sort_keys=True, allow_unicode=False)
            except ImportError:
                content = json.dumps(payload, indent=2, sort_keys=True)

            receipt_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = receipt_path.with_suffix(".yaml.tmp")
            tmp_path.write_text(content, encoding="utf-8")
            tmp_path.replace(receipt_path)

            project.receipt_id = str(receipt_path.relative_to(self.repo_root))
            logger.info(
                "receipt_emitted: project=%s status=%s path=%s",
                project.project_id,
                project.status,
                project.receipt_id,
            )
            return receipt_path

        except Exception as exc:
            logger.error(
                "receipt_emit_failed: project=%s status=%s error=%s",
                project.project_id,
                project.status,
                exc,
            )
            raise RuntimeError(f"Failed to emit receipt for {project.project_id}: {exc}") from exc


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _first_receipt_id(run_dict: dict[str, Any]) -> str | None:
    for item in run_dict.get("work_orders", []):
        if not isinstance(item, dict):
            continue
        text = str(item.get("receipt_id", "")).strip()
        if text:
            return text
    return None


def _refresh_execution_state(manifest: CampaignManifest) -> None:
    manifest.execution_state.ready_queue = [
        project.project_id
        for project in manifest.projects
        if project.status == CampaignProjectStatus.READY.value
    ]
    manifest.execution_state.active_projects = [
        project.project_id
        for project in manifest.projects
        if project.status == CampaignProjectStatus.ACTIVE.value
    ]
    manifest.execution_state.completed_projects = [
        project.project_id
        for project in manifest.projects
        if project.status == CampaignProjectStatus.COMPLETED.value
    ]
    manifest.execution_state.failed_projects = [
        project.project_id
        for project in manifest.projects
        if project.status
        in {CampaignProjectStatus.FAILED.value, CampaignProjectStatus.BLOCKED.value}
    ]
    manifest.execution_state.skipped_projects = [
        project.project_id
        for project in manifest.projects
        if project.status == CampaignProjectStatus.SKIPPED.value
    ]


def _manifest_summary(manifest: CampaignManifest) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for project in manifest.projects:
        counts[project.status] = counts.get(project.status, 0) + 1
    return {
        "mode": "campaign-status",
        "campaign_id": manifest.campaign_id,
        "source_kind": manifest.source_kind,
        "source_ref": manifest.source_ref,
        "planner_model": manifest.planner_model,
        "worker_model": manifest.worker_model,
        "review_model": manifest.review_model,
        "budget_limit_usd": manifest.budget_limit_usd,
        "total_cost_usd": manifest.execution_state.total_cost_usd,
        "counts": counts,
        "stop_reason": _compute_stop_reason(manifest),
        "projects": [
            {
                "project_id": project.project_id,
                "title": project.title,
                "status": project.status,
                "retry_count": project.retry_count,
                "run_id": project.run_id,
                "receipt_id": project.receipt_id,
                "pr_url": project.pr_url,
                "adopted_pr": project.adopted_pr,
                "review_status": project.review.status,
                "dependencies": [dep.to_dict() for dep in project.dependencies],
            }
            for project in manifest.projects
        ],
    }


def _compute_stop_reason(manifest: CampaignManifest) -> str:
    if manifest.execution_state.total_cost_usd >= manifest.budget_limit_usd:
        return CampaignStopReason.BUDGET_EXHAUSTED.value
    started_at = _parse_dt(manifest.execution_state.last_run_at)
    if started_at and manifest.time_limit_hours > 0:
        elapsed_hours = (datetime.now(UTC) - started_at).total_seconds() / 3600.0
        if elapsed_hours >= manifest.time_limit_hours:
            return CampaignStopReason.TIME_LIMIT_EXCEEDED.value
    statuses = {project.status for project in manifest.projects}
    if statuses and statuses.issubset(
        {
            CampaignProjectStatus.COMPLETED.value,
            CampaignProjectStatus.SKIPPED.value,
        }
    ):
        return CampaignStopReason.CAMPAIGN_COMPLETE.value
    blocked_like = {
        CampaignProjectStatus.BLOCKED.value,
        CampaignProjectStatus.FAILED.value,
        CampaignProjectStatus.SKIPPED.value,
    }
    if statuses and statuses.issubset(blocked_like | {CampaignProjectStatus.COMPLETED.value}):
        return CampaignStopReason.CAMPAIGN_BLOCKED.value
    # Check for unreachable pending projects: if every non-terminal project
    # has at least one dependency in a terminal-but-not-completed state, the
    # campaign is effectively blocked even though raw statuses include pending.
    terminal_not_completed = blocked_like
    active_statuses = {
        CampaignProjectStatus.ACTIVE.value,
        CampaignProjectStatus.DELIVERED.value,
    }
    if not statuses & active_statuses:
        project_status_map = {p.project_id: p.status for p in manifest.projects}
        reachable = [
            p
            for p in manifest.projects
            if p.status
            in {
                CampaignProjectStatus.PENDING.value,
                CampaignProjectStatus.READY.value,
                CampaignProjectStatus.NEEDS_REVISION.value,
            }
        ]
        if reachable and all(
            any(
                project_status_map.get(d.project_id) in terminal_not_completed
                for d in p.dependencies
            )
            for p in reachable
            if p.dependencies
        ):
            # Every remaining non-terminal project with dependencies has at least
            # one dependency that will never complete.  If there are also no
            # dependency-free pending projects that could still make progress,
            # the campaign is blocked.
            dependency_free = [p for p in reachable if not p.dependencies]
            if not dependency_free:
                return CampaignStopReason.CAMPAIGN_BLOCKED.value
    return CampaignStopReason.STILL_RUNNING.value


def _extract_first_json_object(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, ValueError):
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _extract_json_list(text: str) -> list[str]:
    payload = _extract_first_json_object(text)
    findings = payload.get("findings")
    if isinstance(findings, list):
        return [str(item) for item in findings if str(item).strip()]
    return []


# ---------------------------------------------------------------------------
# Receipt emission helpers
# ---------------------------------------------------------------------------


def _derive_phase(campaign_id: str) -> str | None:
    cid = str(campaign_id or "").lower()
    for prefix, label in (
        ("phase0a", "0a"),
        ("phase0b", "0b"),
        ("phase1", "1"),
        ("phase2", "2"),
    ):
        if cid.startswith(prefix):
            return label
    return None


def _failure_classification_from_outcome(outcome: str | None) -> str | None:
    mapping = {
        CampaignRunOutcome.CRASH.value: "worker_crash",
        CampaignRunOutcome.TIMEOUT.value: "timeout",
        CampaignRunOutcome.BLOCKED.value: "stall",
        CampaignRunOutcome.NEEDS_HUMAN.value: "stall",
        CampaignRunOutcome.CLEAN_EXIT_NO_DELIVERABLE.value: "stall",
    }
    return mapping.get(str(outcome or ""))


def _receipt_final_status(project_status: str) -> str:
    return {
        "completed": "completed",
        "failed": "failed",
        "skipped": "failed",
        "blocked": "rejected",
    }.get(project_status, "failed")


def _receipt_review_verdict(review_status: str) -> str:
    return {
        CampaignReviewStatus.PASSED.value: "passed",
        CampaignReviewStatus.CHANGES_REQUESTED.value: "failed",
        CampaignReviewStatus.BLOCKED_NONREVIEWABLE.value: "failed",
        CampaignReviewStatus.PENDING.value: "skipped",
    }.get(review_status, "skipped")


def _duration_seconds_from_run(run_dict: dict[str, Any] | None) -> int | None:
    if not run_dict:
        return None
    for wo in run_dict.get("work_orders", []):
        if not isinstance(wo, dict):
            continue
        started = str(wo.get("started_at", "")).strip()
        ended = str(wo.get("completed_at", "")).strip()
        if started and ended:
            try:
                t_start = datetime.fromisoformat(started)
                t_end = datetime.fromisoformat(ended)
                return max(0, int((t_end - t_start).total_seconds()))
            except (ValueError, TypeError):
                continue
    return None


def _changed_files_from_run(run_dict: dict[str, Any] | None) -> list[str]:
    if not run_dict:
        return []
    paths: list[str] = []
    seen: set[str] = set()
    for wo in run_dict.get("work_orders", []):
        if not isinstance(wo, dict):
            continue
        for p in wo.get("changed_paths", []):
            text = str(p).strip()
            if text and text not in seen:
                seen.add(text)
                paths.append(text)
    return paths


def _worker_branch_from_run(
    project: CampaignProject, run_dict: dict[str, Any] | None
) -> str | None:
    if project.branch:
        return project.branch
    if not run_dict:
        return None
    for wo in run_dict.get("work_orders", []):
        if isinstance(wo, dict):
            branch = str(wo.get("branch", "")).strip()
            if branch:
                return branch
    return None


def _worker_commit_from_run(
    project: CampaignProject, run_dict: dict[str, Any] | None
) -> str | None:
    if project.commit_shas:
        return project.commit_shas[-1]
    if not run_dict:
        return None
    for wo in run_dict.get("work_orders", []):
        if not isinstance(wo, dict):
            continue
        sha = str(wo.get("head_sha", "")).strip()
        if sha:
            return sha
        shas = [str(s).strip() for s in wo.get("commit_shas", []) if str(s).strip()]
        if shas:
            return shas[-1]
    return None
