"""Manifest-driven tranche inspection, planning, and bounded lane execution."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aragora.nomic.dev_coordination import DevCoordinationStore, WorkLease
from aragora.swarm.campaign import locked_manifest_path

UTC = timezone.utc
DEFAULT_TRANCHE_ARTIFACT_ROOT = ".aragora/tranche_artifacts"
DEFAULT_TRANCHE_MANIFEST_DIR = ".aragora/tranches"
_GITHUB_REF_RE = re.compile(r"^https://github\.com/([^/]+)/([^/]+)/(pull|issues)/(\d+)$")
_REUSABLE_PREPARED_STATUSES = {"prepared", "running"}
_TRUTHY_BOOL_STRINGS = {"1", "true", "yes", "y", "on"}
_FALSY_BOOL_STRINGS = {"0", "false", "no", "n", "off"}


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list.")
    return [str(item).strip() for item in value if str(item).strip()]


def _dict_value(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object.")
    return dict(value)


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUTHY_BOOL_STRINGS:
            return True
        if normalized in _FALSY_BOOL_STRINGS:
            return False
    return default


def _normalize_scope(value: str) -> str:
    return value.replace("\\", "/").strip().strip("/")


def _scope_prefix(value: str) -> str:
    normalized = _normalize_scope(value)
    if normalized.endswith("/**"):
        normalized = normalized[:-3]
    return normalized.rstrip("/")


def _scope_patterns_overlap(left: str, right: str) -> bool:
    left_norm = _normalize_scope(left)
    right_norm = _normalize_scope(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    left_recursive = left_norm.endswith("/**")
    right_recursive = right_norm.endswith("/**")
    left_prefix = _scope_prefix(left_norm)
    right_prefix = _scope_prefix(right_norm)
    if left_recursive and (
        right_prefix == left_prefix or right_prefix.startswith(left_prefix + "/")
    ):
        return True
    if right_recursive and (
        left_prefix == right_prefix or left_prefix.startswith(right_prefix + "/")
    ):
        return True
    if left_prefix == right_prefix:
        return True
    return False


def _paths_overlap(left: list[str], right: list[str]) -> bool:
    return any(
        _scope_patterns_overlap(left_item, right_item) for left_item in left for right_item in right
    )


@dataclass(slots=True)
class TrancheReference:
    kind: str
    url: str
    state: str = ""
    meaning: str = ""
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "kind": self.kind,
            "url": self.url,
            "state": self.state,
            "meaning": self.meaning,
        }
        if self.label:
            payload["label"] = self.label
        payload.update(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrancheReference:
        kind = str(data.get("kind", "")).strip()
        url = str(data.get("url", "")).strip()
        if not kind:
            raise ValueError("Tranche reference kind is required.")
        if not url:
            raise ValueError("Tranche reference url is required.")
        metadata = {
            key: value
            for key, value in data.items()
            if key not in {"kind", "url", "state", "meaning", "label"}
        }
        return cls(
            kind=kind,
            url=url,
            state=str(data.get("state", "")).strip(),
            meaning=str(data.get("meaning", "")).strip(),
            label=_optional_text(data.get("label")),
            metadata=metadata,
        )


@dataclass(slots=True)
class TrancheGate:
    source_ref: str
    state: str
    required_for: list[str] = field(default_factory=list)
    satisfy_when: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "source_ref": self.source_ref,
            "state": self.state,
            "required_for": list(self.required_for),
            "satisfy_when": self.satisfy_when,
        }
        payload.update(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrancheGate:
        source_ref = str(data.get("source_ref", "")).strip()
        state = str(data.get("state", "")).strip()
        if not source_ref:
            raise ValueError("Tranche gate source_ref is required.")
        if not state:
            raise ValueError("Tranche gate state is required.")
        metadata = {
            key: value
            for key, value in data.items()
            if key not in {"source_ref", "state", "required_for", "satisfy_when"}
        }
        return cls(
            source_ref=source_ref,
            state=state,
            required_for=_string_list(data.get("required_for"), field_name="required_for"),
            satisfy_when=str(data.get("satisfy_when", "")).strip(),
            metadata=metadata,
        )


@dataclass(slots=True)
class TrancheLane:
    lane_id: str
    owner_role: str
    branch: dict[str, Any] = field(default_factory=dict)
    worktree: dict[str, Any] = field(default_factory=dict)
    allowed_write_scope: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    verification_commands: list[str] = field(default_factory=list)
    stop_conditions: list[str] = field(default_factory=list)
    expected_receipts_artifacts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "lane_id": self.lane_id,
            "owner_role": self.owner_role,
            "branch": dict(self.branch),
            "worktree": dict(self.worktree),
            "allowed_write_scope": list(self.allowed_write_scope),
            "dependencies": list(self.dependencies),
            "verification_commands": list(self.verification_commands),
            "stop_conditions": list(self.stop_conditions),
            "expected_receipts_artifacts": list(self.expected_receipts_artifacts),
        }
        payload.update(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrancheLane:
        lane_id = str(data.get("lane_id", "")).strip()
        owner_role = str(data.get("owner_role", "")).strip()
        if not lane_id:
            raise ValueError("Tranche lane lane_id is required.")
        if not owner_role:
            raise ValueError("Tranche lane owner_role is required.")
        metadata = {
            key: value
            for key, value in data.items()
            if key
            not in {
                "lane_id",
                "owner_role",
                "branch",
                "worktree",
                "allowed_write_scope",
                "dependencies",
                "verification_commands",
                "stop_conditions",
                "expected_receipts_artifacts",
            }
        }
        return cls(
            lane_id=lane_id,
            owner_role=owner_role,
            branch=_dict_value(data.get("branch"), field_name="branch"),
            worktree=_dict_value(data.get("worktree"), field_name="worktree"),
            allowed_write_scope=_string_list(
                data.get("allowed_write_scope"), field_name="allowed_write_scope"
            ),
            dependencies=_string_list(data.get("dependencies"), field_name="dependencies"),
            verification_commands=_string_list(
                data.get("verification_commands"), field_name="verification_commands"
            ),
            stop_conditions=_string_list(data.get("stop_conditions"), field_name="stop_conditions"),
            expected_receipts_artifacts=_string_list(
                data.get("expected_receipts_artifacts"),
                field_name="expected_receipts_artifacts",
            ),
            metadata=metadata,
        )

    @property
    def claimable(self) -> bool:
        return bool(self.allowed_write_scope)


@dataclass(slots=True)
class TrancheManifest:
    manifest_id: str
    repo: dict[str, Any]
    references: dict[str, dict[str, TrancheReference]]
    gates: dict[str, TrancheGate]
    lanes: list[TrancheLane]
    terminal_outcomes: dict[str, Any]
    manifest_version: int = 1
    generated_on: str | None = None
    objective: str = ""
    shared_constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "manifest_id": self.manifest_id,
            "generated_on": self.generated_on,
            "repo": dict(self.repo),
            "objective": self.objective,
            "shared_constraints": dict(self.shared_constraints),
            "references": {
                group: {ref_id: ref.to_dict() for ref_id, ref in refs.items()}
                for group, refs in self.references.items()
            },
            "gates": {gate_id: gate.to_dict() for gate_id, gate in self.gates.items()},
            "lanes": [lane.to_dict() for lane in self.lanes],
            "terminal_outcomes": dict(self.terminal_outcomes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrancheManifest:
        manifest_id = str(data.get("manifest_id", "")).strip()
        if not manifest_id:
            raise ValueError("Tranche manifest manifest_id is required.")
        repo = _dict_value(data.get("repo"), field_name="repo")
        if not repo:
            raise ValueError("Tranche manifest repo is required.")
        references_raw = _dict_value(data.get("references"), field_name="references")
        gates_raw = _dict_value(data.get("gates"), field_name="gates")
        lanes_raw = data.get("lanes")
        if not isinstance(lanes_raw, list):
            raise ValueError("Tranche manifest lanes must be a list.")
        terminal_outcomes = _dict_value(
            data.get("terminal_outcomes"), field_name="terminal_outcomes"
        )
        if not terminal_outcomes:
            raise ValueError("Tranche manifest terminal_outcomes is required.")
        references: dict[str, dict[str, TrancheReference]] = {}
        for group, items in references_raw.items():
            if not isinstance(items, dict):
                raise ValueError(f"Tranche references group {group!r} must be an object.")
            references[str(group)] = {
                str(ref_id): TrancheReference.from_dict(dict(ref_data))
                for ref_id, ref_data in items.items()
                if isinstance(ref_data, dict)
            }
        gates = {
            str(gate_id): TrancheGate.from_dict(dict(gate_data))
            for gate_id, gate_data in gates_raw.items()
            if isinstance(gate_data, dict)
        }
        lanes = [TrancheLane.from_dict(dict(item)) for item in lanes_raw if isinstance(item, dict)]
        return cls(
            manifest_version=int(data.get("manifest_version", 1) or 1),
            manifest_id=manifest_id,
            generated_on=_optional_text(data.get("generated_on")),
            repo=repo,
            objective=str(data.get("objective", "")).strip(),
            shared_constraints=_dict_value(
                data.get("shared_constraints"), field_name="shared_constraints"
            ),
            references=references,
            gates=gates,
            lanes=lanes,
            terminal_outcomes=terminal_outcomes,
        )

    def reference(self, ref_id: str) -> tuple[str, TrancheReference] | None:
        for group, refs in self.references.items():
            if ref_id in refs:
                return group, refs[ref_id]
        return None

    def lane(self, lane_id: str) -> TrancheLane:
        for lane in self.lanes:
            if lane.lane_id == lane_id:
                return lane
        raise KeyError(f"Unknown tranche lane: {lane_id}")

    def to_yaml(self) -> str:
        try:
            import yaml

            return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=False)
        except ImportError:
            return json.dumps(self.to_dict(), indent=2, sort_keys=False)

    @classmethod
    def from_text(cls, text: str) -> TrancheManifest:
        try:
            import yaml

            payload = yaml.safe_load(text) or {}
        except ImportError:
            payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("Tranche manifest must deserialize to an object.")
        return cls.from_dict(payload)


@dataclass(slots=True)
class TrancheLaneArtifact:
    lane_id: str
    source_ref: str
    status: str
    commands: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    run_id: str | None = None
    worktree_path: str | None = None
    residual_risk: str = ""
    timestamp: str = field(default_factory=lambda: _utcnow().isoformat())
    next_actions: list[str] = field(default_factory=list)
    blocked_reason: str | None = None
    blocking_question: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def set_blocker(self, *, reason: Any = None, question: Any = None) -> None:
        self.blocked_reason = _optional_text(reason)
        self.blocking_question = _optional_text(question)
        metadata = dict(self.metadata) if isinstance(self.metadata, dict) else {}
        if self.blocked_reason or self.blocking_question:
            blocker: dict[str, Any] = {}
            if self.blocked_reason:
                blocker["reason"] = self.blocked_reason
            if self.blocking_question:
                blocker["question"] = self.blocking_question
            metadata["blocker"] = blocker
        else:
            metadata.pop("blocker", None)
        self.metadata = metadata

    def clear_blocker(self) -> None:
        self.set_blocker()

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "source_ref": self.source_ref,
            "status": self.status,
            "commands": list(self.commands),
            "urls": list(self.urls),
            "run_id": self.run_id,
            "worktree_path": self.worktree_path,
            "residual_risk": self.residual_risk,
            "timestamp": self.timestamp,
            "next_actions": list(self.next_actions),
            "blocked_reason": self.blocked_reason,
            "blocking_question": self.blocking_question,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrancheLaneArtifact:
        lane_id = str(data.get("lane_id", "")).strip()
        source_ref = str(data.get("source_ref", "")).strip()
        status = str(data.get("status", "")).strip()
        if not lane_id:
            raise ValueError("Tranche lane artifact lane_id is required.")
        if not source_ref:
            raise ValueError("Tranche lane artifact source_ref is required.")
        if not status:
            raise ValueError("Tranche lane artifact status is required.")
        metadata = _dict_value(data.get("metadata"), field_name="metadata")
        blocker = metadata.get("blocker", {})
        if not isinstance(blocker, dict):
            blocker = {}
        return cls(
            lane_id=lane_id,
            source_ref=source_ref,
            status=status,
            commands=_string_list(data.get("commands"), field_name="commands"),
            urls=_string_list(data.get("urls"), field_name="urls"),
            run_id=_optional_text(data.get("run_id")),
            worktree_path=_optional_text(data.get("worktree_path")),
            residual_risk=str(data.get("residual_risk", "")).strip(),
            timestamp=str(data.get("timestamp", "")).strip() or _utcnow().isoformat(),
            next_actions=_string_list(data.get("next_actions"), field_name="next_actions"),
            blocked_reason=_optional_text(data.get("blocked_reason"))
            or _optional_text(blocker.get("reason")),
            blocking_question=_optional_text(data.get("blocking_question"))
            or _optional_text(blocker.get("question")),
            metadata=metadata,
        )


class TrancheArtifactStore:
    """File-backed lane artifact store for tranche proofs and runbooks."""

    def __init__(self, repo_root: Path, root: str | Path = DEFAULT_TRANCHE_ARTIFACT_ROOT) -> None:
        self.repo_root = repo_root.resolve()
        root_path = Path(root)
        self.root = root_path if root_path.is_absolute() else (self.repo_root / root_path).resolve()

    def path_for(self, manifest_id: str, lane_id: str) -> Path:
        return self.root / manifest_id / f"{lane_id}.yaml"

    def save(self, manifest_id: str, artifact: TrancheLaneArtifact) -> Path:
        path = self.path_for(manifest_id, artifact.lane_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with locked_manifest_path(path):
            path.write_text(_dump_yaml_like(artifact.to_dict()), encoding="utf-8")
        return path

    def load(self, manifest_id: str, lane_id: str) -> TrancheLaneArtifact | None:
        path = self.path_for(manifest_id, lane_id)
        if not path.exists():
            return None
        return TrancheLaneArtifact.from_dict(_load_yaml_like(path))

    def list(self, manifest_id: str) -> list[TrancheLaneArtifact]:
        root = self.root / manifest_id
        if not root.exists():
            return []
        artifacts = [
            TrancheLaneArtifact.from_dict(_load_yaml_like(path))
            for path in sorted(root.glob("*.yaml"))
        ]
        return sorted(artifacts, key=lambda item: item.timestamp, reverse=True)


class TranchePlanner:
    """Compile prompt bundles into tranche manifests."""

    def __init__(self, *, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()

    def default_manifest_path(self, manifest_id: str) -> Path:
        return (
            self.repo_root / DEFAULT_TRANCHE_MANIFEST_DIR / manifest_id / "tranche.yaml"
        ).resolve()

    def plan_from_prompt_bundle(
        self,
        bundle_path: Path,
        *,
        output_path: Path | None = None,
    ) -> tuple[TrancheManifest, Path]:
        payload = _load_yaml_like(bundle_path)
        manifest = TrancheManifest.from_dict(
            _prompt_bundle_to_manifest_dict(payload, repo_root=self.repo_root)
        )
        destination = (output_path or self.default_manifest_path(manifest.manifest_id)).resolve()
        save_tranche_manifest(destination, manifest)
        return manifest, destination


class TrancheExecutor:
    """Prepare and run claimable tranche lanes through the bounded boss path."""

    def __init__(
        self,
        *,
        repo_root: Path,
        artifact_store: TrancheArtifactStore | None = None,
        reference_client: GhReferenceClient | None = None,
        reviewer: Any | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.artifact_store = artifact_store or TrancheArtifactStore(self.repo_root)
        self.reference_client = reference_client or GhReferenceClient()
        self.reviewer = reviewer

    def prepare(
        self,
        manifest: TrancheManifest,
        *,
        lane_id: str = "",
        all_ready: bool = False,
        owner_agent: str | None = None,
        owner_session_id: str | None = None,
        base_branch: str | None = None,
    ) -> dict[str, Any]:
        inspector = TrancheInspector(
            repo_root=self.repo_root,
            reference_client=self.reference_client,
            artifact_store=self.artifact_store,
        )
        inspection = inspector.inspect(manifest)
        lanes = _select_claimable_lanes(
            manifest,
            inspection=inspection,
            lane_id=lane_id,
            all_ready=all_ready,
        )
        prepared: list[dict[str, Any]] = []
        for lane_payload in lanes:
            lane = manifest.lane(str(lane_payload["lane_id"]))
            artifact = self._prepare_lane_workspace(
                manifest,
                lane,
                owner_agent=owner_agent,
                owner_session_id=owner_session_id,
                base_branch=base_branch,
            )
            self.artifact_store.save(manifest.manifest_id, artifact)
            prepared.append(artifact.to_dict())
        return {
            "mode": "tranche-prepare",
            "manifest_id": manifest.manifest_id,
            "prepared_lanes": prepared,
        }

    async def run(
        self,
        manifest: TrancheManifest,
        *,
        lane_id: str = "",
        all_ready: bool = False,
        owner_agent: str | None = None,
        owner_session_id: str | None = None,
        target_branch: str = "main",
        max_ticks: int = 360,
        wait_for_completion: bool = True,
        skip_review: bool = False,
    ) -> dict[str, Any]:
        from aragora.swarm.boss_loop import dispatch_bounded_spec

        inspector = TrancheInspector(
            repo_root=self.repo_root,
            reference_client=self.reference_client,
            artifact_store=self.artifact_store,
        )
        inspection = inspector.inspect(manifest)
        lanes = _select_claimable_lanes(
            manifest,
            inspection=inspection,
            lane_id=lane_id,
            all_ready=all_ready,
        )
        results: list[dict[str, Any]] = []
        for lane_payload in lanes:
            lane = manifest.lane(str(lane_payload["lane_id"]))
            prepared = self.artifact_store.load(manifest.manifest_id, lane.lane_id)
            if (
                prepared is None
                or not prepared.worktree_path
                or prepared.status not in _REUSABLE_PREPARED_STATUSES
            ):
                prepared = self._prepare_lane_workspace(
                    manifest,
                    lane,
                    owner_agent=owner_agent,
                    owner_session_id=owner_session_id,
                    base_branch=target_branch,
                )
            spec = _lane_spec_from_manifest(manifest, lane)
            target_agent = _lane_target_agent(lane, fallback=owner_agent or "codex")
            review_model = _lane_review_model(lane, target_agent=target_agent)
            result = await dispatch_bounded_spec(
                spec,
                target_branch=target_branch,
                budget_limit_usd=_lane_budget_limit_usd(lane),
                max_ticks=max(1, int(max_ticks)),
                wait_for_completion=wait_for_completion,
                repo_path=self.repo_root,
                default_target_agent=target_agent,
                default_reviewer_agent=review_model,
                use_managed_session_script=bool(
                    lane.metadata.get("use_managed_session_script", True)
                ),
            )
            artifact = await self._artifact_from_run_result(
                manifest,
                lane,
                prepared=prepared,
                result=result,
                review_model=review_model,
                skip_review=skip_review,
            )
            self.artifact_store.save(manifest.manifest_id, artifact)
            results.append(
                {
                    "lane_id": lane.lane_id,
                    "status": artifact.status,
                    "run_id": artifact.run_id,
                    "worktree_path": artifact.worktree_path,
                    "urls": list(artifact.urls),
                    "next_actions": list(artifact.next_actions),
                    "blocked_reason": artifact.blocked_reason,
                    "blocking_question": artifact.blocking_question,
                    "metadata": dict(artifact.metadata),
                }
            )
        return {
            "mode": "tranche-run",
            "manifest_id": manifest.manifest_id,
            "wait_for_completion": wait_for_completion,
            "results": results,
        }

    def _prepare_lane_workspace(
        self,
        manifest: TrancheManifest,
        lane: TrancheLane,
        *,
        owner_agent: str | None,
        owner_session_id: str | None,
        base_branch: str | None,
    ) -> TrancheLaneArtifact:
        base = _manifest_base_branch(manifest, fallback=base_branch or "main")
        target_agent = _lane_target_agent(lane, fallback=owner_agent or "codex")
        ensure_cmd = [
            "python3",
            str(self.repo_root / "scripts" / "codex_worktree_autopilot.py"),
            "ensure",
            "--agent",
            target_agent,
            "--base",
            base,
            "--force-new",
            "--reconcile",
            "--print-path",
        ]
        ensure_proc = subprocess.run(
            ensure_cmd,
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if ensure_proc.returncode != 0:
            raise RuntimeError(
                ensure_proc.stderr.strip()
                or ensure_proc.stdout.strip()
                or f"Failed to prepare worktree for lane {lane.lane_id}"
            )
        worktree_path = str(ensure_proc.stdout.strip())
        if not worktree_path:
            raise RuntimeError(f"Worktree autopilot returned no path for lane {lane.lane_id}")
        branch = _lane_branch_name(manifest, lane, target_agent=target_agent)
        branch_cmd = ["git", "switch", "-C", branch, f"origin/{base}"]
        branch_proc = subprocess.run(
            branch_cmd,
            cwd=worktree_path,
            text=True,
            capture_output=True,
            check=False,
        )
        if branch_proc.returncode != 0:
            raise RuntimeError(
                branch_proc.stderr.strip()
                or branch_proc.stdout.strip()
                or f"Failed to switch worktree branch for lane {lane.lane_id}"
            )
        return TrancheLaneArtifact(
            lane_id=lane.lane_id,
            source_ref=_lane_primary_source_ref(manifest, lane),
            status="prepared",
            commands=[shlex.join(ensure_cmd), shlex.join(branch_cmd)],
            urls=_lane_source_urls(lane),
            worktree_path=worktree_path,
            residual_risk="Workspace prepared; execution and review are still pending.",
            next_actions=[f"Run tranche lane {lane.lane_id} when ready."],
            metadata={
                "branch": branch,
                "target_agent": target_agent,
                "owner_agent": owner_agent or target_agent,
                "owner_session_id": owner_session_id or Path(worktree_path).name,
                "prepared_scope": list(lane.allowed_write_scope),
                "controller_worktree_path": worktree_path,
            },
        )

    async def _artifact_from_run_result(
        self,
        manifest: TrancheManifest,
        lane: TrancheLane,
        *,
        prepared: TrancheLaneArtifact,
        result: dict[str, Any],
        review_model: str,
        skip_review: bool,
    ) -> TrancheLaneArtifact:
        run_dict = result.get("run") if isinstance(result.get("run"), dict) else {}
        deliverable = (
            result.get("deliverable") if isinstance(result.get("deliverable"), dict) else {}
        )
        urls = list(dict.fromkeys(_lane_source_urls(lane) + _deliverable_urls(deliverable)))
        metadata = dict(prepared.metadata)
        metadata.update(
            {
                "result_status": str(result.get("status", "")).strip(),
                "result_outcome": str(result.get("outcome", "")).strip(),
            }
        )
        if run_dict:
            metadata["run_status"] = str(run_dict.get("status", "")).strip()
            for key in ("receipt_id", "lease_id"):
                value = _first_work_order_text(run_dict, key)
                if value:
                    metadata[key] = value
        worker_worktree_path = _first_worker_worktree_path(run_dict) or prepared.worktree_path
        artifact = TrancheLaneArtifact(
            lane_id=lane.lane_id,
            source_ref=_lane_primary_source_ref(manifest, lane),
            status=_artifact_status_from_dispatch_result(result),
            commands=list(dict.fromkeys(prepared.commands + list(lane.verification_commands))),
            urls=urls,
            run_id=_optional_text(result.get("run_id")),
            worktree_path=worker_worktree_path,
            residual_risk=_residual_risk_from_dispatch_result(result),
            next_actions=_next_actions_from_dispatch_result(result),
            metadata=metadata,
        )
        if artifact.status == "needs_human":
            artifact.set_blocker(
                reason=_dispatch_result_blocked_reason(result, run_dict=run_dict),
                question=_dispatch_result_blocking_question(result, run_dict=run_dict),
            )
        else:
            artifact.clear_blocker()
        if run_dict:
            artifact.metadata["run"] = {
                "run_id": run_dict.get("run_id"),
                "status": run_dict.get("status"),
            }
        if deliverable:
            artifact.metadata["deliverable"] = dict(deliverable)
        if (
            not skip_review
            and artifact.status == "completed"
            and run_dict
            and isinstance(run_dict, dict)
        ):
            gate = await self._review_lane(
                manifest,
                lane,
                run_dict=run_dict,
                review_model=review_model,
            )
            artifact.metadata["review"] = gate.to_dict()
            artifact.next_actions = _review_next_actions(gate)
            artifact.residual_risk = _review_residual_risk(gate)
            artifact.status = _artifact_status_from_review(gate.status)
        return artifact

    async def _review_lane(
        self,
        manifest: TrancheManifest,
        lane: TrancheLane,
        *,
        run_dict: dict[str, Any],
        review_model: str,
    ) -> Any:
        from aragora.swarm.campaign import CampaignProject, CampaignReviewer

        reviewer = self.reviewer or CampaignReviewer()
        spec = _lane_spec_from_manifest(manifest, lane)
        project = CampaignProject(
            project_id=lane.lane_id,
            title=_lane_title(lane),
            source_refs=list(_lane_source_urls(lane)),
            spec=spec,
            file_scope_hints=list(spec.file_scope_hints),
            acceptance_criteria=list(spec.acceptance_criteria),
            constraints=list(spec.constraints),
        )
        worker_model = _lane_target_agent(lane, fallback="codex")
        return await reviewer.review(
            project=project,
            worker_model=worker_model,
            review_model=review_model,
            enforce_cross_model_review=_coerce_bool(
                lane.metadata.get("enforce_cross_model_review"),
                default=True,
            ),
            run_dict=dict(run_dict),
            budget_context={"project_estimated_cost_usd": _lane_budget_limit_usd(lane)},
            repo_root=self.repo_root,
            target_branch=_manifest_base_branch(manifest, fallback="main"),
        )


@dataclass(slots=True)
class GitHubReferenceTarget:
    owner: str
    repo: str
    kind: str
    number: int


class GhReferenceClient:
    """Minimal GitHub reference resolver backed by the gh CLI."""

    def _run_json(self, args: list[str]) -> dict[str, Any]:
        proc = subprocess.run(
            ["gh", *args],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "gh command failed")
        payload = json.loads(proc.stdout or "{}")
        if not isinstance(payload, dict):
            raise RuntimeError("gh command did not return a JSON object")
        return payload

    def get_issue(self, repo: str, number: int) -> dict[str, Any]:
        return self._run_json(
            [
                "issue",
                "view",
                str(number),
                "--repo",
                repo,
                "--json",
                "number,state,closedAt,title,url,labels",
            ]
        )

    def get_pr(self, repo: str, number: int) -> dict[str, Any]:
        return self._run_json(
            [
                "pr",
                "view",
                str(number),
                "--repo",
                repo,
                "--json",
                "number,state,mergedAt,title,url,mergeable,mergeStateStatus,reviewDecision,headRefName,baseRefName",
            ]
        )


class TrancheInspector:
    """Read-only tranche inspection against live GitHub and local scope rules."""

    def __init__(
        self,
        *,
        repo_root: Path,
        reference_client: GhReferenceClient | None = None,
        artifact_store: TrancheArtifactStore | None = None,
        coordination_store: DevCoordinationStore | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.reference_client = reference_client or GhReferenceClient()
        self.artifact_store = artifact_store or TrancheArtifactStore(self.repo_root)
        self.coordination_store = coordination_store

    def inspect(self, manifest: TrancheManifest) -> dict[str, Any]:
        references = self._resolve_references(manifest)
        gates = self._resolve_gates(manifest, references)
        scope_conflicts = self._declared_scope_conflicts(manifest)
        artifacts = self.artifact_store.list(manifest.manifest_id)
        lanes = self._resolve_lanes(manifest, references, gates, artifacts, scope_conflicts)
        blockers = []
        blockers.extend(
            item["reason"]
            for item in references.values()
            if item["group"] == "live_target" and item["status"] in {"blocked", "stale"}
        )
        blockers.extend(
            f"Declared write-scope overlap between {item['left_lane_id']} and {item['right_lane_id']}"
            for item in scope_conflicts
        )
        recommended = self._recommended_action(gates, lanes, blockers)
        return {
            "mode": "tranche-inspect",
            "generated_at": _utcnow().isoformat(),
            "manifest_id": manifest.manifest_id,
            "manifest_version": manifest.manifest_version,
            "repo": dict(manifest.repo),
            "preflight_status": "blocked" if blockers else "ok",
            "preflight_blockers": blockers,
            "references": references,
            "gates": gates,
            "lanes": lanes,
            "scope_conflicts": scope_conflicts,
            "artifacts": [item.to_dict() for item in artifacts],
            "recommended_action": recommended,
        }

    def prepare_lane_claim(
        self,
        manifest: TrancheManifest,
        *,
        lane_id: str,
        task_id: str,
        title: str,
        owner_agent: str,
        owner_session_id: str,
        branch: str,
        worktree_path: str,
        expected_tests: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkLease:
        lane = manifest.lane(lane_id)
        if not lane.claimable:
            raise ValueError(f"Lane {lane_id} is read-only and cannot claim a write lease.")
        store = self.coordination_store or DevCoordinationStore(repo_root=self.repo_root)
        return store.claim_lease(
            task_id=task_id,
            title=title or lane_id,
            owner_agent=owner_agent,
            owner_session_id=owner_session_id,
            branch=branch,
            worktree_path=worktree_path,
            allowed_globs=list(lane.allowed_write_scope),
            claimed_paths=[],
            expected_tests=list(expected_tests or lane.verification_commands),
            metadata={
                "tranche_manifest_id": manifest.manifest_id,
                "tranche_lane_id": lane_id,
                **dict(metadata or {}),
            },
        )

    def _resolve_references(self, manifest: TrancheManifest) -> dict[str, dict[str, Any]]:
        resolved: dict[str, dict[str, Any]] = {}
        for group, refs in manifest.references.items():
            for ref_id, ref in refs.items():
                target = parse_github_reference_url(ref.url)
                repo = f"{target.owner}/{target.repo}"
                payload = (
                    self.reference_client.get_pr(repo, target.number)
                    if target.kind == "pull_request"
                    else self.reference_client.get_issue(repo, target.number)
                )
                observed_state = _observed_reference_state(target.kind, payload)
                status = self._classify_reference_status(
                    group=group,
                    kind=target.kind,
                    declared_state=ref.state,
                    observed_state=observed_state,
                )
                resolved[ref_id] = {
                    "group": group,
                    "kind": target.kind,
                    "repo": repo,
                    "number": target.number,
                    "url": ref.url,
                    "title": str(payload.get("title", "")).strip(),
                    "declared_state": ref.state,
                    "observed_state": observed_state,
                    "status": status,
                    "actionable": target.kind == "issue" and observed_state == "open",
                    "drifted": bool(ref.state) and ref.state.lower() != observed_state.lower(),
                    "meaning": ref.meaning,
                    "label": ref.label,
                    "labels": [
                        str(item.get("name", "")).strip()
                        for item in payload.get("labels", [])
                        if isinstance(item, dict) and str(item.get("name", "")).strip()
                    ],
                    "closed_at": payload.get("closedAt"),
                    "merged_at": payload.get("mergedAt"),
                    "mergeable": payload.get("mergeable"),
                    "merge_state_status": payload.get("mergeStateStatus"),
                    "review_decision": payload.get("reviewDecision"),
                    "reason": _reference_reason(group, target.kind, observed_state, status),
                }
        return resolved

    def _classify_reference_status(
        self,
        *,
        group: str,
        kind: str,
        declared_state: str,
        observed_state: str,
    ) -> str:
        declared = declared_state.lower()
        observed = observed_state.lower()
        if group == "retired_targets":
            return "stale" if observed in {"closed", "merged"} else "unexpectedly_open"
        if group == "source_refs":
            if observed == "open":
                return "actionable"
            if observed in {"closed", "merged"}:
                return "resolved"
            return observed
        if group == "live_target":
            if kind == "issue":
                return "actionable" if observed == "open" else "blocked"
            return "actionable" if observed == "open" else "blocked"
        if observed == "merged":
            return "satisfied"
        if observed == "open":
            return "pending" if declared not in {"merged", "closed"} else "drifted"
        if observed == "closed":
            return "resolved" if declared == "closed" else "stale"
        return observed

    def _resolve_gates(
        self,
        manifest: TrancheManifest,
        references: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        resolved: dict[str, dict[str, Any]] = {}
        for gate_id, gate in manifest.gates.items():
            ref = references.get(gate.source_ref)
            if ref is None:
                state = "blocked"
                reason = f"Missing reference {gate.source_ref}"
            else:
                state, reason = self._evaluate_gate(gate, ref)
            resolved[gate_id] = {
                "source_ref": gate.source_ref,
                "declared_state": gate.state,
                "state": state,
                "required_for": list(gate.required_for),
                "satisfy_when": gate.satisfy_when,
                "reason": reason,
            }
        return resolved

    def _evaluate_gate(
        self,
        gate: TrancheGate,
        reference: dict[str, Any],
    ) -> tuple[str, str]:
        observed = str(reference.get("observed_state", "")).lower()
        declared = gate.state.lower()
        satisfy_when = gate.satisfy_when.lower()
        if observed == "merged":
            if "merged" in satisfy_when or declared in {"pending", "satisfied"}:
                return "satisfied", f"{gate.source_ref} merged"
            return "blocked", f"{gate.source_ref} merged unexpectedly"
        if observed == "open":
            if declared == "pending":
                return "pending", f"{gate.source_ref} still open"
            return "blocked", f"{gate.source_ref} still open"
        if observed == "closed":
            if (
                declared == "satisfied"
                and str(reference.get("declared_state", "")).lower() == "closed"
            ):
                return "satisfied", f"{gate.source_ref} reached declared closed state"
            return "blocked", f"{gate.source_ref} closed without satisfying gate"
        return "blocked", f"{gate.source_ref} resolved to unsupported state {observed}"

    def _declared_scope_conflicts(self, manifest: TrancheManifest) -> list[dict[str, Any]]:
        conflicts: list[dict[str, Any]] = []
        writable = [lane for lane in manifest.lanes if lane.allowed_write_scope]
        for index, left in enumerate(writable):
            for right in writable[index + 1 :]:
                if not _paths_overlap(left.allowed_write_scope, right.allowed_write_scope):
                    continue
                conflicts.append(
                    {
                        "left_lane_id": left.lane_id,
                        "right_lane_id": right.lane_id,
                        "left_scope": list(left.allowed_write_scope),
                        "right_scope": list(right.allowed_write_scope),
                    }
                )
        return conflicts

    def _resolve_lanes(
        self,
        manifest: TrancheManifest,
        references: dict[str, dict[str, Any]],
        gates: dict[str, dict[str, Any]],
        artifacts: list[TrancheLaneArtifact],
        scope_conflicts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        artifacts_by_lane: dict[str, list[dict[str, Any]]] = {}
        for artifact in artifacts:
            artifacts_by_lane.setdefault(artifact.lane_id, []).append(artifact.to_dict())
        conflicting_lanes = {item["left_lane_id"] for item in scope_conflicts}.union(
            {item["right_lane_id"] for item in scope_conflicts}
        )
        lanes: list[dict[str, Any]] = []
        for lane in manifest.lanes:
            blockers: list[str] = []
            dependency_status: list[dict[str, Any]] = []
            for dependency in lane.dependencies:
                if dependency in gates:
                    gate = gates[dependency]
                    satisfied = gate["state"] == "satisfied"
                    dependency_status.append(
                        {
                            "dependency": dependency,
                            "kind": "gate",
                            "state": gate["state"],
                            "satisfied": satisfied,
                            "reason": gate["reason"],
                        }
                    )
                    if not satisfied:
                        blockers.append(gate["reason"])
                    continue
                ref = references.get(dependency)
                if ref is None:
                    dependency_status.append(
                        {
                            "dependency": dependency,
                            "kind": "unknown",
                            "state": "missing",
                            "satisfied": False,
                            "reason": f"Unknown dependency {dependency}",
                        }
                    )
                    blockers.append(f"Unknown dependency {dependency}")
                    continue
                satisfied = ref["status"] in {"actionable", "satisfied", "resolved"}
                dependency_status.append(
                    {
                        "dependency": dependency,
                        "kind": "reference",
                        "state": ref["status"],
                        "satisfied": satisfied,
                        "reason": ref["reason"],
                    }
                )
                if not satisfied:
                    blockers.append(ref["reason"])
            if lane.lane_id in conflicting_lanes:
                blockers.append("Lane has declared write-scope overlap with another writable lane.")
            readiness = "ready" if not blockers else "blocked"
            lanes.append(
                {
                    "lane_id": lane.lane_id,
                    "owner_role": lane.owner_role,
                    "claimable": lane.claimable,
                    "readiness": readiness,
                    "allowed_write_scope": list(lane.allowed_write_scope),
                    "dependencies": dependency_status,
                    "blockers": blockers,
                    "artifacts": artifacts_by_lane.get(lane.lane_id, []),
                    "branch": dict(lane.branch),
                    "worktree": dict(lane.worktree),
                    "verification_commands": list(lane.verification_commands),
                    "stop_conditions": list(lane.stop_conditions),
                    "expected_receipts_artifacts": list(lane.expected_receipts_artifacts),
                }
            )
        return lanes

    def _recommended_action(
        self,
        gates: dict[str, dict[str, Any]],
        lanes: list[dict[str, Any]],
        blockers: list[str],
    ) -> dict[str, Any]:
        if blockers:
            return {
                "kind": "stop_and_replan",
                "reason": blockers[0],
            }
        for gate_id, gate in gates.items():
            if gate["state"] == "pending":
                affected = [
                    lane["lane_id"]
                    for lane in lanes
                    if gate_id in {dep["dependency"] for dep in lane["dependencies"]}
                ]
                return {
                    "kind": "resolve_gate",
                    "gate_id": gate_id,
                    "source_ref": gate["source_ref"],
                    "reason": gate["reason"],
                    "affected_lanes": affected,
                }
        for lane in lanes:
            if lane["readiness"] == "ready":
                return {
                    "kind": "run_lane",
                    "lane_id": lane["lane_id"],
                    "reason": "All dependencies satisfied",
                }
        return {
            "kind": "idle",
            "reason": "No ready lane found",
        }


def parse_github_reference_url(url: str) -> GitHubReferenceTarget:
    match = _GITHUB_REF_RE.match(str(url).strip())
    if match is None:
        raise ValueError(f"Unsupported GitHub reference URL: {url}")
    owner, repo, raw_kind, number_text = match.groups()
    return GitHubReferenceTarget(
        owner=owner,
        repo=repo,
        kind="pull_request" if raw_kind == "pull" else "issue",
        number=int(number_text),
    )


def load_tranche_manifest(path: Path) -> TrancheManifest:
    return TrancheManifest.from_text(path.read_text(encoding="utf-8"))


def save_tranche_manifest(path: Path, manifest: TrancheManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with locked_manifest_path(path):
        path.write_text(manifest.to_yaml(), encoding="utf-8")


def render_tranche_inspection_text(payload: dict[str, Any]) -> str:
    lines = [
        f"manifest_id={payload.get('manifest_id', '')}",
        f"preflight={payload.get('preflight_status', '')}",
    ]
    recommended = payload.get("recommended_action", {})
    if isinstance(recommended, dict):
        lines.append(
            "recommended={kind} {detail}".format(
                kind=recommended.get("kind", ""),
                detail=recommended.get("lane_id") or recommended.get("gate_id") or "",
            ).strip()
        )
    blockers = [str(item) for item in payload.get("preflight_blockers", []) if str(item).strip()]
    for blocker in blockers[:3]:
        lines.append(f"blocker: {blocker}")
    for gate_id, gate in list((payload.get("gates") or {}).items())[:5]:
        if not isinstance(gate, dict):
            continue
        lines.append(f"gate {gate_id}: {gate.get('state')} ({gate.get('reason', '')})")
    for lane in [item for item in payload.get("lanes", []) if isinstance(item, dict)][:5]:
        lines.append(
            "lane {lane_id}: {readiness} claimable={claimable}".format(
                lane_id=lane.get("lane_id", ""),
                readiness=lane.get("readiness", ""),
                claimable=lane.get("claimable", False),
            )
        )
        for blocker in [item for item in lane.get("blockers", []) if str(item).strip()][:2]:
            lines.append(f"  blocker: {blocker}")
    return "\n".join(lines)


def _dump_yaml_like(data: dict[str, Any]) -> str:
    try:
        import yaml

        return yaml.safe_dump(data, sort_keys=False, allow_unicode=False)
    except ImportError:
        return json.dumps(data, indent=2, sort_keys=False)


def _load_yaml_like(path: Path) -> dict[str, Any]:
    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except ImportError:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object payload in {path}")
    return payload


def _observed_reference_state(kind: str, payload: dict[str, Any]) -> str:
    state = str(payload.get("state", "")).strip().lower()
    if kind == "pull_request" and _optional_text(payload.get("mergedAt")):
        return "merged"
    return state or "unknown"


def _reference_reason(group: str, kind: str, observed_state: str, status: str) -> str:
    if group == "retired_targets":
        return f"Retired {kind} is {observed_state}"
    if group == "live_target" and status == "actionable":
        return f"Live target is {observed_state}"
    if group == "live_target":
        return f"Live target is {observed_state}; dispatch should not proceed"
    return f"Reference is {observed_state}"


def _prompt_bundle_to_manifest_dict(
    payload: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    manifest_id = _optional_text(payload.get("manifest_id")) or _optional_text(
        payload.get("bundle_id")
    )
    if not manifest_id:
        raise ValueError("Prompt bundle is missing manifest_id or bundle_id.")
    lanes_raw = payload.get("lanes")
    if not isinstance(lanes_raw, list) or not lanes_raw:
        raise ValueError("Prompt bundle lanes must be a non-empty list.")

    repo = _dict_value(payload.get("repo"), field_name="repo")
    base_ref = str(repo.get("base_ref") or payload.get("base_ref") or "origin/main").strip()
    repo.setdefault("root", str(repo_root.resolve()))
    repo.setdefault("base_ref", base_ref)
    repo.setdefault(
        "base_sha", _git_rev_parse(repo_root, base_ref) or _git_rev_parse(repo_root, "HEAD")
    )
    repo.setdefault("name", _repo_name_from_remote(repo_root) or repo_root.name)

    references = {
        str(group): {
            str(ref_id): dict(ref_data)
            for ref_id, ref_data in items.items()
            if isinstance(ref_data, dict)
        }
        for group, items in _dict_value(payload.get("references"), field_name="references").items()
        if isinstance(items, dict)
    }
    references.setdefault("source_refs", {})
    known_ref_ids = {ref_id for refs in references.values() for ref_id in refs}

    planned_lanes: list[dict[str, Any]] = []
    for index, item in enumerate(lanes_raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt bundle lane #{index} must be an object.")
        lane = dict(item)
        lane_id = str(lane.get("lane_id", "")).strip() or f"lane-{index}"
        prompt = _optional_text(lane.get("prompt"))
        if not prompt:
            raise ValueError(f"Prompt bundle lane {lane_id!r} is missing prompt.")
        owner_role = str(lane.get("owner_role", "")).strip()
        if not owner_role:
            raise ValueError(f"Prompt bundle lane {lane_id!r} is missing owner_role.")
        source_refs = _string_list(lane.get("source_refs"), field_name="source_refs")
        derived_dependencies: list[str] = []
        for source_url in source_refs:
            try:
                target = parse_github_reference_url(source_url)
            except ValueError:
                continue
            ref_id = _reference_id_for_target(target)
            if ref_id not in known_ref_ids:
                references["source_refs"][ref_id] = {
                    "kind": target.kind,
                    "url": source_url,
                    "state": "open",
                    "meaning": f"Source ref for lane {lane_id}",
                }
                known_ref_ids.add(ref_id)
            derived_dependencies.append(ref_id)
        dependencies = _string_list(lane.get("dependencies"), field_name="dependencies")
        if not dependencies and derived_dependencies:
            lane["dependencies"] = derived_dependencies
        target_agent = (
            _optional_text(lane.get("target_agent"))
            or _optional_text(lane.get("worker_model"))
            or "codex"
        )
        lane.setdefault("branch", {"convention": f"{target_agent}/<{_slugify(lane_id)}>"})
        lane.setdefault("worktree", {"convention": f".worktrees/{target_agent}-auto/<session-id>"})
        lane["lane_id"] = lane_id
        lane["prompt"] = prompt
        planned_lanes.append(lane)

    terminal_outcomes = _dict_value(
        payload.get("terminal_outcomes"), field_name="terminal_outcomes"
    ) or {
        "success": {"definition": "All required tranche lanes completed and reviewed."},
        "needs_human": {"definition": "A lane or gate blocked without a safe automated next step."},
        "stop_and_replan": {
            "definition": "Targets drifted or execution fell outside the bounded contract."
        },
    }
    return {
        "manifest_version": int(payload.get("manifest_version", 1) or 1),
        "manifest_id": manifest_id,
        "generated_on": _optional_text(payload.get("generated_on")) or _utcnow().date().isoformat(),
        "repo": repo,
        "objective": str(payload.get("objective", "")).strip(),
        "shared_constraints": _dict_value(
            payload.get("shared_constraints"), field_name="shared_constraints"
        ),
        "references": references,
        "gates": _dict_value(payload.get("gates"), field_name="gates"),
        "lanes": planned_lanes,
        "terminal_outcomes": terminal_outcomes,
    }


def _select_claimable_lanes(
    manifest: TrancheManifest,
    *,
    inspection: dict[str, Any],
    lane_id: str,
    all_ready: bool,
) -> list[dict[str, Any]]:
    lanes = [item for item in inspection.get("lanes", []) if isinstance(item, dict)]
    lane_map = {str(item.get("lane_id", "")): item for item in lanes}
    selected_id = str(lane_id or "").strip()
    if selected_id:
        lane = lane_map.get(selected_id)
        if lane is None:
            raise KeyError(f"Unknown tranche lane: {selected_id}")
        if not bool(lane.get("claimable")):
            raise ValueError(f"Lane {selected_id} is read-only and cannot be prepared or run.")
        if str(lane.get("readiness", "")).strip() != "ready":
            blockers = [str(item) for item in lane.get("blockers", []) if str(item).strip()]
            raise ValueError(blockers[0] if blockers else f"Lane {selected_id} is not ready.")
        return [lane]
    if all_ready:
        ready = [
            lane
            for lane in lanes
            if bool(lane.get("claimable")) and str(lane.get("readiness", "")).strip() == "ready"
        ]
        if ready:
            return ready
        raise ValueError("No ready claimable lanes found in tranche manifest.")

    recommended = inspection.get("recommended_action")
    if not isinstance(recommended, dict):
        raise ValueError("Tranche inspection did not return a recommended action.")
    if recommended.get("kind") != "run_lane":
        raise ValueError(str(recommended.get("reason", "Tranche is not ready to run.")))
    recommended_lane_id = str(recommended.get("lane_id", "")).strip()
    lane = lane_map.get(recommended_lane_id)
    if lane is None:
        raise KeyError(f"Recommended tranche lane {recommended_lane_id!r} was not found.")
    if not bool(lane.get("claimable")):
        raise ValueError(f"Recommended lane {recommended_lane_id} is read-only.")
    return [lane]


def _lane_spec_from_manifest(manifest: TrancheManifest, lane: TrancheLane) -> Any:
    from aragora.swarm.spec import SwarmSpec

    prompt = _optional_text(lane.metadata.get("prompt"))
    if not prompt:
        raise ValueError(f"Lane {lane.lane_id} is missing prompt metadata.")
    target_agent = _lane_target_agent(lane, fallback="codex")
    review_model = _lane_review_model(lane, target_agent=target_agent)
    acceptance = _metadata_string_list(lane, "acceptance_criteria") or [
        f"Run and satisfy: {item}" for item in lane.verification_commands
    ]
    constraints = _metadata_string_list(lane, "constraints")
    constraints.extend(_shared_constraints_as_list(manifest.shared_constraints))
    file_scope_hints = _metadata_string_list(lane, "file_scope_hints") or list(
        lane.allowed_write_scope
    )
    explicit_work_order = {
        "work_order_id": lane.lane_id,
        "title": _lane_title(lane),
        "description": prompt,
        "file_scope": list(dict.fromkeys(file_scope_hints)),
        "target_agent": target_agent,
        "reviewer_agent": review_model,
        "expected_tests": list(lane.verification_commands),
        "success_criteria": {
            "acceptance_criteria": list(dict.fromkeys(acceptance)),
            "tests": list(lane.verification_commands),
        },
        "estimated_complexity": str(lane.metadata.get("estimated_complexity", "medium")).strip()
        or "medium",
        "approval_required": _coerce_bool(
            lane.metadata.get("requires_approval"),
            default=True,
        ),
        "metadata": {
            "tranche_manifest_id": manifest.manifest_id,
            "tranche_lane_id": lane.lane_id,
        },
    }
    return SwarmSpec(
        raw_goal=_lane_execution_prompt(manifest, lane, prompt=prompt),
        refined_goal=_lane_execution_prompt(manifest, lane, prompt=prompt),
        acceptance_criteria=list(dict.fromkeys(acceptance)),
        constraints=list(dict.fromkeys(constraints)),
        budget_limit_usd=_lane_budget_limit_usd(lane),
        file_scope_hints=list(dict.fromkeys(file_scope_hints)),
        work_orders=[explicit_work_order],
        requires_approval=_coerce_bool(
            lane.metadata.get("requires_approval"),
            default=True,
        ),
        user_expertise="developer",
        estimated_complexity=str(lane.metadata.get("estimated_complexity", "medium")).strip()
        or "medium",
        track_hints=[lane.owner_role] if lane.owner_role else [],
    )


def _lane_execution_prompt(manifest: TrancheManifest, lane: TrancheLane, *, prompt: str) -> str:
    parts = [prompt.strip()]
    if manifest.objective:
        parts.append(f"Tranche objective: {manifest.objective}")
    source_urls = _lane_source_urls(lane)
    if source_urls:
        parts.append("Source refs:\n- " + "\n- ".join(source_urls))
    if lane.allowed_write_scope:
        parts.append("Allowed write scope:\n- " + "\n- ".join(lane.allowed_write_scope))
    if lane.verification_commands:
        parts.append("Verification commands:\n- " + "\n- ".join(lane.verification_commands))
    if lane.stop_conditions:
        parts.append("Stop and escalate when:\n- " + "\n- ".join(lane.stop_conditions))
    if lane.expected_receipts_artifacts:
        parts.append(
            "Expected receipts/artifacts:\n- " + "\n- ".join(lane.expected_receipts_artifacts)
        )
    return "\n\n".join(part for part in parts if part.strip())


def _shared_constraints_as_list(shared_constraints: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for key, value in sorted(shared_constraints.items()):
        if isinstance(value, list):
            for item in value:
                text = str(item).strip()
                if text:
                    result.append(f"{key}: {text}")
            continue
        text = str(value).strip()
        if text:
            result.append(f"{key}: {text}")
    return result


def _metadata_string_list(lane: TrancheLane, key: str) -> list[str]:
    value = lane.metadata.get(key)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _manifest_base_branch(manifest: TrancheManifest, *, fallback: str) -> str:
    base_ref = str(manifest.repo.get("base_ref") or fallback).strip() or fallback
    if base_ref.startswith("origin/"):
        return base_ref[len("origin/") :] or fallback
    return base_ref


def _lane_target_agent(lane: TrancheLane, *, fallback: str) -> str:
    return (
        _optional_text(lane.metadata.get("target_agent"))
        or _optional_text(lane.metadata.get("worker_model"))
        or fallback
    )


def _lane_review_model(lane: TrancheLane, *, target_agent: str) -> str:
    requested = _optional_text(lane.metadata.get("review_model"))
    if requested:
        if (
            _coerce_bool(lane.metadata.get("enforce_cross_model_review"), default=True)
            and requested == target_agent
        ):
            return "claude" if target_agent == "codex" else "codex"
        return requested
    return "claude" if target_agent == "codex" else "codex"


def _lane_budget_limit_usd(lane: TrancheLane) -> float:
    value = lane.metadata.get("budget_limit_usd", 5.0)
    try:
        return float(value or 5.0)
    except (TypeError, ValueError):
        return 5.0


def _lane_title(lane: TrancheLane) -> str:
    return _optional_text(lane.metadata.get("title")) or lane.lane_id


def _lane_source_urls(lane: TrancheLane) -> list[str]:
    value = lane.metadata.get("source_refs")
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _lane_primary_source_ref(manifest: TrancheManifest, lane: TrancheLane) -> str:
    source_urls = _lane_source_urls(lane)
    if source_urls:
        return source_urls[0]
    for dependency in lane.dependencies:
        if manifest.reference(dependency) is not None:
            return dependency
    return lane.lane_id


def _lane_branch_name(manifest: TrancheManifest, lane: TrancheLane, *, target_agent: str) -> str:
    current = _optional_text(lane.branch.get("current")) if isinstance(lane.branch, dict) else None
    if current:
        return current
    convention = (
        _optional_text(lane.branch.get("convention")) if isinstance(lane.branch, dict) else None
    )
    slug = _slugify(f"{manifest.manifest_id}-{lane.lane_id}")
    if convention:
        if "<" in convention and ">" in convention:
            return re.sub(r"<[^>]+>", slug, convention)
        return convention
    return f"{target_agent}/{slug}"


def _artifact_status_from_dispatch_result(result: dict[str, Any]) -> str:
    status = str(result.get("status", "")).strip().lower()
    return {
        "running": "running",
        "completed": "completed",
        "needs_human": "needs_human",
        "failed": "failed",
    }.get(status, status or "unknown")


def _artifact_status_from_review(status: str) -> str:
    lowered = str(status).strip().lower()
    if lowered == "passed":
        return "review_passed"
    if lowered == "changes_requested":
        return "changes_requested"
    return "review_blocked"


def _residual_risk_from_dispatch_result(result: dict[str, Any]) -> str:
    status = str(result.get("status", "")).strip().lower()
    if status == "running":
        return "Lane dispatched and is still executing."
    if status == "needs_human":
        reasons = [str(item).strip() for item in result.get("reasons", []) if str(item).strip()]
        return reasons[0] if reasons else "Lane requires human follow-up."
    if status == "failed":
        return str(result.get("error", "")).strip() or "Lane execution failed."
    return ""


def _dispatch_result_blocked_reason(
    result: dict[str, Any],
    *,
    run_dict: dict[str, Any],
) -> str | None:
    explicit = _optional_text(result.get("blocked_reason"))
    if explicit:
        return explicit
    for work_order in run_dict.get("work_orders", []):
        if not isinstance(work_order, dict):
            continue
        reason = _optional_text(work_order.get("failure_reason"))
        if reason:
            return reason
        status = str(work_order.get("status", "")).strip().lower()
        if status == "scope_violation":
            return "scope_violation"
    outcome = _optional_text(result.get("outcome"))
    if outcome == "clean_exit_no_deliverable":
        return outcome
    return _optional_text(result.get("status"))


def _dispatch_result_blocking_question(
    result: dict[str, Any],
    *,
    run_dict: dict[str, Any],
) -> str | None:
    explicit = _optional_text(result.get("blocking_question"))
    if explicit:
        return explicit
    for work_order in run_dict.get("work_orders", []):
        if not isinstance(work_order, dict):
            continue
        question = _optional_text(work_order.get("blocking_question"))
        if question:
            return question
    if _dispatch_result_blocked_reason(result, run_dict=run_dict):
        return "What human input is required before rerunning this lane?"
    return None


def _next_actions_from_dispatch_result(result: dict[str, Any]) -> list[str]:
    status = str(result.get("status", "")).strip().lower()
    run_id = _optional_text(result.get("run_id"))
    if status == "running":
        if run_id:
            return [f"Inspect active supervisor run {run_id} before starting another tranche tick."]
        return ["Inspect the active supervisor run before starting another tranche tick."]
    if status == "needs_human":
        reasons = [str(item).strip() for item in result.get("reasons", []) if str(item).strip()]
        return reasons or ["Review the reported blockers and decide whether to replan."]
    if status == "completed":
        return ["Run cross-model review and integrate the resulting deliverable."]
    if status == "failed":
        error = str(result.get("error", "")).strip()
        return [error] if error else ["Inspect the failed lane result and replan."]
    return []


def _review_next_actions(gate: Any) -> list[str]:
    findings = [str(item).strip() for item in getattr(gate, "findings", []) if str(item).strip()]
    status = str(getattr(gate, "status", "")).strip().lower()
    if status == "passed":
        return ["Review passed; proceed with PR validation and merge gating."]
    return findings or ["Review returned a blocking decision."]


def _review_residual_risk(gate: Any) -> str:
    findings = [str(item).strip() for item in getattr(gate, "findings", []) if str(item).strip()]
    return findings[0] if findings else ""


def _deliverable_urls(deliverable: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for key in ("pr_url", "adopted_pr"):
        value = str(deliverable.get(key, "")).strip()
        if value:
            urls.append(value)
    return list(dict.fromkeys(urls))


def _first_worker_worktree_path(run_dict: dict[str, Any]) -> str | None:
    return _first_work_order_text(run_dict, "worktree_path")


def _first_work_order_text(run_dict: dict[str, Any], key: str) -> str | None:
    for item in run_dict.get("work_orders", []):
        if not isinstance(item, dict):
            continue
        value = _optional_text(item.get(key))
        if value:
            return value
    return None


def _reference_id_for_target(target: GitHubReferenceTarget) -> str:
    prefix = "pr" if target.kind == "pull_request" else "issue"
    return f"{prefix}_{target.number}"


def _git_rev_parse(repo_root: Path, ref: str) -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", ref],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    return value or None


def _repo_name_from_remote(repo_root: Path) -> str | None:
    proc = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    remote = proc.stdout.strip()
    match = re.search(r"github\.com[:/]+([^/]+)/([^/.]+)(?:\.git)?$", remote)
    if not match:
        return None
    return f"{match.group(1)}/{match.group(2)}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(value).strip().lower()).strip("-")
    return slug or "lane"
