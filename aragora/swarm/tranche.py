"""Manifest-driven tranche inspection and artifact persistence for swarm work."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aragora.nomic.dev_coordination import DevCoordinationStore, WorkLease
from aragora.swarm.campaign import locked_manifest_path

UTC = timezone.utc
DEFAULT_TRANCHE_ARTIFACT_ROOT = ".aragora/tranche_artifacts"
_GITHUB_REF_RE = re.compile(r"^https://github\.com/([^/]+)/([^/]+)/(pull|issues)/(\d+)$")


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
    metadata: dict[str, Any] = field(default_factory=dict)

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
            metadata=_dict_value(data.get("metadata"), field_name="metadata"),
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
