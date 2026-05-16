"""Read-only source adapters for the Aragora work board."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aragora.work.models import WorkItem
from aragora.work.scoring import is_current_status, stale_factor


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _read_jsonl(path: Path, *, limit: int = 200) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return rows
    for raw in lines[-limit:]:
        raw = raw.strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            rows.append(data)
    return rows


def _health(source: str, status: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"source": source, "status": status, "detail": detail, **extra}


def _tier_from_labels(labels: list[str]) -> int | None:
    normalized = {label.lower().replace("_", "-").strip() for label in labels}
    for tier in (4, 3, 2, 1, 0):
        if f"tier-{tier}" in normalized or f"tier {tier}" in normalized:
            return tier
    return None


def _labels_are_human_gated(labels: list[str], tier: int | None) -> bool:
    normalized = {label.lower().replace("_", "-").strip() for label in labels}
    return bool(
        (tier is not None and tier >= 3)
        or normalized
        & {
            "human-gated",
            "human-preapproval",
            "tier-3",
            "tier-4",
            "semantic-risk",
            "security",
            "merge-authority",
        }
    )


def collect_github_prs(repo_root: Path) -> tuple[list[WorkItem], dict[str, Any]]:
    if shutil.which("gh") is None:
        return [], _health("github_pr", "degraded", "gh executable not found")

    fields = (
        "number,title,url,state,isDraft,headRefName,headRefOid,"
        "updatedAt,createdAt,reviewDecision,mergeStateStatus,labels,assignees"
    )
    try:
        result = subprocess.run(
            ["gh", "pr", "list", "--state", "open", "--limit", "50", "--json", fields],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
            timeout=12,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], _health("github_pr", "degraded", f"gh pr list failed: {exc}")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        return [], _health("github_pr", "degraded", detail[0] if detail else "gh pr list failed")
    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return [], _health("github_pr", "degraded", "gh pr list returned non-json")
    if not isinstance(payload, list):
        return [], _health("github_pr", "degraded", "gh pr list returned non-list")

    items: list[WorkItem] = []
    for pr in payload:
        if not isinstance(pr, dict):
            continue
        number = pr.get("number")
        labels = [
            str(label.get("name")) for label in pr.get("labels") or [] if isinstance(label, dict)
        ]
        tier = _tier_from_labels(labels)
        assignees = [
            str(assignee.get("login"))
            for assignee in pr.get("assignees") or []
            if isinstance(assignee, dict)
        ]
        item = WorkItem(
            id=f"pr:{number}",
            source="github_pr",
            item_type="pull_request",
            title=str(pr.get("title") or f"PR #{number}"),
            status="draft" if pr.get("isDraft") else str(pr.get("state") or "open").lower(),
            scope="current",
            url=str(pr.get("url") or "") or None,
            owner=assignees[0] if assignees else None,
            branch=str(pr.get("headRefName") or "") or None,
            created_at=str(pr.get("createdAt") or "") or None,
            updated_at=str(pr.get("updatedAt") or "") or None,
            tags=labels,
            metadata={
                "number": number,
                "is_draft": bool(pr.get("isDraft")),
                "head_sha": pr.get("headRefOid"),
                "review_decision": pr.get("reviewDecision"),
                "merge_state_status": pr.get("mergeStateStatus"),
                "labels": labels,
                "tier": tier,
                "human_gate": _labels_are_human_gated(labels, tier),
            },
        )
        items.append(item)
    return items, _health("github_pr", "ok", f"{len(items)} open PR(s)")


def collect_automation_outbox(repo_root: Path) -> tuple[list[WorkItem], dict[str, Any]]:
    outbox = repo_root / ".aragora" / "automation-outbox"
    if not outbox.exists():
        return [], _health("automation_outbox", "missing", f"{outbox} not found")
    items: list[WorkItem] = []
    for path in sorted(outbox.glob("*.json")):
        data = _read_json(path)
        if data is None:
            continue
        title = str(data.get("task") or data.get("title") or data.get("summary") or path.stem)
        branch = data.get("branch") or data.get("head_ref") or data.get("branch_name")
        items.append(
            WorkItem(
                id=f"automation-outbox:{path.stem}",
                source="automation_outbox",
                item_type="handoff",
                title=title,
                status=str(data.get("status") or data.get("reason") or "pending"),
                scope="current",
                branch=str(branch) if branch else None,
                updated_at=str(data.get("updated_at") or data.get("recorded_at") or "") or None,
                evidence_refs=[str(path.relative_to(repo_root))],
                metadata={"path": str(path), "idempotency_key": data.get("idempotency_key")},
            )
        )
    return items, _health("automation_outbox", "ok", f"{len(items)} pending handoff(s)")


def collect_automation_receipts(
    repo_root: Path, *, scope: str
) -> tuple[list[WorkItem], dict[str, Any]]:
    receipts = repo_root / ".aragora" / "automation-receipts"
    if not receipts.exists():
        return [], _health("automation_receipt", "missing", f"{receipts} not found")
    files = sorted(receipts.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:80]
    items: list[WorkItem] = []
    for path in files:
        data = _read_json(path)
        if data is None:
            continue
        status = str(data.get("status") or data.get("reason") or "recorded")
        updated_at = str(data.get("recorded_at") or data.get("updated_at") or "") or None
        active = is_current_status(status) and stale_factor(updated_at) > 0.35
        if scope == "current" and not active:
            continue
        title = str(data.get("task") or data.get("idempotency_key") or path.stem)
        items.append(
            WorkItem(
                id=f"automation-receipt:{path.stem}",
                source="automation_receipt",
                item_type="receipt",
                title=title,
                status=status,
                scope="current" if active else "historical",
                url=data.get("existing_pr_url")
                or data.get("created_issue_url")
                or data.get("existing_issue_url"),
                updated_at=updated_at,
                evidence_refs=[str(path.relative_to(repo_root))],
                metadata={
                    "reason": data.get("reason"),
                    "repo": data.get("repo"),
                    "source_file": data.get("source_file"),
                },
            )
        )
    return items, _health("automation_receipt", "ok", f"{len(items)} receipt item(s)")


def collect_broker_runs(repo_root: Path, *, scope: str) -> tuple[list[WorkItem], dict[str, Any]]:
    runs_dir = repo_root / ".aragora" / "agent_bridge" / "runs"
    if not runs_dir.exists():
        return [], _health("broker_run", "missing", f"{runs_dir} not found")
    items: list[WorkItem] = []
    for run_path in sorted(runs_dir.glob("*/run.json")):
        data = _read_json(run_path)
        if data is None:
            continue
        status = str(data.get("status") or "unknown")
        active = is_current_status(status)
        if scope == "current" and not active:
            continue
        run_id = str(data.get("run_id") or run_path.parent.name)
        participants = data.get("participants") or []
        owner = None
        if participants and isinstance(participants[0], dict):
            owner = str(participants[0].get("harness") or "") or None
        items.append(
            WorkItem(
                id=f"broker-run:{run_id}",
                source="broker_run",
                item_type="broker_run",
                title=str(data.get("task") or run_id).splitlines()[0][:180],
                status=status,
                scope="current" if active else "historical",
                owner=owner,
                created_at=str(data.get("created_at") or "") or None,
                updated_at=str(data.get("updated_at") or data.get("completed_at") or "") or None,
                evidence_refs=[str(run_path.relative_to(repo_root))],
                metadata={
                    "run_id": run_id,
                    "next_actor": data.get("next_actor"),
                    "worktree_path": data.get("worktree_path"),
                    "participants": participants,
                },
            )
        )
    return items, _health("broker_run", "ok", f"{len(items)} broker run item(s)")


def collect_beads_and_convoys(
    repo_root: Path, *, scope: str
) -> tuple[list[WorkItem], dict[str, Any]]:
    items: list[WorkItem] = []
    paths = [
        (repo_root / ".beads" / "beads.jsonl", "bead"),
        (repo_root / ".aragora_beads" / "beads.jsonl", "bead"),
        (repo_root / ".aragora_beads" / "convoys.jsonl", "convoy"),
    ]
    for path, kind in paths:
        if not path.exists():
            continue
        for row in _read_jsonl(path):
            status = str(row.get("status") or "unknown")
            updated_at = str(row.get("updated_at") or row.get("created_at") or "") or None
            active = is_current_status(status) and stale_factor(updated_at) >= 0.35
            if scope == "current" and not active:
                continue
            raw_id = row.get("id") or row.get("bead_id") or row.get("convoy_id")
            if not raw_id:
                continue
            deps = row.get("dependencies") or row.get("depends_on") or []
            if not isinstance(deps, list):
                deps = []
            items.append(
                WorkItem(
                    id=f"{kind}:{raw_id}",
                    source=kind,
                    item_type=kind,
                    title=str(row.get("title") or row.get("name") or raw_id),
                    status=status,
                    scope="current" if active else "historical",
                    owner=row.get("claimed_by") or row.get("assigned_agent"),
                    created_at=str(row.get("created_at") or "") or None,
                    updated_at=updated_at,
                    dependencies=[f"{kind}:{dep}" for dep in deps],
                    evidence_refs=[str(path.relative_to(repo_root))],
                    tags=[str(tag) for tag in row.get("tags") or []],
                    metadata={
                        "priority": row.get("priority"),
                        "bead_ids": row.get("bead_ids") or [],
                        "metadata": row.get("metadata") or {},
                    },
                )
            )
    return items, _health("bead_convoy", "ok", f"{len(items)} bead/convoy item(s)")


def collect_mission_files(repo_root: Path, *, scope: str) -> tuple[list[WorkItem], dict[str, Any]]:
    mission_roots = [
        repo_root / "docs" / "missions",
        repo_root / ".aragora" / "goal-conductor",
        repo_root / ".aragora" / "initiatives",
    ]
    items: list[WorkItem] = []
    for root in mission_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*"))[:120]:
            if not path.is_file() or path.suffix.lower() not in {".md", ".json", ".yaml", ".yml"}:
                continue
            stat = path.stat()
            updated = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
            active = stale_factor(updated) >= 0.65 and ".aragora" in path.parts
            if scope == "current" and not active:
                continue
            title = path.stem
            if path.suffix.lower() == ".md":
                try:
                    first = next(
                        (
                            line.lstrip("# ").strip()
                            for line in path.read_text(
                                encoding="utf-8", errors="replace"
                            ).splitlines()
                            if line.strip()
                        ),
                        "",
                    )
                except OSError:
                    first = ""
                title = first or title
            items.append(
                WorkItem(
                    id=f"mission:{path.relative_to(repo_root)}",
                    source="mission_file",
                    item_type="mission",
                    title=title[:180],
                    status="context",
                    scope="current" if active else "historical",
                    updated_at=updated,
                    evidence_refs=[str(path.relative_to(repo_root))],
                    metadata={"path": str(path), "size_bytes": stat.st_size},
                )
            )
    return items, _health("mission_file", "ok", f"{len(items)} mission file item(s)")
