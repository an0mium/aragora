"""Runbook dispatcher for coordination directives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunbookDirective:
    target: str
    task: str
    scope: list[str]
    constraints: list[str]
    status: str


def resolve_runbook_path(runbook: str, repo_root: Path) -> Path:
    path = Path(runbook)
    if path.suffix in {".yaml", ".yml"}:
        return path if path.is_absolute() else (repo_root / path).resolve()
    local_path = (repo_root / ".aragora" / "runbooks" / f"{runbook}.yaml").resolve()
    if local_path.exists():
        return local_path
    docs_path = (repo_root / "docs" / "runbooks" / f"{runbook}.yaml").resolve()
    return docs_path


def _normalize_directive(item: dict[str, Any], default_status: str) -> RunbookDirective:
    target = str(item.get("target") or "").strip()
    task = str(item.get("task") or "").strip()
    if not target:
        raise ValueError("runbook directive missing required field: target")
    if not task:
        raise ValueError(f"runbook directive for {target} missing required field: task")
    scope = [str(value).strip() for value in (item.get("scope") or []) if str(value).strip()]
    constraints = [
        str(value).strip() for value in (item.get("constraints") or []) if str(value).strip()
    ]
    status = str(item.get("status") or default_status or "active").strip() or "active"
    return RunbookDirective(
        target=target,
        task=task,
        scope=scope,
        constraints=constraints,
        status=status,
    )


def _load_runbook(path: Path) -> dict[str, Any]:
    import yaml

    raw = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(raw) or {}
    if not isinstance(payload, dict):
        raise ValueError("runbook must be a YAML mapping")
    return payload


def dispatch_runbook(
    runbook_path: Path,
    *,
    issued_by: str,
    repo_root: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    from aragora.swarm.session_coordinator import set_assignment

    payload = _load_runbook(runbook_path)
    runbook_name = str(payload.get("name") or runbook_path.stem)
    default_status = str(payload.get("default_status") or "active")
    directives_raw = payload.get("directives") or []
    if not isinstance(directives_raw, list):
        raise ValueError("runbook directives must be a list")

    directives: list[RunbookDirective] = []
    for item in directives_raw:
        if not isinstance(item, dict):
            raise ValueError("runbook directive must be a mapping")
        directives.append(_normalize_directive(item, default_status))

    results: list[dict[str, Any]] = []
    for directive in directives:
        if dry_run:
            results.append(
                {
                    "target": directive.target,
                    "task": directive.task,
                    "scope": list(directive.scope),
                    "constraints": list(directive.constraints),
                    "status": directive.status,
                }
            )
            continue
        result = set_assignment(
            directive.target,
            directive.task,
            scope=list(directive.scope),
            constraints=list(directive.constraints),
            status=directive.status,
            issued_by=issued_by,
            repo_root=repo_root,
        )
        results.append(result)

    return {
        "name": runbook_name,
        "path": str(runbook_path),
        "issued_by": issued_by,
        "directives": results,
        "dry_run": dry_run,
    }
