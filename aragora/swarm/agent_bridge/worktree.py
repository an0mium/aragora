from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from typing import Any


@dataclass(frozen=True, slots=True)
class WorktreeLease:
    agent_slug: str
    branch: str
    path: Path
    base_branch: str
    session_id: str


def ensure_worktree(
    repo_root: Path,
    *,
    agent_slug: str,
    base_branch: str,
    force_new: bool = False,
    session_id: str | None = None,
    reconcile: bool = False,
    strategy: str = "merge",
    managed_dir: str = ".worktrees/codex-auto",
) -> WorktreeLease:
    module = _load_autopilot_module(Path(repo_root).resolve())
    resolved_repo_root = Path(module._repo_root_from(Path(repo_root).resolve()))
    managed_root = (resolved_repo_root / managed_dir).resolve()
    state_file = module._state_path(managed_root)

    entries = module._get_worktree_entries(resolved_repo_root)
    active_paths = module._active_path_set(entries)
    active_branches_by_path = {str(entry.path): entry.branch for entry in entries}
    state = module._load_state(state_file)
    state, _ = module._prune_stale_state(state, active_paths)

    session: dict[str, Any] | None = None
    if not force_new:
        session = module._choose_reusable_session(
            state,
            agent=agent_slug,
            session_id=session_id,
            active_paths=active_paths,
            active_branches_by_path=active_branches_by_path,
        )
        if session is None and module._evict_branch_mismatched_session(
            resolved_repo_root,
            state,
            agent=agent_slug,
            session_id=session_id,
            active_branches_by_path=active_branches_by_path,
        ):
            entries = module._get_worktree_entries(resolved_repo_root)
            active_paths = module._active_path_set(entries)
            active_branches_by_path = {str(entry.path): entry.branch for entry in entries}

    ttl = timedelta(hours=int(module.DEFAULT_TTL_HOURS))
    if session is None:
        session = module._create_managed_worktree(
            resolved_repo_root,
            managed_root,
            agent=agent_slug,
            base=base_branch,
            session_id=session_id,
        )
        entries = module._get_worktree_entries(resolved_repo_root)
        active_paths = module._active_path_set(entries)
    else:
        session_path = Path(str(session["path"]))
        status = module._worktree_status(resolved_repo_root, session_path, base_branch)
        needs_replacement = bool(status["dirty"]) or int(status["ahead"]) > 0
        if reconcile and not needs_replacement and int(status["behind"]) > 0:
            ok, reconcile_status = module._integrate_worktree(
                resolved_repo_root,
                session_path,
                base_branch,
                strategy,
            )
            session["reconcile_status"] = reconcile_status
            if not ok:
                needs_replacement = True
        if needs_replacement:
            session = module._create_managed_worktree(
                resolved_repo_root,
                managed_root,
                agent=agent_slug,
                base=base_branch,
                session_id=session_id,
            )
            entries = module._get_worktree_entries(resolved_repo_root)
            active_paths = module._active_path_set(entries)
        else:
            session["last_seen_at"] = module._utc_now().isoformat()

    if session is None:
        raise RuntimeError("agent-bridge worktree session was not assigned")
    module._annotate_session(
        resolved_repo_root,
        session,
        active_paths=active_paths,
        ttl=ttl,
        base_branch=base_branch,
    )
    module._upsert_session(state, session)
    module._save_state(state_file, state)

    return WorktreeLease(
        agent_slug=agent_slug,
        branch=str(session["branch"]),
        path=Path(str(session["path"])),
        base_branch=base_branch,
        session_id=str(session["session_id"]),
    )


def _load_autopilot_module(repo_root: Path) -> ModuleType:
    script_path = repo_root / "scripts" / "codex_worktree_autopilot.py"
    spec = importlib.util.spec_from_file_location("aragora_codex_worktree_autopilot", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load worktree autopilot module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
