from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import time
from pathlib import Path

from aragora.swarm.env_utils import git_safe_env
from aragora.swarm.worker_launcher import LaunchConfig, WorkerLauncher


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(detail or f"Command failed: {' '.join(cmd)}")


def _branch_name() -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"preflight/{stamp}"


def _worktree_path(repo_root: Path, branch: str) -> Path:
    return repo_root / ".worktrees" / f"preflight-{branch.replace('/', '-')}"


def _work_order(agent: str) -> dict[str, object]:
    filename = "scratch/preflight_worker_check.txt"
    return {
        "work_order_id": f"preflight-{int(time.time())}",
        "target_agent": agent,
        "title": "Preflight worker check",
        "description": (
            "Create a file named `scratch/preflight_worker_check.txt` with a single line "
            "timestamp. Commit it with message `chore: preflight worker check`. "
            "Do not modify any other files."
        ),
        "file_scope": [filename],
        "expected_tests": [],
        "metadata": {"admin_approved": True},
    }


async def _run_worker(
    *,
    repo_root: Path,
    worktree_path: Path,
    branch: str,
    agent: str,
) -> None:
    config = LaunchConfig(
        allow_claude_dangerously_skip_permissions=True,
        allow_codex_full_auto=True,
    )
    launcher = WorkerLauncher(config=config)
    work_order = _work_order(agent)
    worker = await launcher.launch_and_wait(
        work_order,
        worktree_path=str(worktree_path),
        branch=branch,
        timeout=900.0,
    )
    if not worker.commit_shas:
        raise RuntimeError("Preflight worker did not produce a commit.")


def _create_pr(repo_root: Path, branch: str) -> None:
    _run(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            "synaptent/aragora",
            "--head",
            branch,
            "--base",
            "main",
            "--title",
            "[preflight] worker check",
            "--body",
            "Preflight validation of worker read/write/commit/push.",
            "--draft",
        ],
        cwd=repo_root,
    )


def _close_pr(repo_root: Path, branch: str) -> None:
    _run(
        [
            "gh",
            "pr",
            "close",
            "--repo",
            "synaptent/aragora",
            branch,
            "--delete-branch",
            "--comment",
            "Preflight complete — closing.",
        ],
        cwd=repo_root,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run swarm worker preflight.")
    parser.add_argument(
        "--repo-root",
        default=str(Path.cwd()),
        help="Repository root",
    )
    parser.add_argument(
        "--agent",
        default=os.environ.get("WORKER_MODEL", "claude"),
        help="Target agent (claude or codex)",
    )
    parser.add_argument(
        "--skip-publication",
        action="store_true",
        help="Skip push/PR steps (debug only).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    branch = _branch_name()
    worktree_path = _worktree_path(repo_root, branch)
    worktree_path.parent.mkdir(parents=True, exist_ok=True)

    _run(["git", "worktree", "add", "-b", branch, str(worktree_path), "main"], cwd=repo_root)
    try:
        asyncio.run(
            _run_worker(
                repo_root=repo_root,
                worktree_path=worktree_path,
                branch=branch,
                agent=str(args.agent),
            )
        )

        if args.skip_publication:
            return 0

        _run(["git", "push", "origin", "HEAD"], cwd=worktree_path, env=git_safe_env())
        _create_pr(repo_root, branch)
        _close_pr(repo_root, branch)
        return 0
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree_path)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        subprocess.run(
            ["git", "branch", "-D", branch],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )


if __name__ == "__main__":
    raise SystemExit(main())
