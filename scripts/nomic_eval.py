#!/usr/bin/env python3
"""
Evaluation harness for comparing Nomic multi-agent runs vs single-agent baselines.

This script runs tasks in isolated git worktrees, captures diffs and logs,
and emits JSON summaries for comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from aragora.nomic.evaluation import EvalTask, build_task_prompt, load_tasks


def _run(
    cmd: list[str], cwd: Path | None = None, env: dict | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)


def _run_logged(
    cmd: list[str],
    cwd: Path,
    env: dict,
    log_path: Path,
    timeout: int | None = None,
) -> dict:
    start = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        log_file.write(f"$ {' '.join(cmd)}\n\n")
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = 124
            log_file.write("\n[timeout] Command timed out\n")
    return {"returncode": code, "duration_seconds": time.time() - start}


def _ensure_worktree(repo: Path, worktree_dir: Path, branch: str, base_ref: str) -> None:
    if worktree_dir.exists():
        raise RuntimeError(f"Worktree already exists: {worktree_dir}")
    result = _run(["git", "worktree", "add", "-b", branch, str(worktree_dir), base_ref], cwd=repo)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git worktree add failed")


def _cleanup_worktree(repo: Path, worktree_dir: Path, branch: str) -> None:
    _run(["git", "worktree", "remove", "--force", str(worktree_dir)], cwd=repo)
    _run(["git", "branch", "-D", branch], cwd=repo)


def _git_diff_stats(repo: Path) -> dict:
    stats = _run(["git", "diff", "--numstat"], cwd=repo)
    added = 0
    removed = 0
    files = []
    if stats.returncode == 0:
        for line in stats.stdout.strip().splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                add = int(parts[0]) if parts[0].isdigit() else 0
                rem = int(parts[1]) if parts[1].isdigit() else 0
                added += add
                removed += rem
                files.append(parts[2])
    diff_stat = _run(["git", "diff", "--stat"], cwd=repo).stdout.strip()
    return {
        "files_changed": len(files),
        "files": files,
        "lines_added": added,
        "lines_removed": removed,
        "diff_stat": diff_stat,
    }


def _parse_outcome(log_path: Path) -> dict:
    if not log_path.exists():
        return {"outcome": "unknown", "verification_passed": False}
    lines = log_path.read_text().splitlines()[-300:]
    outcome = "unknown"
    verification_passed = False
    for line in reversed(lines):
        if "Cycle" in line and "outcome:" in line:
            outcome = line.split("outcome:", 1)[-1].strip()
            break
    for line in lines:
        if "Verification passed" in line:
            verification_passed = True
            break
    return {"outcome": outcome, "verification_passed": verification_passed}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _build_env(base_env: dict, overrides: dict) -> dict:
    env = base_env.copy()
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)
    return env


def _run_variant(
    repo: Path,
    task: EvalTask,
    variant: str,
    output_dir: Path,
    base_ref: str,
    cleanup: bool,
    single_agent: str,
    nomic_timeout: int | None,
) -> dict:
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    task_dir = output_dir / task.task_id
    variant_dir = task_dir / variant / timestamp
    worktree_dir = variant_dir / "worktree"
    branch = f"nomic-eval/{task.task_id}-{variant}-{timestamp}"

    _ensure_worktree(repo, worktree_dir, branch, base_ref)

    proposal_path = variant_dir / "proposal.txt"
    proposal_path.write_text(build_task_prompt(task))

    env_overrides = {
        "NOMIC_AUTO_COMMIT": "0",
        "NOMIC_AUTO_PUSH": "0",
        "NOMIC_AUTO_PR": "0",
        "NOMIC_SINGLE_AGENT": "1" if variant == "single" else "0",
        "NOMIC_SINGLE_AGENT_NAME": single_agent if variant == "single" else "",
        "ARAGORA_HYBRID_IMPLEMENT": "0" if variant == "single" else "1",
    }
    env = _build_env(os.environ, env_overrides)

    cmd = [
        sys.executable,
        str(repo / "scripts" / "nomic_loop.py"),
        "run",
        "--cycles",
        "1",
        "--proposal-file",
        str(proposal_path),
        "--path",
        str(worktree_dir),
        "--no-stream",
    ]
    log_path = variant_dir / "nomic_eval.log"
    run_result = _run_logged(cmd, cwd=repo, env=env, log_path=log_path, timeout=nomic_timeout)

    diff_stats = _git_diff_stats(worktree_dir)
    nomic_log_path = worktree_dir / ".nomic" / "nomic_loop.log"
    outcome = _parse_outcome(nomic_log_path)

    result = {
        "variant": variant,
        "task_id": task.task_id,
        "title": task.title,
        "started_at": timestamp,
        "duration_seconds": run_result["duration_seconds"],
        "returncode": run_result["returncode"],
        "worktree": str(worktree_dir),
        "log_path": str(log_path),
        "nomic_log": str(nomic_log_path),
        "diff": diff_stats,
        "outcome": outcome["outcome"],
        "verification_passed": outcome["verification_passed"],
    }

    _write_json(variant_dir / "result.json", result)

    if cleanup:
        _cleanup_worktree(repo, worktree_dir, branch)

    return result


def _compare_results(single: dict | None, multi: dict | None) -> dict:
    if not single or not multi:
        return {"winner": "unknown", "note": "Incomplete comparison"}
    winner = "tie"
    if single["verification_passed"] and not multi["verification_passed"]:
        winner = "single"
    elif multi["verification_passed"] and not single["verification_passed"]:
        winner = "multi"
    return {
        "winner": winner,
        "single_passed": single["verification_passed"],
        "multi_passed": multi["verification_passed"],
        "single_files_changed": single["diff"]["files_changed"],
        "multi_files_changed": multi["diff"]["files_changed"],
        "single_lines_added": single["diff"]["lines_added"],
        "multi_lines_added": multi["diff"]["lines_added"],
        "single_lines_removed": single["diff"]["lines_removed"],
        "multi_lines_removed": multi["diff"]["lines_removed"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Nomic evaluation harness")
    parser.add_argument("--tasks", type=str, required=True, help="Path to tasks JSON")
    parser.add_argument("--output-dir", type=str, default=".nomic/eval", help="Output directory")
    parser.add_argument("--mode", choices=["single", "multi", "shadow"], default="shadow")
    parser.add_argument("--base-ref", type=str, default="main", help="Git ref for worktrees")
    parser.add_argument("--cleanup", action="store_true", help="Remove worktrees after runs")
    parser.add_argument(
        "--single-agent", type=str, default="codex", help="Single-agent name for baseline"
    )
    parser.add_argument("--task-id", type=str, help="Run only a single task by id")
    parser.add_argument("--timeout", type=int, help="Timeout for each run (seconds)")
    args = parser.parse_args()

    repo = Path(__file__).parent.parent
    tasks = load_tasks(Path(args.tasks))
    if args.task_id:
        tasks = [t for t in tasks if t.task_id == args.task_id]
        if not tasks:
            print(f"No task found: {args.task_id}")
            return 2

    output_dir = repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "repo": str(repo),
        "base_ref": args.base_ref,
        "mode": args.mode,
        "tasks": [],
    }

    for task in tasks:
        task_entry = {
            "task": asdict(task),
            "single": None,
            "multi": None,
            "comparison": None,
        }
        if args.mode in ("single", "shadow"):
            task_entry["single"] = _run_variant(
                repo=repo,
                task=task,
                variant="single",
                output_dir=output_dir,
                base_ref=args.base_ref,
                cleanup=args.cleanup,
                single_agent=args.single_agent,
                nomic_timeout=args.timeout,
            )
        if args.mode in ("multi", "shadow"):
            task_entry["multi"] = _run_variant(
                repo=repo,
                task=task,
                variant="multi",
                output_dir=output_dir,
                base_ref=args.base_ref,
                cleanup=args.cleanup,
                single_agent=args.single_agent,
                nomic_timeout=args.timeout,
            )
        task_entry["comparison"] = _compare_results(task_entry["single"], task_entry["multi"])
        report["tasks"].append(task_entry)

    _write_json(output_dir / "report.json", report)
    print(f"Wrote report to {output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
