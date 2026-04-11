from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_boss_cycle.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run_script(
    tmp_path: Path,
    *args: str,
    env: dict[str, str] | None = None,
    boss_status: int = 0,
) -> subprocess.CompletedProcess[str]:
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    log_path = tmp_path / "python3.log"
    status_path = tmp_path / "boss-status"
    status_path.write_text(str(boss_status), encoding="utf-8")
    _write_executable(
        stub_dir / "python3",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$ARAGORA_TEST_PYTHON3_LOG"
if [[ "$1" == "-u" && "$2" == "-m" && "$3" == "aragora.cli.main" ]]; then
  exit "$(cat "$ARAGORA_TEST_BOSS_STATUS_FILE")"
fi
exit 0
""",
    )
    merged_env = os.environ.copy()
    merged_env["PATH"] = f"{stub_dir}:{merged_env['PATH']}"
    merged_env["ARAGORA_TEST_PYTHON3_LOG"] = str(log_path)
    merged_env["ARAGORA_TEST_BOSS_STATUS_FILE"] = str(status_path)
    if env:
        merged_env.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT_PATH), *args],
        cwd=REPO_ROOT,
        env=merged_env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_successful_boss_loop_runs_post_loop_refill(tmp_path: Path) -> None:
    result = _run_script(
        tmp_path,
        "--boss-repo",
        "example/repo",
        "--label",
        "priority-ready",
        env={"ARAGORA_POST_LOOP_MAX_ISSUES": "7"},
    )

    assert result.returncode == 0
    log_lines = (tmp_path / "python3.log").read_text(encoding="utf-8").splitlines()
    assert log_lines == [
        "-u -m aragora.cli.main swarm boss-loop --boss-repo example/repo --label priority-ready",
        "scripts/generate_boss_issues.py --repo example/repo --max-issues 7 --label priority-ready",
    ]
    assert "Running post-loop issue refill" in result.stdout


def test_nonzero_boss_loop_skips_post_loop_refill(tmp_path: Path) -> None:
    result = _run_script(tmp_path, "--boss-repo", "example/repo", boss_status=9)

    assert result.returncode == 9
    log_lines = (tmp_path / "python3.log").read_text(encoding="utf-8").splitlines()
    assert log_lines == ["-u -m aragora.cli.main swarm boss-loop --boss-repo example/repo"]
    assert "Skipping post-loop issue refill because boss loop exited non-zero." in result.stderr


def test_post_loop_dry_run_and_label_override_apply_to_refill(tmp_path: Path) -> None:
    result = _run_script(
        tmp_path,
        "--label",
        "boss-ready",
        env={
            "ARAGORA_POST_LOOP_DRY_RUN": "1",
            "ARAGORA_POST_LOOP_LABEL": "post-loop-ready",
        },
    )

    assert result.returncode == 0
    log_lines = (tmp_path / "python3.log").read_text(encoding="utf-8").splitlines()
    assert log_lines == [
        "-u -m aragora.cli.main swarm boss-loop --label boss-ready",
        "scripts/generate_boss_issues.py --repo synaptent/aragora --max-issues 20 --label post-loop-ready --dry-run",
    ]
    assert "label=post-loop-ready" in result.stdout
