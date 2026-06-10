from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "audit_endpoints.py"


def _run_pipe(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-o", "pipefail", "-c", command],
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_help_pipe_to_head_is_pipe_safe() -> None:
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(SCRIPT_PATH))} --help | head -1"

    result = _run_pipe(command)

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "API Endpoint Audit" not in result.stdout
    assert "BrokenPipeError" not in result.stderr


def test_report_pipe_to_head_is_pipe_safe(tmp_path: Path) -> None:
    handlers_dir = tmp_path / "handlers"
    frontend_dir = tmp_path / "frontend"
    handlers_dir.mkdir()
    frontend_dir.mkdir()
    (handlers_dir / "example.py").write_text('"""GET /api/example/items/{item_id}"""\n')
    (frontend_dir / "app.ts").write_text('fetch("/api/example/items/${itemId}")\n')
    command = " ".join(
        [
            shlex.quote(sys.executable),
            shlex.quote(str(SCRIPT_PATH)),
            "--handlers-dir",
            shlex.quote(str(handlers_dir)),
            "--frontend-dir",
            shlex.quote(str(frontend_dir)),
            "|",
            "head",
            "-1",
        ]
    )

    result = _run_pipe(command)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "API Endpoint Audit\n"
    assert "BrokenPipeError" not in result.stderr
