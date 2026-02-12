"""Tests for scripts/check_agent_registry_sync.py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_agent_registry_sync.py"


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        env=dict(os.environ),
    )


def test_docs_only_passes_for_repo_agents_doc():
    result = _run("--docs-only")
    assert result.returncode == 0
    assert "docs-only check passed" in result.stdout.lower()


def test_docs_only_fails_when_declared_count_mismatches(tmp_path: Path):
    broken = tmp_path / "AGENTS.md"
    broken.write_text(
        """
## Agent Types
Aragora currently registers 3 agent types.

| Agent Type | Notes |
|---|---|
| `a` | |
| `b` | |

ALLOWED_AGENT_TYPES, 1 types.

## Agent Creation
""".strip()
    )

    result = _run("--docs-only", "--agents-doc", str(broken))
    assert result.returncode == 1
    assert "mismatch" in result.stdout.lower() or "declared" in result.stdout.lower()
