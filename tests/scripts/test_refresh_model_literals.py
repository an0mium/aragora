"""Tests for ``scripts/refresh_model_literals.py``.

Controller ruling (frontier-model-refresh, Task 8, 2026-09-04): the sweep
must SKIP files that legitimately contain retired ids — the catalog and
upgrade-map source itself, legacy pricing/routing tables old receipts
still resolve through, tests that assert retired ids on purpose, and the
script's own source. That skip list lives in ``SKIP_PATHS`` and is matched
by path suffix so it works regardless of the cwd the sweep runs from.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/refresh_model_literals.py")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def test_rewrites_bare_and_openrouter_spellings(tmp_path: Path) -> None:
    f = tmp_path / "x.py"
    f.write_text('A = "gpt-4o"\nB = "anthropic/claude-fable-5"\nC = "claude-fable-5-1"\n')
    r = _run("--paths", str(tmp_path), "--write", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0, r.stderr
    assert f.read_text() == (
        'A = "gpt-6-astra"\nB = "anthropic/claude-fable-5.1"\nC = "claude-fable-5-1"\n'
    )


def test_check_fails_on_retired_literal_and_respects_allowlist(tmp_path: Path) -> None:
    f = tmp_path / "old.md"
    f.write_text("we shipped gpt-4 in 2024\n")
    allow = tmp_path / "allow.txt"
    allow.write_text("")
    assert _run("--paths", str(tmp_path), "--check", "--allowlist", str(allow)).returncode == 1
    allow.write_text(f"{f}\n")
    assert _run("--paths", str(tmp_path), "--check", "--allowlist", str(allow)).returncode == 0


def test_does_not_touch_lockfiles_or_git(tmp_path: Path) -> None:
    (tmp_path / "package-lock.json").write_text('{"x":"gpt-4"}')
    r = _run("--paths", str(tmp_path), "--write", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0 and (tmp_path / "package-lock.json").read_text() == '{"x":"gpt-4"}'


def test_skip_paths_are_never_rewritten_or_reported(tmp_path: Path) -> None:
    """Files at known SKIP_PATHS suffixes must be left alone entirely.

    These are the catalog/upgrade-map source, legacy pricing/routing
    tables, tests/models/, and the sweep script itself — see the
    SKIP_PATHS comment in scripts/refresh_model_literals.py for why each
    one legitimately contains retired ids on purpose.
    """
    skip_files = {
        tmp_path / "aragora" / "models" / "catalog.py": 'RETIRED = "gpt-4o"\n',
        tmp_path / "aragora" / "billing" / "usage.py": 'LEGACY = "gpt-4"\n',
        tmp_path / "tests" / "models" / "test_retired_on_purpose.py": 'OLD = "grok-3"\n',
        tmp_path / "scripts" / "refresh_model_literals.py": 'SELF = "claude-3-opus"\n',
    }
    for path, content in skip_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    r = _run("--paths", str(tmp_path), "--write", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0, r.stderr
    for path, content in skip_files.items():
        assert path.read_text() == content, f"{path} was rewritten but should be skipped"

    r = _run("--paths", str(tmp_path), "--check", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0, f"skip-path files were reported as offenders: {r.stdout}"
