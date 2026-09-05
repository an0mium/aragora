"""Tests for ``scripts/refresh_model_literals.py``.

Controller ruling (frontier-model-refresh, Task 8, 2026-09-04): the sweep
must SKIP files that legitimately contain retired ids — the catalog and
upgrade-map source itself, legacy pricing/routing tables old receipts
still resolve through, tests that assert retired ids on purpose, and the
script's own source. That skip list lives in ``SKIP_PATHS`` and is matched
by path suffix so it works regardless of the cwd the sweep runs from.

Fix round 1 (2026-09-05): two Important findings from review — (1)
``--check`` output was non-deterministic (``rglob`` discovery order), now
fixed by sorting scanned files and offenders; (2) the historical-allowlist
membership check compared a raw (possibly cwd- or absolute-path-flavored)
string against repo-relative allowlist entries, now fixed by normalizing
both sides to repo-root-relative POSIX paths via ``REPO_ROOT``.

The allowlist-normalization test loads the script as a module (rather than
via subprocess against the real repo) and monkeypatches its ``REPO_ROOT``
to a throwaway tmp_path tree. This repo's own dev checkout lives under a
directory literally named ``.worktrees`` (see ``SKIP_DIRS`` in the script),
so a subprocess run with a genuinely absolute --paths into the real repo
gets zero files back regardless of the allowlist fix — an unrelated,
pre-existing SKIP_DIRS hazard, not something this test should be defeated
by (SKIP_DIRS entries are out of scope for this round; see the script's
module docstring / task-8-report.md fix-round-1 section).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path("scripts/refresh_model_literals.py")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def _load_module() -> Any:
    """Load scripts/refresh_model_literals.py as a fresh, isolated module."""
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "refresh_model_literals.py"
    spec = importlib.util.spec_from_file_location("refresh_model_literals_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_check_output_is_deterministic_and_sorted_by_path(tmp_path: Path) -> None:
    """Two --check runs over the same tree must print byte-identical,
    path-sorted output — not whatever order the filesystem/rglob happens
    to discover files in.
    """
    for name in ("zeta.py", "alpha.py", "mu.py", "beta.py"):
        (tmp_path / name).write_text('X = "gpt-4"\n')

    r1 = _run("--paths", str(tmp_path), "--check", "--allowlist", str(tmp_path / "none.txt"))
    r2 = _run("--paths", str(tmp_path), "--check", "--allowlist", str(tmp_path / "none.txt"))
    assert r1.returncode == 1 and r2.returncode == 1
    assert r1.stdout == r2.stdout, "identical --check runs produced different output"

    offender_lines = [ln for ln in r1.stdout.splitlines() if ": retired model id " in ln]
    assert len(offender_lines) == 4
    reported_paths = [ln.split(":", 1)[0] for ln in offender_lines]
    assert reported_paths == sorted(reported_paths), (
        f"offenders not sorted by path: {reported_paths}"
    )


def test_allowlist_matches_regardless_of_cwd_or_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The historical allowlist stores repo-relative paths (generated via
    ``git ls-files`` from the repo root). Membership must still match when
    the sweep is invoked from an unrelated cwd with an absolute --paths —
    not just when run from the repo root with relative --paths.

    Exercises this against a throwaway fake repo root (monkeypatched onto
    the loaded module) rather than the real one, so the test is hermetic
    and not confounded by this checkout's own SKIP_DIRS(".worktrees")
    layout — see the module docstring above.
    """
    module = _load_module()
    fake_repo_root = (tmp_path / "fake_repo").resolve()
    fixture_file = fake_repo_root / "tests" / "scripts" / "offender.py"
    fixture_file.parent.mkdir(parents=True)
    fixture_file.write_text('X = "gpt-4"\n')
    repo_relative = fixture_file.relative_to(fake_repo_root).as_posix()
    assert repo_relative == "tests/scripts/offender.py"

    monkeypatch.setattr(module, "REPO_ROOT", fake_repo_root)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    empty_allow = tmp_path / "empty_allow.txt"
    empty_allow.write_text("")
    sanity = module.main(["--paths", str(fixture_file), "--check", "--allowlist", str(empty_allow)])
    assert sanity == 1, "fixture should be a genuine offender without an allowlist entry"

    allow = tmp_path / "allow.txt"
    allow.write_text(f"{repo_relative}\n")
    result = module.main(["--paths", str(fixture_file), "--check", "--allowlist", str(allow)])
    assert result == 0, (
        "repo-relative allowlist entry did not match a file given as an "
        "absolute path while running from an unrelated cwd"
    )
