"""Tests for the shared ``scripts/aragora_runtime.sh`` interpreter resolver.

The resolver must pick a usable interpreter AT RUNTIME so a moved/removed
``.venv`` degrades gracefully instead of stranding a long-running service on a
stale absolute path.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "scripts" / "aragora_runtime.sh"

requires_bash = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash is required to exercise the resolver"
)

# Absolute shebang so the stub execs under a minimal/empty PATH.
_OK = "#!/bin/bash\nexit 0\n"
# Fails the import probe (-c ...) but is otherwise executable.
_BAD = '#!/bin/bash\nif [[ "${1:-}" == "-c" ]]; then exit 1; fi\nexit 0\n'
# Only succeeds for an empty probe (-c "").
_EMPTY_ONLY = (
    "#!/bin/bash\n"
    'if [[ "${1:-}" == "-c" ]]; then\n'
    '  if [[ -z "${2:-}" ]]; then exit 0; else exit 1; fi\n'
    "fi\nexit 0\n"
)


def _mkpy(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _resolve(
    *,
    repo_root: Path,
    path_dirs: list[str],
    aragora_python: str | None = None,
    probe: str = "import pydantic",
) -> subprocess.CompletedProcess[str]:
    repo_root.mkdir(parents=True, exist_ok=True)
    env = {
        "PATH": ":".join(path_dirs),
        "ARAGORA_REPO_ROOT": str(repo_root),
        "HOME": str(repo_root),
    }
    if aragora_python is not None:
        env["ARAGORA_PYTHON"] = aragora_python
    return subprocess.run(
        ["/bin/bash", "-c", f'source "{HELPER}"; resolve_aragora_python "$1"', "_", probe],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_honors_aragora_python_when_probe_passes(tmp_path: Path) -> None:
    ok = _mkpy(tmp_path / "ok" / "python3", _OK)
    result = _resolve(repo_root=tmp_path / "root", path_dirs=[], aragora_python=str(ok))
    assert result.returncode == 0
    assert result.stdout.strip() == str(ok)


def test_skips_unusable_aragora_python_and_falls_back(tmp_path: Path) -> None:
    bad = _mkpy(tmp_path / "bad" / "python3-bad", _BAD)
    good = _mkpy(tmp_path / "bin" / "python3", _OK)
    result = _resolve(
        repo_root=tmp_path / "root",
        path_dirs=[str(good.parent)],
        aragora_python=str(bad),
    )
    assert result.returncode == 0
    assert result.stdout.strip() == str(good)
    assert "Skipping ARAGORA_PYTHON" in result.stderr


def test_prefers_repo_venv_interpreter(tmp_path: Path) -> None:
    root = tmp_path / "root"
    venv_py = _mkpy(root / ".venv" / "bin" / "python3", _OK)
    other = _mkpy(tmp_path / "bin" / "python3", _OK)
    result = _resolve(repo_root=root, path_dirs=[str(other.parent)])
    assert result.returncode == 0
    assert result.stdout.strip() == str(venv_py)


def test_falls_back_to_path_python3_without_venv(tmp_path: Path) -> None:
    good = _mkpy(tmp_path / "bin" / "python3", _OK)
    result = _resolve(repo_root=tmp_path / "root", path_dirs=[str(good.parent)])
    assert result.returncode == 0
    assert result.stdout.strip() == str(good)


def test_fails_loudly_when_no_interpreter_usable(tmp_path: Path) -> None:
    bad = _mkpy(tmp_path / "bin" / "python3", _BAD)
    result = _resolve(repo_root=tmp_path / "root", path_dirs=[str(bad.parent)])
    assert result.returncode == 1
    assert "No usable python interpreter" in result.stderr


def test_empty_probe_accepts_any_executable(tmp_path: Path) -> None:
    only_empty = _mkpy(tmp_path / "bin" / "python3", _EMPTY_ONLY)
    # With the default pydantic probe this interpreter would be rejected...
    rejected = _resolve(repo_root=tmp_path / "root", path_dirs=[str(only_empty.parent)])
    assert rejected.returncode == 1
    # ...but an empty probe skips the import check and accepts it.
    accepted = _resolve(repo_root=tmp_path / "root", path_dirs=[str(only_empty.parent)], probe="")
    assert accepted.returncode == 0
    assert accepted.stdout.strip() == str(only_empty)


def test_helper_is_sourced_not_executed() -> None:
    # The helper defines functions and must be sourced; it should not be marked
    # executable (callers use `source`).
    assert not os.access(HELPER, os.X_OK)


# ---------------------------------------------------------------------------
# Contract tests against the REAL host environment (no PATH sandboxing).
# These pin the resolver's launch-time behavior for systemd/launchd wrappers.
# ---------------------------------------------------------------------------


@requires_bash
def test_contract_empty_probe_resolves_real_interpreter() -> None:
    """An empty probe must resolve some executable interpreter on this host."""
    result = subprocess.run(
        ["bash", "-c", f'source "{HELPER}" && resolve_aragora_python ""'],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    resolved = result.stdout.strip()
    assert resolved, "resolver printed nothing"
    assert os.access(resolved, os.X_OK), f"not executable: {resolved}"


@requires_bash
def test_contract_nonexistent_aragora_python_falls_back_with_diagnostic() -> None:
    """Characterization: a bogus ARAGORA_PYTHON is skipped with a diagnostic.

    The resolver does NOT fail closed on a bad ARAGORA_PYTHON; it warns on
    stderr ("Skipping ARAGORA_PYTHON ...") and continues down the fallback
    chain (.venv, python3, python, pyenv). With an empty probe the host always
    has at least one fallback, so the call still succeeds.
    """
    env = dict(os.environ, ARAGORA_PYTHON="/nonexistent")
    result = subprocess.run(
        ["bash", "-c", f'source "{HELPER}" && resolve_aragora_python ""'],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert "Skipping ARAGORA_PYTHON" in result.stderr
    assert "/nonexistent" in result.stderr
    assert result.returncode == 0, result.stderr
    resolved = result.stdout.strip()
    assert resolved != "/nonexistent"
    assert os.access(resolved, os.X_OK)


@requires_bash
def test_contract_repo_root_env_honored(tmp_path: Path) -> None:
    """aragora_repo_root() must prefer an existing ARAGORA_REPO_ROOT dir."""
    override = tmp_path / "elsewhere"
    override.mkdir()
    env = dict(os.environ, ARAGORA_REPO_ROOT=str(override))
    result = subprocess.run(
        ["bash", "-c", f'source "{HELPER}" && aragora_repo_root'],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert Path(result.stdout.strip()) == override


@requires_bash
def test_contract_repo_root_ignores_missing_override_dir(tmp_path: Path) -> None:
    """A nonexistent ARAGORA_REPO_ROOT is ignored; derived root wins."""
    env = dict(os.environ, ARAGORA_REPO_ROOT=str(tmp_path / "does-not-exist"))
    result = subprocess.run(
        ["bash", "-c", f'source "{HELPER}" && aragora_repo_root'],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert Path(result.stdout.strip()).resolve() == REPO_ROOT.resolve()
