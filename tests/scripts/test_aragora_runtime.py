"""Tests for the shared ``scripts/aragora_runtime.sh`` interpreter resolver.

The resolver must pick a usable interpreter AT RUNTIME so a moved/removed
``.venv`` degrades gracefully instead of stranding a long-running service on a
stale absolute path.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "scripts" / "aragora_runtime.sh"

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
