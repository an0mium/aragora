"""The launchd installers must generate plists that resolve Python AT RUNTIME.

Regression guard for the install-time interpreter-capture bug: the generated
plist must invoke a repo-owned wrapper script and must never bake an absolute
interpreter path (``export ARAGORA_PYTHON=...`` or a captured virtualenv Python)
that goes stale when the venv moves or is removed.
"""

from __future__ import annotations

import plistlib
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_OK_PY = "#!/bin/bash\nexit 0\n"
_STUB = "#!/bin/bash\nexit 0\n"

# installer, label, wrapper referenced by the generated plist, extra CLI args
CASES = [
    (
        "install_merge_arbiter_launchd.sh",
        "com.aragora.swarm-merge-arbiter",
        "scripts/run_merge_arbiter.sh",
        [],
    ),
    (
        "install_boss_loop_launchd.sh",
        "com.aragora.swarm-boss-loop",
        "scripts/run_boss_cycle.sh",
        ["--label", "boss-ready"],
    ),
]

_NEEDED_SCRIPTS = [
    "aragora_runtime.sh",
    "run_merge_arbiter.sh",
    "run_pr_watch.sh",
    "run_boss_cycle.sh",
]


def _mk(path: Path, body: str, *, executable: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    if executable:
        path.chmod(0o755)
    return path


def _run_installer(installer: str, args: list[str], tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in [installer, *_NEEDED_SCRIPTS]:
        shutil.copy2(REPO_ROOT / "scripts" / name, repo / "scripts" / name)

    bindir = tmp_path / "bin"
    fake_py = _mk(bindir / "python3", _OK_PY, executable=True)
    _mk(bindir / "launchctl", _STUB, executable=True)

    home = tmp_path / "home"
    home.mkdir()

    env = {
        "PATH": f"{bindir}:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(home),
        "USER": "tester",
    }
    proc = subprocess.run(
        ["/bin/bash", str(repo / "scripts" / installer), *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"installer failed: {proc.stderr}\n{proc.stdout}"
    return home, fake_py


@pytest.mark.parametrize("installer,label,wrapper,args", CASES)
def test_generated_plist_invokes_wrapper_not_captured_interpreter(
    installer: str, label: str, wrapper: str, args: list[str], tmp_path: Path
) -> None:
    home, fake_py = _run_installer(installer, args, tmp_path)
    plist_path = home / "Library" / "LaunchAgents" / f"{label}.plist"
    assert plist_path.exists(), "installer did not write the plist into HOME"

    raw = plist_path.read_text(encoding="utf-8")
    data = plistlib.loads(plist_path.read_bytes())
    program_args = data["ProgramArguments"]
    command = program_args[-1]

    # Plist is well-formed and runs the repo-owned wrapper.
    assert program_args[:2] == ["/bin/bash", "-lc"]
    assert wrapper in command

    # No baked interpreter: neither an exported ARAGORA_PYTHON nor a captured
    # .venv interpreter nor the install-time interpreter path may appear.
    assert "ARAGORA_PYTHON=" not in command
    assert ".venv/bin/" + "python" not in raw
    assert str(fake_py) not in raw
    # No real user home leaked into the generated unit.
    assert "/Users/" not in raw


def _merge_arbiter_wrapper_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "scripts" / "run_merge_arbiter.sh", scripts / "run_merge_arbiter.sh")
    fake_python = _mk(
        repo / "fake-python",
        """#!/usr/bin/env bash
printf '%s\n' "$*" >> "${FAKE_PYTHON_LOG:?}"
case "$*" in
  *"scripts/auto_evidence_cycle.py"*)
    if [[ "${FAKE_AUTO_EVIDENCE_RC:-0}" != "0" ]]; then
      echo "fake auto-evidence failure" >&2
      exit "${FAKE_AUTO_EVIDENCE_RC}"
    fi
    ;;
esac
exit 0
""",
        executable=True,
    )
    _mk(
        scripts / "aragora_runtime.sh",
        f"#!/usr/bin/env bash\nresolve_aragora_python() {{ echo {fake_python}; }}\n",
        executable=True,
    )
    return repo, repo / "fake-python.log"


def test_merge_arbiter_starts_with_default_env(tmp_path: Path) -> None:
    repo, log = _merge_arbiter_wrapper_repo(tmp_path)
    proc = subprocess.run(
        ["/bin/bash", str(repo / "scripts" / "run_merge_arbiter.sh")],
        cwd=repo,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "FAKE_PYTHON_LOG": str(log)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "Starting swarm merge-arbiter" in proc.stdout
    recorded = log.read_text(encoding="utf-8")
    assert "scripts/auto_evidence_cycle.py" not in recorded
    assert "-u -m aragora.cli.main swarm merge-arbiter" in recorded


def test_merge_arbiter_skips_legacy_auto_evidence_without_override(
    tmp_path: Path,
) -> None:
    repo, log = _merge_arbiter_wrapper_repo(tmp_path)
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "FAKE_PYTHON_LOG": str(log),
        "ARAGORA_AUTO_EVIDENCE": "1",
    }
    proc = subprocess.run(
        ["/bin/bash", str(repo / "scripts" / "run_merge_arbiter.sh")],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "Skipping ARAGORA_AUTO_EVIDENCE=1" in proc.stderr
    assert "ARAGORA_ALLOW_LEGACY_AUTO_EVIDENCE_APPLY=1" in proc.stderr
    assert "No legacy evidence collection or posting will run" in proc.stderr
    assert "merge-quorum throughput may drop" in proc.stderr
    assert "Starting swarm merge-arbiter" in proc.stdout
    recorded = log.read_text(encoding="utf-8")
    assert "scripts/auto_evidence_cycle.py" not in recorded
    assert "-u -m aragora.cli.main swarm merge-arbiter" in recorded


def test_merge_arbiter_reports_legacy_auto_evidence_failure_with_override(
    tmp_path: Path,
) -> None:
    repo, log = _merge_arbiter_wrapper_repo(tmp_path)
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "FAKE_PYTHON_LOG": str(log),
        "ARAGORA_AUTO_EVIDENCE": "1",
        "ARAGORA_ALLOW_LEGACY_AUTO_EVIDENCE_APPLY": "1",
        "FAKE_AUTO_EVIDENCE_RC": "23",
    }
    proc = subprocess.run(
        ["/bin/bash", str(repo / "scripts" / "run_merge_arbiter.sh")],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "Auto-evidence cycle reported failures" in proc.stderr
    assert "Starting swarm merge-arbiter" in proc.stdout
    recorded = log.read_text(encoding="utf-8")
    assert "scripts/auto_evidence_cycle.py --apply --max-scan 40" in recorded
    assert "-u -m aragora.cli.main swarm merge-arbiter" in recorded


def test_merge_arbiter_legacy_override_runs_auto_evidence_step(tmp_path: Path) -> None:
    repo, log = _merge_arbiter_wrapper_repo(tmp_path)
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "FAKE_PYTHON_LOG": str(log),
        "ARAGORA_AUTO_EVIDENCE": "1",
        "ARAGORA_ALLOW_LEGACY_AUTO_EVIDENCE_APPLY": "1",
    }
    proc = subprocess.run(
        ["/bin/bash", str(repo / "scripts" / "run_merge_arbiter.sh")],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "Running bounded auto-evidence cycle" in proc.stdout
    assert "Starting swarm merge-arbiter" in proc.stdout
    recorded = log.read_text(encoding="utf-8")
    assert "scripts/auto_evidence_cycle.py --apply --max-scan 40" in recorded
    assert "-u -m aragora.cli.main swarm merge-arbiter" in recorded
