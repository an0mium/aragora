"""Contract tests for the maintained local mypy execution surfaces."""

from __future__ import annotations

from pathlib import Path
import re
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]


def _locked_mypy_version() -> str:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    package = next(item for item in lock["package"] if item["name"] == "mypy")
    return package["version"]


def test_precommit_mypy_pin_matches_locked_toolchain() -> None:
    locked_version = _locked_mypy_version()
    config = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    match = re.search(r"(?m)^\s+- mypy==(?P<version>[0-9.]+)\s*$", config)

    assert locked_version == "2.1.0"
    assert match is not None
    assert match.group("version") == locked_version


def test_pristine_health_uses_locked_dev_runtime() -> None:
    script = (REPO_ROOT / "scripts/pristine_main_health.py").read_text(encoding="utf-8")

    assert (
        'LOCKED_DEV_RUN = ("uv", "run", "--locked", "--extra", "dev", "--extra", "test")' in script
    )
    assert 'shutil.which("mypy")' not in script
