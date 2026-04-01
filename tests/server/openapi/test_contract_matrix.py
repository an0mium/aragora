"""SDK/OpenAPI contract verification smoke tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    root = Path(__file__).resolve().parents[3]
    if (root / "pyproject.toml").exists():
        return root
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists():
        return cwd
    return root


def test_verify_sdk_contracts_strict_passes() -> None:
    """Contract drift should be evaluated through the maintained verifier script."""
    repo = _repo_root()
    completed = subprocess.run(
        [sys.executable, "scripts/verify_sdk_contracts.py", "--strict"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "verify_sdk_contracts.py --strict failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
