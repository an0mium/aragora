"""Package gates must scan real code without inheriting root type exemptions."""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("package", ["debate", "verify"])
def test_package_ruff_extends_root_with_strict_rules(package: str) -> None:
    root = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["ruff"]
    local = tomllib.loads((ROOT / f"aragora-{package}/pyproject.toml").read_text())["tool"]
    assert f"aragora-{package}*" not in root.get("exclude", [])
    assert local["ruff"]["extend"] == "../pyproject.toml"
    assert {"N", "C901"} <= set(local["ruff"]["lint"]["select"])
    assert local["ruff"]["lint"]["mccabe"]["max-complexity"] == 10
    assert set(root["lint"]["ignore"]) <= set(local["ruff"]["lint"]["ignore"])


@pytest.mark.parametrize("package", ["debate", "verify"])
def test_package_typecheck_uses_local_strict_config_and_pin(package: str) -> None:
    config = tomllib.loads((ROOT / f"aragora-{package}/pyproject.toml").read_text())["tool"]
    assert config["mypy"]["strict"] is True
    for override in config["mypy"].get("overrides", []):
        assert not override.get("ignore_errors")
        assert override.get("disallow_untyped_defs", True)
        assert not override.get("allow_untyped_defs")
        assert "no-untyped-def" not in override.get("disable_error_code", [])
    result = subprocess.run(
        ["make", "-n", f"readiness-typecheck-{package}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "mypy --strict src" in result.stdout
    assert "cd aragora-" + package in result.stdout
    assert '"2.1.0"' in result.stdout
    assert "strict mypy lands in M3" not in result.stdout


def test_root_no_longer_exempts_package_modules() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["mypy"]
    modules = {
        module
        for override in config["overrides"]
        for module in override["module"]
        if override.get("disallow_untyped_defs") is False
    }
    assert not modules & {
        "aragora_debate._mock",
        "aragora_debate.styled_mock",
        "aragora_verify.verifier",
    }
