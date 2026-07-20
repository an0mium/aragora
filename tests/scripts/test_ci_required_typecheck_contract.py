"""Keep the local required-check target aligned with the CI typecheck path."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
TYPECHECK_COMMAND = "bash scripts/test_tiers.sh typecheck"


def _make_target_commands(target: str) -> list[str]:
    lines = (ROOT / "Makefile").read_text(encoding="utf-8").splitlines()
    start = lines.index(f"{target}:")
    commands: list[str] = []

    for line in lines[start + 1 :]:
        if line.startswith("\t"):
            commands.append(line.removeprefix("\t").strip())
            continue
        if line and not line.lstrip().startswith("#"):
            break

    return commands


def _workflow_full_typecheck_command() -> str:
    workflow = yaml.safe_load((ROOT / ".github/workflows/lint.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["typecheck-run"]["steps"]
    step = next(item for item in steps if item.get("name") == "Run full typecheck tier")
    return str(step["run"])


def test_ci_required_uses_the_workflow_typecheck_command() -> None:
    commands = _make_target_commands("ci-required")
    workflow_command = _workflow_full_typecheck_command()

    assert commands.count(TYPECHECK_COMMAND) == 1
    assert TYPECHECK_COMMAND in workflow_command


def test_ci_required_does_not_bypass_the_typecheck_tier() -> None:
    commands = _make_target_commands("ci-required")
    direct_mypy = re.compile(r"(^|\s)(?:python\s+-m\s+)?mypy(?:\s|$)")

    assert not any(direct_mypy.search(command) for command in commands)
