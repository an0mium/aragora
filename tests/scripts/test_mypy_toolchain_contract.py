"""Pin the baseline-producing mypy toolchain across every execution surface."""

from __future__ import annotations

from pathlib import Path
import re
import tomllib

from packaging.requirements import Requirement
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLCHAIN = {
    "mypy": "2.3.0",
    "mypy-baseline": "0.7.4",
    "types-croniter": "6.2.2.20260518",
    "types-jsonschema": "4.26.0.20260518",
    "types-pyyaml": "6.0.12.20260518",
    "types-python-dateutil": "2.9.0.20260716",
    "types-redis": "4.6.0.20241004",
    "types-requests": "2.33.0.20260712",
    "types-setuptools": "83.0.0.20260716",
}


def _pins(requirements: list[str]) -> dict[str, str]:
    pins: dict[str, str] = {}
    for value in requirements:
        if value.startswith("-"):
            continue
        requirement = Requirement(value)
        if requirement.name.lower() in TOOLCHAIN:
            pins[requirement.name.lower()] = str(requirement.specifier)
    return pins


def _expected_pins() -> dict[str, str]:
    return {name: f"=={version}" for name, version in TOOLCHAIN.items()}


def _shell_array_values(text: str, name: str) -> list[str]:
    match = re.search(rf"{name}=\(\n(?P<body>.*?)\n\)", text, re.DOTALL)
    assert match is not None
    return re.findall(r'^\s*"([^"]+)"', match.group("body"), re.MULTILINE)


def test_pyproject_and_lock_pin_the_canonical_toolchain() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert _pins(project["project"]["optional-dependencies"]["dev"]) == _expected_pins()

    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    versions = {
        package["name"].lower(): package["version"]
        for package in lock["package"]
        if package["name"].lower() in TOOLCHAIN
    }
    assert versions == TOOLCHAIN

    aragora = next(package for package in lock["package"] if package["name"] == "aragora")
    declared = [
        item["name"] + item.get("specifier", "") for item in aragora["metadata"]["requires-dist"]
    ]
    assert _pins(declared) == _expected_pins()


def test_precommit_delegates_to_the_locked_toolchain() -> None:
    config = yaml.load(
        (REPO_ROOT / ".pre-commit-config.yaml").read_text(),
        Loader=yaml.BaseLoader,
    )
    hooks = [hook for repo in config["repos"] for hook in repo["hooks"]]
    typecheck = next(hook for hook in hooks if hook["id"] == "typecheck-changed")
    assert typecheck["language"] == "system"
    assert typecheck["require_serial"] == "true"
    assert "additional_dependencies" not in typecheck
    assert "uv run --locked --extra dev bash scripts/test_tiers.sh typecheck" in typecheck["entry"]
    assert "uv run --locked --extra dev mypy" in typecheck["entry"]


def test_legacy_installer_pins_the_canonical_toolchain() -> None:
    installer = (REPO_ROOT / "scripts" / "ci_install_project.sh").read_text()
    dependencies = _shell_array_values(installer, "LEGACY_CONTROL_PLANE_DEV_DEPS")
    assert _pins(dependencies) == _expected_pins()


def test_workflow_typecheck_jobs_use_the_locked_toolchain() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "lint.yml").read_text()
    uv_install_lines = [
        line.strip()
        for line in workflow.splitlines()
        if "-m pip install" in line and "uv==0.11.19" in line
    ]
    assert len(uv_install_lines) == 2
    assert workflow.count("uv sync --locked --extra dev") == 2
    assert workflow.count("uv run --locked --extra dev") >= 3
    assert re.search(r"\bpip install(?: --user)? mypy(?:\s|$)", workflow) is None
