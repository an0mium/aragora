from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
LIVE_PACKAGE_DIR = ROOT / "aragora" / "live"

NODE_ENGINE_RANGE = ">=24.18.0 <25"
LIVE_NODE_VERSION = "24.18.0"

EXPECTED_SELECTOR_VERSIONS = {
    ("deploy-frontend.yml", "build-standalone"): LIVE_NODE_VERSION,
    ("deploy-frontend.yml", "deploy-vercel"): LIVE_NODE_VERSION,
    ("deploy-secure.yml", "deploy-vercel"): LIVE_NODE_VERSION,
    ("e2e.yml", "e2e"): LIVE_NODE_VERSION,
    ("e2e.yml", "visual-regression"): LIVE_NODE_VERSION,
    ("integration-gate.yml", "frontend-build"): LIVE_NODE_VERSION,
    ("lighthouse.yml", "lighthouse"): LIVE_NODE_VERSION,
    ("lighthouse.yml", "accessibility-audit"): LIVE_NODE_VERSION,
    ("lighthouse.yml", "mobile-responsiveness"): LIVE_NODE_VERSION,
    ("lint.yml", "frontend-lint"): LIVE_NODE_VERSION,
    ("live-deploy-mode-gate.yml", "gate"): LIVE_NODE_VERSION,
    ("merge-group-frontend-typecheck.yml", "frontend-typecheck"): LIVE_NODE_VERSION,
    ("nightly-full-matrix.yml", "frontend-build"): LIVE_NODE_VERSION,
    ("onramp-integration.yml", "playground-build"): LIVE_NODE_VERSION,
    ("production-monitor.yml", "production-health"): LIVE_NODE_VERSION,
    ("production-monitor.yml", "production-auth"): LIVE_NODE_VERSION,
    ("production-monitor.yml", "production-full"): LIVE_NODE_VERSION,
    ("release.yml", "sdk-parity-gate"): "20",
    ("release.yml", "security-gate"): LIVE_NODE_VERSION,
    ("release.yml", "frontend-build"): LIVE_NODE_VERSION,
    ("test.yml", "version-check"): "20",
    ("test.yml", "sdk-typescript-compile-gate"): "20",
    ("test.yml", "frontend"): LIVE_NODE_VERSION,
    ("test.yml", "frontend-typecheck"): LIVE_NODE_VERSION,
}

NON_LIVE_SELECTORS = {
    ("release.yml", "sdk-parity-gate"),
    ("test.yml", "version-check"),
    ("test.yml", "sdk-typescript-compile-gate"),
}

APPROVED_WORKFLOWS = {workflow_name for workflow_name, _job_name in EXPECTED_SELECTOR_VERSIONS}


def _load_workflow(workflow_name: str) -> dict[str, Any]:
    workflow = yaml.safe_load((WORKFLOWS_DIR / workflow_name).read_text(encoding="utf-8"))
    assert isinstance(workflow, dict)
    return workflow


def _workflow_triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    """Return triggers despite PyYAML 1.1 parsing bare ``on`` as ``True``."""
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    return triggers


def _setup_node_steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for step in job.get("steps", []) if step.get("uses") == "actions/setup-node@v4"]


def _resolve_node_version(
    workflow: dict[str, Any], job: dict[str, Any], step: dict[str, Any]
) -> str:
    raw_version = step.get("with", {}).get("node-version")
    if raw_version != "${{ env.NODE_VERSION }}":
        return str(raw_version)

    environment = {
        **(workflow.get("env") or {}),
        **(job.get("env") or {}),
    }
    assert "NODE_VERSION" in environment
    return str(environment["NODE_VERSION"])


def _cache_dependency_paths(step: dict[str, Any]) -> list[str]:
    value = step.get("with", {}).get("cache-dependency-path")
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return str(value).splitlines()


def _job_operates_on_live(job: dict[str, Any]) -> bool:
    for step in job.get("steps", []):
        working_directory = str(step.get("working-directory", ""))
        if working_directory == "aragora/live" or working_directory.startswith("aragora/live/"):
            return True
        if "aragora/live/package-lock.json" in _cache_dependency_paths(step):
            return True
    return False


def test_package_and_lockfile_declare_the_same_node_engine() -> None:
    package = json.loads((LIVE_PACKAGE_DIR / "package.json").read_text(encoding="utf-8"))
    lockfile = json.loads((LIVE_PACKAGE_DIR / "package-lock.json").read_text(encoding="utf-8"))

    assert package["engines"]["node"] == NODE_ENGINE_RANGE
    assert lockfile["packages"][""]["engines"]["node"] == NODE_ENGINE_RANGE


def test_approved_workflows_own_the_complete_node_selector_inventory() -> None:
    assert APPROVED_WORKFLOWS == {
        "deploy-frontend.yml",
        "deploy-secure.yml",
        "e2e.yml",
        "integration-gate.yml",
        "lighthouse.yml",
        "lint.yml",
        "live-deploy-mode-gate.yml",
        "merge-group-frontend-typecheck.yml",
        "nightly-full-matrix.yml",
        "onramp-integration.yml",
        "production-monitor.yml",
        "release.yml",
        "test.yml",
    }

    actual_versions: dict[tuple[str, str], str] = {}
    live_selectors: set[tuple[str, str]] = set()
    for workflow_name in APPROVED_WORKFLOWS:
        workflow = _load_workflow(workflow_name)
        for job_name, job in workflow["jobs"].items():
            setup_node_steps = _setup_node_steps(job)
            assert len(setup_node_steps) <= 1, (
                f"{workflow_name}:{job_name} has multiple setup-node selectors"
            )
            if not setup_node_steps:
                continue

            selector = (workflow_name, job_name)
            actual_versions[selector] = _resolve_node_version(workflow, job, setup_node_steps[0])
            if _job_operates_on_live(job):
                live_selectors.add(selector)

    assert actual_versions == EXPECTED_SELECTOR_VERSIONS
    assert live_selectors == set(EXPECTED_SELECTOR_VERSIONS) - NON_LIVE_SELECTORS


def test_live_deploy_mode_gate_builds_package_changes_on_node_24() -> None:
    workflow = _load_workflow("live-deploy-mode-gate.yml")
    triggers = _workflow_triggers(workflow)
    assert "aragora/live/**" in triggers["pull_request"]["paths"]

    scope_steps = workflow["jobs"]["scope"]["steps"]
    path_filter = next(step for step in scope_steps if step.get("uses") == "dorny/paths-filter@v3")
    assert "aragora/live/package*.json" in path_filter["with"]["filters"]

    gate = workflow["jobs"]["gate"]
    setup_node = _setup_node_steps(gate)
    assert len(setup_node) == 1
    assert _resolve_node_version(workflow, gate, setup_node[0]) == LIVE_NODE_VERSION
    assert any(
        step.get("working-directory") == "aragora/live" and "npm ci" in step.get("run", "")
        for step in gate["steps"]
    )
    assert any(
        step.get("working-directory") == "aragora/live" and "npm run build" in step.get("run", "")
        for step in gate["steps"]
    )
