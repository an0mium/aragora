"""Regression contract for the future required fast gate and its companion."""

from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path
import re
import subprocess

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github/workflows"
FAST_SHARDS = {
    "agents",
    "reasoning",
    "workflow-client",
    "knowledge",
    "storage",
    "infra",
    "server-handlers-am",
    "server-handlers-nz",
    "server-rest",
    "handlers-features",
    "handlers-amk-no-features",
    "handlers-lz",
}
DEBATE_SHARDS = {"debate-phases", "debate-1", "debate-2", "debate-3"}
CURRENT_CONTEXTS = [
    "lint",
    "typecheck",
    "sdk-parity",
    "Generate & Validate",
    "TypeScript SDK Type Check",
    "aragora-merge-quorum",
]


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text())


def _triggers(workflow: dict) -> dict:
    # PyYAML's YAML 1.1 loader treats the unquoted GitHub Actions key as True.
    return workflow.get("on", workflow.get(True))


def test_fast_gate_dependencies_timeout_and_shared_shards() -> None:
    jobs = _workflow("test.yml")["jobs"]
    worker = jobs["test-fast-gate-run"]
    assert worker["needs"] == ["test-shard-scope"]
    assert worker["timeout-minutes"] == 10
    assert worker["if"] == jobs["test-fast"]["if"]
    assert "!github.event.pull_request.draft" in worker["if"]
    assert worker["strategy"]["fail-fast"] is False
    assert worker["strategy"]["matrix"] == jobs["test-fast"]["strategy"]["matrix"]
    assert worker["steps"] == jobs["test-fast"]["steps"]
    assert {c["name"] for c in worker["strategy"]["matrix"]["category"]} == FAST_SHARDS
    assert "continue-on-error" not in worker
    assert all(not step.get("continue-on-error") for step in worker["steps"])
    gate = jobs["test-fast-gate"]
    assert gate["needs"] == ["test-fast-gate-run"]
    assert gate["if"] == "always()"
    assert not gate.get("continue-on-error")


@pytest.mark.parametrize("result", ["failure", "cancelled", "success", "skipped", "", "unknown"])
def test_umbrella_fails_closed_offline(result: str) -> None:
    gate = _workflow("test.yml")["jobs"]["test-fast-gate"]
    step = next(s for s in gate["steps"] if "WORKER_RESULT" in s.get("env", {}))
    assert step["env"]["WORKER_RESULT"] == "${{ needs.test-fast-gate-run.result }}"
    assert "${{" not in step["run"]
    assert not step.get("continue-on-error")
    outcome = subprocess.run(
        ["bash", "-c", step["run"]],
        env={**os.environ, "WORKER_RESULT": result},
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert (outcome.returncode == 0) == (result in {"success", "skipped"}), outcome.stdout


def test_baseline_decoupling_and_all_dependencies_resolve() -> None:
    jobs = _workflow("test.yml")["jobs"]
    assert "baseline-determinism" in jobs
    assert jobs["test-fast"]["needs"] == ["test-shard-scope"]
    for job in jobs.values():
        needs = job.get("needs", [])
        assert set([needs] if isinstance(needs, str) else needs) <= jobs.keys()


def test_debate_shards_move_to_main_and_nightly_only() -> None:
    workflow = _workflow("test-debate-shards.yml")
    assert workflow["name"] == "Tests (debate shards)"
    triggers = _triggers(workflow)
    assert set(triggers) == {"push", "schedule", "workflow_dispatch"}
    assert triggers["push"]["branches"] == ["main"]
    assert triggers["schedule"] == [{"cron": "0 4 * * *"}]
    jobs = workflow["jobs"]
    assert len(jobs) == 1
    job = next(iter(jobs.values()))
    categories = job["strategy"]["matrix"]["category"]
    assert {c["name"] for c in categories} == DEBATE_SHARDS
    assert {c["resolver"] for c in categories} == DEBATE_SHARDS
    scripts = "\n".join(s.get("run", "") for s in job["steps"])
    assert "scripts/ci_resolve_test_shard.py" in scripts
    assert "--check" in scripts
    assert "pytest" in scripts
    assert not job.get("continue-on-error")
    for test_job in _workflow("test.yml")["jobs"].values():
        for categories in test_job.get("strategy", {}).get("matrix", {}).values():
            if isinstance(categories, list):
                assert not any(
                    isinstance(c, dict) and c.get("name") in DEBATE_SHARDS for c in categories
                )


def test_companion_inverts_paths_and_always_reports_success() -> None:
    primary_pr = _triggers(_workflow("test.yml"))["pull_request"]
    companion = _workflow("test-fast-gate-companion.yml")
    triggers = _triggers(companion)
    assert set(triggers) == {"pull_request"}
    assert triggers["pull_request"] == {
        "types": primary_pr["types"],
        "branches": primary_pr["branches"],
        "paths-ignore": primary_pr["paths"],
    }
    assert list(companion["jobs"]) == ["test-fast-gate"]
    gate = companion["jobs"]["test-fast-gate"]
    assert gate.get("name", "test-fast-gate") == "test-fast-gate"
    assert "if" not in gate
    assert not gate.get("needs")
    assert len(gate["steps"]) == 1
    assert "uses" not in gate["steps"][0]  # No checkout or token needed for the no-op.
    outcome = subprocess.run(
        ["bash", "-c", gate["steps"][0]["run"]],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert outcome.returncode == 0
    for files, expected in [
        (["README.md"], (False, True)),
        (["docs/CI_LANES.md", "docs-site/README.md"], (False, True)),
        (["aragora/debate/orchestrator.py"], (True, False)),
        (["aragora/debate/orchestrator.py", "README.md"], (True, True)),
    ]:
        matched = [any(fnmatch.fnmatch(f, p) for p in primary_pr["paths"]) for f in files]
        assert (any(matched), not all(matched)) == expected, files


def test_existing_triggers_and_concurrency_are_preserved() -> None:
    workflow = _workflow("test.yml")
    triggers = _triggers(workflow)
    assert set(triggers) == {"pull_request", "schedule", "workflow_dispatch"}
    assert triggers["schedule"] == [{"cron": "0 4 * * *"}]
    assert workflow["concurrency"] == {
        "group": (
            "tests-${{ github.event.pull_request.number || github.ref }}"
            "${{ startsWith(github.ref, 'refs/heads/dev/') && format('-{0}', github.sha) || '' }}"
        ),
        "cancel-in-progress": (
            "${{ github.event_name == 'pull_request' && github.event.action == 'synchronize' }}"
        ),
    }


def test_required_manifest_and_auto_revert_references_remain_valid() -> None:
    manifest = json.loads((ROOT / "scripts/ci/required_workflow_manifest.json").read_text())
    assert "Tests" in manifest["workflow_names"]
    assert ".github/workflows/test.yml" in manifest["workflow_paths"]
    names = {_workflow(p.name)["name"] for p in WORKFLOWS.glob("*.yml")}
    revert = _triggers(_workflow("main-required-checks-auto-revert.yml"))
    assert set(revert["workflow_run"]["workflows"]) <= names
    source = (ROOT / "scripts/auto_revert_main_required_failures.py").read_text()
    assert "def get_required_contexts(" in source
    assert 'self.get(f"/repos/{self.repo}/branches/{branch}/protection")' in source
    assert "return list(DEFAULT_REQUIRED_CONTEXTS)" in source


@pytest.mark.parametrize(
    "workflow_name,job_id",
    [("test.yml", "test-fast-gate-run"), ("test-debate-shards.yml", "test-debate-shards")],
)
def test_added_workers_pin_checkout_and_run_tests(workflow_name: str, job_id: str) -> None:
    steps = _workflow(workflow_name)["jobs"][job_id]["steps"]
    checkout = next(s for s in steps if s.get("uses", "").startswith("actions/checkout@"))
    assert re.fullmatch(r"actions/checkout@[0-9a-f]{40}", checkout["uses"])
    assert checkout["with"]["persist-credentials"] is False
    for step in steps:
        if step.get("uses", "").startswith("actions/upload-artifact@"):
            assert re.fullmatch(r"actions/upload-artifact@[0-9a-f]{40}", step["uses"])
    run_tests = next(
        s["run"]
        for s in steps
        if "pytest ${{ steps.shard_paths.outputs.args }}" in s.get("run", "")
    )
    for flag in ("-n 4", "--timeout=120", "--dist loadscope", "not slow", "not integration"):
        assert flag in run_tests


def test_docs_keep_gate_future_and_preserve_exact_patch_payload() -> None:
    docs = (ROOT / "docs/CI_LANES.md").read_text()
    assert "FUTURE required check" in docs
    assert "at most 10 minutes" in docs
    assert "test-fast-gate-companion.yml" in docs
    assert "test-debate-shards.yml" in docs
    assert "M10" in docs and "M11" in docs
    assert (
        "gh api -X PATCH repos/synaptent/aragora/branches/main/protection/"
        "required_status_checks --input /tmp/aragora-readiness/required-checks.json"
    ) in docs
    payload = json.loads(re.search(r"```json\n(.*?)\n```", docs, flags=re.DOTALL).group(1))
    assert payload == {
        "strict": False,
        "contexts": [*CURRENT_CONTEXTS, "test-fast-gate"],
        "checks": [{"context": c, "app_id": 15368} for c in [*CURRENT_CONTEXTS, "test-fast-gate"]],
    }
