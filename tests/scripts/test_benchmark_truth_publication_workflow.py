from __future__ import annotations

from pathlib import Path

import yaml


def _benchmark_truth_publication_workflow() -> dict[str, object]:
    workflow_path = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "benchmark-truth-publication.yml"
    )
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    if not isinstance(workflow, dict):
        raise AssertionError("benchmark-truth-publication workflow not found")
    return workflow


def _benchmark_truth_publication_run() -> str:
    workflow = _benchmark_truth_publication_workflow()
    jobs = workflow.get("jobs", {})
    publish_job = jobs.get("publish-benchmark-truth", {})
    steps = publish_job.get("steps", [])
    for step in steps:
        if step.get("name") == "Verify runtime prerequisites":
            return str(step.get("run", ""))
    raise AssertionError("Verify runtime prerequisites step not found")


def _benchmark_truth_publication_steps() -> list[dict[str, object]]:
    workflow = _benchmark_truth_publication_workflow()
    jobs = workflow.get("jobs", {})
    publish_job = jobs.get("publish-benchmark-truth", {})
    steps = publish_job.get("steps", [])
    if not isinstance(steps, list):
        raise AssertionError("publish-benchmark-truth steps not found")
    return [step for step in steps if isinstance(step, dict)]


def _workflow_on(workflow: dict[str, object]) -> dict[str, object]:
    on = workflow.get("on", workflow.get(True))
    if not isinstance(on, dict):
        raise AssertionError("workflow triggers not found")
    return on


def _workflow_step(name: str) -> dict[str, object]:
    for step in _benchmark_truth_publication_steps():
        if str(step.get("name", "")) == name:
            return step
    raise AssertionError(f"{name} step not found")


def test_runtime_prereq_creates_metrics_dir_and_allows_fresh_recurrence() -> None:
    run = _benchmark_truth_publication_run()
    assert 'METRICS_PATH=".aragora/overnight/boss_metrics.jsonl"' in run
    assert 'mkdir -p "$(dirname "$METRICS_PATH")"' in run
    assert "recurrence will generate a fresh window" in run
    assert 'test -f "$METRICS_PATH"' not in run


def test_installs_dependencies_before_recurrence() -> None:
    steps = _benchmark_truth_publication_steps()
    names = [str(step.get("name", "")) for step in steps]
    install_index = names.index("Install dependencies")
    recurrence_index = names.index("Refresh recurring benchmark corpus metrics")
    assert install_index < recurrence_index
    install_run = str(steps[install_index].get("run", ""))
    assert "python3 -m pip install --upgrade pip setuptools --quiet" in install_run
    assert 'python3 -m pip install -e ".[dev]" --quiet' in install_run


def test_installs_github_cli_before_runtime_prerequisites() -> None:
    steps = _benchmark_truth_publication_steps()
    names = [str(step.get("name", "")) for step in steps]
    gh_index = names.index("Install GitHub CLI")
    prereq_index = names.index("Verify runtime prerequisites")
    assert gh_index < prereq_index
    gh_step = steps[gh_index]
    gh_env = gh_step.get("env")
    assert gh_env == {
        "GH_CLI_VERSION": "2.89.0",
        "GITHUB_TOKEN": "${{ github.token }}",
    }
    gh_run = str(gh_step.get("run", ""))
    assert "https://api.github.com/repos/cli/cli/releases/latest" not in gh_run
    assert "installed_version=\"$(gh --version | awk 'NR==1 {print $3}')\"" in gh_run
    assert "Replacing preinstalled gh" in gh_run
    assert "https://api.github.com/repos/cli/cli/releases/tags/v${gh_version}" in gh_run
    assert 'asset.get("digest")' in gh_run
    assert "hashlib.sha256" in gh_run
    assert 'curl -fsSL -o "$archive_path"' in gh_run
    assert 'archive="gh_${gh_version}_${os}_${gh_arch}.tar.gz"' in gh_run
    assert 'echo "$gh_root/gh_${gh_version}_${os}_${gh_arch}/bin" >> "$GITHUB_PATH"' in gh_run


def test_installs_codex_cli_before_runtime_prerequisites() -> None:
    steps = _benchmark_truth_publication_steps()
    names = [str(step.get("name", "")) for step in steps]
    codex_index = names.index("Install Codex CLI")
    prereq_index = names.index("Verify runtime prerequisites")
    assert codex_index < prereq_index
    codex_run = str(steps[codex_index].get("run", ""))
    assert 'codex_root="$RUNNER_TEMP/codex-cli"' in codex_run
    assert 'npm install --global --prefix "$codex_root" @openai/codex' in codex_run
    assert 'echo "$codex_root/bin" >> "$GITHUB_PATH"' in codex_run


def test_resolves_codex_auth_paths_before_runner_refresh() -> None:
    steps = _benchmark_truth_publication_steps()
    names = [str(step.get("name", "")) for step in steps]
    resolve_index = names.index("Resolve Codex runner auth paths")
    refresh_index = names.index("Refresh execution-verified Codex runner")
    assert resolve_index < refresh_index

    run = str(_workflow_step("Resolve Codex runner auth paths").get("run", ""))
    assert 'RUNNER_USER="${ARAGORA_USER_ID:-${USER:-$(id -un)}}"' in run
    assert 'RUNNER_HOME="$(python3 -c' in run
    assert "pwd.getpwnam" in run
    assert "pwd.getpwall" in run
    assert 'mkdir -p "$RUNNER_HOME/.codex" "$RUNNER_HOME/.aragora"' in run
    assert 'echo "HOME=$RUNNER_HOME" >> "$GITHUB_ENV"' in run
    assert 'echo "CODEX_HOME=$RUNNER_HOME/.codex" >> "$GITHUB_ENV"' in run
    assert (
        'echo "ARAGORA_RUNNER_REGISTRY_PATH=$RUNNER_HOME/.aragora/swarm_runners.json" >> "$GITHUB_ENV"'
        in run
    )


def test_refreshes_execution_verified_codex_runner_before_recurrence() -> None:
    steps = _benchmark_truth_publication_steps()
    names = [str(step.get("name", "")) for step in steps]
    refresh_index = names.index("Refresh execution-verified Codex runner")
    recurrence_index = names.index("Refresh recurring benchmark corpus metrics")
    assert refresh_index < recurrence_index

    run = str(_workflow_step("Refresh execution-verified Codex runner").get("run", ""))
    assert "python3 -m aragora.cli.main swarm runner maintain" in run
    assert "--runner-type codex" in run
    assert "--probe-limit 1" in run
    assert 'RUNNER_REPORT="$RUNNER_TEMP/codex-runner-maintain.json"' in run
    assert "failed_runner = next(" in run
    assert "print(json.dumps(diagnostic, indent=2))" in run
    assert "routing blocked_reason=" in run
    assert 'payload.get("routing_after") or {}' in run
    assert "No execution-verified Codex runner selected after refresh." in run


def test_runs_daily_and_manual_dispatch() -> None:
    workflow = _benchmark_truth_publication_workflow()
    on = _workflow_on(workflow)
    assert on.get("workflow_dispatch") is None or on.get("workflow_dispatch") == {}
    schedule = on.get("schedule")
    assert isinstance(schedule, list)
    assert schedule == [{"cron": "20 13 * * *"}]


def test_publishes_refresh_via_branch_only_and_delegates_pr_creation() -> None:
    workflow = _benchmark_truth_publication_workflow()
    permissions = workflow.get("permissions")
    assert permissions == {
        "contents": "write",
        "issues": "write",
        "pull-requests": "read",
    }

    publish_run = str(_workflow_step("Publish tracked trust-loop surfaces").get("run", ""))
    run = str(
        _workflow_step("Commit and publish refreshed trust-loop surfaces branch").get("run", "")
    )
    assert "--freshness-map docs/benchmarks/benchmark_corpus_freshness.json \\" in publish_run
    assert "--ensure-issues \\" in publish_run
    assert 'branch="benchmark-truth-publication/${GITHUB_RUN_ID}"' in run
    assert 'git checkout -b "$branch"' in run
    assert 'git push origin "$branch"' in run
    assert 'git commit -m "${title}"' in run
    assert "[skip ci]" not in run
    assert "gh pr create" not in run
    assert "git push origin HEAD:main" not in run
