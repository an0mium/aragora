from __future__ import annotations

from pathlib import Path

from scripts.check_required_check_priority_policy import (
    check_repo,
    find_required_check_priority_violations,
)


def _valid_workflow_text() -> str:
    return """
jobs:
  prioritize-required-checks:
    steps:
      - name: Cancel non-required workflow runs for this PR head
        uses: actions/github-script@v7
        with:
          script: |
            const alwaysKeepWorkflowPaths = new Set([
              '.github/workflows/aragora-review-gate.yml',
              '.github/workflows/autopilot-worktree-e2e.yml',
              '.github/workflows/core-suites.yml',
              '.github/workflows/lint.yml',
              '.github/workflows/live-deploy-mode-gate.yml',
              '.github/workflows/sdk-parity.yml',
              '.github/workflows/sdk-test.yml',
              '.github/workflows/test.yml',
              '.github/workflows/openapi.yml',
              '.github/workflows/pr-admission-controller.yml',
              '.github/workflows/quality-smoke.yml',
              '.github/workflows/release-readiness.yml',
              '.github/workflows/security-gate.yml',
              '.github/workflows/smoke.yml',
              '.github/workflows/required-check-priority.yml',
            ]);
            const alwaysKeepWorkflowNames = new Set([
              'Aragora Code Review',
              'Autopilot Worktree E2E',
              'Core Suites (Decision Integrity)',
              'Live Deploy Mode Gate',
              'PR Admission Controller',
              'Quality Pipeline Smoke',
              'Required Check Priority',
              'Release Readiness Gate',
              'Security Gate',
              'Lint',
              'SDK Parity Check',
              'SDK Tests',
              'Smoke Tests',
              'Tests',
              'OpenAPI Spec',
            ]);
"""


def test_policy_accepts_required_keep_entries() -> None:
    violations = find_required_check_priority_violations(_valid_workflow_text())
    assert violations == []


def test_policy_requires_required_keep_workflow_path() -> None:
    text = _valid_workflow_text().replace(
        ".github/workflows/openapi.yml", ".github/workflows/other.yml"
    )
    violations = find_required_check_priority_violations(text)
    assert violations
    assert any(
        "missing required keep workflow path: .github/workflows/openapi.yml" == v
        for v in violations
    )


def test_policy_requires_context_mapped_workflow_path_in_keep_list() -> None:
    text = _valid_workflow_text().replace(
        ".github/workflows/lint.yml", ".github/workflows/lint-alt.yml"
    )
    violations = find_required_check_priority_violations(text)
    assert violations
    assert any(
        "required context `lint` maps to workflow path not in keep-list: .github/workflows/lint.yml"
        == v
        for v in violations
    )


def test_policy_requires_required_keep_workflow_name() -> None:
    text = _valid_workflow_text().replace("SDK Tests", "SDK Smoke")
    violations = find_required_check_priority_violations(text)
    assert violations
    assert any("missing required keep workflow name: SDK Tests" == v for v in violations)


def test_policy_detects_stale_workflow_paths() -> None:
    text = _valid_workflow_text().replace(
        ".github/workflows/autopilot-worktree-e2e.yml",
        ".github/workflows/does-not-exist.yml",
    )
    repo_root = Path(__file__).resolve().parents[2]
    violations = find_required_check_priority_violations(text, repo_root=repo_root)
    assert violations
    assert any("does not exist" in v for v in violations)


def test_policy_detects_missing_context_marker_in_mapped_workflow(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    wf_dir = repo_root / ".github" / "workflows"
    wf_dir.mkdir(parents=True)

    (wf_dir / "lint.yml").write_text(
        "name: Lint\njobs:\n  lint:\n    runs-on: ubuntu-latest\n", encoding="utf-8"
    )
    (wf_dir / "aragora-review-gate.yml").write_text(
        "name: Aragora Code Review\njobs:\n  aragora-review:\n    runs-on: ubuntu-latest\n",
        encoding="utf-8",
    )
    (wf_dir / "autopilot-worktree-e2e.yml").write_text(
        "name: Autopilot Worktree E2E\njobs:\n  scope:\n    name: Autopilot Scope\n",
        encoding="utf-8",
    )
    (wf_dir / "core-suites.yml").write_text(
        "name: Core Suites (Decision Integrity)\njobs:\n  core:\n    runs-on: ubuntu-latest\n",
        encoding="utf-8",
    )
    (wf_dir / "sdk-parity.yml").write_text(
        "name: SDK Parity Check\njobs:\n  sdk-parity:\n    runs-on: ubuntu-latest\n",
        encoding="utf-8",
    )
    (wf_dir / "sdk-test.yml").write_text(
        "name: SDK Tests\njobs:\n  typescript-sdk:\n    name: TypeScript SDK Type Check\n",
        encoding="utf-8",
    )
    (wf_dir / "test.yml").write_text(
        "name: Tests\njobs:\n  python-tests:\n    runs-on: ubuntu-latest\n",
        encoding="utf-8",
    )
    (wf_dir / "openapi.yml").write_text(
        "name: OpenAPI Spec\njobs:\n  generate:\n    name: Generate & Validate\n",
        encoding="utf-8",
    )
    (wf_dir / "live-deploy-mode-gate.yml").write_text(
        "name: Live Deploy Mode Gate\njobs:\n  gate:\n    name: Validate Live Deploy Mode\n",
        encoding="utf-8",
    )
    (wf_dir / "pr-admission-controller.yml").write_text(
        "name: PR Admission Controller\njobs:\n  enforce:\n    name: PR Admission Signal (Advisory)\n",
        encoding="utf-8",
    )
    (wf_dir / "quality-smoke.yml").write_text(
        "name: Quality Pipeline Smoke\njobs:\n  quality:\n    runs-on: ubuntu-latest\n",
        encoding="utf-8",
    )
    (wf_dir / "required-check-priority.yml").write_text(
        "name: Required Check Priority\n", encoding="utf-8"
    )
    (wf_dir / "release-readiness.yml").write_text(
        "name: Release Readiness Gate\njobs:\n  release:\n    name: Release Readiness\n",
        encoding="utf-8",
    )
    (wf_dir / "security-gate.yml").write_text(
        "name: Security Gate\njobs:\n  summary:\n    name: Security Gate Summary\n",
        encoding="utf-8",
    )
    (wf_dir / "smoke.yml").write_text(
        "name: Smoke Tests\njobs:\n  smoke:\n    name: Smoke Tests\n",
        encoding="utf-8",
    )

    text = """
jobs:
  prioritize-required-checks:
    steps:
      - name: Cancel non-required workflow runs for this PR head
        uses: actions/github-script@v7
        with:
          script: |
            const alwaysKeepWorkflowPaths = new Set([
              '.github/workflows/aragora-review-gate.yml',
              '.github/workflows/autopilot-worktree-e2e.yml',
              '.github/workflows/core-suites.yml',
              '.github/workflows/lint.yml',
              '.github/workflows/live-deploy-mode-gate.yml',
              '.github/workflows/sdk-parity.yml',
              '.github/workflows/sdk-test.yml',
              '.github/workflows/test.yml',
              '.github/workflows/openapi.yml',
              '.github/workflows/pr-admission-controller.yml',
              '.github/workflows/quality-smoke.yml',
              '.github/workflows/required-check-priority.yml',
              '.github/workflows/release-readiness.yml',
              '.github/workflows/security-gate.yml',
              '.github/workflows/smoke.yml',
            ]);
            const alwaysKeepWorkflowNames = new Set([
              'Aragora Code Review',
              'Autopilot Worktree E2E',
              'Core Suites (Decision Integrity)',
              'Live Deploy Mode Gate',
              'PR Admission Controller',
              'Quality Pipeline Smoke',
              'Required Check Priority',
              'Release Readiness Gate',
              'Security Gate',
              'Lint',
              'SDK Parity Check',
              'SDK Tests',
              'Smoke Tests',
              'Tests',
              'OpenAPI Spec',
            ]);
"""
    violations = find_required_check_priority_violations(text, repo_root=repo_root)
    assert violations
    assert any(
        "required context marker `typecheck` not found in mapped workflow: .github/workflows/lint.yml"
        == v
        for v in violations
    )


def test_repo_required_check_priority_policy_passes_for_current_tree() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    violations = check_repo(repo_root)
    assert violations == []
