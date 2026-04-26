from __future__ import annotations

from pathlib import Path

from scripts.check_branch_mutation_policy import (
    check_repo,
    find_mutating_workflow_violations,
    find_strategy_default_violations,
)


def test_find_strategy_default_violations_detects_missing_patterns() -> None:
    files = {
        "scripts/codex_session.sh": "ENSURE_ARGS+=(--reconcile --strategy merge)",
    }
    violations = find_strategy_default_violations(files)
    assert violations
    assert any(v.path == "scripts/codex_session.sh" for v in violations)


def test_find_mutating_workflow_violations_rejects_unallowlisted_push() -> None:
    workflows = {
        "rogue.yml": "on:\n  workflow_dispatch:\njobs:\n  x:\n    steps:\n      - run: git push origin main",
    }
    violations = find_mutating_workflow_violations(workflows)
    assert violations
    assert "not in mutation allowlist" in violations[0].message


def test_find_mutating_workflow_violations_requires_testfixer_prefix() -> None:
    workflows = {
        "testfixer-auto.yml": (
            "on:\n  workflow_run:\n    workflows: [Test]\n"
            'jobs:\n  x:\n    steps:\n      - run: |\n          branch="autofix/tmp"\n          git push origin "$branch"'
        ),
    }
    violations = find_mutating_workflow_violations(workflows)
    assert violations
    assert "testfixer/*" in violations[0].message


def test_find_mutating_workflow_violations_requires_safe_benchmark_truth_publication() -> None:
    workflows = {
        "benchmark-truth-publication.yml": (
            "on:\n  pull_request:\n  workflow_dispatch:\n"
            "jobs:\n  x:\n    steps:\n      - run: |\n"
            '          branch="unsafe/tmp"\n'
            '          git push origin "$branch"\n'
        ),
    }
    violations = find_mutating_workflow_violations(workflows)
    assert violations
    assert any(
        "must not be triggered by pull_request/pull_request_target" in v.message for v in violations
    )
    assert any(
        "must push only to benchmark-truth-publication/* branch namespace" in v.message
        for v in violations
    )
    assert any("must create a draft pull request in-band" in v.message for v in violations)


def test_find_mutating_workflow_violations_requires_draft_benchmark_publication_pr() -> None:
    workflows = {
        "benchmark-truth-publication.yml": (
            "on:\n  workflow_dispatch:\n"
            "jobs:\n  x:\n    steps:\n      - run: |\n"
            '          branch="benchmark-truth-publication/123"\n'
            '          git push origin "$branch"\n'
            '          gh pr create --base main --head "$branch"\n'
        ),
    }
    violations = find_mutating_workflow_violations(workflows)
    assert violations
    assert any(
        "must create the benchmark publication pull request as a draft" in v.message
        for v in violations
    )


def test_repo_branch_mutation_policy_passes_for_current_tree() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    violations = check_repo(repo_root)
    assert violations == []
