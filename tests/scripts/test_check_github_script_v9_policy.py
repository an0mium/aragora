from __future__ import annotations

from pathlib import Path

from scripts.check_github_script_v9_policy import (
    check_repo,
    find_github_script_v9_violations,
)


def test_detects_actions_github_require_in_v9_script() -> None:
    text = """
jobs:
  audit:
    steps:
      - uses: actions/github-script@v9
        with:
          script: |
            const { getOctokit } = require('@actions/github');
"""

    violations = find_github_script_v9_violations(text)

    assert len(violations) == 2
    assert "getOctokit" in violations[0][2]
    assert "@actions/github" in violations[1][2]


def test_detects_get_octokit_let_redeclaration_in_v9_script() -> None:
    text = """
jobs:
  audit:
    steps:
      - uses: actions/github-script@v9
        with:
          script: |
            let getOctokit = github.getOctokit;
"""

    violations = find_github_script_v9_violations(text)

    assert len(violations) == 1
    assert "injected getOctokit" in violations[0][2]


def test_detects_actions_github_internal_import_in_v9_script() -> None:
    text = """
jobs:
  audit:
    steps:
      - uses: actions/github-script@v9
        with:
          script: |
            const utils = require('@actions/github/lib/utils');
"""

    violations = find_github_script_v9_violations(text)

    assert len(violations) == 1
    assert "internals" in violations[0][2]


def test_allows_node_builtin_require_and_v7_composite_script() -> None:
    text = """
jobs:
  audit:
    steps:
      - uses: actions/github-script@v9
        with:
          script: |
            const fs = require('fs');
            await github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: fs.readFileSync('summary.md', 'utf8'),
            });
      - uses: actions/github-script@v7
        with:
          script: |
            const { getOctokit } = require('@actions/github');
"""

    violations = find_github_script_v9_violations(text)

    assert violations == []


def test_repo_github_script_v9_policy_passes_for_current_tree() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    violations = check_repo(repo_root)

    assert violations == []
