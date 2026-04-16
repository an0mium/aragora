from __future__ import annotations

from pathlib import Path

from scripts.check_workflow_checkout_integrity_policy import check_repo


ARCHIVE_RESTORE_BLOCK = """
name: "Verify checkout integrity"
description: "Ensures checkout is complete, recovers from sparse-checkout corruption"
inputs:
  checkout-path:
    description: "Relative path containing the checked out repository"
    required: false
    default: "."
runs:
  using: "composite"
  steps:
    - name: Verify checkout integrity
      shell: bash
      run: |
        git sparse-checkout disable || true
        git archive "${GITHUB_SHA:-HEAD}" | tar -x || git archive HEAD | tar -x
"""


def _write_checkout_action(repo_root: Path, text: str = ARCHIVE_RESTORE_BLOCK) -> None:
    action_path = repo_root / ".github/actions/checkout-integrity/action.yml"
    action_path.parent.mkdir(parents=True, exist_ok=True)
    action_path.write_text(text, encoding="utf-8")


def _write_workflow(repo_root: Path, name: str, text: str) -> None:
    workflow_path = repo_root / ".github/workflows" / name
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_path.write_text(text, encoding="utf-8")


def test_detects_self_hosted_checkout_without_guard(tmp_path: Path) -> None:
    _write_checkout_action(tmp_path)
    _write_workflow(
        tmp_path,
        "missing-guard.yml",
        """
jobs:
  probe:
    runs-on: aragora
    steps:
      - uses: actions/checkout@v4
""",
    )

    violations = check_repo(tmp_path)

    assert len(violations) == 1
    assert violations[0].path == ".github/workflows/missing-guard.yml"
    assert "without a checkout-integrity guard" in violations[0].message


def test_detects_subdirectory_checkout_without_matching_action(tmp_path: Path) -> None:
    _write_checkout_action(tmp_path)
    _write_workflow(
        tmp_path,
        "subdir-mismatch.yml",
        """
jobs:
  probe:
    runs-on:
      - self-hosted
      - aragora
    steps:
      - uses: actions/checkout@v4
        with:
          path: repo
      - uses: ./.github/actions/checkout-integrity
""",
    )

    violations = check_repo(tmp_path)

    assert len(violations) == 1
    assert "checkout-path=repo" in violations[0].message


def test_allows_subdirectory_checkout_with_matching_shared_action(tmp_path: Path) -> None:
    _write_checkout_action(tmp_path)
    _write_workflow(
        tmp_path,
        "subdir-ok.yml",
        """
jobs:
  probe:
    runs-on:
      - self-hosted
      - aragora
    steps:
      - uses: actions/checkout@v4
        with:
          path: repo
      - uses: ./repo/.github/actions/checkout-integrity
        with:
          checkout-path: repo
""",
    )

    violations = check_repo(tmp_path)

    assert violations == []


def test_detects_stale_checkout_action(tmp_path: Path) -> None:
    _write_checkout_action(
        tmp_path,
        """
name: "Verify checkout integrity"
description: "Old action"
runs:
  using: "composite"
  steps:
    - name: Verify checkout integrity
      shell: bash
      run: git sparse-checkout disable || true
""",
    )
    _write_workflow(
        tmp_path,
        "root-guard.yml",
        """
jobs:
  lint:
    runs-on: aragora
    steps:
      - uses: actions/checkout@v4
      - uses: ./.github/actions/checkout-integrity
""",
    )

    violations = check_repo(tmp_path)

    assert len(violations) == 2
    messages = {violation.message for violation in violations}
    assert "checkout-integrity action must include archive restore fallback" in messages
    assert "checkout-integrity action must support checkout-path inputs" in messages


def test_repo_policy_passes_for_current_tree() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    violations = check_repo(repo_root)
    assert violations == []
