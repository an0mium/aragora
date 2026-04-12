from __future__ import annotations

import sys

from aragora.swarm.issue_scanner import BossIssueCandidate
from scripts import generate_boss_issues as mod


def _candidate(name: str, *, file_scope: list[str]) -> BossIssueCandidate:
    return BossIssueCandidate(
        category="test_coverage",
        title=f"Add tests for {name}",
        description=f"Create tests covering {name}.",
        file_scope=file_scope,
        new_files=[f"tests/test_{name}.py"],
        validation_command=f"pytest tests/test_{name}.py -v",
        acceptance_criteria=["All tests pass"],
    )


def test_main_dry_run_fetches_and_filters_like_real_mode(
    monkeypatch,
    capsys,
) -> None:
    duplicate = _candidate("duplicate_module", file_scope=["aragora/duplicate_module.py"])
    pr_conflict = _candidate("pr_conflict_module", file_scope=["aragora/pr_conflict_module.py"])
    eligible = _candidate("eligible_module", file_scope=["aragora/eligible_module.py"])

    scan_calls: list[tuple[object, object, object]] = []
    fetch_existing_calls: list[str] = []
    fetch_pr_calls: list[str] = []
    create_calls: list[tuple[str, str, str, str]] = []

    monkeypatch.setattr(
        mod,
        "scan_all",
        lambda repo_root, categories=None, min_success_rate=0.3: (
            scan_calls.append((repo_root, categories, min_success_rate))
            or [duplicate, pr_conflict, eligible]
        ),
    )
    monkeypatch.setattr(
        mod,
        "fetch_existing_boss_issues",
        lambda repo: (
            fetch_existing_calls.append(repo)
            or [{"title": "Other issue", "body": f"<!-- fingerprint:{duplicate.fingerprint} -->"}]
        ),
    )
    monkeypatch.setattr(
        mod,
        "fetch_open_pr_files",
        lambda repo: (fetch_pr_calls.append(repo) or {"aragora/pr_conflict_module.py"}),
    )
    monkeypatch.setattr(mod, "validate_body", lambda body: (True, ""))
    monkeypatch.setattr(
        mod,
        "create_github_issue",
        lambda repo, title, body, label: (create_calls.append((repo, title, body, label)) or True),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_boss_issues.py",
            "--repo",
            "org/repo",
            "--dry-run",
            "--max-issues",
            "5",
        ],
    )

    mod.main()

    out = capsys.readouterr().out
    assert scan_calls
    assert scan_calls[0][2] == 0.3
    assert fetch_existing_calls == ["org/repo"]
    assert fetch_pr_calls == ["org/repo"]
    assert "DRY RUN — would create 1 issues" in out
    assert eligible.title in out
    assert "Skipped: 1 duplicates, 1 PR conflicts, 0 validation failures" in out
    assert not create_calls


def test_main_create_mode_trims_to_max_and_writes_fingerprint(
    monkeypatch,
    capsys,
) -> None:
    first = _candidate("first_module", file_scope=["aragora/first_module.py"])
    second = _candidate("second_module", file_scope=["aragora/second_module.py"])

    created: list[tuple[str, str, str, str]] = []
    monkeypatch.setattr(
        mod,
        "scan_all",
        lambda repo_root, categories=None, min_success_rate=0.3: [first, second],
    )
    monkeypatch.setattr(mod, "fetch_existing_boss_issues", lambda repo: [])
    monkeypatch.setattr(mod, "fetch_open_pr_files", lambda repo: set())
    monkeypatch.setattr(mod, "validate_body", lambda body: (True, ""))
    monkeypatch.setattr(
        mod,
        "create_github_issue",
        lambda repo, title, body, label: (created.append((repo, title, body, label)) or True),
    )
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_boss_issues.py",
            "--repo",
            "org/repo",
            "--max-issues",
            "1",
            "--label",
            "boss-ready",
        ],
    )

    mod.main()

    out = capsys.readouterr().out
    assert "Done: 1 created, 0 failed" in out
    assert len(created) == 1
    repo, title, body, label = created[0]
    assert repo == "org/repo"
    assert title == first.title
    assert label == "boss-ready"
    assert f"<!-- fingerprint:{first.fingerprint} -->" in body


def test_main_passes_explicit_min_success_rate(monkeypatch, capsys) -> None:
    eligible = _candidate("eligible_module", file_scope=["aragora/eligible_module.py"])
    scan_calls: list[tuple[object, object, object]] = []

    monkeypatch.setattr(
        mod,
        "scan_all",
        lambda repo_root, categories=None, min_success_rate=0.3: (
            scan_calls.append((repo_root, categories, min_success_rate)) or [eligible]
        ),
    )
    monkeypatch.setattr(mod, "fetch_existing_boss_issues", lambda repo: [])
    monkeypatch.setattr(mod, "fetch_open_pr_files", lambda repo: set())
    monkeypatch.setattr(mod, "validate_body", lambda body: (True, ""))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_boss_issues.py",
            "--dry-run",
            "--min-success-rate",
            "0.5",
        ],
    )

    mod.main()

    _ = capsys.readouterr()
    assert scan_calls
    assert scan_calls[0][2] == 0.5


def test_maybe_decompose_candidates_noop_when_disabled(monkeypatch) -> None:
    parent = _candidate("parent_module", file_scope=["aragora/parent_module.py"])

    class UnexpectedBridge:
        def __init__(self, repo_root):  # noqa: D401, ANN001
            raise AssertionError("bridge should not be constructed when disabled")

    monkeypatch.setattr(mod, "DecompositionBridge", UnexpectedBridge)

    result = mod.maybe_decompose_candidates(
        [parent],
        enabled=False,
        max_children_per_parent=5,
        repo_root=mod.REPO_ROOT,
    )

    assert result == [parent]


def test_maybe_decompose_candidates_replaces_parent_when_children_emitted(monkeypatch) -> None:
    parent = _candidate("parent_module", file_scope=["aragora/parent_module.py"])
    child_a = BossIssueCandidate(
        category="test_coverage",
        title="Child A",
        description="Write focused tests for module A with bounded scope and validation.",
        file_scope=["aragora/module_a.py"],
        new_files=["tests/test_module_a.py"],
        validation_command="python3 -m pytest -q tests/test_module_a.py",
    )
    child_b = BossIssueCandidate(
        category="test_coverage",
        title="Child B",
        description="Write focused tests for module B with bounded scope and validation.",
        file_scope=["aragora/module_b.py"],
        new_files=["tests/test_module_b.py"],
        validation_command="python3 -m pytest -q tests/test_module_b.py",
    )

    class FakeBridge:
        def __init__(self, repo_root):  # noqa: D401, ANN001
            self.repo_root = repo_root

        def decompose_issue_sync(self, title, body, *, max_children):  # noqa: ANN001
            assert title == parent.title
            assert "## Task" in body
            assert max_children == 3
            return [child_a, child_b]

    monkeypatch.setattr(mod, "DecompositionBridge", FakeBridge)

    result = mod.maybe_decompose_candidates(
        [parent],
        enabled=True,
        max_children_per_parent=3,
        repo_root=mod.REPO_ROOT,
    )

    assert result == [child_a, child_b]


def test_maybe_decompose_candidates_keeps_parent_when_child_set_is_not_meaningful(
    monkeypatch,
) -> None:
    parent = _candidate("parent_module", file_scope=["aragora/parent_module.py"])
    single_child = BossIssueCandidate(
        category="test_coverage",
        title="Only child",
        description="Only child with bounded scope and validation command.",
        file_scope=["aragora/module_a.py"],
        validation_command="python3 -m ruff check aragora/module_a.py",
    )

    class FakeBridge:
        def __init__(self, repo_root):  # noqa: D401, ANN001
            self.repo_root = repo_root

        def decompose_issue_sync(self, title, body, *, max_children):  # noqa: ANN001
            return [single_child]

    monkeypatch.setattr(mod, "DecompositionBridge", FakeBridge)

    result = mod.maybe_decompose_candidates(
        [parent],
        enabled=True,
        max_children_per_parent=5,
        repo_root=mod.REPO_ROOT,
    )

    assert result == [parent]


def test_main_passes_decomposition_flags(monkeypatch, capsys) -> None:
    eligible = _candidate("eligible_module", file_scope=["aragora/eligible_module.py"])
    scan_calls: list[tuple[object, object, object]] = []
    decompose_calls: list[tuple[list[BossIssueCandidate], bool, int, object]] = []

    monkeypatch.setattr(
        mod,
        "scan_all",
        lambda repo_root, categories=None, min_success_rate=0.3: (
            scan_calls.append((repo_root, categories, min_success_rate)) or [eligible]
        ),
    )
    monkeypatch.setattr(
        mod,
        "maybe_decompose_candidates",
        lambda candidates, *, enabled, max_children_per_parent, repo_root: (
            decompose_calls.append((list(candidates), enabled, max_children_per_parent, repo_root))
            or list(candidates)
        ),
    )
    monkeypatch.setattr(mod, "fetch_existing_boss_issues", lambda repo: [])
    monkeypatch.setattr(mod, "fetch_open_pr_files", lambda repo: set())
    monkeypatch.setattr(mod, "validate_body", lambda body: (True, ""))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_boss_issues.py",
            "--dry-run",
            "--decompose-low-quality",
            "--max-children-per-parent",
            "4",
        ],
    )

    mod.main()

    _ = capsys.readouterr()
    assert scan_calls
    assert decompose_calls
    assert decompose_calls[0][1] is True
    assert decompose_calls[0][2] == 4
