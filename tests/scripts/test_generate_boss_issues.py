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

    scan_calls: list[tuple[object, object, float]] = []
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
    assert len(scan_calls) == 2
    assert scan_calls[0][2] == 0.3
    assert scan_calls[1][2] == 0.0
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


def test_main_reports_low_signal_suppression_and_allows_override(
    monkeypatch,
    capsys,
) -> None:
    low_signal = BossIssueCandidate(
        category="handler_validation",
        title="Add request body validation to demo handler",
        description="Validate request bodies before handler logic runs.",
        file_scope=["aragora/server/demo.py"],
        validation_command="pytest tests/server/demo.py -q",
        acceptance_criteria=["Reject invalid request bodies"],
        expected_success_rate=0.2,
    )

    scan_calls: list[float] = []

    def fake_scan_all(repo_root, categories=None, min_success_rate=0.3):
        scan_calls.append(min_success_rate)
        if min_success_rate > 0:
            return []
        return [low_signal]

    monkeypatch.setattr(mod, "scan_all", fake_scan_all)
    monkeypatch.setattr(mod, "fetch_existing_boss_issues", lambda repo: [])
    monkeypatch.setattr(mod, "fetch_open_pr_files", lambda repo: set())
    monkeypatch.setattr(mod, "validate_body", lambda body: (True, ""))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_boss_issues.py",
            "--repo",
            "org/repo",
            "--dry-run",
            "--categories",
            "handler_validation",
        ],
    )
    mod.main()

    out = capsys.readouterr().out
    assert scan_calls == [0.3, 0.0]
    assert "DRY RUN — would create 0 issues" in out
    assert "Suppressed 1 low-signal candidates across 1 categories at min success rate 30%" in out
    assert "Use --min-success-rate 0 to include them." in out

    scan_calls.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_boss_issues.py",
            "--repo",
            "org/repo",
            "--dry-run",
            "--categories",
            "handler_validation",
            "--min-success-rate",
            "0",
        ],
    )
    mod.main()

    out = capsys.readouterr().out
    assert scan_calls == [0.0]
    assert "Suppressed" not in out
    assert "DRY RUN — would create 1 issues" in out
    assert low_signal.title in out
