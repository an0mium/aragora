from __future__ import annotations

from types import SimpleNamespace

import scripts.create_b0_cohort as create_b0_cohort
from aragora.swarm.issue_scanner import BossIssueCandidate
from aragora.swarm.task_sanitizer import SanitizationOutcome, SanitizationResult


def _candidate(title: str) -> BossIssueCandidate:
    return BossIssueCandidate(
        category="test_coverage",
        title=title,
        description="Add focused tests for the module to cover edge cases and errors.",
        file_scope=["aragora/foo.py"],
        new_files=["tests/test_foo.py"],
        validation_command="pytest tests/test_foo.py -v",
        acceptance_criteria=["All tests pass"],
        estimated_complexity="small",
    )


def _install_dummy_sanitizer(monkeypatch, outcome: SanitizationOutcome) -> None:
    class DummySanitizer:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            self._outcome = outcome

        def sanitize(self, title: str, body: str) -> SanitizationResult:
            combined = f"{title}\n\n{body}"
            return SanitizationResult(
                outcome=self._outcome,
                original_text=combined,
                sanitized_text=combined,
                reason="ok" if self._outcome == SanitizationOutcome.ACCEPTED else "blocked",
                confidence=0.9,
                checks_failed=[],
            )

    monkeypatch.setattr(create_b0_cohort, "TaskSanitizer", DummySanitizer)


def _install_dummy_bridge(monkeypatch, children: list[BossIssueCandidate] | None = None) -> None:
    class DummyBridge:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            self._children = list(children or [])

        def decompose_issue_sync(self, *args, **kwargs) -> list[BossIssueCandidate]:  # noqa: ANN002, ANN003
            return list(self._children)

    monkeypatch.setattr(create_b0_cohort, "DecompositionBridge", DummyBridge)


def test_dry_run_outputs_required_fields(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        create_b0_cohort, "scan_all", lambda *args, **kwargs: [_candidate("Add unit tests for foo")]
    )
    _install_dummy_bridge(monkeypatch, children=[])
    _install_dummy_sanitizer(monkeypatch, SanitizationOutcome.ACCEPTED)
    monkeypatch.setattr(create_b0_cohort, "assess_issue_body_sanitation", lambda *_: (True, ""))

    exit_code = create_b0_cohort.main(["--dry-run", "--max-issues", "1"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "[B0-cohort]" in output
    assert "File scope: aragora/foo.py, tests/test_foo.py" in output
    assert "Validation command: pytest tests/test_foo.py -v" in output
    assert "TaskSanitizer accepts: yes" in output
    assert "assess_issue_body_sanitation accepts: yes" in output


def test_publish_requires_all_candidates_pass(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        create_b0_cohort, "scan_all", lambda *args, **kwargs: [_candidate("Add unit tests for foo")]
    )
    _install_dummy_bridge(monkeypatch, children=[])
    _install_dummy_sanitizer(monkeypatch, SanitizationOutcome.DROPPED)
    monkeypatch.setattr(create_b0_cohort, "assess_issue_body_sanitation", lambda *_: (True, ""))

    calls = SimpleNamespace(count=0)

    def _fake_create_issue(*args, **kwargs) -> bool:  # noqa: ANN002, ANN003
        calls.count += 1
        return True

    monkeypatch.setattr(create_b0_cohort, "_create_issue", _fake_create_issue)
    exit_code = create_b0_cohort.main(["--publish", "--max-issues", "1"])
    output = capsys.readouterr().out
    assert exit_code == 1
    assert "Publish aborted" in output
    assert calls.count == 0


def test_skip_decomposition_bypasses_bridge(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        create_b0_cohort, "scan_all", lambda *args, **kwargs: [_candidate("Add unit tests for foo")]
    )

    class ExplodingBridge:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            pass

        def decompose_issue_sync(self, *args, **kwargs) -> list[BossIssueCandidate]:  # noqa: ANN002, ANN003
            raise AssertionError("bridge should not run when --skip-decomposition is set")

    monkeypatch.setattr(create_b0_cohort, "DecompositionBridge", ExplodingBridge)
    _install_dummy_sanitizer(monkeypatch, SanitizationOutcome.ACCEPTED)
    monkeypatch.setattr(create_b0_cohort, "assess_issue_body_sanitation", lambda *_: (True, ""))

    exit_code = create_b0_cohort.main(["--dry-run", "--max-issues", "1", "--skip-decomposition"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "[B0-cohort]" in output


def test_publish_applies_requested_label(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        create_b0_cohort, "scan_all", lambda *args, **kwargs: [_candidate("Add unit tests for foo")]
    )
    _install_dummy_bridge(monkeypatch, children=[])
    _install_dummy_sanitizer(monkeypatch, SanitizationOutcome.ACCEPTED)
    monkeypatch.setattr(create_b0_cohort, "assess_issue_body_sanitation", lambda *_: (True, ""))

    calls = []

    def _fake_create_issue(repo: str, title: str, body: str, *, label: str) -> bool:
        calls.append((repo, title, body, label))
        return True

    monkeypatch.setattr(create_b0_cohort, "_create_issue_with_label", _fake_create_issue)
    exit_code = create_b0_cohort.main(["--publish", "--max-issues", "1", "--label", "boss-ready"])
    assert exit_code == 0
    capsys.readouterr()
    assert len(calls) == 1
    assert calls[0][3] == "boss-ready"


def test_requires_exact_count(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        create_b0_cohort, "scan_all", lambda *args, **kwargs: [_candidate("Add unit tests for foo")]
    )
    _install_dummy_bridge(monkeypatch, children=[])
    _install_dummy_sanitizer(monkeypatch, SanitizationOutcome.ACCEPTED)
    monkeypatch.setattr(create_b0_cohort, "assess_issue_body_sanitation", lambda *_: (True, ""))

    exit_code = create_b0_cohort.main(["--dry-run", "--max-issues", "2"])
    output = capsys.readouterr().out
    assert exit_code == 1
    assert "only 1 candidate issues available" in output


def test_select_cohort_skips_duplicate_title_against_open_issue(monkeypatch) -> None:
    _install_dummy_bridge(monkeypatch, children=[])
    _install_dummy_sanitizer(monkeypatch, SanitizationOutcome.ACCEPTED)
    sanitizer = create_b0_cohort.TaskSanitizer()
    bridge = create_b0_cohort.DecompositionBridge()
    candidate = _candidate("Add unit tests for foo")
    open_issue = create_b0_cohort.OpenCohortIssue(
        number=123,
        title="[B0-cohort] Add unit tests for foo",
        body="## Task\n\nSame task without scope details.",
    )

    reviews = create_b0_cohort._select_cohort(
        [candidate],
        max_issues=1,
        bridge=bridge,
        sanitizer=sanitizer,
        existing_aliases=create_b0_cohort._open_issue_aliases([open_issue]),
        skip_decomposition=True,
    )

    assert reviews == []


def test_select_cohort_skips_duplicate_file_scope_against_open_issue(monkeypatch) -> None:
    _install_dummy_bridge(monkeypatch, children=[])
    _install_dummy_sanitizer(monkeypatch, SanitizationOutcome.ACCEPTED)
    sanitizer = create_b0_cohort.TaskSanitizer()
    bridge = create_b0_cohort.DecompositionBridge()
    candidate = _candidate("Different title")
    open_issue = create_b0_cohort.OpenCohortIssue(
        number=124,
        title="[B0-cohort] Existing other title",
        body=(
            "## Task\n\nBody\n\n### File Scope\n"
            "- `aragora/foo.py`\n"
            "- `tests/test_foo.py` (create)\n"
        ),
    )

    reviews = create_b0_cohort._select_cohort(
        [candidate],
        max_issues=1,
        bridge=bridge,
        sanitizer=sanitizer,
        existing_aliases=create_b0_cohort._open_issue_aliases([open_issue]),
        skip_decomposition=True,
    )

    assert reviews == []


def test_select_cohort_skips_duplicate_file_scope_from_legacy_body(monkeypatch) -> None:
    _install_dummy_bridge(monkeypatch, children=[])
    _install_dummy_sanitizer(monkeypatch, SanitizationOutcome.ACCEPTED)
    sanitizer = create_b0_cohort.TaskSanitizer()
    bridge = create_b0_cohort.DecompositionBridge()
    candidate = _candidate("Completely different title")
    open_issue = create_b0_cohort.OpenCohortIssue(
        number=1241,
        title="[B0-cohort] Legacy phrasing",
        body="## Task\n\nAdd focused unit tests for `aragora/foo.py`.\n",
    )

    reviews = create_b0_cohort._select_cohort(
        [candidate],
        max_issues=1,
        bridge=bridge,
        sanitizer=sanitizer,
        existing_aliases=create_b0_cohort._open_issue_aliases([open_issue]),
        skip_decomposition=True,
    )

    assert reviews == []


def test_select_cohort_skips_duplicate_fingerprint_against_open_issue(monkeypatch) -> None:
    _install_dummy_bridge(monkeypatch, children=[])
    _install_dummy_sanitizer(monkeypatch, SanitizationOutcome.ACCEPTED)
    sanitizer = create_b0_cohort.TaskSanitizer()
    bridge = create_b0_cohort.DecompositionBridge()
    candidate = _candidate("Fresh title")
    open_issue = create_b0_cohort.OpenCohortIssue(
        number=125,
        title="[B0-cohort] Unrelated title",
        body=f"<!-- fingerprint:{candidate.fingerprint} -->\n\n## Task\n\nExisting issue",
    )

    reviews = create_b0_cohort._select_cohort(
        [candidate],
        max_issues=1,
        bridge=bridge,
        sanitizer=sanitizer,
        existing_aliases=create_b0_cohort._open_issue_aliases([open_issue]),
        skip_decomposition=True,
    )

    assert reviews == []


def test_replenishment_only_selects_genuinely_new_children(monkeypatch) -> None:
    _install_dummy_sanitizer(monkeypatch, SanitizationOutcome.ACCEPTED)
    parent = _candidate("Parent title")
    duplicate_child = _candidate("Already seeded child")
    unique_child = BossIssueCandidate(
        category="test_coverage",
        title="Brand new child",
        description="Add focused tests for another module to cover edge cases and errors.",
        file_scope=["aragora/bar.py"],
        new_files=["tests/test_bar.py"],
        validation_command="pytest tests/test_bar.py -v",
        acceptance_criteria=["All tests pass"],
        estimated_complexity="small",
    )
    _install_dummy_bridge(monkeypatch, children=[duplicate_child, unique_child])
    sanitizer = create_b0_cohort.TaskSanitizer()
    bridge = create_b0_cohort.DecompositionBridge()
    open_issue = create_b0_cohort.OpenCohortIssue(
        number=126,
        title="[B0-cohort] Already seeded child",
        body=f"<!-- fingerprint:{duplicate_child.fingerprint} -->\n\n## Task\n\nExisting issue",
    )

    reviews = create_b0_cohort._select_cohort(
        [parent],
        max_issues=2,
        bridge=bridge,
        sanitizer=sanitizer,
        existing_aliases=create_b0_cohort._open_issue_aliases([open_issue]),
        skip_decomposition=False,
    )

    assert len(reviews) == 1
    assert [child.prefixed_title for child in reviews[0].children] == [
        "[B0-cohort] Brand new child"
    ]


def test_count_open_b0_duplicate_issues_uses_key_order(monkeypatch) -> None:
    issues = [
        create_b0_cohort.OpenCohortIssue(
            number=1,
            title="[B0-cohort] First",
            body="<!-- fingerprint:abc123 -->\n\n## Task\n\nA",
        ),
        create_b0_cohort.OpenCohortIssue(
            number=2,
            title="[B0-cohort] Second",
            body="<!-- fingerprint:abc123 -->\n\n## Task\n\nB",
        ),
        create_b0_cohort.OpenCohortIssue(
            number=3,
            title="[B0-cohort] Third",
            body="## Task\n\nC\n\n### File Scope\n- `aragora/foo.py`\n",
        ),
        create_b0_cohort.OpenCohortIssue(
            number=4,
            title="[B0-cohort] Fourth",
            body="## Task\n\nD\n\n### File Scope\n- `aragora/foo.py`\n",
        ),
        create_b0_cohort.OpenCohortIssue(
            number=5,
            title="[B0-cohort] Add unit tests for baz",
            body="## Task\n\nE",
        ),
        create_b0_cohort.OpenCohortIssue(
            number=6,
            title="[B0-cohort] Add unit tests for baz",
            body="## Task\n\nF",
        ),
    ]
    monkeypatch.setattr(create_b0_cohort, "_fetch_open_b0_cohort_issues", lambda repo: issues)

    duplicates = create_b0_cohort.count_open_b0_duplicate_issues("synaptent/aragora")

    assert duplicates == 3
