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
