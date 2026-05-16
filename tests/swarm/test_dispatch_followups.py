from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aragora.swarm.dispatch_followups import (
    _extract_acceptance_criteria,
    annotate_result_with_conductor,
    collect_worker_changed_paths,
    enforce_acceptance_binding,
    inject_closes_into_published_pr,
    maybe_upgrade_dispatch_spec,
    maybe_upgrade_dispatch_spec_from_corpus,
)
from aragora.swarm.issue_upgrader import UpgradedIssue
from aragora.swarm.spec import SwarmSpec


def test_maybe_upgrade_dispatch_spec_uses_issue_category_for_unbounded_spec() -> None:
    issue = SimpleNamespace(number=17, title="Narrow broad except in helper")
    spec = SwarmSpec(
        raw_goal="Fix the helper.",
        refined_goal="Fix the helper.",
        requires_approval=True,
        user_expertise="developer",
    )
    upgraded = UpgradedIssue(
        original_title=issue.title,
        original_body="body",
        upgraded_title=issue.title,
        upgraded_body=(
            "## Task\n\n"
            "Narrow the broad exception handler in `aragora/swarm/helper.py`.\n\n"
            "### File Scope\n"
            "- `aragora/swarm/helper.py`\n"
        ),
        module_summary="Helper summary",
        functions_found=["render_helper"],
        loc=12,
        imports=[],
        complexity="simple",
        upgrade_method="heuristic",
    )

    with patch(
        "aragora.swarm.dispatch_followups.upgrade_issue_heuristic",
        return_value=upgraded,
    ) as upgrade_mock:
        result = maybe_upgrade_dispatch_spec(
            issue=issue,
            spec=spec,
            sanitized_issue_body="body",
            repo_root=Path("/repo"),
        )

    assert result is spec
    assert "aragora/swarm/helper.py" in result.file_scope_hints
    upgrade_mock.assert_called_once_with(
        issue.title,
        "body",
        repo_root=Path("/repo"),
        category="broad_exception",
        acceptance_criteria=[],
    )
    assert result.acceptance_criteria == []


def test_maybe_upgrade_dispatch_spec_preserves_upgraded_acceptance_criteria() -> None:
    issue = SimpleNamespace(number=19, title="Narrow broad except in helper")
    spec = SwarmSpec(
        raw_goal="Fix the helper.",
        refined_goal="Fix the helper.",
        requires_approval=True,
        user_expertise="developer",
    )
    upgraded = UpgradedIssue(
        original_title=issue.title,
        original_body="body",
        upgraded_title=issue.title,
        upgraded_body=(
            "## Task\n\n"
            "Narrow the broad exception handler in `aragora/swarm/helper.py`.\n\n"
            "### File Scope\n"
            "- `aragora/swarm/helper.py`\n\n"
            "### Validation\n"
            "```bash\n"
            "ruff check aragora/swarm/helper.py\n"
            "```\n\n"
            "### Acceptance Criteria\n"
            "- `ruff check aragora/swarm/helper.py` passes\n"
            "- Keep the lane scoped to `aragora/swarm/helper.py`.\n"
        ),
        module_summary="Helper summary",
        functions_found=["render_helper"],
        loc=12,
        imports=[],
        complexity="simple",
        upgrade_method="heuristic",
    )

    with patch(
        "aragora.swarm.dispatch_followups.upgrade_issue_heuristic",
        return_value=upgraded,
    ):
        result = maybe_upgrade_dispatch_spec(
            issue=issue,
            spec=spec,
            sanitized_issue_body="body",
            repo_root=Path("/repo"),
        )

    assert result is spec
    assert result.acceptance_criteria == [
        "`ruff check aragora/swarm/helper.py` passes",
        "Keep the lane scoped to `aragora/swarm/helper.py`.",
    ]
    assert "aragora/swarm/helper.py" in result.file_scope_hints


def test_extract_acceptance_criteria_accepts_acceptance_heading_alias() -> None:
    body = (
        "## Acceptance\n"
        "- `ruff check aragora/swarm/helper.py` passes\n"
        "- Keep the lane scoped to `aragora/swarm/helper.py`.\n"
    )

    assert _extract_acceptance_criteria(body) == [
        "`ruff check aragora/swarm/helper.py` passes",
        "Keep the lane scoped to `aragora/swarm/helper.py`.",
    ]


def test_maybe_upgrade_dispatch_spec_leaves_bounded_spec_unchanged() -> None:
    issue = SimpleNamespace(number=18, title="Add unit tests for helper")
    spec = SwarmSpec(
        raw_goal="Add tests for aragora/swarm/helper.py",
        refined_goal="Add tests",
        acceptance_criteria=["pytest tests/swarm/test_helper.py -q"],
        file_scope_hints=["aragora/swarm/helper.py"],
    )

    with patch("aragora.swarm.dispatch_followups.upgrade_issue_heuristic") as upgrade_mock:
        result = maybe_upgrade_dispatch_spec(
            issue=issue,
            spec=spec,
            sanitized_issue_body="body",
            repo_root=Path("/repo"),
        )

    assert result is spec
    upgrade_mock.assert_not_called()


def test_annotate_result_with_conductor_augments_non_success_result() -> None:
    result = {"status": "needs_human", "reasons": ["boom"]}
    step = SimpleNamespace(
        next_action="retry_same",
        next_prompt="retry prompt",
        terminal_class=SimpleNamespace(value="worker_timeout"),
    )

    with patch("aragora.swarm.conductor.Conductor") as conductor_cls:
        conductor_cls.return_value.evaluate_worker_output.return_value = step
        annotated = annotate_result_with_conductor(
            issue_number=22,
            result=result,
            repo_root=Path("/repo"),
        )

    assert annotated["conductor_next_action"] == "retry_same"
    assert annotated["conductor_next_prompt"] == "retry prompt"
    assert annotated["conductor_terminal_class"] == "worker_timeout"


def test_annotate_result_with_conductor_ignores_completed_result() -> None:
    result = {"status": "completed"}
    assert (
        annotate_result_with_conductor(
            issue_number=23,
            result=result,
            repo_root=Path("/repo"),
        )
        is result
    )


# ---------------------------------------------------------------------------
# v1.3 — acceptance binding wiring tests
# ---------------------------------------------------------------------------


def _worker_result_with_changed_paths(
    changed_paths: list[str],
    *,
    status: str = "completed",
    deliverable_type: str = "branch",
) -> dict[str, object]:
    return {
        "status": status,
        "deliverable": {
            "type": deliverable_type,
            "branch": "aragora/boss-harvest/issue-99-demo",
            "commit_shas": ["abc123"],
        },
        "run": {
            "work_orders": [
                {
                    "work_order_id": "wo-1",
                    "changed_paths": list(changed_paths),
                }
            ],
        },
    }


def test_collect_worker_changed_paths_merges_run_and_deliverable_paths() -> None:
    worker_result = {
        "run": {
            "work_orders": [
                {"changed_paths": ["a.py", "b.py"]},
                {"changed_paths": ["b.py", "c.py"]},
            ],
        },
        "deliverable": {"changed_paths": ["c.py", "d.py"]},
    }
    assert collect_worker_changed_paths(worker_result) == ["a.py", "b.py", "c.py", "d.py"]


def test_enforce_acceptance_binding_passes_for_satisfying_delivery(tmp_path: Path) -> None:
    spec = SwarmSpec(
        raw_goal="tests",
        refined_goal="Add tests",
        acceptance_criteria=["pytest tests/scripts/test_reconcile_b0_pr_truth.py -v passes"],
        file_scope_hints=["scripts/reconcile_b0_pr_truth.py"],
    )
    worker_result = _worker_result_with_changed_paths(
        [
            "scripts/reconcile_b0_pr_truth.py",
            "tests/scripts/test_reconcile_b0_pr_truth.py",
        ]
    )
    metrics_path = tmp_path / "metrics.jsonl"
    issue_body = (
        "## Files\n- `tests/scripts/test_reconcile_b0_pr_truth.py` (new)\n\n"
        "## Acceptance\n- pytest tests/scripts/test_reconcile_b0_pr_truth.py -v passes\n"
    )
    result = enforce_acceptance_binding(
        issue_number=5899,
        issue_body=issue_body,
        spec=spec,
        worker_result=worker_result,
        metrics_path=metrics_path,
    )
    assert result["acceptance_gate_passed"] is True
    assert result["closes_issue_number"] == 5899
    assert result["status"] == "completed"
    # telemetry row written
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    assert rows and rows[0]["event"] == "acceptance_gate"
    assert rows[0]["passed"] is True


def test_enforce_acceptance_binding_rejects_tangential_delivery(tmp_path: Path) -> None:
    """Cycle 1 #5899 replay: worker edited prod only; no test file."""
    spec = SwarmSpec(
        raw_goal="tests",
        refined_goal="Add tests",
        acceptance_criteria=["pytest tests/scripts/test_reconcile_b0_pr_truth.py -v passes"],
        file_scope_hints=["scripts/reconcile_b0_pr_truth.py"],
    )
    worker_result = _worker_result_with_changed_paths(["scripts/reconcile_b0_pr_truth.py"])
    issue_body = (
        "## Files\n- `tests/scripts/test_reconcile_b0_pr_truth.py` (new)\n\n"
        "## Acceptance\n- pytest tests/scripts/test_reconcile_b0_pr_truth.py -v passes\n"
    )
    result = enforce_acceptance_binding(
        issue_number=5899,
        issue_body=issue_body,
        spec=spec,
        worker_result=worker_result,
        metrics_path=tmp_path / "m.jsonl",
    )
    assert result["acceptance_gate_passed"] is False
    assert result["status"] == "needs_human"
    assert result["outcome"] == "acceptance_gate_failed"
    # Both test_presence_missing AND expected_file_not_created should flag.
    assert "test_presence_missing" in result["failure_classes"]
    assert "expected_file_not_created" in result["failure_classes"]
    assert "closes_issue_number" not in result
    assert result["reasons"], "gate should populate human-readable reasons"


def test_enforce_acceptance_binding_falls_back_to_issue_acceptance_heading(
    tmp_path: Path,
) -> None:
    spec = SwarmSpec(raw_goal="tests", refined_goal="Add tests")
    worker_result = _worker_result_with_changed_paths(["aragora/swarm/dispatch_followups.py"])
    issue_body = "## Acceptance\n- pytest tests/swarm/test_dispatch_followups.py -q\n"

    result = enforce_acceptance_binding(
        issue_number=5904,
        issue_body=issue_body,
        spec=spec,
        worker_result=worker_result,
        metrics_path=tmp_path / "m.jsonl",
    )

    assert result["acceptance_gate_passed"] is False
    assert result["status"] == "needs_human"
    assert result["acceptance_gate"]["checks_run"] == ["test_presence", "file_creation"]
    assert "test_presence_missing" in result["failure_classes"]
    assert "expected_file_not_created" in result["failure_classes"]


def test_enforce_acceptance_binding_skips_when_no_deliverable() -> None:
    """No deliverable → gate is a no-op; the result passes through unchanged."""
    spec = SwarmSpec(raw_goal="x", refined_goal="x", acceptance_criteria=["pytest foo.py"])
    worker_result: dict[str, object] = {"status": "completed"}
    out = enforce_acceptance_binding(
        issue_number=1,
        issue_body="",
        spec=spec,
        worker_result=worker_result,
    )
    assert out is worker_result
    assert "acceptance_gate" not in out


def test_enforce_acceptance_binding_skips_for_in_flight_result() -> None:
    spec = SwarmSpec(raw_goal="x", refined_goal="x")
    wr = _worker_result_with_changed_paths(["aragora/x.py"], status="running")
    out = enforce_acceptance_binding(
        issue_number=1,
        issue_body="",
        spec=spec,
        worker_result=wr,
    )
    assert "acceptance_gate" not in out


def test_enforce_acceptance_binding_skips_when_no_changed_paths() -> None:
    spec = SwarmSpec(
        raw_goal="x",
        refined_goal="x",
        acceptance_criteria=["pytest tests/foo.py"],
    )
    wr = _worker_result_with_changed_paths([])  # empty changed paths
    out = enforce_acceptance_binding(
        issue_number=1,
        issue_body="",
        spec=spec,
        worker_result=wr,
    )
    assert "acceptance_gate" not in out


# ---------------------------------------------------------------------------
# Closes #N injection — uses injected fetcher/setter stubs
# ---------------------------------------------------------------------------


def test_inject_closes_into_published_pr_prepends_closer() -> None:
    captured: dict[str, str] = {}

    def fetcher() -> str:
        return "Existing body."

    def setter(new_body: str) -> bool:
        captured["body"] = new_body
        return True

    result = inject_closes_into_published_pr(
        pr_url="https://github.com/org/repo/pull/1",
        issue_number=42,
        body_fetcher=fetcher,
        body_setter=setter,
    )
    assert result["injected"] is True
    assert captured["body"] == "Closes #42\n\nExisting body."


def test_inject_closes_into_published_pr_is_idempotent() -> None:
    def fetcher() -> str:
        return "Closes #42\n\nAlready references."

    def setter(new_body: str) -> bool:  # pragma: no cover — should not be called
        raise AssertionError("setter should not be invoked")

    result = inject_closes_into_published_pr(
        pr_url="https://github.com/org/repo/pull/1",
        issue_number=42,
        body_fetcher=fetcher,
        body_setter=setter,
    )
    assert result["injected"] is False
    assert result["action"] == "already_closes"


def test_inject_closes_into_published_pr_handles_fetch_failure() -> None:
    def fetcher() -> None:
        return None

    def setter(new_body: str) -> bool:  # pragma: no cover
        raise AssertionError("setter should not be invoked on fetch failure")

    result = inject_closes_into_published_pr(
        pr_url="https://github.com/org/repo/pull/1",
        issue_number=42,
        body_fetcher=fetcher,
        body_setter=setter,
    )
    assert result["injected"] is False
    assert result["action"] == "fetch_failed"


def test_inject_closes_into_published_pr_handles_edit_failure() -> None:
    def fetcher() -> str:
        return "Body"

    def setter(new_body: str) -> bool:
        return False

    result = inject_closes_into_published_pr(
        pr_url="https://github.com/org/repo/pull/1",
        issue_number=42,
        body_fetcher=fetcher,
        body_setter=setter,
    )
    assert result["injected"] is False
    assert result["action"] == "edit_failed"


def test_inject_closes_into_published_pr_refuses_invalid_inputs() -> None:
    assert (
        inject_closes_into_published_pr(
            pr_url="",
            issue_number=42,
            body_fetcher=lambda: "body",
            body_setter=lambda b: True,
        )["action"]
        == "skipped"
    )
    assert (
        inject_closes_into_published_pr(
            pr_url="https://github.com/org/repo/pull/1",
            issue_number=0,
            body_fetcher=lambda: "body",
            body_setter=lambda b: True,
        )["action"]
        == "skipped"
    )


# ---------------------------------------------------------------------------
# PR-1 of #7209 — corpus-aware dispatch upgrade tests (flag-gated, default OFF)
# ---------------------------------------------------------------------------


def _write_corpus(repo_root: Path, issues: list[dict]) -> None:
    """Materialize a minimal docs/benchmarks/corpus.json under ``repo_root``."""
    target = repo_root / "docs" / "benchmarks"
    target.mkdir(parents=True, exist_ok=True)
    (target / "corpus.json").write_text(
        json.dumps({"corpus_id": "tw-01-bounded-execution-v1", "issues": issues}),
        encoding="utf-8",
    )


def _corpus_issue(
    issue_id: int,
    *,
    execution_class: str = "missing_test_coverage",
    scope_hint: list[str] | None = None,
    known_constraints: list[str] | None = None,
    terminal_classes: list[str] | None = None,
) -> dict:
    return {
        "issue_id": issue_id,
        "execution_class": execution_class,
        "scope_hint": list(
            scope_hint
            if scope_hint is not None
            else [
                "aragora/utils/sql_helpers.py",
                "tests/utils/test_sql_helpers.py",
            ]
        ),
        "known_constraints": list(
            known_constraints
            if known_constraints is not None
            else [
                "add unit tests covering happy and edge cases for the named module",
                "do not modify production code under aragora/ unless strictly required by the test scaffold",
            ]
        ),
        "dispatch_provenance": {
            "terminal_classes": list(
                terminal_classes
                if terminal_classes is not None
                else ["blocked_not_dispatch_bounded"]
            )
        },
    }


def test_corpus_aware_dispatch_off_by_default_returns_spec_unchanged(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ARAGORA_CORPUS_AWARE_DISPATCH", raising=False)
    _write_corpus(tmp_path, [_corpus_issue(5185)])
    issue = SimpleNamespace(
        number=5185, title="[B0-cohort] Add unit tests for utils/sql_helpers.py"
    )
    spec = SwarmSpec(raw_goal="Add tests")

    result = maybe_upgrade_dispatch_spec_from_corpus(issue=issue, spec=spec, repo_root=tmp_path)

    assert result is spec
    assert not spec.is_dispatch_bounded()
    assert spec.file_scope_hints == []
    assert spec.acceptance_criteria == []
    assert spec.constraints == []


def test_corpus_aware_dispatch_on_corpus_member_whitelisted_augments_spec(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ARAGORA_CORPUS_AWARE_DISPATCH", "1")
    _write_corpus(tmp_path, [_corpus_issue(5185)])
    issue = SimpleNamespace(
        number=5185, title="[B0-cohort] Add unit tests for utils/sql_helpers.py"
    )
    spec = SwarmSpec(raw_goal="Add tests")

    result = maybe_upgrade_dispatch_spec_from_corpus(issue=issue, spec=spec, repo_root=tmp_path)

    assert result is spec
    assert spec.is_dispatch_bounded()
    assert "aragora/utils/sql_helpers.py" in spec.file_scope_hints
    assert "tests/utils/test_sql_helpers.py" in spec.file_scope_hints
    assert any("happy path" in c and "edge case" in c for c in spec.acceptance_criteria), (
        spec.acceptance_criteria
    )
    assert any("pytest" in c for c in spec.acceptance_criteria), spec.acceptance_criteria
    assert any("production code" in c for c in spec.acceptance_criteria), spec.acceptance_criteria
    assert "add unit tests covering happy and edge cases for the named module" in spec.constraints


def test_corpus_aware_dispatch_on_non_corpus_issue_returns_unchanged(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ARAGORA_CORPUS_AWARE_DISPATCH", "1")
    _write_corpus(tmp_path, [_corpus_issue(5185)])
    # Issue #9999 is not in the corpus.
    issue = SimpleNamespace(number=9999, title="Some random feature request")
    spec = SwarmSpec(raw_goal="Add tests")

    result = maybe_upgrade_dispatch_spec_from_corpus(issue=issue, spec=spec, repo_root=tmp_path)

    assert result is spec
    assert not spec.is_dispatch_bounded()
    assert spec.file_scope_hints == []
    assert spec.acceptance_criteria == []
    assert spec.constraints == []


def test_corpus_aware_dispatch_on_non_whitelisted_execution_class_returns_unchanged(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ARAGORA_CORPUS_AWARE_DISPATCH", "1")
    # ``schema_migration`` is intentionally outside the PR-1 whitelist.
    _write_corpus(
        tmp_path,
        [_corpus_issue(7777, execution_class="schema_migration")],
    )
    issue = SimpleNamespace(number=7777, title="Run a database migration")
    spec = SwarmSpec(raw_goal="Migrate")

    result = maybe_upgrade_dispatch_spec_from_corpus(issue=issue, spec=spec, repo_root=tmp_path)

    assert result is spec
    assert not spec.is_dispatch_bounded()
    assert spec.file_scope_hints == []
    assert spec.acceptance_criteria == []
    assert spec.constraints == []


def test_corpus_aware_dispatch_on_non_admission_terminal_class_returns_unchanged(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ARAGORA_CORPUS_AWARE_DISPATCH", "1")
    _write_corpus(
        tmp_path,
        [
            _corpus_issue(
                5426, execution_class="small_refactor", terminal_classes=["deliverable_pr_created"]
            )
        ],
    )
    issue = SimpleNamespace(number=5426, title="Add operator-snapshot subcommand")
    spec = SwarmSpec(raw_goal="Add a command")

    result = maybe_upgrade_dispatch_spec_from_corpus(issue=issue, spec=spec, repo_root=tmp_path)

    assert result is spec
    assert not spec.is_dispatch_bounded()
    assert spec.file_scope_hints == []
    assert spec.acceptance_criteria == []
    assert spec.constraints == []


def test_corpus_aware_dispatch_already_bounded_spec_is_untouched(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ARAGORA_CORPUS_AWARE_DISPATCH", "1")
    _write_corpus(tmp_path, [_corpus_issue(5185)])
    issue = SimpleNamespace(
        number=5185, title="[B0-cohort] Add unit tests for utils/sql_helpers.py"
    )
    spec = SwarmSpec(
        raw_goal="Add tests",
        acceptance_criteria=["existing criterion"],
        file_scope_hints=["existing/path.py"],
    )

    result = maybe_upgrade_dispatch_spec_from_corpus(issue=issue, spec=spec, repo_root=tmp_path)

    assert result is spec
    # Already-bounded spec must not be merged with corpus row -- we short-circuit
    # so as not to widen scope behind the operator's back.
    assert spec.acceptance_criteria == ["existing criterion"]
    assert spec.file_scope_hints == ["existing/path.py"]
    assert spec.constraints == []


def test_corpus_aware_dispatch_handles_each_whitelisted_execution_class(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ARAGORA_CORPUS_AWARE_DISPATCH", "1")
    for execution_class in (
        "missing_test_coverage",
        "small_refactor",
        "validation_tightening",
        "exception_narrowing",
    ):
        sub_root = tmp_path / execution_class
        _write_corpus(
            sub_root,
            [_corpus_issue(5185, execution_class=execution_class)],
        )
        issue = SimpleNamespace(number=5185, title=f"corpus issue for {execution_class}")
        spec = SwarmSpec(raw_goal="goal")
        result = maybe_upgrade_dispatch_spec_from_corpus(issue=issue, spec=spec, repo_root=sub_root)
        assert result.is_dispatch_bounded(), execution_class
        assert spec.acceptance_criteria, execution_class


def test_corpus_aware_dispatch_missing_corpus_file_is_safe(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ARAGORA_CORPUS_AWARE_DISPATCH", "1")
    # ``tmp_path`` has no docs/benchmarks/corpus.json -- read should fall back silently.
    issue = SimpleNamespace(
        number=5185, title="[B0-cohort] Add unit tests for utils/sql_helpers.py"
    )
    spec = SwarmSpec(raw_goal="Add tests")

    result = maybe_upgrade_dispatch_spec_from_corpus(issue=issue, spec=spec, repo_root=tmp_path)

    assert result is spec
    assert not spec.is_dispatch_bounded()


def test_maybe_upgrade_dispatch_spec_chains_into_corpus_path(monkeypatch, tmp_path: Path) -> None:
    """End-to-end: ``maybe_upgrade_dispatch_spec`` consults the corpus first."""
    monkeypatch.setenv("ARAGORA_CORPUS_AWARE_DISPATCH", "1")
    _write_corpus(tmp_path, [_corpus_issue(5185)])
    issue = SimpleNamespace(
        number=5185,
        title="[B0-cohort] Add unit tests for utils/sql_helpers.py",
    )
    spec = SwarmSpec(raw_goal="Add tests")

    # When corpus path bounds the spec, the heuristic upgrader must NOT be called.
    with patch("aragora.swarm.dispatch_followups.upgrade_issue_heuristic") as heuristic_mock:
        result = maybe_upgrade_dispatch_spec(
            issue=issue,
            spec=spec,
            sanitized_issue_body="body",
            repo_root=tmp_path,
        )

    assert result is spec
    assert spec.is_dispatch_bounded()
    heuristic_mock.assert_not_called()


# ---------------------------------------------------------------------------
# PR-2 of #7209 — credential-envelope corpus synthesis tests (flag-gated, default OFF)
# ---------------------------------------------------------------------------


def _envelope_with_all_missing_slices():
    """Build a CredentialEnvelope whose ``missing_slices()`` returns all 5.

    Constructed via ``from_environment({})`` rather than synthesizing
    placeholder slices manually so the test exercises the same constructor
    surface the production code uses.
    """
    from aragora.swarm.credential_envelope import CredentialEnvelope

    return CredentialEnvelope.from_environment({})


def _envelope_with_no_missing_slices():
    """Build a CredentialEnvelope that already passes ``is_complete()``."""
    from aragora.swarm.credential_envelope import CredentialEnvelope

    env = {
        "ARAGORA_RUNNER_PROFILE": "default",
        "CLAUDE_COMMAND": "/usr/bin/claude",
        "ARAGORA_RUNNER_AUTH_MODE": "profile",
        "SSH_AUTH_SOCK": "/tmp/ssh-agent.sock",
        "GH_TOKEN": "gh-token-stub",
        "OPENAI_API_KEY": "sk-stub",
        "ARAGORA_PROVIDER": "openai",
        "ARAGORA_CAN_RUN_PYTEST": "1",
        "ARAGORA_CAN_RUN_RUFF": "1",
    }
    return CredentialEnvelope.from_environment(env)


def test_credential_envelope_probe_off_by_default_returns_none(monkeypatch, tmp_path: Path) -> None:
    from aragora.swarm.dispatch_followups import (
        maybe_synthesize_credential_envelope_from_corpus,
    )

    monkeypatch.delenv("ARAGORA_CREDENTIAL_ENVELOPE_PROBE", raising=False)
    _write_corpus(
        tmp_path,
        [
            _corpus_issue(
                5789,
                execution_class="exception_narrowing",
                terminal_classes=["blocked_auth_failure"],
            )
        ],
    )
    issue = SimpleNamespace(number=5789, title="Narrow broad except in helper")
    envelope = _envelope_with_all_missing_slices()
    assert envelope.missing_slices(), "fixture pre-condition"

    result = maybe_synthesize_credential_envelope_from_corpus(
        issue=issue, envelope=envelope, repo_root=tmp_path
    )

    assert result is None


def test_credential_envelope_probe_no_missing_slices_returns_none(
    monkeypatch, tmp_path: Path
) -> None:
    from aragora.swarm.dispatch_followups import (
        maybe_synthesize_credential_envelope_from_corpus,
    )

    monkeypatch.setenv("ARAGORA_CREDENTIAL_ENVELOPE_PROBE", "1")
    _write_corpus(
        tmp_path,
        [
            _corpus_issue(
                5789,
                execution_class="exception_narrowing",
                terminal_classes=["blocked_auth_failure"],
            )
        ],
    )
    issue = SimpleNamespace(number=5789, title="Narrow broad except in helper")
    envelope = _envelope_with_no_missing_slices()
    assert envelope.missing_slices() == [], "fixture pre-condition"

    result = maybe_synthesize_credential_envelope_from_corpus(
        issue=issue, envelope=envelope, repo_root=tmp_path
    )

    assert result is None


def test_credential_envelope_probe_non_corpus_issue_returns_none(
    monkeypatch, tmp_path: Path
) -> None:
    from aragora.swarm.dispatch_followups import (
        maybe_synthesize_credential_envelope_from_corpus,
    )

    monkeypatch.setenv("ARAGORA_CREDENTIAL_ENVELOPE_PROBE", "1")
    _write_corpus(
        tmp_path,
        [
            _corpus_issue(
                5789,
                execution_class="exception_narrowing",
                terminal_classes=["blocked_auth_failure"],
            )
        ],
    )
    issue = SimpleNamespace(number=9999, title="Some unrelated issue")
    envelope = _envelope_with_all_missing_slices()

    result = maybe_synthesize_credential_envelope_from_corpus(
        issue=issue, envelope=envelope, repo_root=tmp_path
    )

    assert result is None


def test_credential_envelope_probe_non_whitelisted_execution_class_returns_none(
    monkeypatch, tmp_path: Path
) -> None:
    from aragora.swarm.dispatch_followups import (
        maybe_synthesize_credential_envelope_from_corpus,
    )

    monkeypatch.setenv("ARAGORA_CREDENTIAL_ENVELOPE_PROBE", "1")
    # ``docs_reconciliation`` is outside the PR-1 execution_class whitelist.
    _write_corpus(
        tmp_path,
        [
            _corpus_issue(
                5844,
                execution_class="docs_reconciliation",
                terminal_classes=["blocked_auth_failure"],
            )
        ],
    )
    issue = SimpleNamespace(number=5844, title="Reconcile docs/STATUS.md")
    envelope = _envelope_with_all_missing_slices()

    result = maybe_synthesize_credential_envelope_from_corpus(
        issue=issue, envelope=envelope, repo_root=tmp_path
    )

    assert result is None


def test_credential_envelope_probe_no_auth_terminal_class_returns_none(
    monkeypatch, tmp_path: Path
) -> None:
    from aragora.swarm.dispatch_followups import (
        maybe_synthesize_credential_envelope_from_corpus,
    )

    monkeypatch.setenv("ARAGORA_CREDENTIAL_ENVELOPE_PROBE", "1")
    # Corpus member with only ``blocked_not_dispatch_bounded`` terminal history.
    _write_corpus(
        tmp_path,
        [
            _corpus_issue(
                5185,
                execution_class="missing_test_coverage",
                terminal_classes=["blocked_not_dispatch_bounded"],
            )
        ],
    )
    issue = SimpleNamespace(number=5185, title="Add unit tests for utils/sql_helpers.py")
    envelope = _envelope_with_all_missing_slices()

    result = maybe_synthesize_credential_envelope_from_corpus(
        issue=issue, envelope=envelope, repo_root=tmp_path
    )

    assert result is None


def test_credential_envelope_probe_synthesizes_when_all_conditions_met(
    monkeypatch, tmp_path: Path
) -> None:
    from aragora.swarm.dispatch_followups import (
        maybe_synthesize_credential_envelope_from_corpus,
    )

    monkeypatch.setenv("ARAGORA_CREDENTIAL_ENVELOPE_PROBE", "1")
    _write_corpus(
        tmp_path,
        [
            _corpus_issue(
                5789,
                execution_class="exception_narrowing",
                terminal_classes=["blocked_auth_failure"],
            )
        ],
    )
    issue = SimpleNamespace(number=5789, title="Narrow broad except in helper")
    envelope = _envelope_with_all_missing_slices()
    assert envelope.missing_slices() == [
        "runner",
        "git",
        "github_api",
        "provider",
        "verification",
    ], "fixture pre-condition"

    result = maybe_synthesize_credential_envelope_from_corpus(
        issue=issue, envelope=envelope, repo_root=tmp_path
    )

    assert result is not None
    assert result is not envelope, "synthesizer returns a new envelope, not the input"
    assert result.missing_slices() == [], (
        "synthesized envelope must have no missing slices",
        result.missing_slices(),
    )
    # Corpus markers are present so downstream telemetry can identify probe envelopes.
    assert result.runner.profile == "corpus_admitted"
    assert result.provider.provider_name == "corpus_admitted"
    assert result.runner.auth_mode == "profile"


def test_credential_envelope_probe_preserves_already_complete_slices(
    monkeypatch, tmp_path: Path
) -> None:
    from aragora.swarm.credential_envelope import CredentialEnvelope
    from aragora.swarm.dispatch_followups import (
        maybe_synthesize_credential_envelope_from_corpus,
    )

    monkeypatch.setenv("ARAGORA_CREDENTIAL_ENVELOPE_PROBE", "1")
    _write_corpus(
        tmp_path,
        [
            _corpus_issue(
                5789,
                execution_class="exception_narrowing",
                terminal_classes=["blocked_auth_failure"],
            )
        ],
    )
    issue = SimpleNamespace(number=5789, title="Narrow broad except in helper")
    # An envelope where ``runner`` and ``provider`` are already complete but
    # everything else is missing.
    partial_env = {
        "ARAGORA_RUNNER_PROFILE": "real_operator_profile",
        "ARAGORA_RUNNER_AUTH_MODE": "profile",
        "OPENAI_API_KEY": "sk-real-operator",
        "ARAGORA_PROVIDER": "openai",
    }
    envelope = CredentialEnvelope.from_environment(partial_env)
    pre_missing = set(envelope.missing_slices())
    assert "runner" not in pre_missing
    assert "provider" not in pre_missing
    assert {"git", "github_api", "verification"} <= pre_missing

    result = maybe_synthesize_credential_envelope_from_corpus(
        issue=issue, envelope=envelope, repo_root=tmp_path
    )

    assert result is not None
    # Real operator slices preserved verbatim:
    assert result.runner.profile == "real_operator_profile"
    assert result.provider.provider_name == "openai"
    # Synthesized slices are now complete:
    assert result.missing_slices() == []


def test_credential_envelope_probe_handles_missing_corpus_file(monkeypatch, tmp_path: Path) -> None:
    from aragora.swarm.dispatch_followups import (
        maybe_synthesize_credential_envelope_from_corpus,
    )

    monkeypatch.setenv("ARAGORA_CREDENTIAL_ENVELOPE_PROBE", "1")
    # ``tmp_path`` has no docs/benchmarks/corpus.json -- the lookup must
    # fall through silently rather than raise.
    issue = SimpleNamespace(number=5789, title="Narrow broad except in helper")
    envelope = _envelope_with_all_missing_slices()

    result = maybe_synthesize_credential_envelope_from_corpus(
        issue=issue, envelope=envelope, repo_root=tmp_path
    )

    assert result is None


def test_credential_envelope_probe_handles_envelope_without_missing_slices_method(
    monkeypatch, tmp_path: Path
) -> None:
    """Defensive: a non-envelope object should yield None rather than raise."""
    from aragora.swarm.dispatch_followups import (
        maybe_synthesize_credential_envelope_from_corpus,
    )

    monkeypatch.setenv("ARAGORA_CREDENTIAL_ENVELOPE_PROBE", "1")
    _write_corpus(
        tmp_path,
        [
            _corpus_issue(
                5789,
                execution_class="exception_narrowing",
                terminal_classes=["blocked_auth_failure"],
            )
        ],
    )
    issue = SimpleNamespace(number=5789, title="Narrow broad except in helper")
    envelope = SimpleNamespace()  # no ``missing_slices`` attribute

    result = maybe_synthesize_credential_envelope_from_corpus(
        issue=issue, envelope=envelope, repo_root=tmp_path
    )

    assert result is None
