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
    upgrade_unbounded_spec,
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


def _write_corpus_entry(
    repo_root: Path,
    *,
    issue_id: int = 5185,
    execution_class: str = "missing_test_coverage",
) -> None:
    corpus_path = repo_root / "docs" / "benchmarks" / "corpus.json"
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text(
        json.dumps(
            {
                "issues": [
                    {
                        "issue_id": issue_id,
                        "title": "[B0-cohort] Add unit tests for utils/sql_helpers.py",
                        "execution_class": execution_class,
                        "scope_hint": [
                            "aragora/utils/sql_helpers.py",
                            "tests/utils/test_sql_helpers.py",
                        ],
                        "known_constraints": [
                            "add unit tests covering happy and edge cases",
                            "do not modify production code unless strictly required",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_upgrade_unbounded_spec_corpus_aware_attaches_scope_hint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_corpus_entry(tmp_path)
    monkeypatch.setenv("ARAGORA_CORPUS_AWARE_DISPATCH", "1")
    spec = SwarmSpec(raw_goal="Fix issue #5185", refined_goal="Fix issue #5185")

    with patch(
        "aragora.swarm.dispatch_followups.upgrade_spec",
        side_effect=AssertionError("corpus-aware upgrade should avoid LLM upgrade"),
    ):
        result = upgrade_unbounded_spec(
            spec,
            issue_number=5185,
            issue_title="[B0-cohort] Add unit tests for utils/sql_helpers.py",
            issue_body="Sparse issue body.",
            repo_root=tmp_path,
            metrics_path=tmp_path / "metrics.jsonl",
        )

    assert result is spec
    assert result.is_dispatch_bounded() is True
    assert result.file_scope_hints == [
        "aragora/utils/sql_helpers.py",
        "tests/utils/test_sql_helpers.py",
    ]
    assert "add unit tests covering happy and edge cases" in result.constraints
    assert "Add focused tests for the behavior in aragora/utils/sql_helpers.py." in (
        result.acceptance_criteria
    )
    assert "Cover at least one happy path and one edge or failure path." in (
        result.acceptance_criteria
    )
    assert result.work_orders == [
        {
            "work_order_id": "corpus-5185",
            "title": "Corpus-aware dispatch bounds for issue #5185",
            "execution_class": "missing_test_coverage",
            "changed_paths": [
                "aragora/utils/sql_helpers.py",
                "tests/utils/test_sql_helpers.py",
            ],
            "acceptance_criteria": result.acceptance_criteria,
            "constraints": result.constraints,
        }
    ]


def test_upgrade_unbounded_spec_corpus_aware_off_by_default(tmp_path: Path, monkeypatch) -> None:
    _write_corpus_entry(tmp_path)
    monkeypatch.delenv("ARAGORA_CORPUS_AWARE_DISPATCH", raising=False)
    spec = SwarmSpec(raw_goal="Fix issue #5185", refined_goal="Fix issue #5185")

    with patch(
        "aragora.swarm.dispatch_followups.upgrade_spec",
        return_value=SimpleNamespace(status="needs_clarification", upgraded_spec=None),
    ) as upgrade_mock:
        result = upgrade_unbounded_spec(
            spec,
            issue_number=5185,
            issue_title="[B0-cohort] Add unit tests for utils/sql_helpers.py",
            issue_body="Sparse issue body.",
            repo_root=tmp_path,
            metrics_path=tmp_path / "metrics.jsonl",
        )

    assert result is None
    assert spec.file_scope_hints == []
    assert spec.acceptance_criteria == []
    assert spec.constraints == []
    upgrade_mock.assert_called_once()


def test_upgrade_unbounded_spec_non_corpus_issue_passes_through(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_corpus_entry(tmp_path, issue_id=5185)
    monkeypatch.setenv("ARAGORA_CORPUS_AWARE_DISPATCH", "1")
    spec = SwarmSpec(raw_goal="Fix issue #9999", refined_goal="Fix issue #9999")

    with patch(
        "aragora.swarm.dispatch_followups.upgrade_spec",
        return_value=SimpleNamespace(status="needs_clarification", upgraded_spec=None),
    ) as upgrade_mock:
        result = upgrade_unbounded_spec(
            spec,
            issue_number=9999,
            issue_title="Unrelated issue",
            issue_body="Sparse issue body.",
            repo_root=tmp_path,
            metrics_path=tmp_path / "metrics.jsonl",
        )

    assert result is None
    assert spec.file_scope_hints == []
    assert spec.acceptance_criteria == []
    assert spec.constraints == []
    upgrade_mock.assert_called_once()


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
