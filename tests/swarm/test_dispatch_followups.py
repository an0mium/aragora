from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aragora.swarm.dispatch_followups import (
    annotate_result_with_conductor,
    maybe_upgrade_dispatch_spec,
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
