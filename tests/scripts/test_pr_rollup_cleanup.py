from __future__ import annotations

import json
from typing import Any

from scripts.pr_rollup_cleanup import CommandResult, evaluate_rollup_cleanup


EXPECTED_HEAD = "66c921bcc6607abb3dd50664ac3369057beda664"


class FakeRunner:
    def __init__(
        self,
        *,
        pr_payload: dict[str, Any],
        required_payload: list[dict[str, Any]],
        rerun_returncode: int = 0,
    ) -> None:
        self.pr_payload = pr_payload
        self.required_payload = required_payload
        self.rerun_returncode = rerun_returncode
        self.commands: list[list[str]] = []

    def __call__(self, args: list[str], timeout: int = 120) -> CommandResult:
        self.commands.append(args)
        if args[:3] == ["gh", "pr", "view"]:
            return CommandResult(args, 0, json.dumps(self.pr_payload), "")
        if args[:3] == ["gh", "pr", "checks"]:
            return CommandResult(args, 0, json.dumps(self.required_payload), "")
        if args[:3] == ["gh", "run", "rerun"]:
            return CommandResult(args, self.rerun_returncode, "", "")
        raise AssertionError(f"unexpected command: {args}")


def _pr_payload(
    *,
    head: str = EXPECTED_HEAD,
    state: str = "OPEN",
    is_draft: bool = False,
    rollup: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "number": 7561,
        "title": "fix(review-queue): count quorum by model lineage",
        "state": state,
        "isDraft": is_draft,
        "headRefOid": head,
        "mergeable": "MERGEABLE",
        "mergeStateStatus": "UNSTABLE",
        "url": "https://github.com/synaptent/aragora/pull/7561",
        "statusCheckRollup": rollup if rollup is not None else [_success_rollup()],
    }


def _required_pass() -> dict[str, Any]:
    return {
        "name": "lint",
        "workflow": "Lint",
        "state": "SUCCESS",
        "bucket": "pass",
        "link": "https://example.test/lint",
    }


def _required_fail() -> dict[str, Any]:
    return {
        "name": "lint",
        "workflow": "Lint",
        "state": "FAILURE",
        "bucket": "fail",
        "link": "https://example.test/lint",
    }


def _success_rollup() -> dict[str, Any]:
    return {
        "__typename": "CheckRun",
        "name": "lint",
        "workflowName": "Lint",
        "status": "COMPLETED",
        "conclusion": "SUCCESS",
        "detailsUrl": "https://github.com/synaptent/aragora/actions/runs/10/job/11",
    }


def _cancelled_build_docs() -> dict[str, Any]:
    return {
        "__typename": "CheckRun",
        "name": "build",
        "workflowName": "Build Documentation (PR Check)",
        "status": "COMPLETED",
        "conclusion": "CANCELLED",
        "detailsUrl": "https://github.com/synaptent/aragora/actions/runs/26720995791/job/78747915280",
    }


def _queued_shadow(name: str = "Mac TypeScript SDK Shadow") -> dict[str, Any]:
    return {
        "__typename": "CheckRun",
        "name": name,
        "workflowName": "Self-Hosted Shadow CI",
        "status": "QUEUED",
        "conclusion": "",
        "detailsUrl": "https://github.com/synaptent/aragora/actions/runs/26720995823/job/78747937670",
    }


def test_exact_head_mismatch_blocks_before_required_checks() -> None:
    runner = FakeRunner(
        pr_payload=_pr_payload(head="different"),
        required_payload=[_required_pass()],
    )

    result = evaluate_rollup_cleanup(
        pr_number=7561,
        expected_head=EXPECTED_HEAD,
        runner=runner,
    )

    assert result["blocker"] == "exact_head_mismatch"
    assert result["safe_to_apply"] is False
    assert not any(command[:3] == ["gh", "pr", "checks"] for command in runner.commands)


def test_apply_without_exact_head_blocks_before_pr_view() -> None:
    runner = FakeRunner(
        pr_payload=_pr_payload(rollup=[_cancelled_build_docs()]),
        required_payload=[_required_pass()],
    )

    result = evaluate_rollup_cleanup(
        pr_number=7561,
        expected_head=None,
        apply=True,
        runner=runner,
    )

    assert result["blocker"] == "exact_head_required_for_apply"
    assert result["safe_to_apply"] is False
    assert runner.commands == []


def test_required_check_failure_blocks_rerun() -> None:
    runner = FakeRunner(
        pr_payload=_pr_payload(rollup=[_cancelled_build_docs()]),
        required_payload=[_required_fail()],
    )

    result = evaluate_rollup_cleanup(
        pr_number=7561,
        expected_head=EXPECTED_HEAD,
        runner=runner,
    )

    assert result["blocker"] == "required_checks_not_green"
    assert result["required_checks"]["green"] is False
    assert result["safe_to_apply"] is False


def test_self_hosted_pending_blocks_cancelled_build_docs_rerun() -> None:
    runner = FakeRunner(
        pr_payload=_pr_payload(rollup=[_cancelled_build_docs(), _queued_shadow()]),
        required_payload=[_required_pass()],
    )

    result = evaluate_rollup_cleanup(
        pr_number=7561,
        expected_head=EXPECTED_HEAD,
        runner=runner,
    )

    assert result["proposed_action"] == "wait"
    assert result["blocker"] == "self_hosted_shadow_pending"
    assert len(result["rollup"]["pending_self_hosted"]) == 1
    assert result["safe_to_apply"] is False


def test_no_action_clean_state() -> None:
    runner = FakeRunner(
        pr_payload=_pr_payload(rollup=[_success_rollup()]),
        required_payload=[_required_pass()],
    )

    result = evaluate_rollup_cleanup(
        pr_number=7561,
        expected_head=EXPECTED_HEAD,
        runner=runner,
    )

    assert result["ok"] is True
    assert result["proposed_action"] == "none"
    assert result["blocker"] is None
    assert result["safe_to_apply"] is False


def test_apply_safe_reruns_cancelled_build_docs_once() -> None:
    runner = FakeRunner(
        pr_payload=_pr_payload(rollup=[_cancelled_build_docs()]),
        required_payload=[_required_pass()],
    )

    result = evaluate_rollup_cleanup(
        pr_number=7561,
        expected_head=EXPECTED_HEAD,
        apply=True,
        runner=runner,
    )

    assert result["ok"] is True
    assert result["safe_to_apply"] is True
    assert result["proposed_action"] == "rerun_build_documentation"
    assert result["rerun_run_id"] == "26720995791"
    assert result["rerun_performed"] is True
    assert runner.commands.count(["gh", "run", "rerun", "26720995791"]) == 1
