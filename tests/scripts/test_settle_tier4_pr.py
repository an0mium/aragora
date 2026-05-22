"""Tests for ``scripts/settle_tier4_pr.py`` pure guard helpers."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_module(script_name: str) -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


settler = _load_module("settle_tier4_pr.py")
HEAD_COMMITTED_AT = "2026-05-22T00:00:00Z"
AUTH_CREATED_AT = "2026-05-22T00:05:00Z"


def _authorized_comment(head: str, *, association: str = "OWNER") -> dict[str, str]:
    return {
        "authorAssociation": association,
        "createdAt": AUTH_CREATED_AT,
        "body": (
            "Tier-4 Human Settlement Authorization\n"
            f"Authorized Head SHA: {head}\n"
            "Authorized Action: admin_squash_merge on PR #7423\n"
            "Authorized Action: branch_protection_reconcile on main\n"
        ),
    }


def _valid_checks() -> list[dict[str, str]]:
    return [
        {"name": "lint", "state": "SUCCESS"},
        {"name": "aragora-merge-quorum", "state": "SUCCESS"},
    ]


def test_missing_operator_comment_blocks_settlement() -> None:
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head="57c740022e3c432718462efa12ca79f1df4f674d",
        pr_view={
            "headRefOid": "57c740022e3c432718462efa12ca79f1df4f674d",
            "state": "OPEN",
            "isDraft": False,
            "mergeStateStatus": "BLOCKED",
            "headCommittedDate": HEAD_COMMITTED_AT,
            "comments": [{"body": "looks good"}],
            "reviews": [],
        },
        merge_packet={"admin_squash_allowed": False, "not_ready": ["human_risk_settlement"]},
    )

    assert result["ok"] is False
    assert "missing repo-visible Tier 4 operator settlement comment" in result["blockers"]


def test_exact_head_operator_comment_allows_check_result() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view={
            "headRefOid": head,
            "state": "OPEN",
            "isDraft": False,
            "mergeStateStatus": "BLOCKED",
            "headCommittedDate": HEAD_COMMITTED_AT,
            "comments": [_authorized_comment(head)],
            "reviews": [],
        },
        merge_packet={"admin_squash_allowed": False, "not_ready": ["human_risk_settlement"]},
        required_checks=_valid_checks(),
    )

    assert result["ok"] is True
    assert result["blockers"] == []


def test_numeric_not_ready_is_allowed_when_packet_marks_tier4_human_settlement() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view={
            "headRefOid": head,
            "state": "OPEN",
            "isDraft": False,
            "mergeStateStatus": "BLOCKED",
            "headCommittedDate": HEAD_COMMITTED_AT,
            "comments": [_authorized_comment(head)],
            "reviews": [],
        },
        merge_packet={
            "not_ready": [7423],
            "human_risk_settlement_required": [7423],
            "entries": [
                {
                    "pr_number": 7423,
                    "status": "human_preapproval_required",
                    "requires_human_risk_settlement": True,
                }
            ],
        },
        required_checks=[{"name": "lint", "state": "SUCCESS"}],
    )

    assert result["ok"] is True
    assert result["blockers"] == []


def test_untrusted_author_comment_does_not_authorize() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view={
            "headRefOid": head,
            "state": "OPEN",
            "isDraft": False,
            "mergeStateStatus": "BLOCKED",
            "headCommittedDate": HEAD_COMMITTED_AT,
            "comments": [_authorized_comment(head, association="CONTRIBUTOR")],
            "reviews": [],
        },
        merge_packet={"admin_squash_allowed": False, "not_ready": ["human_risk_settlement"]},
        required_checks=_valid_checks(),
    )

    assert result["ok"] is False
    assert "missing repo-visible Tier 4 operator settlement comment" in result["blockers"]


def test_stale_authorization_comment_does_not_authorize() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    stale = _authorized_comment(head)
    stale["createdAt"] = "2026-05-21T23:59:00Z"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view={
            "headRefOid": head,
            "state": "OPEN",
            "isDraft": False,
            "mergeStateStatus": "BLOCKED",
            "headCommittedDate": HEAD_COMMITTED_AT,
            "comments": [stale],
            "reviews": [],
        },
        merge_packet={"admin_squash_allowed": False, "not_ready": ["human_risk_settlement"]},
        required_checks=_valid_checks(),
    )

    assert result["ok"] is False
    assert "missing repo-visible Tier 4 operator settlement comment" in result["blockers"]


def test_head_mismatch_blocks_before_authorization() -> None:
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head="expected",
        pr_view={
            "headRefOid": "actual",
            "state": "OPEN",
            "isDraft": False,
            "mergeStateStatus": "BLOCKED",
            "headCommittedDate": HEAD_COMMITTED_AT,
            "comments": [],
            "reviews": [],
        },
        merge_packet={},
    )

    assert result["ok"] is False
    assert "head mismatch: expected expected, got actual" in result["blockers"]


def test_failed_required_check_blocks_settlement() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view={
            "headRefOid": head,
            "state": "OPEN",
            "isDraft": False,
            "mergeStateStatus": "BLOCKED",
            "headCommittedDate": HEAD_COMMITTED_AT,
            "comments": [_authorized_comment(head)],
            "reviews": [],
        },
        merge_packet={"admin_squash_allowed": False, "not_ready": ["human_risk_settlement"]},
        required_checks=[
            {"name": "lint", "state": "FAILURE"},
            {"name": "aragora-merge-quorum", "state": "SUCCESS"},
        ],
    )

    assert result["ok"] is False
    assert "required check lint is FAILURE" in result["blockers"]


def test_missing_merge_quorum_check_is_allowed_before_apply_reconciles_protection() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view={
            "headRefOid": head,
            "state": "OPEN",
            "isDraft": False,
            "mergeStateStatus": "BLOCKED",
            "headCommittedDate": HEAD_COMMITTED_AT,
            "comments": [_authorized_comment(head)],
            "reviews": [],
        },
        merge_packet={"admin_squash_allowed": False, "not_ready": ["human_risk_settlement"]},
        required_checks=[{"name": "lint", "state": "SUCCESS"}],
    )

    assert result["ok"] is True
    assert result["blockers"] == []


def test_present_failed_merge_quorum_required_check_blocks_settlement() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view={
            "headRefOid": head,
            "state": "OPEN",
            "isDraft": False,
            "mergeStateStatus": "BLOCKED",
            "headCommittedDate": HEAD_COMMITTED_AT,
            "comments": [_authorized_comment(head)],
            "reviews": [],
        },
        merge_packet={"admin_squash_allowed": False, "not_ready": ["human_risk_settlement"]},
        required_checks=[
            {"name": "lint", "state": "SUCCESS"},
            {"name": "aragora-merge-quorum", "state": "FAILURE"},
        ],
    )

    assert result["ok"] is False
    assert "required check aragora-merge-quorum is FAILURE" in result["blockers"]


def test_unexpected_merge_packet_blocker_blocks_settlement() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view={
            "headRefOid": head,
            "state": "OPEN",
            "isDraft": False,
            "mergeStateStatus": "BLOCKED",
            "headCommittedDate": HEAD_COMMITTED_AT,
            "comments": [_authorized_comment(head)],
            "reviews": [],
        },
        merge_packet={"not_ready": ["human_risk_settlement", "model_quorum"]},
        required_checks=_valid_checks(),
    )

    assert result["ok"] is False
    assert "merge-packet has unexpected blockers: model_quorum" in result["blockers"]


def test_apply_uses_valid_command_sequence(monkeypatch: Any, tmp_path: Path) -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    commands: list[tuple[list[str], str | None]] = []

    monkeypatch.setattr(
        settler,
        "_load_live_inputs",
        lambda pr, cwd: (
            {
                "headRefOid": head,
                "state": "OPEN",
                "isDraft": False,
                "mergeStateStatus": "BLOCKED",
                "headCommittedDate": HEAD_COMMITTED_AT,
                "comments": [_authorized_comment(head)],
                "reviews": [],
            },
            {"not_ready": ["human_risk_settlement"]},
            _valid_checks(),
        ),
    )
    monkeypatch.setattr(
        settler,
        "_run_command",
        lambda command, cwd, input_text=None: commands.append((command, input_text)),
    )
    monkeypatch.setattr(
        settler,
        "_required_status_check_patch",
        lambda repo, cwd: (["gh", "api", "--method", "PATCH", "checks"], '{"contexts": []}'),
    )
    monkeypatch.setattr(
        settler,
        "_branch_protection_snapshot",
        lambda repo, cwd: {},
    )

    rc = settler.main(["--apply", "--pr", "7423", "--head", head, "--cwd", str(tmp_path)])

    assert rc == 0
    assert commands[0][0] == [
        "gh",
        "pr",
        "merge",
        "7423",
        "--squash",
        "--admin",
        "--match-head-commit",
        head,
    ]
    assert "required_approving_review_count" in str(commands[1][1])
    assert commands[-1][0][-1].endswith("/protection/enforce_admins")
