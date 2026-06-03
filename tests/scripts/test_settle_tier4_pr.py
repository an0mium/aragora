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


def _authorized_comment(
    head: str,
    *,
    association: str = "OWNER",
    author: str = "owner-user",
    include_branch_protection: bool = True,
) -> dict[str, Any]:
    branch_line = (
        "Authorized Action: branch_protection_reconcile on main\n"
        if include_branch_protection
        else ""
    )
    return {
        "authorAssociation": association,
        "author": {"login": author},
        "createdAt": AUTH_CREATED_AT,
        "url": "https://github.example/pr/7423#issuecomment-1",
        "body": (
            "Tier-4 Human Settlement Authorization\n"
            f"Authorized Head SHA: {head}\n"
            "Authorized Action: admin_squash_merge on PR #7423\n"
            f"{branch_line}"
        ),
    }


def _pr_view(
    head: str,
    *,
    comments: list[dict[str, Any]],
    human_settlement_state: str | None = "SUCCESS",
) -> dict[str, Any]:
    status_rollup = (
        [{"context": "aragora/human-settlement", "state": human_settlement_state}]
        if human_settlement_state is not None
        else []
    )
    return {
        "headRefOid": head,
        "state": "OPEN",
        "isDraft": False,
        "mergeStateStatus": "BLOCKED",
        "headCommittedDate": HEAD_COMMITTED_AT,
        "comments": comments,
        "reviews": [],
        "statusCheckRollup": status_rollup,
    }


def _tier4_packet(
    pr: int = 7423,
    *,
    counted_reviewer_ids: list[str] | None = None,
    dogfood_evidence: list[dict[str, str]] | None = None,
    unresolved_dissent: bool = False,
) -> dict[str, Any]:
    return {
        "not_ready": [pr],
        "human_risk_settlement_required": [pr],
        "entries": [
            {
                "pr_number": pr,
                "status": "human_preapproval_required",
                "requires_human_risk_settlement": True,
                "unresolved_dissent": unresolved_dissent,
                "counted_reviewer_ids": (
                    ["codex", "grok"] if counted_reviewer_ids is None else counted_reviewer_ids
                ),
                "dogfood_evidence": (
                    [{"reviewer_id": "codex"}] if dogfood_evidence is None else dogfood_evidence
                ),
            }
        ],
    }


def _valid_checks() -> list[dict[str, str]]:
    return [
        {"name": "lint", "state": "SUCCESS"},
        {"name": "aragora-merge-quorum", "state": "SUCCESS"},
    ]


def test_missing_operator_comment_blocks_settlement() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(head, comments=[{"body": "looks good"}]),
        merge_packet=_tier4_packet(),
        required_checks=_valid_checks(),
    )

    assert result["ok"] is False
    assert "missing repo-visible Tier 4 operator settlement comment" in result["blockers"]


def test_exact_head_operator_comment_allows_check_result() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(
            head,
            comments=[_authorized_comment(head, include_branch_protection=False)],
        ),
        merge_packet=_tier4_packet(),
        required_checks=_valid_checks(),
    )

    assert result["ok"] is True
    assert result["blockers"] == []


def test_member_operator_comment_with_status_and_evidence_allows_check_result() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(
            head,
            comments=[
                _authorized_comment(
                    head,
                    association="MEMBER",
                    author="trusted-member",
                    include_branch_protection=False,
                )
            ],
        ),
        merge_packet=_tier4_packet(),
        required_checks=_valid_checks(),
        trusted_operator_logins=["trusted-member"],
        permission_checker=lambda login: login == "trusted-member",
    )

    assert result["ok"] is True
    assert result["blockers"] == []


def test_operator_comment_without_human_status_does_not_authorize() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(
            head,
            comments=[_authorized_comment(head)],
            human_settlement_state=None,
        ),
        merge_packet=_tier4_packet(),
        required_checks=_valid_checks(),
    )

    assert result["ok"] is False
    assert "missing or unsuccessful aragora/human-settlement status" in result["blockers"]
    assert "missing repo-visible Tier 4 operator settlement comment" not in result["blockers"]


def test_operator_comment_without_counted_evidence_does_not_authorize() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(head, comments=[_authorized_comment(head)]),
        merge_packet=_tier4_packet(counted_reviewer_ids=["codex"]),
        required_checks=_valid_checks(),
    )

    assert result["ok"] is False
    assert "missing Tier 4 model/dogfood settlement evidence" in result["blockers"]
    assert "missing repo-visible Tier 4 operator settlement comment" not in result["blockers"]


def test_missing_required_checks_report_distinct_blocker() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(head, comments=[_authorized_comment(head)]),
        merge_packet=_tier4_packet(),
        required_checks=[],
    )

    assert result["ok"] is False
    assert "required checks are missing" in result["blockers"]
    assert "missing repo-visible Tier 4 operator settlement comment" not in result["blockers"]


def test_branch_protection_mode_requires_branch_protection_token() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(
            head,
            comments=[_authorized_comment(head, include_branch_protection=False)],
        ),
        merge_packet=_tier4_packet(),
        required_checks=_valid_checks(),
        require_branch_protection_token=True,
    )

    assert result["ok"] is False
    assert "missing repo-visible Tier 4 operator settlement comment" in result["blockers"]


def test_branch_protection_mode_accepts_branch_protection_token() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(head, comments=[_authorized_comment(head)]),
        merge_packet=_tier4_packet(),
        required_checks=_valid_checks(),
        require_branch_protection_token=True,
    )

    assert result["ok"] is True
    assert result["blockers"] == []


def test_numeric_not_ready_is_allowed_when_packet_marks_tier4_human_settlement() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(
            head,
            comments=[_authorized_comment(head, include_branch_protection=False)],
        ),
        merge_packet=_tier4_packet(),
        required_checks=[{"name": "lint", "state": "SUCCESS"}],
    )

    assert result["ok"] is True
    assert result["blockers"] == []


def test_untrusted_author_comment_does_not_authorize() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(
            head,
            comments=[_authorized_comment(head, association="CONTRIBUTOR")],
        ),
        merge_packet=_tier4_packet(),
        required_checks=_valid_checks(),
    )

    assert result["ok"] is False
    assert "missing repo-visible Tier 4 operator settlement comment" in result["blockers"]


def test_untrusted_member_comment_does_not_authorize() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(
            head,
            comments=[_authorized_comment(head, association="MEMBER", author="random-member")],
        ),
        merge_packet=_tier4_packet(),
        required_checks=_valid_checks(),
    )

    assert result["ok"] is False
    assert "missing repo-visible Tier 4 operator settlement comment" in result["blockers"]


def test_configured_trusted_member_comment_authorizes(monkeypatch: Any) -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    monkeypatch.setenv("ARAGORA_TIER4_TRUSTED_OPERATORS", "trusted-member")
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(
            head,
            comments=[_authorized_comment(head, association="MEMBER", author="trusted-member")],
        ),
        merge_packet=_tier4_packet(),
        required_checks=_valid_checks(),
        permission_checker=lambda login: login == "trusted-member",
    )

    assert result["ok"] is True
    assert result["blockers"] == []


def test_configured_trusted_member_permission_check_runs_once(monkeypatch: Any) -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    checked_logins: list[str] = []
    monkeypatch.setenv("ARAGORA_TIER4_TRUSTED_OPERATORS", "trusted-member")

    def permission_checker(login: str) -> bool:
        checked_logins.append(login)
        return login == "trusted-member"

    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(
            head,
            comments=[_authorized_comment(head, association="MEMBER", author="trusted-member")],
        ),
        merge_packet=_tier4_packet(),
        required_checks=_valid_checks(),
        permission_checker=permission_checker,
    )

    assert result["ok"] is True
    assert checked_logins == ["trusted-member"]


def test_trusted_member_comment_requires_admin_permission(monkeypatch: Any) -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    monkeypatch.setenv("ARAGORA_TIER4_TRUSTED_OPERATORS", "trusted-member")
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(
            head,
            comments=[_authorized_comment(head, association="MEMBER", author="trusted-member")],
        ),
        merge_packet=_tier4_packet(),
        required_checks=_valid_checks(),
        permission_checker=lambda login: False,
    )

    assert result["ok"] is False
    assert "missing repo-visible Tier 4 operator settlement comment" in result["blockers"]


def test_admin_member_comment_does_not_require_explicit_allowlist() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(
            head,
            comments=[_authorized_comment(head, association="MEMBER", author="an0mium")],
        ),
        merge_packet=_tier4_packet(),
        required_checks=_valid_checks(),
        permission_checker=lambda login: login == "an0mium",
    )

    assert result["ok"] is True
    assert result["blockers"] == []


def test_cli_trusted_operator_login_authorizes_member_comment(
    monkeypatch: Any, tmp_path: Path
) -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    monkeypatch.setattr(
        settler,
        "_load_live_inputs",
        lambda pr, cwd: (
            _pr_view(
                head,
                comments=[_authorized_comment(head, association="MEMBER", author="trusted-member")],
            ),
            _tier4_packet(),
            _valid_checks(),
        ),
    )
    monkeypatch.setattr(
        settler,
        "_login_has_admin_permission",
        lambda login, repo, cwd: login == "trusted-member",
    )

    rc = settler.main(
        [
            "--check",
            "--pr",
            "7423",
            "--head",
            head,
            "--trusted-operator-login",
            "trusted-member",
            "--cwd",
            str(tmp_path),
        ]
    )

    assert rc == 0


def test_collaborator_permission_payload_only_treats_admin_as_admin() -> None:
    assert settler._collaborator_permission_is_admin({"permission": "admin"}) is True
    assert settler._collaborator_permission_is_admin({"role_name": "admin"}) is True
    for permission in ("maintain", "write", "triage", "read"):
        assert settler._collaborator_permission_is_admin({"permission": permission}) is False


def test_authorization_comment_for_different_head_does_not_authorize() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(head, comments=[_authorized_comment("different-head")]),
        merge_packet=_tier4_packet(),
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
        pr_view=_pr_view(head, comments=[stale]),
        merge_packet=_tier4_packet(),
        required_checks=_valid_checks(),
        trusted_operator_logins=["trusted-member"],
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


def test_head_mismatch_skips_member_permission_check(monkeypatch: Any) -> None:
    monkeypatch.setenv("ARAGORA_TIER4_TRUSTED_OPERATORS", "trusted-member")

    def permission_checker(login: str) -> bool:
        raise AssertionError(f"permission check should be lazy, got {login}")

    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head="expected",
        pr_view={
            "headRefOid": "actual",
            "state": "OPEN",
            "isDraft": False,
            "mergeStateStatus": "BLOCKED",
            "headCommittedDate": HEAD_COMMITTED_AT,
            "comments": [
                _authorized_comment(
                    "expected",
                    association="MEMBER",
                    author="trusted-member",
                )
            ],
            "reviews": [],
            "statusCheckRollup": [{"context": "aragora/human-settlement", "state": "SUCCESS"}],
        },
        merge_packet=_tier4_packet(),
        required_checks=_valid_checks(),
        permission_checker=permission_checker,
    )

    diagnostic = result["authorization_diagnostics"][0]
    assert result["admin_permission_evaluation"] == "skipped_early_gate_blockers"
    assert diagnostic["admin_permission_required"] is True
    assert diagnostic["admin_permission_evaluated"] is False
    assert (
        "trusted member admin permission was not evaluated because earlier gate blockers are present"
        in diagnostic["rejection_reasons"]
    )


def test_failed_required_check_blocks_settlement() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(head, comments=[_authorized_comment(head)]),
        merge_packet=_tier4_packet(),
        required_checks=[
            {"name": "lint", "state": "FAILURE"},
            {"name": "aragora-merge-quorum", "state": "SUCCESS"},
        ],
    )

    assert result["ok"] is False
    assert "required check lint is FAILURE" in result["blockers"]
    assert "missing repo-visible Tier 4 operator settlement comment" not in result["blockers"]


def test_failed_required_check_skips_member_permission_check(monkeypatch: Any) -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    monkeypatch.setenv("ARAGORA_TIER4_TRUSTED_OPERATORS", "trusted-member")

    def permission_checker(login: str) -> bool:
        raise AssertionError(f"permission check should be lazy, got {login}")

    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(
            head,
            comments=[
                _authorized_comment(
                    head,
                    association="MEMBER",
                    author="trusted-member",
                )
            ],
        ),
        merge_packet=_tier4_packet(),
        required_checks=[
            {"name": "lint", "state": "FAILURE"},
            {"name": "aragora-merge-quorum", "state": "SUCCESS"},
        ],
        permission_checker=permission_checker,
    )

    diagnostic = result["authorization_diagnostics"][0]
    assert result["admin_permission_evaluation"] == "skipped_early_gate_blockers"
    assert diagnostic["admin_permission_required"] is True
    assert diagnostic["admin_permission_evaluated"] is False
    assert (
        "trusted member admin permission was not evaluated because earlier gate blockers are present"
        in diagnostic["rejection_reasons"]
    )


def test_missing_merge_quorum_check_is_allowed_before_apply_reconciles_protection() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(head, comments=[_authorized_comment(head)]),
        merge_packet=_tier4_packet(),
        required_checks=[{"name": "lint", "state": "SUCCESS"}],
    )

    assert result["ok"] is True
    assert result["blockers"] == []


def test_present_failed_merge_quorum_required_check_blocks_settlement() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(head, comments=[_authorized_comment(head)]),
        merge_packet=_tier4_packet(),
        required_checks=[
            {"name": "lint", "state": "SUCCESS"},
            {"name": "aragora-merge-quorum", "state": "FAILURE"},
        ],
    )

    assert result["ok"] is False
    assert "required check aragora-merge-quorum is FAILURE" in result["blockers"]
    assert "missing repo-visible Tier 4 operator settlement comment" not in result["blockers"]


def test_unexpected_merge_packet_blocker_blocks_settlement() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    packet = _tier4_packet()
    packet["not_ready"] = ["human_risk_settlement", "model_quorum"]
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(head, comments=[_authorized_comment(head)]),
        merge_packet=packet,
        required_checks=_valid_checks(),
    )

    assert result["ok"] is False
    assert "merge-packet has unexpected blockers: model_quorum" in result["blockers"]


def test_unexpected_packet_blocker_does_not_report_missing_trusted_member_comment(
    monkeypatch: Any,
) -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    packet = _tier4_packet()
    packet["not_ready"] = ["human_risk_settlement", "model_quorum"]
    monkeypatch.setenv("ARAGORA_TIER4_TRUSTED_OPERATORS", "trusted-member")

    def permission_checker(login: str) -> bool:
        raise AssertionError(f"permission check should be lazy, got {login}")

    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(
            head,
            comments=[
                _authorized_comment(
                    head,
                    association="MEMBER",
                    author="trusted-member",
                )
            ],
        ),
        merge_packet=packet,
        required_checks=_valid_checks(),
        permission_checker=permission_checker,
    )

    diagnostic = result["authorization_diagnostics"][0]
    assert result["ok"] is False
    assert "merge-packet has unexpected blockers: model_quorum" in result["blockers"]
    assert "missing repo-visible Tier 4 operator settlement comment" not in result["blockers"]
    assert result["admin_permission_evaluation"] == "skipped_early_gate_blockers"
    assert diagnostic["admin_permission_required"] is True
    assert diagnostic["admin_permission_evaluated"] is False
    assert (
        "trusted member admin permission was not evaluated because earlier gate blockers are present"
        in diagnostic["rejection_reasons"]
    )


def test_authorization_diagnostics_explain_member_rejection() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(head, comments=[_authorized_comment(head, association="MEMBER")]),
        merge_packet=_tier4_packet(),
        required_checks=_valid_checks(),
        trusted_operator_logins=["trusted-member"],
    )

    assert result["ok"] is False
    assert result["required_author_associations"] == ["OWNER"]
    assert "Tier-4 Human Settlement Authorization" in result["settlement_comment_template"]
    assert "PR: #7423" in result["settlement_comment_template"]
    assert f"Exact head: {head}" in result["settlement_comment_template"]
    assert (
        "admin_squash_merge and branch_protection_reconcile"
        in result["settlement_comment_template"]
    )
    diagnostics = result["authorization_diagnostics"]
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic["kind"] == "comment"
    assert diagnostic["author"] == "owner-user"
    assert diagnostic["authorAssociation"] == "MEMBER"
    assert diagnostic["url"] == "https://github.example/pr/7423#issuecomment-1"
    assert diagnostic["marker_present"] is True
    assert diagnostic["trusted_author_association"] is False
    assert diagnostic["fresh_after_head_commit"] is True
    assert diagnostic["exact_head_present"] is True
    assert diagnostic["merge_action_present"] is True
    assert diagnostic["branch_protection_action_present"] is True
    assert diagnostic["accepted"] is False
    assert diagnostic["rejection_reasons"] == [
        "MEMBER login owner-user is not in trusted operator allowlist"
    ]


def test_authorization_diagnostics_accept_owner_comment() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(head, comments=[_authorized_comment(head)]),
        merge_packet=_tier4_packet(),
        required_checks=_valid_checks(),
    )

    assert result["ok"] is True
    assert result["authorization_diagnostics"][0]["accepted"] is True
    assert result["authorization_diagnostics"][0]["rejection_reasons"] == []


def test_authorization_diagnostics_report_stale_comment() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    stale = _authorized_comment(head)
    stale["createdAt"] = "2026-05-21T23:59:00Z"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(head, comments=[stale]),
        merge_packet=_tier4_packet(),
        required_checks=_valid_checks(),
    )

    assert result["authorization_diagnostics"][0]["fresh_after_head_commit"] is False
    assert result["authorization_diagnostics"][0]["rejection_reasons"] == [
        "authorization is older than head commit"
    ]


def test_authorization_diagnostics_report_missing_head_and_tokens() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    bad = _authorized_comment("different-head")
    bad["body"] = (
        "Tier-4 Human Settlement Authorization\n"
        "PR: #7423\n"
        "Exact head: different-head\n"
        "Authorized action: admin_squash_merge only\n"
    )
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view=_pr_view(head, comments=[bad]),
        merge_packet=_tier4_packet(),
        required_checks=_valid_checks(),
        require_branch_protection_token=True,
    )

    diagnostic = result["authorization_diagnostics"][0]
    assert diagnostic["exact_head_present"] is False
    assert diagnostic["merge_action_present"] is True
    assert diagnostic["branch_protection_action_present"] is False
    assert diagnostic["rejection_reasons"] == [
        "exact head is missing",
        "branch_protection_reconcile action is missing",
    ]


def test_check_json_includes_authorization_diagnostics(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    monkeypatch.setattr(
        settler,
        "_load_live_inputs",
        lambda pr, cwd: (
            _pr_view(head, comments=[_authorized_comment(head, association="MEMBER")]),
            _tier4_packet(),
            _valid_checks(),
        ),
    )

    rc = settler.main(
        [
            "--check",
            "--pr",
            "7423",
            "--head",
            head,
            "--trusted-operator-login",
            "trusted-member",
            "--cwd",
            str(tmp_path),
            "--json",
        ]
    )

    assert rc == 1
    payload = settler.json.loads(capsys.readouterr().out)
    gate = payload["gate"]
    assert gate["required_author_associations"] == ["OWNER"]
    assert gate["authorization_diagnostics"][0]["rejection_reasons"] == [
        "MEMBER login owner-user is not in trusted operator allowlist"
    ]


def test_apply_uses_valid_command_sequence(monkeypatch: Any, tmp_path: Path) -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    commands: list[tuple[list[str], str | None]] = []

    monkeypatch.setattr(
        settler,
        "_load_live_inputs",
        lambda pr, cwd: (
            _pr_view(head, comments=[_authorized_comment(head)]),
            _tier4_packet(),
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


def test_apply_merge_only_authorization_skips_branch_protection(
    monkeypatch: Any, tmp_path: Path
) -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    commands: list[tuple[list[str], str | None]] = []

    monkeypatch.setattr(
        settler,
        "_load_live_inputs",
        lambda pr, cwd: (
            _pr_view(
                head,
                comments=[_authorized_comment(head, include_branch_protection=False)],
            ),
            _tier4_packet(),
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
        "_branch_protection_snapshot",
        lambda repo, cwd: {"unexpected": "snapshot"},
    )

    rc = settler.main(["--apply", "--pr", "7423", "--head", head, "--cwd", str(tmp_path)])

    assert rc == 0
    assert commands == [
        (
            [
                "gh",
                "pr",
                "merge",
                "7423",
                "--squash",
                "--admin",
                "--match-head-commit",
                head,
            ],
            None,
        )
    ]
