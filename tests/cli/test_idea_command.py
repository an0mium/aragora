from __future__ import annotations

import argparse
import asyncio
import json
from unittest.mock import AsyncMock, patch

from aragora.cli.commands.idea import (
    _create_intake_issue,
    _create_triage_issues,
    _compose_initiative_brief,
    _issue_body_for_brief,
    _run_idea_intake,
    _run_idea_review,
    _run_idea_triage,
    _triage_issue_body_for_handoff,
    cmd_idea,
)
from aragora.cli.parser import build_parser


def _idea_args(**overrides) -> argparse.Namespace:
    defaults = {
        "idea_command": "intake",
        "idea": "Turn founder notes into initiatives",
        "from_file": None,
        "skip_clarify": True,
        "priority": "high",
        "track": "2",
        "repo": "synaptent/aragora",
        "max_tasks": 3,
        "risk": "medium",
        "merge_class": "manual",
        "autonomy_mode": "checkpoint",
        "worker_model": "claude",
        "review_model": "codex",
        "create_issue": False,
        "create_issues": False,
        "json": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_idea_intake_parser_registers_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "idea",
            "intake",
            "Clarify founder roadmap",
            "--priority",
            "critical",
            "--track",
            "3",
            "--create-issue",
            "--json",
        ]
    )

    assert args.command == "idea"
    assert args.idea_command == "intake"
    assert args.priority == "critical"
    assert args.track == "3"
    assert args.create_issue is True
    assert args.func.__name__ == "cmd_idea"


def test_idea_triage_parser_registers_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "idea",
            "triage",
            "Clarify founder roadmap",
            "--max-tasks",
            "3",
            "--create-issues",
            "--json",
        ]
    )

    assert args.command == "idea"
    assert args.idea_command == "triage"
    assert args.max_tasks == 3
    assert args.create_issues is True


def test_idea_review_parser_registers_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "idea",
            "review",
            "Clarify founder roadmap",
            "--max-tasks",
            "2",
            "--create-issues",
            "--json",
        ]
    )

    assert args.command == "idea"
    assert args.idea_command == "review"
    assert args.max_tasks == 2
    assert args.create_issues is True


def test_cmd_idea_json_emits_only_json_for_intake(capsys) -> None:
    args = _idea_args(json=True)
    payload = {"status": "brief_created", "brief": {"title": "Founder roadmap"}}

    with patch(
        "aragora.cli.commands.idea._run_idea_intake_with_cleanup",
        AsyncMock(return_value=payload),
    ):
        cmd_idea(args)

    out = capsys.readouterr().out
    assert json.loads(out)["status"] == "brief_created"


def test_compose_initiative_brief_uses_spec_metadata() -> None:
    brief = _compose_initiative_brief(
        idea="Turn founder notes into initiatives",
        spec={
            "title": "Founder intake",
            "user_goal": "Turn vague founder notes into executable work.",
            "desired_outcome": "A structured intake brief exists before coding starts.",
            "success_criteria": ["The system asks clarifying questions before execution."],
            "clarification_status": "needs_clarification",
            "open_questions": ["Should intake always create a GitHub issue?"],
            "raw": "frontend API issue",
        },
        priority="high",
        track="2",
        risk="medium",
        merge_class="manual",
        autonomy_mode="checkpoint",
        worker_model="claude",
        review_model="codex",
    )

    assert brief["title"] == "Founder intake"
    assert brief["sequencing_priority"] == "high"
    assert brief["clarification_completeness_status"] == "needs_clarification"
    assert brief["preferred_worker_agent"] == "claude"
    assert brief["preferred_reviewer_agent"] == "codex"
    assert brief["success_criteria"] == ["The system asks clarifying questions before execution."]
    assert brief["open_questions"] == ["Should intake always create a GitHub issue?"]
    assert brief["summary"] == "A structured intake brief exists before coding starts."
    assert brief["acceptance_criteria"] == [
        "The system asks clarifying questions before execution."
    ]
    assert brief["validation"]


def test_run_idea_intake_returns_brief_without_issue_creation() -> None:
    with patch(
        "aragora.cli.commands.idea._generate_spec",
        AsyncMock(
            return_value={
                "title": "Founder intake",
                "user_goal": "Turn vague founder notes into executable work.",
                "desired_outcome": "A structured intake brief exists before coding starts.",
                "success_criteria": ["The system asks clarifying questions before execution."],
                "clarification_status": "decision_complete",
                "open_questions": [],
                "raw": "frontend server",
            }
        ),
    ):
        result = asyncio.run(
            _run_idea_intake(
                idea="Turn founder notes into initiatives",
                skip_clarify=True,
                priority="high",
                track="2",
                repo="synaptent/aragora",
                risk="medium",
                merge_class="manual",
                autonomy_mode="checkpoint",
                worker_model="claude",
                review_model="codex",
                create_issue=False,
            )
        )

    assert result["status"] == "brief_created"
    assert result["repo"] == "synaptent/aragora"
    assert result["brief"]["preferred_reviewer_agent"] == "codex"
    assert result["brief"]["clarification_completeness_status"] == "decision_complete"
    assert "issue" not in result


def test_run_idea_triage_stops_when_clarification_is_incomplete() -> None:
    with patch(
        "aragora.cli.commands.idea._generate_spec",
        AsyncMock(
            return_value={
                "title": "Founder intake",
                "user_goal": "Turn vague founder notes into executable work.",
                "desired_outcome": "A structured intake brief exists before coding starts.",
                "success_criteria": ["The system asks clarifying questions before execution."],
                "clarification_status": "needs_clarification",
                "open_questions": ["Should intake always create a GitHub issue?"],
                "raw": "frontend server",
            }
        ),
    ):
        result = asyncio.run(
            _run_idea_triage(
                idea="Turn founder notes into initiatives",
                skip_clarify=False,
                priority="high",
                track="2",
                repo="synaptent/aragora",
                max_tasks=3,
                risk="medium",
                merge_class="manual",
                autonomy_mode="checkpoint",
                worker_model="claude",
                review_model="codex",
                create_issues=False,
            )
        )

    assert result["status"] == "needs_clarification"
    assert result["handoffs"] == []
    assert result["next_actions"] == ["Should intake always create a GitHub issue?"]


def test_run_idea_triage_returns_handoffs() -> None:
    with (
        patch(
            "aragora.cli.commands.idea._generate_spec",
            AsyncMock(
                return_value={
                    "title": "Founder intake",
                    "user_goal": "Turn vague founder notes into executable work.",
                    "desired_outcome": "A structured intake brief exists before coding starts.",
                    "success_criteria": ["The system asks clarifying questions before execution."],
                    "clarification_status": "decision_complete",
                    "open_questions": [],
                    "raw": "server/auth/oidc.py frontend",
                }
            ),
        ),
        patch(
            "aragora.cli.commands.idea._generate_founder_handoffs",
            AsyncMock(
                return_value=[
                    {
                        "task_title": "Implement intake endpoint",
                        "risk": "high",
                        "merge_class": "manual",
                        "autonomy_mode": "checkpoint",
                        "preferred_worker_agent": "claude",
                        "preferred_reviewer_agent": "codex",
                        "file_scope": ["server/auth/oidc.py"],
                        "acceptance_criteria": ["Done"],
                        "validation": ["pytest tests/a.py -q"],
                        "repo_evidence": ["server/auth/oidc.py"],
                        "policy_reasons": ["sensitive_scope:auth_rbac"],
                        "description": "Implement intake endpoint",
                    }
                ]
            ),
        ),
    ):
        result = asyncio.run(
            _run_idea_triage(
                idea="Turn founder notes into initiatives",
                skip_clarify=True,
                priority="high",
                track="2",
                repo="synaptent/aragora",
                max_tasks=3,
                risk="medium",
                merge_class="manual",
                autonomy_mode="checkpoint",
                worker_model="claude",
                review_model="codex",
                create_issues=False,
            )
        )

    assert result["status"] == "triaged"
    assert result["handoffs"][0]["preferred_worker_agent"] == "claude"
    assert result["handoffs"][0]["policy_reasons"] == ["sensitive_scope:auth_rbac"]


def test_run_idea_review_returns_structured_findings_and_followups() -> None:
    with (
        patch(
            "aragora.cli.commands.idea._generate_spec",
            AsyncMock(
                return_value={
                    "title": "Founder intake",
                    "user_goal": "Turn vague founder notes into executable work.",
                    "desired_outcome": "A structured intake brief exists before coding starts.",
                    "success_criteria": ["The system asks clarifying questions before execution."],
                    "clarification_status": "decision_complete",
                    "open_questions": [],
                    "raw": "server/auth/oidc.py frontend",
                }
            ),
        ),
        patch(
            "aragora.cli.commands.idea._generate_founder_handoffs",
            AsyncMock(
                return_value=[
                    {
                        "handoff_id": "handoff_1",
                        "task_title": "Implement intake endpoint",
                        "risk": "high",
                        "merge_class": "manual",
                        "autonomy_mode": "checkpoint",
                        "preferred_worker_agent": "claude",
                        "preferred_reviewer_agent": "claude",
                        "file_scope": ["server/auth/oidc.py"],
                        "acceptance_criteria": [],
                        "validation": ["python3 -m pytest tests/ -q -k 'not benchmark'"],
                        "repo_evidence": [],
                        "policy_reasons": ["sensitive_scope:auth_rbac"],
                        "description": "Implement intake endpoint",
                    }
                ]
            ),
        ),
        patch(
            "aragora.cli.commands.idea._model_review_founder_handoffs",
            AsyncMock(
                return_value={
                    "status": "approved",
                    "summary": "Model founder review found no additional issues.",
                    "findings": [],
                    "followups": [],
                }
            ),
        ),
    ):
        result = asyncio.run(
            _run_idea_review(
                idea="Turn founder notes into initiatives",
                skip_clarify=True,
                priority="high",
                track="2",
                repo="synaptent/aragora",
                max_tasks=3,
                risk="medium",
                merge_class="manual",
                autonomy_mode="checkpoint",
                worker_model="claude",
                review_model="codex",
                create_issues=False,
            )
        )

    assert result["status"] == "changes_requested"
    assert result["review"]["status"] == "changes_requested"
    assert len(result["review"]["findings"]) >= 3
    assert result["review"]["followups"]
    assert result["review"]["followups"][0]["task_title"].startswith("Implement intake endpoint:")


def test_run_idea_review_merges_model_review_findings() -> None:
    with (
        patch(
            "aragora.cli.commands.idea._generate_spec",
            AsyncMock(
                return_value={
                    "title": "Founder intake",
                    "user_goal": "Turn vague founder notes into executable work.",
                    "desired_outcome": "A structured intake brief exists before coding starts.",
                    "success_criteria": ["The system asks clarifying questions before execution."],
                    "clarification_status": "decision_complete",
                    "open_questions": [],
                    "raw": "server/intake.py frontend",
                }
            ),
        ),
        patch(
            "aragora.cli.commands.idea._generate_founder_handoffs",
            AsyncMock(
                return_value=[
                    {
                        "handoff_id": "handoff_1",
                        "task_title": "Implement intake endpoint",
                        "risk": "medium",
                        "merge_class": "manual",
                        "autonomy_mode": "checkpoint",
                        "preferred_worker_agent": "claude",
                        "preferred_reviewer_agent": "codex",
                        "file_scope": ["server/intake.py"],
                        "acceptance_criteria": ["Founder intake exists"],
                        "validation": ["python3 -m pytest tests/api/test_intake.py -q"],
                        "repo_evidence": ["server/intake.py"],
                        "policy_reasons": [],
                        "description": "Implement intake endpoint",
                    }
                ]
            ),
        ),
        patch(
            "aragora.cli.commands.idea._model_review_founder_handoffs",
            AsyncMock(
                return_value={
                    "status": "changes_requested",
                    "summary": "Model review detected a missing non-goal boundary.",
                    "findings": [
                        {
                            "finding_id": "model_finding_1",
                            "severity": "medium",
                            "category": "non_goal_boundary",
                            "title": "Non-goals are still implicit",
                            "detail": "The handoff could sprawl into adjacent intake UX work.",
                            "handoff_ids": ["handoff_1"],
                            "recommended_action": "Explicitly bound the handoff to backend intake API work.",
                        }
                    ],
                    "followups": [
                        {
                            "handoff_id": "review_followup_intake-boundary",
                            "task_title": "Implement intake endpoint: Add non-goal boundary",
                            "file_scope": ["server/intake.py"],
                        }
                    ],
                }
            ),
        ),
    ):
        result = asyncio.run(
            _run_idea_review(
                idea="Turn founder notes into initiatives",
                skip_clarify=True,
                priority="high",
                track="2",
                repo="synaptent/aragora",
                max_tasks=3,
                risk="medium",
                merge_class="manual",
                autonomy_mode="checkpoint",
                worker_model="claude",
                review_model="codex",
                create_issues=False,
            )
        )

    assert result["review"]["status"] == "changes_requested"
    assert any(item.get("category") == "non_goal_boundary" for item in result["review"]["findings"])
    assert any(
        item.get("task_title") == "Implement intake endpoint: Add non-goal boundary"
        for item in result["review"]["followups"]
    )


def test_issue_body_mentions_queue_metadata() -> None:
    body = _issue_body_for_brief(
        {
            "title": "Founder intake",
            "user_goal": "Turn vague founder notes into executable work.",
            "desired_business_outcome": "A structured intake brief exists before coding starts.",
            "success_criteria": ["The system asks clarifying questions before execution."],
            "constraints": [],
            "explicit_non_goals": [],
            "affected_product_surfaces": ["frontend", "server"],
            "proof_evidence_expected": "Validation and review notes recorded.",
            "sequencing_priority": "high",
            "clarification_completeness_status": "needs_clarification",
            "open_questions": ["Should intake always create a GitHub issue?"],
            "risk": "medium",
            "merge_class": "manual",
            "autonomy_mode": "checkpoint",
            "track": "2",
            "preferred_worker_agent": "claude",
            "preferred_reviewer_agent": "codex",
        }
    )

    assert "## Initiative Brief" in body
    assert "## Queue Metadata" in body
    assert "- Summary: " in body
    assert "## Acceptance Criteria" in body
    assert "## Validation" in body
    assert "Preferred Worker Agent: claude" in body
    assert "Open questions: Should intake always create a GitHub issue?" in body


def test_triage_issue_body_mentions_scope_and_policy() -> None:
    body = _triage_issue_body_for_handoff(
        {
            "task_title": "Implement intake endpoint",
            "why_now": "Bound the highest-leverage slice first.",
            "repo_evidence": ["server/auth/oidc.py"],
            "file_scope": ["server/auth/oidc.py"],
            "risk": "high",
            "merge_class": "manual",
            "autonomy_mode": "checkpoint",
            "preferred_worker_agent": "claude",
            "preferred_reviewer_agent": "codex",
            "policy_reasons": ["sensitive_scope:auth_rbac"],
            "acceptance_criteria": ["Done"],
            "validation": ["pytest tests/a.py -q"],
            "description": "Implement intake endpoint",
        },
        brief={
            "title": "Founder intake",
            "user_goal": "Turn vague founder notes into executable work.",
            "desired_business_outcome": "A structured intake brief exists before coding starts.",
        },
    )

    assert "## File Scope" in body
    assert "server/auth/oidc.py" in body
    assert "Policy Notes: sensitive_scope:auth_rbac" in body


def test_create_intake_issue_keeps_incomplete_brief_as_idea_intake_only() -> None:
    brief = {
        "title": "Founder intake",
        "summary": "Clarify founder notes before queue creation.",
        "user_goal": "Turn vague founder notes into executable work.",
        "desired_business_outcome": "A structured intake brief exists before coding starts.",
        "success_criteria": ["Open questions are captured before execution."],
        "acceptance_criteria": ["Open questions are captured before execution."],
        "validation": [
            "Artifact remains `idea-intake` only until clarification reaches decision_complete."
        ],
        "constraints": [],
        "explicit_non_goals": [],
        "affected_product_surfaces": ["frontend", "server"],
        "proof_evidence_expected": "Validation and review notes recorded.",
        "sequencing_priority": "high",
        "clarification_completeness_status": "needs_clarification",
        "open_questions": ["Should intake always create a GitHub issue?"],
        "risk": "medium",
        "merge_class": "manual",
        "autonomy_mode": "checkpoint",
        "track": "2",
        "preferred_worker_agent": "claude",
        "preferred_reviewer_agent": "codex",
    }
    proc = type(
        "Proc",
        (),
        {
            "returncode": 0,
            "stdout": "https://github.com/synaptent/aragora/issues/321\n",
            "stderr": "",
        },
    )()

    with patch("subprocess.run", return_value=proc) as run:
        issue = asyncio.run(_create_intake_issue(brief, repo="synaptent/aragora"))

    assert issue["number"] == 321
    cmd = run.call_args.args[0]
    assert "[Idea Intake] Founder intake" in cmd
    assert "idea-intake" in cmd
    assert "initiative" not in cmd


def test_create_triage_issues_rejects_placeholder_handoff_titles() -> None:
    with patch("subprocess.run") as run:
        try:
            asyncio.run(
                _create_triage_issues(
                    [
                        {
                            "task_title": "Untitled handoff",
                            "acceptance_criteria": ["Done"],
                            "validation": ["pytest tests/a.py -q"],
                            "merge_class": "manual",
                            "autonomy_mode": "checkpoint",
                            "labels": ["boss-ready"],
                        }
                    ],
                    brief={"title": "Founder intake"},
                    repo="synaptent/aragora",
                )
            )
        except ValueError as exc:
            assert "placeholder" in str(exc)
        else:
            raise AssertionError("Expected triage issue creation to reject placeholder title")

    run.assert_not_called()
