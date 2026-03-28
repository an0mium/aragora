from __future__ import annotations

import argparse
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aragora.cli.commands.idea import (
    _compose_initiative_brief,
    _create_issue_with_optional_labels,
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


def test_run_idea_triage_creates_initiative_and_linked_issues() -> None:
    handoff = {
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
            AsyncMock(return_value=[handoff]),
        ),
        patch(
            "aragora.cli.commands.idea._create_intake_issue",
            AsyncMock(
                return_value={
                    "number": 700,
                    "url": "https://github.com/synaptent/aragora/issues/700",
                }
            ),
        ),
        patch(
            "aragora.cli.commands.idea._create_triage_issues",
            AsyncMock(
                return_value=[
                    {
                        "number": 701,
                        "url": "https://github.com/synaptent/aragora/issues/701",
                        "title": "Implement intake endpoint",
                        "initiative_issue_number": 700,
                    }
                ]
            ),
        ) as create_triage_issues,
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
                create_issues=True,
            )
        )

    assert result["initiative_issue"]["number"] == 700
    assert result["issues"][0]["initiative_issue_number"] == 700
    create_triage_issues.assert_awaited_once_with(
        [handoff],
        brief=result["brief"],
        repo="synaptent/aragora",
        initiative_issue={
            "number": 700,
            "url": "https://github.com/synaptent/aragora/issues/700",
        },
    )


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


def test_run_idea_review_creates_initiative_and_linked_followup_issues() -> None:
    followup = {
        "handoff_id": "review_followup_intake-boundary",
        "task_title": "Implement intake endpoint: Add non-goal boundary",
        "description": "Bound backend scope",
        "acceptance_criteria": ["Boundary is explicit"],
        "validation": ["pytest tests/api/test_intake.py -q"],
        "file_scope": ["server/intake.py"],
        "risk": "medium",
        "merge_class": "manual",
        "autonomy_mode": "checkpoint",
        "policy_reasons": ["founder_review_followup"],
        "preferred_worker_agent": "claude",
        "preferred_reviewer_agent": "codex",
        "labels": ["boss-ready", "review-followup"],
    }
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
            "aragora.cli.commands.idea._review_founder_handoffs",
            AsyncMock(
                return_value={
                    "status": "approved_with_followups",
                    "summary": "One follow-up remains.",
                    "findings": [],
                    "followups": [followup],
                }
            ),
        ),
        patch(
            "aragora.cli.commands.idea._create_intake_issue",
            AsyncMock(
                return_value={
                    "number": 800,
                    "url": "https://github.com/synaptent/aragora/issues/800",
                }
            ),
        ),
        patch(
            "aragora.cli.commands.idea._create_triage_issues",
            AsyncMock(
                return_value=[
                    {
                        "number": 801,
                        "url": "https://github.com/synaptent/aragora/issues/801",
                        "title": "Implement intake endpoint: Add non-goal boundary",
                        "initiative_issue_number": 800,
                    }
                ]
            ),
        ) as create_triage_issues,
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
                create_issues=True,
            )
        )

    assert result["initiative_issue"]["number"] == 800
    assert result["issues"][0]["initiative_issue_number"] == 800
    create_triage_issues.assert_awaited_once_with(
        [followup],
        brief=result["brief"],
        repo="synaptent/aragora",
        initiative_issue={
            "number": 800,
            "url": "https://github.com/synaptent/aragora/issues/800",
        },
    )


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
        initiative_issue={
            "number": 700,
            "url": "https://github.com/synaptent/aragora/issues/700",
        },
    )

    assert "## File Scope" in body
    assert "server/auth/oidc.py" in body
    assert "Initiative issue: #700 https://github.com/synaptent/aragora/issues/700" in body
    assert "Policy Notes: sensitive_scope:auth_rbac" in body


def test_create_issue_with_optional_labels_ensures_missing_labels() -> None:
    missing_label = SimpleNamespace(
        returncode=1,
        stdout="",
        stderr="could not add label: 'boss-ready' not found",
    )
    label_created = SimpleNamespace(returncode=0, stdout="", stderr="")
    issue_created = SimpleNamespace(
        returncode=0,
        stdout="https://github.com/synaptent/aragora/issues/123\n",
        stderr="",
    )

    with patch(
        "subprocess.run",
        side_effect=[missing_label, label_created, label_created, issue_created],
    ) as run:
        result = asyncio.run(
            _create_issue_with_optional_labels(
                repo="synaptent/aragora",
                title="Founder intake",
                body="body",
                requested_labels=["boss-ready", "risk:medium"],
            )
        )

    assert result["number"] == 123
    assert result["labels_applied"] == ["boss-ready", "risk:medium"]
    assert result["labels_ensured"] == ["boss-ready", "risk:medium"]
    label_create_calls = [
        call.args[0]
        for call in run.call_args_list
        if call.args and call.args[0][:3] == ["gh", "label", "create"]
    ]
    assert len(label_create_calls) == 2
