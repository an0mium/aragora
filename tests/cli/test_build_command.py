from __future__ import annotations

import argparse
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.cli.commands.build import (
    _create_issues,
    _decompose_tasks,
    _dispatch_owner_binding,
    _dispatch_to_boss_loop,
    _generate_spec,
    _preflight_boss_routing,
    _queueable_tasks_from_review,
    _run_build_pipeline,
    cmd_build,
)
from aragora.cli.parser import build_parser
from aragora.prompt_engine.conductor import ConductorResult
from aragora.prompt_engine.types import ClarifyingQuestion, IntentType, PromptIntent, Specification


def _build_args(**overrides) -> argparse.Namespace:
    defaults = {
        "idea": "Add real-time streaming",
        "from_file": None,
        "dry_run": True,
        "skip_clarify": True,
        "max_tasks": 3,
        "repo": "synaptent/aragora",
        "worker_model": "claude",
        "review_model": "codex",
        "risk": "medium",
        "merge_class": "manual",
        "autonomy_mode": "full-auto",
        "json": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_parser_registers_new_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "build",
            "Ship founder intake",
            "--repo",
            "synaptent/aragora",
            "--worker-model",
            "claude",
            "--review-model",
            "codex",
            "--risk",
            "high",
            "--merge-class",
            "manual",
            "--autonomy-mode",
            "checkpoint",
            "--json",
        ]
    )

    assert args.command == "build"
    assert args.idea == "Ship founder intake"
    assert args.worker_model == "claude"
    assert args.review_model == "codex"
    assert args.risk == "high"
    assert args.autonomy_mode == "checkpoint"
    assert args.func.__name__ == "cmd_build"


def test_cmd_build_json_emits_only_json(capsys) -> None:
    args = _build_args(json=True)
    payload = {"status": "dry_run_complete", "stages": {"tasks": []}}

    with patch(
        "aragora.cli.commands.build._run_build_pipeline_with_cleanup",
        AsyncMock(return_value=payload),
    ):
        cmd_build(args)

    out = capsys.readouterr().out
    assert json.loads(out)["status"] == "dry_run_complete"


def test_run_build_pipeline_dry_run_returns_tasks() -> None:
    with (
        patch(
            "aragora.cli.commands.build._generate_spec",
            AsyncMock(return_value={"title": "Spec", "sections": [], "raw": "spec"}),
        ),
        patch(
            "aragora.cli.commands.build._plan_reviewed_tasks",
            AsyncMock(
                return_value={
                    "brief": {"clarification_completeness_status": "decision_complete"},
                    "handoffs": [
                        {
                            "handoff_id": "handoff_1",
                            "task_title": "Task A",
                        }
                    ],
                    "review": {
                        "status": "approved",
                        "summary": "No findings.",
                        "findings": [],
                        "followups": [],
                    },
                    "tasks": [
                        {
                            "title": "Task A",
                            "description": "Implement thing in server/auth/oidc.py",
                            "acceptance_criteria": ["It works"],
                            "verification": "pytest tests/a.py -q",
                            "file_scope_hints": ["server/auth/oidc.py"],
                            "risk": "high",
                            "merge_class": "manual",
                            "autonomy_mode": "checkpoint",
                            "policy_reasons": ["sensitive_scope:auth_rbac"],
                            "preferred_worker_agent": "claude",
                            "preferred_reviewer_agent": "codex",
                        }
                    ],
                }
            ),
        ),
    ):
        result = asyncio.run(
            _run_build_pipeline(
                idea="Ship thing",
                dry_run=True,
                skip_clarify=True,
                repo="synaptent/aragora",
                worker_model="claude",
                review_model="codex",
                emit_progress=False,
            )
        )

    assert result["status"] == "dry_run_complete"
    assert result["repo"] == "synaptent/aragora"
    assert result["routing_defaults"]["worker_model"] == "claude"
    assert result["stages"]["review"]["status"] == "approved"
    assert result["stages"]["tasks"][0]["preferred_worker_agent"] == "claude"
    assert result["stages"]["tasks"][0]["risk"] == "high"
    assert result["stages"]["tasks"][0]["autonomy_mode"] == "checkpoint"
    assert "server/auth/oidc.py" in result["stages"]["tasks"][0]["file_scope_hints"]


def test_run_build_pipeline_blocks_before_dispatch_when_routing_is_blocked() -> None:
    with (
        patch(
            "aragora.cli.commands.build._generate_spec",
            AsyncMock(return_value={"title": "Spec", "sections": [], "raw": "spec"}),
        ),
        patch(
            "aragora.cli.commands.build._plan_reviewed_tasks",
            AsyncMock(
                return_value={
                    "brief": {"clarification_completeness_status": "decision_complete"},
                    "handoffs": [],
                    "review": {
                        "status": "approved",
                        "summary": "",
                        "findings": [],
                        "followups": [],
                    },
                    "tasks": [
                        {
                            "title": "Task A",
                            "description": "Implement thing",
                            "acceptance_criteria": ["It works"],
                            "verification": "pytest tests/a.py -q",
                        }
                    ],
                }
            ),
        ),
        patch("aragora.cli.commands.build._create_issues", AsyncMock(return_value=[11])),
        patch(
            "aragora.cli.commands.build._preflight_boss_routing",
            return_value={
                "owner_binding": {"user_id": "armand", "workspace_id": "aragora"},
                "blocked": True,
                "routing": {"blocked_reason": "no_eligible_registered_runners"},
            },
        ),
        patch("aragora.cli.commands.build._dispatch_to_boss_loop", AsyncMock()) as dispatch,
    ):
        result = asyncio.run(
            _run_build_pipeline(
                idea="Ship thing",
                dry_run=False,
                skip_clarify=True,
                repo="synaptent/aragora",
                worker_model="claude",
                review_model="codex",
                emit_progress=False,
            )
        )

    assert result["status"] == "blocked_no_runner"
    assert result["stages"]["routing"]["blocked"] is True
    dispatch.assert_not_awaited()


def test_run_build_pipeline_preserves_rich_issue_records_and_dispatches_numbers() -> None:
    with (
        patch(
            "aragora.cli.commands.build._generate_spec",
            AsyncMock(return_value={"title": "Spec", "sections": [], "raw": "spec"}),
        ),
        patch(
            "aragora.cli.commands.build._plan_reviewed_tasks",
            AsyncMock(
                return_value={
                    "brief": {
                        "title": "Ship thing",
                        "clarification_completeness_status": "decision_complete",
                    },
                    "handoffs": [],
                    "review": {
                        "status": "approved",
                        "summary": "",
                        "findings": [],
                        "followups": [],
                    },
                    "tasks": [
                        {
                            "title": "Task A",
                            "description": "Implement thing",
                            "acceptance_criteria": ["It works"],
                            "verification": "pytest tests/a.py -q",
                        }
                    ],
                }
            ),
        ),
        patch(
            "aragora.cli.commands.idea._create_intake_issue",
            AsyncMock(
                return_value={
                    "number": 500,
                    "url": "https://github.com/synaptent/aragora/issues/500",
                }
            ),
        ),
        patch(
            "aragora.cli.commands.build._create_issues",
            AsyncMock(
                return_value=[
                    {
                        "number": 11,
                        "url": "https://github.com/synaptent/aragora/issues/11",
                        "title": "Task A",
                        "initiative_issue_number": 500,
                    }
                ]
            ),
        ),
        patch(
            "aragora.cli.commands.build._preflight_boss_routing",
            return_value={
                "owner_binding": {"user_id": "armand", "workspace_id": "aragora"},
                "blocked": False,
                "routing": {"blocked_reason": None},
            },
        ),
        patch(
            "aragora.cli.commands.build._dispatch_to_boss_loop",
            AsyncMock(return_value={"pid": 4242, "log": ".aragora/builds/run.log"}),
        ) as dispatch,
    ):
        result = asyncio.run(
            _run_build_pipeline(
                idea="Ship thing",
                dry_run=False,
                skip_clarify=True,
                repo="synaptent/aragora",
                worker_model="claude",
                review_model="codex",
                emit_progress=False,
            )
        )

    assert result["status"] == "dispatched"
    assert result["stages"]["initiative_issue"]["number"] == 500
    assert result["stages"]["issues"] == [
        {
            "number": 11,
            "url": "https://github.com/synaptent/aragora/issues/11",
            "title": "Task A",
            "initiative_issue_number": 500,
        }
    ]
    dispatch.assert_awaited_once_with(
        [11],
        repo="synaptent/aragora",
        worker_model="claude",
        review_model="codex",
        autonomy_mode="full-auto",
    )


def test_generate_spec_reads_conductor_result() -> None:
    conductor_result = ConductorResult(
        specification=Specification(
            title="Realtime debate streaming",
            problem_statement="Frontend debate output is only visible after completion.",
            proposed_solution="Stream agent output incrementally to the viewer over SSE.",
            success_criteria=["Users can see agent output as it arrives."],
            confidence=0.86,
        ),
        intent=PromptIntent(
            raw_prompt="Add realtime debate streaming",
            intent_type=IntentType.FEATURE,
            summary="Ship realtime debate streaming to the frontend.",
        ),
        questions=[],
        stages_completed=["decompose", "specify"],
    )

    with patch("aragora.prompt_engine.conductor.PromptConductor") as conductor_cls:
        conductor_cls.return_value.run = AsyncMock(return_value=conductor_result)
        spec = asyncio.run(_generate_spec("Add realtime debate streaming", skip_clarify=True))

    assert spec["title"] == "Realtime debate streaming"
    assert spec["clarification_status"] == "decision_complete"
    assert spec["user_goal"] == "Ship realtime debate streaming to the frontend."
    assert "Stream agent output incrementally" in spec["raw"]


def test_preflight_boss_routing_uses_dispatch_owner_binding(monkeypatch) -> None:
    monkeypatch.setenv("ARAGORA_USER_ID", "founder-1")
    monkeypatch.setenv("ARAGORA_WORKSPACE_ID", "workspace-9")
    registry = MagicMock()
    registry.resolve_boss_routing.return_value.to_dict.return_value = {
        "blocked_reason": None,
        "selected_runner_ids": ["claude-runner-1"],
    }

    with (
        patch("aragora.swarm.runner_registry.LocalRunnerRegistry", return_value=registry),
        patch("aragora.swarm.runner_registry.refresh_discovered_runners", return_value=[]),
    ):
        payload = _preflight_boss_routing(repo="synaptent/aragora", worker_model="claude")

    assert payload["blocked"] is False
    assert payload["owner_binding"] == {"user_id": "founder-1", "workspace_id": "workspace-9"}


def test_preflight_boss_routing_probes_toward_verified_target(monkeypatch) -> None:
    monkeypatch.setenv("ARAGORA_USER_ID", "founder-1")
    monkeypatch.setenv("ARAGORA_WORKSPACE_ID", "workspace-9")
    monkeypatch.setenv("ARAGORA_BUILD_VERIFIED_RUNNER_TARGET", "2")
    monkeypatch.setenv("ARAGORA_BUILD_RUNNER_PROBE_LIMIT", "1")
    registry = MagicMock()
    inspection = SimpleNamespace(runner_id="claude-runner-2", profile="max-02")
    probe = SimpleNamespace(
        status="passed",
        to_dict=lambda: {
            "runner_id": "claude-runner-2",
            "runner_type": "claude",
            "probe_status": "passed",
        },
    )
    registry.resolve_boss_routing.side_effect = [
        SimpleNamespace(
            to_dict=lambda: {
                "blocked_reason": None,
                "selected_runner_ids": ["claude-runner-1", "claude-runner-2"],
                "selected_runners": [
                    {"runner_id": "claude-runner-1", "probe_status": "passed"},
                    {"runner_id": "claude-runner-2", "probe_status": None},
                ],
            }
        ),
        SimpleNamespace(
            to_dict=lambda: {
                "blocked_reason": None,
                "selected_runner_ids": ["claude-runner-1", "claude-runner-2"],
                "selected_runners": [
                    {"runner_id": "claude-runner-1", "probe_status": "passed"},
                    {"runner_id": "claude-runner-2", "probe_status": "passed"},
                ],
            }
        ),
    ]

    with (
        patch("aragora.swarm.runner_registry.LocalRunnerRegistry", return_value=registry),
        patch(
            "aragora.swarm.runner_registry.refresh_discovered_runners", return_value=[inspection]
        ),
        patch(
            "aragora.swarm.runner_registry.prioritized_probe_candidates", return_value=[inspection]
        ),
        patch("aragora.swarm.runner_registry.probe_runner_execution", return_value=probe),
    ):
        payload = _preflight_boss_routing(repo="synaptent/aragora", worker_model="claude")

    assert payload["blocked"] is False
    assert payload["probe"]["auto_probe_triggered"] is True
    assert payload["probe"]["attempted"] == 1
    assert payload["probe"]["passed"] == 1


def test_generate_spec_marks_unanswered_questions_as_needing_clarification() -> None:
    conductor_result = ConductorResult(
        specification=Specification(
            title="Founder intake",
            problem_statement="Ideas enter the system without structured clarification.",
            proposed_solution="Add an intake stage before queue creation.",
            confidence=0.72,
        ),
        intent=PromptIntent(
            raw_prompt="Build founder intake",
            intent_type=IntentType.STRATEGIC,
            summary="Turn vague founder ideas into executable work.",
        ),
        questions=[
            ClarifyingQuestion(
                question="Should this create a GitHub issue automatically?",
                why_it_matters="It changes whether intake is advisory or canonical backlog creation.",
            )
        ],
        stages_completed=["decompose", "interrogate", "specify"],
    )

    with patch("aragora.prompt_engine.conductor.PromptConductor") as conductor_cls:
        conductor_cls.return_value.run = AsyncMock(return_value=conductor_result)
        spec = asyncio.run(_generate_spec("Build founder intake", skip_clarify=False))

    assert spec["clarification_status"] == "needs_clarification"
    assert spec["open_questions"] == ["Should this create a GitHub issue automatically?"]


def test_decompose_tasks_uses_subtask_success_criteria() -> None:
    subtask = SimpleNamespace(
        title="Harden onboarding queueing",
        description="Carry founder review validation into queueable tasks.",
        success_criteria={
            "acceptance_criteria": ["Queue item preserves validation details"],
            "tests": ["pytest tests/cli/test_build_command.py -q"],
        },
    )

    with patch.dict(
        "sys.modules",
        {
            "aragora.nomic.task_decomposer": SimpleNamespace(
                TaskDecomposer=lambda: SimpleNamespace(
                    analyze=lambda _text: SimpleNamespace(subtasks=[subtask])
                )
            )
        },
    ):
        tasks = asyncio.run(_decompose_tasks({"raw": "demo spec"}))

    assert tasks == [
        {
            "title": "Harden onboarding queueing",
            "description": "Carry founder review validation into queueable tasks.",
            "acceptance_criteria": ["Queue item preserves validation details"],
            "verification": "pytest tests/cli/test_build_command.py -q",
        }
    ]


def test_create_issues_includes_queue_metadata() -> None:
    tasks = [
        {
            "title": "Task A",
            "description": "Implement thing",
            "acceptance_criteria": ["It works"],
            "verification": "pytest tests/a.py -q",
            "user_goal": "Ship thing",
            "desired_outcome": "Working thing",
            "affected_surfaces": ["frontend", "server"],
            "file_scope_hints": ["server/auth/oidc.py"],
            "proof_expected": "pytest tests/a.py -q",
            "clarification_status": "decision_complete",
            "open_questions": ["Should this ship behind a flag?"],
            "risk": "medium",
            "merge_class": "manual",
            "autonomy_mode": "full-auto",
            "policy_reasons": ["sensitive_scope:auth_rbac"],
            "labels": ["review-followup", "risk:high"],
            "preferred_worker_agent": "claude",
            "preferred_reviewer_agent": "codex",
        }
    ]
    with patch(
        "aragora.cli.commands.idea._create_issue_with_optional_labels",
        AsyncMock(
            return_value={
                "number": 999,
                "url": "https://github.com/synaptent/aragora/issues/999",
                "labels_requested": ["boss-ready", "review-followup", "risk:high"],
                "labels_applied": ["boss-ready", "review-followup", "risk:high"],
                "labels_ensured": [],
                "fallback_reason": "",
            }
        ),
    ) as create_issue:
        issues = asyncio.run(
            _create_issues(
                tasks,
                repo="synaptent/aragora",
                initiative_issue={
                    "number": 321,
                    "url": "https://github.com/synaptent/aragora/issues/321",
                },
            )
        )

    assert issues == [
        {
            "number": 999,
            "url": "https://github.com/synaptent/aragora/issues/999",
            "labels_requested": ["boss-ready", "review-followup", "risk:high"],
            "labels_applied": ["boss-ready", "review-followup", "risk:high"],
            "labels_ensured": [],
            "fallback_reason": "",
            "title": "Task A",
            "initiative_issue_number": 321,
        }
    ]
    body = create_issue.await_args.kwargs["body"]
    assert "## Initiative Brief" in body
    assert "## Queue Metadata" in body
    assert "Initiative issue: #321 https://github.com/synaptent/aragora/issues/321" in body
    assert "Preferred Worker Agent: claude" in body
    assert "Preferred Reviewer Agent: codex" in body
    assert "Open questions: Should this ship behind a flag?" in body
    assert "File scope hints: server/auth/oidc.py" in body
    assert "Policy Notes: sensitive_scope:auth_rbac" in body
    assert "review-followup" in create_issue.await_args.kwargs["requested_labels"]


def test_queueable_tasks_from_review_prefers_followups_and_unblocked_handoffs() -> None:
    tasks = _queueable_tasks_from_review(
        brief={
            "user_goal": "Ship thing",
            "desired_business_outcome": "Working thing",
            "affected_surfaces": ["frontend", "server"],
            "proof_evidence_expected": "pytest tests/a.py -q",
            "clarification_completeness_status": "decision_complete",
            "open_questions": [],
        },
        spec={"user_goal": "Ship thing", "desired_outcome": "Working thing"},
        handoffs=[
            {
                "handoff_id": "handoff_1",
                "task_title": "Blocked task",
                "description": "Blocked",
                "acceptance_criteria": ["Done"],
                "validation": ["pytest tests/blocked.py -q"],
                "file_scope": ["server/auth/oidc.py"],
                "risk": "high",
                "merge_class": "manual",
                "autonomy_mode": "checkpoint",
                "policy_reasons": ["sensitive_scope:auth_rbac"],
                "preferred_worker_agent": "claude",
                "preferred_reviewer_agent": "codex",
            },
            {
                "handoff_id": "handoff_2",
                "task_title": "Clean task",
                "description": "Safe",
                "acceptance_criteria": ["Done"],
                "validation": ["pytest tests/clean.py -q"],
                "file_scope": ["frontend/src/app/page.tsx"],
                "risk": "medium",
                "merge_class": "manual",
                "autonomy_mode": "checkpoint",
                "policy_reasons": [],
                "preferred_worker_agent": "claude",
                "preferred_reviewer_agent": "codex",
            },
        ],
        review={
            "status": "changes_requested",
            "findings": [
                {
                    "severity": "high",
                    "handoff_ids": ["handoff_1"],
                }
            ],
            "followups": [
                {
                    "handoff_id": "followup_1",
                    "task_title": "Blocked task: Tighten validation contract",
                    "description": "Fix planning contract",
                    "acceptance_criteria": ["Validation is specific"],
                    "validation": ["pytest tests/planning.py -q"],
                    "file_scope": ["server/auth/oidc.py"],
                    "risk": "high",
                    "merge_class": "manual",
                    "autonomy_mode": "checkpoint",
                    "policy_reasons": ["founder_review_followup"],
                    "preferred_worker_agent": "claude",
                    "preferred_reviewer_agent": "codex",
                }
            ],
        },
        repo="synaptent/aragora",
    )

    assert [item["title"] for item in tasks] == [
        "Blocked task: Tighten validation contract",
        "Clean task",
    ]


def test_dispatch_to_boss_loop_uses_scoped_issue_list(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USER", "armand")
    popen = MagicMock()
    popen.return_value.pid = 4242

    with patch("subprocess.Popen", popen):
        result = asyncio.run(
            _dispatch_to_boss_loop(
                [11, 12],
                repo="synaptent/aragora",
                worker_model="claude",
                review_model="codex",
                autonomy_mode="full-auto",
            )
        )

    command = popen.call_args.args[0][2]
    assert "export ARAGORA_USER_ID=armand" in command
    assert "export ARAGORA_WORKSPACE_ID=aragora" in command
    assert "--boss-issue-list 11,12" in command
    assert "--worker-model claude" in command
    assert "--review-model codex" in command
    assert "--boss-repo synaptent/aragora" in command
    assert result["pid"] == 4242
    assert result["issues"] == [11, 12]
    assert result["owner_binding"] == {"user_id": "armand", "workspace_id": "aragora"}


def test_dispatch_owner_binding_prefers_explicit_env(monkeypatch) -> None:
    monkeypatch.setenv("ARAGORA_USER_ID", "founder-1")
    monkeypatch.setenv("ARAGORA_WORKSPACE_ID", "workspace-9")

    binding = _dispatch_owner_binding(repo="synaptent/aragora")

    assert binding == {"user_id": "founder-1", "workspace_id": "workspace-9"}


def test_dispatch_owner_binding_defaults_workspace_from_repo(monkeypatch) -> None:
    monkeypatch.delenv("ARAGORA_USER_ID", raising=False)
    monkeypatch.delenv("ARAGORA_ACTOR_ID", raising=False)
    monkeypatch.delenv("ARAGORA_WORKSPACE_ID", raising=False)
    monkeypatch.delenv("ARAGORA_WORKSPACE", raising=False)
    monkeypatch.setenv("USER", "armand")

    binding = _dispatch_owner_binding(repo="synaptent/aragora")

    assert binding == {"user_id": "armand", "workspace_id": "aragora"}
