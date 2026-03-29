"""Idea intake, founder-triage, and founder-review commands."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from aragora.cli.commands.build import (
    _close_shared_agent_connector,
    _default_repo,
    _generate_spec,
    _infer_surfaces,
)
from aragora.nomic.pipeline_bridge import NomicPipelineBridge
from aragora.nomic.task_decomposer import TaskDecomposer
from aragora.swarm.delivery_policy import apply_delivery_policy
from aragora.swarm.spec import SwarmSpec

logger = logging.getLogger(__name__)

_PLACEHOLDER_TITLES = {
    "untitled initiative",
    "untitled handoff",
    "untitled task",
    "initiative",
    "handoff",
    "task",
}


def cmd_idea(args: argparse.Namespace) -> None:
    """Dispatch idea subcommands."""
    command = str(getattr(args, "idea_command", "") or "").strip().lower()
    if command not in {"intake", "triage", "review"}:
        print('Usage: aragora idea {intake,triage,review} "your idea here"')
        sys.exit(1)

    idea = _read_idea_text(args)
    if not idea:
        print(f'Usage: aragora idea {command} "your idea here"')
        sys.exit(1)

    if command == "intake":
        result = asyncio.run(
            _run_idea_intake_with_cleanup(
                idea=idea,
                skip_clarify=bool(getattr(args, "skip_clarify", False)),
                priority=_text(getattr(args, "priority", "medium")) or "medium",
                track=_text(getattr(args, "track", "1")) or "1",
                repo=_text(getattr(args, "repo", None)),
                risk=_text(getattr(args, "risk", "medium")) or "medium",
                merge_class=_text(getattr(args, "merge_class", "manual")) or "manual",
                autonomy_mode=_text(getattr(args, "autonomy_mode", "checkpoint")) or "checkpoint",
                worker_model=_text(getattr(args, "worker_model", "claude")) or "claude",
                review_model=_text(getattr(args, "review_model", "codex")) or "codex",
                create_issue=bool(getattr(args, "create_issue", False)),
            )
        )
    elif command == "triage":
        result = asyncio.run(
            _run_idea_triage_with_cleanup(
                idea=idea,
                skip_clarify=bool(getattr(args, "skip_clarify", False)),
                priority=_text(getattr(args, "priority", "medium")) or "medium",
                track=_text(getattr(args, "track", "1")) or "1",
                repo=_text(getattr(args, "repo", None)),
                max_tasks=int(getattr(args, "max_tasks", 4) or 4),
                risk=_text(getattr(args, "risk", "medium")) or "medium",
                merge_class=_text(getattr(args, "merge_class", "manual")) or "manual",
                autonomy_mode=_text(getattr(args, "autonomy_mode", "checkpoint")) or "checkpoint",
                worker_model=_text(getattr(args, "worker_model", "claude")) or "claude",
                review_model=_text(getattr(args, "review_model", "codex")) or "codex",
                create_issues=bool(getattr(args, "create_issues", False)),
            )
        )
    else:
        result = asyncio.run(
            _run_idea_review_with_cleanup(
                idea=idea,
                skip_clarify=bool(getattr(args, "skip_clarify", False)),
                priority=_text(getattr(args, "priority", "medium")) or "medium",
                track=_text(getattr(args, "track", "1")) or "1",
                repo=_text(getattr(args, "repo", None)),
                max_tasks=int(getattr(args, "max_tasks", 4) or 4),
                risk=_text(getattr(args, "risk", "medium")) or "medium",
                merge_class=_text(getattr(args, "merge_class", "manual")) or "manual",
                autonomy_mode=_text(getattr(args, "autonomy_mode", "checkpoint")) or "checkpoint",
                worker_model=_text(getattr(args, "worker_model", "claude")) or "claude",
                review_model=_text(getattr(args, "review_model", "codex")) or "codex",
                create_issues=bool(getattr(args, "create_issues", False)),
            )
        )

    if bool(getattr(args, "json", False)):
        print(json.dumps(result, indent=2, default=str))
    elif command == "intake":
        _print_intake_result(result)
    elif command == "triage":
        _print_triage_result(result)
    else:
        _print_review_result(result)


def _read_idea_text(args: argparse.Namespace) -> str:
    idea = str(getattr(args, "idea", None) or "").strip()
    from_file = _text(getattr(args, "from_file", None))
    if from_file:
        idea = Path(from_file).read_text().strip()
    return idea


def _text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _collapse_whitespace(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _is_placeholder_title(value: Any) -> bool:
    title = _collapse_whitespace(value).strip(" -:_")
    if not title:
        return True
    normalized = title.lower()
    if normalized in _PLACEHOLDER_TITLES:
        return True
    return normalized.startswith("untitled ")


def _derive_initiative_title(*, idea: str, spec: dict[str, Any]) -> str:
    for candidate in (
        spec.get("title"),
        spec.get("user_goal"),
        spec.get("desired_outcome"),
        spec.get("raw"),
        idea,
    ):
        title = _collapse_whitespace(candidate).strip(" -:_")
        if title and not _is_placeholder_title(title):
            return title[:120]
    return ""


def _derive_initiative_summary(*, idea: str, spec: dict[str, Any], title: str) -> str:
    candidates = [
        spec.get("desired_outcome"),
        spec.get("proposed_solution"),
        spec.get("user_goal"),
        spec.get("raw"),
        idea,
    ]
    for candidate in candidates:
        summary = _collapse_whitespace(candidate)
        if summary:
            if summary == title:
                continue
            return summary[:300]
    return title[:300]


def _initiative_acceptance_criteria(
    *,
    success_criteria: list[str],
    clarification_status: str,
    open_questions: list[str],
) -> list[str]:
    criteria = [item for item in success_criteria if item]
    if criteria:
        return criteria
    fallback = ["A usable initiative brief exists with explicit summary and queue metadata."]
    if clarification_status == "decision_complete":
        fallback.append(
            "Founder triage can decompose the initiative into bounded work with file scope and validation."
        )
    else:
        fallback.append("Open questions remain captured before any boss-ready work is created.")
    if open_questions:
        fallback.append(
            f"{len(open_questions)} open question(s) are explicitly preserved in the intake artifact."
        )
    return fallback


def _initiative_validation_steps(
    *,
    clarification_status: str,
    open_questions: list[str],
) -> list[str]:
    validation = [
        "Brief includes title, summary, merge class, autonomy mode, and preferred worker/reviewer defaults."
    ]
    if clarification_status == "decision_complete":
        validation.append(
            "Founder triage yields bounded handoffs with acceptance criteria, validation, and file scope."
        )
    else:
        validation.append(
            "Artifact remains `idea-intake` only until clarification reaches decision_complete."
        )
    if open_questions:
        validation.append(
            "Open questions are visible on the issue and block queue-ready execution."
        )
    return validation


def _brief_issue_prefix(brief: dict[str, Any]) -> str:
    completeness = str(brief.get("clarification_completeness_status", "")).strip().lower()
    return "Initiative" if completeness == "decision_complete" else "Idea Intake"


def _brief_issue_labels(brief: dict[str, Any]) -> list[str]:
    completeness = str(brief.get("clarification_completeness_status", "")).strip().lower()
    if completeness != "decision_complete":
        return ["idea-intake"]
    return [
        "initiative",
        f"risk:{brief.get('risk', 'medium')}",
        f"merge:{brief.get('merge_class', 'manual')}",
        f"track:{brief.get('track', '1')}",
        f"autonomy:{brief.get('autonomy_mode', 'checkpoint')}",
    ]


def _brief_persistence_errors(brief: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if _is_placeholder_title(brief.get("title")):
        errors.append("title must be non-empty and not a placeholder")
    if not _text(brief.get("summary")):
        errors.append("summary is required")
    if not [item for item in brief.get("acceptance_criteria", []) if _text(item)]:
        errors.append("acceptance criteria are required")
    if not [item for item in brief.get("validation", []) if _text(item)]:
        errors.append("validation steps are required")
    if not _text(brief.get("merge_class")):
        errors.append("merge class is required")
    if not _text(brief.get("autonomy_mode")):
        errors.append("autonomy mode is required")
    return errors


def _handoff_persistence_errors(handoff: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if _is_placeholder_title(handoff.get("task_title")):
        errors.append("task title must be non-empty and not a placeholder")
    if not [item for item in handoff.get("acceptance_criteria", []) if _text(item)]:
        errors.append("acceptance criteria are required")
    if not [item for item in handoff.get("validation", []) if _text(item)]:
        errors.append("validation steps are required")
    if not _text(handoff.get("merge_class")):
        errors.append("merge class is required")
    if not _text(handoff.get("autonomy_mode")):
        errors.append("autonomy mode is required")
    return errors


async def _run_idea_intake_with_cleanup(**kwargs: Any) -> dict[str, Any]:
    """Run intake and release shared API connectors before exit."""
    try:
        return await _run_idea_intake(**kwargs)
    finally:
        await _close_shared_agent_connector()


async def _run_idea_triage_with_cleanup(**kwargs: Any) -> dict[str, Any]:
    """Run founder triage and release shared API connectors before exit."""
    try:
        return await _run_idea_triage(**kwargs)
    finally:
        await _close_shared_agent_connector()


async def _run_idea_review_with_cleanup(**kwargs: Any) -> dict[str, Any]:
    """Run founder review and release shared API connectors before exit."""
    try:
        return await _run_idea_review(**kwargs)
    finally:
        await _close_shared_agent_connector()


async def _run_idea_intake(
    *,
    idea: str,
    skip_clarify: bool,
    priority: str,
    track: str,
    repo: str | None,
    risk: str,
    merge_class: str,
    autonomy_mode: str,
    worker_model: str,
    review_model: str,
    create_issue: bool,
) -> dict[str, Any]:
    """Turn a vague idea into a structured initiative brief."""
    repo_name = repo or _default_repo()
    spec = await _generate_spec(idea, skip_clarify=skip_clarify)
    brief = _compose_initiative_brief(
        idea=idea,
        spec=spec,
        priority=priority,
        track=track,
        risk=risk,
        merge_class=merge_class,
        autonomy_mode=autonomy_mode,
        worker_model=worker_model,
        review_model=review_model,
    )
    result: dict[str, Any] = {
        "status": "brief_created",
        "repo": repo_name,
        "brief": brief,
    }
    if create_issue:
        result["issue"] = await _create_intake_issue(brief, repo=repo_name)
    return result


async def _run_idea_triage(
    *,
    idea: str,
    skip_clarify: bool,
    priority: str,
    track: str,
    repo: str | None,
    max_tasks: int,
    risk: str,
    merge_class: str,
    autonomy_mode: str,
    worker_model: str,
    review_model: str,
    create_issues: bool,
) -> dict[str, Any]:
    """Convert a clarified initiative into founder handoffs."""
    repo_name = repo or _default_repo()
    spec = await _generate_spec(idea, skip_clarify=skip_clarify)
    brief = _compose_initiative_brief(
        idea=idea,
        spec=spec,
        priority=priority,
        track=track,
        risk=risk,
        merge_class=merge_class,
        autonomy_mode=autonomy_mode,
        worker_model=worker_model,
        review_model=review_model,
    )
    if brief["clarification_completeness_status"] != "decision_complete":
        return {
            "status": "needs_clarification",
            "repo": repo_name,
            "brief": brief,
            "handoffs": [],
            "next_actions": list(brief.get("open_questions", []) or []),
        }

    handoffs = await _generate_founder_handoffs(
        brief=brief,
        spec=spec,
        max_tasks=max_tasks,
        worker_model=worker_model,
        review_model=review_model,
    )
    result: dict[str, Any] = {
        "status": "triaged",
        "repo": repo_name,
        "brief": brief,
        "handoffs": handoffs,
    }
    if create_issues:
        result["issues"] = await _create_triage_issues(
            handoffs,
            brief=brief,
            repo=repo_name,
        )
    return result


async def _run_idea_review(
    *,
    idea: str,
    skip_clarify: bool,
    priority: str,
    track: str,
    repo: str | None,
    max_tasks: int,
    risk: str,
    merge_class: str,
    autonomy_mode: str,
    worker_model: str,
    review_model: str,
    create_issues: bool,
) -> dict[str, Any]:
    """Review founder handoffs and generate structured follow-up tasks."""
    repo_name = repo or _default_repo()
    spec = await _generate_spec(idea, skip_clarify=skip_clarify)
    brief = _compose_initiative_brief(
        idea=idea,
        spec=spec,
        priority=priority,
        track=track,
        risk=risk,
        merge_class=merge_class,
        autonomy_mode=autonomy_mode,
        worker_model=worker_model,
        review_model=review_model,
    )
    if brief["clarification_completeness_status"] != "decision_complete":
        return {
            "status": "needs_clarification",
            "repo": repo_name,
            "brief": brief,
            "handoffs": [],
            "review": {"status": "blocked", "findings": [], "followups": []},
            "next_actions": list(brief.get("open_questions", []) or []),
        }

    handoffs = await _generate_founder_handoffs(
        brief=brief,
        spec=spec,
        max_tasks=max_tasks,
        worker_model=worker_model,
        review_model=review_model,
    )
    review = await _review_founder_handoffs(
        brief=brief,
        handoffs=handoffs,
        review_model=review_model,
    )
    result: dict[str, Any] = {
        "status": str(review.get("status", "approved")).strip() or "approved",
        "repo": repo_name,
        "brief": brief,
        "handoffs": handoffs,
        "review": review,
    }
    followups = list(review.get("followups", []) or [])
    if create_issues and followups:
        result["issues"] = await _create_triage_issues(
            followups,
            brief=brief,
            repo=repo_name,
        )
    return result


def _compose_initiative_brief(
    *,
    idea: str,
    spec: dict[str, Any],
    priority: str,
    track: str,
    risk: str,
    merge_class: str,
    autonomy_mode: str,
    worker_model: str,
    review_model: str,
) -> dict[str, Any]:
    """Create a machine-readable initiative brief from spec output."""
    surfaces = list(dict.fromkeys(spec.get("affected_surfaces") or []))
    if not surfaces:
        surfaces = _infer_surfaces(
            {"title": spec.get("title", ""), "description": spec.get("raw", "")}
        )
    success_criteria = [
        str(item).strip() for item in spec.get("success_criteria", []) if str(item).strip()
    ]
    if not success_criteria:
        criteria_section = next(
            (
                str(section.get("content", "")).strip()
                for section in spec.get("sections", [])
                if str(section.get("name", "")).strip() == "success_criteria"
            ),
            "",
        )
        if criteria_section:
            success_criteria = [
                part.strip() for part in criteria_section.split(";") if part.strip()
            ]

    open_questions = [
        str(item).strip() for item in spec.get("open_questions", []) if str(item).strip()
    ]
    clarification_status = _text(spec.get("clarification_status")) or "draft"
    title = _derive_initiative_title(idea=idea, spec=spec)
    desired_outcome = _text(spec.get("desired_outcome")) or title
    summary = _derive_initiative_summary(idea=idea, spec=spec, title=title)
    proof_expected = "Acceptance criteria satisfied, targeted validation attached, and adversarial review notes recorded."
    acceptance_criteria = _initiative_acceptance_criteria(
        success_criteria=success_criteria,
        clarification_status=clarification_status,
        open_questions=open_questions,
    )
    validation = _initiative_validation_steps(
        clarification_status=clarification_status,
        open_questions=open_questions,
    )

    return {
        "title": title,
        "summary": summary,
        "user_goal": _text(spec.get("user_goal")) or idea,
        "desired_business_outcome": desired_outcome,
        "success_criteria": success_criteria,
        "acceptance_criteria": acceptance_criteria,
        "validation": validation,
        "constraints": [],
        "explicit_non_goals": [],
        "affected_product_surfaces": surfaces,
        "affected_surfaces": surfaces,
        "proof_evidence_expected": proof_expected,
        "sequencing_priority": priority,
        "clarification_completeness_status": clarification_status,
        "open_questions": open_questions,
        "risk": risk,
        "merge_class": merge_class,
        "autonomy_mode": autonomy_mode,
        "track": track,
        "preferred_worker_agent": worker_model,
        "preferred_reviewer_agent": review_model,
    }


async def _generate_founder_handoffs(
    *,
    brief: dict[str, Any],
    spec: dict[str, Any],
    max_tasks: int,
    worker_model: str,
    review_model: str,
) -> list[dict[str, Any]]:
    """Generate structured founder handoffs from a clarified initiative."""
    file_scope_hints = SwarmSpec.infer_file_scope_hints(_text(spec.get("raw")) or "")
    success_criteria = [
        str(item).strip() for item in brief.get("success_criteria", []) if str(item).strip()
    ]
    constraints = [str(item).strip() for item in brief.get("constraints", []) if str(item).strip()]
    planning_prompt = _text(spec.get("raw")) or _text(brief.get("user_goal")) or ""
    decomposer = TaskDecomposer()
    planner_timeout = float(os.environ.get("ARAGORA_IDEA_TRIAGE_TIMEOUT_SECONDS", "120") or 120)

    try:
        decomposition = await decomposer.analyze_with_model(
            planning_prompt,
            planner_model=worker_model,
            timeout_seconds=planner_timeout,
            file_scope_hints=file_scope_hints or None,
            acceptance_criteria=success_criteria or None,
            constraints=constraints or None,
        )
    except Exception as exc:
        logger.warning("Idea triage planner fell back to heuristic decomposition: %s", exc)
        decomposition = decomposer.analyze(
            planning_prompt, file_scope_hints=file_scope_hints or None
        )

    subtasks = list(getattr(decomposition, "subtasks", []) or [])[: max(1, max_tasks)]
    bridge = NomicPipelineBridge(repo_path=Path.cwd())
    work_orders = [item.to_dict() for item in bridge.build_work_orders(subtasks)]
    handoffs: list[dict[str, Any]] = []
    for index, work_order in enumerate(work_orders[: max(1, max_tasks)], start=1):
        file_scope = [
            str(item).strip() for item in work_order.get("file_scope", []) if str(item).strip()
        ]
        if not file_scope:
            file_scope = list(file_scope_hints)
        desired_reviewer = (
            review_model
            if review_model != worker_model
            else ("codex" if worker_model != "codex" else "claude")
        )
        policy = apply_delivery_policy(
            file_scope=file_scope,
            requested_risk=_text(brief.get("risk")) or "medium",
            requested_merge_class=_text(brief.get("merge_class")) or "manual",
            requested_autonomy_mode=_text(brief.get("autonomy_mode")) or "checkpoint",
        )
        validation = _work_order_validation(work_order)
        acceptance = _work_order_acceptance_criteria(work_order, brief=brief, validation=validation)
        handoffs.append(
            {
                "handoff_id": f"handoff_{index}",
                "task_title": _text(work_order.get("title")) or f"Handoff {index}",
                "why_now": (
                    f"Advances the initiative '{brief.get('title', '')}' with a bounded slice that "
                    f"fits the current {brief.get('sequencing_priority', 'medium')} priority lane."
                ),
                "repo_evidence": _repo_evidence_for_scope(file_scope),
                "acceptance_criteria": acceptance,
                "validation": validation,
                "risk": policy["effective_risk"],
                "merge_class": policy["effective_merge_class"],
                "autonomy_mode": policy["effective_autonomy_mode"],
                "preferred_worker_agent": worker_model,
                "preferred_reviewer_agent": desired_reviewer,
                "file_scope": file_scope,
                "policy_reasons": list(policy.get("policy_reasons", []) or []),
                "labels": [
                    "boss-ready",
                    f"risk:{policy['effective_risk']}",
                    f"merge:{'auto' if policy['effective_merge_class'] == 'low_risk' else 'manual'}",
                    f"track:{brief.get('track', '1')}",
                    f"autonomy:{policy['effective_autonomy_mode']}",
                ],
                "description": _text(work_order.get("description")) or "",
            }
        )
    return handoffs


def _work_order_validation(work_order: dict[str, Any]) -> list[str]:
    tests = [
        str(item).strip() for item in work_order.get("expected_tests", []) if str(item).strip()
    ]
    return tests or ["python3 -m pytest tests/ -q -k 'not benchmark'"]


def _work_order_acceptance_criteria(
    work_order: dict[str, Any],
    *,
    brief: dict[str, Any],
    validation: list[str],
) -> list[str]:
    criteria: list[str] = []
    success = dict(work_order.get("success_criteria") or {})
    for key, value in success.items():
        key_text = str(key).strip().lower()
        if key_text == "tests":
            continue
        criteria.extend(_stringify_success_value(str(key), value))
    criteria.extend(f"Run and satisfy: {command}" for command in validation)
    if not criteria:
        criteria.extend(
            str(item).strip() for item in brief.get("success_criteria", []) if str(item).strip()
        )
    if not criteria:
        criteria.append("Implementation complete for the declared file scope.")
    return list(dict.fromkeys(criteria))


def _stringify_success_value(key: str, value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        nested: list[str] = []
        for nested_key, nested_value in value.items():
            nested.extend(_stringify_success_value(f"{key}.{nested_key}", nested_value))
        return nested
    if isinstance(value, list):
        return [f"{key}: {str(item).strip()}" for item in value if str(item).strip()]
    text = str(value).strip()
    return [f"{key}: {text}"] if text else []


def _repo_evidence_for_scope(file_scope: list[str]) -> list[str]:
    """Collect concrete repo hints for a file scope."""
    repo_root = Path.cwd()
    evidence: list[str] = []
    for raw_scope in file_scope:
        scope = str(raw_scope).strip().rstrip("/")
        if not scope:
            continue
        if (repo_root / scope).exists():
            evidence.append(scope)
            continue
        top_level = scope.split("/", 1)[0]
        if top_level and (repo_root / top_level).exists():
            evidence.append(top_level)
    return list(dict.fromkeys(evidence))[:5]


def _deterministic_review_founder_handoffs(
    *,
    brief: dict[str, Any],
    handoffs: list[dict[str, Any]],
    review_model: str,
) -> dict[str, Any]:
    """Apply deterministic founder-review checks to generated handoffs."""
    findings: list[dict[str, Any]] = []
    followups: list[dict[str, Any]] = []

    if not handoffs:
        findings.append(
            {
                "finding_id": "finding_1",
                "severity": "high",
                "category": "missing_handoffs",
                "title": "No executable handoffs were produced",
                "detail": "Founder triage returned zero handoffs, so the queue cannot advance.",
                "handoff_ids": [],
                "recommended_action": "Regenerate the plan with bounded tasks and explicit file scope.",
            }
        )

    generic_validation = "python3 -m pytest tests/ -q -k 'not benchmark'"
    scope_index: dict[str, list[str]] = {}
    followup_keys: set[tuple[str, str]] = set()

    for handoff in handoffs:
        handoff_id = _text(handoff.get("handoff_id")) or "handoff"
        task_title = _text(handoff.get("task_title")) or handoff_id
        file_scope = [
            str(item).strip() for item in handoff.get("file_scope", []) if str(item).strip()
        ]
        acceptance = [
            str(item).strip()
            for item in handoff.get("acceptance_criteria", [])
            if str(item).strip()
        ]
        validation = [
            str(item).strip() for item in handoff.get("validation", []) if str(item).strip()
        ]
        repo_evidence = [
            str(item).strip() for item in handoff.get("repo_evidence", []) if str(item).strip()
        ]
        worker = _text(handoff.get("preferred_worker_agent")) or "claude"
        reviewer = _text(handoff.get("preferred_reviewer_agent")) or "codex"
        policy_reasons = [
            str(item).strip() for item in handoff.get("policy_reasons", []) if str(item).strip()
        ]

        for path in file_scope:
            scope_index.setdefault(path, []).append(handoff_id)

        if not acceptance:
            findings.append(
                _review_finding(
                    finding_id=f"finding_{len(findings) + 1}",
                    severity="high",
                    category="acceptance_contract",
                    title=f"{task_title}: acceptance criteria missing",
                    detail="The handoff lacks explicit acceptance criteria, so success is unverifiable.",
                    handoff_ids=[handoff_id],
                    recommended_action="Add concrete acceptance criteria before dispatch.",
                )
            )
            followup = _review_followup(
                brief=brief,
                handoff=handoff,
                review_model=review_model,
                title_suffix="Add acceptance criteria contract",
                description=(
                    "Define concrete acceptance criteria for this handoff so execution can be "
                    "reviewed against a stable contract."
                ),
                acceptance_criteria=[
                    "Acceptance criteria are explicit, bounded, and outcome-based.",
                    "Acceptance criteria are persisted on the queue item before execution.",
                ],
                validation=["python3 -m pytest tests/ -q -k 'not benchmark'"],
            )
            _append_followup(followups, followup_keys, followup)

        if not validation:
            findings.append(
                _review_finding(
                    finding_id=f"finding_{len(findings) + 1}",
                    severity="high",
                    category="validation_contract",
                    title=f"{task_title}: validation contract missing",
                    detail="The handoff has no validation commands, so the boss loop cannot verify completion.",
                    handoff_ids=[handoff_id],
                    recommended_action="Add targeted validation commands before dispatch.",
                )
            )
            followup = _review_followup(
                brief=brief,
                handoff=handoff,
                review_model=review_model,
                title_suffix="Add targeted validation contract",
                description=(
                    "Add targeted validation commands for this handoff so execution can be "
                    "proven before merge."
                ),
                acceptance_criteria=[
                    "Validation commands target the declared file scope.",
                    "Validation commands are narrow enough to catch regressions without scanning the whole repo.",
                ],
                validation=[generic_validation],
            )
            _append_followup(followups, followup_keys, followup)
        elif all(command == generic_validation for command in validation):
            findings.append(
                _review_finding(
                    finding_id=f"finding_{len(findings) + 1}",
                    severity="medium",
                    category="validation_contract",
                    title=f"{task_title}: validation is too generic",
                    detail=(
                        "The handoff uses only a broad default pytest command, which is weak as a "
                        "task-specific validation contract."
                    ),
                    handoff_ids=[handoff_id],
                    recommended_action="Replace the generic test command with task-specific validation.",
                )
            )
            followup = _review_followup(
                brief=brief,
                handoff=handoff,
                review_model=review_model,
                title_suffix="Tighten validation contract",
                description=(
                    "Replace broad default validation with targeted commands tied to the declared "
                    "file scope and acceptance criteria."
                ),
                acceptance_criteria=[
                    "Validation commands are targeted to this handoff's scope.",
                    "At least one validation command directly exercises the changed behavior.",
                ],
                validation=[generic_validation],
            )
            _append_followup(followups, followup_keys, followup)

        if not repo_evidence:
            findings.append(
                _review_finding(
                    finding_id=f"finding_{len(findings) + 1}",
                    severity="medium",
                    category="repo_evidence",
                    title=f"{task_title}: repo evidence missing",
                    detail=(
                        "The handoff does not reference concrete repo files or modules, which weakens "
                        "the architectural grounding for implementation."
                    ),
                    handoff_ids=[handoff_id],
                    recommended_action="Capture concrete repo evidence before queueing the task.",
                )
            )

        if not file_scope:
            findings.append(
                _review_finding(
                    finding_id=f"finding_{len(findings) + 1}",
                    severity="high",
                    category="file_scope",
                    title=f"{task_title}: file scope missing",
                    detail="The handoff does not declare file scope, so safe parallel execution is impossible.",
                    handoff_ids=[handoff_id],
                    recommended_action="Add explicit file scope before dispatch.",
                )
            )

        if worker == reviewer:
            findings.append(
                _review_finding(
                    finding_id=f"finding_{len(findings) + 1}",
                    severity="medium",
                    category="cross_model_review",
                    title=f"{task_title}: reviewer duplicates worker",
                    detail=(
                        "The preferred reviewer matches the preferred worker, which undermines "
                        "heterogeneous adversarial review."
                    ),
                    handoff_ids=[handoff_id],
                    recommended_action="Route review to a different model family.",
                )
            )

        if policy_reasons and (
            _text(handoff.get("merge_class")) != "manual"
            or _text(handoff.get("autonomy_mode")) != "checkpoint"
        ):
            findings.append(
                _review_finding(
                    finding_id=f"finding_{len(findings) + 1}",
                    severity="high",
                    category="safety_policy",
                    title=f"{task_title}: sensitive scope is not fully downgraded",
                    detail=(
                        "The handoff touches sensitive scope but is not marked manual/checkpoint. "
                        f"Reasons: {', '.join(policy_reasons)}."
                    ),
                    handoff_ids=[handoff_id],
                    recommended_action="Force manual merge class and checkpoint autonomy.",
                )
            )

    overlap_map = {path: ids for path, ids in scope_index.items() if len(set(ids)) > 1}
    for path, handoff_ids in overlap_map.items():
        findings.append(
            _review_finding(
                finding_id=f"finding_{len(findings) + 1}",
                severity="high",
                category="scope_overlap",
                title=f"Overlapping file scope detected: {path}",
                detail=(
                    "Multiple handoffs claim the same file scope, which will create queue conflicts "
                    "and ambiguous ownership."
                ),
                handoff_ids=list(dict.fromkeys(handoff_ids)),
                recommended_action="Split or resequence the overlapping handoffs before dispatch.",
            )
        )

    severities = {str(item.get("severity", "")).strip().lower() for item in findings}
    if {"high", "medium"} & severities:
        status = "changes_requested"
    elif findings:
        status = "approved_with_followups"
    else:
        status = "approved"

    summary = (
        f"Founder review completed with {len(findings)} finding(s) and "
        f"{len(followups)} follow-up task(s)."
    )
    return {
        "status": status,
        "summary": summary,
        "findings": findings,
        "followups": followups,
    }


async def _review_founder_handoffs(
    *,
    brief: dict[str, Any],
    handoffs: list[dict[str, Any]],
    review_model: str,
) -> dict[str, Any]:
    deterministic_review = _deterministic_review_founder_handoffs(
        brief=brief,
        handoffs=handoffs,
        review_model=review_model,
    )
    model_review = await _model_review_founder_handoffs(
        brief=brief,
        handoffs=handoffs,
        review_model=review_model,
    )
    return _merge_review_results(
        deterministic_review=deterministic_review,
        model_review=model_review,
    )


async def _model_review_founder_handoffs(
    *,
    brief: dict[str, Any],
    handoffs: list[dict[str, Any]],
    review_model: str,
) -> dict[str, Any]:
    if not handoffs:
        return {"status": "approved", "summary": "", "findings": [], "followups": []}

    try:
        from aragora.agents import create_agent

        agent = create_agent(
            review_model,
            name="founder-review-red-team",
            role="critic",
            enable_fallback=False,
        )
        timeout_seconds = float(
            os.environ.get("ARAGORA_FOUNDER_REVIEW_TIMEOUT_SECONDS", "90") or 90
        )
        response = await asyncio.wait_for(
            agent.generate(_founder_review_model_prompt(brief=brief, handoffs=handoffs)),
            timeout=timeout_seconds,
        )
    except Exception as exc:
        logger.warning("model founder review unavailable: %s", exc)
        return {
            "status": "approved",
            "summary": f"Model founder review unavailable: {exc}",
            "findings": [],
            "followups": [],
        }

    payload = _extract_first_json_object(response)
    findings = _normalize_model_review_findings(payload, handoffs=handoffs)
    followups = _model_review_followups(
        brief=brief,
        handoffs=handoffs,
        findings=findings,
        review_model=review_model,
    )
    return {
        "status": _review_status_from_findings(findings, followups=followups),
        "summary": _text(payload.get("summary"))
        or f"Model founder review produced {len(findings)} finding(s).",
        "findings": findings,
        "followups": followups,
    }


def _merge_review_results(
    *,
    deterministic_review: dict[str, Any],
    model_review: dict[str, Any],
) -> dict[str, Any]:
    findings = [
        item
        for item in [
            *(deterministic_review.get("findings", []) or []),
            *(model_review.get("findings", []) or []),
        ]
        if isinstance(item, dict)
    ]
    followups: list[dict[str, Any]] = []
    seen_followups: set[tuple[str, str]] = set()
    for followup in deterministic_review.get("followups", []) or []:
        if isinstance(followup, dict):
            _append_followup(followups, seen_followups, followup)
    for followup in model_review.get("followups", []) or []:
        if isinstance(followup, dict):
            _append_followup(followups, seen_followups, followup)

    deterministic_summary = _text(deterministic_review.get("summary"))
    model_summary = _text(model_review.get("summary"))
    summary_parts = [item for item in [deterministic_summary, model_summary] if item]
    return {
        "status": _review_status_from_findings(findings, followups=followups),
        "summary": " ".join(summary_parts)
        or (
            f"Founder review completed with {len(findings)} finding(s) and "
            f"{len(followups)} follow-up task(s)."
        ),
        "findings": findings,
        "followups": followups,
        "layers": {
            "deterministic": {
                "findings": len(
                    [
                        item
                        for item in deterministic_review.get("findings", [])
                        if isinstance(item, dict)
                    ]
                ),
                "followups": len(
                    [
                        item
                        for item in deterministic_review.get("followups", [])
                        if isinstance(item, dict)
                    ]
                ),
            },
            "model": {
                "findings": len(
                    [item for item in model_review.get("findings", []) if isinstance(item, dict)]
                ),
                "followups": len(
                    [item for item in model_review.get("followups", []) if isinstance(item, dict)]
                ),
            },
        },
    }


def _review_status_from_findings(
    findings: list[dict[str, Any]],
    *,
    followups: list[dict[str, Any]] | None = None,
) -> str:
    severities = {str(item.get("severity", "")).strip().lower() for item in findings}
    if {"high", "medium"} & severities:
        return "changes_requested"
    if findings or (followups or []):
        return "approved_with_followups"
    return "approved"


def _founder_review_model_prompt(
    *,
    brief: dict[str, Any],
    handoffs: list[dict[str, Any]],
) -> str:
    compact_handoffs = [
        {
            "handoff_id": _text(item.get("handoff_id")),
            "task_title": _text(item.get("task_title")),
            "why_now": _text(item.get("why_now")),
            "repo_evidence": list(item.get("repo_evidence", []) or []),
            "file_scope": list(item.get("file_scope", []) or []),
            "acceptance_criteria": list(item.get("acceptance_criteria", []) or []),
            "validation": list(item.get("validation", []) or []),
            "risk": _text(item.get("risk")),
            "merge_class": _text(item.get("merge_class")),
            "autonomy_mode": _text(item.get("autonomy_mode")),
            "preferred_worker_agent": _text(item.get("preferred_worker_agent")),
            "preferred_reviewer_agent": _text(item.get("preferred_reviewer_agent")),
            "policy_reasons": list(item.get("policy_reasons", []) or []),
        }
        for item in handoffs
    ]
    prompt_payload = {
        "initiative": {
            "title": brief.get("title"),
            "user_goal": brief.get("user_goal"),
            "desired_business_outcome": brief.get("desired_business_outcome"),
            "success_criteria": list(brief.get("success_criteria", []) or []),
            "constraints": list(brief.get("constraints", []) or []),
            "explicit_non_goals": list(brief.get("explicit_non_goals", []) or []),
            "affected_product_surfaces": list(brief.get("affected_product_surfaces", []) or []),
        },
        "handoffs": compact_handoffs,
    }
    return (
        "You are performing adversarial founder review on proposed execution handoffs. "
        "Look for hidden ambiguity, weak acceptance criteria, missing validation, unsafe merge/autonomy "
        "policy, architectural debt, overlap, or poor cross-model review pairing.\n\n"
        "Return JSON only with this shape:\n"
        "{\n"
        '  "summary": "short summary",\n'
        '  "findings": [\n'
        "    {\n"
        '      "severity": "low|medium|high",\n'
        '      "category": "short_category",\n'
        '      "title": "short title",\n'
        '      "detail": "why this matters",\n'
        '      "handoff_ids": ["handoff_id"],\n'
        '      "recommended_action": "specific fix"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Only include findings that materially improve execution quality or safety.\n\n"
        f"{json.dumps(prompt_payload, indent=2, sort_keys=True)}"
    )


def _extract_first_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, ValueError):
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        parsed = json.loads(raw[start : end + 1])
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _normalize_model_review_findings(
    payload: dict[str, Any],
    *,
    handoffs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    known_handoff_ids = {
        _text(item.get("handoff_id")) or "" for item in handoffs if _text(item.get("handoff_id"))
    }
    findings_raw = payload.get("findings")
    if not isinstance(findings_raw, list):
        return []
    findings: list[dict[str, Any]] = []
    for item in findings_raw:
        if not isinstance(item, dict):
            continue
        title = _text(item.get("title"))
        detail = _text(item.get("detail"))
        if not title or not detail:
            continue
        handoff_ids: list[str] = []
        raw_handoff_ids = item.get("handoff_ids")
        if isinstance(raw_handoff_ids, list):
            handoff_ids = [
                str(value).strip()
                for value in raw_handoff_ids
                if str(value).strip() and str(value).strip() in known_handoff_ids
            ]
        elif _text(raw_handoff_ids) and _text(raw_handoff_ids) in known_handoff_ids:
            handoff_ids = [_text(raw_handoff_ids) or ""]
        severity = _text(item.get("severity")) or "medium"
        if severity not in {"low", "medium", "high"}:
            severity = "medium"
        findings.append(
            _review_finding(
                finding_id=f"model_finding_{len(findings) + 1}",
                severity=severity,
                category=_text(item.get("category")) or "model_review",
                title=title,
                detail=detail,
                handoff_ids=handoff_ids,
                recommended_action=(
                    _text(item.get("recommended_action"))
                    or "Address this founder-review concern before autonomous execution."
                ),
            )
        )
    return findings


def _model_review_followups(
    *,
    brief: dict[str, Any],
    handoffs: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    review_model: str,
) -> list[dict[str, Any]]:
    followups: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    handoff_by_id = {
        _text(item.get("handoff_id")) or "": item
        for item in handoffs
        if _text(item.get("handoff_id"))
    }
    generic_validation = ["python3 -m pytest tests/ -q -k 'not benchmark'"]
    for finding in findings:
        severity = _text(finding.get("severity")) or "medium"
        if severity not in {"high", "medium"}:
            continue
        for handoff_id in [
            str(item).strip() for item in finding.get("handoff_ids", []) if str(item).strip()
        ]:
            handoff = handoff_by_id.get(handoff_id)
            if not isinstance(handoff, dict):
                continue
            followup = _review_followup(
                brief=brief,
                handoff=handoff,
                review_model=review_model,
                title_suffix=_truncate_title(
                    f"Address model review: {_text(finding.get('title')) or 'follow-up'}"
                ),
                description=_text(finding.get("detail"))
                or "Resolve the model-detected founder review concern before execution.",
                acceptance_criteria=[
                    _text(finding.get("recommended_action"))
                    or "Model-detected founder review concern is addressed.",
                    "The handoff remains bounded and reviewable after the correction.",
                ],
                validation=list(handoff.get("validation", []) or []) or generic_validation,
            )
            _append_followup(followups, seen, followup)
    return followups


def _truncate_title(value: str, *, limit: int = 72) -> str:
    text = _text(value) or ""
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _review_finding(
    *,
    finding_id: str,
    severity: str,
    category: str,
    title: str,
    detail: str,
    handoff_ids: list[str],
    recommended_action: str,
) -> dict[str, Any]:
    return {
        "finding_id": finding_id,
        "severity": severity,
        "category": category,
        "title": title,
        "detail": detail,
        "handoff_ids": handoff_ids,
        "recommended_action": recommended_action,
    }


def _review_followup(
    *,
    brief: dict[str, Any],
    handoff: dict[str, Any],
    review_model: str,
    title_suffix: str,
    description: str,
    acceptance_criteria: list[str],
    validation: list[str],
) -> dict[str, Any]:
    file_scope = [str(item).strip() for item in handoff.get("file_scope", []) if str(item).strip()]
    policy = apply_delivery_policy(
        file_scope=file_scope,
        requested_risk="medium",
        requested_merge_class="manual",
        requested_autonomy_mode="checkpoint",
    )
    task_title = _text(handoff.get("task_title")) or "Untitled handoff"
    return {
        "handoff_id": f"review_followup_{_slug(task_title)}",
        "task_title": f"{task_title}: {title_suffix}",
        "why_now": (
            f"Founder review flagged this correction before autonomous execution of "
            f"'{brief.get('title', '')}'."
        ),
        "repo_evidence": list(handoff.get("repo_evidence", []) or []),
        "acceptance_criteria": acceptance_criteria,
        "validation": validation,
        "risk": policy["effective_risk"],
        "merge_class": policy["effective_merge_class"],
        "autonomy_mode": policy["effective_autonomy_mode"],
        "preferred_worker_agent": _text(handoff.get("preferred_worker_agent")) or "claude",
        "preferred_reviewer_agent": review_model,
        "file_scope": file_scope,
        "policy_reasons": list(policy.get("policy_reasons", []) or [])
        + ["founder_review_followup"],
        "labels": [
            "boss-ready",
            "review-followup",
            f"risk:{policy['effective_risk']}",
            f"merge:{'auto' if policy['effective_merge_class'] == 'low_risk' else 'manual'}",
            f"track:{brief.get('track', '1')}",
            f"autonomy:{policy['effective_autonomy_mode']}",
        ],
        "description": description,
    }


def _append_followup(
    followups: list[dict[str, Any]],
    seen: set[tuple[str, str]],
    followup: dict[str, Any],
) -> None:
    key = (
        _text(followup.get("task_title")) or "",
        "|".join(str(item).strip() for item in followup.get("file_scope", []) if str(item).strip()),
    )
    if key in seen:
        return
    seen.add(key)
    followups.append(followup)


def _slug(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value or ""))
    while "--" in text:
        text = text.replace("--", "-")
    return text.strip("-") or "item"


async def _create_intake_issue(brief: dict[str, Any], *, repo: str) -> dict[str, Any]:
    """Persist an initiative brief as a GitHub issue."""
    errors = _brief_persistence_errors(brief)
    if errors:
        raise ValueError(f"Cannot persist intake brief: {'; '.join(errors)}")
    prefix = _brief_issue_prefix(brief)
    return await _create_issue_with_optional_labels(
        repo=repo,
        title=f"[{prefix}] {brief.get('title', '').strip()}",
        body=_issue_body_for_brief(brief),
        requested_labels=_brief_issue_labels(brief),
    )


async def _create_triage_issues(
    handoffs: list[dict[str, Any]],
    *,
    brief: dict[str, Any],
    repo: str,
) -> list[dict[str, Any]]:
    """Persist founder handoffs as boss-ready GitHub issues."""
    created: list[dict[str, Any]] = []
    for handoff in handoffs:
        errors = _handoff_persistence_errors(handoff)
        if errors:
            raise ValueError(
                f"Cannot persist boss-ready issue for {handoff.get('task_title', 'unknown task')!r}: "
                + "; ".join(errors)
            )
        created.append(
            await _create_issue_with_optional_labels(
                repo=repo,
                title=str(handoff.get("task_title", "")).strip(),
                body=_triage_issue_body_for_handoff(handoff, brief=brief),
                requested_labels=list(handoff.get("labels", []) or []),
            )
        )
    return created


async def _create_issue_with_optional_labels(
    *,
    repo: str,
    title: str,
    body: str,
    requested_labels: list[str],
) -> dict[str, Any]:
    """Create an issue and retry without labels if the repo is missing them."""
    issue_cmd = ["gh", "issue", "create", "--repo", repo, "--title", title]
    for label in requested_labels:
        issue_cmd.extend(["--label", label])
    issue_cmd.extend(["--body", body])

    result = subprocess.run(issue_cmd, capture_output=True, text=True, timeout=30)
    applied_labels = list(requested_labels)
    fallback_reason = ""
    if result.returncode != 0 and "label" in result.stderr.lower():
        fallback_reason = result.stderr.strip()
        logger.warning(
            "Issue label application failed, retrying without labels: %s", fallback_reason
        )
        result = subprocess.run(
            ["gh", "issue", "create", "--repo", repo, "--title", title, "--body", body],
            capture_output=True,
            text=True,
            timeout=30,
        )
        applied_labels = []

    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(result.stderr.strip() or "Failed to create GitHub issue")

    issue_url = result.stdout.strip()
    return {
        "number": int(issue_url.rstrip("/").split("/")[-1]),
        "url": issue_url,
        "labels_requested": requested_labels,
        "labels_applied": applied_labels,
        "fallback_reason": fallback_reason,
    }


def _issue_body_for_brief(brief: dict[str, Any]) -> str:
    """Render the intake issue body."""
    success_criteria = list(brief.get("success_criteria", []) or [])
    acceptance_criteria = [
        str(item).strip() for item in brief.get("acceptance_criteria", []) if str(item).strip()
    ]
    validation = [str(item).strip() for item in brief.get("validation", []) if str(item).strip()]
    open_questions = list(brief.get("open_questions", []) or [])
    surfaces = list(brief.get("affected_product_surfaces", []) or [])
    next_step = (
        "Run founder triage to decompose this initiative into bounded `boss-ready` issues with "
        "explicit acceptance criteria, validation commands, file scope, and merge policy."
        if str(brief.get("clarification_completeness_status", "")).strip() == "decision_complete"
        else "Keep this as `idea-intake` only. Resolve the open questions before any `boss-ready` issue is created."
    )
    return f"""## Initiative Brief
- Summary: {brief.get("summary", "")[:300]}
- User goal: {brief.get("user_goal", "")[:300]}
- Desired business outcome: {brief.get("desired_business_outcome", "")[:300]}
- Success criteria: {", ".join(success_criteria) or "to be refined"}
- Constraints: {", ".join(brief.get("constraints", [])) or "none captured yet"}
- Explicit non-goals: {", ".join(brief.get("explicit_non_goals", [])) or "none captured yet"}
- Affected product surfaces: {", ".join(surfaces) or "unknown"}
- Proof/evidence expected: {brief.get("proof_evidence_expected", "")[:300]}
- Sequencing priority: {brief.get("sequencing_priority", "medium")}
- Clarification completeness: {brief.get("clarification_completeness_status", "draft")}
- Open questions: {", ".join(open_questions) or "none"}

## Queue Metadata
- Risk: {brief.get("risk", "medium")}
- Merge Class: {brief.get("merge_class", "manual")}
- Autonomy Mode: {brief.get("autonomy_mode", "checkpoint")}
- Track: {brief.get("track", "1")}
- Preferred Worker Agent: {brief.get("preferred_worker_agent", "claude")}
- Preferred Reviewer Agent: {brief.get("preferred_reviewer_agent", "codex")}

## Acceptance Criteria
{chr(10).join(f"- [ ] {item}" for item in acceptance_criteria) or "- [ ] Clarification requirements captured"}

## Validation
{chr(10).join(f"- {item}" for item in validation) or "- Validation still needs to be defined"}

## Next Step
{next_step}
"""


def _triage_issue_body_for_handoff(handoff: dict[str, Any], *, brief: dict[str, Any]) -> str:
    """Render a queue-ready issue body from a founder handoff."""
    repo_evidence = list(handoff.get("repo_evidence", []) or [])
    file_scope = list(handoff.get("file_scope", []) or [])
    acceptance = list(handoff.get("acceptance_criteria", []) or [])
    validation = list(handoff.get("validation", []) or [])
    return f"""## Why Now
{handoff.get("why_now", "")}

## Initiative Context
- Initiative: {brief.get("title", "")}
- User goal: {brief.get("user_goal", "")[:300]}
- Desired business outcome: {brief.get("desired_business_outcome", "")[:300]}

## Repo Evidence
{chr(10).join(f"- {item}" for item in repo_evidence) or "- none captured"}

## File Scope
{chr(10).join(f"- {item}" for item in file_scope) or "- none captured"}

## Queue Metadata
- Risk: {handoff.get("risk", "medium")}
- Merge Class: {handoff.get("merge_class", "manual")}
- Autonomy Mode: {handoff.get("autonomy_mode", "checkpoint")}
- Preferred Worker Agent: {handoff.get("preferred_worker_agent", "claude")}
- Preferred Reviewer Agent: {handoff.get("preferred_reviewer_agent", "codex")}
- Policy Notes: {", ".join(handoff.get("policy_reasons", [])) or "none"}

## Acceptance Criteria
{chr(10).join(f"- [ ] {item}" for item in acceptance) or "- [ ] Implementation complete"}

## Validation
```bash
{chr(10).join(validation) if validation else "python3 -m pytest tests/ -q"}
```

## Description
{handoff.get("description", "")[:500]}

## Definition of Done
Deliverable created, bounded scope respected, validation evidence attached, and reviewer findings recorded.
"""


def _print_intake_result(result: dict[str, Any]) -> None:
    """Print a human-readable intake summary."""
    brief = result.get("brief", {})
    print("\n============================================================")
    print("Initiative Brief")
    print("============================================================")
    print(f"Title: {brief.get('title', '?')}")
    print(f"Goal: {brief.get('user_goal', '?')}")
    print(f"Outcome: {brief.get('desired_business_outcome', '?')}")
    print(f"Priority: {brief.get('sequencing_priority', '?')}")
    print(f"Clarification: {brief.get('clarification_completeness_status', '?')}")
    if brief.get("open_questions"):
        print("Open questions:")
        for question in brief["open_questions"]:
            print(f"- {question}")
    if result.get("issue"):
        issue = result["issue"]
        print(f"Issue: #{issue.get('number')} {issue.get('url')}")


def _print_triage_result(result: dict[str, Any]) -> None:
    """Print a human-readable founder-triage summary."""
    brief = result.get("brief", {})
    print("\n============================================================")
    print("Founder Handoffs")
    print("============================================================")
    print(f"Initiative: {brief.get('title', '?')}")
    print(f"Status: {result.get('status', '?')}")
    handoffs = list(result.get("handoffs", []) or [])
    for index, handoff in enumerate(handoffs, start=1):
        print(f"{index}. {handoff.get('task_title', 'untitled')}")
        print(
            f"   worker={handoff.get('preferred_worker_agent', '?')} reviewer={handoff.get('preferred_reviewer_agent', '?')}"
        )
        print(
            f"   risk={handoff.get('risk', '?')} merge={handoff.get('merge_class', '?')} autonomy={handoff.get('autonomy_mode', '?')}"
        )
        scope = ", ".join(handoff.get("file_scope", []) or []) or "none"
        print(f"   scope={scope}")


def _print_review_result(result: dict[str, Any]) -> None:
    """Print a human-readable founder-review summary."""
    brief = result.get("brief", {})
    review = result.get("review", {}) if isinstance(result.get("review"), dict) else {}
    findings = [item for item in review.get("findings", []) if isinstance(item, dict)]
    followups = [item for item in review.get("followups", []) if isinstance(item, dict)]
    print("\n============================================================")
    print("Founder Review")
    print("============================================================")
    print(f"Initiative: {brief.get('title', '?')}")
    print(f"Status: {result.get('status', '?')}")
    print(f"Summary: {review.get('summary', 'No summary available.')}")
    if findings:
        print("Findings:")
        for finding in findings:
            print(
                f"- [{finding.get('severity', '?')}] {finding.get('title', 'untitled')} "
                f"({finding.get('category', '?')})"
            )
    if followups:
        print("Follow-up tasks:")
        for followup in followups:
            print(f"- {followup.get('task_title', 'untitled')}")
