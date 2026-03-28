"""aragora build — Turn a vague idea into executed, reviewed, merged code.

Usage:
    aragora build "I want real-time streaming of agent debate responses"
    aragora build "Add a provider selection UI to settings" --dry-run
    aragora build --from-file ideas.txt

Pipeline:
    1. Clarify: Ask questions to understand the idea (skip with --skip-clarify)
    2. Specify: Run aragora spec to produce structured specification
    3. Debate: Multi-agent debate on the specification quality
    4. Decompose: Break into bounded tasks with acceptance criteria
    5. Plan: Sequence tasks and identify dependencies
    6. Execute: Dispatch to boss loop for implementation
    7. Review: Adversarial review of each change
    8. Iterate: Fix issues found in review
    9. Merge: Land clean changes on main
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from aragora.swarm.delivery_policy import apply_delivery_policy
from aragora.swarm.spec import SwarmSpec

logger = logging.getLogger(__name__)


def cmd_build(args: argparse.Namespace) -> None:
    """Turn a vague idea into executed, reviewed, merged code."""
    idea = getattr(args, "idea", None) or ""
    from_file = getattr(args, "from_file", None)
    dry_run = getattr(args, "dry_run", False)
    skip_clarify = getattr(args, "skip_clarify", False)
    max_tasks = getattr(args, "max_tasks", 5)
    as_json = getattr(args, "json", False)
    repo = str(getattr(args, "repo", None) or "").strip() or None
    worker_model = str(getattr(args, "worker_model", "claude") or "claude").strip() or "claude"
    review_model = str(getattr(args, "review_model", "codex") or "codex").strip() or "codex"
    risk = str(getattr(args, "risk", "medium") or "medium").strip().lower() or "medium"
    merge_class = (
        str(getattr(args, "merge_class", "manual") or "manual").strip().lower() or "manual"
    )
    autonomy_mode = (
        str(getattr(args, "autonomy_mode", "full-auto") or "full-auto").strip().lower()
        or "full-auto"
    )

    if from_file:
        idea = Path(from_file).read_text().strip()
    if not idea:
        print('Usage: aragora build "your idea here"')
        sys.exit(1)

    result = asyncio.run(
        _run_build_pipeline_with_cleanup(
            idea=idea,
            dry_run=dry_run,
            skip_clarify=skip_clarify,
            max_tasks=max_tasks,
            repo=repo,
            worker_model=worker_model,
            review_model=review_model,
            risk=risk,
            merge_class=merge_class,
            autonomy_mode=autonomy_mode,
            emit_progress=not as_json,
        )
    )

    if as_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_result(result)


async def _run_build_pipeline_with_cleanup(**kwargs: Any) -> dict[str, Any]:
    """Run the build pipeline and release shared API connectors before exit."""
    try:
        return await _run_build_pipeline(**kwargs)
    finally:
        await _close_shared_agent_connector()


async def _run_build_pipeline(
    *,
    idea: str,
    dry_run: bool = False,
    skip_clarify: bool = False,
    max_tasks: int = 5,
    repo: str | None = None,
    worker_model: str = "claude",
    review_model: str = "codex",
    risk: str = "medium",
    merge_class: str = "manual",
    autonomy_mode: str = "full-auto",
    emit_progress: bool = True,
) -> dict[str, Any]:
    """Execute the full build pipeline."""
    repo_name = repo or _default_repo()
    result: dict[str, Any] = {
        "idea": idea,
        "dry_run": dry_run,
        "repo": repo_name,
        "routing_defaults": {
            "worker_model": worker_model,
            "review_model": review_model,
        },
        "queue_policy": {
            "risk": risk,
            "merge_class": merge_class,
            "autonomy_mode": autonomy_mode,
        },
        "stages": {},
        "status": "running",
    }
    start = time.monotonic()

    # Stage 1: Specification
    _progress(emit_progress, "\n[1/5] Generating specification from idea...")
    _progress(emit_progress, f"  Idea: {idea[:100]}{'...' if len(idea) > 100 else ''}")
    spec = await _generate_spec(idea, skip_clarify=skip_clarify)
    result["stages"]["spec"] = spec
    _progress(emit_progress, f"  ✓ Spec generated ({len(spec.get('sections', []))} sections)")

    # Stage 2: Founder triage and review
    _progress(emit_progress, "\n[2/5] Running founder triage and review...")
    planning = await _plan_reviewed_tasks(
        idea=idea,
        spec=spec,
        repo=repo_name,
        risk=risk,
        merge_class=merge_class,
        autonomy_mode=autonomy_mode,
        max_tasks=max_tasks,
        worker_model=worker_model,
        review_model=review_model,
    )
    result["stages"]["brief"] = planning["brief"]
    result["stages"]["handoffs"] = planning["handoffs"]
    result["stages"]["review"] = planning["review"]
    tasks = list(planning.get("tasks", []) or [])

    if planning["brief"]["clarification_completeness_status"] != "decision_complete":
        result["status"] = "needs_clarification"
        result["elapsed_seconds"] = time.monotonic() - start
        _progress(
            emit_progress,
            "  ! Clarification is incomplete. Capture the open questions before queue dispatch.",
        )
        return result

    if not tasks:
        _progress(
            emit_progress,
            "  ! Founder review produced no queueable tasks, falling back to bounded task decomposition.",
        )
        fallback_tasks = await _decompose_tasks(spec, max_tasks=max_tasks)
        tasks = _annotate_tasks(
            fallback_tasks,
            spec=spec,
            idea=idea,
            repo=repo_name,
            risk=risk,
            merge_class=merge_class,
            autonomy_mode=autonomy_mode,
            worker_model=worker_model,
            review_model=review_model,
        )

    result["stages"]["tasks"] = tasks
    _progress(emit_progress, f"  ✓ {len(tasks)} tasks identified")
    for i, task in enumerate(tasks, 1):
        _progress(emit_progress, f"    {i}. {task['title']}")

    if dry_run:
        result["status"] = "dry_run_complete"
        result["elapsed_seconds"] = time.monotonic() - start
        _progress(
            emit_progress,
            f"\n[DRY RUN] Would create {len(tasks)} issues and dispatch to boss loop.",
        )
        _progress(emit_progress, "  Run without --dry-run to execute.")
        return result

    # Stage 3: Create GitHub issues
    _progress(emit_progress, "\n[3/5] Creating GitHub issues...")
    issues = await _create_issues(tasks, repo=repo_name)
    result["stages"]["issues"] = issues
    _progress(emit_progress, f"  ✓ {len(issues)} issues created")

    if not issues:
        result["status"] = "issue_creation_failed"
        result["elapsed_seconds"] = time.monotonic() - start
        return result

    routing_preflight = _preflight_boss_routing(repo=repo_name, worker_model=worker_model)
    result["stages"]["routing"] = routing_preflight
    if routing_preflight.get("blocked"):
        result["status"] = "blocked_no_runner"
        result["elapsed_seconds"] = time.monotonic() - start
        _progress(
            emit_progress,
            "  ! Boss-loop routing is blocked; skipping dispatch until a matching runner is available.",
        )
        return result

    # Stage 4: Dispatch to boss loop
    _progress(emit_progress, f"\n[4/5] Dispatching to boss loop (--autonomy {autonomy_mode})...")
    dispatch_result = await _dispatch_to_boss_loop(
        issues,
        repo=repo_name,
        worker_model=worker_model,
        review_model=review_model,
        autonomy_mode=autonomy_mode,
    )
    result["stages"]["dispatch"] = dispatch_result
    _progress(emit_progress, f"  ✓ Boss loop started (PID: {dispatch_result.get('pid', '?')})")

    # Stage 5: Summary
    result["status"] = "dispatched"
    result["elapsed_seconds"] = time.monotonic() - start
    _progress(emit_progress, "\n[5/5] Pipeline complete!")
    _progress(emit_progress, f"  Issues: {', '.join(f'#{i}' for i in issues)}")
    _progress(
        emit_progress,
        f"  Monitor: tail -f {dispatch_result.get('log', '.aragora/overnight/code-improvements.log')}",
    )

    return result


async def _plan_reviewed_tasks(
    *,
    idea: str,
    spec: dict[str, Any],
    repo: str,
    risk: str,
    merge_class: str,
    autonomy_mode: str,
    max_tasks: int,
    worker_model: str,
    review_model: str,
) -> dict[str, Any]:
    """Generate founder handoffs, review them, and select queueable tasks."""
    from aragora.cli.commands.idea import (
        _compose_initiative_brief,
        _generate_founder_handoffs,
        _review_founder_handoffs,
    )

    brief = _compose_initiative_brief(
        idea=idea,
        spec=spec,
        priority="medium",
        track="1",
        risk=risk,
        merge_class=merge_class,
        autonomy_mode=autonomy_mode,
        worker_model=worker_model,
        review_model=review_model,
    )
    if brief["clarification_completeness_status"] != "decision_complete":
        return {
            "brief": brief,
            "handoffs": [],
            "review": {
                "status": "blocked",
                "summary": "Clarification incomplete; founder review skipped.",
                "findings": [],
                "followups": [],
            },
            "tasks": [],
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
    tasks = _queueable_tasks_from_review(
        brief=brief, spec=spec, handoffs=handoffs, review=review, repo=repo
    )
    return {
        "brief": brief,
        "handoffs": handoffs,
        "review": review,
        "tasks": tasks,
    }


async def _generate_spec(idea: str, *, skip_clarify: bool = False) -> dict[str, Any]:
    """Generate a structured specification from a vague idea."""
    try:
        from aragora.prompt_engine.conductor import PromptConductor, ConductorConfig

        config = ConductorConfig(
            skip_interrogation=skip_clarify,
            skip_research=True,  # Fast mode for build pipeline
        )
        conductor = PromptConductor(config=config)
        timeout_seconds = float(os.environ.get("ARAGORA_BUILD_SPEC_TIMEOUT_SECONDS", "180") or 180)
        result = await asyncio.wait_for(conductor.run(prompt=idea), timeout=timeout_seconds)
        specification = result.specification
        intent = result.intent
        questions = list(result.questions or [])
        unanswered_questions = [
            q.question.strip()
            for q in questions
            if getattr(q, "question", "").strip() and not getattr(q, "is_answered", False)
        ]
        success_criteria = [
            c.get("description", "") if isinstance(c, dict) else getattr(c, "description", str(c))
            for c in getattr(specification, "success_criteria", [])
        ]
        raw_sections = [
            getattr(specification, "title", "").strip(),
            getattr(specification, "problem_statement", "").strip(),
            getattr(specification, "proposed_solution", "").strip(),
        ]
        if success_criteria:
            raw_sections.append(
                "Success criteria:\n"
                + "\n".join(f"- {criterion}" for criterion in success_criteria)
            )
        raw_text = "\n\n".join(section for section in raw_sections if section)
        return {
            "title": getattr(specification, "title", idea[:80]),
            "sections": [
                {
                    "name": "problem_statement",
                    "content": getattr(specification, "problem_statement", "")[:200],
                },
                {
                    "name": "proposed_solution",
                    "content": getattr(specification, "proposed_solution", "")[:200],
                },
                {
                    "name": "success_criteria",
                    "content": "; ".join(item for item in success_criteria if item)[:200],
                },
            ],
            "confidence": getattr(specification, "confidence", 0.0),
            "raw": raw_text or idea,
            "user_goal": getattr(intent, "summary", "") or idea,
            "desired_outcome": getattr(specification, "proposed_solution", "")[:500],
            "success_criteria": [criterion for criterion in success_criteria if criterion],
            "clarification_status": (
                "needs_clarification" if unanswered_questions else "decision_complete"
            ),
            "open_questions": unanswered_questions,
        }
    except Exception as exc:
        logger.warning("Spec generation failed: %s", exc)
        return {
            "title": idea[:80],
            "sections": [],
            "confidence": 0.0,
            "raw": idea,
            "fallback": True,
        }


async def _decompose_tasks(spec: dict[str, Any], *, max_tasks: int = 5) -> list[dict[str, Any]]:
    """Break a specification into bounded implementation tasks."""
    try:
        from aragora.nomic.task_decomposer import TaskDecomposer

        decomposer = TaskDecomposer()
        analysis = decomposer.analyze(spec.get("raw", spec.get("title", "")))
        raw_tasks = analysis.subtasks if hasattr(analysis, "subtasks") else []
        raw_tasks = raw_tasks[:max_tasks]
        return [
            {
                "title": t.title if hasattr(t, "title") else str(t)[:80],
                "description": t.description if hasattr(t, "description") else str(t),
                "acceptance_criteria": _task_acceptance_criteria(t),
                "verification": _task_verification_command(t),
            }
            for t in (raw_tasks if isinstance(raw_tasks, list) else [raw_tasks])
        ]
    except Exception as exc:
        logger.warning("Task decomposition failed: %s, using spec as single task", exc)
        return [
            {
                "title": spec.get("title", "Implement idea"),
                "description": spec.get("raw", ""),
                "acceptance_criteria": ["Implementation complete", "Tests pass"],
                "verification": "python -m pytest tests/ -q -k 'not benchmark'",
            }
        ]


def _task_acceptance_criteria(task: Any) -> list[str]:
    acceptance = _string_list(getattr(task, "acceptance_criteria", []))
    if acceptance:
        return acceptance

    success_criteria = getattr(task, "success_criteria", {}) or {}
    if not isinstance(success_criteria, dict):
        return []

    acceptance = _string_list(success_criteria.get("acceptance_criteria"))
    if acceptance:
        return acceptance

    derived: list[str] = []
    for key, value in success_criteria.items():
        if key == "tests":
            continue
        values = _string_list(value)
        if values:
            derived.extend(f"{key}: {item}" for item in values)
            continue
        text = str(value).strip()
        if text:
            derived.append(f"{key}: {text}")
    return derived


def _task_verification_command(task: Any) -> str:
    verification = str(getattr(task, "verification_command", "") or "").strip()
    if verification:
        return verification

    verification_commands = _string_list(getattr(task, "verification_commands", []))
    if verification_commands:
        return "\n".join(verification_commands)

    success_criteria = getattr(task, "success_criteria", {}) or {}
    if isinstance(success_criteria, dict):
        tests = _string_list(success_criteria.get("tests"))
        if tests:
            return "\n".join(tests)

    return "pytest tests/ -q"


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        return [text for item in value if (text := str(item).strip())]
    return []


async def _create_issues(tasks: list[dict[str, Any]], *, repo: str) -> list[int]:
    """Create GitHub issues for each task."""
    from aragora.cli.commands.idea import _create_issue_with_optional_labels

    issue_numbers = []
    for task in tasks:
        labels = [
            "boss-ready",
            *[str(item).strip() for item in task.get("labels", []) if str(item).strip()],
        ]
        deduped_labels = list(dict.fromkeys(labels))
        body = f"""## Initiative Brief
- User goal: {task.get("user_goal", "")[:200]}
- Desired outcome: {task.get("desired_outcome", "")[:200]}
- Affected surfaces: {", ".join(task.get("affected_surfaces", [])) or "unknown"}
- File scope hints: {", ".join(task.get("file_scope_hints", [])) or "none"}
- Proof/evidence expected: {task.get("proof_expected", "")[:200]}
- Clarification completeness: {task.get("clarification_status", "draft")}
- Open questions: {", ".join(task.get("open_questions", [])) or "none"}

## Queue Metadata
- Risk: {task.get("risk", "medium")}
- Merge Class: {task.get("merge_class", "manual")}
- Autonomy Mode: {task.get("autonomy_mode", "full-auto")}
- Preferred Worker Agent: {task.get("preferred_worker_agent", "claude")}
- Preferred Reviewer Agent: {task.get("preferred_reviewer_agent", "codex")}
- Policy Notes: {", ".join(task.get("policy_reasons", [])) or "none"}

## Acceptance Criteria
{chr(10).join(f"- [ ] {c}" for c in task.get("acceptance_criteria", ["Implementation complete"]))}
- [ ] All tests pass (no new failures)
- [ ] Ruff clean on modified files

## Validation
```bash
{task.get("verification", "python -m pytest tests/ -q")}
```

## Description
{task.get("description", "")[:500]}

## Definition of Done
Implementation complete, tests pass, PR opened with evidence.
"""
        try:
            issue = await _create_issue_with_optional_labels(
                repo=repo,
                title=task["title"],
                body=body,
                requested_labels=deduped_labels,
            )
            num = int(issue["number"])
            issue_numbers.append(num)
            logger.info("Created issue #%d: %s", num, task["title"])
        except Exception as exc:
            logger.warning("Issue creation failed: %s", exc)

    return issue_numbers


async def _dispatch_to_boss_loop(
    issue_numbers: list[int],
    *,
    repo: str,
    worker_model: str,
    review_model: str,
    autonomy_mode: str,
) -> dict[str, Any]:
    """Launch the boss loop against the created issues."""
    import subprocess

    build_run_id = f"build-{int(time.time())}"
    log_path = Path(".aragora") / "builds" / f"{build_run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    owner_binding = _dispatch_owner_binding(repo=repo)
    cmd = [
        "bash",
        "-lc",
        (
            f"cd {sh_quote(str(Path.cwd()))} && "
            f"export ARAGORA_USER_ID={sh_quote(owner_binding['user_id'])} && "
            f"export ARAGORA_WORKSPACE_ID={sh_quote(owner_binding['workspace_id'])} && "
            "exec python3 -u -m aragora.cli.main swarm boss-loop "
            f"--boss-repo {sh_quote(repo)} "
            f"--boss-issue-list {sh_quote(','.join(str(num) for num in issue_numbers))} "
            f"--max-ticks {len(issue_numbers) * 2} "
            "--interval 30 "
            "--max-consecutive-failures 5 "
            f"--autonomy {sh_quote(autonomy_mode)} "
            f"--worker-model {sh_quote(worker_model)} "
            f"--review-model {sh_quote(review_model)} "
            "--max-hours 10"
        ),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=open(log_path, "w"),  # noqa: SIM115
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {
        "pid": proc.pid,
        "issues": issue_numbers,
        "log": str(log_path),
        "owner_binding": owner_binding,
    }


def _dispatch_owner_binding(*, repo: str) -> dict[str, str]:
    repo_name = str(repo or "").strip()
    workspace_default = repo_name.rsplit("/", 1)[-1].strip() or "aragora"
    user_id = (
        str(os.environ.get("ARAGORA_USER_ID") or os.environ.get("ARAGORA_ACTOR_ID") or "").strip()
        or str(os.environ.get("USER") or "").strip()
        or "armand"
    )
    workspace_id = (
        str(
            os.environ.get("ARAGORA_WORKSPACE_ID") or os.environ.get("ARAGORA_WORKSPACE") or ""
        ).strip()
        or workspace_default
    )
    return {
        "user_id": user_id,
        "workspace_id": workspace_id,
    }


def _preflight_boss_routing(*, repo: str, worker_model: str) -> dict[str, Any]:
    from aragora.swarm.runner_registry import (
        LocalRunnerRegistry,
        authorization_context_from_env,
        prioritized_probe_candidates,
        probe_runner_execution,
        refresh_discovered_runners,
    )

    owner_binding = _dispatch_owner_binding(repo=repo)
    env = {
        **os.environ,
        "ARAGORA_USER_ID": owner_binding["user_id"],
        "ARAGORA_WORKSPACE_ID": owner_binding["workspace_id"],
    }
    owner_context = authorization_context_from_env(env)
    registry = LocalRunnerRegistry()
    discovered = refresh_discovered_runners(
        worker_model,
        registry=registry,
        owner_context=owner_context,
        env=env,
        repo_root=Path.cwd(),
    )
    routing = registry.resolve_boss_routing(
        owner_context=owner_context,
        requested_runner_type=worker_model,
    ).to_dict()
    probe_summary = {
        "auto_probe_triggered": False,
        "attempted": 0,
        "passed": 0,
        "failed": 0,
        "verified_target": 0,
        "results": [],
    }
    if worker_model == "claude":
        try:
            verified_target = max(
                1,
                int(str(os.environ.get("ARAGORA_BUILD_VERIFIED_RUNNER_TARGET", "3") or "3")),
            )
        except ValueError:
            verified_target = 3
        try:
            probe_limit = max(
                1,
                int(str(os.environ.get("ARAGORA_BUILD_RUNNER_PROBE_LIMIT", "2") or "2")),
            )
        except ValueError:
            probe_limit = 2
        selected = [item for item in routing.get("selected_runners", []) if isinstance(item, dict)]
        selected_verified = len(
            [item for item in selected if str(item.get("probe_status", "")).strip() == "passed"]
        )
        probe_summary["verified_target"] = verified_target
        if selected_verified < verified_target:
            candidates = prioritized_probe_candidates(
                registry=registry,
                runner_type=worker_model,
                discovered_inspections=discovered,
                owner_context=owner_context,
                selected_runners=selected,
            )
            for inspection in candidates[:probe_limit]:
                probe = probe_runner_execution(
                    inspection,
                    repo_root=Path.cwd(),
                )
                registry.record_probe(
                    inspection,
                    probe,
                    owner_context=owner_context,
                )
                probe_summary["results"].append(probe.to_dict())
                probe_summary["attempted"] += 1
                if probe.status == "passed":
                    probe_summary["passed"] += 1
                elif probe.status == "failed":
                    probe_summary["failed"] += 1
            if probe_summary["attempted"]:
                probe_summary["auto_probe_triggered"] = True
                routing = registry.resolve_boss_routing(
                    owner_context=owner_context,
                    requested_runner_type=worker_model,
                ).to_dict()
    return {
        "owner_binding": owner_binding,
        "blocked": bool(routing.get("blocked_reason")),
        "routing": routing,
        "probe": probe_summary,
    }


def _queueable_tasks_from_review(
    *,
    brief: dict[str, Any],
    spec: dict[str, Any],
    handoffs: list[dict[str, Any]],
    review: dict[str, Any],
    repo: str,
) -> list[dict[str, Any]]:
    """Select the queue tasks that should proceed after founder review."""
    findings = [item for item in review.get("findings", []) if isinstance(item, dict)]
    blocked_handoff_ids = {
        str(handoff_id).strip()
        for finding in findings
        if str(finding.get("severity", "")).strip().lower() in {"high", "medium"}
        for handoff_id in finding.get("handoff_ids", [])
        if str(handoff_id).strip()
    }
    followups = [item for item in review.get("followups", []) if isinstance(item, dict)]
    approved_handoffs = [
        handoff
        for handoff in handoffs
        if str(handoff.get("handoff_id", "")).strip() not in blocked_handoff_ids
    ]

    queueable: list[dict[str, Any]] = []
    for item in [*followups, *approved_handoffs]:
        queueable.append(_handoff_to_build_task(item, brief=brief, spec=spec, repo=repo))
    return queueable


def _handoff_to_build_task(
    handoff: dict[str, Any],
    *,
    brief: dict[str, Any],
    spec: dict[str, Any],
    repo: str,
) -> dict[str, Any]:
    """Normalize a founder handoff or review follow-up into a build queue task."""
    validation = [str(item).strip() for item in handoff.get("validation", []) if str(item).strip()]
    file_scope = [str(item).strip() for item in handoff.get("file_scope", []) if str(item).strip()]
    return {
        "title": str(handoff.get("task_title", "Untitled task")).strip(),
        "description": str(handoff.get("description", "")).strip(),
        "acceptance_criteria": [
            str(item).strip()
            for item in handoff.get("acceptance_criteria", [])
            if str(item).strip()
        ],
        "verification": "\n".join(validation) if validation else "python3 -m pytest tests/ -q",
        "user_goal": str(brief.get("user_goal", "")).strip()
        or str(spec.get("user_goal", "")).strip(),
        "desired_outcome": str(brief.get("desired_business_outcome", "")).strip()
        or str(spec.get("desired_outcome", "")).strip(),
        "affected_surfaces": list(brief.get("affected_surfaces", []) or []),
        "file_scope_hints": file_scope,
        "proof_expected": str(brief.get("proof_evidence_expected", "")).strip(),
        "clarification_status": str(brief.get("clarification_completeness_status", "")).strip()
        or "draft",
        "open_questions": list(brief.get("open_questions", []) or []),
        "risk": str(handoff.get("risk", "medium")).strip() or "medium",
        "merge_class": str(handoff.get("merge_class", "manual")).strip() or "manual",
        "autonomy_mode": str(handoff.get("autonomy_mode", "checkpoint")).strip() or "checkpoint",
        "policy_reasons": list(handoff.get("policy_reasons", []) or []),
        "labels": list(handoff.get("labels", []) or []),
        "preferred_worker_agent": str(handoff.get("preferred_worker_agent", "claude")).strip()
        or "claude",
        "preferred_reviewer_agent": str(handoff.get("preferred_reviewer_agent", "codex")).strip()
        or "codex",
        "repo": repo,
    }


def _annotate_tasks(
    tasks: list[dict[str, Any]],
    *,
    spec: dict[str, Any],
    idea: str,
    repo: str,
    risk: str,
    merge_class: str,
    autonomy_mode: str,
    worker_model: str,
    review_model: str,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for task in tasks:
        item = dict(task)
        file_scope_hints = SwarmSpec.infer_file_scope_hints(
            "\n".join(
                [
                    str(item.get("title", "")),
                    str(item.get("description", "")),
                    str(item.get("verification", "")),
                ]
            )
        )
        policy = apply_delivery_policy(
            file_scope=file_scope_hints,
            requested_risk=risk,
            requested_merge_class=merge_class,
            requested_autonomy_mode=autonomy_mode,
        )
        item.setdefault("user_goal", str(spec.get("user_goal", "")).strip() or idea[:500])
        item.setdefault(
            "desired_outcome",
            str(spec.get("desired_outcome", "")).strip()
            or item.get("description", "")[:300]
            or item.get("title", ""),
        )
        item.setdefault("affected_surfaces", _infer_surfaces(item))
        item.setdefault("file_scope_hints", file_scope_hints)
        item.setdefault("proof_expected", item.get("verification", ""))
        item.setdefault(
            "clarification_status",
            str(spec.get("clarification_status", "")).strip() or "draft",
        )
        item.setdefault("open_questions", list(spec.get("open_questions", []) or []))
        item.setdefault("risk", policy["effective_risk"])
        item.setdefault("merge_class", policy["effective_merge_class"])
        item.setdefault("autonomy_mode", policy["effective_autonomy_mode"])
        item.setdefault("policy_reasons", list(policy.get("policy_reasons", []) or []))
        item.setdefault("preferred_worker_agent", worker_model)
        item.setdefault("preferred_reviewer_agent", review_model)
        item.setdefault("repo", repo)
        annotated.append(item)
    return annotated


def _infer_surfaces(task: dict[str, Any]) -> list[str]:
    text = "\n".join(
        [
            str(task.get("title", "")),
            str(task.get("description", "")),
            str(task.get("verification", "")),
        ]
    ).lower()
    surfaces: list[str] = []
    if "front" in text or "live" in text or "ui" in text:
        surfaces.append("frontend")
    if "api" in text or "handler" in text or "route" in text or "server" in text:
        surfaces.append("server")
    if "receipt" in text or "storage" in text or "db" in text:
        surfaces.append("storage")
    if "test" in text or "pytest" in text or "jest" in text:
        surfaces.append("testing")
    return surfaces or ["unknown"]


def _default_repo() -> str:
    return (
        str(os.environ.get("ARAGORA_BUILD_REPO", "synaptent/aragora")).strip()
        or "synaptent/aragora"
    )


def _progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message)


def sh_quote(value: str) -> str:
    import shlex

    return shlex.quote(str(value))


async def _close_shared_agent_connector() -> None:
    """Best-effort cleanup for shared API connectors used by prompt_engine agents."""
    try:
        from aragora.agents.api_agents.common import close_shared_connector

        await close_shared_connector()
    except Exception as exc:
        logger.debug("Shared API connector cleanup failed: %s", exc)


def _print_result(result: dict[str, Any]) -> None:
    """Print a human-readable summary."""
    print(f"\n{'=' * 60}")
    print(f"Build Pipeline: {result['status']}")
    print(f"{'=' * 60}")
    if result.get("elapsed_seconds"):
        print(f"Time: {result['elapsed_seconds']:.1f}s")
    if "stages" in result:
        if "spec" in result["stages"]:
            print(f"Spec: {result['stages']['spec'].get('title', '?')}")
        if "tasks" in result["stages"]:
            print(f"Tasks: {len(result['stages']['tasks'])}")
        if "issues" in result["stages"]:
            issues = result["stages"]["issues"]
            print(f"Issues: {', '.join(f'#{n}' for n in issues)}")
        if "dispatch" in result["stages"]:
            print(f"Boss loop PID: {result['stages']['dispatch'].get('pid', '?')}")
    print(f"{'=' * 60}")
