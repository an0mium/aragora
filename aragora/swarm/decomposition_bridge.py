"""Bridge vague boss issues into bounded child candidates.

This module intentionally reuses existing swarm/nomic primitives instead of
introducing a parallel decomposition stack:

1. Parent issue text is normalized into a ``SwarmSpec``.
2. ``TaskDecomposer`` extracts subtasks when possible.
3. Broad subtasks fall back to ``build_micro_work_orders``.
4. Each child is converted into a ``BossIssueCandidate``.
5. Each child is gated through ``TaskSanitizer`` and boss-validation.

The output is a bounded list of child candidates suitable for queueing as
worker-friendly follow-up issues.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from aragora.nomic.task_decomposer import SubTask, TaskDecomposer
from aragora.swarm.boss_validation import (
    assess_issue_body_sanitation,
    extract_declared_new_file_paths,
    extract_issue_validation_contract,
    extract_pre_dispatch_validation_commands,
    sanitize_issue_body_for_dispatch,
)
from aragora.swarm.issue_scanner import BossIssueCandidate, infer_issue_category_from_title
from aragora.swarm.micro_decomposer import build_micro_work_orders
from aragora.swarm.prompt_refiner import refine_worker_prompt
from aragora.swarm.spec import SwarmSpec
from aragora.swarm.task_sanitizer import SanitizationOutcome, TaskSanitizer

logger = logging.getLogger(__name__)

_MAX_SCOPE_PATHS = 5


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _normalize_scope_paths(values: list[str]) -> list[str]:
    return _ordered_unique(
        [
            normalized
            for normalized in (SwarmSpec.sanitize_file_scope_entry(value) for value in values)
            if normalized
        ]
    )


def _partition_paths(
    repo_root: Path,
    paths: list[str],
    *,
    declared_new: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    declared = set(_normalize_scope_paths(list(declared_new or [])))
    file_scope: list[str] = []
    new_files: list[str] = []
    for path in _normalize_scope_paths(paths):
        if path in declared or not (repo_root / path).exists():
            new_files.append(path)
        else:
            file_scope.append(path)
    return file_scope[:_MAX_SCOPE_PATHS], new_files[:_MAX_SCOPE_PATHS]


def _normalized_complexity(value: str) -> str:
    lowered = str(value or "").strip().lower()
    if lowered in {"low", "small", "trivial"}:
        return "small"
    return "medium"


def _success_criteria_lines(success_criteria: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key, value in dict(success_criteria or {}).items():
        left = str(key or "").strip()
        right = str(value or "").strip()
        if left and right:
            lines.append(f"{left} {right}")
    return _ordered_unique(lines)


class DecompositionBridge:
    """Transform a vague parent issue into bounded child candidates."""

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root).resolve()
        self._task_decomposer = TaskDecomposer()
        self._task_sanitizer = TaskSanitizer(repo_root=self.repo_root)

    async def decompose_issue(
        self,
        title: str,
        body: str,
        *,
        max_children: int = 5,
    ) -> list[BossIssueCandidate]:
        if max_children <= 0:
            return []

        spec = self._build_parent_spec(title, body)
        parent_category = infer_issue_category_from_title(title) or "decomposed_issue"
        actionable = spec.refined_goal or spec.raw_goal or str(title or "").strip()

        decomposition = self._task_decomposer.analyze(
            actionable,
            file_scope_hints=list(spec.file_scope_hints),
            acceptance_criteria=list(spec.acceptance_criteria),
            constraints=list(spec.constraints),
        )

        candidates: list[BossIssueCandidate] = []
        if decomposition.subtasks:
            for subtask in decomposition.subtasks:
                candidates.extend(
                    await self._candidates_for_subtask(
                        subtask,
                        parent_title=title,
                        parent_category=parent_category,
                        parent_spec=spec,
                    )
                )
        elif spec.file_scope_hints:
            candidates.extend(
                await self._candidates_from_micro_work_orders(
                    goal=actionable,
                    parent_title=title,
                    parent_category=parent_category,
                    file_scope_hints=list(spec.file_scope_hints),
                    acceptance_criteria=list(spec.acceptance_criteria),
                    constraints=list(spec.constraints),
                )
            )

        accepted: list[BossIssueCandidate] = []
        seen_paths: set[str] = set()
        for candidate in candidates:
            if len(accepted) >= max_children:
                break
            candidate_paths = set(candidate.file_scope + candidate.new_files)
            if not candidate_paths:
                continue
            if candidate_paths & seen_paths:
                continue
            if candidate.estimated_complexity not in {"small", "medium"}:
                continue
            gated = self._gate_candidate(candidate)
            if gated is None:
                continue
            accepted.append(gated)
            seen_paths.update(gated.file_scope)
            seen_paths.update(gated.new_files)

        return accepted[:max_children]

    def decompose_issue_sync(
        self,
        title: str,
        body: str,
        **kwargs: Any,
    ) -> list[BossIssueCandidate]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.decompose_issue(title, body, **kwargs))
        raise RuntimeError("decompose_issue_sync() cannot run inside an active event loop")

    def _build_parent_spec(self, title: str, body: str) -> SwarmSpec:
        issue_body = str(body or "").strip()
        raw_goal = "\n\n".join(
            part for part in [str(title or "").strip(), issue_body] if part
        ).strip()
        spec = SwarmSpec.from_direct_goal(
            raw_goal,
            budget_limit_usd=None,
            requires_approval=True,
            user_expertise="developer",
            use_llm=False,
        )
        dispatch_body = sanitize_issue_body_for_dispatch(issue_body)
        if dispatch_body:
            spec.refined_goal = dispatch_body
        validation_contract = extract_issue_validation_contract(issue_body)
        if validation_contract:
            spec.acceptance_criteria = _ordered_unique(
                [*spec.acceptance_criteria, *validation_contract]
            )
        spec.constraints = _ordered_unique(
            [*spec.constraints, *SwarmSpec.infer_constraints([title, issue_body])]
        )
        spec.file_scope_hints = _ordered_unique(
            [
                *spec.file_scope_hints,
                *SwarmSpec.infer_file_scope_hints(issue_body),
                *extract_declared_new_file_paths(issue_body),
            ]
        )
        return spec

    async def _candidates_for_subtask(
        self,
        subtask: SubTask,
        *,
        parent_title: str,
        parent_category: str,
        parent_spec: SwarmSpec,
    ) -> list[BossIssueCandidate]:
        scope = _normalize_scope_paths(list(subtask.file_scope))
        if self._needs_micro_decomposition(subtask):
            return await self._candidates_from_micro_work_orders(
                goal=subtask.description
                or subtask.title
                or parent_spec.refined_goal
                or parent_title,
                parent_title=subtask.title or parent_title,
                parent_category=parent_category,
                file_scope_hints=scope or list(parent_spec.file_scope_hints),
                acceptance_criteria=(
                    _success_criteria_lines(subtask.success_criteria)
                    or list(parent_spec.acceptance_criteria)
                ),
                constraints=list(parent_spec.constraints),
            )

        return [
            await self._candidate_from_parts(
                title=subtask.title or parent_title,
                category=parent_category,
                description=subtask.description or parent_spec.refined_goal or parent_title,
                scope_paths=scope or list(parent_spec.file_scope_hints),
                acceptance_criteria=(
                    _success_criteria_lines(subtask.success_criteria)
                    or list(parent_spec.acceptance_criteria)
                ),
                estimated_complexity=_normalized_complexity(subtask.estimated_complexity),
            )
        ]

    async def _candidates_from_micro_work_orders(
        self,
        *,
        goal: str,
        parent_title: str,
        parent_category: str,
        file_scope_hints: list[str],
        acceptance_criteria: list[str],
        constraints: list[str],
    ) -> list[BossIssueCandidate]:
        orders = build_micro_work_orders(
            goal=goal,
            file_scope_hints=file_scope_hints,
            acceptance_criteria=acceptance_criteria,
            constraints=constraints,
            repo_root=self.repo_root,
        )
        candidates: list[BossIssueCandidate] = []
        for order in orders:
            title = str(order.get("title", "")).strip()
            if "validation" in title.lower():
                continue
            scope_paths = [
                str(path).strip() for path in order.get("file_scope", []) if str(path).strip()
            ]
            if not scope_paths:
                continue
            candidates.append(
                await self._candidate_from_parts(
                    title=title or parent_title,
                    category=parent_category,
                    description=str(order.get("description", "")).strip() or goal,
                    scope_paths=scope_paths,
                    acceptance_criteria=acceptance_criteria,
                    estimated_complexity=_normalized_complexity(
                        str(order.get("estimated_complexity", "low"))
                    ),
                )
            )
        return candidates

    async def _candidate_from_parts(
        self,
        *,
        title: str,
        category: str,
        description: str,
        scope_paths: list[str],
        acceptance_criteria: list[str],
        estimated_complexity: str,
    ) -> BossIssueCandidate:
        normalized_scope = _normalize_scope_paths(scope_paths)
        file_scope, new_files = _partition_paths(self.repo_root, normalized_scope)

        refinement = await refine_worker_prompt(
            title,
            description,
            repo_path=self.repo_root,
        )
        refined_description = (
            str(refinement.get("refined_prompt", "")).strip() or str(description or "").strip()
        )
        validation_command = self._choose_validation_command(
            description=refined_description,
            file_scope=file_scope,
            new_files=new_files,
            acceptance_criteria=acceptance_criteria,
            refinement=refinement,
        )
        acceptance = _ordered_unique(
            [
                *acceptance_criteria,
                *(extract_issue_validation_contract(refined_description) or []),
                *([f"`{validation_command}` passes"] if validation_command else []),
            ]
        )
        return BossIssueCandidate(
            category=category,
            title=title.strip() or "Decomposed child issue",
            description=refined_description,
            file_scope=file_scope,
            new_files=new_files,
            validation_command=validation_command,
            acceptance_criteria=acceptance,
            estimated_complexity=estimated_complexity,
        )

    def _needs_micro_decomposition(self, subtask: SubTask) -> bool:
        scope = _normalize_scope_paths(list(subtask.file_scope))
        if not scope:
            return False
        if len(scope) > 1:
            return True
        if any(path.endswith("/") for path in scope):
            return True
        lowered = str(subtask.estimated_complexity or "").strip().lower()
        return lowered not in {"low", "small", "trivial"}

    def _choose_validation_command(
        self,
        *,
        description: str,
        file_scope: list[str],
        new_files: list[str],
        acceptance_criteria: list[str],
        refinement: dict[str, Any],
    ) -> str:
        command_sources = [
            *extract_pre_dispatch_validation_commands(description),
            *extract_pre_dispatch_validation_commands("\n".join(acceptance_criteria)),
        ]
        if command_sources:
            return command_sources[0]

        test_patterns = [
            str(item).strip() for item in refinement.get("test_patterns", []) if str(item).strip()
        ]
        if test_patterns:
            return f"python3 -m pytest -q {test_patterns[0]}"

        test_targets = [
            path
            for path in [*file_scope, *new_files]
            if path.startswith("tests/") and path.endswith(".py")
        ]
        if test_targets:
            return f"python3 -m pytest -q {test_targets[0]}"

        lint_targets = [*file_scope, *new_files]
        if lint_targets:
            return "python3 -m ruff check " + " ".join(lint_targets[:_MAX_SCOPE_PATHS])
        return "python3 -m ruff check aragora/"

    def _render_candidate_body(self, candidate: BossIssueCandidate) -> str:
        parts = [f"## Task\n\n{candidate.description.strip()}"]
        scope_lines = [f"- `{path}`" for path in candidate.file_scope]
        scope_lines.extend(f"- `{path}` (create)" for path in candidate.new_files)
        if scope_lines:
            parts.append("### File Scope\n" + "\n".join(scope_lines))
        if candidate.validation_command:
            parts.append(f"### Validation\n- {candidate.validation_command}")
        if candidate.acceptance_criteria:
            parts.append(
                "### Acceptance Criteria\n"
                + "\n".join(f"- {item}" for item in candidate.acceptance_criteria)
            )
        parts.append(
            "### Constraints\n"
            f"- Estimated complexity: {candidate.estimated_complexity}\n"
            "- Child issue emitted by decomposition bridge\n"
            "- Keep changes focused to the declared file scope"
        )
        return "\n\n".join(parts).strip()

    def _extract_body_text(self, title: str, composed_text: str) -> str:
        text = str(composed_text or "").strip()
        stripped_title = str(title or "").strip()
        if stripped_title and text.startswith(stripped_title):
            remainder = text[len(stripped_title) :].lstrip()
            if remainder.startswith("\n"):
                remainder = remainder.lstrip()
            return remainder.strip()
        return text

    def _gate_candidate(self, candidate: BossIssueCandidate) -> BossIssueCandidate | None:
        body = self._render_candidate_body(candidate)
        sanitation = self._task_sanitizer.sanitize(candidate.title, body)
        if sanitation.outcome in {
            SanitizationOutcome.DROPPED,
            SanitizationOutcome.QUARANTINED,
        }:
            return None

        effective_text = sanitation.sanitized_text or f"{candidate.title}\n\n{body}"
        effective_body = self._extract_body_text(candidate.title, effective_text)

        commands = extract_pre_dispatch_validation_commands(effective_body)
        if commands:
            candidate.validation_command = commands[0]

        paths = _normalize_scope_paths(
            [
                *SwarmSpec.infer_file_scope_hints(effective_body),
                *extract_declared_new_file_paths(effective_body),
            ]
        )
        if paths:
            file_scope, new_files = _partition_paths(
                self.repo_root,
                paths,
                declared_new=extract_declared_new_file_paths(effective_body),
            )
            if file_scope or new_files:
                candidate.file_scope = file_scope
                candidate.new_files = new_files

        criteria = extract_issue_validation_contract(effective_body)
        if criteria:
            candidate.acceptance_criteria = _ordered_unique(criteria)

        if len(candidate.file_scope) + len(candidate.new_files) > _MAX_SCOPE_PATHS:
            return None

        sanitation_ok, _ = assess_issue_body_sanitation(effective_body)
        if not sanitation_ok:
            return None
        if not candidate.validation_command:
            return None
        if not (candidate.file_scope or candidate.new_files):
            return None
        return candidate


__all__ = ["DecompositionBridge"]
