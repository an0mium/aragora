"""Intake bridge — turn a seeded mission-intake feature into branch-backed work.

``aragora mission seed --goal "..."`` creates a single intake feature with no
``metadata.branch``. :class:`~aragora.missions.dispatch.BossLoopDispatch`
(correctly) refuses to drive such a feature — the merge gate needs a live
branch — so before this bridge a freshly seeded mission parked forever (#8758).

:class:`IntakeBridgeDispatch` wraps any inner ``Dispatch``. When the picked
feature is intake-shaped it decomposes the goal via the existing nomic
:class:`~aragora.nomic.task_decomposer.TaskDecomposer` (heuristic by default —
its LLM paths are key-gated and fall through silently, so no API keys are
required) into 1+ child Features, each carrying a deterministic
``metadata.branch`` so it is claimable by the existing dispatch/lease
machinery. The children ride back through the orchestrator's own
propose/accept handoff triage (``follow_ups`` + ``accept_follow_ups=True``),
so no new state-mutation path is introduced: triage inserts the children,
completes the intake (non-terminal), and folds the provenance note into
``notes``. Child ids derive from subtask titles (not list position), so a
crash-retried decomposition converges on the same ids and triage's duplicate
check makes the re-tick a no-op.

Failure to decompose parks the intake with a diagnostic reason — it never
crashes the tick loop. The bridge is default-ON for auto-drain and only ever
activates on intake-shaped features (i.e. freshly seeded missions); the
``ARAGORA_DISABLE_MISSION_INTAKE_BRIDGE=1`` kill-switch restores the previous
park-on-intake behavior.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .orchestrator import Dispatch, Handoff
from .state import Feature

if TYPE_CHECKING:
    from aragora.nomic.task_decomposer import SubTask

logger = logging.getLogger(__name__)

INTAKE_FEATURE_ID = "mission-intake"
INTAKE_KIND = "intake"
DISABLE_ENV = "ARAGORA_DISABLE_MISSION_INTAKE_BRIDGE"

# (goal, path allowlist hints) -> subtasks. Injectable for tests/alternate planners.
Decompose = Callable[[str, list[str]], "list[SubTask]"]


def is_intake_feature(feature: Feature) -> bool:
    """True iff ``feature`` is seed-shaped work awaiting decomposition.

    Intake = no live ``metadata.branch`` AND explicitly marked as intake
    (the seeded ``mission-intake`` id, or a ``metadata.kind == "intake"``
    marker). A branch-backed feature is never intake, whatever its id.
    """
    branch = feature.metadata.get("branch")
    if isinstance(branch, str) and branch.strip():
        return False
    if feature.metadata.get("kind") == INTAKE_KIND:
        return True
    return feature.id == INTAKE_FEATURE_ID


def intake_bridge_enabled() -> bool:
    """Default ON; ``ARAGORA_DISABLE_MISSION_INTAKE_BRIDGE=1`` opts out."""
    return os.environ.get(DISABLE_ENV, "").strip().lower() not in {"1", "true", "yes"}


class IntakeBridgeDispatch:
    """Dispatch wrapper: decompose intake features, delegate everything else.

    Composes existing primitives only: ``TaskDecomposer`` produces subtasks,
    and the orchestrator's handoff triage (``accept_follow_ups``) inserts the
    children and completes the intake. Provenance (which decomposer, when,
    child ids) travels on the children's metadata and the intake's notes.
    """

    def __init__(
        self,
        inner: Dispatch,
        *,
        decompose: Decompose | None = None,
        max_children: int = 5,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.inner = inner
        self.max_children = max_children
        self.clock = clock
        if decompose is None:
            self._decompose: Decompose = _task_decomposer_decompose
            self.decomposer_name = "nomic.TaskDecomposer.analyze"
        else:
            self._decompose = decompose
            self.decomposer_name = getattr(decompose, "__qualname__", repr(decompose))

    def __call__(self, feature: Feature) -> Handoff:
        if not is_intake_feature(feature):
            return self.inner(feature)

        goal = feature.description.strip()
        if not goal:
            return Handoff(
                success=False,
                terminal=True,
                blocked_reason="intake feature has no goal/description to decompose",
                discovered=[f"intake feature {feature.id} arrived with a blank goal"],
            )

        paths = _path_hints(feature)
        try:
            subtasks = list(self._decompose(goal, paths))[: self.max_children]
        except Exception as exc:  # noqa: BLE001 - decomposer is an external boundary; park, never crash the tick loop
            logger.exception("intake decomposition failed for feature %s", feature.id)
            return Handoff(
                success=False,
                terminal=True,
                blocked_reason=f"intake decomposition failed: {exc}",
                discovered=[f"intake feature {feature.id} parked: {self.decomposer_name} raised"],
            )

        children = self._child_features(feature, subtasks)
        note = (
            f"intake decomposed via {self.decomposer_name} into "
            f"{len(children)} feature(s): {', '.join(c.id for c in children)}"
        )
        logger.info("feature %s: %s", feature.id, note)
        return Handoff(
            success=True,
            follow_ups=children,
            accept_follow_ups=True,
            discovered=[note],
        )

    # ---- child construction ---------------------------------------------------

    def _child_features(self, intake: Feature, subtasks: list[SubTask]) -> list[Feature]:
        """Deterministic branch-backed children (mirror the goal when empty).

        An empty decomposition means the goal is a bounded, single-unit change —
        mirror it into one child rather than parking a perfectly workable goal.
        """
        if not subtasks:
            return [self._mirrored_child(intake)]

        decomposed_at = self.clock()
        # First pass: stable ids from titles (not positions), so a crash-retried
        # decomposition converges on the same ids even if subtask order shifts.
        child_ids: dict[str, str] = {}
        used: set[str] = set()
        for idx, subtask in enumerate(subtasks, start=1):
            base = f"{intake.id}-{_slug(subtask.title or subtask.id or str(idx))}"
            child_id = base if base not in used else f"{base}-{idx}"
            used.add(child_id)
            child_ids[subtask.id] = child_id

        children: list[Feature] = []
        for subtask in subtasks:
            child_id = child_ids[subtask.id]
            preconditions = [
                f"feature:{child_ids[dep]}" for dep in subtask.dependencies if dep in child_ids
            ]
            metadata = self._child_metadata(intake, child_id, decomposed_at)
            if subtask.file_scope:
                metadata["paths"] = sorted(set(subtask.file_scope))
            description = subtask.description.strip() or subtask.title.strip() or intake.description
            children.append(
                Feature(
                    id=child_id,
                    description=description,
                    milestone=intake.milestone,
                    skill=intake.skill,
                    preconditions=preconditions,
                    metadata=metadata,
                )
            )
        return children

    def _mirrored_child(self, intake: Feature) -> Feature:
        child_id = f"{intake.id}-{_slug(intake.description)}"
        return Feature(
            id=child_id,
            description=intake.description.strip(),
            milestone=intake.milestone,
            skill=intake.skill,
            metadata=self._child_metadata(intake, child_id, self.clock()),
        )

    def _child_metadata(
        self, intake: Feature, child_id: str, decomposed_at: float
    ) -> dict[str, Any]:
        metadata = {k: v for k, v in intake.metadata.items() if k != "kind"}
        metadata.update(
            branch=f"mission/{child_id}",
            intake_parent=intake.id,
            decomposer=self.decomposer_name,
            decomposed_at=decomposed_at,
        )
        return metadata


def _task_decomposer_decompose(goal: str, paths: list[str]) -> list[SubTask]:
    """Default planner: the existing nomic TaskDecomposer.

    Heuristic by default — its LLM extraction paths are key-gated and return
    empty when no provider is configured, so this never *requires* API keys.
    Imported lazily so the missions spine stays importable without the nomic
    extras loaded.
    """
    from aragora.nomic.task_decomposer import TaskDecomposer

    result = TaskDecomposer().analyze(goal, file_scope_hints=paths or None)
    return list(result.subtasks)


def _path_hints(feature: Feature) -> list[str]:
    raw = feature.metadata.get("paths")
    if not isinstance(raw, list):
        return []
    return [str(p).strip() for p in raw if str(p).strip()]


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:48].rstrip("-") or "work"
