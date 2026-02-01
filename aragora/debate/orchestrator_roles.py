"""Cognitive role rotation helpers for Arena debates.

Extracted from orchestrator.py to reduce its size. These functions handle
RolesManager initialization, initial role assignment, and stance management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aragora.debate.roles_manager import RolesManager

if TYPE_CHECKING:
    from aragora.debate.orchestrator import Arena


def init_roles_and_stances(arena: Arena) -> None:
    """Initialize cognitive role rotation and agent stances.

    Creates the RolesManager, assigns initial cognitive roles to agents,
    sets debate stances for round 0, and applies agreement intensity.

    Args:
        arena: Arena instance to initialize.
    """
    arena.roles_manager = RolesManager(
        agents=arena.agents,
        protocol=arena.protocol,
        prompt_builder=arena.prompt_builder if hasattr(arena, "prompt_builder") else None,
        calibration_tracker=(
            arena.calibration_tracker if hasattr(arena, "calibration_tracker") else None
        ),
        persona_manager=arena.persona_manager if hasattr(arena, "persona_manager") else None,
    )
    arena.role_rotator = arena.roles_manager.role_rotator
    arena.role_matcher = arena.roles_manager.role_matcher
    arena.current_role_assignments = arena.roles_manager.current_role_assignments
    arena.roles_manager.assign_initial_roles()
    arena.roles_manager.assign_stances(round_num=0)
    arena.roles_manager.apply_agreement_intensity()


__all__ = ["init_roles_and_stances"]
