"""
Roles Manager for debate agent configuration.

Handles role assignment, stance assignment, and agreement intensity
for agents participating in debates. Extracted from orchestrator.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from aragora.agents.calibration import CalibrationTracker
    from aragora.agents.personas import PersonaManager
    from aragora.core import Agent
    from aragora.debate.prompt_builder import PromptBuilder
    from aragora.debate.protocol import DebateProtocol
    from aragora.debate.role_matcher import RoleMatcher
    from aragora.debate.roles import (
        RoleAssignment,
        RoleRotator,
    )

logger = logging.getLogger(__name__)


class RolesManager:
    """
    Manages agent roles, stances, and agreement intensity.

    Extracted from Arena to separate role management concerns from
    debate orchestration.

    Usage:
        manager = RolesManager(agents, protocol, prompt_builder)
        manager.assign_initial_roles()
        manager.assign_stances(round_num=0)
        manager.apply_agreement_intensity()
    """

    def __init__(
        self,
        agents: list["Agent"],
        protocol: "DebateProtocol",
        prompt_builder: Optional["PromptBuilder"] = None,
        calibration_tracker: Optional["CalibrationTracker"] = None,
        persona_manager: Optional["PersonaManager"] = None,
    ):
        """
        Initialize roles manager.

        Args:
            agents: List of agents to manage
            protocol: Debate protocol with role configuration
            prompt_builder: Optional prompt builder for stance guidance
            calibration_tracker: Optional calibration tracker for role matching
            persona_manager: Optional persona manager for role matching
        """
        self.agents = agents
        self.protocol = protocol
        self.prompt_builder = prompt_builder
        self.calibration_tracker = calibration_tracker
        self.persona_manager = persona_manager

        # Cognitive role rotation (Heavy3-inspired)
        self.role_rotator: Optional["RoleRotator"] = None
        self.role_matcher: Optional["RoleMatcher"] = None
        self.current_role_assignments: dict[str, "RoleAssignment"] = {}

        self._init_role_systems()

    def _init_role_systems(self) -> None:
        """Initialize role rotation or matching systems based on protocol."""
        from aragora.debate.role_matcher import RoleMatcher, RoleMatchingConfig
        from aragora.debate.roles import RoleRotationConfig, RoleRotator

        # Role matching takes priority over simple rotation
        if self.protocol.role_matching:
            config = self.protocol.role_matching_config or RoleMatchingConfig()
            self.role_matcher = RoleMatcher(
                calibration_tracker=self.calibration_tracker,
                persona_manager=self.persona_manager,
                config=config,
            )
            logger.info("role_matcher_enabled strategy=%s", config.strategy)
        elif self.protocol.role_rotation:
            rotation_config = self.protocol.role_rotation_config or RoleRotationConfig()
            self.role_rotator = RoleRotator(rotation_config)

    def assign_initial_roles(self) -> None:
        """Assign initial roles to agents based on protocol with safety bounds."""
        # If agents already have roles, respect them
        if all(a.role for a in self.agents):
            return

        n_agents = len(self.agents)

        # Safety: Ensure at least 1 critic and 1 synthesizer when we have 3+ agents
        # This prevents all-proposer scenarios that break debate dynamics
        max_proposers = max(1, n_agents - 2) if n_agents >= 3 else 1
        proposers_needed = min(self.protocol.proposer_count, max_proposers)

        for i, agent in enumerate(self.agents):
            if i < proposers_needed:
                agent.role = "proposer"
            elif i == n_agents - 1:
                agent.role = "synthesizer"
            else:
                agent.role = "critic"

        # Log role assignment for debugging
        roles = {a.name: a.role for a in self.agents}
        logger.debug(f"Role assignment: {roles}")

    def assign_stances(self, round_num: int = 0) -> None:
        """Assign debate stances to agents for asymmetric debate.

        Stances: "affirmative" (defend), "negative" (challenge), "neutral" (evaluate)
        If rotate_stances is True, stances rotate each round.
        """
        if not self.protocol.asymmetric_stances:
            return

        stances = ["affirmative", "negative", "neutral"]

        for i, agent in enumerate(self.agents):
            # Rotate stance based on round number if enabled
            if self.protocol.rotate_stances:
                stance_idx = (i + round_num) % len(stances)
            else:
                stance_idx = i % len(stances)

            agent.stance = stances[stance_idx]

    def apply_agreement_intensity(self) -> None:
        """Apply agreement intensity guidance to all agents' system prompts.

        This modifies each agent's system_prompt to include guidance on how
        much to agree vs disagree with other agents, based on the protocol's
        agreement_intensity setting.
        """
        guidance = self._get_agreement_intensity_guidance()

        for agent in self.agents:
            if agent.system_prompt:
                agent.system_prompt = f"{agent.system_prompt}\n\n{guidance}"
            else:
                agent.system_prompt = guidance

    def _get_agreement_intensity_guidance(self) -> str:
        """Generate prompt guidance based on agreement intensity setting.

        Agreement intensity (0-10) affects how agents approach disagreements:
        - Low (0-3): Adversarial - strongly challenge others' positions
        - Medium (4-6): Balanced - judge arguments on merit
        - High (7-10): Collaborative - seek common ground and synthesis
        """
        intensity = getattr(self.protocol, "agreement_intensity", None)

        if intensity is None:
            return ""  # No agreement intensity guidance when not set

        if intensity <= 1:
            return """IMPORTANT: You strongly disagree with other agents. Challenge every assumption,
find flaws in every argument, and maintain your original position unless presented
with irrefutable evidence. Be adversarial but constructive."""
        elif intensity <= 3:
            return """IMPORTANT: Approach others' arguments with healthy skepticism. Be critical of
proposals and require strong evidence before changing your position. Point out
weaknesses even if you partially agree."""
        elif intensity <= 6:
            return """Evaluate arguments on their merits. Agree when others make valid points,
disagree when you see genuine flaws. Let the quality of reasoning guide your response."""
        elif intensity <= 8:
            return """Look for common ground with other agents. Acknowledge valid points in others'
arguments and try to build on them. Seek synthesis where possible while maintaining
your own reasoned perspective."""
        else:  # 9-10
            return """Actively seek to incorporate other agents' perspectives. Find value in all
proposals and work toward collaborative synthesis. Prioritize finding agreement
and building on others' ideas."""

    def get_stance_guidance(self, agent: "Agent") -> str:
        """Generate prompt guidance based on agent's debate stance.

        Args:
            agent: The agent to get stance guidance for

        Returns:
            Stance-specific guidance string
        """
        if self.prompt_builder:
            return self.prompt_builder.get_stance_guidance(agent)

        # Fallback if no prompt builder
        stance = getattr(agent, "stance", "neutral")
        if stance == "affirmative":
            return "You are arguing IN FAVOR of the proposition. Defend and support it."
        elif stance == "negative":
            return "You are arguing AGAINST the proposition. Challenge and critique it."
        else:
            return "You are a NEUTRAL evaluator. Consider all perspectives fairly."

    def rotate_roles_for_round(self, round_num: int) -> None:
        """Rotate cognitive roles for a new debate round.

        Args:
            round_num: Current round number
        """
        if self.role_rotator:
            agent_names = [a.name for a in self.agents]
            total_rounds = self.protocol.rounds
            self.current_role_assignments = self.role_rotator.get_assignments(
                agent_names, round_num, total_rounds
            )
        elif self.role_matcher:
            # Role matcher uses task-based matching, not round rotation
            pass

    def match_roles_for_task(self, task: str, round_num: int = 0) -> dict[str, "RoleAssignment"]:
        """Match agents to optimal roles for a specific task.

        Args:
            task: The debate task description (used as domain hint)
            round_num: Current round number for rotation fallback

        Returns:
            Dict mapping agent name to role assignment
        """
        if self.role_matcher:
            agent_names = [a.name for a in self.agents]
            result = self.role_matcher.match_roles(agent_names, round_num, debate_domain=task)
            self.current_role_assignments = result.assignments
        return self.current_role_assignments


__all__ = ["RolesManager"]
