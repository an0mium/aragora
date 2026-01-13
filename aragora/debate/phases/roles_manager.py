"""
Role and stance management logic extracted from Arena.

Provides utilities for:
- Role assignment (proposer, critic, synthesizer)
- Stance assignment (affirmative, negative, neutral)
- Agreement intensity guidance
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.core import Agent
    from aragora.debate.protocol import DebateProtocol

logger = logging.getLogger(__name__)


class RolesManager:
    """Manages role and stance assignments for debate agents.

    Roles:
    - proposer: Generates initial proposals
    - critic: Critiques proposals from others
    - synthesizer: Synthesizes final verdict

    Stances (for asymmetric debates):
    - affirmative: Defend and support proposals
    - negative: Challenge and critique proposals
    - neutral: Evaluate objectively
    """

    def __init__(self, protocol: "DebateProtocol", agents: list["Agent"]):
        """Initialize roles manager.

        Args:
            protocol: Debate protocol with role configuration
            agents: List of all agents in the debate
        """
        self.protocol = protocol
        self.agents = agents

    def assign_roles(self) -> None:
        """Assign roles to agents based on protocol with safety bounds.

        Safety guarantees:
        - At least 1 critic and 1 synthesizer when 3+ agents
        - Prevents all-proposer scenarios that break debate dynamics
        """
        # If agents already have roles, respect them
        if all(a.role for a in self.agents):
            return

        n_agents = len(self.agents)

        # Safety: Ensure at least 1 critic and 1 synthesizer when we have 3+ agents
        max_proposers = max(1, n_agents - 2) if n_agents >= 3 else 1
        proposers_needed = min(self.protocol.proposer_count, max_proposers)

        for i, agent in enumerate(self.agents):
            if i < proposers_needed:
                agent.role = "proposer"
            elif i == n_agents - 1:
                agent.role = "synthesizer"
            else:
                agent.role = "critic"

        # Log role assignment
        roles = {a.name: a.role for a in self.agents}
        logger.debug(f"Role assignment: {roles}")

    def assign_stances(self, round_num: int = 0) -> None:
        """Assign debate stances to agents for asymmetric debate.

        Stances rotate each round if rotate_stances is enabled.

        Args:
            round_num: Current round number (for rotation)
        """
        if not self.protocol.asymmetric_stances:
            return

        stances = ["affirmative", "negative", "neutral"]
        for i, agent in enumerate(self.agents):
            if self.protocol.rotate_stances:
                stance_idx = (i + round_num) % len(stances)
            else:
                stance_idx = i % len(stances)

            agent.stance = stances[stance_idx]

    def apply_agreement_intensity(self) -> None:
        """Apply agreement intensity guidance to all agents' system prompts.

        Modifies each agent's system_prompt to include guidance on how
        much to agree vs disagree with other agents.
        """
        guidance = self.get_agreement_intensity_guidance()
        if not guidance:
            return

        for agent in self.agents:
            if agent.system_prompt:
                agent.system_prompt = f"{agent.system_prompt}\n\n{guidance}"
            else:
                agent.system_prompt = guidance

    def get_agreement_intensity_guidance(self) -> str:
        """Generate prompt guidance based on agreement intensity setting.

        Agreement intensity (0-10) affects how agents approach disagreements:
        - Low (0-3): Adversarial - strongly challenge others' positions
        - Medium (4-6): Balanced - judge arguments on merit
        - High (7-10): Collaborative - seek common ground and synthesis

        Returns:
            Guidance string to append to agent prompts
        """
        intensity = getattr(self.protocol, "agreement_intensity", 5)

        if intensity <= 3:
            return """DEBATE MODE: ADVERSARIAL
You should strongly challenge and question other agents' positions.
Look for weaknesses, contradictions, and unsupported claims.
Push back on assertions that lack evidence.
Your role is to stress-test ideas through rigorous critique."""

        elif intensity >= 7:
            return """DEBATE MODE: COLLABORATIVE
You should seek common ground and synthesize different perspectives.
Build upon others' ideas constructively.
Focus on finding points of agreement while addressing differences.
Your role is to help the group converge on the best solution."""

        else:
            return """DEBATE MODE: BALANCED
Evaluate arguments on their merits objectively.
Acknowledge good points while also noting weaknesses.
Neither automatically agree nor disagree - let evidence guide you.
Your role is to contribute fair and reasoned analysis."""

    def get_stance_guidance(self, agent: "Agent") -> str:
        """Generate prompt guidance based on agent's debate stance.

        Args:
            agent: The agent to get guidance for

        Returns:
            Stance guidance string
        """
        if not self.protocol.asymmetric_stances:
            return ""

        stance = getattr(agent, "stance", None)
        if not stance:
            return ""

        if stance == "affirmative":
            return """DEBATE STANCE: AFFIRMATIVE
You are assigned to DEFEND and SUPPORT proposals. Your role is to:
- Find strengths and merits in arguments
- Build upon existing ideas
- Advocate for the proposal's value
- Counter criticisms constructively
Even if you personally disagree, argue the affirmative position."""

        elif stance == "negative":
            return """DEBATE STANCE: NEGATIVE
You are assigned to CHALLENGE and CRITIQUE proposals. Your role is to:
- Identify weaknesses and potential problems
- Question assumptions and evidence
- Present counterarguments
- Play devil's advocate
Even if you personally agree, argue the negative position."""

        elif stance == "neutral":
            return """DEBATE STANCE: NEUTRAL
You are assigned to EVALUATE objectively. Your role is to:
- Weigh both sides fairly
- Identify the strongest arguments from each position
- Note where consensus exists vs disagreement
- Provide balanced analysis
Do not advocate - analyze impartially."""

        return ""

    def get_role_summary(self) -> dict[str, list[str]]:
        """Get summary of current role assignments.

        Returns:
            Dict mapping role -> list of agent names
        """
        summary: dict[str, list[str]] = {
            "proposer": [],
            "critic": [],
            "synthesizer": [],
        }

        for agent in self.agents:
            role = getattr(agent, "role", "unknown")
            if role in summary:
                summary[role].append(agent.name)
            else:
                summary.setdefault(role, []).append(agent.name)

        return summary

    def get_stance_summary(self) -> dict[str, list[str]]:
        """Get summary of current stance assignments.

        Returns:
            Dict mapping stance -> list of agent names
        """
        summary: dict[str, list[str]] = {
            "affirmative": [],
            "negative": [],
            "neutral": [],
        }

        for agent in self.agents:
            stance = getattr(agent, "stance", None)
            if stance in summary:
                summary[stance].append(agent.name)

        return summary
