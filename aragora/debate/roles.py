"""
Cognitive Role System for Multi-Agent Debates.

Inspired by Heavy3.ai's Deep Audit feature which assigns explicit cognitive roles:
- Analyst: Conducts deep investigation
- Skeptic: Challenges underlying assumptions
- Lateral Thinker: Explores unconventional angles
- Synthesizer: Cross-examines findings and delivers verdicts

This module enables role rotation across rounds, ensuring each agent explores
different cognitive perspectives throughout the debate.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class CognitiveRole(Enum):
    """
    Cognitive roles that shape how an agent approaches a debate.

    Each role brings a distinct perspective:
    - ANALYST: Deep, thorough investigation of the topic
    - SKEPTIC: Challenges assumptions and seeks weaknesses
    - LATERAL_THINKER: Explores unconventional solutions
    - SYNTHESIZER: Integrates perspectives and finds common ground
    - ADVOCATE: Strongly defends a particular position
    - DEVIL_ADVOCATE: Argues against the emerging consensus
    - QUALITY_CHALLENGER: Evidence-Powered Trickster - challenges hollow consensus
    """

    ANALYST = "analyst"
    SKEPTIC = "skeptic"
    LATERAL_THINKER = "lateral_thinker"
    SYNTHESIZER = "synthesizer"
    ADVOCATE = "advocate"
    DEVIL_ADVOCATE = "devil_advocate"
    QUALITY_CHALLENGER = "quality_challenger"


# Role prompts that modify agent behavior
ROLE_PROMPTS = {
    CognitiveRole.ANALYST: """
## Your Cognitive Role: ANALYST

You are the Analyst. Your job is deep, thorough investigation.

Approach:
- Examine every claim systematically
- Look for evidence and data to support or refute positions
- Identify gaps in reasoning or missing information
- Ask probing questions that get to the root of the issue
- Document your findings with precision

Key questions to ask:
- What evidence exists for this claim?
- What are the key assumptions being made?
- What data would help us decide?
- Are there relevant precedents or case studies?
""",
    CognitiveRole.SKEPTIC: """
## Your Cognitive Role: SKEPTIC

You are the Skeptic. Your job is to challenge underlying assumptions.

Approach:
- Question everything, especially things everyone seems to agree on
- Identify hidden assumptions and unstated premises
- Look for weaknesses in arguments that others might overlook
- Probe for edge cases and failure modes
- Don't accept claims without rigorous justification

Key questions to ask:
- Why do we believe this is true?
- What would prove this wrong?
- What are we assuming without evidence?
- What could go wrong that we haven't considered?
""",
    CognitiveRole.LATERAL_THINKER: """
## Your Cognitive Role: LATERAL THINKER

You are the Lateral Thinker. Your job is to explore unconventional angles.

Approach:
- Think outside the box - consider solutions others might dismiss
- Draw analogies from different domains and fields
- Challenge the problem framing itself
- Look for creative reframes that change the constraints
- Propose unexpected alternatives

Key questions to ask:
- Is there a completely different way to think about this?
- What would someone from a different field suggest?
- What if we inverted the problem?
- What unconventional approach might actually work?
""",
    CognitiveRole.SYNTHESIZER: """
## Your Cognitive Role: SYNTHESIZER

You are the Synthesizer. Your job is to integrate perspectives and find common ground.

Approach:
- Identify the strongest elements from each proposal
- Find underlying agreements even in apparent disagreements
- Build bridges between different viewpoints
- Create hybrid solutions that incorporate multiple ideas
- Summarize and distill key insights

Key questions to ask:
- What do all proposals have in common?
- Can we combine the best elements from each?
- What's the core insight we should preserve?
- How do we resolve apparent contradictions?
""",
    CognitiveRole.ADVOCATE: """
## Your Cognitive Role: ADVOCATE

You are the Advocate. Your job is to build the strongest possible case for a position.

Approach:
- Develop the most compelling arguments for your position
- Anticipate and address counterarguments proactively
- Marshal evidence and examples to support your case
- Present your position with confidence and clarity
- Defend against critiques constructively

Key questions to ask:
- What's the strongest argument for this approach?
- How do we address the likely objections?
- What evidence best supports this position?
- Why is this the right choice given the constraints?
""",
    CognitiveRole.DEVIL_ADVOCATE: """
## Your Cognitive Role: DEVIL'S ADVOCATE

You are the Devil's Advocate. Your job is to argue against the emerging consensus.

Approach:
- Actively challenge the position that seems to be winning
- Present the strongest possible counterarguments
- Expose risks and downsides others might minimize
- Ensure we don't reach premature consensus
- Be constructively adversarial

Key questions to ask:
- What are we overlooking because we want this to work?
- What are the hidden costs or risks?
- Who would disagree with this and why?
- What's the strongest argument against the leading position?
""",
    CognitiveRole.QUALITY_CHALLENGER: """
## Your Cognitive Role: QUALITY CHALLENGER (Evidence-Powered Trickster)

You are the Quality Challenger. Your job is to detect and challenge hollow consensus.

Hollow consensus occurs when agents agree without substantive evidence - when
positions converge through social dynamics rather than rigorous reasoning.

Approach:
- Demand concrete evidence for all major claims
- Challenge vague language ("generally", "often", "significant")
- Require specific data, citations, or examples
- Question the logical chain connecting premises to conclusions
- Expose when agreement masks shallow understanding

Key challenges to pose:
- "What specific data supports this claim?"
- "Can you provide a concrete example of this working in practice?"
- "What would falsify this position?"
- "Why should we accept this premise without evidence?"
- "The agreement here seems premature - what have we not examined?"

Quality indicators to demand:
- Citations to sources
- Specific numbers and metrics
- Concrete examples and case studies
- Logical reasoning chains
- Acknowledgment of limitations and uncertainties

Your goal is intellectual rigor, not obstruction. When evidence is provided,
acknowledge it. When reasoning is sound, accept it. But never let the debate
reach consensus without substantive backing.
""",
}


@dataclass
class RoleAssignment:
    """An assignment of a cognitive role to an agent for a specific round."""

    agent_name: str
    role: CognitiveRole
    round_num: int
    role_prompt: str = ""

    def __post_init__(self):
        if not self.role_prompt:
            self.role_prompt = ROLE_PROMPTS.get(self.role, "")


@dataclass
class RoleRotationConfig:
    """Configuration for role rotation across debate rounds."""

    # Whether to rotate roles each round
    enabled: bool = True

    # Available roles to assign
    roles: list[CognitiveRole] = field(
        default_factory=lambda: [
            CognitiveRole.ANALYST,
            CognitiveRole.SKEPTIC,
            CognitiveRole.LATERAL_THINKER,
            CognitiveRole.SYNTHESIZER,
        ]
    )

    # Ensure each agent gets each role at least once (requires enough rounds)
    ensure_coverage: bool = True

    # Force synthesizer role in final round
    synthesizer_final_round: bool = True


class RoleRotator:
    """
    Manages cognitive role rotation across debate rounds.

    Ensures diverse perspectives by rotating which cognitive role
    each agent takes each round.
    """

    def __init__(self, config: Optional[RoleRotationConfig] = None):
        self.config = config or RoleRotationConfig()
        self._rotation_state: dict[str, int] = {}  # agent -> role index

    def get_assignments(
        self,
        agent_names: list[str],
        round_num: int,
        total_rounds: int,
    ) -> dict[str, RoleAssignment]:
        """
        Get role assignments for a specific round.

        Args:
            agent_names: Names of participating agents
            round_num: Current round number (0-indexed)
            total_rounds: Total number of debate rounds

        Returns:
            Dict mapping agent name to RoleAssignment
        """
        if not self.config.enabled:
            return {}

        assignments = {}
        available_roles = list(self.config.roles)

        # Final round: assign synthesizer to one agent if configured
        is_final_round = round_num == total_rounds - 1
        synthesizer_agent = None

        if is_final_round and self.config.synthesizer_final_round and agent_names:
            # Give synthesizer to the agent who hasn't been synthesizer yet
            # or the first agent if all have been
            synthesizer_agent = agent_names[0]
            assignments[synthesizer_agent] = RoleAssignment(
                agent_name=synthesizer_agent,
                role=CognitiveRole.SYNTHESIZER,
                round_num=round_num,
            )

        # Assign other roles with rotation
        for i, agent_name in enumerate(agent_names):
            if agent_name == synthesizer_agent:
                continue

            # Get current role index for this agent
            if agent_name not in self._rotation_state:
                self._rotation_state[agent_name] = i % len(available_roles)

            role_idx = (self._rotation_state[agent_name] + round_num) % len(available_roles)
            role = available_roles[role_idx]

            assignments[agent_name] = RoleAssignment(
                agent_name=agent_name,
                role=role,
                round_num=round_num,
            )

        return assignments

    def get_role_prompt(self, agent_name: str, round_num: int) -> str:
        """Get the role prompt for an agent in a specific round."""
        # Note: This requires assignments to have been generated first
        return ROLE_PROMPTS.get(CognitiveRole.ANALYST, "")  # Default

    def format_role_context(self, assignment: RoleAssignment) -> str:
        """Format a role assignment for inclusion in agent prompt."""
        return f"""
=== COGNITIVE ROLE ASSIGNMENT (Round {assignment.round_num + 1}) ===
{assignment.role_prompt}
=== END ROLE ASSIGNMENT ===
"""


def create_role_rotation(
    agents: list,
    total_rounds: int,
    config: Optional[RoleRotationConfig] = None,
) -> list[dict[str, RoleAssignment]]:
    """
    Create a complete role rotation schedule for a debate.

    Args:
        agents: List of agent objects (must have .name attribute)
        total_rounds: Number of debate rounds
        config: Optional rotation configuration

    Returns:
        List of dicts, one per round, mapping agent name to assignment
    """
    rotator = RoleRotator(config)
    agent_names = [a.name for a in agents]

    schedule = []
    for round_num in range(total_rounds):
        assignments = rotator.get_assignments(agent_names, round_num, total_rounds)
        schedule.append(assignments)

    return schedule


def inject_role_into_prompt(
    base_prompt: str,
    assignment: RoleAssignment,
) -> str:
    """
    Inject a role assignment into an agent's prompt.

    Args:
        base_prompt: The original prompt
        assignment: The role assignment for this round

    Returns:
        Modified prompt with role context prepended
    """
    rotator = RoleRotator()
    role_context = rotator.format_role_context(assignment)
    return f"{role_context}\n\n{base_prompt}"
