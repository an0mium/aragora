"""
Prompt builder for debate agents.

Extracted from Arena to improve code organization and testability.
Handles construction of prompts for proposals, revisions, and judgments.
"""

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from aragora.core import Agent
    from aragora.debate.protocol import DebateProtocol
    from aragora.core import Environment
    from aragora.memory.critique_store import CritiqueStore
    from aragora.memory.continuum import ContinuumMemory
    from aragora.memory.consensus import DissentRetriever
    from aragora.agents.flip_detector import FlipDetector
    from aragora.agents.personas import PersonaManager
    from aragora.debate.roles import RoleRotator, RoleAssignment
    from aragora.core import Critique
    from aragora.evidence.collector import EvidencePack

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builds prompts for debate agents.

    Encapsulates all prompt construction logic including context injection
    for personas, roles, patterns, and historical data.
    """

    def __init__(
        self,
        protocol: "DebateProtocol",
        env: "Environment",
        memory: Optional["CritiqueStore"] = None,
        continuum_memory: Optional["ContinuumMemory"] = None,
        dissent_retriever: Optional["DissentRetriever"] = None,
        role_rotator: Optional["RoleRotator"] = None,
        persona_manager: Optional["PersonaManager"] = None,
        flip_detector: Optional["FlipDetector"] = None,
        evidence_pack: Optional["EvidencePack"] = None,
    ) -> None:
        """Initialize prompt builder with debate context.

        Args:
            protocol: Debate configuration (rounds, intensity, stances)
            env: Task environment with task description and context
            memory: Optional critique pattern store
            continuum_memory: Optional cross-debate memory
            dissent_retriever: Optional historical dissent retriever
            role_rotator: Optional cognitive role rotation
            persona_manager: Optional agent persona management
            flip_detector: Optional position consistency tracking
            evidence_pack: Optional evidence pack with research snippets
        """
        self.protocol = protocol
        self.env = env
        self.memory = memory
        self.continuum_memory = continuum_memory
        self.dissent_retriever = dissent_retriever
        self.role_rotator = role_rotator
        self.persona_manager = persona_manager
        self.flip_detector = flip_detector
        self.evidence_pack = evidence_pack

        # Current state (set externally by Arena)
        self.current_role_assignments: dict[str, "RoleAssignment"] = {}
        self._historical_context_cache: str = ""
        self._continuum_context_cache: str = ""
        self.user_suggestions: list = []

    def format_patterns_for_prompt(self, patterns: list[dict]) -> str:
        """Format learned patterns as prompt context for agents.

        This enables pattern-based learning: agents are warned about
        recurring issues from past debates before they make the same mistakes.

        Args:
            patterns: List of pattern dicts with 'category', 'pattern', 'occurrences'

        Returns:
            Formatted string to inject into debate context
        """
        if not patterns:
            return ""

        lines = ["## LEARNED PATTERNS (From Previous Debates)"]
        lines.append("Be especially careful about these recurring issues:\n")

        for p in patterns[:5]:  # Limit to top 5 patterns
            category = p.get("category", "general")
            pattern = p.get("pattern", "")
            occurrences = p.get("occurrences", 0)
            severity = p.get("avg_severity", 0)

            severity_label = ""
            if severity >= 0.7:
                severity_label = " [HIGH SEVERITY]"
            elif severity >= 0.4:
                severity_label = " [MEDIUM]"

            lines.append(f"- **{category.upper()}**{severity_label}: {pattern}")
            lines.append(f"  (Occurred in {occurrences} past debates)")

        lines.append("\nAddress these proactively to improve debate quality.")
        return "\n".join(lines)

    def get_stance_guidance(self, agent: "Agent") -> str:
        """Generate prompt guidance based on agent's debate stance."""
        if not self.protocol.asymmetric_stances:
            return ""

        stance = getattr(agent, "stance", None)
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
- Identify weaknesses, flaws, and risks
- Play devil's advocate
- Raise objections and counterarguments
- Stress-test the proposal
Even if you personally agree, argue the negative position."""

        else:  # neutral
            return """DEBATE STANCE: NEUTRAL
You are assigned to EVALUATE FAIRLY. Your role is to:
- Weigh arguments from both sides impartially
- Identify the strongest and weakest points
- Seek balanced synthesis
- Judge on merit, not position"""

    def get_agreement_intensity_guidance(self) -> str:
        """Generate prompt guidance based on agreement intensity setting.

        Agreement intensity (0-10) affects how agents approach disagreements:
        - Low (0-3): Adversarial - strongly challenge others' positions
        - Medium (4-6): Balanced - judge arguments on merit
        - High (7-10): Collaborative - seek common ground and synthesis
        """
        intensity = self.protocol.agreement_intensity

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

    def format_successful_patterns(self, limit: int = 3) -> str:
        """Format successful critique patterns for prompt injection."""
        if not self.memory:
            return ""
        try:
            patterns = self.memory.retrieve_patterns(min_success=2, limit=limit)
            if not patterns:
                return ""

            lines = ["## SUCCESSFUL PATTERNS (from past debates)"]
            for p in patterns:
                issue_preview = p.issue_text[:100] + "..." if len(p.issue_text) > 100 else p.issue_text
                fix_preview = p.suggestion_text[:80] + "..." if len(p.suggestion_text) > 80 else p.suggestion_text
                lines.append(f"- **{p.issue_type}**: {issue_preview}")
                if fix_preview:
                    lines.append(f"  Fix: {fix_preview} ({p.success_count} successes)")
            return "\n".join(lines)
        except Exception as e:
            logger.debug(f"Successful patterns formatting error: {e}")
            return ""

    def get_role_context(self, agent: "Agent") -> str:
        """Get cognitive role context for an agent in the current round."""
        if not self.role_rotator or agent.name not in self.current_role_assignments:
            return ""

        assignment = self.current_role_assignments[agent.name]
        return self.role_rotator.format_role_context(assignment)

    def get_persona_context(self, agent: "Agent") -> str:
        """Get persona context for agent specialization."""
        if not self.persona_manager:
            return ""

        # Try to get persona from database
        persona = self.persona_manager.get_persona(agent.name)
        if not persona:
            # Try default persona based on agent type (e.g., "claude_proposer" -> "claude")
            agent_type = agent.name.split("_")[0].lower()
            from aragora.agents.personas import DEFAULT_PERSONAS
            if agent_type in DEFAULT_PERSONAS:
                # DEFAULT_PERSONAS contains Persona objects directly
                persona = DEFAULT_PERSONAS[agent_type]
            else:
                return ""

        return persona.to_prompt_context()

    def get_flip_context(self, agent: "Agent") -> str:
        """Get flip/consistency context for agent self-awareness.

        This helps agents be aware of their position history and avoid
        unnecessary flip-flopping while still allowing genuine position changes.
        """
        if not self.flip_detector:
            return ""

        try:
            consistency = self.flip_detector.get_agent_consistency(agent.name)

            # Skip if no position history yet
            if consistency.total_positions == 0:
                return ""

            # Only inject context if there are notable flips
            if consistency.total_flips == 0:
                return ""

            # Build context based on flip patterns
            lines = ["## Position Consistency Note"]

            # Warn about contradictions specifically
            if consistency.contradictions > 0:
                lines.append(
                    f"You have {consistency.contradictions} prior position contradiction(s) on record. "
                    "Consider your stance carefully before arguing against positions you previously held."
                )

            # Note retractions
            if consistency.retractions > 0:
                lines.append(
                    f"You have retracted {consistency.retractions} previous position(s). "
                    "If changing positions again, clearly explain your reasoning."
                )

            # Add overall consistency score
            score = consistency.consistency_score
            if score < 0.7:
                lines.append(
                    f"Your consistency score is {score:.0%}. Prioritize coherent positions."
                )

            # Note domains with instability
            if consistency.domains_with_flips:
                domains = ", ".join(consistency.domains_with_flips[:3])
                lines.append(f"Domains with position changes: {domains}")

            return "\n".join(lines) if len(lines) > 1 else ""

        except Exception as e:
            logger.debug(f"Flip context formatting error: {e}")
            return ""

    def get_continuum_context(self) -> str:
        """Get cached continuum memory context."""
        return self._continuum_context_cache

    def format_evidence_for_prompt(self, max_snippets: int = 5) -> str:
        """Format evidence pack as citable references for agent prompts.

        Returns a formatted string with evidence snippets that agents can
        cite using [EVID-N] notation.
        """
        if not self.evidence_pack or not self.evidence_pack.snippets:
            return ""

        lines = ["## AVAILABLE EVIDENCE"]
        lines.append("Reference these sources by ID when making factual claims:\n")

        for i, snippet in enumerate(self.evidence_pack.snippets[:max_snippets], 1):
            evid_id = f"[EVID-{i}]"
            title = snippet.title[:80] if snippet.title else "Untitled"
            source = snippet.source or "Unknown"
            # Get reliability score with safe fallback (handles MagicMock, None, etc.)
            reliability = getattr(snippet, 'reliability_score', 0.5)
            if not isinstance(reliability, (int, float)):
                reliability = 0.5

            # Format the snippet
            lines.append(f"{evid_id} \"{title}\" ({source})")
            lines.append(f"  Reliability: {reliability:.0%}")
            if snippet.url:
                lines.append(f"  URL: {snippet.url}")

            # Include truncated snippet content
            content = snippet.snippet[:200] if snippet.snippet else ""
            if content:
                if len(snippet.snippet) > 200:
                    content += "..."
                lines.append(f"  > {content}")
            lines.append("")  # Blank line between snippets

        lines.append("When stating facts, cite evidence as [EVID-N]. Uncited claims may be challenged.")
        return "\n".join(lines)

    def set_evidence_pack(self, evidence_pack: Optional["EvidencePack"]) -> None:
        """Update the evidence pack (called by orchestrator between rounds)."""
        self.evidence_pack = evidence_pack

    def build_proposal_prompt(
        self,
        agent: "Agent",
        audience_section: str = "",
    ) -> str:
        """Build the initial proposal prompt.

        Args:
            agent: The agent to build the prompt for
            audience_section: Optional pre-formatted audience suggestions section
        """
        context_str = f"\n\nContext: {self.env.context}" if self.env.context else ""

        # Add research status indicator
        research_status = ""
        if not self.env.context or "No research context" in str(self.env.context):
            research_status = "\n\n[RESEARCH STATUS: No external research was performed. Base your response on your training knowledge and clearly state any limitations or uncertainties about specific entities, websites, or current events.]"
        elif "EVIDENCE CONTEXT" in str(self.env.context):
            research_status = "\n\n[RESEARCH STATUS: Research context has been provided above. Use this information in your response and cite it where applicable.]"

        stance_str = self.get_stance_guidance(agent)
        stance_section = f"\n\n{stance_str}" if stance_str else ""

        # Include cognitive role context if role rotation enabled
        role_section = self.get_role_context(agent)
        if role_section:
            role_section = f"\n\n{role_section}"

        # Include persona context for agent specialization
        persona_section = ""
        persona_context = self.get_persona_context(agent)
        if persona_context:
            persona_section = f"\n\n{persona_context}"

        # Include flip/consistency context for self-awareness
        flip_section = ""
        flip_context = self.get_flip_context(agent)
        if flip_context:
            flip_section = f"\n\n{flip_context}"

        # Include historical context if available (capped at 800 chars)
        historical_section = ""
        if self._historical_context_cache:
            historical = self._historical_context_cache[:800]
            historical_section = f"\n\n{historical}"

        # Include continuum memory context (cross-debate learnings)
        continuum_section = ""
        continuum_context = self.get_continuum_context()
        if continuum_context:
            continuum_section = f"\n\n{continuum_context}"

        # Include historical dissents and minority views
        dissent_section = ""
        if self.dissent_retriever:
            try:
                dissent_context = self.dissent_retriever.get_debate_preparation_context(
                    topic=self.env.task
                )
                if dissent_context:
                    dissent_section = f"\n\n## Historical Minority Views\n{dissent_context[:600]}"
            except Exception as e:
                logger.debug(f"Dissent retrieval error: {e}")

        # Include successful patterns from past debates
        patterns_section = ""
        patterns = self.format_successful_patterns(limit=3)
        if patterns:
            patterns_section = f"\n\n{patterns}"

        # Include evidence citations if available
        evidence_section = ""
        evidence_context = self.format_evidence_for_prompt(max_snippets=5)
        if evidence_context:
            evidence_section = f"\n\n{evidence_context}"

        # Format audience section if provided
        if audience_section:
            audience_section = f"\n\n{audience_section}"

        return f"""You are acting as a {agent.role} in a multi-agent debate.{stance_section}{role_section}{persona_section}{flip_section}
{historical_section}{continuum_section}{dissent_section}{patterns_section}{evidence_section}{audience_section}
Task: {self.env.task}{context_str}{research_status}

IMPORTANT: If this task mentions a specific website, company, product, or current topic, you MUST:
1. State what you know vs what you would need to research
2. If research context was provided above, use it. If not, acknowledge the limitation.
3. Do NOT make up facts or speculate about specific entities you don't have verified information about.

Please provide your best proposal to address this task. Be thorough and specific.
Your proposal will be critiqued by other agents, so anticipate potential objections."""

    def build_revision_prompt(
        self,
        agent: "Agent",
        original: str,
        critiques: list["Critique"],
        audience_section: str = "",
    ) -> str:
        """Build the revision prompt including critiques.

        Args:
            agent: The agent revising their proposal
            original: The original proposal text
            critiques: List of critiques received
            audience_section: Optional pre-formatted audience suggestions section
        """
        critiques_str = "\n\n".join(c.to_prompt() for c in critiques)
        intensity_guidance = self.get_agreement_intensity_guidance()
        stance_str = self.get_stance_guidance(agent)
        stance_section = f"\n\n{stance_str}" if stance_str else ""

        # Include cognitive role context if role rotation enabled
        role_section = self.get_role_context(agent)
        if role_section:
            role_section = f"\n\n{role_section}"

        # Include persona context for agent specialization
        persona_section = ""
        persona_context = self.get_persona_context(agent)
        if persona_context:
            persona_section = f"\n\n{persona_context}"

        # Include flip/consistency context (especially relevant during revisions)
        flip_section = ""
        flip_context = self.get_flip_context(agent)
        if flip_context:
            flip_section = f"\n\n{flip_context}"

        # Include successful patterns that may help address critiques
        patterns_section = ""
        patterns = self.format_successful_patterns(limit=2)
        if patterns:
            patterns_section = f"\n\n{patterns}"

        # Include evidence for strengthening revised claims
        evidence_section = ""
        evidence_context = self.format_evidence_for_prompt(max_snippets=3)
        if evidence_context:
            evidence_section = f"\n\n{evidence_context}"

        # Format audience section if provided
        if audience_section:
            audience_section = f"\n\n{audience_section}"

        return f"""You are revising your proposal based on critiques from other agents.{role_section}{persona_section}{flip_section}

{intensity_guidance}{stance_section}{patterns_section}{evidence_section}{audience_section}

Original Task: {self.env.task}

Your Original Proposal:
{original}

Critiques Received:
{critiques_str}

Please provide a revised proposal that addresses the valid critiques.
Use evidence citations [EVID-N] to support strengthened claims.
Explain what you changed and why. If you disagree with a critique, explain your reasoning."""

    def build_judge_prompt(
        self,
        proposals: dict[str, str],
        task: str,
        critiques: list["Critique"],
    ) -> str:
        """Build the judge/synthesizer prompt."""
        proposals_str = "\n\n---\n\n".join(
            f"[{agent}]:\n{prop}" for agent, prop in proposals.items()
        )
        critiques_str = "\n".join(
            f"- {c.agent}: {', '.join(c.issues[:2])}" for c in critiques[:5]
        )

        # Include evidence for final synthesis
        evidence_section = ""
        evidence_context = self.format_evidence_for_prompt(max_snippets=5)
        if evidence_context:
            evidence_section = f"\n\n{evidence_context}\n"

        return f"""You are the synthesizer/judge in a multi-agent debate.

Task: {task}
{evidence_section}
Proposals:
{proposals_str}

Key Critiques:
{critiques_str}

Synthesize the best elements of all proposals into a final answer.
Reference evidence [EVID-N] to support key claims in your synthesis.
Address the most important critiques raised. Explain your synthesis."""

    def build_judge_vote_prompt(
        self, candidates: list["Agent"], proposals: dict[str, str]
    ) -> str:
        """Build prompt for voting on who should judge."""
        candidate_names = ", ".join(a.name for a in candidates)
        proposals_summary = "\n".join(
            f"- {name}: {prop[:300]}..." for name, prop in proposals.items()
        )

        return f"""Based on the proposals in this debate, vote for which agent should synthesize the final answer.

Candidates: {candidate_names}

Proposals summary:
{proposals_summary}

Consider: Which agent showed the most balanced, thorough, and fair reasoning?
Vote by stating ONLY the agent's name. You cannot vote for yourself."""
