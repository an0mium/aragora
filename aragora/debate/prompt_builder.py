"""
Prompt builder for debate agents.

Extracted from Arena to improve code organization and testability.
Handles construction of prompts for proposals, revisions, and judgments.
"""

import logging
from typing import TYPE_CHECKING, Optional

# Import QuestionClassifier for LLM-based classification
try:
    from aragora.server.question_classifier import QuestionClassifier, QuestionClassification
except ImportError:
    QuestionClassifier = None  # type: ignore[misc,assignment]
    QuestionClassification = None  # type: ignore[misc,assignment]

if TYPE_CHECKING:
    from aragora.agents.calibration import CalibrationTracker
    from aragora.agents.flip_detector import FlipDetector
    from aragora.agents.personas import PersonaManager
    from aragora.core import Agent, Critique, Environment
    from aragora.debate.protocol import DebateProtocol
    from aragora.debate.roles import RoleAssignment, RoleRotator
    from aragora.evidence.collector import EvidencePack
    from aragora.memory.consensus import DissentRetriever
    from aragora.memory.continuum import ContinuumMemory
    from aragora.memory.store import CritiqueStore
    from aragora.ranking.elo import EloSystem
    from aragora.rlm.types import RLMContext

# Check for RLM availability
try:
    from aragora.rlm import HierarchicalCompressor, AbstractionLevel, RLMContextAdapter
    HAS_RLM = True
except ImportError:
    HAS_RLM = False
    HierarchicalCompressor = None  # type: ignore[misc,assignment]
    AbstractionLevel = None  # type: ignore[misc,assignment]
    RLMContextAdapter = None  # type: ignore[misc,assignment]

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
        calibration_tracker: Optional["CalibrationTracker"] = None,
        elo_system: Optional["EloSystem"] = None,
        domain: str = "general",
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
            calibration_tracker: Optional calibration tracker for confidence feedback
            elo_system: Optional ELO system for agent ranking context
            domain: Debate domain for domain-specific ELO lookup
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
        self.calibration_tracker = calibration_tracker
        self.elo_system = elo_system
        self.domain = domain

        # Current state (set externally by Arena)
        self.current_role_assignments: dict[str, "RoleAssignment"] = {}
        self._historical_context_cache: str = ""
        self._continuum_context_cache: str = ""
        self.user_suggestions: list = []

        # Question classification cache (populated by classify_question_async)
        self._classification: Optional["QuestionClassification"] = None
        self._question_classifier: Optional["QuestionClassifier"] = None

        # RLM hierarchical context (set by ContextInitializer when enabled)
        self._rlm_context: Optional["RLMContext"] = None
        self._enable_rlm_hints: bool = HAS_RLM  # Show agents how to query context

        # RLM context adapter for external environment pattern
        # Content is registered here and accessed via query, not stuffed in prompt
        self._rlm_adapter: Optional["RLMContextAdapter"] = None
        if HAS_RLM and RLMContextAdapter is not None:
            self._rlm_adapter = RLMContextAdapter()

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
                issue_preview = (
                    p.issue_text[:100] + "..." if len(p.issue_text) > 100 else p.issue_text
                )
                fix_preview = (
                    p.suggestion_text[:80] + "..."
                    if len(p.suggestion_text) > 80
                    else p.suggestion_text
                )
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

    def get_round_phase_context(self, round_number: int) -> str:
        """Get structured phase context for the current debate round.

        Args:
            round_number: 1-indexed round number

        Returns:
            Formatted guidance string for the current phase
        """
        phase = self.protocol.get_round_phase(round_number)
        if not phase:
            return ""

        lines = [f"## ROUND {round_number}: {phase.name.upper()}"]
        lines.append(f"**Phase Objective:** {phase.description}")
        lines.append(f"**Focus:** {phase.focus}")
        lines.append(f"**Cognitive Mode:** {phase.cognitive_mode}")
        lines.append("")

        # Add phase-specific guidance
        if phase.cognitive_mode == "Analyst":
            lines.append("Analyze thoroughly. Establish facts and key considerations.")
        elif phase.cognitive_mode == "Skeptic":
            lines.append("Challenge assumptions. Find weaknesses and edge cases.")
        elif phase.cognitive_mode == "Lateral Thinker":
            lines.append("Think creatively. Explore unconventional approaches and analogies.")
        elif phase.cognitive_mode == "Devil's Advocate":
            lines.append("Argue the opposing view. Surface risks and unintended consequences.")
        elif phase.cognitive_mode == "Synthesizer":
            lines.append("Integrate insights. Find common ground and build consensus.")
        elif phase.cognitive_mode == "Examiner":
            lines.append("Question directly. Clarify remaining disputes and test convictions.")
        elif phase.cognitive_mode == "Researcher":
            lines.append(
                "Gather evidence. Research background context, find supporting data, identify key sources."
            )
        elif phase.cognitive_mode == "Adjudicator":
            lines.append(
                "Render judgment. Weigh all arguments fairly, select the strongest position, explain your verdict."
            )
        elif phase.cognitive_mode == "Integrator":
            lines.append(
                "Connect the dots. Identify emerging patterns, bridge opposing views, highlight key trade-offs."
            )

        return "\n".join(lines)

    async def classify_question_async(self, use_llm: bool = True) -> str:
        """Classify the debate question using LLM (async).

        This method should be called once at debate start to classify the
        question using Claude for accurate domain detection. Results are
        cached for subsequent calls to get_persona_context().

        Args:
            use_llm: If True, use Claude for classification (more accurate).
                     If False, use fast keyword-based classification.

        Returns:
            The detected domain category.
        """
        if self._classification is not None:
            return self._classification.category

        if QuestionClassifier is None:
            logger.debug("QuestionClassifier not available, using keyword fallback")
            return self._detect_question_domain_keywords(self.env.task)

        try:
            if self._question_classifier is None:
                self._question_classifier = QuestionClassifier()

            if use_llm:
                # Use LLM-based classification (more accurate)
                self._classification = self._question_classifier.classify(self.env.task)
                logger.info(
                    f"LLM classification: category={self._classification.category}, "
                    f"confidence={self._classification.confidence:.2f}, "
                    f"personas={self._classification.recommended_personas}"
                )
            else:
                # Use fast keyword-based classification
                self._classification = self._question_classifier.classify_simple(self.env.task)
                logger.debug(f"Keyword classification: category={self._classification.category}")

            return self._classification.category

        except Exception as e:
            logger.warning(f"Question classification failed, using keyword fallback: {e}")
            return self._detect_question_domain_keywords(self.env.task)

    def _detect_question_domain(self, question: str) -> str:
        """Detect question domain for persona selection.

        Uses cached LLM classification if available, otherwise falls back
        to keyword-based detection.

        Returns 'philosophical', 'ethics', 'technical', 'ethical', or 'general'.
        """
        # Use cached LLM classification if available
        if self._classification is not None:
            # Map classifier categories to our domain categories
            category = self._classification.category
            if category in ("ethical", "philosophical"):
                return "philosophical"  # Both use philosophical personas
            elif category == "technical":
                return "technical"
            elif category in ("legal", "security", "financial", "healthcare", "scientific"):
                return "technical"  # Specialized domains use technical personas
            else:
                return "general"

        # Fall back to keyword-based detection
        return self._detect_question_domain_keywords(question)

    def _detect_question_domain_keywords(self, question: str) -> str:
        """Keyword-based domain detection (fallback when LLM unavailable).

        Returns 'philosophical', 'ethics', 'technical', or 'general'.
        """
        import re

        lower = question.lower()

        def word_match(keywords: list[str]) -> bool:
            """Check if any keyword appears as a whole word."""
            for kw in keywords:
                # Use word boundary regex to avoid substring matches
                # e.g., "api" shouldn't match "capitalism"
                if re.search(rf"\b{re.escape(kw)}\b", lower):
                    return True
            return False

        # Philosophical/Life/Theological domains - use philosophical personas
        philosophical_keywords = [
            "meaning",
            "meaningful",
            "life",
            "purpose",
            "existence",
            "happiness",
            "soul",
            "consciousness",
            "free will",
            "morality",
            "good life",
            "philosophy",
            "wisdom",
            "death",
            "love",
            "truth",
            "human condition",
            "fulfillment",
            "wellbeing",
            "flourishing",
            # Theological/Religious keywords
            "heaven",
            "hell",
            "afterlife",
            "divine",
            "god",
            "theological",
            "theology",
            "religious",
            "faith",
            "spiritual",
            "prayer",
            "scripture",
            "bible",
            "sin",
            "redemption",
            "salvation",
            "eternal",
            "sacred",
            "holy",
            "angel",
            "fetus",
            "abortion",
        ]
        if word_match(philosophical_keywords):
            return "philosophical"

        # Ethics domain - questions about right/wrong, good/bad, should/shouldn't
        ethics_keywords = [
            "should",
            "ethical",
            "moral",
            "right",
            "wrong",
            "justice",
            "fair",
            "harm",
            "good or bad",
            "bad or good",
        ]
        if word_match(ethics_keywords):
            return "ethics"

        # Technical domain - keep existing personas (use word boundaries to avoid
        # false positives like "capitalism" matching "api")
        technical_keywords = [
            "code",
            "api",
            "software",
            "architecture",
            "database",
            "security",
            "testing",
            "function",
            "class",
            "microservice",
            "deployment",
            "infrastructure",
            "programming",
            "algorithm",
            "backend",
            "frontend",
        ]
        if word_match(technical_keywords):
            return "technical"

        return "general"

    def get_persona_context(self, agent: "Agent") -> str:
        """Get persona context for agent specialization.

        For philosophical or ethical questions, returns appropriate guidance
        to avoid agents framing responses in technical terms.
        """
        # Detect domain from task/environment
        question_domain = self._detect_question_domain(self.env.task)

        # For philosophical questions, override technical personas with humanistic guidance
        if question_domain == "philosophical":
            return (
                "Approach this question as a thoughtful observer of the human condition. "
                "Draw on wisdom traditions, philosophy, psychology, and lived experience. "
                "Avoid framing your answer in technical or software metaphors. "
                "Focus on what makes life meaningful, purposeful, and fulfilling."
            )

        # For ethics questions, emphasize ethical reasoning
        if question_domain == "ethics":
            return (
                "Approach this as an ethical question requiring nuanced moral reasoning. "
                "Consider multiple ethical frameworks, stakeholder perspectives, and real-world consequences. "
                "Acknowledge complexity and avoid reductive technical framings."
            )

        # For general questions, use a wise humanistic persona (not technical)
        if question_domain == "general":
            return (
                "Approach this as a thoughtful, experienced, and friendly advisor. "
                "Draw on broad knowledge, practical wisdom, and common sense. "
                "Be clear, helpful, and accessible. Avoid technical jargon unless the "
                "question specifically calls for it."
            )

        # Technical domain: use existing persona system for technical questions
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

    def set_rlm_context(self, context: Optional["RLMContext"]) -> None:
        """Set hierarchical RLM context for drill-down access.

        This context enables agents to query compressed debate history
        at different abstraction levels rather than working with truncated text.

        Args:
            context: RLMContext from HierarchicalCompressor, or None to disable
        """
        self._rlm_context = context
        if context:
            logger.debug(
                "[rlm] Set hierarchical context with %d levels",
                len(context.levels) if hasattr(context, 'levels') else 0
            )

    def get_rlm_context_hint(self) -> str:
        """Get RLM context hint for agent prompts.

        When hierarchical context is available, this informs agents that
        they can query for more detail on specific topics instead of
        relying only on the truncated summary.

        Returns:
            Formatted hint string, or empty if RLM not available
        """
        if not self._enable_rlm_hints or not self._rlm_context:
            return ""

        # Build hint based on available abstraction levels
        levels_available = []
        if hasattr(self._rlm_context, 'levels'):
            for level in self._rlm_context.levels:
                levels_available.append(level.name if hasattr(level, 'name') else str(level))

        if not levels_available:
            return ""

        return f"""## HIERARCHICAL CONTEXT AVAILABLE
The debate history has been compressed into multiple abstraction levels
for efficient processing. You are seeing a SUMMARY view.

**Available detail levels:** {', '.join(levels_available)}

If you need more detail on a specific topic mentioned in the context,
you can request drill-down by including in your response:
  [QUERY: <your specific question about the context>]

The system will provide relevant details from the full history."""

    def get_rlm_abstract(self, max_chars: int = 2000) -> str:
        """Get abstract-level summary from RLM context.

        Returns the highest abstraction level available, suitable for
        injection into prompts when context limits are tight.

        Args:
            max_chars: Maximum characters to return

        Returns:
            Abstract summary or empty string
        """
        if not self._rlm_context:
            return ""

        try:
            # Try to get ABSTRACT level first, then SUMMARY
            if AbstractionLevel and hasattr(self._rlm_context, 'get_at_level'):
                abstract = self._rlm_context.get_at_level(AbstractionLevel.ABSTRACT)
                if abstract:
                    return abstract[:max_chars]

                summary = self._rlm_context.get_at_level(AbstractionLevel.SUMMARY)
                if summary:
                    return summary[:max_chars]

            # Fallback to original content truncated
            if hasattr(self._rlm_context, 'original_content'):
                return self._rlm_context.original_content[:max_chars] + "..."

        except Exception as e:
            logger.debug(f"RLM abstract retrieval error: {e}")

        return ""

    def get_language_constraint(self) -> str:
        """Get language enforcement instruction for agent prompts.

        Returns instruction requiring agents to respond in the configured
        debate language. This prevents multilingual models (DeepSeek, Kimi, Qwen)
        from code-switching mid-response.

        Returns:
            Language constraint instruction, or empty string if disabled.
        """
        from aragora.config import DEFAULT_DEBATE_LANGUAGE, ENFORCE_RESPONSE_LANGUAGE

        if not ENFORCE_RESPONSE_LANGUAGE:
            return ""

        lang = getattr(self.protocol, "language", None) or DEFAULT_DEBATE_LANGUAGE
        return (
            f"\n\n**LANGUAGE REQUIREMENT**: You MUST respond entirely in {lang}. "
            f"Do not use any other language. If you need to reference foreign terms, "
            f"provide a {lang} translation in parentheses."
        )

    def _inject_belief_context(self, limit: int = 3) -> str:
        """Retrieve and format historical belief cruxes for prompt injection.

        Searches ContinuumMemory for relevant past debates and extracts
        crux_claims from their metadata. These cruxes represent key points
        of contention from similar debates that agents should be aware of.

        Args:
            limit: Maximum number of memories to search

        Returns:
            Formatted string with historical cruxes, or empty string
        """
        if not self.continuum_memory:
            return ""

        try:
            # Retrieve relevant memories for the current task
            memories = self.continuum_memory.retrieve(
                query=self.env.task,
                limit=limit,
            )

            if not memories:
                return ""

            # Extract crux_claims from memory metadata
            all_cruxes: list[str] = []
            for mem in memories:
                metadata = getattr(mem, "metadata", {}) or {}
                cruxes = metadata.get("crux_claims", [])
                if isinstance(cruxes, list):
                    all_cruxes.extend(str(c) for c in cruxes if c)

            # Deduplicate and limit
            unique_cruxes = list(dict.fromkeys(all_cruxes))[:5]

            if not unique_cruxes:
                return ""

            # Format as prompt context
            lines = ["## Historical Disagreement Points"]
            lines.append(
                "Past debates on similar topics identified these key points of contention:"
            )
            for crux in unique_cruxes:
                lines.append(f"- {crux}")
            lines.append("\nConsider whether your proposal addresses these concerns.")

            return "\n".join(lines)

        except Exception as e:
            logger.debug(f"Belief context injection error: {e}")
            return ""

    def _inject_calibration_context(self, agent: "Agent") -> str:
        """Inject calibration feedback into agent prompts.

        Retrieves the agent's historical calibration performance and provides
        feedback to help them improve confidence estimates. This creates
        a feedback loop where poorly calibrated agents receive guidance.

        Args:
            agent: The agent to get calibration feedback for

        Returns:
            Formatted calibration context string, or empty string if no data
        """
        if not self.calibration_tracker:
            return ""

        try:
            summary = self.calibration_tracker.get_calibration_summary(agent.name)

            # Need at least 5 predictions for meaningful feedback
            if summary.total_predictions < 5:
                return ""

            brier = summary.brier_score

            # Only provide feedback for poorly calibrated agents (Brier > 0.25)
            # 0.25 is roughly random guessing at 50% confidence
            if brier <= 0.25:
                return ""

            lines = ["## Calibration Feedback"]
            lines.append(
                f"Your historical prediction accuracy needs improvement (Brier score: {brier:.2f})."
            )

            if summary.is_overconfident:
                lines.append(
                    "You tend to be OVERCONFIDENT - your certainty often exceeds your accuracy."
                )
                lines.append("Consider expressing more uncertainty in your claims.")
            elif summary.is_underconfident:
                lines.append(
                    "You tend to be UNDERCONFIDENT - your accuracy is better than your expressed certainty."
                )
                lines.append("You can express more confidence in well-supported claims.")

            lines.append("\nAdjust your certainty levels in this debate accordingly.")

            return "\n".join(lines)

        except Exception as e:
            logger.debug(f"Calibration context injection error: {e}")
            return ""

    def get_elo_context(self, agent: "Agent", all_agents: list["Agent"]) -> str:
        """Inject ELO ranking context for agent awareness of relative expertise (B3).

        Provides agents with information about their own and other agents'
        ELO ratings and calibration scores. This enables:
        - More informed deference to domain experts
        - Appropriate confidence based on track record
        - Strategic adaptation based on relative standings

        Args:
            agent: The agent to get ELO context for
            all_agents: All agents in the debate for comparison

        Returns:
            Formatted ELO context string, or empty string if no data
        """
        if not self.elo_system:
            return ""

        try:
            # Get all agent names for batch lookup
            agent_names = [a.name for a in all_agents]

            # Batch fetch ratings
            ratings_batch = self.elo_system.get_ratings_batch(agent_names)
            if not ratings_batch:
                return ""

            # Get domain-specific ELO if domain is set
            domain_suffix = ""
            if self.domain and self.domain != "general":
                domain_suffix = f" ({self.domain})"

            lines = [f"## Agent Rankings{domain_suffix}"]
            lines.append("Consider these rankings when weighing arguments:\n")

            # Sort by ELO for display
            sorted_ratings = sorted(
                [(name, rating) for name, rating in ratings_batch.items()],
                key=lambda x: x[1].elo,
                reverse=True,
            )

            for rank, (name, rating) in enumerate(sorted_ratings, 1):
                elo = rating.elo
                wins = getattr(rating, "wins", 0)
                losses = getattr(rating, "losses", 0)
                total = wins + losses

                # Mark this agent
                marker = " (you)" if name == agent.name else ""

                # Show calibration if available
                calib_str = ""
                if self.calibration_tracker:
                    try:
                        summary = self.calibration_tracker.get_calibration_summary(name)
                        if summary.total_predictions >= 5:
                            accuracy = 1.0 - summary.brier_score  # Convert Brier to accuracy
                            calib_str = f", {accuracy:.0%} calibration"
                    except Exception as e:
                        logger.debug(f"Failed to get calibration summary for {name}: {e}")

                lines.append(
                    f"  {rank}. {name}: {elo:.0f} ELO ({total} debates{calib_str}){marker}"
                )

            # Add guidance for this agent
            self_rating = ratings_batch.get(agent.name)
            if self_rating:
                lines.append("")
                if self_rating.elo >= 1600:
                    lines.append(
                        "You have a strong track record. Lead with confidence but remain open to critique."
                    )
                elif self_rating.elo <= 1400:
                    lines.append("Consider carefully weighing arguments from higher-ranked agents.")
                else:
                    lines.append(
                        "Engage constructively and let the quality of arguments guide the debate."
                    )

            return "\n".join(lines)

        except Exception as e:
            logger.debug(f"ELO context injection error: {e}")
            return ""

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
            reliability = getattr(snippet, "reliability_score", 0.5)
            if not isinstance(reliability, (int, float)):
                reliability = 0.5

            # Format the snippet
            lines.append(f'{evid_id} "{title}" ({source})')
            lines.append(f"  Reliability: {reliability:.0%}")
            if snippet.url:
                lines.append(f"  URL: {snippet.url}")

            # Include snippet content using RLM external environment pattern
            # Full content is registered externally, summary shown in prompt
            if snippet.snippet:
                if self._rlm_adapter and len(snippet.snippet) > 200:
                    # RLM pattern: register full content, show summary with hint
                    content = self._rlm_adapter.format_for_prompt(
                        content=snippet.snippet,
                        max_chars=200,
                        content_type="evidence",
                        include_hint=self._enable_rlm_hints,
                    )
                else:
                    # Fallback: simple truncation
                    content = snippet.snippet[:200]
                    if len(snippet.snippet) > 200:
                        content += "..."
                lines.append(f"  > {content}")
            lines.append("")  # Blank line between snippets

        lines.append(
            "When stating facts, cite evidence as [EVID-N]. Uncited claims may be challenged."
        )
        return "\n".join(lines)

    def set_evidence_pack(self, evidence_pack: Optional["EvidencePack"]) -> None:
        """Update the evidence pack (called by orchestrator between rounds)."""
        self.evidence_pack = evidence_pack

    def build_proposal_prompt(
        self,
        agent: "Agent",
        audience_section: str = "",
        all_agents: Optional[list["Agent"]] = None,
    ) -> str:
        """Build the initial proposal prompt.

        Args:
            agent: The agent to build the prompt for
            audience_section: Optional pre-formatted audience suggestions section
            all_agents: Optional list of all agents for ELO context injection
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

        # Include historical context if available
        # Prefer RLM abstract for semantic compression over simple truncation
        historical_section = ""
        if self._rlm_context:
            # Use RLM abstract which preserves semantic content
            rlm_abstract = self.get_rlm_abstract(max_chars=800)
            if rlm_abstract:
                historical_section = f"\n\n## Prior Context (Compressed)\n{rlm_abstract}"
                # Add hint about drill-down capability
                rlm_hint = self.get_rlm_context_hint()
                if rlm_hint:
                    historical_section += f"\n{rlm_hint}"
        elif self._historical_context_cache:
            # Fallback to simple truncation if RLM not available
            historical = self._historical_context_cache[:800]
            historical_section = f"\n\n{historical}"

        # Include continuum memory context (cross-debate learnings)
        continuum_section = ""
        continuum_context = self.get_continuum_context()
        if continuum_context:
            continuum_section = f"\n\n{continuum_context}"

        # Include historical belief cruxes (key disagreement points from past debates)
        belief_section = ""
        belief_context = self._inject_belief_context(limit=3)
        if belief_context:
            belief_section = f"\n\n{belief_context}"

        # Include historical dissents and minority views
        # RLM pattern: register full dissent context, show summary in prompt
        dissent_section = ""
        if self.dissent_retriever:
            try:
                dissent_context = self.dissent_retriever.get_debate_preparation_context(
                    topic=self.env.task
                )
                if dissent_context:
                    if self._rlm_adapter and len(dissent_context) > 600:
                        # RLM: full content in external store, summary in prompt
                        formatted = self._rlm_adapter.format_for_prompt(
                            content=dissent_context,
                            max_chars=600,
                            content_type="dissent",
                            include_hint=self._enable_rlm_hints,
                        )
                        dissent_section = f"\n\n## Historical Minority Views\n{formatted}"
                    else:
                        # Fallback: simple truncation
                        dissent_section = f"\n\n## Historical Minority Views\n{dissent_context[:600]}"
            except Exception as e:
                logger.debug(f"Dissent retrieval error: {e}")

        # Include successful patterns from past debates
        patterns_section = ""
        patterns = self.format_successful_patterns(limit=3)
        if patterns:
            patterns_section = f"\n\n{patterns}"

        # Include calibration feedback for poorly calibrated agents
        calibration_section = ""
        calibration_context = self._inject_calibration_context(agent)
        if calibration_context:
            calibration_section = f"\n\n{calibration_context}"

        # Include ELO ranking context for agent awareness of relative expertise (B3)
        elo_section = ""
        if all_agents:
            elo_context = self.get_elo_context(agent, all_agents)
            if elo_context:
                elo_section = f"\n\n{elo_context}"

        # Include evidence citations if available
        evidence_section = ""
        evidence_context = self.format_evidence_for_prompt(max_snippets=5)
        if evidence_context:
            evidence_section = f"\n\n{evidence_context}"

        # Format audience section if provided
        if audience_section:
            audience_section = f"\n\n{audience_section}"

        return f"""You are acting as a {agent.role} in a multi-agent debate (decision stress-test).{stance_section}{role_section}{persona_section}{flip_section}
{historical_section}{continuum_section}{belief_section}{dissent_section}{patterns_section}{calibration_section}{elo_section}{evidence_section}{audience_section}
Task: {self.env.task}{context_str}{research_status}

IMPORTANT: If this task mentions a specific website, company, product, or current topic, you MUST:
1. State what you know vs what you would need to research
2. If research context was provided above, use it. If not, acknowledge the limitation.
3. Do NOT make up facts or speculate about specific entities you don't have verified information about.

Please provide your best proposal to address this task. Be thorough and specific.
Your proposal will be critiqued by other agents, so anticipate potential objections.{self.get_language_constraint()}"""

    def build_revision_prompt(
        self,
        agent: "Agent",
        original: str,
        critiques: list["Critique"],
        audience_section: str = "",
        all_agents: Optional[list["Agent"]] = None,
        round_number: int = 0,
    ) -> str:
        """Build the revision prompt including critiques.

        Args:
            agent: The agent revising their proposal
            original: The original proposal text
            critiques: List of critiques received
            audience_section: Optional pre-formatted audience suggestions section
            all_agents: Optional list of all agents for ELO context injection
            round_number: Current debate round (1-indexed) for phase-specific prompts
        """
        critiques_str = "\n\n".join(c.to_prompt() for c in critiques)
        intensity_guidance = self.get_agreement_intensity_guidance()
        stance_str = self.get_stance_guidance(agent)
        stance_section = f"\n\n{stance_str}" if stance_str else ""

        # Include structured round phase context if using structured phases
        round_phase_section = ""
        if round_number > 0:
            round_phase_context = self.get_round_phase_context(round_number)
            if round_phase_context:
                round_phase_section = f"\n\n{round_phase_context}"

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

        # Include historical belief cruxes (key disagreement points)
        belief_section = ""
        belief_context = self._inject_belief_context(limit=2)
        if belief_context:
            belief_section = f"\n\n{belief_context}"

        # Include calibration feedback for poorly calibrated agents
        calibration_section = ""
        calibration_context = self._inject_calibration_context(agent)
        if calibration_context:
            calibration_section = f"\n\n{calibration_context}"

        # Include ELO ranking context for agent awareness of relative expertise (B3)
        elo_section = ""
        if all_agents:
            elo_context = self.get_elo_context(agent, all_agents)
            if elo_context:
                elo_section = f"\n\n{elo_context}"

        # Include evidence for strengthening revised claims
        evidence_section = ""
        evidence_context = self.format_evidence_for_prompt(max_snippets=3)
        if evidence_context:
            evidence_section = f"\n\n{evidence_context}"

        # Format audience section if provided
        if audience_section:
            audience_section = f"\n\n{audience_section}"

        return f"""You are revising your proposal based on critiques from other agents.{round_phase_section}{role_section}{persona_section}{flip_section}

{intensity_guidance}{stance_section}{patterns_section}{belief_section}{calibration_section}{elo_section}{evidence_section}{audience_section}

Original Task: {self.env.task}

Your Original Proposal:
{original}

Critiques Received:
{critiques_str}

Please provide a revised proposal that addresses the valid critiques.
Use evidence citations [EVID-N] to support strengthened claims.
Explain what you changed and why. If you disagree with a critique, explain your reasoning.{self.get_language_constraint()}"""

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
        critiques_str = "\n".join(f"- {c.agent}: {', '.join(c.issues[:2])}" for c in critiques[:5])

        # Include evidence for final synthesis
        evidence_section = ""
        evidence_context = self.format_evidence_for_prompt(max_snippets=5)
        if evidence_context:
            evidence_section = f"\n\n{evidence_context}\n"

        return f"""You are the synthesizer/judge in a multi-agent debate (decision stress-test).

Task: {task}
{evidence_section}
Proposals:
{proposals_str}

Key Critiques:
{critiques_str}

Synthesize the best elements of all proposals into a final answer.
Reference evidence [EVID-N] to support key claims in your synthesis.
Address the most important critiques raised. Explain your synthesis."""

    def build_judge_vote_prompt(self, candidates: list["Agent"], proposals: dict[str, str]) -> str:
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
