"""
Prompt builder for debate agents.

Extracted from Arena to improve code organization and testability.
Handles construction of prompts for proposals, revisions, and judgments.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
from typing import Any, TYPE_CHECKING, Optional

# Import QuestionClassifier for LLM-based classification
try:
    from aragora.server.question_classifier import QuestionClassifier, QuestionClassification
except ImportError:
    QuestionClassifier: Any = None  # type: ignore[no-redef]
    QuestionClassification: Any = None  # type: ignore[no-redef]

if TYPE_CHECKING:
    from aragora.agents.calibration import CalibrationTracker
    from aragora.agents.flip_detector import FlipDetector
    from aragora.agents.personas import PersonaManager
    from aragora.core import Agent, Critique, Environment
    from aragora.debate.protocol import DebateProtocol
    from aragora.debate.roles import RoleAssignment, RoleRotator
    from aragora.evidence.collector import EvidencePack
    from aragora.knowledge.mound.adapters import SupermemoryAdapter, ContextInjectionResult
    from aragora.memory.consensus import DissentRetriever
    from aragora.memory.continuum import ContinuumMemory
    from aragora.memory.store import CritiqueStore
    from aragora.pulse.types import TrendingTopic
    from aragora.ranking.elo import EloSystem
    from aragora.rlm.types import RLMContext

# Check for RLM availability (use factory for consistent initialization)
try:
    from aragora.rlm import AbstractionLevel, RLMContextAdapter, HAS_OFFICIAL_RLM

    HAS_RLM = True
except ImportError:
    HAS_RLM = False
    HAS_OFFICIAL_RLM = False
    AbstractionLevel: Any = None  # type: ignore[no-redef]
    RLMContextAdapter: Any = None  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


# --- Module-level cached functions for pure computations ---
@functools.lru_cache(maxsize=8)
def _get_stance_guidance_impl(asymmetric_stances: bool, stance: str | None) -> str:
    """Cached implementation of stance guidance generation."""
    if not asymmetric_stances:
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
- Identify weaknesses, flaws, and risks
- Play devil's advocate
- Raise objections and counterarguments
- Stress-test the proposal
Even if you personally agree, argue the negative position."""
    else:
        return """DEBATE STANCE: NEUTRAL
You are assigned to EVALUATE FAIRLY. Your role is to:
- Weigh arguments from both sides impartially
- Identify the strongest and weakest points
- Seek balanced synthesis
- Judge on merit, not position"""


@functools.lru_cache(maxsize=16)
def _get_agreement_intensity_impl(intensity: int | None) -> str:
    """Cached implementation of agreement intensity guidance generation."""
    if intensity is None:
        return ""
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
    else:
        return """Actively seek to incorporate other agents' perspectives. Find value in all
proposals and work toward collaborative synthesis. Prioritize finding agreement
and building on others' ideas."""


@functools.lru_cache(maxsize=128)
def _detect_domain_keywords_impl(question: str) -> str:
    """Cached keyword-based domain detection."""
    import re

    lower = question.lower()

    def word_match(keywords: list[str]) -> bool:
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", lower):
                return True
        return False

    philosophical_kw = [
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
    if word_match(philosophical_kw):
        return "philosophical"
    ethics_kw = [
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
    if word_match(ethics_kw):
        return "ethics"
    technical_kw = [
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
    if word_match(technical_kw):
        return "technical"
    return "general"


@functools.lru_cache(maxsize=32)
def _format_round_phase_impl(
    round_number: int,
    phase_name: str,
    phase_description: str,
    phase_focus: str,
    cognitive_mode: str,
) -> str:
    """Cached implementation of round phase context formatting."""
    lines = [f"## ROUND {round_number}: {phase_name.upper()}"]
    lines.append(f"**Phase Objective:** {phase_description}")
    lines.append(f"**Focus:** {phase_focus}")
    lines.append(f"**Cognitive Mode:** {cognitive_mode}")
    lines.append("")
    mode_guidance = {
        "Analyst": "Analyze thoroughly. Establish facts and key considerations.",
        "Skeptic": "Challenge assumptions. Find weaknesses and edge cases.",
        "Lateral Thinker": "Think creatively. Explore unconventional approaches.",
        "Devil's Advocate": "Argue the opposing view. Surface risks.",
        "Synthesizer": "Integrate insights. Find common ground and build consensus.",
        "Examiner": "Question directly. Clarify remaining disputes.",
        "Researcher": "Gather evidence. Research background context.",
        "Adjudicator": "Render judgment. Weigh all arguments fairly.",
        "Integrator": "Connect the dots. Identify emerging patterns.",
    }
    if cognitive_mode in mode_guidance:
        lines.append(mode_guidance[cognitive_mode])
    return "\n".join(lines)


@functools.lru_cache(maxsize=8)
def _get_language_constraint_impl(enforce: bool, language: str) -> str:
    """Cached implementation of language constraint generation."""
    if not enforce:
        return ""
    return (
        f"\n\n**LANGUAGE REQUIREMENT**: You MUST respond entirely in {language}. "
        f"Do not use any other language. If you need to reference foreign terms, "
        f"provide a {language} translation in parentheses."
    )


def _hash_patterns(patterns: list[dict]) -> str:
    """Create a stable hash key for a list of pattern dicts."""
    serialized = str(
        [
            (p.get("category", ""), p.get("pattern", ""), p.get("occurrences", 0))
            for p in patterns[:5]
        ]
    )
    return hashlib.md5(serialized.encode()).hexdigest()


def clear_all_prompt_caches() -> None:
    """Clear all module-level LRU caches. Call at session boundaries."""
    _get_stance_guidance_impl.cache_clear()
    _get_agreement_intensity_impl.cache_clear()
    _detect_domain_keywords_impl.cache_clear()
    _format_round_phase_impl.cache_clear()
    _get_language_constraint_impl.cache_clear()
    logger.debug("Module-level prompt caches cleared")


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
        supermemory_adapter: Optional["SupermemoryAdapter"] = None,
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
            supermemory_adapter: Optional adapter for external memory context injection
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

        # Trending topics for pulse injection (set externally via set_trending_topics)
        self.trending_topics: list["TrendingTopic"] = []

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

        # Instance-level caches for mutable argument methods
        self._pattern_cache: dict[str, str] = {}
        self._evidence_cache: dict[str, str] = {}
        self._trending_cache: dict[str, str] = {}
        self._cache_max_size: int = 100

        # Supermemory external memory integration (cross-session context)
        self.supermemory_adapter = supermemory_adapter
        self._supermemory_context: Optional["ContextInjectionResult"] = None
        self._supermemory_context_cache: str = ""

    def clear_caches(self) -> None:
        """Clear all caches. Call at session boundaries (e.g., debate end)."""
        self._pattern_cache.clear()
        self._evidence_cache.clear()
        self._trending_cache.clear()
        clear_all_prompt_caches()
        logger.debug("PromptBuilder caches cleared")

    def _evict_cache_if_needed(self, cache: dict) -> None:
        """Evict oldest entries if cache exceeds max size."""
        if len(cache) > self._cache_max_size:
            keys_to_remove = list(cache.keys())[: self._cache_max_size // 2]
            for key in keys_to_remove:
                cache.pop(key, None)

    def format_patterns_for_prompt(self, patterns: list[dict]) -> str:
        """Format learned patterns as prompt context for agents.

        This enables pattern-based learning: agents are warned about
        recurring issues from past debates before they make the same mistakes.

        Results are cached to avoid repeated string operations for identical inputs.

        Args:
            patterns: List of pattern dicts with 'category', 'pattern', 'occurrences'

        Returns:
            Formatted string to inject into debate context
        """
        if not patterns:
            return ""

        # Check cache first
        cache_key = _hash_patterns(patterns)
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key]

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
        result = "\n".join(lines)

        # Cache the result
        self._evict_cache_if_needed(self._pattern_cache)
        self._pattern_cache[cache_key] = result

        return result

    def get_stance_guidance(self, agent: "Agent") -> str:
        """Generate prompt guidance based on agent's debate stance.

        Uses cached implementation for repeated calls with same stance.
        """
        stance = getattr(agent, "stance", None)
        return _get_stance_guidance_impl(self.protocol.asymmetric_stances, stance)

    def get_agreement_intensity_guidance(self) -> str:
        """Generate prompt guidance based on agreement intensity setting.

        Uses cached implementation for repeated calls with same intensity.
        """
        return _get_agreement_intensity_impl(self.protocol.agreement_intensity)

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
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(f"Successful patterns formatting error: {e}")
            return ""
        except Exception as e:
            logger.warning(f"Unexpected patterns formatting error: {e}")
            return ""

    def get_role_context(self, agent: "Agent") -> str:
        """Get cognitive role context for an agent in the current round."""
        if not self.role_rotator or agent.name not in self.current_role_assignments:
            return ""

        assignment = self.current_role_assignments[agent.name]
        return self.role_rotator.format_role_context(assignment)

    def get_round_phase_context(self, round_number: int) -> str:
        """Get structured phase context for the current debate round.

        Uses cached implementation for repeated calls with same phase.

        Args:
            round_number: 1-indexed round number

        Returns:
            Formatted guidance string for the current phase
        """
        phase = self.protocol.get_round_phase(round_number)
        if not phase:
            return ""

        return _format_round_phase_impl(
            round_number,
            phase.name,
            phase.description,
            phase.focus,
            phase.cognitive_mode,
        )

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
                # classify() is now async with AsyncAnthropic client
                self._classification = await self._question_classifier.classify(self.env.task)
                logger.info(
                    f"LLM classification: category={self._classification.category}, "
                    f"confidence={self._classification.confidence:.2f}, "
                    f"personas={self._classification.recommended_personas}"
                )
            else:
                # Use fast keyword-based classification (sync, no API call)
                self._classification = self._question_classifier.classify_simple(self.env.task)
                logger.debug(f"Keyword classification: category={self._classification.category}")

            return self._classification.category

        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            logger.warning(f"Question classification timed out: {e}")
            return self._detect_question_domain_keywords(self.env.task)
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Question classification failed with data error: {e}")
            return self._detect_question_domain_keywords(self.env.task)
        except Exception as e:
            logger.exception(f"Unexpected question classification error: {e}")
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

        Uses cached implementation for repeated calls with same question.

        Returns 'philosophical', 'ethics', 'technical', or 'general'.
        """
        return _detect_domain_keywords_impl(question)

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

        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(f"Flip context formatting error: {e}")
            return ""
        except Exception as e:
            logger.warning(f"Unexpected flip context formatting error: {e}")
            return ""

    def get_continuum_context(self) -> str:
        """Get cached continuum memory context."""
        return self._continuum_context_cache

    async def inject_supermemory_context(
        self,
        debate_topic: str | None = None,
        debate_id: str | None = None,
        container_tag: str | None = None,
        limit: int | None = None,
    ) -> str:
        """Inject context from Supermemory external memory.

        Called at debate start to load relevant context from cross-session
        memory. Results are cached for subsequent prompt builds.

        Args:
            debate_topic: Optional topic to search for (defaults to task)
            debate_id: Optional debate ID for tracking
            container_tag: Optional container filter
            limit: Max items to retrieve

        Returns:
            Formatted context string for prompt injection
        """
        if not self.supermemory_adapter:
            return ""

        try:
            topic = debate_topic or self.env.task
            result = await self.supermemory_adapter.inject_context(
                debate_topic=topic,
                debate_id=debate_id,
                container_tag=container_tag,
                limit=limit,
            )

            self._supermemory_context = result

            if not result.context_content:
                return ""

            # Format as prompt context
            lines = ["## External Memory Context"]
            lines.append("Relevant memories from previous sessions:\n")

            for i, content in enumerate(result.context_content[:5], 1):
                # Truncate long memories
                truncated = content[:400] if len(content) > 400 else content
                if len(content) > 400:
                    truncated += "..."
                lines.append(f"[MEM-{i}] {truncated}")
                lines.append("")

            lines.append(
                f"({result.memories_injected} memories loaded, "
                f"~{result.total_tokens_estimate} tokens)"
            )
            lines.append("Consider these historical insights when formulating your response.")

            self._supermemory_context_cache = "\n".join(lines)
            return self._supermemory_context_cache

        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            logger.warning(f"Supermemory context injection timed out: {e}")
            return ""
        except (AttributeError, TypeError, KeyError) as e:
            logger.debug(f"Supermemory context injection error: {e}")
            return ""
        except Exception as e:
            logger.warning(f"Unexpected supermemory context injection error: {e}")
            return ""

    def get_supermemory_context(self) -> str:
        """Get cached supermemory context for prompt injection.

        Returns:
            Formatted supermemory context, or empty string if not available
        """
        return self._supermemory_context_cache

    def set_supermemory_adapter(self, adapter: Optional["SupermemoryAdapter"]) -> None:
        """Set the supermemory adapter for external memory integration.

        Args:
            adapter: SupermemoryAdapter instance or None to disable
        """
        self.supermemory_adapter = adapter
        # Clear cached context when adapter changes
        self._supermemory_context = None
        self._supermemory_context_cache = ""

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
                len(context.levels) if hasattr(context, "levels") else 0,
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
        if hasattr(self._rlm_context, "levels"):
            for level in self._rlm_context.levels:
                levels_available.append(level.name if hasattr(level, "name") else str(level))

        if not levels_available:
            return ""

        return f"""## HIERARCHICAL CONTEXT AVAILABLE
The debate history has been compressed into multiple abstraction levels
for efficient processing. You are seeing a SUMMARY view.

**Available detail levels:** {", ".join(levels_available)}

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
            if AbstractionLevel and hasattr(self._rlm_context, "get_at_level"):
                abstract = self._rlm_context.get_at_level(AbstractionLevel.ABSTRACT)
                if abstract:
                    return abstract[:max_chars]

                summary = self._rlm_context.get_at_level(AbstractionLevel.SUMMARY)
                if summary:
                    return summary[:max_chars]

            # Fallback to original content truncated
            if hasattr(self._rlm_context, "original_content"):
                return self._rlm_context.original_content[:max_chars] + "..."

        except (AttributeError, TypeError, KeyError) as e:
            logger.debug(f"RLM abstract retrieval error: {e}")
        except Exception as e:
            logger.warning(f"Unexpected RLM abstract retrieval error: {e}")

        return ""

    def get_language_constraint(self) -> str:
        """Get language enforcement instruction for agent prompts.

        Uses cached implementation for repeated calls with same settings.

        Returns:
            Language constraint instruction, or empty string if disabled.
        """
        from aragora.config import DEFAULT_DEBATE_LANGUAGE, ENFORCE_RESPONSE_LANGUAGE

        lang = getattr(self.protocol, "language", None) or DEFAULT_DEBATE_LANGUAGE
        return _get_language_constraint_impl(ENFORCE_RESPONSE_LANGUAGE, lang)

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

        except (AttributeError, TypeError, KeyError) as e:
            logger.debug(f"Belief context injection error: {e}")
            return ""
        except Exception as e:
            logger.warning(f"Unexpected belief context injection error: {e}")
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

        except (AttributeError, TypeError, KeyError) as e:
            logger.debug(f"Calibration context injection error: {e}")
            return ""
        except Exception as e:
            logger.warning(f"Unexpected calibration context injection error: {e}")
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
                    except (AttributeError, TypeError, KeyError) as e:
                        logger.debug(f"Failed to get calibration summary for {name}: {e}")
                    except Exception as e:
                        logger.warning(
                            f"Unexpected error getting calibration summary for {name}: {e}"
                        )

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

        except (AttributeError, TypeError, KeyError) as e:
            logger.debug(f"ELO context injection error: {e}")
            return ""
        except Exception as e:
            logger.warning(f"Unexpected ELO context injection error: {e}")
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
        """Update the evidence pack (called by orchestrator between rounds).

        Clears evidence cache to ensure fresh formatting.
        """
        self.evidence_pack = evidence_pack
        self._evidence_cache.clear()

    def set_trending_topics(self, topics: list["TrendingTopic"]) -> None:
        """Update trending topics for context injection.

        Clears trending cache to ensure fresh formatting.

        Args:
            topics: List of TrendingTopic objects from Pulse system
        """
        self.trending_topics = topics or []
        self._trending_cache.clear()

    def format_trending_for_prompt(self, max_topics: int | None = None) -> str:
        """Format trending topics as context for agent prompts.

        Returns a formatted string with current trending topics that may
        be relevant to the debate task.

        Args:
            max_topics: Maximum number of trending topics to include.
                       Defaults to protocol.trending_injection_max_topics.

        Returns:
            Formatted trending context, or empty string if disabled or no topics
        """
        # Check if trending injection is enabled in protocol
        if not getattr(self.protocol, "enable_trending_injection", False):
            return ""

        if not self.trending_topics:
            return ""

        # Use protocol config for max topics if not specified
        if max_topics is None:
            max_topics = getattr(self.protocol, "trending_injection_max_topics", 3)

        # Check if relevance filtering is enabled
        use_relevance_filter = getattr(self.protocol, "trending_relevance_filter", True)

        if use_relevance_filter:
            # Filter for relevance to task if possible
            task_lower = self.env.task.lower() if self.env else ""
            relevant_topics = []

            for topic in self.trending_topics[: max_topics * 2]:  # Get more for filtering
                # Simple relevance check - topic keywords in task or vice versa
                topic_text = topic.topic.lower() if hasattr(topic, "topic") else str(topic).lower()
                if any(word in task_lower for word in topic_text.split() if len(word) > 3):
                    relevant_topics.append(topic)
                elif len(relevant_topics) < max_topics:
                    relevant_topics.append(topic)

                if len(relevant_topics) >= max_topics:
                    break

            if not relevant_topics:
                relevant_topics = self.trending_topics[:max_topics]
        else:
            # No filtering - just take top N topics
            relevant_topics = self.trending_topics[:max_topics]

        lines = ["## CURRENT TRENDING CONTEXT"]
        lines.append("These topics are currently trending and may provide timely context:\n")

        for topic in relevant_topics:
            topic_name = getattr(topic, "topic", str(topic))
            platform = getattr(topic, "platform", "unknown")
            volume = getattr(topic, "volume", 0)
            category = getattr(topic, "category", "general")

            lines.append(f"- **{topic_name}** ({platform})")
            if volume:
                lines.append(f"  Engagement: {volume:,} | Category: {category}")

        lines.append("")
        lines.append("Consider how current events may relate to the debate topic.")
        return "\n".join(lines)

    def build_proposal_prompt(
        self,
        agent: "Agent",
        audience_section: str = "",
        all_agents: list["Agent"] | None = None,
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

        # Include supermemory external context (cross-session learnings)
        supermemory_section = ""
        supermemory_context = self.get_supermemory_context()
        if supermemory_context:
            supermemory_section = f"\n\n{supermemory_context}"

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
                        dissent_section = (
                            f"\n\n## Historical Minority Views\n{dissent_context[:600]}"
                        )
            except (AttributeError, TypeError, KeyError) as e:
                logger.debug(f"Dissent retrieval error: {e}")
            except Exception as e:
                logger.warning(f"Unexpected dissent retrieval error: {e}")

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

        # Include trending topics for timely context (Pulse integration)
        trending_section = ""
        trending_context = self.format_trending_for_prompt(max_topics=3)
        if trending_context:
            trending_section = f"\n\n{trending_context}"

        # Format audience section if provided
        if audience_section:
            audience_section = f"\n\n{audience_section}"

        return f"""You are acting as a {agent.role} in a multi-agent debate (decision stress-test).{stance_section}{role_section}{persona_section}{flip_section}
{historical_section}{continuum_section}{supermemory_section}{belief_section}{dissent_section}{patterns_section}{calibration_section}{elo_section}{evidence_section}{trending_section}{audience_section}
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
        all_agents: list["Agent"] | None = None,
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

        # Include trending topics for timely context (Pulse integration)
        trending_section = ""
        trending_context = self.format_trending_for_prompt(max_topics=2)
        if trending_context:
            trending_section = f"\n\n{trending_context}"

        # Format audience section if provided
        if audience_section:
            audience_section = f"\n\n{audience_section}"

        return f"""You are revising your proposal based on critiques from other agents.{round_phase_section}{role_section}{persona_section}{flip_section}

{intensity_guidance}{stance_section}{patterns_section}{belief_section}{calibration_section}{elo_section}{evidence_section}{trending_section}{audience_section}

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
