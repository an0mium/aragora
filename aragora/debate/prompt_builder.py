"""
Prompt builder for debate agents.

Extracted from Arena to improve code organization and testability.
Handles construction of prompts for proposals, revisions, and judgments.

The implementation is split across mixin classes:
- PromptContextMixin (prompt_context_providers.py) - Context gathering methods
- PromptAssemblyMixin (prompt_assemblers.py) - Prompt assembly (build_*) methods
"""

from __future__ import annotations

import functools
import hashlib
import logging
from typing import Any, TYPE_CHECKING

from aragora.debate.context_budgeter import ContextBudgeter, ContextSection
from .prompt_assemblers import PromptAssemblyMixin
from .prompt_context_providers import PromptContextMixin

if TYPE_CHECKING:
    from aragora.agents.calibration import CalibrationTracker
    from aragora.agents.flip_detector import FlipDetector
    from aragora.agents.personas import PersonaManager
    from aragora.core import Environment
    from aragora.debate.protocol import DebateProtocol
    from aragora.debate.roles import RoleAssignment, RoleRotator
    from aragora.evidence.collector import EvidencePack
    from aragora.knowledge.mound.adapters import SupermemoryAdapter, ContextInjectionResult
    from aragora.memory.consensus import DissentRetriever
    from aragora.memory.continuum import ContinuumMemory
    from aragora.memory.store import CritiqueStore
    from aragora.pulse.types import TrendingTopic
    from aragora.ranking.elo import EloSystem
    from aragora.server.question_classifier import QuestionClassification, QuestionClassifier
    from aragora.rlm.types import RLMContext

# Check for RLM availability (use factory for consistent initialization)
AbstractionLevel: Any
RLMContextAdapter: Any

try:
    from aragora.rlm import AbstractionLevel, RLMContextAdapter, HAS_OFFICIAL_RLM

    HAS_RLM = True
except ImportError:
    HAS_RLM = False
    HAS_OFFICIAL_RLM = False
    AbstractionLevel = None
    RLMContextAdapter = None

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
    return hashlib.md5(serialized.encode(), usedforsecurity=False).hexdigest()


def clear_all_prompt_caches() -> None:
    """Clear all module-level LRU caches. Call at session boundaries."""
    _get_stance_guidance_impl.cache_clear()
    _get_agreement_intensity_impl.cache_clear()
    _detect_domain_keywords_impl.cache_clear()
    _format_round_phase_impl.cache_clear()
    _get_language_constraint_impl.cache_clear()
    logger.debug("Module-level prompt caches cleared")


class PromptBuilder(PromptContextMixin, PromptAssemblyMixin):
    """Builds prompts for debate agents.

    Encapsulates all prompt construction logic including context injection
    for personas, roles, patterns, and historical data.
    """

    def __init__(
        self,
        protocol: DebateProtocol,
        env: Environment,
        memory: CritiqueStore | None = None,
        continuum_memory: ContinuumMemory | None = None,
        dissent_retriever: DissentRetriever | None = None,
        role_rotator: RoleRotator | None = None,
        persona_manager: PersonaManager | None = None,
        flip_detector: FlipDetector | None = None,
        evidence_pack: EvidencePack | None = None,
        calibration_tracker: CalibrationTracker | None = None,
        elo_system: EloSystem | None = None,
        domain: str = "general",
        supermemory_adapter: SupermemoryAdapter | None = None,
        claims_kernel: Any | None = None,
        include_prior_claims: bool = False,
        enable_introspection: bool = True,
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

        # Prior claims kernel for injecting related claims from previous debates
        self.claims_kernel = claims_kernel
        self.include_prior_claims = include_prior_claims

        # Agent introspection (self-awareness of reputation, expertise)
        self.enable_introspection = enable_introspection

        # Trending topics for pulse injection (set externally via set_trending_topics)
        self.trending_topics: list[TrendingTopic] = []

        # Pulse topics for enhanced context (set via set_pulse_topics)
        self._pulse_topics: list[dict] = []

        # Current state (set externally by Arena)
        self.current_role_assignments: dict[str, RoleAssignment] = {}
        self._historical_context_cache: str = ""
        self._continuum_context_cache: str = ""
        self.user_suggestions: list = []

        # Question classification cache (populated by classify_question_async)
        self._classification: QuestionClassification | None = None
        self._question_classifier: QuestionClassifier | None = None

        # RLM hierarchical context (set by ContextInitializer when enabled)
        self._rlm_context: RLMContext | None = None
        self._enable_rlm_hints: bool = HAS_RLM  # Show agents how to query context

        # RLM context adapter for external environment pattern
        self._rlm_adapter: RLMContextAdapter | None = None
        if HAS_RLM and RLMContextAdapter is not None:
            self._rlm_adapter = RLMContextAdapter()

        # Instance-level caches for mutable argument methods
        self._pattern_cache: dict[str, str] = {}
        self._evidence_cache: dict[str, str] = {}
        self._trending_cache: dict[str, str] = {}
        self._cache_max_size: int = 100

        # Context budgeter for mixed input types
        self._context_budgeter = ContextBudgeter()

        # Supermemory external memory integration (cross-session context)
        self.supermemory_adapter = supermemory_adapter
        self._supermemory_context: ContextInjectionResult | None = None
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

    def _get_introspection_context(self, agent_name: str) -> str:
        """Get introspection context for an agent.

        Returns formatted self-awareness section (reputation, expertise)
        for injection into agent prompts, or empty string on error.
        """
        if not self.enable_introspection:
            return ""
        try:
            from aragora.introspection.api import (
                get_agent_introspection,
                format_introspection_section,
            )

            snapshot = get_agent_introspection(
                agent_name,
                memory=self.memory,
                persona_manager=self.persona_manager,
            )
            section = format_introspection_section(snapshot, max_chars=600)
            return section
        except Exception:
            return ""

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _apply_context_budget(
        self,
        *,
        env_context: str,
        sections: list[ContextSection],
    ) -> tuple[str, str]:
        """Apply context budgets and return (context_block, env_context_str)."""
        budgeter = self._context_budgeter

        trimmed_env = env_context
        remaining = budgeter.total_tokens
        if trimmed_env:
            trimmed_env = budgeter.truncate_section("env_context", trimmed_env)
            remaining = max(remaining - self._estimate_tokens(trimmed_env), 0)

        budgeted_sections = budgeter.apply(sections, total_tokens=remaining)
        context_block = "\n\n".join(
            section.content for section in budgeted_sections if section.content
        )
        if context_block:
            context_block = f"\n\n{context_block}"

        if trimmed_env:
            trimmed_env = f"\n\n{trimmed_env}"

        return context_block, trimmed_env
