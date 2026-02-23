"""
Context provider mixin for PromptBuilder.

Provides methods that gather and format various context sections
for injection into debate prompts: personas, evidence, calibration,
ELO rankings, belief cruxes, trending topics, RLM, supermemory, etc.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.core import Agent
    from aragora.evidence.collector import EvidencePack
    from aragora.knowledge.mound.adapters import SupermemoryAdapter
    from aragora.pulse.types import TrendingTopic
    from aragora.rlm.types import RLMContext

# Check for RLM availability
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

# Check for QuestionClassifier
QuestionClassifier: Any
QuestionClassification: Any

try:
    from aragora.server.question_classifier import QuestionClassifier, QuestionClassification
except ImportError:
    QuestionClassifier = None
    QuestionClassification = None

logger = logging.getLogger(__name__)


class PromptContextMixin:
    """Mixin providing context gathering methods for prompt construction."""

    # These attributes are defined in the main PromptBuilder class
    protocol: Any
    env: Any
    memory: Any
    continuum_memory: Any
    dissent_retriever: Any
    role_rotator: Any
    persona_manager: Any
    flip_detector: Any
    evidence_pack: Any
    calibration_tracker: Any
    elo_system: Any
    domain: str
    trending_topics: list
    current_role_assignments: dict
    _historical_context_cache: str
    _continuum_context_cache: str
    _classification: Any
    _question_classifier: Any
    _rlm_context: Any
    _enable_rlm_hints: bool
    _rlm_adapter: Any
    _pattern_cache: dict
    _evidence_cache: dict
    _trending_cache: dict
    _cache_max_size: int
    supermemory_adapter: Any
    _supermemory_context: Any
    _supermemory_context_cache: str
    _evict_cache_if_needed: Any
    claims_kernel: Any
    include_prior_claims: bool
    _pulse_topics: list
    _knowledge_context: str
    _km_item_ids: list
    _outcome_context: str

    def get_deliberation_template_context(self) -> str:
        """Get deliberation template context for prompt injection.

        Returns formatted template context string, or empty string if no template
        is configured or the template cannot be found.
        """
        template_name = getattr(self.protocol, "deliberation_template", None)
        if not template_name:
            return ""

        try:
            from aragora.deliberation.templates.registry import get_template

            template = get_template(template_name)
            if template is None:
                return ""

            lines = [f"## DELIBERATION TEMPLATE: {template.name}"]
            lines.append(
                f"Category: {template.category.value if hasattr(template.category, 'value') else template.category}"
            )
            lines.append(f"Description: {template.description}")

            if template.system_prompt_additions:
                lines.append(f"\nGuidance: {template.system_prompt_additions}")

            if template.personas:
                lines.append("\nAssigned Personas:")
                for persona in template.personas:
                    lines.append(f"- {persona}")

            return "\n".join(lines)
        except (ImportError, AttributeError, TypeError):
            return ""

    def format_patterns_for_prompt(self, patterns: list[dict]) -> str:
        """Format learned patterns as prompt context for agents.

        Results are cached to avoid repeated string operations for identical inputs.
        """
        from .prompt_builder import _hash_patterns

        if not patterns:
            return ""

        cache_key = _hash_patterns(patterns)
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key]

        lines = ["## LEARNED PATTERNS (From Previous Debates)"]
        lines.append("Be especially careful about these recurring issues:\n")

        for p in patterns[:5]:
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

        self._evict_cache_if_needed(self._pattern_cache)
        self._pattern_cache[cache_key] = result

        return result

    def get_stance_guidance(self, agent: Agent) -> str:
        """Generate prompt guidance based on agent's debate stance."""
        from .prompt_builder import _get_stance_guidance_impl

        stance = getattr(agent, "stance", None)
        return _get_stance_guidance_impl(self.protocol.asymmetric_stances, stance)

    def get_agreement_intensity_guidance(self) -> str:
        """Generate prompt guidance based on agreement intensity setting."""
        from .prompt_builder import _get_agreement_intensity_impl

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
            logger.debug("Successful patterns formatting error: %s", e)
            return ""
        except (RuntimeError, KeyError) as e:
            logger.warning("Unexpected patterns formatting error: %s", e)
            return ""

    def get_role_context(self, agent: Agent) -> str:
        """Get cognitive role context for an agent in the current round."""
        if not self.role_rotator or agent.name not in self.current_role_assignments:
            return ""

        assignment = self.current_role_assignments[agent.name]
        return self.role_rotator.format_role_context(assignment)

    def get_round_phase_context(self, round_number: int) -> str:
        """Get structured phase context for the current debate round."""
        from .prompt_builder import _format_round_phase_impl

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
        """Classify the debate question using LLM (async)."""
        if self._classification is not None:
            return self._classification.category

        if QuestionClassifier is None:
            logger.debug("QuestionClassifier not available, using keyword fallback")
            return self._detect_question_domain_keywords(self.env.task)

        try:
            if self._question_classifier is None:
                self._question_classifier = QuestionClassifier()

            if use_llm:
                self._classification = await self._question_classifier.classify(self.env.task)
                logger.info(
                    f"LLM classification: category={self._classification.category}, "
                    f"confidence={self._classification.confidence:.2f}, "
                    f"personas={self._classification.recommended_personas}"
                )
            else:
                self._classification = self._question_classifier.classify_simple(self.env.task)
                logger.debug("Keyword classification: category=%s", self._classification.category)

            return self._classification.category

        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            logger.warning("Question classification timed out: %s", e)
            return self._detect_question_domain_keywords(self.env.task)
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning("Question classification failed with data error: %s", e)
            return self._detect_question_domain_keywords(self.env.task)
        except (RuntimeError, KeyError, OSError, ConnectionError) as e:
            logger.exception("Unexpected question classification error: %s", e)
            return self._detect_question_domain_keywords(self.env.task)
        except Exception as e:
            logger.warning("Question classification failed (API or other error): %s", e)
            return self._detect_question_domain_keywords(self.env.task)

    def _detect_question_domain(self, question: str) -> str:
        """Detect question domain for persona selection."""
        if self._classification is not None:
            category = self._classification.category
            if category in ("ethical", "philosophical"):
                return "philosophical"
            elif category == "technical":
                return "technical"
            elif category in ("legal", "security", "financial", "healthcare", "scientific"):
                return "technical"
            else:
                return "general"

        return self._detect_question_domain_keywords(question)

    def _detect_question_domain_keywords(self, question: str) -> str:
        """Keyword-based domain detection (fallback when LLM unavailable)."""
        from .prompt_builder import _detect_domain_keywords_impl

        return _detect_domain_keywords_impl(question)

    def get_persona_context(self, agent: Agent) -> str:
        """Get persona context for agent specialization."""
        question_domain = self._detect_question_domain(self.env.task)

        if question_domain == "philosophical":
            return (
                "Approach this question as a thoughtful observer of the human condition. "
                "Draw on wisdom traditions, philosophy, psychology, and lived experience. "
                "Avoid framing your answer in technical or software metaphors. "
                "Focus on what makes life meaningful, purposeful, and fulfilling."
            )

        if question_domain == "ethics":
            return (
                "Approach this as an ethical question requiring nuanced moral reasoning. "
                "Consider multiple ethical frameworks, stakeholder perspectives, and real-world consequences. "
                "Acknowledge complexity and avoid reductive technical framings."
            )

        if question_domain == "general":
            return (
                "Approach this as a thoughtful, experienced, and friendly advisor. "
                "Draw on broad knowledge, practical wisdom, and common sense. "
                "Be clear, helpful, and accessible. Avoid technical jargon unless the "
                "question specifically calls for it."
            )

        if not self.persona_manager:
            return ""

        persona = self.persona_manager.get_persona(agent.name)
        if not persona:
            agent_type = agent.name.split("_")[0].lower()
            from aragora.agents.personas import DEFAULT_PERSONAS

            if agent_type in DEFAULT_PERSONAS:
                persona = DEFAULT_PERSONAS[agent_type]
            else:
                return ""

        return persona.to_prompt_context()

    def get_flip_context(self, agent: Agent) -> str:
        """Get flip/consistency context for agent self-awareness."""
        if not self.flip_detector:
            return ""

        try:
            consistency = self.flip_detector.get_agent_consistency(agent.name)

            if consistency.total_positions == 0:
                return ""
            if consistency.total_flips == 0:
                return ""

            lines = ["## Position Consistency Note"]

            if consistency.contradictions > 0:
                lines.append(
                    f"You have {consistency.contradictions} prior position contradiction(s) on record. "
                    "Consider your stance carefully before arguing against positions you previously held."
                )

            if consistency.retractions > 0:
                lines.append(
                    f"You have retracted {consistency.retractions} previous position(s). "
                    "If changing positions again, clearly explain your reasoning."
                )

            score = consistency.consistency_score
            if score < 0.7:
                lines.append(
                    f"Your consistency score is {score:.0%}. Prioritize coherent positions."
                )

            if consistency.domains_with_flips:
                domains = ", ".join(consistency.domains_with_flips[:3])
                lines.append(f"Domains with position changes: {domains}")

            return "\n".join(lines) if len(lines) > 1 else ""

        except (AttributeError, TypeError, ValueError) as e:
            logger.debug("Flip context formatting error: %s", e)
            return ""
        except (RuntimeError, KeyError) as e:
            logger.warning("Unexpected flip context formatting error: %s", e)
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
        """Inject context from Supermemory external memory."""
        if not self.supermemory_adapter:
            return ""
        if self._supermemory_context_cache:
            return self._supermemory_context_cache

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

            lines = ["## External Memory Context"]
            lines.append("Relevant memories from previous sessions:\n")

            for i, content in enumerate(result.context_content[:5], 1):
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
            logger.warning("Supermemory context injection timed out: %s", e)
            return ""
        except (AttributeError, TypeError, KeyError) as e:
            logger.debug("Supermemory context injection error: %s", e)
            return ""
        except (RuntimeError, ValueError, OSError, ConnectionError) as e:
            logger.warning("Unexpected supermemory context injection error: %s", e)
            return ""

    def get_supermemory_context(self) -> str:
        """Get cached supermemory context for prompt injection."""
        return self._supermemory_context_cache

    def get_knowledge_mound_context(self) -> str:
        """Get structured Knowledge Mound context for prompt injection.

        Returns the organizational knowledge context that was set via
        set_knowledge_context(). This is injected as a dedicated prompt
        section (parallel to supermemory) rather than appended to env.context.

        Returns:
            Formatted knowledge context string, or empty string if not set.
        """
        return self._knowledge_context

    def set_knowledge_context(
        self, context: str, item_ids: list[str] | None = None
    ) -> None:
        """Set structured Knowledge Mound context for prompt injection.

        This allows the context initializer to provide KM content as a
        dedicated prompt section rather than appending to env.context.
        The content will be injected with a clear "Organizational Knowledge"
        header so agents know it represents institutional memory.

        Args:
            context: The knowledge context string to inject into prompts.
            item_ids: Optional list of KM item IDs used (for outcome tracking).
        """
        self._knowledge_context = context or ""
        if item_ids is not None:
            self._km_item_ids = list(item_ids)

    def get_outcome_context(self) -> str:
        """Get past decision outcome context for prompt injection.

        Returns the outcome context that was set via set_outcome_context().
        This is injected as a dedicated prompt section so agents see past
        decision successes and failures alongside other KM context.

        Returns:
            Formatted outcome context string, or empty string if not set.
        """
        return self._outcome_context

    def set_outcome_context(self, context: str) -> None:
        """Set past decision outcome context for prompt injection.

        This allows the context initializer to provide outcome data as a
        dedicated prompt section. Agents will see a "Past Decision Outcomes"
        header with outcome types, impact scores, and lessons learned.

        Args:
            context: The outcome context string to inject into prompts.
        """
        self._outcome_context = context or ""

    def get_codebase_context(self) -> str:
        """Get codebase context for code-grounded debate prompt injection.

        Returns the codebase structure context that was set via
        set_codebase_context(). This is injected as a dedicated prompt
        section so agents can reference actual code during debates.

        Returns:
            Formatted codebase context string, or empty string if not set.
        """
        return self._codebase_context

    def set_codebase_context(self, context: str) -> None:
        """Set codebase context for code-grounded debate prompt injection.

        This allows the context initializer to provide codebase structure
        as a dedicated prompt section. Agents will see a "Codebase Context"
        header with file paths, symbols, and dependency information.

        Args:
            context: The codebase context string to inject into prompts.
        """
        self._codebase_context = context or ""

    def get_prior_claims_context(self, limit: int = 5) -> str:
        """Get prior claims related to the current topic for context injection.

        Queries the ClaimsKernel for claims matching the debate topic
        and formats them as a prompt section.

        Args:
            limit: Maximum number of prior claims to include

        Returns:
            Formatted string with prior claims context, or empty string
        """
        if not self.include_prior_claims or not self.claims_kernel:
            return ""

        try:
            topic = self.env.task if hasattr(self.env, "task") else ""
            if not topic:
                return ""

            related = self.claims_kernel.get_related_claims(topic, limit=limit)
            if not related:
                return ""

            lines = ["## PRIOR CLAIMS (From Previous Debates)"]
            lines.append("Consider these established positions when formulating your response:\n")

            for claim in related:
                type_label = claim.claim_type.value.upper()
                author = claim.author
                confidence = f"{claim.adjusted_confidence:.0%}"
                status = claim.status
                statement = claim.statement[:150]
                if len(claim.statement) > 150:
                    statement += "..."
                lines.append(
                    f"- [{type_label}] **{author}** ({confidence} confidence, {status}): "
                    f"{statement}"
                )

            return "\n".join(lines)
        except (AttributeError, TypeError, KeyError) as e:
            logger.debug("Prior claims context error: %s", e)
            return ""

    def set_supermemory_adapter(self, adapter: SupermemoryAdapter | None) -> None:
        """Set the supermemory adapter for external memory integration."""
        self.supermemory_adapter = adapter
        self._supermemory_context = None
        self._supermemory_context_cache = ""

    def set_rlm_context(self, context: RLMContext | None) -> None:
        """Set hierarchical RLM context for drill-down access."""
        self._rlm_context = context
        if context:
            logger.debug(
                "[rlm] Set hierarchical context with %d levels",
                len(context.levels) if hasattr(context, "levels") else 0,
            )

    def get_rlm_context_hint(self) -> str:
        """Get RLM context hint for agent prompts."""
        if not self._enable_rlm_hints or not self._rlm_context:
            return ""

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
        """Get abstract-level summary from RLM context."""
        if not self._rlm_context:
            return ""

        try:
            if AbstractionLevel and hasattr(self._rlm_context, "get_at_level"):
                abstract = self._rlm_context.get_at_level(AbstractionLevel.ABSTRACT)
                if abstract:
                    return abstract[:max_chars]

                summary = self._rlm_context.get_at_level(AbstractionLevel.SUMMARY)
                if summary:
                    return summary[:max_chars]

            if hasattr(self._rlm_context, "original_content"):
                return self._rlm_context.original_content[:max_chars] + "..."

        except (AttributeError, TypeError, KeyError) as e:
            logger.debug("RLM abstract retrieval error: %s", e)
        except (RuntimeError, ValueError) as e:
            logger.warning("Unexpected RLM abstract retrieval error: %s", e)

        return ""

    def get_language_constraint(self) -> str:
        """Get language enforcement instruction for agent prompts."""
        from aragora.config import DEFAULT_DEBATE_LANGUAGE, ENFORCE_RESPONSE_LANGUAGE
        from .prompt_builder import _get_language_constraint_impl

        lang = getattr(self.protocol, "language", None) or DEFAULT_DEBATE_LANGUAGE
        return _get_language_constraint_impl(ENFORCE_RESPONSE_LANGUAGE, lang)

    def _inject_belief_context(self, limit: int = 3) -> str:
        """Retrieve and format historical belief cruxes for prompt injection."""
        if not self.continuum_memory:
            return ""

        try:
            memories = self.continuum_memory.retrieve(
                query=self.env.task,
                limit=limit,
            )

            if not memories:
                return ""

            all_cruxes: list[str] = []
            for mem in memories:
                metadata = getattr(mem, "metadata", {}) or {}
                cruxes = metadata.get("crux_claims", [])
                if isinstance(cruxes, list):
                    all_cruxes.extend(str(c) for c in cruxes if c)

            unique_cruxes = list(dict.fromkeys(all_cruxes))[:5]

            if not unique_cruxes:
                return ""

            lines = ["## Historical Disagreement Points"]
            lines.append(
                "Past debates on similar topics identified these key points of contention:"
            )
            for crux in unique_cruxes:
                lines.append(f"- {crux}")
            lines.append("\nConsider whether your proposal addresses these concerns.")

            return "\n".join(lines)

        except (AttributeError, TypeError, KeyError) as e:
            logger.debug("Belief context injection error: %s", e)
            return ""
        except (RuntimeError, OSError) as e:
            logger.warning("Unexpected belief context injection error: %s", e)
            return ""

    def _inject_calibration_context(self, agent: Agent) -> str:
        """Inject calibration feedback into agent prompts."""
        if not self.calibration_tracker:
            return ""

        try:
            summary = self.calibration_tracker.get_calibration_summary(agent.name)

            if summary.total_predictions < 5:
                return ""

            brier = summary.brier_score
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
            logger.debug("Calibration context injection error: %s", e)
            return ""
        except (RuntimeError, OSError) as e:
            logger.warning("Unexpected calibration context injection error: %s", e)
            return ""

    def get_elo_context(self, agent: Agent, all_agents: list[Agent]) -> str:
        """Inject ELO ranking context for agent awareness of relative expertise."""
        if not self.elo_system:
            return ""

        try:
            agent_names = [a.name for a in all_agents]
            ratings_batch = self.elo_system.get_ratings_batch(agent_names)
            if not ratings_batch:
                return ""

            domain_suffix = ""
            if self.domain and self.domain != "general":
                domain_suffix = f" ({self.domain})"

            lines = [f"## Agent Rankings{domain_suffix}"]
            lines.append("Consider these rankings when weighing arguments:\n")

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
                marker = " (you)" if name == agent.name else ""

                calib_str = ""
                if self.calibration_tracker:
                    try:
                        summary = self.calibration_tracker.get_calibration_summary(name)
                        if summary.total_predictions >= 5:
                            accuracy = 1.0 - summary.brier_score
                            calib_str = f", {accuracy:.0%} calibration"
                    except (AttributeError, TypeError, KeyError) as e:
                        logger.debug("Failed to get calibration summary for %s: %s", name, e)
                    except (RuntimeError, OSError) as e:
                        logger.warning(
                            "Unexpected error getting calibration summary for %s: %s", name, e
                        )

                lines.append(
                    f"  {rank}. {name}: {elo:.0f} ELO ({total} debates{calib_str}){marker}"
                )

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
            logger.debug("ELO context injection error: %s", e)
            return ""
        except (RuntimeError, OSError) as e:
            logger.warning("Unexpected ELO context injection error: %s", e)
            return ""

    def format_evidence_for_prompt(self, max_snippets: int = 5) -> str:
        """Format evidence pack as citable references for agent prompts."""
        if not self.evidence_pack or not self.evidence_pack.snippets:
            return ""

        lines = ["## AVAILABLE EVIDENCE"]
        lines.append("Reference these sources by ID when making factual claims:\n")

        for i, snippet in enumerate(self.evidence_pack.snippets[:max_snippets], 1):
            evid_id = f"[EVID-{i}]"
            title = snippet.title[:80] if snippet.title else "Untitled"
            source = snippet.source or "Unknown"
            reliability = getattr(snippet, "reliability_score", 0.5)
            if not isinstance(reliability, (int, float)):
                reliability = 0.5

            lines.append(f'{evid_id} "{title}" ({source})')
            lines.append(f"  Reliability: {reliability:.0%}")
            if snippet.url:
                lines.append(f"  URL: {snippet.url}")

            if snippet.snippet:
                if self._rlm_adapter and len(snippet.snippet) > 200:
                    content = self._rlm_adapter.format_for_prompt(
                        content=snippet.snippet,
                        max_chars=200,
                        content_type="evidence",
                        include_hint=self._enable_rlm_hints,
                    )
                else:
                    content = snippet.snippet[:200]
                    if len(snippet.snippet) > 200:
                        content += "..."
                lines.append(f"  > {content}")
            lines.append("")

        lines.append(
            "When stating facts, cite evidence as [EVID-N]. Uncited claims may be challenged."
        )
        return "\n".join(lines)

    def set_evidence_pack(self, evidence_pack: EvidencePack | None) -> None:
        """Update the evidence pack (called by orchestrator between rounds)."""
        self.evidence_pack = evidence_pack
        self._evidence_cache.clear()

    def set_trending_topics(self, topics: list[TrendingTopic]) -> None:
        """Update trending topics for context injection."""
        self.trending_topics = topics or []
        self._trending_cache.clear()

    def format_trending_for_prompt(self, max_topics: int | None = None) -> str:
        """Format trending topics as context for agent prompts."""
        if not getattr(self.protocol, "enable_trending_injection", False):
            return ""

        if not self.trending_topics:
            return ""

        if max_topics is None:
            max_topics = getattr(self.protocol, "trending_injection_max_topics", 3)

        use_relevance_filter = getattr(self.protocol, "trending_relevance_filter", True)

        if use_relevance_filter:
            task_lower = self.env.task.lower() if self.env else ""
            relevant_topics = []

            for topic in self.trending_topics[: max_topics * 2]:
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

    def format_pulse_context(self, max_topics: int = 5) -> str:
        """Format Pulse trending topics with source, velocity, and recency.

        Unlike format_trending_for_prompt which uses TrendingTopic objects,
        this method works with enriched pulse data dicts that include
        recency (hours_ago) information from the ScheduledDebateStore.

        Args:
            max_topics: Maximum number of pulse topics to include

        Returns:
            Formatted string with pulse context, or empty string
        """
        if not self._pulse_topics:
            return ""

        topics = self._pulse_topics[:max_topics]
        if not topics:
            return ""

        lines = ["## PULSE: TRENDING CONTEXT"]
        lines.append("Real-time signals from monitored sources:\n")

        for topic in topics:
            name = topic.get("topic", "")
            platform = topic.get("platform", "unknown")
            volume = topic.get("volume", 0)
            category = topic.get("category", "")
            hours_ago = topic.get("hours_ago", 0.0)

            velocity_label = ""
            if volume >= 10000:
                velocity_label = " [HIGH VELOCITY]"
            elif volume >= 1000:
                velocity_label = " [RISING]"

            recency = f"{hours_ago:.1f}h ago" if hours_ago else "recent"

            lines.append(f"- **{name}**{velocity_label}")
            detail_parts = [f"Source: {platform}"]
            if volume:
                detail_parts.append(f"Engagement: {volume:,}")
            if category:
                detail_parts.append(f"Category: {category}")
            detail_parts.append(f"Age: {recency}")
            lines.append(f"  {' | '.join(detail_parts)}")

        lines.append("")
        lines.append("Factor in these signals if relevant to the decision at hand.")
        return "\n".join(lines)

    def set_pulse_topics(self, topics: list[dict]) -> None:
        """Set pulse topics for context injection.

        Args:
            topics: List of dicts with topic, platform, volume, category, hours_ago
        """
        self._pulse_topics = topics
