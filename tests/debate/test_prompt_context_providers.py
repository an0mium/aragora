"""
Tests for PromptContextMixin in aragora/debate/prompt_context_providers.py.

Covers all 26 method groups with ~65+ tests including happy paths,
edge cases, error handling, caching, and timeout behavior.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.prompt_context_providers import PromptContextMixin


# ---------------------------------------------------------------------------
# Concrete test harness: inherits the mixin and provides all required attrs
# ---------------------------------------------------------------------------

class ConcreteContextProvider(PromptContextMixin):
    """Concrete class that wires up all attributes the mixin expects."""

    def __init__(self):
        self.protocol = MagicMock()
        self.env = MagicMock()
        self.env.task = "Design a rate limiter for API endpoints"
        self.memory = None
        self.continuum_memory = None
        self.dissent_retriever = None
        self.role_rotator = None
        self.persona_manager = None
        self.flip_detector = None
        self.evidence_pack = None
        self.calibration_tracker = None
        self.elo_system = None
        self.domain = "general"
        self.trending_topics = []
        self.current_role_assignments = {}
        self._historical_context_cache = ""
        self._continuum_context_cache = ""
        self._classification = None
        self._question_classifier = None
        self._rlm_context = None
        self._enable_rlm_hints = False
        self._rlm_adapter = None
        self._pattern_cache = {}
        self._evidence_cache = {}
        self._trending_cache = {}
        self._cache_max_size = 100
        self.supermemory_adapter = None
        self._supermemory_context = None
        self._supermemory_context_cache = ""
        self.claims_kernel = None
        self.include_prior_claims = False
        self._pulse_topics = []
        self._knowledge_context = ""
        self._km_item_ids = []
        self._codebase_context = ""

    def _evict_cache_if_needed(self, cache):
        if len(cache) > self._cache_max_size:
            oldest = next(iter(cache))
            del cache[oldest]


# ---------------------------------------------------------------------------
# Helper mock classes
# ---------------------------------------------------------------------------

class MockAgent:
    def __init__(self, name: str = "claude_proposer", stance: str | None = None):
        self.name = name
        self.stance = stance


class MockConsistency:
    def __init__(
        self,
        total_positions=10,
        total_flips=2,
        contradictions=1,
        retractions=1,
        consistency_score=0.8,
        domains_with_flips=None,
    ):
        self.total_positions = total_positions
        self.total_flips = total_flips
        self.contradictions = contradictions
        self.retractions = retractions
        self.consistency_score = consistency_score
        self.domains_with_flips = domains_with_flips or []


class MockPattern:
    def __init__(self, issue_text, suggestion_text, issue_type="logic", success_count=3):
        self.issue_text = issue_text
        self.suggestion_text = suggestion_text
        self.issue_type = issue_type
        self.success_count = success_count


class MockCalibrationSummary:
    def __init__(
        self,
        total_predictions=10,
        brier_score=0.3,
        is_overconfident=False,
        is_underconfident=False,
    ):
        self.total_predictions = total_predictions
        self.brier_score = brier_score
        self.is_overconfident = is_overconfident
        self.is_underconfident = is_underconfident


class MockRating:
    def __init__(self, elo=1500, wins=5, losses=3):
        self.elo = elo
        self.wins = wins
        self.losses = losses


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provider():
    return ConcreteContextProvider()


@pytest.fixture
def agent():
    return MockAgent()


# ===================================================================
# 1. get_deliberation_template_context
# ===================================================================

class TestGetDeliberationTemplateContext:
    def test_no_template_configured(self, provider):
        """Returns empty when protocol has no deliberation_template."""
        provider.protocol.deliberation_template = None
        assert provider.get_deliberation_template_context() == ""

    def test_template_found(self, provider):
        """Returns formatted template context when found."""
        provider.protocol.deliberation_template = "socratic"
        mock_template = MagicMock()
        mock_template.name = "Socratic Method"
        mock_template.category.value = "philosophical"
        mock_template.description = "Guided questioning"
        mock_template.system_prompt_additions = "Ask probing questions"
        mock_template.personas = ["Questioner", "Responder"]

        with patch(
            "aragora.deliberation.templates.registry.get_template",
            return_value=mock_template,
        ):
            result = provider.get_deliberation_template_context()

        assert "DELIBERATION TEMPLATE: Socratic Method" in result
        assert "philosophical" in result
        assert "Guided questioning" in result
        assert "Ask probing questions" in result
        assert "Questioner" in result
        assert "Responder" in result

    def test_template_not_found(self, provider):
        """Returns empty when get_template returns None."""
        provider.protocol.deliberation_template = "nonexistent"
        with patch(
            "aragora.deliberation.templates.registry.get_template",
            return_value=None,
        ):
            assert provider.get_deliberation_template_context() == ""

    def test_import_error(self, provider):
        """Returns empty on ImportError (registry not available)."""
        provider.protocol.deliberation_template = "socratic"
        with patch(
            "aragora.debate.prompt_context_providers.PromptContextMixin"
            ".get_deliberation_template_context",
            wraps=provider.get_deliberation_template_context,
        ):
            with patch.dict("sys.modules", {"aragora.deliberation.templates.registry": None}):
                # Importing from None module raises ImportError
                result = provider.get_deliberation_template_context()
        assert result == ""


# ===================================================================
# 2. format_patterns_for_prompt
# ===================================================================

class TestFormatPatternsForPrompt:
    def test_empty_patterns(self, provider):
        assert provider.format_patterns_for_prompt([]) == ""

    def test_cached_result(self, provider):
        """Second call returns cached result without recomputing."""
        patterns = [{"category": "logic", "pattern": "Circular reasoning", "occurrences": 3, "avg_severity": 0.8}]
        result1 = provider.format_patterns_for_prompt(patterns)
        result2 = provider.format_patterns_for_prompt(patterns)
        assert result1 == result2
        assert "LEARNED PATTERNS" in result1

    def test_severity_labels(self, provider):
        """Severity labels applied correctly for high, medium, and low."""
        patterns = [
            {"category": "logic", "pattern": "High sev", "occurrences": 5, "avg_severity": 0.8},
            {"category": "evidence", "pattern": "Med sev", "occurrences": 2, "avg_severity": 0.5},
            {"category": "tone", "pattern": "Low sev", "occurrences": 1, "avg_severity": 0.2},
        ]
        result = provider.format_patterns_for_prompt(patterns)
        assert "[HIGH SEVERITY]" in result
        assert "[MEDIUM]" in result
        # Low severity should have no label
        assert result.count("[HIGH SEVERITY]") == 1
        assert result.count("[MEDIUM]") == 1

    def test_limits_to_five_patterns(self, provider):
        """Only first 5 patterns are included."""
        patterns = [
            {"category": f"cat{i}", "pattern": f"Pattern {i}", "occurrences": i}
            for i in range(10)
        ]
        result = provider.format_patterns_for_prompt(patterns)
        assert "Pattern 4" in result
        assert "Pattern 5" not in result


# ===================================================================
# 3. get_stance_guidance
# ===================================================================

class TestGetStanceGuidance:
    def test_no_stance(self, provider, agent):
        """Returns empty when asymmetric_stances is False."""
        provider.protocol.asymmetric_stances = False
        agent.stance = None
        result = provider.get_stance_guidance(agent)
        assert result == ""

    def test_with_affirmative_stance(self, provider, agent):
        provider.protocol.asymmetric_stances = True
        agent.stance = "affirmative"
        result = provider.get_stance_guidance(agent)
        assert "AFFIRMATIVE" in result
        assert "DEFEND" in result


# ===================================================================
# 4. get_agreement_intensity_guidance
# ===================================================================

class TestGetAgreementIntensityGuidance:
    def test_none_intensity(self, provider):
        provider.protocol.agreement_intensity = None
        assert provider.get_agreement_intensity_guidance() == ""

    def test_low_intensity(self, provider):
        provider.protocol.agreement_intensity = 1
        result = provider.get_agreement_intensity_guidance()
        assert "disagree" in result.lower()

    def test_high_intensity(self, provider):
        provider.protocol.agreement_intensity = 9
        result = provider.get_agreement_intensity_guidance()
        assert "agreement" in result.lower() or "synthesis" in result.lower()


# ===================================================================
# 5. format_successful_patterns
# ===================================================================

class TestFormatSuccessfulPatterns:
    def test_no_memory(self, provider):
        provider.memory = None
        assert provider.format_successful_patterns() == ""

    def test_empty_patterns(self, provider):
        provider.memory = MagicMock()
        provider.memory.retrieve_patterns.return_value = []
        assert provider.format_successful_patterns() == ""

    def test_formatted_output(self, provider):
        p1 = MockPattern("Short issue", "Fix it", "logic", 5)
        p2 = MockPattern("A" * 150, "B" * 100, "evidence", 2)
        provider.memory = MagicMock()
        provider.memory.retrieve_patterns.return_value = [p1, p2]

        result = provider.format_successful_patterns(limit=2)

        assert "SUCCESSFUL PATTERNS" in result
        assert "logic" in result
        assert "5 successes" in result
        # Long texts should be truncated
        assert "..." in result

    def test_attribute_error_returns_empty(self, provider):
        provider.memory = MagicMock()
        provider.memory.retrieve_patterns.side_effect = AttributeError("no attr")
        assert provider.format_successful_patterns() == ""


# ===================================================================
# 6. get_role_context
# ===================================================================

class TestGetRoleContext:
    def test_no_rotator(self, provider, agent):
        provider.role_rotator = None
        assert provider.get_role_context(agent) == ""

    def test_agent_not_assigned(self, provider, agent):
        provider.role_rotator = MagicMock()
        provider.current_role_assignments = {}
        assert provider.get_role_context(agent) == ""

    def test_agent_assigned(self, provider, agent):
        provider.role_rotator = MagicMock()
        assignment = MagicMock()
        provider.current_role_assignments = {agent.name: assignment}
        provider.role_rotator.format_role_context.return_value = "You are the Analyst."

        result = provider.get_role_context(agent)

        assert result == "You are the Analyst."
        provider.role_rotator.format_role_context.assert_called_once_with(assignment)


# ===================================================================
# 7. get_round_phase_context
# ===================================================================

class TestGetRoundPhaseContext:
    def test_no_phase(self, provider):
        provider.protocol.get_round_phase.return_value = None
        assert provider.get_round_phase_context(1) == ""

    def test_with_phase(self, provider):
        phase = MagicMock()
        phase.name = "Exploration"
        phase.description = "Explore the solution space"
        phase.focus = "breadth over depth"
        phase.cognitive_mode = "Lateral Thinker"
        provider.protocol.get_round_phase.return_value = phase

        result = provider.get_round_phase_context(2)

        assert "ROUND 2" in result
        assert "EXPLORATION" in result
        assert "breadth over depth" in result
        assert "Lateral Thinker" in result


# ===================================================================
# 8. classify_question_async
# ===================================================================

class TestClassifyQuestionAsync:
    @pytest.mark.asyncio
    async def test_cached_classification(self, provider):
        """Returns cached category without calling classifier again."""
        provider._classification = MagicMock()
        provider._classification.category = "technical"
        result = await provider.classify_question_async()
        assert result == "technical"

    @pytest.mark.asyncio
    async def test_no_classifier_module(self, provider):
        """Falls back to keyword detection when QuestionClassifier is None."""
        provider._classification = None
        with patch("aragora.debate.prompt_context_providers.QuestionClassifier", None):
            result = await provider.classify_question_async()
        # env.task = "Design a rate limiter for API endpoints" -> technical domain
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_llm_classification_success(self, provider):
        """Successful LLM classification stores and returns result."""
        mock_classifier = MagicMock()
        mock_classification = MagicMock()
        mock_classification.category = "technical"
        mock_classification.confidence = 0.95
        mock_classification.recommended_personas = ["engineer"]
        mock_classifier.classify = AsyncMock(return_value=mock_classification)

        provider._question_classifier = mock_classifier
        with patch(
            "aragora.debate.prompt_context_providers.QuestionClassifier",
            MagicMock(return_value=mock_classifier),
        ):
            result = await provider.classify_question_async(use_llm=True)

        assert result == "technical"
        assert provider._classification is mock_classification

    @pytest.mark.asyncio
    async def test_timeout_falls_back_to_keywords(self, provider):
        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock(side_effect=asyncio.TimeoutError("timeout"))
        provider._question_classifier = mock_classifier

        with patch(
            "aragora.debate.prompt_context_providers.QuestionClassifier",
            MagicMock(return_value=mock_classifier),
        ):
            result = await provider.classify_question_async(use_llm=True)

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_value_error_falls_back(self, provider):
        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock(side_effect=ValueError("bad input"))
        provider._question_classifier = mock_classifier

        with patch(
            "aragora.debate.prompt_context_providers.QuestionClassifier",
            MagicMock(return_value=mock_classifier),
        ):
            result = await provider.classify_question_async(use_llm=True)

        assert isinstance(result, str)


# ===================================================================
# 9. _detect_question_domain
# ===================================================================

class TestDetectQuestionDomain:
    def test_from_classification_ethical(self, provider):
        provider._classification = MagicMock()
        provider._classification.category = "ethical"
        assert provider._detect_question_domain("anything") == "philosophical"

    def test_from_classification_technical(self, provider):
        provider._classification = MagicMock()
        provider._classification.category = "technical"
        assert provider._detect_question_domain("anything") == "technical"

    def test_from_classification_general(self, provider):
        provider._classification = MagicMock()
        provider._classification.category = "business"
        assert provider._detect_question_domain("anything") == "general"

    def test_no_classification_falls_back_to_keywords(self, provider):
        provider._classification = None
        result = provider._detect_question_domain("What is the meaning of life?")
        assert result == "philosophical"


# ===================================================================
# 10. _detect_question_domain_keywords
# ===================================================================

class TestDetectQuestionDomainKeywords:
    def test_philosophical(self, provider):
        assert provider._detect_question_domain_keywords("What is the meaning of life?") == "philosophical"

    def test_ethics(self, provider):
        assert provider._detect_question_domain_keywords("Is it ethical to use AI for hiring?") == "ethics"

    def test_technical(self, provider):
        assert provider._detect_question_domain_keywords("How to refactor the API architecture?") == "technical"

    def test_general(self, provider):
        assert provider._detect_question_domain_keywords("What color paint for the office?") == "general"


# ===================================================================
# 11. get_persona_context
# ===================================================================

class TestGetPersonaContext:
    def test_philosophical_domain(self, provider, agent):
        """Philosophical domain returns hardcoded philosophical guidance."""
        provider._classification = MagicMock()
        provider._classification.category = "philosophical"
        result = provider.get_persona_context(agent)
        assert "human condition" in result
        assert "wisdom" in result

    def test_ethics_domain(self, provider, agent):
        provider._classification = MagicMock()
        provider._classification.category = "ethical"
        # "ethical" maps to "philosophical" in _detect_question_domain
        result = provider.get_persona_context(agent)
        assert "human condition" in result

    def test_general_domain(self, provider, agent):
        provider.env.task = "What color for the office?"
        provider._classification = None
        result = provider.get_persona_context(agent)
        assert "advisor" in result

    def test_technical_with_persona_manager(self, provider, agent):
        """Technical domain with persona manager returns persona context."""
        provider._classification = MagicMock()
        provider._classification.category = "technical"
        mock_persona = MagicMock()
        mock_persona.to_prompt_context.return_value = "You are a security expert."
        provider.persona_manager = MagicMock()
        provider.persona_manager.get_persona.return_value = mock_persona

        result = provider.get_persona_context(agent)

        assert result == "You are a security expert."

    def test_technical_no_persona_manager(self, provider, agent):
        """Technical domain without persona manager returns empty."""
        provider._classification = MagicMock()
        provider._classification.category = "technical"
        provider.persona_manager = None
        result = provider.get_persona_context(agent)
        assert result == ""


# ===================================================================
# 12. get_flip_context
# ===================================================================

class TestGetFlipContext:
    def test_no_detector(self, provider, agent):
        provider.flip_detector = None
        assert provider.get_flip_context(agent) == ""

    def test_no_positions(self, provider, agent):
        provider.flip_detector = MagicMock()
        provider.flip_detector.get_agent_consistency.return_value = MockConsistency(
            total_positions=0, total_flips=0
        )
        assert provider.get_flip_context(agent) == ""

    def test_no_flips(self, provider, agent):
        provider.flip_detector = MagicMock()
        provider.flip_detector.get_agent_consistency.return_value = MockConsistency(
            total_positions=10, total_flips=0
        )
        assert provider.get_flip_context(agent) == ""

    def test_contradictions_mentioned(self, provider, agent):
        provider.flip_detector = MagicMock()
        provider.flip_detector.get_agent_consistency.return_value = MockConsistency(
            contradictions=3, retractions=0, consistency_score=0.9
        )
        result = provider.get_flip_context(agent)
        assert "3 prior position contradiction(s)" in result

    def test_low_consistency_score(self, provider, agent):
        provider.flip_detector = MagicMock()
        provider.flip_detector.get_agent_consistency.return_value = MockConsistency(
            contradictions=0, retractions=1, consistency_score=0.5,
            domains_with_flips=["security"]
        )
        result = provider.get_flip_context(agent)
        assert "50%" in result
        assert "security" in result

    def test_error_returns_empty(self, provider, agent):
        provider.flip_detector = MagicMock()
        provider.flip_detector.get_agent_consistency.side_effect = RuntimeError("db fail")
        assert provider.get_flip_context(agent) == ""


# ===================================================================
# 13. get_continuum_context
# ===================================================================

class TestGetContinuumContext:
    def test_returns_cached_value(self, provider):
        provider._continuum_context_cache = "cached continuum data"
        assert provider.get_continuum_context() == "cached continuum data"

    def test_empty_cache(self, provider):
        assert provider.get_continuum_context() == ""


# ===================================================================
# 14. inject_supermemory_context
# ===================================================================

class TestInjectSupermemoryContext:
    @pytest.mark.asyncio
    async def test_no_adapter(self, provider):
        provider.supermemory_adapter = None
        assert await provider.inject_supermemory_context() == ""

    @pytest.mark.asyncio
    async def test_cached_result(self, provider):
        provider.supermemory_adapter = MagicMock()
        provider._supermemory_context_cache = "previously cached"
        result = await provider.inject_supermemory_context()
        assert result == "previously cached"

    @pytest.mark.asyncio
    async def test_successful_injection(self, provider):
        mock_result = MagicMock()
        mock_result.context_content = ["Memory snippet 1", "Memory snippet 2"]
        mock_result.memories_injected = 2
        mock_result.total_tokens_estimate = 150

        adapter = AsyncMock()
        adapter.inject_context = AsyncMock(return_value=mock_result)
        provider.supermemory_adapter = adapter

        result = await provider.inject_supermemory_context()

        assert "External Memory Context" in result
        assert "Memory snippet 1" in result
        assert "2 memories loaded" in result
        assert provider._supermemory_context_cache == result

    @pytest.mark.asyncio
    async def test_empty_content(self, provider):
        mock_result = MagicMock()
        mock_result.context_content = []
        adapter = AsyncMock()
        adapter.inject_context = AsyncMock(return_value=mock_result)
        provider.supermemory_adapter = adapter

        result = await provider.inject_supermemory_context()
        assert result == ""

    @pytest.mark.asyncio
    async def test_timeout_returns_empty(self, provider):
        adapter = AsyncMock()
        adapter.inject_context = AsyncMock(side_effect=asyncio.TimeoutError("slow"))
        provider.supermemory_adapter = adapter

        result = await provider.inject_supermemory_context()
        assert result == ""


# ===================================================================
# 15. get_supermemory_context
# ===================================================================

class TestGetSupermemoryContext:
    def test_returns_cached(self, provider):
        provider._supermemory_context_cache = "cached supermemory"
        assert provider.get_supermemory_context() == "cached supermemory"

    def test_empty(self, provider):
        assert provider.get_supermemory_context() == ""


# ===================================================================
# 16. get_knowledge_mound_context / set_knowledge_context
# ===================================================================

class TestKnowledgeMoundContext:
    def test_getter_empty(self, provider):
        assert provider.get_knowledge_mound_context() == ""

    def test_setter_and_getter(self, provider):
        provider.set_knowledge_context("org knowledge", item_ids=["km-1", "km-2"])
        assert provider.get_knowledge_mound_context() == "org knowledge"
        assert provider._km_item_ids == ["km-1", "km-2"]

    def test_setter_none_context_becomes_empty(self, provider):
        provider.set_knowledge_context(None)
        assert provider.get_knowledge_mound_context() == ""

    def test_setter_no_item_ids_preserves_existing(self, provider):
        provider._km_item_ids = ["old"]
        provider.set_knowledge_context("new context")
        assert provider._km_item_ids == ["old"]


# ===================================================================
# 17. get_codebase_context / set_codebase_context
# ===================================================================

class TestCodebaseContext:
    def test_getter_empty(self, provider):
        assert provider.get_codebase_context() == ""

    def test_setter_and_getter(self, provider):
        provider.set_codebase_context("file: main.py\nclass Arena")
        assert provider.get_codebase_context() == "file: main.py\nclass Arena"

    def test_setter_none_becomes_empty(self, provider):
        provider.set_codebase_context(None)
        assert provider.get_codebase_context() == ""


# ===================================================================
# 18. get_prior_claims_context
# ===================================================================

class TestGetPriorClaimsContext:
    def test_disabled(self, provider):
        provider.include_prior_claims = False
        assert provider.get_prior_claims_context() == ""

    def test_no_kernel(self, provider):
        provider.include_prior_claims = True
        provider.claims_kernel = None
        assert provider.get_prior_claims_context() == ""

    def test_empty_results(self, provider):
        provider.include_prior_claims = True
        provider.claims_kernel = MagicMock()
        provider.claims_kernel.get_related_claims.return_value = []
        assert provider.get_prior_claims_context() == ""

    def test_formatted_claims(self, provider):
        claim = MagicMock()
        claim.claim_type.value = "factual"
        claim.author = "claude_proposer"
        claim.adjusted_confidence = 0.85
        claim.status = "supported"
        claim.statement = "Rate limiting prevents abuse"

        provider.include_prior_claims = True
        provider.claims_kernel = MagicMock()
        provider.claims_kernel.get_related_claims.return_value = [claim]

        result = provider.get_prior_claims_context()

        assert "PRIOR CLAIMS" in result
        assert "FACTUAL" in result
        assert "claude_proposer" in result
        assert "85%" in result
        assert "Rate limiting prevents abuse" in result


# ===================================================================
# 19. set_rlm_context / get_rlm_context_hint / get_rlm_abstract
# ===================================================================

class TestRLMContext:
    def test_set_rlm_context(self, provider):
        ctx = MagicMock()
        ctx.levels = [MagicMock(name="summary"), MagicMock(name="detail")]
        provider.set_rlm_context(ctx)
        assert provider._rlm_context is ctx

    def test_set_rlm_context_none(self, provider):
        provider.set_rlm_context(None)
        assert provider._rlm_context is None

    def test_hint_disabled(self, provider):
        provider._enable_rlm_hints = False
        assert provider.get_rlm_context_hint() == ""

    def test_hint_no_context(self, provider):
        provider._enable_rlm_hints = True
        provider._rlm_context = None
        assert provider.get_rlm_context_hint() == ""

    def test_hint_with_levels(self, provider):
        provider._enable_rlm_hints = True
        level1 = MagicMock()
        level1.name = "Summary"
        level2 = MagicMock()
        level2.name = "Detail"
        ctx = MagicMock()
        ctx.levels = [level1, level2]
        provider._rlm_context = ctx

        result = provider.get_rlm_context_hint()

        assert "HIERARCHICAL CONTEXT AVAILABLE" in result
        assert "Summary" in result
        assert "Detail" in result

    def test_abstract_no_context(self, provider):
        provider._rlm_context = None
        assert provider.get_rlm_abstract() == ""

    def test_abstract_with_level(self, provider):
        ctx = MagicMock()
        ctx.get_at_level.return_value = "Abstract summary of the debate."
        provider._rlm_context = ctx

        with patch("aragora.debate.prompt_context_providers.AbstractionLevel") as mock_level:
            mock_level.ABSTRACT = "abstract"
            result = provider.get_rlm_abstract(max_chars=100)

        assert result == "Abstract summary of the debate."

    def test_abstract_truncation(self, provider):
        ctx = MagicMock()
        long_text = "A" * 3000
        ctx.get_at_level.return_value = long_text
        provider._rlm_context = ctx

        with patch("aragora.debate.prompt_context_providers.AbstractionLevel") as mock_level:
            mock_level.ABSTRACT = "abstract"
            result = provider.get_rlm_abstract(max_chars=500)

        assert len(result) == 500


# ===================================================================
# 20. get_language_constraint
# ===================================================================

class TestGetLanguageConstraint:
    def test_enforcement_enabled(self, provider):
        provider.protocol.language = "French"
        # _get_language_constraint_impl is an lru_cache function; patch the
        # config values it receives via the local imports in get_language_constraint.
        with patch("aragora.config.ENFORCE_RESPONSE_LANGUAGE", True):
            with patch("aragora.config.DEFAULT_DEBATE_LANGUAGE", "English"):
                # Clear the lru_cache so our patched values take effect
                from aragora.debate.prompt_builder import _get_language_constraint_impl
                _get_language_constraint_impl.cache_clear()
                result = provider.get_language_constraint()
        assert "French" in result
        assert "LANGUAGE REQUIREMENT" in result

    def test_enforcement_disabled(self, provider):
        provider.protocol.language = "French"
        with patch("aragora.config.ENFORCE_RESPONSE_LANGUAGE", False):
            with patch("aragora.config.DEFAULT_DEBATE_LANGUAGE", "English"):
                from aragora.debate.prompt_builder import _get_language_constraint_impl
                _get_language_constraint_impl.cache_clear()
                result = provider.get_language_constraint()
        assert result == ""


# ===================================================================
# 21. _inject_belief_context
# ===================================================================

class TestInjectBeliefContext:
    def test_no_continuum_memory(self, provider):
        provider.continuum_memory = None
        assert provider._inject_belief_context() == ""

    def test_no_cruxes(self, provider):
        mem = MagicMock()
        mem.metadata = {"crux_claims": []}
        provider.continuum_memory = MagicMock()
        provider.continuum_memory.retrieve.return_value = [mem]
        assert provider._inject_belief_context() == ""

    def test_formatted_cruxes(self, provider):
        mem = MagicMock()
        mem.metadata = {"crux_claims": ["Scalability is critical", "Cost matters"]}
        provider.continuum_memory = MagicMock()
        provider.continuum_memory.retrieve.return_value = [mem]

        result = provider._inject_belief_context()

        assert "Historical Disagreement Points" in result
        assert "Scalability is critical" in result
        assert "Cost matters" in result

    def test_error_returns_empty(self, provider):
        provider.continuum_memory = MagicMock()
        provider.continuum_memory.retrieve.side_effect = AttributeError("no method")
        assert provider._inject_belief_context() == ""


# ===================================================================
# 22. _inject_calibration_context
# ===================================================================

class TestInjectCalibrationContext:
    def test_no_tracker(self, provider, agent):
        provider.calibration_tracker = None
        assert provider._inject_calibration_context(agent) == ""

    def test_few_predictions(self, provider, agent):
        provider.calibration_tracker = MagicMock()
        provider.calibration_tracker.get_calibration_summary.return_value = MockCalibrationSummary(
            total_predictions=3
        )
        assert provider._inject_calibration_context(agent) == ""

    def test_good_calibration(self, provider, agent):
        """Brier <= 0.25 means well-calibrated, no feedback needed."""
        provider.calibration_tracker = MagicMock()
        provider.calibration_tracker.get_calibration_summary.return_value = MockCalibrationSummary(
            total_predictions=20, brier_score=0.15
        )
        assert provider._inject_calibration_context(agent) == ""

    def test_overconfident(self, provider, agent):
        provider.calibration_tracker = MagicMock()
        provider.calibration_tracker.get_calibration_summary.return_value = MockCalibrationSummary(
            total_predictions=20, brier_score=0.40, is_overconfident=True
        )
        result = provider._inject_calibration_context(agent)
        assert "OVERCONFIDENT" in result
        assert "0.40" in result

    def test_underconfident(self, provider, agent):
        provider.calibration_tracker = MagicMock()
        provider.calibration_tracker.get_calibration_summary.return_value = MockCalibrationSummary(
            total_predictions=20, brier_score=0.35, is_underconfident=True
        )
        result = provider._inject_calibration_context(agent)
        assert "UNDERCONFIDENT" in result


# ===================================================================
# 23. get_elo_context
# ===================================================================

class TestGetEloContext:
    def test_no_elo_system(self, provider, agent):
        provider.elo_system = None
        assert provider.get_elo_context(agent, [agent]) == ""

    def test_empty_ratings(self, provider, agent):
        provider.elo_system = MagicMock()
        provider.elo_system.get_ratings_batch.return_value = {}
        assert provider.get_elo_context(agent, [agent]) == ""

    def test_with_ratings(self, provider, agent):
        other = MockAgent(name="gpt4_critic")
        provider.elo_system = MagicMock()
        provider.elo_system.get_ratings_batch.return_value = {
            agent.name: MockRating(elo=1550, wins=10, losses=5),
            other.name: MockRating(elo=1450, wins=7, losses=8),
        }

        result = provider.get_elo_context(agent, [agent, other])

        assert "Agent Rankings" in result
        assert "claude_proposer" in result
        assert "gpt4_critic" in result
        assert "(you)" in result

    def test_high_elo_encouragement(self, provider, agent):
        provider.elo_system = MagicMock()
        provider.elo_system.get_ratings_batch.return_value = {
            agent.name: MockRating(elo=1650, wins=20, losses=5),
        }
        result = provider.get_elo_context(agent, [agent])
        assert "strong track record" in result

    def test_low_elo_guidance(self, provider, agent):
        provider.elo_system = MagicMock()
        provider.elo_system.get_ratings_batch.return_value = {
            agent.name: MockRating(elo=1350, wins=3, losses=12),
        }
        result = provider.get_elo_context(agent, [agent])
        assert "higher-ranked" in result

    def test_domain_suffix(self, provider, agent):
        provider.domain = "security"
        provider.elo_system = MagicMock()
        provider.elo_system.get_ratings_batch.return_value = {
            agent.name: MockRating(elo=1500),
        }
        result = provider.get_elo_context(agent, [agent])
        assert "(security)" in result


# ===================================================================
# 24. format_evidence_for_prompt
# ===================================================================

class TestFormatEvidenceForPrompt:
    def test_no_evidence_pack(self, provider):
        provider.evidence_pack = None
        assert provider.format_evidence_for_prompt() == ""

    def test_empty_snippets(self, provider):
        provider.evidence_pack = MagicMock()
        provider.evidence_pack.snippets = []
        assert provider.format_evidence_for_prompt() == ""

    def test_with_snippets(self, provider):
        snippet = MagicMock()
        snippet.title = "Rate Limiting Best Practices"
        snippet.source = "RFC 6585"
        snippet.reliability_score = 0.9
        snippet.url = "https://example.com/rfc"
        snippet.snippet = "Token bucket algorithm overview."

        provider.evidence_pack = MagicMock()
        provider.evidence_pack.snippets = [snippet]

        result = provider.format_evidence_for_prompt()

        assert "AVAILABLE EVIDENCE" in result
        assert "[EVID-1]" in result
        assert "Rate Limiting Best Practices" in result
        assert "RFC 6585" in result
        assert "90%" in result
        assert "https://example.com/rfc" in result
        assert "Token bucket algorithm overview" in result

    def test_long_snippet_truncated(self, provider):
        snippet = MagicMock()
        snippet.title = "Title"
        snippet.source = "Source"
        snippet.reliability_score = 0.5
        snippet.url = None
        snippet.snippet = "X" * 300

        provider.evidence_pack = MagicMock()
        provider.evidence_pack.snippets = [snippet]
        provider._rlm_adapter = None

        result = provider.format_evidence_for_prompt()

        assert "..." in result


# ===================================================================
# 25. format_trending_for_prompt
# ===================================================================

class TestFormatTrendingForPrompt:
    def test_trending_disabled(self, provider):
        provider.protocol.enable_trending_injection = False
        assert provider.format_trending_for_prompt() == ""

    def test_no_topics(self, provider):
        provider.protocol.enable_trending_injection = True
        provider.trending_topics = []
        assert provider.format_trending_for_prompt() == ""

    def test_with_topics(self, provider):
        topic = MagicMock()
        topic.topic = "API Security"
        topic.platform = "hackernews"
        topic.volume = 5000
        topic.category = "technology"

        provider.protocol.enable_trending_injection = True
        provider.protocol.trending_injection_max_topics = 3
        provider.protocol.trending_relevance_filter = False
        provider.trending_topics = [topic]

        result = provider.format_trending_for_prompt()

        assert "CURRENT TRENDING CONTEXT" in result
        assert "API Security" in result
        assert "hackernews" in result
        assert "5,000" in result

    def test_relevance_filter(self, provider):
        """Relevance filter prefers topics matching task keywords."""
        relevant = MagicMock()
        relevant.topic = "Rate Limiting Techniques"
        relevant.platform = "reddit"
        relevant.volume = 100
        relevant.category = "tech"

        irrelevant = MagicMock()
        irrelevant.topic = "Cooking Recipes"
        irrelevant.platform = "twitter"
        irrelevant.volume = 9000
        irrelevant.category = "food"

        provider.protocol.enable_trending_injection = True
        provider.protocol.trending_injection_max_topics = 1
        provider.protocol.trending_relevance_filter = True
        provider.env.task = "Design a rate limiter"
        provider.trending_topics = [relevant, irrelevant]

        result = provider.format_trending_for_prompt(max_topics=1)

        assert "Rate Limiting" in result


# ===================================================================
# 26. format_pulse_context
# ===================================================================

class TestFormatPulseContext:
    def test_empty_topics(self, provider):
        provider._pulse_topics = []
        assert provider.format_pulse_context() == ""

    def test_velocity_labels(self, provider):
        provider._pulse_topics = [
            {"topic": "AI Safety", "platform": "hackernews", "volume": 15000, "category": "tech", "hours_ago": 1.5},
            {"topic": "New Framework", "platform": "reddit", "volume": 2000, "category": "dev", "hours_ago": 3.0},
            {"topic": "Minor Update", "platform": "twitter", "volume": 50, "category": "misc", "hours_ago": 0.5},
        ]
        result = provider.format_pulse_context()

        assert "PULSE: TRENDING CONTEXT" in result
        assert "[HIGH VELOCITY]" in result
        assert "[RISING]" in result
        assert "AI Safety" in result
        assert "15,000" in result

    def test_recency_display(self, provider):
        provider._pulse_topics = [
            {"topic": "Fresh", "platform": "test", "volume": 100, "category": "", "hours_ago": 0.0},
            {"topic": "Older", "platform": "test", "volume": 100, "category": "", "hours_ago": 2.5},
        ]
        result = provider.format_pulse_context()
        assert "recent" in result
        assert "2.5h ago" in result

    def test_max_topics_limit(self, provider):
        provider._pulse_topics = [
            {"topic": f"Topic {i}", "platform": "test", "volume": 100, "category": "", "hours_ago": 0}
            for i in range(10)
        ]
        result = provider.format_pulse_context(max_topics=3)
        assert "Topic 0" in result
        assert "Topic 2" in result
        assert "Topic 3" not in result


# ===================================================================
# Setter helper methods
# ===================================================================

class TestSetterMethods:
    def test_set_evidence_pack(self, provider):
        pack = MagicMock()
        provider._evidence_cache = {"old_key": "old_val"}
        provider.set_evidence_pack(pack)
        assert provider.evidence_pack is pack
        assert provider._evidence_cache == {}

    def test_set_trending_topics(self, provider):
        topics = [MagicMock()]
        provider._trending_cache = {"old": "data"}
        provider.set_trending_topics(topics)
        assert provider.trending_topics == topics
        assert provider._trending_cache == {}

    def test_set_trending_topics_none(self, provider):
        provider.set_trending_topics(None)
        assert provider.trending_topics == []

    def test_set_supermemory_adapter(self, provider):
        adapter = MagicMock()
        provider._supermemory_context_cache = "old cache"
        provider._supermemory_context = MagicMock()
        provider.set_supermemory_adapter(adapter)
        assert provider.supermemory_adapter is adapter
        assert provider._supermemory_context is None
        assert provider._supermemory_context_cache == ""

    def test_set_pulse_topics(self, provider):
        topics = [{"topic": "test"}]
        provider.set_pulse_topics(topics)
        assert provider._pulse_topics == topics
