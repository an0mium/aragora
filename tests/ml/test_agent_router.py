"""
Tests for aragora.ml.agent_router module.

Tests cover:
- TaskType enum values
- AgentCapabilities dataclass and default capabilities
- RoutingDecision dataclass and serialization
- AgentRouterConfig defaults and custom values
- AgentRouter initialization and agent registration
- Task classification from descriptions
- Agent scoring for task types
- Historical performance tracking
- ELO score normalization
- Diversity score calculation
- Cost score calculation
- Main route() method with various scenarios
- Constraint handling (require_code, require_vision, max_cost)
- Performance recording and history trimming
- ELO updating
- Agent statistics retrieval
- Global instance management (get_agent_router)
"""

from __future__ import annotations

from collections import defaultdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aragora.ml.agent_router import (
    AgentCapabilities,
    AgentRouter,
    AgentRouterConfig,
    RoutingDecision,
    TaskType,
    get_agent_router,
)


# =============================================================================
# TestTaskType - Enum Tests
# =============================================================================


class TestTaskType:
    """Tests for TaskType enum."""

    def test_all_types_defined(self):
        """Should define all expected task types."""
        assert TaskType.CODING.value == "coding"
        assert TaskType.ANALYSIS.value == "analysis"
        assert TaskType.CREATIVE.value == "creative"
        assert TaskType.REASONING.value == "reasoning"
        assert TaskType.RESEARCH.value == "research"
        assert TaskType.MATH.value == "math"
        assert TaskType.GENERAL.value == "general"

    def test_type_count(self):
        """Should have expected number of types."""
        assert len(TaskType) == 7

    def test_types_are_strings(self):
        """Task type values should be strings."""
        for task_type in TaskType:
            assert isinstance(task_type.value, str)

    def test_string_comparison(self):
        """Should support string comparison due to str mixin."""
        assert TaskType.CODING == "coding"
        assert TaskType.GENERAL == "general"


# =============================================================================
# TestAgentCapabilities - Dataclass Tests
# =============================================================================


class TestAgentCapabilitiesInit:
    """Tests for AgentCapabilities initialization."""

    def test_creates_with_required_fields(self):
        """Should create with agent_id only."""
        caps = AgentCapabilities(agent_id="test-agent")
        assert caps.agent_id == "test-agent"
        assert caps.strengths == []
        assert caps.weaknesses == []
        assert caps.speed_tier == 2
        assert caps.cost_tier == 2
        assert caps.max_context == 8000
        assert caps.supports_code is True
        assert caps.supports_vision is False
        assert caps.elo_rating == 1000.0

    def test_creates_with_all_fields(self):
        """Should create with all fields."""
        caps = AgentCapabilities(
            agent_id="claude",
            strengths=[TaskType.ANALYSIS, TaskType.REASONING],
            weaknesses=[TaskType.MATH],
            speed_tier=1,
            cost_tier=3,
            max_context=200000,
            supports_code=True,
            supports_vision=True,
            elo_rating=1200.0,
        )
        assert caps.agent_id == "claude"
        assert len(caps.strengths) == 2
        assert len(caps.weaknesses) == 1
        assert caps.speed_tier == 1
        assert caps.cost_tier == 3
        assert caps.max_context == 200000
        assert caps.supports_vision is True
        assert caps.elo_rating == 1200.0


class TestAgentCapabilitiesDefaultCapabilities:
    """Tests for AgentCapabilities.default_capabilities()."""

    def test_returns_dict_of_capabilities(self):
        """Should return dict mapping agent_id to capabilities."""
        defaults = AgentCapabilities.default_capabilities()
        assert isinstance(defaults, dict)
        assert all(isinstance(v, AgentCapabilities) for v in defaults.values())

    def test_known_agents_present(self):
        """Should include all known agents."""
        defaults = AgentCapabilities.default_capabilities()
        expected_agents = [
            "claude",
            "claude-sonnet",
            "gpt-4",
            "gpt-4o",
            "codex",
            "gemini",
            "grok",
            "mistral-large",
            "deepseek",
            "llama",
        ]
        for agent in expected_agents:
            assert agent in defaults, f"{agent} not in default capabilities"

    def test_claude_capabilities(self):
        """Claude should have correct capabilities."""
        defaults = AgentCapabilities.default_capabilities()
        claude = defaults["claude"]
        assert TaskType.ANALYSIS in claude.strengths
        assert TaskType.REASONING in claude.strengths
        assert claude.max_context == 200000
        assert claude.supports_vision is True
        assert claude.elo_rating == 1100

    def test_codex_capabilities(self):
        """Codex should be strong at coding, weak at creative."""
        defaults = AgentCapabilities.default_capabilities()
        codex = defaults["codex"]
        assert TaskType.CODING in codex.strengths
        assert TaskType.CREATIVE in codex.weaknesses
        assert TaskType.RESEARCH in codex.weaknesses

    def test_deepseek_capabilities(self):
        """DeepSeek should be strong at coding and math."""
        defaults = AgentCapabilities.default_capabilities()
        ds = defaults["deepseek"]
        assert TaskType.CODING in ds.strengths
        assert TaskType.MATH in ds.strengths
        assert ds.cost_tier == 1  # Cheap

    def test_gemini_large_context(self):
        """Gemini should have very large context window."""
        defaults = AgentCapabilities.default_capabilities()
        gemini = defaults["gemini"]
        assert gemini.max_context == 1000000


# =============================================================================
# TestRoutingDecision - Dataclass Tests
# =============================================================================


class TestRoutingDecisionInit:
    """Tests for RoutingDecision initialization."""

    def test_creates_with_required_fields(self):
        """Should create with all required fields."""
        decision = RoutingDecision(
            selected_agents=["claude", "gpt-4"],
            task_type=TaskType.CODING,
            confidence=0.85,
            reasoning=["task_type=coding", "claude_strong_at_coding"],
        )
        assert decision.selected_agents == ["claude", "gpt-4"]
        assert decision.task_type == TaskType.CODING
        assert decision.confidence == 0.85
        assert len(decision.reasoning) == 2

    def test_default_values(self):
        """Should have sensible defaults."""
        decision = RoutingDecision(
            selected_agents=[],
            task_type=TaskType.GENERAL,
            confidence=0.0,
            reasoning=[],
        )
        assert decision.agent_scores == {}
        assert decision.diversity_score == 0.0


class TestRoutingDecisionToDict:
    """Tests for RoutingDecision.to_dict()."""

    def test_returns_dict(self):
        """to_dict should return dictionary."""
        decision = RoutingDecision(
            selected_agents=["claude"],
            task_type=TaskType.CODING,
            confidence=0.8567,
            reasoning=["reason1"],
            diversity_score=0.7123,
        )
        d = decision.to_dict()
        assert d["selected_agents"] == ["claude"]
        assert d["task_type"] == "coding"
        assert d["confidence"] == 0.857
        assert d["reasoning"] == ["reason1"]
        assert d["diversity_score"] == 0.712

    def test_rounds_values(self):
        """Should round confidence and diversity to 3 decimals."""
        decision = RoutingDecision(
            selected_agents=[],
            task_type=TaskType.GENERAL,
            confidence=0.123456,
            reasoning=[],
            diversity_score=0.654321,
        )
        d = decision.to_dict()
        assert d["confidence"] == 0.123
        assert d["diversity_score"] == 0.654


# =============================================================================
# TestAgentRouterConfig - Configuration Tests
# =============================================================================


class TestAgentRouterConfigDefaults:
    """Tests for AgentRouterConfig default values."""

    def test_default_weights(self):
        """Default weights should sum to 1.0."""
        config = AgentRouterConfig()
        total = (
            config.weight_task_match
            + config.weight_historical
            + config.weight_elo
            + config.weight_diversity
            + config.weight_cost
        )
        assert abs(total - 1.0) < 0.01

    def test_default_values(self):
        """Should have sensible defaults."""
        config = AgentRouterConfig()
        assert config.weight_task_match == 0.35
        assert config.weight_historical == 0.25
        assert config.weight_elo == 0.20
        assert config.weight_diversity == 0.10
        assert config.weight_cost == 0.10
        assert config.prefer_diversity is True
        assert config.max_same_provider == 2
        assert config.min_confidence_threshold == 0.3
        assert config.use_embeddings is True


class TestAgentRouterConfigCustom:
    """Tests for AgentRouterConfig custom values."""

    def test_accepts_custom_values(self):
        """Should accept custom configuration."""
        config = AgentRouterConfig(
            weight_task_match=0.5,
            weight_historical=0.2,
            weight_elo=0.1,
            weight_diversity=0.1,
            weight_cost=0.1,
            prefer_diversity=False,
            use_embeddings=False,
        )
        assert config.weight_task_match == 0.5
        assert config.prefer_diversity is False
        assert config.use_embeddings is False


# =============================================================================
# TestAgentRouterInit - Initialization Tests
# =============================================================================


class TestAgentRouterInit:
    """Tests for AgentRouter initialization."""

    def test_creates_with_default_config(self):
        """Should create with default config."""
        router = AgentRouter()
        assert router.config is not None
        assert isinstance(router.config, AgentRouterConfig)

    def test_creates_with_custom_config(self):
        """Should accept custom config."""
        config = AgentRouterConfig(use_embeddings=False)
        router = AgentRouter(config)
        assert router.config.use_embeddings is False

    def test_initializes_with_default_capabilities(self):
        """Should initialize with default agent capabilities."""
        router = AgentRouter()
        assert "claude" in router._capabilities
        assert "gpt-4" in router._capabilities
        assert "codex" in router._capabilities

    def test_initializes_empty_history(self):
        """Should initialize with empty historical performance."""
        router = AgentRouter()
        assert isinstance(router._historical_performance, defaultdict)


# =============================================================================
# TestAgentRouterRegisterAgent - Agent Registration Tests
# =============================================================================


class TestAgentRouterRegisterAgent:
    """Tests for AgentRouter.register_agent()."""

    def test_registers_new_agent(self):
        """Should register a new agent."""
        router = AgentRouter()
        caps = AgentCapabilities(
            agent_id="custom-agent",
            strengths=[TaskType.CODING],
            elo_rating=1200,
        )
        router.register_agent(caps)
        assert "custom-agent" in router._capabilities
        assert router._capabilities["custom-agent"].elo_rating == 1200

    def test_updates_existing_agent(self):
        """Should update existing agent capabilities."""
        router = AgentRouter()
        original_elo = router._capabilities["claude"].elo_rating

        new_caps = AgentCapabilities(
            agent_id="claude",
            strengths=[TaskType.CODING],
            elo_rating=1300,
        )
        router.register_agent(new_caps)
        assert router._capabilities["claude"].elo_rating == 1300
        assert router._capabilities["claude"].elo_rating != original_elo


# =============================================================================
# TestAgentRouterClassifyTask - Task Classification Tests
# =============================================================================


class TestAgentRouterClassifyTask:
    """Tests for AgentRouter._classify_task()."""

    @pytest.fixture
    def router(self):
        return AgentRouter(AgentRouterConfig(use_embeddings=False))

    def test_classifies_coding_task(self, router):
        """Should classify coding-related tasks."""
        task_type, confidence = router._classify_task("Implement a REST API in Python")
        assert task_type == TaskType.CODING
        assert confidence > 0.3

    def test_classifies_math_task(self, router):
        """Should classify math-related tasks."""
        task_type, confidence = router._classify_task("Calculate the derivative of x^2")
        assert task_type == TaskType.MATH
        assert confidence > 0.3

    def test_classifies_analysis_task(self, router):
        """Should classify analysis-related tasks."""
        task_type, confidence = router._classify_task("Analyze the pros and cons of microservices")
        assert task_type == TaskType.ANALYSIS
        assert confidence > 0.3

    def test_classifies_creative_task(self, router):
        """Should classify creative-related tasks."""
        task_type, confidence = router._classify_task("Write a short story about space exploration")
        assert task_type == TaskType.CREATIVE
        assert confidence > 0.3

    def test_classifies_reasoning_task(self, router):
        """Should classify reasoning-related tasks."""
        task_type, confidence = router._classify_task(
            "Reason about why the hypothesis is incorrect"
        )
        assert task_type == TaskType.REASONING
        assert confidence > 0.3

    def test_classifies_research_task(self, router):
        """Should classify research-related tasks."""
        task_type, confidence = router._classify_task(
            "Research the latest sources on climate change"
        )
        assert task_type == TaskType.RESEARCH
        assert confidence > 0.3

    def test_returns_general_for_ambiguous(self, router):
        """Should return GENERAL for ambiguous tasks."""
        task_type, confidence = router._classify_task("do thing now please")
        assert task_type == TaskType.GENERAL
        assert confidence == 0.5

    def test_confidence_bounded(self, router):
        """Confidence should be bounded between 0 and 1."""
        task_type, confidence = router._classify_task("implement code function class algorithm")
        assert 0.0 <= confidence <= 1.0

    def test_case_insensitive(self, router):
        """Should be case insensitive."""
        task_type1, _ = router._classify_task("IMPLEMENT a Python API")
        task_type2, _ = router._classify_task("implement a python api")
        assert task_type1 == task_type2

    def test_multiple_keyword_matches(self, router):
        """Tasks matching multiple keywords should get higher confidence."""
        # Single keyword
        _, conf1 = router._classify_task("implement")
        # Multiple keywords
        _, conf2 = router._classify_task("implement a function in python with an algorithm")
        assert conf2 >= conf1


# =============================================================================
# TestAgentRouterScoreAgentForTask - Agent Scoring Tests
# =============================================================================


class TestAgentRouterScoreAgentForTask:
    """Tests for AgentRouter._score_agent_for_task()."""

    @pytest.fixture
    def router(self):
        return AgentRouter()

    def test_scores_strength_higher(self, router):
        """Agent strong at task type should score higher."""
        # Claude is strong at analysis
        score = router._score_agent_for_task("claude", TaskType.ANALYSIS)
        assert score > 0.5

    def test_scores_weakness_lower(self, router):
        """Agent weak at task type should score lower."""
        # Codex is weak at creative
        score = router._score_agent_for_task("codex", TaskType.CREATIVE)
        assert score < 0.5

    def test_unknown_agent_neutral(self, router):
        """Unknown agent should get neutral score."""
        score = router._score_agent_for_task("unknown-agent", TaskType.CODING)
        assert score == 0.5

    def test_score_bounded(self, router):
        """Score should be bounded 0-1."""
        for agent in ["claude", "codex", "gpt-4", "unknown"]:
            for task_type in TaskType:
                score = router._score_agent_for_task(agent, task_type)
                assert 0.0 <= score <= 1.0

    def test_neutral_for_neither_strength_nor_weakness(self, router):
        """Agent with no opinion on task should get base score."""
        # Claude doesn't list MATH as strength or weakness
        score = router._score_agent_for_task("claude", TaskType.MATH)
        assert score == 0.5


# =============================================================================
# TestAgentRouterHistoricalScore - Historical Performance Tests
# =============================================================================


class TestAgentRouterHistoricalScore:
    """Tests for AgentRouter._get_historical_score()."""

    @pytest.fixture
    def router(self):
        return AgentRouter()

    def test_no_history_neutral(self, router):
        """No history should return neutral score."""
        score = router._get_historical_score("claude", TaskType.CODING)
        assert score == 0.5

    def test_all_success_high_score(self, router):
        """All successes should give high score."""
        for _ in range(10):
            router.record_performance("claude", "coding", True)
        score = router._get_historical_score("claude", TaskType.CODING)
        assert score == 1.0

    def test_all_failure_low_score(self, router):
        """All failures should give low score."""
        for _ in range(10):
            router.record_performance("claude", "coding", False)
        score = router._get_historical_score("claude", TaskType.CODING)
        assert score == 0.0

    def test_mixed_results(self, router):
        """Mixed results should give intermediate score."""
        for _ in range(5):
            router.record_performance("claude", "coding", True)
        for _ in range(5):
            router.record_performance("claude", "coding", False)
        score = router._get_historical_score("claude", TaskType.CODING)
        assert score == 0.5

    def test_uses_recent_history(self, router):
        """Should use only recent history (last 50)."""
        # Add 100 failures
        for _ in range(100):
            router.record_performance("claude", "coding", False)
        # Then 50 successes (these are the recent ones)
        for _ in range(50):
            router.record_performance("claude", "coding", True)

        score = router._get_historical_score("claude", TaskType.CODING)
        # Last 50 are all successes
        assert score > 0.3  # Should reflect recent success rate


# =============================================================================
# TestAgentRouterEloScore - ELO Score Tests
# =============================================================================


class TestAgentRouterEloScore:
    """Tests for AgentRouter._get_elo_score()."""

    @pytest.fixture
    def router(self):
        return AgentRouter()

    def test_known_agent_score(self, router):
        """Known agent should get normalized ELO score."""
        score = router._get_elo_score("claude")
        # Claude ELO is 1100, normalized: (1100-800)/400 = 0.75
        assert score == pytest.approx(0.75)

    def test_unknown_agent_neutral(self, router):
        """Unknown agent should get neutral score."""
        score = router._get_elo_score("unknown")
        assert score == 0.5

    def test_score_bounded(self, router):
        """Score should be bounded 0-1."""
        for agent_id in router._capabilities:
            score = router._get_elo_score(agent_id)
            assert 0.0 <= score <= 1.0

    def test_high_elo_high_score(self, router):
        """High ELO should give high score."""
        router._capabilities["test"] = AgentCapabilities(agent_id="test", elo_rating=1200)
        score = router._get_elo_score("test")
        assert score == 1.0  # (1200-800)/400 = 1.0

    def test_low_elo_low_score(self, router):
        """Low ELO should give low score."""
        router._capabilities["test"] = AgentCapabilities(agent_id="test", elo_rating=800)
        score = router._get_elo_score("test")
        assert score == 0.0  # (800-800)/400 = 0.0


# =============================================================================
# TestAgentRouterDiversityScore - Diversity Score Tests
# =============================================================================


class TestAgentRouterDiversityScore:
    """Tests for AgentRouter._calculate_diversity_score()."""

    @pytest.fixture
    def router(self):
        return AgentRouter()

    def test_single_agent_zero_diversity(self, router):
        """Single agent should have zero diversity."""
        score = router._calculate_diversity_score(["claude"], ["claude", "gpt-4"])
        assert score == 0.0

    def test_diverse_team_high_score(self, router):
        """Team from different providers should have high diversity."""
        score = router._calculate_diversity_score(
            ["claude", "gpt-4", "gemini"],
            ["claude", "gpt-4", "gemini"],
        )
        assert score > 0.5

    def test_same_provider_lower_diversity(self, router):
        """Team from same provider should have lower diversity."""
        score = router._calculate_diversity_score(
            ["claude", "claude-sonnet"],
            ["claude", "claude-sonnet", "gpt-4"],
        )
        # Both are anthropic, so provider diversity is lower
        same_provider_score = score

        diverse_score = router._calculate_diversity_score(
            ["claude", "gpt-4"],
            ["claude", "gpt-4", "gemini"],
        )
        assert diverse_score >= same_provider_score

    def test_empty_team_zero_diversity(self, router):
        """Empty team should have zero diversity."""
        score = router._calculate_diversity_score([], ["claude", "gpt-4"])
        assert score == 0.0

    def test_provider_inference(self, router):
        """Should correctly infer providers from agent names."""
        # Test that different providers are detected
        score = router._calculate_diversity_score(
            ["claude", "gpt-4", "gemini", "mistral-large"],
            ["claude", "gpt-4", "gemini", "mistral-large"],
        )
        # 4 different providers out of 4 agents = high provider diversity
        assert score > 0.5


# =============================================================================
# TestAgentRouterCostScore - Cost Score Tests
# =============================================================================


class TestAgentRouterCostScore:
    """Tests for AgentRouter._get_cost_score()."""

    @pytest.fixture
    def router(self):
        return AgentRouter()

    def test_cheap_agent_high_score(self, router):
        """Cheap agent should have high cost score."""
        # llama is cost tier 1
        score = router._get_cost_score("llama")
        assert score == 1.0  # 1.0 - (1-1)/2 = 1.0

    def test_expensive_agent_low_score(self, router):
        """Expensive agent should have low cost score."""
        # claude is cost tier 3
        score = router._get_cost_score("claude")
        assert score == 0.0  # 1.0 - (3-1)/2 = 0.0

    def test_medium_agent_middle_score(self, router):
        """Medium cost agent should have middle score."""
        # gpt-4o is cost tier 2
        score = router._get_cost_score("gpt-4o")
        assert score == 0.5  # 1.0 - (2-1)/2 = 0.5

    def test_unknown_agent_neutral(self, router):
        """Unknown agent should get neutral score."""
        score = router._get_cost_score("unknown")
        assert score == 0.5


# =============================================================================
# TestAgentRouterRoute - Main Routing Tests
# =============================================================================


class TestAgentRouterRouteBasic:
    """Basic tests for AgentRouter.route()."""

    @pytest.fixture
    def router(self):
        return AgentRouter(AgentRouterConfig(use_embeddings=False))

    def test_returns_routing_decision(self, router):
        """Should return RoutingDecision instance."""
        decision = router.route(
            task="Implement a function",
            available_agents=["claude", "gpt-4"],
        )
        assert isinstance(decision, RoutingDecision)

    def test_empty_agents_returns_empty(self, router):
        """Should handle empty agent list."""
        decision = router.route(task="Test", available_agents=[])
        assert decision.selected_agents == []
        assert decision.task_type == TaskType.GENERAL
        assert decision.confidence == 0.0
        assert "no_agents_available" in decision.reasoning

    def test_selects_team_size_agents(self, router):
        """Should select requested team size."""
        decision = router.route(
            task="Implement a function",
            available_agents=["claude", "gpt-4", "codex", "gemini", "grok"],
            team_size=3,
        )
        assert len(decision.selected_agents) == 3

    def test_selects_fewer_if_not_enough(self, router):
        """Should select fewer agents if not enough available."""
        decision = router.route(
            task="Test",
            available_agents=["claude"],
            team_size=3,
        )
        assert len(decision.selected_agents) == 1

    def test_includes_agent_scores(self, router):
        """Should include individual agent scores."""
        decision = router.route(
            task="Implement a function",
            available_agents=["claude", "gpt-4"],
        )
        assert "claude" in decision.agent_scores
        assert "gpt-4" in decision.agent_scores

    def test_includes_diversity_score(self, router):
        """Should include diversity score."""
        decision = router.route(
            task="Test",
            available_agents=["claude", "gpt-4", "gemini"],
            team_size=3,
        )
        assert 0.0 <= decision.diversity_score <= 1.0


class TestAgentRouterRouteTaskMatching:
    """Tests for task-type specific routing."""

    @pytest.fixture
    def router(self):
        return AgentRouter(AgentRouterConfig(use_embeddings=False, prefer_diversity=False))

    def test_coding_task_favors_coders(self, router):
        """Coding task should favor coding-strong agents."""
        decision = router.route(
            task="Implement a Python function",
            available_agents=["claude", "codex", "grok"],
            team_size=2,
        )
        # Codex should be selected (strong at coding)
        assert "codex" in decision.selected_agents

    def test_analysis_task_favors_analysts(self, router):
        """Analysis task should favor analysis-strong agents."""
        decision = router.route(
            task="Analyze the pros and cons of this approach",
            available_agents=["claude", "codex", "llama"],
            team_size=1,
        )
        # Claude is strong at analysis
        assert "claude" in decision.selected_agents

    def test_task_type_in_reasoning(self, router):
        """Should include task type in reasoning."""
        decision = router.route(
            task="Write a creative story",
            available_agents=["claude", "gpt-4"],
        )
        assert any("task_type=" in r for r in decision.reasoning)


class TestAgentRouterRouteConstraints:
    """Tests for constraint handling in routing."""

    @pytest.fixture
    def router(self):
        return AgentRouter(AgentRouterConfig(use_embeddings=False, prefer_diversity=False))

    def test_require_code_constraint(self, router):
        """Should penalize agents that don't support code."""
        # Register an agent without code support
        router.register_agent(
            AgentCapabilities(
                agent_id="no-code-agent",
                supports_code=False,
                elo_rating=1200,  # Very high ELO
            )
        )

        decision = router.route(
            task="Write code",
            available_agents=["no-code-agent", "codex"],
            team_size=1,
            constraints={"require_code": True},
        )
        # Should prefer codex despite lower ELO
        assert decision.selected_agents[0] == "codex"

    def test_require_vision_constraint(self, router):
        """Should penalize agents without vision support."""
        decision = router.route(
            task="Analyze this image",
            available_agents=["claude", "llama"],
            team_size=1,
            constraints={"require_vision": True},
        )
        # Claude supports vision, llama doesn't
        assert "claude" in decision.selected_agents

    def test_max_cost_constraint(self, router):
        """Should penalize expensive agents when max_cost set."""
        decision = router.route(
            task="Simple task",
            available_agents=["claude", "llama"],
            team_size=1,
            constraints={"max_cost": 1},
        )
        # llama is cost tier 1, claude is cost tier 3
        assert "llama" in decision.selected_agents

    def test_no_constraints(self, router):
        """Should work without constraints."""
        decision = router.route(
            task="Test",
            available_agents=["claude", "gpt-4"],
        )
        assert len(decision.selected_agents) > 0

    def test_empty_constraints(self, router):
        """Should handle empty constraints dict."""
        decision = router.route(
            task="Test",
            available_agents=["claude"],
            constraints={},
        )
        assert len(decision.selected_agents) == 1


class TestAgentRouterRouteDiversity:
    """Tests for diversity optimization in routing."""

    def test_diversity_prefers_different_providers(self):
        """With diversity enabled, should prefer different providers."""
        router = AgentRouter(
            AgentRouterConfig(
                use_embeddings=False,
                prefer_diversity=True,
            )
        )

        decision = router.route(
            task="General task",
            available_agents=["claude", "claude-sonnet", "gpt-4", "gemini"],
            team_size=3,
        )
        # Should prefer diverse team
        assert decision.diversity_score > 0.0

    def test_without_diversity_selects_top_scoring(self):
        """Without diversity, should select purely by score."""
        router = AgentRouter(
            AgentRouterConfig(
                use_embeddings=False,
                prefer_diversity=False,
            )
        )

        decision = router.route(
            task="General task",
            available_agents=["claude", "gpt-4", "gemini"],
            team_size=3,
        )
        assert len(decision.selected_agents) == 3

    def test_good_diversity_noted_in_reasoning(self):
        """Should note good diversity in reasoning."""
        router = AgentRouter(AgentRouterConfig(use_embeddings=False))

        decision = router.route(
            task="General task",
            available_agents=["claude", "gpt-4", "gemini", "deepseek", "grok"],
            team_size=4,
        )
        # Diverse team should have diversity noted
        if decision.diversity_score > 0.5:
            assert "good_team_diversity" in decision.reasoning


class TestAgentRouterRouteConfidence:
    """Tests for confidence scoring in routing."""

    @pytest.fixture
    def router(self):
        return AgentRouter(AgentRouterConfig(use_embeddings=False))

    def test_confidence_bounded(self, router):
        """Confidence should be bounded."""
        decision = router.route(
            task="Implement a function in Python",
            available_agents=["claude", "gpt-4", "codex"],
            team_size=3,
        )
        assert 0.0 <= decision.confidence <= 2.0  # Can be above 1 with diversity bonus

    def test_higher_confidence_for_clear_task(self, router):
        """Clear task type should have higher confidence."""
        clear_decision = router.route(
            task="Implement a Python function with algorithm",
            available_agents=["claude", "codex"],
        )
        ambiguous_decision = router.route(
            task="do stuff",
            available_agents=["claude", "codex"],
        )
        assert clear_decision.confidence >= ambiguous_decision.confidence


# =============================================================================
# TestAgentRouterRecordPerformance - Performance Recording Tests
# =============================================================================


class TestAgentRouterRecordPerformance:
    """Tests for AgentRouter.record_performance()."""

    @pytest.fixture
    def router(self):
        return AgentRouter()

    def test_records_success(self, router):
        """Should record successful performance."""
        router.record_performance("claude", "coding", True)
        history = router._historical_performance["claude"]["coding"]
        assert len(history) == 1
        assert history[0] is True

    def test_records_failure(self, router):
        """Should record failed performance."""
        router.record_performance("claude", "coding", False)
        history = router._historical_performance["claude"]["coding"]
        assert len(history) == 1
        assert history[0] is False

    def test_accumulates_history(self, router):
        """Should accumulate performance history."""
        for i in range(10):
            router.record_performance("claude", "coding", i % 2 == 0)
        history = router._historical_performance["claude"]["coding"]
        assert len(history) == 10

    def test_trims_history_at_500(self, router):
        """Should trim history to last 500 entries."""
        for _ in range(600):
            router.record_performance("claude", "coding", True)
        history = router._historical_performance["claude"]["coding"]
        assert len(history) == 500

    def test_separates_task_types(self, router):
        """Should track different task types separately."""
        router.record_performance("claude", "coding", True)
        router.record_performance("claude", "analysis", False)

        coding_history = router._historical_performance["claude"]["coding"]
        analysis_history = router._historical_performance["claude"]["analysis"]

        assert len(coding_history) == 1
        assert len(analysis_history) == 1
        assert coding_history[0] is True
        assert analysis_history[0] is False


# =============================================================================
# TestAgentRouterUpdateElo - ELO Update Tests
# =============================================================================


class TestAgentRouterUpdateElo:
    """Tests for AgentRouter.update_elo()."""

    @pytest.fixture
    def router(self):
        return AgentRouter()

    def test_updates_existing_agent(self, router):
        """Should update ELO for existing agent."""
        router.update_elo("claude", 1500.0)
        assert router._capabilities["claude"].elo_rating == 1500.0

    def test_creates_agent_if_unknown(self, router):
        """Should create capabilities for unknown agent."""
        router.update_elo("new-agent", 1200.0)
        assert "new-agent" in router._capabilities
        assert router._capabilities["new-agent"].elo_rating == 1200.0
        assert router._capabilities["new-agent"].agent_id == "new-agent"


# =============================================================================
# TestAgentRouterGetAgentStats - Agent Statistics Tests
# =============================================================================


class TestAgentRouterGetAgentStats:
    """Tests for AgentRouter.get_agent_stats()."""

    @pytest.fixture
    def router(self):
        return AgentRouter()

    def test_returns_stats_for_known_agent(self, router):
        """Should return stats for registered agent."""
        stats = router.get_agent_stats("claude")
        assert stats["agent_id"] == "claude"
        assert stats["registered"] is True
        assert "elo_rating" in stats
        assert "strengths" in stats
        assert "weaknesses" in stats
        assert "cost_tier" in stats
        assert "speed_tier" in stats

    def test_returns_minimal_for_unknown_agent(self, router):
        """Should return minimal stats for unknown agent."""
        stats = router.get_agent_stats("unknown")
        assert stats["agent_id"] == "unknown"
        assert stats["registered"] is False
        assert "elo_rating" not in stats

    def test_includes_performance_history(self, router):
        """Should include performance stats when available."""
        router.record_performance("claude", "coding", True)
        router.record_performance("claude", "coding", False)
        router.record_performance("claude", "analysis", True)

        stats = router.get_agent_stats("claude")
        assert stats["total_tasks"] == 3
        assert stats["overall_success_rate"] == pytest.approx(2 / 3)
        assert "task_breakdown" in stats
        assert stats["task_breakdown"]["coding"] == 0.5
        assert stats["task_breakdown"]["analysis"] == 1.0

    def test_strengths_as_string_values(self, router):
        """Should return strengths as string values."""
        stats = router.get_agent_stats("claude")
        assert all(isinstance(s, str) for s in stats["strengths"])


# =============================================================================
# TestAgentRouterEmbeddingService - Embedding Service Tests
# =============================================================================


class TestAgentRouterEmbeddingService:
    """Tests for AgentRouter._get_embedding_service()."""

    def test_returns_none_when_disabled(self):
        """Should return None when embeddings disabled."""
        router = AgentRouter(AgentRouterConfig(use_embeddings=False))
        service = router._get_embedding_service()
        assert service is None

    def test_disables_on_error(self):
        """Should disable embeddings on import error."""
        router = AgentRouter(AgentRouterConfig(use_embeddings=True))

        with patch(
            "aragora.ml.embeddings.get_embedding_service",
            side_effect=ImportError("No embedding service"),
        ):
            router._get_embedding_service()

        assert router.config.use_embeddings is False

    def test_caches_service(self):
        """Should cache embedding service."""
        router = AgentRouter(AgentRouterConfig(use_embeddings=True))
        mock_service = MagicMock()

        with patch(
            "aragora.ml.embeddings.get_embedding_service",
            return_value=mock_service,
        ):
            service1 = router._get_embedding_service()
            service2 = router._get_embedding_service()

        assert service1 is service2


# =============================================================================
# TestGetAgentRouter - Global Instance Tests
# =============================================================================


class TestGetAgentRouter:
    """Tests for get_agent_router global function."""

    def test_returns_agent_router(self):
        """Should return AgentRouter instance."""
        with patch("aragora.ml.agent_router._agent_router", None):
            router = get_agent_router()
            assert isinstance(router, AgentRouter)

    def test_returns_same_instance(self):
        """Should return same instance on subsequent calls."""
        with patch("aragora.ml.agent_router._agent_router", None):
            router1 = get_agent_router()
            router2 = get_agent_router()
            assert router1 is router2


# =============================================================================
# Integration Tests
# =============================================================================


class TestAgentRouterIntegration:
    """Integration tests for agent routing workflow."""

    def test_full_routing_workflow(self):
        """Should support full routing workflow: route -> record -> improve."""
        router = AgentRouter(AgentRouterConfig(use_embeddings=False))

        # Route a task
        decision = router.route(
            task="Implement a sorting algorithm in Python",
            available_agents=["claude", "codex", "gpt-4", "gemini"],
            team_size=3,
        )
        assert len(decision.selected_agents) == 3
        assert decision.task_type == TaskType.CODING

        # Record outcomes
        for agent in decision.selected_agents:
            router.record_performance(agent, decision.task_type.value, True)

        # Verify history recorded
        for agent in decision.selected_agents:
            stats = router.get_agent_stats(agent)
            assert stats.get("total_tasks", 0) > 0

    def test_consistent_routing(self):
        """Same inputs should produce same routing."""
        router = AgentRouter(AgentRouterConfig(use_embeddings=False))

        decision1 = router.route(
            task="Implement a REST API",
            available_agents=["claude", "gpt-4", "codex"],
            team_size=2,
        )
        decision2 = router.route(
            task="Implement a REST API",
            available_agents=["claude", "gpt-4", "codex"],
            team_size=2,
        )

        assert decision1.selected_agents == decision2.selected_agents
        assert decision1.task_type == decision2.task_type

    def test_routing_adapts_to_performance(self):
        """Routing should change based on recorded performance."""
        router = AgentRouter(
            AgentRouterConfig(
                use_embeddings=False,
                prefer_diversity=False,
                weight_historical=0.8,  # Heavy weight on history
                weight_task_match=0.1,
                weight_elo=0.05,
                weight_cost=0.05,
            )
        )

        # Record poor performance for claude on coding
        for _ in range(20):
            router.record_performance("claude", "coding", False)

        # Record excellent performance for llama on coding
        for _ in range(20):
            router.record_performance("llama", "coding", True)

        # Route coding task
        decision = router.route(
            task="Implement a function",
            available_agents=["claude", "llama"],
            team_size=1,
        )

        # llama should be preferred due to historical performance
        assert decision.selected_agents[0] == "llama"

    def test_routing_with_elo_updates(self):
        """Should incorporate ELO updates in routing."""
        router = AgentRouter(
            AgentRouterConfig(
                use_embeddings=False,
                prefer_diversity=False,
            )
        )

        # Give llama very high ELO
        router.update_elo("llama", 1500)
        # Give claude very low ELO
        router.update_elo("claude", 800)

        # ELO should influence routing
        decision = router.route(
            task="General task",
            available_agents=["claude", "llama"],
            team_size=1,
        )

        # Higher ELO should be reflected in scores
        assert decision.agent_scores["llama"] > decision.agent_scores["claude"]

    def test_diverse_team_selection(self):
        """Should select diverse teams when configured."""
        router = AgentRouter(
            AgentRouterConfig(
                use_embeddings=False,
                prefer_diversity=True,
            )
        )

        decision = router.route(
            task="Analyze this problem",
            available_agents=["claude", "claude-sonnet", "gpt-4", "gemini", "deepseek"],
            team_size=3,
        )

        # Should prefer agents from different providers
        providers = set()
        for agent in decision.selected_agents:
            if "claude" in agent:
                providers.add("anthropic")
            elif "gpt" in agent:
                providers.add("openai")
            elif "gemini" in agent:
                providers.add("google")
            else:
                providers.add(agent)

        # With diversity optimization, should have at least 2 providers
        assert len(providers) >= 2

    def test_to_dict_serialization(self):
        """Routing decision should be fully serializable."""
        router = AgentRouter(AgentRouterConfig(use_embeddings=False))

        decision = router.route(
            task="Test task",
            available_agents=["claude", "gpt-4"],
        )

        d = decision.to_dict()
        assert isinstance(d, dict)
        assert isinstance(d["selected_agents"], list)
        assert isinstance(d["task_type"], str)
        assert isinstance(d["confidence"], float)
        assert isinstance(d["reasoning"], list)
