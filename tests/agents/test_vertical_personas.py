"""
Tests for vertical persona framework.

Tests cover:
- Vertical and TaskComplexity enums
- VerticalConfig dataclass creation and defaults
- VERTICAL_CONFIGS constant completeness
- VerticalPersonaManager initialization
- get_vertical_config for all verticals and fallback
- get_personas_for_vertical persona retrieval
- get_persona_by_expertise filtering and sorting
- recommend_team composition, models, compliance, cost
- _select_personas_for_task task-specific mapping
- _estimate_cost_tier cost classification
- _generate_recommendation_reasoning output format
- get_compliance_frameworks per vertical
- get_typical_tasks per vertical
- detect_vertical_from_task keyword detection
- get_vertical_personas convenience function
- Edge cases (empty inputs, unknown tasks, boundary values)
"""

import pytest

from aragora.agents.personas import DEFAULT_PERSONAS, EXPERTISE_DOMAINS, Persona
from aragora.agents.vertical_personas import (
    VERTICAL_CONFIGS,
    TaskComplexity,
    Vertical,
    VerticalConfig,
    VerticalPersonaManager,
    VerticalTeamRecommendation,
    get_vertical_personas,
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestVerticalEnum:
    """Tests for the Vertical enum."""

    def test_all_verticals_defined(self):
        """All expected industry verticals are present."""
        expected = {"SOFTWARE", "LEGAL", "HEALTHCARE", "ACCOUNTING", "ACADEMIC", "GENERAL"}
        actual = {v.name for v in Vertical}
        assert actual == expected

    def test_vertical_values(self):
        """Vertical enum values are lowercase strings."""
        for v in Vertical:
            assert isinstance(v.value, str)
            assert v.value == v.value.lower()

    def test_vertical_from_value(self):
        """Verticals can be constructed from string values."""
        assert Vertical("software") is Vertical.SOFTWARE
        assert Vertical("legal") is Vertical.LEGAL
        assert Vertical("healthcare") is Vertical.HEALTHCARE
        assert Vertical("accounting") is Vertical.ACCOUNTING
        assert Vertical("academic") is Vertical.ACADEMIC
        assert Vertical("general") is Vertical.GENERAL

    def test_invalid_vertical_raises(self):
        """Invalid vertical string raises ValueError."""
        with pytest.raises(ValueError):
            Vertical("nonexistent")


class TestTaskComplexityEnum:
    """Tests for the TaskComplexity enum."""

    def test_all_complexity_levels_defined(self):
        """All expected complexity levels are present."""
        expected = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        actual = {c.name for c in TaskComplexity}
        assert actual == expected

    def test_complexity_values(self):
        """Complexity enum values are lowercase strings."""
        for c in TaskComplexity:
            assert isinstance(c.value, str)
            assert c.value == c.value.lower()


# ---------------------------------------------------------------------------
# VerticalConfig dataclass tests
# ---------------------------------------------------------------------------


class TestVerticalConfig:
    """Tests for the VerticalConfig dataclass."""

    def test_create_minimal_config(self):
        """Create a VerticalConfig with required fields only."""
        config = VerticalConfig(
            vertical=Vertical.GENERAL,
            description="Test vertical",
            primary_personas=["claude"],
            compliance_frameworks=[],
            preferred_models={TaskComplexity.LOW: ["claude"]},
            expertise_domains=["security"],
            typical_tasks=["analysis"],
        )
        assert config.vertical is Vertical.GENERAL
        assert config.default_model == "claude"
        assert config.requires_high_accuracy is False
        assert config.max_temperature == 0.7

    def test_custom_defaults(self):
        """Custom default_model, requires_high_accuracy, max_temperature."""
        config = VerticalConfig(
            vertical=Vertical.LEGAL,
            description="Legal",
            primary_personas=[],
            compliance_frameworks=[],
            preferred_models={},
            expertise_domains=[],
            typical_tasks=[],
            default_model="gpt4",
            requires_high_accuracy=True,
            max_temperature=0.3,
        )
        assert config.default_model == "gpt4"
        assert config.requires_high_accuracy is True
        assert config.max_temperature == 0.3


# ---------------------------------------------------------------------------
# VERTICAL_CONFIGS constant tests
# ---------------------------------------------------------------------------


class TestVerticalConfigs:
    """Tests for the VERTICAL_CONFIGS mapping."""

    def test_all_verticals_have_config(self):
        """Every Vertical enum member has a configuration entry."""
        for v in Vertical:
            assert v in VERTICAL_CONFIGS, f"Missing config for {v}"

    def test_configs_are_vertical_config_instances(self):
        """All config entries are VerticalConfig instances."""
        for v, config in VERTICAL_CONFIGS.items():
            assert isinstance(config, VerticalConfig)
            assert config.vertical is v

    def test_software_config_details(self):
        """Spot-check the SOFTWARE vertical configuration."""
        cfg = VERTICAL_CONFIGS[Vertical.SOFTWARE]
        assert "code_security_specialist" in cfg.primary_personas
        assert "owasp" in cfg.compliance_frameworks
        assert cfg.requires_high_accuracy is True
        assert cfg.max_temperature == 0.6

    def test_legal_config_details(self):
        """Spot-check the LEGAL vertical configuration."""
        cfg = VERTICAL_CONFIGS[Vertical.LEGAL]
        assert "contract_analyst" in cfg.primary_personas
        assert "aba_ethics" in cfg.compliance_frameworks
        assert cfg.max_temperature == 0.4

    def test_healthcare_config_details(self):
        """Spot-check the HEALTHCARE vertical configuration."""
        cfg = VERTICAL_CONFIGS[Vertical.HEALTHCARE]
        assert "hipaa_auditor" in cfg.primary_personas
        assert "hipaa" in cfg.compliance_frameworks
        assert cfg.requires_high_accuracy is True

    def test_accounting_config_details(self):
        """Spot-check the ACCOUNTING vertical configuration."""
        cfg = VERTICAL_CONFIGS[Vertical.ACCOUNTING]
        assert "financial_auditor" in cfg.primary_personas
        assert "sox" in cfg.compliance_frameworks
        assert "gaap" in cfg.compliance_frameworks

    def test_academic_config_details(self):
        """Spot-check the ACADEMIC vertical configuration."""
        cfg = VERTICAL_CONFIGS[Vertical.ACADEMIC]
        assert "peer_reviewer" in cfg.primary_personas
        assert "irb" in cfg.compliance_frameworks
        assert cfg.requires_high_accuracy is False

    def test_general_config_details(self):
        """Spot-check the GENERAL vertical configuration."""
        cfg = VERTICAL_CONFIGS[Vertical.GENERAL]
        assert "claude" in cfg.primary_personas
        assert cfg.compliance_frameworks == []
        assert cfg.max_temperature == 0.7

    def test_all_configs_have_preferred_models_for_every_complexity(self):
        """Every config defines preferred models for all complexity levels."""
        for v, cfg in VERTICAL_CONFIGS.items():
            for complexity in TaskComplexity:
                assert complexity in cfg.preferred_models, f"{v} missing models for {complexity}"


# ---------------------------------------------------------------------------
# VerticalPersonaManager tests
# ---------------------------------------------------------------------------


class TestVerticalPersonaManagerInit:
    """Tests for VerticalPersonaManager initialization."""

    def test_default_init(self):
        """Manager can be created without arguments."""
        manager = VerticalPersonaManager()
        assert manager._persona_manager is None
        assert manager._vertical_configs is VERTICAL_CONFIGS

    def test_init_with_custom_persona_manager(self):
        """Manager accepts an optional PersonaManager."""
        sentinel = object()
        manager = VerticalPersonaManager(persona_manager=sentinel)
        assert manager._persona_manager is sentinel


class TestGetVerticalConfig:
    """Tests for get_vertical_config."""

    def test_known_vertical(self):
        """Returns the correct config for a known vertical."""
        manager = VerticalPersonaManager()
        cfg = manager.get_vertical_config(Vertical.LEGAL)
        assert cfg.vertical is Vertical.LEGAL

    def test_all_verticals_return_config(self):
        """Every vertical returns a VerticalConfig."""
        manager = VerticalPersonaManager()
        for v in Vertical:
            cfg = manager.get_vertical_config(v)
            assert isinstance(cfg, VerticalConfig)


class TestGetPersonasForVertical:
    """Tests for get_personas_for_vertical."""

    def test_returns_persona_list(self):
        """Returns a list of Persona objects."""
        manager = VerticalPersonaManager()
        personas = manager.get_personas_for_vertical(Vertical.SOFTWARE)
        assert isinstance(personas, list)
        for p in personas:
            assert isinstance(p, Persona)

    def test_software_personas_subset_of_defaults(self):
        """Returned personas come from DEFAULT_PERSONAS."""
        manager = VerticalPersonaManager()
        personas = manager.get_personas_for_vertical(Vertical.SOFTWARE)
        for p in personas:
            assert p.agent_name in DEFAULT_PERSONAS

    def test_legal_returns_legal_personas(self):
        """Legal vertical includes contract_analyst."""
        manager = VerticalPersonaManager()
        personas = manager.get_personas_for_vertical(Vertical.LEGAL)
        names = [p.agent_name for p in personas]
        assert "contract_analyst" in names

    def test_healthcare_returns_clinical_personas(self):
        """Healthcare vertical includes clinical_reviewer."""
        manager = VerticalPersonaManager()
        personas = manager.get_personas_for_vertical(Vertical.HEALTHCARE)
        names = [p.agent_name for p in personas]
        assert "clinical_reviewer" in names

    def test_skips_unknown_persona_names(self):
        """Persona names not in DEFAULT_PERSONAS are silently skipped."""
        manager = VerticalPersonaManager()
        # Inject a config with a bogus persona name
        fake_config = VerticalConfig(
            vertical=Vertical.GENERAL,
            description="test",
            primary_personas=["nonexistent_persona", "claude"],
            compliance_frameworks=[],
            preferred_models={TaskComplexity.LOW: ["claude"]},
            expertise_domains=[],
            typical_tasks=[],
        )
        manager._vertical_configs = {Vertical.GENERAL: fake_config}
        personas = manager.get_personas_for_vertical(Vertical.GENERAL)
        names = [p.agent_name for p in personas]
        assert "nonexistent_persona" not in names
        assert "claude" in names


class TestGetPersonaByExpertise:
    """Tests for get_persona_by_expertise."""

    def test_returns_sorted_by_score(self):
        """Results are sorted by descending expertise score."""
        manager = VerticalPersonaManager()
        results = manager.get_persona_by_expertise(Vertical.SOFTWARE, "security", min_score=0.0)
        if len(results) >= 2:
            scores = [p.expertise.get("security", 0.0) for p in results]
            assert scores == sorted(scores, reverse=True)

    def test_min_score_filters(self):
        """Personas below min_score are excluded."""
        manager = VerticalPersonaManager()
        results = manager.get_persona_by_expertise(Vertical.SOFTWARE, "security", min_score=0.99)
        for p in results:
            assert p.expertise.get("security", 0.0) >= 0.99

    def test_no_matches_returns_empty(self):
        """No matching expertise returns empty list."""
        manager = VerticalPersonaManager()
        results = manager.get_persona_by_expertise(
            Vertical.SOFTWARE, "totally_fake_domain", min_score=0.5
        )
        assert results == []

    def test_returns_personas_with_matching_domain(self):
        """Returns personas that actually have the requested domain."""
        manager = VerticalPersonaManager()
        # "architecture" is an expertise for architecture_reviewer in SOFTWARE
        results = manager.get_persona_by_expertise(Vertical.SOFTWARE, "architecture", min_score=0.5)
        assert len(results) > 0
        for p in results:
            assert p.expertise.get("architecture", 0.0) >= 0.5


# ---------------------------------------------------------------------------
# recommend_team tests
# ---------------------------------------------------------------------------


class TestRecommendTeam:
    """Tests for recommend_team."""

    def test_returns_recommendation_dataclass(self):
        """Returns a VerticalTeamRecommendation."""
        manager = VerticalPersonaManager()
        rec = manager.recommend_team(Vertical.LEGAL, "contract_review")
        assert isinstance(rec, VerticalTeamRecommendation)

    def test_recommendation_fields(self):
        """Recommendation has all expected fields populated."""
        manager = VerticalPersonaManager()
        rec = manager.recommend_team(
            Vertical.SOFTWARE,
            "code_review",
            complexity=TaskComplexity.HIGH,
            team_size=3,
        )
        assert rec.vertical is Vertical.SOFTWARE
        assert rec.task_type == "code_review"
        assert rec.complexity is TaskComplexity.HIGH
        assert isinstance(rec.personas, list)
        assert isinstance(rec.models, list)
        assert isinstance(rec.compliance_frameworks, list)
        assert isinstance(rec.max_temperature, float)
        assert rec.estimated_cost_tier in {"low", "medium", "high"}
        assert isinstance(rec.reasoning, str) and len(rec.reasoning) > 0

    def test_team_size_respected(self):
        """Persona list length does not exceed team_size."""
        manager = VerticalPersonaManager()
        for size in (1, 2, 5):
            rec = manager.recommend_team(Vertical.LEGAL, "contract_review", team_size=size)
            assert len(rec.personas) <= size

    def test_models_truncated_to_team_size(self):
        """Models list is truncated to team_size."""
        manager = VerticalPersonaManager()
        rec = manager.recommend_team(
            Vertical.SOFTWARE,
            "code_review",
            complexity=TaskComplexity.LOW,
            team_size=1,
        )
        assert len(rec.models) <= 1

    def test_compliance_included_by_default(self):
        """Compliance frameworks are included when include_compliance=True (default)."""
        manager = VerticalPersonaManager()
        rec = manager.recommend_team(Vertical.LEGAL, "contract_review")
        assert len(rec.compliance_frameworks) > 0

    def test_compliance_excluded(self):
        """Compliance frameworks are empty when include_compliance=False."""
        manager = VerticalPersonaManager()
        rec = manager.recommend_team(Vertical.LEGAL, "contract_review", include_compliance=False)
        assert rec.compliance_frameworks == []

    def test_unknown_task_uses_primary_personas(self):
        """An unknown task type falls back to vertical primary personas."""
        manager = VerticalPersonaManager()
        rec = manager.recommend_team(Vertical.LEGAL, "unknown_task_xyz", team_size=3)
        # Should still have personas from the vertical's primary list
        assert len(rec.personas) > 0

    def test_max_temperature_matches_config(self):
        """Recommendation max_temperature matches the vertical config."""
        manager = VerticalPersonaManager()
        for v in Vertical:
            rec = manager.recommend_team(v, "analysis")
            expected = VERTICAL_CONFIGS[v].max_temperature
            assert rec.max_temperature == expected


# ---------------------------------------------------------------------------
# _estimate_cost_tier tests
# ---------------------------------------------------------------------------


class TestEstimateCostTier:
    """Tests for _estimate_cost_tier."""

    def test_low_cost(self):
        """LOW complexity with small team is low cost."""
        manager = VerticalPersonaManager()
        assert manager._estimate_cost_tier(TaskComplexity.LOW, 1) == "low"
        assert manager._estimate_cost_tier(TaskComplexity.LOW, 2) == "low"

    def test_medium_cost(self):
        """LOW or MEDIUM complexity with moderate team is medium cost."""
        manager = VerticalPersonaManager()
        assert manager._estimate_cost_tier(TaskComplexity.LOW, 3) == "medium"
        assert manager._estimate_cost_tier(TaskComplexity.MEDIUM, 2) == "medium"
        assert manager._estimate_cost_tier(TaskComplexity.MEDIUM, 3) == "medium"

    def test_high_cost(self):
        """HIGH/CRITICAL complexity or large teams are high cost."""
        manager = VerticalPersonaManager()
        assert manager._estimate_cost_tier(TaskComplexity.HIGH, 3) == "high"
        assert manager._estimate_cost_tier(TaskComplexity.CRITICAL, 1) == "high"
        assert manager._estimate_cost_tier(TaskComplexity.MEDIUM, 4) == "high"


# ---------------------------------------------------------------------------
# _generate_recommendation_reasoning tests
# ---------------------------------------------------------------------------


class TestGenerateRecommendationReasoning:
    """Tests for _generate_recommendation_reasoning."""

    def test_reasoning_contains_vertical(self):
        """Reasoning mentions the vertical."""
        manager = VerticalPersonaManager()
        reasoning = manager._generate_recommendation_reasoning(
            Vertical.LEGAL,
            "contract_review",
            TaskComplexity.HIGH,
            ["contract_analyst", "compliance_officer"],
            ["claude"],
        )
        assert "legal" in reasoning

    def test_reasoning_contains_task_type(self):
        """Reasoning mentions the task type."""
        manager = VerticalPersonaManager()
        reasoning = manager._generate_recommendation_reasoning(
            Vertical.SOFTWARE,
            "code_review",
            TaskComplexity.MEDIUM,
            ["code_quality_reviewer"],
            ["claude"],
        )
        assert "code_review" in reasoning

    def test_reasoning_contains_complexity(self):
        """Reasoning mentions the complexity level."""
        manager = VerticalPersonaManager()
        reasoning = manager._generate_recommendation_reasoning(
            Vertical.HEALTHCARE,
            "clinical_review",
            TaskComplexity.CRITICAL,
            ["clinical_reviewer"],
            ["claude"],
        )
        assert "critical" in reasoning

    def test_reasoning_contains_model(self):
        """Reasoning mentions the primary model."""
        manager = VerticalPersonaManager()
        reasoning = manager._generate_recommendation_reasoning(
            Vertical.GENERAL,
            "analysis",
            TaskComplexity.LOW,
            ["claude"],
            ["gpt4"],
        )
        assert "gpt4" in reasoning


# ---------------------------------------------------------------------------
# get_compliance_frameworks and get_typical_tasks tests
# ---------------------------------------------------------------------------


class TestComplianceAndTasks:
    """Tests for get_compliance_frameworks and get_typical_tasks."""

    def test_compliance_frameworks_returns_list(self):
        """get_compliance_frameworks returns a list of strings."""
        manager = VerticalPersonaManager()
        frameworks = manager.get_compliance_frameworks(Vertical.HEALTHCARE)
        assert isinstance(frameworks, list)
        assert all(isinstance(f, str) for f in frameworks)

    def test_general_has_no_compliance_frameworks(self):
        """GENERAL vertical has an empty compliance list."""
        manager = VerticalPersonaManager()
        frameworks = manager.get_compliance_frameworks(Vertical.GENERAL)
        assert frameworks == []

    def test_typical_tasks_returns_list(self):
        """get_typical_tasks returns a list of strings."""
        manager = VerticalPersonaManager()
        tasks = manager.get_typical_tasks(Vertical.ACCOUNTING)
        assert isinstance(tasks, list)
        assert len(tasks) > 0
        assert all(isinstance(t, str) for t in tasks)

    def test_typical_tasks_match_config(self):
        """Returned tasks match the vertical config."""
        manager = VerticalPersonaManager()
        for v in Vertical:
            tasks = manager.get_typical_tasks(v)
            assert tasks == VERTICAL_CONFIGS[v].typical_tasks


# ---------------------------------------------------------------------------
# detect_vertical_from_task tests
# ---------------------------------------------------------------------------


class TestDetectVerticalFromTask:
    """Tests for detect_vertical_from_task keyword detection."""

    def test_software_detection(self):
        """Software keywords map to SOFTWARE vertical."""
        manager = VerticalPersonaManager()
        assert manager.detect_vertical_from_task("Review this code for bugs") is Vertical.SOFTWARE

    def test_legal_detection(self):
        """Legal keywords map to LEGAL vertical."""
        manager = VerticalPersonaManager()
        assert (
            manager.detect_vertical_from_task("Analyze this contract for liability")
            is Vertical.LEGAL
        )

    def test_healthcare_detection(self):
        """Healthcare keywords map to HEALTHCARE vertical."""
        manager = VerticalPersonaManager()
        assert (
            manager.detect_vertical_from_task("Review patient medical records for HIPAA")
            is Vertical.HEALTHCARE
        )

    def test_accounting_detection(self):
        """Accounting keywords map to ACCOUNTING vertical."""
        manager = VerticalPersonaManager()
        assert (
            manager.detect_vertical_from_task("Perform a financial audit of SOX compliance")
            is Vertical.ACCOUNTING
        )

    def test_academic_detection(self):
        """Academic keywords map to ACADEMIC vertical."""
        manager = VerticalPersonaManager()
        assert (
            manager.detect_vertical_from_task("Peer review this research methodology")
            is Vertical.ACADEMIC
        )

    def test_no_keywords_returns_general(self):
        """Task with no matching keywords falls back to GENERAL."""
        manager = VerticalPersonaManager()
        assert manager.detect_vertical_from_task("What is the meaning of life?") is Vertical.GENERAL

    def test_empty_string_returns_general(self):
        """Empty task description returns GENERAL."""
        manager = VerticalPersonaManager()
        assert manager.detect_vertical_from_task("") is Vertical.GENERAL

    def test_case_insensitive(self):
        """Detection is case-insensitive."""
        manager = VerticalPersonaManager()
        assert manager.detect_vertical_from_task("REVIEW THIS CODE") is Vertical.SOFTWARE

    def test_highest_score_wins(self):
        """When multiple verticals match, the one with the most keyword hits wins."""
        manager = VerticalPersonaManager()
        # Heavy legal language should outweigh a stray keyword
        result = manager.detect_vertical_from_task(
            "Review the contract clause for litigation liability and compliance with legal agreement"
        )
        assert result is Vertical.LEGAL


# ---------------------------------------------------------------------------
# get_vertical_personas convenience function tests
# ---------------------------------------------------------------------------


class TestGetVerticalPersonasFunction:
    """Tests for the get_vertical_personas module-level function."""

    def test_returns_persona_list(self):
        """Function returns a list of Persona objects."""
        personas = get_vertical_personas(Vertical.SOFTWARE)
        assert isinstance(personas, list)
        for p in personas:
            assert isinstance(p, Persona)

    def test_matches_manager_output(self):
        """Function output matches VerticalPersonaManager.get_personas_for_vertical."""
        manager = VerticalPersonaManager()
        for v in Vertical:
            func_result = get_vertical_personas(v)
            manager_result = manager.get_personas_for_vertical(v)
            assert [p.agent_name for p in func_result] == [p.agent_name for p in manager_result]


# ---------------------------------------------------------------------------
# _select_personas_for_task tests
# ---------------------------------------------------------------------------


class TestSelectPersonasForTask:
    """Tests for the _select_personas_for_task internal method."""

    def test_known_task_selects_mapped_personas(self):
        """Known task types select from the task-persona mapping."""
        manager = VerticalPersonaManager()
        config = VERTICAL_CONFIGS[Vertical.LEGAL]
        selected = manager._select_personas_for_task(config, "contract_review", 5, False)
        assert "contract_analyst" in selected

    def test_fills_from_primary_personas(self):
        """Remaining slots are filled from vertical primary personas."""
        manager = VerticalPersonaManager()
        config = VERTICAL_CONFIGS[Vertical.SOFTWARE]
        selected = manager._select_personas_for_task(config, "code_review", 5, False)
        # Should contain task-specific plus primary fill
        assert len(selected) <= 5
        assert len(selected) > 0

    def test_compliance_persona_added_when_flag_set(self):
        """When include_compliance is True and room remains, a compliance persona is added."""
        manager = VerticalPersonaManager()
        config = VERTICAL_CONFIGS[Vertical.GENERAL]
        # Use a large team_size and an unknown task to force falling through
        selected = manager._select_personas_for_task(config, "unknown", 10, True)
        compliance_candidates = {"compliance_officer", "sox", "hipaa", "gdpr"}
        has_compliance = any(p in compliance_candidates for p in selected)
        assert has_compliance

    def test_unknown_task_uses_primaries_only(self):
        """Unknown task fills entirely from primary personas."""
        manager = VerticalPersonaManager()
        config = VERTICAL_CONFIGS[Vertical.ACADEMIC]
        selected = manager._select_personas_for_task(config, "completely_unknown", 3, False)
        for persona_name in selected:
            assert persona_name in config.primary_personas or persona_name in DEFAULT_PERSONAS


# ---------------------------------------------------------------------------
# VerticalTeamRecommendation dataclass tests
# ---------------------------------------------------------------------------


class TestVerticalTeamRecommendation:
    """Tests for the VerticalTeamRecommendation dataclass."""

    def test_create_recommendation(self):
        """Create a VerticalTeamRecommendation with all fields."""
        rec = VerticalTeamRecommendation(
            vertical=Vertical.LEGAL,
            task_type="contract_review",
            complexity=TaskComplexity.HIGH,
            personas=["contract_analyst"],
            models=["claude"],
            compliance_frameworks=["aba_ethics"],
            max_temperature=0.4,
            estimated_cost_tier="high",
            reasoning="Test reasoning",
        )
        assert rec.vertical is Vertical.LEGAL
        assert rec.task_type == "contract_review"
        assert rec.complexity is TaskComplexity.HIGH
        assert rec.personas == ["contract_analyst"]
        assert rec.models == ["claude"]
        assert rec.compliance_frameworks == ["aba_ethics"]
        assert rec.max_temperature == 0.4
        assert rec.estimated_cost_tier == "high"
        assert rec.reasoning == "Test reasoning"
