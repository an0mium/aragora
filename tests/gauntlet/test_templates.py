"""Tests for aragora.gauntlet.templates module.

Covers GauntletTemplate enum, _TEMPLATES dict, get_template, list_templates,
get_template_by_domain, and get_template_by_tags.
"""

from __future__ import annotations

import pytest

from aragora.gauntlet.templates import (
    GauntletTemplate,
    _TEMPLATES,
    get_template,
    get_template_by_domain,
    get_template_by_tags,
    list_templates,
)
from aragora.gauntlet.config import GauntletConfig, PassFailCriteria


# ---------------------------------------------------------------------------
# 1. GauntletTemplate enum has all 12 expected values
# ---------------------------------------------------------------------------

EXPECTED_MEMBERS = [
    "API_ROBUSTNESS",
    "DECISION_QUALITY",
    "COMPLIANCE_AUDIT",
    "SECURITY_ASSESSMENT",
    "ARCHITECTURE_REVIEW",
    "PROMPT_INJECTION",
    "FINANCIAL_RISK",
    "GDPR_COMPLIANCE",
    "AI_ACT_COMPLIANCE",
    "CODE_REVIEW",
    "QUICK_SANITY",
    "COMPREHENSIVE",
]

EXPECTED_VALUES = [
    "api_robustness",
    "decision_quality",
    "compliance_audit",
    "security_assessment",
    "architecture_review",
    "prompt_injection",
    "financial_risk",
    "gdpr_compliance",
    "ai_act_compliance",
    "code_review",
    "quick_sanity",
    "comprehensive",
]


class TestGauntletTemplateEnum:
    """Tests for the GauntletTemplate enum itself."""

    def test_enum_has_exactly_12_members(self):
        assert len(GauntletTemplate) == 12

    @pytest.mark.parametrize("member_name", EXPECTED_MEMBERS)
    def test_enum_has_expected_member(self, member_name: str):
        assert hasattr(GauntletTemplate, member_name)

    @pytest.mark.parametrize(
        "member_name,expected_value",
        list(zip(EXPECTED_MEMBERS, EXPECTED_VALUES)),
    )
    def test_enum_values_match(self, member_name: str, expected_value: str):
        member = GauntletTemplate[member_name]
        assert member.value == expected_value

    def test_all_member_names_accounted_for(self):
        actual_names = {m.name for m in GauntletTemplate}
        assert actual_names == set(EXPECTED_MEMBERS)


# ---------------------------------------------------------------------------
# 2. All templates in _TEMPLATES are valid GauntletConfigs
# ---------------------------------------------------------------------------


class TestTemplatesDict:
    """Tests for the _TEMPLATES dictionary."""

    def test_templates_dict_has_12_entries(self):
        assert len(_TEMPLATES) == 12

    def test_all_enum_members_have_template(self):
        for member in GauntletTemplate:
            assert member in _TEMPLATES, f"Missing template for {member.name}"

    @pytest.mark.parametrize("template", list(GauntletTemplate))
    def test_template_is_gauntlet_config(self, template: GauntletTemplate):
        config = _TEMPLATES[template]
        assert isinstance(config, GauntletConfig)

    @pytest.mark.parametrize("template", list(GauntletTemplate))
    def test_template_has_required_attributes(self, template: GauntletTemplate):
        config = _TEMPLATES[template]
        # name
        assert isinstance(config.name, str)
        assert len(config.name) > 0
        # description
        assert isinstance(config.description, str)
        assert len(config.description) > 0
        # template_id
        assert config.template_id == template.value
        # domain
        assert isinstance(config.domain, str)
        assert len(config.domain) > 0
        # input_type
        assert isinstance(config.input_type, str)
        assert len(config.input_type) > 0
        # agents non-empty
        assert isinstance(config.agents, list)
        assert len(config.agents) > 0
        # tags non-empty
        assert isinstance(config.tags, list)
        assert len(config.tags) > 0
        # criteria
        assert isinstance(config.criteria, PassFailCriteria)
        # timeout positive
        assert config.timeout_seconds > 0

    @pytest.mark.parametrize("template", list(GauntletTemplate))
    def test_template_has_attack_categories(self, template: GauntletTemplate):
        config = _TEMPLATES[template]
        assert isinstance(config.attack_categories, list)
        assert len(config.attack_categories) > 0

    @pytest.mark.parametrize("template", list(GauntletTemplate))
    def test_template_probes_configuration_valid(self, template: GauntletTemplate):
        config = _TEMPLATES[template]
        assert config.probes_per_category >= 1
        assert config.max_total_probes >= 1


# ---------------------------------------------------------------------------
# 3 & 4. get_template with enum and string arguments
# ---------------------------------------------------------------------------


class TestGetTemplate:
    """Tests for the get_template function."""

    @pytest.mark.parametrize("template", list(GauntletTemplate))
    def test_get_template_with_enum_returns_config(self, template: GauntletTemplate):
        config = get_template(template)
        assert isinstance(config, GauntletConfig)

    @pytest.mark.parametrize("template", list(GauntletTemplate))
    def test_get_template_with_enum_returns_correct_name(self, template: GauntletTemplate):
        config = get_template(template)
        original = _TEMPLATES[template]
        assert config.name == original.name

    @pytest.mark.parametrize("value", EXPECTED_VALUES)
    def test_get_template_with_string_returns_config(self, value: str):
        config = get_template(value)
        assert isinstance(config, GauntletConfig)

    def test_get_template_string_api_robustness(self):
        config = get_template("api_robustness")
        assert config.name == "API Robustness Gauntlet"
        assert config.domain == "api"
        assert config.template_id == "api_robustness"

    def test_get_template_string_security_assessment(self):
        config = get_template("security_assessment")
        assert config.name == "Security Assessment Gauntlet"
        assert config.domain == "security"

    def test_get_template_string_comprehensive(self):
        config = get_template("comprehensive")
        assert config.name == "Comprehensive Gauntlet"
        assert config.domain == "general"

    # 5. get_template with unknown string raises ValueError
    def test_get_template_unknown_string_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown template"):
            get_template("nonexistent_template")

    def test_get_template_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            get_template("")

    def test_get_template_uppercase_string_raises_value_error(self):
        # Enum values are lowercase; uppercase should not match
        with pytest.raises(ValueError):
            get_template("API_ROBUSTNESS")

    # 6. get_template returns a copy (mutating returned config doesn't affect original)
    def test_get_template_returns_copy_not_same_object(self):
        config1 = get_template(GauntletTemplate.API_ROBUSTNESS)
        config2 = get_template(GauntletTemplate.API_ROBUSTNESS)
        assert config1 is not config2

    def test_get_template_scalar_mutation_does_not_affect_original(self):
        """Scalar attribute mutations on the copy do not affect the original."""
        config = get_template(GauntletTemplate.API_ROBUSTNESS)
        original_name = _TEMPLATES[GauntletTemplate.API_ROBUSTNESS].name
        original_timeout = _TEMPLATES[GauntletTemplate.API_ROBUSTNESS].timeout_seconds

        # Mutate scalar fields on the copy
        config.name = "MUTATED NAME"
        config.timeout_seconds = 9999

        # Original scalars must be unaffected
        original = _TEMPLATES[GauntletTemplate.API_ROBUSTNESS]
        assert original.name == original_name
        assert original.timeout_seconds == original_timeout

    def test_get_template_copy_lists_are_separate_via_replacement(self):
        """Replacing list fields on the copy does not affect the original."""
        original_agents = list(_TEMPLATES[GauntletTemplate.API_ROBUSTNESS].agents)
        config = get_template(GauntletTemplate.API_ROBUSTNESS)

        # Replace the list entirely (not in-place mutation)
        config.agents = ["totally_new_agent"]

        original = _TEMPLATES[GauntletTemplate.API_ROBUSTNESS]
        assert original.agents == original_agents

    def test_get_template_copy_has_equal_data(self):
        original = _TEMPLATES[GauntletTemplate.DECISION_QUALITY]
        copy = get_template(GauntletTemplate.DECISION_QUALITY)
        assert copy.name == original.name
        assert copy.description == original.description
        assert copy.template_id == original.template_id
        assert copy.domain == original.domain
        assert copy.input_type == original.input_type
        assert copy.tags == original.tags
        assert copy.agents == original.agents
        assert copy.timeout_seconds == original.timeout_seconds
        assert copy.criteria.max_critical_findings == original.criteria.max_critical_findings
        assert copy.criteria.max_high_findings == original.criteria.max_high_findings


# ---------------------------------------------------------------------------
# 7 & 8. list_templates
# ---------------------------------------------------------------------------

REQUIRED_LIST_KEYS = {
    "id",
    "name",
    "description",
    "domain",
    "input_type",
    "tags",
    "estimated_duration_seconds",
    "criteria_level",
}


class TestListTemplates:
    """Tests for the list_templates function."""

    def test_list_templates_returns_12_items(self):
        result = list_templates()
        assert len(result) == 12

    def test_list_templates_returns_list_of_dicts(self):
        result = list_templates()
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, dict)

    def test_list_templates_each_item_has_required_keys(self):
        result = list_templates()
        for item in result:
            missing = REQUIRED_LIST_KEYS - set(item.keys())
            assert not missing, f"Item {item.get('id', '?')} missing keys: {missing}"

    def test_list_templates_ids_match_enum_values(self):
        result = list_templates()
        ids = {item["id"] for item in result}
        expected_ids = {t.value for t in GauntletTemplate}
        assert ids == expected_ids

    def test_list_templates_criteria_level_values(self):
        """criteria_level must be one of strict, standard, lenient."""
        result = list_templates()
        valid_levels = {"strict", "standard", "lenient"}
        for item in result:
            assert item["criteria_level"] in valid_levels, (
                f"Template {item['id']} has invalid criteria_level: {item['criteria_level']}"
            )

    # 8. Verify criteria_level classification logic
    def test_strict_criteria_level(self):
        """strict: max_critical=0 AND max_high=0."""
        result = list_templates()
        by_id = {item["id"]: item for item in result}

        # SECURITY_ASSESSMENT has max_critical=0, max_high=0 -> strict
        assert by_id["security_assessment"]["criteria_level"] == "strict"
        # PROMPT_INJECTION has max_critical=0, max_high=0 -> strict
        assert by_id["prompt_injection"]["criteria_level"] == "strict"
        # COMPLIANCE_AUDIT uses PassFailCriteria.strict() -> strict
        assert by_id["compliance_audit"]["criteria_level"] == "strict"
        # FINANCIAL_RISK uses PassFailCriteria.strict() -> strict
        assert by_id["financial_risk"]["criteria_level"] == "strict"
        # GDPR_COMPLIANCE uses PassFailCriteria.strict() -> strict
        assert by_id["gdpr_compliance"]["criteria_level"] == "strict"
        # AI_ACT_COMPLIANCE uses PassFailCriteria.strict() -> strict
        assert by_id["ai_act_compliance"]["criteria_level"] == "strict"

    def test_standard_criteria_level(self):
        """standard: max_high <= 3 (but not strict)."""
        result = list_templates()
        by_id = {item["id"]: item for item in result}

        # API_ROBUSTNESS: max_critical=0, max_high=2 -> standard
        assert by_id["api_robustness"]["criteria_level"] == "standard"
        # DECISION_QUALITY: max_critical=0, max_high=3 -> standard
        assert by_id["decision_quality"]["criteria_level"] == "standard"
        # ARCHITECTURE_REVIEW: max_critical=0, max_high=2 -> standard
        assert by_id["architecture_review"]["criteria_level"] == "standard"
        # CODE_REVIEW: max_critical=0, max_high=2 -> standard
        assert by_id["code_review"]["criteria_level"] == "standard"
        # COMPREHENSIVE: max_critical=0, max_high=3 -> standard
        assert by_id["comprehensive"]["criteria_level"] == "standard"

    def test_lenient_criteria_level(self):
        """lenient: max_high > 3."""
        result = list_templates()
        by_id = {item["id"]: item for item in result}

        # QUICK_SANITY uses PassFailCriteria.lenient() -> max_high=5 -> lenient
        assert by_id["quick_sanity"]["criteria_level"] == "lenient"

    def test_criteria_level_classification_matches_criteria_values(self):
        """Cross-check criteria_level against actual PassFailCriteria values."""
        result = list_templates()
        by_id = {item["id"]: item for item in result}

        for template in GauntletTemplate:
            config = _TEMPLATES[template]
            criteria = config.criteria
            item = by_id[template.value]

            if criteria.max_critical_findings == 0 and criteria.max_high_findings == 0:
                expected_level = "strict"
            elif criteria.max_high_findings <= 3:
                expected_level = "standard"
            else:
                expected_level = "lenient"

            assert item["criteria_level"] == expected_level, (
                f"Template {template.value}: expected {expected_level}, "
                f"got {item['criteria_level']} "
                f"(max_critical={criteria.max_critical_findings}, "
                f"max_high={criteria.max_high_findings})"
            )

    def test_list_templates_estimated_duration_positive(self):
        result = list_templates()
        for item in result:
            assert item["estimated_duration_seconds"] > 0, (
                f"Template {item['id']} has non-positive duration"
            )

    def test_list_templates_tags_are_lists(self):
        result = list_templates()
        for item in result:
            assert isinstance(item["tags"], list)
            assert len(item["tags"]) > 0


# ---------------------------------------------------------------------------
# 9, 10, 11. get_template_by_domain
# ---------------------------------------------------------------------------


class TestGetTemplateByDomain:
    """Tests for the get_template_by_domain function."""

    def test_security_domain_returns_expected_templates(self):
        result = get_template_by_domain("security")
        result_set = set(result)
        assert GauntletTemplate.SECURITY_ASSESSMENT in result_set
        assert GauntletTemplate.PROMPT_INJECTION in result_set
        assert len(result) == 2

    def test_compliance_domain_returns_expected_templates(self):
        result = get_template_by_domain("compliance")
        result_set = set(result)
        assert GauntletTemplate.COMPLIANCE_AUDIT in result_set
        assert GauntletTemplate.GDPR_COMPLIANCE in result_set
        assert GauntletTemplate.AI_ACT_COMPLIANCE in result_set
        assert len(result) == 3

    def test_api_domain(self):
        result = get_template_by_domain("api")
        assert result == [GauntletTemplate.API_ROBUSTNESS]

    def test_strategy_domain(self):
        result = get_template_by_domain("strategy")
        assert result == [GauntletTemplate.DECISION_QUALITY]

    def test_architecture_domain(self):
        result = get_template_by_domain("architecture")
        assert result == [GauntletTemplate.ARCHITECTURE_REVIEW]

    def test_finance_domain(self):
        result = get_template_by_domain("finance")
        assert result == [GauntletTemplate.FINANCIAL_RISK]

    def test_code_domain(self):
        result = get_template_by_domain("code")
        assert result == [GauntletTemplate.CODE_REVIEW]

    def test_general_domain(self):
        result = get_template_by_domain("general")
        result_set = set(result)
        assert GauntletTemplate.QUICK_SANITY in result_set
        assert GauntletTemplate.COMPREHENSIVE in result_set
        assert len(result) == 2

    def test_unknown_domain_returns_empty_list(self):
        result = get_template_by_domain("nonexistent_domain")
        assert result == []

    def test_empty_domain_returns_empty_list(self):
        result = get_template_by_domain("")
        assert result == []

    def test_returns_list_of_gauntlet_templates(self):
        result = get_template_by_domain("security")
        for item in result:
            assert isinstance(item, GauntletTemplate)


# ---------------------------------------------------------------------------
# 12 & 13. get_template_by_tags
# ---------------------------------------------------------------------------


class TestGetTemplateByTags:
    """Tests for the get_template_by_tags function."""

    def test_security_tag_returns_templates_with_security(self):
        result = get_template_by_tags(["security"])
        result_set = set(result)
        # SECURITY_ASSESSMENT has tags: ["security", "red-team", "vulnerability"]
        assert GauntletTemplate.SECURITY_ASSESSMENT in result_set
        # CODE_REVIEW has tags: ["code", "review", "bugs", "security"]
        assert GauntletTemplate.CODE_REVIEW in result_set

    def test_compliance_tag(self):
        result = get_template_by_tags(["compliance"])
        result_set = set(result)
        assert GauntletTemplate.COMPLIANCE_AUDIT in result_set
        assert GauntletTemplate.GDPR_COMPLIANCE in result_set
        assert GauntletTemplate.AI_ACT_COMPLIANCE in result_set

    def test_multiple_tags_match_any(self):
        """get_template_by_tags matches templates that have ANY of the given tags."""
        result = get_template_by_tags(["finance", "architecture"])
        result_set = set(result)
        assert GauntletTemplate.FINANCIAL_RISK in result_set
        assert GauntletTemplate.ARCHITECTURE_REVIEW in result_set

    def test_no_matching_tags_returns_empty(self):
        result = get_template_by_tags(["does_not_exist_xyz"])
        assert result == []

    def test_empty_tags_returns_empty(self):
        result = get_template_by_tags([])
        assert result == []

    def test_returns_list_of_gauntlet_templates(self):
        result = get_template_by_tags(["api"])
        for item in result:
            assert isinstance(item, GauntletTemplate)

    def test_gdpr_tag(self):
        result = get_template_by_tags(["gdpr"])
        assert GauntletTemplate.GDPR_COMPLIANCE in result

    def test_ai_act_tag(self):
        result = get_template_by_tags(["ai-act"])
        assert GauntletTemplate.AI_ACT_COMPLIANCE in result

    def test_quick_tag(self):
        result = get_template_by_tags(["quick"])
        assert GauntletTemplate.QUICK_SANITY in result

    def test_comprehensive_tag(self):
        result = get_template_by_tags(["comprehensive"])
        assert GauntletTemplate.COMPREHENSIVE in result

    def test_prompt_tag(self):
        result = get_template_by_tags(["prompt"])
        assert GauntletTemplate.PROMPT_INJECTION in result

    def test_robustness_tag(self):
        result = get_template_by_tags(["robustness"])
        assert GauntletTemplate.API_ROBUSTNESS in result

    def test_red_team_tag(self):
        result = get_template_by_tags(["red-team"])
        assert GauntletTemplate.SECURITY_ASSESSMENT in result

    def test_overlapping_results_deduplicated(self):
        """When a template matches multiple tags, it should appear only once."""
        # SECURITY_ASSESSMENT has ["security", "red-team", "vulnerability"]
        result = get_template_by_tags(["security", "red-team", "vulnerability"])
        count = sum(1 for t in result if t == GauntletTemplate.SECURITY_ASSESSMENT)
        assert count == 1


# ---------------------------------------------------------------------------
# 14. Each template has required attributes (comprehensive per-template checks)
# ---------------------------------------------------------------------------


class TestTemplateAttributeCompleteness:
    """Ensure every template in _TEMPLATES has all expected attributes populated."""

    @pytest.mark.parametrize("template", list(GauntletTemplate))
    def test_template_id_matches_enum_value(self, template: GauntletTemplate):
        config = _TEMPLATES[template]
        assert config.template_id == template.value

    @pytest.mark.parametrize("template", list(GauntletTemplate))
    def test_agents_non_empty_strings(self, template: GauntletTemplate):
        config = _TEMPLATES[template]
        for agent in config.agents:
            assert isinstance(agent, str)
            assert len(agent) > 0

    @pytest.mark.parametrize("template", list(GauntletTemplate))
    def test_tags_non_empty_strings(self, template: GauntletTemplate):
        config = _TEMPLATES[template]
        for tag in config.tags:
            assert isinstance(tag, str)
            assert len(tag) > 0

    @pytest.mark.parametrize("template", list(GauntletTemplate))
    def test_criteria_has_valid_thresholds(self, template: GauntletTemplate):
        config = _TEMPLATES[template]
        c = config.criteria
        assert c.max_critical_findings >= 0
        assert c.max_high_findings >= 0
        assert 0.0 <= c.min_robustness_score <= 1.0
        assert 0.0 <= c.min_confidence <= 1.0

    @pytest.mark.parametrize("template", list(GauntletTemplate))
    def test_timeout_positive(self, template: GauntletTemplate):
        config = _TEMPLATES[template]
        assert config.timeout_seconds > 0

    @pytest.mark.parametrize("template", list(GauntletTemplate))
    def test_max_total_probes_positive(self, template: GauntletTemplate):
        config = _TEMPLATES[template]
        assert config.max_total_probes > 0

    @pytest.mark.parametrize("template", list(GauntletTemplate))
    def test_config_round_trips_via_dict(self, template: GauntletTemplate):
        """to_dict -> from_dict should produce equivalent config."""
        config = _TEMPLATES[template]
        roundtripped = GauntletConfig.from_dict(config.to_dict())
        assert roundtripped.name == config.name
        assert roundtripped.template_id == config.template_id
        assert roundtripped.domain == config.domain
        assert roundtripped.agents == config.agents
        assert roundtripped.tags == config.tags
        assert roundtripped.criteria.max_critical_findings == config.criteria.max_critical_findings
        assert roundtripped.criteria.max_high_findings == config.criteria.max_high_findings
