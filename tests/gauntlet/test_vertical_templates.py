"""Tests for aragora.gauntlet.vertical_templates module.

Covers:
- VerticalDomain enum values and membership
- ComplianceMapping dataclass creation and defaults
- VerticalTemplate dataclass creation, defaults, and serialization
- Pre-built templates (GDPR, HIPAA, SOC2, PCI-DSS, AI Governance)
- Template registry (VERTICAL_TEMPLATES)
- get_template() lookup
- list_templates() summary generation
- get_templates_for_domain() filtering
- create_custom_template() with and without base templates
- Edge cases (invalid IDs, missing params, empty inputs)
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from aragora.gauntlet.vertical_templates import (
    AI_GOVERNANCE_MAPPINGS,
    GDPR_MAPPINGS,
    HIPAA_MAPPINGS,
    PCIDSS_MAPPINGS,
    SOC2_MAPPINGS,
    TEMPLATE_AI_GOVERNANCE,
    TEMPLATE_GDPR,
    TEMPLATE_HIPAA,
    TEMPLATE_PCI_DSS,
    TEMPLATE_SOC2,
    VERTICAL_TEMPLATES,
    ComplianceMapping,
    VerticalDomain,
    VerticalTemplate,
    create_custom_template,
    get_template,
    get_templates_for_domain,
    list_templates,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_mapping() -> ComplianceMapping:
    """A minimal ComplianceMapping for testing."""
    return ComplianceMapping(
        framework="TestFramework",
        control_id="T-01",
        control_name="Test Control",
        description="A test compliance control",
    )


@pytest.fixture
def sample_mapping_weighted() -> ComplianceMapping:
    """A ComplianceMapping with custom severity_weight."""
    return ComplianceMapping(
        framework="TestFramework",
        control_id="T-02",
        control_name="Weighted Control",
        description="A weighted test control",
        severity_weight=2.5,
    )


@pytest.fixture
def sample_template(sample_mapping: ComplianceMapping) -> VerticalTemplate:
    """A minimal VerticalTemplate for testing."""
    return VerticalTemplate(
        id="test-template",
        name="Test Template",
        domain=VerticalDomain.CUSTOM,
        description="A test template for unit testing",
        personas=["tester", "analyst"],
        priority_categories=["testing", "validation"],
        compliance_mappings=[sample_mapping],
        report_sections=["summary", "findings"],
        severity_thresholds={"critical": 0.9, "high": 0.7, "medium": 0.4, "low": 0.1},
        metadata={"author": "test_suite"},
    )


@pytest.fixture
def all_template_ids() -> list[str]:
    """All registered template IDs."""
    return [
        "gdpr-compliance",
        "hipaa-healthcare",
        "soc2-trust",
        "pci-dss-payment",
        "ai-ml-governance",
    ]


@pytest.fixture
def all_prebuilt_templates() -> list[VerticalTemplate]:
    """All pre-built template objects."""
    return [
        TEMPLATE_GDPR,
        TEMPLATE_HIPAA,
        TEMPLATE_SOC2,
        TEMPLATE_PCI_DSS,
        TEMPLATE_AI_GOVERNANCE,
    ]


# ===========================================================================
# 1. VerticalDomain Enum
# ===========================================================================


EXPECTED_DOMAIN_MEMBERS = [
    "GDPR",
    "HIPAA",
    "SOC2",
    "PCI_DSS",
    "ISO_27001",
    "FEDRAMP",
    "FINRA",
    "AI_GOVERNANCE",
    "CUSTOM",
]

EXPECTED_DOMAIN_VALUES = [
    "gdpr",
    "hipaa",
    "soc2",
    "pci_dss",
    "iso_27001",
    "fedramp",
    "finra",
    "ai_governance",
    "custom",
]


class TestVerticalDomainEnum:
    """Tests for the VerticalDomain enum."""

    def test_enum_has_exactly_9_members(self):
        assert len(VerticalDomain) == 9

    @pytest.mark.parametrize("member_name", EXPECTED_DOMAIN_MEMBERS)
    def test_enum_has_expected_member(self, member_name: str):
        assert hasattr(VerticalDomain, member_name)

    @pytest.mark.parametrize(
        "member_name,expected_value",
        list(zip(EXPECTED_DOMAIN_MEMBERS, EXPECTED_DOMAIN_VALUES)),
    )
    def test_enum_values_match(self, member_name: str, expected_value: str):
        member = VerticalDomain[member_name]
        assert member.value == expected_value

    def test_all_member_names_accounted_for(self):
        actual_names = {m.name for m in VerticalDomain}
        assert actual_names == set(EXPECTED_DOMAIN_MEMBERS)

    def test_domain_is_string_enum(self):
        """VerticalDomain inherits from str, so members are also strings."""
        for member in VerticalDomain:
            assert isinstance(member, str)
            assert member == member.value

    def test_domain_can_be_constructed_from_value(self):
        assert VerticalDomain("gdpr") == VerticalDomain.GDPR
        assert VerticalDomain("hipaa") == VerticalDomain.HIPAA
        assert VerticalDomain("custom") == VerticalDomain.CUSTOM

    def test_domain_invalid_value_raises(self):
        with pytest.raises(ValueError):
            VerticalDomain("nonexistent")


# ===========================================================================
# 2. ComplianceMapping Dataclass
# ===========================================================================


class TestComplianceMapping:
    """Tests for the ComplianceMapping dataclass."""

    def test_creation_with_defaults(self, sample_mapping: ComplianceMapping):
        assert sample_mapping.framework == "TestFramework"
        assert sample_mapping.control_id == "T-01"
        assert sample_mapping.control_name == "Test Control"
        assert sample_mapping.description == "A test compliance control"
        assert sample_mapping.severity_weight == 1.0

    def test_creation_with_custom_weight(self, sample_mapping_weighted: ComplianceMapping):
        assert sample_mapping_weighted.severity_weight == 2.5

    def test_severity_weight_default_is_one(self):
        m = ComplianceMapping(
            framework="F",
            control_id="C-1",
            control_name="Control",
            description="Desc",
        )
        assert m.severity_weight == 1.0

    def test_fields_are_stored_correctly(self):
        m = ComplianceMapping(
            framework="ISO 27001",
            control_id="A.9.1.1",
            control_name="Access control policy",
            description="Access control policy description",
            severity_weight=1.8,
        )
        assert m.framework == "ISO 27001"
        assert m.control_id == "A.9.1.1"
        assert m.control_name == "Access control policy"
        assert m.severity_weight == 1.8


# ===========================================================================
# 3. VerticalTemplate Dataclass
# ===========================================================================


class TestVerticalTemplate:
    """Tests for the VerticalTemplate dataclass."""

    def test_creation_with_all_fields(self, sample_template: VerticalTemplate):
        assert sample_template.id == "test-template"
        assert sample_template.name == "Test Template"
        assert sample_template.domain == VerticalDomain.CUSTOM
        assert sample_template.description == "A test template for unit testing"
        assert sample_template.personas == ["tester", "analyst"]
        assert sample_template.priority_categories == ["testing", "validation"]
        assert len(sample_template.compliance_mappings) == 1
        assert sample_template.report_sections == ["summary", "findings"]
        assert sample_template.severity_thresholds == {
            "critical": 0.9,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.1,
        }
        assert sample_template.metadata == {"author": "test_suite"}

    def test_creation_minimal_defaults(self):
        """Creating a VerticalTemplate with only required fields uses defaults."""
        t = VerticalTemplate(
            id="minimal",
            name="Minimal",
            domain=VerticalDomain.CUSTOM,
            description="Minimal template",
        )
        assert t.personas == []
        assert t.priority_categories == []
        assert t.compliance_mappings == []
        assert t.report_sections == []
        assert t.severity_thresholds == {}
        assert t.metadata == {}

    def test_default_factory_independence(self):
        """Default mutable fields do not share state between instances."""
        t1 = VerticalTemplate(id="t1", name="T1", domain=VerticalDomain.CUSTOM, description="D1")
        t2 = VerticalTemplate(id="t2", name="T2", domain=VerticalDomain.CUSTOM, description="D2")
        t1.personas.append("persona_only_in_t1")
        assert "persona_only_in_t1" not in t2.personas
        assert t1.personas is not t2.personas

    def test_to_dict_returns_dict(self, sample_template: VerticalTemplate):
        result = sample_template.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_has_all_keys(self, sample_template: VerticalTemplate):
        result = sample_template.to_dict()
        expected_keys = {
            "id",
            "name",
            "domain",
            "description",
            "personas",
            "priority_categories",
            "compliance_mappings",
            "report_sections",
            "severity_thresholds",
            "metadata",
        }
        assert set(result.keys()) == expected_keys

    def test_to_dict_domain_is_string_value(self, sample_template: VerticalTemplate):
        result = sample_template.to_dict()
        assert result["domain"] == "custom"
        assert isinstance(result["domain"], str)

    def test_to_dict_compliance_mappings_serialized(self, sample_template: VerticalTemplate):
        result = sample_template.to_dict()
        mappings = result["compliance_mappings"]
        assert isinstance(mappings, list)
        assert len(mappings) == 1
        m = mappings[0]
        assert m["framework"] == "TestFramework"
        assert m["control_id"] == "T-01"
        assert m["control_name"] == "Test Control"
        assert m["description"] == "A test compliance control"
        assert m["severity_weight"] == 1.0

    def test_to_dict_compliance_mappings_multiple(self):
        """Verify multiple compliance mappings serialize correctly."""
        mappings = [
            ComplianceMapping("F1", "C1", "Name1", "Desc1", 1.0),
            ComplianceMapping("F2", "C2", "Name2", "Desc2", 2.0),
            ComplianceMapping("F3", "C3", "Name3", "Desc3", 3.0),
        ]
        t = VerticalTemplate(
            id="multi",
            name="Multi",
            domain=VerticalDomain.GDPR,
            description="Multi-mapping template",
            compliance_mappings=mappings,
        )
        result = t.to_dict()
        assert len(result["compliance_mappings"]) == 3
        assert result["compliance_mappings"][0]["framework"] == "F1"
        assert result["compliance_mappings"][1]["severity_weight"] == 2.0
        assert result["compliance_mappings"][2]["control_id"] == "C3"

    def test_to_dict_empty_lists(self):
        """A template with all empty lists serializes cleanly."""
        t = VerticalTemplate(
            id="empty",
            name="Empty",
            domain=VerticalDomain.CUSTOM,
            description="Empty template",
        )
        result = t.to_dict()
        assert result["personas"] == []
        assert result["priority_categories"] == []
        assert result["compliance_mappings"] == []
        assert result["report_sections"] == []
        assert result["severity_thresholds"] == {}
        assert result["metadata"] == {}

    def test_to_dict_preserves_metadata(self):
        t = VerticalTemplate(
            id="meta",
            name="Meta",
            domain=VerticalDomain.CUSTOM,
            description="Metadata test",
            metadata={"key1": "value1", "nested": {"a": 1}},
        )
        result = t.to_dict()
        assert result["metadata"] == {"key1": "value1", "nested": {"a": 1}}


# ===========================================================================
# 4. Pre-built Compliance Mapping Lists
# ===========================================================================


class TestPrebuiltMappings:
    """Tests for the module-level compliance mapping lists."""

    @pytest.mark.parametrize(
        "mappings,framework,expected_count",
        [
            (GDPR_MAPPINGS, "GDPR", 6),
            (HIPAA_MAPPINGS, "HIPAA", 6),
            (SOC2_MAPPINGS, "SOC 2", 6),
            (PCIDSS_MAPPINGS, "PCI-DSS", 6),
            (AI_GOVERNANCE_MAPPINGS, "AI Governance", 6),
        ],
    )
    def test_mapping_list_has_expected_count(
        self,
        mappings: list[ComplianceMapping],
        framework: str,
        expected_count: int,
    ):
        assert len(mappings) == expected_count

    @pytest.mark.parametrize(
        "mappings,framework",
        [
            (GDPR_MAPPINGS, "GDPR"),
            (HIPAA_MAPPINGS, "HIPAA"),
            (SOC2_MAPPINGS, "SOC 2"),
            (PCIDSS_MAPPINGS, "PCI-DSS"),
            (AI_GOVERNANCE_MAPPINGS, "AI Governance"),
        ],
    )
    def test_all_mappings_have_correct_framework(
        self,
        mappings: list[ComplianceMapping],
        framework: str,
    ):
        for m in mappings:
            assert m.framework == framework

    @pytest.mark.parametrize(
        "mappings",
        [GDPR_MAPPINGS, HIPAA_MAPPINGS, SOC2_MAPPINGS, PCIDSS_MAPPINGS, AI_GOVERNANCE_MAPPINGS],
    )
    def test_all_mappings_are_compliance_mapping_instances(self, mappings: list[ComplianceMapping]):
        for m in mappings:
            assert isinstance(m, ComplianceMapping)

    @pytest.mark.parametrize(
        "mappings",
        [GDPR_MAPPINGS, HIPAA_MAPPINGS, SOC2_MAPPINGS, PCIDSS_MAPPINGS, AI_GOVERNANCE_MAPPINGS],
    )
    def test_all_mappings_have_nonempty_fields(self, mappings: list[ComplianceMapping]):
        for m in mappings:
            assert len(m.control_id) > 0
            assert len(m.control_name) > 0
            assert len(m.description) > 0

    @pytest.mark.parametrize(
        "mappings",
        [GDPR_MAPPINGS, HIPAA_MAPPINGS, SOC2_MAPPINGS, PCIDSS_MAPPINGS, AI_GOVERNANCE_MAPPINGS],
    )
    def test_all_mappings_have_positive_severity_weight(self, mappings: list[ComplianceMapping]):
        for m in mappings:
            assert m.severity_weight > 0

    @pytest.mark.parametrize(
        "mappings",
        [GDPR_MAPPINGS, HIPAA_MAPPINGS, SOC2_MAPPINGS, PCIDSS_MAPPINGS, AI_GOVERNANCE_MAPPINGS],
    )
    def test_mappings_have_unique_control_ids(self, mappings: list[ComplianceMapping]):
        control_ids = [m.control_id for m in mappings]
        assert len(control_ids) == len(set(control_ids)), "Duplicate control_ids found"


# ===========================================================================
# 5. Pre-built Templates (per-vertical)
# ===========================================================================


class TestTemplateGDPR:
    """Tests for the GDPR pre-built template."""

    def test_id(self):
        assert TEMPLATE_GDPR.id == "gdpr-compliance"

    def test_name(self):
        assert TEMPLATE_GDPR.name == "GDPR Compliance Audit"

    def test_domain(self):
        assert TEMPLATE_GDPR.domain == VerticalDomain.GDPR

    def test_description_nonempty(self):
        assert len(TEMPLATE_GDPR.description) > 0
        assert "GDPR" in TEMPLATE_GDPR.description

    def test_personas(self):
        assert "gdpr_auditor" in TEMPLATE_GDPR.personas
        assert "privacy_advocate" in TEMPLATE_GDPR.personas
        assert "security_analyst" in TEMPLATE_GDPR.personas
        assert len(TEMPLATE_GDPR.personas) == 3

    def test_priority_categories(self):
        expected = [
            "data_privacy",
            "consent_management",
            "data_retention",
            "cross_border_transfer",
            "breach_notification",
            "data_subject_rights",
        ]
        assert TEMPLATE_GDPR.priority_categories == expected

    def test_compliance_mappings_are_gdpr(self):
        assert TEMPLATE_GDPR.compliance_mappings is GDPR_MAPPINGS

    def test_report_sections(self):
        assert "executive_summary" in TEMPLATE_GDPR.report_sections
        assert "recommendations" in TEMPLATE_GDPR.report_sections
        assert len(TEMPLATE_GDPR.report_sections) == 7

    def test_severity_thresholds(self):
        t = TEMPLATE_GDPR.severity_thresholds
        assert t["critical"] == 0.9
        assert t["high"] == 0.7
        assert t["medium"] == 0.4
        assert t["low"] == 0.1

    def test_metadata(self):
        assert "regulatory_authority" in TEMPLATE_GDPR.metadata
        assert "max_fine" in TEMPLATE_GDPR.metadata
        assert "compliance_deadline" in TEMPLATE_GDPR.metadata


class TestTemplateHIPAA:
    """Tests for the HIPAA pre-built template."""

    def test_id(self):
        assert TEMPLATE_HIPAA.id == "hipaa-healthcare"

    def test_name(self):
        assert TEMPLATE_HIPAA.name == "HIPAA Healthcare Compliance"

    def test_domain(self):
        assert TEMPLATE_HIPAA.domain == VerticalDomain.HIPAA

    def test_description_nonempty(self):
        assert len(TEMPLATE_HIPAA.description) > 0
        assert "HIPAA" in TEMPLATE_HIPAA.description

    def test_personas(self):
        assert "hipaa_auditor" in TEMPLATE_HIPAA.personas
        assert len(TEMPLATE_HIPAA.personas) == 3

    def test_priority_categories(self):
        assert "phi_protection" in TEMPLATE_HIPAA.priority_categories
        assert "access_control" in TEMPLATE_HIPAA.priority_categories
        assert len(TEMPLATE_HIPAA.priority_categories) == 6

    def test_compliance_mappings_are_hipaa(self):
        assert TEMPLATE_HIPAA.compliance_mappings is HIPAA_MAPPINGS

    def test_severity_thresholds(self):
        t = TEMPLATE_HIPAA.severity_thresholds
        assert t["critical"] == 0.85
        assert t["high"] == 0.65

    def test_metadata_covered_entities(self):
        assert "covered_entities" in TEMPLATE_HIPAA.metadata
        assert isinstance(TEMPLATE_HIPAA.metadata["covered_entities"], list)
        assert len(TEMPLATE_HIPAA.metadata["covered_entities"]) == 3


class TestTemplateSOC2:
    """Tests for the SOC 2 pre-built template."""

    def test_id(self):
        assert TEMPLATE_SOC2.id == "soc2-trust"

    def test_name(self):
        assert TEMPLATE_SOC2.name == "SOC 2 Trust Services"

    def test_domain(self):
        assert TEMPLATE_SOC2.domain == VerticalDomain.SOC2

    def test_description_nonempty(self):
        assert len(TEMPLATE_SOC2.description) > 0
        assert "SOC 2" in TEMPLATE_SOC2.description

    def test_personas(self):
        assert "soc2_auditor" in TEMPLATE_SOC2.personas
        assert len(TEMPLATE_SOC2.personas) == 3

    def test_priority_categories(self):
        expected_categories = {
            "security",
            "availability",
            "processing_integrity",
            "confidentiality",
            "privacy",
            "change_management",
        }
        assert set(TEMPLATE_SOC2.priority_categories) == expected_categories

    def test_compliance_mappings_are_soc2(self):
        assert TEMPLATE_SOC2.compliance_mappings is SOC2_MAPPINGS

    def test_metadata_trust_criteria(self):
        assert "trust_criteria" in TEMPLATE_SOC2.metadata
        trust = TEMPLATE_SOC2.metadata["trust_criteria"]
        assert len(trust) == 5
        assert "Security" in trust
        assert "Privacy" in trust


class TestTemplatePCIDSS:
    """Tests for the PCI-DSS pre-built template."""

    def test_id(self):
        assert TEMPLATE_PCI_DSS.id == "pci-dss-payment"

    def test_name(self):
        assert TEMPLATE_PCI_DSS.name == "PCI-DSS Payment Security"

    def test_domain(self):
        assert TEMPLATE_PCI_DSS.domain == VerticalDomain.PCI_DSS

    def test_description_nonempty(self):
        assert len(TEMPLATE_PCI_DSS.description) > 0
        assert (
            "PCI" in TEMPLATE_PCI_DSS.description
            or "payment" in TEMPLATE_PCI_DSS.description.lower()
        )

    def test_personas(self):
        assert "pci_auditor" in TEMPLATE_PCI_DSS.personas
        assert "penetration_tester" in TEMPLATE_PCI_DSS.personas
        assert len(TEMPLATE_PCI_DSS.personas) == 3

    def test_priority_categories(self):
        assert "cardholder_data" in TEMPLATE_PCI_DSS.priority_categories
        assert "encryption" in TEMPLATE_PCI_DSS.priority_categories
        assert len(TEMPLATE_PCI_DSS.priority_categories) == 6

    def test_compliance_mappings_are_pcidss(self):
        assert TEMPLATE_PCI_DSS.compliance_mappings is PCIDSS_MAPPINGS

    def test_severity_thresholds_stricter(self):
        """PCI-DSS has stricter thresholds (higher critical threshold)."""
        t = TEMPLATE_PCI_DSS.severity_thresholds
        assert t["critical"] == 0.95
        assert t["high"] == 0.8

    def test_metadata_compliance_levels(self):
        assert "compliance_levels" in TEMPLATE_PCI_DSS.metadata
        levels = TEMPLATE_PCI_DSS.metadata["compliance_levels"]
        assert len(levels) == 4
        assert "Level 1" in levels


class TestTemplateAIGovernance:
    """Tests for the AI/ML Governance pre-built template."""

    def test_id(self):
        assert TEMPLATE_AI_GOVERNANCE.id == "ai-ml-governance"

    def test_name(self):
        assert TEMPLATE_AI_GOVERNANCE.name == "AI/ML Governance"

    def test_domain(self):
        assert TEMPLATE_AI_GOVERNANCE.domain == VerticalDomain.AI_GOVERNANCE

    def test_description_nonempty(self):
        assert len(TEMPLATE_AI_GOVERNANCE.description) > 0
        assert "AI" in TEMPLATE_AI_GOVERNANCE.description

    def test_personas(self):
        assert "ai_ethicist" in TEMPLATE_AI_GOVERNANCE.personas
        assert "fairness_auditor" in TEMPLATE_AI_GOVERNANCE.personas
        assert len(TEMPLATE_AI_GOVERNANCE.personas) == 4

    def test_priority_categories(self):
        assert "bias_fairness" in TEMPLATE_AI_GOVERNANCE.priority_categories
        assert "explainability" in TEMPLATE_AI_GOVERNANCE.priority_categories
        assert "safety" in TEMPLATE_AI_GOVERNANCE.priority_categories
        assert len(TEMPLATE_AI_GOVERNANCE.priority_categories) == 6

    def test_compliance_mappings_are_ai_governance(self):
        assert TEMPLATE_AI_GOVERNANCE.compliance_mappings is AI_GOVERNANCE_MAPPINGS

    def test_metadata_frameworks(self):
        assert "frameworks" in TEMPLATE_AI_GOVERNANCE.metadata
        frameworks = TEMPLATE_AI_GOVERNANCE.metadata["frameworks"]
        assert "EU AI Act" in frameworks
        assert "NIST AI RMF" in frameworks

    def test_metadata_risk_categories(self):
        assert "risk_categories" in TEMPLATE_AI_GOVERNANCE.metadata
        categories = TEMPLATE_AI_GOVERNANCE.metadata["risk_categories"]
        assert "Unacceptable" in categories
        assert "High" in categories


# ===========================================================================
# 6. Template Registry (VERTICAL_TEMPLATES dict)
# ===========================================================================


class TestVerticalTemplatesRegistry:
    """Tests for the VERTICAL_TEMPLATES registry dict."""

    def test_registry_has_5_entries(self):
        assert len(VERTICAL_TEMPLATES) == 5

    def test_registry_keys_match_template_ids(self, all_template_ids: list[str]):
        assert set(VERTICAL_TEMPLATES.keys()) == set(all_template_ids)

    def test_registry_values_are_vertical_template_instances(self):
        for key, val in VERTICAL_TEMPLATES.items():
            assert isinstance(val, VerticalTemplate), f"{key} is not a VerticalTemplate"

    def test_registry_key_matches_template_id_field(self):
        for key, template in VERTICAL_TEMPLATES.items():
            assert key == template.id, (
                f"Registry key '{key}' does not match template.id '{template.id}'"
            )

    def test_registry_contains_gdpr(self):
        assert "gdpr-compliance" in VERTICAL_TEMPLATES
        assert VERTICAL_TEMPLATES["gdpr-compliance"] is TEMPLATE_GDPR

    def test_registry_contains_hipaa(self):
        assert "hipaa-healthcare" in VERTICAL_TEMPLATES
        assert VERTICAL_TEMPLATES["hipaa-healthcare"] is TEMPLATE_HIPAA

    def test_registry_contains_soc2(self):
        assert "soc2-trust" in VERTICAL_TEMPLATES
        assert VERTICAL_TEMPLATES["soc2-trust"] is TEMPLATE_SOC2

    def test_registry_contains_pci_dss(self):
        assert "pci-dss-payment" in VERTICAL_TEMPLATES
        assert VERTICAL_TEMPLATES["pci-dss-payment"] is TEMPLATE_PCI_DSS

    def test_registry_contains_ai_governance(self):
        assert "ai-ml-governance" in VERTICAL_TEMPLATES
        assert VERTICAL_TEMPLATES["ai-ml-governance"] is TEMPLATE_AI_GOVERNANCE

    def test_all_templates_have_unique_domains(self):
        """Each pre-built template covers a distinct domain."""
        domains = [t.domain for t in VERTICAL_TEMPLATES.values()]
        assert len(domains) == len(set(domains))

    def test_all_templates_have_nonempty_personas(self):
        for template_id, template in VERTICAL_TEMPLATES.items():
            assert len(template.personas) > 0, f"Template '{template_id}' has no personas"

    def test_all_templates_have_nonempty_priority_categories(self):
        for template_id, template in VERTICAL_TEMPLATES.items():
            assert len(template.priority_categories) > 0, (
                f"Template '{template_id}' has no priority_categories"
            )

    def test_all_templates_have_nonempty_compliance_mappings(self):
        for template_id, template in VERTICAL_TEMPLATES.items():
            assert len(template.compliance_mappings) > 0, (
                f"Template '{template_id}' has no compliance_mappings"
            )

    def test_all_templates_have_nonempty_report_sections(self):
        for template_id, template in VERTICAL_TEMPLATES.items():
            assert len(template.report_sections) > 0, (
                f"Template '{template_id}' has no report_sections"
            )

    def test_all_templates_have_severity_thresholds(self):
        for template_id, template in VERTICAL_TEMPLATES.items():
            assert len(template.severity_thresholds) > 0, (
                f"Template '{template_id}' has no severity_thresholds"
            )

    def test_all_templates_serializable_via_to_dict(self):
        for template_id, template in VERTICAL_TEMPLATES.items():
            d = template.to_dict()
            assert isinstance(d, dict)
            assert d["id"] == template_id


# ===========================================================================
# 7. get_template() function
# ===========================================================================


class TestGetTemplate:
    """Tests for the get_template lookup function."""

    @pytest.mark.parametrize(
        "template_id",
        [
            "gdpr-compliance",
            "hipaa-healthcare",
            "soc2-trust",
            "pci-dss-payment",
            "ai-ml-governance",
        ],
    )
    def test_get_template_returns_correct_instance(self, template_id: str):
        result = get_template(template_id)
        assert result is not None
        assert result.id == template_id

    def test_get_template_returns_same_object_from_registry(self):
        result = get_template("gdpr-compliance")
        assert result is TEMPLATE_GDPR

    def test_get_template_returns_none_for_unknown_id(self):
        result = get_template("nonexistent-template")
        assert result is None

    def test_get_template_returns_none_for_empty_string(self):
        result = get_template("")
        assert result is None

    def test_get_template_case_sensitive(self):
        """get_template is case-sensitive; uppercase should not match."""
        result = get_template("GDPR-COMPLIANCE")
        assert result is None

    def test_get_template_partial_match_fails(self):
        """Partial IDs should not match."""
        result = get_template("gdpr")
        assert result is None
        result = get_template("hipaa")
        assert result is None

    def test_get_template_with_extra_whitespace_fails(self):
        result = get_template(" gdpr-compliance ")
        assert result is None

    def test_get_template_returns_vertical_template_type(self):
        result = get_template("soc2-trust")
        assert isinstance(result, VerticalTemplate)


# ===========================================================================
# 8. list_templates() function
# ===========================================================================


REQUIRED_LIST_KEYS = {
    "id",
    "name",
    "domain",
    "description",
    "personas",
    "categories_count",
    "mappings_count",
}


class TestListTemplates:
    """Tests for the list_templates function."""

    def test_returns_list(self):
        result = list_templates()
        assert isinstance(result, list)

    def test_returns_5_items(self):
        result = list_templates()
        assert len(result) == 5

    def test_each_item_is_dict(self):
        result = list_templates()
        for item in result:
            assert isinstance(item, dict)

    def test_each_item_has_required_keys(self):
        result = list_templates()
        for item in result:
            missing = REQUIRED_LIST_KEYS - set(item.keys())
            assert not missing, f"Item {item.get('id', '?')} missing keys: {missing}"

    def test_ids_match_registry(self):
        result = list_templates()
        ids = {item["id"] for item in result}
        expected_ids = set(VERTICAL_TEMPLATES.keys())
        assert ids == expected_ids

    def test_domains_are_string_values(self):
        result = list_templates()
        for item in result:
            assert isinstance(item["domain"], str)
            # The domain value should be the enum .value string
            assert item["domain"] in {d.value for d in VerticalDomain}

    def test_personas_are_lists(self):
        result = list_templates()
        for item in result:
            assert isinstance(item["personas"], list)
            assert len(item["personas"]) > 0

    def test_categories_count_positive(self):
        result = list_templates()
        for item in result:
            assert isinstance(item["categories_count"], int)
            assert item["categories_count"] > 0

    def test_mappings_count_positive(self):
        result = list_templates()
        for item in result:
            assert isinstance(item["mappings_count"], int)
            assert item["mappings_count"] > 0

    def test_categories_count_matches_template(self):
        """categories_count matches len(priority_categories) of original template."""
        result = list_templates()
        by_id = {item["id"]: item for item in result}
        for template_id, template in VERTICAL_TEMPLATES.items():
            assert by_id[template_id]["categories_count"] == len(template.priority_categories)

    def test_mappings_count_matches_template(self):
        """mappings_count matches len(compliance_mappings) of original template."""
        result = list_templates()
        by_id = {item["id"]: item for item in result}
        for template_id, template in VERTICAL_TEMPLATES.items():
            assert by_id[template_id]["mappings_count"] == len(template.compliance_mappings)

    def test_description_nonempty(self):
        result = list_templates()
        for item in result:
            assert isinstance(item["description"], str)
            assert len(item["description"]) > 0

    def test_name_nonempty(self):
        result = list_templates()
        for item in result:
            assert isinstance(item["name"], str)
            assert len(item["name"]) > 0


# ===========================================================================
# 9. get_templates_for_domain() function
# ===========================================================================


class TestGetTemplatesForDomain:
    """Tests for the get_templates_for_domain function."""

    @pytest.mark.parametrize(
        "domain,expected_id",
        [
            (VerticalDomain.GDPR, "gdpr-compliance"),
            (VerticalDomain.HIPAA, "hipaa-healthcare"),
            (VerticalDomain.SOC2, "soc2-trust"),
            (VerticalDomain.PCI_DSS, "pci-dss-payment"),
            (VerticalDomain.AI_GOVERNANCE, "ai-ml-governance"),
        ],
    )
    def test_returns_template_for_domain(self, domain: VerticalDomain, expected_id: str):
        result = get_templates_for_domain(domain)
        assert len(result) >= 1
        ids = [t.id for t in result]
        assert expected_id in ids

    @pytest.mark.parametrize(
        "domain,expected_id",
        [
            (VerticalDomain.GDPR, "gdpr-compliance"),
            (VerticalDomain.HIPAA, "hipaa-healthcare"),
            (VerticalDomain.SOC2, "soc2-trust"),
            (VerticalDomain.PCI_DSS, "pci-dss-payment"),
            (VerticalDomain.AI_GOVERNANCE, "ai-ml-governance"),
        ],
    )
    def test_returns_exactly_one_for_each_used_domain(
        self, domain: VerticalDomain, expected_id: str
    ):
        """Each domain has exactly one pre-built template."""
        result = get_templates_for_domain(domain)
        assert len(result) == 1
        assert result[0].id == expected_id

    def test_unused_domain_returns_empty(self):
        """Domains without pre-built templates return empty list."""
        result = get_templates_for_domain(VerticalDomain.ISO_27001)
        assert result == []

    def test_fedramp_domain_returns_empty(self):
        result = get_templates_for_domain(VerticalDomain.FEDRAMP)
        assert result == []

    def test_finra_domain_returns_empty(self):
        result = get_templates_for_domain(VerticalDomain.FINRA)
        assert result == []

    def test_custom_domain_returns_empty(self):
        """No pre-built templates use the CUSTOM domain."""
        result = get_templates_for_domain(VerticalDomain.CUSTOM)
        assert result == []

    def test_returns_list_of_vertical_templates(self):
        result = get_templates_for_domain(VerticalDomain.GDPR)
        for item in result:
            assert isinstance(item, VerticalTemplate)

    def test_all_returned_templates_match_domain(self):
        for domain in VerticalDomain:
            result = get_templates_for_domain(domain)
            for t in result:
                assert t.domain == domain


# ===========================================================================
# 10. create_custom_template() function
# ===========================================================================


class TestCreateCustomTemplate:
    """Tests for the create_custom_template function."""

    def test_basic_creation(self):
        t = create_custom_template(
            id="custom-test",
            name="Custom Test",
            description="A custom test template",
            personas=["tester"],
            priority_categories=["testing"],
        )
        assert isinstance(t, VerticalTemplate)
        assert t.id == "custom-test"
        assert t.name == "Custom Test"
        assert t.domain == VerticalDomain.CUSTOM
        assert t.description == "A custom test template"
        assert t.personas == ["tester"]
        assert t.priority_categories == ["testing"]

    def test_domain_always_custom(self):
        """Custom templates always have CUSTOM domain regardless of base."""
        t = create_custom_template(
            id="custom-from-gdpr",
            name="Custom GDPR",
            description="Based on GDPR",
            personas=["auditor"],
            priority_categories=["privacy"],
            base_template="gdpr-compliance",
        )
        assert t.domain == VerticalDomain.CUSTOM

    def test_without_base_has_default_report_sections(self):
        """Without a base template, default report sections are used."""
        t = create_custom_template(
            id="custom-default",
            name="Custom Default",
            description="No base template",
            personas=["analyst"],
            priority_categories=["general"],
        )
        assert t.report_sections == ["executive_summary", "findings", "recommendations"]

    def test_without_base_has_default_severity_thresholds(self):
        """Without a base template, default severity thresholds are used."""
        t = create_custom_template(
            id="custom-default",
            name="Custom Default",
            description="No base template",
            personas=["analyst"],
            priority_categories=["general"],
        )
        expected = {"critical": 0.9, "high": 0.7, "medium": 0.4, "low": 0.1}
        assert t.severity_thresholds == expected

    def test_without_base_has_empty_mappings(self):
        """Without a base template, compliance mappings are empty."""
        t = create_custom_template(
            id="custom-no-mappings",
            name="Custom No Mappings",
            description="No mappings",
            personas=["analyst"],
            priority_categories=["general"],
        )
        assert t.compliance_mappings == []

    def test_without_base_has_empty_metadata(self):
        """Without a base template, metadata is empty."""
        t = create_custom_template(
            id="custom-no-meta",
            name="Custom No Meta",
            description="No metadata",
            personas=["analyst"],
            priority_categories=["general"],
        )
        assert t.metadata == {}

    def test_with_base_inherits_compliance_mappings(self):
        """With a base template, compliance mappings are inherited."""
        t = create_custom_template(
            id="custom-gdpr",
            name="Custom GDPR",
            description="Based on GDPR",
            personas=["custom_auditor"],
            priority_categories=["custom_privacy"],
            base_template="gdpr-compliance",
        )
        assert len(t.compliance_mappings) == len(GDPR_MAPPINGS)
        for orig, inherited in zip(GDPR_MAPPINGS, t.compliance_mappings):
            assert orig.control_id == inherited.control_id
            assert orig.framework == inherited.framework

    def test_with_base_inherits_report_sections(self):
        """With a base template, report sections are inherited."""
        t = create_custom_template(
            id="custom-hipaa",
            name="Custom HIPAA",
            description="Based on HIPAA",
            personas=["custom_auditor"],
            priority_categories=["custom_phi"],
            base_template="hipaa-healthcare",
        )
        assert t.report_sections == TEMPLATE_HIPAA.report_sections

    def test_with_base_inherits_severity_thresholds(self):
        """With a base template, severity thresholds are inherited."""
        t = create_custom_template(
            id="custom-pci",
            name="Custom PCI",
            description="Based on PCI-DSS",
            personas=["custom_auditor"],
            priority_categories=["custom_payment"],
            base_template="pci-dss-payment",
        )
        assert t.severity_thresholds == TEMPLATE_PCI_DSS.severity_thresholds

    def test_with_base_inherits_metadata(self):
        """With a base template, metadata is inherited."""
        t = create_custom_template(
            id="custom-soc2",
            name="Custom SOC2",
            description="Based on SOC2",
            personas=["custom_auditor"],
            priority_categories=["custom_security"],
            base_template="soc2-trust",
        )
        assert t.metadata == TEMPLATE_SOC2.metadata

    def test_with_base_uses_custom_personas(self):
        """Custom personas override the base template's personas."""
        custom_personas = ["my_auditor", "my_analyst"]
        t = create_custom_template(
            id="custom-personas",
            name="Custom Personas",
            description="Custom personas",
            personas=custom_personas,
            priority_categories=["general"],
            base_template="gdpr-compliance",
        )
        assert t.personas == custom_personas
        assert t.personas != TEMPLATE_GDPR.personas

    def test_with_base_uses_custom_priority_categories(self):
        """Custom priority_categories override the base template's categories."""
        custom_cats = ["cat_a", "cat_b"]
        t = create_custom_template(
            id="custom-cats",
            name="Custom Categories",
            description="Custom categories",
            personas=["analyst"],
            priority_categories=custom_cats,
            base_template="hipaa-healthcare",
        )
        assert t.priority_categories == custom_cats
        assert t.priority_categories != TEMPLATE_HIPAA.priority_categories

    def test_with_invalid_base_template_returns_defaults(self):
        """If base_template ID is invalid, defaults are used (not an error)."""
        t = create_custom_template(
            id="custom-invalid-base",
            name="Custom Invalid Base",
            description="Invalid base",
            personas=["analyst"],
            priority_categories=["general"],
            base_template="nonexistent-template",
        )
        # Should fall back to defaults (empty mappings, default sections)
        assert t.compliance_mappings == []
        assert t.report_sections == ["executive_summary", "findings", "recommendations"]
        assert t.severity_thresholds == {
            "critical": 0.9,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.1,
        }
        assert t.metadata == {}

    def test_with_none_base_template(self):
        """base_template=None means no inheritance."""
        t = create_custom_template(
            id="custom-none-base",
            name="Custom None Base",
            description="None base",
            personas=["analyst"],
            priority_categories=["general"],
            base_template=None,
        )
        assert t.compliance_mappings == []
        assert t.report_sections == ["executive_summary", "findings", "recommendations"]

    def test_inherited_mappings_are_separate_copy(self):
        """Inherited compliance_mappings list should be a new list (not same object)."""
        t = create_custom_template(
            id="custom-copy-test",
            name="Copy Test",
            description="Test copy",
            personas=["analyst"],
            priority_categories=["general"],
            base_template="gdpr-compliance",
        )
        assert t.compliance_mappings is not TEMPLATE_GDPR.compliance_mappings
        assert t.compliance_mappings == TEMPLATE_GDPR.compliance_mappings

    def test_inherited_report_sections_are_separate_copy(self):
        """Inherited report_sections list should be a new list."""
        t = create_custom_template(
            id="custom-sections-copy",
            name="Sections Copy",
            description="Test sections copy",
            personas=["analyst"],
            priority_categories=["general"],
            base_template="hipaa-healthcare",
        )
        assert t.report_sections is not TEMPLATE_HIPAA.report_sections

    def test_inherited_thresholds_are_separate_copy(self):
        """Inherited severity_thresholds dict should be a new dict."""
        t = create_custom_template(
            id="custom-thresholds-copy",
            name="Thresholds Copy",
            description="Test thresholds copy",
            personas=["analyst"],
            priority_categories=["general"],
            base_template="soc2-trust",
        )
        assert t.severity_thresholds is not TEMPLATE_SOC2.severity_thresholds

    def test_inherited_metadata_is_separate_copy(self):
        """Inherited metadata dict should be a new dict."""
        t = create_custom_template(
            id="custom-metadata-copy",
            name="Metadata Copy",
            description="Test metadata copy",
            personas=["analyst"],
            priority_categories=["general"],
            base_template="pci-dss-payment",
        )
        assert t.metadata is not TEMPLATE_PCI_DSS.metadata

    @pytest.mark.parametrize(
        "base_id",
        [
            "gdpr-compliance",
            "hipaa-healthcare",
            "soc2-trust",
            "pci-dss-payment",
            "ai-ml-governance",
        ],
    )
    def test_create_custom_from_each_base(self, base_id: str):
        """Custom template creation works with every pre-built base."""
        t = create_custom_template(
            id=f"custom-from-{base_id}",
            name=f"Custom from {base_id}",
            description=f"Custom template based on {base_id}",
            personas=["custom_persona"],
            priority_categories=["custom_category"],
            base_template=base_id,
        )
        base = VERTICAL_TEMPLATES[base_id]
        assert t.domain == VerticalDomain.CUSTOM
        assert len(t.compliance_mappings) == len(base.compliance_mappings)
        assert t.severity_thresholds == base.severity_thresholds
        assert t.metadata == base.metadata


# ===========================================================================
# 11. Template Composition / Combination
# ===========================================================================


class TestTemplateComposition:
    """Tests for composing and combining templates."""

    def test_merge_personas_from_two_templates(self):
        """Simulate merging personas from two templates into a custom one."""
        combined_personas = list(set(TEMPLATE_GDPR.personas + TEMPLATE_HIPAA.personas))
        t = create_custom_template(
            id="merged-privacy",
            name="Merged Privacy",
            description="GDPR + HIPAA personas",
            personas=combined_personas,
            priority_categories=["privacy"],
        )
        for persona in TEMPLATE_GDPR.personas:
            assert persona in t.personas
        for persona in TEMPLATE_HIPAA.personas:
            assert persona in t.personas

    def test_merge_categories_from_two_templates(self):
        """Simulate merging priority categories from two templates."""
        combined_categories = list(
            set(TEMPLATE_SOC2.priority_categories + TEMPLATE_PCI_DSS.priority_categories)
        )
        t = create_custom_template(
            id="merged-security",
            name="Merged Security",
            description="SOC2 + PCI-DSS categories",
            personas=["security_analyst"],
            priority_categories=combined_categories,
        )
        for cat in TEMPLATE_SOC2.priority_categories:
            assert cat in t.priority_categories
        for cat in TEMPLATE_PCI_DSS.priority_categories:
            assert cat in t.priority_categories

    def test_custom_template_to_dict_round_trip(self):
        """A custom template can be serialized and has correct structure."""
        t = create_custom_template(
            id="rt-test",
            name="Round Trip Test",
            description="Testing serialization",
            personas=["tester"],
            priority_categories=["testing"],
            base_template="gdpr-compliance",
        )
        d = t.to_dict()
        assert d["id"] == "rt-test"
        assert d["domain"] == "custom"
        assert d["personas"] == ["tester"]
        assert d["priority_categories"] == ["testing"]
        assert len(d["compliance_mappings"]) == len(GDPR_MAPPINGS)

    def test_chain_inheritance_simulation(self):
        """
        Simulate chaining: create a custom template, then use it as conceptual base.

        Since create_custom_template only accepts registry IDs, we simulate
        by manually composing from a custom template's fields.
        """
        # Step 1: Create first custom template based on GDPR
        t1 = create_custom_template(
            id="layer1",
            name="Layer 1",
            description="First layer",
            personas=["layer1_persona"],
            priority_categories=["layer1_category"],
            base_template="gdpr-compliance",
        )
        # Step 2: Manually create second layer using t1's fields
        t2 = VerticalTemplate(
            id="layer2",
            name="Layer 2",
            domain=VerticalDomain.CUSTOM,
            description="Second layer",
            personas=["layer2_persona"],
            priority_categories=["layer2_category"],
            compliance_mappings=list(t1.compliance_mappings),
            report_sections=list(t1.report_sections),
            severity_thresholds=dict(t1.severity_thresholds),
            metadata=dict(t1.metadata),
        )
        # Layer 2 should carry GDPR mappings from layer 1
        assert len(t2.compliance_mappings) == len(GDPR_MAPPINGS)
        assert t2.severity_thresholds == TEMPLATE_GDPR.severity_thresholds


# ===========================================================================
# 12. Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases and boundary tests."""

    def test_get_template_with_none_like_string(self):
        result = get_template("None")
        assert result is None

    def test_get_template_with_special_characters(self):
        result = get_template("gdpr-compliance;DROP TABLE")
        assert result is None

    def test_empty_personas_list(self):
        """Template with empty personas is valid."""
        t = VerticalTemplate(
            id="no-personas",
            name="No Personas",
            domain=VerticalDomain.CUSTOM,
            description="No personas",
            personas=[],
        )
        assert t.personas == []
        d = t.to_dict()
        assert d["personas"] == []

    def test_empty_priority_categories(self):
        """Template with empty priority_categories is valid."""
        t = VerticalTemplate(
            id="no-categories",
            name="No Categories",
            domain=VerticalDomain.CUSTOM,
            description="No categories",
            priority_categories=[],
        )
        assert t.priority_categories == []

    def test_empty_compliance_mappings(self):
        """Template with empty compliance_mappings serializes correctly."""
        t = VerticalTemplate(
            id="no-mappings",
            name="No Mappings",
            domain=VerticalDomain.CUSTOM,
            description="No mappings",
        )
        d = t.to_dict()
        assert d["compliance_mappings"] == []

    def test_to_dict_with_complex_metadata(self):
        """Metadata can contain nested dicts and lists."""
        t = VerticalTemplate(
            id="complex-meta",
            name="Complex Meta",
            domain=VerticalDomain.CUSTOM,
            description="Complex metadata",
            metadata={
                "nested": {"deep": {"value": 42}},
                "list_value": [1, 2, 3],
                "bool_value": True,
                "null_value": None,
            },
        )
        d = t.to_dict()
        assert d["metadata"]["nested"]["deep"]["value"] == 42
        assert d["metadata"]["list_value"] == [1, 2, 3]
        assert d["metadata"]["bool_value"] is True
        assert d["metadata"]["null_value"] is None

    def test_severity_weight_zero(self):
        """A compliance mapping with severity_weight of 0 is valid."""
        m = ComplianceMapping(
            framework="Test",
            control_id="Z-0",
            control_name="Zero Weight",
            description="Zero weight control",
            severity_weight=0.0,
        )
        assert m.severity_weight == 0.0

    def test_severity_weight_negative(self):
        """Negative severity_weight is technically allowed (no validation)."""
        m = ComplianceMapping(
            framework="Test",
            control_id="N-1",
            control_name="Negative Weight",
            description="Negative weight control",
            severity_weight=-1.0,
        )
        assert m.severity_weight == -1.0

    def test_list_templates_returns_fresh_list_each_time(self):
        """Each call to list_templates returns a new list."""
        r1 = list_templates()
        r2 = list_templates()
        assert r1 is not r2
        assert r1 == r2

    def test_get_templates_for_domain_returns_fresh_list(self):
        """Each call to get_templates_for_domain returns a new list."""
        r1 = get_templates_for_domain(VerticalDomain.GDPR)
        r2 = get_templates_for_domain(VerticalDomain.GDPR)
        assert r1 is not r2
        assert len(r1) == len(r2)

    def test_create_custom_with_empty_personas(self):
        t = create_custom_template(
            id="empty-personas",
            name="Empty Personas",
            description="Empty",
            personas=[],
            priority_categories=["general"],
        )
        assert t.personas == []

    def test_create_custom_with_empty_categories(self):
        t = create_custom_template(
            id="empty-cats",
            name="Empty Cats",
            description="Empty",
            personas=["analyst"],
            priority_categories=[],
        )
        assert t.priority_categories == []

    def test_create_custom_with_many_personas(self):
        many_personas = [f"persona_{i}" for i in range(100)]
        t = create_custom_template(
            id="many-personas",
            name="Many Personas",
            description="Many",
            personas=many_personas,
            priority_categories=["general"],
        )
        assert len(t.personas) == 100

    def test_template_with_unicode_fields(self):
        """Unicode content in fields is valid."""
        t = VerticalTemplate(
            id="unicode-test",
            name="Test de conformite RGPD",
            domain=VerticalDomain.CUSTOM,
            description="Verification de conformite pour le RGPD europeen",
            personas=["auditeur_rgpd"],
            metadata={"region": "Europe"},
        )
        d = t.to_dict()
        assert "RGPD" in d["name"]
        assert "europeen" in d["description"]


# ===========================================================================
# 13. Specific GDPR Mapping Details
# ===========================================================================


class TestGDPRMappingDetails:
    """Detailed tests for GDPR compliance mappings."""

    def test_article_5_mapping(self):
        m = GDPR_MAPPINGS[0]
        assert m.control_id == "Art. 5"
        assert m.control_name == "Principles relating to processing"
        assert m.severity_weight == 1.5

    def test_article_33_mapping(self):
        m = GDPR_MAPPINGS[5]
        assert m.control_id == "Art. 33"
        assert m.control_name == "Notification of breach"
        assert m.severity_weight == 1.6

    def test_all_gdpr_control_ids(self):
        expected_ids = {"Art. 5", "Art. 6", "Art. 17", "Art. 25", "Art. 32", "Art. 33"}
        actual_ids = {m.control_id for m in GDPR_MAPPINGS}
        assert actual_ids == expected_ids


# ===========================================================================
# 14. Specific HIPAA Mapping Details
# ===========================================================================


class TestHIPAAMappingDetails:
    """Detailed tests for HIPAA compliance mappings."""

    def test_access_control_mapping(self):
        m = HIPAA_MAPPINGS[0]
        assert m.control_id == "164.312(a)(1)"
        assert m.control_name == "Access Control"
        assert m.severity_weight == 1.5

    def test_transmission_security_mapping(self):
        m = HIPAA_MAPPINGS[4]
        assert m.control_id == "164.312(e)(1)"
        assert m.control_name == "Transmission Security"
        assert m.severity_weight == 1.6

    def test_all_hipaa_control_ids(self):
        expected_ids = {
            "164.312(a)(1)",
            "164.312(b)",
            "164.312(c)(1)",
            "164.312(d)",
            "164.312(e)(1)",
            "164.308(a)(6)",
        }
        actual_ids = {m.control_id for m in HIPAA_MAPPINGS}
        assert actual_ids == expected_ids


# ===========================================================================
# 15. Specific AI Governance Mapping Details
# ===========================================================================


class TestAIGovernanceMappingDetails:
    """Detailed tests for AI Governance compliance mappings."""

    def test_bias_detection_mapping(self):
        m = AI_GOVERNANCE_MAPPINGS[0]
        assert m.control_id == "BIAS-01"
        assert m.control_name == "Bias Detection"
        assert m.severity_weight == 1.5

    def test_safety_guardrails_mapping(self):
        m = AI_GOVERNANCE_MAPPINGS[3]
        assert m.control_id == "SAFE-01"
        assert m.control_name == "Safety Guardrails"
        assert m.severity_weight == 1.6

    def test_all_ai_governance_control_ids(self):
        expected_ids = {"BIAS-01", "TRANS-01", "FAIR-01", "SAFE-01", "PRIV-01", "AUDIT-01"}
        actual_ids = {m.control_id for m in AI_GOVERNANCE_MAPPINGS}
        assert actual_ids == expected_ids


# ===========================================================================
# 16. Cross-Template Consistency
# ===========================================================================


class TestCrossTemplateConsistency:
    """Tests verifying consistency across all pre-built templates."""

    def test_all_templates_have_executive_summary_section(self):
        """Every pre-built template should include an executive_summary report section."""
        for template_id, template in VERTICAL_TEMPLATES.items():
            assert "executive_summary" in template.report_sections, (
                f"Template '{template_id}' missing executive_summary"
            )

    def test_all_templates_have_recommendations_section(self):
        """Every pre-built template should include a recommendations report section."""
        for template_id, template in VERTICAL_TEMPLATES.items():
            assert "recommendations" in template.report_sections, (
                f"Template '{template_id}' missing recommendations"
            )

    def test_all_templates_have_four_severity_levels(self):
        """Every pre-built template should define critical/high/medium/low thresholds."""
        expected_keys = {"critical", "high", "medium", "low"}
        for template_id, template in VERTICAL_TEMPLATES.items():
            assert set(template.severity_thresholds.keys()) == expected_keys, (
                f"Template '{template_id}' has unexpected severity keys: "
                f"{set(template.severity_thresholds.keys())}"
            )

    def test_severity_thresholds_ordered(self):
        """For all templates, critical > high > medium > low."""
        for template_id, template in VERTICAL_TEMPLATES.items():
            t = template.severity_thresholds
            assert t["critical"] > t["high"] > t["medium"] > t["low"], (
                f"Template '{template_id}' thresholds not in descending order: {t}"
            )

    def test_severity_thresholds_between_zero_and_one(self):
        """All threshold values should be between 0 and 1 (inclusive)."""
        for template_id, template in VERTICAL_TEMPLATES.items():
            for level, value in template.severity_thresholds.items():
                assert 0.0 <= value <= 1.0, (
                    f"Template '{template_id}' threshold '{level}' = {value} out of [0,1]"
                )

    def test_all_templates_have_at_least_3_personas(self):
        for template_id, template in VERTICAL_TEMPLATES.items():
            assert len(template.personas) >= 3, (
                f"Template '{template_id}' has fewer than 3 personas"
            )

    def test_all_templates_have_6_mappings(self):
        for template_id, template in VERTICAL_TEMPLATES.items():
            assert len(template.compliance_mappings) == 6, (
                f"Template '{template_id}' has {len(template.compliance_mappings)} mappings, expected 6"
            )

    def test_all_templates_have_nonempty_metadata(self):
        for template_id, template in VERTICAL_TEMPLATES.items():
            assert len(template.metadata) > 0, f"Template '{template_id}' has empty metadata"

    def test_no_duplicate_template_names(self):
        names = [t.name for t in VERTICAL_TEMPLATES.values()]
        assert len(names) == len(set(names)), "Duplicate template names found"

    def test_no_duplicate_template_ids(self):
        ids = [t.id for t in VERTICAL_TEMPLATES.values()]
        assert len(ids) == len(set(ids)), "Duplicate template IDs found"
