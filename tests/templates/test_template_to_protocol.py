"""
Tests for the template_to_protocol function.

Tests cover:
- Protocol conversion from templates
- Round calculation from phases
- Topology selection based on template structure
- Protocol configuration overrides
- Consensus threshold mapping
"""

import pytest

from aragora.debate.protocol import DebateProtocol
from aragora.templates import (
    CODE_REVIEW_TEMPLATE,
    DESIGN_DOC_TEMPLATE,
    INCIDENT_RESPONSE_TEMPLATE,
    RESEARCH_SYNTHESIS_TEMPLATE,
    SECURITY_AUDIT_TEMPLATE,
    ARCHITECTURE_REVIEW_TEMPLATE,
    HEALTHCARE_COMPLIANCE_TEMPLATE,
    FINANCIAL_RISK_TEMPLATE,
    DebatePhase,
    DebateRole,
    DebateTemplate,
    TemplateType,
    template_to_protocol,
)


class TestProtocolCreation:
    """Tests for basic protocol creation from templates."""

    def test_returns_debate_protocol(self):
        """Test that template_to_protocol returns a DebateProtocol."""
        protocol = template_to_protocol(CODE_REVIEW_TEMPLATE)
        assert isinstance(protocol, DebateProtocol)

    def test_code_review_protocol(self):
        """Test protocol creation from CODE_REVIEW template."""
        protocol = template_to_protocol(CODE_REVIEW_TEMPLATE)

        assert protocol.rounds >= CODE_REVIEW_TEMPLATE.max_rounds
        assert protocol.consensus_threshold == CODE_REVIEW_TEMPLATE.consensus_threshold

    def test_design_doc_protocol(self):
        """Test protocol creation from DESIGN_DOC template."""
        protocol = template_to_protocol(DESIGN_DOC_TEMPLATE)

        assert protocol.rounds >= DESIGN_DOC_TEMPLATE.max_rounds
        assert protocol.consensus_threshold == DESIGN_DOC_TEMPLATE.consensus_threshold

    def test_incident_response_protocol(self):
        """Test protocol creation from INCIDENT_RESPONSE template."""
        protocol = template_to_protocol(INCIDENT_RESPONSE_TEMPLATE)

        assert protocol.rounds >= INCIDENT_RESPONSE_TEMPLATE.max_rounds
        assert protocol.consensus_threshold == INCIDENT_RESPONSE_TEMPLATE.consensus_threshold

    def test_research_synthesis_protocol(self):
        """Test protocol creation from RESEARCH_SYNTHESIS template."""
        protocol = template_to_protocol(RESEARCH_SYNTHESIS_TEMPLATE)

        assert protocol.consensus_threshold == RESEARCH_SYNTHESIS_TEMPLATE.consensus_threshold

    def test_security_audit_protocol(self):
        """Test protocol creation from SECURITY_AUDIT template."""
        protocol = template_to_protocol(SECURITY_AUDIT_TEMPLATE)

        assert protocol.consensus_threshold == SECURITY_AUDIT_TEMPLATE.consensus_threshold

    def test_all_templates_convertible(self):
        """Test that all predefined templates can be converted."""
        templates = [
            CODE_REVIEW_TEMPLATE,
            DESIGN_DOC_TEMPLATE,
            INCIDENT_RESPONSE_TEMPLATE,
            RESEARCH_SYNTHESIS_TEMPLATE,
            SECURITY_AUDIT_TEMPLATE,
            ARCHITECTURE_REVIEW_TEMPLATE,
            HEALTHCARE_COMPLIANCE_TEMPLATE,
            FINANCIAL_RISK_TEMPLATE,
        ]

        for template in templates:
            protocol = template_to_protocol(template)
            assert isinstance(protocol, DebateProtocol)


class TestRoundCalculation:
    """Tests for round calculation from template phases."""

    @pytest.fixture
    def minimal_template(self):
        """Create a minimal template for testing."""
        role = DebateRole(
            name="test",
            description="test",
            objectives=[],
            evaluation_criteria=[],
        )
        return DebateTemplate(
            template_id="test",
            template_type=TemplateType.CODE_REVIEW,
            name="Test",
            description="Test",
            roles=[role],
            phases=[],
            recommended_agents=1,
            max_rounds=3,
            consensus_threshold=0.5,
            rubric={},
            output_format="",
            domain="test",
        )

    def test_rounds_from_single_phase(self, minimal_template):
        """Test round calculation with a single phase."""
        minimal_template.phases = [
            DebatePhase(
                name="test",
                description="test",
                duration_rounds=2,
                roles_active=[],
                objectives=[],
                outputs=[],
            )
        ]

        protocol = template_to_protocol(minimal_template)

        # Should use max of phase total (2) and max_rounds (3)
        assert protocol.rounds >= 2

    def test_rounds_from_multiple_phases(self, minimal_template):
        """Test round calculation with multiple phases."""
        minimal_template.phases = [
            DebatePhase(
                name="phase1",
                description="test",
                duration_rounds=2,
                roles_active=[],
                objectives=[],
                outputs=[],
            ),
            DebatePhase(
                name="phase2",
                description="test",
                duration_rounds=3,
                roles_active=[],
                objectives=[],
                outputs=[],
            ),
        ]

        protocol = template_to_protocol(minimal_template)

        # Total phase rounds = 5, max_rounds = 3
        # Should use max of 5 and 3 = 5
        assert protocol.rounds >= 5

    def test_rounds_uses_max_rounds_when_larger(self, minimal_template):
        """Test that max_rounds is used when larger than phase total."""
        minimal_template.max_rounds = 10
        minimal_template.phases = [
            DebatePhase(
                name="test",
                description="test",
                duration_rounds=2,
                roles_active=[],
                objectives=[],
                outputs=[],
            )
        ]

        protocol = template_to_protocol(minimal_template)

        # max_rounds (10) > phase total (2)
        assert protocol.rounds >= 10

    def test_code_review_total_phase_rounds(self):
        """Test CODE_REVIEW template phase round total."""
        total = sum(p.duration_rounds for p in CODE_REVIEW_TEMPLATE.phases)
        protocol = template_to_protocol(CODE_REVIEW_TEMPLATE)

        assert protocol.rounds >= min(total, CODE_REVIEW_TEMPLATE.max_rounds)


class TestTopologySelection:
    """Tests for topology selection based on template structure."""

    @pytest.fixture
    def minimal_template(self):
        """Create a minimal template for testing."""
        return DebateTemplate(
            template_id="test",
            template_type=TemplateType.CODE_REVIEW,
            name="Test",
            description="Test",
            roles=[],
            phases=[],
            recommended_agents=1,
            max_rounds=3,
            consensus_threshold=0.5,
            rubric={},
            output_format="",
            domain="test",
        )

    def test_research_synthesis_uses_all_to_all(self):
        """Test that RESEARCH_SYNTHESIS uses all-to-all topology."""
        protocol = template_to_protocol(RESEARCH_SYNTHESIS_TEMPLATE)

        assert protocol.topology == "all-to-all"

    def test_templates_with_many_roles_use_round_robin(self, minimal_template):
        """Test that templates with 4+ roles use round-robin."""
        roles = [
            DebateRole(
                name=f"role{i}",
                description="test",
                objectives=[],
                evaluation_criteria=[],
            )
            for i in range(5)
        ]
        minimal_template.roles = roles
        minimal_template.template_type = TemplateType.SECURITY_AUDIT

        protocol = template_to_protocol(minimal_template)

        assert protocol.topology == "round-robin"

    def test_templates_with_few_roles_use_all_to_all(self, minimal_template):
        """Test that templates with fewer than 4 roles use all-to-all."""
        roles = [
            DebateRole(
                name=f"role{i}",
                description="test",
                objectives=[],
                evaluation_criteria=[],
            )
            for i in range(2)
        ]
        minimal_template.roles = roles
        # Use a type that isn't research synthesis
        minimal_template.template_type = TemplateType.POLICY_REVIEW

        protocol = template_to_protocol(minimal_template)

        assert protocol.topology == "all-to-all"

    def test_code_review_topology(self):
        """Test CODE_REVIEW template topology selection."""
        protocol = template_to_protocol(CODE_REVIEW_TEMPLATE)

        # CODE_REVIEW has 5 roles, should use round-robin
        if len(CODE_REVIEW_TEMPLATE.roles) >= 4:
            assert protocol.topology == "round-robin"


class TestProtocolOverrides:
    """Tests for protocol configuration overrides."""

    def test_override_rounds(self):
        """Test overriding the rounds parameter."""
        protocol = template_to_protocol(CODE_REVIEW_TEMPLATE, overrides={"rounds": 15})

        assert protocol.rounds == 15

    def test_override_topology(self):
        """Test overriding the topology parameter."""
        protocol = template_to_protocol(RESEARCH_SYNTHESIS_TEMPLATE, overrides={"topology": "star"})

        assert protocol.topology == "star"

    def test_override_consensus(self):
        """Test overriding the consensus parameter."""
        protocol = template_to_protocol(CODE_REVIEW_TEMPLATE, overrides={"consensus": "unanimous"})

        assert protocol.consensus == "unanimous"

    def test_override_consensus_threshold(self):
        """Test overriding the consensus_threshold parameter."""
        protocol = template_to_protocol(
            CODE_REVIEW_TEMPLATE, overrides={"consensus_threshold": 0.95}
        )

        assert protocol.consensus_threshold == 0.95

    def test_override_role_rotation(self):
        """Test overriding the role_rotation parameter."""
        protocol = template_to_protocol(CODE_REVIEW_TEMPLATE, overrides={"role_rotation": False})

        assert protocol.role_rotation is False

    def test_override_early_stopping(self):
        """Test overriding the early_stopping parameter."""
        protocol = template_to_protocol(CODE_REVIEW_TEMPLATE, overrides={"early_stopping": False})

        assert protocol.early_stopping is False

    def test_override_convergence_detection(self):
        """Test overriding the convergence_detection parameter."""
        protocol = template_to_protocol(
            CODE_REVIEW_TEMPLATE, overrides={"convergence_detection": False}
        )

        assert protocol.convergence_detection is False

    def test_override_enable_calibration(self):
        """Test overriding the enable_calibration parameter."""
        protocol = template_to_protocol(
            CODE_REVIEW_TEMPLATE, overrides={"enable_calibration": False}
        )

        assert protocol.enable_calibration is False

    def test_multiple_overrides(self):
        """Test applying multiple overrides at once."""
        protocol = template_to_protocol(
            CODE_REVIEW_TEMPLATE,
            overrides={
                "rounds": 20,
                "topology": "mesh",
                "consensus_threshold": 0.9,
                "early_stopping": False,
            },
        )

        assert protocol.rounds == 20
        assert protocol.topology == "mesh"
        assert protocol.consensus_threshold == 0.9
        assert protocol.early_stopping is False

    def test_none_overrides(self):
        """Test that None overrides parameter works."""
        protocol1 = template_to_protocol(CODE_REVIEW_TEMPLATE, overrides=None)
        protocol2 = template_to_protocol(CODE_REVIEW_TEMPLATE)

        assert protocol1.rounds == protocol2.rounds
        assert protocol1.topology == protocol2.topology

    def test_empty_overrides(self):
        """Test that empty overrides dict works."""
        protocol1 = template_to_protocol(CODE_REVIEW_TEMPLATE, overrides={})
        protocol2 = template_to_protocol(CODE_REVIEW_TEMPLATE)

        assert protocol1.rounds == protocol2.rounds
        assert protocol1.topology == protocol2.topology


class TestProtocolDefaults:
    """Tests for protocol default values from templates."""

    def test_default_consensus_is_majority(self):
        """Test that default consensus mode is majority."""
        protocol = template_to_protocol(CODE_REVIEW_TEMPLATE)

        assert protocol.consensus == "majority"

    def test_default_role_rotation_enabled(self):
        """Test that role rotation is enabled by default."""
        protocol = template_to_protocol(CODE_REVIEW_TEMPLATE)

        assert protocol.role_rotation is True

    def test_default_require_reasoning_enabled(self):
        """Test that require_reasoning is enabled by default."""
        protocol = template_to_protocol(CODE_REVIEW_TEMPLATE)

        assert protocol.require_reasoning is True

    def test_default_early_stopping_enabled(self):
        """Test that early stopping is enabled by default."""
        protocol = template_to_protocol(CODE_REVIEW_TEMPLATE)

        assert protocol.early_stopping is True

    def test_default_convergence_detection_enabled(self):
        """Test that convergence detection is enabled by default."""
        protocol = template_to_protocol(CODE_REVIEW_TEMPLATE)

        assert protocol.convergence_detection is True

    def test_default_calibration_enabled(self):
        """Test that calibration is enabled by default."""
        protocol = template_to_protocol(CODE_REVIEW_TEMPLATE)

        assert protocol.enable_calibration is True


class TestConsensusThresholdMapping:
    """Tests for consensus threshold mapping from templates."""

    def test_code_review_threshold(self):
        """Test CODE_REVIEW consensus threshold."""
        protocol = template_to_protocol(CODE_REVIEW_TEMPLATE)
        assert protocol.consensus_threshold == 0.7

    def test_design_doc_threshold(self):
        """Test DESIGN_DOC consensus threshold."""
        protocol = template_to_protocol(DESIGN_DOC_TEMPLATE)
        assert protocol.consensus_threshold == 0.6

    def test_incident_response_threshold(self):
        """Test INCIDENT_RESPONSE consensus threshold."""
        protocol = template_to_protocol(INCIDENT_RESPONSE_TEMPLATE)
        assert protocol.consensus_threshold == 0.7

    def test_research_synthesis_threshold(self):
        """Test RESEARCH_SYNTHESIS consensus threshold."""
        protocol = template_to_protocol(RESEARCH_SYNTHESIS_TEMPLATE)
        assert protocol.consensus_threshold == 0.6

    def test_security_audit_threshold(self):
        """Test SECURITY_AUDIT consensus threshold."""
        protocol = template_to_protocol(SECURITY_AUDIT_TEMPLATE)
        assert protocol.consensus_threshold == 0.8

    def test_architecture_review_threshold(self):
        """Test ARCHITECTURE_REVIEW consensus threshold."""
        protocol = template_to_protocol(ARCHITECTURE_REVIEW_TEMPLATE)
        assert protocol.consensus_threshold == 0.7

    def test_healthcare_compliance_threshold(self):
        """Test HEALTHCARE_COMPLIANCE consensus threshold."""
        protocol = template_to_protocol(HEALTHCARE_COMPLIANCE_TEMPLATE)
        assert protocol.consensus_threshold == 0.8

    def test_financial_risk_threshold(self):
        """Test FINANCIAL_RISK consensus threshold."""
        protocol = template_to_protocol(FINANCIAL_RISK_TEMPLATE)
        assert protocol.consensus_threshold == 0.75


class TestProtocolCompatibility:
    """Tests for protocol compatibility with debate system."""

    def test_protocol_has_required_fields(self):
        """Test that generated protocol has all required fields."""
        protocol = template_to_protocol(CODE_REVIEW_TEMPLATE)

        # Check required DebateProtocol fields exist
        assert hasattr(protocol, "rounds")
        assert hasattr(protocol, "topology")
        assert hasattr(protocol, "consensus")
        assert hasattr(protocol, "consensus_threshold")

    def test_protocol_rounds_positive(self):
        """Test that protocol rounds is positive."""
        for template in [
            CODE_REVIEW_TEMPLATE,
            DESIGN_DOC_TEMPLATE,
            INCIDENT_RESPONSE_TEMPLATE,
        ]:
            protocol = template_to_protocol(template)
            assert protocol.rounds > 0

    def test_protocol_threshold_valid_range(self):
        """Test that consensus threshold is in valid range."""
        for template in [
            CODE_REVIEW_TEMPLATE,
            SECURITY_AUDIT_TEMPLATE,
            HEALTHCARE_COMPLIANCE_TEMPLATE,
        ]:
            protocol = template_to_protocol(template)
            assert 0.0 <= protocol.consensus_threshold <= 1.0
