"""
Tests for Gauntlet compliance personas.

Verifies that each regulatory persona:
- Has valid structure and required fields
- Generates proper system prompts
- Contains attack prompts with expected findings
- Follows the base persona interface
"""

from __future__ import annotations

import pytest

from aragora.gauntlet.personas.base import (
    AttackSeverity,
    PersonaAttack,
    RegulatoryPersona,
)


class TestAttackSeverity:
    """Test AttackSeverity enum."""

    def test_severity_values(self):
        """Test all severity levels exist."""
        assert AttackSeverity.CRITICAL.value == "critical"
        assert AttackSeverity.HIGH.value == "high"
        assert AttackSeverity.MEDIUM.value == "medium"
        assert AttackSeverity.LOW.value == "low"

    def test_severity_ordering(self):
        """Test severity can be compared by name."""
        severities = [s.value for s in AttackSeverity]
        assert "critical" in severities
        assert len(severities) == 4


class TestPersonaAttack:
    """Test PersonaAttack dataclass."""

    def test_basic_attack(self):
        """Test creating a basic attack."""
        attack = PersonaAttack(
            id="test-001",
            name="Test Attack",
            prompt="Test prompt",
            category="test",
        )
        assert attack.id == "test-001"
        assert attack.name == "Test Attack"
        assert attack.prompt == "Test prompt"
        assert attack.category == "test"
        assert attack.expected_findings == []
        assert attack.severity_weight == 1.0

    def test_attack_with_findings(self):
        """Test attack with expected findings."""
        attack = PersonaAttack(
            id="test-002",
            name="Attack with Findings",
            prompt="Find issues",
            category="compliance",
            expected_findings=["Issue 1", "Issue 2"],
            severity_weight=1.5,
        )
        assert len(attack.expected_findings) == 2
        assert attack.severity_weight == 1.5


class TestRegulatoryPersona:
    """Test base RegulatoryPersona class."""

    def test_default_values(self):
        """Test default persona values."""
        persona = RegulatoryPersona()
        assert persona.name == "Base Persona"
        assert persona.description == "Base regulatory persona"
        assert persona.regulation == "General"
        assert persona.version == "1.0"
        assert persona.attack_prompts == []
        assert persona.compliance_checks == []

    def test_system_prompt_generation(self):
        """Test system prompt includes regulation name."""
        persona = RegulatoryPersona(
            name="Test Persona",
            description="Test description",
            regulation="Test Regulation",
        )
        prompt = persona.get_system_prompt()
        assert "Test Regulation" in prompt
        assert "Test description" in prompt
        assert "adversarial compliance reviewer" in prompt


class TestGDPRPersona:
    """Test GDPR compliance persona."""

    @pytest.fixture
    def gdpr_persona(self):
        """Create GDPR persona."""
        from aragora.gauntlet.personas.gdpr import GDPRPersona

        return GDPRPersona()

    def test_persona_metadata(self, gdpr_persona):
        """Test GDPR persona has correct metadata."""
        assert "GDPR" in gdpr_persona.name
        assert gdpr_persona.regulation == "GDPR (EU 2016/679)"
        assert gdpr_persona.version

    def test_attack_prompts_exist(self, gdpr_persona):
        """Test GDPR persona has attack prompts."""
        assert len(gdpr_persona.attack_prompts) > 0

    def test_attack_prompts_have_required_fields(self, gdpr_persona):
        """Test each attack has required fields."""
        for attack in gdpr_persona.attack_prompts:
            assert attack.id.startswith("gdpr-")
            assert attack.name
            assert attack.prompt
            assert attack.category
            assert isinstance(attack.severity_weight, float)

    def test_context_preamble(self, gdpr_persona):
        """Test GDPR context includes key principles."""
        preamble = gdpr_persona.context_preamble
        assert "Art." in preamble  # References GDPR articles
        assert "data" in preamble.lower()

    def test_system_prompt(self, gdpr_persona):
        """Test system prompt generation."""
        prompt = gdpr_persona.get_system_prompt()
        assert "GDPR" in prompt


class TestHIPAAPersona:
    """Test HIPAA compliance persona."""

    @pytest.fixture
    def hipaa_persona(self):
        """Create HIPAA persona."""
        from aragora.gauntlet.personas.hipaa import HIPAAPersona

        return HIPAAPersona()

    def test_persona_metadata(self, hipaa_persona):
        """Test HIPAA persona has correct metadata."""
        assert "HIPAA" in hipaa_persona.name or "HIPAA" in hipaa_persona.regulation
        assert hipaa_persona.version

    def test_attack_prompts_exist(self, hipaa_persona):
        """Test HIPAA persona has attack prompts."""
        assert len(hipaa_persona.attack_prompts) > 0

    def test_attack_ids_unique(self, hipaa_persona):
        """Test attack IDs are unique."""
        ids = [a.id for a in hipaa_persona.attack_prompts]
        assert len(ids) == len(set(ids))


class TestSOC2Persona:
    """Test SOC 2 compliance persona."""

    @pytest.fixture
    def soc2_persona(self):
        """Create SOC 2 persona."""
        from aragora.gauntlet.personas.soc2 import SOC2Persona

        return SOC2Persona()

    def test_persona_metadata(self, soc2_persona):
        """Test SOC 2 persona has correct metadata."""
        assert "SOC" in soc2_persona.name or "SOC" in soc2_persona.regulation
        assert soc2_persona.version

    def test_attack_prompts_exist(self, soc2_persona):
        """Test SOC 2 persona has attack prompts."""
        assert len(soc2_persona.attack_prompts) > 0

    def test_trust_services_criteria(self, soc2_persona):
        """Test SOC 2 references trust services criteria."""
        # SOC 2 should reference security, availability, processing integrity,
        # confidentiality, or privacy criteria
        all_prompts = " ".join(a.prompt for a in soc2_persona.attack_prompts)
        preamble = soc2_persona.context_preamble
        combined = (all_prompts + preamble).lower()

        tsc_terms = ["security", "availability", "confidentiality", "privacy"]
        assert any(term in combined for term in tsc_terms)


class TestPCIDSSPersona:
    """Test PCI DSS compliance persona."""

    @pytest.fixture
    def pci_persona(self):
        """Create PCI DSS persona."""
        from aragora.gauntlet.personas.pci_dss import PCIDSSPersona

        return PCIDSSPersona()

    def test_persona_metadata(self, pci_persona):
        """Test PCI DSS persona has correct metadata."""
        assert "PCI" in pci_persona.name or "PCI" in pci_persona.regulation
        assert pci_persona.version

    def test_attack_prompts_exist(self, pci_persona):
        """Test PCI DSS persona has attack prompts."""
        assert len(pci_persona.attack_prompts) > 0


class TestAIActPersona:
    """Test EU AI Act compliance persona."""

    @pytest.fixture
    def ai_act_persona(self):
        """Create AI Act persona."""
        from aragora.gauntlet.personas.ai_act import AIActPersona

        return AIActPersona()

    def test_persona_metadata(self, ai_act_persona):
        """Test AI Act persona has correct metadata."""
        assert "AI" in ai_act_persona.name or "AI" in ai_act_persona.regulation
        assert ai_act_persona.version

    def test_attack_prompts_exist(self, ai_act_persona):
        """Test AI Act persona has attack prompts."""
        assert len(ai_act_persona.attack_prompts) > 0

    def test_risk_categories(self, ai_act_persona):
        """Test AI Act references risk categories."""
        # AI Act has risk-based approach
        all_prompts = " ".join(a.prompt for a in ai_act_persona.attack_prompts)
        preamble = ai_act_persona.context_preamble
        combined = (all_prompts + preamble).lower()

        # Should reference risk or transparency
        assert "risk" in combined or "transparency" in combined


class TestSOXPersona:
    """Test SOX compliance persona."""

    @pytest.fixture
    def sox_persona(self):
        """Create SOX persona."""
        from aragora.gauntlet.personas.sox import SOXPersona

        return SOXPersona()

    def test_persona_metadata(self, sox_persona):
        """Test SOX persona has correct metadata."""
        assert "SOX" in sox_persona.name or "Sarbanes" in sox_persona.regulation
        assert sox_persona.version

    def test_attack_prompts_exist(self, sox_persona):
        """Test SOX persona has attack prompts."""
        assert len(sox_persona.attack_prompts) > 0


class TestNISTCSFPersona:
    """Test NIST CSF compliance persona."""

    @pytest.fixture
    def nist_persona(self):
        """Create NIST CSF persona."""
        from aragora.gauntlet.personas.nist_csf import NISTCSFPersona

        return NISTCSFPersona()

    def test_persona_metadata(self, nist_persona):
        """Test NIST CSF persona has correct metadata."""
        assert "NIST" in nist_persona.name or "NIST" in nist_persona.regulation
        assert nist_persona.version

    def test_attack_prompts_exist(self, nist_persona):
        """Test NIST CSF persona has attack prompts."""
        assert len(nist_persona.attack_prompts) > 0

    def test_framework_functions(self, nist_persona):
        """Test NIST CSF references core functions."""
        # NIST CSF has 5 functions: Identify, Protect, Detect, Respond, Recover
        all_prompts = " ".join(a.prompt for a in nist_persona.attack_prompts)
        preamble = nist_persona.context_preamble
        combined = (all_prompts + preamble).lower()

        functions = ["identify", "protect", "detect", "respond", "recover"]
        assert any(func in combined for func in functions)


class TestSecurityPersona:
    """Test general security persona."""

    @pytest.fixture
    def security_persona(self):
        """Create security persona."""
        from aragora.gauntlet.personas.security import SecurityPersona

        return SecurityPersona()

    def test_persona_metadata(self, security_persona):
        """Test security persona has correct metadata."""
        assert (
            "Security" in security_persona.name
            or "security" in security_persona.description.lower()
        )
        assert security_persona.version

    def test_attack_prompts_exist(self, security_persona):
        """Test security persona has attack prompts."""
        assert len(security_persona.attack_prompts) > 0


class TestAllPersonasConsistency:
    """Test consistency across all personas."""

    @pytest.fixture
    def all_personas(self):
        """Load all compliance personas."""
        from aragora.gauntlet.personas.ai_act import AIActPersona
        from aragora.gauntlet.personas.gdpr import GDPRPersona
        from aragora.gauntlet.personas.hipaa import HIPAAPersona
        from aragora.gauntlet.personas.nist_csf import NISTCSFPersona
        from aragora.gauntlet.personas.pci_dss import PCIDSSPersona
        from aragora.gauntlet.personas.security import SecurityPersona
        from aragora.gauntlet.personas.soc2 import SOC2Persona
        from aragora.gauntlet.personas.sox import SOXPersona

        return [
            GDPRPersona(),
            HIPAAPersona(),
            SOC2Persona(),
            PCIDSSPersona(),
            AIActPersona(),
            SOXPersona(),
            NISTCSFPersona(),
            SecurityPersona(),
        ]

    def test_all_have_attack_prompts(self, all_personas):
        """Test all personas have at least one attack prompt."""
        for persona in all_personas:
            assert len(persona.attack_prompts) > 0, f"{persona.name} has no attacks"

    def test_all_have_version(self, all_personas):
        """Test all personas have a version."""
        for persona in all_personas:
            assert persona.version, f"{persona.name} missing version"

    def test_all_generate_system_prompts(self, all_personas):
        """Test all personas can generate system prompts."""
        for persona in all_personas:
            prompt = persona.get_system_prompt()
            assert len(prompt) > 50, f"{persona.name} has empty system prompt"

    def test_attack_ids_globally_unique(self, all_personas):
        """Test attack IDs are unique across all personas."""
        all_ids = []
        for persona in all_personas:
            all_ids.extend(a.id for a in persona.attack_prompts)

        assert len(all_ids) == len(set(all_ids)), "Duplicate attack IDs found"

    def test_all_inherit_from_base(self, all_personas):
        """Test all personas inherit from RegulatoryPersona."""
        for persona in all_personas:
            assert isinstance(persona, RegulatoryPersona)
