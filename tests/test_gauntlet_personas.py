"""
Tests for aragora.gauntlet.personas module.

Tests regulatory persona loading, attack generation, and compliance checks.
"""

import pytest
from dataclasses import fields

from aragora.gauntlet.personas import (
    RegulatoryPersona,
    PersonaAttack,
    GDPRPersona,
    HIPAAPersona,
    AIActPersona,
    SecurityPersona,
    SOC2Persona,
    SOXPersona,
    PCIDSSPersona,
    NISTCSFPersona,
    get_persona,
    list_personas,
    PERSONAS,
)
from aragora.gauntlet.personas.base import AttackSeverity


class TestAttackSeverity:
    """Tests for AttackSeverity enum."""

    def test_all_severities_exist(self):
        """All expected severity levels exist."""
        assert AttackSeverity.CRITICAL.value == "critical"
        assert AttackSeverity.HIGH.value == "high"
        assert AttackSeverity.MEDIUM.value == "medium"
        assert AttackSeverity.LOW.value == "low"

    def test_severity_count(self):
        """Exactly 4 severity levels."""
        assert len(AttackSeverity) == 4


class TestPersonaAttack:
    """Tests for PersonaAttack dataclass."""

    def test_minimal_attack(self):
        """Create attack with minimal fields."""
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
        """Create attack with expected findings."""
        attack = PersonaAttack(
            id="test-002",
            name="Data Protection Attack",
            prompt="Check for data protection violations",
            category="privacy",
            expected_findings=["Consent issues", "Data retention"],
            severity_weight=1.5,
        )

        assert len(attack.expected_findings) == 2
        assert "Consent issues" in attack.expected_findings
        assert attack.severity_weight == 1.5


class TestRegulatoryPersona:
    """Tests for RegulatoryPersona base class."""

    def test_default_persona(self):
        """Default persona has expected defaults."""
        persona = RegulatoryPersona()

        assert persona.name == "Base Persona"
        assert persona.description == "Base regulatory persona"
        assert persona.regulation == "General"
        assert persona.version == "1.0"
        assert persona.attack_prompts == []
        assert persona.compliance_checks == []

    def test_get_system_prompt(self):
        """get_system_prompt includes regulation and description."""
        persona = RegulatoryPersona(
            name="Test Persona",
            description="Tests compliance",
            regulation="Test Regulation",
        )

        prompt = persona.get_system_prompt()

        assert "Test Regulation" in prompt
        assert "Tests compliance" in prompt
        assert "compliance reviewer" in prompt.lower()

    def test_get_attack_prompt(self):
        """get_attack_prompt combines system prompt and attack details."""
        persona = RegulatoryPersona(regulation="GDPR")
        attack = PersonaAttack(
            id="test-001",
            name="Consent Check",
            prompt="Verify consent mechanisms",
            category="consent",
            expected_findings=["Missing consent", "Invalid consent"],
        )

        prompt = persona.get_attack_prompt("Target document content", attack)

        assert "GDPR" in prompt
        assert "Consent Check" in prompt
        assert "Verify consent mechanisms" in prompt
        assert "Target document content" in prompt
        assert "Missing consent" in prompt

    def test_get_attacks_for_category(self):
        """get_attacks_for_category filters correctly."""
        attacks = [
            PersonaAttack(id="1", name="A1", prompt="P1", category="privacy"),
            PersonaAttack(id="2", name="A2", prompt="P2", category="security"),
            PersonaAttack(id="3", name="A3", prompt="P3", category="privacy"),
        ]
        persona = RegulatoryPersona(attack_prompts=attacks)

        privacy_attacks = persona.get_attacks_for_category("privacy")
        security_attacks = persona.get_attacks_for_category("security")
        other_attacks = persona.get_attacks_for_category("other")

        assert len(privacy_attacks) == 2
        assert len(security_attacks) == 1
        assert len(other_attacks) == 0

    def test_to_dict(self):
        """to_dict returns expected structure."""
        attacks = [
            PersonaAttack(id="1", name="A1", prompt="P1", category="cat1"),
            PersonaAttack(id="2", name="A2", prompt="P2", category="cat2"),
        ]
        persona = RegulatoryPersona(
            name="Test",
            description="Test persona",
            regulation="Test Reg",
            version="2.0",
            attack_prompts=attacks,
            compliance_checks=["Check 1", "Check 2"],
        )

        data = persona.to_dict()

        assert data["name"] == "Test"
        assert data["regulation"] == "Test Reg"
        assert data["version"] == "2.0"
        assert data["attack_count"] == 2
        assert len(data["compliance_checks"]) == 2
        assert "cat1" in data["categories"]
        assert "cat2" in data["categories"]


class TestGDPRPersona:
    """Tests for GDPR persona."""

    def test_gdpr_persona_properties(self):
        """GDPR persona has expected properties."""
        persona = GDPRPersona()

        assert "GDPR" in persona.name
        assert "GDPR" in persona.regulation
        assert len(persona.attack_prompts) > 0
        assert len(persona.compliance_checks) > 0

    def test_gdpr_has_data_protection_attacks(self):
        """GDPR has data protection focused attacks."""
        persona = GDPRPersona()
        categories = set(a.category for a in persona.attack_prompts)

        # Should have privacy-related categories
        assert len(categories) > 0
        # Check for common GDPR categories
        prompt_text = " ".join(a.prompt.lower() for a in persona.attack_prompts)
        assert "data" in prompt_text or "consent" in prompt_text or "privacy" in prompt_text


class TestHIPAAPersona:
    """Tests for HIPAA persona."""

    def test_hipaa_persona_properties(self):
        """HIPAA persona has expected properties."""
        persona = HIPAAPersona()

        assert "HIPAA" in persona.name or "HIPAA" in persona.regulation
        assert len(persona.attack_prompts) > 0

    def test_hipaa_has_healthcare_attacks(self):
        """HIPAA has healthcare focused attacks."""
        persona = HIPAAPersona()
        prompt_text = " ".join(a.prompt.lower() for a in persona.attack_prompts)

        # Should mention PHI, healthcare, or patient
        assert "phi" in prompt_text or "health" in prompt_text or "patient" in prompt_text


class TestAIActPersona:
    """Tests for EU AI Act persona."""

    def test_ai_act_persona_properties(self):
        """AI Act persona has expected properties."""
        persona = AIActPersona()

        assert "AI" in persona.name or "AI" in persona.regulation
        assert len(persona.attack_prompts) > 0

    def test_ai_act_has_ai_attacks(self):
        """AI Act has AI-focused attacks."""
        persona = AIActPersona()
        prompt_text = " ".join(a.prompt.lower() for a in persona.attack_prompts)

        # Should mention AI, model, or algorithm
        assert "ai" in prompt_text or "model" in prompt_text or "algorithm" in prompt_text


class TestSecurityPersona:
    """Tests for Security persona."""

    def test_security_persona_properties(self):
        """Security persona has expected properties."""
        persona = SecurityPersona()

        assert "Security" in persona.name or "Security" in persona.regulation
        assert len(persona.attack_prompts) > 0

    def test_security_has_security_attacks(self):
        """Security persona has security focused attacks."""
        persona = SecurityPersona()
        prompt_text = " ".join(a.prompt.lower() for a in persona.attack_prompts)

        # Should mention security-related terms
        assert any(term in prompt_text for term in ["security", "vulnerability", "attack", "threat"])


class TestSOC2Persona:
    """Tests for SOC2 persona."""

    def test_soc2_persona_properties(self):
        """SOC2 persona has expected properties."""
        persona = SOC2Persona()

        assert "SOC" in persona.name or "SOC" in persona.regulation
        assert len(persona.attack_prompts) > 0


class TestSOXPersona:
    """Tests for SOX persona."""

    def test_sox_persona_properties(self):
        """SOX persona has expected properties."""
        persona = SOXPersona()

        assert "SOX" in persona.name or "Sarbanes" in persona.regulation or "SOX" in persona.regulation
        assert len(persona.attack_prompts) > 0


class TestPCIDSSPersona:
    """Tests for PCI-DSS persona."""

    def test_pci_dss_persona_properties(self):
        """PCI-DSS persona has expected properties."""
        persona = PCIDSSPersona()

        assert "PCI" in persona.name or "PCI" in persona.regulation
        assert len(persona.attack_prompts) > 0


class TestNISTCSFPersona:
    """Tests for NIST CSF persona."""

    def test_nist_csf_persona_properties(self):
        """NIST CSF persona has expected properties."""
        persona = NISTCSFPersona()

        assert "NIST" in persona.name or "NIST" in persona.regulation
        assert len(persona.attack_prompts) > 0


class TestGetPersona:
    """Tests for get_persona function."""

    def test_get_gdpr_persona(self):
        """get_persona returns GDPR persona."""
        persona = get_persona("gdpr")

        assert isinstance(persona, GDPRPersona)

    def test_get_hipaa_persona(self):
        """get_persona returns HIPAA persona."""
        persona = get_persona("hipaa")

        assert isinstance(persona, HIPAAPersona)

    def test_get_ai_act_persona(self):
        """get_persona returns AI Act persona."""
        persona = get_persona("ai_act")

        assert isinstance(persona, AIActPersona)

    def test_get_security_persona(self):
        """get_persona returns Security persona."""
        persona = get_persona("security")

        assert isinstance(persona, SecurityPersona)

    def test_get_pci_dss_with_hyphen(self):
        """get_persona supports hyphen alias for PCI-DSS."""
        persona = get_persona("pci-dss")

        assert isinstance(persona, PCIDSSPersona)

    def test_get_nist_csf_with_hyphen(self):
        """get_persona supports hyphen alias for NIST-CSF."""
        persona = get_persona("nist-csf")

        assert isinstance(persona, NISTCSFPersona)

    def test_get_unknown_persona_raises(self):
        """get_persona raises ValueError for unknown persona."""
        with pytest.raises(ValueError) as exc_info:
            get_persona("unknown_persona")

        assert "Unknown persona" in str(exc_info.value)
        assert "unknown_persona" in str(exc_info.value)


class TestListPersonas:
    """Tests for list_personas function."""

    def test_list_personas_returns_list(self):
        """list_personas returns a list."""
        personas = list_personas()

        assert isinstance(personas, list)
        assert len(personas) > 0

    def test_list_personas_contains_expected(self):
        """list_personas contains expected persona names."""
        personas = list_personas()

        assert "gdpr" in personas
        assert "hipaa" in personas
        assert "ai_act" in personas
        assert "security" in personas
        assert "soc2" in personas
        assert "sox" in personas

    def test_all_listed_personas_are_valid(self):
        """All listed personas can be retrieved."""
        personas = list_personas()

        for name in personas:
            # Should not raise
            persona = get_persona(name)
            assert persona is not None


class TestPersonasRegistry:
    """Tests for PERSONAS registry."""

    def test_registry_contains_all_personas(self):
        """Registry contains all persona types."""
        assert "gdpr" in PERSONAS
        assert "hipaa" in PERSONAS
        assert "ai_act" in PERSONAS
        assert "security" in PERSONAS
        assert "soc2" in PERSONAS
        assert "sox" in PERSONAS
        assert "pci_dss" in PERSONAS
        assert "nist_csf" in PERSONAS

    def test_registry_has_aliases(self):
        """Registry includes hyphen aliases."""
        assert "pci-dss" in PERSONAS
        assert "nist-csf" in PERSONAS

    def test_all_registry_values_are_classes(self):
        """All registry values are persona classes."""
        for name, cls in PERSONAS.items():
            assert issubclass(cls, RegulatoryPersona), f"{name} is not a RegulatoryPersona subclass"


class TestPersonaCompleteness:
    """Tests ensuring all personas are complete."""

    @pytest.mark.parametrize("persona_name", [
        "gdpr", "hipaa", "ai_act", "security", "soc2", "sox", "pci_dss", "nist_csf"
    ])
    def test_persona_has_attacks(self, persona_name):
        """Each persona has at least one attack."""
        persona = get_persona(persona_name)
        assert len(persona.attack_prompts) > 0, f"{persona_name} has no attacks"

    @pytest.mark.parametrize("persona_name", [
        "gdpr", "hipaa", "ai_act", "security", "soc2", "sox", "pci_dss", "nist_csf"
    ])
    def test_persona_has_name(self, persona_name):
        """Each persona has a name."""
        persona = get_persona(persona_name)
        assert persona.name and persona.name != "Base Persona", f"{persona_name} has no custom name"

    @pytest.mark.parametrize("persona_name", [
        "gdpr", "hipaa", "ai_act", "security", "soc2", "sox", "pci_dss", "nist_csf"
    ])
    def test_persona_has_regulation(self, persona_name):
        """Each persona has a regulation reference."""
        persona = get_persona(persona_name)
        assert persona.regulation and persona.regulation != "General", f"{persona_name} has no regulation"

    @pytest.mark.parametrize("persona_name", [
        "gdpr", "hipaa", "ai_act", "security", "soc2", "sox", "pci_dss", "nist_csf"
    ])
    def test_persona_to_dict_is_complete(self, persona_name):
        """Each persona's to_dict includes required fields."""
        persona = get_persona(persona_name)
        data = persona.to_dict()

        assert "name" in data
        assert "regulation" in data
        assert "attack_count" in data
        assert data["attack_count"] > 0
