"""
Tests for aragora.gauntlet extensions.

Tests DecisionReceipt, RiskHeatmap, and Regulatory Personas.
"""

import pytest
from datetime import datetime


class TestDecisionReceipt:
    """Tests for DecisionReceipt."""

    def test_receipt_creation_manual(self):
        """Test creating a receipt manually."""
        from aragora.gauntlet.receipt import DecisionReceipt, ConsensusProof

        receipt = DecisionReceipt(
            receipt_id="receipt-test-001",
            gauntlet_id="gauntlet-test-001",
            timestamp=datetime.now().isoformat(),
            input_summary="Test input content",
            input_hash="abc123",
            risk_summary={"critical": 1, "high": 2, "medium": 3, "low": 4, "total": 10},
            attacks_attempted=20,
            attacks_successful=5,
            probes_run=15,
            vulnerabilities_found=10,
            verdict="CONDITIONAL",
            confidence=0.75,
            robustness_score=0.8,
        )

        assert receipt.receipt_id == "receipt-test-001"
        assert receipt.verdict == "CONDITIONAL"
        assert receipt.artifact_hash  # Should be auto-generated

    def test_receipt_integrity_verification(self):
        """Test that integrity verification works."""
        from aragora.gauntlet.receipt import DecisionReceipt

        receipt = DecisionReceipt(
            receipt_id="receipt-test-002",
            gauntlet_id="gauntlet-test-002",
            timestamp=datetime.now().isoformat(),
            input_summary="Test content",
            input_hash="def456",
            risk_summary={"critical": 0, "total": 0},
            attacks_attempted=10,
            attacks_successful=0,
            probes_run=5,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.95,
            robustness_score=0.9,
        )

        # Verify integrity
        assert receipt.verify_integrity()

        # Tamper and check
        receipt.verdict = "FAIL"
        assert not receipt.verify_integrity()

    def test_receipt_to_markdown(self):
        """Test markdown generation."""
        from aragora.gauntlet.receipt import DecisionReceipt

        receipt = DecisionReceipt(
            receipt_id="receipt-test-003",
            gauntlet_id="gauntlet-test-003",
            timestamp=datetime.now().isoformat(),
            input_summary="Test content for markdown",
            input_hash="ghi789",
            risk_summary={"critical": 2, "high": 1, "medium": 0, "low": 0, "total": 3},
            attacks_attempted=15,
            attacks_successful=3,
            probes_run=10,
            vulnerabilities_found=3,
            verdict="FAIL",
            confidence=0.85,
            robustness_score=0.5,
            verdict_reasoning="Critical vulnerabilities found",
        )

        md = receipt.to_markdown()

        assert "# Decision Receipt" in md
        assert "FAIL" in md
        assert "Critical" in md
        assert "85" in md  # confidence

    def test_receipt_to_json(self):
        """Test JSON export."""
        from aragora.gauntlet.receipt import DecisionReceipt
        import json

        receipt = DecisionReceipt(
            receipt_id="receipt-test-004",
            gauntlet_id="gauntlet-test-004",
            timestamp=datetime.now().isoformat(),
            input_summary="JSON test",
            input_hash="jkl012",
            risk_summary={"total": 0},
            attacks_attempted=5,
            attacks_successful=0,
            probes_run=5,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.85,
        )

        json_str = receipt.to_json()
        data = json.loads(json_str)

        assert data["receipt_id"] == "receipt-test-004"
        assert data["verdict"] == "PASS"


class TestRiskHeatmap:
    """Tests for RiskHeatmap."""

    def test_heatmap_creation_manual(self):
        """Test creating a heatmap manually."""
        from aragora.gauntlet.heatmap import RiskHeatmap, HeatmapCell

        cells = [
            HeatmapCell(category="security", severity="critical", count=2),
            HeatmapCell(category="security", severity="high", count=3),
            HeatmapCell(category="logic", severity="medium", count=5),
        ]

        heatmap = RiskHeatmap(
            cells=cells,
            categories=["security", "logic"],
            total_findings=10,
        )

        assert heatmap.total_findings == 10
        assert len(heatmap.cells) == 3
        assert heatmap.get_category_total("security") == 5

    def test_heatmap_cell_intensity(self):
        """Test cell intensity calculation."""
        from aragora.gauntlet.heatmap import HeatmapCell

        cell_0 = HeatmapCell(category="test", severity="low", count=0)
        cell_1 = HeatmapCell(category="test", severity="low", count=1)
        cell_10 = HeatmapCell(category="test", severity="low", count=10)

        assert cell_0.intensity == 0.0
        assert 0 < cell_1.intensity < cell_10.intensity

    def test_heatmap_to_svg(self):
        """Test SVG generation."""
        from aragora.gauntlet.heatmap import RiskHeatmap, HeatmapCell

        cells = [
            HeatmapCell(category="security", severity="critical", count=1),
            HeatmapCell(category="security", severity="high", count=2),
        ]

        heatmap = RiskHeatmap(
            cells=cells,
            categories=["security"],
            total_findings=3,
        )

        svg = heatmap.to_svg()

        assert "<svg" in svg
        assert "security" in svg.lower() or "CRITICAL" in svg

    def test_heatmap_to_ascii(self):
        """Test ASCII table generation."""
        from aragora.gauntlet.heatmap import RiskHeatmap, HeatmapCell

        cells = [
            HeatmapCell(category="security", severity="critical", count=2),
            HeatmapCell(category="security", severity="high", count=0),
            HeatmapCell(category="security", severity="medium", count=1),
            HeatmapCell(category="security", severity="low", count=0),
        ]

        heatmap = RiskHeatmap(
            cells=cells,
            categories=["security"],
            total_findings=3,
        )

        ascii_table = heatmap.to_ascii()

        assert "security" in ascii_table
        assert "CRIT" in ascii_table
        assert "TOTAL" in ascii_table

    def test_heatmap_empty(self):
        """Test empty heatmap handling."""
        from aragora.gauntlet.heatmap import RiskHeatmap

        heatmap = RiskHeatmap()

        svg = heatmap.to_svg()
        assert "No data" in svg

        ascii_table = heatmap.to_ascii()
        assert "No findings" in ascii_table


class TestRegulatoryPersonas:
    """Tests for regulatory personas."""

    def test_gdpr_persona(self):
        """Test GDPR persona."""
        from aragora.gauntlet.personas import GDPRPersona

        persona = GDPRPersona()

        assert persona.name == "GDPR Compliance Auditor"
        assert persona.regulation == "GDPR (EU 2016/679)"
        assert len(persona.attack_prompts) > 0
        assert len(persona.compliance_checks) > 0

        # Test system prompt generation
        system_prompt = persona.get_system_prompt()
        assert "GDPR" in system_prompt
        assert "compliance" in system_prompt.lower()

    def test_hipaa_persona(self):
        """Test HIPAA persona."""
        from aragora.gauntlet.personas import HIPAAPersona

        persona = HIPAAPersona()

        assert persona.name == "HIPAA Compliance Auditor"
        assert "HIPAA" in persona.regulation
        assert len(persona.attack_prompts) > 0

        # Check for PHI-related attacks
        attack_categories = [a.category for a in persona.attack_prompts]
        assert "phi_handling" in attack_categories

    def test_ai_act_persona(self):
        """Test EU AI Act persona."""
        from aragora.gauntlet.personas import AIActPersona

        persona = AIActPersona()

        assert persona.name == "EU AI Act Compliance Auditor"
        assert "AI Act" in persona.regulation
        assert len(persona.attack_prompts) > 0

        # Check for AI-specific attacks
        attack_categories = [a.category for a in persona.attack_prompts]
        assert "risk_classification" in attack_categories
        assert "human_oversight" in attack_categories

    def test_security_persona(self):
        """Test Security red team persona."""
        from aragora.gauntlet.personas import SecurityPersona

        persona = SecurityPersona()

        assert persona.name == "Security Red Team"
        assert "OWASP" in persona.regulation
        assert len(persona.attack_prompts) > 0

        # Check for security attacks
        attack_categories = [a.category for a in persona.attack_prompts]
        assert "injection" in attack_categories
        assert "authentication" in attack_categories

    def test_get_persona(self):
        """Test getting personas by name."""
        from aragora.gauntlet.personas import get_persona, list_personas

        personas = list_personas()
        assert "gdpr" in personas
        assert "hipaa" in personas
        assert "ai_act" in personas
        assert "security" in personas

        gdpr = get_persona("gdpr")
        assert gdpr.name == "GDPR Compliance Auditor"

    def test_get_persona_invalid(self):
        """Test getting invalid persona raises error."""
        from aragora.gauntlet.personas import get_persona

        with pytest.raises(ValueError, match="Unknown persona"):
            get_persona("invalid_persona")

    def test_attack_prompt_generation(self):
        """Test generating attack prompts."""
        from aragora.gauntlet.personas import GDPRPersona

        persona = GDPRPersona()
        attack = persona.attack_prompts[0]

        target = "This system collects user emails without consent."
        prompt = persona.get_attack_prompt(target, attack)

        assert target in prompt
        assert attack.name in prompt
        assert "GDPR" in prompt


class TestPersonaAttack:
    """Tests for PersonaAttack dataclass."""

    def test_persona_attack_creation(self):
        """Test creating a PersonaAttack."""
        from aragora.gauntlet.personas.base import PersonaAttack

        attack = PersonaAttack(
            id="test-001",
            name="Test Attack",
            prompt="Find vulnerabilities in data handling",
            category="data_handling",
            expected_findings=["Missing encryption", "No access controls"],
            severity_weight=1.5,
        )

        assert attack.id == "test-001"
        assert attack.severity_weight == 1.5
        assert len(attack.expected_findings) == 2


class TestIntegration:
    """Integration tests for gauntlet extensions."""

    def test_full_workflow_mock(self):
        """Test full workflow with mock data."""
        from aragora.gauntlet.receipt import DecisionReceipt, ProvenanceRecord
        from aragora.gauntlet.heatmap import RiskHeatmap, HeatmapCell

        # Create heatmap
        cells = [
            HeatmapCell(category="security", severity="critical", count=1),
            HeatmapCell(category="compliance", severity="high", count=2),
            HeatmapCell(category="logic", severity="medium", count=3),
        ]
        heatmap = RiskHeatmap(
            cells=cells,
            categories=["security", "compliance", "logic"],
            total_findings=6,
            highest_risk_category="logic",
            highest_risk_severity="critical",
        )

        # Create receipt
        receipt = DecisionReceipt(
            receipt_id="receipt-integration-001",
            gauntlet_id="gauntlet-integration-001",
            timestamp=datetime.now().isoformat(),
            input_summary="Integration test input",
            input_hash="integration123",
            risk_summary={
                "critical": 1,
                "high": 2,
                "medium": 3,
                "low": 0,
                "total": 6
            },
            attacks_attempted=25,
            attacks_successful=6,
            probes_run=20,
            vulnerabilities_found=6,
            verdict="CONDITIONAL",
            confidence=0.7,
            robustness_score=0.75,
            provenance_chain=[
                ProvenanceRecord(
                    timestamp=datetime.now().isoformat(),
                    event_type="attack",
                    agent="red_team_1",
                    description="[CRITICAL] Security vulnerability found",
                ),
            ],
        )

        # Verify everything works together
        assert heatmap.total_findings == receipt.vulnerabilities_found
        assert receipt.verify_integrity()
        assert heatmap.highest_risk_severity == "critical"

        # Export both
        json_receipt = receipt.to_json()
        svg_heatmap = heatmap.to_svg()

        assert "integration" in json_receipt
        assert "<svg" in svg_heatmap
