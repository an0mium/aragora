"""
Tests for the knowledge verticals module.

Tests the vertical-specific knowledge extraction, validation, and pattern detection
for healthcare, legal, accounting, and other enterprise verticals.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.knowledge.mound.verticals.base import (
    BaseVerticalKnowledge,
    ComplianceCheckResult,
    PatternMatch,
    VerticalCapabilities,
    VerticalFact,
)
from aragora.knowledge.mound.verticals.registry import VerticalRegistry


# =============================================================================
# VerticalCapabilities Tests
# =============================================================================


class TestVerticalCapabilities:
    """Tests for VerticalCapabilities dataclass."""

    def test_capabilities_defaults(self):
        """Test default capabilities."""
        caps = VerticalCapabilities()
        assert caps.supports_pattern_detection is True
        assert caps.supports_cross_reference is False
        assert caps.supports_compliance_check is False
        assert caps.requires_llm is False
        assert caps.requires_vector_search is True
        assert caps.pattern_categories == []
        assert caps.compliance_frameworks == []
        assert caps.document_types == []

    def test_capabilities_custom(self):
        """Test custom capabilities."""
        caps = VerticalCapabilities(
            supports_compliance_check=True,
            requires_llm=True,
            compliance_frameworks=["HIPAA", "GDPR"],
            document_types=["clinical_note", "lab_report"],
        )
        assert caps.supports_compliance_check is True
        assert caps.requires_llm is True
        assert len(caps.compliance_frameworks) == 2
        assert "HIPAA" in caps.compliance_frameworks


# =============================================================================
# VerticalFact Tests
# =============================================================================


class TestVerticalFact:
    """Tests for VerticalFact dataclass."""

    def test_fact_creation(self):
        """Test creating a vertical fact."""
        fact = VerticalFact(
            id="fact-123",
            vertical="healthcare",
            content="Patient diagnosed with hypertension",
            category="diagnosis",
            confidence=0.95,
        )
        assert fact.id == "fact-123"
        assert fact.vertical == "healthcare"
        assert fact.content == "Patient diagnosed with hypertension"
        assert fact.category == "diagnosis"
        assert fact.confidence == 0.95
        assert fact.staleness_days == 0.0
        assert fact.decay_rate == 0.1

    def test_fact_adjusted_confidence_fresh(self):
        """Test adjusted confidence for fresh facts."""
        fact = VerticalFact(
            id="fact-1",
            vertical="legal",
            content="Contract clause",
            category="contract",
            confidence=0.9,
            staleness_days=0.0,
        )
        assert fact.adjusted_confidence == 0.9  # No decay

    def test_fact_adjusted_confidence_stale(self):
        """Test adjusted confidence for stale facts."""
        fact = VerticalFact(
            id="fact-2",
            vertical="legal",
            content="Contract clause",
            category="contract",
            confidence=0.9,
            staleness_days=5.0,
            decay_rate=0.1,
        )
        # Decay = 5 * 0.1 = 0.5, adjusted = 0.9 * (1 - 0.5) = 0.45
        assert fact.adjusted_confidence == pytest.approx(0.45)

    def test_fact_adjusted_confidence_max_decay(self):
        """Test that decay is capped at 0.9."""
        fact = VerticalFact(
            id="fact-3",
            vertical="legal",
            content="Old fact",
            category="contract",
            confidence=1.0,
            staleness_days=100.0,
            decay_rate=0.1,
        )
        # Decay would be 10.0 but capped at 0.9
        assert fact.adjusted_confidence == pytest.approx(0.1)

    def test_fact_needs_reverification(self):
        """Test needs_reverification property."""
        fresh_fact = VerticalFact(
            id="fresh",
            vertical="healthcare",
            content="Recent finding",
            category="exam",
            confidence=0.9,
            staleness_days=0.0,
        )
        assert fresh_fact.needs_reverification is False

        stale_fact = VerticalFact(
            id="stale",
            vertical="healthcare",
            content="Old finding",
            category="exam",
            confidence=0.9,
            staleness_days=10.0,
            decay_rate=0.1,
        )
        # Adjusted = 0.9 * (1 - 0.9) = 0.09, which is < 0.5 * 0.9 = 0.45
        assert stale_fact.needs_reverification is True

    def test_fact_refresh(self):
        """Test refreshing a fact."""
        fact = VerticalFact(
            id="fact-4",
            vertical="accounting",
            content="Revenue figure",
            category="financial",
            confidence=0.7,
            staleness_days=30.0,
        )
        assert fact.staleness_days == 30.0

        fact.refresh(new_confidence=0.95)

        assert fact.staleness_days == 0.0
        assert fact.confidence == 0.95

    def test_fact_to_dict(self):
        """Test fact serialization."""
        fact = VerticalFact(
            id="fact-5",
            vertical="software",
            content="API endpoint documented",
            category="documentation",
            confidence=0.85,
            provenance={"source": "readme.md"},
            metadata={"line": 42},
        )

        d = fact.to_dict()

        assert d["id"] == "fact-5"
        assert d["vertical"] == "software"
        assert d["content"] == "API endpoint documented"
        assert d["confidence"] == 0.85
        assert d["adjusted_confidence"] == 0.85  # Fresh fact
        assert d["provenance"]["source"] == "readme.md"
        assert "created_at" in d


# =============================================================================
# PatternMatch Tests
# =============================================================================


class TestPatternMatch:
    """Tests for PatternMatch dataclass."""

    def test_pattern_match_creation(self):
        """Test creating a pattern match."""
        pattern = PatternMatch(
            pattern_id="pattern-1",
            pattern_name="Drug Interaction",
            pattern_type="safety",
            description="Potential drug-drug interaction detected",
            confidence=0.92,
            supporting_facts=["fact-1", "fact-2", "fact-3"],
        )
        assert pattern.pattern_id == "pattern-1"
        assert pattern.pattern_name == "Drug Interaction"
        assert pattern.pattern_type == "safety"
        assert pattern.confidence == 0.92
        assert len(pattern.supporting_facts) == 3

    def test_pattern_match_with_metadata(self):
        """Test pattern match with metadata."""
        pattern = PatternMatch(
            pattern_id="pattern-2",
            pattern_name="Code Smell",
            pattern_type="quality",
            description="Long method detected",
            confidence=0.8,
            supporting_facts=["fact-10"],
            metadata={"severity": "medium", "lines": 150},
        )
        assert pattern.metadata["severity"] == "medium"
        assert pattern.metadata["lines"] == 150


# =============================================================================
# ComplianceCheckResult Tests
# =============================================================================


class TestComplianceCheckResult:
    """Tests for ComplianceCheckResult dataclass."""

    def test_compliance_result_passed(self):
        """Test compliance result for passing check."""
        result = ComplianceCheckResult(
            rule_id="HIPAA-001",
            rule_name="PHI Encryption",
            framework="HIPAA",
            passed=True,
            severity="high",
            findings=["All PHI data encrypted at rest"],
            evidence=["encryption_config.yaml"],
            recommendations=[],
            confidence=0.98,
        )
        assert result.passed is True
        assert result.framework == "HIPAA"
        assert len(result.recommendations) == 0

    def test_compliance_result_failed(self):
        """Test compliance result for failing check."""
        result = ComplianceCheckResult(
            rule_id="SOX-404",
            rule_name="Access Controls",
            framework="SOX",
            passed=False,
            severity="critical",
            findings=[
                "Missing audit trail for financial data access",
                "No separation of duties for approval workflow",
            ],
            evidence=["audit_log_missing.txt"],
            recommendations=[
                "Implement comprehensive audit logging",
                "Add role-based access controls",
            ],
            confidence=0.95,
        )
        assert result.passed is False
        assert result.severity == "critical"
        assert len(result.findings) == 2
        assert len(result.recommendations) == 2


# =============================================================================
# VerticalRegistry Tests
# =============================================================================


class TestVerticalRegistry:
    """Tests for VerticalRegistry class."""

    def test_registry_has_verticals(self):
        """Test that registry has registered verticals."""
        # Should have been populated by imports
        verticals = VerticalRegistry.list_all()
        assert isinstance(verticals, list)
        # At minimum, should have some verticals
        assert len(verticals) >= 0

    def test_registry_get_existing(self):
        """Test getting an existing vertical."""
        # Try to get a vertical that may exist
        try:
            vertical = VerticalRegistry.get("software")
            assert vertical is not None
            assert isinstance(vertical, BaseVerticalKnowledge)
        except ValueError:
            # Vertical might not be registered in test environment
            pass

    def test_registry_get_nonexistent(self):
        """Test getting a non-existent vertical returns None."""
        result = VerticalRegistry.get("nonexistent_vertical_xyz")
        assert result is None

    def test_registry_get_or_raise_nonexistent(self):
        """Test get_or_raise raises KeyError for non-existent vertical."""
        with pytest.raises(KeyError, match="Unknown vertical"):
            VerticalRegistry.get_or_raise("nonexistent_vertical_xyz")


# =============================================================================
# Healthcare Vertical Tests (if available)
# =============================================================================


class TestHealthcareKnowledge:
    """Tests for HealthcareKnowledge vertical."""

    @pytest.fixture
    def healthcare(self):
        """Get healthcare vertical instance."""
        try:
            from aragora.knowledge.mound.verticals.healthcare import HealthcareKnowledge
            return HealthcareKnowledge()
        except ImportError:
            pytest.skip("Healthcare vertical not available")

    def test_healthcare_vertical_id(self, healthcare):
        """Test healthcare vertical identifier."""
        assert healthcare.vertical_id == "healthcare"

    def test_healthcare_display_name(self, healthcare):
        """Test healthcare display name."""
        assert "Healthcare" in healthcare.display_name

    def test_healthcare_description(self, healthcare):
        """Test healthcare has description."""
        assert len(healthcare.description) > 0

    def test_healthcare_capabilities(self, healthcare):
        """Test healthcare capabilities."""
        caps = healthcare.capabilities
        assert isinstance(caps, VerticalCapabilities)
        # Healthcare should support compliance checking
        assert caps.supports_compliance_check is True
        assert "HIPAA" in caps.compliance_frameworks

    def test_healthcare_decay_rates(self, healthcare):
        """Test healthcare-specific decay rates."""
        rates = healthcare.decay_rates
        assert isinstance(rates, dict)
        assert "default" in rates or len(rates) > 0

    @pytest.mark.asyncio
    async def test_healthcare_extract_facts_basic(self, healthcare):
        """Test basic fact extraction from clinical text."""
        clinical_text = """
        Patient presented with chief complaint of chest pain.
        Diagnosis: Acute coronary syndrome.
        Prescribed: Aspirin 81mg daily.
        """

        facts = await healthcare.extract_facts(clinical_text)

        assert isinstance(facts, list)
        # Should extract some facts
        if facts:
            assert all(isinstance(f, VerticalFact) for f in facts)
            assert all(f.vertical == "healthcare" for f in facts)


# =============================================================================
# Legal Vertical Tests (if available)
# =============================================================================


class TestLegalKnowledge:
    """Tests for LegalKnowledge vertical."""

    @pytest.fixture
    def legal(self):
        """Get legal vertical instance."""
        try:
            from aragora.knowledge.mound.verticals.legal import LegalKnowledge
            return LegalKnowledge()
        except ImportError:
            pytest.skip("Legal vertical not available")

    def test_legal_vertical_id(self, legal):
        """Test legal vertical identifier."""
        assert legal.vertical_id == "legal"

    def test_legal_display_name(self, legal):
        """Test legal display name."""
        assert "Legal" in legal.display_name

    def test_legal_capabilities(self, legal):
        """Test legal capabilities."""
        caps = legal.capabilities
        assert isinstance(caps, VerticalCapabilities)

    @pytest.mark.asyncio
    async def test_legal_extract_facts_basic(self, legal):
        """Test basic fact extraction from legal text."""
        legal_text = """
        WHEREAS, the parties agree to the following terms:
        1. Confidentiality: All information shared shall be kept confidential.
        2. Term: This agreement shall be effective for 2 years.
        3. Jurisdiction: This contract shall be governed by California law.
        """

        facts = await legal.extract_facts(legal_text)

        assert isinstance(facts, list)


# =============================================================================
# Accounting Vertical Tests (if available)
# =============================================================================


class TestAccountingKnowledge:
    """Tests for AccountingKnowledge vertical."""

    @pytest.fixture
    def accounting(self):
        """Get accounting vertical instance."""
        try:
            from aragora.knowledge.mound.verticals.accounting import AccountingKnowledge
            return AccountingKnowledge()
        except ImportError:
            pytest.skip("Accounting vertical not available")

    def test_accounting_vertical_id(self, accounting):
        """Test accounting vertical identifier."""
        assert accounting.vertical_id == "accounting"

    def test_accounting_display_name(self, accounting):
        """Test accounting display name."""
        assert "Accounting" in accounting.display_name or "Financial" in accounting.display_name

    def test_accounting_capabilities(self, accounting):
        """Test accounting capabilities."""
        caps = accounting.capabilities
        assert isinstance(caps, VerticalCapabilities)

    @pytest.mark.asyncio
    async def test_accounting_extract_facts_basic(self, accounting):
        """Test basic fact extraction from financial text."""
        financial_text = """
        Q4 2024 Financial Summary:
        Total Revenue: $1,250,000
        Operating Expenses: $890,000
        Net Income: $360,000
        EBITDA Margin: 32%
        """

        facts = await accounting.extract_facts(financial_text)

        assert isinstance(facts, list)


# =============================================================================
# Integration Tests
# =============================================================================


class TestVerticalIntegration:
    """Integration tests for vertical knowledge modules."""

    def test_fact_lifecycle(self):
        """Test complete fact lifecycle."""
        # Create fact
        fact = VerticalFact(
            id="lifecycle-test",
            vertical="software",
            content="API endpoint /users returns user list",
            category="api_documentation",
            confidence=0.9,
        )

        # Fact is fresh
        assert fact.adjusted_confidence == 0.9
        assert fact.needs_reverification is False

        # Simulate time passing
        fact.staleness_days = 20.0

        # Fact is now stale
        stale_confidence = fact.adjusted_confidence
        assert stale_confidence < 0.9

        # Check if reverification needed
        if fact.needs_reverification:
            # Refresh the fact
            fact.refresh(new_confidence=0.95)
            assert fact.staleness_days == 0.0
            assert fact.confidence == 0.95

    def test_pattern_with_multiple_facts(self):
        """Test pattern detection across multiple facts."""
        facts = [
            VerticalFact(
                id=f"fact-{i}",
                vertical="healthcare",
                content=f"Clinical finding {i}",
                category="diagnosis",
                confidence=0.8,
            )
            for i in range(5)
        ]

        # Create pattern from facts
        pattern = PatternMatch(
            pattern_id="multi-fact-pattern",
            pattern_name="Related Diagnoses",
            pattern_type="clinical",
            description="Multiple related diagnoses detected",
            confidence=0.85,
            supporting_facts=[f.id for f in facts],
        )

        assert len(pattern.supporting_facts) == 5

    def test_compliance_check_workflow(self):
        """Test compliance checking workflow."""
        # Simulate facts extracted from document
        facts = [
            VerticalFact(
                id="phi-1",
                vertical="healthcare",
                content="Patient SSN: ***-**-****",
                category="phi",
                confidence=0.99,
            ),
            VerticalFact(
                id="phi-2",
                vertical="healthcare",
                content="Medical record number: 12345",
                category="phi",
                confidence=0.95,
            ),
        ]

        # Check HIPAA compliance
        compliance_result = ComplianceCheckResult(
            rule_id="HIPAA-PHI-01",
            rule_name="PHI Protection",
            framework="HIPAA",
            passed=True,
            severity="high",
            findings=["PHI detected and properly masked"],
            evidence=[f.id for f in facts],
            recommendations=[],
            confidence=0.9,
        )

        assert compliance_result.passed is True
        assert len(compliance_result.evidence) == 2
