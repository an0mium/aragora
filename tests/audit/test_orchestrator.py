"""
Tests for Multi-Vertical Audit Orchestrator.

Tests the orchestrator module that provides:
- Parallel auditor execution
- Finding aggregation and deduplication
- Cross-auditor correlation
- Profile-based configuration
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def mock_session():
    """Create a mock AuditSession for testing."""
    from aragora.audit.document_auditor import AuditSession

    return AuditSession(
        id="session-test-123",
        created_by="user-123",
        document_ids=["doc-1", "doc-2"],
        model="claude-3.5-sonnet",
    )


@pytest.fixture
def sample_chunks():
    """Create sample document chunks for testing."""
    return [
        {
            "id": "chunk-1",
            "document_id": "doc-1",
            "content": "This is test content for chunk 1",
            "chunk_type": "text",
        },
        {
            "id": "chunk-2",
            "document_id": "doc-1",
            "content": "This is test content for chunk 2",
            "chunk_type": "text",
        },
    ]


# ===========================================================================
# Tests: AuditVertical Enum
# ===========================================================================


class TestAuditVertical:
    """Tests for AuditVertical enum."""

    def test_core_verticals_exist(self):
        """Test core verticals are defined."""
        from aragora.audit.orchestrator import AuditVertical

        assert AuditVertical.SECURITY.value == "security"
        assert AuditVertical.COMPLIANCE.value == "compliance"
        assert AuditVertical.QUALITY.value == "quality"
        assert AuditVertical.CONSISTENCY.value == "consistency"

    def test_domain_verticals_exist(self):
        """Test domain verticals are defined."""
        from aragora.audit.orchestrator import AuditVertical

        assert AuditVertical.LEGAL.value == "legal"
        assert AuditVertical.ACCOUNTING.value == "accounting"
        assert AuditVertical.SOFTWARE.value == "software"
        assert AuditVertical.HEALTHCARE.value == "healthcare"
        assert AuditVertical.REGULATORY.value == "regulatory"
        assert AuditVertical.ACADEMIC.value == "academic"


# ===========================================================================
# Tests: AuditProfile Dataclass
# ===========================================================================


class TestAuditProfile:
    """Tests for AuditProfile dataclass."""

    def test_creation(self):
        """Test AuditProfile creation."""
        from aragora.audit.orchestrator import AuditProfile, AuditVertical

        profile = AuditProfile(
            name="Test Profile",
            description="Test description",
            verticals=[AuditVertical.SECURITY, AuditVertical.COMPLIANCE],
            priority_order=[AuditVertical.SECURITY, AuditVertical.COMPLIANCE],
        )

        assert profile.name == "Test Profile"
        assert len(profile.verticals) == 2
        assert profile.parallel_execution is True
        assert profile.max_concurrent == 5

    def test_default_values(self):
        """Test AuditProfile default values."""
        from aragora.audit.orchestrator import AuditProfile, AuditVertical

        profile = AuditProfile(
            name="Minimal",
            description="Minimal profile",
            verticals=[AuditVertical.QUALITY],
            priority_order=[AuditVertical.QUALITY],
        )

        assert profile.confidence_threshold == 0.5
        assert profile.include_low_severity is False
        assert profile.custom_config == {}


# ===========================================================================
# Tests: AUDIT_PROFILES Configuration
# ===========================================================================


class TestAuditProfiles:
    """Tests for pre-defined audit profiles."""

    def test_enterprise_full_profile(self):
        """Test enterprise_full profile exists and is complete."""
        from aragora.audit.orchestrator import AUDIT_PROFILES, AuditVertical

        profile = AUDIT_PROFILES["enterprise_full"]

        assert profile.name == "Enterprise Full Audit"
        assert len(profile.verticals) == len(AuditVertical)
        assert profile.parallel_execution is True

    def test_healthcare_hipaa_profile(self):
        """Test healthcare_hipaa profile configuration."""
        from aragora.audit.orchestrator import AUDIT_PROFILES, AuditVertical

        profile = AUDIT_PROFILES["healthcare_hipaa"]

        assert AuditVertical.HEALTHCARE in profile.verticals
        assert AuditVertical.SECURITY in profile.verticals
        assert profile.confidence_threshold == 0.6

    def test_financial_sox_profile(self):
        """Test financial_sox profile configuration."""
        from aragora.audit.orchestrator import AUDIT_PROFILES, AuditVertical

        profile = AUDIT_PROFILES["financial_sox"]

        assert AuditVertical.ACCOUNTING in profile.verticals
        assert AuditVertical.REGULATORY in profile.verticals

    def test_quick_security_profile(self):
        """Test quick_security profile is non-parallel."""
        from aragora.audit.orchestrator import AUDIT_PROFILES, AuditVertical

        profile = AUDIT_PROFILES["quick_security"]

        assert profile.verticals == [AuditVertical.SECURITY]
        assert profile.parallel_execution is False


# ===========================================================================
# Tests: OrchestratorResult Dataclass
# ===========================================================================


class TestOrchestratorResult:
    """Tests for OrchestratorResult dataclass."""

    def test_creation(self):
        """Test OrchestratorResult creation."""
        from aragora.audit.orchestrator import OrchestratorResult

        now = datetime.now(timezone.utc)
        result = OrchestratorResult(
            session_id="session-123",
            profile="test_profile",
            verticals_run=["security", "compliance"],
            findings=[],
            findings_by_vertical={},
            findings_by_severity={"critical": 0, "high": 1, "medium": 2, "low": 0, "info": 0},
            total_chunks_processed=10,
            duration_ms=1500.0,
            errors=[],
            started_at=now,
            completed_at=now,
        )

        assert result.session_id == "session-123"
        assert result.profile == "test_profile"
        assert result.total_chunks_processed == 10

    def test_to_dict(self):
        """Test OrchestratorResult.to_dict."""
        from aragora.audit.orchestrator import OrchestratorResult

        now = datetime.now(timezone.utc)
        result = OrchestratorResult(
            session_id="session-123",
            profile="test_profile",
            verticals_run=["security"],
            findings=[],
            findings_by_vertical={"security": []},
            findings_by_severity={"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0},
            total_chunks_processed=5,
            duration_ms=500.0,
            errors=[],
            started_at=now,
            completed_at=now,
        )

        result_dict = result.to_dict()

        assert "session_id" in result_dict
        assert "findings_count" in result_dict
        assert "duration_ms" in result_dict
        assert result_dict["findings_count"] == 0


# ===========================================================================
# Tests: AuditOrchestrator Initialization
# ===========================================================================


class TestAuditOrchestratorInit:
    """Tests for AuditOrchestrator initialization."""

    def test_init_with_profile(self):
        """Test initialization with profile name."""
        from aragora.audit.orchestrator import AuditOrchestrator

        orchestrator = AuditOrchestrator(profile="quick_security")

        assert orchestrator._profile.name == "Quick Security Scan"

    def test_init_with_custom_verticals(self):
        """Test initialization with custom verticals."""
        from aragora.audit.orchestrator import AuditOrchestrator, AuditVertical

        orchestrator = AuditOrchestrator(verticals=[AuditVertical.SECURITY, AuditVertical.QUALITY])

        assert orchestrator._profile.name == "Custom"
        assert len(orchestrator._profile.verticals) == 2

    def test_init_default_profile(self):
        """Test initialization defaults to enterprise_full."""
        from aragora.audit.orchestrator import AuditOrchestrator

        orchestrator = AuditOrchestrator()

        assert orchestrator._profile.name == "Enterprise Full Audit"

    def test_init_with_workspace(self):
        """Test initialization with workspace ID."""
        from aragora.audit.orchestrator import AuditOrchestrator

        orchestrator = AuditOrchestrator(profile="quick_security", workspace_id="ws-123")

        assert orchestrator._workspace_id == "ws-123"


# ===========================================================================
# Tests: AuditOrchestrator Class Methods
# ===========================================================================


class TestAuditOrchestratorClassMethods:
    """Tests for AuditOrchestrator class methods."""

    def test_list_profiles(self):
        """Test list_profiles returns all profiles."""
        from aragora.audit.orchestrator import AuditOrchestrator, AUDIT_PROFILES

        profiles = AuditOrchestrator.list_profiles()

        assert len(profiles) == len(AUDIT_PROFILES)
        assert all("name" in p for p in profiles)
        assert all("description" in p for p in profiles)
        assert all("verticals" in p for p in profiles)

    def test_list_verticals(self):
        """Test list_verticals returns all verticals."""
        from aragora.audit.orchestrator import AuditOrchestrator, AuditVertical

        verticals = AuditOrchestrator.list_verticals()

        assert len(verticals) == len(AuditVertical)
        assert all("id" in v for v in verticals)
        assert all("name" in v for v in verticals)
        assert all("auditor" in v for v in verticals)


# ===========================================================================
# Tests: VERTICAL_AUDITORS Mapping
# ===========================================================================


class TestVerticalAuditors:
    """Tests for VERTICAL_AUDITORS mapping."""

    def test_all_verticals_have_auditors(self):
        """Test all verticals are mapped to auditors."""
        from aragora.audit.orchestrator import VERTICAL_AUDITORS, AuditVertical

        for vertical in AuditVertical:
            assert vertical in VERTICAL_AUDITORS

    def test_auditor_classes_are_valid(self):
        """Test auditor classes can be instantiated."""
        from aragora.audit.orchestrator import VERTICAL_AUDITORS

        for vertical, auditor_class in VERTICAL_AUDITORS.items():
            # Just check it's a class, not instantiate (some may require setup)
            assert isinstance(auditor_class, type)
