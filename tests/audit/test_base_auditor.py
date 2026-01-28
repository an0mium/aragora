"""
Tests for Base Auditor Abstract Class.

Tests the base_auditor module that provides:
- AuditContext for passing session information
- ChunkData for normalized chunk access
- AuditorCapabilities declaration
- BaseAuditor abstract class
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from unittest.mock import AsyncMock, MagicMock
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
        document_ids=["doc-1"],
        model="claude-3.5-sonnet",
    )


@pytest.fixture
def audit_context(mock_session):
    """Create an AuditContext for testing."""
    from aragora.audit.base_auditor import AuditContext

    return AuditContext(
        session=mock_session,
        workspace_id="ws-123",
        model="claude-3.5-sonnet",
    )


@pytest.fixture
def sample_chunk():
    """Create a sample ChunkData for testing."""
    from aragora.audit.base_auditor import ChunkData

    return ChunkData(
        id="chunk-123",
        document_id="doc-1",
        content="This is test content for the chunk",
        chunk_type="text",
        page_number=1,
    )


# ===========================================================================
# Tests: AuditContext Dataclass
# ===========================================================================


class TestAuditContext:
    """Tests for AuditContext dataclass."""

    def test_creation(self, mock_session):
        """Test AuditContext creation."""
        from aragora.audit.base_auditor import AuditContext

        context = AuditContext(
            session=mock_session,
            workspace_id="ws-123",
            user_id="user-456",
        )

        assert context.session == mock_session
        assert context.workspace_id == "ws-123"
        assert context.user_id == "user-456"

    def test_default_values(self, mock_session):
        """Test AuditContext default values."""
        from aragora.audit.base_auditor import AuditContext

        context = AuditContext(session=mock_session)

        assert context.model == "claude-3.5-sonnet"
        assert context.max_findings_per_chunk == 50
        assert context.confidence_threshold == 0.5
        assert context.include_low_severity is True
        assert context.custom_params == {}

    def test_create_finding(self, audit_context):
        """Test AuditContext.create_finding helper."""
        from aragora.audit.document_auditor import FindingSeverity

        finding = audit_context.create_finding(
            document_id="doc-1",
            title="Test Finding",
            description="This is a test finding",
            severity=FindingSeverity.MEDIUM,
            category="test_category",
            confidence=0.85,
            evidence_text="evidence here",
        )

        assert finding.document_id == "doc-1"
        assert finding.title == "Test Finding"
        assert finding.severity == FindingSeverity.MEDIUM
        assert finding.confidence == 0.85
        assert finding.session_id == audit_context.session.id

    def test_create_finding_with_tags(self, audit_context):
        """Test create_finding with tags."""
        from aragora.audit.document_auditor import FindingSeverity

        finding = audit_context.create_finding(
            document_id="doc-1",
            title="Tagged Finding",
            description="Finding with tags",
            severity=FindingSeverity.HIGH,
            category="security",
            tags=["pii", "sensitive"],
        )

        assert finding.tags == ["pii", "sensitive"]


# ===========================================================================
# Tests: ChunkData Dataclass
# ===========================================================================


class TestChunkData:
    """Tests for ChunkData dataclass."""

    def test_creation(self):
        """Test ChunkData creation."""
        from aragora.audit.base_auditor import ChunkData

        chunk = ChunkData(
            id="chunk-1",
            document_id="doc-1",
            content="Test content",
            chunk_type="text",
        )

        assert chunk.id == "chunk-1"
        assert chunk.document_id == "doc-1"
        assert chunk.content == "Test content"
        assert chunk.chunk_type == "text"

    def test_default_values(self):
        """Test ChunkData default values."""
        from aragora.audit.base_auditor import ChunkData

        chunk = ChunkData(
            id="chunk-1",
            document_id="doc-1",
            content="Content",
        )

        assert chunk.chunk_type == "text"
        assert chunk.page_number is None
        assert chunk.heading_context == []
        assert chunk.metadata == {}

    def test_from_dict(self):
        """Test ChunkData.from_dict."""
        from aragora.audit.base_auditor import ChunkData

        data = {
            "id": "chunk-1",
            "document_id": "doc-1",
            "content": "Content from dict",
            "chunk_type": "code",
            "page_number": 5,
            "heading_context": ["Chapter 1", "Section A"],
        }

        chunk = ChunkData.from_dict(data)

        assert chunk.id == "chunk-1"
        assert chunk.chunk_type == "code"
        assert chunk.page_number == 5
        assert chunk.heading_context == ["Chapter 1", "Section A"]

    def test_from_dict_missing_values(self):
        """Test ChunkData.from_dict with missing values."""
        from aragora.audit.base_auditor import ChunkData

        data = {"content": "Just content"}

        chunk = ChunkData.from_dict(data)

        assert chunk.id == ""
        assert chunk.document_id == ""
        assert chunk.chunk_type == "text"

    def test_to_dict(self, sample_chunk):
        """Test ChunkData.to_dict."""
        result = sample_chunk.to_dict()

        assert result["id"] == "chunk-123"
        assert result["document_id"] == "doc-1"
        assert result["content"] == "This is test content for the chunk"
        assert result["chunk_type"] == "text"
        assert result["page_number"] == 1


# ===========================================================================
# Tests: AuditorCapabilities Dataclass
# ===========================================================================


class TestAuditorCapabilities:
    """Tests for AuditorCapabilities dataclass."""

    def test_creation(self):
        """Test AuditorCapabilities creation."""
        from aragora.audit.base_auditor import AuditorCapabilities

        caps = AuditorCapabilities(
            supports_chunk_analysis=True,
            supports_cross_document=True,
            requires_llm=True,
        )

        assert caps.supports_chunk_analysis is True
        assert caps.supports_cross_document is True
        assert caps.requires_llm is True

    def test_default_values(self):
        """Test AuditorCapabilities default values."""
        from aragora.audit.base_auditor import AuditorCapabilities

        caps = AuditorCapabilities()

        assert caps.supports_chunk_analysis is True
        assert caps.supports_cross_document is False
        assert caps.supports_streaming is False
        assert caps.requires_llm is False
        assert caps.min_chunk_size == 10
        assert caps.max_chunk_size == 100000

    def test_finding_categories(self):
        """Test finding_categories field."""
        from aragora.audit.base_auditor import AuditorCapabilities

        caps = AuditorCapabilities(finding_categories=["security", "compliance", "quality"])

        assert len(caps.finding_categories) == 3
        assert "security" in caps.finding_categories

    def test_supported_doc_types_sync(self):
        """Test supported_doc_types syncs with supported_document_types."""
        from aragora.audit.base_auditor import AuditorCapabilities

        caps = AuditorCapabilities(supported_doc_types=["pdf", "docx"])

        assert caps.supported_document_types == ["pdf", "docx"]


# ===========================================================================
# Tests: BaseAuditor Abstract Class
# ===========================================================================


class TestBaseAuditor:
    """Tests for BaseAuditor abstract class."""

    def test_cannot_instantiate_directly(self):
        """Test BaseAuditor cannot be instantiated directly."""
        from aragora.audit.base_auditor import BaseAuditor

        with pytest.raises(TypeError):
            BaseAuditor()

    def test_concrete_subclass_can_instantiate(self):
        """Test concrete subclass can be instantiated."""
        from aragora.audit.base_auditor import (
            AuditContext,
            AuditorCapabilities,
            BaseAuditor,
            ChunkData,
        )
        from aragora.audit.document_auditor import AuditFinding

        class TestAuditor(BaseAuditor):
            @property
            def audit_type_id(self) -> str:
                return "test_auditor"

            @property
            def display_name(self) -> str:
                return "Test Auditor"

            @property
            def description(self) -> str:
                return "Test auditor for unit tests"

            async def analyze_chunk(self, chunk: ChunkData, context: AuditContext):
                return []

        auditor = TestAuditor()

        assert auditor.audit_type_id == "test_auditor"
        assert auditor.display_name == "Test Auditor"

    def test_default_version(self):
        """Test default version is 1.0.0."""
        from aragora.audit.base_auditor import (
            AuditContext,
            BaseAuditor,
            ChunkData,
        )

        class TestAuditor(BaseAuditor):
            @property
            def audit_type_id(self) -> str:
                return "test"

            @property
            def display_name(self) -> str:
                return "Test"

            @property
            def description(self) -> str:
                return "Test"

            async def analyze_chunk(self, chunk: ChunkData, context: AuditContext):
                return []

        auditor = TestAuditor()

        assert auditor.version == "1.0.0"

    def test_default_author(self):
        """Test default author is aragora."""
        from aragora.audit.base_auditor import (
            AuditContext,
            BaseAuditor,
            ChunkData,
        )

        class TestAuditor(BaseAuditor):
            @property
            def audit_type_id(self) -> str:
                return "test"

            @property
            def display_name(self) -> str:
                return "Test"

            @property
            def description(self) -> str:
                return "Test"

            async def analyze_chunk(self, chunk: ChunkData, context: AuditContext):
                return []

        auditor = TestAuditor()

        assert auditor.author == "aragora"

    def test_default_capabilities(self):
        """Test default capabilities."""
        from aragora.audit.base_auditor import (
            AuditContext,
            BaseAuditor,
            ChunkData,
        )

        class TestAuditor(BaseAuditor):
            @property
            def audit_type_id(self) -> str:
                return "test"

            @property
            def display_name(self) -> str:
                return "Test"

            @property
            def description(self) -> str:
                return "Test"

            async def analyze_chunk(self, chunk: ChunkData, context: AuditContext):
                return []

        auditor = TestAuditor()
        caps = auditor.capabilities

        assert caps.supports_chunk_analysis is True
        assert caps.supports_cross_document is False

    def test_validate_finding_confidence_threshold(self, audit_context):
        """Test validate_finding filters by confidence threshold."""
        from aragora.audit.base_auditor import (
            AuditContext,
            BaseAuditor,
            ChunkData,
        )
        from aragora.audit.document_auditor import FindingSeverity

        class TestAuditor(BaseAuditor):
            @property
            def audit_type_id(self) -> str:
                return "test"

            @property
            def display_name(self) -> str:
                return "Test"

            @property
            def description(self) -> str:
                return "Test"

            async def analyze_chunk(self, chunk: ChunkData, context: AuditContext):
                return []

        auditor = TestAuditor()

        # Finding below threshold
        low_conf_finding = audit_context.create_finding(
            document_id="doc-1",
            title="Low confidence",
            description="Below threshold",
            severity=FindingSeverity.MEDIUM,
            category="test",
            confidence=0.3,  # Below 0.5 threshold
        )

        assert auditor.validate_finding(low_conf_finding, audit_context) is False

        # Finding above threshold
        high_conf_finding = audit_context.create_finding(
            document_id="doc-1",
            title="High confidence",
            description="Above threshold",
            severity=FindingSeverity.MEDIUM,
            category="test",
            confidence=0.8,
        )

        assert auditor.validate_finding(high_conf_finding, audit_context) is True

    def test_validate_finding_low_severity_filter(self, mock_session):
        """Test validate_finding filters low severity when configured."""
        from aragora.audit.base_auditor import (
            AuditContext,
            BaseAuditor,
            ChunkData,
        )
        from aragora.audit.document_auditor import FindingSeverity

        class TestAuditor(BaseAuditor):
            @property
            def audit_type_id(self) -> str:
                return "test"

            @property
            def display_name(self) -> str:
                return "Test"

            @property
            def description(self) -> str:
                return "Test"

            async def analyze_chunk(self, chunk: ChunkData, context: AuditContext):
                return []

        auditor = TestAuditor()

        # Context that filters low severity
        context = AuditContext(
            session=mock_session,
            include_low_severity=False,
        )

        low_severity_finding = context.create_finding(
            document_id="doc-1",
            title="Low severity",
            description="Should be filtered",
            severity=FindingSeverity.LOW,
            category="test",
            confidence=0.9,
        )

        assert auditor.validate_finding(low_severity_finding, context) is False

    def test_repr(self):
        """Test __repr__ format."""
        from aragora.audit.base_auditor import (
            AuditContext,
            BaseAuditor,
            ChunkData,
        )

        class TestAuditor(BaseAuditor):
            @property
            def audit_type_id(self) -> str:
                return "test_repr"

            @property
            def display_name(self) -> str:
                return "Test"

            @property
            def description(self) -> str:
                return "Test"

            async def analyze_chunk(self, chunk: ChunkData, context: AuditContext):
                return []

        auditor = TestAuditor()
        repr_str = repr(auditor)

        assert "TestAuditor" in repr_str
        assert "test_repr" in repr_str
        assert "1.0.0" in repr_str

    @pytest.mark.asyncio
    async def test_cross_document_analysis_default(self, audit_context, sample_chunk):
        """Test default cross_document_analysis returns empty list."""
        from aragora.audit.base_auditor import (
            AuditContext,
            BaseAuditor,
            ChunkData,
        )

        class TestAuditor(BaseAuditor):
            @property
            def audit_type_id(self) -> str:
                return "test"

            @property
            def display_name(self) -> str:
                return "Test"

            @property
            def description(self) -> str:
                return "Test"

            async def analyze_chunk(self, chunk: ChunkData, context: AuditContext):
                return []

        auditor = TestAuditor()
        result = await auditor.cross_document_analysis([sample_chunk], audit_context)

        assert result == []

    @pytest.mark.asyncio
    async def test_pre_audit_hook_default(self, audit_context):
        """Test default pre_audit_hook does nothing."""
        from aragora.audit.base_auditor import (
            AuditContext,
            BaseAuditor,
            ChunkData,
        )

        class TestAuditor(BaseAuditor):
            @property
            def audit_type_id(self) -> str:
                return "test"

            @property
            def display_name(self) -> str:
                return "Test"

            @property
            def description(self) -> str:
                return "Test"

            async def analyze_chunk(self, chunk: ChunkData, context: AuditContext):
                return []

        auditor = TestAuditor()

        # Should not raise
        await auditor.pre_audit_hook(audit_context)

    @pytest.mark.asyncio
    async def test_post_audit_hook_default(self, audit_context):
        """Test default post_audit_hook returns findings unchanged."""
        from aragora.audit.base_auditor import (
            AuditContext,
            BaseAuditor,
            ChunkData,
        )
        from aragora.audit.document_auditor import FindingSeverity

        class TestAuditor(BaseAuditor):
            @property
            def audit_type_id(self) -> str:
                return "test"

            @property
            def display_name(self) -> str:
                return "Test"

            @property
            def description(self) -> str:
                return "Test"

            async def analyze_chunk(self, chunk: ChunkData, context: AuditContext):
                return []

        auditor = TestAuditor()

        findings = [
            audit_context.create_finding(
                document_id="doc-1",
                title="Finding 1",
                description="Desc",
                severity=FindingSeverity.MEDIUM,
                category="test",
            )
        ]

        result = await auditor.post_audit_hook(findings, audit_context)

        assert result == findings
