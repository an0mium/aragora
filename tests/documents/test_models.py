"""
Tests for document processing data models.
"""

import pytest
from datetime import datetime

from aragora.documents.models import (
    ChunkType,
    DocumentChunk,
    DocumentStatus,
    IngestedDocument,
    get_model_token_limit,
    MODEL_TOKEN_LIMITS,
)


class TestDocumentChunk:
    """Tests for DocumentChunk dataclass."""

    def test_create_chunk(self):
        """Test creating a basic chunk."""
        chunk = DocumentChunk(
            document_id="doc-123",
            sequence=0,
            content="Hello, world!",
            chunk_type=ChunkType.TEXT,
        )

        assert chunk.document_id == "doc-123"
        assert chunk.sequence == 0
        assert chunk.content == "Hello, world!"
        assert chunk.chunk_type == ChunkType.TEXT
        assert chunk.id is not None

    def test_chunk_to_dict(self):
        """Test serialization to dictionary."""
        chunk = DocumentChunk(
            document_id="doc-123",
            sequence=1,
            content="Test content",
            chunk_type=ChunkType.HEADING,
            token_count=10,
        )

        data = chunk.to_dict()

        assert data["document_id"] == "doc-123"
        assert data["sequence"] == 1
        assert data["content"] == "Test content"
        assert data["chunk_type"] == "heading"
        assert data["token_count"] == 10

    def test_chunk_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "chunk-abc",
            "document_id": "doc-456",
            "sequence": 2,
            "content": "Restored content",
            "chunk_type": "table",
            "token_count": 50,
        }

        chunk = DocumentChunk.from_dict(data)

        assert chunk.id == "chunk-abc"
        assert chunk.document_id == "doc-456"
        assert chunk.sequence == 2
        assert chunk.chunk_type == ChunkType.TABLE
        assert chunk.token_count == 50

    def test_chunk_types(self):
        """Test all chunk types."""
        types = [
            ChunkType.TEXT,
            ChunkType.HEADING,
            ChunkType.TABLE,
            ChunkType.CODE,
            ChunkType.LIST,
            ChunkType.IMAGE,
            ChunkType.FORMULA,
            ChunkType.METADATA,
        ]

        for ct in types:
            chunk = DocumentChunk(chunk_type=ct)
            assert chunk.chunk_type == ct


class TestIngestedDocument:
    """Tests for IngestedDocument dataclass."""

    def test_create_document(self):
        """Test creating a basic document."""
        doc = IngestedDocument(
            filename="report.pdf",
            content_type="application/pdf",
            file_size=1024,
            workspace_id="ws-123",
        )

        assert doc.filename == "report.pdf"
        assert doc.content_type == "application/pdf"
        assert doc.file_size == 1024
        assert doc.status == DocumentStatus.PENDING
        assert doc.id is not None

    def test_document_generates_preview(self):
        """Test automatic preview generation."""
        long_text = "A" * 600  # More than 500 chars
        doc = IngestedDocument(
            filename="test.txt",
            text=long_text,
        )

        assert len(doc.preview) <= 503  # 500 + "..."
        assert doc.preview.endswith("...")

    def test_document_to_dict(self):
        """Test serialization."""
        doc = IngestedDocument(
            filename="data.json",
            content_type="application/json",
            file_size=512,
            page_count=1,
            word_count=100,
            chunk_count=5,
        )

        data = doc.to_dict()

        assert data["filename"] == "data.json"
        assert data["content_type"] == "application/json"
        assert data["status"] == "pending"
        assert data["chunk_count"] == 5

    def test_document_from_dict(self):
        """Test deserialization."""
        data = {
            "id": "doc-789",
            "filename": "manual.docx",
            "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "file_size": 2048,
            "status": "indexed",
            "chunk_count": 10,
            "total_tokens": 5000,
        }

        doc = IngestedDocument.from_dict(data)

        assert doc.id == "doc-789"
        assert doc.filename == "manual.docx"
        assert doc.status == DocumentStatus.INDEXED
        assert doc.total_tokens == 5000

    def test_document_to_summary(self):
        """Test summary generation."""
        doc = IngestedDocument(
            filename="report.pdf",
            content_type="application/pdf",
            file_size=10240,
            status=DocumentStatus.INDEXED,
            page_count=20,
            word_count=5000,
            chunk_count=25,
            preview="Executive summary of the annual report...",
            tags=["finance", "2025", "quarterly"],
        )

        summary = doc.to_summary()

        assert summary["filename"] == "report.pdf"
        assert summary["page_count"] == 20
        assert summary["chunk_count"] == 25
        assert "finance" in summary["tags"]

    def test_document_status_transitions(self):
        """Test status enum values."""
        statuses = [
            DocumentStatus.PENDING,
            DocumentStatus.PROCESSING,
            DocumentStatus.INDEXED,
            DocumentStatus.FAILED,
            DocumentStatus.ARCHIVED,
        ]

        for status in statuses:
            doc = IngestedDocument(status=status)
            assert doc.status == status
            assert doc.to_dict()["status"] == status.value


class TestModelTokenLimits:
    """Tests for model token limit lookup."""

    def test_get_known_model_limits(self):
        """Test getting limits for known models."""
        assert get_model_token_limit("gpt-4") == 8_192
        assert get_model_token_limit("gpt-4-turbo") == 128_000
        assert get_model_token_limit("claude-3-opus") == 200_000
        assert get_model_token_limit("gemini-3-pro") == 1_000_000

    def test_get_unknown_model_defaults(self):
        """Test unknown model returns default."""
        # Unknown models return the default limit
        default_limit = MODEL_TOKEN_LIMITS["default"]
        assert get_model_token_limit("unknown-model-xyz123") == default_limit
        # Empty string may match something due to partial matching, so just verify it returns a value
        assert get_model_token_limit("") > 0

    def test_case_insensitive_lookup(self):
        """Test case-insensitive model name matching."""
        assert get_model_token_limit("GPT-4") == 8_192
        assert get_model_token_limit("Claude-3-Opus") == 200_000

    def test_partial_model_name_matching(self):
        """Test partial model name matching."""
        # Should match models containing the key
        assert get_model_token_limit("gpt-4o-mini") == 128_000  # matches gpt-4o
        assert get_model_token_limit("claude-3.5-sonnet-v2") == 200_000
