"""
Tests for DocumentsAPI resource.

Tests cover:
- Document management (list, get, upload, delete, formats)
- Batch processing (upload, status, results, cancel)
- Document content (chunks, context)
- Audit sessions (create, list, get, start/pause/resume/cancel)
- Audit findings and reports
- MIME type guessing
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aragora.client import AragoraClient
from aragora.client.models import (
    AuditFinding,
    AuditReport,
    AuditSession,
    AuditSessionCreateResponse,
    AuditSessionStatus,
    AuditType,
    BatchJobResults,
    BatchJobStatus,
    BatchUploadResponse,
    Document,
    DocumentChunk,
    DocumentContext,
    DocumentUploadResponse,
    FindingSeverity,
    ProcessingStats,
    SupportedFormats,
)
from aragora.client.resources.documents import DocumentsAPI


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def client():
    """Create a basic client for testing."""
    return AragoraClient(base_url="http://test.example.com", api_key="test-key")


@pytest.fixture
def docs_api(client):
    """Create a DocumentsAPI instance for testing."""
    return client.documents


@pytest.fixture
def temp_file():
    """Create a temporary file for upload testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Test document content")
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_files():
    """Create multiple temporary files for batch upload testing."""
    paths = []
    for i in range(3):
        with tempfile.NamedTemporaryFile(mode="w", suffix=f"_{i}.txt", delete=False) as f:
            f.write(f"Test document {i} content")
            paths.append(f.name)
    yield paths
    for path in paths:
        Path(path).unlink(missing_ok=True)


# ============================================================================
# Document Management Tests
# ============================================================================


class TestListDocuments:
    """Tests for list() method."""

    def test_list_documents(self, docs_api, client):
        """Test listing documents."""
        now = datetime.now()
        client._get = MagicMock(
            return_value={
                "documents": [
                    {
                        "id": "doc1",
                        "filename": "contract.pdf",
                        "mime_type": "application/pdf",
                        "size_bytes": 1024,
                        "status": "completed",
                        "created_at": now,
                        "chunk_count": 10,
                    },
                    {
                        "id": "doc2",
                        "filename": "report.docx",
                        "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        "size_bytes": 2048,
                        "status": "processing",
                        "created_at": now,
                        "chunk_count": 5,
                    },
                ],
            }
        )

        result = docs_api.list()

        client._get.assert_called_once()
        call_args = client._get.call_args
        assert call_args[0][0] == "/api/documents"
        assert call_args[1]["params"]["limit"] == 50
        assert len(result) == 2
        assert all(isinstance(d, Document) for d in result)

    def test_list_with_status_filter(self, docs_api, client):
        """Test listing documents with status filter."""
        client._get = MagicMock(return_value={"documents": []})

        docs_api.list(status="completed")

        call_args = client._get.call_args
        assert call_args[1]["params"]["status"] == "completed"

    def test_list_with_pagination(self, docs_api, client):
        """Test listing documents with pagination."""
        client._get = MagicMock(return_value={"documents": []})

        docs_api.list(limit=100, offset=50)

        call_args = client._get.call_args
        assert call_args[1]["params"]["limit"] == 100
        assert call_args[1]["params"]["offset"] == 50


class TestGetDocument:
    """Tests for get() method."""

    def test_get_document(self, docs_api, client):
        """Test getting a document by ID."""
        now = datetime.now()
        client._get = MagicMock(
            return_value={
                "id": "doc123",
                "filename": "contract.pdf",
                "mime_type": "application/pdf",
                "size_bytes": 5000,
                "status": "completed",
                "created_at": now,
                "chunk_count": 25,
            }
        )

        result = docs_api.get("doc123")

        client._get.assert_called_once_with("/api/documents/doc123")
        assert isinstance(result, Document)
        assert result.id == "doc123"
        assert result.filename == "contract.pdf"


class TestUploadDocument:
    """Tests for upload() method."""

    def test_upload_document(self, docs_api, client, temp_file):
        """Test uploading a document."""
        client._post = MagicMock(
            return_value={
                "document_id": "doc123",
                "status": "pending",
                "filename": Path(temp_file).name,
            }
        )

        result = docs_api.upload(temp_file)

        client._post.assert_called_once()
        call_args = client._post.call_args
        assert call_args[0][0] == "/api/documents/upload"
        assert "content" in call_args[0][1]
        assert "filename" in call_args[0][1]
        assert isinstance(result, DocumentUploadResponse)
        assert result.document_id == "doc123"

    def test_upload_with_metadata(self, docs_api, client, temp_file):
        """Test uploading with metadata."""
        client._post = MagicMock(
            return_value={
                "document_id": "doc123",
                "filename": Path(temp_file).name,
                "status": "pending",
            }
        )

        docs_api.upload(temp_file, metadata={"category": "legal"})

        call_args = client._post.call_args
        assert call_args[0][1]["metadata"] == {"category": "legal"}

    def test_upload_file_not_found(self, docs_api):
        """Test upload raises error for missing file."""
        with pytest.raises(FileNotFoundError):
            docs_api.upload("/nonexistent/file.pdf")


class TestDeleteDocument:
    """Tests for delete() method."""

    def test_delete_document(self, docs_api, client):
        """Test deleting a document."""
        client._delete = MagicMock(return_value={"success": True})

        result = docs_api.delete("doc123")

        client._delete.assert_called_once_with("/api/documents/doc123")
        assert result is True


class TestSupportedFormats:
    """Tests for formats() method."""

    def test_get_formats(self, docs_api, client):
        """Test getting supported formats."""
        client._get = MagicMock(
            return_value={
                "formats": ["pdf", "docx", "txt", "md"],
                "mime_types": ["application/pdf", "text/plain"],
            }
        )

        result = docs_api.formats()

        client._get.assert_called_once_with("/api/documents/formats")
        assert isinstance(result, SupportedFormats)


# ============================================================================
# Batch Processing Tests
# ============================================================================


class TestBatchUpload:
    """Tests for batch_upload() method."""

    def test_batch_upload(self, docs_api, client, temp_files):
        """Test batch uploading documents."""
        now = datetime.now()
        client._post = MagicMock(
            return_value={
                "job_id": "job123",
                "document_count": 3,
                "status": "pending",
                "created_at": now,
            }
        )

        result = docs_api.batch_upload(temp_files)

        client._post.assert_called_once()
        call_args = client._post.call_args
        assert call_args[0][0] == "/api/documents/batch"
        assert len(call_args[0][1]["files"]) == 3
        assert isinstance(result, BatchUploadResponse)
        assert result.job_id == "job123"

    def test_batch_upload_with_metadata(self, docs_api, client, temp_files):
        """Test batch upload with metadata."""
        now = datetime.now()
        client._post = MagicMock(
            return_value={
                "job_id": "job123",
                "document_count": 3,
                "status": "pending",
                "created_at": now,
            }
        )

        docs_api.batch_upload(temp_files, metadata={"batch": "test"})

        call_args = client._post.call_args
        assert call_args[0][1]["metadata"] == {"batch": "test"}

    def test_batch_upload_file_not_found(self, docs_api):
        """Test batch upload raises error for missing file."""
        with pytest.raises(FileNotFoundError):
            docs_api.batch_upload(["/nonexistent/file1.pdf", "/nonexistent/file2.pdf"])


class TestBatchStatus:
    """Tests for batch_status() method."""

    def test_batch_status(self, docs_api, client):
        """Test getting batch job status."""
        now = datetime.now()
        client._get = MagicMock(
            return_value={
                "job_id": "job123",
                "status": "processing",
                "progress": 0.5,
                "document_count": 10,
                "completed_count": 5,
                "created_at": now,
            }
        )

        result = docs_api.batch_status("job123")

        client._get.assert_called_once_with("/api/documents/batch/job123")
        assert isinstance(result, BatchJobStatus)
        assert result.progress == 0.5


class TestBatchResults:
    """Tests for batch_results() method."""

    def test_batch_results(self, docs_api, client):
        """Test getting batch job results."""
        now = datetime.now()
        client._get = MagicMock(
            return_value={
                "job_id": "job123",
                "documents": [
                    {
                        "id": "doc1",
                        "filename": "file1.txt",
                        "mime_type": "text/plain",
                        "size_bytes": 100,
                        "status": "completed",
                        "created_at": now,
                        "chunk_count": 5,
                    },
                    {
                        "id": "doc2",
                        "filename": "file2.txt",
                        "mime_type": "text/plain",
                        "size_bytes": 200,
                        "status": "completed",
                        "created_at": now,
                        "chunk_count": 3,
                    },
                ],
            }
        )

        result = docs_api.batch_results("job123")

        client._get.assert_called_once_with("/api/documents/batch/job123/results")
        assert isinstance(result, BatchJobResults)


class TestBatchCancel:
    """Tests for batch_cancel() method."""

    def test_batch_cancel(self, docs_api, client):
        """Test canceling a batch job."""
        client._delete = MagicMock(return_value={"success": True})

        result = docs_api.batch_cancel("job123")

        client._delete.assert_called_once_with("/api/documents/batch/job123")
        assert result is True


class TestProcessingStats:
    """Tests for processing_stats() method."""

    def test_processing_stats(self, docs_api, client):
        """Test getting processing statistics."""
        client._get = MagicMock(
            return_value={
                "total_documents": 100,
                "pending": 5,
                "processing": 10,
                "completed": 80,
                "failed": 5,
            }
        )

        result = docs_api.processing_stats()

        client._get.assert_called_once_with("/api/documents/processing/stats")
        assert isinstance(result, ProcessingStats)


# ============================================================================
# Document Content Tests
# ============================================================================


class TestDocumentChunks:
    """Tests for chunks() method."""

    def test_get_chunks(self, docs_api, client):
        """Test getting document chunks."""
        client._get = MagicMock(
            return_value={
                "chunks": [
                    {
                        "id": "chunk1",
                        "document_id": "doc123",
                        "chunk_index": 0,
                        "content": "First chunk",
                        "token_count": 10,
                    },
                    {
                        "id": "chunk2",
                        "document_id": "doc123",
                        "chunk_index": 1,
                        "content": "Second chunk",
                        "token_count": 12,
                    },
                ],
            }
        )

        result = docs_api.chunks("doc123")

        client._get.assert_called_once()
        call_args = client._get.call_args
        assert call_args[0][0] == "/api/documents/doc123/chunks"
        assert len(result) == 2
        assert all(isinstance(c, DocumentChunk) for c in result)

    def test_get_chunks_with_pagination(self, docs_api, client):
        """Test getting chunks with pagination."""
        client._get = MagicMock(return_value={"chunks": []})

        docs_api.chunks("doc123", limit=50, offset=25)

        call_args = client._get.call_args
        assert call_args[1]["params"]["limit"] == 50
        assert call_args[1]["params"]["offset"] == 25


class TestDocumentContext:
    """Tests for context() method."""

    def test_get_context(self, docs_api, client):
        """Test getting document context."""
        client._get = MagicMock(
            return_value={
                "document_id": "doc123",
                "context": "Combined document content...",
                "total_tokens": 5000,
                "chunks_used": 10,
            }
        )

        result = docs_api.context("doc123")

        client._get.assert_called_once()
        call_args = client._get.call_args
        assert call_args[0][0] == "/api/documents/doc123/context"
        assert isinstance(result, DocumentContext)

    def test_get_context_with_options(self, docs_api, client):
        """Test getting context with custom options."""
        client._get = MagicMock(
            return_value={
                "document_id": "doc123",
                "context": "Content...",
                "total_tokens": 10000,
                "chunks_used": 15,
            }
        )

        docs_api.context("doc123", max_tokens=50000, model="gpt-4")

        call_args = client._get.call_args
        assert call_args[1]["params"]["max_tokens"] == 50000
        assert call_args[1]["params"]["model"] == "gpt-4"


# ============================================================================
# Audit Session Tests
# ============================================================================


class TestCreateAudit:
    """Tests for create_audit() method."""

    def test_create_audit(self, docs_api, client):
        """Test creating an audit session."""
        client._post = MagicMock(
            return_value={
                "session_id": "sess123",
                "status": "pending",
                "document_count": 2,
                "audit_types": ["security", "compliance"],
            }
        )

        result = docs_api.create_audit(["doc1", "doc2"])

        client._post.assert_called_once()
        call_args = client._post.call_args
        assert call_args[0][0] == "/api/audit/sessions"
        assert call_args[0][1]["document_ids"] == ["doc1", "doc2"]
        assert isinstance(result, AuditSessionCreateResponse)

    def test_create_audit_with_options(self, docs_api, client):
        """Test creating audit with custom options."""
        client._post = MagicMock(
            return_value={
                "session_id": "sess123",
                "status": "pending",
            }
        )

        docs_api.create_audit(
            ["doc1"],
            audit_types=["security"],
            model="gpt-4",
            strict=True,
        )

        call_args = client._post.call_args
        assert call_args[0][1]["audit_types"] == ["security"]
        assert call_args[0][1]["model"] == "gpt-4"


class TestListAudits:
    """Tests for list_audits() method."""

    def test_list_audits(self, docs_api, client):
        """Test listing audit sessions."""
        now = datetime.now()
        client._get = MagicMock(
            return_value={
                "sessions": [
                    {
                        "id": "sess1",
                        "status": "completed",
                        "progress": 1.0,
                        "finding_count": 15,
                        "created_at": now,
                    },
                ],
            }
        )

        result = docs_api.list_audits()

        client._get.assert_called_once()
        call_args = client._get.call_args
        assert call_args[0][0] == "/api/audit/sessions"
        assert len(result) == 1
        assert isinstance(result[0], AuditSession)


class TestGetAudit:
    """Tests for get_audit() method."""

    def test_get_audit(self, docs_api, client):
        """Test getting audit session details."""
        now = datetime.now()
        client._get = MagicMock(
            return_value={
                "id": "sess123",
                "status": "running",
                "progress": 0.75,
                "finding_count": 10,
                "created_at": now,
            }
        )

        result = docs_api.get_audit("sess123")

        client._get.assert_called_once_with("/api/audit/sessions/sess123")
        assert isinstance(result, AuditSession)
        assert result.progress == 0.75


class TestAuditLifecycle:
    """Tests for audit lifecycle methods (start, pause, resume, cancel)."""

    def test_start_audit(self, docs_api, client):
        """Test starting an audit session."""
        now = datetime.now()
        client._post = MagicMock(
            return_value={
                "id": "sess123",
                "status": "running",
                "created_at": now,
            }
        )

        result = docs_api.start_audit("sess123")

        client._post.assert_called_once_with("/api/audit/sessions/sess123/start", {})
        assert isinstance(result, AuditSession)

    def test_pause_audit(self, docs_api, client):
        """Test pausing an audit session."""
        now = datetime.now()
        client._post = MagicMock(
            return_value={
                "id": "sess123",
                "status": "paused",
                "created_at": now,
            }
        )

        result = docs_api.pause_audit("sess123")

        client._post.assert_called_once_with("/api/audit/sessions/sess123/pause", {})
        assert isinstance(result, AuditSession)

    def test_resume_audit(self, docs_api, client):
        """Test resuming an audit session."""
        now = datetime.now()
        client._post = MagicMock(
            return_value={
                "id": "sess123",
                "status": "running",
                "created_at": now,
            }
        )

        result = docs_api.resume_audit("sess123")

        client._post.assert_called_once_with("/api/audit/sessions/sess123/resume", {})
        assert isinstance(result, AuditSession)

    def test_cancel_audit(self, docs_api, client):
        """Test canceling an audit session."""
        now = datetime.now()
        client._post = MagicMock(
            return_value={
                "id": "sess123",
                "status": "cancelled",
                "created_at": now,
            }
        )

        result = docs_api.cancel_audit("sess123")

        client._post.assert_called_once_with("/api/audit/sessions/sess123/cancel", {})
        assert isinstance(result, AuditSession)


class TestAuditFindings:
    """Tests for audit_findings() method."""

    def test_get_findings(self, docs_api, client):
        """Test getting audit findings."""
        client._get = MagicMock(
            return_value={
                "findings": [
                    {
                        "id": "f1",
                        "session_id": "sess123",
                        "audit_type": "security",
                        "category": "credentials",
                        "severity": "high",
                        "title": "Exposed Key",
                        "description": "API key found",
                    },
                ],
            }
        )

        result = docs_api.audit_findings("sess123")

        client._get.assert_called_once()
        call_args = client._get.call_args
        assert call_args[0][0] == "/api/audit/sessions/sess123/findings"
        assert len(result) == 1
        assert isinstance(result[0], AuditFinding)

    def test_get_findings_with_filters(self, docs_api, client):
        """Test getting findings with filters."""
        client._get = MagicMock(return_value={"findings": []})

        docs_api.audit_findings("sess123", severity="critical", audit_type="security")

        call_args = client._get.call_args
        assert call_args[1]["params"]["severity"] == "critical"
        assert call_args[1]["params"]["audit_type"] == "security"


class TestAuditReport:
    """Tests for audit_report() method."""

    def test_get_report(self, docs_api, client):
        """Test generating an audit report."""
        now = datetime.now()
        client._get = MagicMock(
            return_value={
                "session_id": "sess123",
                "format": "json",
                "content": '{"findings": []}',
                "generated_at": now,
            }
        )

        result = docs_api.audit_report("sess123")

        client._get.assert_called_once()
        call_args = client._get.call_args
        assert call_args[0][0] == "/api/audit/sessions/sess123/report"
        assert isinstance(result, AuditReport)

    def test_get_report_different_format(self, docs_api, client):
        """Test generating report in different format."""
        now = datetime.now()
        client._get = MagicMock(
            return_value={
                "session_id": "sess123",
                "format": "markdown",
                "content": "# Audit Report",
                "generated_at": now,
            }
        )

        docs_api.audit_report("sess123", format="markdown")

        call_args = client._get.call_args
        assert call_args[1]["params"]["format"] == "markdown"


class TestIntervene:
    """Tests for intervene() method."""

    def test_intervene(self, docs_api, client):
        """Test submitting intervention."""
        now = datetime.now()
        client._post = MagicMock(
            return_value={
                "id": "sess123",
                "status": "running",
                "created_at": now,
            }
        )

        result = docs_api.intervene("sess123", "approve", "Looks good")

        client._post.assert_called_once()
        call_args = client._post.call_args
        assert call_args[0][0] == "/api/audit/sessions/sess123/intervene"
        assert call_args[0][1]["action"] == "approve"
        assert call_args[0][1]["message"] == "Looks good"


# ============================================================================
# MIME Type Tests
# ============================================================================


class TestMimeTypeGuessing:
    """Tests for _guess_mime_type() helper."""

    def test_pdf_mime_type(self, docs_api):
        """Test PDF MIME type."""
        assert docs_api._guess_mime_type("document.pdf") == "application/pdf"

    def test_docx_mime_type(self, docs_api):
        """Test DOCX MIME type."""
        result = docs_api._guess_mime_type("document.docx")
        assert "wordprocessingml" in result

    def test_txt_mime_type(self, docs_api):
        """Test TXT MIME type."""
        assert docs_api._guess_mime_type("file.txt") == "text/plain"

    def test_python_mime_type(self, docs_api):
        """Test Python file MIME type."""
        assert docs_api._guess_mime_type("script.py") == "text/x-python"

    def test_json_mime_type(self, docs_api):
        """Test JSON MIME type."""
        assert docs_api._guess_mime_type("data.json") == "application/json"

    def test_unknown_mime_type(self, docs_api):
        """Test unknown extension falls back to octet-stream."""
        assert docs_api._guess_mime_type("file.xyz") == "application/octet-stream"

    def test_case_insensitive(self, docs_api):
        """Test MIME type guessing is case-insensitive."""
        assert docs_api._guess_mime_type("file.PDF") == "application/pdf"
        assert docs_api._guess_mime_type("file.TXT") == "text/plain"


# ============================================================================
# Client Integration Tests
# ============================================================================


class TestDocumentsAPIIntegration:
    """Tests for DocumentsAPI integration with AragoraClient."""

    def test_documents_accessible_from_client(self):
        """Test documents API is accessible from client."""
        client = AragoraClient(base_url="http://test.example.com")
        assert hasattr(client, "documents")
        assert isinstance(client.documents, DocumentsAPI)

    def test_documents_shares_client(self):
        """Test documents API shares the same client."""
        client = AragoraClient(base_url="http://test.example.com")
        assert client.documents._client is client
