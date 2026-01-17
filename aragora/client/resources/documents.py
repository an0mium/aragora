"""DocumentsAPI resource for the Aragora client.

Provides SDK methods for document management, batch processing, and auditing.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

from ..models import (
    AuditFinding,
    AuditReport,
    AuditSession,
    AuditSessionCreateRequest,
    AuditSessionCreateResponse,
    BatchJobResults,
    BatchJobStatus,
    BatchUploadResponse,
    Document,
    DocumentChunk,
    DocumentContext,
    DocumentUploadResponse,
    ProcessingStats,
    SupportedFormats,
)

if TYPE_CHECKING:
    from ..client import AragoraClient


class DocumentsAPI:
    """API interface for document management and auditing."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    # =========================================================================
    # Document Management
    # =========================================================================

    def list(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
    ) -> List[Document]:
        """
        List uploaded documents.

        Args:
            limit: Maximum number of documents to return.
            offset: Number of documents to skip.
            status: Filter by status (pending, processing, completed, failed).

        Returns:
            List of Document objects.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        response = self._client._get("/api/documents", params=params)
        documents = response.get("documents", response) if isinstance(response, dict) else response
        return [Document(**d) for d in documents]

    async def list_async(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
    ) -> List[Document]:
        """Async version of list()."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        response = await self._client._get_async("/api/documents", params=params)
        documents = response.get("documents", response) if isinstance(response, dict) else response
        return [Document(**d) for d in documents]

    def get(self, document_id: str) -> Document:
        """
        Get a document by ID.

        Args:
            document_id: The document ID.

        Returns:
            Document with full details.
        """
        response = self._client._get(f"/api/documents/{document_id}")
        return Document(**response)

    async def get_async(self, document_id: str) -> Document:
        """Async version of get()."""
        response = await self._client._get_async(f"/api/documents/{document_id}")
        return Document(**response)

    def upload(
        self,
        file_path: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> DocumentUploadResponse:
        """
        Upload a document for processing.

        Args:
            file_path: Path to the file to upload.
            metadata: Optional metadata to attach to the document.

        Returns:
            DocumentUploadResponse with document_id and status.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, "rb") as f:
            content = base64.b64encode(f.read()).decode("utf-8")

        data = {
            "filename": path.name,
            "content": content,
            "content_type": self._guess_mime_type(path.name),
        }
        if metadata:
            data["metadata"] = metadata

        response = self._client._post("/api/documents/upload", data)
        return DocumentUploadResponse(**response)

    async def upload_async(
        self,
        file_path: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> DocumentUploadResponse:
        """Async version of upload()."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, "rb") as f:
            content = base64.b64encode(f.read()).decode("utf-8")

        data = {
            "filename": path.name,
            "content": content,
            "content_type": self._guess_mime_type(path.name),
        }
        if metadata:
            data["metadata"] = metadata

        response = await self._client._post_async("/api/documents/upload", data)
        return DocumentUploadResponse(**response)

    def delete(self, document_id: str) -> bool:
        """
        Delete a document by ID.

        Args:
            document_id: The document ID to delete.

        Returns:
            True if deletion was successful.
        """
        response = self._client._delete(f"/api/documents/{document_id}")
        return response.get("success", True)

    async def delete_async(self, document_id: str) -> bool:
        """Async version of delete()."""
        response = await self._client._delete_async(f"/api/documents/{document_id}")
        return response.get("success", True)

    def formats(self) -> SupportedFormats:
        """
        Get supported document formats.

        Returns:
            SupportedFormats with list of supported formats and MIME types.
        """
        response = self._client._get("/api/documents/formats")
        return SupportedFormats(**response)

    async def formats_async(self) -> SupportedFormats:
        """Async version of formats()."""
        response = await self._client._get_async("/api/documents/formats")
        return SupportedFormats(**response)

    # =========================================================================
    # Batch Processing
    # =========================================================================

    def batch_upload(
        self,
        file_paths: List[str],
        metadata: Optional[dict[str, Any]] = None,
    ) -> BatchUploadResponse:
        """
        Upload multiple documents as a batch.

        Args:
            file_paths: List of file paths to upload.
            metadata: Optional metadata to attach to all documents.

        Returns:
            BatchUploadResponse with job_id for tracking.
        """
        files = []
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(path, "rb") as f:
                content = base64.b64encode(f.read()).decode("utf-8")

            files.append(
                {
                    "filename": path.name,
                    "content": content,
                    "content_type": self._guess_mime_type(path.name),
                }
            )

        data: dict[str, Any] = {"files": files}
        if metadata:
            data["metadata"] = metadata

        response = self._client._post("/api/documents/batch", data)
        return BatchUploadResponse(**response)

    async def batch_upload_async(
        self,
        file_paths: List[str],
        metadata: Optional[dict[str, Any]] = None,
    ) -> BatchUploadResponse:
        """Async version of batch_upload()."""
        files = []
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(path, "rb") as f:
                content = base64.b64encode(f.read()).decode("utf-8")

            files.append(
                {
                    "filename": path.name,
                    "content": content,
                    "content_type": self._guess_mime_type(path.name),
                }
            )

        data: dict[str, Any] = {"files": files}
        if metadata:
            data["metadata"] = metadata

        response = await self._client._post_async("/api/documents/batch", data)
        return BatchUploadResponse(**response)

    def batch_status(self, job_id: str) -> BatchJobStatus:
        """
        Get the status of a batch processing job.

        Args:
            job_id: The batch job ID.

        Returns:
            BatchJobStatus with progress and counts.
        """
        response = self._client._get(f"/api/documents/batch/{job_id}")
        return BatchJobStatus(**response)

    async def batch_status_async(self, job_id: str) -> BatchJobStatus:
        """Async version of batch_status()."""
        response = await self._client._get_async(f"/api/documents/batch/{job_id}")
        return BatchJobStatus(**response)

    def batch_results(self, job_id: str) -> BatchJobResults:
        """
        Get the results of a completed batch job.

        Args:
            job_id: The batch job ID.

        Returns:
            BatchJobResults with processed documents.
        """
        response = self._client._get(f"/api/documents/batch/{job_id}/results")
        return BatchJobResults(**response)

    async def batch_results_async(self, job_id: str) -> BatchJobResults:
        """Async version of batch_results()."""
        response = await self._client._get_async(f"/api/documents/batch/{job_id}/results")
        return BatchJobResults(**response)

    def batch_cancel(self, job_id: str) -> bool:
        """
        Cancel a batch processing job.

        Args:
            job_id: The batch job ID to cancel.

        Returns:
            True if cancellation was successful.
        """
        response = self._client._delete(f"/api/documents/batch/{job_id}")
        return response.get("success", True)

    async def batch_cancel_async(self, job_id: str) -> bool:
        """Async version of batch_cancel()."""
        response = await self._client._delete_async(f"/api/documents/batch/{job_id}")
        return response.get("success", True)

    def processing_stats(self) -> ProcessingStats:
        """
        Get document processing statistics.

        Returns:
            ProcessingStats with counts and totals.
        """
        response = self._client._get("/api/documents/processing/stats")
        return ProcessingStats(**response)

    async def processing_stats_async(self) -> ProcessingStats:
        """Async version of processing_stats()."""
        response = await self._client._get_async("/api/documents/processing/stats")
        return ProcessingStats(**response)

    # =========================================================================
    # Document Content
    # =========================================================================

    def chunks(
        self,
        document_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[DocumentChunk]:
        """
        Get chunks for a document.

        Args:
            document_id: The document ID.
            limit: Maximum number of chunks to return.
            offset: Number of chunks to skip.

        Returns:
            List of DocumentChunk objects.
        """
        params = {"limit": limit, "offset": offset}
        response = self._client._get(f"/api/documents/{document_id}/chunks", params=params)
        chunks = response.get("chunks", response) if isinstance(response, dict) else response
        return [DocumentChunk(**c) for c in chunks]

    async def chunks_async(
        self,
        document_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[DocumentChunk]:
        """Async version of chunks()."""
        params = {"limit": limit, "offset": offset}
        response = await self._client._get_async(
            f"/api/documents/{document_id}/chunks", params=params
        )
        chunks = response.get("chunks", response) if isinstance(response, dict) else response
        return [DocumentChunk(**c) for c in chunks]

    def context(
        self,
        document_id: str,
        max_tokens: int = 100000,
        model: str = "gemini-1.5-flash",
    ) -> DocumentContext:
        """
        Get LLM-ready context from a document.

        Combines document chunks into a format suitable for LLM context,
        respecting token limits for the target model.

        Args:
            document_id: The document ID.
            max_tokens: Maximum tokens to include (default: 100K).
            model: Target model for token counting.

        Returns:
            DocumentContext with combined content.
        """
        params = {"max_tokens": max_tokens, "model": model}
        response = self._client._get(f"/api/documents/{document_id}/context", params=params)
        return DocumentContext(**response)

    async def context_async(
        self,
        document_id: str,
        max_tokens: int = 100000,
        model: str = "gemini-1.5-flash",
    ) -> DocumentContext:
        """Async version of context()."""
        params = {"max_tokens": max_tokens, "model": model}
        response = await self._client._get_async(
            f"/api/documents/{document_id}/context", params=params
        )
        return DocumentContext(**response)

    # =========================================================================
    # Audit Sessions
    # =========================================================================

    def create_audit(
        self,
        document_ids: List[str],
        audit_types: Optional[List[str]] = None,
        model: str = "gemini-1.5-flash",
        **options: Any,
    ) -> AuditSessionCreateResponse:
        """
        Create a new audit session for documents.

        Args:
            document_ids: List of document IDs to audit.
            audit_types: Types of audits to run (security, compliance, consistency, quality).
                        Defaults to all types if not specified.
            model: Model to use for analysis (default: gemini-1.5-flash).
            **options: Additional audit options.

        Returns:
            AuditSessionCreateResponse with session_id.
        """
        request = AuditSessionCreateRequest(
            document_ids=document_ids,
            audit_types=audit_types or ["security", "compliance", "consistency", "quality"],
            model=model,
            options=options,
        )

        response = self._client._post("/api/audit/sessions", request.model_dump())
        return AuditSessionCreateResponse(**response)

    async def create_audit_async(
        self,
        document_ids: List[str],
        audit_types: Optional[List[str]] = None,
        model: str = "gemini-1.5-flash",
        **options: Any,
    ) -> AuditSessionCreateResponse:
        """Async version of create_audit()."""
        request = AuditSessionCreateRequest(
            document_ids=document_ids,
            audit_types=audit_types or ["security", "compliance", "consistency", "quality"],
            model=model,
            options=options,
        )

        response = await self._client._post_async("/api/audit/sessions", request.model_dump())
        return AuditSessionCreateResponse(**response)

    def list_audits(
        self,
        limit: int = 20,
        offset: int = 0,
        status: Optional[str] = None,
    ) -> List[AuditSession]:
        """
        List audit sessions.

        Args:
            limit: Maximum number of sessions to return.
            offset: Number of sessions to skip.
            status: Filter by status.

        Returns:
            List of AuditSession objects.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        response = self._client._get("/api/audit/sessions", params=params)
        sessions = response.get("sessions", response) if isinstance(response, dict) else response
        return [AuditSession(**s) for s in sessions]

    async def list_audits_async(
        self,
        limit: int = 20,
        offset: int = 0,
        status: Optional[str] = None,
    ) -> List[AuditSession]:
        """Async version of list_audits()."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        response = await self._client._get_async("/api/audit/sessions", params=params)
        sessions = response.get("sessions", response) if isinstance(response, dict) else response
        return [AuditSession(**s) for s in sessions]

    def get_audit(self, session_id: str) -> AuditSession:
        """
        Get audit session details.

        Args:
            session_id: The audit session ID.

        Returns:
            AuditSession with full details.
        """
        response = self._client._get(f"/api/audit/sessions/{session_id}")
        return AuditSession(**response)

    async def get_audit_async(self, session_id: str) -> AuditSession:
        """Async version of get_audit()."""
        response = await self._client._get_async(f"/api/audit/sessions/{session_id}")
        return AuditSession(**response)

    def start_audit(self, session_id: str) -> AuditSession:
        """
        Start an audit session.

        Args:
            session_id: The audit session ID to start.

        Returns:
            Updated AuditSession.
        """
        response = self._client._post(f"/api/audit/sessions/{session_id}/start", {})
        return AuditSession(**response)

    async def start_audit_async(self, session_id: str) -> AuditSession:
        """Async version of start_audit()."""
        response = await self._client._post_async(f"/api/audit/sessions/{session_id}/start", {})
        return AuditSession(**response)

    def pause_audit(self, session_id: str) -> AuditSession:
        """
        Pause an audit session.

        Args:
            session_id: The audit session ID to pause.

        Returns:
            Updated AuditSession.
        """
        response = self._client._post(f"/api/audit/sessions/{session_id}/pause", {})
        return AuditSession(**response)

    async def pause_audit_async(self, session_id: str) -> AuditSession:
        """Async version of pause_audit()."""
        response = await self._client._post_async(f"/api/audit/sessions/{session_id}/pause", {})
        return AuditSession(**response)

    def resume_audit(self, session_id: str) -> AuditSession:
        """
        Resume a paused audit session.

        Args:
            session_id: The audit session ID to resume.

        Returns:
            Updated AuditSession.
        """
        response = self._client._post(f"/api/audit/sessions/{session_id}/resume", {})
        return AuditSession(**response)

    async def resume_audit_async(self, session_id: str) -> AuditSession:
        """Async version of resume_audit()."""
        response = await self._client._post_async(f"/api/audit/sessions/{session_id}/resume", {})
        return AuditSession(**response)

    def cancel_audit(self, session_id: str) -> AuditSession:
        """
        Cancel an audit session.

        Args:
            session_id: The audit session ID to cancel.

        Returns:
            Updated AuditSession.
        """
        response = self._client._post(f"/api/audit/sessions/{session_id}/cancel", {})
        return AuditSession(**response)

    async def cancel_audit_async(self, session_id: str) -> AuditSession:
        """Async version of cancel_audit()."""
        response = await self._client._post_async(f"/api/audit/sessions/{session_id}/cancel", {})
        return AuditSession(**response)

    def audit_findings(
        self,
        session_id: str,
        severity: Optional[str] = None,
        audit_type: Optional[str] = None,
    ) -> List[AuditFinding]:
        """
        Get findings from an audit session.

        Args:
            session_id: The audit session ID.
            severity: Filter by severity (critical, high, medium, low, info).
            audit_type: Filter by audit type (security, compliance, consistency, quality).

        Returns:
            List of AuditFinding objects.
        """
        params: dict[str, Any] = {}
        if severity:
            params["severity"] = severity
        if audit_type:
            params["audit_type"] = audit_type

        response = self._client._get(f"/api/audit/sessions/{session_id}/findings", params=params)
        findings = response.get("findings", response) if isinstance(response, dict) else response
        return [AuditFinding(**f) for f in findings]

    async def audit_findings_async(
        self,
        session_id: str,
        severity: Optional[str] = None,
        audit_type: Optional[str] = None,
    ) -> List[AuditFinding]:
        """Async version of audit_findings()."""
        params: dict[str, Any] = {}
        if severity:
            params["severity"] = severity
        if audit_type:
            params["audit_type"] = audit_type

        response = await self._client._get_async(
            f"/api/audit/sessions/{session_id}/findings", params=params
        )
        findings = response.get("findings", response) if isinstance(response, dict) else response
        return [AuditFinding(**f) for f in findings]

    def audit_report(
        self,
        session_id: str,
        format: str = "json",
    ) -> AuditReport:
        """
        Generate an audit report.

        Args:
            session_id: The audit session ID.
            format: Report format (json, markdown, html, pdf).

        Returns:
            AuditReport with content.
        """
        params = {"format": format}
        response = self._client._get(f"/api/audit/sessions/{session_id}/report", params=params)
        return AuditReport(**response)

    async def audit_report_async(
        self,
        session_id: str,
        format: str = "json",
    ) -> AuditReport:
        """Async version of audit_report()."""
        params = {"format": format}
        response = await self._client._get_async(
            f"/api/audit/sessions/{session_id}/report", params=params
        )
        return AuditReport(**response)

    def intervene(
        self,
        session_id: str,
        action: str,
        message: str = "",
        **data: Any,
    ) -> AuditSession:
        """
        Submit a human intervention to an audit session.

        Args:
            session_id: The audit session ID.
            action: Intervention action (approve, reject, modify, skip).
            message: Optional message explaining the intervention.
            **data: Additional intervention data.

        Returns:
            Updated AuditSession.
        """
        payload = {"action": action, "message": message, **data}
        response = self._client._post(f"/api/audit/sessions/{session_id}/intervene", payload)
        return AuditSession(**response)

    async def intervene_async(
        self,
        session_id: str,
        action: str,
        message: str = "",
        **data: Any,
    ) -> AuditSession:
        """Async version of intervene()."""
        payload = {"action": action, "message": message, **data}
        response = await self._client._post_async(
            f"/api/audit/sessions/{session_id}/intervene", payload
        )
        return AuditSession(**response)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _guess_mime_type(self, filename: str) -> str:
        """Guess MIME type from filename extension."""
        ext = os.path.splitext(filename)[1].lower()
        mime_map = {
            ".pdf": "application/pdf",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xls": "application/vnd.ms-excel",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".ppt": "application/vnd.ms-powerpoint",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".json": "application/json",
            ".xml": "application/xml",
            ".csv": "text/csv",
            ".html": "text/html",
            ".htm": "text/html",
            ".py": "text/x-python",
            ".js": "text/javascript",
            ".ts": "text/typescript",
            ".java": "text/x-java",
            ".c": "text/x-c",
            ".cpp": "text/x-c++",
            ".h": "text/x-c",
            ".hpp": "text/x-c++",
            ".go": "text/x-go",
            ".rs": "text/x-rust",
            ".rb": "text/x-ruby",
            ".php": "text/x-php",
            ".yaml": "text/yaml",
            ".yml": "text/yaml",
            ".toml": "text/toml",
            ".ini": "text/plain",
            ".cfg": "text/plain",
            ".conf": "text/plain",
            ".log": "text/plain",
            ".sh": "text/x-shellscript",
            ".bash": "text/x-shellscript",
            ".zsh": "text/x-shellscript",
            ".sql": "text/x-sql",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
            ".webp": "image/webp",
            ".zip": "application/zip",
            ".tar": "application/x-tar",
            ".gz": "application/gzip",
        }
        return mime_map.get(ext, "application/octet-stream")
