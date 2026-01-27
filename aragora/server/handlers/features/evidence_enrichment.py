"""
Evidence Enrichment endpoint handlers.

Endpoints:
- POST /api/findings/{finding_id}/evidence - Enrich a finding with evidence
- POST /api/findings/batch-evidence - Batch evidence enrichment
- GET  /api/findings/{finding_id}/evidence - Get evidence for a finding
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

from aragora.server.http_utils import run_async as _run_async

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    require_user_auth,
    safe_error_message,
)
from aragora.rbac.decorators import require_permission

if TYPE_CHECKING:
    from aragora.audit.document_auditor import AuditFinding
    from aragora.storage.documents import DocumentStore

logger = logging.getLogger(__name__)


def _get_evidence_enrichment(finding: "AuditFinding") -> Any:
    """Get evidence enrichment from finding using getattr to avoid type errors."""
    return getattr(finding, "_evidence_enrichment", None)


def _set_evidence_enrichment(finding: "AuditFinding", enrichment: Any) -> None:
    """Set evidence enrichment on finding using setattr to avoid type errors."""
    setattr(finding, "_evidence_enrichment", enrichment)


class EvidenceEnrichmentHandler(BaseHandler):
    """Handler for evidence enrichment endpoints."""

    ROUTES = [
        "/api/v1/findings/batch-evidence",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        # Handle /api/findings/{finding_id}/evidence pattern
        if path.startswith("/api/v1/findings/") and path.endswith("/evidence"):
            return True
        return False

    @require_permission("evidence:read")
    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route GET requests to appropriate methods."""
        if path.startswith("/api/v1/findings/") and path.endswith("/evidence"):
            # Extract finding_id from /api/findings/{finding_id}/evidence
            parts = path.split("/")
            if len(parts) == 5:
                finding_id = parts[3]
                return self._get_finding_evidence(finding_id)
        return None

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route POST requests to appropriate methods."""
        if path == "/api/v1/findings/batch-evidence":
            return self._batch_enrich(handler)

        if path.startswith("/api/v1/findings/") and path.endswith("/evidence"):
            # Extract finding_id from /api/findings/{finding_id}/evidence
            parts = path.split("/")
            if len(parts) == 5:
                finding_id = parts[3]
                return self._enrich_finding(handler, finding_id)

        return None

    def _get_document_store(self) -> Optional["DocumentStore"]:
        """Get document store from handler context."""
        return self.ctx.get("document_store")

    @require_user_auth
    @handle_errors("get finding evidence")
    def _get_finding_evidence(self, finding_id: str, user=None) -> HandlerResult:
        """
        Get evidence associated with a finding.

        Response:
        {
            "finding_id": "...",
            "evidence": {
                "sources": [...],
                "original_confidence": 0.7,
                "adjusted_confidence": 0.85,
                "evidence_summary": "..."
            }
        }
        """
        # Get finding from audit system
        try:
            from aragora.audit import get_document_auditor

            auditor = get_document_auditor()

            # Try to find the finding across sessions
            finding = None
            for session in auditor._sessions.values():
                for f in session.findings:
                    if f.id == finding_id:
                        finding = f
                        break
                if finding:
                    break

            if not finding:
                return error_response(f"Finding not found: {finding_id}", 404)

            # Check if evidence is already attached
            evidence_enrichment = _get_evidence_enrichment(finding)
            if evidence_enrichment is not None:
                return json_response(
                    {
                        "finding_id": finding_id,
                        "evidence": evidence_enrichment.to_dict(),
                    }
                )

            return json_response(
                {
                    "finding_id": finding_id,
                    "evidence": None,
                    "message": "No evidence collected yet. POST to enrich.",
                }
            )

        except Exception as e:
            logger.error(f"Failed to get evidence for {finding_id}: {e}")
            return error_response(safe_error_message(e, "Failed to get evidence"), 500)

    @require_user_auth
    @handle_errors("enrich finding")
    def _enrich_finding(self, handler, finding_id: str, user=None) -> HandlerResult:
        """
        Enrich a finding with evidence.

        Request body (optional):
        {
            "document_content": "...",  // Optional: provide doc content
            "related_documents": {"doc2": "..."},  // Optional: related doc content
            "config": {
                "max_sources_per_finding": 5,
                "enable_external_sources": true
            }
        }

        Response:
        {
            "finding_id": "...",
            "enrichment": {
                "sources": [...],
                "original_confidence": 0.7,
                "adjusted_confidence": 0.85,
                "evidence_summary": "...",
                "collection_time_ms": 234
            }
        }
        """
        body = self.read_json_body(handler) or {}

        document_content = body.get("document_content")
        related_documents = body.get("related_documents", {})
        config_dict = body.get("config", {})

        try:
            # Get document store from handler context
            document_store = self._get_document_store()
            result = _run_async(
                self._run_enrichment(
                    finding_id=finding_id,
                    document_content=document_content,
                    related_documents=related_documents,
                    config_dict=config_dict,
                    document_store=document_store,
                )
            )
            return json_response(result)
        except ValueError as e:
            return error_response(str(e), 404)
        except Exception as e:
            logger.error(f"Failed to enrich finding {finding_id}: {e}")
            return error_response(safe_error_message(e, "Enrichment"), 500)

    async def _run_enrichment(
        self,
        finding_id: str,
        document_content: Optional[str],
        related_documents: dict[str, str],
        config_dict: dict[str, Any],
        document_store: Optional["DocumentStore"],
    ) -> dict[str, Any]:
        """Run evidence enrichment for a finding."""
        from aragora.audit import get_document_auditor
        from aragora.audit.evidence_adapter import FindingEvidenceCollector, EvidenceConfig

        auditor = get_document_auditor()

        # Find the finding
        finding = None
        doc_content = document_content
        for session in auditor._sessions.values():
            for f in session.findings:
                if f.id == finding_id:
                    finding = f
                    # Try to get document content from store if not provided
                    if not doc_content and document_store is not None:
                        doc = document_store.get(f.document_id)
                        if doc:
                            doc_content = doc.content
                    break
            if finding:
                break

        if not finding:
            raise ValueError(f"Finding not found: {finding_id}")

        # Build config
        config = EvidenceConfig(
            max_sources_per_finding=config_dict.get("max_sources_per_finding", 5),
            enable_external_sources=config_dict.get("enable_external_sources", True),
            enable_cross_reference=config_dict.get("enable_cross_reference", True),
        )

        # Run enrichment
        collector = FindingEvidenceCollector(config=config)
        enrichment = await collector.enrich_finding(
            finding=finding,
            document_content=doc_content,
            related_documents=related_documents,
        )

        # Store enrichment on finding
        _set_evidence_enrichment(finding, enrichment)

        return {
            "finding_id": finding_id,
            "enrichment": enrichment.to_dict(),
        }

    @require_user_auth
    @handle_errors("batch enrich")
    def _batch_enrich(self, handler, user=None) -> HandlerResult:
        """
        Batch enrich multiple findings with evidence.

        Request body:
        {
            "finding_ids": ["f1", "f2", "f3"],
            "config": {...}
        }

        Response:
        {
            "enrichments": {
                "f1": {...},
                "f2": {...}
            },
            "errors": {
                "f3": "Finding not found"
            }
        }
        """
        body = self.read_json_body(handler)
        if not body:
            return error_response("Request body required", 400)

        finding_ids = body.get("finding_ids", [])
        if not finding_ids:
            return error_response("'finding_ids' list required", 400)

        config_dict = body.get("config", {})

        try:
            # Get document store from handler context
            document_store = self._get_document_store()
            result = _run_async(
                self._run_batch_enrichment(
                    finding_ids=finding_ids,
                    config_dict=config_dict,
                    document_store=document_store,
                )
            )
            return json_response(result)
        except Exception as e:
            logger.error(f"Batch enrichment failed: {e}")
            return error_response(safe_error_message(e, "Batch enrichment"), 500)

    async def _run_batch_enrichment(
        self,
        finding_ids: list[str],
        config_dict: dict[str, Any],
        document_store: Optional["DocumentStore"],
    ) -> dict[str, Any]:
        """Run batch evidence enrichment."""
        from aragora.audit import get_document_auditor
        from aragora.audit.evidence_adapter import FindingEvidenceCollector, EvidenceConfig

        auditor = get_document_auditor()

        # Find all findings and their documents
        findings = []
        documents: dict[str, str] = {}

        for session in auditor._sessions.values():
            for f in session.findings:
                if f.id in finding_ids:
                    findings.append(f)
                    # Get document content
                    if document_store is not None and f.document_id not in documents:
                        doc = document_store.get(f.document_id)
                        if doc:
                            documents[f.document_id] = doc.content

        # Track errors for missing findings
        found_ids = {f.id for f in findings}
        errors = {fid: "Finding not found" for fid in finding_ids if fid not in found_ids}

        # Build config
        config = EvidenceConfig(
            max_sources_per_finding=config_dict.get("max_sources_per_finding", 5),
            enable_external_sources=config_dict.get("enable_external_sources", True),
            enable_cross_reference=config_dict.get("enable_cross_reference", True),
        )

        # Run batch enrichment
        collector = FindingEvidenceCollector(config=config)
        enrichments = await collector.enrich_findings_batch(
            findings=findings,
            documents=documents,
            max_concurrent=config_dict.get("max_concurrent", 5),
        )

        # Store enrichments on findings
        for finding in findings:
            if finding.id in enrichments:
                _set_evidence_enrichment(finding, enrichments[finding.id])

        return {
            "enrichments": {fid: enrichment.to_dict() for fid, enrichment in enrichments.items()},
            "errors": errors,
            "processed": len(enrichments),
            "failed": len(errors),
        }
