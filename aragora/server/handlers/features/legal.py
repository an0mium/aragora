"""
Legal E-Signature API Handler.

Provides REST APIs for e-signature workflows via DocuSign:
- Create and send signature envelopes
- Track signing status
- Download signed documents
- Manage templates
- Webhook handling for status updates

Endpoints:
- POST /api/v1/legal/envelopes              - Create new envelope
- GET  /api/v1/legal/envelopes              - List envelopes
- GET  /api/v1/legal/envelopes/{id}         - Get envelope details
- POST /api/v1/legal/envelopes/{id}/void    - Void envelope
- POST /api/v1/legal/envelopes/{id}/resend  - Resend notifications
- GET  /api/v1/legal/envelopes/{id}/documents/{doc_id} - Download document
- GET  /api/v1/legal/envelopes/{id}/certificate - Download certificate
- POST /api/v1/legal/webhooks/docusign      - DocuSign webhook handler
- GET  /api/v1/legal/status                 - Connection status
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, Optional

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Connector Instance Management
# =============================================================================

_connector_instances: Dict[str, Any] = {}  # tenant_id -> DocuSignConnector


async def get_docusign_connector(tenant_id: str):
    """Get or create DocuSign connector for tenant."""
    if tenant_id not in _connector_instances:
        try:
            from aragora.connectors.legal.docusign import DocuSignConnector

            connector = DocuSignConnector()
            if connector.is_configured:
                _connector_instances[tenant_id] = connector
            else:
                return None
        except ImportError:
            return None
    return _connector_instances.get(tenant_id)


# =============================================================================
# Handler Class
# =============================================================================


class LegalHandler(BaseHandler):
    """Handler for legal e-signature API endpoints."""

    ROUTES = [
        "/api/v1/legal/envelopes",
        "/api/v1/legal/envelopes/{envelope_id}",
        "/api/v1/legal/envelopes/{envelope_id}/void",
        "/api/v1/legal/envelopes/{envelope_id}/resend",
        "/api/v1/legal/envelopes/{envelope_id}/documents/{document_id}",
        "/api/v1/legal/envelopes/{envelope_id}/certificate",
        "/api/v1/legal/webhooks/docusign",
        "/api/v1/legal/status",
        "/api/v1/legal/templates",
    ]

    def __init__(self, server_context: Optional[Dict[str, Any]] = None):
        """Initialize handler with optional server context."""
        super().__init__(server_context or {})

    async def handle(self, request: Any, path: str, method: str) -> HandlerResult:
        """Route requests to appropriate handler methods."""
        try:
            tenant_id = self._get_tenant_id(request)

            # Status check
            if path == "/api/v1/legal/status" and method == "GET":
                return await self._handle_status(request, tenant_id)

            # List/create envelopes
            if path == "/api/v1/legal/envelopes":
                if method == "GET":
                    return await self._handle_list_envelopes(request, tenant_id)
                elif method == "POST":
                    return await self._handle_create_envelope(request, tenant_id)

            # Templates
            if path == "/api/v1/legal/templates" and method == "GET":
                return await self._handle_list_templates(request, tenant_id)

            # Webhook
            if path == "/api/v1/legal/webhooks/docusign" and method == "POST":
                return await self._handle_docusign_webhook(request, tenant_id)

            # Envelope-specific paths
            if path.startswith("/api/v1/legal/envelopes/"):
                parts = path.split("/")
                if len(parts) >= 5:
                    envelope_id = parts[4]

                    # GET /envelopes/{id}
                    if len(parts) == 5 and method == "GET":
                        return await self._handle_get_envelope(request, tenant_id, envelope_id)

                    # Actions on envelope
                    if len(parts) == 6:
                        action = parts[5]
                        if action == "void" and method == "POST":
                            return await self._handle_void_envelope(request, tenant_id, envelope_id)
                        elif action == "resend" and method == "POST":
                            return await self._handle_resend_envelope(
                                request, tenant_id, envelope_id
                            )
                        elif action == "certificate" and method == "GET":
                            return await self._handle_download_certificate(
                                request, tenant_id, envelope_id
                            )

                    # Download document
                    if len(parts) == 7 and parts[5] == "documents" and method == "GET":
                        document_id = parts[6]
                        return await self._handle_download_document(
                            request, tenant_id, envelope_id, document_id
                        )

            return error_response("Not found", 404)

        except Exception as e:
            logger.exception(f"Error in legal handler: {e}")
            return error_response(f"Internal error: {str(e)}", 500)

    def _get_tenant_id(self, request: Any) -> str:
        """Extract tenant ID from request context."""
        return getattr(request, "tenant_id", "default")

    # =========================================================================
    # Status
    # =========================================================================

    async def _handle_status(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get DocuSign connection status."""
        try:
            from aragora.connectors.legal.docusign import DocuSignConnector

            connector = DocuSignConnector()

            return success_response(
                {
                    "configured": connector.is_configured,
                    "authenticated": connector.is_authenticated,
                    "environment": connector.environment.value,
                    "integration_key_set": bool(connector.integration_key),
                    "account_id_set": bool(connector.account_id),
                }
            )
        except ImportError:
            return success_response(
                {
                    "configured": False,
                    "error": "DocuSign connector not installed",
                }
            )

    # =========================================================================
    # Envelopes
    # =========================================================================

    async def _handle_list_envelopes(self, request: Any, tenant_id: str) -> HandlerResult:
        """List envelopes with filtering.

        Query params:
        - status: Filter by status (sent, completed, voided)
        - from_date: Start date (ISO format)
        - to_date: End date (ISO format)
        - limit: Max results (default 25)
        - offset: Pagination offset
        """
        connector = await get_docusign_connector(tenant_id)
        if not connector:
            return error_response("DocuSign not configured", 503)

        if not connector.is_authenticated:
            try:
                await connector.authenticate_jwt()
            except Exception as e:
                return error_response(f"Authentication failed: {e}", 401)

        params = self._get_query_params(request)
        status = params.get("status")
        from_date = params.get("from_date")
        to_date = params.get("to_date")
        limit = int(params.get("limit", 25))

        try:
            envelopes = await connector.list_envelopes(
                status=status,
                from_date=from_date,
                to_date=to_date,
                count=limit,
            )

            return success_response(
                {
                    "envelopes": [e.to_dict() for e in envelopes],
                    "count": len(envelopes),
                }
            )
        except Exception as e:
            logger.error(f"Failed to list envelopes: {e}")
            return error_response(f"Failed to list envelopes: {e}", 500)

    async def _handle_create_envelope(self, request: Any, tenant_id: str) -> HandlerResult:
        """Create and send a new envelope.

        Request body:
        {
            "email_subject": "Please sign this document",
            "email_body": "Optional message",
            "recipients": [
                {"email": "signer@example.com", "name": "John Doe", "type": "signer"}
            ],
            "documents": [
                {"name": "contract.pdf", "content_base64": "..."}
            ],
            "status": "sent",  // "sent" to send immediately, "created" for draft
            "tabs": [
                {"type": "signature", "page": 1, "x": 100, "y": 500, "recipient_id": "1"}
            ]
        }
        """
        connector = await get_docusign_connector(tenant_id)
        if not connector:
            return error_response("DocuSign not configured", 503)

        if not connector.is_authenticated:
            try:
                await connector.authenticate_jwt()
            except Exception as e:
                return error_response(f"Authentication failed: {e}", 401)

        body = await self._get_json_body(request)

        # Validate required fields
        if not body.get("email_subject"):
            return error_response("email_subject is required", 400)
        if not body.get("recipients"):
            return error_response("recipients is required", 400)
        if not body.get("documents"):
            return error_response("documents is required", 400)

        try:
            from aragora.connectors.legal.docusign import (
                Document,
                EnvelopeCreateRequest,
                Recipient,
                RecipientType,
                SignatureTab,
            )

            # Build recipients
            recipients = []
            for i, r in enumerate(body["recipients"], 1):
                recipient_type = RecipientType(r.get("type", "signer"))
                recipients.append(
                    Recipient(
                        email=r["email"],
                        name=r["name"],
                        recipient_type=recipient_type,
                        routing_order=r.get("routing_order", i),
                        recipient_id=r.get("recipient_id", str(i)),
                    )
                )

            # Build documents
            documents = []
            for i, d in enumerate(body["documents"], 1):
                content = base64.b64decode(d["content_base64"])
                documents.append(
                    Document(
                        document_id=d.get("document_id", str(i)),
                        name=d["name"],
                        content=content,
                        file_extension=d.get("extension", "pdf"),
                        order=i,
                    )
                )

            # Build tabs
            tabs = None
            if body.get("tabs"):
                tabs = []
                for t in body["tabs"]:
                    tabs.append(
                        SignatureTab(
                            tab_type=t.get("type", "signature"),
                            page_number=t.get("page", 1),
                            x_position=t.get("x", 100),
                            y_position=t.get("y", 100),
                            recipient_id=t.get("recipient_id", "1"),
                        )
                    )

            # Create request
            create_request = EnvelopeCreateRequest(
                email_subject=body["email_subject"],
                email_body=body.get("email_body", ""),
                recipients=recipients,
                documents=documents,
                status=body.get("status", "sent"),
                tabs=tabs,
            )

            envelope = await connector.create_envelope(create_request)

            logger.info(f"[Legal] Created envelope {envelope.envelope_id} for tenant {tenant_id}")

            return success_response(
                {
                    "envelope": envelope.to_dict(),
                    "message": "Envelope created successfully",
                },
                201,
            )

        except Exception as e:
            logger.error(f"Failed to create envelope: {e}")
            return error_response(f"Failed to create envelope: {e}", 500)

    async def _handle_get_envelope(
        self, request: Any, tenant_id: str, envelope_id: str
    ) -> HandlerResult:
        """Get envelope details."""
        connector = await get_docusign_connector(tenant_id)
        if not connector:
            return error_response("DocuSign not configured", 503)

        if not connector.is_authenticated:
            try:
                await connector.authenticate_jwt()
            except Exception as e:
                return error_response(f"Authentication failed: {e}", 401)

        try:
            envelope = await connector.get_envelope(envelope_id)
            if not envelope:
                return error_response("Envelope not found", 404)

            return success_response({"envelope": envelope.to_dict()})

        except Exception as e:
            logger.error(f"Failed to get envelope {envelope_id}: {e}")
            return error_response(f"Failed to get envelope: {e}", 500)

    async def _handle_void_envelope(
        self, request: Any, tenant_id: str, envelope_id: str
    ) -> HandlerResult:
        """Void an envelope.

        Request body:
        {
            "reason": "Contract cancelled by mutual agreement"
        }
        """
        connector = await get_docusign_connector(tenant_id)
        if not connector:
            return error_response("DocuSign not configured", 503)

        if not connector.is_authenticated:
            try:
                await connector.authenticate_jwt()
            except Exception as e:
                return error_response(f"Authentication failed: {e}", 401)

        body = await self._get_json_body(request)
        reason = body.get("reason", "Voided by user")

        try:
            success = await connector.void_envelope(envelope_id, reason)
            if success:
                logger.info(f"[Legal] Voided envelope {envelope_id} for tenant {tenant_id}")
                return success_response(
                    {"message": "Envelope voided successfully", "envelope_id": envelope_id}
                )
            else:
                return error_response("Failed to void envelope", 500)

        except Exception as e:
            logger.error(f"Failed to void envelope {envelope_id}: {e}")
            return error_response(f"Failed to void envelope: {e}", 500)

    async def _handle_resend_envelope(
        self, request: Any, tenant_id: str, envelope_id: str
    ) -> HandlerResult:
        """Resend envelope notifications to recipients."""
        connector = await get_docusign_connector(tenant_id)
        if not connector:
            return error_response("DocuSign not configured", 503)

        if not connector.is_authenticated:
            try:
                await connector.authenticate_jwt()
            except Exception as e:
                return error_response(f"Authentication failed: {e}", 401)

        try:
            success = await connector.resend_envelope(envelope_id)
            if success:
                logger.info(f"[Legal] Resent envelope {envelope_id} for tenant {tenant_id}")
                return success_response(
                    {
                        "message": "Notifications resent successfully",
                        "envelope_id": envelope_id,
                    }
                )
            else:
                return error_response("Failed to resend notifications", 500)

        except Exception as e:
            logger.error(f"Failed to resend envelope {envelope_id}: {e}")
            return error_response(f"Failed to resend notifications: {e}", 500)

    # =========================================================================
    # Documents
    # =========================================================================

    async def _handle_download_document(
        self, request: Any, tenant_id: str, envelope_id: str, document_id: str
    ) -> HandlerResult:
        """Download a signed document."""
        connector = await get_docusign_connector(tenant_id)
        if not connector:
            return error_response("DocuSign not configured", 503)

        if not connector.is_authenticated:
            try:
                await connector.authenticate_jwt()
            except Exception as e:
                return error_response(f"Authentication failed: {e}", 401)

        try:
            content = await connector.download_document(envelope_id, document_id)

            # Return base64-encoded content for JSON response
            return success_response(
                {
                    "envelope_id": envelope_id,
                    "document_id": document_id,
                    "content_base64": base64.b64encode(content).decode("utf-8"),
                    "content_type": "application/pdf",
                }
            )

        except Exception as e:
            logger.error(f"Failed to download document {document_id} from {envelope_id}: {e}")
            return error_response(f"Failed to download document: {e}", 500)

    async def _handle_download_certificate(
        self, request: Any, tenant_id: str, envelope_id: str
    ) -> HandlerResult:
        """Download the signing certificate of completion."""
        connector = await get_docusign_connector(tenant_id)
        if not connector:
            return error_response("DocuSign not configured", 503)

        if not connector.is_authenticated:
            try:
                await connector.authenticate_jwt()
            except Exception as e:
                return error_response(f"Authentication failed: {e}", 401)

        try:
            content = await connector.download_certificate(envelope_id)

            return success_response(
                {
                    "envelope_id": envelope_id,
                    "content_base64": base64.b64encode(content).decode("utf-8"),
                    "content_type": "application/pdf",
                }
            )

        except Exception as e:
            logger.error(f"Failed to download certificate for {envelope_id}: {e}")
            return error_response(f"Failed to download certificate: {e}", 500)

    # =========================================================================
    # Templates
    # =========================================================================

    async def _handle_list_templates(self, request: Any, tenant_id: str) -> HandlerResult:
        """List available DocuSign templates."""
        # Templates would require additional connector methods
        # For now return a placeholder
        return success_response(
            {
                "templates": [],
                "message": "Template listing not yet implemented",
            }
        )

    # =========================================================================
    # Webhooks
    # =========================================================================

    async def _handle_docusign_webhook(self, request: Any, tenant_id: str) -> HandlerResult:
        """Handle DocuSign Connect webhook notifications.

        DocuSign sends envelope status updates to this endpoint.
        """
        try:
            body = await self._get_json_body(request)

            # Extract envelope info
            envelope_id = body.get("envelopeId")
            status = body.get("status")
            event_time = body.get("statusChangedDateTime")

            logger.info(
                f"[Legal] DocuSign webhook: envelope={envelope_id} status={status} time={event_time}"
            )

            # Emit event for downstream processing
            await self._emit_connector_event(
                event_type="docusign_envelope_status",
                tenant_id=tenant_id,
                data={
                    "envelope_id": envelope_id,
                    "status": status,
                    "event_time": event_time,
                },
            )

            return success_response(
                {
                    "received": True,
                    "envelope_id": envelope_id,
                    "status": status,
                    "event_time": event_time,
                }
            )

        except Exception as e:
            logger.error(f"Error processing DocuSign webhook: {e}")
            # Return 200 to prevent retries for malformed payloads
            return success_response({"received": True, "error": str(e)})

    # =========================================================================
    # Utilities
    # =========================================================================

    def _get_query_params(self, request: Any) -> Dict[str, str]:
        """Extract query parameters from request."""
        if hasattr(request, "query"):
            return dict(request.query)
        if hasattr(request, "query_string"):
            from urllib.parse import parse_qs

            return {k: v[0] for k, v in parse_qs(request.query_string).items()}
        return {}

    async def _get_json_body(self, request: Any) -> Dict[str, Any]:
        """Parse JSON body from request."""
        if hasattr(request, "json"):
            if callable(request.json):
                return await request.json()
            return request.json
        return {}

    async def _emit_connector_event(
        self,
        event_type: str,
        tenant_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Emit a connector event for downstream processing.

        Events can trigger workflows, update caches, or send notifications.
        """
        try:
            from aragora.events.types import StreamEventType

            event_data = {
                "connector": "docusign",
                "event_type": event_type,
                "tenant_id": tenant_id,
                **data,
            }

            # Log structured event for processing pipelines
            logger.info(
                f"[Legal] Connector event: {event_type}",
                extra={"event_data": event_data},
            )

            # If we have a server context with an emitter, emit the event
            if self.server_context and "emitter" in self.server_context:
                emitter = self.server_context["emitter"]
                emitter.emit(
                    StreamEventType.CONNECTOR_DOCUSIGN_ENVELOPE_STATUS.value,
                    event_data,
                )
        except Exception as e:
            logger.debug(f"[Legal] Event emission skipped: {e}")


# =============================================================================
# Factory
# =============================================================================


def create_legal_handler(server_context: Optional[Dict[str, Any]] = None) -> LegalHandler:
    """Create a legal handler instance."""
    return LegalHandler(server_context)


__all__ = ["LegalHandler", "create_legal_handler"]
