"""
Receipt handling methods for gauntlet stress-tests.

This module contains:
- _get_receipt: Get decision receipt for a gauntlet run
- _verify_receipt: Verify a signed decision receipt
- _auto_persist_receipt: Auto-persist receipt after completion
- _risk_level_from_score: Helper for risk level calculation
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime
from typing import Any

from aragora.rbac.decorators import require_permission

from ..base import HandlerResult, error_response, get_string_param, json_response
from ..openapi_decorator import api_endpoint
from .storage import get_gauntlet_runs


def _get_storage_proxy():
    """Resolve storage accessor dynamically for test patching."""
    from . import _get_storage as get_storage

    return get_storage()


logger = logging.getLogger(__name__)


class GauntletReceiptsMixin:
    """Mixin providing gauntlet receipt methods."""

    @api_endpoint(
        method="GET",
        path="/api/v1/gauntlet/{gauntlet_id}/receipt",
        summary="Get decision receipt",
        description="Get the cryptographic decision receipt for a completed gauntlet run.",
        tags=["Gauntlet", "Receipts"],
        operation_id="get_gauntlet_receipt",
        parameters=[
            {"name": "gauntlet_id", "in": "path", "required": True, "schema": {"type": "string"}},
            {
                "name": "format",
                "in": "query",
                "schema": {"type": "string", "enum": ["json", "html", "md", "sarif", "pdf", "csv"]},
            },
            {"name": "signed", "in": "query", "schema": {"type": "string", "default": "true"}},
        ],
        responses={
            "200": {"description": "Decision receipt in requested format"},
            "400": {"description": "Gauntlet not completed"},
            "401": {"description": "Authentication required"},
            "404": {"description": "Gauntlet run not found"},
        },
    )
    @require_permission("gauntlet:read")
    async def _get_receipt(self, gauntlet_id: str, query_params: dict) -> HandlerResult:
        """Get decision receipt for gauntlet run."""
        from aragora.gauntlet.errors import gauntlet_error_response
        from aragora.gauntlet.receipt import DecisionReceipt

        gauntlet_runs = get_gauntlet_runs()

        run = None
        result = None
        result_obj = None

        # Check in-memory first
        if gauntlet_id in gauntlet_runs:
            run = gauntlet_runs[gauntlet_id]
            if run["status"] != "completed":
                body, status = gauntlet_error_response(
                    "not_completed", {"gauntlet_id": gauntlet_id}
                )
                return json_response(body, status=status)
            result = run["result"]
            result_obj = run.get("result_obj")
        else:
            # Check persistent storage
            try:
                storage = _get_storage_proxy()
                stored = storage.get(gauntlet_id)
                if stored:
                    result = stored
                else:
                    body, status = gauntlet_error_response(
                        "gauntlet_not_found", {"gauntlet_id": gauntlet_id}
                    )
                    return json_response(body, status=status)
            except (OSError, RuntimeError, ValueError) as e:
                logger.warning(f"Storage lookup failed for {gauntlet_id}: {e}")
                body, status = gauntlet_error_response("storage_error", {"reason": str(e)})
                return json_response(body, status=status)

        # Generate receipt
        if result_obj:
            receipt = DecisionReceipt.from_mode_result(
                result_obj,
                input_hash=run.get("input_hash") if run else None,
            )
        else:
            receipt = DecisionReceipt(
                receipt_id=f"receipt-{gauntlet_id[-12:]}",
                gauntlet_id=gauntlet_id,
                timestamp=run.get("completed_at", "") if run else datetime.now().isoformat(),
                input_summary=run["input_summary"] if run else result.get("input_summary", ""),
                input_hash=(
                    run.get("input_hash", gauntlet_id)
                    if run
                    else result.get("input_hash", gauntlet_id)
                ),
                risk_summary={
                    "critical": result.get("critical_count", 0),
                    "high": result.get("high_count", 0),
                    "medium": result.get("medium_count", 0),
                    "low": result.get("low_count", 0),
                    "total": result.get("total_findings", 0),
                },
                attacks_attempted=0,
                attacks_successful=0,
                probes_run=0,
                vulnerabilities_found=result.get("total_findings", 0),
                verdict=result.get("verdict", "UNKNOWN").upper(),
                confidence=result.get("confidence", 0),
                robustness_score=result.get("robustness_score", 0),
            )

        # Return format based on query param
        # Supported formats: json (default), html, md, sarif, pdf, csv
        format_type = get_string_param(query_params, "format", "json")

        # Sign the receipt by default (use signed=false query param to disable)
        skip_signing = get_string_param(query_params, "signed", "true") == "false"
        if not skip_signing:
            try:
                receipt.sign()
            except (ImportError, ValueError) as e:
                logger.warning(f"Receipt signing failed: {e}")
                # Continue with unsigned receipt

        # Prepare receipt data
        receipt_data = receipt.to_dict()

        def _notify_export(export_format: str, size_bytes: int | None = None) -> None:
            try:
                from aragora.integrations.receipt_webhooks import get_receipt_notifier

                notifier = get_receipt_notifier()
                notifier.notify_receipt_exported(
                    receipt_id=receipt.receipt_id,
                    debate_id=gauntlet_id,
                    export_format=export_format,
                    file_size=size_bytes,
                )
            except Exception as e:
                logger.debug(f"Receipt export webhook skipped: {e}")

        if format_type == "html":
            html_bytes = receipt.to_html().encode("utf-8")
            _notify_export("html", len(html_bytes))
            return HandlerResult(
                status_code=200,
                content_type="text/html",
                body=html_bytes,
            )
        elif format_type == "md":
            md_bytes = receipt.to_markdown().encode("utf-8")
            _notify_export("markdown", len(md_bytes))
            return HandlerResult(
                status_code=200,
                content_type="text/markdown",
                body=md_bytes,
            )
        elif format_type == "sarif":
            # SARIF 2.1.0 format for security tool integration
            sarif_bytes = receipt.to_sarif_json().encode("utf-8")
            _notify_export("sarif", len(sarif_bytes))
            return HandlerResult(
                status_code=200,
                content_type="application/sarif+json",
                body=sarif_bytes,
                headers={"Content-Disposition": f'attachment; filename="{gauntlet_id}.sarif"'},
            )
        elif format_type == "pdf":
            # PDF format (requires weasyprint)
            try:
                pdf_bytes = receipt.to_pdf()
                _notify_export("pdf", len(pdf_bytes))
                return HandlerResult(
                    status_code=200,
                    content_type="application/pdf",
                    body=pdf_bytes,
                    headers={
                        "Content-Disposition": f'attachment; filename="{gauntlet_id}-receipt.pdf"'
                    },
                )
            except ImportError:
                return error_response(
                    "PDF export requires weasyprint. Install with: pip install weasyprint",
                    501,
                )
        elif format_type == "csv":
            # CSV format for spreadsheet import
            csv_bytes = receipt.to_csv().encode("utf-8")
            _notify_export("csv", len(csv_bytes))
            return HandlerResult(
                status_code=200,
                content_type="text/csv",
                body=csv_bytes,
                headers={
                    "Content-Disposition": f'attachment; filename="{gauntlet_id}-findings.csv"'
                },
            )
        else:
            _notify_export("json", len(json.dumps(receipt_data)))
            return json_response(receipt_data)

    @api_endpoint(
        method="POST",
        path="/api/v1/gauntlet/{gauntlet_id}/receipt/verify",
        summary="Verify decision receipt",
        description="Verify the cryptographic signature and integrity of a signed decision receipt.",
        tags=["Gauntlet", "Receipts"],
        parameters=[
            {"name": "gauntlet_id", "in": "path", "required": True, "schema": {"type": "string"}}
        ],
        responses={
            "200": {"description": "Verification result with detailed status"},
            "400": {"description": "Invalid or missing request body"},
            "401": {"description": "Authentication required"},
        },
    )
    @require_permission("gauntlet:read")
    async def _verify_receipt(self, gauntlet_id: str, handler: Any) -> HandlerResult:
        """Verify a signed decision receipt.

        Validates:
        1. Cryptographic signature authenticity
        2. Artifact hash integrity (content not tampered)
        3. Receipt ID matches gauntlet ID

        Request body should be a SignedReceipt dict with:
        - receipt: The receipt data
        - signature: Base64-encoded signature
        - signature_metadata: Algorithm, timestamp, key_id

        Returns verification result with detailed status.
        """
        from aragora.gauntlet.receipt import DecisionReceipt
        from aragora.gauntlet.signing import SignedReceipt, verify_receipt

        # Parse request body
        from typing import cast

        data = cast(Any, self).read_json_body(handler)  # Mixin method from base handler
        if data is None:
            return error_response("Invalid or missing request body", 400)

        # Validate required fields
        if "receipt" not in data or "signature" not in data:
            return error_response("Missing required fields: 'receipt' and 'signature'", 400)

        if "signature_metadata" not in data:
            return error_response("Missing required field: 'signature_metadata'", 400)

        try:
            # Parse signed receipt
            signed_receipt = SignedReceipt.from_dict(data)
        except (KeyError, TypeError, ValueError) as e:
            return error_response(f"Invalid signed receipt format: {e}", 400)

        # Initialize verification result
        verification_result = {
            "gauntlet_id": gauntlet_id,
            "receipt_id": signed_receipt.receipt_data.get("receipt_id"),
            "verified": False,
            "signature_valid": False,
            "integrity_valid": False,
            "id_match": False,
            "errors": [],
            "warnings": [],
            "verified_at": datetime.now().isoformat(),
        }

        # Check receipt ID matches gauntlet ID
        receipt_gauntlet_id = signed_receipt.receipt_data.get("gauntlet_id")
        if receipt_gauntlet_id == gauntlet_id:
            verification_result["id_match"] = True
        else:
            verification_result["errors"].append(
                f"Receipt gauntlet_id '{receipt_gauntlet_id}' does not match "
                f"requested gauntlet_id '{gauntlet_id}'"
            )

        # Verify cryptographic signature
        try:
            signature_valid = verify_receipt(signed_receipt)
            verification_result["signature_valid"] = signature_valid
            if not signature_valid:
                verification_result["errors"].append("Cryptographic signature is invalid")
        except (ImportError, ValueError, RuntimeError) as e:
            verification_result["errors"].append(f"Signature verification failed: {e}")

        # Verify artifact hash integrity
        try:
            receipt_dict = signed_receipt.receipt_data
            # Reconstruct DecisionReceipt to check integrity
            receipt = DecisionReceipt(
                receipt_id=receipt_dict.get("receipt_id", ""),
                gauntlet_id=receipt_dict.get("gauntlet_id", ""),
                timestamp=receipt_dict.get("timestamp", ""),
                input_summary=receipt_dict.get("input_summary", ""),
                input_hash=receipt_dict.get("input_hash", ""),
                risk_summary=receipt_dict.get("risk_summary", {}),
                attacks_attempted=receipt_dict.get("attacks_attempted", 0),
                attacks_successful=receipt_dict.get("attacks_successful", 0),
                probes_run=receipt_dict.get("probes_run", 0),
                vulnerabilities_found=receipt_dict.get("vulnerabilities_found", 0),
                verdict=receipt_dict.get("verdict", ""),
                confidence=receipt_dict.get("confidence", 0.0),
                robustness_score=receipt_dict.get("robustness_score", 0.0),
                artifact_hash=receipt_dict.get("artifact_hash", ""),
            )

            integrity_valid = receipt.verify_integrity()
            verification_result["integrity_valid"] = integrity_valid
            if not integrity_valid:
                verification_result["errors"].append(
                    "Artifact hash mismatch - receipt content may have been tampered"
                )
        except (KeyError, TypeError, ValueError) as e:
            verification_result["errors"].append(f"Integrity verification failed: {e}")

        # Set overall verification status
        verification_result["verified"] = (
            verification_result["signature_valid"]
            and verification_result["integrity_valid"]
            and verification_result["id_match"]
        )

        # Add metadata about the verification
        verification_result["signature_metadata"] = {
            "algorithm": signed_receipt.signature_metadata.algorithm,
            "key_id": signed_receipt.signature_metadata.key_id,
            "signed_at": signed_receipt.signature_metadata.timestamp,
        }

        # Emit webhook based on verification result
        try:
            from aragora.integrations.receipt_webhooks import get_receipt_notifier

            notifier = get_receipt_notifier()
            receipt_id = signed_receipt.receipt_data.get("receipt_id", "")
            receipt_hash = signed_receipt.receipt_data.get(
                "artifact_hash", ""
            ) or signed_receipt.receipt_data.get("checksum", "")
            computed_hash = ""
            try:
                computed_hash = receipt._calculate_hash()
            except Exception as e:
                logger.debug(f"Error calculating receipt hash: {e}")
                computed_hash = ""

            if verification_result["verified"]:
                notifier.notify_receipt_verified(
                    receipt_id=receipt_id,
                    debate_id=gauntlet_id,
                    hash=receipt_hash,
                    computed_hash=computed_hash,
                    valid=True,
                )
            else:
                notifier.notify_receipt_integrity_failed(
                    receipt_id=receipt_id,
                    debate_id=gauntlet_id,
                    expected_hash=receipt_hash,
                    computed_hash=computed_hash,
                    error_message="; ".join(verification_result.get("errors", []))
                    or "verification failed",
                )
        except Exception as e:
            logger.debug(f"Receipt verification webhook skipped: {e}")

        # Return appropriate status code
        if verification_result["verified"]:
            return json_response(verification_result)
        else:
            # Return 200 with verification failure details (not a client error)
            return json_response(verification_result)

    async def _auto_persist_receipt(self, result: Any, gauntlet_id: str) -> None:
        """Auto-persist decision receipt after gauntlet completion.

        Generates and stores a decision receipt for compliance and audit trail.
        Optionally signs the receipt if ARAGORA_AUTO_SIGN_RECEIPTS=true.
        """
        gauntlet_runs = get_gauntlet_runs()

        try:
            from aragora.gauntlet.receipt import DecisionReceipt
            from aragora.storage.receipt_store import StoredReceipt, get_receipt_store

            # Get run data for input hash
            run = gauntlet_runs.get(gauntlet_id, {})

            # Generate receipt from result
            receipt = DecisionReceipt.from_mode_result(
                result,
                input_hash=run.get("input_hash"),
            )

            # Sign the receipt
            receipt.sign()

            # Create stored receipt
            stored = StoredReceipt(
                receipt_id=receipt.receipt_id,
                gauntlet_id=gauntlet_id,
                debate_id=getattr(result, "debate_id", None),
                created_at=time.time(),
                expires_at=None,  # Receipts don't expire by default
                verdict=receipt.verdict,
                confidence=receipt.confidence,
                risk_level=self._risk_level_from_score(receipt.robustness_score),
                risk_score=1.0 - receipt.robustness_score,  # Invert: higher score = lower risk
                checksum=hashlib.sha256(str(receipt.to_dict()).encode()).hexdigest(),
                data=receipt.to_dict(),
            )

            # Save to receipt store with signature data
            store = get_receipt_store()
            receipt_dict = receipt.to_dict()

            # Pass signature separately for the store
            signed_receipt = None
            if receipt.signature:
                signed_receipt = {
                    "signature": receipt.signature,
                    "signature_metadata": {
                        "algorithm": receipt.signature_algorithm,
                        "key_id": receipt.signature_key_id,
                        "timestamp": receipt.signed_at,
                    },
                }

            store.save(receipt_dict, signed_receipt=signed_receipt)
            logger.info(f"Decision receipt auto-persisted: {receipt.receipt_id}")

            # Emit receipt generated webhook
            try:
                from aragora.integrations.receipt_webhooks import get_receipt_notifier

                notifier = get_receipt_notifier()
                debate_id = getattr(result, "debate_id", None) or gauntlet_id
                agents = getattr(result, "agents_involved", None) or getattr(result, "agents", None)
                rounds = getattr(result, "rounds_completed", None) or getattr(
                    result, "rounds_used", None
                )
                findings_count = getattr(result, "total_findings", None)
                if findings_count is None:
                    findings_count = len(getattr(receipt, "vulnerability_details", []) or [])
                notifier.notify_receipt_generated(
                    receipt_id=receipt.receipt_id,
                    debate_id=debate_id,
                    verdict=receipt.verdict,
                    confidence=receipt.confidence,
                    hash=stored.checksum,
                    agents=agents,
                    rounds=rounds,
                    findings_count=findings_count,
                )
            except Exception as e:
                logger.debug(f"Receipt webhook notification skipped: {e}")

            # Auto-ingest receipt to Knowledge Mound for cross-debate learning
            try:
                from aragora.knowledge.mound.adapters.receipt_adapter import ReceiptAdapter
                from aragora.knowledge.mound import get_knowledge_mound
                from typing import cast, Any

                mound = get_knowledge_mound()
                if mound:
                    adapter = ReceiptAdapter(auto_ingest=True)
                    adapter.set_mound(mound)

                    # Extract workspace_id from run context if available
                    workspace_id = run.get("workspace_id") or run.get("tenant_id")

                    # Ingest receipt - extracts claims, findings, dissenting views
                    # Cast receipt to Any since the adapter accepts multiple receipt types
                    ingest_result = await adapter.ingest_receipt(
                        receipt=cast(Any, receipt),
                        workspace_id=workspace_id,
                    )

                    # Check if ingest_result has success attribute (dataclass) or get method (dict)
                    success = (
                        getattr(ingest_result, "success", None)
                        if hasattr(ingest_result, "success")
                        else not bool(getattr(ingest_result, "errors", []))
                    )
                    if success:
                        claims_count = getattr(ingest_result, "claims_ingested", 0)
                        findings_count = getattr(ingest_result, "findings_ingested", 0)
                        logger.info(
                            f"Receipt {receipt.receipt_id} ingested to KM: "
                            f"{claims_count} claims, {findings_count} findings"
                        )
                    else:
                        errors = getattr(ingest_result, "errors", [])
                        error_msg = errors[0] if errors else "unknown"
                        logger.debug(f"Receipt KM ingestion returned non-success: {error_msg}")
            except ImportError:
                logger.debug("Knowledge Mound not available for receipt ingestion")
            except Exception as e:
                logger.debug(f"Receipt KM ingestion skipped: {e}")

            # Optional auto-signing
            if os.environ.get("ARAGORA_AUTO_SIGN_RECEIPTS", "").lower() in ("true", "1", "yes"):
                try:
                    from aragora.gauntlet.signing import sign_receipt

                    signed = sign_receipt(receipt.to_dict())
                    store.update_signature(
                        receipt.receipt_id,
                        signature=signed.signature,
                        algorithm=signed.signature_metadata.algorithm,
                        key_id=signed.signature_metadata.key_id,
                    )
                    logger.info(f"Receipt auto-signed: {receipt.receipt_id}")
                except (ImportError, ValueError) as sign_err:
                    logger.warning(f"Auto-signing failed for {receipt.receipt_id}: {sign_err}")

        except ImportError as e:
            logger.debug(f"Receipt persistence skipped (module not available): {e}")
        except Exception as e:
            logger.warning(f"Failed to auto-persist receipt for {gauntlet_id}: {e}")

    def _risk_level_from_score(self, robustness_score: float) -> str:
        """Determine risk level from robustness score."""
        if robustness_score >= 0.8:
            return "LOW"
        elif robustness_score >= 0.6:
            return "MEDIUM"
        elif robustness_score >= 0.4:
            return "HIGH"
        else:
            return "CRITICAL"
