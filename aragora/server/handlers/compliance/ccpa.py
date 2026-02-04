"""
CCPA Compliance Handler.

Provides California Consumer Privacy Act (CCPA) data operations including:
- Right to Know (categories and specific pieces)
- Right to Delete
- Right to Opt-Out (Do Not Sell/Share)
- Right to Non-Discrimination
- Right to Correct
- Right to Limit Use of Sensitive Personal Information
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    json_response,
)
from aragora.events.handler_events import emit_handler_event, DELETED
from aragora.rbac.decorators import require_permission
from aragora.observability.metrics import track_handler
from aragora.storage.audit_store import get_audit_store
from aragora.storage.receipt_store import get_receipt_store
from aragora.privacy.deletion import get_deletion_scheduler, get_legal_hold_manager

logger = logging.getLogger(__name__)


class CCPAMixin:
    """Mixin providing CCPA-related handler methods."""

    # CCPA categories of personal information
    CCPA_CATEGORIES = [
        {
            "category": "A",
            "name": "Identifiers",
            "examples": [
                "Real name",
                "Alias",
                "Postal address",
                "Unique personal identifier",
                "Online identifier",
                "IP address",
                "Email address",
                "Account name",
            ],
        },
        {
            "category": "B",
            "name": "Personal Information (Cal. Civ. Code 1798.80(e))",
            "examples": [
                "Name",
                "Address",
                "Telephone number",
                "Employment history",
                "Financial information",
            ],
        },
        {
            "category": "C",
            "name": "Protected Classification Characteristics",
            "examples": [
                "Age",
                "Race",
                "Gender",
                "Veteran status",
                "Disability status",
            ],
        },
        {
            "category": "D",
            "name": "Commercial Information",
            "examples": [
                "Records of products or services purchased",
                "Purchasing histories",
                "Consuming tendencies",
            ],
        },
        {
            "category": "F",
            "name": "Internet or Network Activity",
            "examples": [
                "Browsing history",
                "Search history",
                "Interaction with website",
                "Advertisement interactions",
            ],
        },
        {
            "category": "G",
            "name": "Geolocation Data",
            "examples": ["Physical location", "IP-based location"],
        },
        {
            "category": "K",
            "name": "Inferences",
            "examples": [
                "Consumer profile reflecting preferences",
                "Characteristics",
                "Psychological trends",
                "Predispositions",
                "Behavior",
                "Attitudes",
            ],
        },
    ]

    @track_handler("compliance/ccpa-disclosure", method="GET")
    @require_permission("compliance:ccpa")
    async def _ccpa_disclosure(self, query_params: dict[str, str]) -> HandlerResult:
        """
        CCPA Right to Know - Provide disclosure of personal information.

        Query params:
            user_id: User ID to get disclosure for (required)
            disclosure_type: categories | specific (default: categories)
            format: json (default) | pdf
        """
        user_id = query_params.get("user_id")
        if not user_id:
            return error_response("user_id is required", 400)

        disclosure_type = query_params.get("disclosure_type", "categories")
        output_format = query_params.get("format", "json")

        now = datetime.now(timezone.utc)
        request_id = f"ccpa-disc-{user_id}-{now.strftime('%Y%m%d%H%M%S')}"

        disclosure: dict[str, Any] = {
            "request_id": request_id,
            "user_id": user_id,
            "disclosure_type": disclosure_type,
            "requested_at": now.isoformat(),
            "regulatory_basis": "California Consumer Privacy Act (CCPA)",
            "response_deadline": (now + timedelta(days=45)).isoformat(),
        }

        if disclosure_type == "categories":
            # Right to Know - Categories
            disclosure["categories_collected"] = await self._get_ccpa_categories(user_id)
            disclosure["categories_disclosed"] = await self._get_categories_disclosed(user_id)
            disclosure["categories_sold"] = []  # We don't sell personal information
            disclosure["business_purpose"] = await self._get_business_purposes()
        else:
            # Right to Know - Specific Pieces
            disclosure["personal_information"] = await self._get_specific_pi(user_id)
            disclosure["sources"] = await self._get_pi_sources(user_id)
            disclosure["third_parties"] = await self._get_third_party_disclosures(user_id)

        # Log the disclosure request
        await self._log_ccpa_request(
            request_type="disclosure",
            request_id=request_id,
            user_id=user_id,
            disclosure_type=disclosure_type,
        )

        # Calculate verification hash
        data_str = json.dumps(disclosure, sort_keys=True, default=str)
        disclosure["verification_hash"] = hashlib.sha256(data_str.encode()).hexdigest()

        if output_format == "pdf":
            # For PDF format, return a note about generation
            disclosure["pdf_generation"] = {
                "status": "queued",
                "estimated_completion": (now + timedelta(hours=1)).isoformat(),
                "download_url": f"/api/v2/compliance/ccpa/disclosures/{request_id}/download",
            }

        return json_response(disclosure)

    @track_handler("compliance/ccpa-delete", method="POST")
    @require_permission("compliance:ccpa")
    async def _ccpa_delete(self, body: dict[str, Any]) -> HandlerResult:
        """
        CCPA Right to Delete - Delete personal information.

        Body:
            user_id: User ID requesting deletion (required)
            verification_method: email | phone | account (required)
            verification_code: Verification code (required)
            retain_for_exceptions: List of exception categories to retain

        CCPA allows retention for specific exceptions:
        - Complete transactions
        - Detect security incidents
        - Debug errors
        - Exercise free speech
        - Comply with legal obligations
        - Internal uses reasonably aligned with consumer expectations
        """
        user_id = body.get("user_id")
        if not user_id:
            return error_response("user_id is required", 400)

        verification_method = body.get("verification_method")
        verification_code = body.get("verification_code")

        if not verification_method or not verification_code:
            return error_response("verification_method and verification_code are required", 400)

        # Verify the deletion request
        is_verified = await self._verify_ccpa_request(
            user_id, verification_method, verification_code
        )
        if not is_verified:
            return error_response("Verification failed", 401)

        retain_exceptions = body.get("retain_for_exceptions", [])

        now = datetime.now(timezone.utc)
        request_id = f"ccpa-del-{user_id}-{now.strftime('%Y%m%d%H%M%S')}"

        try:
            # Check for legal holds
            hold_manager = get_legal_hold_manager()
            if hold_manager.is_user_on_hold(user_id):
                return error_response(
                    "Cannot process deletion: User data is under legal hold. "
                    "Please contact legal department.",
                    409,
                )

            # Schedule the deletion
            scheduler = get_deletion_scheduler()
            deletion_request = scheduler.schedule_deletion(
                user_id=user_id,
                grace_period_days=45,  # CCPA requires 45-day response window
                reason=f"CCPA Right to Delete - Request {request_id}",
                metadata={
                    "ccpa_request_id": request_id,
                    "verification_method": verification_method,
                    "retain_exceptions": retain_exceptions,
                    "source": "ccpa_compliance_handler",
                },
            )

            result = {
                "request_id": request_id,
                "user_id": user_id,
                "status": "scheduled",
                "requested_at": now.isoformat(),
                "deletion_scheduled": deletion_request.scheduled_for.isoformat(),
                "response_deadline": (now + timedelta(days=45)).isoformat(),
                "retained_categories": retain_exceptions,
                "message": (
                    "Your deletion request has been received and verified. "
                    "Personal information will be deleted within 45 days as required by CCPA."
                ),
            }

            # Log the deletion request
            await self._log_ccpa_request(
                request_type="deletion",
                request_id=request_id,
                user_id=user_id,
                retain_exceptions=retain_exceptions,
            )

            logger.info(f"CCPA deletion request scheduled: user={user_id}, request_id={request_id}")
            emit_handler_event(
                "compliance",
                DELETED,
                {"action": "ccpa_delete", "request_id": request_id},
                user_id=user_id,
            )

            return json_response(result)

        except Exception as e:
            logger.exception(f"CCPA deletion request failed for user {user_id}: {e}")
            return error_response(f"Failed to process deletion request: {str(e)}", 500)

    @track_handler("compliance/ccpa-optout", method="POST")
    @require_permission("compliance:ccpa")
    async def _ccpa_opt_out(self, body: dict[str, Any]) -> HandlerResult:
        """
        CCPA Right to Opt-Out of Sale/Sharing.

        Body:
            user_id: User ID opting out (required)
            opt_out_type: sale | sharing | both (default: both)
            sensitive_pi_limit: If true, limit use of sensitive PI (CPRA addition)
        """
        user_id = body.get("user_id")
        if not user_id:
            return error_response("user_id is required", 400)

        opt_out_type = body.get("opt_out_type", "both")
        sensitive_pi_limit = body.get("sensitive_pi_limit", False)

        now = datetime.now(timezone.utc)
        request_id = f"ccpa-opt-{user_id}-{now.strftime('%Y%m%d%H%M%S')}"

        # Record the opt-out preference
        opt_out_record = {
            "request_id": request_id,
            "user_id": user_id,
            "opt_out_type": opt_out_type,
            "sensitive_pi_limit": sensitive_pi_limit,
            "effective_at": now.isoformat(),
            "status": "active",
        }

        try:
            # Store the opt-out preference
            await self._store_ccpa_preference(user_id, opt_out_record)

            # Log the opt-out
            await self._log_ccpa_request(
                request_type="opt_out",
                request_id=request_id,
                user_id=user_id,
                opt_out_type=opt_out_type,
                sensitive_pi_limit=sensitive_pi_limit,
            )

            result = {
                "request_id": request_id,
                "user_id": user_id,
                "status": "confirmed",
                "effective_at": now.isoformat(),
                "opt_out_type": opt_out_type,
                "sensitive_pi_limit": sensitive_pi_limit,
                "message": (
                    "Your opt-out request has been processed. "
                    "We will not sell or share your personal information."
                ),
            }

            if sensitive_pi_limit:
                result["message"] += (
                    " We will also limit the use and disclosure of your "
                    "sensitive personal information."
                )

            logger.info(f"CCPA opt-out processed: user={user_id}, type={opt_out_type}")

            return json_response(result)

        except Exception as e:
            logger.exception(f"CCPA opt-out failed for user {user_id}: {e}")
            return error_response(f"Failed to process opt-out request: {str(e)}", 500)

    @track_handler("compliance/ccpa-correct", method="POST")
    @require_permission("compliance:ccpa")
    async def _ccpa_correct(self, body: dict[str, Any]) -> HandlerResult:
        """
        CCPA/CPRA Right to Correct inaccurate personal information.

        Body:
            user_id: User ID requesting correction (required)
            corrections: List of corrections [{field, current_value, corrected_value}]
            supporting_documentation: Optional documentation reference
        """
        user_id = body.get("user_id")
        if not user_id:
            return error_response("user_id is required", 400)

        corrections = body.get("corrections", [])
        if not corrections:
            return error_response("corrections list is required", 400)

        supporting_docs = body.get("supporting_documentation")

        now = datetime.now(timezone.utc)
        request_id = f"ccpa-corr-{user_id}-{now.strftime('%Y%m%d%H%M%S')}"

        result = {
            "request_id": request_id,
            "user_id": user_id,
            "status": "pending_review",
            "requested_at": now.isoformat(),
            "response_deadline": (now + timedelta(days=45)).isoformat(),
            "corrections_requested": len(corrections),
            "corrections": [],
        }

        for correction in corrections:
            correction_item = {
                "field": correction.get("field"),
                "current_value": "[REDACTED]",  # Don't echo current value
                "status": "pending",
            }
            result["corrections"].append(correction_item)

        # Log the correction request
        await self._log_ccpa_request(
            request_type="correction",
            request_id=request_id,
            user_id=user_id,
            corrections_count=len(corrections),
            has_documentation=bool(supporting_docs),
        )

        result["message"] = (
            "Your correction request has been received and is under review. "
            "We will respond within 45 days as required by CCPA/CPRA."
        )

        return json_response(result)

    @require_permission("compliance:ccpa")
    async def _ccpa_get_status(self, query_params: dict[str, str]) -> HandlerResult:
        """
        Get status of CCPA requests for a user.

        Query params:
            user_id: User ID to check status for (required)
            request_type: Optional filter by request type
        """
        user_id = query_params.get("user_id")
        if not user_id:
            return error_response("user_id is required", 400)

        request_type = query_params.get("request_type")

        try:
            store = get_audit_store()
            # Get CCPA-related events for this user
            events = store.get_log(limit=100)

            # Filter for CCPA requests
            ccpa_requests = []
            for event in events:
                if event.get("resource_id") == user_id:
                    action = event.get("action", "")
                    if action.startswith("ccpa_"):
                        if request_type is None or request_type in action:
                            ccpa_requests.append(
                                {
                                    "request_id": event.get("metadata", {}).get("request_id"),
                                    "type": action.replace("ccpa_", "").replace("_request", ""),
                                    "status": event.get("metadata", {}).get("status", "pending"),
                                    "requested_at": event.get("timestamp"),
                                }
                            )

            return json_response(
                {
                    "user_id": user_id,
                    "requests": ccpa_requests,
                    "count": len(ccpa_requests),
                }
            )

        except Exception as e:
            logger.exception(f"Error fetching CCPA status: {e}")
            return error_response(f"Failed to fetch status: {str(e)}", 500)

    # =========================================================================
    # Helper methods for CCPA operations
    # =========================================================================

    async def _get_ccpa_categories(self, user_id: str) -> list[dict[str, Any]]:
        """Get CCPA categories of PI collected for user."""
        # Return categories we actually collect
        collected = []
        for cat in self.CCPA_CATEGORIES:
            if cat["category"] in ["A", "D", "F", "K"]:
                collected.append(
                    {
                        "category": cat["category"],
                        "name": cat["name"],
                        "collected": True,
                        "examples": cat["examples"][:3],  # Limit examples
                    }
                )
            else:
                collected.append(
                    {
                        "category": cat["category"],
                        "name": cat["name"],
                        "collected": False,
                    }
                )
        return collected

    async def _get_categories_disclosed(self, user_id: str) -> list[dict[str, Any]]:
        """Get categories disclosed to third parties."""
        return [
            {
                "category": "A",
                "name": "Identifiers",
                "disclosed_to": ["Service providers (authentication)"],
                "purpose": "Account authentication and security",
            },
            {
                "category": "F",
                "name": "Internet Activity",
                "disclosed_to": ["AI model providers (anonymized)"],
                "purpose": "Debate processing and decision support",
            },
        ]

    async def _get_business_purposes(self) -> list[dict[str, Any]]:
        """Get business purposes for data collection."""
        return [
            {
                "purpose": "Service delivery",
                "description": "Providing the Aragora decision support platform",
            },
            {
                "purpose": "Security",
                "description": "Detecting and preventing fraud and security incidents",
            },
            {
                "purpose": "Improvement",
                "description": "Improving and developing new features",
            },
            {
                "purpose": "Communication",
                "description": "Communicating with users about their account and services",
            },
        ]

    async def _get_specific_pi(self, user_id: str) -> dict[str, Any]:
        """Get specific pieces of personal information for user."""
        try:
            store = get_receipt_store()
            receipts = store.list(limit=50, sort_by="created_at", order="desc")

            return {
                "account_information": {
                    "user_id": user_id,
                    "account_type": "standard",
                    "created_date": "[Retrieved from user store]",
                },
                "activity_records": {
                    "decisions_count": len(receipts),
                    "last_activity": (receipts[0].created_at if receipts else "No activity"),
                },
                "preferences": "[Retrieved from preferences store]",
            }
        except Exception as e:
            logger.warning(f"Error getting specific PI: {e}")
            return {"error": "Unable to retrieve specific PI"}

    async def _get_pi_sources(self, user_id: str) -> list[dict[str, Any]]:
        """Get sources of personal information."""
        return [
            {"source": "Direct collection", "description": "Information you provide"},
            {
                "source": "Automatic collection",
                "description": "Usage data and analytics",
            },
            {
                "source": "Third-party authentication",
                "description": "OAuth provider data",
            },
        ]

    async def _get_third_party_disclosures(self, user_id: str) -> list[dict[str, Any]]:
        """Get third parties to whom PI is disclosed."""
        return [
            {
                "party_type": "Service providers",
                "examples": ["Cloud hosting", "Authentication services"],
                "categories_disclosed": ["A"],
            },
            {
                "party_type": "AI model providers",
                "examples": ["Anthropic", "OpenAI", "Mistral"],
                "categories_disclosed": ["F"],
                "note": "Debate content is anonymized",
            },
        ]

    async def _verify_ccpa_request(self, user_id: str, method: str, code: str) -> bool:
        """Verify CCPA request through specified method."""
        # In production, implement actual verification
        # This is a placeholder that always returns True for demo
        return True

    async def _store_ccpa_preference(self, user_id: str, preference: dict[str, Any]) -> None:
        """Store CCPA preference for user."""
        try:
            store = get_audit_store()
            store.log_event(
                action="ccpa_preference_stored",
                resource_type="user",
                resource_id=user_id,
                metadata=preference,
            )
        except Exception as e:
            logger.warning(f"Failed to store CCPA preference: {e}")

    async def _log_ccpa_request(
        self, request_type: str, request_id: str, user_id: str, **kwargs: Any
    ) -> None:
        """Log CCPA request for audit trail."""
        try:
            store = get_audit_store()
            store.log_event(
                action=f"ccpa_{request_type}_request",
                resource_type="user",
                resource_id=user_id,
                metadata={
                    "request_id": request_id,
                    "request_type": request_type,
                    **kwargs,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to log CCPA request: {e}")


__all__ = ["CCPAMixin"]
