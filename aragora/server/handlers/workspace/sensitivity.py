"""
Sensitivity Classification Handlers for Workspace Package.

This module contains handlers for content sensitivity classification:
- handle_classify_content: Classify content sensitivity level
- handle_get_level_policy: Get policy for a specific sensitivity level

Stability: STABLE
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aragora.billing.jwt_auth import extract_user_from_request
from aragora.privacy import (
    AuditAction,
    AuditOutcome,
    SensitivityLevel,
)
from aragora.privacy.audit_log import Actor, Resource
from aragora.protocols import HTTPRequestHandler
from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)
from aragora.server.handlers.openapi_decorator import api_endpoint
from aragora.server.handlers.utils.rate_limit import rate_limit

if TYPE_CHECKING:
    from aragora.server.handlers.workspace import WorkspaceHandler

logger = logging.getLogger(__name__)


@api_endpoint(
    method="POST",
    path="/api/v1/classify",
    summary="Classify content sensitivity",
    tags=["Classification"],
)
@rate_limit(requests_per_minute=60, limiter_name="classify")
@handle_errors("classify content")
def handle_classify_content(
    handler_instance: "WorkspaceHandler",
    handler: HTTPRequestHandler,
) -> HandlerResult:
    """Classify content sensitivity.

    Args:
        handler_instance: The WorkspaceHandler instance
        handler: The HTTP request handler

    Returns:
        HandlerResult with classification details
    """
    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)
    if not auth_ctx.is_authenticated:
        return error_response("Not authenticated", 401)

    body = handler_instance.read_json_body(handler)
    if body is None:
        return error_response("Invalid JSON body", 400)

    content = body.get("content")
    if not content:
        return error_response("content is required", 400)

    document_id = body.get("document_id", "")
    metadata = body.get("metadata", {})

    classifier = handler_instance._get_classifier()
    result = handler_instance._run_async(
        classifier.classify(
            content=content,
            document_id=document_id,
            metadata=metadata,
        )
    )

    # Log to audit if document_id provided
    if document_id:
        audit_log = handler_instance._get_audit_log()
        handler_instance._run_async(
            audit_log.log(
                action=AuditAction.CLASSIFY_DOCUMENT,
                actor=Actor(id=auth_ctx.user_id, type="user"),
                resource=Resource(
                    id=document_id,
                    type="document",
                    sensitivity_level=result.level.value,
                ),
                outcome=AuditOutcome.SUCCESS,
                details={"level": result.level.value, "confidence": result.confidence},
            )
        )

    return json_response({"classification": result.to_dict()})


@api_endpoint(
    method="GET",
    path="/api/v1/classify/policy/{level}",
    summary="Get policy for sensitivity level",
    tags=["Classification"],
)
@handle_errors("get level policy")
def handle_get_level_policy(
    handler_instance: "WorkspaceHandler",
    handler: HTTPRequestHandler,
    level: str,
) -> HandlerResult:
    """Get recommended policy for a sensitivity level.

    Args:
        handler_instance: The WorkspaceHandler instance
        handler: The HTTP request handler
        level: The sensitivity level to get policy for

    Returns:
        HandlerResult with policy details for the level
    """
    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)
    if not auth_ctx.is_authenticated:
        return error_response("Not authenticated", 401)

    try:
        sensitivity_level = SensitivityLevel(level)
    except ValueError:
        valid_levels = [lvl.value for lvl in SensitivityLevel]
        return error_response(f"Invalid level: {level}. Valid: {', '.join(valid_levels)}", 400)

    classifier = handler_instance._get_classifier()
    policy = classifier.get_level_policy(sensitivity_level)

    return json_response({"level": level, "policy": policy})


__all__ = [
    "handle_classify_content",
    "handle_get_level_policy",
]
