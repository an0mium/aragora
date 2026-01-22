"""
Formal verification endpoint handlers.

Endpoints:
- GET /api/verification/status - Get status of formal verification backends
- POST /api/verification/formal-verify - Verify a claim using Z3 SMT solver
"""

from __future__ import annotations

import logging
from typing import Optional

from aragora.server.http_utils import run_async
from aragora.server.middleware.rate_limit import rate_limit
from aragora.server.validation.schema import VERIFICATION_SCHEMA, validate_against_schema
from aragora.utils.optional_imports import try_import

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)

logger = logging.getLogger(__name__)

# Optional formal verification imports
_fv_imports, FORMAL_VERIFICATION_AVAILABLE = try_import(
    "aragora.verification.formal", "FormalVerificationManager", "get_formal_verification_manager"
)
get_formal_verification_manager = _fv_imports.get("get_formal_verification_manager")


def _safe_float(value, default: float = 0.0) -> float:
    """Safely convert value to float, returning default on failure."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


class VerificationHandler(BaseHandler):
    """Handler for formal verification endpoints."""

    ROUTES = [
        "/api/v1/verification/status",
        "/api/v1/verification/formal-verify",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(self, path: str, query_params: dict, handler=None) -> Optional[HandlerResult]:
        """Route GET requests to appropriate methods."""
        if path == "/api/v1/verification/status":
            return self._get_status()
        return None

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route POST requests to appropriate methods."""
        if path == "/api/v1/verification/formal-verify":
            return self._verify_claim(handler)
        return None

    @handle_errors("formal verification status")
    def _get_status(self) -> HandlerResult:
        """Get status of formal verification backends.

        Returns availability of Z3 and Lean backends.
        """
        if not FORMAL_VERIFICATION_AVAILABLE:
            return json_response(
                {
                    "available": False,
                    "hint": "Install z3-solver: pip install z3-solver",
                    "backends": [],
                }
            )

        manager = get_formal_verification_manager()
        status = manager.status_report()
        return json_response(
            {
                "available": status.get("any_available", False),
                "backends": status.get("backends", []),
            }
        )

    @rate_limit(requests_per_minute=10, burst=3, limiter_name="formal_verification")
    @handle_errors("formal verification")
    def _verify_claim(self, handler) -> HandlerResult:
        """Attempt formal verification of a claim using Z3 SMT solver.

        POST body:
            claim: The claim to verify (required)
            claim_type: Optional hint (assertion, logical, arithmetic, etc.)
            context: Optional additional context
            timeout: Timeout in seconds (default: 30, max: 120)

        Returns:
            status: proof_found, proof_failed, translation_failed, etc.
            is_verified: True if claim was formally proven
            formal_statement: The SMT-LIB2 translation (if successful)
            proof_hash: Hash of the proof (if found)
        """
        if not FORMAL_VERIFICATION_AVAILABLE:
            return json_response(
                {
                    "error": "Formal verification not available",
                    "hint": "Install z3-solver: pip install z3-solver",
                },
                status=503,
            )

        # Read request body
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body or body too large", status=400)

        # Schema validation for input sanitization
        validation_result = validate_against_schema(body, VERIFICATION_SCHEMA)
        if not validation_result.is_valid:
            return error_response(validation_result.error, status=400)

        claim = body.get("claim", "").strip()

        claim_type = body.get("claim_type")
        context = body.get("context", "")
        timeout = min(_safe_float(body.get("timeout", 30), 30.0), 120.0)

        # Get the formal verification manager
        manager = get_formal_verification_manager()

        # Check backend availability
        status_report = manager.status_report()
        if not status_report.get("any_available"):
            return json_response(
                {
                    "error": "No formal verification backends available",
                    "backends": status_report.get("backends", []),
                },
                status=503,
            )

        # Run verification asynchronously
        # Use run_async() for safe sync/async bridging
        result = run_async(
            manager.attempt_formal_verification(
                claim=claim,
                claim_type=claim_type,
                context=context,
                timeout_seconds=timeout,
            )
        )

        # Build response with user-friendly enhancements
        response = result.to_dict()
        response["claim"] = claim
        if claim_type:
            response["claim_type"] = claim_type

        # Add user-friendly hints for failure cases
        status = result.status.value if hasattr(result.status, "value") else str(result.status)

        if status == "translation_failed":
            response["hint"] = (
                "The claim could not be translated to a formal logical statement. "
                "Try rephrasing with explicit logical structure."
            )
            response["examples"] = [
                "For all x, x + 0 = x",
                "If A > B and B > C, then A > C",
                "There exists a number greater than any given number",
                "All prime numbers greater than 2 are odd",
            ]
            response["suggestions"] = [
                "Use explicit quantifiers (for all, there exists)",
                "Use clear logical connectives (and, or, implies, if-then)",
                "Express mathematical relationships explicitly (equals, greater than)",
                "Avoid ambiguous natural language",
            ]

        elif status == "not_supported":
            response["hint"] = (
                "This type of claim is not suitable for formal verification. "
                "Formal verification works best with logical or mathematical statements."
            )
            response["supported_types"] = [
                "Logical statements (implications, conjunctions)",
                "Arithmetic equalities and inequalities",
                "Quantified statements (for all, there exists)",
                "Set membership and relations",
            ]

        elif status == "proof_failed":
            response["hint"] = (
                "The formal system found the claim to be false or unprovable. "
                "Check your assumptions or try a weaker formulation."
            )
            if response.get("proof_text"):
                response["counterexample_note"] = (
                    "A counterexample was found, meaning the claim is false as stated."
                )

        elif status == "timeout":
            response["hint"] = (
                "The proof search exceeded the time limit. "
                "Try a simpler claim or increase the timeout."
            )
            response["max_timeout"] = 120

        return json_response(response)
