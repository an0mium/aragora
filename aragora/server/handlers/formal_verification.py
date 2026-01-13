"""
Formal Verification API Endpoints.

Endpoints:
- POST /api/verify/claim - Verify a single claim
- POST /api/verify/batch - Batch verification of multiple claims
- GET /api/verify/status - Get backend availability status
- POST /api/verify/translate - Translate claim to formal language only
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    handle_errors,
    safe_json_parse,
)
from .utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)


def _init_verification():
    """Deferred import to avoid circular dependencies."""
    from aragora.verification.formal import (
        FormalVerificationManager,
        FormalProofStatus,
        FormalLanguage,
        TranslationModel,
        get_formal_verification_manager,
    )

    return {
        "FormalVerificationManager": FormalVerificationManager,
        "FormalProofStatus": FormalProofStatus,
        "FormalLanguage": FormalLanguage,
        "TranslationModel": TranslationModel,
        "get_formal_verification_manager": get_formal_verification_manager,
    }


class FormalVerificationHandler(BaseHandler):
    """Handler for formal verification endpoints."""

    ROUTES = [
        "/api/verify/claim",
        "/api/verify/batch",
        "/api/verify/status",
        "/api/verify/translate",
    ]

    def __init__(self, server_context: dict = None):
        super().__init__(server_context or {})
        self._manager = None

    def _get_manager(self):
        """Get or create the formal verification manager."""
        if self._manager is None:
            mods = _init_verification()
            self._manager = mods["get_formal_verification_manager"]()
        return self._manager

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES or path.startswith("/api/verify/")

    async def handle_async(
        self,
        handler,
        method: str,
        path: str,
        body: Optional[bytes] = None,
    ) -> HandlerResult:
        """Route and handle formal verification requests."""
        if path == "/api/verify/claim" and method == "POST":
            return await self._handle_verify_claim(handler, body)
        elif path == "/api/verify/batch" and method == "POST":
            return await self._handle_verify_batch(handler, body)
        elif path == "/api/verify/status" and method == "GET":
            return self._handle_verify_status(handler)
        elif path == "/api/verify/translate" and method == "POST":
            return await self._handle_translate(handler, body)
        else:
            return error_response(f"Unknown path: {path}", 404)

    @handle_errors("formal verification claim")
    @rate_limit(rpm=30)
    async def _handle_verify_claim(self, handler, body: Optional[bytes]) -> HandlerResult:
        """
        POST /api/verify/claim - Verify a single claim.

        Request body:
        {
            "claim": "For all natural numbers n, n + 0 = n",
            "claim_type": "MATHEMATICAL",  // optional
            "context": "Basic arithmetic identity",  // optional
            "timeout": 60.0  // optional, seconds
        }

        Response:
        {
            "status": "proof_found",
            "language": "lean4",
            "is_verified": true,
            "formal_statement": "theorem claim_1 : ∀ n : ℕ, n + 0 = n := by simp",
            "proof_hash": "abc123...",
            "translation_time_ms": 1234.5,
            "proof_search_time_ms": 567.8,
            "error_message": "",
            "prover_version": "Lean 4.3.0"
        }
        """
        if not body:
            return error_response("Request body required", 400)

        data = safe_json_parse(body)
        if data is None:
            return error_response("Invalid JSON body", 400)

        claim = data.get("claim", "").strip()
        if not claim:
            return error_response("claim field required", 400)

        claim_type = data.get("claim_type")
        context = data.get("context", "")
        timeout = min(float(data.get("timeout", 60.0)), 300.0)  # Max 5 min

        manager = self._get_manager()
        result = await manager.attempt_formal_verification(
            claim=claim,
            claim_type=claim_type,
            context=context,
            timeout_seconds=timeout,
        )

        return json_response(result.to_dict())

    @handle_errors("formal verification batch")
    @rate_limit(rpm=10)
    async def _handle_verify_batch(self, handler, body: Optional[bytes]) -> HandlerResult:
        """
        POST /api/verify/batch - Batch verification of multiple claims.

        Request body:
        {
            "claims": [
                {"claim": "...", "claim_type": "...", "context": "..."},
                {"claim": "..."}
            ],
            "timeout_per_claim": 30.0,
            "max_concurrent": 3
        }

        Response:
        {
            "results": [
                {"status": "proof_found", ...},
                {"status": "translation_failed", ...}
            ],
            "summary": {
                "total": 2,
                "verified": 1,
                "failed": 1,
                "timeout": 0
            }
        }
        """
        if not body:
            return error_response("Request body required", 400)

        data = safe_json_parse(body)
        if data is None:
            return error_response("Invalid JSON body", 400)

        claims_data = data.get("claims", [])
        if not claims_data or not isinstance(claims_data, list):
            return error_response("claims array required", 400)

        if len(claims_data) > 20:
            return error_response("Maximum 20 claims per batch", 400)

        timeout_per = min(float(data.get("timeout_per_claim", 30.0)), 120.0)
        max_concurrent = min(int(data.get("max_concurrent", 3)), 5)

        manager = self._get_manager()
        mods = _init_verification()
        FormalProofStatus = mods["FormalProofStatus"]

        semaphore = asyncio.Semaphore(max_concurrent)

        async def verify_one(claim_info: dict):
            async with semaphore:
                claim = claim_info.get("claim", "").strip()
                if not claim:
                    return {"status": "error", "error_message": "Empty claim"}
                return await manager.attempt_formal_verification(
                    claim=claim,
                    claim_type=claim_info.get("claim_type"),
                    context=claim_info.get("context", ""),
                    timeout_seconds=timeout_per,
                )

        results = await asyncio.gather(
            *[verify_one(c) for c in claims_data],
            return_exceptions=True,
        )

        # Process results
        processed = []
        summary = {"total": len(results), "verified": 0, "failed": 0, "timeout": 0}

        for r in results:
            if isinstance(r, Exception):
                processed.append(
                    {
                        "status": "error",
                        "error_message": str(r),
                        "is_verified": False,
                    }
                )
                summary["failed"] += 1
            else:
                result_dict = r.to_dict()
                processed.append(result_dict)
                if r.status == FormalProofStatus.PROOF_FOUND:
                    summary["verified"] += 1
                elif r.status == FormalProofStatus.TIMEOUT:
                    summary["timeout"] += 1
                else:
                    summary["failed"] += 1

        return json_response({"results": processed, "summary": summary})

    @handle_errors("formal verification status")
    def _handle_verify_status(self, handler) -> HandlerResult:
        """
        GET /api/verify/status - Get backend availability status.

        Response:
        {
            "backends": [
                {"language": "z3_smt", "available": true},
                {"language": "lean4", "available": false}
            ],
            "any_available": true,
            "deepseek_prover_available": true
        }
        """
        manager = self._get_manager()
        status = manager.status_report()

        # Check DeepSeek-Prover availability
        try:
            from aragora.verification.deepseek_prover import DeepSeekProverTranslator

            translator = DeepSeekProverTranslator()
            status["deepseek_prover_available"] = translator.is_available
        except ImportError:
            status["deepseek_prover_available"] = False

        return json_response(status)

    @handle_errors("formal verification translate")
    @rate_limit(rpm=30)
    async def _handle_translate(self, handler, body: Optional[bytes]) -> HandlerResult:
        """
        POST /api/verify/translate - Translate claim to formal language only.

        Request body:
        {
            "claim": "For all natural numbers n, n + 0 = n",
            "context": "",  // optional
            "target_language": "lean4"  // optional: lean4, z3_smt
        }

        Response:
        {
            "success": true,
            "formal_statement": "theorem claim_1 : ∀ n : ℕ, n + 0 = n := by simp",
            "language": "lean4",
            "model_used": "deepseek/deepseek-prover-v2",
            "confidence": 0.85,
            "translation_time_ms": 1234.5,
            "error_message": ""
        }
        """
        if not body:
            return error_response("Request body required", 400)

        data = safe_json_parse(body)
        if data is None:
            return error_response("Invalid JSON body", 400)

        claim = data.get("claim", "").strip()
        if not claim:
            return error_response("claim field required", 400)

        context = data.get("context", "")
        target = data.get("target_language", "lean4")

        mods = _init_verification()

        if target == "lean4":
            # Try DeepSeek-Prover first
            try:
                from aragora.verification.deepseek_prover import DeepSeekProverTranslator

                translator = DeepSeekProverTranslator()
                if translator.is_available:
                    result = await translator.translate(claim, context)
                    return json_response(
                        {
                            "success": result.success,
                            "formal_statement": result.lean_code,
                            "language": "lean4",
                            "model_used": result.model_used,
                            "confidence": result.confidence,
                            "translation_time_ms": result.translation_time_ms,
                            "error_message": result.error_message,
                        }
                    )
            except ImportError:
                pass

            # Fallback to LeanBackend
            from aragora.verification.formal import LeanBackend

            backend = LeanBackend()
            formal_statement = await backend.translate(claim, context)
            return json_response(
                {
                    "success": formal_statement is not None,
                    "formal_statement": formal_statement,
                    "language": "lean4",
                    "model_used": "claude/openai",
                    "confidence": 0.6 if formal_statement else 0.0,
                    "translation_time_ms": 0,
                    "error_message": "" if formal_statement else "Translation failed",
                }
            )

        elif target == "z3_smt":
            from aragora.verification.formal import Z3Backend

            backend = Z3Backend()
            formal_statement = await backend.translate(claim, context)
            return json_response(
                {
                    "success": formal_statement is not None,
                    "formal_statement": formal_statement,
                    "language": "z3_smt",
                    "model_used": "pattern/llm",
                    "confidence": 0.7 if formal_statement else 0.0,
                    "translation_time_ms": 0,
                    "error_message": "" if formal_statement else "Translation failed",
                }
            )

        else:
            return error_response(f"Unknown target language: {target}", 400)


__all__ = ["FormalVerificationHandler"]
