"""
Formal Verification API Endpoints.

Endpoints:
- POST /api/verify/claim - Verify a single claim
- POST /api/verify/batch - Batch verification of multiple claims
- GET /api/verify/status - Get backend availability status
- POST /api/verify/translate - Translate claim to formal language only
- GET /api/verify/history - Get verification history
- GET /api/verify/history/{id} - Get specific verification result
- GET /api/verify/history/{id}/tree - Get proof tree for visualization
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    get_clamped_int_param,
    handle_errors,
    json_response,
    safe_json_parse,
)
from ..utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)


# =============================================================================
# Verification History Storage
# =============================================================================

MAX_HISTORY_SIZE = 1000
HISTORY_TTL_SECONDS = 86400  # 24 hours


@dataclass
class VerificationHistoryEntry:
    """A single verification history entry."""

    id: str
    claim: str
    claim_type: Optional[str]
    context: str
    result: dict
    timestamp: float
    proof_tree: Optional[list] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "claim": self.claim,
            "claim_type": self.claim_type,
            "context": self.context,
            "result": self.result,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "has_proof_tree": self.proof_tree is not None,
        }


# In-memory history storage (OrderedDict for FIFO eviction)
# Used as cache, backed by GovernanceStore for persistence
_verification_history: OrderedDict[str, VerificationHistoryEntry] = OrderedDict()

# Lazy-loaded governance store
_governance_store = None


def _get_governance_store():
    """Get or create governance store for persistence."""
    global _governance_store
    if _governance_store is None:
        try:
            from aragora.storage.governance_store import get_governance_store

            _governance_store = get_governance_store()
        except ImportError:
            logger.debug("GovernanceStore not available, using in-memory only")
    return _governance_store


def _generate_verification_id(claim: str, timestamp: float) -> str:
    """Generate a unique ID for a verification entry."""
    data = f"{claim}:{timestamp}".encode()
    return hashlib.sha256(data).hexdigest()[:16]


def _add_to_history(
    claim: str,
    claim_type: Optional[str],
    context: str,
    result: dict,
    proof_tree: Optional[list] = None,
) -> str:
    """Add a verification result to history."""
    timestamp = time.time()
    entry_id = _generate_verification_id(claim, timestamp)

    entry = VerificationHistoryEntry(
        id=entry_id,
        claim=claim,
        claim_type=claim_type,
        context=context,
        result=result,
        timestamp=timestamp,
        proof_tree=proof_tree,
    )

    # Add to in-memory cache
    _verification_history[entry_id] = entry

    # Evict old entries if over limit
    while len(_verification_history) > MAX_HISTORY_SIZE:
        _verification_history.popitem(last=False)

    # Persist to GovernanceStore
    store = _get_governance_store()
    if store:
        try:
            store.save_verification(
                verification_id=entry_id,
                claim=claim,
                context=context,
                result=result,
                verified_by="formal_verification",
                claim_type=claim_type,
                confidence=result.get("confidence", 0.0) if isinstance(result, dict) else 0.0,
                proof_tree=proof_tree,
            )
            logger.debug(f"Persisted verification {entry_id} to GovernanceStore")
        except Exception as e:
            logger.warning(f"Failed to persist verification to GovernanceStore: {e}")

    return entry_id


def _cleanup_old_history():
    """Remove entries older than TTL."""
    cutoff = time.time() - HISTORY_TTL_SECONDS
    to_remove = [k for k, v in _verification_history.items() if v.timestamp < cutoff]
    for k in to_remove:
        del _verification_history[k]


def _build_proof_tree(result: dict) -> Optional[list]:
    """Build a proof tree structure from verification result."""
    if not result.get("is_verified"):
        return None

    formal_statement = result.get("formal_statement", "")
    if not formal_statement:
        return None

    # Parse the formal statement into tree nodes
    nodes = []

    # Root node: the claim
    nodes.append(
        {
            "id": "root",
            "type": "claim",
            "content": result.get("claim", "Original claim"),
            "children": ["translation"],
        }
    )

    # Translation node
    nodes.append(
        {
            "id": "translation",
            "type": "translation",
            "content": formal_statement,
            "language": result.get("language", "unknown"),
            "children": ["verification"],
        }
    )

    # Verification node
    status = result.get("status", "unknown")
    nodes.append(
        {
            "id": "verification",
            "type": "verification",
            "content": f"Status: {status}",
            "is_verified": result.get("is_verified", False),
            "proof_hash": result.get("proof_hash"),
            "children": [],
        }
    )

    # If we have proof steps, add them
    proof_steps = result.get("proof_steps", [])
    if proof_steps:
        nodes[-1]["children"] = [f"step_{i}" for i in range(len(proof_steps))]
        for i, step in enumerate(proof_steps):
            nodes.append(
                {
                    "id": f"step_{i}",
                    "type": "proof_step",
                    "content": step,
                    "step_number": i + 1,
                    "children": [],
                }
            )

    return nodes


def _init_verification():
    """Deferred import to avoid circular dependencies."""
    from aragora.verification.formal import (
        FormalLanguage,
        FormalProofStatus,
        FormalVerificationManager,
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
        "/api/v1/verify/claim",
        "/api/v1/verify/batch",
        "/api/v1/verify/status",
        "/api/v1/verify/translate",
        "/api/v1/verify/history",
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
        if path in self.ROUTES:
            return True
        if path.startswith("/api/v1/verify/history/"):
            return True
        return path.startswith("/api/v1/verify/")

    async def handle_async(
        self,
        handler,
        method: str,
        path: str,
        body: Optional[bytes] = None,
        query_params: Optional[dict] = None,
    ) -> HandlerResult:
        """Route and handle formal verification requests."""
        if path == "/api/v1/verify/claim" and method == "POST":
            return await self._handle_verify_claim(handler, body)
        elif path == "/api/v1/verify/batch" and method == "POST":
            return await self._handle_verify_batch(handler, body)
        elif path == "/api/v1/verify/status" and method == "GET":
            return self._handle_verify_status(handler)
        elif path == "/api/v1/verify/translate" and method == "POST":
            return await self._handle_translate(handler, body)
        elif path == "/api/v1/verify/history" and method == "GET":
            return self._handle_get_history(query_params or {})
        elif path.startswith("/api/v1/verify/history/") and method == "GET":
            return self._handle_get_history_entry(path)
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

        result_dict = result.to_dict()

        # Build proof tree and store in history
        proof_tree = _build_proof_tree({**result_dict, "claim": claim})
        entry_id = _add_to_history(
            claim=claim,
            claim_type=claim_type,
            context=context,
            result=result_dict,
            proof_tree=proof_tree,
        )

        # Include entry_id in response for retrieval
        result_dict["history_id"] = entry_id

        return json_response(result_dict)

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
            if isinstance(r, BaseException):
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

        _init_verification()  # Ensure verification module is loaded

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

            lean_backend = LeanBackend()
            formal_statement = await lean_backend.translate(claim, context)
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

            z3_backend = Z3Backend()
            formal_statement = await z3_backend.translate(claim, context)
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

    @handle_errors("verification history")
    def _handle_get_history(self, query_params: dict) -> HandlerResult:
        """
        GET /api/verify/history - Get verification history.

        Query params:
            limit: Max entries to return (default 20, max 100)
            offset: Skip first N entries (default 0)
            status: Filter by status (proof_found, translation_failed, etc.)

        Response:
        {
            "entries": [...],
            "total": 150,
            "limit": 20,
            "offset": 0
        }
        """
        # Cleanup old entries periodically
        _cleanup_old_history()

        limit = get_clamped_int_param(query_params, "limit", 20, 1, 100)
        offset = get_clamped_int_param(query_params, "offset", 0, 0, 10000)
        status_filter = query_params.get("status", [""])[0] if query_params.get("status") else None

        # Try to get from GovernanceStore first (authoritative)
        store = _get_governance_store()
        if store:
            try:
                records = store.list_verifications(limit=limit + offset + 50)
                all_entries = []
                for rec in records:
                    entry = VerificationHistoryEntry(
                        id=rec.verification_id,
                        claim=rec.claim,
                        claim_type=rec.claim_type,
                        context=rec.context,
                        result=rec.to_dict().get("result", {}),
                        timestamp=rec.timestamp.timestamp()
                        if hasattr(rec.timestamp, "timestamp")
                        else time.time(),
                        proof_tree=rec.to_dict().get("proof_tree"),
                    )
                    all_entries.append(entry)

                # Apply status filter
                if status_filter:
                    all_entries = [
                        e for e in all_entries if e.result.get("status") == status_filter
                    ]

                total = len(all_entries)
                paginated = all_entries[offset : offset + limit]

                return json_response(
                    {
                        "entries": [e.to_dict() for e in paginated],
                        "total": total,
                        "limit": limit,
                        "offset": offset,
                        "source": "persistent",
                    }
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load from GovernanceStore, falling back to in-memory: {e}"
                )

        # Fallback to in-memory cache
        all_entries = list(reversed(_verification_history.values()))

        # Apply status filter
        if status_filter:
            all_entries = [e for e in all_entries if e.result.get("status") == status_filter]

        total = len(all_entries)

        # Apply pagination
        paginated = all_entries[offset : offset + limit]

        return json_response(
            {
                "entries": [e.to_dict() for e in paginated],
                "total": total,
                "limit": limit,
                "offset": offset,
                "source": "in_memory",
            }
        )

    @handle_errors("verification history entry")
    def _handle_get_history_entry(self, path: str) -> HandlerResult:
        """
        GET /api/verify/history/{id} - Get specific verification result.
        GET /api/verify/history/{id}/tree - Get proof tree for visualization.

        Response for /history/{id}:
        {
            "id": "abc123",
            "claim": "...",
            "result": {...},
            "timestamp": 1234567890,
            "proof_tree": [...]
        }

        Response for /history/{id}/tree:
        {
            "nodes": [
                {"id": "root", "type": "claim", "content": "...", "children": ["translation"]},
                {"id": "translation", "type": "translation", "content": "...", "children": ["verification"]},
                ...
            ]
        }
        """
        # Parse the path to extract ID and optional /tree suffix
        parts = path.replace("/api/v1/verify/history/", "").split("/")
        entry_id = parts[0]
        is_tree_request = len(parts) > 1 and parts[1] == "tree"

        if not entry_id:
            return error_response("Entry ID required", 400)

        # Try in-memory cache first
        entry = _verification_history.get(entry_id)

        # If not in memory, try GovernanceStore
        if not entry:
            store = _get_governance_store()
            if store:
                try:
                    rec = store.get_verification(entry_id)
                    if rec:
                        entry = VerificationHistoryEntry(
                            id=rec.verification_id,
                            claim=rec.claim,
                            claim_type=rec.claim_type,
                            context=rec.context,
                            result=rec.to_dict().get("result", {}),
                            timestamp=rec.timestamp.timestamp()
                            if hasattr(rec.timestamp, "timestamp")
                            else time.time(),
                            proof_tree=rec.to_dict().get("proof_tree"),
                        )
                        # Cache in memory for future lookups
                        _verification_history[entry_id] = entry
                except Exception as e:
                    logger.warning(f"Failed to load verification from store: {e}")

        if not entry:
            return error_response(f"Entry not found: {entry_id}", 404)

        if is_tree_request:
            # Return proof tree for visualization
            proof_tree = entry.proof_tree
            if not proof_tree:
                # Try to build it from the result
                proof_tree = _build_proof_tree({**entry.result, "claim": entry.claim})

            if not proof_tree:
                return json_response(
                    {
                        "nodes": [],
                        "message": "No proof tree available for this verification",
                    }
                )

            return json_response({"nodes": proof_tree})

        # Return full entry
        response = entry.to_dict()
        response["result"] = entry.result
        if entry.proof_tree:
            response["proof_tree"] = entry.proof_tree

        return json_response(response)


__all__ = ["FormalVerificationHandler"]
