"""
Explainability API handler.

Provides endpoints for understanding debate decisions:
- GET /api/v1/debates/{id}/explanation - Full decision explanation
- GET /api/v1/debates/{id}/evidence - Evidence chain
- GET /api/v1/debates/{id}/votes/pivots - Vote influence analysis
- GET /api/v1/debates/{id}/counterfactuals - Counterfactual analysis
- GET /api/v1/debates/{id}/summary - Human-readable summary

Batch operations:
- POST /api/v1/explainability/batch - Process multiple debates
- GET /api/v1/explainability/batch/{batch_id}/status - Get batch status
- GET /api/v1/explainability/batch/{batch_id}/results - Get batch results
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    get_string_param,
    json_response,
)
from aragora.server.handlers.utils.rate_limit import rate_limit
from aragora.server.handlers.explainability_store import (
    BatchJob as StoreBatchJob,
    get_batch_job_store,
)

logger = logging.getLogger(__name__)

# Cache for built Decision objects (simple TTL cache)
_decision_cache: Dict[str, Any] = {}
_cache_timestamps: Dict[str, float] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes


def _get_cached_decision(debate_id: str) -> Optional[Any]:
    """Get cached decision if not expired."""
    import time

    if debate_id not in _decision_cache:
        return None

    timestamp = _cache_timestamps.get(debate_id, 0)
    if time.time() - timestamp > CACHE_TTL_SECONDS:
        del _decision_cache[debate_id]
        del _cache_timestamps[debate_id]
        return None

    return _decision_cache[debate_id]


def _cache_decision(debate_id: str, decision: Any) -> None:
    """Cache a decision."""
    _decision_cache[debate_id] = decision
    _cache_timestamps[debate_id] = time.time()

    # Prune old entries
    if len(_decision_cache) > 100:
        oldest = min(_cache_timestamps, key=_cache_timestamps.get)  # type: ignore
        del _decision_cache[oldest]
        del _cache_timestamps[oldest]


# ============================================================================
# Batch Processing Types
# ============================================================================


class BatchStatus(Enum):
    """Status of a batch explainability job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL = "partial"  # Some debates failed
    FAILED = "failed"


@dataclass
class BatchDebateResult:
    """Result for a single debate in a batch."""

    debate_id: str
    status: str  # success, error, not_found
    explanation: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "debate_id": self.debate_id,
            "status": self.status,
            "processing_time_ms": self.processing_time_ms,
        }
        if self.explanation:
            result["explanation"] = self.explanation
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class BatchJob:
    """A batch explainability job."""

    batch_id: str
    debate_ids: List[str]
    status: BatchStatus = BatchStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    results: List[BatchDebateResult] = field(default_factory=list)
    processed_count: int = 0
    options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "status": self.status.value,
            "total_debates": len(self.debate_ids),
            "processed_count": self.processed_count,
            "success_count": sum(1 for r in self.results if r.status == "success"),
            "error_count": sum(1 for r in self.results if r.status in ("error", "not_found")),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress_pct": (
                round((self.processed_count / len(self.debate_ids)) * 100, 1)
                if self.debate_ids
                else 0
            ),
        }


# Batch job storage - uses backend-abstracted store
BATCH_JOB_TTL = 3600  # 1 hour retention
MAX_BATCH_SIZE = 100


def _convert_to_store_job(job: BatchJob) -> StoreBatchJob:
    """Convert local BatchJob to store BatchJob."""
    return StoreBatchJob(
        batch_id=job.batch_id,
        debate_ids=job.debate_ids,
        status=job.status.value,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        results=[r.to_dict() for r in job.results],
        processed_count=job.processed_count,
        options=job.options,
        error=None,
    )


async def _save_batch_job_async(job: BatchJob) -> None:
    """Save batch job to storage (async version for use in async contexts)."""
    store = get_batch_job_store()
    store_job = _convert_to_store_job(job)
    await store.save_job(store_job)


def _save_batch_job(job: BatchJob) -> None:
    """Save batch job to storage (sync wrapper for async store)."""
    store = get_batch_job_store()
    store_job = _convert_to_store_job(job)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create a future and don't wait
            # This is a fallback - prefer using _save_batch_job_async in async contexts
            asyncio.ensure_future(store.save_job(store_job))
            return
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(store.save_job(store_job))


def _get_batch_job(batch_id: str) -> Optional[BatchJob]:
    """Get batch job from storage (sync wrapper for async store)."""
    store = get_batch_job_store()
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    store_job = loop.run_until_complete(store.get_job(batch_id))
    if store_job is None:
        return None
    # Convert store BatchJob to local BatchJob
    job = BatchJob(
        batch_id=store_job.batch_id,
        debate_ids=store_job.debate_ids,
        status=BatchStatus(store_job.status),
        created_at=store_job.created_at,
        started_at=store_job.started_at,
        completed_at=store_job.completed_at,
        processed_count=store_job.processed_count,
        options=store_job.options,
    )
    # Restore results
    for r in store_job.results:
        job.results.append(
            BatchDebateResult(
                debate_id=r["debate_id"],
                status=r["status"],
                explanation=r.get("explanation"),
                error=r.get("error"),
                processing_time_ms=r.get("processing_time_ms", 0),
            )
        )
    return job


class ExplainabilityHandler(BaseHandler):
    """Handler for debate explainability endpoints."""

    # API v1 routes
    ROUTES = [
        "/api/v1/debates/*/explanation",
        "/api/v1/debates/*/evidence",
        "/api/v1/debates/*/votes/pivots",
        "/api/v1/debates/*/counterfactuals",
        "/api/v1/debates/*/summary",
        "/api/v1/explain/*",
        # Batch endpoints
        "/api/v1/explainability/batch",
        "/api/v1/explainability/batch/*/status",
        "/api/v1/explainability/batch/*/results",
        # Compare endpoint
        "/api/v1/explainability/compare",
        # Legacy routes (deprecated)
        "/api/v1/debates/*/explanation",
        "/api/v1/explain/*",
    ]

    def __init__(self, server_context: Optional[Dict] = None):
        """Initialize with server context for richer explanations."""
        super().__init__(server_context)  # type: ignore[arg-type]
        self.elo_system = (server_context or {}).get("elo_system")
        self.calibration_tracker = None
        # Try to get calibration tracker from global
        try:
            from aragora.ranking.calibration import get_calibration_tracker

            self.calibration_tracker = get_calibration_tracker()
        except (ImportError, Exception):
            pass

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        # Batch endpoints support POST and GET
        if path == "/api/v1/explainability/batch":
            return method == "POST"
        if path.startswith("/api/v1/explainability/batch/") and path.endswith("/status"):
            return method == "GET"
        if path.startswith("/api/v1/explainability/batch/") and path.endswith("/results"):
            return method == "GET"

        # Compare endpoint supports POST
        if path == "/api/v1/explainability/compare":
            return method == "POST"

        # Single debate endpoints only support GET
        if method != "GET":
            return False

        # Check versioned routes
        if path.startswith("/api/v1/debates/") and any(
            path.endswith(suffix)
            for suffix in [
                "/explanation",
                "/evidence",
                "/votes/pivots",
                "/counterfactuals",
                "/summary",
            ]
        ):
            return True

        # Check explain shortcut
        if path.startswith("/api/v1/explain/"):
            return True

        # Legacy routes
        if path.startswith("/api/v1/debates/") and path.endswith("/explanation"):
            return True
        if path.startswith("/api/v1/explain/"):
            return True

        return False

    def _is_legacy_route(self, path: str) -> bool:
        """Check if this is a legacy (non-versioned) route."""
        return not path.startswith("/api/v1/")

    @rate_limit(rpm=60)
    def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route explainability requests."""
        # Handle batch endpoints first
        if path == "/api/v1/explainability/batch":
            return self._handle_batch_create(handler)
        if path.startswith("/api/v1/explainability/batch/") and path.endswith("/status"):
            batch_id = path.split("/")[-2]
            return self._handle_batch_status(batch_id)
        if path.startswith("/api/v1/explainability/batch/") and path.endswith("/results"):
            batch_id = path.split("/")[-2]
            return self._handle_batch_results(batch_id, query_params)

        # Handle compare endpoint
        if path == "/api/v1/explainability/compare":
            return self._handle_compare(handler)

        # Add deprecation headers for legacy routes
        is_legacy = self._is_legacy_route(path)

        # Normalize path
        if path.startswith("/api/v1/"):
            normalized = path[8:]  # Remove /api/v1/
        else:
            normalized = path[5:]  # Remove /api/

        # Extract debate_id
        parts = normalized.split("/")

        # Handle /explain/{id} shortcut
        if parts[0] == "explain" and len(parts) >= 2:
            debate_id = parts[1]
            return self._handle_full_explanation(debate_id, query_params, is_legacy)

        # Handle /debates/{id}/...
        if parts[0] == "debates" and len(parts) >= 3:
            debate_id = parts[1]
            endpoint = "/".join(parts[2:])

            if endpoint == "explanation":
                return self._handle_full_explanation(debate_id, query_params, is_legacy)
            elif endpoint == "evidence":
                return self._handle_evidence(debate_id, query_params, is_legacy)
            elif endpoint == "votes/pivots":
                return self._handle_vote_pivots(debate_id, query_params, is_legacy)
            elif endpoint == "counterfactuals":
                return self._handle_counterfactuals(debate_id, query_params, is_legacy)
            elif endpoint == "summary":
                return self._handle_summary(debate_id, query_params, is_legacy)

        return error_response("Invalid explainability endpoint", 400)

    def _add_headers(self, result: HandlerResult, is_legacy: bool) -> HandlerResult:
        """Add version and deprecation headers."""
        if result.headers is None:
            result.headers = {}

        result.headers["X-API-Version"] = "v1"

        if is_legacy:
            result.headers["Deprecation"] = "true"
            result.headers["Sunset"] = "2026-06-01"

        return result

    async def _get_or_build_decision(self, debate_id: str) -> Optional[Any]:
        """Get decision from cache or build it."""
        # Check cache
        decision = _get_cached_decision(debate_id)
        if decision:
            return decision

        # Get debate result from storage
        try:
            from aragora.server.storage import get_debates_db

            db = get_debates_db()
            if not db:
                return None

            debate_data = db.get(debate_id)
            if not debate_data:
                return None

            # Build decision with available tracking systems
            from aragora.explainability import ExplanationBuilder

            builder = ExplanationBuilder(
                elo_system=self.elo_system,
                calibration_tracker=self.calibration_tracker,
            )

            # Convert dict to simple object for builder
            class ResultProxy:
                def __init__(self, data: Dict[str, Any]):
                    for k, v in data.items():
                        setattr(self, k, v)

            result = ResultProxy(debate_data)
            result.id = debate_id  # type: ignore[attr-defined]  # Ensure id is set

            decision = await builder.build(result)
            _cache_decision(debate_id, decision)

            return decision

        except Exception as e:
            logger.error(f"Failed to build decision for {debate_id}: {e}")
            return None

    def _handle_full_explanation(
        self, debate_id: str, query_params: Dict[str, Any], is_legacy: bool
    ) -> HandlerResult:
        """Handle full explanation request."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            decision = loop.run_until_complete(self._get_or_build_decision(debate_id))

            if not decision:
                return error_response(f"Debate not found: {debate_id}", 404)

            # Get format preference
            format_type = get_string_param(query_params, "format", "json")

            if format_type == "summary":
                from aragora.explainability import ExplanationBuilder

                builder = ExplanationBuilder()
                summary = builder.generate_summary(decision)

                result = HandlerResult(
                    status_code=200,
                    content_type="text/markdown",
                    body=summary.encode("utf-8"),
                )
            else:
                result = json_response(decision.to_dict())

            return self._add_headers(result, is_legacy)

        except Exception as e:
            logger.error(f"Explanation error for {debate_id}: {e}")
            return error_response(f"Failed to generate explanation: {str(e)[:100]}", 500)

    def _handle_evidence(
        self, debate_id: str, query_params: Dict[str, Any], is_legacy: bool
    ) -> HandlerResult:
        """Handle evidence chain request."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            decision = loop.run_until_complete(self._get_or_build_decision(debate_id))

            if not decision:
                return error_response(f"Debate not found: {debate_id}", 404)

            # Get filter params
            limit = int(get_string_param(query_params, "limit", "20"))
            min_relevance = float(get_string_param(query_params, "min_relevance", "0.0"))

            evidence = decision.evidence_chain
            if min_relevance > 0:
                evidence = [e for e in evidence if e.relevance_score >= min_relevance]

            evidence = sorted(evidence, key=lambda e: e.relevance_score, reverse=True)[:limit]

            result = json_response(
                {
                    "debate_id": debate_id,
                    "evidence_count": len(evidence),
                    "evidence_quality_score": decision.evidence_quality_score,
                    "evidence": [e.to_dict() for e in evidence],
                }
            )

            return self._add_headers(result, is_legacy)

        except Exception as e:
            logger.error(f"Evidence error for {debate_id}: {e}")
            return error_response(f"Failed to get evidence: {str(e)[:100]}", 500)

    def _handle_vote_pivots(
        self, debate_id: str, query_params: Dict[str, Any], is_legacy: bool
    ) -> HandlerResult:
        """Handle vote pivot analysis request."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            decision = loop.run_until_complete(self._get_or_build_decision(debate_id))

            if not decision:
                return error_response(f"Debate not found: {debate_id}", 404)

            # Get filter params
            min_influence = float(get_string_param(query_params, "min_influence", "0.0"))

            pivots = decision.vote_pivots
            if min_influence > 0:
                pivots = [p for p in pivots if p.influence_score >= min_influence]

            result = json_response(
                {
                    "debate_id": debate_id,
                    "total_votes": len(decision.vote_pivots),
                    "pivotal_votes": len(pivots),
                    "agent_agreement_score": decision.agent_agreement_score,
                    "votes": [p.to_dict() for p in pivots],
                }
            )

            return self._add_headers(result, is_legacy)

        except Exception as e:
            logger.error(f"Vote pivot error for {debate_id}: {e}")
            return error_response(f"Failed to get vote pivots: {str(e)[:100]}", 500)

    def _handle_counterfactuals(
        self, debate_id: str, query_params: Dict[str, Any], is_legacy: bool
    ) -> HandlerResult:
        """Handle counterfactual analysis request."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            decision = loop.run_until_complete(self._get_or_build_decision(debate_id))

            if not decision:
                return error_response(f"Debate not found: {debate_id}", 404)

            # Get filter params
            min_sensitivity = float(get_string_param(query_params, "min_sensitivity", "0.0"))

            counterfactuals = decision.counterfactuals
            if min_sensitivity > 0:
                counterfactuals = [c for c in counterfactuals if c.sensitivity >= min_sensitivity]

            result = json_response(
                {
                    "debate_id": debate_id,
                    "counterfactual_count": len(counterfactuals),
                    "counterfactuals": [c.to_dict() for c in counterfactuals],
                }
            )

            return self._add_headers(result, is_legacy)

        except Exception as e:
            logger.error(f"Counterfactual error for {debate_id}: {e}")
            return error_response(f"Failed to get counterfactuals: {str(e)[:100]}", 500)

    def _handle_summary(
        self, debate_id: str, query_params: Dict[str, Any], is_legacy: bool
    ) -> HandlerResult:
        """Handle human-readable summary request."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            decision = loop.run_until_complete(self._get_or_build_decision(debate_id))

            if not decision:
                return error_response(f"Debate not found: {debate_id}", 404)

            from aragora.explainability import ExplanationBuilder

            builder = ExplanationBuilder()
            summary = builder.generate_summary(decision)

            # Get format preference
            format_type = get_string_param(query_params, "format", "markdown")

            if format_type == "json":
                result = json_response(
                    {
                        "debate_id": debate_id,
                        "summary": summary,
                        "confidence": decision.confidence,
                        "consensus_reached": decision.consensus_reached,
                    }
                )
            elif format_type == "html":
                import markdown

                html_content = f"""
<!DOCTYPE html>
<html>
<head><title>Decision Summary - {debate_id}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 2rem auto; padding: 1rem; }}
h2 {{ color: #333; }}
h3 {{ color: #666; }}
</style>
</head>
<body>
{markdown.markdown(summary)}
</body>
</html>
"""
                result = HandlerResult(
                    status_code=200,
                    content_type="text/html",
                    body=html_content.encode("utf-8"),
                )
            else:
                result = HandlerResult(
                    status_code=200,
                    content_type="text/markdown",
                    body=summary.encode("utf-8"),
                )

            return self._add_headers(result, is_legacy)

        except ImportError:
            # markdown not available, return plain text
            from aragora.explainability import ExplanationBuilder

            builder = ExplanationBuilder()
            summary = builder.generate_summary(decision)

            result = HandlerResult(
                status_code=200,
                content_type="text/plain",
                body=summary.encode("utf-8"),
            )
            return self._add_headers(result, is_legacy)

        except Exception as e:
            logger.error(f"Summary error for {debate_id}: {e}")
            return error_response(f"Failed to generate summary: {str(e)[:100]}", 500)

    # ========================================================================
    # Batch Processing Methods
    # ========================================================================

    @rate_limit(rpm=20)
    def _handle_batch_create(self, handler: Any) -> HandlerResult:
        """Create a new batch explainability job.

        Request body:
        {
            "debate_ids": ["debate-1", "debate-2", ...],
            "options": {
                "include_evidence": true,
                "include_counterfactuals": false,
                "include_vote_pivots": false,
                "format": "full"  # full, summary, minimal
            }
        }
        """
        try:
            # Parse request body
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length == 0:
                return error_response("Request body required", 400)

            body = handler.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        debate_ids = data.get("debate_ids", [])
        if not debate_ids:
            return error_response("debate_ids array required", 400)

        if not isinstance(debate_ids, list):
            return error_response("debate_ids must be an array", 400)

        if len(debate_ids) > MAX_BATCH_SIZE:
            return error_response(f"Maximum batch size is {MAX_BATCH_SIZE}", 400)

        # Validate IDs are strings
        debate_ids = [str(d) for d in debate_ids]

        # Create batch job
        batch_id = f"batch-{uuid.uuid4().hex[:12]}"
        options = data.get("options", {})

        job = BatchJob(
            batch_id=batch_id,
            debate_ids=debate_ids,
            options=options,
        )
        _save_batch_job(job)

        # Start processing in background
        self._start_batch_processing(job)

        return json_response(
            {
                "batch_id": batch_id,
                "status": job.status.value,
                "total_debates": len(debate_ids),
                "status_url": f"/api/v1/explainability/batch/{batch_id}/status",
                "results_url": f"/api/v1/explainability/batch/{batch_id}/results",
            },
            status=202,
        )

    def _start_batch_processing(self, job: BatchJob) -> None:
        """Start processing batch job asynchronously."""
        import threading

        def process():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._process_batch(job))
            except Exception as e:
                logger.error(f"Batch processing error for {job.batch_id}: {e}")
                job.status = BatchStatus.FAILED
            finally:
                loop.close()

        thread = threading.Thread(target=process, daemon=True)
        thread.start()

    async def _process_batch(self, job: BatchJob) -> None:
        """Process all debates in the batch."""
        job.status = BatchStatus.PROCESSING
        job.started_at = time.time()
        await _save_batch_job_async(job)  # Persist initial processing state

        options = job.options
        include_evidence = options.get("include_evidence", True)
        include_counterfactuals = options.get("include_counterfactuals", False)
        include_vote_pivots = options.get("include_vote_pivots", False)
        format_type = options.get("format", "full")

        # Process debates (could parallelize with asyncio.gather for performance)
        for debate_id in job.debate_ids:
            start_time = time.time()
            try:
                decision = await self._get_or_build_decision(debate_id)

                if decision is None:
                    job.results.append(
                        BatchDebateResult(
                            debate_id=debate_id,
                            status="not_found",
                            error=f"Debate not found: {debate_id}",
                            processing_time_ms=(time.time() - start_time) * 1000,
                        )
                    )
                else:
                    # Build explanation based on options
                    explanation = self._build_explanation_dict(
                        decision,
                        include_evidence=include_evidence,
                        include_counterfactuals=include_counterfactuals,
                        include_vote_pivots=include_vote_pivots,
                        format_type=format_type,
                    )

                    job.results.append(
                        BatchDebateResult(
                            debate_id=debate_id,
                            status="success",
                            explanation=explanation,
                            processing_time_ms=(time.time() - start_time) * 1000,
                        )
                    )

            except Exception as e:
                logger.error(f"Error processing {debate_id} in batch: {e}")
                job.results.append(
                    BatchDebateResult(
                        debate_id=debate_id,
                        status="error",
                        error=str(e)[:200],
                        processing_time_ms=(time.time() - start_time) * 1000,
                    )
                )

            job.processed_count += 1
            # Persist progress periodically (every 5 items or when done)
            if job.processed_count % 5 == 0 or job.processed_count == len(job.debate_ids):
                await _save_batch_job_async(job)

        # Set final status
        job.completed_at = time.time()
        error_count = sum(1 for r in job.results if r.status != "success")

        if error_count == 0:
            job.status = BatchStatus.COMPLETED
        elif error_count == len(job.debate_ids):
            job.status = BatchStatus.FAILED
        else:
            job.status = BatchStatus.PARTIAL

        # Persist final state
        await _save_batch_job_async(job)

        # Emit WebSocket event
        try:
            from aragora.events.types import StreamEventType
            from aragora.events.emitter import global_emitter

            global_emitter().emit(
                StreamEventType.EXPLAINABILITY_COMPLETE.value,
                {
                    "batch_id": job.batch_id,
                    "status": job.status.value,
                    "total": len(job.debate_ids),
                    "success_count": len(job.debate_ids) - error_count,
                    "error_count": error_count,
                },
            )
        except Exception:
            pass  # WebSocket events are optional

    def _build_explanation_dict(
        self,
        decision: Any,
        include_evidence: bool = True,
        include_counterfactuals: bool = False,
        include_vote_pivots: bool = False,
        format_type: str = "full",
    ) -> Dict[str, Any]:
        """Build explanation dictionary based on options."""
        if format_type == "minimal":
            return {
                "debate_id": getattr(decision, "debate_id", None),
                "confidence": decision.confidence,
                "consensus_reached": decision.consensus_reached,
                "primary_factors": (
                    [
                        {"name": f.name, "contribution": f.contribution}
                        for f in decision.contributing_factors[:3]
                    ]
                    if hasattr(decision, "contributing_factors")
                    else []
                ),
            }

        if format_type == "summary":
            from aragora.explainability import ExplanationBuilder

            builder = ExplanationBuilder()
            return {
                "debate_id": getattr(decision, "debate_id", None),
                "summary": builder.generate_summary(decision),
                "confidence": decision.confidence,
            }

        # Full format
        result = decision.to_dict() if hasattr(decision, "to_dict") else {}

        if not include_evidence and "evidence_chain" in result:
            del result["evidence_chain"]

        if not include_counterfactuals and "counterfactuals" in result:
            del result["counterfactuals"]

        if not include_vote_pivots and "vote_pivots" in result:
            del result["vote_pivots"]

        return result

    def _handle_batch_status(self, batch_id: str) -> HandlerResult:
        """Get status of a batch job."""
        job = _get_batch_job(batch_id)
        if not job:
            return error_response(f"Batch job not found: {batch_id}", 404)

        return json_response(job.to_dict())

    def _handle_batch_results(self, batch_id: str, query_params: Dict[str, Any]) -> HandlerResult:
        """Get results of a completed batch job."""
        job = _get_batch_job(batch_id)
        if not job:
            return error_response(f"Batch job not found: {batch_id}", 404)

        # Allow fetching partial results while processing
        include_partial = (
            get_string_param(query_params, "include_partial", "false").lower() == "true"
        )

        if job.status == BatchStatus.PENDING:
            return error_response("Batch job not yet started", 202)

        if job.status == BatchStatus.PROCESSING and not include_partial:
            return json_response(  # type: ignore[call-arg]
                {
                    **job.to_dict(),
                    "message": "Batch still processing. Use ?include_partial=true for partial results.",
                },
                status_code=202,
            )

        # Pagination
        offset = int(get_string_param(query_params, "offset", "0"))
        limit = int(get_string_param(query_params, "limit", "50"))
        limit = min(limit, 100)  # Cap at 100

        paginated_results = job.results[offset : offset + limit]

        return json_response(
            {
                **job.to_dict(),
                "results": [r.to_dict() for r in paginated_results],
                "pagination": {
                    "offset": offset,
                    "limit": limit,
                    "total": len(job.results),
                    "has_more": offset + limit < len(job.results),
                },
            }
        )

    # ========================================================================
    # Compare Explanations
    # ========================================================================

    @rate_limit(rpm=30)
    def _handle_compare(self, handler: Any) -> HandlerResult:
        """Compare explanations between multiple debates.

        Request body:
        {
            "debate_ids": ["debate-1", "debate-2"],
            "compare_fields": ["contributing_factors", "evidence_quality", "confidence"]
        }
        """
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length == 0:
                return error_response("Request body required", 400)

            body = handler.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        debate_ids = data.get("debate_ids", [])
        if len(debate_ids) < 2:
            return error_response("At least 2 debate_ids required for comparison", 400)

        if len(debate_ids) > 10:
            return error_response("Maximum 10 debates can be compared at once", 400)

        compare_fields = data.get(
            "compare_fields",
            ["confidence", "consensus_reached", "contributing_factors", "evidence_quality"],
        )

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            # Fetch all decisions
            debates = {}
            for debate_id in debate_ids:
                decision = loop.run_until_complete(self._get_or_build_decision(debate_id))
                if decision:
                    debates[debate_id] = decision

            if len(debates) < 2:
                return error_response("Need at least 2 valid debates to compare", 404)

            # Build comparison
            comparison = {
                "debates_compared": list(debates.keys()),
                "comparison": {},
            }

            if "confidence" in compare_fields:
                comparison["comparison"]["confidence"] = {  # type: ignore[index]
                    debate_id: decision.confidence for debate_id, decision in debates.items()
                }
                confidences = list(comparison["comparison"]["confidence"].values())  # type: ignore[index]
                comparison["comparison"]["confidence_stats"] = {  # type: ignore[index]
                    "min": min(confidences),
                    "max": max(confidences),
                    "avg": sum(confidences) / len(confidences),
                    "spread": max(confidences) - min(confidences),
                }

            if "consensus_reached" in compare_fields:
                comparison["comparison"]["consensus_reached"] = {  # type: ignore[index]
                    debate_id: decision.consensus_reached for debate_id, decision in debates.items()
                }
                comparison["comparison"]["consensus_agreement"] = (  # type: ignore[index]
                    len(set(d.consensus_reached for d in debates.values())) == 1
                )

            if "contributing_factors" in compare_fields:
                factor_names: dict[str, dict[str, float]] = {}
                for debate_id, decision in debates.items():
                    if hasattr(decision, "contributing_factors"):
                        for f in decision.contributing_factors[:5]:
                            if f.name not in factor_names:
                                factor_names[f.name] = {}
                            factor_names[f.name][debate_id] = f.contribution

                comparison["comparison"]["contributing_factors"] = factor_names  # type: ignore[index]
                comparison["comparison"]["common_factors"] = [  # type: ignore[index]
                    name for name, vals in factor_names.items() if len(vals) == len(debates)
                ]

            if "evidence_quality" in compare_fields:
                comparison["comparison"]["evidence_quality"] = {  # type: ignore[index]
                    debate_id: getattr(decision, "evidence_quality_score", None)
                    for debate_id, decision in debates.items()
                }

            return json_response(comparison)

        except Exception as e:
            logger.error(f"Compare error: {e}")
            return error_response(f"Failed to compare debates: {str(e)[:100]}", 500)


# Handler factory
_explainability_handler: Optional["ExplainabilityHandler"] = None


def get_explainability_handler(server_context: Optional[Dict] = None) -> "ExplainabilityHandler":
    """Get or create the explainability handler instance."""
    global _explainability_handler
    if _explainability_handler is None:
        if server_context is None:
            server_context = {}
        _explainability_handler = ExplainabilityHandler(server_context)
    return _explainability_handler


__all__ = [
    "ExplainabilityHandler",
    "get_explainability_handler",
    "BatchStatus",
    "BatchJob",
    "BatchDebateResult",
]
