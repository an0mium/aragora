"""
Receipt Endpoints (FastAPI v2).

Provides async receipt management endpoints:
- List receipts with pagination
- Get receipt by ID
- Verify receipt integrity
- Export receipt in various formats (json, markdown, sarif)
- Batch verify multiple receipts
- Batch export multiple receipts
- Search receipts by query/date/debate_id
- Receipt statistics
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ..middleware.error_handling import NotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["Receipts"])


# =============================================================================
# Pydantic Models
# =============================================================================


class ExportFormat(str, Enum):
    """Supported export formats."""

    json = "json"
    markdown = "markdown"
    sarif = "sarif"


class ReceiptSummary(BaseModel):
    """Summary of a receipt for list views."""

    receipt_id: str
    gauntlet_id: str
    timestamp: str | None = None
    verdict: str = ""
    confidence: float = 0.0
    risk_level: str = "MEDIUM"
    risk_score: float = 0.0
    robustness_score: float = 0.0
    findings_count: int = 0
    checksum: str = ""

    model_config = {"extra": "allow"}


class ReceiptListResponse(BaseModel):
    """Response for receipt listing."""

    receipts: list[ReceiptSummary]
    total: int
    limit: int
    offset: int


class ReceiptDetail(BaseModel):
    """Full receipt details."""

    receipt_id: str
    gauntlet_id: str
    timestamp: str | None = None
    input_summary: str = ""
    input_type: str = "spec"
    schema_version: str = "1.0"
    verdict: str = ""
    confidence: float = 0.0
    risk_level: str = "MEDIUM"
    risk_score: float = 0.0
    robustness_score: float = 0.0
    coverage_score: float = 0.0
    verification_coverage: float = 0.0
    findings: list[dict[str, Any]] = Field(default_factory=list)
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    mitigations: list[str] = Field(default_factory=list)
    dissenting_views: list[dict[str, Any]] = Field(default_factory=list)
    unresolved_tensions: list[str] = Field(default_factory=list)
    verified_claims: list[dict[str, Any]] = Field(default_factory=list)
    unverified_claims: list[str] = Field(default_factory=list)
    agents_involved: list[str] = Field(default_factory=list)
    rounds_completed: int = 0
    duration_seconds: float = 0.0
    audit_trail_id: str | None = None
    checksum: str = ""

    model_config = {"extra": "allow"}


class VerifyResponse(BaseModel):
    """Response for receipt verification."""

    receipt_id: str
    verified: bool
    integrity_valid: bool
    checksum_match: bool
    details: dict[str, Any] = Field(default_factory=dict)


class ExportResponse(BaseModel):
    """Response for receipt export."""

    receipt_id: str
    format: str
    content: str


class BatchVerifyRequest(BaseModel):
    """Request body for POST /receipts/batch-verify."""

    receipt_ids: list[str] = Field(
        ..., min_length=1, max_length=100, description="Receipt IDs to verify (max 100)"
    )


class BatchVerifyResult(BaseModel):
    """Result for a single receipt in batch verification."""

    receipt_id: str
    verified: bool
    integrity_valid: bool
    error: str | None = None


class BatchVerifyResponse(BaseModel):
    """Response for batch verification."""

    results: list[BatchVerifyResult]
    total: int
    verified_count: int
    failed_count: int


class BatchExportRequest(BaseModel):
    """Request body for POST /receipts/batch-export."""

    receipt_ids: list[str] = Field(
        ..., min_length=1, max_length=100, description="Receipt IDs to export (max 100)"
    )
    format: str = Field("json", description="Export format (json, markdown, sarif)")


class BatchExportItem(BaseModel):
    """A single exported receipt in the batch."""

    receipt_id: str
    format: str
    content: str


class BatchExportResponse(BaseModel):
    """Response for batch export."""

    items: list[BatchExportItem]
    total_requested: int
    exported_count: int
    failed_ids: list[str] = Field(default_factory=list)


class ReceiptSearchResponse(BaseModel):
    """Response for receipt search."""

    receipts: list[ReceiptSummary]
    query: str
    total: int
    limit: int
    offset: int


class ReceiptStatsResponse(BaseModel):
    """Response for receipt statistics."""

    total: int = 0
    verified: int = 0
    by_verdict: dict[str, int] = Field(default_factory=dict)
    by_risk_level: dict[str, int] = Field(default_factory=dict)
    by_framework: dict[str, int] = Field(default_factory=dict)
    generated_at: str = ""


# =============================================================================
# Dependencies
# =============================================================================


async def get_receipt_store(request: Request):
    """Dependency to get the receipt store.

    Tries the storage.receipt_store from app context first,
    then falls back to the global receipt store.
    """
    ctx = getattr(request.app.state, "context", None)
    if ctx:
        store = ctx.get("receipt_store")
        if store:
            return store

    # Fall back to global receipt store
    try:
        from aragora.storage.receipt_store import get_receipt_store as _get_store

        return _get_store()
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        logger.warning("Receipt store not available: %s", e)
        raise HTTPException(status_code=503, detail="Receipt store not available")


# =============================================================================
# Helpers
# =============================================================================


def _to_receipt_summary(r: Any) -> ReceiptSummary:
    """Convert a receipt dict or object to a ReceiptSummary."""
    if isinstance(r, dict):
        data = r.get("data", r)
        return ReceiptSummary(
            receipt_id=data.get("receipt_id", r.get("receipt_id", "")),
            gauntlet_id=data.get("gauntlet_id", r.get("gauntlet_id", "")),
            timestamp=data.get("timestamp", r.get("timestamp")),
            verdict=data.get("verdict", r.get("verdict", "")),
            confidence=data.get("confidence", r.get("confidence", 0.0)),
            risk_level=data.get("risk_level", r.get("risk_level", "MEDIUM")),
            risk_score=data.get("risk_score", r.get("risk_score", 0.0)),
            robustness_score=data.get("robustness_score", r.get("robustness_score", 0.0)),
            findings_count=len(data.get("findings", r.get("findings", []))),
            checksum=data.get("checksum", r.get("checksum", "")),
        )
    return ReceiptSummary(
        receipt_id=getattr(r, "receipt_id", ""),
        gauntlet_id=getattr(r, "gauntlet_id", ""),
        timestamp=str(getattr(r, "timestamp", "")) if hasattr(r, "timestamp") else None,
        verdict=getattr(r, "verdict", ""),
        confidence=getattr(r, "confidence", 0.0),
        risk_level=getattr(r, "risk_level", "MEDIUM"),
        risk_score=getattr(r, "risk_score", 0.0),
        robustness_score=getattr(r, "robustness_score", 0.0),
        findings_count=len(getattr(r, "findings", [])),
        checksum=getattr(r, "checksum", ""),
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/receipts", response_model=ReceiptListResponse)
async def list_receipts(
    request: Request,
    limit: int = Query(50, ge=1, le=100, description="Max results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    verdict: str | None = Query(None, description="Filter by verdict"),
    store=Depends(get_receipt_store),
) -> ReceiptListResponse:
    """List all receipts with pagination."""
    try:
        if hasattr(store, "list_recent"):
            results = store.list_recent(limit=limit, offset=offset, verdict=verdict)
        elif hasattr(store, "list_all"):
            all_receipts = store.list_all()
            if verdict:
                all_receipts = [
                    r
                    for r in all_receipts
                    if (r.get("verdict") if isinstance(r, dict) else getattr(r, "verdict", ""))
                    == verdict
                ]
            results = all_receipts[offset : offset + limit]
        else:
            results = []

        if hasattr(store, "count"):
            total = store.count(verdict=verdict)
        else:
            total = len(results)

        receipts = [_to_receipt_summary(r) for r in results]

        return ReceiptListResponse(receipts=receipts, total=total, limit=limit, offset=offset)

    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error listing receipts: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list receipts")


# --- Fixed-path routes MUST come before {receipt_id} parameterized routes ---


@router.get("/receipts/search", response_model=ReceiptSearchResponse)
async def search_receipts(
    request: Request,
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(50, ge=1, le=100, description="Max results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    verdict: str | None = Query(None, description="Filter by verdict"),
    risk_level: str | None = Query(None, description="Filter by risk level"),
    debate_id: str | None = Query(None, description="Filter by debate ID"),
    date_from: str | None = Query(None, description="Start date (ISO format)"),
    date_to: str | None = Query(None, description="End date (ISO format)"),
    store=Depends(get_receipt_store),
) -> ReceiptSearchResponse:
    """Search receipts by query, date range, debate ID, and other filters."""
    try:
        results: list[Any] = []
        total = 0

        search_kwargs: dict[str, Any] = {"limit": limit, "offset": offset}
        if verdict:
            search_kwargs["verdict"] = verdict
        if risk_level:
            search_kwargs["risk_level"] = risk_level
        if debate_id:
            search_kwargs["debate_id"] = debate_id
        if date_from:
            search_kwargs["date_from"] = date_from
        if date_to:
            search_kwargs["date_to"] = date_to

        if hasattr(store, "search"):
            raw_results = store.search(query=q, **search_kwargs)
            results = list(raw_results)
            if hasattr(store, "search_count"):
                total = store.search_count(query=q, verdict=verdict, risk_level=risk_level)
            else:
                total = len(results)
        elif hasattr(store, "list_all"):
            all_receipts = store.list_all()
            query_lower = q.lower()
            for r in all_receipts:
                data = r if isinstance(r, dict) else (r.to_dict() if hasattr(r, "to_dict") else {})
                if query_lower in str(data).lower():
                    results.append(data)
            total = len(results)
            results = results[offset : offset + limit]

        receipts = [_to_receipt_summary(r) for r in results]

        return ReceiptSearchResponse(
            receipts=receipts, query=q, total=total, limit=limit, offset=offset
        )

    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error searching receipts: %s", e)
        raise HTTPException(status_code=500, detail="Failed to search receipts")


@router.get("/receipts/stats", response_model=ReceiptStatsResponse)
async def get_receipt_stats(
    request: Request,
    store=Depends(get_receipt_store),
) -> ReceiptStatsResponse:
    """Get receipt statistics including counts by verdict and risk level."""
    try:
        stats: dict[str, Any] = {}

        if hasattr(store, "get_stats"):
            raw_stats = store.get_stats()
            if isinstance(raw_stats, dict):
                stats = raw_stats

        return ReceiptStatsResponse(
            total=stats.get("total", stats.get("total_count", 0)),
            verified=stats.get("verified", stats.get("verified_count", 0)),
            by_verdict=stats.get("by_verdict", {}),
            by_risk_level=stats.get("by_risk_level", {}),
            by_framework=stats.get("by_framework", {}),
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting receipt stats: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get receipt stats")


@router.post("/receipts/batch-verify", response_model=BatchVerifyResponse)
async def batch_verify_receipts(
    body: BatchVerifyRequest,
    store=Depends(get_receipt_store),
) -> BatchVerifyResponse:
    """Verify multiple receipts at once (up to 100)."""
    try:
        results: list[BatchVerifyResult] = []
        verified_count = 0

        for rid in body.receipt_ids:
            try:
                receipt_data = None
                if hasattr(store, "get"):
                    receipt_data = store.get(rid)
                elif hasattr(store, "get_by_id"):
                    receipt_data = store.get_by_id(rid)

                if not receipt_data:
                    results.append(
                        BatchVerifyResult(
                            receipt_id=rid,
                            verified=False,
                            integrity_valid=False,
                            error="Receipt not found",
                        )
                    )
                    continue

                try:
                    from aragora.export.decision_receipt import DecisionReceipt

                    if isinstance(receipt_data, dict):
                        data = receipt_data.get("data", receipt_data)
                        receipt = DecisionReceipt.from_dict(data)
                    elif hasattr(receipt_data, "to_dict"):
                        receipt = DecisionReceipt.from_dict(receipt_data.to_dict())
                    else:
                        results.append(
                            BatchVerifyResult(
                                receipt_id=rid,
                                verified=False,
                                integrity_valid=False,
                                error="Cannot reconstruct receipt",
                            )
                        )
                        continue

                    integrity_valid = receipt.verify_integrity()
                    results.append(
                        BatchVerifyResult(
                            receipt_id=rid,
                            verified=integrity_valid,
                            integrity_valid=integrity_valid,
                        )
                    )
                    if integrity_valid:
                        verified_count += 1

                except (ImportError, ValueError, TypeError, KeyError) as e:
                    logger.warning("Batch verify failed for %s: %s", rid, e)
                    results.append(
                        BatchVerifyResult(
                            receipt_id=rid,
                            verified=False,
                            integrity_valid=False,
                            error="Verification failed",
                        )
                    )

            except (RuntimeError, OSError, AttributeError) as e:
                logger.warning("Batch verify error for %s: %s", rid, e)
                results.append(
                    BatchVerifyResult(
                        receipt_id=rid,
                        verified=False,
                        integrity_valid=False,
                        error="Verification error",
                    )
                )

        return BatchVerifyResponse(
            results=results,
            total=len(results),
            verified_count=verified_count,
            failed_count=len(results) - verified_count,
        )

    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error in batch verify: %s", e)
        raise HTTPException(status_code=500, detail="Failed to batch verify receipts")


@router.post("/receipts/batch-export", response_model=BatchExportResponse)
async def batch_export_receipts(
    body: BatchExportRequest,
    store=Depends(get_receipt_store),
) -> BatchExportResponse:
    """Export multiple receipts at once (up to 100)."""
    try:
        items: list[BatchExportItem] = []
        failed_ids: list[str] = []

        export_format = body.format.lower()
        if export_format not in ("json", "markdown", "sarif"):
            raise HTTPException(
                status_code=422,
                detail="Unsupported format. Supported: json, markdown, sarif",
            )

        for rid in body.receipt_ids:
            try:
                receipt_data = None
                if hasattr(store, "get"):
                    receipt_data = store.get(rid)
                elif hasattr(store, "get_by_id"):
                    receipt_data = store.get_by_id(rid)

                if not receipt_data:
                    failed_ids.append(rid)
                    continue

                from aragora.export.decision_receipt import DecisionReceipt

                if isinstance(receipt_data, dict):
                    data = receipt_data.get("data", receipt_data)
                    receipt = DecisionReceipt.from_dict(data)
                elif hasattr(receipt_data, "to_dict"):
                    receipt = DecisionReceipt.from_dict(receipt_data.to_dict())
                else:
                    failed_ids.append(rid)
                    continue

                if export_format == "markdown":
                    content = receipt.to_markdown()
                elif export_format == "sarif":
                    content = receipt.to_sarif_json()
                else:
                    content = receipt.to_json()

                items.append(BatchExportItem(receipt_id=rid, format=export_format, content=content))

            except (ImportError, ValueError, TypeError, KeyError, OSError) as e:
                logger.warning("Batch export failed for %s: %s", rid, e)
                failed_ids.append(rid)

        return BatchExportResponse(
            items=items,
            total_requested=len(body.receipt_ids),
            exported_count=len(items),
            failed_ids=failed_ids,
        )

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error in batch export: %s", e)
        raise HTTPException(status_code=500, detail="Failed to batch export receipts")


# --- Parameterized routes below ---


@router.get("/receipts/{receipt_id}", response_model=ReceiptDetail)
async def get_receipt(
    receipt_id: str,
    store=Depends(get_receipt_store),
) -> ReceiptDetail:
    """Get receipt by ID with full details including findings and verification data."""
    try:
        receipt_data = None

        if hasattr(store, "get"):
            receipt_data = store.get(receipt_id)
        elif hasattr(store, "get_by_id"):
            receipt_data = store.get_by_id(receipt_id)

        if not receipt_data:
            raise NotFoundError(f"Receipt {receipt_id} not found")

        if isinstance(receipt_data, dict):
            data = receipt_data.get("data", receipt_data)
            return ReceiptDetail(
                receipt_id=data.get("receipt_id", receipt_id),
                gauntlet_id=data.get("gauntlet_id", ""),
                timestamp=data.get("timestamp"),
                input_summary=data.get("input_summary", ""),
                input_type=data.get("input_type", "spec"),
                schema_version=data.get("schema_version", "1.0"),
                verdict=data.get("verdict", ""),
                confidence=data.get("confidence", 0.0),
                risk_level=data.get("risk_level", "MEDIUM"),
                risk_score=data.get("risk_score", 0.0),
                robustness_score=data.get("robustness_score", 0.0),
                coverage_score=data.get("coverage_score", 0.0),
                verification_coverage=data.get("verification_coverage", 0.0),
                findings=data.get("findings", []),
                critical_count=data.get("critical_count", 0),
                high_count=data.get("high_count", 0),
                medium_count=data.get("medium_count", 0),
                low_count=data.get("low_count", 0),
                mitigations=data.get("mitigations", []),
                dissenting_views=data.get("dissenting_views", []),
                unresolved_tensions=data.get("unresolved_tensions", []),
                verified_claims=data.get("verified_claims", []),
                unverified_claims=data.get("unverified_claims", []),
                agents_involved=data.get("agents_involved", []),
                rounds_completed=data.get("rounds_completed", 0),
                duration_seconds=data.get("duration_seconds", 0.0),
                audit_trail_id=data.get("audit_trail_id"),
                checksum=data.get("checksum", ""),
            )
        else:
            return ReceiptDetail(
                receipt_id=getattr(receipt_data, "receipt_id", receipt_id),
                gauntlet_id=getattr(receipt_data, "gauntlet_id", ""),
                timestamp=str(getattr(receipt_data, "timestamp", ""))
                if hasattr(receipt_data, "timestamp")
                else None,
                input_summary=getattr(receipt_data, "input_summary", ""),
                input_type=getattr(receipt_data, "input_type", "spec"),
                schema_version=getattr(receipt_data, "schema_version", "1.0"),
                verdict=getattr(receipt_data, "verdict", ""),
                confidence=getattr(receipt_data, "confidence", 0.0),
                risk_level=getattr(receipt_data, "risk_level", "MEDIUM"),
                risk_score=getattr(receipt_data, "risk_score", 0.0),
                robustness_score=getattr(receipt_data, "robustness_score", 0.0),
                coverage_score=getattr(receipt_data, "coverage_score", 0.0),
                verification_coverage=getattr(receipt_data, "verification_coverage", 0.0),
                findings=[
                    f if isinstance(f, dict) else f.__dict__
                    for f in getattr(receipt_data, "findings", [])
                ],
                critical_count=getattr(receipt_data, "critical_count", 0),
                high_count=getattr(receipt_data, "high_count", 0),
                medium_count=getattr(receipt_data, "medium_count", 0),
                low_count=getattr(receipt_data, "low_count", 0),
                mitigations=getattr(receipt_data, "mitigations", []),
                dissenting_views=[
                    d if isinstance(d, dict) else d.__dict__
                    for d in getattr(receipt_data, "dissenting_views", [])
                ],
                unresolved_tensions=getattr(receipt_data, "unresolved_tensions", []),
                verified_claims=[
                    v if isinstance(v, dict) else v.__dict__
                    for v in getattr(receipt_data, "verified_claims", [])
                ],
                unverified_claims=getattr(receipt_data, "unverified_claims", []),
                agents_involved=getattr(receipt_data, "agents_involved", []),
                rounds_completed=getattr(receipt_data, "rounds_completed", 0),
                duration_seconds=getattr(receipt_data, "duration_seconds", 0.0),
                audit_trail_id=getattr(receipt_data, "audit_trail_id", None),
                checksum=getattr(receipt_data, "checksum", ""),
            )

    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting receipt %s: %s", receipt_id, e)
        raise HTTPException(status_code=500, detail="Failed to get receipt")


@router.post("/receipts/{receipt_id}/verify", response_model=VerifyResponse)
async def verify_receipt_post(
    receipt_id: str,
    store=Depends(get_receipt_store),
) -> VerifyResponse:
    """Verify receipt integrity (POST variant)."""
    return await verify_receipt(receipt_id=receipt_id, store=store)


@router.get("/receipts/{receipt_id}/verify", response_model=VerifyResponse)
async def verify_receipt(
    receipt_id: str,
    store=Depends(get_receipt_store),
) -> VerifyResponse:
    """Verify receipt integrity by checking the checksum."""
    try:
        receipt_data = None

        if hasattr(store, "get"):
            receipt_data = store.get(receipt_id)
        elif hasattr(store, "get_by_id"):
            receipt_data = store.get_by_id(receipt_id)

        if not receipt_data:
            raise NotFoundError(f"Receipt {receipt_id} not found")

        try:
            from aragora.export.decision_receipt import DecisionReceipt

            if isinstance(receipt_data, dict):
                data = receipt_data.get("data", receipt_data)
                receipt = DecisionReceipt.from_dict(data)
            elif hasattr(receipt_data, "to_dict"):
                receipt = DecisionReceipt.from_dict(receipt_data.to_dict())
            else:
                return VerifyResponse(
                    receipt_id=receipt_id,
                    verified=False,
                    integrity_valid=False,
                    checksum_match=False,
                    details={"error": "Cannot reconstruct receipt for verification"},
                )

            integrity_valid = receipt.verify_integrity()
            stored_checksum = (
                receipt_data.get("checksum", "")
                if isinstance(receipt_data, dict)
                else getattr(receipt_data, "checksum", "")
            )
            checksum_match = receipt.checksum == stored_checksum if stored_checksum else True

            return VerifyResponse(
                receipt_id=receipt_id,
                verified=integrity_valid and checksum_match,
                integrity_valid=integrity_valid,
                checksum_match=checksum_match,
                details={
                    "computed_checksum": receipt.checksum,
                    "stored_checksum": stored_checksum,
                    "verdict": receipt.verdict,
                    "confidence": receipt.confidence,
                },
            )

        except (ImportError, ValueError, TypeError, KeyError) as e:
            logger.warning("Receipt verification failed for %s: %s", receipt_id, e)
            return VerifyResponse(
                receipt_id=receipt_id,
                verified=False,
                integrity_valid=False,
                checksum_match=False,
                details={"error": "Verification failed"},
            )

    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error verifying receipt %s: %s", receipt_id, e)
        raise HTTPException(status_code=500, detail="Failed to verify receipt")


@router.get("/receipts/{receipt_id}/export", response_model=ExportResponse)
async def export_receipt(
    receipt_id: str,
    format: ExportFormat = Query(ExportFormat.json, description="Export format"),
    store=Depends(get_receipt_store),
) -> ExportResponse:
    """Export receipt in the specified format (json, markdown, sarif)."""
    try:
        receipt_data = None

        if hasattr(store, "get"):
            receipt_data = store.get(receipt_id)
        elif hasattr(store, "get_by_id"):
            receipt_data = store.get_by_id(receipt_id)

        if not receipt_data:
            raise NotFoundError(f"Receipt {receipt_id} not found")

        try:
            from aragora.export.decision_receipt import DecisionReceipt

            if isinstance(receipt_data, dict):
                data = receipt_data.get("data", receipt_data)
                receipt = DecisionReceipt.from_dict(data)
            elif hasattr(receipt_data, "to_dict"):
                receipt = DecisionReceipt.from_dict(receipt_data.to_dict())
            else:
                raise ValueError("Cannot reconstruct receipt for export")

            if format == ExportFormat.markdown:
                content = receipt.to_markdown()
            elif format == ExportFormat.sarif:
                content = receipt.to_sarif_json()
            else:
                content = receipt.to_json()

            return ExportResponse(
                receipt_id=receipt_id,
                format=format.value,
                content=content,
            )

        except ImportError as e:
            raise HTTPException(
                status_code=501,
                detail=f"Export module not available: {e}",
            )

    except NotFoundError:
        raise
    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error exporting receipt %s: %s", receipt_id, e)
        raise HTTPException(status_code=500, detail="Failed to export receipt")
