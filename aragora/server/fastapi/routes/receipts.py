"""
Receipt Endpoints (FastAPI v2).

Provides async receipt management endpoints:
- List receipts with pagination
- Get receipt by ID
- Verify receipt integrity
- Export receipt in various formats (json, markdown, sarif)
"""

from __future__ import annotations

import logging
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
        logger.warning(f"Receipt store not available: {e}")
        raise HTTPException(status_code=503, detail="Receipt store not available")


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
    """
    List all receipts with pagination.

    Returns a paginated list of receipt summaries.
    """
    try:
        # Try list_recent for paginated results
        if hasattr(store, "list_recent"):
            results = store.list_recent(limit=limit, offset=offset, verdict=verdict)
        elif hasattr(store, "list_all"):
            all_receipts = store.list_all()
            if verdict:
                all_receipts = [
                    r for r in all_receipts
                    if (r.get("verdict") if isinstance(r, dict) else getattr(r, "verdict", "")) == verdict
                ]
            results = all_receipts[offset:offset + limit]
        else:
            results = []

        # Get total count
        if hasattr(store, "count"):
            total = store.count(verdict=verdict)
        else:
            total = len(results)

        # Convert to summaries
        receipts = []
        for r in results:
            if isinstance(r, dict):
                data = r.get("data", r)
                summary = ReceiptSummary(
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
            else:
                summary = ReceiptSummary(
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
            receipts.append(summary)

        return ReceiptListResponse(
            receipts=receipts,
            total=total,
            limit=limit,
            offset=offset,
        )

    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception(f"Error listing receipts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list receipts: {e}")


@router.get("/receipts/{receipt_id}", response_model=ReceiptDetail)
async def get_receipt(
    receipt_id: str,
    store=Depends(get_receipt_store),
) -> ReceiptDetail:
    """
    Get receipt by ID.

    Returns full receipt details including findings and verification data.
    """
    try:
        receipt_data = None

        if hasattr(store, "get"):
            receipt_data = store.get(receipt_id)
        elif hasattr(store, "get_by_id"):
            receipt_data = store.get_by_id(receipt_id)

        if not receipt_data:
            raise NotFoundError(f"Receipt {receipt_id} not found")

        # Handle both dict and object responses
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
        logger.exception(f"Error getting receipt {receipt_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get receipt: {e}")


@router.get("/receipts/{receipt_id}/verify", response_model=VerifyResponse)
async def verify_receipt(
    receipt_id: str,
    store=Depends(get_receipt_store),
) -> VerifyResponse:
    """
    Verify receipt integrity.

    Checks the receipt checksum to detect tampering.
    Returns verification result with detailed status.
    """
    try:
        receipt_data = None

        if hasattr(store, "get"):
            receipt_data = store.get(receipt_id)
        elif hasattr(store, "get_by_id"):
            receipt_data = store.get_by_id(receipt_id)

        if not receipt_data:
            raise NotFoundError(f"Receipt {receipt_id} not found")

        # Reconstruct a DecisionReceipt to verify integrity
        try:
            from aragora.export.decision_receipt import DecisionReceipt

            if isinstance(receipt_data, dict):
                data = receipt_data.get("data", receipt_data)
                receipt = DecisionReceipt.from_dict(data)
            elif hasattr(receipt_data, "to_dict"):
                receipt = DecisionReceipt.from_dict(receipt_data.to_dict())
            else:
                # Cannot verify without proper data
                return VerifyResponse(
                    receipt_id=receipt_id,
                    verified=False,
                    integrity_valid=False,
                    checksum_match=False,
                    details={"error": "Cannot reconstruct receipt for verification"},
                )

            integrity_valid = receipt.verify_integrity()
            stored_checksum = (
                receipt_data.get("checksum", "") if isinstance(receipt_data, dict)
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
            logger.warning(f"Receipt verification failed for {receipt_id}: {e}")
            return VerifyResponse(
                receipt_id=receipt_id,
                verified=False,
                integrity_valid=False,
                checksum_match=False,
                details={"error": f"Verification failed: {e}"},
            )

    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception(f"Error verifying receipt {receipt_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to verify receipt: {e}")


@router.get("/receipts/{receipt_id}/export", response_model=ExportResponse)
async def export_receipt(
    receipt_id: str,
    format: ExportFormat = Query(ExportFormat.json, description="Export format"),
    store=Depends(get_receipt_store),
) -> ExportResponse:
    """
    Export receipt in the specified format.

    Supported formats: json, markdown, sarif.
    """
    try:
        receipt_data = None

        if hasattr(store, "get"):
            receipt_data = store.get(receipt_id)
        elif hasattr(store, "get_by_id"):
            receipt_data = store.get_by_id(receipt_id)

        if not receipt_data:
            raise NotFoundError(f"Receipt {receipt_id} not found")

        # Reconstruct a DecisionReceipt for export
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
        logger.exception(f"Error exporting receipt {receipt_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export receipt: {e}")
