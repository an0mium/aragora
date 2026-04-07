"""
Legacy v1 compatibility routes for the live audit/receipt UI.

These routes keep the current frontend's audit-trail page working against the
FastAPI app while the UI finishes moving from the old v1 surface to v2.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from aragora.rbac.models import AuthorizationContext

from ..dependencies.auth import require_permission

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Legacy Compatibility"])


def _get_store(request: Request) -> Any | None:
    """Resolve the audit/receipt store from app context or the default factory."""
    context = getattr(request.app.state, "context", None)
    if isinstance(context, dict):
        for key in ("audit_trail_store", "receipt_audit_store"):
            store = context.get(key)
            if store is not None:
                return store

    from aragora.storage.audit_trail_store import get_audit_trail_store

    return get_audit_trail_store()


async def _call_store(store: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    """Call sync or async store methods without blocking the event loop."""
    method = getattr(store, method_name)
    if inspect.iscoroutinefunction(method):
        return await method(*args, **kwargs)

    result = await asyncio.to_thread(method, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _to_plain_dict(item: Any) -> dict[str, Any]:
    """Normalize stored trail/receipt records into plain dicts."""
    if isinstance(item, dict):
        return dict(item)

    if hasattr(item, "to_dict"):
        converted = item.to_dict()
        if isinstance(converted, dict):
            return dict(converted)

    nested = getattr(item, "data", None)
    if isinstance(nested, dict):
        return dict(nested)

    return {}


def _trail_summary(item: Any) -> dict[str, Any]:
    data = _to_plain_dict(item)
    return {
        "trail_id": data.get("trail_id", ""),
        "gauntlet_id": data.get("gauntlet_id"),
        "created_at": data.get("created_at"),
        "verdict": data.get("verdict"),
        "confidence": data.get("confidence"),
        "total_findings": data.get("total_findings"),
        "duration_seconds": data.get("duration_seconds"),
        "checksum": data.get("checksum"),
    }


def _receipt_summary(item: Any) -> dict[str, Any]:
    data = _to_plain_dict(item)
    findings = data.get("findings")
    findings_count = len(findings) if isinstance(findings, list) else data.get("findings_count")
    return {
        "receipt_id": data.get("receipt_id", ""),
        "gauntlet_id": data.get("gauntlet_id"),
        "timestamp": data.get("timestamp"),
        "verdict": data.get("verdict"),
        "confidence": data.get("confidence"),
        "risk_level": data.get("risk_level"),
        "findings_count": findings_count,
        "checksum": data.get("checksum"),
    }


async def _get_trail_or_404(request: Request, trail_id: str) -> dict[str, Any]:
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=404, detail=f"Audit trail not found: {trail_id}")

    trail = await _call_store(store, "get_trail", trail_id)
    data = _to_plain_dict(trail)
    if not data:
        raise HTTPException(status_code=404, detail=f"Audit trail not found: {trail_id}")
    return data


async def _get_receipt_or_404(request: Request, receipt_id: str) -> dict[str, Any]:
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=404, detail=f"Receipt not found: {receipt_id}")

    receipt = await _call_store(store, "get_receipt", receipt_id)
    data = _to_plain_dict(receipt)
    if not data:
        raise HTTPException(status_code=404, detail=f"Receipt not found: {receipt_id}")
    return data


@router.get("/api/v1/audit-trails", include_in_schema=False)
async def list_audit_trails_v1(
    request: Request,
    limit: int = Query(20, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    verdict: str | None = Query(None),
    _auth: AuthorizationContext = Depends(require_permission("audit:read")),
) -> dict[str, Any]:
    del _auth
    store = _get_store(request)
    if store is None:
        return {"trails": [], "total": 0, "limit": limit, "offset": offset}

    trails = await _call_store(store, "list_trails", limit=limit, offset=offset, verdict=verdict)
    total = await _call_store(store, "count_trails", verdict=verdict)
    return {
        "trails": [_trail_summary(item) for item in trails],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/api/v1/audit-trails/{trail_id}", include_in_schema=False)
async def get_audit_trail_v1(
    request: Request,
    trail_id: str,
    _auth: AuthorizationContext = Depends(require_permission("audit:read")),
) -> dict[str, Any]:
    del _auth
    return await _get_trail_or_404(request, trail_id)


@router.post("/api/v1/audit-trails/{trail_id}/verify", include_in_schema=False)
async def verify_audit_trail_v1(
    request: Request,
    trail_id: str,
    _auth: AuthorizationContext = Depends(require_permission("audit:verify")),
) -> dict[str, Any]:
    del _auth
    trail = await _get_trail_or_404(request, trail_id)
    stored_checksum = str(trail.get("checksum", "") or "")

    try:
        from aragora.export.audit_trail import AuditTrail

        audit_trail = AuditTrail.from_json(json.dumps(trail))
        computed_checksum = audit_trail.checksum
        valid = audit_trail.verify_integrity()
        return {
            "trail_id": trail_id,
            "valid": valid,
            "stored_checksum": stored_checksum,
            "computed_checksum": computed_checksum,
            "match": stored_checksum == computed_checksum,
        }
    except (ImportError, ValueError, TypeError, KeyError, AttributeError) as e:
        logger.warning("Audit trail verification failed for %s: %s", trail_id, e)
        return {
            "trail_id": trail_id,
            "valid": False,
            "stored_checksum": stored_checksum,
            "computed_checksum": "",
            "match": False,
            "error": "Audit trail verification failed",
        }


@router.get("/api/v1/receipts", include_in_schema=False)
async def list_receipts_v1(
    request: Request,
    limit: int = Query(20, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    verdict: str | None = Query(None),
    risk_level: str | None = Query(None),
    _auth: AuthorizationContext = Depends(require_permission("audit:receipts.read")),
) -> dict[str, Any]:
    del _auth
    store = _get_store(request)
    if store is None:
        return {"receipts": [], "total": 0, "limit": limit, "offset": offset}

    receipts = await _call_store(
        store,
        "list_receipts",
        limit=limit,
        offset=offset,
        verdict=verdict,
        risk_level=risk_level,
    )
    total = await _call_store(
        store,
        "count_receipts",
        verdict=verdict,
        risk_level=risk_level,
    )
    return {
        "receipts": [_receipt_summary(item) for item in receipts],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/api/v1/receipts/{receipt_id}", include_in_schema=False)
async def get_receipt_v1(
    request: Request,
    receipt_id: str,
    _auth: AuthorizationContext = Depends(require_permission("audit:receipts.read")),
) -> dict[str, Any]:
    del _auth
    return await _get_receipt_or_404(request, receipt_id)


@router.post("/api/v1/receipts/{receipt_id}/verify", include_in_schema=False)
async def verify_receipt_v1(
    request: Request,
    receipt_id: str,
    _auth: AuthorizationContext = Depends(require_permission("audit:receipts.verify")),
) -> dict[str, Any]:
    del _auth
    receipt = await _get_receipt_or_404(request, receipt_id)

    content = json.dumps(
        {
            "receipt_id": receipt.get("receipt_id"),
            "gauntlet_id": receipt.get("gauntlet_id"),
            "verdict": receipt.get("verdict"),
            "confidence": receipt.get("confidence"),
        },
        sort_keys=True,
    )
    computed_checksum = hashlib.sha256(content.encode()).hexdigest()[:16]
    stored_checksum = str(receipt.get("checksum", "") or "")
    return {
        "receipt_id": receipt_id,
        "valid": computed_checksum == stored_checksum,
        "stored_checksum": stored_checksum,
        "computed_checksum": computed_checksum,
        "match": computed_checksum == stored_checksum,
    }
