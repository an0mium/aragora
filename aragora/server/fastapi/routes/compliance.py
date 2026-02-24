"""
Compliance Endpoints (FastAPI v2).

Migrated from: aragora/server/handlers/compliance/ (aiohttp handler)

Provides async compliance management endpoints:
- GET  /api/v2/compliance/status             - Compliance framework status
- GET  /api/v2/compliance/policies           - List policies
- POST /api/v2/compliance/artifacts/generate - Generate compliance artifacts
- GET  /api/v2/compliance/audit-log          - Query audit log

Migration Notes:
    This module replaces the legacy compliance handler endpoints with native
    FastAPI routes. Key improvements:
    - Pydantic request/response models with automatic validation
    - FastAPI dependency injection for auth and storage
    - Proper HTTP status codes (422 for validation, 404 for not found)
    - OpenAPI schema auto-generation
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from aragora.rbac.models import AuthorizationContext

from ..dependencies.auth import require_permission
from ..middleware.error_handling import NotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["Compliance"])


# =============================================================================
# Pydantic Models
# =============================================================================


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""

    soc2 = "soc2"
    gdpr = "gdpr"
    hipaa = "hipaa"
    eu_ai_act = "eu_ai_act"
    sox = "sox"


class ControlStatus(BaseModel):
    """Status of a single compliance control."""

    control_id: str
    name: str
    description: str = ""
    status: str = "not_assessed"
    evidence_count: int = 0
    last_assessed: str | None = None

    model_config = {"extra": "allow"}


class ComplianceStatusResponse(BaseModel):
    """Response for compliance framework status."""

    framework: str = "soc2"
    overall_status: str = "not_assessed"
    controls_total: int = 0
    controls_passing: int = 0
    controls_failing: int = 0
    controls_not_assessed: int = 0
    coverage_percent: float = 0.0
    controls: list[ControlStatus] = Field(default_factory=list)
    last_assessment: str | None = None


class PolicySummary(BaseModel):
    """Summary of a compliance policy."""

    id: str
    name: str
    description: str = ""
    framework: str = ""
    status: str = "active"
    enforcement: str = "enforced"
    created_at: str | None = None
    updated_at: str | None = None

    model_config = {"extra": "allow"}


class PolicyListResponse(BaseModel):
    """Response for policy listing."""

    policies: list[PolicySummary]
    total: int
    limit: int
    offset: int


class GenerateArtifactRequest(BaseModel):
    """Request body for POST /compliance/artifacts/generate."""

    framework: str = Field(
        ...,
        description="Compliance framework (soc2, gdpr, hipaa, eu_ai_act, sox)",
    )
    artifact_type: str = Field(
        "full_bundle",
        description="Artifact type to generate (full_bundle, technical_doc, risk_assessment)",
    )
    debate_ids: list[str] = Field(
        default_factory=list,
        description="Debate IDs to include as evidence",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional generation options",
    )


class GenerateArtifactResponse(BaseModel):
    """Response for POST /compliance/artifacts/generate."""

    success: bool
    artifact_id: str
    framework: str
    artifact_type: str
    content: dict[str, Any] = Field(default_factory=dict)
    integrity_hash: str = ""


class AuditLogEntry(BaseModel):
    """A single audit log entry."""

    id: str
    timestamp: str
    action: str
    actor: str = ""
    resource_type: str = ""
    resource_id: str = ""
    details: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class AuditLogResponse(BaseModel):
    """Response for audit log query."""

    entries: list[AuditLogEntry]
    total: int
    limit: int
    offset: int


# =============================================================================
# Dependencies
# =============================================================================


async def get_compliance_framework(request: Request):
    """Dependency to get the compliance framework from app state."""
    ctx = getattr(request.app.state, "context", None)
    if ctx:
        fw = ctx.get("compliance_framework")
        if fw:
            return fw

    # Fall back to global compliance framework
    try:
        from aragora.compliance.framework import get_compliance_framework as _get_fw

        return _get_fw()
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        logger.debug("Compliance framework not available: %s", e)
        return None


async def get_audit_store(request: Request):
    """Dependency to get the audit log store from app state."""
    ctx = getattr(request.app.state, "context", None)
    if ctx:
        store = ctx.get("audit_store")
        if store:
            return store

    # Fall back to global audit store
    try:
        from aragora.audit.log import get_audit_store as _get_store

        return _get_store()
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        logger.debug("Audit store not available: %s", e)
        return None


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/compliance/status", response_model=ComplianceStatusResponse)
async def get_compliance_status(
    request: Request,
    framework: str = Query("soc2", description="Compliance framework to check"),
    fw=Depends(get_compliance_framework),
) -> ComplianceStatusResponse:
    """
    Get compliance framework status.

    Returns the current compliance status including control assessments
    and coverage metrics.
    """
    if not fw:
        return ComplianceStatusResponse(
            framework=framework,
            overall_status="not_configured",
        )

    try:
        status_data: dict[str, Any] = {}

        if hasattr(fw, "get_status"):
            raw_status = fw.get_status(framework=framework)
            if isinstance(raw_status, dict):
                status_data = raw_status
        elif hasattr(fw, "assess"):
            raw_status = fw.assess(framework=framework)
            if isinstance(raw_status, dict):
                status_data = raw_status

        # Extract controls
        controls: list[ControlStatus] = []
        raw_controls = status_data.get("controls", [])
        for ctrl in raw_controls:
            if isinstance(ctrl, dict):
                controls.append(ControlStatus(
                    control_id=ctrl.get("control_id", ctrl.get("id", "")),
                    name=ctrl.get("name", ""),
                    description=ctrl.get("description", ""),
                    status=ctrl.get("status", "not_assessed"),
                    evidence_count=ctrl.get("evidence_count", 0),
                    last_assessed=ctrl.get("last_assessed"),
                ))
            else:
                controls.append(ControlStatus(
                    control_id=getattr(ctrl, "control_id", getattr(ctrl, "id", "")),
                    name=getattr(ctrl, "name", ""),
                    description=getattr(ctrl, "description", ""),
                    status=getattr(ctrl, "status", "not_assessed"),
                    evidence_count=getattr(ctrl, "evidence_count", 0),
                    last_assessed=getattr(ctrl, "last_assessed", None),
                ))

        passing = sum(1 for c in controls if c.status == "passing")
        failing = sum(1 for c in controls if c.status == "failing")
        not_assessed = sum(1 for c in controls if c.status == "not_assessed")
        total = len(controls)
        coverage = passing / total * 100 if total > 0 else 0.0

        overall = status_data.get("overall_status", "not_assessed")
        if not overall or overall == "not_assessed":
            if failing > 0:
                overall = "non_compliant"
            elif passing == total and total > 0:
                overall = "compliant"
            elif passing > 0:
                overall = "partially_compliant"

        return ComplianceStatusResponse(
            framework=framework,
            overall_status=overall,
            controls_total=total,
            controls_passing=passing,
            controls_failing=failing,
            controls_not_assessed=not_assessed,
            coverage_percent=round(coverage, 1),
            controls=controls,
            last_assessment=status_data.get("last_assessment"),
        )

    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting compliance status: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get compliance status")


@router.get("/compliance/policies", response_model=PolicyListResponse)
async def list_policies(
    request: Request,
    limit: int = Query(50, ge=1, le=100, description="Max results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    framework: str | None = Query(None, description="Filter by framework"),
    status: str | None = Query(None, description="Filter by status"),
    fw=Depends(get_compliance_framework),
) -> PolicyListResponse:
    """
    List compliance policies.

    Returns a paginated list of compliance policies with optional filters.
    """
    if not fw:
        return PolicyListResponse(policies=[], total=0, limit=limit, offset=offset)

    try:
        policies_raw: list[Any] = []

        if hasattr(fw, "list_policies"):
            policies_raw = fw.list_policies(
                limit=limit, offset=offset, framework=framework, status=status,
            )
        elif hasattr(fw, "get_policies"):
            all_policies = fw.get_policies()
            if framework:
                all_policies = [
                    p for p in all_policies
                    if (p.get("framework") if isinstance(p, dict) else getattr(p, "framework", ""))
                    == framework
                ]
            if status:
                all_policies = [
                    p for p in all_policies
                    if (p.get("status") if isinstance(p, dict) else getattr(p, "status", ""))
                    == status
                ]
            policies_raw = all_policies[offset: offset + limit]

        # Get total count
        if hasattr(fw, "count_policies"):
            total = fw.count_policies(framework=framework, status=status)
        else:
            total = len(policies_raw)

        # Convert to summaries
        policies: list[PolicySummary] = []
        for p in policies_raw:
            if isinstance(p, dict):
                policies.append(PolicySummary(
                    id=p.get("id", p.get("policy_id", "")),
                    name=p.get("name", ""),
                    description=p.get("description", ""),
                    framework=p.get("framework", ""),
                    status=p.get("status", "active"),
                    enforcement=p.get("enforcement", "enforced"),
                    created_at=p.get("created_at"),
                    updated_at=p.get("updated_at"),
                ))
            else:
                policies.append(PolicySummary(
                    id=getattr(p, "id", getattr(p, "policy_id", "")),
                    name=getattr(p, "name", ""),
                    description=getattr(p, "description", ""),
                    framework=getattr(p, "framework", ""),
                    status=getattr(p, "status", "active"),
                    enforcement=getattr(p, "enforcement", "enforced"),
                    created_at=str(getattr(p, "created_at", ""))
                    if hasattr(p, "created_at") else None,
                    updated_at=str(getattr(p, "updated_at", ""))
                    if hasattr(p, "updated_at") else None,
                ))

        return PolicyListResponse(
            policies=policies,
            total=total,
            limit=limit,
            offset=offset,
        )

    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error listing policies: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list policies")


@router.post("/compliance/artifacts/generate", response_model=GenerateArtifactResponse)
async def generate_compliance_artifact(
    body: GenerateArtifactRequest,
    auth: AuthorizationContext = Depends(require_permission("compliance:write")),
    fw=Depends(get_compliance_framework),
) -> GenerateArtifactResponse:
    """
    Generate compliance artifacts.

    Generates compliance documentation and evidence bundles for the
    specified framework. Supports EU AI Act, SOC 2, GDPR, HIPAA, and SOX.
    Requires `compliance:write` permission.
    """
    try:
        import uuid

        artifact_id = f"ca_{uuid.uuid4().hex[:12]}"
        content: dict[str, Any] = {}
        integrity_hash = ""

        # Try the dedicated artifact generator first
        try:
            from aragora.compliance.eu_ai_act import ComplianceArtifactGenerator

            if body.framework in ("eu_ai_act", "gdpr"):
                generator = ComplianceArtifactGenerator()
                if hasattr(generator, "generate_bundle"):
                    bundle = generator.generate_bundle(
                        debate_ids=body.debate_ids,
                        artifact_type=body.artifact_type,
                        **body.options,
                    )
                    if isinstance(bundle, dict):
                        content = bundle
                        integrity_hash = bundle.get("integrity_hash", "")
                    elif hasattr(bundle, "to_dict"):
                        content = bundle.to_dict()
                        integrity_hash = getattr(bundle, "integrity_hash", "")
        except (ImportError, RuntimeError, ValueError) as e:
            logger.debug("Dedicated artifact generator not available: %s", e)

        # Fall back to generic compliance framework
        if not content and fw:
            if hasattr(fw, "generate_artifact"):
                result = fw.generate_artifact(
                    framework=body.framework,
                    artifact_type=body.artifact_type,
                    debate_ids=body.debate_ids,
                    **body.options,
                )
                if isinstance(result, dict):
                    content = result
                    integrity_hash = result.get("integrity_hash", "")
            elif hasattr(fw, "generate"):
                result = fw.generate(framework=body.framework, **body.options)
                if isinstance(result, dict):
                    content = result

        # Generate a basic artifact if no backend available
        if not content:
            content = {
                "framework": body.framework,
                "artifact_type": body.artifact_type,
                "status": "generated",
                "debate_ids": body.debate_ids,
                "note": "Basic artifact generated without compliance engine",
            }

        logger.info(
            "Generated compliance artifact: %s (framework=%s, type=%s)",
            artifact_id, body.framework, body.artifact_type,
        )

        return GenerateArtifactResponse(
            success=True,
            artifact_id=artifact_id,
            framework=body.framework,
            artifact_type=body.artifact_type,
            content=content,
            integrity_hash=integrity_hash,
        )

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error generating compliance artifact: %s", e)
        raise HTTPException(status_code=500, detail="Failed to generate compliance artifact")


@router.get("/compliance/audit-log", response_model=AuditLogResponse)
async def query_audit_log(
    request: Request,
    limit: int = Query(50, ge=1, le=500, description="Max entries to return"),
    offset: int = Query(0, ge=0, description="Number of entries to skip"),
    action: str | None = Query(None, description="Filter by action type"),
    actor: str | None = Query(None, description="Filter by actor"),
    resource_type: str | None = Query(None, description="Filter by resource type"),
    auth: AuthorizationContext = Depends(require_permission("audit:read")),
    store=Depends(get_audit_store),
) -> AuditLogResponse:
    """
    Query the audit log.

    Returns paginated audit log entries with optional filters.
    Requires `audit:read` permission.
    """
    if not store:
        return AuditLogResponse(entries=[], total=0, limit=limit, offset=offset)

    try:
        entries_raw: list[Any] = []

        # Build query kwargs
        query_kwargs: dict[str, Any] = {"limit": limit, "offset": offset}
        if action:
            query_kwargs["action"] = action
        if actor:
            query_kwargs["actor"] = actor
        if resource_type:
            query_kwargs["resource_type"] = resource_type

        if hasattr(store, "query"):
            entries_raw = store.query(**query_kwargs)
        elif hasattr(store, "list_entries"):
            entries_raw = store.list_entries(**query_kwargs)
        elif hasattr(store, "get_entries"):
            all_entries = store.get_entries()
            # Apply filters
            if action:
                all_entries = [
                    e for e in all_entries
                    if (e.get("action") if isinstance(e, dict) else getattr(e, "action", ""))
                    == action
                ]
            if actor:
                all_entries = [
                    e for e in all_entries
                    if (e.get("actor") if isinstance(e, dict) else getattr(e, "actor", ""))
                    == actor
                ]
            if resource_type:
                all_entries = [
                    e for e in all_entries
                    if (e.get("resource_type") if isinstance(e, dict) else getattr(e, "resource_type", ""))
                    == resource_type
                ]
            entries_raw = all_entries[offset: offset + limit]

        # Get total count
        if hasattr(store, "count"):
            total = store.count(action=action, actor=actor, resource_type=resource_type)
        else:
            total = len(entries_raw)

        # Convert to response entries
        entries: list[AuditLogEntry] = []
        for entry in entries_raw:
            if isinstance(entry, dict):
                entries.append(AuditLogEntry(
                    id=entry.get("id", entry.get("entry_id", "")),
                    timestamp=entry.get("timestamp", ""),
                    action=entry.get("action", ""),
                    actor=entry.get("actor", ""),
                    resource_type=entry.get("resource_type", ""),
                    resource_id=entry.get("resource_id", ""),
                    details=entry.get("details", {}),
                ))
            else:
                entries.append(AuditLogEntry(
                    id=getattr(entry, "id", getattr(entry, "entry_id", "")),
                    timestamp=str(getattr(entry, "timestamp", "")),
                    action=getattr(entry, "action", ""),
                    actor=getattr(entry, "actor", ""),
                    resource_type=getattr(entry, "resource_type", ""),
                    resource_id=getattr(entry, "resource_id", ""),
                    details=getattr(entry, "details", {}),
                ))

        return AuditLogResponse(
            entries=entries,
            total=total,
            limit=limit,
            offset=offset,
        )

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error querying audit log: %s", e)
        raise HTTPException(status_code=500, detail="Failed to query audit log")
