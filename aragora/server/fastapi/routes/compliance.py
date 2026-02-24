"""
Compliance Endpoints (FastAPI v2).

Migrated from: aragora/server/handlers/compliance/ (aiohttp handler)

Provides async compliance management endpoints:
- GET  /api/v2/compliance/status                    - Compliance framework status
- GET  /api/v2/compliance/controls                  - List compliance controls
- GET  /api/v2/compliance/policies                  - List policies
- GET  /api/v2/compliance/frameworks                - List frameworks
- GET  /api/v2/compliance/frameworks/{framework_id} - Framework details
- GET  /api/v2/compliance/violations                - List violations
- GET  /api/v2/compliance/audit-log                 - Query audit log
- POST /api/v2/compliance/check                     - Run compliance check
- POST /api/v2/compliance/artifacts/generate        - Generate compliance artifacts
- GET  /api/v2/compliance/report/{debate_id}        - Compliance report for debate

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


class ControlListResponse(BaseModel):
    """Response for compliance controls listing."""

    controls: list[ControlStatus]
    total: int
    framework: str = ""
    limit: int
    offset: int


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
                controls.append(
                    ControlStatus(
                        control_id=ctrl.get("control_id", ctrl.get("id", "")),
                        name=ctrl.get("name", ""),
                        description=ctrl.get("description", ""),
                        status=ctrl.get("status", "not_assessed"),
                        evidence_count=ctrl.get("evidence_count", 0),
                        last_assessed=ctrl.get("last_assessed"),
                    )
                )
            else:
                controls.append(
                    ControlStatus(
                        control_id=getattr(ctrl, "control_id", getattr(ctrl, "id", "")),
                        name=getattr(ctrl, "name", ""),
                        description=getattr(ctrl, "description", ""),
                        status=getattr(ctrl, "status", "not_assessed"),
                        evidence_count=getattr(ctrl, "evidence_count", 0),
                        last_assessed=getattr(ctrl, "last_assessed", None),
                    )
                )

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


@router.get("/compliance/controls", response_model=ControlListResponse)
async def list_compliance_controls(
    request: Request,
    framework: str = Query("soc2", description="Compliance framework"),
    status: str | None = Query(
        None, description="Filter by status (passing, failing, not_assessed)"
    ),
    limit: int = Query(50, ge=1, le=200, description="Max results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    fw=Depends(get_compliance_framework),
) -> ControlListResponse:
    """
    List compliance controls for a given framework.

    Returns individual control statuses with evidence counts and assessment dates.
    Supports filtering by status (passing, failing, not_assessed).
    """
    if not fw:
        return ControlListResponse(
            controls=[],
            total=0,
            framework=framework,
            limit=limit,
            offset=offset,
        )

    try:
        controls: list[ControlStatus] = []

        # Try dedicated controls listing
        if hasattr(fw, "list_controls"):
            raw_controls = fw.list_controls(framework=framework, status=status)
        elif hasattr(fw, "get_controls"):
            raw_controls = fw.get_controls(framework=framework)
        elif hasattr(fw, "get_status"):
            # Fall back to extracting controls from status
            raw_status = fw.get_status(framework=framework)
            raw_controls = raw_status.get("controls", []) if isinstance(raw_status, dict) else []
        else:
            raw_controls = []

        for ctrl in raw_controls:
            if isinstance(ctrl, dict):
                ctrl_status = ctrl.get("status", "not_assessed")
                if status and ctrl_status != status:
                    continue
                controls.append(
                    ControlStatus(
                        control_id=ctrl.get("control_id", ctrl.get("id", "")),
                        name=ctrl.get("name", ""),
                        description=ctrl.get("description", ""),
                        status=ctrl_status,
                        evidence_count=ctrl.get("evidence_count", 0),
                        last_assessed=ctrl.get("last_assessed"),
                    )
                )
            else:
                ctrl_status = getattr(ctrl, "status", "not_assessed")
                if status and ctrl_status != status:
                    continue
                controls.append(
                    ControlStatus(
                        control_id=getattr(ctrl, "control_id", getattr(ctrl, "id", "")),
                        name=getattr(ctrl, "name", ""),
                        description=getattr(ctrl, "description", ""),
                        status=ctrl_status,
                        evidence_count=getattr(ctrl, "evidence_count", 0),
                        last_assessed=getattr(ctrl, "last_assessed", None),
                    )
                )

        total = len(controls)
        paginated = controls[offset : offset + limit]

        return ControlListResponse(
            controls=paginated,
            total=total,
            framework=framework,
            limit=limit,
            offset=offset,
        )

    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error listing compliance controls: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list compliance controls")


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
                limit=limit,
                offset=offset,
                framework=framework,
                status=status,
            )
        elif hasattr(fw, "get_policies"):
            all_policies = fw.get_policies()
            if framework:
                all_policies = [
                    p
                    for p in all_policies
                    if (p.get("framework") if isinstance(p, dict) else getattr(p, "framework", ""))
                    == framework
                ]
            if status:
                all_policies = [
                    p
                    for p in all_policies
                    if (p.get("status") if isinstance(p, dict) else getattr(p, "status", ""))
                    == status
                ]
            policies_raw = all_policies[offset : offset + limit]

        # Get total count
        if hasattr(fw, "count_policies"):
            total = fw.count_policies(framework=framework, status=status)
        else:
            total = len(policies_raw)

        # Convert to summaries
        policies: list[PolicySummary] = []
        for p in policies_raw:
            if isinstance(p, dict):
                policies.append(
                    PolicySummary(
                        id=p.get("id", p.get("policy_id", "")),
                        name=p.get("name", ""),
                        description=p.get("description", ""),
                        framework=p.get("framework", ""),
                        status=p.get("status", "active"),
                        enforcement=p.get("enforcement", "enforced"),
                        created_at=p.get("created_at"),
                        updated_at=p.get("updated_at"),
                    )
                )
            else:
                policies.append(
                    PolicySummary(
                        id=getattr(p, "id", getattr(p, "policy_id", "")),
                        name=getattr(p, "name", ""),
                        description=getattr(p, "description", ""),
                        framework=getattr(p, "framework", ""),
                        status=getattr(p, "status", "active"),
                        enforcement=getattr(p, "enforcement", "enforced"),
                        created_at=str(getattr(p, "created_at", ""))
                        if hasattr(p, "created_at")
                        else None,
                        updated_at=str(getattr(p, "updated_at", ""))
                        if hasattr(p, "updated_at")
                        else None,
                    )
                )

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
            artifact_id,
            body.framework,
            body.artifact_type,
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
                    e
                    for e in all_entries
                    if (e.get("action") if isinstance(e, dict) else getattr(e, "action", ""))
                    == action
                ]
            if actor:
                all_entries = [
                    e
                    for e in all_entries
                    if (e.get("actor") if isinstance(e, dict) else getattr(e, "actor", "")) == actor
                ]
            if resource_type:
                all_entries = [
                    e
                    for e in all_entries
                    if (
                        e.get("resource_type")
                        if isinstance(e, dict)
                        else getattr(e, "resource_type", "")
                    )
                    == resource_type
                ]
            entries_raw = all_entries[offset : offset + limit]

        # Get total count
        if hasattr(store, "count"):
            total = store.count(action=action, actor=actor, resource_type=resource_type)
        else:
            total = len(entries_raw)

        # Convert to response entries
        entries: list[AuditLogEntry] = []
        for entry in entries_raw:
            if isinstance(entry, dict):
                entries.append(
                    AuditLogEntry(
                        id=entry.get("id", entry.get("entry_id", "")),
                        timestamp=entry.get("timestamp", ""),
                        action=entry.get("action", ""),
                        actor=entry.get("actor", ""),
                        resource_type=entry.get("resource_type", ""),
                        resource_id=entry.get("resource_id", ""),
                        details=entry.get("details", {}),
                    )
                )
            else:
                entries.append(
                    AuditLogEntry(
                        id=getattr(entry, "id", getattr(entry, "entry_id", "")),
                        timestamp=str(getattr(entry, "timestamp", "")),
                        action=getattr(entry, "action", ""),
                        actor=getattr(entry, "actor", ""),
                        resource_type=getattr(entry, "resource_type", ""),
                        resource_id=getattr(entry, "resource_id", ""),
                        details=getattr(entry, "details", {}),
                    )
                )

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


# =============================================================================
# New Pydantic Models (Frameworks, Check, Violations, Report)
# =============================================================================


class FrameworkDetail(BaseModel):
    """Detail of a compliance framework."""

    id: str
    name: str
    description: str = ""
    version: str = ""
    category: str = ""
    rule_count: int = 0
    applicable_verticals: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class FrameworkListResponse(BaseModel):
    """Response for framework listing."""

    frameworks: list[FrameworkDetail]
    total: int


class ComplianceCheckRequest(BaseModel):
    """Request body for POST /compliance/check."""

    content: str = Field(..., min_length=1, description="Content to check for compliance")
    frameworks: list[str] | None = Field(None, description="Framework IDs to check (None = all)")
    min_severity: str = Field("low", description="Minimum severity to report")


class ComplianceIssueDetail(BaseModel):
    """A single compliance issue found during a check."""

    framework: str = ""
    rule_id: str = ""
    severity: str = ""
    description: str = ""
    recommendation: str = ""
    matched_text: str = ""
    line_number: int | None = None

    model_config = {"extra": "allow"}


class ComplianceCheckResponse(BaseModel):
    """Response for compliance check."""

    compliant: bool
    score: float = 1.0
    frameworks_checked: list[str] = Field(default_factory=list)
    issue_count: int = 0
    critical_count: int = 0
    high_count: int = 0
    issues: list[ComplianceIssueDetail] = Field(default_factory=list)
    checked_at: str = ""


class ViolationEntry(BaseModel):
    """A single compliance violation."""

    id: str = ""
    framework: str = ""
    rule_id: str = ""
    severity: str = ""
    description: str = ""
    resource_type: str = ""
    resource_id: str = ""
    detected_at: str | None = None
    status: str = "active"

    model_config = {"extra": "allow"}


class ViolationsResponse(BaseModel):
    """Response for violations listing."""

    violations: list[ViolationEntry]
    total: int
    limit: int
    offset: int


class ComplianceReportResponse(BaseModel):
    """Response for compliance report generation."""

    debate_id: str
    frameworks_assessed: list[str] = Field(default_factory=list)
    overall_compliant: bool = True
    score: float = 1.0
    issue_count: int = 0
    issues: list[ComplianceIssueDetail] = Field(default_factory=list)
    generated_at: str = ""


# =============================================================================
# New Endpoints (Frameworks, Check, Violations, Report)
# =============================================================================


def _get_framework_manager(fw: Any) -> Any:
    """Get or create a ComplianceFrameworkManager."""
    if fw and hasattr(fw, "list_frameworks"):
        return fw
    try:
        from aragora.compliance.framework import ComplianceFrameworkManager

        return ComplianceFrameworkManager()
    except (ImportError, RuntimeError, ValueError) as e:
        logger.debug("ComplianceFrameworkManager not available: %s", e)
        return None


@router.get("/compliance/frameworks", response_model=FrameworkListResponse)
async def list_frameworks(
    request: Request,
    fw=Depends(get_compliance_framework),
) -> FrameworkListResponse:
    """List all available compliance frameworks."""
    mgr = _get_framework_manager(fw)
    if not mgr:
        return FrameworkListResponse(frameworks=[], total=0)

    try:
        frameworks: list[FrameworkDetail] = []
        raw_frameworks = mgr.list_frameworks()
        for f in raw_frameworks:
            if isinstance(f, dict):
                frameworks.append(
                    FrameworkDetail(
                        id=f.get("id", ""),
                        name=f.get("name", ""),
                        description=f.get("description", ""),
                        version=f.get("version", ""),
                        category=f.get("category", ""),
                        rule_count=f.get("rule_count", 0),
                        applicable_verticals=f.get("applicable_verticals", []),
                    )
                )
            else:
                frameworks.append(
                    FrameworkDetail(
                        id=getattr(f, "id", ""),
                        name=getattr(f, "name", ""),
                        description=getattr(f, "description", ""),
                        version=getattr(f, "version", ""),
                        category=getattr(f, "category", ""),
                        rule_count=len(getattr(f, "rules", [])),
                        applicable_verticals=getattr(f, "applicable_verticals", []),
                    )
                )

        return FrameworkListResponse(frameworks=frameworks, total=len(frameworks))

    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error listing frameworks: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list frameworks")


@router.get("/compliance/frameworks/{framework_id}", response_model=FrameworkDetail)
async def get_framework_detail(
    framework_id: str,
    fw=Depends(get_compliance_framework),
) -> FrameworkDetail:
    """Get compliance framework details by ID."""
    mgr = _get_framework_manager(fw)
    if not mgr:
        raise HTTPException(status_code=503, detail="Compliance framework not available")

    try:
        framework_obj = None
        if hasattr(mgr, "get_framework"):
            framework_obj = mgr.get_framework(framework_id)

        if not framework_obj:
            raise NotFoundError(f"Framework {framework_id} not found")

        if isinstance(framework_obj, dict):
            return FrameworkDetail(
                id=framework_obj.get("id", framework_id),
                name=framework_obj.get("name", ""),
                description=framework_obj.get("description", ""),
                version=framework_obj.get("version", ""),
                category=framework_obj.get("category", ""),
                rule_count=framework_obj.get("rule_count", 0),
                applicable_verticals=framework_obj.get("applicable_verticals", []),
            )

        return FrameworkDetail(
            id=getattr(framework_obj, "id", framework_id),
            name=getattr(framework_obj, "name", ""),
            description=getattr(framework_obj, "description", ""),
            version=getattr(framework_obj, "version", ""),
            category=getattr(framework_obj, "category", ""),
            rule_count=len(getattr(framework_obj, "rules", [])),
            applicable_verticals=getattr(framework_obj, "applicable_verticals", []),
        )

    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting framework %s: %s", framework_id, e)
        raise HTTPException(status_code=500, detail="Failed to get framework")


@router.post("/compliance/check", response_model=ComplianceCheckResponse)
async def check_compliance(
    body: ComplianceCheckRequest,
    fw=Depends(get_compliance_framework),
) -> ComplianceCheckResponse:
    """Run a compliance check on provided content against one or more frameworks."""
    from datetime import datetime, timezone

    mgr = _get_framework_manager(fw)
    if not mgr:
        raise HTTPException(status_code=503, detail="Compliance framework not available")

    try:
        check_kwargs: dict[str, Any] = {}
        if body.frameworks:
            check_kwargs["frameworks"] = body.frameworks

        result = mgr.check(body.content, **check_kwargs)

        def _issue_to_detail(i: Any) -> ComplianceIssueDetail:
            if isinstance(i, dict):
                return ComplianceIssueDetail(**i)
            sev = getattr(i, "severity", "")
            return ComplianceIssueDetail(
                framework=getattr(i, "framework", ""),
                rule_id=getattr(i, "rule_id", ""),
                severity=sev.value if hasattr(sev, "value") else str(sev),
                description=getattr(i, "description", ""),
                recommendation=getattr(i, "recommendation", ""),
                matched_text=getattr(i, "matched_text", ""),
                line_number=getattr(i, "line_number", None),
            )

        if hasattr(result, "to_dict"):
            result_dict = result.to_dict()
            issues = [_issue_to_detail(i) for i in getattr(result, "issues", [])]
            checked_at_val = getattr(result, "checked_at", None)
            checked_at_str = (
                checked_at_val.isoformat()
                if hasattr(checked_at_val, "isoformat")
                else datetime.now(timezone.utc).isoformat()
            )
            return ComplianceCheckResponse(
                compliant=getattr(result, "compliant", True),
                score=getattr(result, "score", 1.0),
                frameworks_checked=getattr(result, "frameworks_checked", []),
                issue_count=result_dict.get("issue_count", len(issues)),
                critical_count=result_dict.get("critical_count", 0),
                high_count=result_dict.get("high_count", 0),
                issues=issues,
                checked_at=checked_at_str,
            )

        if isinstance(result, dict):
            issues = [_issue_to_detail(i) for i in result.get("issues", [])]
            return ComplianceCheckResponse(
                compliant=result.get("compliant", True),
                score=result.get("score", 1.0),
                frameworks_checked=result.get("frameworks_checked", []),
                issue_count=result.get("issue_count", len(issues)),
                critical_count=result.get("critical_count", 0),
                high_count=result.get("high_count", 0),
                issues=issues,
                checked_at=result.get("checked_at", datetime.now(timezone.utc).isoformat()),
            )

        return ComplianceCheckResponse(
            compliant=True,
            score=1.0,
            checked_at=datetime.now(timezone.utc).isoformat(),
        )

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error running compliance check: %s", e)
        raise HTTPException(status_code=500, detail="Failed to run compliance check")


@router.get("/compliance/violations", response_model=ViolationsResponse)
async def list_violations(
    request: Request,
    limit: int = Query(50, ge=1, le=100, description="Max results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    framework: str | None = Query(None, description="Filter by framework"),
    severity: str | None = Query(None, description="Filter by severity"),
    status: str | None = Query(None, description="Filter by status (active, resolved)"),
    fw=Depends(get_compliance_framework),
) -> ViolationsResponse:
    """List active compliance violations with optional filters."""
    if not fw:
        return ViolationsResponse(violations=[], total=0, limit=limit, offset=offset)

    try:
        query_kwargs: dict[str, Any] = {"limit": limit, "offset": offset}
        if framework:
            query_kwargs["framework"] = framework
        if severity:
            query_kwargs["severity"] = severity
        if status:
            query_kwargs["status"] = status

        raw_violations: list[Any] = []
        if hasattr(fw, "list_violations"):
            raw_violations = fw.list_violations(**query_kwargs)
        elif hasattr(fw, "get_violations"):
            raw_violations = fw.get_violations(**query_kwargs)

        violations: list[ViolationEntry] = []
        for v in raw_violations:
            if isinstance(v, dict):
                violations.append(
                    ViolationEntry(
                        id=v.get("id", v.get("violation_id", "")),
                        framework=v.get("framework", ""),
                        rule_id=v.get("rule_id", ""),
                        severity=v.get("severity", ""),
                        description=v.get("description", ""),
                        resource_type=v.get("resource_type", ""),
                        resource_id=v.get("resource_id", ""),
                        detected_at=v.get("detected_at"),
                        status=v.get("status", "active"),
                    )
                )
            else:
                sev = getattr(v, "severity", "")
                violations.append(
                    ViolationEntry(
                        id=getattr(v, "id", getattr(v, "violation_id", "")),
                        framework=getattr(v, "framework", ""),
                        rule_id=getattr(v, "rule_id", ""),
                        severity=sev.value if hasattr(sev, "value") else str(sev),
                        description=getattr(v, "description", ""),
                        resource_type=getattr(v, "resource_type", ""),
                        resource_id=getattr(v, "resource_id", ""),
                        detected_at=str(getattr(v, "detected_at", ""))
                        if hasattr(v, "detected_at")
                        else None,
                        status=getattr(v, "status", "active"),
                    )
                )

        total = len(violations)
        if hasattr(fw, "count_violations"):
            total = fw.count_violations(framework=framework, severity=severity, status=status)

        return ViolationsResponse(violations=violations, total=total, limit=limit, offset=offset)

    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error listing violations: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list violations")


@router.get("/compliance/report/{debate_id}", response_model=ComplianceReportResponse)
async def get_compliance_report(
    debate_id: str,
    frameworks: str | None = Query(None, description="Comma-separated framework IDs to assess"),
    fw=Depends(get_compliance_framework),
) -> ComplianceReportResponse:
    """Generate a compliance report for a specific debate."""
    from datetime import datetime, timezone

    try:
        framework_list = [f.strip() for f in frameworks.split(",")] if frameworks else None

        debate_content = ""
        try:
            from aragora.storage.debate_storage import DebateStorage

            storage = DebateStorage()
            debate_data = storage.get(debate_id)
            if debate_data:
                if isinstance(debate_data, dict):
                    debate_content = str(debate_data.get("result", debate_data.get("task", "")))
                elif hasattr(debate_data, "result"):
                    debate_content = str(getattr(debate_data, "result", ""))
        except (ImportError, RuntimeError, OSError, ValueError, TypeError) as e:
            logger.debug("Could not load debate %s: %s", debate_id, e)

        mgr = _get_framework_manager(fw)
        if mgr and debate_content and hasattr(mgr, "check"):
            check_kwargs: dict[str, Any] = {}
            if framework_list:
                check_kwargs["frameworks"] = framework_list
            result = mgr.check(debate_content, **check_kwargs)

            issues: list[ComplianceIssueDetail] = []
            for i in getattr(result, "issues", []):
                sev = getattr(i, "severity", "")
                issues.append(
                    ComplianceIssueDetail(
                        framework=getattr(i, "framework", ""),
                        rule_id=getattr(i, "rule_id", ""),
                        severity=sev.value if hasattr(sev, "value") else str(sev),
                        description=getattr(i, "description", ""),
                        recommendation=getattr(i, "recommendation", ""),
                        matched_text=getattr(i, "matched_text", ""),
                        line_number=getattr(i, "line_number", None),
                    )
                )

            return ComplianceReportResponse(
                debate_id=debate_id,
                frameworks_assessed=getattr(result, "frameworks_checked", framework_list or []),
                overall_compliant=getattr(result, "compliant", True),
                score=getattr(result, "score", 1.0),
                issue_count=len(issues),
                issues=issues,
                generated_at=datetime.now(timezone.utc).isoformat(),
            )

        return ComplianceReportResponse(
            debate_id=debate_id,
            frameworks_assessed=framework_list or [],
            overall_compliant=True,
            score=1.0,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error generating compliance report for %s: %s", debate_id, e)
        raise HTTPException(status_code=500, detail="Failed to generate compliance report")
