"""
MCP Audit Tools for Aragora.

Provides document audit capabilities for AI harnesses like Claude Code, Codex, etc.

Tools:
- list_audit_presets: List available audit preset configurations
- list_audit_types: List registered audit types with capabilities
- create_audit_session: Create new audit session
- run_audit: Start an audit session
- get_audit_status: Get session status and progress
- get_audit_findings: Get findings from completed audit
- update_finding_status: Update finding workflow state
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def list_audit_presets_tool() -> dict[str, Any]:
    """
    List available audit presets for different industries.

    Presets include pre-configured rules for:
    - Legal Due Diligence (contracts, compliance, jurisdiction)
    - Financial Audit (accounting irregularities, SOX compliance)
    - Code Security (vulnerabilities, secrets, licenses)

    Returns:
        Dictionary with presets list
    """
    try:
        from aragora.audit.registry import audit_registry

        audit_registry.auto_discover()
        presets = audit_registry.list_presets()

        return {
            "success": True,
            "presets": [
                {
                    "name": p.name,
                    "description": p.description,
                    "audit_types": p.audit_types,
                    "custom_rules_count": len(p.custom_rules),
                    "consensus_threshold": p.consensus_threshold,
                }
                for p in presets
            ],
            "count": len(presets),
        }
    except Exception as e:
        logger.error(f"Failed to list audit presets: {e}")
        return {"success": False, "error": str(e)}


async def list_audit_types_tool() -> dict[str, Any]:
    """
    List registered audit types and their capabilities.

    Returns information about available auditors:
    - security: Credentials, injection vulnerabilities, data exposure
    - compliance: GDPR, HIPAA, SOC2, contractual violations
    - consistency: Cross-document contradictions
    - quality: Ambiguity, completeness, documentation quality

    Returns:
        Dictionary with audit types list
    """
    try:
        from aragora.audit.registry import audit_registry

        audit_registry.auto_discover()
        types = audit_registry.list_audit_types()

        return {
            "success": True,
            "audit_types": [
                {
                    "id": t.id,
                    "display_name": t.display_name,
                    "description": t.description,
                    "version": t.version,
                    "capabilities": t.capabilities,
                }
                for t in types
            ],
            "count": len(types),
        }
    except Exception as e:
        logger.error(f"Failed to list audit types: {e}")
        return {"success": False, "error": str(e)}


async def get_audit_preset_tool(preset_name: str) -> dict[str, Any]:
    """
    Get details of a specific audit preset.

    Args:
        preset_name: Name of the preset (e.g., "Legal Due Diligence")

    Returns:
        Dictionary with preset configuration
    """
    try:
        from aragora.audit.registry import audit_registry

        audit_registry.auto_discover()
        preset = audit_registry.get_preset(preset_name)

        if not preset:
            return {"success": False, "error": f"Preset '{preset_name}' not found"}

        return {
            "success": True,
            "preset": {
                "name": preset.name,
                "description": preset.description,
                "audit_types": preset.audit_types,
                "custom_rules": [
                    {
                        "pattern": r.get("pattern", ""),
                        "severity": r.get("severity", "medium"),
                        "category": r.get("category", ""),
                        "title": r.get("title", ""),
                    }
                    for r in preset.custom_rules
                ],
                "consensus_threshold": preset.consensus_threshold,
                "agents": preset.agents,
                "parameters": preset.parameters,
            },
        }
    except Exception as e:
        logger.error(f"Failed to get preset: {e}")
        return {"success": False, "error": str(e)}


async def create_audit_session_tool(
    document_ids: str,
    audit_types: str = "security,compliance,consistency,quality",
    preset: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """
    Create a new document audit session.

    Args:
        document_ids: Comma-separated list of document IDs to audit
        audit_types: Comma-separated audit types (default: all types)
        preset: Optional preset name to use instead of audit_types
        name: Optional session name

    Returns:
        Dictionary with created session info
    """
    try:
        from aragora.audit import DocumentAuditor, AuditConfig

        doc_ids = [d.strip() for d in document_ids.split(",") if d.strip()]
        if not doc_ids:
            return {"success": False, "error": "No document IDs provided"}

        # Get audit types from preset or parameter
        types = [t.strip() for t in audit_types.split(",")]
        if preset:
            from aragora.audit.registry import audit_registry

            audit_registry.auto_discover()
            preset_config = audit_registry.get_preset(preset)
            if preset_config:
                types = preset_config.audit_types

        config = AuditConfig(
            use_hive_mind=len(doc_ids) > 1,
            consensus_verification=True,
        )

        auditor = DocumentAuditor(config=config)
        session = await auditor.create_session(
            document_ids=doc_ids,
            audit_types=types,
            name=name or f"Audit-{','.join(types)}",
        )

        return {
            "success": True,
            "session": {
                "id": session.id,
                "name": session.name,
                "document_count": len(doc_ids),
                "audit_types": types,
                "status": session.status.value,
            },
        }
    except Exception as e:
        logger.error(f"Failed to create audit session: {e}")
        return {"success": False, "error": str(e)}


async def run_audit_tool(session_id: str) -> dict[str, Any]:
    """
    Start running an audit session.

    Args:
        session_id: ID of the audit session to start

    Returns:
        Dictionary with run status
    """
    try:
        from aragora.audit import DocumentAuditor

        auditor = DocumentAuditor()
        result = await auditor.run_audit(session_id)

        return {
            "success": True,
            "session_id": session_id,
            "status": result.status.value if hasattr(result, "status") else "completed",
            "findings_count": len(result.findings) if hasattr(result, "findings") else 0,
        }
    except Exception as e:
        logger.error(f"Failed to run audit: {e}")
        return {"success": False, "error": str(e)}


async def get_audit_status_tool(session_id: str) -> dict[str, Any]:
    """
    Get status and progress of an audit session.

    Args:
        session_id: ID of the audit session

    Returns:
        Dictionary with session status
    """
    try:
        from aragora.audit import DocumentAuditor

        auditor = DocumentAuditor()
        session = await auditor.get_session(session_id)

        if not session:
            return {"success": False, "error": f"Session {session_id} not found"}

        return {
            "success": True,
            "session": {
                "id": session.id,
                "name": session.name,
                "status": session.status.value,
                "progress": session.progress,
                "current_phase": session.current_phase,
                "total_chunks": session.total_chunks,
                "processed_chunks": session.processed_chunks,
                "findings_count": len(session.findings),
            },
        }
    except Exception as e:
        logger.error(f"Failed to get audit status: {e}")
        return {"success": False, "error": str(e)}


async def get_audit_findings_tool(
    session_id: str,
    severity: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Get findings from an audit session.

    Args:
        session_id: ID of the audit session
        severity: Filter by severity (critical, high, medium, low, info)
        status: Filter by status (open, triaging, investigating, resolved, etc.)
        limit: Maximum number of findings to return

    Returns:
        Dictionary with findings list
    """
    try:
        from aragora.audit import DocumentAuditor

        auditor = DocumentAuditor()
        findings = await auditor.get_findings(session_id)

        # Apply filters
        if severity:
            findings = [f for f in findings if f.severity.value == severity]
        if status:
            findings = [f for f in findings if f.status.value == status]

        # Limit results
        findings = findings[:limit]

        return {
            "success": True,
            "session_id": session_id,
            "findings": [
                {
                    "id": f.id,
                    "title": f.title,
                    "description": f.description,
                    "severity": f.severity.value,
                    "status": f.status.value,
                    "audit_type": f.audit_type.value,
                    "category": f.category,
                    "confidence": f.confidence,
                    "evidence_text": f.evidence_text[:500] if f.evidence_text else "",
                    "recommendation": f.recommendation,
                    "document_id": f.document_id,
                }
                for f in findings
            ],
            "count": len(findings),
        }
    except Exception as e:
        logger.error(f"Failed to get findings: {e}")
        return {"success": False, "error": str(e)}


async def update_finding_status_tool(
    finding_id: str,
    status: str,
    comment: str = "",
) -> dict[str, Any]:
    """
    Update the workflow status of a finding.

    Args:
        finding_id: ID of the finding to update
        status: New status (open, triaging, investigating, remediating, resolved, false_positive, accepted_risk)
        comment: Optional comment explaining the change

    Returns:
        Dictionary with update result
    """
    try:
        from aragora.audit.findings.workflow import (
            FindingWorkflow,
            FindingWorkflowData,
            WorkflowState,
        )

        # Load or create workflow data
        data = FindingWorkflowData(finding_id=finding_id)
        workflow = FindingWorkflow(data)

        try:
            target_state = WorkflowState(status)
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid status: {status}. Valid: {[s.value for s in WorkflowState]}",
            }

        if not workflow.can_transition_to(target_state):
            return {
                "success": False,
                "error": f"Cannot transition from {workflow.state.value} to {status}",
                "valid_transitions": [s.value for s in workflow.get_valid_transitions()],
            }

        event = workflow.transition_to(
            target_state,
            user_id="mcp_tool",
            user_name="MCP Tool",
            comment=comment,
        )

        return {
            "success": True,
            "finding_id": finding_id,
            "previous_state": event.from_state.value if event.from_state else None,
            "current_state": workflow.state.value,
            "comment": comment,
        }
    except Exception as e:
        logger.error(f"Failed to update finding status: {e}")
        return {"success": False, "error": str(e)}


async def run_quick_audit_tool(
    document_ids: str,
    preset: str = "Code Security",
) -> dict[str, Any]:
    """
    Run a quick audit using a preset and return findings summary.

    This is a convenience tool that creates a session, runs the audit,
    and returns a summary of findings.

    Args:
        document_ids: Comma-separated list of document IDs
        preset: Preset to use (default: Code Security)

    Returns:
        Dictionary with audit results summary
    """
    try:
        # Create session
        session_result = await create_audit_session_tool(
            document_ids=document_ids,
            preset=preset,
        )

        if not session_result.get("success"):
            return session_result

        session_id = session_result["session"]["id"]

        # Run audit
        run_result = await run_audit_tool(session_id)
        if not run_result.get("success"):
            return run_result

        # Get findings summary
        findings_result = await get_audit_findings_tool(session_id)

        # Create summary by severity
        findings = findings_result.get("findings", [])
        severity_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0,
        }
        for f in findings:
            sev = f.get("severity", "info")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        return {
            "success": True,
            "session_id": session_id,
            "preset_used": preset,
            "document_count": len(document_ids.split(",")),
            "total_findings": len(findings),
            "findings_by_severity": severity_counts,
            "critical_findings": [f for f in findings if f.get("severity") == "critical"][
                :5
            ],  # Top 5 critical
            "high_findings": [f for f in findings if f.get("severity") == "high"][:5],  # Top 5 high
        }
    except Exception as e:
        logger.error(f"Failed to run quick audit: {e}")
        return {"success": False, "error": str(e)}


__all__ = [
    "list_audit_presets_tool",
    "list_audit_types_tool",
    "get_audit_preset_tool",
    "create_audit_session_tool",
    "run_audit_tool",
    "get_audit_status_tool",
    "get_audit_findings_tool",
    "update_finding_status_tool",
    "run_quick_audit_tool",
]
