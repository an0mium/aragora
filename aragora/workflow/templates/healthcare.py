"""
Healthcare Industry Workflow Templates.

HIPAA-compliant templates for healthcare data review and compliance.
"""

from typing import Dict, Any

HIPAA_ASSESSMENT_TEMPLATE: Dict[str, Any] = {
    "name": "HIPAA Risk Assessment",
    "description": "Comprehensive HIPAA security risk assessment workflow",
    "category": "healthcare",
    "version": "1.0",
    "tags": ["healthcare", "hipaa", "security", "compliance"],
    "steps": [
        {
            "id": "scope",
            "type": "task",
            "name": "Define Assessment Scope",
            "description": "Identify systems and data in scope for HIPAA assessment",
            "config": {
                "task_type": "validate",
                "validation_rules": ["phi_systems_identified", "scope_approved"],
            },
        },
        {
            "id": "asset_inventory",
            "type": "memory_read",
            "name": "Asset Inventory",
            "description": "Retrieve inventory of PHI-containing systems",
            "config": {
                "query_template": "PHI systems inventory for {organization}",
                "domains": ["healthcare/administrative", "technical/infrastructure"],
            },
        },
        {
            "id": "privacy_assessment",
            "type": "debate",
            "name": "Privacy Rule Assessment",
            "description": "Assess compliance with HIPAA Privacy Rule",
            "config": {
                "agents": ["hipaa_auditor", "compliance_officer", "clinical_reviewer"],
                "rounds": 3,
                "topic_template": "Assess Privacy Rule compliance for: {asset_inventory}",
            },
        },
        {
            "id": "security_assessment",
            "type": "debate",
            "name": "Security Rule Assessment",
            "description": "Assess compliance with HIPAA Security Rule",
            "config": {
                "agents": ["hipaa_auditor", "security_engineer", "code_security_specialist"],
                "rounds": 3,
                "topic_template": "Assess Security Rule compliance: {asset_inventory}",
            },
        },
        {
            "id": "breach_readiness",
            "type": "debate",
            "name": "Breach Notification Readiness",
            "description": "Assess breach notification procedures",
            "config": {
                "agents": ["hipaa_auditor", "compliance_officer"],
                "topic_template": "Assess breach notification readiness",
            },
        },
        {
            "id": "risk_scoring",
            "type": "task",
            "name": "Risk Scoring",
            "description": "Calculate risk scores for identified vulnerabilities",
            "config": {
                "task_type": "function",
                "function_name": "calculate_hipaa_risk_score",
            },
        },
        {
            "id": "ciso_review",
            "type": "human_checkpoint",
            "name": "CISO Review",
            "description": "Chief Information Security Officer review",
            "config": {
                "approval_type": "sign_off",
                "required_role": "ciso",
                "checklist": [
                    "Verify technical control assessment",
                    "Confirm risk ratings",
                    "Approve remediation priorities",
                ],
            },
        },
        {
            "id": "generate_report",
            "type": "task",
            "name": "Generate Assessment Report",
            "description": "Generate HIPAA risk assessment report",
            "config": {
                "task_type": "transform",
                "template": "hipaa_risk_assessment_report",
            },
        },
        {
            "id": "remediation_plan",
            "type": "task",
            "name": "Remediation Plan",
            "description": "Generate prioritized remediation plan",
            "config": {
                "task_type": "transform",
                "template": "hipaa_remediation_plan",
            },
        },
        {
            "id": "archive",
            "type": "memory_write",
            "name": "Archive Assessment",
            "description": "Archive assessment for compliance records",
            "config": {
                "domain": "healthcare/compliance",
                "retention_years": 6,  # HIPAA requires 6-year retention
            },
        },
    ],
    "transitions": [
        {"from": "scope", "to": "asset_inventory"},
        {"from": "asset_inventory", "to": "privacy_assessment"},
        {"from": "privacy_assessment", "to": "security_assessment"},
        {"from": "security_assessment", "to": "breach_readiness"},
        {"from": "breach_readiness", "to": "risk_scoring"},
        {"from": "risk_scoring", "to": "ciso_review"},
        {"from": "ciso_review", "to": "generate_report", "condition": "approved"},
        {"from": "generate_report", "to": "remediation_plan"},
        {"from": "remediation_plan", "to": "archive"},
    ],
}

CLINICAL_REVIEW_TEMPLATE: Dict[str, Any] = {
    "name": "Clinical Document Review",
    "description": "Multi-agent review of clinical documentation and protocols",
    "category": "healthcare",
    "version": "1.0",
    "tags": ["healthcare", "clinical", "documentation"],
    "steps": [
        {
            "id": "document_intake",
            "type": "task",
            "name": "Document Intake",
            "description": "Ingest and classify clinical documents",
            "config": {
                "task_type": "transform",
                "input_key": "clinical_documents",
                "output_key": "parsed_documents",
            },
        },
        {
            "id": "phi_redaction",
            "type": "task",
            "name": "PHI Redaction",
            "description": "Apply Safe Harbor PHI redaction",
            "config": {
                "task_type": "function",
                "function_name": "redact_phi_safe_harbor",
            },
        },
        {
            "id": "terminology_check",
            "type": "debate",
            "name": "Medical Terminology Review",
            "description": "Verify medical terminology accuracy",
            "config": {
                "agents": ["clinical_reviewer", "medical_coder"],
                "topic_template": "Review medical terminology in: {redacted_documents}",
            },
        },
        {
            "id": "protocol_compliance",
            "type": "debate",
            "name": "Protocol Compliance Check",
            "description": "Verify adherence to clinical protocols",
            "config": {
                "agents": ["clinical_reviewer", "research_analyst_clinical"],
                "rounds": 2,
                "topic_template": "Check protocol compliance: {redacted_documents}",
            },
        },
        {
            "id": "quality_assessment",
            "type": "decision",
            "name": "Quality Gate",
            "description": "Determine if documentation meets quality standards",
            "config": {
                "condition": "quality_score >= 0.8",
                "true_target": "approval",
                "false_target": "revision_request",
            },
        },
        {
            "id": "revision_request",
            "type": "human_checkpoint",
            "name": "Request Revision",
            "description": "Request documentation revision from clinical staff",
            "config": {
                "approval_type": "revision",
                "notification_roles": ["clinical_staff", "physician"],
            },
        },
        {
            "id": "approval",
            "type": "human_checkpoint",
            "name": "Clinical Approval",
            "description": "Final clinical review and approval",
            "config": {
                "approval_type": "sign_off",
                "required_role": "attending_physician",
            },
        },
        {
            "id": "store",
            "type": "memory_write",
            "name": "Store Review Results",
            "description": "Persist review findings",
            "config": {
                "domain": "healthcare/clinical",
            },
        },
    ],
    "transitions": [
        {"from": "document_intake", "to": "phi_redaction"},
        {"from": "phi_redaction", "to": "terminology_check"},
        {"from": "terminology_check", "to": "protocol_compliance"},
        {"from": "protocol_compliance", "to": "quality_assessment"},
        {"from": "quality_assessment", "to": "revision_request", "condition": "quality_low"},
        {"from": "quality_assessment", "to": "approval", "condition": "quality_ok"},
        {"from": "revision_request", "to": "document_intake"},
        {"from": "approval", "to": "store"},
    ],
}

PHI_AUDIT_TEMPLATE: Dict[str, Any] = {
    "name": "PHI Access Audit",
    "description": "Audit trail review for PHI access compliance",
    "category": "healthcare",
    "version": "1.0",
    "tags": ["healthcare", "hipaa", "audit", "phi"],
    "steps": [
        {
            "id": "collect_logs",
            "type": "task",
            "name": "Collect Access Logs",
            "description": "Gather PHI access logs for audit period",
            "config": {
                "task_type": "aggregate",
                "sources": ["access_logs", "audit_trails", "authentication_logs"],
            },
        },
        {
            "id": "normalize",
            "type": "task",
            "name": "Normalize Log Data",
            "description": "Standardize log format for analysis",
            "config": {
                "task_type": "transform",
            },
        },
        {
            "id": "anomaly_detection",
            "type": "task",
            "name": "Anomaly Detection",
            "description": "Detect unusual access patterns",
            "config": {
                "task_type": "function",
                "function_name": "detect_phi_access_anomalies",
            },
        },
        {
            "id": "minimum_necessary",
            "type": "debate",
            "name": "Minimum Necessary Review",
            "description": "Assess compliance with minimum necessary standard",
            "config": {
                "agents": ["hipaa_auditor", "compliance_officer"],
                "topic_template": "Review minimum necessary compliance: {anomalies}",
            },
        },
        {
            "id": "violation_assessment",
            "type": "decision",
            "name": "Violation Assessment",
            "description": "Determine if violations occurred",
            "config": {
                "condition": "potential_violations_found",
                "true_target": "investigation",
                "false_target": "clean_report",
            },
        },
        {
            "id": "investigation",
            "type": "debate",
            "name": "Violation Investigation",
            "description": "Investigate potential HIPAA violations",
            "config": {
                "agents": ["hipaa_auditor", "compliance_officer", "forensic_accountant"],
                "rounds": 3,
            },
        },
        {
            "id": "privacy_officer_review",
            "type": "human_checkpoint",
            "name": "Privacy Officer Review",
            "description": "Privacy Officer review of findings",
            "config": {
                "approval_type": "review",
                "required_role": "privacy_officer",
                "checklist": [
                    "Verify investigation completeness",
                    "Assess breach notification requirements",
                    "Determine corrective actions",
                ],
            },
        },
        {
            "id": "clean_report",
            "type": "task",
            "name": "Generate Clean Audit Report",
            "description": "Generate audit report with no violations",
            "config": {
                "task_type": "transform",
                "template": "phi_audit_clean_report",
            },
        },
        {
            "id": "violation_report",
            "type": "task",
            "name": "Generate Violation Report",
            "description": "Generate detailed violation report",
            "config": {
                "task_type": "transform",
                "template": "phi_audit_violation_report",
            },
        },
        {
            "id": "archive",
            "type": "memory_write",
            "name": "Archive Audit",
            "description": "Archive audit for compliance records",
            "config": {
                "domain": "healthcare/audit",
                "retention_years": 6,
            },
        },
    ],
    "transitions": [
        {"from": "collect_logs", "to": "normalize"},
        {"from": "normalize", "to": "anomaly_detection"},
        {"from": "anomaly_detection", "to": "minimum_necessary"},
        {"from": "minimum_necessary", "to": "violation_assessment"},
        {"from": "violation_assessment", "to": "investigation", "condition": "violations_found"},
        {"from": "violation_assessment", "to": "clean_report", "condition": "no_violations"},
        {"from": "investigation", "to": "privacy_officer_review"},
        {"from": "privacy_officer_review", "to": "violation_report", "condition": "confirmed"},
        {"from": "privacy_officer_review", "to": "clean_report", "condition": "dismissed"},
        {"from": "clean_report", "to": "archive"},
        {"from": "violation_report", "to": "archive"},
    ],
}
