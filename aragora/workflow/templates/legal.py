"""
Legal Industry Workflow Templates.

Templates for legal document review and compliance workflows.
"""

from typing import Dict, Any

CONTRACT_REVIEW_TEMPLATE: Dict[str, Any] = {
    "name": "Contract Review",
    "description": "Multi-agent contract analysis with risk assessment and clause review",
    "category": "legal",
    "version": "1.0",
    "tags": ["legal", "contracts", "risk-assessment"],
    "steps": [
        {
            "id": "ingest",
            "type": "task",
            "name": "Document Ingestion",
            "description": "Extract and parse contract document",
            "config": {
                "task_type": "transform",
                "input_key": "document",
                "output_key": "parsed_contract",
            },
        },
        {
            "id": "classify",
            "type": "task",
            "name": "Contract Classification",
            "description": "Identify contract type and key characteristics",
            "config": {
                "task_type": "function",
                "function_name": "classify_contract",
            },
        },
        {
            "id": "extract_clauses",
            "type": "memory_read",
            "name": "Extract Key Clauses",
            "description": "Identify and extract important contract clauses",
            "config": {
                "query_template": "Extract key clauses from: {parsed_contract}",
                "domains": ["legal/contracts"],
            },
        },
        {
            "id": "risk_debate",
            "type": "debate",
            "name": "Risk Assessment Debate",
            "description": "Multi-agent debate on contract risks",
            "config": {
                "topic_template": "Analyze risks in this contract: {parsed_contract}",
                "agents": ["contract_analyst", "compliance_officer", "litigation_support"],
                "rounds": 3,
                "consensus_threshold": 0.7,
            },
        },
        {
            "id": "human_review",
            "type": "human_checkpoint",
            "name": "Legal Review",
            "description": "Senior attorney review of identified risks",
            "config": {
                "approval_type": "review",
                "checklist": [
                    "Verify risk categorization accuracy",
                    "Confirm suggested mitigations",
                    "Review negotiation recommendations",
                ],
                "timeout_hours": 48,
                "required_role": "senior_attorney",
            },
        },
        {
            "id": "generate_report",
            "type": "task",
            "name": "Generate Report",
            "description": "Create contract review report",
            "config": {
                "task_type": "transform",
                "template": "contract_review_report",
            },
        },
        {
            "id": "store_findings",
            "type": "memory_write",
            "name": "Store Analysis",
            "description": "Persist findings to knowledge base",
            "config": {
                "domain": "legal/contracts",
                "confidence": 0.85,
            },
        },
    ],
    "transitions": [
        {"from": "ingest", "to": "classify"},
        {"from": "classify", "to": "extract_clauses"},
        {"from": "extract_clauses", "to": "risk_debate"},
        {"from": "risk_debate", "to": "human_review"},
        {"from": "human_review", "to": "generate_report", "condition": "approved"},
        {"from": "human_review", "to": "risk_debate", "condition": "rejected"},
        {"from": "generate_report", "to": "store_findings"},
    ],
}

DUE_DILIGENCE_TEMPLATE: Dict[str, Any] = {
    "name": "Due Diligence Review",
    "description": "Comprehensive M&A due diligence workflow with multi-document analysis",
    "category": "legal",
    "version": "1.0",
    "tags": ["legal", "m&a", "due-diligence"],
    "steps": [
        {
            "id": "document_collection",
            "type": "task",
            "name": "Document Collection",
            "description": "Gather and organize due diligence documents",
            "config": {
                "task_type": "aggregate",
                "sources": ["contracts", "financials", "corporate_records"],
            },
        },
        {
            "id": "parallel_review",
            "type": "parallel",
            "name": "Parallel Document Reviews",
            "description": "Concurrent review of different document categories",
            "branches": [
                {
                    "id": "corporate_review",
                    "steps": [
                        {
                            "id": "corp_analysis",
                            "type": "debate",
                            "name": "Corporate Structure Analysis",
                            "config": {
                                "agents": ["m_and_a_counsel", "compliance_officer"],
                                "topic_template": "Analyze corporate structure: {corporate_docs}",
                            },
                        },
                    ],
                },
                {
                    "id": "contract_review",
                    "steps": [
                        {
                            "id": "material_contracts",
                            "type": "debate",
                            "name": "Material Contracts Review",
                            "config": {
                                "agents": ["contract_analyst", "m_and_a_counsel"],
                                "topic_template": "Review material contracts: {contract_docs}",
                            },
                        },
                    ],
                },
                {
                    "id": "ip_review",
                    "steps": [
                        {
                            "id": "ip_analysis",
                            "type": "debate",
                            "name": "IP Portfolio Analysis",
                            "config": {
                                "agents": ["contract_analyst", "litigation_support"],
                                "topic_template": "Analyze IP portfolio: {ip_docs}",
                            },
                        },
                    ],
                },
            ],
        },
        {
            "id": "risk_consolidation",
            "type": "debate",
            "name": "Risk Consolidation",
            "description": "Consolidate findings and assess overall risk profile",
            "config": {
                "agents": ["m_and_a_counsel", "compliance_officer", "contract_analyst"],
                "rounds": 2,
            },
        },
        {
            "id": "executive_summary",
            "type": "task",
            "name": "Executive Summary",
            "description": "Generate executive summary of due diligence findings",
            "config": {
                "task_type": "transform",
                "template": "dd_executive_summary",
            },
        },
        {
            "id": "partner_review",
            "type": "human_checkpoint",
            "name": "Partner Review",
            "description": "Managing partner review and sign-off",
            "config": {
                "approval_type": "sign_off",
                "required_role": "partner",
            },
        },
    ],
    "transitions": [
        {"from": "document_collection", "to": "parallel_review"},
        {"from": "parallel_review", "to": "risk_consolidation"},
        {"from": "risk_consolidation", "to": "executive_summary"},
        {"from": "executive_summary", "to": "partner_review"},
    ],
}

COMPLIANCE_AUDIT_TEMPLATE: Dict[str, Any] = {
    "name": "Compliance Audit",
    "description": "Regulatory compliance audit workflow with gap analysis",
    "category": "legal",
    "version": "1.0",
    "tags": ["legal", "compliance", "audit", "regulatory"],
    "steps": [
        {
            "id": "scope_definition",
            "type": "task",
            "name": "Define Audit Scope",
            "description": "Define regulatory frameworks and scope",
            "config": {
                "task_type": "validate",
                "validation_rules": ["framework_selected", "scope_defined"],
            },
        },
        {
            "id": "evidence_collection",
            "type": "memory_read",
            "name": "Collect Evidence",
            "description": "Gather compliance evidence from knowledge base",
            "config": {
                "query_template": "Find compliance evidence for {framework}",
                "domains": ["compliance/audit", "operational/policies"],
            },
        },
        {
            "id": "control_assessment",
            "type": "debate",
            "name": "Control Assessment",
            "description": "Multi-agent assessment of compliance controls",
            "config": {
                "agents": ["compliance_officer", "internal_auditor", "sox"],
                "rounds": 3,
                "topic_template": "Assess {framework} compliance controls: {evidence}",
            },
        },
        {
            "id": "gap_analysis",
            "type": "decision",
            "name": "Gap Analysis",
            "description": "Identify compliance gaps",
            "config": {
                "condition": "gaps_identified",
                "true_target": "remediation_planning",
                "false_target": "report_generation",
            },
        },
        {
            "id": "remediation_planning",
            "type": "debate",
            "name": "Remediation Planning",
            "description": "Develop remediation recommendations",
            "config": {
                "agents": ["compliance_officer", "internal_auditor"],
                "topic_template": "Plan remediation for gaps: {gaps}",
            },
        },
        {
            "id": "management_review",
            "type": "human_checkpoint",
            "name": "Management Review",
            "description": "Management review of audit findings",
            "config": {
                "approval_type": "review",
                "checklist": [
                    "Verify finding accuracy",
                    "Confirm risk ratings",
                    "Approve remediation timeline",
                ],
            },
        },
        {
            "id": "report_generation",
            "type": "task",
            "name": "Generate Audit Report",
            "description": "Generate formal compliance audit report",
            "config": {
                "task_type": "transform",
                "template": "compliance_audit_report",
            },
        },
        {
            "id": "archive",
            "type": "memory_write",
            "name": "Archive Findings",
            "description": "Archive audit findings for future reference",
            "config": {
                "domain": "compliance/audit",
                "retention_years": 7,
            },
        },
    ],
    "transitions": [
        {"from": "scope_definition", "to": "evidence_collection"},
        {"from": "evidence_collection", "to": "control_assessment"},
        {"from": "control_assessment", "to": "gap_analysis"},
        {"from": "gap_analysis", "to": "remediation_planning", "condition": "gaps_identified"},
        {"from": "gap_analysis", "to": "report_generation", "condition": "no_gaps"},
        {"from": "remediation_planning", "to": "management_review"},
        {"from": "management_review", "to": "report_generation", "condition": "approved"},
        {"from": "report_generation", "to": "archive"},
    ],
}
