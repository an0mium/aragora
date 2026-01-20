"""
DevOps Workflow Templates.

Templates for CI/CD pipelines, infrastructure review, and incident response.
"""

from typing import Any, Dict

CICD_PIPELINE_REVIEW_TEMPLATE: Dict[str, Any] = {
    "name": "CI/CD Pipeline Review",
    "description": "Comprehensive CI/CD pipeline security and reliability review",
    "category": "devops",
    "version": "1.0",
    "tags": ["devops", "cicd", "security", "automation"],
    "steps": [
        {
            "id": "gather_configs",
            "type": "task",
            "name": "Gather Pipeline Configs",
            "description": "Collect pipeline configurations and secrets management",
            "config": {
                "task_type": "aggregate",
                "sources": ["pipeline_yaml", "secrets_config", "env_vars"],
            },
        },
        {
            "id": "security_review",
            "type": "debate",
            "name": "Pipeline Security Review",
            "description": "Review pipeline for security vulnerabilities",
            "config": {
                "agents": ["devops_engineer", "security_engineer"],
                "rounds": 2,
                "topic_template": "Review pipeline security: {pipeline_config}",
            },
        },
        {
            "id": "secrets_audit",
            "type": "debate",
            "name": "Secrets Management Audit",
            "description": "Audit secrets handling and rotation",
            "config": {
                "agents": ["security_engineer", "devops_engineer"],
                "topic_template": "Audit secrets management: {secrets_config}",
            },
        },
        {
            "id": "dependency_scan",
            "type": "task",
            "name": "Dependency Security Scan",
            "description": "Scan pipeline dependencies for vulnerabilities",
            "config": {
                "task_type": "function",
                "function_name": "scan_pipeline_dependencies",
            },
        },
        {
            "id": "performance_review",
            "type": "debate",
            "name": "Pipeline Performance Review",
            "description": "Review build times and optimization opportunities",
            "config": {
                "agents": ["devops_engineer", "performance_engineer"],
                "topic_template": "Review pipeline performance",
            },
        },
        {
            "id": "rollback_review",
            "type": "debate",
            "name": "Rollback Mechanism Review",
            "description": "Review deployment rollback procedures",
            "config": {
                "agents": ["devops_engineer", "sre"],
                "topic_template": "Review rollback mechanisms",
            },
        },
        {
            "id": "observability_check",
            "type": "debate",
            "name": "Observability Check",
            "description": "Review logging, metrics, and alerting",
            "config": {
                "agents": ["sre", "devops_engineer"],
                "topic_template": "Review pipeline observability",
            },
        },
        {
            "id": "compliance_gate",
            "type": "decision",
            "name": "Security Compliance Gate",
            "description": "Check security compliance requirements",
            "config": {
                "condition": "security_score >= 0.8",
                "true_target": "sre_review",
                "false_target": "remediation_plan",
            },
        },
        {
            "id": "remediation_plan",
            "type": "debate",
            "name": "Security Remediation Plan",
            "description": "Generate security improvement recommendations",
            "config": {
                "agents": ["security_engineer", "devops_engineer"],
                "topic_template": "Generate remediation plan for: {security_findings}",
            },
        },
        {
            "id": "sre_review",
            "type": "human_checkpoint",
            "name": "SRE Team Review",
            "description": "Site Reliability Engineering team approval",
            "config": {
                "approval_type": "sign_off",
                "required_role": "sre_lead",
                "checklist": [
                    "Security controls verified",
                    "Rollback tested",
                    "Monitoring configured",
                    "On-call procedures documented",
                ],
            },
        },
        {
            "id": "generate_report",
            "type": "task",
            "name": "Generate Review Report",
            "description": "Generate CI/CD review report",
            "config": {
                "task_type": "transform",
                "template": "cicd_review_report",
            },
        },
        {
            "id": "store",
            "type": "memory_write",
            "name": "Store Review",
            "description": "Persist review to knowledge base",
            "config": {
                "domain": "devops/cicd",
            },
        },
    ],
    "transitions": [
        {"from": "gather_configs", "to": "security_review"},
        {"from": "security_review", "to": "secrets_audit"},
        {"from": "secrets_audit", "to": "dependency_scan"},
        {"from": "dependency_scan", "to": "performance_review"},
        {"from": "performance_review", "to": "rollback_review"},
        {"from": "rollback_review", "to": "observability_check"},
        {"from": "observability_check", "to": "compliance_gate"},
        {"from": "compliance_gate", "to": "remediation_plan", "condition": "needs_remediation"},
        {"from": "compliance_gate", "to": "sre_review", "condition": "compliant"},
        {"from": "remediation_plan", "to": "gather_configs"},
        {"from": "sre_review", "to": "generate_report"},
        {"from": "generate_report", "to": "store"},
    ],
}

INCIDENT_RESPONSE_TEMPLATE: Dict[str, Any] = {
    "name": "Incident Response Workflow",
    "description": "Structured incident response with multi-agent analysis",
    "category": "devops",
    "version": "1.0",
    "tags": ["devops", "incident", "response", "sre"],
    "steps": [
        {
            "id": "incident_triage",
            "type": "task",
            "name": "Incident Triage",
            "description": "Collect and classify incident information",
            "config": {
                "task_type": "function",
                "function_name": "triage_incident",
            },
        },
        {
            "id": "severity_classification",
            "type": "debate",
            "name": "Severity Classification",
            "description": "Multi-agent severity and impact assessment",
            "config": {
                "agents": ["sre", "devops_engineer", "security_engineer"],
                "rounds": 1,
                "topic_template": "Classify incident severity: {incident_details}",
            },
        },
        {
            "id": "severity_gate",
            "type": "decision",
            "name": "Severity Gate",
            "description": "Route based on severity level",
            "config": {
                "condition": "severity in ['sev1', 'sev2']",
                "true_target": "war_room",
                "false_target": "root_cause_analysis",
            },
        },
        {
            "id": "war_room",
            "type": "human_checkpoint",
            "name": "Activate War Room",
            "description": "Notify stakeholders and activate war room",
            "config": {
                "approval_type": "acknowledge",
                "notification_channels": [
                    "incident_response_team",
                    "engineering_leads",
                    "executives",
                ],
                "urgent": True,
            },
        },
        {
            "id": "root_cause_analysis",
            "type": "debate",
            "name": "Root Cause Analysis",
            "description": "Multi-agent root cause investigation",
            "config": {
                "agents": ["sre", "devops_engineer", "architect"],
                "rounds": 3,
                "topic_template": "Investigate root cause: {incident_logs} {metrics}",
            },
        },
        {
            "id": "mitigation_options",
            "type": "debate",
            "name": "Mitigation Options",
            "description": "Generate and evaluate mitigation strategies",
            "config": {
                "agents": ["sre", "devops_engineer", "architect"],
                "rounds": 2,
                "topic_template": "Generate mitigation options for: {root_cause}",
            },
        },
        {
            "id": "mitigation_approval",
            "type": "human_checkpoint",
            "name": "Mitigation Approval",
            "description": "Incident commander approval for mitigation",
            "config": {
                "approval_type": "sign_off",
                "required_role": "incident_commander",
                "checklist": [
                    "Root cause identified",
                    "Mitigation tested in staging",
                    "Rollback plan ready",
                    "Communication prepared",
                ],
            },
        },
        {
            "id": "execute_mitigation",
            "type": "task",
            "name": "Execute Mitigation",
            "description": "Apply mitigation steps",
            "config": {
                "task_type": "function",
                "function_name": "execute_mitigation_plan",
            },
        },
        {
            "id": "verify_resolution",
            "type": "task",
            "name": "Verify Resolution",
            "description": "Verify incident is resolved",
            "config": {
                "task_type": "function",
                "function_name": "verify_resolution",
            },
        },
        {
            "id": "resolution_gate",
            "type": "decision",
            "name": "Resolution Gate",
            "description": "Check if incident is fully resolved",
            "config": {
                "condition": "incident_resolved == True",
                "true_target": "postmortem_prep",
                "false_target": "root_cause_analysis",
            },
        },
        {
            "id": "postmortem_prep",
            "type": "debate",
            "name": "Postmortem Preparation",
            "description": "Prepare blameless postmortem",
            "config": {
                "agents": ["sre", "devops_engineer", "technical_writer"],
                "rounds": 2,
                "topic_template": "Prepare postmortem: {incident_timeline} {root_cause} {mitigation}",
            },
        },
        {
            "id": "action_items",
            "type": "debate",
            "name": "Generate Action Items",
            "description": "Generate follow-up action items",
            "config": {
                "agents": ["sre", "architect", "product_manager"],
                "topic_template": "Generate action items from postmortem",
            },
        },
        {
            "id": "create_tickets",
            "type": "task",
            "name": "Create Action Tickets",
            "description": "Create tickets for action items",
            "config": {
                "task_type": "http",
                "endpoint": "/api/tickets/create-batch",
            },
        },
        {
            "id": "store",
            "type": "memory_write",
            "name": "Store Incident Record",
            "description": "Persist incident to knowledge base",
            "config": {
                "domain": "devops/incidents",
            },
        },
    ],
    "transitions": [
        {"from": "incident_triage", "to": "severity_classification"},
        {"from": "severity_classification", "to": "severity_gate"},
        {"from": "severity_gate", "to": "war_room", "condition": "critical"},
        {"from": "severity_gate", "to": "root_cause_analysis", "condition": "non_critical"},
        {"from": "war_room", "to": "root_cause_analysis"},
        {"from": "root_cause_analysis", "to": "mitigation_options"},
        {"from": "mitigation_options", "to": "mitigation_approval"},
        {"from": "mitigation_approval", "to": "execute_mitigation"},
        {"from": "execute_mitigation", "to": "verify_resolution"},
        {"from": "verify_resolution", "to": "resolution_gate"},
        {"from": "resolution_gate", "to": "root_cause_analysis", "condition": "not_resolved"},
        {"from": "resolution_gate", "to": "postmortem_prep", "condition": "resolved"},
        {"from": "postmortem_prep", "to": "action_items"},
        {"from": "action_items", "to": "create_tickets"},
        {"from": "create_tickets", "to": "store"},
    ],
}

INFRASTRUCTURE_AUDIT_TEMPLATE: Dict[str, Any] = {
    "name": "Infrastructure Security Audit",
    "description": "Cloud infrastructure security and compliance audit",
    "category": "devops",
    "version": "1.0",
    "tags": ["devops", "infrastructure", "security", "cloud"],
    "steps": [
        {
            "id": "inventory",
            "type": "task",
            "name": "Resource Inventory",
            "description": "Gather cloud resource inventory",
            "config": {
                "task_type": "function",
                "function_name": "get_cloud_inventory",
            },
        },
        {
            "id": "network_review",
            "type": "debate",
            "name": "Network Security Review",
            "description": "Review network configuration and segmentation",
            "config": {
                "agents": ["security_engineer", "devops_engineer"],
                "rounds": 2,
                "topic_template": "Review network security: {network_config}",
            },
        },
        {
            "id": "iam_review",
            "type": "debate",
            "name": "IAM Policy Review",
            "description": "Review identity and access management",
            "config": {
                "agents": ["security_engineer", "compliance_officer"],
                "rounds": 2,
                "topic_template": "Review IAM policies: {iam_config}",
            },
        },
        {
            "id": "encryption_review",
            "type": "debate",
            "name": "Encryption Review",
            "description": "Review data encryption at rest and in transit",
            "config": {
                "agents": ["security_engineer", "pci_dss"],
                "topic_template": "Review encryption: {encryption_config}",
            },
        },
        {
            "id": "logging_review",
            "type": "debate",
            "name": "Logging and Monitoring Review",
            "description": "Review audit logging and monitoring",
            "config": {
                "agents": ["sre", "security_engineer"],
                "topic_template": "Review logging: {logging_config}",
            },
        },
        {
            "id": "compliance_check",
            "type": "gauntlet",
            "name": "Compliance Check",
            "description": "Run compliance checks against standards",
            "config": {
                "profile": "security",
                "input_type": "architecture",
            },
        },
        {
            "id": "ciso_review",
            "type": "human_checkpoint",
            "name": "CISO Review",
            "description": "Chief Information Security Officer approval",
            "config": {
                "approval_type": "sign_off",
                "required_role": "ciso",
                "checklist": [
                    "Network segmentation adequate",
                    "Least privilege enforced",
                    "Encryption standards met",
                    "Audit logging complete",
                ],
            },
        },
        {
            "id": "generate_report",
            "type": "task",
            "name": "Generate Audit Report",
            "description": "Generate infrastructure audit report",
            "config": {
                "task_type": "transform",
                "template": "infrastructure_audit_report",
            },
        },
        {
            "id": "store",
            "type": "memory_write",
            "name": "Store Audit",
            "description": "Persist audit to knowledge base",
            "config": {
                "domain": "devops/audits",
            },
        },
    ],
    "transitions": [
        {"from": "inventory", "to": "network_review"},
        {"from": "network_review", "to": "iam_review"},
        {"from": "iam_review", "to": "encryption_review"},
        {"from": "encryption_review", "to": "logging_review"},
        {"from": "logging_review", "to": "compliance_check"},
        {"from": "compliance_check", "to": "ciso_review"},
        {"from": "ciso_review", "to": "generate_report"},
        {"from": "generate_report", "to": "store"},
    ],
}
