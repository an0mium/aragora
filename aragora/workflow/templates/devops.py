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

# =============================================================================
# PagerDuty Incident Management Workflow
# =============================================================================

PAGERDUTY_INCIDENT_WORKFLOW_TEMPLATE: Dict[str, Any] = {
    "name": "PagerDuty Incident Management",
    "description": "End-to-end incident management with PagerDuty integration",
    "category": "devops",
    "version": "1.0",
    "tags": ["devops", "incident", "pagerduty", "alerting", "on-call"],
    "connectors": ["pagerduty"],
    "steps": [
        {
            "id": "receive_alert",
            "type": "task",
            "name": "Receive Alert",
            "description": "Receive and parse incoming alert/finding",
            "config": {
                "task_type": "transform",
                "input_key": "alert_data",
                "output_key": "parsed_alert",
            },
        },
        {
            "id": "severity_assessment",
            "type": "debate",
            "name": "Severity Assessment",
            "description": "Multi-agent assessment of incident severity",
            "config": {
                "agents": ["sre", "security_engineer", "devops_engineer"],
                "rounds": 1,
                "topic_template": "Assess severity of alert: {parsed_alert}",
            },
        },
        {
            "id": "urgency_decision",
            "type": "decision",
            "name": "Determine Urgency",
            "description": "Route based on urgency level",
            "config": {
                "condition": "assessed_severity in ['critical', 'high']",
                "true_target": "create_high_urgency_incident",
                "false_target": "create_low_urgency_incident",
            },
        },
        {
            "id": "create_high_urgency_incident",
            "type": "connector",
            "name": "Create High Urgency Incident",
            "description": "Create PagerDuty incident with high urgency",
            "config": {
                "connector": "pagerduty",
                "operation": "create_incident",
                "params": {
                    "title": "{parsed_alert.title}",
                    "service_id": "{service_id}",
                    "urgency": "high",
                    "body": "{parsed_alert.description}",
                    "escalation_policy_id": "{escalation_policy}",
                },
                "output_key": "incident",
            },
        },
        {
            "id": "create_low_urgency_incident",
            "type": "connector",
            "name": "Create Low Urgency Incident",
            "description": "Create PagerDuty incident with low urgency",
            "config": {
                "connector": "pagerduty",
                "operation": "create_incident",
                "params": {
                    "title": "{parsed_alert.title}",
                    "service_id": "{service_id}",
                    "urgency": "low",
                    "body": "{parsed_alert.description}",
                },
                "output_key": "incident",
            },
        },
        {
            "id": "get_on_call",
            "type": "connector",
            "name": "Get On-Call Responder",
            "description": "Identify current on-call responder",
            "config": {
                "connector": "pagerduty",
                "operation": "get_on_call",
                "params": {
                    "schedule_ids": "{on_call_schedules}",
                },
                "output_key": "on_call_responder",
            },
        },
        {
            "id": "add_context_note",
            "type": "connector",
            "name": "Add Context Note",
            "description": "Add AI-generated context note to incident",
            "config": {
                "connector": "pagerduty",
                "operation": "add_note",
                "params": {
                    "incident_id": "{incident.id}",
                    "content": "AI Analysis:\n{severity_assessment_result}\n\nSuggested Actions:\n{suggested_actions}",
                },
            },
        },
        {
            "id": "root_cause_investigation",
            "type": "debate",
            "name": "Root Cause Investigation",
            "description": "Multi-agent root cause analysis",
            "config": {
                "agents": ["sre", "devops_engineer", "architect"],
                "rounds": 3,
                "topic_template": "Investigate root cause: {incident.title}\nContext: {incident_logs} {metrics}",
            },
        },
        {
            "id": "add_rca_note",
            "type": "connector",
            "name": "Add RCA Note",
            "description": "Document root cause analysis findings",
            "config": {
                "connector": "pagerduty",
                "operation": "add_note",
                "params": {
                    "incident_id": "{incident.id}",
                    "content": "Root Cause Analysis:\n{rca_findings}",
                },
            },
        },
        {
            "id": "mitigation_planning",
            "type": "debate",
            "name": "Mitigation Planning",
            "description": "Generate mitigation strategies",
            "config": {
                "agents": ["sre", "devops_engineer", "security_engineer"],
                "rounds": 2,
                "topic_template": "Generate mitigation plan for: {rca_findings}",
            },
        },
        {
            "id": "human_approval",
            "type": "human_checkpoint",
            "name": "Mitigation Approval",
            "description": "Incident commander approval for mitigation",
            "config": {
                "approval_type": "sign_off",
                "required_role": "incident_commander",
                "checklist": [
                    "Root cause identified and documented",
                    "Mitigation plan reviewed",
                    "Rollback plan available",
                    "Customer communication prepared",
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
            "description": "Verify incident is resolved via health checks",
            "config": {
                "task_type": "function",
                "function_name": "run_health_checks",
                "output_key": "health_status",
            },
        },
        {
            "id": "resolution_check",
            "type": "decision",
            "name": "Resolution Check",
            "description": "Verify incident is fully resolved",
            "config": {
                "condition": "health_status.all_healthy == True",
                "true_target": "resolve_incident",
                "false_target": "root_cause_investigation",
            },
        },
        {
            "id": "resolve_incident",
            "type": "connector",
            "name": "Resolve PagerDuty Incident",
            "description": "Mark incident as resolved in PagerDuty",
            "config": {
                "connector": "pagerduty",
                "operation": "resolve",
                "params": {
                    "incident_id": "{incident.id}",
                    "resolution_note": "Resolved via mitigation plan.\n\nRoot Cause: {rca_findings}\nResolution: {mitigation_summary}",
                },
            },
        },
        {
            "id": "postmortem_generation",
            "type": "debate",
            "name": "Generate Postmortem",
            "description": "Create blameless postmortem document",
            "config": {
                "agents": ["sre", "technical_writer", "architect"],
                "rounds": 2,
                "topic_template": "Generate postmortem for incident {incident.incident_number}: {incident_timeline}",
            },
        },
        {
            "id": "action_items",
            "type": "debate",
            "name": "Generate Action Items",
            "description": "Identify follow-up action items",
            "config": {
                "agents": ["sre", "product_manager", "engineering_lead"],
                "rounds": 1,
                "topic_template": "Generate action items from postmortem",
            },
        },
        {
            "id": "store_incident",
            "type": "memory_write",
            "name": "Archive Incident Record",
            "description": "Store incident data in knowledge base",
            "config": {
                "domain": "devops/incidents",
                "metadata": {
                    "pagerduty_id": "{incident.id}",
                    "incident_number": "{incident.incident_number}",
                    "severity": "{assessed_severity}",
                    "mttr_minutes": "{resolution_time_minutes}",
                },
            },
        },
    ],
    "transitions": [
        {"from": "receive_alert", "to": "severity_assessment"},
        {"from": "severity_assessment", "to": "urgency_decision"},
        {
            "from": "urgency_decision",
            "to": "create_high_urgency_incident",
            "condition": "high_urgency",
        },
        {
            "from": "urgency_decision",
            "to": "create_low_urgency_incident",
            "condition": "low_urgency",
        },
        {"from": "create_high_urgency_incident", "to": "get_on_call"},
        {"from": "create_low_urgency_incident", "to": "get_on_call"},
        {"from": "get_on_call", "to": "add_context_note"},
        {"from": "add_context_note", "to": "root_cause_investigation"},
        {"from": "root_cause_investigation", "to": "add_rca_note"},
        {"from": "add_rca_note", "to": "mitigation_planning"},
        {"from": "mitigation_planning", "to": "human_approval"},
        {"from": "human_approval", "to": "execute_mitigation", "condition": "approved"},
        {"from": "execute_mitigation", "to": "verify_resolution"},
        {"from": "verify_resolution", "to": "resolution_check"},
        {"from": "resolution_check", "to": "root_cause_investigation", "condition": "not_resolved"},
        {"from": "resolution_check", "to": "resolve_incident", "condition": "resolved"},
        {"from": "resolve_incident", "to": "postmortem_generation"},
        {"from": "postmortem_generation", "to": "action_items"},
        {"from": "action_items", "to": "store_incident"},
    ],
}
