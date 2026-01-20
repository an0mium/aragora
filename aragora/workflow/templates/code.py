"""
Code Review Workflow Templates.

Templates for security audits, architecture reviews, and code quality analysis.
"""

from typing import Dict, Any

SECURITY_AUDIT_TEMPLATE: Dict[str, Any] = {
    "name": "Security Code Audit",
    "description": "Comprehensive security audit with OWASP Top 10 coverage",
    "category": "code",
    "version": "1.0",
    "tags": ["code", "security", "owasp", "audit"],
    "steps": [
        {
            "id": "scan_setup",
            "type": "task",
            "name": "Configure Security Scans",
            "description": "Set up SAST, DAST, and SCA scanning",
            "config": {
                "task_type": "function",
                "function_name": "configure_security_scans",
            },
        },
        {
            "id": "static_analysis",
            "type": "task",
            "name": "Static Analysis",
            "description": "Run static application security testing",
            "config": {
                "task_type": "function",
                "function_name": "run_sast_scan",
            },
        },
        {
            "id": "dependency_scan",
            "type": "task",
            "name": "Dependency Scanning",
            "description": "Scan dependencies for known vulnerabilities",
            "config": {
                "task_type": "function",
                "function_name": "run_sca_scan",
            },
        },
        {
            "id": "owasp_review",
            "type": "debate",
            "name": "OWASP Top 10 Review",
            "description": "Multi-agent review against OWASP Top 10",
            "config": {
                "agents": [
                    "code_security_specialist",
                    "security_engineer",
                    "architecture_reviewer",
                ],
                "rounds": 3,
                "topic_template": "Review code against OWASP Top 10: {scan_results}",
            },
        },
        {
            "id": "injection_review",
            "type": "debate",
            "name": "Injection Vulnerability Review",
            "description": "Deep dive on injection vulnerabilities",
            "config": {
                "agents": ["code_security_specialist", "api_design_reviewer"],
                "topic_template": "Analyze injection risks in: {code_paths}",
            },
        },
        {
            "id": "auth_review",
            "type": "debate",
            "name": "Authentication Review",
            "description": "Review authentication and session management",
            "config": {
                "agents": ["security_engineer", "code_security_specialist"],
                "topic_template": "Review auth implementation: {auth_code}",
            },
        },
        {
            "id": "crypto_review",
            "type": "debate",
            "name": "Cryptography Review",
            "description": "Review cryptographic implementations",
            "config": {
                "agents": ["code_security_specialist", "pci_dss"],
                "topic_template": "Review crypto usage: {crypto_code}",
            },
        },
        {
            "id": "severity_classification",
            "type": "task",
            "name": "Classify Findings",
            "description": "Classify vulnerabilities by severity",
            "config": {
                "task_type": "function",
                "function_name": "classify_vulnerabilities_cvss",
            },
        },
        {
            "id": "critical_gate",
            "type": "decision",
            "name": "Critical Findings Gate",
            "description": "Check for critical vulnerabilities",
            "config": {
                "condition": "critical_count > 0",
                "true_target": "security_lead_review",
                "false_target": "generate_report",
            },
        },
        {
            "id": "security_lead_review",
            "type": "human_checkpoint",
            "name": "Security Lead Review",
            "description": "Security lead review of critical findings",
            "config": {
                "approval_type": "review",
                "required_role": "security_lead",
                "checklist": [
                    "Verify critical findings",
                    "Assess exploitability",
                    "Confirm remediation urgency",
                ],
                "notification_channels": ["security_team", "engineering_leads"],
            },
        },
        {
            "id": "generate_report",
            "type": "task",
            "name": "Generate Security Report",
            "description": "Generate comprehensive security audit report",
            "config": {
                "task_type": "transform",
                "template": "security_audit_report",
            },
        },
        {
            "id": "create_tickets",
            "type": "task",
            "name": "Create Remediation Tickets",
            "description": "Create tickets for vulnerability remediation",
            "config": {
                "task_type": "http",
                "endpoint": "/api/tickets/create-batch",
            },
        },
        {
            "id": "store_findings",
            "type": "memory_write",
            "name": "Store Audit Findings",
            "description": "Persist findings to security knowledge base",
            "config": {
                "domain": "technical/security",
            },
        },
    ],
    "transitions": [
        {"from": "scan_setup", "to": "static_analysis"},
        {"from": "static_analysis", "to": "dependency_scan"},
        {"from": "dependency_scan", "to": "owasp_review"},
        {"from": "owasp_review", "to": "injection_review"},
        {"from": "injection_review", "to": "auth_review"},
        {"from": "auth_review", "to": "crypto_review"},
        {"from": "crypto_review", "to": "severity_classification"},
        {"from": "severity_classification", "to": "critical_gate"},
        {"from": "critical_gate", "to": "security_lead_review", "condition": "has_critical"},
        {"from": "critical_gate", "to": "generate_report", "condition": "no_critical"},
        {"from": "security_lead_review", "to": "generate_report"},
        {"from": "generate_report", "to": "create_tickets"},
        {"from": "create_tickets", "to": "store_findings"},
    ],
}

ARCHITECTURE_REVIEW_TEMPLATE: Dict[str, Any] = {
    "name": "Architecture Review",
    "description": "System architecture review with scalability and resilience analysis",
    "category": "code",
    "version": "1.0",
    "tags": ["code", "architecture", "design", "scalability"],
    "steps": [
        {
            "id": "gather_artifacts",
            "type": "task",
            "name": "Gather Architecture Artifacts",
            "description": "Collect architecture diagrams, ADRs, and documentation",
            "config": {
                "task_type": "aggregate",
                "sources": ["diagrams", "adrs", "documentation"],
            },
        },
        {
            "id": "component_analysis",
            "type": "debate",
            "name": "Component Analysis",
            "description": "Analyze system components and boundaries",
            "config": {
                "agents": ["architecture_reviewer", "data_architect"],
                "rounds": 2,
                "topic_template": "Analyze system components: {architecture_docs}",
            },
        },
        {
            "id": "coupling_review",
            "type": "debate",
            "name": "Coupling and Cohesion Review",
            "description": "Assess component coupling and cohesion",
            "config": {
                "agents": ["architecture_reviewer", "code_quality_reviewer"],
                "topic_template": "Review coupling patterns: {component_analysis}",
            },
        },
        {
            "id": "scalability_analysis",
            "type": "debate",
            "name": "Scalability Analysis",
            "description": "Analyze horizontal and vertical scaling capabilities",
            "config": {
                "agents": ["architecture_reviewer", "performance_engineer", "devops_engineer"],
                "rounds": 2,
                "topic_template": "Analyze scalability: {architecture_docs}",
            },
        },
        {
            "id": "resilience_review",
            "type": "debate",
            "name": "Resilience Review",
            "description": "Review fault tolerance and disaster recovery",
            "config": {
                "agents": ["architecture_reviewer", "devops_engineer", "security_engineer"],
                "topic_template": "Review resilience patterns: {architecture_docs}",
            },
        },
        {
            "id": "data_flow_analysis",
            "type": "debate",
            "name": "Data Flow Analysis",
            "description": "Analyze data flow and consistency patterns",
            "config": {
                "agents": ["data_architect", "architecture_reviewer"],
                "topic_template": "Analyze data flows: {architecture_docs}",
            },
        },
        {
            "id": "technical_debt",
            "type": "debate",
            "name": "Technical Debt Assessment",
            "description": "Identify and assess technical debt",
            "config": {
                "agents": ["architecture_reviewer", "code_quality_reviewer"],
                "topic_template": "Assess technical debt in architecture",
            },
        },
        {
            "id": "recommendations",
            "type": "debate",
            "name": "Generate Recommendations",
            "description": "Synthesize findings into recommendations",
            "config": {
                "agents": ["architecture_reviewer", "data_architect", "performance_engineer"],
                "rounds": 2,
            },
        },
        {
            "id": "architect_review",
            "type": "human_checkpoint",
            "name": "Principal Architect Review",
            "description": "Principal architect sign-off",
            "config": {
                "approval_type": "sign_off",
                "required_role": "principal_architect",
            },
        },
        {
            "id": "generate_adr",
            "type": "task",
            "name": "Generate ADRs",
            "description": "Generate Architecture Decision Records",
            "config": {
                "task_type": "transform",
                "template": "adr_template",
            },
        },
        {
            "id": "store",
            "type": "memory_write",
            "name": "Store Review",
            "description": "Persist architecture review",
            "config": {
                "domain": "technical/architecture",
            },
        },
    ],
    "transitions": [
        {"from": "gather_artifacts", "to": "component_analysis"},
        {"from": "component_analysis", "to": "coupling_review"},
        {"from": "coupling_review", "to": "scalability_analysis"},
        {"from": "scalability_analysis", "to": "resilience_review"},
        {"from": "resilience_review", "to": "data_flow_analysis"},
        {"from": "data_flow_analysis", "to": "technical_debt"},
        {"from": "technical_debt", "to": "recommendations"},
        {"from": "recommendations", "to": "architect_review"},
        {"from": "architect_review", "to": "generate_adr", "condition": "approved"},
        {"from": "generate_adr", "to": "store"},
    ],
}

CODE_QUALITY_TEMPLATE: Dict[str, Any] = {
    "name": "Code Quality Review",
    "description": "Comprehensive code quality analysis with automated and manual review",
    "category": "code",
    "version": "1.0",
    "tags": ["code", "quality", "review", "maintainability"],
    "steps": [
        {
            "id": "lint_check",
            "type": "task",
            "name": "Run Linters",
            "description": "Execute code linters and formatters",
            "config": {
                "task_type": "function",
                "function_name": "run_linters",
            },
        },
        {
            "id": "type_check",
            "type": "task",
            "name": "Type Checking",
            "description": "Run static type analysis",
            "config": {
                "task_type": "function",
                "function_name": "run_type_checker",
            },
        },
        {
            "id": "complexity_analysis",
            "type": "task",
            "name": "Complexity Analysis",
            "description": "Analyze cyclomatic and cognitive complexity",
            "config": {
                "task_type": "function",
                "function_name": "analyze_complexity",
            },
        },
        {
            "id": "test_coverage",
            "type": "task",
            "name": "Test Coverage Analysis",
            "description": "Analyze test coverage metrics",
            "config": {
                "task_type": "function",
                "function_name": "analyze_coverage",
            },
        },
        {
            "id": "readability_review",
            "type": "debate",
            "name": "Readability Review",
            "description": "Multi-agent code readability assessment",
            "config": {
                "agents": ["code_quality_reviewer", "claude", "deepseek"],
                "rounds": 2,
                "topic_template": "Review code readability: {code_sample}",
            },
        },
        {
            "id": "pattern_review",
            "type": "debate",
            "name": "Design Patterns Review",
            "description": "Review design pattern usage and anti-patterns",
            "config": {
                "agents": ["code_quality_reviewer", "architecture_reviewer"],
                "topic_template": "Review design patterns: {code_sample}",
            },
        },
        {
            "id": "documentation_check",
            "type": "debate",
            "name": "Documentation Review",
            "description": "Review code documentation quality",
            "config": {
                "agents": ["code_quality_reviewer", "api_design_reviewer"],
                "topic_template": "Review documentation: {code_docs}",
            },
        },
        {
            "id": "quality_gate",
            "type": "decision",
            "name": "Quality Gate",
            "description": "Check if code meets quality thresholds",
            "config": {
                "condition": "quality_score >= quality_threshold",
                "true_target": "approval",
                "false_target": "improvement_suggestions",
            },
        },
        {
            "id": "improvement_suggestions",
            "type": "debate",
            "name": "Generate Improvements",
            "description": "Generate specific improvement suggestions",
            "config": {
                "agents": [
                    "code_quality_reviewer",
                    "architecture_reviewer",
                    "performance_engineer",
                ],
                "topic_template": "Suggest improvements for: {quality_issues}",
            },
        },
        {
            "id": "developer_feedback",
            "type": "human_checkpoint",
            "name": "Developer Feedback",
            "description": "Present findings to development team",
            "config": {
                "approval_type": "review",
                "notification_roles": ["developer", "tech_lead"],
            },
        },
        {
            "id": "approval",
            "type": "human_checkpoint",
            "name": "Tech Lead Approval",
            "description": "Tech lead sign-off on code quality",
            "config": {
                "approval_type": "sign_off",
                "required_role": "tech_lead",
            },
        },
        {
            "id": "generate_report",
            "type": "task",
            "name": "Generate Quality Report",
            "description": "Generate code quality report",
            "config": {
                "task_type": "transform",
                "template": "code_quality_report",
            },
        },
        {
            "id": "store",
            "type": "memory_write",
            "name": "Store Metrics",
            "description": "Store quality metrics for trending",
            "config": {
                "domain": "technical/quality",
            },
        },
    ],
    "transitions": [
        {"from": "lint_check", "to": "type_check"},
        {"from": "type_check", "to": "complexity_analysis"},
        {"from": "complexity_analysis", "to": "test_coverage"},
        {"from": "test_coverage", "to": "readability_review"},
        {"from": "readability_review", "to": "pattern_review"},
        {"from": "pattern_review", "to": "documentation_check"},
        {"from": "documentation_check", "to": "quality_gate"},
        {"from": "quality_gate", "to": "improvement_suggestions", "condition": "below_threshold"},
        {"from": "quality_gate", "to": "approval", "condition": "meets_threshold"},
        {"from": "improvement_suggestions", "to": "developer_feedback"},
        {"from": "developer_feedback", "to": "lint_check"},  # Loop back for re-review
        {"from": "approval", "to": "generate_report"},
        {"from": "generate_report", "to": "store"},
    ],
}
