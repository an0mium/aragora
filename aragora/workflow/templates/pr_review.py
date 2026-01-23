"""
Pull Request Review Workflow Templates.

Multi-agent workflows for reviewing pull requests:
- Code quality analysis
- Security vulnerability scanning
- Design pattern review
- Performance impact assessment
- Documentation completeness check

These workflows provide comprehensive PR reviews using multiple
specialized agents with consensus-based recommendations.
"""

from typing import Dict, Any

PR_REVIEW_TEMPLATE: Dict[str, Any] = {
    "name": "Pull Request Review",
    "description": "Comprehensive multi-agent PR review workflow",
    "category": "development",
    "version": "1.0",
    "tags": ["code", "review", "pr", "github", "quality"],
    "steps": [
        {
            "id": "fetch_pr_details",
            "type": "task",
            "name": "Fetch PR Details",
            "description": "Retrieve PR information from GitHub",
            "config": {
                "task_type": "http",
                "endpoint": "/api/github/pr/{pr_number}",
                "method": "GET",
            },
        },
        {
            "id": "analyze_changes",
            "type": "task",
            "name": "Analyze Changes",
            "description": "Parse and categorize code changes",
            "config": {
                "task_type": "function",
                "function_name": "analyze_pr_changes",
                "inputs": ["pr_diff", "file_list"],
            },
        },
        {
            "id": "code_quality_review",
            "type": "debate",
            "name": "Code Quality Review",
            "description": "Multi-agent review of code quality",
            "config": {
                "agents": [
                    "claude",
                    "code_quality_reviewer",
                    "deepseek",
                ],
                "rounds": 2,
                "topic_template": "Review code quality for PR #{pr_number}: {pr_title}\n\nChanges:\n{changes_summary}",
                "focus_areas": [
                    "readability",
                    "maintainability",
                    "naming_conventions",
                    "code_organization",
                    "error_handling",
                ],
            },
        },
        {
            "id": "security_review",
            "type": "debate",
            "name": "Security Review",
            "description": "Multi-agent security analysis",
            "config": {
                "agents": [
                    "code_security_specialist",
                    "claude",
                    "security_engineer",
                ],
                "rounds": 2,
                "topic_template": "Security review for PR #{pr_number}:\n{changes_summary}\n\nOWASP considerations required.",
                "focus_areas": [
                    "injection_vulnerabilities",
                    "authentication",
                    "authorization",
                    "data_exposure",
                    "input_validation",
                ],
            },
        },
        {
            "id": "architecture_review",
            "type": "debate",
            "name": "Architecture Review",
            "description": "Review architectural impact of changes",
            "config": {
                "agents": [
                    "architecture_reviewer",
                    "claude",
                    "data_architect",
                ],
                "rounds": 2,
                "topic_template": "Review architecture impact for PR #{pr_number}:\n{changes_summary}\n\nAffected components: {affected_components}",
            },
        },
        {
            "id": "performance_review",
            "type": "debate",
            "name": "Performance Review",
            "description": "Assess performance impact of changes",
            "config": {
                "agents": [
                    "performance_engineer",
                    "claude",
                    "deepseek",
                ],
                "rounds": 2,
                "topic_template": "Review performance impact for PR #{pr_number}:\n{changes_summary}",
                "focus_areas": [
                    "time_complexity",
                    "space_complexity",
                    "database_queries",
                    "api_calls",
                    "caching",
                ],
            },
        },
        {
            "id": "test_coverage_check",
            "type": "task",
            "name": "Check Test Coverage",
            "description": "Verify test coverage for changed code",
            "config": {
                "task_type": "function",
                "function_name": "check_test_coverage",
                "inputs": ["changed_files", "test_files"],
            },
        },
        {
            "id": "coverage_gate",
            "type": "decision",
            "name": "Coverage Adequate?",
            "description": "Check if test coverage meets threshold",
            "config": {
                "condition": "coverage.percentage >= coverage_threshold",
                "true_target": "documentation_check",
                "false_target": "suggest_tests",
            },
        },
        {
            "id": "suggest_tests",
            "type": "debate",
            "name": "Suggest Additional Tests",
            "description": "Generate test suggestions for uncovered code",
            "config": {
                "agents": ["claude", "code_quality_reviewer"],
                "rounds": 1,
                "topic_template": "Suggest tests for uncovered code in PR #{pr_number}:\n{uncovered_code}",
            },
        },
        {
            "id": "documentation_check",
            "type": "task",
            "name": "Documentation Check",
            "description": "Verify documentation is updated",
            "config": {
                "task_type": "function",
                "function_name": "check_documentation",
                "inputs": ["changed_files", "doc_files"],
            },
        },
        {
            "id": "synthesize_review",
            "type": "debate",
            "name": "Synthesize Review",
            "description": "Combine all review findings",
            "config": {
                "agents": ["claude", "architecture_reviewer"],
                "rounds": 1,
                "topic_template": "Synthesize review findings:\nQuality: {quality_findings}\nSecurity: {security_findings}\nArchitecture: {architecture_findings}\nPerformance: {performance_findings}",
            },
        },
        {
            "id": "generate_review_comments",
            "type": "task",
            "name": "Generate Review Comments",
            "description": "Create GitHub review comments",
            "config": {
                "task_type": "function",
                "function_name": "format_review_comments",
                "inputs": ["synthesized_review", "changed_files"],
            },
        },
        {
            "id": "determine_verdict",
            "type": "decision",
            "name": "Determine Verdict",
            "description": "Decide review outcome",
            "config": {
                "conditions": [
                    {
                        "expression": "has_critical_issues",
                        "target": "request_changes",
                    },
                    {
                        "expression": "has_suggestions_only",
                        "target": "approve_with_comments",
                    },
                    {
                        "expression": "no_issues",
                        "target": "approve",
                    },
                ],
                "default_target": "request_changes",
            },
        },
        {
            "id": "request_changes",
            "type": "task",
            "name": "Request Changes",
            "description": "Submit review requesting changes",
            "config": {
                "task_type": "http",
                "endpoint": "/api/github/pr/{pr_number}/review",
                "method": "POST",
                "body": {
                    "event": "REQUEST_CHANGES",
                    "body": "{review_summary}",
                    "comments": "{review_comments}",
                },
            },
        },
        {
            "id": "approve_with_comments",
            "type": "task",
            "name": "Approve with Comments",
            "description": "Approve PR with suggestions",
            "config": {
                "task_type": "http",
                "endpoint": "/api/github/pr/{pr_number}/review",
                "method": "POST",
                "body": {
                    "event": "APPROVE",
                    "body": "{review_summary}",
                    "comments": "{review_comments}",
                },
            },
        },
        {
            "id": "approve",
            "type": "task",
            "name": "Approve",
            "description": "Approve PR",
            "config": {
                "task_type": "http",
                "endpoint": "/api/github/pr/{pr_number}/review",
                "method": "POST",
                "body": {
                    "event": "APPROVE",
                    "body": "LGTM! {review_summary}",
                },
            },
        },
        {
            "id": "store_review",
            "type": "memory_write",
            "name": "Store Review",
            "description": "Persist review for learning",
            "config": {
                "domain": "technical/code_reviews",
            },
        },
    ],
    "transitions": [
        {"from": "fetch_pr_details", "to": "analyze_changes"},
        {"from": "analyze_changes", "to": "code_quality_review"},
        {"from": "code_quality_review", "to": "security_review"},
        {"from": "security_review", "to": "architecture_review"},
        {"from": "architecture_review", "to": "performance_review"},
        {"from": "performance_review", "to": "test_coverage_check"},
        {"from": "test_coverage_check", "to": "coverage_gate"},
        {"from": "coverage_gate", "to": "documentation_check", "condition": "adequate"},
        {"from": "coverage_gate", "to": "suggest_tests", "condition": "inadequate"},
        {"from": "suggest_tests", "to": "documentation_check"},
        {"from": "documentation_check", "to": "synthesize_review"},
        {"from": "synthesize_review", "to": "generate_review_comments"},
        {"from": "generate_review_comments", "to": "determine_verdict"},
        {"from": "determine_verdict", "to": "request_changes", "condition": "critical_issues"},
        {
            "from": "determine_verdict",
            "to": "approve_with_comments",
            "condition": "suggestions_only",
        },
        {"from": "determine_verdict", "to": "approve", "condition": "no_issues"},
        {"from": "request_changes", "to": "store_review"},
        {"from": "approve_with_comments", "to": "store_review"},
        {"from": "approve", "to": "store_review"},
    ],
    "inputs": {
        "pr_number": {
            "type": "integer",
            "required": True,
            "description": "Pull request number",
        },
        "repository": {
            "type": "string",
            "required": True,
            "description": "Repository name (owner/repo)",
        },
        "coverage_threshold": {
            "type": "number",
            "required": False,
            "default": 80,
            "description": "Minimum test coverage percentage",
        },
    },
    "outputs": {
        "review_result": {
            "type": "string",
            "enum": ["APPROVE", "APPROVE_WITH_COMMENTS", "REQUEST_CHANGES"],
            "description": "Final review verdict",
        },
        "review_comments": {
            "type": "array",
            "description": "List of review comments",
        },
        "summary": {
            "type": "string",
            "description": "Review summary",
        },
    },
}


QUICK_PR_REVIEW_TEMPLATE: Dict[str, Any] = {
    "name": "Quick PR Review",
    "description": "Fast PR review for small changes",
    "category": "development",
    "version": "1.0",
    "tags": ["code", "review", "pr", "quick"],
    "steps": [
        {
            "id": "fetch_pr",
            "type": "task",
            "name": "Fetch PR",
            "description": "Get PR details",
            "config": {
                "task_type": "http",
                "endpoint": "/api/github/pr/{pr_number}",
            },
        },
        {
            "id": "quick_review",
            "type": "debate",
            "name": "Quick Review",
            "description": "Fast multi-agent review",
            "config": {
                "agents": ["claude", "code_quality_reviewer"],
                "rounds": 1,
                "topic_template": "Quick review PR #{pr_number}: {pr_title}\n\nChanges:\n{diff}",
            },
        },
        {
            "id": "submit_review",
            "type": "task",
            "name": "Submit Review",
            "description": "Post review to GitHub",
            "config": {
                "task_type": "http",
                "endpoint": "/api/github/pr/{pr_number}/review",
                "method": "POST",
            },
        },
    ],
    "transitions": [
        {"from": "fetch_pr", "to": "quick_review"},
        {"from": "quick_review", "to": "submit_review"},
    ],
}


PR_SECURITY_REVIEW_TEMPLATE: Dict[str, Any] = {
    "name": "PR Security Review",
    "description": "Security-focused PR review",
    "category": "security",
    "version": "1.0",
    "tags": ["security", "review", "pr", "owasp"],
    "steps": [
        {
            "id": "fetch_pr",
            "type": "task",
            "name": "Fetch PR",
            "description": "Get PR details",
            "config": {
                "task_type": "http",
                "endpoint": "/api/github/pr/{pr_number}",
            },
        },
        {
            "id": "scan_dependencies",
            "type": "task",
            "name": "Scan Dependencies",
            "description": "Check for vulnerable dependencies",
            "config": {
                "task_type": "function",
                "function_name": "scan_pr_dependencies",
            },
        },
        {
            "id": "owasp_review",
            "type": "debate",
            "name": "OWASP Top 10 Review",
            "description": "Review against OWASP Top 10",
            "config": {
                "agents": [
                    "code_security_specialist",
                    "security_engineer",
                    "claude",
                ],
                "rounds": 2,
                "topic_template": "OWASP Top 10 security review for PR #{pr_number}:\n{changes}",
            },
        },
        {
            "id": "secrets_scan",
            "type": "task",
            "name": "Secrets Scan",
            "description": "Scan for exposed secrets",
            "config": {
                "task_type": "function",
                "function_name": "scan_for_secrets",
            },
        },
        {
            "id": "security_verdict",
            "type": "decision",
            "name": "Security Verdict",
            "description": "Determine security assessment",
            "config": {
                "conditions": [
                    {
                        "expression": "has_critical_vulns or has_exposed_secrets",
                        "target": "block_pr",
                    },
                    {"expression": "has_high_vulns", "target": "request_fixes"},
                    {"expression": "has_medium_vulns", "target": "warn_and_approve"},
                ],
                "default_target": "approve_security",
            },
        },
        {
            "id": "block_pr",
            "type": "task",
            "name": "Block PR",
            "description": "Block merge due to critical security issues",
            "config": {
                "task_type": "http",
                "endpoint": "/api/github/pr/{pr_number}/review",
                "body": {
                    "event": "REQUEST_CHANGES",
                    "body": "SECURITY BLOCK: {security_issues}",
                },
            },
        },
        {
            "id": "request_fixes",
            "type": "task",
            "name": "Request Security Fixes",
            "description": "Request fixes for high-severity issues",
            "config": {
                "task_type": "http",
                "endpoint": "/api/github/pr/{pr_number}/review",
                "body": {
                    "event": "REQUEST_CHANGES",
                    "body": "Security issues require attention: {security_issues}",
                },
            },
        },
        {
            "id": "warn_and_approve",
            "type": "task",
            "name": "Warn and Approve",
            "description": "Approve with security warnings",
            "config": {
                "task_type": "http",
                "endpoint": "/api/github/pr/{pr_number}/review",
                "body": {
                    "event": "APPROVE",
                    "body": "Security approved with notes: {security_notes}",
                },
            },
        },
        {
            "id": "approve_security",
            "type": "task",
            "name": "Security Approved",
            "description": "Security review passed",
            "config": {
                "task_type": "http",
                "endpoint": "/api/github/pr/{pr_number}/review",
                "body": {
                    "event": "APPROVE",
                    "body": "Security review passed.",
                },
            },
        },
    ],
    "transitions": [
        {"from": "fetch_pr", "to": "scan_dependencies"},
        {"from": "scan_dependencies", "to": "owasp_review"},
        {"from": "owasp_review", "to": "secrets_scan"},
        {"from": "secrets_scan", "to": "security_verdict"},
        {"from": "security_verdict", "to": "block_pr", "condition": "critical"},
        {"from": "security_verdict", "to": "request_fixes", "condition": "high"},
        {"from": "security_verdict", "to": "warn_and_approve", "condition": "medium"},
        {"from": "security_verdict", "to": "approve_security", "condition": "pass"},
    ],
}
