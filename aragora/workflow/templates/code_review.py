"""
Code Review Workflow Templates.

Multi-agent workflows for automated code review:
- Parallel specialized reviews (security, performance, quality)
- Conflict resolution through debate
- GitHub comment generation
- Approval workflow integration

Integrates with CodeReviewOrchestrator for comprehensive reviews.
"""

from typing import Any, Dict

CODE_REVIEW_WORKFLOW: Dict[str, Any] = {
    "name": "Automated Code Review",
    "description": "Multi-agent code review with specialized reviewers",
    "category": "development",
    "version": "1.0",
    "tags": ["code", "review", "security", "performance", "quality"],
    "config": {
        "reviewers": ["security", "performance", "maintainability", "test_coverage"],
        "auto_approve_threshold": 0,  # Max high-severity findings for auto-approve
        "require_human_approval": True,
    },
    "steps": [
        {
            "id": "fetch_code",
            "type": "task",
            "name": "Fetch Code to Review",
            "description": "Retrieve code from PR diff or file paths",
            "config": {
                "task_type": "function",
                "function_name": "fetch_review_target",
                "inputs": ["pr_url", "file_paths", "branch"],
            },
            "outputs": ["code_files", "diff", "context"],
        },
        {
            "id": "parallel_review",
            "type": "parallel",
            "name": "Parallel Specialized Reviews",
            "description": "Run all specialized reviewers in parallel",
            "branches": [
                {
                    "id": "security_scan",
                    "type": "task",
                    "name": "Security Review",
                    "description": "Scan for security vulnerabilities",
                    "config": {
                        "task_type": "agent",
                        "agent_type": "security_reviewer",
                        "focus_areas": [
                            "injection_vulnerabilities",
                            "authentication_issues",
                            "authorization_gaps",
                            "sensitive_data_exposure",
                            "hardcoded_secrets",
                        ],
                    },
                    "outputs": ["security_findings"],
                },
                {
                    "id": "performance_analysis",
                    "type": "task",
                    "name": "Performance Review",
                    "description": "Analyze performance implications",
                    "config": {
                        "task_type": "agent",
                        "agent_type": "performance_reviewer",
                        "focus_areas": [
                            "algorithm_complexity",
                            "memory_usage",
                            "database_queries",
                            "caching_opportunities",
                            "async_patterns",
                        ],
                    },
                    "outputs": ["performance_findings"],
                },
                {
                    "id": "maintainability_check",
                    "type": "task",
                    "name": "Maintainability Review",
                    "description": "Review code quality and patterns",
                    "config": {
                        "task_type": "agent",
                        "agent_type": "maintainability_reviewer",
                        "focus_areas": [
                            "code_clarity",
                            "design_patterns",
                            "solid_principles",
                            "error_handling",
                            "documentation",
                        ],
                    },
                    "outputs": ["maintainability_findings"],
                },
                {
                    "id": "test_coverage_check",
                    "type": "task",
                    "name": "Test Coverage Review",
                    "description": "Check test coverage and quality",
                    "config": {
                        "task_type": "agent",
                        "agent_type": "test_coverage_reviewer",
                        "focus_areas": [
                            "missing_tests",
                            "edge_cases",
                            "assertion_quality",
                            "test_independence",
                            "mock_usage",
                        ],
                    },
                    "outputs": ["coverage_findings"],
                },
            ],
        },
        {
            "id": "aggregate_findings",
            "type": "task",
            "name": "Aggregate Findings",
            "description": "Combine and deduplicate findings from all reviewers",
            "config": {
                "task_type": "function",
                "function_name": "aggregate_review_findings",
                "inputs": [
                    "security_findings",
                    "performance_findings",
                    "maintainability_findings",
                    "coverage_findings",
                ],
            },
            "outputs": ["all_findings", "finding_summary"],
        },
        {
            "id": "check_conflicts",
            "type": "condition",
            "name": "Check for Conflicting Recommendations",
            "description": "Determine if findings conflict",
            "config": {
                "condition": "has_conflicting_findings(all_findings)",
            },
            "on_true": "debate_conflicts",
            "on_false": "determine_approval",
        },
        {
            "id": "debate_conflicts",
            "type": "debate",
            "name": "Resolve Conflicting Findings",
            "description": "Multi-agent debate to resolve conflicts",
            "config": {
                "agents": [
                    "security_reviewer",
                    "performance_reviewer",
                    "maintainability_reviewer",
                ],
                "rounds": 2,
                "topic_template": "Resolve conflicting code review recommendations:\n{conflict_summary}",
                "consensus_threshold": 0.7,
            },
            "outputs": ["consensus_findings", "debate_notes"],
        },
        {
            "id": "determine_approval",
            "type": "task",
            "name": "Determine Approval Status",
            "description": "Decide if code should be approved",
            "config": {
                "task_type": "function",
                "function_name": "determine_review_approval",
                "inputs": ["all_findings", "consensus_findings"],
                "rules": {
                    "block_on_critical": True,
                    "block_on_high_count": 3,
                    "require_human_for_medium": True,
                },
            },
            "outputs": ["approval_status", "approval_reason"],
        },
        {
            "id": "generate_comments",
            "type": "task",
            "name": "Generate Review Comments",
            "description": "Format findings as GitHub comments",
            "config": {
                "task_type": "function",
                "function_name": "generate_github_comments",
                "inputs": ["all_findings", "pr_url"],
                "max_comments": 20,
                "prioritize_by": "severity",
            },
            "outputs": ["comments"],
        },
        {
            "id": "human_checkpoint",
            "type": "approval",
            "name": "Human Review Checkpoint",
            "description": "Optional human review before posting",
            "config": {
                "skip_if": "approval_status == 'approved'",
                "timeout_hours": 24,
                "auto_approve_on_timeout": False,
            },
        },
        {
            "id": "post_review",
            "type": "task",
            "name": "Post Review to GitHub",
            "description": "Submit review comments to PR",
            "config": {
                "task_type": "function",
                "function_name": "post_github_review",
                "inputs": ["pr_url", "comments", "approval_status"],
                "requires_permission": True,
            },
            "outputs": ["review_url"],
        },
    ],
}

QUICK_SECURITY_SCAN: Dict[str, Any] = {
    "name": "Quick Security Scan",
    "description": "Fast security-focused code review",
    "category": "security",
    "version": "1.0",
    "tags": ["code", "security", "quick", "scan"],
    "steps": [
        {
            "id": "pattern_scan",
            "type": "task",
            "name": "Pattern-Based Security Scan",
            "description": "Scan for known vulnerability patterns",
            "config": {
                "task_type": "function",
                "function_name": "security_pattern_scan",
                "patterns": [
                    "injection",
                    "hardcoded_secrets",
                    "unsafe_deserialization",
                    "xss",
                    "sql_injection",
                ],
            },
        },
        {
            "id": "report",
            "type": "task",
            "name": "Generate Security Report",
            "description": "Create security scan report",
            "config": {
                "task_type": "function",
                "function_name": "generate_security_report",
                "include_recommendations": True,
            },
        },
    ],
}

ARCHITECTURE_REVIEW_WORKFLOW: Dict[str, Any] = {
    "name": "Architecture Review",
    "description": "Review architectural impact of code changes",
    "category": "development",
    "version": "1.0",
    "tags": ["code", "architecture", "design", "patterns"],
    "steps": [
        {
            "id": "analyze_structure",
            "type": "task",
            "name": "Analyze Code Structure",
            "description": "Parse and understand code organization",
            "config": {
                "task_type": "function",
                "function_name": "analyze_code_structure",
                "outputs": ["modules", "dependencies", "imports"],
            },
        },
        {
            "id": "pattern_detection",
            "type": "task",
            "name": "Detect Design Patterns",
            "description": "Identify design patterns in use",
            "config": {
                "task_type": "function",
                "function_name": "detect_design_patterns",
                "patterns": [
                    "factory",
                    "singleton",
                    "observer",
                    "strategy",
                    "decorator",
                    "repository",
                ],
            },
        },
        {
            "id": "architecture_debate",
            "type": "debate",
            "name": "Architecture Review Discussion",
            "description": "Multi-agent architectural analysis",
            "config": {
                "agents": [
                    "software_architect",
                    "claude",
                    "system_designer",
                ],
                "rounds": 2,
                "focus_areas": [
                    "solid_principles",
                    "coupling",
                    "cohesion",
                    "scalability",
                    "maintainability",
                ],
            },
        },
        {
            "id": "recommendations",
            "type": "task",
            "name": "Generate Recommendations",
            "description": "Compile architectural recommendations",
            "config": {
                "task_type": "function",
                "function_name": "compile_architecture_recommendations",
            },
        },
    ],
}

# Export all templates
CODE_REVIEW_TEMPLATES = {
    "code_review": CODE_REVIEW_WORKFLOW,
    "quick_security_scan": QUICK_SECURITY_SCAN,
    "architecture_review": ARCHITECTURE_REVIEW_WORKFLOW,
}

__all__ = [
    "CODE_REVIEW_WORKFLOW",
    "QUICK_SECURITY_SCAN",
    "ARCHITECTURE_REVIEW_WORKFLOW",
    "CODE_REVIEW_TEMPLATES",
]
