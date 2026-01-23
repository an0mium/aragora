"""
Feature Implementation Workflow Templates.

Multi-agent workflows for implementing new features:
- Requirements gathering and clarification
- Design and architecture planning
- Test-first development
- Implementation with code review
- Final review and documentation

These workflows leverage multi-agent debate for design decisions,
test generation, and code review.
"""

from typing import Dict, Any

FEATURE_IMPLEMENTATION_TEMPLATE: Dict[str, Any] = {
    "name": "Feature Implementation",
    "description": "End-to-end feature implementation with multi-agent collaboration",
    "category": "development",
    "version": "1.0",
    "tags": ["code", "feature", "implementation", "tdd", "review"],
    "steps": [
        {
            "id": "gather_requirements",
            "type": "task",
            "name": "Gather Requirements",
            "description": "Collect and structure feature requirements",
            "config": {
                "task_type": "function",
                "function_name": "gather_feature_requirements",
                "inputs": ["feature_description", "context"],
            },
        },
        {
            "id": "clarify_requirements",
            "type": "human_checkpoint",
            "name": "Clarify Requirements",
            "description": "Review and clarify requirements with stakeholder",
            "config": {
                "approval_type": "review",
                "checklist": [
                    "Requirements are clear and complete",
                    "Edge cases are identified",
                    "Success criteria defined",
                    "Dependencies identified",
                ],
                "allow_modifications": True,
            },
        },
        {
            "id": "design_debate",
            "type": "debate",
            "name": "Architecture Design",
            "description": "Multi-agent debate on feature architecture",
            "config": {
                "agents": [
                    "claude",
                    "architecture_reviewer",
                    "deepseek",
                ],
                "rounds": 3,
                "topic_template": "Design the architecture for: {feature_description}\nRequirements: {requirements}",
                "consensus_type": "majority",
            },
        },
        {
            "id": "design_review",
            "type": "human_checkpoint",
            "name": "Design Review",
            "description": "Review and approve the proposed design",
            "config": {
                "approval_type": "sign_off",
                "checklist": [
                    "Design aligns with requirements",
                    "Architecture is sound",
                    "No major concerns",
                ],
                "required_role": "tech_lead",
            },
        },
        {
            "id": "generate_test_cases",
            "type": "debate",
            "name": "Generate Test Cases",
            "description": "Multi-agent generation of test cases",
            "config": {
                "agents": [
                    "claude",
                    "code_quality_reviewer",
                    "deepseek",
                ],
                "rounds": 2,
                "topic_template": "Generate comprehensive test cases for: {feature_description}\nDesign: {design}",
            },
        },
        {
            "id": "write_tests",
            "type": "task",
            "name": "Write Tests",
            "description": "Generate test code from test cases",
            "config": {
                "task_type": "function",
                "function_name": "generate_test_code",
                "inputs": ["test_cases", "codebase_context"],
            },
        },
        {
            "id": "test_review",
            "type": "human_checkpoint",
            "name": "Test Review",
            "description": "Review generated tests before implementation",
            "config": {
                "approval_type": "review",
                "allow_modifications": True,
            },
        },
        {
            "id": "implementation_planning",
            "type": "debate",
            "name": "Implementation Planning",
            "description": "Plan implementation steps with multi-agent input",
            "config": {
                "agents": [
                    "claude",
                    "codex",
                    "deepseek",
                ],
                "rounds": 2,
                "topic_template": "Plan implementation steps for: {feature_description}\nDesign: {design}\nTests: {tests}",
            },
        },
        {
            "id": "implement_feature",
            "type": "task",
            "name": "Implement Feature",
            "description": "Generate implementation code",
            "config": {
                "task_type": "function",
                "function_name": "implement_feature",
                "inputs": ["implementation_plan", "tests", "codebase_context"],
            },
        },
        {
            "id": "run_tests",
            "type": "task",
            "name": "Run Tests",
            "description": "Execute tests against implementation",
            "config": {
                "task_type": "function",
                "function_name": "run_test_suite",
            },
        },
        {
            "id": "test_gate",
            "type": "decision",
            "name": "Tests Passing?",
            "description": "Check if all tests pass",
            "config": {
                "condition": "test_results.all_passed",
                "true_target": "code_review",
                "false_target": "fix_implementation",
            },
        },
        {
            "id": "fix_implementation",
            "type": "debate",
            "name": "Fix Implementation",
            "description": "Debug and fix failing tests",
            "config": {
                "agents": ["claude", "deepseek", "code_security_specialist"],
                "rounds": 2,
                "topic_template": "Fix failing tests: {test_failures}\nCurrent implementation: {implementation}",
            },
        },
        {
            "id": "code_review",
            "type": "debate",
            "name": "Code Review",
            "description": "Multi-agent code review",
            "config": {
                "agents": [
                    "claude",
                    "code_quality_reviewer",
                    "code_security_specialist",
                    "architecture_reviewer",
                ],
                "rounds": 2,
                "topic_template": "Review implementation for: {feature_description}\nCode: {implementation}",
            },
        },
        {
            "id": "review_findings_gate",
            "type": "decision",
            "name": "Review Findings?",
            "description": "Check if review found issues",
            "config": {
                "condition": "review_findings.has_blockers",
                "true_target": "address_review_findings",
                "false_target": "final_review",
            },
        },
        {
            "id": "address_review_findings",
            "type": "task",
            "name": "Address Review Findings",
            "description": "Fix issues found in code review",
            "config": {
                "task_type": "function",
                "function_name": "apply_review_fixes",
            },
        },
        {
            "id": "final_review",
            "type": "human_checkpoint",
            "name": "Final Review",
            "description": "Human sign-off on completed feature",
            "config": {
                "approval_type": "sign_off",
                "checklist": [
                    "Feature meets requirements",
                    "Tests are comprehensive",
                    "Code quality is acceptable",
                    "Documentation is adequate",
                    "Ready for merge",
                ],
                "required_role": "tech_lead",
            },
        },
        {
            "id": "generate_documentation",
            "type": "task",
            "name": "Generate Documentation",
            "description": "Generate feature documentation",
            "config": {
                "task_type": "function",
                "function_name": "generate_documentation",
            },
        },
        {
            "id": "create_pr",
            "type": "task",
            "name": "Create Pull Request",
            "description": "Create PR with feature changes",
            "config": {
                "task_type": "http",
                "endpoint": "/api/github/pr/create",
            },
        },
        {
            "id": "store_knowledge",
            "type": "memory_write",
            "name": "Store Implementation Knowledge",
            "description": "Persist design decisions and learnings",
            "config": {
                "domain": "technical/implementations",
            },
        },
    ],
    "transitions": [
        {"from": "gather_requirements", "to": "clarify_requirements"},
        {"from": "clarify_requirements", "to": "design_debate", "condition": "approved"},
        {"from": "design_debate", "to": "design_review"},
        {"from": "design_review", "to": "generate_test_cases", "condition": "approved"},
        {"from": "generate_test_cases", "to": "write_tests"},
        {"from": "write_tests", "to": "test_review"},
        {"from": "test_review", "to": "implementation_planning", "condition": "approved"},
        {"from": "implementation_planning", "to": "implement_feature"},
        {"from": "implement_feature", "to": "run_tests"},
        {"from": "run_tests", "to": "test_gate"},
        {"from": "test_gate", "to": "code_review", "condition": "tests_pass"},
        {"from": "test_gate", "to": "fix_implementation", "condition": "tests_fail"},
        {"from": "fix_implementation", "to": "implement_feature"},  # Loop back
        {"from": "code_review", "to": "review_findings_gate"},
        {
            "from": "review_findings_gate",
            "to": "address_review_findings",
            "condition": "has_blockers",
        },
        {"from": "review_findings_gate", "to": "final_review", "condition": "no_blockers"},
        {"from": "address_review_findings", "to": "code_review"},  # Loop back for re-review
        {"from": "final_review", "to": "generate_documentation", "condition": "approved"},
        {"from": "generate_documentation", "to": "create_pr"},
        {"from": "create_pr", "to": "store_knowledge"},
    ],
    "inputs": {
        "feature_description": {
            "type": "string",
            "required": True,
            "description": "Description of the feature to implement",
        },
        "context": {
            "type": "object",
            "required": False,
            "description": "Additional context (repo, branch, relevant files)",
        },
        "codebase_context": {
            "type": "object",
            "required": False,
            "description": "Codebase understanding (architecture, patterns, conventions)",
        },
    },
    "outputs": {
        "implementation": {
            "type": "object",
            "description": "Generated implementation code",
        },
        "tests": {
            "type": "object",
            "description": "Generated test code",
        },
        "documentation": {
            "type": "string",
            "description": "Generated documentation",
        },
        "pr_url": {
            "type": "string",
            "description": "URL of created pull request",
        },
    },
}


BUG_FIX_TEMPLATE: Dict[str, Any] = {
    "name": "Bug Fix",
    "description": "Structured bug fix workflow with root cause analysis",
    "category": "development",
    "version": "1.0",
    "tags": ["code", "bugfix", "debug", "review"],
    "steps": [
        {
            "id": "reproduce_bug",
            "type": "task",
            "name": "Reproduce Bug",
            "description": "Create reproduction steps and verify bug exists",
            "config": {
                "task_type": "function",
                "function_name": "create_bug_reproduction",
            },
        },
        {
            "id": "root_cause_analysis",
            "type": "debate",
            "name": "Root Cause Analysis",
            "description": "Multi-agent investigation of root cause",
            "config": {
                "agents": ["claude", "deepseek", "code_security_specialist"],
                "rounds": 2,
                "topic_template": "Analyze root cause of bug: {bug_description}\nStack trace: {stack_trace}\nReproduction: {reproduction}",
            },
        },
        {
            "id": "propose_fix",
            "type": "debate",
            "name": "Propose Fix",
            "description": "Multi-agent proposal of fix approaches",
            "config": {
                "agents": ["claude", "codex", "architecture_reviewer"],
                "rounds": 2,
                "topic_template": "Propose fix for: {root_cause}\nConstraints: {constraints}",
            },
        },
        {
            "id": "approve_fix_approach",
            "type": "human_checkpoint",
            "name": "Approve Fix Approach",
            "description": "Review and approve proposed fix",
            "config": {
                "approval_type": "review",
                "allow_modifications": True,
            },
        },
        {
            "id": "write_regression_test",
            "type": "task",
            "name": "Write Regression Test",
            "description": "Create test that fails with bug, passes with fix",
            "config": {
                "task_type": "function",
                "function_name": "generate_regression_test",
            },
        },
        {
            "id": "implement_fix",
            "type": "task",
            "name": "Implement Fix",
            "description": "Apply the approved fix",
            "config": {
                "task_type": "function",
                "function_name": "implement_bug_fix",
            },
        },
        {
            "id": "verify_fix",
            "type": "task",
            "name": "Verify Fix",
            "description": "Run tests to verify fix works",
            "config": {
                "task_type": "function",
                "function_name": "run_test_suite",
            },
        },
        {
            "id": "fix_verification_gate",
            "type": "decision",
            "name": "Fix Verified?",
            "description": "Check if fix resolves the bug",
            "config": {
                "condition": "verification.passed",
                "true_target": "code_review",
                "false_target": "root_cause_analysis",  # Go back if fix didn't work
            },
        },
        {
            "id": "code_review",
            "type": "debate",
            "name": "Code Review",
            "description": "Review the fix for quality and side effects",
            "config": {
                "agents": ["claude", "code_quality_reviewer", "code_security_specialist"],
                "rounds": 2,
                "topic_template": "Review bug fix: {fix_implementation}\nOriginal bug: {bug_description}",
            },
        },
        {
            "id": "final_approval",
            "type": "human_checkpoint",
            "name": "Final Approval",
            "description": "Sign off on completed fix",
            "config": {
                "approval_type": "sign_off",
                "checklist": [
                    "Bug is fixed",
                    "Regression test added",
                    "No new issues introduced",
                    "Code quality acceptable",
                ],
            },
        },
        {
            "id": "create_pr",
            "type": "task",
            "name": "Create Pull Request",
            "description": "Create PR with fix",
            "config": {
                "task_type": "http",
                "endpoint": "/api/github/pr/create",
            },
        },
    ],
    "transitions": [
        {"from": "reproduce_bug", "to": "root_cause_analysis"},
        {"from": "root_cause_analysis", "to": "propose_fix"},
        {"from": "propose_fix", "to": "approve_fix_approach"},
        {"from": "approve_fix_approach", "to": "write_regression_test", "condition": "approved"},
        {"from": "write_regression_test", "to": "implement_fix"},
        {"from": "implement_fix", "to": "verify_fix"},
        {"from": "verify_fix", "to": "fix_verification_gate"},
        {"from": "fix_verification_gate", "to": "code_review", "condition": "passed"},
        {"from": "fix_verification_gate", "to": "root_cause_analysis", "condition": "failed"},
        {"from": "code_review", "to": "final_approval"},
        {"from": "final_approval", "to": "create_pr", "condition": "approved"},
    ],
}


REFACTORING_TEMPLATE: Dict[str, Any] = {
    "name": "Code Refactoring",
    "description": "Safe refactoring workflow with comprehensive testing",
    "category": "development",
    "version": "1.0",
    "tags": ["code", "refactoring", "quality", "review"],
    "steps": [
        {
            "id": "analyze_code",
            "type": "task",
            "name": "Analyze Code",
            "description": "Analyze code to be refactored",
            "config": {
                "task_type": "function",
                "function_name": "analyze_refactoring_scope",
            },
        },
        {
            "id": "identify_tests",
            "type": "task",
            "name": "Identify Existing Tests",
            "description": "Find tests covering the code to refactor",
            "config": {
                "task_type": "function",
                "function_name": "identify_test_coverage",
            },
        },
        {
            "id": "coverage_check",
            "type": "decision",
            "name": "Sufficient Coverage?",
            "description": "Check if test coverage is adequate",
            "config": {
                "condition": "coverage >= 80",
                "true_target": "plan_refactoring",
                "false_target": "add_tests",
            },
        },
        {
            "id": "add_tests",
            "type": "debate",
            "name": "Add Missing Tests",
            "description": "Generate tests for uncovered code",
            "config": {
                "agents": ["claude", "code_quality_reviewer"],
                "rounds": 2,
                "topic_template": "Generate tests for: {uncovered_code}",
            },
        },
        {
            "id": "plan_refactoring",
            "type": "debate",
            "name": "Plan Refactoring",
            "description": "Multi-agent planning of refactoring approach",
            "config": {
                "agents": [
                    "claude",
                    "architecture_reviewer",
                    "code_quality_reviewer",
                ],
                "rounds": 3,
                "topic_template": "Plan refactoring for: {code_analysis}\nGoals: {refactoring_goals}",
            },
        },
        {
            "id": "approve_plan",
            "type": "human_checkpoint",
            "name": "Approve Refactoring Plan",
            "description": "Review and approve refactoring approach",
            "config": {
                "approval_type": "sign_off",
                "checklist": [
                    "Plan is incremental and safe",
                    "Risks are identified",
                    "Rollback strategy exists",
                ],
            },
        },
        {
            "id": "execute_refactoring",
            "type": "task",
            "name": "Execute Refactoring",
            "description": "Apply refactoring changes",
            "config": {
                "task_type": "function",
                "function_name": "apply_refactoring",
            },
        },
        {
            "id": "run_tests",
            "type": "task",
            "name": "Run Tests",
            "description": "Verify tests still pass",
            "config": {
                "task_type": "function",
                "function_name": "run_test_suite",
            },
        },
        {
            "id": "test_gate",
            "type": "decision",
            "name": "Tests Pass?",
            "description": "Check all tests pass after refactoring",
            "config": {
                "condition": "test_results.all_passed",
                "true_target": "quality_review",
                "false_target": "rollback",
            },
        },
        {
            "id": "rollback",
            "type": "task",
            "name": "Rollback",
            "description": "Rollback failed refactoring",
            "config": {
                "task_type": "function",
                "function_name": "rollback_changes",
            },
        },
        {
            "id": "quality_review",
            "type": "debate",
            "name": "Quality Review",
            "description": "Review refactored code quality",
            "config": {
                "agents": ["claude", "code_quality_reviewer", "architecture_reviewer"],
                "rounds": 2,
                "topic_template": "Review refactored code: {refactored_code}\nOriginal: {original_code}",
            },
        },
        {
            "id": "final_approval",
            "type": "human_checkpoint",
            "name": "Final Approval",
            "description": "Sign off on refactoring",
            "config": {
                "approval_type": "sign_off",
                "checklist": [
                    "Code quality improved",
                    "All tests pass",
                    "No regression",
                    "Documentation updated",
                ],
            },
        },
        {
            "id": "create_pr",
            "type": "task",
            "name": "Create Pull Request",
            "description": "Create PR with refactoring",
            "config": {
                "task_type": "http",
                "endpoint": "/api/github/pr/create",
            },
        },
    ],
    "transitions": [
        {"from": "analyze_code", "to": "identify_tests"},
        {"from": "identify_tests", "to": "coverage_check"},
        {"from": "coverage_check", "to": "plan_refactoring", "condition": "sufficient"},
        {"from": "coverage_check", "to": "add_tests", "condition": "insufficient"},
        {"from": "add_tests", "to": "plan_refactoring"},
        {"from": "plan_refactoring", "to": "approve_plan"},
        {"from": "approve_plan", "to": "execute_refactoring", "condition": "approved"},
        {"from": "execute_refactoring", "to": "run_tests"},
        {"from": "run_tests", "to": "test_gate"},
        {"from": "test_gate", "to": "quality_review", "condition": "passed"},
        {"from": "test_gate", "to": "rollback", "condition": "failed"},
        {"from": "rollback", "to": "plan_refactoring"},  # Try again with different approach
        {"from": "quality_review", "to": "final_approval"},
        {"from": "final_approval", "to": "create_pr", "condition": "approved"},
    ],
}
