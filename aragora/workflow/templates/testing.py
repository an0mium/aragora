"""
Test Generation Workflow Templates.

Templates for automated test generation, coverage analysis, and test quality review.
"""

from typing import Dict, Any

TEST_GENERATION_TEMPLATE: Dict[str, Any] = {
    "name": "Automated Test Generation",
    "description": "Generate comprehensive tests for code with multi-agent review",
    "category": "testing",
    "version": "1.0",
    "tags": ["testing", "coverage", "quality", "automation"],
    "steps": [
        {
            "id": "analyze_coverage",
            "type": "task",
            "name": "Analyze Coverage Gaps",
            "description": "Identify functions and code paths lacking test coverage",
            "config": {
                "task_type": "function",
                "function_name": "analyze_test_coverage",
                "inputs": ["source_files", "test_files"],
            },
        },
        {
            "id": "extract_signatures",
            "type": "task",
            "name": "Extract Function Signatures",
            "description": "Parse source code to extract testable function signatures",
            "config": {
                "task_type": "function",
                "function_name": "extract_function_signatures",
            },
        },
        {
            "id": "prioritize_tests",
            "type": "debate",
            "name": "Prioritize Test Targets",
            "description": "Multi-agent debate on which functions need tests most urgently",
            "config": {
                "agents": [
                    "test_generator",
                    "code_quality_reviewer",
                    "code_security_specialist",
                ],
                "rounds": 2,
                "topic_template": "Which untested functions should be prioritized? Coverage gaps: {coverage_gaps}",
                "consensus_mechanism": "weighted",
            },
        },
        {
            "id": "generate_unit_tests",
            "type": "task",
            "name": "Generate Unit Tests",
            "description": "Generate unit tests for prioritized functions",
            "config": {
                "task_type": "function",
                "function_name": "generate_unit_tests",
                "inputs": ["prioritized_functions"],
            },
        },
        {
            "id": "generate_edge_cases",
            "type": "task",
            "name": "Generate Edge Case Tests",
            "description": "Generate tests for boundary conditions and edge cases",
            "config": {
                "task_type": "function",
                "function_name": "generate_edge_case_tests",
            },
        },
        {
            "id": "review_test_quality",
            "type": "debate",
            "name": "Review Generated Tests",
            "description": "Multi-agent review of generated test quality",
            "config": {
                "agents": [
                    "test_generator",
                    "code_quality_reviewer",
                ],
                "rounds": 2,
                "topic_template": "Review test quality and coverage: {generated_tests}",
            },
        },
        {
            "id": "validate_tests",
            "type": "task",
            "name": "Validate Tests Run",
            "description": "Execute generated tests to ensure they pass",
            "config": {
                "task_type": "shell",
                "command": "pytest {test_file} --tb=short",
                "timeout": 60,
            },
        },
        {
            "id": "tests_pass_gate",
            "type": "decision",
            "name": "Tests Pass Gate",
            "description": "Check if generated tests pass",
            "config": {
                "condition": "test_exit_code == 0",
                "true_target": "human_review",
                "false_target": "fix_tests",
            },
        },
        {
            "id": "fix_tests",
            "type": "task",
            "name": "Fix Failing Tests",
            "description": "Attempt to fix failing generated tests",
            "config": {
                "task_type": "function",
                "function_name": "fix_failing_tests",
                "max_retries": 3,
            },
        },
        {
            "id": "human_review",
            "type": "human_checkpoint",
            "name": "Developer Review",
            "description": "Developer reviews generated tests before merging",
            "config": {
                "approval_type": "review",
                "checklist": [
                    "Tests are meaningful (not just for coverage)",
                    "Assertions verify correct behavior",
                    "Edge cases are properly handled",
                    "Tests are maintainable",
                ],
            },
        },
        {
            "id": "store_results",
            "type": "memory_write",
            "name": "Store Test Generation Results",
            "description": "Save test generation metadata to knowledge base",
            "config": {
                "mound_type": "testing",
                "data_keys": [
                    "generated_tests",
                    "coverage_improvement",
                    "review_feedback",
                ],
            },
        },
    ],
    "transitions": [
        {"from": "analyze_coverage", "to": "extract_signatures"},
        {"from": "extract_signatures", "to": "prioritize_tests"},
        {"from": "prioritize_tests", "to": "generate_unit_tests"},
        {"from": "generate_unit_tests", "to": "generate_edge_cases"},
        {"from": "generate_edge_cases", "to": "review_test_quality"},
        {"from": "review_test_quality", "to": "validate_tests"},
        {"from": "validate_tests", "to": "tests_pass_gate"},
        {"from": "tests_pass_gate", "to": "human_review", "condition": "tests_pass"},
        {"from": "tests_pass_gate", "to": "fix_tests", "condition": "tests_fail"},
        {"from": "fix_tests", "to": "validate_tests"},
        {"from": "human_review", "to": "store_results"},
    ],
}


COVERAGE_ANALYSIS_TEMPLATE: Dict[str, Any] = {
    "name": "Test Coverage Analysis",
    "description": "Analyze and report on test coverage with improvement recommendations",
    "category": "testing",
    "version": "1.0",
    "tags": ["testing", "coverage", "analysis", "metrics"],
    "steps": [
        {
            "id": "run_coverage",
            "type": "task",
            "name": "Run Coverage Analysis",
            "description": "Execute tests with coverage measurement",
            "config": {
                "task_type": "shell",
                "command": "pytest --cov={source_dir} --cov-report=json --cov-report=term",
            },
        },
        {
            "id": "parse_coverage",
            "type": "task",
            "name": "Parse Coverage Report",
            "description": "Parse coverage JSON into structured data",
            "config": {
                "task_type": "function",
                "function_name": "parse_coverage_json",
            },
        },
        {
            "id": "identify_gaps",
            "type": "task",
            "name": "Identify Coverage Gaps",
            "description": "Find files and functions with low coverage",
            "config": {
                "task_type": "function",
                "function_name": "identify_coverage_gaps",
                "threshold": 80,  # Minimum coverage percentage
            },
        },
        {
            "id": "analyze_critical_paths",
            "type": "debate",
            "name": "Analyze Critical Paths",
            "description": "Multi-agent analysis of which uncovered paths are most critical",
            "config": {
                "agents": [
                    "test_generator",
                    "code_security_specialist",
                    "architecture_reviewer",
                ],
                "rounds": 2,
                "topic_template": "Which uncovered code paths are most critical? Gaps: {coverage_gaps}",
            },
        },
        {
            "id": "generate_recommendations",
            "type": "task",
            "name": "Generate Recommendations",
            "description": "Create actionable recommendations for coverage improvement",
            "config": {
                "task_type": "transform",
                "template": "coverage_recommendations",
            },
        },
        {
            "id": "generate_report",
            "type": "task",
            "name": "Generate Coverage Report",
            "description": "Create comprehensive coverage analysis report",
            "config": {
                "task_type": "transform",
                "template": "coverage_report",
                "formats": ["markdown", "html"],
            },
        },
    ],
    "transitions": [
        {"from": "run_coverage", "to": "parse_coverage"},
        {"from": "parse_coverage", "to": "identify_gaps"},
        {"from": "identify_gaps", "to": "analyze_critical_paths"},
        {"from": "analyze_critical_paths", "to": "generate_recommendations"},
        {"from": "generate_recommendations", "to": "generate_report"},
    ],
}


MUTATION_TESTING_TEMPLATE: Dict[str, Any] = {
    "name": "Mutation Testing Analysis",
    "description": "Evaluate test effectiveness using mutation testing",
    "category": "testing",
    "version": "1.0",
    "tags": ["testing", "mutation", "quality", "effectiveness"],
    "steps": [
        {
            "id": "configure_mutants",
            "type": "task",
            "name": "Configure Mutation Operators",
            "description": "Set up mutation operators for the codebase",
            "config": {
                "task_type": "function",
                "function_name": "configure_mutation_operators",
                "operators": [
                    "arithmetic",
                    "conditional",
                    "return_value",
                    "exception",
                ],
            },
        },
        {
            "id": "generate_mutants",
            "type": "task",
            "name": "Generate Mutants",
            "description": "Create mutated versions of source code",
            "config": {
                "task_type": "shell",
                "command": "mutmut run --paths-to-mutate={source_dir}",
                "timeout": 300,
            },
        },
        {
            "id": "analyze_survivors",
            "type": "task",
            "name": "Analyze Surviving Mutants",
            "description": "Identify mutants that survived (tests didn't catch)",
            "config": {
                "task_type": "function",
                "function_name": "analyze_surviving_mutants",
            },
        },
        {
            "id": "prioritize_improvements",
            "type": "debate",
            "name": "Prioritize Test Improvements",
            "description": "Debate which test improvements would catch most mutants",
            "config": {
                "agents": [
                    "test_generator",
                    "code_quality_reviewer",
                ],
                "rounds": 2,
                "topic_template": "How to improve tests to catch these mutants? {surviving_mutants}",
            },
        },
        {
            "id": "generate_killer_tests",
            "type": "task",
            "name": "Generate Mutant-Killing Tests",
            "description": "Generate tests specifically designed to kill surviving mutants",
            "config": {
                "task_type": "function",
                "function_name": "generate_mutant_killers",
            },
        },
        {
            "id": "generate_report",
            "type": "task",
            "name": "Generate Mutation Report",
            "description": "Create mutation testing effectiveness report",
            "config": {
                "task_type": "transform",
                "template": "mutation_testing_report",
            },
        },
    ],
    "transitions": [
        {"from": "configure_mutants", "to": "generate_mutants"},
        {"from": "generate_mutants", "to": "analyze_survivors"},
        {"from": "analyze_survivors", "to": "prioritize_improvements"},
        {"from": "prioritize_improvements", "to": "generate_killer_tests"},
        {"from": "generate_killer_tests", "to": "generate_report"},
    ],
}


# Template registry
TESTING_TEMPLATES = {
    "test_generation": TEST_GENERATION_TEMPLATE,
    "coverage_analysis": COVERAGE_ANALYSIS_TEMPLATE,
    "mutation_testing": MUTATION_TESTING_TEMPLATE,
}


def get_testing_template(name: str) -> Dict[str, Any]:
    """
    Get a testing workflow template by name.

    Args:
        name: Template name

    Returns:
        Template dictionary

    Raises:
        KeyError: If template not found
    """
    if name not in TESTING_TEMPLATES:
        raise KeyError(
            f"Unknown testing template: {name}. Available: {list(TESTING_TEMPLATES.keys())}"
        )
    return TESTING_TEMPLATES[name]
