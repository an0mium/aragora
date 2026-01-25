"""
Refactoring Workflow Templates.

Templates for automated code refactoring with multi-agent debate and verification.
Detects code smells, plans refactoring, and ensures semantic equivalence.
"""

from typing import Any

# Code smell detection thresholds
CODE_SMELL_THRESHOLDS = {
    "long_method_lines": 50,
    "deep_nesting_levels": 4,
    "cyclomatic_complexity": 10,
    "duplicate_similarity": 0.80,
    "god_class_methods": 20,
    "too_many_parameters": 5,
    "large_class_lines": 500,
    "feature_envy_external_calls": 5,
}


REFACTORING_WORKFLOW_TEMPLATE: dict[str, Any] = {
    "name": "Code Refactoring Workflow",
    "description": "Detect code smells, plan refactoring, and verify semantic equivalence",
    "category": "refactoring",
    "version": "1.0",
    "tags": ["refactoring", "code-quality", "maintenance", "debt"],
    "steps": [
        {
            "id": "detect_code_smells",
            "type": "task",
            "name": "Detect Code Smells",
            "description": "Analyze codebase for common code smells and antipatterns",
            "config": {
                "task_type": "function",
                "function_name": "detect_code_smells",
                "inputs": ["source_files"],
                "thresholds": CODE_SMELL_THRESHOLDS,
            },
        },
        {
            "id": "calculate_complexity",
            "type": "task",
            "name": "Calculate Complexity Metrics",
            "description": "Compute cyclomatic complexity, cognitive complexity, and maintainability index",
            "config": {
                "task_type": "function",
                "function_name": "calculate_complexity_metrics",
                "metrics": [
                    "cyclomatic_complexity",
                    "cognitive_complexity",
                    "maintainability_index",
                    "halstead_metrics",
                ],
            },
        },
        {
            "id": "find_duplicates",
            "type": "task",
            "name": "Find Duplicate Code",
            "description": "Detect duplicate and near-duplicate code blocks",
            "config": {
                "task_type": "function",
                "function_name": "find_code_duplicates",
                "min_lines": 5,
                "similarity_threshold": 0.8,
            },
        },
        {
            "id": "identify_candidates",
            "type": "task",
            "name": "Identify Refactoring Candidates",
            "description": "Rank code sections by refactoring priority based on smells and complexity",
            "config": {
                "task_type": "function",
                "function_name": "rank_refactoring_candidates",
                "inputs": ["code_smells", "complexity_metrics", "duplicates"],
                "scoring_weights": {
                    "complexity": 0.3,
                    "smell_severity": 0.3,
                    "code_churn": 0.2,
                    "bug_correlation": 0.2,
                },
            },
        },
        {
            "id": "debate_approach",
            "type": "debate",
            "name": "Debate Refactoring Approach",
            "description": "Multi-agent debate on best refactoring strategy (simplicity vs performance)",
            "config": {
                "agents": [
                    "code_quality_reviewer",
                    "performance_specialist",
                    "architecture_reviewer",
                ],
                "rounds": 3,
                "topic_template": (
                    "What's the best refactoring approach for these candidates? "
                    "Candidates: {refactoring_candidates}. "
                    "Consider: readability, performance, maintainability, and backward compatibility."
                ),
                "consensus_mechanism": "weighted",
            },
        },
        {
            "id": "design_refactor",
            "type": "task",
            "name": "Design Refactoring Plan",
            "description": "Create detailed refactoring plan with specific transformations",
            "config": {
                "task_type": "function",
                "function_name": "design_refactoring_plan",
                "transformations": [
                    "extract_method",
                    "extract_class",
                    "inline_method",
                    "move_method",
                    "rename_symbol",
                    "replace_conditional_with_polymorphism",
                    "introduce_parameter_object",
                    "replace_magic_number",
                    "decompose_conditional",
                ],
            },
        },
        {
            "id": "snapshot_before",
            "type": "task",
            "name": "Capture Pre-Refactor State",
            "description": "Snapshot code state and test results before refactoring",
            "config": {
                "task_type": "function",
                "function_name": "capture_code_snapshot",
                "capture": ["source_hash", "test_results", "metrics"],
            },
        },
        {
            "id": "implement_changes",
            "type": "task",
            "name": "Implement Refactoring",
            "description": "Apply planned refactoring transformations",
            "config": {
                "task_type": "function",
                "function_name": "apply_refactoring",
                "inputs": ["refactoring_plan"],
                "dry_run": False,
            },
        },
        {
            "id": "verify_semantic_equivalence",
            "type": "task",
            "name": "Verify Semantic Equivalence",
            "description": "Ensure refactored code produces same outputs as original",
            "config": {
                "task_type": "function",
                "function_name": "verify_semantic_equivalence",
                "methods": [
                    "test_execution",
                    "output_comparison",
                    "contract_verification",
                ],
            },
        },
        {
            "id": "run_tests",
            "type": "task",
            "name": "Run Test Suite",
            "description": "Execute full test suite on refactored code",
            "config": {
                "task_type": "shell",
                "command": "pytest {test_dir} --tb=short -v",
                "timeout": 300,
            },
        },
        {
            "id": "tests_pass_gate",
            "type": "decision",
            "name": "Tests Pass Gate",
            "description": "Check if all tests pass after refactoring",
            "config": {
                "condition": "test_exit_code == 0",
                "true_target": "compare_metrics",
                "false_target": "rollback_changes",
            },
        },
        {
            "id": "rollback_changes",
            "type": "task",
            "name": "Rollback Refactoring",
            "description": "Revert changes if tests fail",
            "config": {
                "task_type": "function",
                "function_name": "rollback_to_snapshot",
                "inputs": ["pre_refactor_snapshot"],
            },
        },
        {
            "id": "compare_metrics",
            "type": "task",
            "name": "Compare Before/After Metrics",
            "description": "Compare complexity metrics before and after refactoring",
            "config": {
                "task_type": "function",
                "function_name": "compare_metrics",
                "metrics": ["complexity", "maintainability", "lines_of_code"],
            },
        },
        {
            "id": "human_approval",
            "type": "human_checkpoint",
            "name": "Developer Review",
            "description": "Developer reviews refactoring changes before finalizing",
            "config": {
                "approval_type": "review",
                "show_diff": True,
                "checklist": [
                    "Code is more readable than before",
                    "No logic changes introduced",
                    "Variable names are clearer",
                    "Functions are appropriately sized",
                    "Duplication has been reduced",
                ],
            },
        },
        {
            "id": "store_results",
            "type": "memory_write",
            "name": "Store Refactoring Results",
            "description": "Save refactoring metadata to knowledge base",
            "config": {
                "mound_type": "refactoring",
                "data_keys": [
                    "transformations_applied",
                    "metrics_improvement",
                    "smells_fixed",
                    "review_feedback",
                ],
            },
        },
    ],
    "transitions": [
        {"from": "detect_code_smells", "to": "calculate_complexity"},
        {"from": "calculate_complexity", "to": "find_duplicates"},
        {"from": "find_duplicates", "to": "identify_candidates"},
        {"from": "identify_candidates", "to": "debate_approach"},
        {"from": "debate_approach", "to": "design_refactor"},
        {"from": "design_refactor", "to": "snapshot_before"},
        {"from": "snapshot_before", "to": "implement_changes"},
        {"from": "implement_changes", "to": "verify_semantic_equivalence"},
        {"from": "verify_semantic_equivalence", "to": "run_tests"},
        {"from": "run_tests", "to": "tests_pass_gate"},
        {"from": "tests_pass_gate", "to": "compare_metrics", "condition": "tests_pass"},
        {"from": "tests_pass_gate", "to": "rollback_changes", "condition": "tests_fail"},
        {"from": "rollback_changes", "to": "design_refactor"},  # Retry with modified plan
        {"from": "compare_metrics", "to": "human_approval"},
        {"from": "human_approval", "to": "store_results"},
    ],
}


EXTRACT_METHOD_TEMPLATE: dict[str, Any] = {
    "name": "Extract Method Refactoring",
    "description": "Extract code block into a new method with proper parameters",
    "category": "refactoring",
    "version": "1.0",
    "tags": ["refactoring", "extract-method", "simplification"],
    "steps": [
        {
            "id": "analyze_block",
            "type": "task",
            "name": "Analyze Code Block",
            "description": "Analyze selected code block for extractable logic",
            "config": {
                "task_type": "function",
                "function_name": "analyze_code_block",
                "outputs": ["variables_in", "variables_out", "dependencies"],
            },
        },
        {
            "id": "determine_signature",
            "type": "task",
            "name": "Determine Method Signature",
            "description": "Calculate optimal parameters and return type for new method",
            "config": {
                "task_type": "function",
                "function_name": "determine_method_signature",
            },
        },
        {
            "id": "suggest_name",
            "type": "debate",
            "name": "Suggest Method Name",
            "description": "Multi-agent debate on best method name",
            "config": {
                "agents": ["code_quality_reviewer", "documentation_specialist"],
                "rounds": 2,
                "topic_template": (
                    "What should this extracted method be named? "
                    "Logic: {code_block_summary}. Parameters: {parameters}"
                ),
            },
        },
        {
            "id": "extract_method",
            "type": "task",
            "name": "Perform Extraction",
            "description": "Extract code block into new method",
            "config": {
                "task_type": "function",
                "function_name": "extract_method",
                "inputs": ["code_block", "method_name", "signature"],
            },
        },
        {
            "id": "update_call_sites",
            "type": "task",
            "name": "Update Call Sites",
            "description": "Replace original code with method call",
            "config": {
                "task_type": "function",
                "function_name": "replace_with_method_call",
            },
        },
        {
            "id": "verify_extraction",
            "type": "task",
            "name": "Verify Extraction",
            "description": "Run tests to verify extraction didn't break anything",
            "config": {
                "task_type": "shell",
                "command": "pytest {affected_tests} --tb=short",
            },
        },
    ],
    "transitions": [
        {"from": "analyze_block", "to": "determine_signature"},
        {"from": "determine_signature", "to": "suggest_name"},
        {"from": "suggest_name", "to": "extract_method"},
        {"from": "extract_method", "to": "update_call_sites"},
        {"from": "update_call_sites", "to": "verify_extraction"},
    ],
}


REDUCE_COMPLEXITY_TEMPLATE: dict[str, Any] = {
    "name": "Reduce Cyclomatic Complexity",
    "description": "Reduce complexity of methods with high cyclomatic complexity",
    "category": "refactoring",
    "version": "1.0",
    "tags": ["refactoring", "complexity", "simplification"],
    "steps": [
        {
            "id": "identify_complex_methods",
            "type": "task",
            "name": "Identify Complex Methods",
            "description": "Find methods exceeding complexity threshold",
            "config": {
                "task_type": "function",
                "function_name": "find_complex_methods",
                "threshold": 10,
            },
        },
        {
            "id": "analyze_complexity_sources",
            "type": "task",
            "name": "Analyze Complexity Sources",
            "description": "Identify what's causing high complexity (nested ifs, loops, etc.)",
            "config": {
                "task_type": "function",
                "function_name": "analyze_complexity_sources",
                "sources": ["nested_conditionals", "loops", "switch_cases", "boolean_expressions"],
            },
        },
        {
            "id": "suggest_simplifications",
            "type": "debate",
            "name": "Suggest Simplifications",
            "description": "Multi-agent debate on complexity reduction strategies",
            "config": {
                "agents": [
                    "code_quality_reviewer",
                    "performance_specialist",
                ],
                "rounds": 2,
                "topic_template": (
                    "How should we reduce complexity in {method_name}? "
                    "Current complexity: {complexity}. "
                    "Sources: {complexity_sources}"
                ),
                "strategies": [
                    "guard_clauses",
                    "extract_method",
                    "replace_conditional_with_polymorphism",
                    "decompose_conditional",
                    "consolidate_conditional",
                ],
            },
        },
        {
            "id": "apply_simplifications",
            "type": "task",
            "name": "Apply Simplifications",
            "description": "Apply chosen complexity reduction strategies",
            "config": {
                "task_type": "function",
                "function_name": "apply_complexity_reduction",
            },
        },
        {
            "id": "verify_reduction",
            "type": "task",
            "name": "Verify Complexity Reduction",
            "description": "Measure new complexity and ensure reduction",
            "config": {
                "task_type": "function",
                "function_name": "verify_complexity_reduction",
            },
        },
        {
            "id": "run_tests",
            "type": "task",
            "name": "Run Tests",
            "description": "Ensure tests still pass after simplification",
            "config": {
                "task_type": "shell",
                "command": "pytest --tb=short",
            },
        },
    ],
    "transitions": [
        {"from": "identify_complex_methods", "to": "analyze_complexity_sources"},
        {"from": "analyze_complexity_sources", "to": "suggest_simplifications"},
        {"from": "suggest_simplifications", "to": "apply_simplifications"},
        {"from": "apply_simplifications", "to": "verify_reduction"},
        {"from": "verify_reduction", "to": "run_tests"},
    ],
}


ELIMINATE_DUPLICATION_TEMPLATE: dict[str, Any] = {
    "name": "Eliminate Code Duplication",
    "description": "Find and eliminate duplicate code through abstraction",
    "category": "refactoring",
    "version": "1.0",
    "tags": ["refactoring", "duplication", "dry"],
    "steps": [
        {
            "id": "find_duplicates",
            "type": "task",
            "name": "Find Duplicate Code",
            "description": "Detect exact and near-duplicate code blocks",
            "config": {
                "task_type": "function",
                "function_name": "find_code_duplicates",
                "detection_methods": ["token_based", "ast_based", "semantic"],
                "min_lines": 5,
                "similarity_threshold": 0.8,
            },
        },
        {
            "id": "cluster_duplicates",
            "type": "task",
            "name": "Cluster Similar Duplicates",
            "description": "Group related duplicates for batch handling",
            "config": {
                "task_type": "function",
                "function_name": "cluster_duplicates",
            },
        },
        {
            "id": "analyze_variations",
            "type": "task",
            "name": "Analyze Variations",
            "description": "Understand differences between duplicate instances",
            "config": {
                "task_type": "function",
                "function_name": "analyze_duplicate_variations",
                "outputs": ["common_code", "variable_parts", "parameterizable"],
            },
        },
        {
            "id": "debate_abstraction",
            "type": "debate",
            "name": "Debate Abstraction Strategy",
            "description": "Multi-agent debate on best way to eliminate duplicates",
            "config": {
                "agents": [
                    "code_quality_reviewer",
                    "architecture_reviewer",
                ],
                "rounds": 2,
                "topic_template": (
                    "How should we eliminate this duplication? "
                    "Instances: {duplicate_count}. Variations: {variations}"
                ),
                "strategies": [
                    "extract_shared_function",
                    "template_method_pattern",
                    "strategy_pattern",
                    "parameterize_method",
                    "pull_up_to_base_class",
                ],
            },
        },
        {
            "id": "create_abstraction",
            "type": "task",
            "name": "Create Abstraction",
            "description": "Create shared function/class to eliminate duplication",
            "config": {
                "task_type": "function",
                "function_name": "create_shared_abstraction",
            },
        },
        {
            "id": "update_duplicate_sites",
            "type": "task",
            "name": "Update Duplicate Sites",
            "description": "Replace duplicate code with calls to shared abstraction",
            "config": {
                "task_type": "function",
                "function_name": "replace_duplicates_with_abstraction",
            },
        },
        {
            "id": "verify_deduplication",
            "type": "task",
            "name": "Verify Deduplication",
            "description": "Run tests and verify behavior is preserved",
            "config": {
                "task_type": "shell",
                "command": "pytest --tb=short -v",
            },
        },
    ],
    "transitions": [
        {"from": "find_duplicates", "to": "cluster_duplicates"},
        {"from": "cluster_duplicates", "to": "analyze_variations"},
        {"from": "analyze_variations", "to": "debate_abstraction"},
        {"from": "debate_abstraction", "to": "create_abstraction"},
        {"from": "create_abstraction", "to": "update_duplicate_sites"},
        {"from": "update_duplicate_sites", "to": "verify_deduplication"},
    ],
}


# Template registry
REFACTORING_TEMPLATES = {
    "refactoring_workflow": REFACTORING_WORKFLOW_TEMPLATE,
    "extract_method": EXTRACT_METHOD_TEMPLATE,
    "reduce_complexity": REDUCE_COMPLEXITY_TEMPLATE,
    "eliminate_duplication": ELIMINATE_DUPLICATION_TEMPLATE,
}


def get_refactoring_template(name: str) -> dict[str, Any]:
    """
    Get a refactoring workflow template by name.

    Args:
        name: Template name

    Returns:
        Template dictionary

    Raises:
        KeyError: If template not found
    """
    if name not in REFACTORING_TEMPLATES:
        raise KeyError(
            f"Unknown refactoring template: {name}. Available: {list(REFACTORING_TEMPLATES.keys())}"
        )
    return REFACTORING_TEMPLATES[name]


def get_code_smell_thresholds() -> dict[str, int | float]:
    """Get current code smell detection thresholds."""
    return CODE_SMELL_THRESHOLDS.copy()


def update_code_smell_threshold(smell: str, value: int | float) -> None:
    """
    Update a code smell detection threshold.

    Args:
        smell: Name of the smell threshold
        value: New threshold value
    """
    if smell not in CODE_SMELL_THRESHOLDS:
        raise KeyError(
            f"Unknown smell threshold: {smell}. Available: {list(CODE_SMELL_THRESHOLDS.keys())}"
        )
    CODE_SMELL_THRESHOLDS[smell] = value
