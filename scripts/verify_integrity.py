#!/usr/bin/env python3
"""
Genetic Screen: AST-based integrity verification for proprioceptive fixes.

This script uses Python's Abstract Syntax Tree (AST) module to verify that
critical safety mechanisms exist in the codebase. It prevents regression by
checking for the presence of:

1. OutputSanitizer.sanitize_agent_output - prevents null byte crashes
2. _with_timeout - prevents stalled agents from blocking debates
3. ws._bound_loop_id - enables WebSocket state recovery

Usage:
    python scripts/verify_integrity.py           # Run all checks
    python scripts/verify_integrity.py --verbose # Show details
    python scripts/verify_integrity.py --fix     # Show suggested fixes
"""

import ast
import argparse
import sys
from pathlib import Path
from typing import NamedTuple


class CheckResult(NamedTuple):
    """Result of an integrity check."""

    name: str
    passed: bool
    file: str
    details: str


def find_method_calls(tree: ast.AST, method_name: str) -> list[tuple[int, str]]:
    """Find all calls to a method by name."""
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check for Attribute call like obj.method()
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == method_name:
                    calls.append((node.lineno, method_name))
            # Check for Name call like method()
            elif isinstance(node.func, ast.Name):
                if node.func.id == method_name:
                    calls.append((node.lineno, method_name))
    return calls


def find_attribute_assignments(tree: ast.AST, attr_name: str) -> list[tuple[int, str]]:
    """Find all assignments to an attribute by name."""
    assignments = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    if target.attr == attr_name:
                        assignments.append((node.lineno, attr_name))
    return assignments


def find_function_def(tree: ast.AST, func_name: str) -> list[tuple[int, str]]:
    """Find function definitions by name."""
    defs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func_name:
                defs.append((node.lineno, func_name))
    return defs


def check_sanitization_exists(root: Path) -> CheckResult:
    """
    Verify OutputSanitizer.sanitize_agent_output is used in orchestrator.

    This check ensures that agent outputs are sanitized to prevent
    null byte crashes and other encoding issues.
    """
    filepath = root / "aragora" / "debate" / "orchestrator.py"

    if not filepath.exists():
        return CheckResult(
            name="sanitization", passed=False, file=str(filepath), details="File not found"
        )

    try:
        with open(filepath) as f:
            tree = ast.parse(f.read())

        # Look for calls to sanitize_agent_output
        calls = find_method_calls(tree, "sanitize_agent_output")

        if calls:
            return CheckResult(
                name="sanitization",
                passed=True,
                file=str(filepath),
                details=f"Found {len(calls)} call(s) at lines: {[c[0] for c in calls]}",
            )
        else:
            return CheckResult(
                name="sanitization",
                passed=False,
                file=str(filepath),
                details="No calls to sanitize_agent_output found",
            )

    except SyntaxError as e:
        return CheckResult(
            name="sanitization", passed=False, file=str(filepath), details=f"Syntax error: {e}"
        )


def check_timeout_exists(root: Path) -> CheckResult:
    """
    Verify _with_timeout wraps agent generation calls.

    This check ensures that agent calls have active timeouts
    to prevent stalled agents from blocking the entire debate.
    """
    filepath = root / "aragora" / "debate" / "orchestrator.py"

    if not filepath.exists():
        return CheckResult(
            name="timeout", passed=False, file=str(filepath), details="File not found"
        )

    try:
        with open(filepath) as f:
            tree = ast.parse(f.read())

        # Look for _with_timeout function definition
        defs = find_function_def(tree, "_with_timeout")

        # Look for calls to _with_timeout
        calls = find_method_calls(tree, "_with_timeout")

        if defs and calls:
            return CheckResult(
                name="timeout",
                passed=True,
                file=str(filepath),
                details=f"Found definition at line {defs[0][0]}, {len(calls)} call(s)",
            )
        elif defs:
            return CheckResult(
                name="timeout",
                passed=False,
                file=str(filepath),
                details="_with_timeout defined but never called",
            )
        else:
            return CheckResult(
                name="timeout", passed=False, file=str(filepath), details="_with_timeout not found"
            )

    except SyntaxError as e:
        return CheckResult(
            name="timeout", passed=False, file=str(filepath), details=f"Syntax error: {e}"
        )


def check_loop_binding(root: Path) -> CheckResult:
    """
    Verify ws._bound_loop_id binding exists for WebSocket state recovery.

    This check ensures that loop_id is bound to WebSocket instances
    for recovery when clients reconnect or lose state.
    """
    filepath = root / "aragora" / "server" / "stream.py"

    if not filepath.exists():
        return CheckResult(
            name="loop_binding", passed=False, file=str(filepath), details="File not found"
        )

    try:
        with open(filepath) as f:
            tree = ast.parse(f.read())

        # Look for assignments to _bound_loop_id
        assignments = find_attribute_assignments(tree, "_bound_loop_id")

        if assignments:
            return CheckResult(
                name="loop_binding",
                passed=True,
                file=str(filepath),
                details=f"Found {len(assignments)} binding(s) at lines: {[a[0] for a in assignments]}",
            )
        else:
            return CheckResult(
                name="loop_binding",
                passed=False,
                file=str(filepath),
                details="No ws._bound_loop_id assignment found",
            )

    except SyntaxError as e:
        return CheckResult(
            name="loop_binding", passed=False, file=str(filepath), details=f"Syntax error: {e}"
        )


def check_generate_with_agent(root: Path) -> CheckResult:
    """
    Verify _generate_with_agent method exists for autonomic error handling.

    This check ensures that agent generation is wrapped with error
    containment to prevent crashes from propagating.
    """
    filepath = root / "aragora" / "debate" / "orchestrator.py"

    if not filepath.exists():
        return CheckResult(
            name="generate_with_agent", passed=False, file=str(filepath), details="File not found"
        )

    try:
        with open(filepath) as f:
            tree = ast.parse(f.read())

        # Look for _generate_with_agent function definition
        defs = find_function_def(tree, "_generate_with_agent")

        # Look for calls to _generate_with_agent
        calls = find_method_calls(tree, "_generate_with_agent")

        if defs and calls:
            return CheckResult(
                name="generate_with_agent",
                passed=True,
                file=str(filepath),
                details=f"Found definition at line {defs[0][0]}, {len(calls)} call(s)",
            )
        elif defs:
            return CheckResult(
                name="generate_with_agent",
                passed=False,
                file=str(filepath),
                details="_generate_with_agent defined but never called",
            )
        else:
            return CheckResult(
                name="generate_with_agent",
                passed=False,
                file=str(filepath),
                details="_generate_with_agent not found",
            )

    except SyntaxError as e:
        return CheckResult(
            name="generate_with_agent",
            passed=False,
            file=str(filepath),
            details=f"Syntax error: {e}",
        )


def check_sanitization_module(root: Path) -> CheckResult:
    """
    Verify sanitization module exists with OutputSanitizer class.
    """
    filepath = root / "aragora" / "debate" / "sanitization.py"

    if not filepath.exists():
        return CheckResult(
            name="sanitization_module", passed=False, file=str(filepath), details="File not found"
        )

    try:
        with open(filepath) as f:
            tree = ast.parse(f.read())

        # Look for OutputSanitizer class definition
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name == "OutputSanitizer":
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    return CheckResult(
                        name="sanitization_module",
                        passed=True,
                        file=str(filepath),
                        details=f"OutputSanitizer found with methods: {methods}",
                    )

        return CheckResult(
            name="sanitization_module",
            passed=False,
            file=str(filepath),
            details="OutputSanitizer class not found",
        )

    except SyntaxError as e:
        return CheckResult(
            name="sanitization_module",
            passed=False,
            file=str(filepath),
            details=f"Syntax error: {e}",
        )


def run_all_checks(root: Path, verbose: bool = False) -> list[CheckResult]:
    """Run all integrity checks."""
    checks = [
        check_sanitization_module,
        check_sanitization_exists,
        check_timeout_exists,
        check_generate_with_agent,
        check_loop_binding,
    ]

    results = []
    for check in checks:
        result = check(root)
        results.append(result)

        if verbose:
            status = "PASS" if result.passed else "FAIL"
            print(f"[{status}] {result.name}: {result.details}")
            print(f"       File: {result.file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Verify proprioceptive fixes exist in the codebase"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--fix", action="store_true", help="Show suggested fixes for failures")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Root directory of the aragora project",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ARAGORA INTEGRITY VERIFICATION")
    print("Checking for proprioceptive (self-stabilization) fixes...")
    print("=" * 60)
    print()

    results = run_all_checks(args.root, args.verbose)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    print()
    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("-" * 60)

    if failed > 0:
        print()
        print("FAILURES:")
        for result in results:
            if not result.passed:
                print(f"  - {result.name}: {result.details}")
                if args.fix:
                    print(f"    FIX: Check {result.file} for missing implementation")

        print()
        print("Integrity check FAILED. Some proprioceptive fixes are missing.")
        sys.exit(1)
    else:
        print()
        print("All integrity checks PASSED.")
        print("Proprioceptive fixes are in place.")
        sys.exit(0)


if __name__ == "__main__":
    main()
