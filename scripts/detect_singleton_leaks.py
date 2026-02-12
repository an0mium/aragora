#!/usr/bin/env python3
"""Detect singleton patterns in aragora/ not covered by conftest reset.

Scans for the common singleton pattern:
    _foo: SomeType | None = None
    def get_foo() -> SomeType: ...

Then cross-references against the reset list in tests/conftest.py::_reset_lazy_globals_impl
to find any singletons that aren't being cleaned between tests.

Usage:
    python scripts/detect_singleton_leaks.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "aragora"
CONFTEST = ROOT / "tests" / "conftest.py"


def find_singletons(src_dir: Path) -> list[dict]:
    """Find module-level singleton patterns using AST analysis."""
    singletons: list[dict] = []

    for py_file in sorted(src_dir.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue

        try:
            source = py_file.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue

        # Collect module-level private globals assigned to None
        private_globals: dict[str, int] = {}
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.startswith("_"):
                        # Check if assigned to None
                        if isinstance(node.value, ast.Constant) and node.value.value is None:
                            private_globals[target.id] = node.lineno
            elif isinstance(node, ast.AnnAssign) and node.target and isinstance(node.target, ast.Name):
                name = node.target.id
                if name.startswith("_"):
                    # Check if value is None
                    if node.value and isinstance(node.value, ast.Constant) and node.value.value is None:
                        private_globals[name] = node.lineno

        if not private_globals:
            continue

        # Look for getter functions that reference these globals
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("get_"):
                # Check if function body references a `global _foo` statement
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.Global):
                        for global_name in stmt.names:
                            if global_name in private_globals:
                                rel_path = py_file.relative_to(ROOT)
                                module = str(rel_path).replace("/", ".").removesuffix(".py")
                                singletons.append({
                                    "module": module,
                                    "var": global_name,
                                    "getter": node.name,
                                    "line": private_globals[global_name],
                                    "file": str(rel_path),
                                })

    return singletons


def find_covered_resets(conftest: Path) -> set[tuple[str, str]]:
    """Parse _reset_lazy_globals_impl to find (module, var) pairs being reset."""
    covered: set[tuple[str, str]] = set()

    source = conftest.read_text(encoding="utf-8")

    # Find the function body
    match = re.search(r"def _reset_lazy_globals_impl\(\).*?(?=\n(?:def |@pytest|class )|\Z)", source, re.DOTALL)
    if not match:
        print("WARNING: Could not find _reset_lazy_globals_impl in conftest.py")
        return covered

    body = match.group(0)

    # Find all `import aragora.foo.bar as _alias` patterns
    import_pattern = re.compile(r"import\s+(aragora\.[.\w]+)\s+as\s+(\w+)")
    # Find all `_alias._var = ...` patterns
    reset_pattern = re.compile(r"(\w+)\.(_\w+)\s*=")
    # Find all `_alias._var.clear()` patterns
    clear_pattern = re.compile(r"(\w+)\.(_\w+)\.clear\(\)")

    # Build alias → module mapping
    alias_map: dict[str, str] = {}
    for m in import_pattern.finditer(body):
        alias_map[m.group(2)] = m.group(1)

    # Find all resets
    for m in reset_pattern.finditer(body):
        alias, var = m.group(1), m.group(2)
        if alias in alias_map:
            covered.add((alias_map[alias], var))

    for m in clear_pattern.finditer(body):
        alias, var = m.group(1), m.group(2)
        if alias in alias_map:
            covered.add((alias_map[alias], var))

    return covered


def main() -> int:
    singletons = find_singletons(SRC_DIR)
    covered = find_covered_resets(CONFTEST)

    # Cross-reference
    uncovered = []
    for s in singletons:
        key = (s["module"], s["var"])
        if key not in covered:
            uncovered.append(s)

    if not uncovered:
        print(f"All {len(singletons)} singletons are covered by conftest resets.")
        return 0

    print(f"Found {len(uncovered)} uncovered singletons (out of {len(singletons)} total):\n")
    for s in uncovered:
        print(f"  {s['module']}.{s['var']}  (getter: {s['getter']}, line {s['line']})")
        print(f"    {s['file']}:{s['line']}")
        print()

    print(f"Total: {len(uncovered)} uncovered, {len(singletons) - len(uncovered)} covered")
    return 1


if __name__ == "__main__":
    sys.exit(main())
