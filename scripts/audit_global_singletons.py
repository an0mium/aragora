#!/usr/bin/env python3
"""Audit module-level singletons in aragora/ and check they are reset in tests.

Scans for common singleton patterns (_name = None, _name = {}, _name = [])
at module level and verifies each is referenced in the test conftest reset
function. Reports untracked globals as warnings.

Exit 0 always (advisory, not blocking).

Usage:
    python scripts/audit_global_singletons.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ARAGORA_ROOT = Path(__file__).resolve().parent.parent / "aragora"
CONFTEST_PATH = Path(__file__).resolve().parent.parent / "tests" / "conftest.py"

# Patterns to match: module-level assignments like _foo = None, _bar = {}, _baz = []
SENTINEL_VALUES = {type(None), ast.Constant, ast.Dict, ast.List}


def _is_singleton_pattern(node: ast.Assign) -> str | None:
    """Return the variable name if this looks like a resettable singleton."""
    if len(node.targets) != 1:
        return None
    target = node.targets[0]
    if not isinstance(target, ast.Name):
        return None
    name = target.id
    # Must start with _ (private module-level convention)
    if not name.startswith("_") or name.startswith("__"):
        return None
    # Check value is None, {}, or []
    value = node.value
    if isinstance(value, ast.Constant) and value.value is None:
        return name
    if isinstance(value, (ast.Dict, ast.List)) and not (
        getattr(value, "keys", None) and value.keys or getattr(value, "elts", None) and value.elts
    ):
        return name
    return None


def scan_singletons() -> dict[str, list[str]]:
    """Scan aragora/ for module-level singleton patterns.

    Returns: {module_dotpath: [var_name, ...]}
    """
    results: dict[str, list[str]] = {}
    for py_file in sorted(ARAGORA_ROOT.rglob("*.py")):
        # Skip test files, __pycache__, migrations
        rel = py_file.relative_to(ARAGORA_ROOT.parent)
        if "__pycache__" in str(rel) or "test" in py_file.name:
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except SyntaxError:
            continue
        singletons = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                name = _is_singleton_pattern(node)
                if name:
                    singletons.append(name)
        if singletons:
            # Convert file path to dotted module path
            module = str(rel).replace("/", ".").removesuffix(".py")
            results[module] = singletons
    return results


def load_conftest_resets() -> str:
    """Load the conftest file content for searching."""
    if not CONFTEST_PATH.exists():
        return ""
    return CONFTEST_PATH.read_text(encoding="utf-8")


def main() -> None:
    singletons = scan_singletons()
    conftest_text = load_conftest_resets()

    tracked = 0
    untracked: list[tuple[str, str]] = []

    for module, names in sorted(singletons.items()):
        for name in names:
            # Check if this global is referenced in conftest reset logic
            # Look for the module import or direct attribute reference
            if name in conftest_text and module.split(".")[-1] in conftest_text:
                tracked += 1
            else:
                untracked.append((module, name))

    total = tracked + len(untracked)
    print(f"Singleton audit: {total} globals found, {tracked} tracked, {len(untracked)} untracked")
    print()

    if untracked:
        print("Untracked singletons (may need reset in tests/conftest.py):")
        for module, name in untracked:
            print(f"  {module}.{name}")
        print()
        print(
            "Not all untracked singletons need resetting -- only those that"
            " persist mutable state across tests. Review before adding resets."
        )
    else:
        print("All detected singletons are tracked in tests/conftest.py.")

    # Always exit 0 (advisory)
    sys.exit(0)


if __name__ == "__main__":
    main()
