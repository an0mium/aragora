#!/usr/bin/env python3
"""
Check agent registry drift between runtime and documentation.

Validates that the agent count in documentation files matches the
actual runtime registry. Fails with exit code 1 on mismatch.

Usage:
    python scripts/check_agent_registry_drift.py
    python scripts/check_agent_registry_drift.py --fix  # Update doc counts
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def get_runtime_agent_count() -> int:
    """Get the actual agent count by scanning @AgentRegistry.register() decorators."""
    agents_dir = Path("aragora/agents")
    if not agents_dir.exists():
        print("ERROR: aragora/agents/ not found")
        sys.exit(2)

    # Scan all .py files for @AgentRegistry.register("type_name", ...) decorators
    # The type name is on the line following the decorator
    type_names: set[str] = set()
    decorator_re = re.compile(r"@AgentRegistry\.register\(")
    name_re = re.compile(r'^\s*"([a-z][a-z0-9-]*)"')

    for py_file in agents_dir.rglob("*.py"):
        lines = py_file.read_text().splitlines()
        for i, line in enumerate(lines):
            if decorator_re.search(line):
                # Check same line for inline name
                inline = re.search(r'register\(\s*"([a-z][a-z0-9-]*)"', line)
                if inline:
                    type_names.add(inline.group(1))
                # Also check next line for name on separate line
                elif i + 1 < len(lines):
                    m = name_re.match(lines[i + 1])
                    if m:
                        type_names.add(m.group(1))

    return len(type_names)


def check_doc_counts(runtime_count: int, fix: bool = False) -> list[str]:
    """Check documentation files for stale agent counts."""
    errors: list[str] = []

    # Files and patterns to check
    checks = [
        ("AGENTS.md", r"registers (\d+) agent types"),
        ("README.md", r"orchestrates (\d+) (?:AI )?agent"),
        ("README.md", r"# (\d+) registered agent types"),
        ("CLAUDE.md", r"orchestrating (\d+) agent types"),
    ]

    for filename, pattern in checks:
        path = Path(filename)
        if not path.exists():
            continue

        content = path.read_text()
        for m in re.finditer(pattern, content):
            doc_count = int(m.group(1))
            if doc_count != runtime_count:
                errors.append(
                    f"{filename}: says {doc_count} agents, "
                    f"registry has {runtime_count}"
                )
                if fix:
                    content = content.replace(m.group(0), m.group(0).replace(
                        str(doc_count), str(runtime_count)
                    ))

        if fix and errors:
            path.write_text(content)

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check agent registry drift")
    parser.add_argument("--fix", action="store_true", help="Auto-fix doc counts")
    args = parser.parse_args()

    runtime_count = get_runtime_agent_count()
    print(f"Runtime agent registry: {runtime_count} types")

    errors = check_doc_counts(runtime_count, fix=args.fix)

    if errors:
        if args.fix:
            print(f"Fixed {len(errors)} drift(s):")
        else:
            print(f"DRIFT DETECTED ({len(errors)} mismatch(es)):")
        for err in errors:
            print(f"  - {err}")
        return 0 if args.fix else 1

    print("OK: All documentation matches runtime registry")
    return 0


if __name__ == "__main__":
    sys.exit(main())
