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
    """Get runtime agent count from the same registry used by production code."""
    try:
        from aragora.agents.base import list_available_agents
    except Exception as exc:  # pragma: no cover - import/runtime environment failures
        print(f"ERROR: failed to load runtime registry: {exc}")
        sys.exit(2)

    return len(list_available_agents())


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
                errors.append(f"{filename}: says {doc_count} agents, registry has {runtime_count}")
                if fix:
                    content = content.replace(
                        m.group(0), m.group(0).replace(str(doc_count), str(runtime_count))
                    )

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
