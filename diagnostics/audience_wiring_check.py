#!/usr/bin/env python3
"""
Diagnostic script to check Arena instantiations for event_emitter wiring.

This script uses AST parsing to find all Arena() calls in the codebase
and reports which ones are missing the event_emitter argument needed
for audience participation.

Usage:
    python diagnostics/audience_wiring_check.py
"""

import ast
import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ArenaCall:
    """Represents an Arena instantiation found in code."""

    filepath: str
    line: int
    has_event_emitter: bool
    has_loop_id: bool
    context: str  # Surrounding code for context


def find_arena_calls(filepath: Path) -> list[ArenaCall]:
    """Find all Arena() calls in a Python file."""
    try:
        with open(filepath, "r") as f:
            source = f.read()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return []

    calls = []
    lines = source.split("\n")

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check if this is an Arena() call
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name == "Arena":
                # Check for event_emitter and loop_id arguments
                has_event_emitter = False
                has_loop_id = False

                for keyword in node.keywords:
                    if keyword.arg == "event_emitter":
                        has_event_emitter = True
                    if keyword.arg == "loop_id":
                        has_loop_id = True

                # Get context (the line of code)
                line_num = node.lineno
                context = lines[line_num - 1].strip() if line_num <= len(lines) else ""

                calls.append(
                    ArenaCall(
                        filepath=str(filepath),
                        line=line_num,
                        has_event_emitter=has_event_emitter,
                        has_loop_id=has_loop_id,
                        context=context[:100],  # Truncate for display
                    )
                )

    return calls


def scan_codebase(root: Path) -> list[ArenaCall]:
    """Scan entire codebase for Arena instantiations."""
    all_calls = []

    # Directories to skip
    skip_dirs = {".git", "__pycache__", "node_modules", ".comparison", "venv", ".venv"}

    for dirpath, dirnames, filenames in os.walk(root):
        # Filter out skip directories
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]

        for filename in filenames:
            if filename.endswith(".py"):
                filepath = Path(dirpath) / filename
                calls = find_arena_calls(filepath)
                all_calls.extend(calls)

    return all_calls


def main():
    """Run the diagnostic check."""
    root = Path(__file__).parent.parent

    print("=" * 70)
    print("AUDIENCE WIRING DIAGNOSTIC")
    print("Checking Arena instantiations for event_emitter parameter")
    print("=" * 70)
    print()

    calls = scan_codebase(root)

    if not calls:
        print("No Arena() calls found in codebase.")
        return

    # Categorize calls
    missing_emitter = [c for c in calls if not c.has_event_emitter]
    has_emitter = [c for c in calls if c.has_event_emitter]

    print(f"Found {len(calls)} Arena instantiations:")
    print(f"  - {len(has_emitter)} with event_emitter (audience enabled)")
    print(f"  - {len(missing_emitter)} without event_emitter (audience disabled)")
    print()

    if missing_emitter:
        print("-" * 70)
        print("ARENA CALLS MISSING event_emitter:")
        print("-" * 70)
        for call in missing_emitter:
            print(f"\n  File: {call.filepath}:{call.line}")
            print(f"  Code: {call.context}")
            if not call.has_loop_id:
                print(f"  Note: Also missing loop_id")

    if has_emitter:
        print()
        print("-" * 70)
        print("ARENA CALLS WITH event_emitter (OK):")
        print("-" * 70)
        for call in has_emitter:
            print(f"\n  File: {call.filepath}:{call.line}")
            loop_status = "has loop_id" if call.has_loop_id else "missing loop_id"
            print(f"  Status: {loop_status}")

    print()
    print("=" * 70)
    if missing_emitter:
        print(
            f"RESULT: {len(missing_emitter)} Arena calls need event_emitter for audience participation"
        )
    else:
        print("RESULT: All Arena calls have event_emitter wired!")
    print("=" * 70)


if __name__ == "__main__":
    main()
