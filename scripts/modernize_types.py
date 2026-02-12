#!/usr/bin/env python3
"""
Type Hint Modernization Script

Modernizes Python type hints to use Python 3.10+ syntax:
- Optional[T] -> T | None
- Union[A, B] -> A | B
- Dict[K, V] -> dict[K, V]
- List[T] -> list[T]
- Set[T] -> set[T]
- Tuple[T, ...] -> tuple[T, ...]
- FrozenSet[T] -> frozenset[T]

Also:
- Adds `from __future__ import annotations` when needed for forward references
- Removes unused typing imports when possible.

Usage:
    python scripts/modernize_types.py --dry-run     # Preview changes
    python scripts/modernize_types.py               # Apply changes
    python scripts/modernize_types.py --file path   # Process single file
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# Future import line
FUTURE_IMPORT = "from __future__ import annotations"

# Patterns to transform
# Order matters - process more specific patterns first
# NOTE: We intentionally do NOT transform Union[A, B] -> A | B because:
# 1. Type alias assignments like `MyType = Union[A, B]` fail at runtime with `|`
# 2. Union with forward references needs careful handling
# These should be done manually or with a more sophisticated tool.
PATTERNS: list[tuple[str, str]] = [
    # Optional[T] -> T | None (only for simple non-forward-reference types)
    # Skip if T starts with quote (forward reference)
    (r'Optional\[((?!")[^\[\]]+)\]', r"\1 | None"),
    # Handle nested Optional like Optional[List[str]]
    (r"Optional\[(list\[[^\]]+\])\]", r"\1 | None"),
    (r"Optional\[(dict\[[^\]]+\])\]", r"\1 | None"),
    (r"Optional\[(set\[[^\]]+\])\]", r"\1 | None"),
    (r"Optional\[(tuple\[[^\]]+\])\]", r"\1 | None"),
    # Built-in generic types (lowercase) - safe to transform
    (r"\bDict\[", r"dict["),
    (r"\bList\[", r"list["),
    (r"\bSet\[", r"set["),
    (r"\bTuple\[", r"tuple["),
    (r"\bFrozenSet\[", r"frozenset["),
    (r"\bType\[", r"type["),
]

# Types to remove from typing imports after transformation
REMOVABLE_IMPORTS = {"Optional", "Union", "Dict", "List", "Set", "Tuple", "FrozenSet", "Type"}


def needs_future_import(content: str) -> bool:
    """Check if file uses forward references that need from __future__ import annotations.

    Forward references (quoted strings in type hints) with | operator need
    deferred evaluation to work at runtime.
    """
    # Check for forward reference patterns that would fail at runtime
    # e.g., "ClassName" | None or SomeType["Ref"]
    if re.search(r'"\w+"[\s]*\|', content):
        return True
    if re.search(r'\|[\s]*"\w+"', content):
        return True
    return False


def ensure_future_import(content: str) -> str:
    """Ensure `from __future__ import annotations` is present.

    Adds it as the first import if not already present.
    """
    if FUTURE_IMPORT in content:
        return content

    # Find the right place to insert the import
    lines = content.split("\n")
    insert_idx = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Skip docstrings at the start
        if stripped.startswith('"""') or stripped.startswith("'''"):
            # Find the end of the docstring
            if stripped.count('"""') == 1 or stripped.count("'''") == 1:
                # Multiline docstring
                quote = '"""' if '"""' in stripped else "'''"
                for j in range(i + 1, len(lines)):
                    if quote in lines[j]:
                        insert_idx = j + 1
                        break
            else:
                # Single line docstring
                insert_idx = i + 1
            break
        # Skip comments
        elif stripped.startswith("#") or stripped == "":
            insert_idx = i + 1
        else:
            # Found first non-comment, non-docstring line
            insert_idx = i
            break

    # Insert the future import
    lines.insert(insert_idx, FUTURE_IMPORT)
    if insert_idx < len(lines) - 1 and lines[insert_idx + 1].strip():
        # Add blank line after if next line isn't blank
        lines.insert(insert_idx + 1, "")

    return "\n".join(lines)


def modernize_content(content: str) -> tuple[str, int]:
    """Apply type modernization patterns to content.

    Returns (new_content, change_count).
    """
    new_content = content
    total_changes = 0

    for pattern, replacement in PATTERNS:
        # Keep applying until no more matches (handles nested patterns)
        while True:
            new_text = re.sub(pattern, replacement, new_content)
            if new_text == new_content:
                break
            changes = len(re.findall(pattern, new_content))
            total_changes += changes
            new_content = new_text

    # Only add future import if we introduced | None patterns
    # The built-in generic transformations (Dict -> dict) don't need it
    if total_changes > 0 and " | None" in new_content and " | None" not in content:
        new_content = ensure_future_import(new_content)

    return new_content, total_changes


def clean_typing_imports(content: str) -> str:
    """Remove unused typing imports after modernization.

    This is conservative - only removes imports that are clearly unused.
    """
    # Find typing import statements
    import_pattern = r"from typing import ([^;\n]+)"

    def clean_import(match: re.Match) -> str:
        imports = match.group(1)
        # Parse the imports
        items = [item.strip() for item in imports.split(",")]

        # Filter out items that are no longer used
        new_items = []
        for item in items:
            # Remove 'as X' suffix for checking
            base_item = item.split(" as ")[0].strip()

            # Keep if not in removable set
            if base_item not in REMOVABLE_IMPORTS:
                new_items.append(item)
            # Or if still used somewhere in the file (check for lowercase version too)
            elif base_item in content.replace(match.group(0), ""):
                new_items.append(item)

        if not new_items:
            return ""  # Remove entire import line

        return f"from typing import {', '.join(new_items)}"

    new_content = re.sub(import_pattern, clean_import, content)

    # Clean up empty lines left by removed imports
    new_content = re.sub(r"\n\n\n+", "\n\n", new_content)

    return new_content


def process_file(path: Path, dry_run: bool = False) -> tuple[int, bool]:
    """Process a single file.

    Returns (change_count, had_errors).
    """
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  Error reading {path}: {e}")
        return 0, True

    new_content, changes = modernize_content(content)

    if changes == 0:
        return 0, False

    # Also clean up imports
    new_content = clean_typing_imports(new_content)

    if new_content == content:
        return 0, False

    if dry_run:
        print(f"  Would modify {path} ({changes} changes)")
    else:
        try:
            path.write_text(new_content, encoding="utf-8")
            print(f"  Modified {path} ({changes} changes)")
        except Exception as e:
            print(f"  Error writing {path}: {e}")
            return changes, True

    return changes, False


def main():
    parser = argparse.ArgumentParser(description="Modernize Python type hints")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--file", type=Path, help="Process a single file")
    parser.add_argument("--dir", type=Path, default=Path("aragora"), help="Directory to process")
    args = parser.parse_args()

    if args.file:
        files = [args.file]
    else:
        files = sorted(args.dir.rglob("*.py"))

    print(f"Processing {len(files)} files...")
    if args.dry_run:
        print("(dry run - no changes will be written)\n")
    else:
        print()

    total_changes = 0
    modified_files = 0
    errors = 0

    for path in files:
        changes, had_error = process_file(path, args.dry_run)
        if changes > 0:
            total_changes += changes
            modified_files += 1
        if had_error:
            errors += 1

    print("\nSummary:")
    print(f"  Files processed: {len(files)}")
    print(f"  Files modified: {modified_files}")
    print(f"  Total changes: {total_changes}")
    if errors:
        print(f"  Errors: {errors}")

    if args.dry_run and modified_files > 0:
        print("\nRun without --dry-run to apply changes.")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
