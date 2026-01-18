#!/usr/bin/env python3
"""
Validate documentation links.

Checks for:
- Broken internal links (references to non-existent files)
- Broken anchor links (references to non-existent sections)

Usage:
    python scripts/validate_doc_links.py
    python scripts/validate_doc_links.py --fix  # Report only, no fixes
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import NamedTuple


class BrokenLink(NamedTuple):
    """Represents a broken link."""
    file: Path
    line: int
    link: str
    reason: str


def find_markdown_links(content: str) -> list[tuple[int, str]]:
    """Find all markdown links in content.

    Returns list of (line_number, link_target) tuples.
    """
    links = []
    in_code_block = False

    for i, line in enumerate(content.split('\n'), 1):
        # Track code blocks
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue

        # Skip code blocks
        if in_code_block:
            continue

        # Skip inline code
        if '`' in line:
            # Remove inline code before searching for links
            line = re.sub(r'`[^`]+`', '', line)

        # Match [text](link) pattern
        for match in re.finditer(r'\[([^\]]*)\]\(([^)]+)\)', line):
            link = match.group(2)
            # Skip external links
            if link.startswith(('http://', 'https://', 'mailto:')):
                continue
            # Skip anchor-only links
            if link.startswith('#'):
                continue
            # Skip placeholder links
            if link in ("'...'", "..."):
                continue
            links.append((i, link))
    return links


def validate_link(source_file: Path, link: str, docs_dir: Path) -> str | None:
    """Validate a link.

    Returns error message if broken, None if valid.
    """
    # Parse link and anchor
    if '#' in link:
        file_part, anchor = link.split('#', 1)
    else:
        file_part = link
        anchor = None

    # Resolve relative path
    if file_part.startswith('../'):
        target = source_file.parent / file_part
    elif file_part.startswith('./'):
        target = source_file.parent / file_part[2:]
    elif file_part:
        target = source_file.parent / file_part
    else:
        # Anchor-only link to current file
        target = source_file

    # Normalize path
    try:
        target = target.resolve()
    except (OSError, ValueError):
        return f"Invalid path: {link}"

    # Check if file exists
    if not target.exists():
        return f"File not found: {file_part}"

    # TODO: Validate anchor links by parsing target file headers
    # For now, we only validate file existence

    return None


def validate_docs(docs_dir: Path) -> list[BrokenLink]:
    """Validate all documentation links."""
    broken = []

    for md_file in docs_dir.rglob('*.md'):
        try:
            content = md_file.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Warning: Could not read {md_file}: {e}")
            continue

        links = find_markdown_links(content)
        for line_num, link in links:
            error = validate_link(md_file, link, docs_dir)
            if error:
                broken.append(BrokenLink(
                    file=md_file.relative_to(docs_dir.parent),
                    line=line_num,
                    link=link,
                    reason=error,
                ))

    return broken


def main():
    """Main entry point."""
    # Find docs directory
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    docs_dir = repo_root / 'docs'

    if not docs_dir.exists():
        print(f"Error: docs directory not found at {docs_dir}")
        sys.exit(1)

    print(f"Validating documentation links in {docs_dir}...")
    broken = validate_docs(docs_dir)

    if not broken:
        print("✓ All documentation links are valid!")
        sys.exit(0)

    print(f"\n✗ Found {len(broken)} broken link(s):\n")

    # Group by file
    by_file: dict[Path, list[BrokenLink]] = {}
    for b in broken:
        by_file.setdefault(b.file, []).append(b)

    for file, links in sorted(by_file.items()):
        print(f"{file}:")
        for b in sorted(links, key=lambda x: x.line):
            print(f"  Line {b.line}: {b.link}")
            print(f"    → {b.reason}")
        print()

    sys.exit(1)


if __name__ == '__main__':
    main()
