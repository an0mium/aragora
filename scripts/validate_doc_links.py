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
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote


HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
HTML_ID_RE = re.compile(r"""\bid=["']([^"']+)["']""")
GITHUB_PUNCTUATION_RE = re.compile(r"[`~!@#$%^&*()=+\[\]{}\\|;:'\",.<>/?—–]")


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

    for i, line in enumerate(content.split("\n"), 1):
        # Track code blocks
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue

        # Skip code blocks
        if in_code_block:
            continue

        # Skip inline code
        if "`" in line:
            # Remove inline code before searching for links
            line = re.sub(r"`[^`]+`", "", line)

        # Match [text](link) pattern
        for match in re.finditer(r"\[([^\]]*)\]\(([^)]+)\)", line):
            link = match.group(2)
            # Skip external links
            if link.startswith(("http://", "https://", "mailto:")):
                continue
            # Skip placeholder links
            if link in ("'...'", "..."):
                continue
            links.append((i, link))
    return links


def strip_link_title(raw: str) -> str:
    """Return the link target without an optional markdown title."""
    target = raw.strip()
    if target.startswith("<") and ">" in target:
        return target[1 : target.index(">")]
    return target.split()[0] if target.split() else target


def split_link_target(raw: str) -> tuple[str, str]:
    """Split a markdown link target into path and decoded anchor parts."""
    target = strip_link_title(raw)
    path_part, sep, anchor = target.partition("#")
    path_part = unquote(path_part.split("?", 1)[0])
    return path_part, unquote(anchor) if sep else ""


def _anchor_text(value: str) -> str:
    """Return markdown-stripped text before GitHub-style slug normalization."""
    value = unquote(value).lower()
    value = re.sub(r"`([^`]+)`", r"\1", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    return re.sub(r"<[^>]+>", "", value)


def normalize_anchor(value: str) -> str:
    """Normalize a heading or explicit id for loose internal comparisons."""
    value = GITHUB_PUNCTUATION_RE.sub("", _anchor_text(value))
    value = re.sub(r"[-\s]+", " ", value)
    return " ".join(value.split())


def github_slug(value: str) -> str:
    """Return the canonical GitHub-style slug for a heading."""
    value = GITHUB_PUNCTUATION_RE.sub("", _anchor_text(value))
    return re.sub(r"\s", "-", value.strip())


def heading_anchors(path: Path) -> set[str]:
    """Collect markdown heading and explicit HTML id anchors from a file."""
    try:
        resolved = path.resolve()
        stat = resolved.stat()
    except OSError:
        return set()
    return set(_heading_anchors_cached(resolved, stat.st_mtime_ns, stat.st_size))


@lru_cache(maxsize=512)
def _heading_anchors_cached(
    path: Path,
    mtime_ns: int,  # noqa: ARG001 - cache key invalidates stale reads
    size: int,  # noqa: ARG001 - cache key invalidates stale reads
) -> frozenset[str]:
    """Collect exact anchors for a file, cached by path and file stat."""
    anchors: set[str] = set()
    seen_slugs: dict[str, int] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return frozenset()

    for line in text.splitlines():
        heading = HEADING_RE.match(line)
        if heading:
            slug = github_slug(heading.group(1).strip())
            if slug:
                duplicate_index = seen_slugs.get(slug, 0)
                seen_slugs[slug] = duplicate_index + 1
                anchors.add(slug if duplicate_index == 0 else f"{slug}-{duplicate_index}")
        for html_id in HTML_ID_RE.findall(line):
            explicit_id = unquote(html_id).strip()
            if explicit_id:
                anchors.add(explicit_id)
    return frozenset(anchors)


def anchor_exists(path: Path, anchor: str) -> bool:
    """Return whether a markdown file contains the requested heading anchor."""
    wanted_slug = github_slug(anchor)
    wanted_explicit_id = unquote(anchor).strip()
    anchors = heading_anchors(path)
    return wanted_slug in anchors or wanted_explicit_id in anchors


def validate_link(source_file: Path, link: str, docs_dir: Path) -> str | None:
    """Validate a link.

    Returns error message if broken, None if valid.
    """
    # Parse link and anchor
    file_part, anchor = split_link_target(link)

    # Resolve relative path
    if file_part.startswith("/"):
        target = docs_dir.parent / file_part.lstrip("/")
    elif file_part.startswith("../"):
        target = source_file.parent / file_part
    elif file_part.startswith("./"):
        target = source_file.parent / file_part[2:]
    elif file_part:
        target = source_file.parent / file_part
    else:
        # Anchor-only link to current file
        target = source_file

    # Normalize path
    try:
        repo_root = docs_dir.parent.resolve()
        target = target.resolve()
    except (OSError, ValueError):
        return f"Invalid path: {link}"

    if not target.is_relative_to(repo_root):
        return f"Path escapes repository root: {file_part}"

    # Check if file exists
    if not target.exists():
        return f"File not found: {file_part}"

    if anchor and target.suffix.lower() == ".md" and not anchor_exists(target, anchor):
        return f"Anchor not found: #{anchor}"

    return None


def validate_docs(docs_dir: Path) -> list[BrokenLink]:
    """Validate all documentation links."""
    broken = []

    for md_file in docs_dir.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Warning: Could not read {md_file}: {e}")
            continue

        links = find_markdown_links(content)
        for line_num, link in links:
            error = validate_link(md_file, link, docs_dir)
            if error:
                broken.append(
                    BrokenLink(
                        file=md_file.relative_to(docs_dir.parent),
                        line=line_num,
                        link=link,
                        reason=error,
                    )
                )

    return broken


def main():
    """Main entry point."""
    # Find docs directory
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    docs_dir = repo_root / "docs"

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


if __name__ == "__main__":
    main()
