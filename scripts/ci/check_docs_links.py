#!/usr/bin/env python3
"""Internal markdown link + anchor checker (standard library only).

Docs onboarding guard (HEALTH-5 #8262): after the P2 docs restructure every
internal link must still resolve. This checker scans ``README.md`` and
``docs/**/*.md`` and fails, naming each offender, when:

* a relative link points at a file that does not exist, or
* an in-document anchor (``target.md#section`` / ``#section``) points at a
  heading that does not exist in the target markdown file.

External URLs (``http:``, ``mailto:`` ...), images, and links into
``docs/archive/`` are out of scope. The anchor matching mirrors the proven
``scripts/check_docs_consistency.py`` logic so the two checkers agree on a
clean tree.

Usage::

    python3 scripts/ci/check_docs_links.py            # exit 0 when links resolve
    python3 scripts/ci/check_docs_links.py            # exit 1 naming any offender
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

LINK_RE = re.compile(r"(?<!!)\[[^\]\n]+\]\(([^)\n]+)\)")
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
HTML_ID_RE = re.compile(r"""<a\s+[^>]*id=["']([^"']+)["']""", re.IGNORECASE)


@dataclass(frozen=True)
class LinkIssue:
    location: str
    message: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def iter_markdown_files(root: Path) -> list[Path]:
    """README.md plus every tracked markdown file under docs/.

    Falls back to a filesystem walk when ``git ls-files`` is unavailable (e.g.
    a temporary fixture tree that is not a git checkout).
    """
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", "README.md", "docs"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout:
        tracked = [root / p for p in result.stdout.split("\0") if p.endswith(".md")]
        if tracked:
            return tracked

    files: list[Path] = []
    readme = root / "README.md"
    if readme.is_file():
        files.append(readme)
    docs = root / "docs"
    if docs.is_dir():
        files.extend(sorted(docs.rglob("*.md")))
    return files


def is_url(target: str) -> bool:
    return bool(re.match(r"^[a-z][a-z0-9+.-]*:", target, re.IGNORECASE)) or target.startswith("//")


def strip_link_title(raw: str) -> str:
    target = raw.strip()
    if target.startswith("<") and ">" in target:
        return target[1 : target.index(">")]
    return target.split()[0] if target.split() else target


def split_target(target: str) -> tuple[str, str]:
    no_title = strip_link_title(target)
    path_part, sep, anchor = no_title.partition("#")
    path_part = unquote(path_part.split("?", 1)[0])
    return path_part, unquote(anchor) if sep else ""


def resolve_link(root: Path, source: Path, path_part: str) -> Path:
    if not path_part:
        return source
    if path_part.startswith("/"):
        return root / path_part.lstrip("/")
    return (source.parent / path_part).resolve()


def normalize_anchor(value: str) -> str:
    value = unquote(value).lower()
    value = re.sub(r"`([^`]+)`", r"\1", value)
    value = re.sub(r"\[[^\]]+\]\(([^)]+)\)", r"\1", value)
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def github_slug(value: str) -> str:
    return normalize_anchor(value).replace(" ", "-")


def headings_for(path: Path) -> set[str]:
    anchors: set[str] = set()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return anchors
    for line in text.splitlines():
        heading = HEADING_RE.match(line)
        titles = ([heading.group(1).strip()] if heading else []) + HTML_ID_RE.findall(line)
        for title in titles:
            anchors.add(normalize_anchor(title))
            anchors.add(github_slug(title))
            anchors.add(normalize_anchor(title).replace(" ", ""))
    return anchors


def anchor_exists(path: Path, anchor: str) -> bool:
    wanted = normalize_anchor(anchor)
    compact = wanted.replace(" ", "")
    for existing in headings_for(path):
        normalized = normalize_anchor(existing)
        if normalized == wanted or existing == github_slug(anchor):
            return True
        if wanted and normalized.startswith(f"{wanted} "):
            return True
        if compact and normalized.replace(" ", "").startswith(compact):
            return True
    return False


def extract_links(path: Path) -> list[tuple[int, str, str]]:
    """Return (line_number, raw_target, raw_link_markdown) outside code spans."""
    links: list[tuple[int, str, str]] = []
    in_fence = False
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fence = line.lstrip().startswith("```")
        if fence:
            in_fence = not in_fence
        if fence or in_fence:
            continue
        searchable = INLINE_CODE_RE.sub("", line)
        for match in LINK_RE.finditer(searchable):
            links.append((line_no, match.group(1).strip(), match.group(0)))
    return links


def find_broken_links(root: Path, files: list[Path] | None = None) -> list[LinkIssue]:
    root = root.resolve()
    docs_archive = root / "docs" / "archive"
    sources = files if files is not None else iter_markdown_files(root)
    issues: list[LinkIssue] = []
    for source in sources:
        try:
            source.relative_to(docs_archive)
            continue
        except ValueError:
            pass
        rel_source = _rel(source, root)
        for line_no, target, link in extract_links(source):
            if is_url(strip_link_title(target)):
                continue
            path_part, anchor = split_target(target)
            target_path = resolve_link(root, source, path_part)
            try:
                target_path.relative_to(docs_archive)
                continue
            except ValueError:
                pass
            location = f"{rel_source}:{line_no}"
            if not target_path.exists():
                issues.append(LinkIssue(location, f"missing link target {link} -> {target}"))
                continue
            if (
                anchor
                and target_path.suffix.lower() == ".md"
                and not anchor_exists(target_path, anchor)
            ):
                issues.append(LinkIssue(location, f"missing anchor {link} -> {target}"))
    return issues


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repo_root(), help="Repository root")
    args = parser.parse_args(argv)

    issues = find_broken_links(args.root.resolve())
    if issues:
        print(f"check_docs_links: FAIL {len(issues)} broken internal link(s)/anchor(s):")
        for issue in issues:
            print(f"  {issue.location}: {issue.message}")
        print(
            "Fix the link target, update the anchor, or move the file back; "
            "external URLs and docs/archive/ are out of scope."
        )
        return 1

    print("check_docs_links: OK all internal markdown links and anchors resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
