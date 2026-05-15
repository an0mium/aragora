from __future__ import annotations

import textwrap
from pathlib import Path

from scripts.validate_doc_links import find_markdown_links, validate_link


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def test_validate_link_reports_missing_markdown_anchor(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    source = docs / "source.md"
    target = docs / "target.md"
    _write(source, "[bad](target.md#missing-heading)")
    _write(target, "# Present Heading")

    assert validate_link(source, "target.md#missing-heading", docs) == (
        "Anchor not found: #missing-heading"
    )


def test_validate_link_accepts_github_heading_slug(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    source = docs / "source.md"
    target = docs / "target.md"
    _write(source, "[ok](target.md#part-1-stable-heading)")
    _write(target, "## Part 1: Stable Heading")

    assert validate_link(source, "target.md#part-1-stable-heading", docs) is None


def test_validate_link_accepts_anchor_only_links(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    source = docs / "source.md"
    _write(
        source,
        """
        # Local Section

        [ok](#local-section)
        """,
    )

    assert validate_link(source, "#local-section", docs) is None


def test_find_markdown_links_keeps_anchor_only_links() -> None:
    links = find_markdown_links(
        """
        [same file](#local-section)
        Inline code `[ignored](#not-real)` should be skipped.
        """
    )

    assert links == [(2, "#local-section")]
