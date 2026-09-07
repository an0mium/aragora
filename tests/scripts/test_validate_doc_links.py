from __future__ import annotations

import textwrap
from pathlib import Path

from scripts.validate_doc_links import find_markdown_links, github_slug, validate_link


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


def test_validate_link_rejects_prefix_only_markdown_anchor(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    source = docs / "source.md"
    target = docs / "target.md"
    _write(source, "[bad](target.md#canon)")
    _write(target, "## Canonical Metrics")

    assert validate_link(source, "target.md#canon", docs) == "Anchor not found: #canon"


def test_validate_link_uses_github_apostrophe_slug(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    source = docs / "source.md"
    target = docs / "target.md"
    _write(source, "[ok](target.md#whats-partially-working)")
    _write(target, "## What's Partially Working")

    assert github_slug("What's Partially Working") == "whats-partially-working"
    assert validate_link(source, "target.md#whats-partially-working", docs) is None
    assert validate_link(source, "target.md#what-s-partially-working", docs) == (
        "Anchor not found: #what-s-partially-working"
    )


def test_github_slug_strips_punctuation_without_word_breaks() -> None:
    assert github_slug("OAuth 2.0") == "oauth-20"
    assert github_slug("N+1 Query Detection") == "n1-query-detection"
    assert github_slug("5.2 Version Drift") == "52-version-drift"
    assert github_slug("Matrix/Graph Debates") == "matrixgraph-debates"
    assert (
        github_slug("DRIFT-007: `ARAGORA_SINGLE_INSTANCE=true` undocumented")
        == "drift-007-aragora_single_instancetrue-undocumented"
    )
    assert (
        github_slug("Active direction — Open Decision Receipt")
        == "active-direction--open-decision-receipt"
    )


def test_github_slug_uses_visible_markdown_link_text() -> None:
    assert (
        github_slug("ODR GA ([#8223](https://github.com/synaptent/aragora/pull/8223))")
        == "odr-ga-8223"
    )


def test_validate_link_rejects_loose_normalized_markdown_anchor(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    source = docs / "source.md"
    target = docs / "target.md"
    _write(source, "[bad](target.md#active-direction-open-decision-receipt)")
    _write(target, "## Active direction — Open Decision Receipt")

    assert (
        validate_link(
            source,
            "target.md#active-direction-open-decision-receipt",
            docs,
        )
        == "Anchor not found: #active-direction-open-decision-receipt"
    )
    assert (
        validate_link(
            source,
            "target.md#active-direction--open-decision-receipt",
            docs,
        )
        is None
    )


def test_validate_link_accepts_explicit_html_id_anchor(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    source = docs / "source.md"
    target = docs / "target.md"
    _write(source, "[ok](target.md#manual-anchor)")
    _write(target, '<a id="manual-anchor"></a>\n\n## Different Heading')

    assert validate_link(source, "target.md#manual-anchor", docs) is None


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


def test_validate_link_reports_missing_anchor_only_links(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    source = docs / "source.md"
    _write(source, "# Local Section\n\n[bad](#missing-section)")

    assert validate_link(source, "#missing-section", docs) == ("Anchor not found: #missing-section")


def test_validate_link_rejects_absolute_path_escape(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    source = docs / "source.md"
    outside = tmp_path.parent / "outside.md"
    _write(source, "[bad](/../outside.md)")
    _write(outside, "# Outside")

    try:
        assert validate_link(source, "/../outside.md", docs) == (
            "Path escapes repository root: /../outside.md"
        )
    finally:
        outside.unlink(missing_ok=True)


def test_find_markdown_links_keeps_anchor_only_links() -> None:
    links = find_markdown_links(
        """
        [same file](#local-section)
        Inline code `[ignored](#not-real)` should be skipped.
        """
    )

    assert links == [(2, "#local-section")]
