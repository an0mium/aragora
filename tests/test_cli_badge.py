"""
Tests for CLI badge module.

Tests badge URL generation and markdown formatting.
"""

from __future__ import annotations

import argparse
from io import StringIO
import sys

import pytest

from aragora.cli.badge import (
    BADGE_STYLES,
    BADGE_TYPES,
    BADGE_URLS,
    generate_badge_markdown,
    get_badge_url,
    main,
)


class TestBadgeUrls:
    """Tests for badge URL constants."""

    def test_badge_types_exist(self):
        """All badge types are defined."""
        assert "reviewed" in BADGE_TYPES
        assert "consensus" in BADGE_TYPES
        assert "gauntlet" in BADGE_TYPES

    def test_badge_styles_exist(self):
        """All badge styles are defined."""
        assert "flat" in BADGE_STYLES
        assert "flat-square" in BADGE_STYLES
        assert "for-the-badge" in BADGE_STYLES
        assert "plastic" in BADGE_STYLES

    def test_all_badge_type_style_combinations_exist(self):
        """Every badge type has every style URL."""
        for badge_type in BADGE_TYPES:
            assert badge_type in BADGE_URLS
            for style in BADGE_STYLES:
                assert style in BADGE_URLS[badge_type]
                assert BADGE_URLS[badge_type][style].startswith("https://img.shields.io")


class TestGetBadgeUrl:
    """Tests for get_badge_url function."""

    def test_get_valid_badge_url(self):
        """Returns correct URL for valid type and style."""
        url = get_badge_url("reviewed", "flat")
        assert url == BADGE_URLS["reviewed"]["flat"]
        assert "shields.io" in url

    def test_get_badge_url_different_styles(self):
        """Returns different URLs for different styles."""
        flat_url = get_badge_url("consensus", "flat")
        square_url = get_badge_url("consensus", "flat-square")
        assert flat_url != square_url
        assert "flat" in flat_url or "style=flat" in flat_url

    def test_unknown_badge_type_raises_key_error(self):
        """Unknown badge type raises KeyError due to eager evaluation.

        Note: This tests current behavior. The function has a bug where the
        default argument for .get() is eagerly evaluated, causing KeyError
        even when the style is known.
        """
        with pytest.raises(KeyError):
            get_badge_url("unknown_type", "flat")

    def test_unknown_style_falls_back_to_flat(self):
        """Falls back to 'flat' style for unknown style."""
        url = get_badge_url("gauntlet", "unknown_style")
        assert url == BADGE_URLS["gauntlet"]["flat"]

    def test_all_badge_types_return_urls(self):
        """All badge types return valid URLs."""
        for badge_type in BADGE_TYPES:
            url = get_badge_url(badge_type, "flat")
            assert url.startswith("https://")
            assert "shields.io" in url


class TestGenerateBadgeMarkdown:
    """Tests for generate_badge_markdown function."""

    def test_generates_both_formats(self):
        """Returns both markdown and HTML."""
        markdown, html = generate_badge_markdown()
        assert markdown.startswith("[![")
        assert html.startswith("<a href=")

    def test_markdown_contains_badge_url(self):
        """Markdown contains the badge image URL."""
        markdown, _ = generate_badge_markdown("reviewed", "flat")
        badge_url = get_badge_url("reviewed", "flat")
        assert badge_url in markdown

    def test_html_contains_badge_url(self):
        """HTML contains the badge image URL."""
        _, html = generate_badge_markdown("reviewed", "flat")
        badge_url = get_badge_url("reviewed", "flat")
        assert badge_url in html

    def test_default_link_is_aragora_repo(self):
        """Default link points to Aragora repository."""
        markdown, html = generate_badge_markdown()
        assert "github.com/an0mium/aragora" in markdown
        assert "github.com/an0mium/aragora" in html

    def test_custom_repo_in_link(self):
        """Custom repo is included in the link."""
        repo = "owner/my-repo"
        markdown, html = generate_badge_markdown(repo=repo)
        assert f"github.com/{repo}" in markdown
        assert f"github.com/{repo}" in html

    def test_different_badge_types_produce_different_output(self):
        """Different badge types produce different output."""
        reviewed_md, _ = generate_badge_markdown("reviewed")
        consensus_md, _ = generate_badge_markdown("consensus")
        gauntlet_md, _ = generate_badge_markdown("gauntlet")

        assert reviewed_md != consensus_md
        assert consensus_md != gauntlet_md

    def test_different_styles_produce_different_output(self):
        """Different styles produce different output."""
        flat_md, _ = generate_badge_markdown(style="flat")
        square_md, _ = generate_badge_markdown(style="flat-square")
        badge_md, _ = generate_badge_markdown(style="for-the-badge")

        assert flat_md != square_md
        assert square_md != badge_md


class TestMain:
    """Tests for main CLI function."""

    def test_main_prints_output(self, capsys):
        """Main function prints badge information."""
        args = argparse.Namespace(style="flat", repo=None, type="reviewed")
        main(args)

        captured = capsys.readouterr()
        assert "Aragora Badge" in captured.out
        assert "README.md" in captured.out
        assert "markdown" in captured.out.lower()
        assert "html" in captured.out.lower()

    def test_main_with_custom_style(self, capsys):
        """Main function respects custom style."""
        args = argparse.Namespace(style="for-the-badge", repo=None, type="reviewed")
        main(args)

        captured = capsys.readouterr()
        assert "for-the-badge" in captured.out

    def test_main_with_custom_repo(self, capsys):
        """Main function includes custom repo."""
        args = argparse.Namespace(style="flat", repo="test/repo", type="reviewed")
        main(args)

        captured = capsys.readouterr()
        assert "test/repo" in captured.out

    def test_main_with_different_badge_types(self, capsys):
        """Main function works with all badge types."""
        for badge_type in BADGE_TYPES:
            args = argparse.Namespace(style="flat", repo=None, type=badge_type)
            main(args)

            captured = capsys.readouterr()
            assert "Badge" in captured.out


class TestBadgeUrlContent:
    """Tests for badge URL content validation."""

    def test_reviewed_badge_contains_aragora(self):
        """Reviewed badge URL contains Aragora."""
        url = get_badge_url("reviewed", "flat")
        assert "Aragora" in url or "aragora" in url.lower()

    def test_consensus_badge_contains_consensus(self):
        """Consensus badge URL contains 'Consensus'."""
        url = get_badge_url("consensus", "flat")
        assert "Consensus" in url

    def test_gauntlet_badge_contains_gauntlet(self):
        """Gauntlet badge URL contains 'Gauntlet'."""
        url = get_badge_url("gauntlet", "flat")
        assert "Gauntlet" in url

    def test_all_urls_are_https(self):
        """All badge URLs use HTTPS."""
        for badge_type in BADGE_TYPES:
            for style in BADGE_STYLES:
                url = BADGE_URLS[badge_type][style]
                assert url.startswith("https://")
