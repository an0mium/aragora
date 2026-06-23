"""Unit tests for scripts/ci/check_docs_links.py.

Exercises the pure link/anchor logic and the ``main`` CLI wiring against fake
checkouts built under ``tmp_path`` (not a git repo, so the checker's filesystem
fallback discovers the markdown files). The end-to-end behavior on real
origin/main (green on a clean tree, a README tamper trips it, restore greens
again) is covered by the VAL-P2-011 acceptance check.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKER_PATH = REPO_ROOT / "scripts" / "ci" / "check_docs_links.py"

_spec = importlib.util.spec_from_file_location("check_docs_links", _CHECKER_PATH)
cdl = importlib.util.module_from_spec(_spec)
# Register before exec so the frozen-annotation dataclass can resolve its module.
sys.modules["check_docs_links"] = cdl
_spec.loader.exec_module(cdl)


def _write(root: Path, rel: str, body: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


# --- pure logic -------------------------------------------------------------


def test_is_url_distinguishes_schemes():
    assert cdl.is_url("https://example.com")
    assert cdl.is_url("mailto:x@example.com")
    assert cdl.is_url("//cdn.example.com/x")
    assert not cdl.is_url("docs/guide.md")
    assert not cdl.is_url("#section")


def test_split_target_separates_path_and_anchor():
    assert cdl.split_target("guide.md#intro") == ("guide.md", "intro")
    assert cdl.split_target("guide.md") == ("guide.md", "")
    assert cdl.split_target("#intro") == ("", "intro")
    assert cdl.split_target('guide.md "Title"') == ("guide.md", "")
    assert cdl.split_target("<a path.md>") == ("a path.md", "")


def test_github_slug_matches_github_heading_rules():
    assert cdl.github_slug("Canonical Metrics (March 2026 Baseline)") == (
        "canonical-metrics-march-2026-baseline"
    )
    assert cdl.github_slug("`code` & symbols!") == "code-symbols"


def test_anchor_exists_matches_heading(tmp_path):
    target = _write(tmp_path, "doc.md", "# Title\n\n## Canonical Metrics\n\nbody\n")
    assert cdl.anchor_exists(target, "canonical-metrics")
    assert cdl.anchor_exists(target, "Canonical Metrics")
    assert not cdl.anchor_exists(target, "nonexistent-heading")


# --- find_broken_links ------------------------------------------------------


def test_clean_tree_has_no_issues(tmp_path):
    _write(tmp_path, "docs/guide.md", "# Guide\n\n## Setup\n\ndetails\n")
    readme = _write(
        tmp_path,
        "README.md",
        "# Project\n\nSee [the guide](docs/guide.md#setup) and [guide](docs/guide.md).\n",
    )
    assert cdl.find_broken_links(tmp_path, files=[readme]) == []


def test_missing_file_target_is_reported(tmp_path):
    readme = _write(tmp_path, "README.md", "# P\n\nbroken [x](docs/does_not_exist.md)\n")
    issues = cdl.find_broken_links(tmp_path, files=[readme])
    assert len(issues) == 1
    assert "docs/does_not_exist.md" in issues[0].message
    assert issues[0].location == "README.md:3"


def test_missing_anchor_is_reported(tmp_path):
    _write(tmp_path, "docs/guide.md", "# Guide\n\n## Setup\n")
    readme = _write(tmp_path, "README.md", "# P\n\n[bad](docs/guide.md#teardown)\n")
    issues = cdl.find_broken_links(tmp_path, files=[readme])
    assert len(issues) == 1
    assert "missing anchor" in issues[0].message


def test_external_urls_and_images_are_ignored(tmp_path):
    readme = _write(
        tmp_path,
        "README.md",
        "# P\n\n[site](https://example.com/missing)\n\n![logo](docs/missing.png)\n",
    )
    assert cdl.find_broken_links(tmp_path, files=[readme]) == []


def test_fenced_code_links_are_ignored(tmp_path):
    readme = _write(
        tmp_path,
        "README.md",
        "# P\n\n```\n[x](docs/missing.md)\n```\n",
    )
    assert cdl.find_broken_links(tmp_path, files=[readme]) == []


def test_archive_source_and_target_are_excluded(tmp_path):
    archived = _write(tmp_path, "docs/archive/old.md", "# Old\n\n[gone](missing.md)\n")
    readme = _write(tmp_path, "README.md", "# P\n\n[hist](docs/archive/whatever.md)\n")
    assert cdl.find_broken_links(tmp_path, files=[archived, readme]) == []


# --- main CLI wiring --------------------------------------------------------


def test_main_green_on_clean_tree(tmp_path, capsys):
    _write(tmp_path, "docs/guide.md", "# Guide\n\n## Setup\n")
    _write(tmp_path, "README.md", "# P\n\n[guide](docs/guide.md#setup)\n")
    assert cdl.main(["--root", str(tmp_path)]) == 0
    assert "OK" in capsys.readouterr().out


def test_main_flags_tamper_and_names_offender(tmp_path, capsys):
    _write(tmp_path, "docs/guide.md", "# Guide\n\n## Setup\n")
    _write(
        tmp_path,
        "README.md",
        "# P\n\n[guide](docs/guide.md#setup)\n\nbroken [x](docs/__val_no_such_file__.md)\n",
    )
    assert cdl.main(["--root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "docs/__val_no_such_file__.md" in out
    assert "README.md" in out
