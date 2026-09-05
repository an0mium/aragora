"""Tests for scripts/check_version_alignment.py doc-pattern helpers."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_version_alignment.py"

LAST_UPDATED = (
    r"^(> \*\*Last Updated:\*\* )(?P<date>\d{4}-\d{2}-\d{2})( \(v)(?P<version>\d+\.\d+\.\d+)"
    r"( alignment with repo versions\))$"
)
IMAGE_PIN = r"(ghcr\.io/synaptent/aragora/backend:)(\d+\.\d+\.\d+)(\b)"
VERSION_FILE = (
    'VERSION_MAJOR = 2\nVERSION_MINOR = 10\nVERSION_PATCH = 0\nRELEASE_DATE = "2026-09-04"\n'
)
SUPPORT_MATRIX = (
    "| Version | Release | End of Support | Status |\n"
    "|---------|---------|----------------|--------|\n"
    "| **v2.10.x** | 2026-09-04 | Active | **Current** |\n"
    "| v2.9.x | 2026-04-25 | Active | Supported |\n"
)


@pytest.fixture(scope="module")
def cva():
    spec = importlib.util.spec_from_file_location("check_version_alignment", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pattern(cva, name: str) -> str:
    """The real tracked pattern behind one DOC_SOURCES entry."""
    return {entry: pattern for entry, _, pattern in cva.DOC_SOURCES}[name]


def _fake_repo(cva, tmp_path: Path, monkeypatch, entries, *argv: str) -> None:
    """Point main() at a temp tree whose only tracked doc spots are ``entries``."""
    (tmp_path / "aragora").mkdir()
    (tmp_path / "aragora" / "__version__.py").write_text(VERSION_FILE)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cva, "DOC_SOURCES", entries)
    monkeypatch.setattr(sys, "argv", ["check_version_alignment.py", *argv])


def test_get_doc_versions_returns_every_occurrence(cva, tmp_path: Path) -> None:
    doc = tmp_path / "d.md"
    doc.write_text(
        "image: ghcr.io/synaptent/aragora/backend:2.10.0\n"
        "IMG=ghcr.io/synaptent/aragora/backend:2.9.0 docker compose up\n"
    )
    assert cva.get_doc_versions(doc, IMAGE_PIN) == ["2.10.0", "2.9.0"]


def test_fix_doc_version_rewrites_every_occurrence(cva, tmp_path: Path) -> None:
    doc = tmp_path / "d.md"
    doc.write_text(
        "image: ghcr.io/synaptent/aragora/backend:2.9.0\n"
        "IMG=ghcr.io/synaptent/aragora/backend:2.9.0 docker compose up\n"
    )
    assert cva.fix_doc_version(doc, IMAGE_PIN, "2.10.0") is True
    assert cva.get_doc_versions(doc, IMAGE_PIN) == ["2.10.0", "2.10.0"]
    assert cva.fix_doc_version(doc, IMAGE_PIN, "2.10.0") is False


def test_named_version_group_is_read(cva, tmp_path: Path) -> None:
    doc = tmp_path / "d.md"
    doc.write_text("> **Last Updated:** 2026-04-25 (v2.9.0 alignment with repo versions)\n")
    assert cva.get_doc_versions(doc, LAST_UPDATED) == ["2.9.0"]


def test_fix_doc_version_refreshes_date_group_with_version(cva, tmp_path: Path) -> None:
    doc = tmp_path / "d.md"
    doc.write_text("> **Last Updated:** 2026-04-25 (v2.9.0 alignment with repo versions)\n")
    assert cva.fix_doc_version(doc, LAST_UPDATED, "2.10.0", "2026-09-04") is True
    assert doc.read_text() == (
        "> **Last Updated:** 2026-09-04 (v2.10.0 alignment with repo versions)\n"
    )


def test_fix_doc_version_leaves_date_when_no_release_date(cva, tmp_path: Path) -> None:
    doc = tmp_path / "d.md"
    doc.write_text("> **Last Updated:** 2026-04-25 (v2.9.0 alignment with repo versions)\n")
    assert cva.fix_doc_version(doc, LAST_UPDATED, "2.10.0") is True
    assert doc.read_text() == (
        "> **Last Updated:** 2026-04-25 (v2.10.0 alignment with repo versions)\n"
    )


def test_canonical_release_date_matches_version_file(cva, monkeypatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    date = cva.get_canonical_release_date()
    assert date is not None
    text = (REPO_ROOT / "aragora" / "__version__.py").read_text()
    assert f'RELEASE_DATE = "{date}"' in text


def test_fix_reports_failure_when_nothing_could_be_rewritten(
    cva, monkeypatch, capsys, tmp_path: Path
) -> None:
    _fake_repo(cva, tmp_path, monkeypatch, [], "--fix")
    (tmp_path / "pyproject.toml").write_text('version = "2.9.0"\n')
    # Make the pyproject rewrite a no-op so a mismatch survives --fix.
    monkeypatch.setattr(cva, "fix_pyproject_version", lambda *_: False)
    assert cva.main() == 1
    out = capsys.readouterr().out
    assert "could not be fixed" in out
    assert "All versions aligned!" not in out


def test_check_and_fix_are_mutually_exclusive(cva, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["check_version_alignment.py", "--check", "--fix"])
    with pytest.raises(SystemExit) as excinfo:
        cva.main()
    assert excinfo.value.code == 2


def test_every_date_bearing_pattern_names_its_date_group(cva) -> None:
    # A date swallowed by an unnamed group is rewritten by nobody and checked by nobody.
    for name, _, pattern in cva.DOC_SOURCES:
        if r"\d{4}-\d{2}-\d{2}" in pattern:
            assert "(?P<date>" in pattern or "(?P<since>" in pattern, name


def test_upgrade_roadmap_current_line_refreshes_version_and_date(cva, tmp_path: Path) -> None:
    pattern = _pattern(cva, "docs/deployment/UPGRADE_ROADMAP.md")
    doc = tmp_path / "d.md"
    doc.write_text("**Aragora v2.9.0** (released 2026-04-25)\n")
    assert cva.get_doc_versions(doc, pattern) == ["2.9.0"]
    assert cva.get_doc_dates(doc, pattern) == ["2026-04-25"]
    assert cva.fix_doc_version(doc, pattern, "2.10.0", "2026-09-04") is True
    assert doc.read_text() == "**Aragora v2.10.0** (released 2026-09-04)\n"


def test_stale_date_next_to_current_version_fails_check(
    cva, monkeypatch, capsys, tmp_path: Path
) -> None:
    name = "docs/deployment/UPGRADE_ROADMAP.md"
    doc = Path("docs/deployment/UPGRADE_ROADMAP.md")
    _fake_repo(cva, tmp_path, monkeypatch, [(name, doc, _pattern(cva, name))], "--check")
    (tmp_path / doc).parent.mkdir(parents=True)
    (tmp_path / doc).write_text("**Aragora v2.10.0** (released 2026-04-25)\n")
    assert cva.main() == 1
    out = capsys.readouterr().out
    assert f"{name}: 2.10.0 [doc] [MISMATCH] (date 2026-04-25, release date is 2026-09-04)" in out


def test_stale_date_is_rewritten_by_fix(cva, monkeypatch, capsys, tmp_path: Path) -> None:
    name = "docs/deployment/UPGRADE_ROADMAP.md"
    doc = Path("docs/deployment/UPGRADE_ROADMAP.md")
    _fake_repo(cva, tmp_path, monkeypatch, [(name, doc, _pattern(cva, name))], "--fix")
    (tmp_path / doc).parent.mkdir(parents=True)
    (tmp_path / doc).write_text("**Aragora v2.10.0** (released 2026-04-25)\n")
    assert cva.main() == 0
    assert "All versions aligned!" in capsys.readouterr().out
    assert (tmp_path / doc).read_text() == "**Aragora v2.10.0** (released 2026-09-04)\n"


def test_status_release_line_tracks_version_and_date(cva, tmp_path: Path) -> None:
    pattern = _pattern(cva, "docs/STATUS.md")
    assert pattern == _pattern(cva, "docs/status/STATUS.md")
    assert pattern == _pattern(cva, "docs-site/docs/contributing/status.md")
    doc = tmp_path / "d.md"
    doc.write_text("Current released version is **v2.9.0** (released 2026-04-25).\n")
    assert cva.get_doc_versions(doc, pattern) == ["2.9.0"]
    assert cva.get_doc_dates(doc, pattern) == ["2026-04-25"]
    assert cva.fix_doc_version(doc, pattern, "2.10.0", "2026-09-04") is True
    assert doc.read_text() == "Current released version is **v2.10.0** (released 2026-09-04).\n"


def test_zero_match_tracked_pattern_is_a_mismatch(cva, monkeypatch, capsys, tmp_path: Path) -> None:
    name = "docs/deployment/UPGRADE_ROADMAP.md"
    doc = Path("docs/deployment/UPGRADE_ROADMAP.md")
    _fake_repo(cva, tmp_path, monkeypatch, [(name, doc, _pattern(cva, name))], "--check")
    (tmp_path / doc).parent.mkdir(parents=True)
    (tmp_path / doc).write_text("The version line used to live here.\n")
    assert cva.main() == 1
    out = capsys.readouterr().out
    assert f"{name}: (version not found) [doc] [MISMATCH]" in out
    assert "All versions aligned!" not in out


def test_zero_match_tracked_pattern_survives_fix(cva, monkeypatch, capsys, tmp_path: Path) -> None:
    name = "docs/deployment/UPGRADE_ROADMAP.md"
    doc = Path("docs/deployment/UPGRADE_ROADMAP.md")
    _fake_repo(cva, tmp_path, monkeypatch, [(name, doc, _pattern(cva, name))], "--fix")
    (tmp_path / doc).parent.mkdir(parents=True)
    (tmp_path / doc).write_text("The version line used to live here.\n")
    assert cva.main() == 1
    assert "could not be fixed" in capsys.readouterr().out


def test_support_matrix_series_is_compared_to_major_minor(cva, tmp_path: Path) -> None:
    pattern = _pattern(cva, "docs/deployment/UPGRADE_ROADMAP.md (support matrix)")
    doc = tmp_path / "d.md"
    doc.write_text(SUPPORT_MATRIX)
    assert cva.get_doc_versions(doc, pattern) == ["2.10"]
    # The row's own date is history, not RELEASE_DATE: a patch release leaves it alone.
    assert cva.get_doc_dates(doc, pattern) == []
    assert cva._expected_version(pattern, "2.10.3") == "2.10"
    assert cva._expected_version(pattern, "2.11.0") == "2.11"


def test_support_matrix_fix_inserts_current_row_and_demotes_previous(cva, tmp_path: Path) -> None:
    pattern = _pattern(cva, "docs/deployment/UPGRADE_ROADMAP.md (support matrix)")
    doc = tmp_path / "d.md"
    doc.write_text(SUPPORT_MATRIX)
    # Without a release date there is nothing truthful to write.
    assert cva.fix_support_matrix_row(doc, pattern, "2.11") is False
    assert doc.read_text() == SUPPORT_MATRIX
    assert cva.fix_support_matrix_row(doc, pattern, "2.11", "2026-12-01") is True
    assert doc.read_text() == (
        "| Version | Release | End of Support | Status |\n"
        "|---------|---------|----------------|--------|\n"
        "| **v2.11.x** | 2026-12-01 | Active | **Current** |\n"
        "| v2.10.x | 2026-09-04 | Active | Supported |\n"
        "| v2.9.x | 2026-04-25 | Active | Supported |\n"
    )
    assert cva.get_doc_versions(doc, pattern) == ["2.11"]
    assert cva.fix_support_matrix_row(doc, pattern, "2.11", "2026-12-01") is False


def test_main_fix_moves_the_support_matrix_current_row(
    cva, monkeypatch, capsys, tmp_path: Path
) -> None:
    name = "docs/deployment/UPGRADE_ROADMAP.md (support matrix)"
    doc = Path("docs/deployment/UPGRADE_ROADMAP.md")
    _fake_repo(cva, tmp_path, monkeypatch, [(name, doc, _pattern(cva, name))], "--fix")
    (tmp_path / doc).parent.mkdir(parents=True)
    (tmp_path / doc).write_text(
        "| **v2.9.x** | 2026-04-25 | Active | **Current** |\n"
        "| v2.8.x | 2026-02-25 | Active | Supported |\n"
    )
    assert cva.main() == 0
    assert f"{name}: 2.9 [doc] [MISMATCH]" in capsys.readouterr().out
    assert (tmp_path / doc).read_text() == (
        "| **v2.10.x** | 2026-09-04 | Active | **Current** |\n"
        "| v2.9.x | 2026-04-25 | Active | Supported |\n"
        "| v2.8.x | 2026-02-25 | Active | Supported |\n"
    )


def test_lockfile_patterns_touch_only_the_two_root_spots(cva, tmp_path: Path) -> None:
    lock = tmp_path / "package-lock.json"
    lock.write_text(
        json.dumps(
            {
                "name": "aragora-live",
                "version": "2.9.0",
                "lockfileVersion": 3,
                "requires": True,
                "packages": {
                    "": {"name": "aragora-live", "version": "2.9.0"},
                    "node_modules/left-pad": {"name": "left-pad", "version": "2.9.0"},
                },
            },
            indent=2,
        )
        + "\n"
    )
    root = _pattern(cva, "aragora/live/package-lock.json (root)")
    packages_root = _pattern(cva, "aragora/live/package-lock.json (packages root)")
    assert cva.get_doc_versions(lock, root) == ["2.9.0"]
    assert cva.get_doc_versions(lock, packages_root) == ["2.9.0"]
    assert cva.fix_doc_version(lock, root, "2.10.0") is True
    assert cva.fix_doc_version(lock, packages_root, "2.10.0") is True
    data = json.loads(lock.read_text())
    assert data["version"] == "2.10.0"
    assert data["packages"][""]["version"] == "2.10.0"
    assert data["packages"]["node_modules/left-pad"]["version"] == "2.9.0"


def test_uv_lock_pattern_matches_only_the_aragora_package(cva, tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text(
        '[[package]]\nname = "aragora"\nversion = "2.9.0"\nsource = { editable = "." }\n\n'
        '[[package]]\nname = "aragora-verify"\nversion = "2.9.0"\n'
    )
    pattern = _pattern(cva, "uv.lock (aragora)")
    assert cva.get_doc_versions(lock, pattern) == ["2.9.0"]
    assert cva.fix_doc_version(lock, pattern, "2.10.0") is True
    assert lock.read_text() == (
        '[[package]]\nname = "aragora"\nversion = "2.10.0"\nsource = { editable = "." }\n\n'
        '[[package]]\nname = "aragora-verify"\nversion = "2.9.0"\n'
    )


def test_readme_and_catalog_patterns_track_the_hand_aligned_spots(cva, tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "> (46 files) · 360+ RBAC permissions · Python + TypeScript SDKs · v2.9.0.**\n"
    )
    pattern = _pattern(cva, "README.md (metrics block)")
    assert cva.get_doc_versions(readme, pattern) == ["2.9.0"]
    assert cva.fix_doc_version(readme, pattern, "2.10.0") is True
    assert readme.read_text().endswith("Python + TypeScript SDKs · v2.10.0.**\n")

    catalog = tmp_path / "catalog.toml"
    catalog.write_text(
        '  { key = "other", display = "exact", display_value = "1.2.3" },\n'
        '  { key = "project_version", label = "Project version", display = "exact", '
        'display_value = "2.9.0", comparison = "exact_claim" },\n'
    )
    pattern = _pattern(cva, "docs/status/metrics/catalog.toml (project_version)")
    assert cva.get_doc_versions(catalog, pattern) == ["2.9.0"]
    assert cva.fix_doc_version(catalog, pattern, "2.10.0") is True
    assert 'display_value = "1.2.3"' in catalog.read_text()
    assert 'display_value = "2.10.0"' in catalog.read_text()


# Runs the full checker against the live repo, duplicating the CI gate (sdk-parity.yml, test.yml); opt in so unrelated doc edits cannot fail the unit lane.
@pytest.mark.skipif(
    os.environ.get("ARAGORA_RUN_REPO_VERSION_ALIGNMENT_TEST") != "1",
    reason="set ARAGORA_RUN_REPO_VERSION_ALIGNMENT_TEST=1 to run the checker against the live repo",
)
def test_repo_docs_are_aligned(cva, monkeypatch, capsys) -> None:
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(sys, "argv", ["check_version_alignment.py"])
    assert cva.main() == 0
    assert "All versions aligned!" in capsys.readouterr().out
