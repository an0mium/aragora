"""Tests for scripts/check_version_alignment.py doc-pattern helpers."""

from __future__ import annotations

import importlib.util
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


@pytest.fixture(scope="module")
def cva():
    spec = importlib.util.spec_from_file_location("check_version_alignment", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_repo_docs_are_aligned(cva, monkeypatch, capsys) -> None:
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(sys, "argv", ["check_version_alignment.py"])
    assert cva.main() == 0
    assert "All versions aligned!" in capsys.readouterr().out
