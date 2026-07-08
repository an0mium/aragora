"""Packaging metadata guards for the standalone verifier."""

from __future__ import annotations

import tomllib
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = PACKAGE_ROOT / "pyproject.toml"
CHANGELOG = PACKAGE_ROOT / "CHANGELOG.md"


def _project_metadata() -> dict[str, object]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]


def test_package_version_tracks_unreleased_security_floor_bump() -> None:
    project = _project_metadata()

    assert project["version"] == "0.1.2"


def test_standalone_dependency_floor_blocks_known_cryptography_advisory() -> None:
    project = _project_metadata()

    assert "cryptography>=48.0.1" in project["dependencies"]


def test_changelog_records_unreleased_floor_bump() -> None:
    text = CHANGELOG.read_text(encoding="utf-8")

    assert "## [0.1.2] — Unreleased" in text
    assert "cryptography" in text
    assert ">=48.0.1" in text
    assert "GHSA-537c-gmf6-5ccf" in text
