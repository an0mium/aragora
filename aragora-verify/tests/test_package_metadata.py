"""Packaging metadata guards for the standalone verifier."""

from __future__ import annotations

import ast
import re
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
PYPROJECT = PACKAGE_ROOT / "pyproject.toml"
CHANGELOG = PACKAGE_ROOT / "CHANGELOG.md"
INDEPENDENT_VERIFIER_GUIDE = REPO_ROOT / "docs/specs/INDEPENDENT_VERIFIER_GUIDE.md"


def _project_metadata() -> dict[str, object]:
    text = PYPROJECT.read_text(encoding="utf-8")
    version_match = re.search(r'^version = "([^"]+)"$', text, flags=re.MULTILINE)
    dependencies_match = re.search(
        r"^dependencies = \[\n(?P<body>.*?)\n\]",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )

    assert version_match is not None
    assert dependencies_match is not None

    dependencies = re.findall(
        r'^\s*"([^"]+)",$', dependencies_match.group("body"), flags=re.MULTILINE
    )
    return {
        "version": version_match.group(1),
        "dependencies": dependencies,
    }


def test_metadata_guard_is_python_310_compatible() -> None:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert "tomllib" not in imported_names


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


def test_crypto_floor_docs_call_out_wheel_openssl_and_publish_boundary() -> None:
    text = INDEPENDENT_VERIFIER_GUIDE.read_text(encoding="utf-8")

    assert "wheel" in text
    assert "OpenSSL" in text
    assert "0.1.2" in text
    assert "publish" in text
