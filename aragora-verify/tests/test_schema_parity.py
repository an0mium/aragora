"""Byte-parity between the two ``odr_schema.json`` copies (VAL-VERIFY-009).

The ODR v0.1 JSON Schema is intentionally duplicated: the canonical in-tree
copy lives at ``aragora/gauntlet/odr_schema.json`` and a bundled copy ships
inside the standalone ``aragora-verify`` package so the verifier stays
installable and runnable with zero dependency on the monorepo (see
``aragora_verify.schema.load_bundled_schema``). Nothing enforces the two
copies stay identical except this test: if they drift, the offline verifier
could silently accept or reject receipts the emitter does not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_BUNDLED_SCHEMA = Path(__file__).resolve().parents[1] / "src" / "aragora_verify" / "odr_schema.json"
_IN_TREE_SCHEMA = Path(__file__).resolve().parents[2] / "aragora" / "gauntlet" / "odr_schema.json"


def test_bundled_schema_file_exists() -> None:
    assert _BUNDLED_SCHEMA.is_file(), _BUNDLED_SCHEMA


def test_bundled_schema_is_valid_json() -> None:
    doc = json.loads(_BUNDLED_SCHEMA.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    assert doc  # non-empty


def test_bundled_schema_is_byte_identical_to_in_tree_copy() -> None:
    # A monorepo checkout is required for this comparison; a standalone
    # aragora-verify checkout (e.g. an extracted sdist) has no ``aragora/``
    # sibling to diff against.
    if not _IN_TREE_SCHEMA.is_file():
        pytest.skip(f"in-tree schema not present at {_IN_TREE_SCHEMA} (standalone checkout)")
    bundled = _BUNDLED_SCHEMA.read_bytes()
    in_tree = _IN_TREE_SCHEMA.read_bytes()
    assert bundled == in_tree, (
        f"{_BUNDLED_SCHEMA} and {_IN_TREE_SCHEMA} have drifted; the two ODR "
        "schema copies must be kept byte-identical (mirror `diff` the two "
        "paths and sync whichever is stale)"
    )


def test_bundled_schema_matches_package_loader_output() -> None:
    # Guards against the package loader (importlib.resources) resolving a
    # different file than the one this test reads directly off disk -- e.g.
    # a stale installed copy shadowing the source tree.
    from aragora_verify.schema import load_bundled_schema

    on_disk = json.loads(_BUNDLED_SCHEMA.read_text(encoding="utf-8"))
    assert load_bundled_schema() == on_disk
