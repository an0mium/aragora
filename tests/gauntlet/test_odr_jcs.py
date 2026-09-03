"""The JCS canonicalization leaf shared by the ODR emitter and signer.

``odr_export`` (emitter) and ``odr_signing`` (signer) both need the RFC 8785
digest; hosting it in a leaf module lets the signer depend on it without
importing the emitter, and lets the emitter keep re-exporting the public names.
"""

from __future__ import annotations

import ast
from pathlib import Path

from aragora.gauntlet import odr_export, odr_jcs, odr_signing

_GAUNTLET_DIR = Path(odr_signing.__file__).resolve().parent


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_exporter_reexports_the_leaf_implementation():
    assert odr_export.jcs_canonicalize is odr_jcs.jcs_canonicalize
    assert odr_export.odr_content_digest is odr_jcs.odr_content_digest
    assert "jcs_canonicalize" in odr_export.__all__
    assert "odr_content_digest" in odr_export.__all__


def test_signer_does_not_import_the_exporter():
    imports = _imported_modules(_GAUNTLET_DIR / "odr_signing.py")
    assert not any(name.startswith("aragora.gauntlet.odr_export") for name in imports), imports
    assert "aragora.gauntlet.odr_jcs.odr_content_digest" in imports


def test_leaf_module_imports_nothing_from_aragora():
    imports = _imported_modules(_GAUNTLET_DIR / "odr_jcs.py")
    assert not any(name.startswith("aragora") for name in imports), imports


def test_signer_and_exporter_import_in_either_order():
    import importlib
    import sys

    for first, second in (
        ("aragora.gauntlet.odr_signing", "aragora.gauntlet.odr_export"),
        ("aragora.gauntlet.odr_export", "aragora.gauntlet.odr_signing"),
    ):
        for name in ("aragora.gauntlet.odr_signing", "aragora.gauntlet.odr_export"):
            sys.modules.pop(name, None)
        importlib.import_module(first)
        importlib.import_module(second)


def test_leaf_digest_matches_shipped_verifier():
    from aragora_verify import odr_content_digest as verifier_digest

    doc = {
        "odr_version": "0.1",
        "receipt_id": "rcpt-1",
        "claim": {"verdict": "PASS", "score": 0.5},
        "signatures": [{"alg": "Ed25519", "key_id": "k", "signature": "c2ln"}],
    }
    assert odr_jcs.odr_content_digest(doc) == verifier_digest(doc)
    assert odr_jcs.jcs_canonicalize({"b": 1, "a": [1.0, 1e21]}) == b'{"a":[1,1e+21],"b":1}'
