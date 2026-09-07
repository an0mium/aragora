"""The native<->ODR mapping doc must document every ODR top-level field the
emitter produces, so the mapping cannot silently drift from odr_export."""

from __future__ import annotations

from pathlib import Path

from aragora.gauntlet.odr_export import decision_receipt_to_odr

from tests.gauntlet.test_odr_export import _full_receipt  # existing factory (verified)

MAPPING_DOC = Path("docs/specs/odr-native-mapping.md")


def test_mapping_doc_covers_every_odr_field():
    odr = decision_receipt_to_odr(_full_receipt())
    doc = MAPPING_DOC.read_text(encoding="utf-8")
    missing = [k for k in odr.keys() if f"`{k}`" not in doc]
    assert not missing, f"mapping doc missing ODR fields: {missing}"
