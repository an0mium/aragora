from pathlib import Path


RECEIPTS_PAGE = Path("aragora/live/src/app/(app)/receipts/page.tsx")


def test_receipts_page_preserves_structured_dissent_reasons() -> None:
    source = RECEIPTS_PAGE.read_text(encoding="utf-8")

    assert "Array.isArray(record.reasons)" in source
    assert "reasons.join('; ')" in source
    assert "safeString(record.alternative)" in source
