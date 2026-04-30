"""Tests for aragora.epistemic.genealogy_report (DIC-24 / #6218).

Imports directly from modules (not aragora.epistemic) to avoid the yaml
transitive dependency when running with --noconftest.
"""

from __future__ import annotations

import pytest

from aragora.epistemic.genealogy import (
    CodeUnitGenealogy,
    GenealogyEntry,
    InMemoryGenealogyStore,
)
from aragora.epistemic.genealogy_report import (
    GenealogyReport,
    GenealogyUnitSummary,
    build_genealogy_report,
)


def _e(
    kind: str = "decay_signal", eid: str = "e1", ts: str = "2026-04-01T10:00:00Z"
) -> GenealogyEntry:
    return GenealogyEntry(entry_kind=kind, entry_id=eid, checksum="x", timestamp=ts)  # type: ignore[arg-type]


def _store(data: dict[str, list[GenealogyEntry]]) -> InMemoryGenealogyStore:
    s = InMemoryGenealogyStore()
    for uid, entries in data.items():
        for e in entries:
            s.add(uid, e)
    return s


# -- GenealogyUnitSummary.from_genealogy --


def test_summary_empty_chain() -> None:
    s = GenealogyUnitSummary.from_genealogy(CodeUnitGenealogy.build("u", []))
    assert s.entry_count == 0 and s.entry_kinds == () and s.oldest_timestamp is None


def test_summary_multiple_kinds_sorted() -> None:
    g = CodeUnitGenealogy.build(
        "u",
        [
            _e("decay_signal", "d1", "2026-04-01T00:00:00Z"),
            _e("repair_proposal", "r1", "2026-04-02T00:00:00Z"),
            _e("crux_receipt", "c1", "2026-04-03T00:00:00Z"),
        ],
    )
    s = GenealogyUnitSummary.from_genealogy(g)
    assert s.entry_count == 3
    assert s.entry_kinds == ("crux_receipt", "decay_signal", "repair_proposal")
    assert s.oldest_timestamp < s.newest_timestamp  # type: ignore[operator]


def test_summary_chain_checksum_preserved() -> None:
    g = CodeUnitGenealogy.build("u", [_e()])
    assert GenealogyUnitSummary.from_genealogy(g).chain_checksum == g.chain_checksum


# -- build_genealogy_report: empty and flag gate --


def test_empty_ids_skips_flag() -> None:
    r = build_genealogy_report([], InMemoryGenealogyStore(), require_enabled=True)
    assert r.unit_count == 0 and r.summaries == ()


def test_flag_off_raises() -> None:
    with pytest.raises(RuntimeError, match="ARAGORA_GENEALOGY_ENABLED"):
        build_genealogy_report(["u"], _store({"u": [_e()]}), require_enabled=True)


def test_require_disabled_passes() -> None:
    assert (
        build_genealogy_report(["u"], _store({"u": [_e()]}), require_enabled=False).unit_count == 1
    )


# -- build_genealogy_report: content --


def test_multiple_units_sorted_by_id() -> None:
    r = build_genealogy_report(
        ["u.beta", "u.alpha", "u.gamma"],
        _store(
            {
                "u.beta": [_e(eid="b")],
                "u.alpha": [_e(eid="a1"), _e(eid="a2", ts="2026-04-02T00:00:00Z")],
                "u.gamma": [],
            }
        ),
        require_enabled=False,
    )
    assert [s.code_unit_id for s in r.summaries] == ["u.alpha", "u.beta", "u.gamma"]
    assert r.total_entries == 3


def test_unknown_unit_empty_summary() -> None:
    r = build_genealogy_report(["ghost"], InMemoryGenealogyStore(), require_enabled=False)
    assert r.unit_count == 1 and r.total_entries == 0


# -- GenealogyReport helpers --


def _multi() -> GenealogyReport:
    return build_genealogy_report(
        ["u.a", "u.b", "u.c"],
        _store(
            {
                "u.a": [
                    _e("decay_signal", "d1"),
                    _e("decay_signal", "d2", "2026-04-02T00:00:00Z"),
                    _e("crux_receipt", "c1", "2026-04-03T00:00:00Z"),
                ],
                "u.b": [_e("repair_proposal", "r1")],
                "u.c": [],
            }
        ),
        require_enabled=False,
    )


def test_units_by_activity_order() -> None:
    ranked = _multi().units_by_activity()
    assert ranked[0].code_unit_id == "u.a" and ranked[-1].entry_count == 0


def test_units_with_kind_match_and_miss() -> None:
    r = _multi()
    assert [s.code_unit_id for s in r.units_with_kind("crux_receipt")] == ["u.a"]
    assert r.units_with_kind("decision_receipt") == []


def test_to_dict_structure() -> None:
    d = _multi().to_dict()
    assert set(d.keys()) == {"unit_count", "total_entries", "summaries", "generated_at"}
    assert d["unit_count"] == 3 and d["total_entries"] == 4


# -- __init__.py __all__ coverage (file-based, no yaml needed) --


def test_init_all_contains_genealogy_symbols() -> None:
    init_path = __file__.split("tests/epistemic/")[0] + "aragora/epistemic/__init__.py"
    source = open(init_path).read()
    for sym in (
        "GenealogyReport",
        "GenealogyUnitSummary",
        "build_genealogy_report",
        "CodeUnitGenealogy",
        "GenealogyEntry",
        "GenealogyStore",
        "InMemoryGenealogyStore",
        "get_genealogy",
    ):
        assert f'"{sym}"' in source, f"{sym!r} missing from aragora.epistemic.__all__"
