"""Unit tests for aragora.epistemic.genealogy (DIC-24 / #6218)."""

from __future__ import annotations

import pytest

from aragora.epistemic.genealogy import (
    CodeUnitGenealogy,
    GenealogyEntry,
    GenealogyStore,
    InMemoryGenealogyStore,
    _chain_checksum,
    get_genealogy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(
    kind: str = "decay_signal",
    entry_id: str = "test-entry-1",
    ts: str = "2026-04-25T00:00:00Z",
    checksum: str = "abc123",
) -> GenealogyEntry:
    return GenealogyEntry(
        entry_kind=kind,  # type: ignore[arg-type]
        entry_id=entry_id,
        checksum=checksum,
        timestamp=ts,
    )


# ---------------------------------------------------------------------------
# GenealogyEntry
# ---------------------------------------------------------------------------


class TestGenealogyEntry:
    def test_roundtrip_excludes_empty_metadata(self) -> None:
        e = _entry("decision_receipt", "dr-1", "2026-04-25T10:00:00Z")
        d = e.to_dict()
        assert d["entry_kind"] == "decision_receipt"
        assert d["entry_id"] == "dr-1"
        assert "metadata" not in d

    def test_metadata_included_when_present(self) -> None:
        e = GenealogyEntry(
            entry_kind="repair_proposal",
            entry_id="rp-1",
            checksum="deadbeef",
            timestamp="2026-04-25T00:00:00Z",
            metadata={"repair_kind": "pr_candidate"},
        )
        assert e.to_dict()["metadata"]["repair_kind"] == "pr_candidate"

    def test_metadata_defensive_copy_on_construction(self) -> None:
        caller_dict = {"key": "original", "nested": {"inner": "original"}}
        e = GenealogyEntry(
            entry_kind="decay_signal",
            entry_id="e1",
            checksum="abc",
            timestamp="2026-04-25T00:00:00Z",
            metadata=caller_dict,
        )
        caller_dict["key"] = "mutated"
        caller_dict["nested"]["inner"] = "mutated"
        assert e.metadata["key"] == "original"
        assert e.metadata["nested"]["inner"] == "original"

    def test_metadata_defensive_copy_on_to_dict(self) -> None:
        e = GenealogyEntry(
            entry_kind="decay_signal",
            entry_id="e1",
            checksum="abc",
            timestamp="2026-04-25T00:00:00Z",
            metadata={"key": "original", "nested": {"inner": "original"}},
        )
        d = e.to_dict()
        d["metadata"]["key"] = "mutated"
        d["metadata"]["nested"]["inner"] = "mutated"
        assert e.metadata["key"] == "original"
        assert e.metadata["nested"]["inner"] == "original"

    def test_invalid_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="entry_kind"):
            GenealogyEntry(
                entry_kind="unknown_kind",  # type: ignore[arg-type]
                entry_id="x",
                checksum="",
                timestamp="2026-04-25T00:00:00Z",
            )

    def test_empty_entry_id_raises(self) -> None:
        with pytest.raises(ValueError, match="entry_id"):
            GenealogyEntry(
                entry_kind="decay_signal",
                entry_id="",
                checksum="",
                timestamp="2026-04-25T00:00:00Z",
            )

    def test_invalid_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="timestamp"):
            GenealogyEntry(
                entry_kind="decay_signal",
                entry_id="e1",
                checksum="abc",
                timestamp="not-a-date",
            )

    def test_timestamp_normalized_to_canonical_form(self) -> None:
        # +00:00 offset form → Z suffix; millisecond precision enforced
        e = GenealogyEntry(
            entry_kind="decay_signal",
            entry_id="e1",
            checksum="abc",
            timestamp="2026-04-25T12:00:00+00:00",
        )
        assert e.timestamp == "2026-04-25T12:00:00.000Z"

    def test_timestamp_z_suffix_accepted(self) -> None:
        e = _entry(ts="2026-04-25T00:00:00Z")
        assert e.timestamp == "2026-04-25T00:00:00.000Z"

    def test_naive_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="timezone"):
            GenealogyEntry(
                entry_kind="decay_signal",
                entry_id="e1",
                checksum="abc",
                timestamp="2026-04-25T00:00:00",
            )

    def test_all_valid_kinds_accepted(self) -> None:
        for kind in ("decision_receipt", "decay_signal", "crux_receipt", "repair_proposal"):
            e = GenealogyEntry(
                entry_kind=kind,  # type: ignore[arg-type]
                entry_id="eid",
                checksum="c",
                timestamp="2026-04-25T00:00:00Z",
            )
            assert e.entry_kind == kind


# ---------------------------------------------------------------------------
# _chain_checksum
# ---------------------------------------------------------------------------


class TestChainChecksum:
    def test_empty_entries_is_deterministic(self) -> None:
        c1 = _chain_checksum([])
        c2 = _chain_checksum([])
        assert c1 == c2
        assert len(c1) == 64  # SHA-256 hex

    def test_order_independent(self) -> None:
        e1 = _entry("decay_signal", "e1", "2026-04-25T01:00:00Z")
        e2 = _entry("crux_receipt", "e2", "2026-04-25T02:00:00Z")
        assert _chain_checksum([e1, e2]) == _chain_checksum([e2, e1])

    def test_different_entries_give_different_checksums(self) -> None:
        e1 = _entry("decay_signal", "e1", "2026-04-25T01:00:00Z")
        e2 = _entry("crux_receipt", "e2", "2026-04-25T01:00:00Z")
        assert _chain_checksum([e1]) != _chain_checksum([e2])

    def test_checksum_changes_when_content_changes(self) -> None:
        e1 = _entry("decay_signal", "e1", "2026-04-25T01:00:00Z", checksum="aaa")
        e2 = _entry("decay_signal", "e1", "2026-04-25T01:00:00Z", checksum="bbb")
        assert _chain_checksum([e1]) != _chain_checksum([e2])


# ---------------------------------------------------------------------------
# CodeUnitGenealogy
# ---------------------------------------------------------------------------


class TestCodeUnitGenealogy:
    def test_build_empty_code_unit_id_raises(self) -> None:
        with pytest.raises(ValueError, match="code_unit_id"):
            CodeUnitGenealogy.build("", [])

    def test_build_sorts_by_timestamp(self) -> None:
        e1 = _entry("decay_signal", "e1", "2026-04-25T03:00:00Z")
        e2 = _entry("decision_receipt", "e2", "2026-04-25T01:00:00Z")
        e3 = _entry("crux_receipt", "e3", "2026-04-25T02:00:00Z")
        g = CodeUnitGenealogy.build("unit.foo", [e1, e2, e3])
        assert [e.entry_id for e in g.entries] == ["e2", "e3", "e1"]

    def test_build_secondary_sort_by_entry_id(self) -> None:
        # Same timestamp — should sort by entry_id
        e1 = _entry("decay_signal", "z-first", "2026-04-25T01:00:00Z")
        e2 = _entry("crux_receipt", "a-second", "2026-04-25T01:00:00Z")
        g = CodeUnitGenealogy.build("unit.foo", [e1, e2])
        assert g.entries[0].entry_id == "a-second"
        assert g.entries[1].entry_id == "z-first"

    def test_to_dict_has_all_keys(self) -> None:
        g = CodeUnitGenealogy.build("unit.foo", [_entry()])
        d = g.to_dict()
        for key in ("code_unit_id", "entries", "chain_checksum", "generated_at", "entry_count"):
            assert key in d
        assert d["entry_count"] == 1
        assert d["code_unit_id"] == "unit.foo"

    def test_empty_genealogy_to_dict(self) -> None:
        g = CodeUnitGenealogy.build("unit.empty", [])
        d = g.to_dict()
        assert d["entry_count"] == 0
        assert d["entries"] == []
        assert len(d["chain_checksum"]) == 64

    def test_chain_checksum_stable_regardless_of_input_order(self) -> None:
        entries = [_entry("decay_signal", f"e{i}", f"2026-04-25T0{i}:00:00Z") for i in range(1, 4)]
        g1 = CodeUnitGenealogy.build("unit.foo", entries)
        g2 = CodeUnitGenealogy.build("unit.foo", list(reversed(entries)))
        assert g1.chain_checksum == g2.chain_checksum

    def test_chain_checksum_differs_for_different_entries(self) -> None:
        g1 = CodeUnitGenealogy.build("unit.foo", [_entry("decay_signal", "e1")])
        g2 = CodeUnitGenealogy.build("unit.foo", [_entry("crux_receipt", "e2")])
        assert g1.chain_checksum != g2.chain_checksum


# ---------------------------------------------------------------------------
# InMemoryGenealogyStore
# ---------------------------------------------------------------------------


class TestInMemoryGenealogyStore:
    def test_get_entries_empty(self) -> None:
        store = InMemoryGenealogyStore()
        assert store.get_entries("unit.missing") == []

    def test_add_and_retrieve(self) -> None:
        store = InMemoryGenealogyStore()
        e = _entry("repair_proposal", "rp-1")
        store.add("unit.foo", e)
        assert store.get_entries("unit.foo") == [e]
        assert store.get_entries("unit.bar") == []

    def test_multiple_entries_for_same_unit(self) -> None:
        store = InMemoryGenealogyStore()
        e1 = _entry("decay_signal", "e1")
        e2 = _entry("crux_receipt", "e2")
        store.add("unit.foo", e1)
        store.add("unit.foo", e2)
        assert len(store.get_entries("unit.foo")) == 2

    def test_returns_copy_not_reference(self) -> None:
        store = InMemoryGenealogyStore()
        store.add("unit.foo", _entry())
        result = store.get_entries("unit.foo")
        result.clear()
        assert len(store.get_entries("unit.foo")) == 1

    def test_implements_protocol(self) -> None:
        store = InMemoryGenealogyStore()
        assert isinstance(store, GenealogyStore)


# ---------------------------------------------------------------------------
# get_genealogy
# ---------------------------------------------------------------------------


class TestGetGenealogy:
    def test_empty_code_unit_id_raises(self) -> None:
        store = InMemoryGenealogyStore()
        with pytest.raises(ValueError, match="code_unit_id"):
            get_genealogy("", store, require_enabled=False)

    def test_flag_off_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARAGORA_GENEALOGY_ENABLED", raising=False)
        store = InMemoryGenealogyStore()
        with pytest.raises(RuntimeError, match="ARAGORA_GENEALOGY_ENABLED"):
            get_genealogy("unit.foo", store)

    def test_require_enabled_false_bypasses_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARAGORA_GENEALOGY_ENABLED", raising=False)
        store = InMemoryGenealogyStore()
        g = get_genealogy("unit.foo", store, require_enabled=False)
        assert g.code_unit_id == "unit.foo"
        assert g.entries == []

    def test_flag_enabled_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_GENEALOGY_ENABLED", "1")
        store = InMemoryGenealogyStore()
        store.add("unit.bar", _entry("decision_receipt", "dr-1"))
        g = get_genealogy("unit.bar", store)
        assert len(g.entries) == 1
        assert g.entries[0].entry_id == "dr-1"

    def test_flag_enabled_via_true_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_GENEALOGY_ENABLED", "true")
        store = InMemoryGenealogyStore()
        g = get_genealogy("unit.x", store)
        assert g.code_unit_id == "unit.x"

    def test_returns_genealogy_with_sorted_entries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_GENEALOGY_ENABLED", "1")
        store = InMemoryGenealogyStore()
        store.add("unit.chain", _entry("repair_proposal", "rp-1", "2026-04-25T05:00:00Z"))
        store.add("unit.chain", _entry("decision_receipt", "dr-1", "2026-04-25T01:00:00Z"))
        store.add("unit.chain", _entry("decay_signal", "ds-1", "2026-04-25T03:00:00Z"))
        g = get_genealogy("unit.chain", store)
        assert [e.entry_kind for e in g.entries] == [
            "decision_receipt",
            "decay_signal",
            "repair_proposal",
        ]
