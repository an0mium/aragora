"""Tests for DIC-27 operator crux arbitration (aragora/epistemic/arbitration.py)."""

from __future__ import annotations

import pytest

from aragora.epistemic.arbitration import (
    DEFAULT_EXPIRY_DAYS,
    PERSISTENT_CRUX_MIN_CONSECUTIVE,
    PERSISTENT_CRUX_MIN_SCORE,
    CruxArbitration,
    CruxArbitrationReversal,
    PersistentCrux,
    build_arbitration,
    build_reversal,
    crux_arbitration_enabled,
)


# ──────────────────────────── helpers ──────────────────────────────────────────


def _make_crux(**kwargs: object) -> PersistentCrux:
    defaults: dict = dict(
        crux_id="crux_001",
        statement="Expanding B2 guard requires three consecutive green soaks",
        question_family_id="b2_guard_expansion",
        consecutive_debate_count=3,
        load_bearing_score=0.82,
        cruxset_receipt_ids=("crux_rcpt_abc", "crux_rcpt_def", "crux_rcpt_ghi"),
    )
    defaults.update(kwargs)
    return PersistentCrux(**defaults)


def _make_arbitration(**kwargs: object) -> CruxArbitration:
    crux = _make_crux()
    build_kw: dict = dict(operator="alice", side="accept", rationale="Three soaks are sufficient")
    build_kw.update(kwargs)
    return build_arbitration(crux, **build_kw)


# ──────────────────────────── PersistentCrux ───────────────────────────────────


class TestPersistentCrux:
    def test_qualifies_at_exact_thresholds(self) -> None:
        crux = _make_crux(
            consecutive_debate_count=PERSISTENT_CRUX_MIN_CONSECUTIVE,
            load_bearing_score=PERSISTENT_CRUX_MIN_SCORE,
        )
        assert crux.qualifies is True

    def test_not_qualifies_score_below_threshold(self) -> None:
        crux = _make_crux(load_bearing_score=PERSISTENT_CRUX_MIN_SCORE - 0.01)
        assert crux.qualifies is False

    def test_not_qualifies_count_below_threshold(self) -> None:
        crux = _make_crux(consecutive_debate_count=PERSISTENT_CRUX_MIN_CONSECUTIVE - 1)
        assert crux.qualifies is False

    def test_to_dict_includes_qualifies(self) -> None:
        crux = _make_crux()
        d = crux.to_dict()
        assert d["qualifies"] is True
        assert d["crux_id"] == "crux_001"
        assert len(d["cruxset_receipt_ids"]) == 3
        assert d["load_bearing_score"] == round(0.82, 4)

    def test_cruxset_receipt_ids_is_immutable(self) -> None:
        crux = _make_crux()
        assert isinstance(crux.cruxset_receipt_ids, tuple)
        with pytest.raises((AttributeError, TypeError)):
            crux.cruxset_receipt_ids = ("x",)  # type: ignore[misc]


# ──────────────────────────── build_arbitration ────────────────────────────────


class TestBuildArbitration:
    def test_id_prefix(self) -> None:
        arb = _make_arbitration()
        assert arb.arbitration_id.startswith("arb_")

    def test_fields_stored(self) -> None:
        arb = _make_arbitration(side="reject", rationale="needs more data")
        assert arb.operator == "alice"
        assert arb.side == "reject"
        assert arb.rationale == "needs more data"
        assert arb.is_reversed is False
        assert arb.reversal_receipt_id == ""

    def test_checksum_is_64_hex_chars(self) -> None:
        arb = _make_arbitration()
        assert len(arb.checksum) == 64
        assert all(c in "0123456789abcdef" for c in arb.checksum)

    def test_to_dict_preserves_checksum(self) -> None:
        arb = _make_arbitration()
        assert arb.to_dict()["checksum"] == arb.checksum

    def test_to_dict_contains_crux(self) -> None:
        arb = _make_arbitration()
        d = arb.to_dict()
        assert d["crux"]["crux_id"] == "crux_001"
        assert d["is_reversed"] is False

    def test_evidence_citations_default_empty_tuple(self) -> None:
        arb = _make_arbitration()
        assert arb.evidence_citations == ()
        assert isinstance(arb.evidence_citations, tuple)

    def test_evidence_citations_stored_as_tuple(self) -> None:
        crux = _make_crux()
        arb = build_arbitration(
            crux,
            operator="carol",
            side="defer",
            rationale="pending",
            evidence_citations=["docs/status/B0.md", "issue:#5329"],
        )
        assert arb.evidence_citations == ("docs/status/B0.md", "issue:#5329")
        assert isinstance(arb.evidence_citations, tuple)

    def test_evidence_citations_immutable(self) -> None:
        crux = _make_crux()
        arb = build_arbitration(
            crux,
            operator="x",
            side="accept",
            rationale="ok",
            evidence_citations=["a"],
        )
        with pytest.raises((AttributeError, TypeError)):
            arb.evidence_citations.append("b")  # type: ignore[attr-defined]

    def test_checksum_changes_when_citations_differ(self) -> None:
        crux = _make_crux()
        arb_no_cites = build_arbitration(crux, operator="x", side="accept", rationale="ok")
        arb_with_cite = build_arbitration(
            crux,
            operator="x",
            side="accept",
            rationale="ok",
            evidence_citations=["docs/status/B0.md"],
        )
        assert arb_no_cites.checksum != arb_with_cite.checksum

    def test_checksum_covers_crux_state(self) -> None:
        crux_a = _make_crux(statement="Soaks required")
        crux_b = _make_crux(statement="Soaks NOT required")
        arb_a = build_arbitration(crux_a, operator="x", side="accept", rationale="ok")
        arb_b = build_arbitration(crux_b, operator="x", side="accept", rationale="ok")
        assert arb_a.checksum != arb_b.checksum

    def test_expiry_days_respected(self) -> None:
        from datetime import datetime, timezone

        crux = _make_crux()
        arb = build_arbitration(crux, operator="x", side="accept", rationale="ok", expiry_days=30)
        created = datetime.fromisoformat(arb.created_at)
        expires = datetime.fromisoformat(arb.expires_at)
        delta = expires - created
        assert 29 <= delta.days <= 31

    def test_default_not_expired(self) -> None:
        arb = _make_arbitration()
        assert arb.is_expired is False

    def test_distinct_ids_per_call(self) -> None:
        arb1 = _make_arbitration()
        arb2 = _make_arbitration()
        assert arb1.arbitration_id != arb2.arbitration_id
        assert arb1.checksum != arb2.checksum


# ──────────────────────────── is_expired ───────────────────────────────────────


class TestIsExpired:
    def test_malformed_expires_at_fails_closed(self) -> None:
        from dataclasses import replace

        crux = _make_crux()
        arb = build_arbitration(crux, operator="x", side="accept", rationale="ok")
        bad_arb = replace(arb, expires_at="not-a-date")
        assert bad_arb.is_expired is True  # fail closed, not False

    def test_past_expiry_is_expired(self) -> None:
        from dataclasses import replace

        crux = _make_crux()
        arb = build_arbitration(crux, operator="x", side="accept", rationale="ok")
        past_arb = replace(arb, expires_at="2020-01-01T00:00:00+00:00")
        assert past_arb.is_expired is True


# ──────────────────────────── build_reversal ───────────────────────────────────


class TestBuildReversal:
    def test_creates_pair(self) -> None:
        arb = _make_arbitration()
        updated, reversal = build_reversal(arb, reversed_by="bob", reason="new evidence")
        assert updated.is_reversed is True
        assert updated.reversal_receipt_id == reversal.reversal_id
        assert reversal.arbitration_id == arb.arbitration_id
        assert reversal.reversed_by == "bob"
        assert reversal.reason == "new evidence"

    def test_original_checksum_preserved(self) -> None:
        arb = _make_arbitration()
        original_checksum = arb.checksum
        updated, _ = build_reversal(arb, reversed_by="bob", reason="reason")
        assert updated.checksum == original_checksum

    def test_reversal_id_prefix(self) -> None:
        arb = _make_arbitration()
        _, reversal = build_reversal(arb, reversed_by="bob", reason="reason")
        assert reversal.reversal_id.startswith("rev_")

    def test_reversal_checksum_64_chars(self) -> None:
        arb = _make_arbitration()
        _, reversal = build_reversal(arb, reversed_by="bob", reason="reason")
        assert len(reversal.checksum) == 64

    def test_reversal_to_dict(self) -> None:
        arb = _make_arbitration()
        _, reversal = build_reversal(arb, reversed_by="bob", reason="new evidence")
        d = reversal.to_dict()
        assert d["reversed_by"] == "bob"
        assert d["reason"] == "new evidence"
        assert d["arbitration_id"] == arb.arbitration_id

    def test_double_reversal_preserves_receipt(self) -> None:
        arb = _make_arbitration()
        updated1, rev1 = build_reversal(arb, reversed_by="bob", reason="first")
        updated2, rev2 = build_reversal(updated1, reversed_by="carol", reason="second")
        assert updated2.reversal_receipt_id == rev2.reversal_id
        assert rev2.arbitration_id == arb.arbitration_id


# ──────────────────────────── flag gate ────────────────────────────────────────


class TestFlagGate:
    def test_default_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARAGORA_CRUX_ARBITRATION_ENABLED", raising=False)
        assert crux_arbitration_enabled() is False

    def test_truthy_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for value in ("1", "true", "yes", "on", "True", "YES"):
            monkeypatch.setenv("ARAGORA_CRUX_ARBITRATION_ENABLED", value)
            assert crux_arbitration_enabled() is True

    def test_falsy_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for value in ("0", "false", "no", "off", ""):
            monkeypatch.setenv("ARAGORA_CRUX_ARBITRATION_ENABLED", value)
            assert crux_arbitration_enabled() is False
