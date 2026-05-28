"""DIC-27 arbitration rows in the DIC-18 truth map.

Tests for ArbitrationRow and the arbitrations= parameter of build_truth_map.
Flag-gated via ARAGORA_CRUX_ARBITRATION_ENABLED; default off.
Advances issue #6221.
"""

from __future__ import annotations

import dataclasses
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from aragora.epistemic.arbitration import (
    PERSISTENT_CRUX_MIN_CONSECUTIVE,
    PERSISTENT_CRUX_MIN_SCORE,
    CruxArbitration,
    PersistentCrux,
    build_arbitration,
    build_reversal,
)
from aragora.epistemic.claim_verifier import ClaimResult, ClaimStatus
from aragora.epistemic.truth_map import ArbitrationRow, OrgTruthMapReport, build_truth_map


# ─────────────────────────── helpers ────────────────────────────────────────


def _make_crux(crux_id: str = "crux-1") -> PersistentCrux:
    return PersistentCrux(
        crux_id=crux_id,
        statement="Should we expand B2?",
        question_family_id="qf-b2",
        consecutive_debate_count=PERSISTENT_CRUX_MIN_CONSECUTIVE,
        load_bearing_score=PERSISTENT_CRUX_MIN_SCORE,
        cruxset_receipt_ids=("r1",),
    )


def _make_arb(crux_id: str = "crux-1", expiry_days: int = 90) -> CruxArbitration:
    return build_arbitration(
        crux=_make_crux(crux_id),
        operator="alice",
        side="accept",
        rationale="Evidence favours expansion.",
        expiry_days=expiry_days,
    )


def _make_expired_arb(crux_id: str = "crux-expired") -> CruxArbitration:
    arb = _make_arb(crux_id)
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    return dataclasses.replace(arb, expires_at=past)


def _make_reversed_arb(crux_id: str = "crux-rev") -> CruxArbitration:
    arb = _make_arb(crux_id)
    updated, _ = build_reversal(arb, reversed_by="bob", reason="New evidence.")
    return updated


# ─────────────────────────── ArbitrationRow ──────────────────────────────────


class TestArbitrationRow:
    def test_to_dict_round_trips_all_fields(self) -> None:
        now = datetime.now(timezone.utc)
        row = ArbitrationRow(
            arbitration_id="a1",
            crux_id="c1",
            statement="Q?",
            side="accept",
            operator="alice",
            created_at=now.isoformat(),
            expires_at=(now + timedelta(days=90)).isoformat(),
            age_hours=2.5,
            is_expired=False,
            is_reversed=False,
        )
        d = row.to_dict()
        assert d["arbitration_id"] == "a1"
        assert d["crux_id"] == "c1"
        assert d["side"] == "accept"
        assert d["is_expired"] is False
        assert d["age_hours"] == pytest.approx(2.5)


# ─────────────────────────── flag gating ─────────────────────────────────────


class TestArbitrationFlagGating:
    def test_empty_arbitrations_list_gives_zero_counts(self) -> None:
        r = build_truth_map(claim_results=[])
        assert r.arbitrations == []
        assert r.active_arbitrations == 0
        assert r.expired_arbitrations == 0
        assert r.reversed_arbitrations == 0

    def test_arbitrations_ignored_when_flag_off(self) -> None:
        arb = _make_arb()
        with patch.dict(os.environ, {"ARAGORA_CRUX_ARBITRATION_ENABLED": "0"}):
            r = build_truth_map(claim_results=[], arbitrations=[arb])
        assert r.arbitrations == []
        assert r.active_arbitrations == 0

    def test_arbitrations_populated_when_flag_on(self) -> None:
        arb = _make_arb()
        with patch.dict(os.environ, {"ARAGORA_CRUX_ARBITRATION_ENABLED": "1"}):
            r = build_truth_map(claim_results=[], arbitrations=[arb])
        assert len(r.arbitrations) == 1
        assert r.active_arbitrations == 1


# ─────────────────────────── row field accuracy ───────────────────────────────


class TestArbitrationRowFields:
    def test_row_fields_match_arbitration(self) -> None:
        arb = _make_arb("crux-xyz")
        with patch.dict(os.environ, {"ARAGORA_CRUX_ARBITRATION_ENABLED": "1"}):
            r = build_truth_map(claim_results=[], arbitrations=[arb])
        row = r.arbitrations[0]
        assert row.crux_id == "crux-xyz"
        assert row.statement == "Should we expand B2?"
        assert row.side == "accept"
        assert row.operator == "alice"
        assert row.is_expired is False
        assert row.is_reversed is False
        assert row.age_hours >= 0.0

    def test_age_hours_is_non_negative(self) -> None:
        arb = _make_arb()
        with patch.dict(os.environ, {"ARAGORA_CRUX_ARBITRATION_ENABLED": "1"}):
            r = build_truth_map(claim_results=[], arbitrations=[arb])
        assert r.arbitrations[0].age_hours >= 0.0


# ─────────────────────────── expired / reversed ───────────────────────────────


class TestArbitrationCounting:
    def test_expired_counted_separately(self) -> None:
        arb = _make_expired_arb()
        with patch.dict(os.environ, {"ARAGORA_CRUX_ARBITRATION_ENABLED": "1"}):
            r = build_truth_map(claim_results=[], arbitrations=[arb])
        assert r.expired_arbitrations == 1
        assert r.active_arbitrations == 0
        assert r.arbitrations[0].is_expired is True

    def test_reversed_counted_separately(self) -> None:
        arb = _make_reversed_arb()
        with patch.dict(os.environ, {"ARAGORA_CRUX_ARBITRATION_ENABLED": "1"}):
            r = build_truth_map(claim_results=[], arbitrations=[arb])
        assert r.reversed_arbitrations == 1
        assert r.active_arbitrations == 0
        assert r.arbitrations[0].is_reversed is True

    def test_mixed_arbitrations_all_counted(self) -> None:
        active = _make_arb("c-active")
        expired = _make_expired_arb("c-expired")
        rev = _make_reversed_arb("c-rev")
        with patch.dict(os.environ, {"ARAGORA_CRUX_ARBITRATION_ENABLED": "1"}):
            r = build_truth_map(claim_results=[], arbitrations=[active, expired, rev])
        assert len(r.arbitrations) == 3
        assert r.active_arbitrations == 1
        assert r.expired_arbitrations == 1
        assert r.reversed_arbitrations == 1


# ─────────────────────────── to_dict ─────────────────────────────────────────


class TestToDict:
    def test_to_dict_includes_arbitration_keys(self) -> None:
        arb = _make_arb()
        with patch.dict(os.environ, {"ARAGORA_CRUX_ARBITRATION_ENABLED": "1"}):
            d = build_truth_map(claim_results=[], arbitrations=[arb]).to_dict()
        assert "arbitrations" in d
        assert len(d["arbitrations"]) == 1
        assert d["summary"]["active_arbitrations"] == 1
        assert d["summary"]["expired_arbitrations"] == 0
        assert d["summary"]["reversed_arbitrations"] == 0

    def test_to_dict_summary_has_arbitration_counts_when_flag_off(self) -> None:
        arb = _make_arb()
        with patch.dict(os.environ, {"ARAGORA_CRUX_ARBITRATION_ENABLED": "0"}):
            d = build_truth_map(claim_results=[], arbitrations=[arb]).to_dict()
        assert d["arbitrations"] == []
        assert d["summary"]["active_arbitrations"] == 0


# ─────────────────────────── orthogonality ───────────────────────────────────


class TestOrthogonality:
    def test_claim_counts_unaffected_by_arbitrations(self) -> None:
        cr = ClaimResult(claim_id="c1", status=ClaimStatus.PASS, message="", detail={})
        arb = _make_arb()
        with patch.dict(os.environ, {"ARAGORA_CRUX_ARBITRATION_ENABLED": "1"}):
            r = build_truth_map(claim_results=[cr], arbitrations=[arb])
        assert r.total_claims == 1
        assert r.passing_claims == 1
        assert r.active_arbitrations == 1

    def test_no_arbitrations_param_preserves_existing_summary_keys(self) -> None:
        d = build_truth_map(claim_results=[]).to_dict()
        expected = {
            "total_claims", "passing", "failing", "stale",
            "unsupported", "error", "open_crux_count",
            "active_arbitrations", "expired_arbitrations", "reversed_arbitrations",
        }
        assert expected <= d["summary"].keys()
