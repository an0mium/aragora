"""Tests for DIC-16 CruxReceipt (aragora/epistemic/crux_receipt.py)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from aragora.epistemic.crux_receipt import (
    CruxEntry,
    CruxReceipt,
    build_crux_receipt,
    crux_receipt_enabled,
    enable_crux_receipt,
)


@dataclass
class _Claim:
    claim_id: str
    statement: str
    crux_score: float
    uncertainty_score: float
    contesting_agents: list[str]
    affected_claims: list[str]
    resolution_impact: float


@dataclass
class _Result:
    debate_id: str
    question: str
    _cruxes: list[_Claim]
    _barrier: float
    counterfactuals: list[dict[str, Any]] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    rounds: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def top_cruxes(self) -> list[_Claim]:
        return self._cruxes

    def convergence_barrier(self) -> float:
        return self._barrier


_S = [(0.8, 0.60, 0.40), (0.7, 0.55, 0.35), (0.6, 0.50, 0.30)]


def _r(n: int = 2) -> _Result:
    return _Result(
        debate_id="d_x",
        question="Expand now?",
        _cruxes=[
            _Claim(f"c{i}", f"s{i}", _S[i][0], _S[i][1], ["ag_a"], [f"dep{i}"], _S[i][2])
            for i in range(n)
        ],
        _barrier=0.73,
        agents=["ag_a", "ag_b"],
        rounds=3,
        metadata={"mode": "crux_finder"},
    )


def test_flag_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARAGORA_CRUX_RECEIPT_ENABLED", raising=False)
    assert crux_receipt_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "on"])
def test_flag_on(monkeypatch: pytest.MonkeyPatch, val: str) -> None:
    monkeypatch.setenv("ARAGORA_CRUX_RECEIPT_ENABLED", val)
    assert crux_receipt_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "no", ""])
def test_flag_off_falsy(monkeypatch: pytest.MonkeyPatch, val: str) -> None:
    monkeypatch.setenv("ARAGORA_CRUX_RECEIPT_ENABLED", val)
    assert crux_receipt_enabled() is False


def test_enable_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARAGORA_CRUX_RECEIPT_ENABLED", raising=False)
    enable_crux_receipt()
    assert crux_receipt_enabled() is True


def test_crux_entry_to_dict_rounds_scores() -> None:
    e = CruxEntry("x", "s", 0.123456789, 0.9, [], [], 0.333333)
    d = e.to_dict()
    assert d["load_bearing_score"] == round(0.123456789, 4)
    assert d["resolution_impact"] == round(0.333333, 4)


def test_receipt_fields_match_result() -> None:
    rc = build_crux_receipt(_r(n=2))
    assert rc.debate_id == "d_x"
    assert len(rc.cruxes) == 2
    assert rc.cruxes[0].crux_id == "c0"
    assert rc.rounds == 3
    assert rc.convergence_barrier == pytest.approx(0.73)
    assert rc.receipt_id.startswith("crux_rcpt_")


def test_empty_cruxes() -> None:
    rc = build_crux_receipt(_r(n=0))
    assert rc.cruxes == []
    assert len(rc.checksum) == 64


def test_checksum_64_char_hex() -> None:
    rc = build_crux_receipt(_r())
    assert len(rc.checksum) == 64
    assert all(c in "0123456789abcdef" for c in rc.checksum)


def test_two_builds_have_different_checksums() -> None:
    r = _r()
    ra, rb = build_crux_receipt(r), build_crux_receipt(r)
    assert ra.receipt_id != rb.receipt_id
    assert ra.checksum != rb.checksum
    assert ra.cruxes == rb.cruxes


def test_to_dict_json_serializable() -> None:
    rc = build_crux_receipt(_r())
    data = json.loads(json.dumps(rc.to_dict()))
    assert data["debate_id"] == rc.debate_id
    assert len(data["checksum"]) == 64


def test_load_bearing_score_maps_from_crux_score() -> None:
    rc = build_crux_receipt(_r(n=1))
    assert rc.cruxes[0].load_bearing_score == pytest.approx(0.8)


def test_metadata_pass_through() -> None:
    r = _r()
    r.metadata = {"mode": "crux_finder", "tag": 42}
    assert build_crux_receipt(r).metadata["tag"] == 42
