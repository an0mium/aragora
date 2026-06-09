"""Tests for DIC-16 CruxReceipt → Knowledge Mound bridge.

Hermetic: no real KM store, no real Arena, no network.

Stubs: ``yaml`` and a direct-loaded ``aragora.knowledge.unified.types``
are inserted into ``sys.modules`` before aragora imports so these tests
run in the uv-tool pytest venv (no pyyaml / pydantic). The real packages
take precedence in a fully-installed environment.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

# Stub 1: yaml (needed by aragora.epistemic chain)
if "yaml" not in sys.modules:
    _y = types.ModuleType("yaml")
    _y.safe_load = lambda s: {}  # type: ignore[attr-defined]
    sys.modules["yaml"] = _y

# Stub 2: load aragora.knowledge.unified.types directly, bypassing the
# aragora.knowledge __init__ (which imports fact_store → config → pydantic).
_KM_TYPES = "aragora.knowledge.unified.types"
if _KM_TYPES not in sys.modules:
    _p = Path(__file__).parent.parent.parent / "aragora" / "knowledge" / "unified" / "types.py"
    _spec = importlib.util.spec_from_file_location(_KM_TYPES, _p)
    assert _spec and _spec.loader
    _m = importlib.util.module_from_spec(_spec)
    sys.modules[_KM_TYPES] = _m
    _spec.loader.exec_module(_m)  # type: ignore[union-attr]
    for _ns in ("aragora.knowledge", "aragora.knowledge.unified"):
        sys.modules.setdefault(_ns, types.ModuleType(_ns))

from typing import Any

import pytest

from aragora.epistemic.crux_km_bridge import crux_receipt_to_knowledge_items
from aragora.epistemic.crux_receipt import CruxEntry, CruxReceipt

_km = sys.modules[_KM_TYPES]
ConfidenceLevel = _km.ConfidenceLevel  # type: ignore[attr-defined]
KnowledgeSource = _km.KnowledgeSource  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _entry(crux_id: str = "crux.t1", lbs: float = 0.80) -> CruxEntry:
    return CruxEntry(
        crux_id=crux_id,
        statement="The benchmark is fresh.",
        load_bearing_score=lbs,
        uncertainty_score=0.20,
        contesting_agents=["claude", "codex"],
        affected_claims=["claim.a", "claim.b"],
        resolution_impact=0.70,
    )


def _receipt(cruxes: list[CruxEntry] | None = None) -> CruxReceipt:
    return CruxReceipt(
        receipt_id="crux_rcpt_abc123",
        debate_id="debate_xyz",
        question="Should we adopt the new strategy?",
        cruxes=cruxes if cruxes is not None else [_entry()],
        convergence_barrier=0.65,
        counterfactuals=[],
        agents=["claude", "codex"],
        rounds=3,
        metadata={},
        checksum="a" * 64,
    )


# ---------------------------------------------------------------------------
# Flag-off: always returns empty regardless of cruxes
# ---------------------------------------------------------------------------


class TestFlagOff:
    def test_returns_empty_items(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARAGORA_CRUX_RECEIPT_ENABLED", raising=False)
        r = crux_receipt_to_knowledge_items(_receipt())
        assert r.items == [] and r.success is False

    def test_receipt_id_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARAGORA_CRUX_RECEIPT_ENABLED", raising=False)
        assert crux_receipt_to_knowledge_items(_receipt()).receipt_id == "crux_rcpt_abc123"

    def test_crux_count_still_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARAGORA_CRUX_RECEIPT_ENABLED", raising=False)
        assert crux_receipt_to_knowledge_items(_receipt([_entry(), _entry("c2")])).crux_count == 2


# ---------------------------------------------------------------------------
# Flag-on, empty cruxes → still empty
# ---------------------------------------------------------------------------


def test_empty_cruxes_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARAGORA_CRUX_RECEIPT_ENABLED", "1")
    r = crux_receipt_to_knowledge_items(_receipt(cruxes=[]))
    assert r.items == [] and r.success is False and r.crux_count == 0


# ---------------------------------------------------------------------------
# Item shape
# ---------------------------------------------------------------------------


class TestItemShape:
    @pytest.fixture(autouse=True)
    def _on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_CRUX_RECEIPT_ENABLED", "1")

    def test_one_crux_one_item(self) -> None:
        assert len(crux_receipt_to_knowledge_items(_receipt()).items) == 1

    def test_two_cruxes_two_items(self) -> None:
        assert (
            len(crux_receipt_to_knowledge_items(_receipt([_entry("c1"), _entry("c2")])).items) == 2
        )

    def test_item_id_deterministic(self) -> None:
        assert crux_receipt_to_knowledge_items(_receipt()).items[0].id == "crux_km_crux.t1"

    def test_item_content(self) -> None:
        assert (
            crux_receipt_to_knowledge_items(_receipt()).items[0].content
            == "The benchmark is fresh."
        )

    def test_item_source_is_belief(self) -> None:
        assert crux_receipt_to_knowledge_items(_receipt()).items[0].source == KnowledgeSource.BELIEF

    def test_item_source_id(self) -> None:
        assert crux_receipt_to_knowledge_items(_receipt()).items[0].source_id == "crux.t1"

    def test_importance(self) -> None:
        assert crux_receipt_to_knowledge_items(_receipt()).items[0].importance == pytest.approx(
            0.80, abs=1e-4
        )


# ---------------------------------------------------------------------------
# Confidence mapping
# ---------------------------------------------------------------------------


class TestConfidence:
    @pytest.fixture(autouse=True)
    def _on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_CRUX_RECEIPT_ENABLED", "1")

    def _conf(self, score: float) -> Any:
        return crux_receipt_to_knowledge_items(_receipt([_entry(lbs=score)])).items[0].confidence

    def test_high(self) -> None:
        assert self._conf(0.85) == ConfidenceLevel.HIGH

    def test_boundary_high(self) -> None:
        assert self._conf(0.75) == ConfidenceLevel.HIGH

    def test_medium(self) -> None:
        assert self._conf(0.60) == ConfidenceLevel.MEDIUM

    def test_low(self) -> None:
        assert self._conf(0.20) == ConfidenceLevel.LOW


# ---------------------------------------------------------------------------
# Metadata provenance
# ---------------------------------------------------------------------------


class TestMetadata:
    @pytest.fixture(autouse=True)
    def _on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_CRUX_RECEIPT_ENABLED", "1")

    def _m(self) -> dict[str, Any]:
        return crux_receipt_to_knowledge_items(_receipt()).items[0].metadata

    def test_receipt_id(self) -> None:
        assert self._m()["receipt_id"] == "crux_rcpt_abc123"

    def test_debate_id(self) -> None:
        assert self._m()["debate_id"] == "debate_xyz"

    def test_affected_claims(self) -> None:
        assert self._m()["affected_claims"] == ["claim.a", "claim.b"]

    def test_dic_track(self) -> None:
        assert self._m()["dic_track"] == "DIC-16"
