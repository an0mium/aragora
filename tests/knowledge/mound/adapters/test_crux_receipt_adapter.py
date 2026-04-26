"""Tests for DIC-16 CruxReceiptAdapter (KM ingestion of CruxReceipt objects)."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.epistemic.crux_receipt import CruxEntry, CruxReceipt
from aragora.knowledge.mound.adapters.crux_receipt_adapter import (
    CruxIngestionResult,
    CruxReceiptAdapter,
    _stable_id,
)
from aragora.knowledge.unified.types import ConfidenceLevel, KnowledgeSource


def _entry(crux_id: str = "crux-001", score: float = 0.80) -> CruxEntry:
    return CruxEntry(
        crux_id=crux_id,
        statement="X depends on Y",
        load_bearing_score=score,
        uncertainty_score=0.20,
        contesting_agents=["alice"],
        affected_claims=["claim-A"],
        resolution_impact=0.60,
    )


def _receipt(cruxes: list[CruxEntry] | None = None, receipt_id: str = "rcpt-abc") -> CruxReceipt:
    return CruxReceipt(
        receipt_id=receipt_id,
        debate_id="debate-001",
        question="Q?",
        cruxes=cruxes if cruxes is not None else [_entry()],
        convergence_barrier=0.45,
        counterfactuals=[],
        agents=["alice"],
        rounds=3,
        metadata={},
        checksum="a" * 64,
    )


class TestCruxIngestionResult:
    def test_success_and_failure(self) -> None:
        assert CruxIngestionResult("r", 1, ["a"]).success is True
        assert CruxIngestionResult("r", 0, []).success is False
        assert CruxIngestionResult("r", 1, ["a"], errors=["e"]).success is False

    def test_to_dict(self) -> None:
        d = CruxIngestionResult("r", 1, ["a"]).to_dict()
        assert d["cruxes_ingested"] == 1 and d["success"] is True


class TestStableId:
    def test_deterministic_and_hex(self) -> None:
        r = _stable_id("x", "y")
        assert len(r) == 16 and all(c in "0123456789abcdef" for c in r)
        assert _stable_id("a", "b") == _stable_id("a", "b")
        assert _stable_id("a", "b") != _stable_id("b", "a")


class TestCruxToKnowledgeItem:
    def setup_method(self) -> None:
        self.adapter = CruxReceiptAdapter()
        self.now = datetime.now(UTC)

    def _item(self, score: float = 0.80) -> object:
        e = _entry(score=score)
        return self.adapter._crux_to_knowledge_item(e, _receipt(cruxes=[e]), self.now)

    def test_source_belief(self) -> None:
        assert self._item().source == KnowledgeSource.BELIEF

    def test_id_prefix_stable(self) -> None:
        assert self._item().id.startswith("crux_km_")
        assert self._item().id == self._item().id

    def test_importance_and_confidence(self) -> None:
        item = self._item(0.80)
        assert abs(item.importance - 0.80) < 1e-9
        assert item.confidence == ConfidenceLevel.HIGH

    def test_metadata_provenance(self) -> None:
        m = self._item().metadata
        assert m["receipt_id"] == "rcpt-abc"
        assert m["checksum"] == "a" * 64
        assert "DIC-16" in m["dic_issue"]

    def test_cross_references(self) -> None:
        assert "claim-A" in self._item().cross_references


class TestFlagGating:
    def test_skips_when_flag_off(self) -> None:
        os.environ.pop("ARAGORA_CRUX_RECEIPT_ENABLED", None)
        r = asyncio.run(CruxReceiptAdapter().ingest_crux_receipt(_receipt(), require_enabled=True))
        assert r.cruxes_ingested == 0 and r.skipped == 1

    def test_bypasses_when_require_false(self) -> None:
        os.environ.pop("ARAGORA_CRUX_RECEIPT_ENABLED", None)
        r = asyncio.run(CruxReceiptAdapter().ingest_crux_receipt(_receipt(), require_enabled=False))
        assert r.skipped == 0

    def test_proceeds_when_flag_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_CRUX_RECEIPT_ENABLED", "1")
        r = asyncio.run(CruxReceiptAdapter().ingest_crux_receipt(_receipt(), require_enabled=True))
        assert r.skipped == 0


class TestIngestWithMound:
    def _adapter(self) -> tuple[CruxReceiptAdapter, MagicMock]:
        mound = MagicMock()
        mound.store = AsyncMock(return_value="stored-id")
        return CruxReceiptAdapter(mound=mound), mound

    def test_stores_each_crux(self) -> None:
        adapter, mound = self._adapter()
        r = asyncio.run(
            adapter.ingest_crux_receipt(
                _receipt(cruxes=[_entry("c1"), _entry("c2")]), require_enabled=False
            )
        )
        assert r.cruxes_ingested == 2 and mound.store.call_count == 2

    def test_empty_cruxes(self) -> None:
        adapter, mound = self._adapter()
        r = asyncio.run(adapter.ingest_crux_receipt(_receipt(cruxes=[]), require_enabled=False))
        assert r.cruxes_ingested == 0
        mound.store.assert_not_called()

    def test_error_captured(self) -> None:
        mound = MagicMock()
        mound.store = AsyncMock(side_effect=RuntimeError("db down"))
        r = asyncio.run(
            CruxReceiptAdapter(mound=mound).ingest_crux_receipt(_receipt(), require_enabled=False)
        )
        assert r.cruxes_ingested == 0 and any("crux-001" in e for e in r.errors)

    def test_set_mound(self) -> None:
        adapter = CruxReceiptAdapter()
        mound = MagicMock()
        mound.store = AsyncMock(return_value="x")
        adapter.set_mound(mound)
        r = asyncio.run(adapter.ingest_crux_receipt(_receipt(), require_enabled=False))
        assert r.cruxes_ingested == 1

    def test_fallback_ingest(self) -> None:
        mound = MagicMock(spec=["ingest"])
        mound.ingest = AsyncMock()
        r = asyncio.run(
            CruxReceiptAdapter(mound=mound).ingest_crux_receipt(_receipt(), require_enabled=False)
        )
        mound.ingest.assert_called_once()
        assert len(r.knowledge_item_ids) == 1
