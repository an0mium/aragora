"""Tests for aragora.nomic.throughput (epic #9039, issue #9040)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aragora.nomic.throughput import (
    LedgerRecord,
    ThroughputLedger,
    clear_freeze_marker,
    compute_metrics,
    freeze_active,
    freeze_marker_path,
    mix_from_records,
    render_digest,
    write_freeze_marker,
)
from aragora.nomic.work_mix import WorkClass, classify_paths

NOW = datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)


def _merge(ledger: ThroughputLedger, paths, *, title="", labels=(), when=NOW, pr="1"):
    work = classify_paths(paths, identifier=pr, title=title, labels=labels)
    return ledger.record_merge(work, title=title, when=when)


class TestLedgerRecord:
    def test_round_trip(self):
        record = LedgerRecord(kind="merge", timestamp=NOW.isoformat(), data={"a": 1})
        assert LedgerRecord.from_json(record.to_json()) == record

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="unknown ledger record kind"):
            LedgerRecord(kind="bogus", timestamp=NOW.isoformat())


class TestThroughputLedger:
    def test_append_and_read(self, tmp_path):
        ledger = ThroughputLedger(tmp_path)
        _merge(ledger, ["aragora/debate/a.py"], title="feat: x", pr="42")
        ledger.record_artifact("dogfood-report", url="https://example.com", when=NOW)
        records = ledger.records()
        assert [r.kind for r in records] == ["merge", "artifact"]
        assert records[0].data["identifier"] == "42"
        assert records[0].data["work_class"] == "product-core"

    def test_since_filter(self, tmp_path):
        ledger = ThroughputLedger(tmp_path)
        _merge(ledger, ["aragora/a.py"], when=NOW - timedelta(days=10))
        _merge(ledger, ["aragora/b.py"], when=NOW)
        recent = ledger.records(since=NOW - timedelta(days=7))
        assert len(recent) == 1

    def test_missing_file_returns_empty(self, tmp_path):
        assert ThroughputLedger(tmp_path).records() == []


class TestComputeMetrics:
    def _build(self, tmp_path) -> ThroughputLedger:
        ledger = ThroughputLedger(tmp_path)
        _merge(ledger, ["aragora/debate/a.py"], title="feat: consensus", pr="1")
        _merge(ledger, ["docs/artifacts/a.md"], title="docs: artifact", pr="2")
        _merge(ledger, ["scripts/gate.py"], title="fix: gate hang", pr="3")
        _merge(ledger, ["scripts/old.py"], title="feat: old", pr="4", when=NOW - timedelta(days=30))
        ledger.record_artifact("verifier-release", when=NOW)
        ledger.append(LedgerRecord(kind="park", timestamp=NOW.isoformat(), data={"pr": "5"}))
        return ledger

    def test_window_and_shares(self, tmp_path):
        metrics = compute_metrics(self._build(tmp_path).records(), now=NOW)
        assert metrics.merges_total == 3  # 30-day-old merge excluded
        assert metrics.product_share == pytest.approx(2 / 3)
        assert metrics.substrate_share == pytest.approx(1 / 3)
        assert metrics.external_artifacts == 1
        assert metrics.parks == 1
        assert metrics.reverts == 0

    def test_self_repair_ratio(self, tmp_path):
        metrics = compute_metrics(self._build(tmp_path).records(), now=NOW)
        # one substrate merge with fix-shaped title out of three merges
        assert metrics.self_repair_ratio == pytest.approx(1 / 3)

    def test_mix_from_records_matches(self, tmp_path):
        mix = mix_from_records(self._build(tmp_path).records(), now=NOW)
        assert mix.total == 3
        assert mix.counts[WorkClass.SUBSTRATE] == 1

    def test_empty_records(self):
        metrics = compute_metrics([], now=NOW)
        assert metrics.merges_total == 0
        assert metrics.product_share == 0.0
        assert metrics.self_repair_ratio == 0.0


class TestFreezeMarker:
    def test_lifecycle(self, tmp_path):
        assert freeze_active(tmp_path) is None
        path = write_freeze_marker(tmp_path, reason="substrate 40% > 25%")
        assert path == freeze_marker_path(tmp_path)
        payload = freeze_active(tmp_path)
        assert payload is not None
        assert payload["reason"] == "substrate 40% > 25%"
        assert clear_freeze_marker(tmp_path)
        assert freeze_active(tmp_path) is None
        assert not clear_freeze_marker(tmp_path)

    def test_corrupt_marker_still_reports_active(self, tmp_path):
        path = write_freeze_marker(tmp_path, reason="x")
        path.write_text("not json", encoding="utf-8")
        assert freeze_active(tmp_path) == {}


class TestRenderDigest:
    def test_digest_contains_key_sections(self, tmp_path):
        ledger = ThroughputLedger(tmp_path)
        _merge(ledger, ["aragora/debate/a.py"], title="feat: x", pr="1")
        _merge(ledger, ["scripts/gate.py"], title="fix: y", pr="2")
        metrics = compute_metrics(ledger.records(), now=NOW)
        digest = render_digest(
            metrics,
            freeze={"timestamp": NOW.isoformat(), "reason": "test"},
            samples=[{"identifier": "1", "work_class": "product-core", "title": "feat: x"}],
        )
        assert "# Weekly throughput digest" in digest
        assert "Product share | 50%" in digest
        assert "ACTIVE" in digest
        assert "Spot-audit sample" in digest
        assert "- 1: product-core" in digest

    def test_digest_inactive_freeze_and_wow_delta(self, tmp_path):
        ledger = ThroughputLedger(tmp_path)
        _merge(ledger, ["aragora/debate/a.py"], pr="1")
        current = compute_metrics(ledger.records(), now=NOW)
        previous = compute_metrics([], now=NOW - timedelta(days=7))
        digest = render_digest(current, previous=previous)
        assert "inactive" in digest
        assert "WoW" in digest
