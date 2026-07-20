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
# Windows are half-open [now - window, now): a record stamped exactly `now`
# is excluded, so test merges use a realistic slightly-earlier timestamp.
RECENT = NOW - timedelta(hours=1)


def _merge(ledger: ThroughputLedger, paths, *, title="", labels=(), when=RECENT, pr="1"):
    work = classify_paths(paths, identifier=pr, title=title, labels=labels)
    return ledger.record_merge(work, title=title, when=when)


class TestLedgerRecord:
    def test_round_trip(self):
        record = LedgerRecord(kind="merge", timestamp=NOW.isoformat(), data={"a": 1})
        assert LedgerRecord.from_json(record.to_json()) == record

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="unknown ledger record kind"):
            LedgerRecord(kind="bogus", timestamp=NOW.isoformat())

    def test_from_json_rejects_non_object_payloads(self):
        with pytest.raises(TypeError, match="ledger record must be a JSON object"):
            LedgerRecord.from_json("[]")

        with pytest.raises(TypeError, match="ledger record must be a JSON object"):
            LedgerRecord.from_json("1")


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

    def test_corrupt_line_skipped_not_fatal(self, tmp_path):
        # A write truncated by a killed process must not brick the ledger.
        ledger = ThroughputLedger(tmp_path)
        _merge(ledger, ["aragora/a.py"], pr="1")
        with ledger.path.open("a", encoding="utf-8") as handle:
            handle.write('{"kind": "merge", "timesta\n')  # truncated write
            handle.write("not json at all\n")
            handle.write("[]\n")
            handle.write("1\n")
            handle.write('{"kind": "artifact", "timestamp": "2026-07-08T00:00:00", "data": {}}\n')
            handle.write('{"kind": "bogus-kind", "timestamp": "2026-07-08T00:00:00+00:00"}\n')
            # non-dict data payload passes timestamp validation but must be
            # rejected as corrupt, not crash consumers (#9047 openai [P2])
            handle.write(
                '{"kind": "merge", "timestamp": "2026-07-08T00:00:00+00:00", "data": []}\n'
            )
        _merge(ledger, ["aragora/b.py"], pr="2")
        records = ledger.records()
        assert [r.data["identifier"] for r in records] == ["1", "2"]


class TestComputeMetrics:
    def _build(self, tmp_path) -> ThroughputLedger:
        ledger = ThroughputLedger(tmp_path)
        _merge(ledger, ["aragora/debate/a.py"], title="feat: consensus", pr="1")
        _merge(ledger, ["docs/artifacts/a.md"], title="docs: artifact", pr="2")
        _merge(ledger, ["scripts/gate.py"], title="fix: gate hang", pr="3")
        _merge(ledger, ["scripts/old.py"], title="feat: old", pr="4", when=NOW - timedelta(days=30))
        ledger.record_artifact("verifier-release", when=RECENT)
        ledger.append(LedgerRecord(kind="park", timestamp=RECENT.isoformat(), data={"pr": "5"}))
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

    def test_naive_timestamp_records_are_ignored(self, tmp_path, caplog):
        ledger = ThroughputLedger(tmp_path)
        _merge(ledger, ["aragora/a.py"], pr="good")
        ledger.append(
            LedgerRecord(
                kind="artifact",
                timestamp="2026-07-08T11:00:00",
                data={"name": "naive"},
            )
        )

        with caplog.at_level("WARNING", logger="aragora.nomic.throughput"):
            metrics = compute_metrics(ledger.records(), now=NOW)

        assert metrics.merges_total == 1
        assert metrics.external_artifacts == 0
        assert "corrupt ledger line" in caplog.text

    def test_compute_metrics_skips_direct_naive_timestamp_records(self, tmp_path, caplog):
        ledger = ThroughputLedger(tmp_path)
        good = _merge(ledger, ["aragora/a.py"], pr="good")
        naive = LedgerRecord(
            kind="artifact",
            timestamp="2026-07-08T11:00:00",
            data={"name": "naive"},
        )

        with caplog.at_level("WARNING", logger="aragora.nomic.throughput"):
            metrics = compute_metrics([good, naive], now=NOW)

        assert metrics.merges_total == 1
        assert metrics.external_artifacts == 0
        assert "invalid timestamp" in caplog.text

    def test_previous_window_excludes_current_records(self, tmp_path):
        # WoW regression: computing metrics with now=NOW-7d must NOT absorb
        # records from the current week (window is bounded on both ends).
        ledger = ThroughputLedger(tmp_path)
        _merge(ledger, ["scripts/old.py"], pr="prev1", when=NOW - timedelta(days=10))
        _merge(ledger, ["aragora/new.py"], pr="cur1", when=NOW - timedelta(days=1))
        previous = compute_metrics(ledger.records(), now=NOW - timedelta(days=7))
        assert previous.merges_total == 1
        assert previous.substrate_share == 1.0  # prior week was 100% substrate
        current = compute_metrics(ledger.records(), now=NOW)
        assert current.merges_total == 1
        assert current.product_share == 1.0

    def test_exempt_counted_in_metrics_not_mix(self, tmp_path):
        ledger = ThroughputLedger(tmp_path)
        _merge(ledger, ["scripts/x.py"], title="Revert broken gate", pr="1")
        _merge(ledger, ["aragora/a.py"], pr="2")
        metrics = compute_metrics(ledger.records(), now=NOW)
        assert metrics.exempt_merges == 1
        mix = mix_from_records(ledger.records(), now=NOW)
        assert mix.total == 1  # exempt merge excluded from budget math

    def test_semantically_invalid_merge_records_are_skipped(self, tmp_path, caplog):
        ledger = ThroughputLedger(tmp_path)
        _merge(ledger, ["aragora/a.py"], pr="good")
        ledger.append(
            LedgerRecord(
                kind="merge",
                timestamp=RECENT.isoformat(),
                data={
                    "identifier": "bad-class",
                    "work_class": "not-a-work-class",
                    "file_counts": {},
                },
            )
        )
        ledger.append(
            LedgerRecord(
                kind="merge",
                timestamp=RECENT.isoformat(),
                data={
                    "identifier": "bad-file-counts",
                    "work_class": "substrate",
                    "file_counts": ["not", "a", "mapping"],
                },
            )
        )
        ledger.append(
            LedgerRecord(
                kind="merge",
                timestamp=RECENT.isoformat(),
                data={
                    "identifier": "bad-count-value",
                    "work_class": "substrate",
                    "file_counts": {"substrate": "many"},
                },
            )
        )

        with caplog.at_level("WARNING", logger="aragora.nomic.throughput"):
            metrics = compute_metrics(ledger.records(), now=NOW)
            mix = mix_from_records(ledger.records(), now=NOW)

        assert metrics.merges_total == 1
        assert metrics.product_share == 1.0
        assert mix.total == 1
        assert "semantically invalid merge ledger record" in caplog.text


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

    def test_digest_budget_uses_countable_non_exempt_denominator(self, tmp_path):
        ledger = ThroughputLedger(tmp_path)
        _merge(ledger, ["aragora/debate/a.py"], pr="product")
        for index in range(3):
            _merge(
                ledger,
                [f"scripts/security_fix_{index}.py"],
                labels=["security"],
                pr=f"exempt-{index}",
            )

        metrics = compute_metrics(ledger.records(), now=NOW)
        digest = render_digest(metrics)

        assert metrics.merges_total == 4
        assert metrics.exempt_merges == 3
        assert metrics.product_share == 1.0
        assert "Product share | 100%" in digest
        assert "Exempt merges (excluded from mix) | 3" in digest
        assert "**Budget verdict:** OK" in digest
