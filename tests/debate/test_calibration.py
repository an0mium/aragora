"""Tests for aragora.debate.calibration -- prediction accuracy tracking.

Tests cover:
- CalibrationRecord creation and serialization
- InMemoryCalibrationStore and JsonFileCalibrationStore
- CalibrationTracker outcome recording, calibration curves, Brier scores
- Edge cases: empty data, clamped confidence, multiple agents/domains
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aragora.debate.calibration import (
    CalibrationRecord,
    CalibrationTracker,
    InMemoryCalibrationStore,
    JsonFileCalibrationStore,
)


# ---------------------------------------------------------------------------
# CalibrationRecord
# ---------------------------------------------------------------------------


class TestCalibrationRecord:
    def test_defaults(self):
        rec = CalibrationRecord()
        assert rec.record_id.startswith("cal-")
        assert rec.debate_id == ""
        assert rec.agent_id == ""
        assert rec.predicted_confidence == 0.5
        assert rec.actual_outcome is False
        assert rec.domain == "general"
        assert rec.recorded_at != ""
        assert rec.metadata == {}

    def test_custom_fields(self):
        rec = CalibrationRecord(
            debate_id="d1",
            agent_id="claude",
            predicted_confidence=0.9,
            actual_outcome=True,
            domain="performance",
            metadata={"source": "settlement"},
        )
        assert rec.debate_id == "d1"
        assert rec.agent_id == "claude"
        assert rec.predicted_confidence == 0.9
        assert rec.actual_outcome is True
        assert rec.domain == "performance"

    def test_to_dict(self):
        rec = CalibrationRecord(
            debate_id="d1",
            agent_id="claude",
            predicted_confidence=0.8,
            actual_outcome=True,
        )
        d = rec.to_dict()
        assert d["debate_id"] == "d1"
        assert d["agent_id"] == "claude"
        assert d["predicted_confidence"] == 0.8
        assert d["actual_outcome"] is True

    def test_from_dict_roundtrip(self):
        rec = CalibrationRecord(
            record_id="cal-test123",
            debate_id="d1",
            agent_id="gpt4",
            predicted_confidence=0.7,
            actual_outcome=False,
            domain="security",
            metadata={"key": "value"},
        )
        d = rec.to_dict()
        restored = CalibrationRecord.from_dict(d)
        assert restored.record_id == rec.record_id
        assert restored.debate_id == rec.debate_id
        assert restored.agent_id == rec.agent_id
        assert restored.predicted_confidence == rec.predicted_confidence
        assert restored.actual_outcome == rec.actual_outcome
        assert restored.domain == rec.domain

    def test_to_dict_json_serializable(self):
        rec = CalibrationRecord(debate_id="d1")
        serialized = json.dumps(rec.to_dict())
        assert "d1" in serialized


# ---------------------------------------------------------------------------
# InMemoryCalibrationStore
# ---------------------------------------------------------------------------


class TestInMemoryCalibrationStore:
    def test_save_and_list(self):
        store = InMemoryCalibrationStore()
        store.save(CalibrationRecord(debate_id="d1", agent_id="a"))
        store.save(CalibrationRecord(debate_id="d2", agent_id="b"))
        assert len(store.list_records()) == 2

    def test_filter_by_agent(self):
        store = InMemoryCalibrationStore()
        store.save(CalibrationRecord(agent_id="alice"))
        store.save(CalibrationRecord(agent_id="bob"))
        store.save(CalibrationRecord(agent_id="alice"))
        assert len(store.list_records(agent_id="alice")) == 2
        assert len(store.list_records(agent_id="bob")) == 1

    def test_filter_by_domain(self):
        store = InMemoryCalibrationStore()
        store.save(CalibrationRecord(domain="perf"))
        store.save(CalibrationRecord(domain="security"))
        assert len(store.list_records(domain="perf")) == 1

    def test_filter_by_debate_id(self):
        store = InMemoryCalibrationStore()
        store.save(CalibrationRecord(debate_id="d1"))
        store.save(CalibrationRecord(debate_id="d2"))
        store.save(CalibrationRecord(debate_id="d1"))
        assert len(store.list_records(debate_id="d1")) == 2

    def test_multiple_filters(self):
        store = InMemoryCalibrationStore()
        store.save(CalibrationRecord(agent_id="a", domain="perf"))
        store.save(CalibrationRecord(agent_id="a", domain="security"))
        store.save(CalibrationRecord(agent_id="b", domain="perf"))
        assert len(store.list_records(agent_id="a", domain="perf")) == 1

    def test_empty_store(self):
        store = InMemoryCalibrationStore()
        assert len(store.list_records()) == 0


# ---------------------------------------------------------------------------
# JsonFileCalibrationStore
# ---------------------------------------------------------------------------


class TestJsonFileCalibrationStore:
    def test_creates_directory(self, tmp_path: Path):
        store = JsonFileCalibrationStore(tmp_path)
        assert (tmp_path / "calibration").is_dir()

    def test_persistence_roundtrip(self, tmp_path: Path):
        store = JsonFileCalibrationStore(tmp_path)
        store.save(
            CalibrationRecord(
                record_id="cal-1",
                debate_id="d1",
                agent_id="claude",
                predicted_confidence=0.9,
                actual_outcome=True,
            )
        )
        store.save(
            CalibrationRecord(
                record_id="cal-2",
                debate_id="d2",
                agent_id="gpt4",
                predicted_confidence=0.3,
                actual_outcome=False,
            )
        )

        # Load into a new store
        store2 = JsonFileCalibrationStore(tmp_path)
        records = store2.list_records()
        assert len(records) == 2
        assert records[0].record_id == "cal-1"
        assert records[1].record_id == "cal-2"

    def test_corrupt_json_handled(self, tmp_path: Path):
        cal_dir = tmp_path / "calibration"
        cal_dir.mkdir()
        (cal_dir / "records.json").write_text("not valid json!!!")

        store = JsonFileCalibrationStore(tmp_path)
        assert len(store.list_records()) == 0

    def test_filter_after_load(self, tmp_path: Path):
        store = JsonFileCalibrationStore(tmp_path)
        store.save(CalibrationRecord(agent_id="alice"))
        store.save(CalibrationRecord(agent_id="bob"))

        store2 = JsonFileCalibrationStore(tmp_path)
        assert len(store2.list_records(agent_id="alice")) == 1


# ---------------------------------------------------------------------------
# CalibrationTracker: record_outcome
# ---------------------------------------------------------------------------


class TestRecordOutcome:
    def test_basic_record(self):
        tracker = CalibrationTracker()
        rec = tracker.record_outcome("d1", predicted_confidence=0.8, actual_outcome=True)
        assert rec.debate_id == "d1"
        assert rec.predicted_confidence == 0.8
        assert rec.actual_outcome is True
        assert rec.record_id.startswith("cal-")

    def test_record_with_agent(self):
        tracker = CalibrationTracker()
        rec = tracker.record_outcome(
            "d1",
            predicted_confidence=0.7,
            actual_outcome=False,
            agent_id="claude",
        )
        assert rec.agent_id == "claude"

    def test_record_with_domain(self):
        tracker = CalibrationTracker()
        rec = tracker.record_outcome(
            "d1",
            predicted_confidence=0.5,
            actual_outcome=True,
            domain="security",
        )
        assert rec.domain == "security"

    def test_record_with_metadata(self):
        tracker = CalibrationTracker()
        rec = tracker.record_outcome(
            "d1",
            predicted_confidence=0.6,
            actual_outcome=True,
            metadata={"source": "settlement"},
        )
        assert rec.metadata["source"] == "settlement"

    def test_clamps_confidence_high(self):
        tracker = CalibrationTracker()
        rec = tracker.record_outcome("d1", predicted_confidence=1.5, actual_outcome=True)
        assert rec.predicted_confidence == 1.0

    def test_clamps_confidence_low(self):
        tracker = CalibrationTracker()
        rec = tracker.record_outcome("d1", predicted_confidence=-0.3, actual_outcome=False)
        assert rec.predicted_confidence == 0.0


# ---------------------------------------------------------------------------
# CalibrationTracker: get_calibration_curve
# ---------------------------------------------------------------------------


class TestGetCalibrationCurve:
    def test_empty_curve(self):
        tracker = CalibrationTracker()
        curve = tracker.get_calibration_curve()
        assert curve["total_records"] == 0
        assert len(curve["buckets"]) == 10  # 10 buckets

    def test_well_calibrated(self):
        """Agent at 80% confidence, 8/10 correct -> well calibrated."""
        tracker = CalibrationTracker()
        for i in range(10):
            tracker.record_outcome(
                f"d{i}",
                predicted_confidence=0.8,
                actual_outcome=(i < 8),
                agent_id="claude",
            )

        curve = tracker.get_calibration_curve()
        assert curve["total_records"] == 10

        # Find the 70-80% bucket (0.8 falls into 70-80% bucket since 80 < 80 is false,
        # it goes to 80-90%)
        # Actually _bucket_label(0.8): pct=80, checking buckets:
        # 0-10: 80 < 10? no
        # ...
        # 70-80: 80 < 80? no
        # 80-90: 80 < 90? yes -> "80-90%"
        bucket_80 = None
        for b in curve["buckets"]:
            if b["bucket"] == "80-90%":
                bucket_80 = b
                break
        assert bucket_80 is not None
        assert bucket_80["count"] == 10
        assert bucket_80["predicted"] == pytest.approx(0.8)
        assert bucket_80["actual"] == pytest.approx(0.8)

    def test_multiple_buckets(self):
        tracker = CalibrationTracker()
        # Low confidence
        for i in range(5):
            tracker.record_outcome(f"low-{i}", predicted_confidence=0.15, actual_outcome=False)
        # High confidence
        for i in range(5):
            tracker.record_outcome(f"high-{i}", predicted_confidence=0.95, actual_outcome=True)

        curve = tracker.get_calibration_curve()
        assert curve["total_records"] == 10

        # Low confidence in 10-20% bucket
        low_bucket = next(b for b in curve["buckets"] if b["bucket"] == "10-20%")
        assert low_bucket["count"] == 5
        assert low_bucket["actual"] == pytest.approx(0.0)

        # High confidence in 90-100% bucket
        high_bucket = next(b for b in curve["buckets"] if b["bucket"] == "90-100%")
        assert high_bucket["count"] == 5
        assert high_bucket["actual"] == pytest.approx(1.0)

    def test_filter_by_agent(self):
        tracker = CalibrationTracker()
        for i in range(5):
            tracker.record_outcome(
                f"d{i}", predicted_confidence=0.8, actual_outcome=True, agent_id="alice"
            )
        for i in range(5):
            tracker.record_outcome(
                f"d{i + 5}", predicted_confidence=0.3, actual_outcome=False, agent_id="bob"
            )

        curve = tracker.get_calibration_curve(agent_id="alice")
        assert curve["total_records"] == 5

    def test_filter_by_domain(self):
        tracker = CalibrationTracker()
        tracker.record_outcome("d1", 0.8, True, domain="perf")
        tracker.record_outcome("d2", 0.7, False, domain="security")

        curve = tracker.get_calibration_curve(domain="perf")
        assert curve["total_records"] == 1


# ---------------------------------------------------------------------------
# CalibrationTracker: get_brier_score
# ---------------------------------------------------------------------------


class TestGetBrierScore:
    def test_empty(self):
        tracker = CalibrationTracker()
        assert tracker.get_brier_score() == 0.0

    def test_perfect_calibration(self):
        """100% confidence, all correct -> Brier = 0."""
        tracker = CalibrationTracker()
        for i in range(5):
            tracker.record_outcome(f"d{i}", predicted_confidence=1.0, actual_outcome=True)

        assert tracker.get_brier_score() == pytest.approx(0.0)

    def test_worst_calibration(self):
        """100% confidence, all wrong -> Brier = 1."""
        tracker = CalibrationTracker()
        for i in range(5):
            tracker.record_outcome(f"d{i}", predicted_confidence=1.0, actual_outcome=False)

        assert tracker.get_brier_score() == pytest.approx(1.0)

    def test_well_calibrated(self):
        """80% confidence, 8/10 correct -> Brier = 0.04."""
        tracker = CalibrationTracker()
        for i in range(10):
            tracker.record_outcome(f"d{i}", predicted_confidence=0.8, actual_outcome=(i < 8))

        # Brier: 8*(0.8-1.0)^2 + 2*(0.8-0.0)^2 = 8*0.04 + 2*0.64 = 0.32 + 1.28 = 1.60
        # Mean: 1.60/10 = 0.16
        assert tracker.get_brier_score() == pytest.approx(0.16)

    def test_filter_by_agent(self):
        tracker = CalibrationTracker()
        # Good agent
        for i in range(5):
            tracker.record_outcome(
                f"d{i}", predicted_confidence=0.9, actual_outcome=True, agent_id="good"
            )
        # Bad agent
        for i in range(5):
            tracker.record_outcome(
                f"d{i + 5}", predicted_confidence=0.9, actual_outcome=False, agent_id="bad"
            )

        good_brier = tracker.get_brier_score(agent_id="good")
        bad_brier = tracker.get_brier_score(agent_id="bad")
        assert good_brier < bad_brier
        assert good_brier == pytest.approx(0.01)  # (0.9 - 1.0)^2 = 0.01
        assert bad_brier == pytest.approx(0.81)  # (0.9 - 0.0)^2 = 0.81

    def test_filter_by_domain(self):
        tracker = CalibrationTracker()
        tracker.record_outcome("d1", 1.0, True, domain="perf")
        tracker.record_outcome("d2", 1.0, False, domain="security")

        assert tracker.get_brier_score(domain="perf") == pytest.approx(0.0)
        assert tracker.get_brier_score(domain="security") == pytest.approx(1.0)

    def test_50_50_baseline(self):
        """50% confidence -> Brier = 0.25 regardless of outcome."""
        tracker = CalibrationTracker()
        tracker.record_outcome("d1", 0.5, True)
        tracker.record_outcome("d2", 0.5, False)

        # (0.5-1.0)^2 + (0.5-0.0)^2 = 0.25 + 0.25 = 0.50 / 2 = 0.25
        assert tracker.get_brier_score() == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# CalibrationTracker: get_agent_scores
# ---------------------------------------------------------------------------


class TestGetAgentScores:
    def test_empty(self):
        tracker = CalibrationTracker()
        assert tracker.get_agent_scores() == {}

    def test_multiple_agents(self):
        tracker = CalibrationTracker()
        # Perfect agent
        for i in range(5):
            tracker.record_outcome(
                f"d{i}", predicted_confidence=1.0, actual_outcome=True, agent_id="perfect"
            )
        # Bad agent
        for i in range(5):
            tracker.record_outcome(
                f"d{i + 5}", predicted_confidence=1.0, actual_outcome=False, agent_id="bad"
            )

        scores = tracker.get_agent_scores()
        assert "perfect" in scores
        assert "bad" in scores
        assert scores["perfect"] == pytest.approx(0.0)
        assert scores["bad"] == pytest.approx(1.0)

    def test_ignores_no_agent_records(self):
        tracker = CalibrationTracker()
        tracker.record_outcome("d1", 0.5, True)  # No agent_id
        tracker.record_outcome("d2", 0.5, True, agent_id="alice")

        scores = tracker.get_agent_scores()
        assert "alice" in scores
        assert "" not in scores  # Empty agent_id excluded


# ---------------------------------------------------------------------------
# CalibrationTracker: get_summary
# ---------------------------------------------------------------------------


class TestGetSummary:
    def test_empty_summary(self):
        tracker = CalibrationTracker()
        summary = tracker.get_summary()
        assert summary["total_records"] == 0
        assert summary["correct"] == 0
        assert summary["incorrect"] == 0
        assert summary["accuracy"] == 0.0
        assert summary["brier_score"] == 0.0

    def test_populated_summary(self):
        tracker = CalibrationTracker()
        for i in range(10):
            tracker.record_outcome(
                f"d{i}",
                predicted_confidence=0.8,
                actual_outcome=(i < 8),
                agent_id="claude",
            )

        summary = tracker.get_summary()
        assert summary["total_records"] == 10
        assert summary["correct"] == 8
        assert summary["incorrect"] == 2
        assert summary["accuracy"] == pytest.approx(0.8)
        assert summary["brier_score"] == pytest.approx(0.16)
        assert "per_agent" in summary
        assert "claude" in summary["per_agent"]

    def test_summary_with_agent_filter(self):
        tracker = CalibrationTracker()
        tracker.record_outcome("d1", 0.9, True, agent_id="alice")
        tracker.record_outcome("d2", 0.9, False, agent_id="bob")

        summary = tracker.get_summary(agent_id="alice")
        assert summary["total_records"] == 1
        assert summary["correct"] == 1
        # No per_agent when filtered by agent
        assert "per_agent" not in summary


# ---------------------------------------------------------------------------
# CalibrationTracker with persistent store
# ---------------------------------------------------------------------------


class TestCalibrationTrackerPersistence:
    def test_with_json_store(self, tmp_path: Path):
        store = JsonFileCalibrationStore(tmp_path)
        tracker = CalibrationTracker(store=store)

        tracker.record_outcome("d1", 0.9, True, agent_id="claude")
        tracker.record_outcome("d2", 0.3, False, agent_id="gpt4")

        # Load in a new tracker
        store2 = JsonFileCalibrationStore(tmp_path)
        tracker2 = CalibrationTracker(store=store2)

        summary = tracker2.get_summary()
        assert summary["total_records"] == 2
        assert summary["brier_score"] > 0


# ---------------------------------------------------------------------------
# Integration: Settlement + Calibration
# ---------------------------------------------------------------------------


class TestSettlementCalibrationIntegration:
    """Test that settlement data flows correctly into calibration."""

    def test_settlement_feeds_calibration(self):
        """When a settlement is resolved, it should produce calibration data."""
        from aragora.debate.settlement import (
            SettlementTracker,
            VerifiableClaim,
            SettlementRecord,
        )

        calibration_tracker = CalibrationTracker()
        settlement_tracker = SettlementTracker()

        # Create a pending settlement
        claim = VerifiableClaim(
            claim_id="c1",
            debate_id="d1",
            statement="Will improve by 30%",
            author="claude",
            confidence=0.85,
            domain="performance",
        )
        record = SettlementRecord(settlement_id="stl-test1", claim=claim)
        settlement_tracker._records["stl-test1"] = record
        settlement_tracker._debate_index.setdefault("d1", []).append("stl-test1")

        # Settle it
        result = settlement_tracker.settle("stl-test1", "correct")

        # Record the outcome in calibration tracker
        calibration_tracker.record_outcome(
            debate_id="d1",
            predicted_confidence=claim.confidence,
            actual_outcome=(result.score >= 0.5),
            agent_id=claim.author,
            domain=claim.domain,
        )

        # Verify calibration data
        brier = calibration_tracker.get_brier_score(agent_id="claude")
        # confidence=0.85, outcome=1.0: (0.85 - 1.0)^2 = 0.0225
        assert brier == pytest.approx(0.0225)

    def test_epistemic_settlement_with_calibration(self):
        """Full flow: capture settlement -> mark reviewed -> record calibration."""
        from aragora.debate.settlement import EpistemicSettlementTracker

        settlement_tracker = EpistemicSettlementTracker()
        calibration_tracker = CalibrationTracker()

        # Capture a settlement
        from types import SimpleNamespace

        debate_result = SimpleNamespace(
            debate_id="integration-test",
            consensus_reached=True,
            confidence=0.85,
            winner="claude",
            participants=["claude", "gpt4"],
            dissenting_views=["gpt4: I disagree"],
            final_answer="We should use approach A",
            unresolved_tensions=[],
            messages=[],
            votes=[],
        )

        meta = settlement_tracker.capture_settlement(debate_result, review_horizon_days=0)

        # Review and confirm
        settlement_tracker.mark_reviewed("integration-test", "confirmed", "Validated by metrics")

        # Record calibration
        calibration_tracker.record_outcome(
            debate_id="integration-test",
            predicted_confidence=meta.confidence,
            actual_outcome=True,  # Decision was confirmed
            agent_id="claude",
        )

        # Verify
        summary = calibration_tracker.get_summary()
        assert summary["total_records"] == 1
        assert summary["correct"] == 1
