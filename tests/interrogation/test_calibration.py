"""Tests for the Interrogation Calibration System."""

from __future__ import annotations

import pytest
from aragora.interrogation.calibration import (
    CalibrationSnapshot,
    CategoryEffectiveness,
    InterrogationCalibrator,
    QuestionOutcome,
    SessionRecord,
)


@pytest.fixture
def calibrator():
    return InterrogationCalibrator()


def make_outcome(
    category: str = "requirements",
    domain: str = "technical",
    answered: bool = True,
    changed: bool = False,
    **kwargs,
) -> QuestionOutcome:
    defaults = {
        "question_id": "q1",
        "category": category,
        "domain": domain,
        "was_answered": answered,
        "answer_changed_default": changed,
    }
    defaults.update(kwargs)
    return QuestionOutcome(**defaults)


class TestCategoryEffectiveness:
    def test_answer_rate(self):
        eff = CategoryEffectiveness(category="req", domain="tech", times_asked=10, times_answered=8)
        assert eff.answer_rate == 0.8

    def test_change_rate(self):
        eff = CategoryEffectiveness(
            category="req", domain="tech", times_asked=10, times_answered=8, times_changed_default=4
        )
        assert eff.change_rate == 0.5

    def test_value_score(self):
        eff = CategoryEffectiveness(
            category="req",
            domain="tech",
            times_asked=10,
            times_answered=10,
            times_changed_default=10,
        )
        # answer_rate=1.0, change_rate=1.0 → 0.3*1.0 + 0.7*1.0 = 1.0
        assert eff.value_score == 1.0

    def test_zero_division_safe(self):
        eff = CategoryEffectiveness(category="req", domain="tech")
        assert eff.answer_rate == 0.0
        assert eff.change_rate == 0.0
        assert eff.value_score == 0.0


class TestRecording:
    def test_record_session(self, calibrator):
        record = SessionRecord(
            session_id="s1",
            domain="technical",
            outcomes=[
                make_outcome(category="requirements", answered=True, changed=True),
                make_outcome(category="constraints", answered=True, changed=False),
                make_outcome(category="edge_cases", answered=False, changed=False),
            ],
            spec_quality=0.8,
        )
        calibrator.record_session(record)
        eff = calibrator.get_effectiveness("technical", "requirements")
        assert eff is not None
        assert eff.times_asked == 1
        assert eff.times_answered == 1
        assert eff.times_changed_default == 1

    def test_multiple_sessions_accumulate(self, calibrator):
        for i in range(3):
            calibrator.record_session(
                SessionRecord(
                    session_id=f"s{i}",
                    domain="tech",
                    outcomes=[make_outcome(domain="tech", answered=True, changed=(i > 0))],
                    spec_quality=0.7,
                )
            )
        eff = calibrator.get_effectiveness("tech", "requirements")
        assert eff.times_asked == 3
        assert eff.times_answered == 3
        assert eff.times_changed_default == 2


class TestShouldAsk:
    def test_insufficient_data(self, calibrator):
        # No data → should ask by default
        assert calibrator.should_ask("tech", "unknown_category") is True

    def test_high_value_asked(self, calibrator):
        for i in range(5):
            calibrator.record_session(
                SessionRecord(
                    session_id=f"s{i}",
                    domain="tech",
                    outcomes=[make_outcome(domain="tech", answered=True, changed=True)],
                    spec_quality=0.9,
                )
            )
        assert calibrator.should_ask("tech", "requirements") is True

    def test_low_value_skipped(self, calibrator):
        for i in range(5):
            calibrator.record_session(
                SessionRecord(
                    session_id=f"s{i}",
                    domain="tech",
                    outcomes=[make_outcome(domain="tech", answered=False, changed=False)],
                    spec_quality=0.3,
                )
            )
        # Never answered, never changed → value_score ~0 → should_ask=False
        assert calibrator.should_ask("tech", "requirements") is False


class TestPriorityAdjustment:
    def test_neutral_with_no_data(self, calibrator):
        assert calibrator.get_priority_adjustment("tech", "unknown") == 1.0

    def test_boost_high_value(self, calibrator):
        for i in range(3):
            calibrator.record_session(
                SessionRecord(
                    session_id=f"s{i}",
                    domain="tech",
                    outcomes=[make_outcome(domain="tech", answered=True, changed=True)],
                    spec_quality=0.9,
                )
            )
        adj = calibrator.get_priority_adjustment("tech", "requirements")
        assert adj > 1.0  # Should boost


class TestCalibrationSnapshot:
    def test_snapshot(self, calibrator):
        for i in range(3):
            calibrator.record_session(
                SessionRecord(
                    session_id=f"s{i}",
                    domain="tech",
                    outcomes=[
                        make_outcome(
                            domain="tech", category="important", answered=True, changed=True
                        ),
                        make_outcome(
                            domain="tech", category="trivial", answered=False, changed=False
                        ),
                    ],
                    spec_quality=0.8,
                )
            )
        snap = calibrator.get_calibration("tech")
        assert snap.total_sessions == 3
        assert "important" in snap.focus_categories
        assert "trivial" not in snap.focus_categories


class TestSerialization:
    def test_to_dict(self, calibrator):
        calibrator.record_session(
            SessionRecord(
                session_id="s1",
                domain="tech",
                outcomes=[make_outcome(domain="tech", answered=True, changed=True)],
                spec_quality=0.9,
            )
        )
        d = calibrator.to_dict()
        assert "tech" in d
        assert "requirements" in d["tech"]["categories"]
        cat = d["tech"]["categories"]["requirements"]
        assert cat["times_asked"] == 1
        assert cat["value_score"] > 0
