"""Tests for AutonomousAssessmentEngine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest

from aragora.nomic.assessment_engine import (
    AutonomousAssessmentEngine,
    CodebaseHealthReport,
    ImprovementCandidate,
    SignalSource,
)


@pytest.fixture
def engine():
    return AutonomousAssessmentEngine()


# --- Mock helpers ---

def _mock_scanner_assessment(findings=None):
    """Return a mock StrategicAssessment with given findings."""
    assessment = SimpleNamespace(
        findings=findings or [],
        metrics={},
        focus_areas=[],
    )
    return assessment


def _mock_metric_snapshot(tests_failed=0, lint_errors=0):
    """Return a mock MetricSnapshot that supports to_dict."""
    snapshot = MagicMock()
    snapshot.to_dict.return_value = {
        "tests_passed": 100,
        "tests_failed": tests_failed,
        "lint_errors": lint_errors,
        "timestamp": 1000.0,
    }
    return snapshot


def _patch_all_sources_fail():
    """Patch all signal source imports to raise ImportError."""
    return [
        patch(
            "aragora.nomic.assessment_engine.AutonomousAssessmentEngine._collect_scanner_signals",
            return_value=SignalSource(name="scanner", weight=0.3, error="StrategicScanner not available"),
        ),
        patch(
            "aragora.nomic.assessment_engine.AutonomousAssessmentEngine._collect_metrics_signals",
            return_value=SignalSource(name="metrics", weight=0.25, error="MetricsCollector not available"),
        ),
        patch(
            "aragora.nomic.assessment_engine.AutonomousAssessmentEngine._collect_regression_signals",
            return_value=SignalSource(name="regressions", weight=0.2, error="OutcomeTracker not available"),
        ),
        patch(
            "aragora.nomic.assessment_engine.AutonomousAssessmentEngine._collect_queue_signals",
            return_value=SignalSource(name="queue", weight=0.15, error="ImprovementQueue not available"),
        ),
        patch(
            "aragora.nomic.assessment_engine.AutonomousAssessmentEngine._collect_feedback_signals",
            return_value=SignalSource(name="feedback", weight=0.1, error="OutcomeFeedbackBridge not available"),
        ),
    ]


# --- Tests ---


@pytest.mark.asyncio
async def test_assess_returns_health_report(engine):
    """Basic assess() returns CodebaseHealthReport with valid fields."""
    patches = _patch_all_sources_fail()
    for p in patches:
        p.start()
    try:
        report = await engine.assess()
        assert isinstance(report, CodebaseHealthReport)
        assert isinstance(report.health_score, float)
        assert isinstance(report.signal_sources, list)
        assert isinstance(report.improvement_candidates, list)
        assert isinstance(report.metrics_snapshot, dict)
        assert report.assessment_duration_seconds >= 0.0
    finally:
        for p in patches:
            p.stop()


@pytest.mark.asyncio
async def test_health_score_in_range(engine):
    """health_score is always 0.0-1.0."""
    patches = _patch_all_sources_fail()
    for p in patches:
        p.start()
    try:
        report = await engine.assess()
        assert 0.0 <= report.health_score <= 1.0
    finally:
        for p in patches:
            p.stop()


@pytest.mark.asyncio
async def test_candidates_sorted_by_priority(engine):
    """Candidates are sorted highest priority first."""
    scanner_source = SignalSource(
        name="scanner",
        weight=0.3,
        findings=[
            SimpleNamespace(description="low", severity="low", file_path="a.py", category="untested"),
            SimpleNamespace(description="high", severity="high", file_path="b.py", category="complex"),
            SimpleNamespace(description="critical", severity="critical", file_path="c.py", category="stale"),
        ],
    )
    with patch.object(engine, "_collect_scanner_signals", return_value=scanner_source), \
         patch.object(engine, "_collect_metrics_signals", return_value=SignalSource(name="metrics", weight=0.25, error="skip")), \
         patch.object(engine, "_collect_regression_signals", return_value=SignalSource(name="regressions", weight=0.2, error="skip")), \
         patch.object(engine, "_collect_queue_signals", return_value=SignalSource(name="queue", weight=0.15, error="skip")), \
         patch.object(engine, "_collect_feedback_signals", return_value=SignalSource(name="feedback", weight=0.1, error="skip")):
        report = await engine.assess()
        priorities = [c.priority for c in report.improvement_candidates]
        assert priorities == sorted(priorities, reverse=True)


@pytest.mark.asyncio
async def test_scanner_signals_collected(engine):
    """Mock StrategicScanner, verify findings collected."""
    finding = SimpleNamespace(
        description="Module foo has no test",
        severity="high",
        file_path="aragora/foo.py",
        category="untested",
    )
    mock_assessment = _mock_scanner_assessment(findings=[finding])

    with patch("aragora.nomic.assessment_engine.StrategicScanner", create=True) as MockScanner:
        # The import inside _collect_scanner_signals patches this
        pass

    # Patch at the method level to control the source
    scanner_source = SignalSource(name="scanner", weight=0.3, findings=[finding])
    with patch.object(engine, "_collect_scanner_signals", return_value=scanner_source), \
         patch.object(engine, "_collect_metrics_signals", return_value=SignalSource(name="metrics", weight=0.25, error="skip")), \
         patch.object(engine, "_collect_regression_signals", return_value=SignalSource(name="regressions", weight=0.2, error="skip")), \
         patch.object(engine, "_collect_queue_signals", return_value=SignalSource(name="queue", weight=0.15, error="skip")), \
         patch.object(engine, "_collect_feedback_signals", return_value=SignalSource(name="feedback", weight=0.1, error="skip")):
        report = await engine.assess()
        assert len(report.signal_sources) == 5
        scanner = report.signal_sources[0]
        assert scanner.name == "scanner"
        assert len(scanner.findings) == 1
        assert scanner.error is None


@pytest.mark.asyncio
async def test_metrics_signals_collected(engine):
    """Mock MetricsCollector, verify findings collected."""
    metrics_dict = {"tests_passed": 100, "tests_failed": 2, "lint_errors": 5}
    metrics_source = SignalSource(name="metrics", weight=0.25, findings=[metrics_dict])
    with patch.object(engine, "_collect_scanner_signals", return_value=SignalSource(name="scanner", weight=0.3, error="skip")), \
         patch.object(engine, "_collect_metrics_signals", return_value=metrics_source), \
         patch.object(engine, "_collect_regression_signals", return_value=SignalSource(name="regressions", weight=0.2, error="skip")), \
         patch.object(engine, "_collect_queue_signals", return_value=SignalSource(name="queue", weight=0.15, error="skip")), \
         patch.object(engine, "_collect_feedback_signals", return_value=SignalSource(name="feedback", weight=0.1, error="skip")):
        report = await engine.assess()
        m_source = report.signal_sources[1]
        assert m_source.name == "metrics"
        assert len(m_source.findings) == 1
        assert m_source.error is None


@pytest.mark.asyncio
async def test_regression_signals_collected(engine):
    """Mock OutcomeTracker, verify findings collected."""
    regression_data = [
        {"cycle_id": "c1", "regressed_metrics": ["consensus_rate"], "recommendation": "review"}
    ]
    regression_source = SignalSource(name="regressions", weight=0.2, findings=regression_data)
    with patch.object(engine, "_collect_scanner_signals", return_value=SignalSource(name="scanner", weight=0.3, error="skip")), \
         patch.object(engine, "_collect_metrics_signals", return_value=SignalSource(name="metrics", weight=0.25, error="skip")), \
         patch.object(engine, "_collect_regression_signals", return_value=regression_source), \
         patch.object(engine, "_collect_queue_signals", return_value=SignalSource(name="queue", weight=0.15, error="skip")), \
         patch.object(engine, "_collect_feedback_signals", return_value=SignalSource(name="feedback", weight=0.1, error="skip")):
        report = await engine.assess()
        r_source = report.signal_sources[2]
        assert r_source.name == "regressions"
        assert len(r_source.findings) == 1


@pytest.mark.asyncio
async def test_queue_signals_collected(engine):
    """Mock ImprovementQueue, verify findings collected."""
    queue_data = [
        {"goal": "Improve coverage", "description": "Add tests", "category": "test_coverage", "priority": 0.7}
    ]
    queue_source = SignalSource(name="queue", weight=0.15, findings=queue_data)
    with patch.object(engine, "_collect_scanner_signals", return_value=SignalSource(name="scanner", weight=0.3, error="skip")), \
         patch.object(engine, "_collect_metrics_signals", return_value=SignalSource(name="metrics", weight=0.25, error="skip")), \
         patch.object(engine, "_collect_regression_signals", return_value=SignalSource(name="regressions", weight=0.2, error="skip")), \
         patch.object(engine, "_collect_queue_signals", return_value=queue_source), \
         patch.object(engine, "_collect_feedback_signals", return_value=SignalSource(name="feedback", weight=0.1, error="skip")):
        report = await engine.assess()
        q_source = report.signal_sources[3]
        assert q_source.name == "queue"
        assert len(q_source.findings) == 1


@pytest.mark.asyncio
async def test_feedback_signals_collected(engine):
    """Mock OutcomeFeedbackBridge, verify findings collected."""
    feedback_data = [
        {"description": "Agent X overconfident in security domain", "priority": 0.8, "files": [], "category": "feedback"}
    ]
    feedback_source = SignalSource(name="feedback", weight=0.1, findings=feedback_data)
    with patch.object(engine, "_collect_scanner_signals", return_value=SignalSource(name="scanner", weight=0.3, error="skip")), \
         patch.object(engine, "_collect_metrics_signals", return_value=SignalSource(name="metrics", weight=0.25, error="skip")), \
         patch.object(engine, "_collect_regression_signals", return_value=SignalSource(name="regressions", weight=0.2, error="skip")), \
         patch.object(engine, "_collect_queue_signals", return_value=SignalSource(name="queue", weight=0.15, error="skip")), \
         patch.object(engine, "_collect_feedback_signals", return_value=feedback_source):
        report = await engine.assess()
        f_source = report.signal_sources[4]
        assert f_source.name == "feedback"
        assert len(f_source.findings) == 1


@pytest.mark.asyncio
async def test_graceful_degradation_all_sources_fail(engine):
    """All imports fail, still get valid report."""
    patches = _patch_all_sources_fail()
    for p in patches:
        p.start()
    try:
        report = await engine.assess()
        assert isinstance(report, CodebaseHealthReport)
        assert 0.0 <= report.health_score <= 1.0
        assert len(report.signal_sources) == 5
        for s in report.signal_sources:
            assert s.error is not None
        assert len(report.improvement_candidates) == 0
    finally:
        for p in patches:
            p.stop()


def test_scanner_to_candidates_with_findings(engine):
    """Test _scanner_to_candidates conversion with StrategicFinding-like objects."""
    finding = SimpleNamespace(
        description="Module bar has no test",
        severity="high",
        file_path="aragora/bar.py",
        category="untested",
    )
    source = SignalSource(name="scanner", weight=0.3, findings=[finding])
    candidates = engine._scanner_to_candidates(source)
    assert len(candidates) == 1
    assert candidates[0].description == "Module bar has no test"
    assert candidates[0].priority == 0.8  # high -> 0.8
    assert candidates[0].source == "scanner"
    assert candidates[0].files == ["aragora/bar.py"]
    assert candidates[0].category == "untested"


def test_metrics_to_candidates_with_test_failures(engine):
    """Test _metrics_to_candidates with tests_failed > 0."""
    source = SignalSource(
        name="metrics",
        weight=0.25,
        findings=[{"tests_passed": 100, "tests_failed": 5, "lint_errors": 3}],
    )
    candidates = engine._metrics_to_candidates(source)
    assert len(candidates) == 1
    assert "5" in candidates[0].description
    assert candidates[0].priority == 0.9
    assert candidates[0].category == "test"


def test_metrics_to_candidates_with_lint_violations(engine):
    """Test _metrics_to_candidates with lint_errors > 10."""
    source = SignalSource(
        name="metrics",
        weight=0.25,
        findings=[{"tests_passed": 100, "tests_failed": 0, "lint_errors": 55}],
    )
    candidates = engine._metrics_to_candidates(source)
    assert len(candidates) == 1
    assert "55" in candidates[0].description
    assert candidates[0].priority == 0.7  # >= 50 -> 0.7
    assert candidates[0].category == "lint"


@pytest.mark.asyncio
async def test_health_score_perfect_no_issues(engine):
    """No candidates and no source errors -> high health score."""
    empty_sources = [
        SignalSource(name="scanner", weight=0.3, findings=[]),
        SignalSource(name="metrics", weight=0.25, findings=[]),
        SignalSource(name="regressions", weight=0.2, findings=[]),
        SignalSource(name="queue", weight=0.15, findings=[]),
        SignalSource(name="feedback", weight=0.1, findings=[]),
    ]
    with patch.object(engine, "_collect_scanner_signals", return_value=empty_sources[0]), \
         patch.object(engine, "_collect_metrics_signals", return_value=empty_sources[1]), \
         patch.object(engine, "_collect_regression_signals", return_value=empty_sources[2]), \
         patch.object(engine, "_collect_queue_signals", return_value=empty_sources[3]), \
         patch.object(engine, "_collect_feedback_signals", return_value=empty_sources[4]):
        report = await engine.assess()
        assert report.health_score == 1.0


@pytest.mark.asyncio
async def test_health_score_many_issues(engine):
    """Many high-priority candidates -> low health score."""
    findings = [
        SimpleNamespace(description=f"Issue {i}", severity="critical", file_path=f"f{i}.py", category="complex")
        for i in range(20)
    ]
    scanner_source = SignalSource(name="scanner", weight=0.3, findings=findings)
    with patch.object(engine, "_collect_scanner_signals", return_value=scanner_source), \
         patch.object(engine, "_collect_metrics_signals", return_value=SignalSource(name="metrics", weight=0.25, error="skip")), \
         patch.object(engine, "_collect_regression_signals", return_value=SignalSource(name="regressions", weight=0.2, error="skip")), \
         patch.object(engine, "_collect_queue_signals", return_value=SignalSource(name="queue", weight=0.15, error="skip")), \
         patch.object(engine, "_collect_feedback_signals", return_value=SignalSource(name="feedback", weight=0.1, error="skip")):
        report = await engine.assess()
        # 20 critical candidates (priority=0.95) * weight 0.3 * 0.05 = 0.285 total penalty
        assert report.health_score < 0.8


def test_improvement_candidate_to_dict():
    """Test ImprovementCandidate.to_dict()."""
    candidate = ImprovementCandidate(
        description="Fix flaky test",
        priority=0.85,
        source="metrics",
        files=["tests/test_foo.py"],
        category="test",
    )
    d = candidate.to_dict()
    assert d == {
        "description": "Fix flaky test",
        "priority": 0.85,
        "source": "metrics",
        "files": ["tests/test_foo.py"],
        "category": "test",
    }


def test_codebase_health_report_to_dict():
    """Test CodebaseHealthReport.to_dict()."""
    candidate = ImprovementCandidate(
        description="Add tests",
        priority=0.7,
        source="scanner",
        files=["foo.py"],
        category="test",
    )
    source = SignalSource(name="scanner", weight=0.3, findings=[{"x": 1}])
    report = CodebaseHealthReport(
        health_score=0.82,
        signal_sources=[source],
        improvement_candidates=[candidate],
        metrics_snapshot={"tests_passed": 100},
        assessment_duration_seconds=1.5,
    )
    d = report.to_dict()
    assert d["health_score"] == 0.82
    assert len(d["signal_sources"]) == 1
    assert d["signal_sources"][0]["name"] == "scanner"
    assert d["signal_sources"][0]["weight"] == 0.3
    assert d["signal_sources"][0]["findings_count"] == 1
    assert d["signal_sources"][0]["error"] is None
    assert len(d["improvement_candidates"]) == 1
    assert d["improvement_candidates"][0]["description"] == "Add tests"
    assert d["metrics_snapshot"] == {"tests_passed": 100}
    assert d["assessment_duration_seconds"] == 1.5
