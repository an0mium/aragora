"""Tests for the StrategicMemoryStore — cross-session persistence of strategic assessments."""

import time

import pytest

from aragora.nomic.strategic_memory import StrategicMemoryStore
from aragora.nomic.strategic_scanner import StrategicAssessment, StrategicFinding


def _make_finding(
    category: str = "untested",
    severity: str = "high",
    file_path: str = "aragora/foo.py",
    description: str = "Module foo has no tests",
    track: str = "qa",
) -> StrategicFinding:
    return StrategicFinding(
        category=category,
        severity=severity,
        file_path=file_path,
        description=description,
        evidence="0 test files match",
        suggested_action="Add tests",
        track=track,
    )


def _make_assessment(
    objective: str = "improve test coverage",
    findings: list[StrategicFinding] | None = None,
    timestamp: float | None = None,
) -> StrategicAssessment:
    return StrategicAssessment(
        findings=findings or [_make_finding()],
        metrics={"total_modules": 100, "tested_pct": 0.6},
        focus_areas=["testing", "error handling"],
        objective=objective,
        timestamp=timestamp or time.time(),
    )


@pytest.fixture
def store(tmp_path):
    """Create a StrategicMemoryStore with a temporary database."""
    db_path = str(tmp_path / "strategic_memory.db")
    return StrategicMemoryStore(db_path=db_path)


class TestSaveAndLoad:
    """Test save/load roundtrip."""

    def test_save_returns_id(self, store):
        assessment = _make_assessment()
        aid = store.save(assessment)
        assert aid.startswith("sa-")

    def test_save_and_get_latest(self, store):
        assessment = _make_assessment()
        store.save(assessment)
        results = store.get_latest(limit=1)
        assert len(results) == 1
        assert results[0].objective == "improve test coverage"
        assert len(results[0].findings) == 1
        assert results[0].findings[0].category == "untested"

    def test_roundtrip_preserves_all_fields(self, store):
        finding = _make_finding(
            category="complex",
            severity="critical",
            file_path="aragora/debate/orchestrator.py",
            description="Orchestrator is too complex",
            track="core",
        )
        assessment = _make_assessment(
            objective="reduce complexity",
            findings=[finding],
            timestamp=1000.0,
        )
        store.save(assessment)
        results = store.get_latest(limit=1)
        result = results[0]
        assert result.objective == "reduce complexity"
        assert result.timestamp == 1000.0
        assert result.metrics["total_modules"] == 100
        assert result.focus_areas == ["testing", "error handling"]
        f = result.findings[0]
        assert f.category == "complex"
        assert f.severity == "critical"
        assert f.file_path == "aragora/debate/orchestrator.py"
        assert f.description == "Orchestrator is too complex"
        assert f.suggested_action == "Add tests"
        assert f.track == "core"


class TestGetLatest:
    """Test ordering and limiting of get_latest."""

    def test_ordering_by_timestamp(self, store):
        store.save(_make_assessment(objective="first", timestamp=1000.0))
        store.save(_make_assessment(objective="second", timestamp=2000.0))
        store.save(_make_assessment(objective="third", timestamp=3000.0))
        results = store.get_latest(limit=2)
        assert len(results) == 2
        assert results[0].objective == "third"
        assert results[1].objective == "second"

    def test_limit_respected(self, store):
        for i in range(5):
            store.save(_make_assessment(objective=f"obj-{i}", timestamp=float(i)))
        results = store.get_latest(limit=3)
        assert len(results) == 3

    def test_empty_store(self, store):
        results = store.get_latest(limit=3)
        assert results == []


class TestGetForObjective:
    """Test objective-based filtering."""

    def test_substring_match(self, store):
        store.save(_make_assessment(objective="improve test coverage"))
        store.save(_make_assessment(objective="improve error handling"))
        store.save(_make_assessment(objective="reduce complexity"))
        results = store.get_for_objective("improve")
        assert len(results) == 2

    def test_no_match(self, store):
        store.save(_make_assessment(objective="improve test coverage"))
        results = store.get_for_objective("nonexistent")
        assert results == []

    def test_exact_match(self, store):
        store.save(_make_assessment(objective="improve test coverage"))
        results = store.get_for_objective("improve test coverage")
        assert len(results) == 1
        assert results[0].objective == "improve test coverage"


class TestRecurringFindings:
    """Test recurring finding detection."""

    def test_finds_recurring(self, store):
        finding = _make_finding(file_path="aragora/bar.py")
        store.save(_make_assessment(findings=[finding], timestamp=1000.0))
        store.save(_make_assessment(findings=[finding], timestamp=2000.0))
        recurring = store.get_recurring_findings(min_occurrences=2)
        assert len(recurring) >= 1
        paths = [f.file_path for f in recurring]
        assert "aragora/bar.py" in paths

    def test_ignores_single_occurrence(self, store):
        finding = _make_finding(file_path="aragora/unique.py")
        store.save(_make_assessment(findings=[finding]))
        recurring = store.get_recurring_findings(min_occurrences=2)
        unique_paths = [f.file_path for f in recurring]
        assert "aragora/unique.py" not in unique_paths

    def test_different_categories_not_grouped(self, store):
        f1 = _make_finding(category="untested", file_path="aragora/x.py")
        f2 = _make_finding(category="complex", file_path="aragora/x.py")
        store.save(_make_assessment(findings=[f1], timestamp=1000.0))
        store.save(_make_assessment(findings=[f2], timestamp=2000.0))
        # Different categories should not be grouped
        recurring = store.get_recurring_findings(min_occurrences=2)
        assert len(recurring) == 0

    def test_empty_store_returns_empty(self, store):
        recurring = store.get_recurring_findings(min_occurrences=2)
        assert recurring == []
