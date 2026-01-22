"""
Tests for the workflow coverage tracker.

Validates:
- Coverage tracking for step types, patterns, templates, configs
- Coverage report generation
- Thread safety of singleton tracker
- Report serialization
"""

import json
import tempfile
import threading
from pathlib import Path

import pytest

from aragora.workflow.coverage_tracker import (
    KNOWN_CONFIG_DIMENSIONS,
    KNOWN_PATTERNS,
    KNOWN_STEP_TYPES,
    KNOWN_TEMPLATES,
    CoverageEntry,
    CoverageReport,
    WorkflowCoverageTracker,
    get_coverage_report,
    get_tracker,
    print_coverage_summary,
    track_config,
    track_pattern,
    track_step,
    track_template,
)


@pytest.fixture
def fresh_tracker():
    """Get a fresh tracker instance for testing."""
    tracker = WorkflowCoverageTracker()
    tracker.reset()
    yield tracker
    tracker.reset()


class TestCoverageEntry:
    """Tests for CoverageEntry dataclass."""

    def test_entry_creation(self):
        """Test basic entry creation."""
        entry = CoverageEntry(component="task", test_name="test_basic")
        assert entry.component == "task"
        assert entry.test_name == "test_basic"
        assert entry.metadata == {}

    def test_entry_with_metadata(self):
        """Test entry with metadata."""
        entry = CoverageEntry(
            component="parallel",
            test_name="test_parallel",
            metadata={"workers": 5, "status": "success"},
        )
        assert entry.metadata["workers"] == 5

    def test_entry_to_dict(self):
        """Test dictionary conversion."""
        entry = CoverageEntry(component="agent", test_name="test_agent")
        data = entry.to_dict()
        assert data["component"] == "agent"
        assert data["test_name"] == "test_agent"
        assert "timestamp" in data


class TestWorkflowCoverageTracker:
    """Tests for the WorkflowCoverageTracker class."""

    def test_singleton_pattern(self):
        """Test that tracker is a singleton."""
        tracker1 = WorkflowCoverageTracker()
        tracker2 = WorkflowCoverageTracker()
        assert tracker1 is tracker2

    def test_track_step(self, fresh_tracker):
        """Test tracking step coverage."""
        fresh_tracker.track_step("task", "test_basic_task")
        report = fresh_tracker.get_report()
        assert "task" in report.covered_steps

    def test_track_pattern(self, fresh_tracker):
        """Test tracking pattern coverage."""
        fresh_tracker.track_pattern("sequential", "test_sequential_flow")
        report = fresh_tracker.get_report()
        assert "sequential" in report.covered_patterns

    def test_track_template(self, fresh_tracker):
        """Test tracking template coverage."""
        fresh_tracker.track_template("legal_contract_review", "test_legal_template")
        report = fresh_tracker.get_report()
        assert "legal_contract_review" in report.covered_templates

    def test_track_config(self, fresh_tracker):
        """Test tracking config coverage."""
        fresh_tracker.track_config("checkpoint_enabled", "test_checkpoint_flow")
        report = fresh_tracker.get_report()
        assert "checkpoint_enabled" in report.covered_configs

    def test_track_multiple_steps(self, fresh_tracker):
        """Test tracking multiple step types."""
        fresh_tracker.track_step("task", "test_1")
        fresh_tracker.track_step("parallel", "test_2")
        fresh_tracker.track_step("conditional", "test_3")

        report = fresh_tracker.get_report()
        assert len(report.covered_steps) == 3
        assert "task" in report.covered_steps
        assert "parallel" in report.covered_steps
        assert "conditional" in report.covered_steps

    def test_total_tests_count(self, fresh_tracker):
        """Test that unique test names are counted."""
        fresh_tracker.track_step("task", "test_1")
        fresh_tracker.track_step("parallel", "test_2")
        fresh_tracker.track_pattern("sequential", "test_1")  # Same test name

        report = fresh_tracker.get_report()
        assert report.total_tests == 2  # Only 2 unique test names

    def test_reset(self, fresh_tracker):
        """Test reset clears all data."""
        fresh_tracker.track_step("task", "test_1")
        fresh_tracker.track_pattern("sequential", "test_2")
        fresh_tracker.reset()

        report = fresh_tracker.get_report()
        assert len(report.covered_steps) == 0
        assert len(report.covered_patterns) == 0


class TestCoverageReport:
    """Tests for CoverageReport."""

    def test_coverage_calculation(self, fresh_tracker):
        """Test coverage percentages are calculated correctly."""
        # Track 2 of the known step types
        fresh_tracker.track_step("task", "test_1")
        fresh_tracker.track_step("parallel", "test_2")

        report = fresh_tracker.get_report()
        expected_coverage = 2 / len(KNOWN_STEP_TYPES)
        assert abs(report.step_coverage - expected_coverage) < 0.001

    def test_missing_steps_identified(self, fresh_tracker):
        """Test that missing steps are correctly identified."""
        fresh_tracker.track_step("task", "test_1")
        report = fresh_tracker.get_report()

        # All known steps except 'task' should be missing
        assert "task" not in report.missing_steps
        assert len(report.missing_steps) == len(KNOWN_STEP_TYPES) - 1

    def test_overall_coverage(self, fresh_tracker):
        """Test overall coverage calculation."""
        # Track one of each type
        fresh_tracker.track_step("task", "test_1")
        fresh_tracker.track_pattern("sequential", "test_2")
        fresh_tracker.track_template("legal_contract_review", "test_3")
        fresh_tracker.track_config("checkpoint_enabled", "test_4")

        report = fresh_tracker.get_report()

        # Overall should be weighted average
        expected = (
            report.step_coverage * 0.40
            + report.pattern_coverage * 0.25
            + report.template_coverage * 0.20
            + report.config_coverage * 0.15
        )
        assert abs(report.overall_coverage - expected) < 0.001

    def test_report_to_dict(self, fresh_tracker):
        """Test report dictionary serialization."""
        fresh_tracker.track_step("task", "test_1")
        report = fresh_tracker.get_report()
        data = report.to_dict()

        assert "step_coverage" in data
        assert "pattern_coverage" in data
        assert "template_coverage" in data
        assert "config_coverage" in data
        assert "overall_coverage" in data
        assert "covered_steps" in data
        assert "missing_steps" in data
        assert "total_tests" in data
        assert "generated_at" in data

    def test_report_summary(self, fresh_tracker):
        """Test human-readable summary generation."""
        fresh_tracker.track_step("task", "test_1")
        fresh_tracker.track_pattern("sequential", "test_2")

        report = fresh_tracker.get_report()
        summary = report.summary()

        assert "Workflow Coverage Report" in summary
        assert "Step Types:" in summary
        assert "Patterns:" in summary
        assert "OVERALL:" in summary


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_tracking(self, fresh_tracker):
        """Test that concurrent tracking is thread-safe."""
        errors = []

        def track_steps(step_type: str, count: int):
            try:
                for i in range(count):
                    fresh_tracker.track_step(step_type, f"test_{step_type}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=track_steps, args=("task", 100)),
            threading.Thread(target=track_steps, args=("parallel", 100)),
            threading.Thread(target=track_steps, args=("conditional", 100)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        report = fresh_tracker.get_report()
        assert "task" in report.covered_steps
        assert "parallel" in report.covered_steps
        assert "conditional" in report.covered_steps


class TestPersistence:
    """Tests for report persistence."""

    def test_save_report(self, fresh_tracker):
        """Test saving report to file."""
        fresh_tracker.track_step("task", "test_1")
        fresh_tracker.track_pattern("sequential", "test_2")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "coverage.json"
            saved_path = fresh_tracker.save_report(path)

            assert saved_path == path
            assert path.exists()

            with open(path) as f:
                data = json.load(f)

            assert "step_coverage" in data
            assert "task" in data["covered_steps"]

    def test_save_report_creates_directory(self, fresh_tracker):
        """Test that save_report creates parent directories."""
        fresh_tracker.track_step("task", "test_1")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "coverage.json"
            saved_path = fresh_tracker.save_report(path)

            assert saved_path == path
            assert path.exists()


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_tracker(self):
        """Test get_tracker returns singleton."""
        tracker1 = get_tracker()
        tracker2 = get_tracker()
        assert tracker1 is tracker2

    def test_track_step_function(self, fresh_tracker):
        """Test module-level track_step."""
        track_step("loop", "test_loop")
        report = get_coverage_report()
        assert "loop" in report.covered_steps

    def test_track_pattern_function(self, fresh_tracker):
        """Test module-level track_pattern."""
        track_pattern("hive_mind", "test_hive")
        report = get_coverage_report()
        assert "hive_mind" in report.covered_patterns

    def test_track_template_function(self, fresh_tracker):
        """Test module-level track_template."""
        track_template("healthcare_hipaa_compliance", "test_healthcare")
        report = get_coverage_report()
        assert "healthcare_hipaa_compliance" in report.covered_templates

    def test_track_config_function(self, fresh_tracker):
        """Test module-level track_config."""
        track_config("timeout_short", "test_timeout")
        report = get_coverage_report()
        assert "timeout_short" in report.covered_configs


class TestKnownComponents:
    """Tests for known component definitions."""

    def test_known_step_types_not_empty(self):
        """Test that known step types are defined."""
        assert len(KNOWN_STEP_TYPES) > 0
        assert "task" in KNOWN_STEP_TYPES
        assert "parallel" in KNOWN_STEP_TYPES

    def test_known_patterns_not_empty(self):
        """Test that known patterns are defined."""
        assert len(KNOWN_PATTERNS) > 0
        assert "sequential" in KNOWN_PATTERNS
        assert "parallel" in KNOWN_PATTERNS

    def test_known_templates_not_empty(self):
        """Test that known templates are defined."""
        assert len(KNOWN_TEMPLATES) > 0
        assert "legal_contract_review" in KNOWN_TEMPLATES
        assert "healthcare_hipaa_compliance" in KNOWN_TEMPLATES

    def test_known_configs_not_empty(self):
        """Test that known config dimensions are defined."""
        assert len(KNOWN_CONFIG_DIMENSIONS) > 0
        assert "checkpoint_enabled" in KNOWN_CONFIG_DIMENSIONS
        assert "timeout_short" in KNOWN_CONFIG_DIMENSIONS
