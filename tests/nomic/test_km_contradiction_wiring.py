"""Tests for KnowledgeMound contradiction detection wiring.

Validates that:
1. HardenedOrchestrator calls contradiction detection after recording outcomes
2. SelfImproveFeedbackOrchestrator Step 7 generates goals from contradictions
3. High-severity contradictions are injected into the ImprovementQueue
4. Graceful degradation when KM or ContradictionDetector is unavailable
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.feedback_orchestrator import (
    FeedbackGoal,
    FeedbackResult,
    ImprovementQueue,
    SelfImproveFeedbackOrchestrator,
)

# Patch targets: lazy imports land in source modules, so we patch there.
_KM_GET = "aragora.knowledge.mound.get_knowledge_mound"
_DETECTOR_CLS = "aragora.knowledge.mound.ops.contradiction.ContradictionDetector"
_QUEUE_CLS = "aragora.nomic.feedback_orchestrator.ImprovementQueue"


# ---------------------------------------------------------------------------
# Helpers: build fake Contradiction / ContradictionReport objects
# ---------------------------------------------------------------------------


def _make_contradiction(
    *,
    item_a_id: str = "item-a",
    item_b_id: str = "item-b",
    contradiction_type: str = "semantic",
    similarity_score: float = 0.9,
    conflict_score: float = 0.85,
    severity: str | None = None,
) -> MagicMock:
    """Create a mock Contradiction with the fields the wiring code accesses."""
    c = MagicMock()
    c.item_a_id = item_a_id
    c.item_b_id = item_b_id

    # contradiction_type must have a .value for Enum-like access
    type_mock = MagicMock()
    type_mock.value = contradiction_type
    c.contradiction_type = type_mock

    c.similarity_score = similarity_score
    c.conflict_score = conflict_score

    # Severity is a property computed from scores, but we override for tests
    if severity is None:
        score = similarity_score * conflict_score
        if score > 0.8:
            severity = "critical"
        elif score > 0.5:
            severity = "high"
        elif score > 0.3:
            severity = "medium"
        else:
            severity = "low"
    c.severity = severity

    return c


def _make_report(
    contradictions: list[Any] | None = None,
    contradictions_found: int | None = None,
) -> MagicMock:
    """Create a mock ContradictionReport."""
    report = MagicMock()
    contradictions = contradictions or []
    report.contradictions = contradictions
    report.contradictions_found = (
        contradictions_found if contradictions_found is not None else len(contradictions)
    )

    # Build by_severity / by_type summaries
    by_severity: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for c in contradictions:
        by_severity[c.severity] = by_severity.get(c.severity, 0) + 1
        by_type[c.contradiction_type.value] = by_type.get(c.contradiction_type.value, 0) + 1
    report.by_severity = by_severity
    report.by_type = by_type

    return report


def _make_orchestration_result(
    success: bool = True,
    completed: int = 3,
    failed: int = 0,
) -> MagicMock:
    """Create a mock OrchestrationResult."""
    result = MagicMock()
    result.success = success
    result.completed_subtasks = completed
    result.failed_subtasks = failed
    result.improvement_score = 0.5
    result.success_criteria_met = True
    result.duration_seconds = 10.0
    result.assignments = []
    result.metrics_delta = {}
    result.baseline_metrics = {}
    result.after_metrics = {}
    return result


# Shared fixture: mock all 6 pre-existing feedback steps so only step 7 runs live
_STEP_PATCHES = [
    "aragora.nomic.feedback_orchestrator.SelfImproveFeedbackOrchestrator._step_gauntlet",
    "aragora.nomic.feedback_orchestrator.SelfImproveFeedbackOrchestrator._step_introspection",
    "aragora.nomic.feedback_orchestrator.SelfImproveFeedbackOrchestrator._step_genesis",
    "aragora.nomic.feedback_orchestrator.SelfImproveFeedbackOrchestrator._step_learning",
    "aragora.nomic.feedback_orchestrator.SelfImproveFeedbackOrchestrator._step_workspace",
    "aragora.nomic.feedback_orchestrator.SelfImproveFeedbackOrchestrator._step_pulse",
]


def _patch_other_steps():
    """Return a stacked context manager that mocks steps 1-6."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        patches = []
        for target in _STEP_PATCHES:
            if "genesis" in target or "learning" in target or "workspace" in target:
                p = patch(target, return_value=0)
            else:
                p = patch(target, return_value=[])
            patches.append(p)
        mocks = [p.start() for p in patches]
        try:
            yield mocks
        finally:
            for p in patches:
                p.stop()

    return _ctx()


def _patch_km_and_detector(report: MagicMock):
    """Return a stacked context manager that mocks get_knowledge_mound and ContradictionDetector."""
    import contextlib

    mock_detector_instance = MagicMock()
    mock_detector_instance.detect_contradictions = AsyncMock(return_value=report)

    @contextlib.contextmanager
    def _ctx():
        with (
            patch(_KM_GET, return_value=MagicMock()) as km_mock,
            patch(
                _DETECTOR_CLS,
                return_value=mock_detector_instance,
            ) as det_mock,
        ):
            yield km_mock, det_mock, mock_detector_instance

    return _ctx()


# ============================================================================
# A. SelfImproveFeedbackOrchestrator -- Step 7 tests
# ============================================================================


class TestFeedbackStepKnowledgeContradiction:
    """Tests for _step_knowledge_contradiction in the feedback pipeline."""

    def test_step7_runs_in_pipeline(self, tmp_path):
        """Step 7 runs as part of the full pipeline and updates result fields."""
        queue_path = tmp_path / "queue.json"

        critical = _make_contradiction(severity="critical")
        report = _make_report(contradictions=[critical])

        with _patch_other_steps(), _patch_km_and_detector(report):
            orch = SelfImproveFeedbackOrchestrator(queue_path=queue_path)
            result = orch.run("cycle-001", [])

        assert result.contradiction_detections == 1
        assert result.goals_generated == 1
        assert result.steps_completed == 7  # All 7 steps succeed

    def test_step7_generates_goals_for_high_severity(self, tmp_path):
        """Only medium+ severity contradictions produce goals."""
        queue_path = tmp_path / "queue.json"

        contradictions = [
            _make_contradiction(severity="critical", item_a_id="a1", item_b_id="b1"),
            _make_contradiction(severity="high", item_a_id="a2", item_b_id="b2"),
            _make_contradiction(severity="medium", item_a_id="a3", item_b_id="b3"),
            _make_contradiction(severity="low", item_a_id="a4", item_b_id="b4"),
        ]
        report = _make_report(contradictions=contradictions)

        with _patch_km_and_detector(report):
            orch = SelfImproveFeedbackOrchestrator(queue_path=queue_path)
            goals = orch._step_knowledge_contradiction("cycle-002")

        # low is excluded, so 3 goals
        assert len(goals) == 3
        severities = [g.metadata["severity"] for g in goals]
        assert "low" not in severities
        assert "critical" in severities
        assert "high" in severities
        assert "medium" in severities

    def test_step7_priority_mapping(self, tmp_path):
        """Critical -> priority 1, high -> 2, medium -> 3."""
        queue_path = tmp_path / "queue.json"

        contradictions = [
            _make_contradiction(severity="critical"),
            _make_contradiction(severity="high"),
            _make_contradiction(severity="medium"),
        ]
        report = _make_report(contradictions=contradictions)

        with _patch_km_and_detector(report):
            orch = SelfImproveFeedbackOrchestrator(queue_path=queue_path)
            goals = orch._step_knowledge_contradiction("cycle-003")

        priority_by_severity = {g.metadata["severity"]: g.priority for g in goals}
        assert priority_by_severity["critical"] == 1
        assert priority_by_severity["high"] == 2
        assert priority_by_severity["medium"] == 3

    def test_step7_caps_at_10_goals(self, tmp_path):
        """At most 10 goals are returned even with many contradictions."""
        queue_path = tmp_path / "queue.json"

        contradictions = [
            _make_contradiction(
                severity="high",
                item_a_id=f"a{i}",
                item_b_id=f"b{i}",
            )
            for i in range(20)
        ]
        report = _make_report(contradictions=contradictions)

        with _patch_km_and_detector(report):
            orch = SelfImproveFeedbackOrchestrator(queue_path=queue_path)
            goals = orch._step_knowledge_contradiction("cycle-004")

        assert len(goals) == 10

    def test_step7_no_contradictions_returns_empty(self, tmp_path):
        """When no contradictions found, returns empty list."""
        queue_path = tmp_path / "queue.json"
        report = _make_report(contradictions=[])

        with _patch_km_and_detector(report):
            orch = SelfImproveFeedbackOrchestrator(queue_path=queue_path)
            goals = orch._step_knowledge_contradiction("cycle-005")

        assert goals == []

    def test_step7_goal_source_is_km_contradiction(self, tmp_path):
        """All goals from step 7 have source='km_contradiction'."""
        queue_path = tmp_path / "queue.json"

        contradictions = [
            _make_contradiction(severity="high"),
            _make_contradiction(severity="critical"),
        ]
        report = _make_report(contradictions=contradictions)

        with _patch_km_and_detector(report):
            orch = SelfImproveFeedbackOrchestrator(queue_path=queue_path)
            goals = orch._step_knowledge_contradiction("cycle-006")

        for goal in goals:
            assert goal.source == "km_contradiction"

    def test_step7_metadata_contains_cycle_id(self, tmp_path):
        """Goals include cycle_id in metadata."""
        queue_path = tmp_path / "queue.json"

        contradictions = [_make_contradiction(severity="high")]
        report = _make_report(contradictions=contradictions)

        with _patch_km_and_detector(report):
            orch = SelfImproveFeedbackOrchestrator(queue_path=queue_path)
            goals = orch._step_knowledge_contradiction("cycle-007")

        assert goals[0].metadata["cycle_id"] == "cycle-007"

    def test_step7_graceful_on_import_error(self, tmp_path):
        """Step 7 raises ImportError (caught by pipeline's run method)."""
        queue_path = tmp_path / "queue.json"
        orch = SelfImproveFeedbackOrchestrator(queue_path=queue_path)

        # Patching at the source so the lazy import inside the method fails
        with patch(_KM_GET, side_effect=ImportError("KM not installed")):
            # _step_knowledge_contradiction propagates ImportError; the
            # pipeline run() method catches it.
            with pytest.raises(ImportError):
                orch._step_knowledge_contradiction("cycle-008")

    def test_step7_goals_persisted_to_queue(self, tmp_path):
        """Goals from step 7 are persisted to the improvement queue file."""
        queue_path = tmp_path / "queue.json"

        contradictions = [_make_contradiction(severity="critical")]
        report = _make_report(contradictions=contradictions)

        with _patch_other_steps(), _patch_km_and_detector(report):
            orch = SelfImproveFeedbackOrchestrator(queue_path=queue_path)
            result = orch.run("cycle-009", [])

        # Queue should be persisted
        assert queue_path.exists()
        data = json.loads(queue_path.read_text())
        assert len(data) == 1
        assert data[0]["source"] == "km_contradiction"


# ============================================================================
# B. HardenedOrchestrator -- _detect_km_contradictions tests
# ============================================================================


class TestHardenedOrchestratorContradictionDetection:
    """Tests for HardenedOrchestrator._detect_km_contradictions."""

    def _make_orchestrator(self) -> Any:
        """Create a minimal HardenedOrchestrator for testing."""
        from aragora.nomic.hardened_orchestrator import HardenedOrchestrator

        with patch.object(HardenedOrchestrator, "__init__", lambda self, **kw: None):
            orch = HardenedOrchestrator.__new__(HardenedOrchestrator)
            # Set minimal required attributes
            orch.hardened_config = MagicMock()
            orch._spectate_events = []
            orch._receipts = []
            orch._budget_spent_usd = 0.0
            orch._emit_event = MagicMock()
            return orch

    @pytest.mark.asyncio
    async def test_detect_contradictions_emits_event(self):
        """Contradictions found -> 'km_contradictions_detected' event emitted."""
        orch = self._make_orchestrator()
        result = _make_orchestration_result()

        critical = _make_contradiction(severity="critical")
        report = _make_report(contradictions=[critical])

        mock_detector = MagicMock()
        mock_detector.detect_contradictions = AsyncMock(return_value=report)

        mock_queue = MagicMock()

        with (
            patch(_KM_GET, return_value=MagicMock()),
            patch(
                _DETECTOR_CLS,
                return_value=mock_detector,
            ),
            patch(
                "aragora.nomic.feedback_orchestrator.ImprovementQueue",
            ) as mock_queue_cls,
        ):
            mock_queue_cls.load.return_value = mock_queue

            await orch._detect_km_contradictions("test goal", result)

        orch._emit_event.assert_called_once_with(
            "km_contradictions_detected",
            count=1,
            by_severity={"critical": 1},
            by_type=report.by_type,
        )

    @pytest.mark.asyncio
    async def test_detect_contradictions_queues_goals(self):
        """High/critical contradictions are added to ImprovementQueue."""
        orch = self._make_orchestrator()
        result = _make_orchestration_result()

        contradictions = [
            _make_contradiction(severity="critical", item_a_id="c1", item_b_id="c2"),
            _make_contradiction(severity="high", item_a_id="h1", item_b_id="h2"),
        ]
        report = _make_report(contradictions=contradictions)

        mock_detector = MagicMock()
        mock_detector.detect_contradictions = AsyncMock(return_value=report)

        mock_queue = MagicMock()

        with (
            patch(_KM_GET, return_value=MagicMock()),
            patch(
                _DETECTOR_CLS,
                return_value=mock_detector,
            ),
            patch(
                "aragora.nomic.feedback_orchestrator.ImprovementQueue",
            ) as mock_queue_cls,
        ):
            mock_queue_cls.load.return_value = mock_queue

            await orch._detect_km_contradictions("test goal", result)

        # Two goals should be added (critical + high)
        assert mock_queue.add.call_count == 2
        mock_queue.save.assert_called_once()

        # Verify goal properties
        first_goal = mock_queue.add.call_args_list[0][0][0]
        assert isinstance(first_goal, FeedbackGoal)
        assert first_goal.source == "km_contradiction"
        assert first_goal.priority == 1  # Critical
        assert "critical" in first_goal.description

    @pytest.mark.asyncio
    async def test_detect_contradictions_caps_at_5(self):
        """At most 5 contradictions are queued to avoid flooding."""
        orch = self._make_orchestrator()
        result = _make_orchestration_result()

        contradictions = [
            _make_contradiction(severity="critical", item_a_id=f"a{i}", item_b_id=f"b{i}")
            for i in range(10)
        ]
        report = _make_report(contradictions=contradictions)

        mock_detector = MagicMock()
        mock_detector.detect_contradictions = AsyncMock(return_value=report)

        mock_queue = MagicMock()

        with (
            patch(_KM_GET, return_value=MagicMock()),
            patch(
                _DETECTOR_CLS,
                return_value=mock_detector,
            ),
            patch(
                "aragora.nomic.feedback_orchestrator.ImprovementQueue",
            ) as mock_queue_cls,
        ):
            mock_queue_cls.load.return_value = mock_queue

            await orch._detect_km_contradictions("test goal", result)

        assert mock_queue.add.call_count == 5

    @pytest.mark.asyncio
    async def test_no_contradictions_skips_queue(self):
        """No contradictions -> no event, no queue writes."""
        orch = self._make_orchestrator()
        result = _make_orchestration_result()

        report = _make_report(contradictions=[])

        mock_detector = MagicMock()
        mock_detector.detect_contradictions = AsyncMock(return_value=report)

        with (
            patch(_KM_GET, return_value=MagicMock()),
            patch(
                _DETECTOR_CLS,
                return_value=mock_detector,
            ),
        ):
            await orch._detect_km_contradictions("test goal", result)

        # No event emitted, no queue interaction
        orch._emit_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_import_error_degrades_gracefully(self):
        """ImportError from KM -> debug log, no exception."""
        orch = self._make_orchestrator()
        result = _make_orchestration_result()

        with patch(_KM_GET, side_effect=ImportError("KM not available")):
            # Should not raise
            await orch._detect_km_contradictions("test goal", result)

        orch._emit_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_runtime_error_degrades_gracefully(self):
        """RuntimeError during detection -> debug log, no exception."""
        orch = self._make_orchestrator()
        result = _make_orchestration_result()

        mock_detector = MagicMock()
        mock_detector.detect_contradictions = AsyncMock(
            side_effect=RuntimeError("detection failed"),
        )

        with (
            patch(_KM_GET, return_value=MagicMock()),
            patch(
                _DETECTOR_CLS,
                return_value=mock_detector,
            ),
        ):
            # Should not raise
            await orch._detect_km_contradictions("test goal", result)

        orch._emit_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_only_high_critical_queued_not_medium_low(self):
        """Only high and critical severity contradictions go to queue."""
        orch = self._make_orchestrator()
        result = _make_orchestration_result()

        contradictions = [
            _make_contradiction(severity="critical"),
            _make_contradiction(severity="high"),
            _make_contradiction(severity="medium"),
            _make_contradiction(severity="low"),
        ]
        report = _make_report(contradictions=contradictions)

        mock_detector = MagicMock()
        mock_detector.detect_contradictions = AsyncMock(return_value=report)

        mock_queue = MagicMock()

        with (
            patch(_KM_GET, return_value=MagicMock()),
            patch(
                _DETECTOR_CLS,
                return_value=mock_detector,
            ),
            patch(
                "aragora.nomic.feedback_orchestrator.ImprovementQueue",
            ) as mock_queue_cls,
        ):
            mock_queue_cls.load.return_value = mock_queue

            await orch._detect_km_contradictions("test goal", result)

        # Only critical + high = 2 goals
        assert mock_queue.add.call_count == 2

    @pytest.mark.asyncio
    async def test_goal_metadata_includes_goal_text(self):
        """The originating goal text is stored in contradiction goal metadata."""
        orch = self._make_orchestrator()
        result = _make_orchestration_result()

        contradictions = [_make_contradiction(severity="critical")]
        report = _make_report(contradictions=contradictions)

        mock_detector = MagicMock()
        mock_detector.detect_contradictions = AsyncMock(return_value=report)

        mock_queue = MagicMock()

        with (
            patch(_KM_GET, return_value=MagicMock()),
            patch(
                _DETECTOR_CLS,
                return_value=mock_detector,
            ),
            patch(
                "aragora.nomic.feedback_orchestrator.ImprovementQueue",
            ) as mock_queue_cls,
        ):
            mock_queue_cls.load.return_value = mock_queue

            await orch._detect_km_contradictions("Improve SDK coverage", result)

        goal = mock_queue.add.call_args_list[0][0][0]
        assert goal.metadata["goal"] == "Improve SDK coverage"


# ============================================================================
# C. ImprovementQueue persistence round-trip
# ============================================================================


class TestContradictionQueueRoundTrip:
    """Test that contradiction goals survive queue save/load cycles."""

    def test_save_and_load_preserves_contradiction_goals(self, tmp_path):
        """Goals with km_contradiction source survive serialization."""
        queue_path = tmp_path / "queue.json"

        queue = ImprovementQueue()
        queue.add(
            FeedbackGoal(
                description="KM contradiction (critical): semantic conflict",
                source="km_contradiction",
                track="core",
                priority=1,
                estimated_impact="high",
                metadata={
                    "contradiction_type": "semantic",
                    "severity": "critical",
                    "conflict_score": 0.9,
                    "item_a_id": "item-a",
                    "item_b_id": "item-b",
                    "cycle_id": "cycle-010",
                },
            )
        )
        queue.save(queue_path)

        loaded = ImprovementQueue.load(queue_path)
        assert len(loaded.goals) == 1
        goal = loaded.goals[0]
        assert goal.source == "km_contradiction"
        assert goal.metadata["contradiction_type"] == "semantic"
        assert goal.metadata["severity"] == "critical"
        assert goal.metadata["cycle_id"] == "cycle-010"


# ============================================================================
# D. FeedbackResult field validation
# ============================================================================


class TestFeedbackResultContradictionField:
    """Validate the contradiction_detections field on FeedbackResult."""

    def test_default_is_zero(self):
        """FeedbackResult.contradiction_detections defaults to 0."""
        result = FeedbackResult(cycle_id="test")
        assert result.contradiction_detections == 0

    def test_field_is_settable(self):
        """contradiction_detections can be set."""
        result = FeedbackResult(cycle_id="test", contradiction_detections=5)
        assert result.contradiction_detections == 5

    def test_pipeline_updates_contradiction_field(self, tmp_path):
        """Full pipeline run populates contradiction_detections in result."""
        queue_path = tmp_path / "queue.json"

        contradictions = [
            _make_contradiction(severity="high"),
            _make_contradiction(severity="critical"),
        ]
        report = _make_report(contradictions=contradictions)

        with _patch_other_steps(), _patch_km_and_detector(report):
            orch = SelfImproveFeedbackOrchestrator(queue_path=queue_path)
            result = orch.run("cycle-011", [])

        assert result.contradiction_detections == 2
        assert result.goals_generated == 2

    def test_step7_failure_increments_steps_failed(self, tmp_path):
        """When step 7 fails, steps_failed increments and pipeline continues."""
        queue_path = tmp_path / "queue.json"

        with (
            _patch_other_steps(),
            patch(
                _KM_GET,
                side_effect=ImportError("KM not installed"),
            ),
        ):
            orch = SelfImproveFeedbackOrchestrator(queue_path=queue_path)
            result = orch.run("cycle-012", [])

        # Step 7 failed, others succeeded
        assert result.steps_failed == 1
        assert result.steps_completed == 6
        assert result.contradiction_detections == 0
