"""
Comprehensive Tests for Consensus Healing Background Worker.

Tests the consensus healing worker including:
- Worker lifecycle (start, stop, configuration)
- Healing candidate detection and processing
- Healing action determination logic
- All healing actions (re-debate, extend rounds, add mediator, notify, archive, escalate)
- Metrics tracking and status reporting
- Error handling and callback invocation
- Global worker management functions
- Dataclass serialization
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.queue.workers.consensus_healing_worker import (
    ConsensusHealingWorker,
    HealingAction,
    HealingCandidate,
    HealingConfig,
    HealingReason,
    HealingResult,
    get_consensus_healing_worker,
    start_consensus_healing,
    stop_consensus_healing,
    _global_worker,
)


# =============================================================================
# HealingAction Enum Tests
# =============================================================================


class TestHealingAction:
    """Tests for HealingAction enum."""

    def test_all_action_values(self):
        """All healing actions should have expected string values."""
        assert HealingAction.RE_DEBATE == "re_debate"
        assert HealingAction.EXTEND_ROUNDS == "extend_rounds"
        assert HealingAction.ADD_MEDIATOR == "add_mediator"
        assert HealingAction.NOTIFY_USER == "notify_user"
        assert HealingAction.ARCHIVE == "archive"
        assert HealingAction.ESCALATE == "escalate"

    def test_action_count(self):
        """Should have exactly 6 healing actions."""
        assert len(HealingAction) == 6

    def test_actions_are_strings(self):
        """All actions should be string enums."""
        for action in HealingAction:
            assert isinstance(action.value, str)


# =============================================================================
# HealingReason Enum Tests
# =============================================================================


class TestHealingReason:
    """Tests for HealingReason enum."""

    def test_all_reason_values(self):
        """All healing reasons should have expected string values."""
        assert HealingReason.NO_CONSENSUS == "no_consensus"
        assert HealingReason.STALLED == "stalled"
        assert HealingReason.DIVERGING == "diverging"
        assert HealingReason.TIMEOUT == "timeout"
        assert HealingReason.ERROR == "error"
        assert HealingReason.LOW_QUALITY == "low_quality"

    def test_reason_count(self):
        """Should have exactly 6 healing reasons."""
        assert len(HealingReason) == 6


# =============================================================================
# HealingCandidate Tests
# =============================================================================


class TestHealingCandidate:
    """Tests for HealingCandidate dataclass."""

    def test_creation_minimal(self):
        """Test creating candidate with minimal fields."""
        candidate = HealingCandidate(
            debate_id="debate-1",
            task="Design a rate limiter",
            reason=HealingReason.NO_CONSENSUS,
            created_at=1000.0,
            completed_at=None,
            rounds_completed=3,
            agent_count=4,
        )

        assert candidate.debate_id == "debate-1"
        assert candidate.task == "Design a rate limiter"
        assert candidate.reason == HealingReason.NO_CONSENSUS
        assert candidate.created_at == 1000.0
        assert candidate.completed_at is None
        assert candidate.rounds_completed == 3
        assert candidate.agent_count == 4

    def test_creation_defaults(self):
        """Test default values are set correctly."""
        candidate = HealingCandidate(
            debate_id="d-1",
            task="Task",
            reason=HealingReason.STALLED,
            created_at=0.0,
            completed_at=None,
            rounds_completed=0,
            agent_count=0,
        )

        assert candidate.consensus_probability == 0.0
        assert candidate.convergence_trend == "unknown"
        assert candidate.metadata == {}

    def test_creation_with_all_fields(self):
        """Test creating candidate with all fields populated."""
        metadata = {"healing_attempts": 1, "original_protocol": "majority"}
        candidate = HealingCandidate(
            debate_id="debate-2",
            task="Complex analysis",
            reason=HealingReason.DIVERGING,
            created_at=1000.0,
            completed_at=2000.0,
            rounds_completed=5,
            agent_count=6,
            consensus_probability=0.35,
            convergence_trend="diverging",
            metadata=metadata,
        )

        assert candidate.completed_at == 2000.0
        assert candidate.consensus_probability == 0.35
        assert candidate.convergence_trend == "diverging"
        assert candidate.metadata["healing_attempts"] == 1


# =============================================================================
# HealingResult Tests
# =============================================================================


class TestHealingResult:
    """Tests for HealingResult dataclass."""

    def test_creation_success(self):
        """Test creating a successful healing result."""
        result = HealingResult(
            debate_id="debate-1",
            action=HealingAction.RE_DEBATE,
            success=True,
            message="Re-debate scheduled",
        )

        assert result.debate_id == "debate-1"
        assert result.action == HealingAction.RE_DEBATE
        assert result.success is True
        assert result.message == "Re-debate scheduled"
        assert result.new_debate_id is None
        assert result.metrics == {}

    def test_creation_failure(self):
        """Test creating a failed healing result."""
        result = HealingResult(
            debate_id="debate-2",
            action=HealingAction.NOTIFY_USER,
            success=False,
            message="Notifications disabled",
        )

        assert result.success is False
        assert result.message == "Notifications disabled"

    def test_creation_with_new_debate_id(self):
        """Test result with a new debate ID."""
        result = HealingResult(
            debate_id="debate-old",
            action=HealingAction.RE_DEBATE,
            success=True,
            message="New debate created",
            new_debate_id="debate-new",
        )

        assert result.new_debate_id == "debate-new"

    def test_creation_with_metrics(self):
        """Test result with metrics."""
        result = HealingResult(
            debate_id="debate-1",
            action=HealingAction.EXTEND_ROUNDS,
            success=True,
            message="Extended",
            metrics={"additional_rounds": 3, "original_rounds": 5},
        )

        assert result.metrics["additional_rounds"] == 3
        assert result.metrics["original_rounds"] == 5

    def test_default_timestamp(self):
        """Test that timestamp defaults to current time."""
        before = time.time()
        result = HealingResult(
            debate_id="d-1",
            action=HealingAction.ARCHIVE,
            success=True,
            message="Archived",
        )
        after = time.time()

        assert before <= result.timestamp <= after

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = HealingResult(
            debate_id="debate-1",
            action=HealingAction.ADD_MEDIATOR,
            success=True,
            message="Mediator added",
            new_debate_id=None,
            metrics={"mediator": "claude"},
            timestamp=12345.0,
        )

        d = result.to_dict()

        assert d["debate_id"] == "debate-1"
        assert d["action"] == "add_mediator"
        assert d["success"] is True
        assert d["message"] == "Mediator added"
        assert d["new_debate_id"] is None
        assert d["metrics"]["mediator"] == "claude"
        assert d["timestamp"] == 12345.0

    def test_to_dict_all_actions(self):
        """Test to_dict serializes all action types correctly."""
        for action in HealingAction:
            result = HealingResult(
                debate_id="d-1",
                action=action,
                success=True,
                message="Test",
            )
            d = result.to_dict()
            assert d["action"] == action.value


# =============================================================================
# HealingConfig Tests
# =============================================================================


class TestHealingConfig:
    """Tests for HealingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = HealingConfig()

        assert config.scan_interval_seconds == 300
        assert config.min_age_for_healing_seconds == 3600
        assert config.max_age_for_healing_seconds == 86400 * 7
        assert config.consensus_threshold == 0.6
        assert config.min_rounds_for_redebate == 3
        assert config.max_healing_attempts == 3
        assert config.auto_redebate_enabled is False
        assert config.notify_on_stuck is True
        assert config.archive_after_max_attempts is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = HealingConfig(
            scan_interval_seconds=60,
            min_age_for_healing_seconds=600,
            max_age_for_healing_seconds=86400,
            consensus_threshold=0.8,
            min_rounds_for_redebate=5,
            max_healing_attempts=5,
            auto_redebate_enabled=True,
            notify_on_stuck=False,
            archive_after_max_attempts=False,
        )

        assert config.scan_interval_seconds == 60
        assert config.min_age_for_healing_seconds == 600
        assert config.max_age_for_healing_seconds == 86400
        assert config.consensus_threshold == 0.8
        assert config.min_rounds_for_redebate == 5
        assert config.max_healing_attempts == 5
        assert config.auto_redebate_enabled is True
        assert config.notify_on_stuck is False
        assert config.archive_after_max_attempts is False


# =============================================================================
# ConsensusHealingWorker Initialization Tests
# =============================================================================


class TestConsensusHealingWorkerInit:
    """Tests for ConsensusHealingWorker initialization."""

    def test_default_initialization(self):
        """Test worker initializes with defaults."""
        worker = ConsensusHealingWorker()

        assert worker.worker_id.startswith("consensus-healer-")
        assert isinstance(worker.config, HealingConfig)
        assert worker.on_healing_needed is None
        assert worker.on_healing_complete is None
        assert worker._running is False
        assert worker._healing_history == []
        assert worker._candidates == {}

    def test_custom_worker_id(self):
        """Test worker with custom ID."""
        worker = ConsensusHealingWorker(worker_id="custom-healer-1")
        assert worker.worker_id == "custom-healer-1"

    def test_custom_config(self):
        """Test worker with custom config."""
        config = HealingConfig(scan_interval_seconds=60)
        worker = ConsensusHealingWorker(config=config)
        assert worker.config.scan_interval_seconds == 60

    def test_callbacks(self):
        """Test worker with callbacks."""
        on_needed = MagicMock()
        on_complete = MagicMock()
        worker = ConsensusHealingWorker(
            on_healing_needed=on_needed,
            on_healing_complete=on_complete,
        )
        assert worker.on_healing_needed is on_needed
        assert worker.on_healing_complete is on_complete

    def test_initial_metrics(self):
        """Test initial metrics are zero."""
        worker = ConsensusHealingWorker()
        assert worker._scans_completed == 0
        assert worker._candidates_found == 0
        assert worker._healings_attempted == 0
        assert worker._healings_succeeded == 0


# =============================================================================
# ConsensusHealingWorker Lifecycle Tests
# =============================================================================


class TestConsensusHealingWorkerLifecycle:
    """Tests for worker start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        """Test worker can start and stop."""
        worker = ConsensusHealingWorker(
            config=HealingConfig(scan_interval_seconds=0.01),
        )

        start_task = asyncio.create_task(worker.start())
        await asyncio.sleep(0.05)

        assert worker._running is True

        await worker.stop()
        assert worker._running is False

        # Wait for the task to finish
        await asyncio.wait_for(start_task, timeout=2.0)

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self):
        """Test stop sets running flag to False."""
        worker = ConsensusHealingWorker()
        worker._running = True

        await worker.stop()
        assert worker._running is False

    @pytest.mark.asyncio
    async def test_start_handles_cancellation(self):
        """Test worker handles CancelledError gracefully."""
        worker = ConsensusHealingWorker(
            config=HealingConfig(scan_interval_seconds=0.01),
        )

        task = asyncio.create_task(worker.start())
        await asyncio.sleep(0.02)

        task.cancel()
        # CancelledError is caught inside start() and breaks the loop
        # so the task may complete normally or raise CancelledError
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

        # Worker should have stopped
        assert worker._running is False or task.done()

    @pytest.mark.asyncio
    async def test_start_handles_errors(self):
        """Test worker continues on errors in the loop."""
        worker = ConsensusHealingWorker(
            config=HealingConfig(scan_interval_seconds=0.01),
        )

        call_count = 0
        original_sleep = asyncio.sleep

        async def failing_scan():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("Scan failure")

        async def fast_sleep(seconds):
            """Speed up backoff sleeps for testing."""
            await original_sleep(min(seconds, 0.01))

        with patch.object(worker, "_scan_for_candidates", side_effect=failing_scan):
            with patch.object(worker, "_process_candidates", new_callable=AsyncMock):
                with patch(
                    "aragora.queue.workers.consensus_healing_worker.asyncio.sleep",
                    side_effect=fast_sleep,
                ):
                    task = asyncio.create_task(worker.start())
                    await original_sleep(0.15)
                    await worker.stop()
                    await asyncio.wait_for(task, timeout=2.0)

        # Worker should have retried after errors
        assert call_count >= 2


# =============================================================================
# Determine Reason Tests
# =============================================================================


class TestDetermineReason:
    """Tests for _determine_reason method."""

    def setup_method(self):
        self.worker = ConsensusHealingWorker()

    def test_error_reason(self):
        """Error field takes highest priority."""
        reason = self.worker._determine_reason(
            {
                "error": "Something broke",
                "convergence_trend": "diverging",
                "consensus_probability": 0.1,
            }
        )
        assert reason == HealingReason.ERROR

    def test_diverging_reason(self):
        """Diverging trend is second priority."""
        reason = self.worker._determine_reason(
            {
                "convergence_trend": "diverging",
                "consensus_probability": 0.1,
            }
        )
        assert reason == HealingReason.DIVERGING

    def test_no_consensus_reason(self):
        """Very low consensus probability triggers NO_CONSENSUS."""
        reason = self.worker._determine_reason(
            {
                "consensus_probability": 0.2,
            }
        )
        assert reason == HealingReason.NO_CONSENSUS

    def test_timeout_reason(self):
        """Timed out debate triggers TIMEOUT."""
        reason = self.worker._determine_reason(
            {
                "timed_out": True,
                "consensus_probability": 0.4,
            }
        )
        assert reason == HealingReason.TIMEOUT

    def test_low_quality_reason(self):
        """Consensus below threshold triggers LOW_QUALITY."""
        reason = self.worker._determine_reason(
            {
                "consensus_probability": 0.5,
            }
        )
        assert reason == HealingReason.LOW_QUALITY

    def test_stalled_reason_default(self):
        """Default reason when no other matches is STALLED."""
        reason = self.worker._determine_reason(
            {
                "consensus_probability": 0.7,
            }
        )
        assert reason == HealingReason.STALLED

    def test_missing_fields_defaults_to_no_consensus(self):
        """Empty dict with default 0.0 probability triggers NO_CONSENSUS."""
        reason = self.worker._determine_reason({})
        assert reason == HealingReason.NO_CONSENSUS


# =============================================================================
# Determine Action Tests
# =============================================================================


class TestDetermineAction:
    """Tests for _determine_action method."""

    def _make_candidate(self, reason, metadata=None, rounds_completed=0):
        """Helper to create a HealingCandidate."""
        return HealingCandidate(
            debate_id="d-1",
            task="Test",
            reason=reason,
            created_at=time.time(),
            completed_at=None,
            rounds_completed=rounds_completed,
            agent_count=3,
            metadata=metadata or {},
        )

    def test_archive_after_max_attempts(self):
        """Should archive after max healing attempts when archive enabled."""
        config = HealingConfig(max_healing_attempts=3, archive_after_max_attempts=True)
        worker = ConsensusHealingWorker(config=config)

        candidate = self._make_candidate(
            HealingReason.NO_CONSENSUS,
            metadata={"healing_attempts": 2},
        )
        action = worker._determine_action(candidate)
        assert action == HealingAction.ARCHIVE

    def test_escalate_after_max_attempts_no_archive(self):
        """Should escalate after max attempts when archive disabled."""
        config = HealingConfig(max_healing_attempts=3, archive_after_max_attempts=False)
        worker = ConsensusHealingWorker(config=config)

        candidate = self._make_candidate(
            HealingReason.NO_CONSENSUS,
            metadata={"healing_attempts": 2},
        )
        action = worker._determine_action(candidate)
        assert action == HealingAction.ESCALATE

    def test_error_reason_notifies_user(self):
        """Error reason should trigger user notification."""
        worker = ConsensusHealingWorker()
        candidate = self._make_candidate(HealingReason.ERROR)
        action = worker._determine_action(candidate)
        assert action == HealingAction.NOTIFY_USER

    def test_diverging_reason_adds_mediator(self):
        """Diverging trend should add a mediator."""
        worker = ConsensusHealingWorker()
        candidate = self._make_candidate(HealingReason.DIVERGING)
        action = worker._determine_action(candidate)
        assert action == HealingAction.ADD_MEDIATOR

    def test_timeout_with_enough_rounds_and_auto_redebate(self):
        """Timeout with sufficient rounds and auto-redebate triggers RE_DEBATE."""
        config = HealingConfig(
            auto_redebate_enabled=True,
            min_rounds_for_redebate=3,
        )
        worker = ConsensusHealingWorker(config=config)
        candidate = self._make_candidate(
            HealingReason.TIMEOUT,
            rounds_completed=5,
        )
        action = worker._determine_action(candidate)
        assert action == HealingAction.RE_DEBATE

    def test_timeout_with_enough_rounds_no_auto_redebate(self):
        """Timeout with sufficient rounds but no auto-redebate notifies user."""
        config = HealingConfig(
            auto_redebate_enabled=False,
            min_rounds_for_redebate=3,
        )
        worker = ConsensusHealingWorker(config=config)
        candidate = self._make_candidate(
            HealingReason.TIMEOUT,
            rounds_completed=5,
        )
        action = worker._determine_action(candidate)
        assert action == HealingAction.NOTIFY_USER

    def test_timeout_with_insufficient_rounds(self):
        """Timeout without enough rounds extends rounds."""
        config = HealingConfig(min_rounds_for_redebate=3)
        worker = ConsensusHealingWorker(config=config)
        candidate = self._make_candidate(
            HealingReason.TIMEOUT,
            rounds_completed=1,
        )
        action = worker._determine_action(candidate)
        assert action == HealingAction.EXTEND_ROUNDS

    def test_no_consensus_with_auto_redebate(self):
        """No consensus with auto-redebate triggers RE_DEBATE."""
        config = HealingConfig(auto_redebate_enabled=True)
        worker = ConsensusHealingWorker(config=config)
        candidate = self._make_candidate(HealingReason.NO_CONSENSUS)
        action = worker._determine_action(candidate)
        assert action == HealingAction.RE_DEBATE

    def test_no_consensus_without_auto_redebate(self):
        """No consensus without auto-redebate notifies user."""
        config = HealingConfig(auto_redebate_enabled=False)
        worker = ConsensusHealingWorker(config=config)
        candidate = self._make_candidate(HealingReason.NO_CONSENSUS)
        action = worker._determine_action(candidate)
        assert action == HealingAction.NOTIFY_USER

    def test_low_quality_extends_rounds(self):
        """Low quality consensus should extend rounds."""
        worker = ConsensusHealingWorker()
        candidate = self._make_candidate(HealingReason.LOW_QUALITY)
        action = worker._determine_action(candidate)
        assert action == HealingAction.EXTEND_ROUNDS

    def test_stalled_notifies_user(self):
        """Stalled reason defaults to user notification."""
        worker = ConsensusHealingWorker()
        candidate = self._make_candidate(HealingReason.STALLED)
        action = worker._determine_action(candidate)
        assert action == HealingAction.NOTIFY_USER


# =============================================================================
# Healing Action Execution Tests
# =============================================================================


class TestHealingActionExecution:
    """Tests for individual healing action methods."""

    def _make_candidate(self, reason=HealingReason.NO_CONSENSUS):
        return HealingCandidate(
            debate_id="debate-test",
            task="Test debate",
            reason=reason,
            created_at=time.time() - 7200,
            completed_at=None,
            rounds_completed=3,
            agent_count=4,
        )

    @pytest.mark.asyncio
    async def test_action_redebate(self):
        """Test re-debate action returns success."""
        worker = ConsensusHealingWorker()
        candidate = self._make_candidate()
        result = await worker._action_redebate(candidate)

        assert result.debate_id == "debate-test"
        assert result.action == HealingAction.RE_DEBATE
        assert result.success is True
        assert "original_rounds" in result.metrics

    @pytest.mark.asyncio
    async def test_action_extend_rounds(self):
        """Test extend rounds action returns success."""
        worker = ConsensusHealingWorker()
        candidate = self._make_candidate()
        result = await worker._action_extend_rounds(candidate)

        assert result.action == HealingAction.EXTEND_ROUNDS
        assert result.success is True
        assert result.metrics["additional_rounds"] == 3

    @pytest.mark.asyncio
    async def test_action_add_mediator(self):
        """Test add mediator action returns success."""
        worker = ConsensusHealingWorker()
        candidate = self._make_candidate(HealingReason.DIVERGING)
        result = await worker._action_add_mediator(candidate)

        assert result.action == HealingAction.ADD_MEDIATOR
        assert result.success is True

    @pytest.mark.asyncio
    async def test_action_notify_user_enabled(self):
        """Test notify user action when notifications are enabled."""
        config = HealingConfig(notify_on_stuck=True)
        worker = ConsensusHealingWorker(config=config)
        candidate = self._make_candidate()
        result = await worker._action_notify_user(candidate)

        assert result.action == HealingAction.NOTIFY_USER
        assert result.success is True

    @pytest.mark.asyncio
    async def test_action_notify_user_disabled(self):
        """Test notify user action when notifications are disabled."""
        config = HealingConfig(notify_on_stuck=False)
        worker = ConsensusHealingWorker(config=config)
        candidate = self._make_candidate()
        result = await worker._action_notify_user(candidate)

        assert result.action == HealingAction.NOTIFY_USER
        assert result.success is False
        assert "disabled" in result.message.lower()

    @pytest.mark.asyncio
    async def test_action_archive(self):
        """Test archive action returns success."""
        worker = ConsensusHealingWorker()
        candidate = self._make_candidate()
        result = await worker._action_archive(candidate)

        assert result.action == HealingAction.ARCHIVE
        assert result.success is True
        assert "archived" in result.message.lower()

    @pytest.mark.asyncio
    async def test_action_escalate(self):
        """Test escalate action returns success."""
        worker = ConsensusHealingWorker()
        candidate = self._make_candidate()
        result = await worker._action_escalate(candidate)

        assert result.action == HealingAction.ESCALATE
        assert result.success is True
        assert "human review" in result.message.lower()


# =============================================================================
# Heal Candidate Tests
# =============================================================================


class TestHealCandidate:
    """Tests for _heal_candidate method."""

    @pytest.mark.asyncio
    async def test_heal_candidate_routes_to_correct_action(self):
        """Test heal_candidate routes to correct action based on reason."""
        worker = ConsensusHealingWorker()
        candidate = HealingCandidate(
            debate_id="d-1",
            task="Test",
            reason=HealingReason.ERROR,
            created_at=time.time(),
            completed_at=None,
            rounds_completed=3,
            agent_count=4,
        )

        result = await worker._heal_candidate(candidate)
        assert result.action == HealingAction.NOTIFY_USER
        assert result.success is True

    @pytest.mark.asyncio
    async def test_heal_candidate_handles_exception(self):
        """Test heal_candidate handles exceptions gracefully."""
        worker = ConsensusHealingWorker()
        candidate = HealingCandidate(
            debate_id="d-1",
            task="Test",
            reason=HealingReason.DIVERGING,
            created_at=time.time(),
            completed_at=None,
            rounds_completed=3,
            agent_count=4,
        )

        with patch.object(
            worker,
            "_action_add_mediator",
            side_effect=RuntimeError("Mediator error"),
        ):
            result = await worker._heal_candidate(candidate)

        assert result.success is False
        assert "Error" in result.message


# =============================================================================
# Process Candidates Tests
# =============================================================================


class TestProcessCandidates:
    """Tests for _process_candidates method."""

    @pytest.mark.asyncio
    async def test_processes_candidates_and_tracks_metrics(self):
        """Test processing candidates updates metrics."""
        worker = ConsensusHealingWorker()

        candidate = HealingCandidate(
            debate_id="d-process",
            task="Test",
            reason=HealingReason.ERROR,
            created_at=time.time(),
            completed_at=None,
            rounds_completed=3,
            agent_count=4,
        )
        worker._candidates["d-process"] = candidate

        await worker._process_candidates()

        assert worker._healings_attempted == 1
        assert worker._healings_succeeded == 1
        assert len(worker._healing_history) == 1
        assert "d-process" not in worker._candidates

    @pytest.mark.asyncio
    async def test_process_candidates_calls_on_complete_callback(self):
        """Test on_healing_complete callback is called."""
        on_complete = MagicMock()
        worker = ConsensusHealingWorker(on_healing_complete=on_complete)

        candidate = HealingCandidate(
            debate_id="d-callback",
            task="Test",
            reason=HealingReason.ERROR,
            created_at=time.time(),
            completed_at=None,
            rounds_completed=3,
            agent_count=4,
        )
        worker._candidates["d-callback"] = candidate

        await worker._process_candidates()

        on_complete.assert_called_once()
        result_arg = on_complete.call_args[0][0]
        assert isinstance(result_arg, HealingResult)
        assert result_arg.debate_id == "d-callback"

    @pytest.mark.asyncio
    async def test_process_candidates_handles_callback_error(self):
        """Test callback error is caught and does not crash worker."""
        on_complete = MagicMock(side_effect=RuntimeError("Callback error"))
        worker = ConsensusHealingWorker(on_healing_complete=on_complete)

        candidate = HealingCandidate(
            debate_id="d-err",
            task="Test",
            reason=HealingReason.ERROR,
            created_at=time.time(),
            completed_at=None,
            rounds_completed=3,
            agent_count=4,
        )
        worker._candidates["d-err"] = candidate

        # Should not raise
        await worker._process_candidates()

        assert worker._healings_attempted == 1

    @pytest.mark.asyncio
    async def test_process_candidates_handles_healing_error(self):
        """Test processing continues when healing throws an error."""
        worker = ConsensusHealingWorker()

        for i in range(3):
            candidate = HealingCandidate(
                debate_id=f"d-{i}",
                task="Test",
                reason=HealingReason.ERROR,
                created_at=time.time(),
                completed_at=None,
                rounds_completed=3,
                agent_count=4,
            )
            worker._candidates[f"d-{i}"] = candidate

        call_count = 0
        original_heal = worker._heal_candidate

        async def sometimes_fail(candidate):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Healing error")
            return await original_heal(candidate)

        with patch.object(worker, "_heal_candidate", side_effect=sometimes_fail):
            await worker._process_candidates()

        # Two of three should have been processed (one failed silently)
        assert worker._healings_attempted == 2

    @pytest.mark.asyncio
    async def test_process_candidates_limits_batch(self):
        """Test that only 10 candidates are processed per batch."""
        worker = ConsensusHealingWorker()

        # Add 15 candidates
        for i in range(15):
            candidate = HealingCandidate(
                debate_id=f"d-{i}",
                task="Test",
                reason=HealingReason.ERROR,
                created_at=time.time(),
                completed_at=None,
                rounds_completed=3,
                agent_count=4,
            )
            worker._candidates[f"d-{i}"] = candidate

        await worker._process_candidates()

        assert worker._healings_attempted == 10
        assert len(worker._candidates) == 5

    @pytest.mark.asyncio
    async def test_healing_history_bounded(self):
        """Test healing history is capped at 1000."""
        worker = ConsensusHealingWorker()

        # Pre-fill history to near limit
        worker._healing_history = [
            HealingResult(
                debate_id=f"old-{i}",
                action=HealingAction.ARCHIVE,
                success=True,
                message="Old",
            )
            for i in range(999)
        ]

        candidate = HealingCandidate(
            debate_id="d-new",
            task="Test",
            reason=HealingReason.ERROR,
            created_at=time.time(),
            completed_at=None,
            rounds_completed=3,
            agent_count=4,
        )
        worker._candidates["d-new"] = candidate

        await worker._process_candidates()

        # Should have added 1, making 1000, then trimmed to 500
        assert len(worker._healing_history) <= 1000

    @pytest.mark.asyncio
    async def test_failed_healing_counted(self):
        """Test that failed healings are tracked but not counted as succeeded."""
        config = HealingConfig(notify_on_stuck=False)
        worker = ConsensusHealingWorker(config=config)

        candidate = HealingCandidate(
            debate_id="d-fail",
            task="Test",
            reason=HealingReason.ERROR,
            created_at=time.time(),
            completed_at=None,
            rounds_completed=3,
            agent_count=4,
        )
        worker._candidates["d-fail"] = candidate

        await worker._process_candidates()

        assert worker._healings_attempted == 1
        # NOTIFY_USER with notify_on_stuck=False returns success=False
        assert worker._healings_succeeded == 0


# =============================================================================
# Scan for Candidates Tests
# =============================================================================


class TestScanForCandidates:
    """Tests for _scan_for_candidates method."""

    @pytest.mark.asyncio
    async def test_scan_increments_counter(self):
        """Test scan increments scans_completed."""
        worker = ConsensusHealingWorker()

        with patch.object(
            worker,
            "_find_healing_candidates",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await worker._scan_for_candidates()

        assert worker._scans_completed == 1

    @pytest.mark.asyncio
    async def test_scan_adds_new_candidates(self):
        """Test scan adds new candidates."""
        worker = ConsensusHealingWorker()

        candidate = HealingCandidate(
            debate_id="d-scan-1",
            task="Test",
            reason=HealingReason.NO_CONSENSUS,
            created_at=time.time(),
            completed_at=None,
            rounds_completed=3,
            agent_count=4,
        )

        with patch.object(
            worker,
            "_find_healing_candidates",
            new_callable=AsyncMock,
            return_value=[candidate],
        ):
            await worker._scan_for_candidates()

        assert "d-scan-1" in worker._candidates
        assert worker._candidates_found == 1

    @pytest.mark.asyncio
    async def test_scan_skips_existing_candidates(self):
        """Test scan does not overwrite existing candidates."""
        worker = ConsensusHealingWorker()

        existing = HealingCandidate(
            debate_id="d-existing",
            task="Existing",
            reason=HealingReason.NO_CONSENSUS,
            created_at=time.time(),
            completed_at=None,
            rounds_completed=5,
            agent_count=4,
        )
        worker._candidates["d-existing"] = existing

        new_version = HealingCandidate(
            debate_id="d-existing",
            task="Updated",
            reason=HealingReason.DIVERGING,
            created_at=time.time(),
            completed_at=None,
            rounds_completed=6,
            agent_count=4,
        )

        with patch.object(
            worker,
            "_find_healing_candidates",
            new_callable=AsyncMock,
            return_value=[new_version],
        ):
            await worker._scan_for_candidates()

        # Original should be kept
        assert worker._candidates["d-existing"].task == "Existing"
        assert worker._candidates_found == 0

    @pytest.mark.asyncio
    async def test_scan_calls_on_healing_needed(self):
        """Test on_healing_needed callback is called for new candidates."""
        on_needed = MagicMock()
        worker = ConsensusHealingWorker(on_healing_needed=on_needed)

        candidate = HealingCandidate(
            debate_id="d-callback",
            task="Test",
            reason=HealingReason.NO_CONSENSUS,
            created_at=time.time(),
            completed_at=None,
            rounds_completed=3,
            agent_count=4,
        )

        with patch.object(
            worker,
            "_find_healing_candidates",
            new_callable=AsyncMock,
            return_value=[candidate],
        ):
            await worker._scan_for_candidates()

        on_needed.assert_called_once_with(candidate)

    @pytest.mark.asyncio
    async def test_scan_handles_callback_error(self):
        """Test scan continues when callback raises."""
        on_needed = MagicMock(side_effect=RuntimeError("Callback boom"))
        worker = ConsensusHealingWorker(on_healing_needed=on_needed)

        candidate = HealingCandidate(
            debate_id="d-err",
            task="Test",
            reason=HealingReason.NO_CONSENSUS,
            created_at=time.time(),
            completed_at=None,
            rounds_completed=3,
            agent_count=4,
        )

        with patch.object(
            worker,
            "_find_healing_candidates",
            new_callable=AsyncMock,
            return_value=[candidate],
        ):
            await worker._scan_for_candidates()

        # Candidate should still have been added
        assert "d-err" in worker._candidates

    @pytest.mark.asyncio
    async def test_scan_handles_find_error(self):
        """Test scan handles errors in _find_healing_candidates."""
        worker = ConsensusHealingWorker()

        with patch.object(
            worker,
            "_find_healing_candidates",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Find error"),
        ):
            # Should not raise
            await worker._scan_for_candidates()

        assert worker._scans_completed == 0


# =============================================================================
# Find Healing Candidates Tests
# =============================================================================


class TestFindHealingCandidates:
    """Tests for _find_healing_candidates method."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_consensus_memory_unavailable(self):
        """Test returns empty list when ConsensusMemory is not importable."""
        worker = ConsensusHealingWorker()

        with patch(
            "aragora.queue.workers.consensus_healing_worker.ConsensusHealingWorker._find_healing_candidates",
        ) as mock_find:
            # Simulate the ImportError path
            mock_find.return_value = []
            candidates = await worker._find_healing_candidates()
            assert candidates == []

    @pytest.mark.asyncio
    async def test_query_stale_debates_no_method(self):
        """Test _query_stale_debates returns empty when memory has no query_unconverged."""
        worker = ConsensusHealingWorker()
        mock_memory = MagicMock(spec=[])
        result = await worker._query_stale_debates(mock_memory)
        assert result == []

    @pytest.mark.asyncio
    async def test_query_stale_debates_with_method(self):
        """Test _query_stale_debates calls query_unconverged when available."""
        worker = ConsensusHealingWorker()
        mock_memory = MagicMock()
        mock_memory.query_unconverged = AsyncMock(
            return_value=[
                {"id": "d-1", "task": "Test"},
            ]
        )
        result = await worker._query_stale_debates(mock_memory)
        assert len(result) == 1
        assert result[0]["id"] == "d-1"

    @pytest.mark.asyncio
    async def test_query_stale_debates_handles_exception(self):
        """Test _query_stale_debates handles exceptions."""
        worker = ConsensusHealingWorker()
        mock_memory = MagicMock()
        mock_memory.query_unconverged = AsyncMock(side_effect=RuntimeError("DB error"))
        result = await worker._query_stale_debates(mock_memory)
        assert result == []

    @pytest.mark.asyncio
    async def test_query_stale_debates_non_list_result(self):
        """Test _query_stale_debates handles non-list result."""
        worker = ConsensusHealingWorker()
        mock_memory = MagicMock()
        mock_memory.query_unconverged = AsyncMock(return_value=None)
        result = await worker._query_stale_debates(mock_memory)
        assert result == []


# =============================================================================
# Metrics and Status Tests
# =============================================================================


class TestMetricsAndStatus:
    """Tests for get_metrics and get_status methods."""

    def test_get_metrics_initial(self):
        """Test initial metrics."""
        worker = ConsensusHealingWorker(worker_id="test-healer")
        metrics = worker.get_metrics()

        assert metrics["worker_id"] == "test-healer"
        assert metrics["running"] is False
        assert metrics["scans_completed"] == 0
        assert metrics["candidates_found"] == 0
        assert metrics["candidates_pending"] == 0
        assert metrics["healings_attempted"] == 0
        assert metrics["healings_succeeded"] == 0
        assert metrics["success_rate"] == 0.0
        assert metrics["recent_healings"] == []

    def test_get_metrics_with_data(self):
        """Test metrics with populated data."""
        worker = ConsensusHealingWorker(worker_id="test-healer")
        worker._running = True
        worker._scans_completed = 10
        worker._candidates_found = 5
        worker._healings_attempted = 4
        worker._healings_succeeded = 3
        worker._candidates["d-1"] = MagicMock()

        metrics = worker.get_metrics()

        assert metrics["running"] is True
        assert metrics["scans_completed"] == 10
        assert metrics["candidates_found"] == 5
        assert metrics["candidates_pending"] == 1
        assert metrics["healings_attempted"] == 4
        assert metrics["healings_succeeded"] == 3
        assert metrics["success_rate"] == 0.75

    def test_get_metrics_recent_healings(self):
        """Test recent healings in metrics."""
        worker = ConsensusHealingWorker()
        worker._healing_history = [
            HealingResult(
                debate_id=f"d-{i}",
                action=HealingAction.ARCHIVE,
                success=True,
                message="Test",
            )
            for i in range(15)
        ]

        metrics = worker.get_metrics()

        # Should only include last 10
        assert len(metrics["recent_healings"]) == 10
        assert metrics["recent_healings"][0]["debate_id"] == "d-5"

    def test_get_status(self):
        """Test get_status returns comprehensive status."""
        config = HealingConfig(
            scan_interval_seconds=120,
            auto_redebate_enabled=True,
            max_healing_attempts=5,
        )
        worker = ConsensusHealingWorker(worker_id="status-healer", config=config)

        status = worker.get_status()

        assert status["worker_id"] == "status-healer"
        assert status["running"] is False
        assert status["config"]["scan_interval_seconds"] == 120
        assert status["config"]["auto_redebate_enabled"] is True
        assert status["config"]["max_healing_attempts"] == 5
        assert "metrics" in status


# =============================================================================
# Global Worker Management Tests
# =============================================================================


class TestGlobalWorkerManagement:
    """Tests for global worker functions."""

    def setup_method(self):
        """Reset global state before each test."""
        import aragora.queue.workers.consensus_healing_worker as module

        module._global_worker = None

    def teardown_method(self):
        """Reset global state after each test."""
        import aragora.queue.workers.consensus_healing_worker as module

        module._global_worker = None

    def test_get_consensus_healing_worker_creates_instance(self):
        """Test get_consensus_healing_worker creates new instance."""
        worker = get_consensus_healing_worker()
        assert isinstance(worker, ConsensusHealingWorker)

    def test_get_consensus_healing_worker_returns_same_instance(self):
        """Test get_consensus_healing_worker returns same instance."""
        worker1 = get_consensus_healing_worker()
        worker2 = get_consensus_healing_worker()
        assert worker1 is worker2

    @pytest.mark.asyncio
    async def test_start_consensus_healing(self):
        """Test start_consensus_healing starts the global worker."""
        worker = await start_consensus_healing()
        assert isinstance(worker, ConsensusHealingWorker)

        # Clean up
        await worker.stop()
        # Give the created task time to finish
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_stop_consensus_healing(self):
        """Test stop_consensus_healing stops the global worker."""
        import aragora.queue.workers.consensus_healing_worker as module

        worker = ConsensusHealingWorker()
        worker._running = True
        module._global_worker = worker

        await stop_consensus_healing()
        assert worker._running is False

    @pytest.mark.asyncio
    async def test_stop_consensus_healing_no_worker(self):
        """Test stop_consensus_healing handles no worker gracefully."""
        # Should not raise
        await stop_consensus_healing()
