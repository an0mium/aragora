"""
Tests for Consensus Healing Worker.

Tests the background job that monitors and heals debates without consensus.
"""

from __future__ import annotations

import asyncio
import time

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
)


class TestHealingAction:
    """Tests for HealingAction enum."""

    def test_all_actions_defined(self):
        """Test all healing actions exist."""
        actions = [
            HealingAction.RE_DEBATE,
            HealingAction.EXTEND_ROUNDS,
            HealingAction.ADD_MEDIATOR,
            HealingAction.NOTIFY_USER,
            HealingAction.ARCHIVE,
            HealingAction.ESCALATE,
        ]
        for action in actions:
            assert action.value is not None


class TestHealingReason:
    """Tests for HealingReason enum."""

    def test_all_reasons_defined(self):
        """Test all healing reasons exist."""
        reasons = [
            HealingReason.NO_CONSENSUS,
            HealingReason.STALLED,
            HealingReason.DIVERGING,
            HealingReason.TIMEOUT,
            HealingReason.ERROR,
            HealingReason.LOW_QUALITY,
        ]
        for reason in reasons:
            assert reason.value is not None


class TestHealingCandidate:
    """Tests for HealingCandidate dataclass."""

    def test_candidate_creation(self):
        """Test creating a healing candidate."""
        candidate = HealingCandidate(
            debate_id="debate-123",
            task="Design a rate limiter",
            reason=HealingReason.NO_CONSENSUS,
            created_at=time.time() - 3600,
            completed_at=time.time() - 3000,
            rounds_completed=5,
            agent_count=3,
            consensus_probability=0.3,
            convergence_trend="diverging",
        )

        assert candidate.debate_id == "debate-123"
        assert candidate.reason == HealingReason.NO_CONSENSUS
        assert candidate.rounds_completed == 5


class TestHealingResult:
    """Tests for HealingResult dataclass."""

    def test_result_creation(self):
        """Test creating a healing result."""
        result = HealingResult(
            debate_id="debate-123",
            action=HealingAction.NOTIFY_USER,
            success=True,
            message="User notified",
        )

        assert result.debate_id == "debate-123"
        assert result.action == HealingAction.NOTIFY_USER
        assert result.success is True

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = HealingResult(
            debate_id="debate-123",
            action=HealingAction.RE_DEBATE,
            success=True,
            message="Re-debate scheduled",
            new_debate_id="debate-456",
            metrics={"original_rounds": 5},
        )

        d = result.to_dict()
        assert d["debate_id"] == "debate-123"
        assert d["action"] == "re_debate"
        assert d["success"] is True
        assert d["new_debate_id"] == "debate-456"


class TestHealingConfig:
    """Tests for HealingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HealingConfig()

        assert config.scan_interval_seconds == 300
        assert config.min_age_for_healing_seconds == 3600
        assert config.max_healing_attempts == 3
        assert config.auto_redebate_enabled is False
        assert config.notify_on_stuck is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = HealingConfig(
            scan_interval_seconds=60,
            auto_redebate_enabled=True,
            max_healing_attempts=5,
        )

        assert config.scan_interval_seconds == 60
        assert config.auto_redebate_enabled is True
        assert config.max_healing_attempts == 5


class TestConsensusHealingWorker:
    """Tests for ConsensusHealingWorker."""

    def test_worker_creation(self):
        """Test creating a worker."""
        worker = ConsensusHealingWorker()

        assert worker.worker_id is not None
        assert worker._running is False

    def test_worker_with_custom_config(self):
        """Test creating worker with custom config."""
        config = HealingConfig(scan_interval_seconds=60)
        worker = ConsensusHealingWorker(config=config)

        assert worker.config.scan_interval_seconds == 60

    def test_worker_with_callbacks(self):
        """Test creating worker with callbacks."""
        candidates_found = []
        healings_done = []

        def on_healing_needed(candidate: HealingCandidate):
            candidates_found.append(candidate)

        def on_healing_complete(result: HealingResult):
            healings_done.append(result)

        worker = ConsensusHealingWorker(
            on_healing_needed=on_healing_needed,
            on_healing_complete=on_healing_complete,
        )

        assert worker.on_healing_needed is not None
        assert worker.on_healing_complete is not None

    @pytest.mark.asyncio
    async def test_worker_start_stop(self):
        """Test starting and stopping worker."""
        worker = ConsensusHealingWorker(config=HealingConfig(scan_interval_seconds=0.1))

        # Start worker in background
        task = asyncio.create_task(worker.start())

        # Let it run briefly
        await asyncio.sleep(0.2)

        # Stop worker
        await worker.stop()

        # Wait for task to complete
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()

        assert worker._running is False

    def test_determine_reason_error(self):
        """Test determining reason when error present."""
        worker = ConsensusHealingWorker()

        reason = worker._determine_reason({"error": "Something went wrong"})
        assert reason == HealingReason.ERROR

    def test_determine_reason_diverging(self):
        """Test determining reason when diverging."""
        worker = ConsensusHealingWorker()

        reason = worker._determine_reason({"convergence_trend": "diverging"})
        assert reason == HealingReason.DIVERGING

    def test_determine_reason_no_consensus(self):
        """Test determining reason when low consensus probability."""
        worker = ConsensusHealingWorker()

        reason = worker._determine_reason({"consensus_probability": 0.2})
        assert reason == HealingReason.NO_CONSENSUS

    def test_determine_reason_timeout(self):
        """Test determining reason when timed out."""
        worker = ConsensusHealingWorker()

        reason = worker._determine_reason(
            {
                "timed_out": True,
                "consensus_probability": 0.5,
            }
        )
        assert reason == HealingReason.TIMEOUT

    def test_determine_action_notify_user(self):
        """Test action determination defaults to notify."""
        worker = ConsensusHealingWorker()

        candidate = HealingCandidate(
            debate_id="test",
            task="Test task",
            reason=HealingReason.STALLED,
            created_at=time.time(),
            completed_at=None,
            rounds_completed=3,
            agent_count=3,
        )

        action = worker._determine_action(candidate)
        assert action == HealingAction.NOTIFY_USER

    def test_determine_action_archive_after_max_attempts(self):
        """Test archiving after max attempts."""
        config = HealingConfig(max_healing_attempts=3)
        worker = ConsensusHealingWorker(config=config)

        candidate = HealingCandidate(
            debate_id="test",
            task="Test task",
            reason=HealingReason.NO_CONSENSUS,
            created_at=time.time(),
            completed_at=None,
            rounds_completed=5,
            agent_count=3,
            metadata={"healing_attempts": 2},
        )

        action = worker._determine_action(candidate)
        assert action == HealingAction.ARCHIVE

    def test_determine_action_add_mediator_diverging(self):
        """Test adding mediator for diverging debates."""
        worker = ConsensusHealingWorker()

        candidate = HealingCandidate(
            debate_id="test",
            task="Test task",
            reason=HealingReason.DIVERGING,
            created_at=time.time(),
            completed_at=None,
            rounds_completed=3,
            agent_count=3,
        )

        action = worker._determine_action(candidate)
        assert action == HealingAction.ADD_MEDIATOR

    def test_determine_action_redebate_when_enabled(self):
        """Test re-debate action when auto-redebate enabled."""
        config = HealingConfig(auto_redebate_enabled=True)
        worker = ConsensusHealingWorker(config=config)

        candidate = HealingCandidate(
            debate_id="test",
            task="Test task",
            reason=HealingReason.NO_CONSENSUS,
            created_at=time.time(),
            completed_at=None,
            rounds_completed=5,
            agent_count=3,
        )

        action = worker._determine_action(candidate)
        assert action == HealingAction.RE_DEBATE

    @pytest.mark.asyncio
    async def test_action_redebate(self):
        """Test re-debate action execution."""
        worker = ConsensusHealingWorker()

        candidate = HealingCandidate(
            debate_id="test-123",
            task="Test task",
            reason=HealingReason.NO_CONSENSUS,
            created_at=time.time(),
            completed_at=None,
            rounds_completed=5,
            agent_count=3,
        )

        result = await worker._action_redebate(candidate)

        assert result.debate_id == "test-123"
        assert result.action == HealingAction.RE_DEBATE
        assert result.success is True

    @pytest.mark.asyncio
    async def test_action_extend_rounds(self):
        """Test extend rounds action execution."""
        worker = ConsensusHealingWorker()

        candidate = HealingCandidate(
            debate_id="test-123",
            task="Test task",
            reason=HealingReason.STALLED,
            created_at=time.time(),
            completed_at=None,
            rounds_completed=3,
            agent_count=3,
        )

        result = await worker._action_extend_rounds(candidate)

        assert result.debate_id == "test-123"
        assert result.action == HealingAction.EXTEND_ROUNDS
        assert result.success is True

    @pytest.mark.asyncio
    async def test_action_add_mediator(self):
        """Test add mediator action execution."""
        worker = ConsensusHealingWorker()

        candidate = HealingCandidate(
            debate_id="test-123",
            task="Test task",
            reason=HealingReason.DIVERGING,
            created_at=time.time(),
            completed_at=None,
            rounds_completed=5,
            agent_count=3,
        )

        result = await worker._action_add_mediator(candidate)

        assert result.debate_id == "test-123"
        assert result.action == HealingAction.ADD_MEDIATOR
        assert result.success is True

    @pytest.mark.asyncio
    async def test_action_notify_user(self):
        """Test notify user action execution."""
        worker = ConsensusHealingWorker()

        candidate = HealingCandidate(
            debate_id="test-123",
            task="Test task",
            reason=HealingReason.STALLED,
            created_at=time.time(),
            completed_at=None,
            rounds_completed=3,
            agent_count=3,
        )

        result = await worker._action_notify_user(candidate)

        assert result.debate_id == "test-123"
        assert result.action == HealingAction.NOTIFY_USER
        assert result.success is True

    @pytest.mark.asyncio
    async def test_action_notify_user_disabled(self):
        """Test notify user when notifications disabled."""
        config = HealingConfig(notify_on_stuck=False)
        worker = ConsensusHealingWorker(config=config)

        candidate = HealingCandidate(
            debate_id="test-123",
            task="Test task",
            reason=HealingReason.STALLED,
            created_at=time.time(),
            completed_at=None,
            rounds_completed=3,
            agent_count=3,
        )

        result = await worker._action_notify_user(candidate)

        assert result.success is False
        assert "disabled" in result.message

    @pytest.mark.asyncio
    async def test_action_archive(self):
        """Test archive action execution."""
        worker = ConsensusHealingWorker()

        candidate = HealingCandidate(
            debate_id="test-123",
            task="Test task",
            reason=HealingReason.NO_CONSENSUS,
            created_at=time.time(),
            completed_at=None,
            rounds_completed=5,
            agent_count=3,
        )

        result = await worker._action_archive(candidate)

        assert result.debate_id == "test-123"
        assert result.action == HealingAction.ARCHIVE
        assert result.success is True

    @pytest.mark.asyncio
    async def test_action_escalate(self):
        """Test escalate action execution."""
        worker = ConsensusHealingWorker()

        candidate = HealingCandidate(
            debate_id="test-123",
            task="Test task",
            reason=HealingReason.ERROR,
            created_at=time.time(),
            completed_at=None,
            rounds_completed=3,
            agent_count=3,
        )

        result = await worker._action_escalate(candidate)

        assert result.debate_id == "test-123"
        assert result.action == HealingAction.ESCALATE
        assert result.success is True

    def test_get_metrics(self):
        """Test getting worker metrics."""
        worker = ConsensusHealingWorker()

        metrics = worker.get_metrics()

        assert "worker_id" in metrics
        assert "running" in metrics
        assert "scans_completed" in metrics
        assert "candidates_found" in metrics
        assert "healings_attempted" in metrics
        assert "success_rate" in metrics

    def test_get_status(self):
        """Test getting worker status."""
        worker = ConsensusHealingWorker()

        status = worker.get_status()

        assert "worker_id" in status
        assert "running" in status
        assert "config" in status
        assert "metrics" in status


class TestGlobalFunctions:
    """Tests for global worker functions."""

    def test_get_consensus_healing_worker(self):
        """Test getting global worker."""
        worker = get_consensus_healing_worker()
        assert isinstance(worker, ConsensusHealingWorker)

    @pytest.mark.asyncio
    async def test_start_stop_consensus_healing(self):
        """Test starting and stopping global healing."""
        worker = await start_consensus_healing()

        # Let it start
        await asyncio.sleep(0.1)

        assert worker._running is True

        await stop_consensus_healing()

        assert worker._running is False


class TestIntegration:
    """Integration tests for consensus healing."""

    @pytest.mark.asyncio
    async def test_heal_candidate_full_cycle(self):
        """Test full healing cycle for a candidate."""
        results = []

        def on_complete(result: HealingResult):
            results.append(result)

        worker = ConsensusHealingWorker(on_healing_complete=on_complete)

        candidate = HealingCandidate(
            debate_id="integration-test",
            task="Test task",
            reason=HealingReason.NO_CONSENSUS,
            created_at=time.time() - 7200,
            completed_at=time.time() - 7000,
            rounds_completed=5,
            agent_count=3,
            consensus_probability=0.25,
        )

        result = await worker._heal_candidate(candidate)

        assert result.debate_id == "integration-test"
        assert result.success is True

    @pytest.mark.asyncio
    async def test_worker_processes_candidates(self):
        """Test worker processing of candidates."""
        worker = ConsensusHealingWorker(
            config=HealingConfig(
                min_age_for_healing_seconds=0,
                scan_interval_seconds=0.1,
            )
        )

        # Manually add a candidate
        candidate = HealingCandidate(
            debate_id="manual-test",
            task="Test task",
            reason=HealingReason.STALLED,
            created_at=time.time() - 100,
            completed_at=None,
            rounds_completed=3,
            agent_count=3,
        )
        worker._candidates["manual-test"] = candidate

        # Process candidates
        await worker._process_candidates()

        # Candidate should be removed after processing
        assert "manual-test" not in worker._candidates
        assert worker._healings_attempted == 1
