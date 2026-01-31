"""
Tests for Nomic Loop orchestration.

Full loop integration tests:
- End-to-end cycle execution
- Phase transitions
- Error handling
- Safety enforcement
- Cross-cycle learning
- Checkpointing and recovery
"""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestNomicLoopInitialization:
    """Tests for NomicLoop initialization."""

    def test_init_with_required_args(self, mock_aragora_path):
        """Should initialize with required arguments."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
        )

        assert loop.aragora_path == mock_aragora_path

    def test_init_with_max_cycles(self, mock_aragora_path):
        """Should accept max cycles parameter."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            max_cycles=5,
        )

        assert loop.max_cycles == 5

    def test_init_with_safety_config(self, mock_aragora_path):
        """Should accept safety configuration."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            protected_files=["CLAUDE.md", "core.py"],
            require_human_approval=True,
        )

        assert "CLAUDE.md" in loop.protected_files
        assert loop.require_human_approval is True


class TestNomicLoopPhaseTransitions:
    """Tests for phase transitions."""

    @pytest.mark.asyncio
    async def test_transitions_through_all_phases(self, mock_aragora_path, mock_log_fn):
        """Should transition through all phases in order."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        phases_executed = []

        with patch.object(loop, "run_context_phase", new_callable=AsyncMock) as mock_context:
            with patch.object(loop, "run_debate_phase", new_callable=AsyncMock) as mock_debate:
                with patch.object(loop, "run_design_phase", new_callable=AsyncMock) as mock_design:
                    with patch.object(
                        loop, "run_implement_phase", new_callable=AsyncMock
                    ) as mock_impl:
                        with patch.object(
                            loop, "run_verify_phase", new_callable=AsyncMock
                        ) as mock_verify:
                            # Track phase execution
                            mock_context.side_effect = lambda: phases_executed.append(
                                "context"
                            ) or {"success": True}
                            mock_debate.side_effect = lambda *a, **kw: phases_executed.append(
                                "debate"
                            ) or {"consensus": True}
                            mock_design.side_effect = lambda *a, **kw: phases_executed.append(
                                "design"
                            ) or {"approved": True}
                            mock_impl.side_effect = lambda *a, **kw: phases_executed.append(
                                "implement"
                            ) or {"success": True}
                            mock_verify.side_effect = lambda *a, **kw: phases_executed.append(
                                "verify"
                            ) or {"passed": True}

                            await loop.run_cycle()

        assert phases_executed == ["context", "debate", "design", "implement", "verify"]

    @pytest.mark.asyncio
    async def test_stops_on_phase_failure(self, mock_aragora_path, mock_log_fn):
        """Should stop cycle when a phase fails."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        with patch.object(loop, "run_context_phase", new_callable=AsyncMock) as mock_context:
            with patch.object(loop, "run_debate_phase", new_callable=AsyncMock) as mock_debate:
                with patch.object(loop, "run_design_phase", new_callable=AsyncMock) as mock_design:
                    mock_context.return_value = {"success": True}
                    mock_debate.return_value = {"consensus": False}  # No consensus
                    mock_design.return_value = {"approved": True}

                    result = await loop.run_cycle()

                    # Should not proceed to design if no consensus
                    assert result.get("completed", True) is False or mock_design.call_count == 0


class TestNomicLoopSafety:
    """Tests for safety enforcement."""

    @pytest.mark.asyncio
    async def test_enforces_protected_files(self, mock_aragora_path, mock_log_fn):
        """Should prevent modifications to protected files."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            protected_files=["CLAUDE.md", "core.py"],
        )

        changes = {
            "files_modified": ["CLAUDE.md"],
        }

        result = loop.check_safety(changes)

        assert result["safe"] is False
        assert "protected" in result.get("reason", "").lower()

    @pytest.mark.asyncio
    async def test_enforces_change_limits(self, mock_aragora_path, mock_log_fn):
        """Should enforce limits on change volume."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            max_files_per_cycle=5,
        )

        changes = {
            "files_modified": [f"file{i}.py" for i in range(20)],
        }

        result = loop.check_safety(changes)

        assert result["safe"] is False
        assert "too many" in result.get("reason", "").lower()

    @pytest.mark.asyncio
    async def test_requires_human_approval_when_configured(self, mock_aragora_path, mock_log_fn):
        """Should require human approval when configured."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            require_human_approval=True,
        )

        with patch.object(loop, "request_human_approval", new_callable=AsyncMock) as mock_approval:
            mock_approval.return_value = True

            result = await loop.get_approval_for_changes({"files": ["test.py"]})

            mock_approval.assert_called_once()


class TestNomicLoopErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_handles_phase_exception(self, mock_aragora_path, mock_log_fn):
        """Should handle exceptions in phases gracefully."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        with patch.object(loop, "run_context_phase", new_callable=AsyncMock) as mock_context:
            mock_context.side_effect = Exception("Phase failed")

            result = await loop.run_cycle()

            # Should not crash, should return error result
            assert result is not None
            assert result.get("success", True) is False or "error" in result

    @pytest.mark.asyncio
    async def test_creates_checkpoint_on_error(self, mock_aragora_path, mock_log_fn, tmp_path):
        """Should create checkpoint when error occurs."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            checkpoint_dir=tmp_path,
        )

        with patch.object(loop, "run_context_phase", new_callable=AsyncMock) as mock_context:
            with patch.object(loop, "run_debate_phase", new_callable=AsyncMock) as mock_debate:
                with patch.object(loop, "create_checkpoint") as mock_checkpoint:
                    mock_context.return_value = {"success": True}
                    mock_debate.side_effect = Exception("Debate failed")

                    await loop.run_cycle()

                    # Should have created checkpoint
                    assert mock_checkpoint.called


class TestNomicLoopIntegration:
    """Integration tests for complete loop."""

    @pytest.mark.asyncio
    async def test_complete_successful_cycle(
        self,
        mock_aragora_path,
        mock_log_fn,
        mock_debate_result,
        mock_design_result,
        mock_implementation_result,
        mock_verification_result,
    ):
        """Should complete a full successful cycle."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        with patch.object(loop, "run_context_phase", new_callable=AsyncMock) as mock_context:
            with patch.object(loop, "run_debate_phase", new_callable=AsyncMock) as mock_debate:
                with patch.object(loop, "run_design_phase", new_callable=AsyncMock) as mock_design:
                    with patch.object(
                        loop, "run_implement_phase", new_callable=AsyncMock
                    ) as mock_impl:
                        with patch.object(
                            loop, "run_verify_phase", new_callable=AsyncMock
                        ) as mock_verify:
                            mock_context.return_value = {"success": True, "context": "gathered"}
                            mock_debate.return_value = mock_debate_result
                            mock_design.return_value = mock_design_result
                            mock_impl.return_value = mock_implementation_result
                            mock_verify.return_value = mock_verification_result

                            result = await loop.run_cycle()

                            assert result["success"] is True
                            assert result.get("verification", {}).get("passed", False) is True

    @pytest.mark.asyncio
    async def test_multiple_cycles(self, mock_aragora_path, mock_log_fn):
        """Should run multiple cycles when configured."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            max_cycles=3,
        )

        cycles_run = []

        with patch.object(loop, "run_cycle", new_callable=AsyncMock) as mock_cycle:
            mock_cycle.side_effect = lambda: cycles_run.append(1) or {"success": True}

            await loop.run(max_cycles=3)

            assert len(cycles_run) == 3

    @pytest.mark.asyncio
    async def test_stops_on_consecutive_failures(self, mock_aragora_path, mock_log_fn):
        """Should stop after consecutive failures."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            max_consecutive_failures=2,
        )

        failure_count = [0]

        with patch.object(loop, "run_cycle", new_callable=AsyncMock) as mock_cycle:

            def fail_cycle():
                failure_count[0] += 1
                return {"success": False}

            mock_cycle.side_effect = fail_cycle

            await loop.run(max_cycles=10)

            # Should stop after max_consecutive_failures
            assert failure_count[0] <= 3  # At most max_consecutive_failures + 1


class TestNomicLoopCheckpointing:
    """Tests for checkpoint and recovery functionality."""

    def test_create_checkpoint_returns_data(self, mock_aragora_path, mock_log_fn, tmp_path):
        """Should create checkpoint with required data."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            checkpoint_dir=tmp_path,
        )

        # Set up some state
        loop._current_cycle_id = "test123"
        loop._cycle_count = 5
        loop._cycle_context = {"phase": "debate", "data": "test"}

        checkpoint = loop.create_checkpoint()

        assert checkpoint["cycle_id"] == "test123"
        assert checkpoint["cycle_count"] == 5
        assert "timestamp" in checkpoint
        assert checkpoint["context"]["phase"] == "debate"

    def test_create_checkpoint_saves_to_disk(self, mock_aragora_path, mock_log_fn, tmp_path):
        """Should save checkpoint to disk when checkpoint_dir is set."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            checkpoint_dir=tmp_path,
        )

        loop._current_cycle_id = "disk123"
        loop._cycle_count = 3
        loop._cycle_context = {"test": "data"}

        loop.create_checkpoint()

        # Check file was created
        checkpoint_file = tmp_path / "checkpoint_disk123.json"
        assert checkpoint_file.exists()

        # Verify content
        content = json.loads(checkpoint_file.read_text())
        assert content["cycle_id"] == "disk123"

    def test_restore_from_checkpoint(self, mock_aragora_path, mock_log_fn):
        """Should restore state from checkpoint."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        checkpoint = {
            "cycle_id": "restored123",
            "cycle_count": 10,
            "context": {"phase": "implement", "files": ["test.py"]},
        }

        loop.restore_from_checkpoint(checkpoint)

        assert loop._current_cycle_id == "restored123"
        assert loop._cycle_count == 10
        assert loop._cycle_context["phase"] == "implement"

    def test_checkpoints_accumulate(self, mock_aragora_path, mock_log_fn, tmp_path):
        """Should accumulate checkpoints in memory."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            checkpoint_dir=tmp_path,
        )

        loop._current_cycle_id = "cp1"
        loop.create_checkpoint()

        loop._current_cycle_id = "cp2"
        loop.create_checkpoint()

        assert len(loop._checkpoints) == 2


class TestNomicLoopCrossLearning:
    """Tests for cross-cycle learning functionality."""

    @pytest.mark.asyncio
    async def test_record_agent_contribution(self, mock_aragora_path, mock_log_fn):
        """Should record agent contributions to current cycle."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        # Initialize a cycle record
        from aragora.nomic.cycle_record import NomicCycleRecord

        loop._current_record = NomicCycleRecord(
            cycle_id="test",
            started_at=time.time(),
        )

        loop.record_agent_contribution(
            agent_name="claude",
            proposals_made=3,
            proposals_accepted=2,
            critiques_given=5,
            critiques_valuable=4,
        )

        assert "claude" in loop._current_record.agent_contributions
        contrib = loop._current_record.agent_contributions["claude"]
        assert contrib.proposals_made == 3
        assert contrib.proposals_accepted == 2

    @pytest.mark.asyncio
    async def test_record_surprise_event(self, mock_aragora_path, mock_log_fn):
        """Should record surprise events during cycle."""
        from aragora.nomic.loop import NomicLoop
        from aragora.nomic.cycle_record import NomicCycleRecord

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        loop._current_record = NomicCycleRecord(
            cycle_id="surprise_test",
            started_at=time.time(),
        )

        loop.record_surprise(
            phase="implement",
            description="Tests failed unexpectedly",
            expected="All tests pass",
            actual="3 tests failed",
            impact="high",
        )

        assert len(loop._current_record.surprise_events) == 1
        event = loop._current_record.surprise_events[0]
        assert event.phase == "implement"
        assert event.impact == "high"

    @pytest.mark.asyncio
    async def test_record_pattern_reinforcement(self, mock_aragora_path, mock_log_fn):
        """Should record pattern reinforcements."""
        from aragora.nomic.loop import NomicLoop
        from aragora.nomic.cycle_record import NomicCycleRecord

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        loop._current_record = NomicCycleRecord(
            cycle_id="pattern_test",
            started_at=time.time(),
        )

        loop.record_pattern_reinforcement(
            pattern_type="refactor",
            description="Extract method pattern",
            success=True,
            confidence=0.9,
        )

        assert len(loop._current_record.pattern_reinforcements) == 1
        pattern = loop._current_record.pattern_reinforcements[0]
        assert pattern.pattern_type == "refactor"
        assert pattern.success is True

    @pytest.mark.asyncio
    async def test_get_agent_trajectory(self, mock_aragora_path, mock_log_fn, tmp_path):
        """Should retrieve agent trajectory from store."""
        from aragora.nomic.loop import NomicLoop
        from aragora.nomic.cycle_store import CycleLearningStore
        from aragora.nomic.cycle_record import NomicCycleRecord

        # Create a store with some data
        db_path = str(tmp_path / "test_cycles.db")
        store = CycleLearningStore(db_path=db_path)

        # Add a cycle with agent contribution
        record = NomicCycleRecord(cycle_id="traj_test", started_at=time.time())
        record.add_agent_contribution("claude", proposals_made=5, proposals_accepted=4)
        record.mark_complete(success=True)
        store.save_cycle(record)

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )
        loop._cycle_store = store

        trajectory = loop.get_agent_trajectory("claude", n=10)

        assert len(trajectory) >= 1
        assert trajectory[0]["proposals_made"] == 5

    def test_no_recording_when_disabled(self, mock_aragora_path, mock_log_fn):
        """Should not record when cycle learning is disabled."""
        from aragora.nomic.loop import NomicLoop
        from aragora.nomic.cycle_record import NomicCycleRecord

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )
        loop._enable_cycle_learning = False
        loop._current_record = NomicCycleRecord(
            cycle_id="disabled_test",
            started_at=time.time(),
        )

        # This should not raise but also should not save
        with patch.object(loop, "_get_cycle_store") as mock_store:
            loop._finalize_cycle_record(success=True)
            mock_store.assert_not_called()


class TestNomicLoopPhases:
    """Tests for individual phase execution."""

    @pytest.mark.asyncio
    async def test_run_context_phase_default(self, mock_aragora_path, mock_log_fn):
        """Default context phase should return success."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        result = await loop.run_context_phase()

        assert result["success"] is True
        assert "context" in result

    @pytest.mark.asyncio
    async def test_run_debate_phase_default(self, mock_aragora_path, mock_log_fn):
        """Default debate phase should return consensus."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        result = await loop.run_debate_phase()

        assert result["consensus"] is True
        assert "confidence" in result

    @pytest.mark.asyncio
    async def test_run_design_phase_default(self, mock_aragora_path, mock_log_fn):
        """Default design phase should return approved."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        result = await loop.run_design_phase({"consensus": True})

        assert result["approved"] is True
        assert "design" in result

    @pytest.mark.asyncio
    async def test_run_implement_phase_default(self, mock_aragora_path, mock_log_fn):
        """Default implement phase should return success."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        result = await loop.run_implement_phase({"approved": True})

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_run_verify_phase_default(self, mock_aragora_path, mock_log_fn):
        """Default verify phase should return passed."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        result = await loop.run_verify_phase({"success": True})

        assert result["passed"] is True
        assert "test_results" in result


class TestNomicLoopSafetyExtended:
    """Extended safety tests."""

    def test_check_safety_with_protected_file_path(self, mock_aragora_path, mock_log_fn):
        """Should detect protected files by path."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            protected_files=["core.py"],
        )

        # Should block when file name matches
        changes = {"files_modified": ["aragora/core.py"]}
        result = loop.check_safety(changes)

        assert result["safe"] is False

    def test_check_safety_with_files_created(self, mock_aragora_path, mock_log_fn):
        """Should count created files toward limit."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            max_files_per_cycle=5,
        )

        changes = {
            "files_modified": ["a.py", "b.py"],
            "files_created": ["c.py", "d.py", "e.py", "f.py"],
        }
        result = loop.check_safety(changes)

        assert result["safe"] is False

    def test_check_safety_passes_valid_changes(self, mock_aragora_path, mock_log_fn):
        """Should pass valid changes."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            protected_files=["CLAUDE.md"],
            max_files_per_cycle=10,
        )

        changes = {
            "files_modified": ["test.py", "utils.py"],
            "files_created": ["new_feature.py"],
        }
        result = loop.check_safety(changes)

        assert result["safe"] is True

    @pytest.mark.asyncio
    async def test_get_approval_rejects_unsafe(self, mock_aragora_path, mock_log_fn):
        """Should reject approval for unsafe changes."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            protected_files=["CLAUDE.md"],
            require_human_approval=False,
        )

        changes = {"files_modified": ["CLAUDE.md"]}
        result = await loop.get_approval_for_changes(changes)

        assert result is False


class TestNomicLoopCycleContext:
    """Tests for cycle context management."""

    @pytest.mark.asyncio
    async def test_cycle_context_populated_during_run(self, mock_aragora_path, mock_log_fn):
        """Should populate cycle context during run."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        # Mock all phases
        with patch.object(loop, "run_context_phase", new_callable=AsyncMock) as ctx:
            with patch.object(loop, "run_debate_phase", new_callable=AsyncMock) as deb:
                with patch.object(loop, "run_design_phase", new_callable=AsyncMock) as des:
                    with patch.object(loop, "run_implement_phase", new_callable=AsyncMock) as imp:
                        with patch.object(loop, "run_verify_phase", new_callable=AsyncMock) as ver:
                            ctx.return_value = {"success": True}
                            deb.return_value = {"consensus": True}
                            des.return_value = {"approved": True}
                            imp.return_value = {"success": True}
                            ver.return_value = {"passed": True}

                            result = await loop.run_cycle()

        assert result["success"] is True
        assert "context" in loop._cycle_context
        assert "debate" in loop._cycle_context

    @pytest.mark.asyncio
    async def test_cycle_failed_returns_partial_context(self, mock_aragora_path, mock_log_fn):
        """Should return partial context on failure."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        with patch.object(loop, "run_context_phase", new_callable=AsyncMock) as ctx:
            with patch.object(loop, "run_debate_phase", new_callable=AsyncMock) as deb:
                ctx.return_value = {"success": True, "data": "context_data"}
                deb.return_value = {"consensus": False}

                result = await loop.run_cycle()

        assert result["success"] is False
        assert "partial_context" in result
        assert result["partial_context"]["context"]["data"] == "context_data"


class TestNomicLoopRunMethod:
    """Tests for the run() method."""

    @pytest.mark.asyncio
    async def test_run_returns_summary(self, mock_aragora_path, mock_log_fn):
        """Should return summary with all cycle results."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            max_cycles=2,
        )

        with patch.object(loop, "run_cycle", new_callable=AsyncMock) as mock_cycle:
            mock_cycle.side_effect = [
                {"success": True},
                {"success": False},
            ]

            result = await loop.run()

        assert result["cycles_run"] == 2
        assert result["successful_cycles"] == 1
        assert result["failed_cycles"] == 1
        assert len(result["results"]) == 2

    @pytest.mark.asyncio
    async def test_run_accepts_max_cycles_override(self, mock_aragora_path, mock_log_fn):
        """Should accept max_cycles override."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            max_cycles=10,  # Default high
        )

        with patch.object(loop, "run_cycle", new_callable=AsyncMock) as mock_cycle:
            mock_cycle.return_value = {"success": True}

            result = await loop.run(max_cycles=2)  # Override to 2

        assert result["cycles_run"] == 2

    @pytest.mark.asyncio
    async def test_run_resets_consecutive_failures(self, mock_aragora_path, mock_log_fn):
        """Should reset failure counter after success."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            max_consecutive_failures=2,
        )

        with patch.object(loop, "run_cycle", new_callable=AsyncMock) as mock_cycle:
            # Fail, Fail (would stop at 2), but Success resets
            mock_cycle.side_effect = [
                {"success": False},
                {"success": True},  # Resets counter
                {"success": False},
                {"success": False},  # Now stops
            ]

            result = await loop.run(max_cycles=10)

        # Should run all 4 cycles before stopping
        assert result["cycles_run"] == 4


class TestNomicLoopCycleFailed:
    """Tests for the _cycle_failed helper method."""

    def test_cycle_failed_returns_error_info(self, mock_aragora_path, mock_log_fn):
        """Should return proper failure info."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )
        loop._current_cycle_id = "fail_test"
        loop._cycle_context = {"context": {"data": "test"}}

        result = loop._cycle_failed("debate", {"error": "No consensus"})

        assert result["success"] is False
        assert result["completed"] is False
        assert result["failed_phase"] == "debate"
        assert result["cycle_id"] == "fail_test"
        assert "partial_context" in result

    def test_cycle_failed_with_custom_reason(self, mock_aragora_path, mock_log_fn):
        """Should use custom reason when provided."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )
        loop._current_cycle_id = "custom_fail"

        result = loop._cycle_failed("design", {}, "Custom failure reason")

        assert result["reason"] == "Custom failure reason"


class TestNomicLoopLogging:
    """Tests for logging functionality."""

    def test_log_uses_custom_function(self, mock_aragora_path):
        """Should use custom log function when provided."""
        from aragora.nomic.loop import NomicLoop

        log_messages = []

        def custom_log(msg):
            log_messages.append(msg)

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=custom_log,
        )

        loop._log("Test message")

        assert "Test message" in log_messages

    def test_log_uses_default_logger(self, mock_aragora_path):
        """Should use default logger when no log function provided."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
        )

        # Should not raise
        loop._log("Test message with default logger")


class TestNomicLoopCycleStore:
    """Tests for cycle store integration."""

    def test_get_cycle_store_creates_singleton(self, mock_aragora_path, mock_log_fn, tmp_path):
        """Should create cycle store on first access."""
        from aragora.nomic.loop import NomicLoop
        import os

        # Set data dir to tmp_path
        os.environ["ARAGORA_DATA_DIR"] = str(tmp_path)

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )
        loop._cycle_store = None  # Reset

        store = loop._get_cycle_store()

        assert store is not None
        assert loop._cycle_store is store

        # Clean up
        del os.environ["ARAGORA_DATA_DIR"]

    def test_get_cross_cycle_context_returns_empty_when_disabled(
        self, mock_aragora_path, mock_log_fn
    ):
        """Should return empty dict when learning disabled."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )
        loop._enable_cycle_learning = False

        context = loop._get_cross_cycle_context()

        assert context == {}
