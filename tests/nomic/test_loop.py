"""
Tests for Nomic Loop orchestration.

Full loop integration tests:
- End-to-end cycle execution
- Phase transitions
- Error handling
- Safety enforcement
"""

import asyncio
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

        with patch.object(loop, 'run_context_phase', new_callable=AsyncMock) as mock_context:
            with patch.object(loop, 'run_debate_phase', new_callable=AsyncMock) as mock_debate:
                with patch.object(loop, 'run_design_phase', new_callable=AsyncMock) as mock_design:
                    with patch.object(loop, 'run_implement_phase', new_callable=AsyncMock) as mock_impl:
                        with patch.object(loop, 'run_verify_phase', new_callable=AsyncMock) as mock_verify:
                            # Track phase execution
                            mock_context.side_effect = lambda: phases_executed.append("context") or {"success": True}
                            mock_debate.side_effect = lambda *a, **kw: phases_executed.append("debate") or {"consensus": True}
                            mock_design.side_effect = lambda *a, **kw: phases_executed.append("design") or {"approved": True}
                            mock_impl.side_effect = lambda *a, **kw: phases_executed.append("implement") or {"success": True}
                            mock_verify.side_effect = lambda *a, **kw: phases_executed.append("verify") or {"passed": True}

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

        with patch.object(loop, 'run_context_phase', new_callable=AsyncMock) as mock_context:
            with patch.object(loop, 'run_debate_phase', new_callable=AsyncMock) as mock_debate:
                with patch.object(loop, 'run_design_phase', new_callable=AsyncMock) as mock_design:
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

        with patch.object(loop, 'request_human_approval', new_callable=AsyncMock) as mock_approval:
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

        with patch.object(loop, 'run_context_phase', new_callable=AsyncMock) as mock_context:
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

        with patch.object(loop, 'run_context_phase', new_callable=AsyncMock) as mock_context:
            with patch.object(loop, 'run_debate_phase', new_callable=AsyncMock) as mock_debate:
                with patch.object(loop, 'create_checkpoint') as mock_checkpoint:
                    mock_context.return_value = {"success": True}
                    mock_debate.side_effect = Exception("Debate failed")

                    await loop.run_cycle()

                    # Should have created checkpoint
                    assert mock_checkpoint.called


class TestNomicLoopIntegration:
    """Integration tests for complete loop."""

    @pytest.mark.asyncio
    async def test_complete_successful_cycle(
        self, mock_aragora_path, mock_log_fn,
        mock_debate_result, mock_design_result, mock_implementation_result, mock_verification_result
    ):
        """Should complete a full successful cycle."""
        from aragora.nomic.loop import NomicLoop

        loop = NomicLoop(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        with patch.object(loop, 'run_context_phase', new_callable=AsyncMock) as mock_context:
            with patch.object(loop, 'run_debate_phase', new_callable=AsyncMock) as mock_debate:
                with patch.object(loop, 'run_design_phase', new_callable=AsyncMock) as mock_design:
                    with patch.object(loop, 'run_implement_phase', new_callable=AsyncMock) as mock_impl:
                        with patch.object(loop, 'run_verify_phase', new_callable=AsyncMock) as mock_verify:
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

        with patch.object(loop, 'run_cycle', new_callable=AsyncMock) as mock_cycle:
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

        with patch.object(loop, 'run_cycle', new_callable=AsyncMock) as mock_cycle:
            def fail_cycle():
                failure_count[0] += 1
                return {"success": False}

            mock_cycle.side_effect = fail_cycle

            await loop.run(max_cycles=10)

            # Should stop after max_consecutive_failures
            assert failure_count[0] <= 3  # At most max_consecutive_failures + 1
