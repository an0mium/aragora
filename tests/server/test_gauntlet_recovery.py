"""Tests for gauntlet run recovery after server restart.

Verifies that stale inflight gauntlet runs are properly detected
and marked as interrupted during server startup.
"""

import time
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest

from aragora.gauntlet.storage import GauntletInflightRun, GauntletStorage


class TestGauntletRecovery:
    """Test suite for gauntlet run recovery functionality."""

    def test_recover_stale_runs_marks_as_interrupted(self, tmp_path: Path):
        """Stale runs should be marked as interrupted."""
        # Setup storage
        storage = GauntletStorage(db_path=tmp_path / "gauntlet.db")

        # Create an inflight run that simulates being interrupted
        gauntlet_id = "test-run-001"
        storage.save_inflight(
            gauntlet_id=gauntlet_id,
            status="running",
            input_type="text",
            input_summary="Test input",
            input_hash="abc123",
            persona="default",
            agents=["claude", "gpt4"],
            profile="default",
        )
        # Update progress to simulate mid-run state
        storage.update_inflight_status(
            gauntlet_id=gauntlet_id,
            status="running",
            progress_percent=50.0,
            current_phase="evaluation",
        )

        # Verify inflight run exists
        inflight = storage.get_inflight(gauntlet_id)
        assert inflight is not None
        assert inflight.status == "running"

        # Import and run recovery
        from aragora.server.handlers.gauntlet import recover_stale_gauntlet_runs

        # Mock the storage getter
        with patch(
            "aragora.server.handlers.gauntlet._get_storage", return_value=storage
        ):
            recovered = recover_stale_gauntlet_runs(max_age_seconds=0)

        # Should have recovered 1 run
        assert recovered == 1

        # Verify status was updated
        inflight = storage.get_inflight(gauntlet_id)
        assert inflight is not None
        assert inflight.status == "interrupted"

    def test_recover_no_stale_runs(self, tmp_path: Path):
        """Recovery with no stale runs should return 0."""
        storage = GauntletStorage(db_path=tmp_path / "gauntlet.db")

        from aragora.server.handlers.gauntlet import recover_stale_gauntlet_runs

        with patch(
            "aragora.server.handlers.gauntlet._get_storage", return_value=storage
        ):
            recovered = recover_stale_gauntlet_runs(max_age_seconds=0)

        assert recovered == 0

    def test_recover_skips_completed_runs(self, tmp_path: Path):
        """Completed runs should not be recovered."""
        storage = GauntletStorage(db_path=tmp_path / "gauntlet.db")

        # Create a completed inflight run
        gauntlet_id = "test-completed-001"
        storage.save_inflight(
            gauntlet_id=gauntlet_id,
            status="completed",
            input_type="text",
            input_summary="Test input",
            input_hash="xyz789",
            persona="default",
            agents=["claude"],
            profile="default",
        )
        # Mark as completed
        storage.update_inflight_status(
            gauntlet_id=gauntlet_id,
            status="completed",
            progress_percent=100.0,
            current_phase="done",
        )

        from aragora.server.handlers.gauntlet import recover_stale_gauntlet_runs

        with patch(
            "aragora.server.handlers.gauntlet._get_storage", return_value=storage
        ):
            recovered = recover_stale_gauntlet_runs(max_age_seconds=0)

        # Should not recover completed runs
        assert recovered == 0

    def test_recover_multiple_stale_runs(self, tmp_path: Path):
        """Multiple stale runs should all be recovered."""
        storage = GauntletStorage(db_path=tmp_path / "gauntlet.db")

        # Create multiple inflight runs
        for i in range(3):
            status = "running" if i % 2 == 0 else "pending"
            storage.save_inflight(
                gauntlet_id=f"test-run-{i:03d}",
                status=status,
                input_type="text",
                input_summary=f"Test input {i}",
                input_hash=f"hash{i:03d}",
                persona="default",
                agents=["claude"],
                profile="default",
            )
            storage.update_inflight_status(
                gauntlet_id=f"test-run-{i:03d}",
                status=status,
                progress_percent=float(i * 30),
                current_phase="evaluation",
            )

        from aragora.server.handlers.gauntlet import recover_stale_gauntlet_runs

        with patch(
            "aragora.server.handlers.gauntlet._get_storage", return_value=storage
        ):
            recovered = recover_stale_gauntlet_runs(max_age_seconds=0)

        assert recovered == 3

        # Verify all were marked as interrupted
        for i in range(3):
            inflight = storage.get_inflight(f"test-run-{i:03d}")
            assert inflight is not None
            assert inflight.status == "interrupted"


class TestGauntletRecoveryStartup:
    """Test gauntlet recovery integration with startup sequence."""

    def test_init_gauntlet_run_recovery_returns_count(self):
        """init_gauntlet_run_recovery should return count of recovered runs."""
        from aragora.server.startup import init_gauntlet_run_recovery

        with patch(
            "aragora.server.handlers.gauntlet.recover_stale_gauntlet_runs", return_value=5
        ) as mock_recover:
            result = init_gauntlet_run_recovery()

        assert result == 5
        mock_recover.assert_called_once_with(max_age_seconds=7200)

    def test_init_gauntlet_run_recovery_handles_exception(self):
        """Should return 0 if recovery function raises exception."""
        from aragora.server.startup import init_gauntlet_run_recovery

        with patch(
            "aragora.server.handlers.gauntlet.recover_stale_gauntlet_runs",
            side_effect=RuntimeError("Something went wrong"),
        ):
            result = init_gauntlet_run_recovery()

        # Should return 0 on error without crashing
        assert result == 0

    def test_init_gauntlet_run_recovery_returns_zero_on_no_stale_runs(self):
        """Should return 0 when there are no stale runs."""
        from aragora.server.startup import init_gauntlet_run_recovery

        with patch(
            "aragora.server.handlers.gauntlet.recover_stale_gauntlet_runs", return_value=0
        ):
            result = init_gauntlet_run_recovery()

        assert result == 0
