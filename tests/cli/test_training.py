"""
Tests for aragora.cli.training module.

Tests training CLI commands using Typer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip all tests if typer is not available
pytest.importorskip("typer")

from typer.testing import CliRunner  # noqa: E402

from aragora.cli.training import app  # noqa: E402


runner = CliRunner()


# ===========================================================================
# Test Fixtures and Mock Classes
# ===========================================================================


@dataclass
class MockExportMetadata:
    """Mock export metadata."""

    total_records: int = 100
    output_file: str = "test_output.jsonl"


@dataclass
class MockTrainingJob:
    """Mock training job."""

    job_id: str = "job-123"
    model: str = "llama-3.3-70b"
    status: MagicMock = field(default_factory=lambda: MagicMock(value="completed"))
    model_id: str | None = "model-456"
    error: str | None = None
    result: MagicMock | None = None


# ===========================================================================
# Tests: export-sft
# ===========================================================================


class TestExportSft:
    """Tests for export-sft command."""

    def test_export_sft_success(self, tmp_path):
        """Test export-sft with default options."""
        output_file = tmp_path / "sft_data.jsonl"

        mock_exporter = MagicMock()
        mock_exporter.export_to_file.return_value = MockExportMetadata(total_records=50)

        with patch("aragora.training.exporters.SFTExporter", return_value=mock_exporter):
            result = runner.invoke(app, ["export-sft", "-o", str(output_file)])

        assert result.exit_code == 0
        assert "Exported 50 SFT records" in result.stdout

    def test_export_sft_with_options(self, tmp_path):
        """Test export-sft with custom options."""
        output_file = tmp_path / "sft_data.jsonl"

        mock_exporter = MagicMock()
        mock_exporter.export_to_file.return_value = MockExportMetadata(total_records=100)

        with patch("aragora.training.exporters.SFTExporter", return_value=mock_exporter):
            result = runner.invoke(
                app,
                [
                    "export-sft",
                    "-o",
                    str(output_file),
                    "--min-confidence",
                    "0.8",
                    "--min-success-rate",
                    "0.7",
                    "--limit",
                    "500",
                ],
            )

        assert result.exit_code == 0
        mock_exporter.export_to_file.assert_called_once()
        call_args = mock_exporter.export_to_file.call_args
        assert call_args[1]["min_confidence"] == 0.8
        assert call_args[1]["min_success_rate"] == 0.7
        assert call_args[1]["limit"] == 500


# ===========================================================================
# Tests: export-dpo
# ===========================================================================


class TestExportDpo:
    """Tests for export-dpo command."""

    def test_export_dpo_success(self, tmp_path):
        """Test export-dpo with default options."""
        output_file = tmp_path / "dpo_data.jsonl"

        mock_exporter = MagicMock()
        mock_exporter.export_to_file.return_value = MockExportMetadata(total_records=30)

        with patch("aragora.training.exporters.DPOExporter", return_value=mock_exporter):
            result = runner.invoke(app, ["export-dpo", "-o", str(output_file)])

        assert result.exit_code == 0
        assert "Exported 30 DPO records" in result.stdout

    def test_export_dpo_with_options(self, tmp_path):
        """Test export-dpo with custom options."""
        output_file = tmp_path / "dpo_data.jsonl"

        mock_exporter = MagicMock()
        mock_exporter.export_to_file.return_value = MockExportMetadata(total_records=75)

        with patch("aragora.training.exporters.DPOExporter", return_value=mock_exporter):
            result = runner.invoke(
                app,
                [
                    "export-dpo",
                    "-o",
                    str(output_file),
                    "--min-elo-diff",
                    "100.0",
                    "--min-debates",
                    "5",
                    "--limit",
                    "250",
                ],
            )

        assert result.exit_code == 0
        call_args = mock_exporter.export_to_file.call_args
        assert call_args[1]["min_elo_difference"] == 100.0
        assert call_args[1]["min_debates"] == 5


# ===========================================================================
# Tests: export-gauntlet
# ===========================================================================


class TestExportGauntlet:
    """Tests for export-gauntlet command."""

    def test_export_gauntlet_success(self, tmp_path):
        """Test export-gauntlet with default options."""
        output_file = tmp_path / "gauntlet_data.jsonl"

        mock_exporter = MagicMock()
        mock_exporter.export_to_file.return_value = MockExportMetadata(total_records=20)

        with patch("aragora.training.exporters.GauntletExporter", return_value=mock_exporter):
            result = runner.invoke(app, ["export-gauntlet", "-o", str(output_file)])

        assert result.exit_code == 0
        assert "Exported 20 Gauntlet records" in result.stdout


# ===========================================================================
# Tests: export-all
# ===========================================================================


class TestExportAll:
    """Tests for export-all command."""

    def test_export_all_success(self, tmp_path):
        """Test export-all creates all files."""
        mock_sft = MagicMock()
        mock_sft.export_to_file.return_value = MockExportMetadata(total_records=100)

        mock_dpo = MagicMock()
        mock_dpo.export_to_file.return_value = MockExportMetadata(total_records=50)

        mock_gauntlet = MagicMock()
        mock_gauntlet.export_to_file.return_value = MockExportMetadata(total_records=25)

        with patch("aragora.training.exporters.SFTExporter", return_value=mock_sft):
            with patch("aragora.training.exporters.DPOExporter", return_value=mock_dpo):
                with patch(
                    "aragora.training.exporters.GauntletExporter", return_value=mock_gauntlet
                ):
                    result = runner.invoke(app, ["export-all", "-d", str(tmp_path / "training")])

        assert result.exit_code == 0
        assert "SFT: 100 records" in result.stdout
        assert "DPO: 50 records" in result.stdout
        assert "Gauntlet: 25 records" in result.stdout


# ===========================================================================
# Tests: test-connection
# ===========================================================================


class TestTestConnection:
    """Tests for test-connection command."""

    def test_no_api_key(self, monkeypatch):
        """Test error when no API key."""
        monkeypatch.delenv("TINKER_API_KEY", raising=False)

        result = runner.invoke(app, ["test-connection"])

        assert result.exit_code == 1
        assert "TINKER_API_KEY" in result.stdout

    def test_connection_success(self, monkeypatch):
        """Test successful connection."""
        monkeypatch.setenv("TINKER_API_KEY", "test-key")

        mock_client = MagicMock()
        mock_client.test_connection = AsyncMock()
        mock_client.close = AsyncMock()

        with patch("aragora.training.tinker_client.TinkerClient", return_value=mock_client):
            result = runner.invoke(app, ["test-connection"])

        assert result.exit_code == 0
        assert "Connection successful" in result.stdout

    def test_connection_failure(self, monkeypatch):
        """Test connection failure."""
        monkeypatch.setenv("TINKER_API_KEY", "test-key")

        from aragora.training.tinker_client import TinkerAPIError

        mock_client = MagicMock()
        mock_client.test_connection = AsyncMock(side_effect=TinkerAPIError("Auth failed"))
        mock_client.close = AsyncMock()

        with patch("aragora.training.tinker_client.TinkerClient", return_value=mock_client):
            result = runner.invoke(app, ["test-connection"])

        assert result.exit_code == 1
        assert "Connection failed" in result.stdout


# ===========================================================================
# Tests: train-sft
# ===========================================================================


class TestTrainSft:
    """Tests for train-sft command."""

    def test_train_sft_success(self):
        """Test successful SFT training."""
        mock_result = MagicMock()
        mock_result.final_loss = 0.05
        mock_result.training_time_seconds = 3600

        mock_job = MockTrainingJob(result=mock_result)

        mock_scheduler = MagicMock()
        mock_scheduler.schedule_sft = AsyncMock(return_value=mock_job)
        mock_scheduler.wait_for_job = AsyncMock(return_value=mock_job)
        mock_scheduler.close = AsyncMock()

        with patch("aragora.training.TrainingScheduler", return_value=mock_scheduler):
            result = runner.invoke(app, ["train-sft", "--model", "llama-3.3-70b"])

        assert result.exit_code == 0
        assert "Scheduled SFT job" in result.stdout
        assert "Training completed" in result.stdout

    def test_train_sft_no_wait(self):
        """Test SFT training without waiting."""
        mock_job = MockTrainingJob()
        mock_job.status.value = "pending"

        mock_scheduler = MagicMock()
        mock_scheduler.schedule_sft = AsyncMock(return_value=mock_job)
        mock_scheduler.close = AsyncMock()

        with patch("aragora.training.TrainingScheduler", return_value=mock_scheduler):
            result = runner.invoke(app, ["train-sft", "--no-wait"])

        assert result.exit_code == 0
        assert "Scheduled SFT job" in result.stdout
        # Should not contain "Training completed" since we didn't wait


# ===========================================================================
# Tests: train-dpo
# ===========================================================================


class TestTrainDpo:
    """Tests for train-dpo command."""

    def test_train_dpo_success(self):
        """Test successful DPO training."""
        mock_job = MockTrainingJob()

        mock_scheduler = MagicMock()
        mock_scheduler.schedule_dpo = AsyncMock(return_value=mock_job)
        mock_scheduler.wait_for_job = AsyncMock(return_value=mock_job)
        mock_scheduler.close = AsyncMock()

        with patch("aragora.training.TrainingScheduler", return_value=mock_scheduler):
            result = runner.invoke(app, ["train-dpo", "--beta", "0.2"])

        assert result.exit_code == 0
        assert "Scheduled DPO job" in result.stdout
        assert "Beta: 0.2" in result.stdout


# ===========================================================================
# Tests: train-combined
# ===========================================================================


class TestTrainCombined:
    """Tests for train-combined command."""

    def test_train_combined_success(self):
        """Test successful combined training."""
        mock_job = MockTrainingJob()

        mock_scheduler = MagicMock()
        mock_scheduler.schedule_combined = AsyncMock(return_value=mock_job)
        mock_scheduler.wait_for_job = AsyncMock(return_value=mock_job)
        mock_scheduler.close = AsyncMock()

        with patch("aragora.training.TrainingScheduler", return_value=mock_scheduler):
            result = runner.invoke(
                app, ["train-combined", "--sft-limit", "500", "--dpo-limit", "250"]
            )

        assert result.exit_code == 0
        assert "Scheduled combined job" in result.stdout
        assert "SFT (500 examples)" in result.stdout
        assert "DPO (250 examples)" in result.stdout


# ===========================================================================
# Tests: list-models
# ===========================================================================


class TestListModels:
    """Tests for list-models command."""

    def test_list_models_empty(self):
        """Test list-models with no models."""
        mock_client = MagicMock()
        mock_client.list_models = AsyncMock(return_value=[])
        mock_client.close = AsyncMock()

        with patch("aragora.training.tinker_client.TinkerClient", return_value=mock_client):
            result = runner.invoke(app, ["list-models"])

        assert result.exit_code == 0
        assert "No fine-tuned models found" in result.stdout

    def test_list_models_success(self):
        """Test list-models with results."""
        mock_client = MagicMock()
        mock_client.list_models = AsyncMock(
            return_value=[
                {"model_id": "model-1", "base_model": "llama-3.3-70b", "created_at": "2024-01-01"},
                {"model_id": "model-2", "base_model": "llama-3.3-70b", "created_at": "2024-01-02"},
            ]
        )
        mock_client.close = AsyncMock()

        with patch("aragora.training.tinker_client.TinkerClient", return_value=mock_client):
            result = runner.invoke(app, ["list-models"])

        assert result.exit_code == 0
        assert "Found 2 models" in result.stdout
        assert "model-1" in result.stdout
        assert "model-2" in result.stdout


# ===========================================================================
# Tests: sample
# ===========================================================================


class TestSample:
    """Tests for sample command."""

    def test_sample_success(self):
        """Test sample generation."""
        mock_client = MagicMock()
        mock_client.sample = AsyncMock(return_value="Generated text response")
        mock_client.close = AsyncMock()

        with patch("aragora.training.tinker_client.TinkerClient", return_value=mock_client):
            result = runner.invoke(
                app,
                [
                    "sample",
                    "What is the capital of France?",
                    "--temperature",
                    "0.5",
                    "--max-tokens",
                    "512",
                ],
            )

        assert result.exit_code == 0
        assert "Generated text response" in result.stdout


# ===========================================================================
# Tests: stats
# ===========================================================================


class TestStats:
    """Tests for stats command."""

    def test_stats_success(self):
        """Test stats display."""
        mock_store = MagicMock()
        mock_store.get_stats.return_value = {
            "total_debates": 100,
            "consensus_debates": 80,
            "total_critiques": 500,
            "total_patterns": 50,
            "avg_consensus_confidence": 0.85,
            "patterns_by_type": {"reasoning": 30, "evidence": 20},
        }

        mock_elo = MagicMock()
        mock_elo.get_stats.return_value = {
            "total_agents": 10,
            "total_matches": 500,
            "average_elo": 1050,
        }

        with patch("aragora.memory.store.CritiqueStore", return_value=mock_store):
            with patch("aragora.ranking.elo.EloSystem", return_value=mock_elo):
                result = runner.invoke(app, ["stats"])

        assert result.exit_code == 0
        assert "Training Data Statistics" in result.stdout
        assert "Total debates: 100" in result.stdout
        assert "Total agents: 10" in result.stdout
        assert "reasoning: 30" in result.stdout
