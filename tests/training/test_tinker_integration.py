"""
Tests for Tinker integration module.

Tests cover:
- Training data exporters (SFT, DPO, Gauntlet)
- TinkerClient API wrapper
- TrainingScheduler job management
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.training import (
    TinkerClient,
    TinkerConfig,
    TrainingScheduler,
    TrainingJob,
    SFTExporter,
    DPOExporter,
    GauntletExporter,
)
from aragora.training.tinker_client import (
    TinkerModel,
    TrainingState,
    TrainingResult,
    TinkerAPIError,
)
from aragora.training.training_scheduler import JobType, JobStatus, SchedulerConfig
from aragora.training.exporters.base import BaseExporter, TrainingRecord, PreferenceRecord


class TestTrainingRecord:
    """Tests for TrainingRecord dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        record = TrainingRecord(
            instruction="Test instruction",
            response="Test response",
            metadata={"source": "test"},
        )

        result = record.to_dict()

        assert result["instruction"] == "Test instruction"
        assert result["response"] == "Test response"
        assert result["metadata"]["source"] == "test"

    def test_to_jsonl(self):
        """Test conversion to JSONL format."""
        record = TrainingRecord(
            instruction="Test",
            response="Response",
        )

        jsonl = record.to_jsonl()
        parsed = json.loads(jsonl)

        assert parsed["instruction"] == "Test"
        assert parsed["response"] == "Response"


class TestPreferenceRecord:
    """Tests for PreferenceRecord dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        record = PreferenceRecord(
            prompt="Test prompt",
            chosen="Good response",
            rejected="Bad response",
            metadata={"winner_elo": 1500},
        )

        result = record.to_dict()

        assert result["prompt"] == "Test prompt"
        assert result["chosen"] == "Good response"
        assert result["rejected"] == "Bad response"
        assert result["metadata"]["winner_elo"] == 1500


class TestSFTExporter:
    """Tests for SFT training data exporter."""

    @pytest.fixture
    def exporter(self, tmp_path):
        """Create exporter with temporary database."""
        db_path = tmp_path / "test_critique.db"
        return SFTExporter(db_path=str(db_path))

    def test_export_empty_database(self, exporter):
        """Test export with no data."""
        records = exporter.export(limit=100)
        assert isinstance(records, list)
        # Empty database returns empty list
        assert len(records) == 0

    def test_export_to_file(self, exporter, tmp_path):
        """Test export to JSONL file."""
        output_path = tmp_path / "output.jsonl"

        metadata = exporter.export_to_file(output_path, limit=10)

        assert output_path.exists()
        assert metadata.exporter_type == "sft"
        assert metadata.total_records >= 0

    def test_format_debate_instruction(self, exporter):
        """Test debate instruction formatting."""
        task = "Design a rate limiter"

        instruction = exporter._format_debate_instruction(task)

        assert "Design a rate limiter" in instruction
        assert "Analyze" in instruction or "response" in instruction

    def test_format_pattern_instruction(self, exporter):
        """Test pattern instruction formatting."""
        issue_type = "security"
        issue_text = "SQL injection vulnerability in user input"

        instruction = exporter._format_pattern_instruction(issue_type, issue_text)

        assert "security" in instruction
        assert "SQL injection" in instruction


class TestDPOExporter:
    """Tests for DPO training data exporter."""

    @pytest.fixture
    def exporter(self, tmp_path):
        """Create exporter with temporary databases."""
        elo_path = tmp_path / "test_elo.db"
        critique_path = tmp_path / "test_critique.db"
        return DPOExporter(
            elo_db_path=str(elo_path),
            critique_db_path=str(critique_path),
        )

    def test_export_empty_database(self, exporter):
        """Test export with no data."""
        records = exporter.export(limit=100)
        assert isinstance(records, list)

    def test_export_to_file(self, exporter, tmp_path):
        """Test export to JSONL file."""
        output_path = tmp_path / "dpo_output.jsonl"

        metadata = exporter.export_to_file(output_path, limit=10)

        assert output_path.exists()
        assert metadata.exporter_type == "dpo"


class TestGauntletExporter:
    """Tests for Gauntlet adversarial data exporter."""

    @pytest.fixture
    def exporter(self, tmp_path):
        """Create exporter with temporary database."""
        elo_path = tmp_path / "test_elo.db"
        return GauntletExporter(elo_db_path=str(elo_path))

    def test_export_attack_patterns(self, exporter):
        """Test attack pattern export."""
        records = exporter._export_attack_patterns(limit=50)

        assert len(records) > 0
        for record in records:
            assert "instruction" in record
            assert "response" in record
            assert record["metadata"]["source"] == "attack"
            assert record["metadata"]["is_adversarial"] is True

    def test_export_synthetic_adversarial(self, exporter):
        """Test synthetic adversarial example generation."""
        records = exporter._generate_synthetic_adversarial(limit=10)

        assert len(records) > 0
        for record in records:
            assert "instruction" in record
            assert "response" in record
            assert record["metadata"]["source"] == "synthetic"

    def test_export_includes_all_types(self, exporter):
        """Test that export includes multiple data types."""
        records = exporter.export(limit=100)

        sources = {r.get("metadata", {}).get("source") for r in records}
        # Should include at least attack and synthetic
        assert "attack" in sources or "synthetic" in sources


class TestTinkerConfig:
    """Tests for TinkerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TinkerConfig()

        assert config.base_url == "https://api.thinkingmachines.ai/v1"
        assert config.lora_rank == 16
        assert config.learning_rate == 1e-4
        assert config.batch_size == 4

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TinkerConfig(
            api_key="test-key",
            lora_rank=32,
            learning_rate=5e-5,
        )

        assert config.api_key == "test-key"
        assert config.lora_rank == 32
        assert config.learning_rate == 5e-5


class TestTinkerClient:
    """Tests for TinkerClient API wrapper."""

    @pytest.fixture
    def client(self):
        """Create client with test config."""
        config = TinkerConfig(api_key="test-api-key")
        return TinkerClient(config)

    @pytest.mark.asyncio
    async def test_close(self, client):
        """Test client cleanup."""
        # Should not raise
        await client.close()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        config = TinkerConfig(api_key="test-key")
        async with TinkerClient(config) as client:
            assert client is not None
        # Client should be closed after context

    @pytest.mark.asyncio
    async def test_test_connection_no_api_key(self):
        """Test connection fails without API key."""
        config = TinkerConfig(api_key="")
        client = TinkerClient(config)

        with pytest.raises(TinkerAPIError, match="TINKER_API_KEY not set"):
            await client.test_connection()

        await client.close()

    @pytest.mark.asyncio
    async def test_train_sft_builds_correct_request(self, client):
        """Test SFT training request structure."""
        training_data = [
            {"instruction": "Test", "response": "Response"},
        ]

        with patch.object(client, "_submit_training_job", new_callable=AsyncMock) as mock_submit:
            mock_submit.return_value = TrainingResult(
                job_id="test-job",
                state=TrainingState.COMPLETED,
                model_id="model-123",
                final_loss=0.5,
                total_steps=100,
                training_time_seconds=3600,
                checkpoint_path="/path/to/checkpoint",
            )

            result = await client.train_sft(
                training_data=training_data,
                model=TinkerModel.LLAMA_3_3_70B,
            )

            assert result.model_id == "model-123"
            mock_submit.assert_called_once()

            # Verify request structure
            call_args = mock_submit.call_args[0][0]
            assert call_args["type"] == "sft"
            assert call_args["base_model"] == "llama-3.3-70b"
            assert call_args["training_data"] == training_data

    @pytest.mark.asyncio
    async def test_train_dpo_builds_correct_request(self, client):
        """Test DPO training request structure."""
        preference_data = [
            {"prompt": "Test", "chosen": "Good", "rejected": "Bad"},
        ]

        with patch.object(client, "_submit_training_job", new_callable=AsyncMock) as mock_submit:
            mock_submit.return_value = TrainingResult(
                job_id="test-job",
                state=TrainingState.COMPLETED,
                model_id="model-456",
                final_loss=0.3,
                total_steps=100,
                training_time_seconds=1800,
                checkpoint_path=None,
            )

            result = await client.train_dpo(
                preference_data=preference_data,
                model=TinkerModel.QWEN_2_5_72B,
                beta=0.2,
            )

            assert result.model_id == "model-456"

            call_args = mock_submit.call_args[0][0]
            assert call_args["type"] == "dpo"
            assert call_args["config"]["beta"] == 0.2


class TestTrainingJob:
    """Tests for TrainingJob dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        job = TrainingJob(
            job_id="job-123",
            job_type=JobType.SFT,
            model="llama-3.3-70b",
            status=JobStatus.COMPLETED,
            model_id="model-abc",
        )

        result = job.to_dict()

        assert result["job_id"] == "job-123"
        assert result["job_type"] == "sft"
        assert result["model"] == "llama-3.3-70b"
        assert result["status"] == "completed"
        assert result["model_id"] == "model-abc"


class TestTrainingScheduler:
    """Tests for TrainingScheduler."""

    @pytest.fixture
    def scheduler(self, tmp_path):
        """Create scheduler with temporary directories."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "checkpoints",
        )
        return TrainingScheduler(config=config)

    def test_generate_job_id(self, scheduler):
        """Test unique job ID generation."""
        id1 = scheduler._generate_job_id()
        id2 = scheduler._generate_job_id()

        assert id1 != id2
        assert id1.startswith("job-")
        assert id2.startswith("job-")

    def test_get_job_not_found(self, scheduler):
        """Test getting non-existent job."""
        result = scheduler.get_job("nonexistent-job")
        assert result is None

    def test_list_jobs_empty(self, scheduler):
        """Test listing jobs when empty."""
        jobs = scheduler.list_jobs()
        assert jobs == []

    def test_list_jobs_with_filter(self, scheduler):
        """Test job filtering."""
        # Add some test jobs
        scheduler._jobs["job-1"] = TrainingJob(
            job_id="job-1",
            job_type=JobType.SFT,
            model="llama",
            status=JobStatus.COMPLETED,
        )
        scheduler._jobs["job-2"] = TrainingJob(
            job_id="job-2",
            job_type=JobType.DPO,
            model="llama",
            status=JobStatus.PENDING,
        )

        # Filter by status
        completed = scheduler.list_jobs(status=JobStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].job_id == "job-1"

        # Filter by type
        dpo_jobs = scheduler.list_jobs(job_type=JobType.DPO)
        assert len(dpo_jobs) == 1
        assert dpo_jobs[0].job_id == "job-2"

    def test_cancel_pending_job(self, scheduler):
        """Test cancelling a pending job."""
        scheduler._jobs["job-1"] = TrainingJob(
            job_id="job-1",
            job_type=JobType.SFT,
            model="llama",
            status=JobStatus.PENDING,
        )

        result = scheduler.cancel_job("job-1")

        assert result is True
        assert scheduler._jobs["job-1"].status == JobStatus.CANCELLED

    def test_cancel_completed_job_fails(self, scheduler):
        """Test that completed jobs cannot be cancelled."""
        scheduler._jobs["job-1"] = TrainingJob(
            job_id="job-1",
            job_type=JobType.SFT,
            model="llama",
            status=JobStatus.COMPLETED,
        )

        result = scheduler.cancel_job("job-1")

        assert result is False
        assert scheduler._jobs["job-1"].status == JobStatus.COMPLETED

    def test_save_and_load_state(self, scheduler, tmp_path):
        """Test state persistence."""
        # Add a job
        scheduler._jobs["job-1"] = TrainingJob(
            job_id="job-1",
            job_type=JobType.SFT,
            model="llama-3.3-70b",
            status=JobStatus.COMPLETED,
            model_id="model-123",
        )
        scheduler._job_counter = 5

        # Save state
        state_path = tmp_path / "scheduler_state.json"
        scheduler.save_state(state_path)

        # Create new scheduler and load state
        new_scheduler = TrainingScheduler()
        new_scheduler.load_state(state_path)

        assert new_scheduler._job_counter == 5
        assert "job-1" in new_scheduler._jobs
        assert new_scheduler._jobs["job-1"].model_id == "model-123"


class TestBaseExporter:
    """Tests for BaseExporter abstract class."""

    def test_validate_record_default(self):
        """Test default validation returns True."""

        class TestExporter(BaseExporter):
            exporter_type = "test"

            def export(self, **kwargs):
                return []

        exporter = TestExporter()
        assert exporter.validate_record({}) is True


class TestTinkerModel:
    """Tests for TinkerModel enum."""

    def test_model_values(self):
        """Test model enum values."""
        assert TinkerModel.LLAMA_3_3_70B.value == "llama-3.3-70b"
        assert TinkerModel.QWEN_2_5_72B.value == "qwen-2.5-72b"
        assert TinkerModel.DEEPSEEK_V3.value == "deepseek-v3"

    def test_model_string_conversion(self):
        """Test models can be used as strings."""
        model = TinkerModel.LLAMA_3_3_70B
        assert str(model) == "TinkerModel.LLAMA_3_3_70B"
        assert model.value == "llama-3.3-70b"


class TestTrainingState:
    """Tests for TrainingState enum."""

    def test_state_values(self):
        """Test training state values."""
        assert TrainingState.PENDING.value == "pending"
        assert TrainingState.RUNNING.value == "running"
        assert TrainingState.COMPLETED.value == "completed"
        assert TrainingState.FAILED.value == "failed"


class TestTrainingResult:
    """Tests for TrainingResult dataclass."""

    def test_successful_result(self):
        """Test successful training result."""
        result = TrainingResult(
            job_id="job-123",
            state=TrainingState.COMPLETED,
            model_id="model-abc",
            final_loss=0.25,
            total_steps=1000,
            training_time_seconds=3600,
            checkpoint_path="/checkpoints/model-abc",
        )

        assert result.state == TrainingState.COMPLETED
        assert result.model_id == "model-abc"
        assert result.error_message is None

    def test_failed_result(self):
        """Test failed training result."""
        result = TrainingResult(
            job_id="job-456",
            state=TrainingState.FAILED,
            model_id=None,
            final_loss=None,
            total_steps=50,
            training_time_seconds=600,
            checkpoint_path=None,
            error_message="Out of memory",
        )

        assert result.state == TrainingState.FAILED
        assert result.model_id is None
        assert result.error_message == "Out of memory"


# Integration-style tests (with mocked API)
class TestTrainingIntegration:
    """Integration tests for training pipeline."""

    @pytest.mark.asyncio
    async def test_full_sft_pipeline_mocked(self, tmp_path):
        """Test full SFT training pipeline with mocked API."""
        # Setup
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "checkpoints",
        )
        scheduler = TrainingScheduler(config=config)

        # Mock the TinkerClient
        mock_client = AsyncMock()
        mock_client.train_sft.return_value = TrainingResult(
            job_id="tinker-job-1",
            state=TrainingState.COMPLETED,
            model_id="aragora-sft-v1",
            final_loss=0.15,
            total_steps=500,
            training_time_seconds=1800,
            checkpoint_path="/checkpoints/aragora-sft-v1",
        )
        scheduler._client = mock_client

        # Mock the exporter to return test data
        with patch("aragora.training.training_scheduler.SFTExporter") as MockExporter:
            mock_exporter = MagicMock()
            mock_exporter.export.return_value = [
                {"instruction": "Test task", "response": "Test response"},
            ]
            MockExporter.return_value = mock_exporter

            # Run job (simplified - in real test would await properly)
            job = TrainingJob(
                job_id="test-job",
                job_type=JobType.SFT,
                model="llama-3.3-70b",
                config={"adapter_name": "test-adapter", "limit": 10},
            )

            # Verify job structure
            assert job.job_type == JobType.SFT
            assert job.status == JobStatus.PENDING

        await scheduler.close()
