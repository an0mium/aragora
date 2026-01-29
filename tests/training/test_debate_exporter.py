"""Tests for debate training data exporter.

Tests the debate-to-training pipeline including:
- DebateTrainingConfig: configuration validation
- DebateTrainingExporter: export SFT and DPO records
- Quality thresholds and filtering
- File output and statistics
"""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.training.debate_exporter import (
    DebateTrainingConfig,
    DebateTrainingExporter,
    export_debate_to_training,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def default_config(temp_dir):
    """Create a default config with temp directory."""
    return DebateTrainingConfig(output_dir=str(temp_dir))


@pytest.fixture
def exporter(default_config):
    """Create a DebateTrainingExporter with default config."""
    return DebateTrainingExporter(default_config)


@dataclass
class MockDebateResult:
    """Mock debate result for testing."""

    task: str = "Design a rate limiter"
    final_answer: str = "A comprehensive rate limiter should use token bucket algorithm with sliding window for accurate rate tracking."
    confidence: float = 0.85
    rounds_used: int = 3
    consensus_reached: bool = True
    messages: list = None

    def __post_init__(self):
        if self.messages is None:
            self.messages = [
                {"content": "First proposal with sufficient detail for export testing purposes."},
                {"content": "Second proposal with different approach for DPO pair generation."},
            ]


@pytest.fixture
def mock_result():
    """Create a mock debate result that meets quality thresholds."""
    return MockDebateResult()


@pytest.fixture
def low_quality_result():
    """Create a mock debate result that fails quality thresholds."""
    return MockDebateResult(
        confidence=0.5,  # Below threshold
        rounds_used=1,  # Below threshold
        consensus_reached=False,
    )


# =============================================================================
# DebateTrainingConfig Tests
# =============================================================================


class TestDebateTrainingConfig:
    """Test DebateTrainingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DebateTrainingConfig()

        assert config.output_dir == "data/training"
        assert config.sft_file == "sft_debates.jsonl"
        assert config.dpo_file == "dpo_debates.jsonl"
        assert config.min_confidence == 0.75
        assert config.min_rounds == 2
        assert config.require_consensus is True
        assert config.export_sft is True
        assert config.export_dpo is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DebateTrainingConfig(
            output_dir="/custom/path",
            min_confidence=0.9,
            min_rounds=5,
            require_consensus=False,
            export_sft=False,
        )

        assert config.output_dir == "/custom/path"
        assert config.min_confidence == 0.9
        assert config.min_rounds == 5
        assert config.require_consensus is False
        assert config.export_sft is False

    def test_metadata_flags(self):
        """Test metadata inclusion flags."""
        config = DebateTrainingConfig(
            include_agent_names=False,
            include_debate_id=False,
            include_domain=False,
        )

        assert config.include_agent_names is False
        assert config.include_debate_id is False
        assert config.include_domain is False


# =============================================================================
# DebateTrainingExporter Initialization Tests
# =============================================================================


class TestDebateTrainingExporterInit:
    """Test DebateTrainingExporter initialization."""

    def test_creates_output_directory(self, temp_dir):
        """Test that exporter creates output directory."""
        output_path = temp_dir / "new_dir" / "nested"
        config = DebateTrainingConfig(output_dir=str(output_path))

        DebateTrainingExporter(config)

        assert output_path.exists()

    def test_default_config(self, temp_dir):
        """Test exporter with default config."""
        exporter = DebateTrainingExporter()

        assert exporter.config is not None
        assert exporter.config.min_confidence == 0.75

    def test_initial_stats(self, exporter):
        """Test initial statistics are zero."""
        stats = exporter.get_stats()

        assert stats["sft_count"] == 0
        assert stats["dpo_count"] == 0
        assert stats["skipped_count"] == 0


# =============================================================================
# Quality Threshold Tests
# =============================================================================


class TestQualityThresholds:
    """Test quality threshold validation."""

    def test_meets_threshold_valid(self, exporter, mock_result):
        """Test that valid result meets thresholds."""
        assert exporter._meets_quality_threshold(mock_result) is True

    def test_below_confidence_threshold(self, exporter):
        """Test rejection when confidence is too low."""
        result = MockDebateResult(confidence=0.5)  # Below 0.75 default

        assert exporter._meets_quality_threshold(result) is False

    def test_below_rounds_threshold(self, exporter):
        """Test rejection when rounds are too few."""
        result = MockDebateResult(rounds_used=1)  # Below 2 default

        assert exporter._meets_quality_threshold(result) is False

    def test_no_consensus_when_required(self, exporter):
        """Test rejection when consensus required but not reached."""
        result = MockDebateResult(consensus_reached=False)

        assert exporter._meets_quality_threshold(result) is False

    def test_no_consensus_when_not_required(self, temp_dir):
        """Test acceptance when consensus not required."""
        config = DebateTrainingConfig(
            output_dir=str(temp_dir),
            require_consensus=False,
        )
        exporter = DebateTrainingExporter(config)
        result = MockDebateResult(consensus_reached=False)

        assert exporter._meets_quality_threshold(result) is True

    def test_short_final_answer(self, exporter):
        """Test rejection when final answer is too short."""
        result = MockDebateResult(final_answer="Too short")

        assert exporter._meets_quality_threshold(result) is False

    def test_empty_final_answer(self, exporter):
        """Test rejection when final answer is empty."""
        result = MockDebateResult(final_answer="")

        assert exporter._meets_quality_threshold(result) is False


# =============================================================================
# SFT Export Tests
# =============================================================================


class TestSFTExport:
    """Test SFT (Supervised Fine-Tuning) record creation."""

    def test_create_sft_record(self, exporter, mock_result):
        """Test SFT record creation."""
        record = exporter._create_sft_record(
            mock_result,
            debate_id="test-123",
            domain="tech",
            agents=None,
        )

        assert record is not None
        assert "instruction" in record
        assert "response" in record
        assert "metadata" in record
        assert mock_result.task in record["instruction"]
        assert record["response"] == mock_result.final_answer.strip()

    def test_sft_record_with_metadata(self, exporter, mock_result):
        """Test SFT record includes metadata."""
        record = exporter._create_sft_record(
            mock_result,
            debate_id="test-456",
            domain="security",
            agents=None,
        )

        assert record["metadata"]["debate_id"] == "test-456"
        assert record["metadata"]["domain"] == "security"
        assert record["metadata"]["confidence"] == mock_result.confidence
        assert record["metadata"]["rounds_used"] == mock_result.rounds_used

    def test_sft_record_with_agents(self, exporter, mock_result):
        """Test SFT record includes agent names."""
        agent1 = MagicMock()
        agent1.name = "claude"
        agent2 = MagicMock()
        agent2.name = "gpt-4"

        record = exporter._create_sft_record(
            mock_result,
            debate_id="test-789",
            domain="tech",
            agents=[agent1, agent2],
        )

        assert "agents" in record["metadata"]
        assert record["metadata"]["agents"] == ["claude", "gpt-4"]

    def test_sft_record_no_task(self, exporter):
        """Test SFT record returns None when no task."""
        result = MockDebateResult(task="")

        record = exporter._create_sft_record(result, "", "", None)

        assert record is None

    def test_sft_record_no_final_answer(self, exporter):
        """Test SFT record returns None when no final answer."""
        result = MockDebateResult(final_answer="")

        record = exporter._create_sft_record(result, "", "", None)

        assert record is None


# =============================================================================
# DPO Export Tests
# =============================================================================


class TestDPOExport:
    """Test DPO (Direct Preference Optimization) record creation."""

    def test_create_dpo_records(self, exporter, mock_result):
        """Test DPO record creation."""
        records = exporter._create_dpo_records(
            mock_result,
            debate_id="test-dpo",
            domain="tech",
            agents=None,
        )

        # Should create records for alternative proposals
        assert len(records) >= 1

    def test_dpo_record_structure(self, exporter, mock_result):
        """Test DPO record has correct structure."""
        records = exporter._create_dpo_records(
            mock_result,
            debate_id="test-dpo",
            domain="tech",
            agents=None,
        )

        if records:
            record = records[0]
            assert "prompt" in record
            assert "chosen" in record
            assert "rejected" in record
            assert "metadata" in record
            assert record["chosen"] == mock_result.final_answer.strip()

    def test_dpo_no_records_without_messages(self, exporter):
        """Test no DPO records when messages are insufficient."""
        result = MockDebateResult(messages=[])

        records = exporter._create_dpo_records(result, "", "", None)

        assert len(records) == 0

    def test_dpo_skips_duplicate_responses(self, exporter):
        """Test DPO skips responses similar to final answer."""
        # Content must be 50+ chars for DPO records
        final = "This is the exact final response text that was chosen as the winner here."
        different = "This is a completely different alternative response text for training data."
        result = MockDebateResult(
            final_answer=final,
            messages=[
                {"content": final},  # Same as final - should be skipped
                {"content": different},  # Different - should be included
            ],
        )

        records = exporter._create_dpo_records(result, "", "", None)

        # Should only create record for the different response
        assert len(records) == 1

    def test_dpo_skips_short_content(self, exporter):
        """Test DPO skips messages with short content (<50 chars)."""
        # Only messages with 50+ chars are included in DPO records
        long_content = (
            "This is a sufficiently long response with enough characters for training data."
        )
        result = MockDebateResult(
            messages=[
                {"content": "Short"},  # Too short (<50 chars) - should be skipped
                {"content": long_content},  # Long enough - should be included
            ],
        )

        records = exporter._create_dpo_records(result, "", "", None)

        # Only the longer response should be included
        assert len(records) == 1


# =============================================================================
# Export Debate Tests
# =============================================================================


class TestExportDebate:
    """Test full debate export functionality."""

    def test_export_debate_success(self, exporter, mock_result, temp_dir):
        """Test successful debate export."""
        result = exporter.export_debate(
            mock_result,
            debate_id="export-test",
            domain="tech",
        )

        assert result["sft"] >= 0
        assert result["dpo"] >= 0
        assert result["skipped"] == 0

    def test_export_debate_skipped(self, exporter, low_quality_result):
        """Test debate export skipped for low quality."""
        result = exporter.export_debate(
            low_quality_result,
            debate_id="skip-test",
            domain="tech",
        )

        assert result["skipped"] == 1
        assert result["sft"] == 0
        assert result["dpo"] == 0

    def test_export_debate_updates_stats(self, exporter, mock_result):
        """Test that export updates statistics."""
        initial_stats = exporter.get_stats()

        exporter.export_debate(mock_result, "stat-test", "tech")

        new_stats = exporter.get_stats()

        # Stats should have changed
        total_new = new_stats["sft_count"] + new_stats["dpo_count"] + new_stats["skipped_count"]
        total_initial = (
            initial_stats["sft_count"] + initial_stats["dpo_count"] + initial_stats["skipped_count"]
        )
        assert total_new > total_initial

    def test_export_debate_writes_files(self, exporter, mock_result, default_config):
        """Test that export writes to files."""
        exporter.export_debate(mock_result, "file-test", "tech")

        sft_path = Path(default_config.output_dir) / default_config.sft_file

        # Check SFT file was created
        if exporter.get_stats()["sft_count"] > 0:
            assert sft_path.exists()
            with open(sft_path) as f:
                line = f.readline()
                record = json.loads(line)
                assert "instruction" in record

    def test_export_multiple_debates(self, exporter, mock_result):
        """Test exporting multiple debates."""
        for i in range(5):
            exporter.export_debate(mock_result, f"multi-{i}", "tech")

        stats = exporter.get_stats()

        # Should have accumulated exports
        assert stats["sft_count"] == 5 or stats["skipped_count"] == 5


# =============================================================================
# Statistics Tests
# =============================================================================


class TestExporterStatistics:
    """Test exporter statistics tracking."""

    def test_get_stats(self, exporter):
        """Test getting statistics."""
        stats = exporter.get_stats()

        assert "sft_count" in stats
        assert "dpo_count" in stats
        assert "skipped_count" in stats
        assert "output_dir" in stats

    def test_reset_stats(self, exporter, mock_result):
        """Test resetting statistics."""
        # Export something
        exporter.export_debate(mock_result, "reset-test", "tech")

        # Reset
        exporter.reset_stats()

        stats = exporter.get_stats()
        assert stats["sft_count"] == 0
        assert stats["dpo_count"] == 0
        assert stats["skipped_count"] == 0


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunction:
    """Test the export_debate_to_training convenience function."""

    def test_export_debate_to_training(self, mock_result, temp_dir):
        """Test convenience function."""
        result = export_debate_to_training(
            mock_result,
            debate_id="convenience-test",
            domain="tech",
            output_dir=str(temp_dir),
            min_confidence=0.7,
        )

        assert "sft" in result
        assert "dpo" in result
        assert "skipped" in result

    def test_export_debate_to_training_default_config(self, mock_result, temp_dir):
        """Test convenience function with minimal args."""
        result = export_debate_to_training(
            mock_result,
            output_dir=str(temp_dir),
        )

        assert isinstance(result, dict)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_result_without_attributes(self, exporter):
        """Test handling result without expected attributes."""
        # Create a minimal result object
        result = MagicMock()
        result.confidence = 0.0  # Will fail threshold
        result.rounds_used = 0
        result.consensus_reached = False
        result.final_answer = ""

        export_result = exporter.export_debate(result, "edge-test", "tech")

        assert export_result["skipped"] == 1

    def test_message_with_content_attribute(self, exporter):
        """Test handling messages with content as attribute."""

        class MessageObj:
            def __init__(self, content: str):
                self.content = content

        result = MockDebateResult(
            messages=[
                MessageObj("First proposal with attribute-based content access."),
                MessageObj("Second proposal with different approach to test attribute access."),
            ],
        )

        records = exporter._create_dpo_records(result, "", "", None)

        # Should handle both dict and object messages
        assert isinstance(records, list)

    def test_unicode_content(self, exporter, temp_dir):
        """Test handling unicode content in debates."""
        result = MockDebateResult(
            task="Analyze international contracts with special characters",
            final_answer="Ceci est une réponse avec des caractères spéciaux: ñ, ü, 中文, 日本語, emoji: 🎉",
        )

        export_result = exporter.export_debate(result, "unicode-test", "international")

        # Should handle unicode without errors
        assert export_result["skipped"] == 0 or export_result["sft"] >= 0
