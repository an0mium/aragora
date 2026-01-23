"""
Tests for training data export system.

Tests cover:
- BaseExporter and TrainingRecord dataclasses
- SFTExporter functionality
- DPOExporter functionality
- GauntletExporter functionality
- TrainingHandler API endpoints
- Export metadata and validation
"""

import json
import pytest
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

from aragora.training.exporters.base import (
    BaseExporter,
    ExportMetadata,
    TrainingRecord,
    PreferenceRecord,
)


class TestTrainingRecord:
    """Tests for TrainingRecord dataclass."""

    def test_required_fields(self):
        """Should create with required fields."""
        record = TrainingRecord(
            instruction="Analyze this code",
            response="The code has a bug on line 10",
        )

        assert record.instruction == "Analyze this code"
        assert record.response == "The code has a bug on line 10"
        assert record.metadata == {}

    def test_with_metadata(self):
        """Should accept metadata."""
        record = TrainingRecord(
            instruction="Test",
            response="Response",
            metadata={"source": "debate", "confidence": 0.85},
        )

        assert record.metadata["source"] == "debate"
        assert record.metadata["confidence"] == 0.85

    def test_to_dict(self):
        """Should serialize to dictionary."""
        record = TrainingRecord(
            instruction="Instruction",
            response="Response",
            metadata={"key": "value"},
        )

        d = record.to_dict()

        assert d["instruction"] == "Instruction"
        assert d["response"] == "Response"
        assert d["metadata"]["key"] == "value"

    def test_to_jsonl(self):
        """Should serialize to JSONL line."""
        record = TrainingRecord(
            instruction="Test instruction",
            response="Test response",
        )

        line = record.to_jsonl()
        parsed = json.loads(line)

        assert parsed["instruction"] == "Test instruction"
        assert parsed["response"] == "Test response"


class TestPreferenceRecord:
    """Tests for PreferenceRecord dataclass."""

    def test_required_fields(self):
        """Should create with required fields."""
        record = PreferenceRecord(
            prompt="Which is better?",
            chosen="Option A is better because...",
            rejected="Option B is worse because...",
        )

        assert record.prompt == "Which is better?"
        assert record.chosen == "Option A is better because..."
        assert record.rejected == "Option B is worse because..."

    def test_with_metadata(self):
        """Should accept metadata."""
        record = PreferenceRecord(
            prompt="Prompt",
            chosen="Good",
            rejected="Bad",
            metadata={"chosen_score": 0.9, "rejected_score": 0.3},
        )

        assert record.metadata["chosen_score"] == 0.9
        assert record.metadata["rejected_score"] == 0.3

    def test_to_dict(self):
        """Should serialize to dictionary."""
        record = PreferenceRecord(
            prompt="P",
            chosen="C",
            rejected="R",
        )

        d = record.to_dict()

        assert d["prompt"] == "P"
        assert d["chosen"] == "C"
        assert d["rejected"] == "R"

    def test_to_jsonl(self):
        """Should serialize to JSONL line."""
        record = PreferenceRecord(
            prompt="Test",
            chosen="Better",
            rejected="Worse",
        )

        line = record.to_jsonl()
        parsed = json.loads(line)

        assert parsed["chosen"] == "Better"
        assert parsed["rejected"] == "Worse"


class TestExportMetadata:
    """Tests for ExportMetadata dataclass."""

    def test_required_fields(self):
        """Should create with required exporter_type."""
        meta = ExportMetadata(exporter_type="sft")

        assert meta.exporter_type == "sft"

    def test_auto_timestamp(self):
        """Should auto-generate timestamp."""
        meta = ExportMetadata(exporter_type="test")

        assert meta.exported_at is not None
        # Should be ISO format
        datetime.fromisoformat(meta.exported_at)

    def test_default_values(self):
        """Should have correct defaults."""
        meta = ExportMetadata(exporter_type="test")

        assert meta.total_records == 0
        assert meta.filters_applied == {}
        assert meta.source_db == ""


class TestBaseExporter:
    """Tests for BaseExporter abstract class."""

    def test_cannot_instantiate_directly(self):
        """Should not instantiate abstract class."""
        with pytest.raises(TypeError):
            BaseExporter()

    def test_subclass_must_implement_export(self):
        """Subclass must implement export method."""

        class IncompleteExporter(BaseExporter):
            pass

        with pytest.raises(TypeError):
            IncompleteExporter()

    def test_subclass_can_implement(self):
        """Subclass with export method should work."""

        class ConcreteExporter(BaseExporter):
            exporter_type = "concrete"

            def export(self, **kwargs):
                return [{"instruction": "test", "response": "test"}]

        exporter = ConcreteExporter()
        records = exporter.export()

        assert len(records) == 1

    def test_export_to_file(self, tmp_path):
        """Should export to file."""

        class TestExporter(BaseExporter):
            exporter_type = "test"

            def export(self, **kwargs):
                return [
                    {"instruction": "q1", "response": "a1"},
                    {"instruction": "q2", "response": "a2"},
                ]

        exporter = TestExporter()
        output_file = tmp_path / "output.jsonl"

        metadata = exporter.export_to_file(output_file)

        assert output_file.exists()
        assert metadata.total_records == 2
        assert metadata.exporter_type == "test"

        # Verify file contents
        with open(output_file) as f:
            lines = f.readlines()

        assert len(lines) == 2
        assert json.loads(lines[0])["instruction"] == "q1"

    def test_export_to_file_creates_directory(self, tmp_path):
        """Should create parent directory if needed."""

        class TestExporter(BaseExporter):
            exporter_type = "test"

            def export(self, **kwargs):
                return [{"instruction": "test", "response": "test"}]

        exporter = TestExporter()
        output_file = tmp_path / "nested" / "dir" / "output.jsonl"

        metadata = exporter.export_to_file(output_file)

        assert output_file.exists()

    def test_validate_record_default(self):
        """Default validation should return True."""

        class TestExporter(BaseExporter):
            exporter_type = "test"

            def export(self, **kwargs):
                return []

        exporter = TestExporter()
        assert exporter.validate_record({"any": "data"}) is True


class TestSFTExporter:
    """Tests for SFTExporter."""

    @pytest.fixture
    def mock_db(self, tmp_path):
        """Create a mock database with test data."""
        db_path = tmp_path / "test_memory.db"
        conn = sqlite3.connect(db_path)

        # Create debates table
        conn.execute(
            """
            CREATE TABLE debates (
                id TEXT PRIMARY KEY,
                task TEXT,
                final_answer TEXT,
                confidence REAL,
                rounds_used INTEGER,
                consensus_reached INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create critiques table
        conn.execute(
            """
            CREATE TABLE critiques (
                id INTEGER PRIMARY KEY,
                debate_id TEXT,
                agent TEXT,
                target_agent TEXT,
                issues TEXT,
                suggestions TEXT,
                severity TEXT,
                reasoning TEXT,
                led_to_improvement INTEGER
            )
        """
        )

        # Insert test data
        conn.execute(
            """
            INSERT INTO debates (id, task, final_answer, confidence, rounds_used, consensus_reached)
            VALUES ('debate-1', 'Analyze the security of this code',
                    'The code has a SQL injection vulnerability on line 15. To fix this, use parameterized queries instead of string concatenation.',
                    0.85, 3, 1)
        """
        )
        conn.execute(
            """
            INSERT INTO debates (id, task, final_answer, confidence, rounds_used, consensus_reached)
            VALUES ('debate-2', 'Review the architecture',
                    'The architecture follows good separation of concerns. The service layer properly isolates business logic.',
                    0.92, 2, 1)
        """
        )

        conn.commit()
        conn.close()

        return str(db_path)

    def test_export_empty_db(self, tmp_path):
        """Should handle empty database."""
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE debates (
                id TEXT PRIMARY KEY,
                task TEXT,
                final_answer TEXT,
                confidence REAL,
                rounds_used INTEGER,
                consensus_reached INTEGER,
                created_at TEXT
            )
        """
        )
        conn.commit()
        conn.close()

        from aragora.training.exporters.sft_exporter import SFTExporter

        exporter = SFTExporter(db_path=str(db_path))
        records = exporter.export(include_patterns=False, include_critiques=False)

        assert records == []


class TestTrainingHandler:
    """Tests for TrainingHandler API endpoints."""

    @pytest.fixture
    def handler(self):
        """Create a TrainingHandler instance."""
        from aragora.server.handlers.training import TrainingHandler

        ctx = {"storage": MagicMock()}
        return TrainingHandler(ctx)

    def test_routes_defined(self, handler):
        """Should have routes defined."""
        assert "/api/training/export/sft" in handler.ROUTES
        assert "/api/training/export/dpo" in handler.ROUTES
        assert "/api/training/export/gauntlet" in handler.ROUTES
        assert "/api/training/stats" in handler.ROUTES
        assert "/api/training/formats" in handler.ROUTES

    def test_can_handle_training_paths(self, handler):
        """Should handle training paths."""
        assert handler.can_handle("/api/v1/training/export/sft")
        assert handler.can_handle("/api/v1/training/stats")
        assert handler.can_handle("/api/v1/training/formats")

    def test_cannot_handle_other_paths(self, handler):
        """Should not handle non-training paths."""
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/v1/agents")

    def test_handle_formats(self, handler):
        """Should return format documentation."""
        result = handler.handle_formats("/api/training/formats", {}, MagicMock())

        assert result.status_code == 200
        body = json.loads(result.body)

        assert "formats" in body
        assert "sft" in body["formats"]
        assert "dpo" in body["formats"]
        assert "gauntlet" in body["formats"]
        assert "output_formats" in body
        assert "json" in body["output_formats"]
        assert "jsonl" in body["output_formats"]

    def test_handle_stats(self, handler):
        """Should return stats."""
        result = handler.handle_stats("/api/training/stats", {}, MagicMock())

        assert result.status_code == 200
        body = json.loads(result.body)

        assert "available_exporters" in body
        assert "export_directory" in body

    @patch("aragora.server.handlers.training.TrainingHandler._get_sft_exporter")
    def test_handle_export_sft_with_mock(self, mock_get_exporter, handler):
        """Should export SFT data."""
        # Create mock exporter
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = [
            {"instruction": "test", "response": "test response"},
        ]
        mock_get_exporter.return_value = mock_exporter

        result = handler.handle_export_sft(
            "/api/training/export/sft",
            {"limit": "10", "min_confidence": "0.8"},
            MagicMock(),
        )

        assert result.status_code == 200
        body = json.loads(result.body)

        assert body["export_type"] == "sft"
        assert body["total_records"] == 1
        assert len(body["records"]) == 1
        assert body["parameters"]["min_confidence"] == 0.8
        assert body["parameters"]["limit"] == 10

    @patch("aragora.server.handlers.training.TrainingHandler._get_sft_exporter")
    def test_handle_export_sft_jsonl_format(self, mock_get_exporter, handler):
        """Should export in JSONL format."""
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = [
            {"instruction": "q1", "response": "a1"},
            {"instruction": "q2", "response": "a2"},
        ]
        mock_get_exporter.return_value = mock_exporter

        result = handler.handle_export_sft(
            "/api/training/export/sft",
            {"format": "jsonl"},
            MagicMock(),
        )

        assert result.status_code == 200
        body = json.loads(result.body)

        assert body["format"] == "jsonl"
        assert "data" in body
        assert "records" not in body

        # Parse JSONL
        lines = body["data"].strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["instruction"] == "q1"

    @patch("aragora.server.handlers.training.TrainingHandler._get_sft_exporter")
    def test_handle_export_sft_missing_exporter(self, mock_get_exporter, handler):
        """Should return error when exporter unavailable."""
        mock_get_exporter.return_value = None

        result = handler.handle_export_sft(
            "/api/training/export/sft",
            {},
            MagicMock(),
        )

        assert result.status_code == 500
        body = json.loads(result.body)
        error = body.get("error", {})
        if isinstance(error, dict):
            assert "SFT exporter not available" in error.get("message", "")
        else:
            assert "SFT exporter not available" in str(error)

    @patch("aragora.server.handlers.training.TrainingHandler._get_dpo_exporter")
    def test_handle_export_dpo_with_mock(self, mock_get_exporter, handler):
        """Should export DPO data."""
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = [
            {"prompt": "test", "chosen": "good", "rejected": "bad"},
        ]
        mock_get_exporter.return_value = mock_exporter

        result = handler.handle_export_dpo(
            "/api/training/export/dpo",
            {"min_confidence_diff": "0.2"},
            MagicMock(),
        )

        assert result.status_code == 200
        body = json.loads(result.body)

        assert body["export_type"] == "dpo"
        assert body["total_records"] == 1
        assert body["parameters"]["min_confidence_diff"] == 0.2

    @patch("aragora.server.handlers.training.TrainingHandler._get_gauntlet_exporter")
    def test_handle_export_gauntlet_with_mock(self, mock_get_exporter, handler):
        """Should export Gauntlet data."""
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = [
            {"instruction": "adversarial prompt", "response": "safe response"},
        ]
        mock_get_exporter.return_value = mock_exporter

        result = handler.handle_export_gauntlet(
            "/api/training/export/gauntlet",
            {"persona": "gdpr", "min_severity": "0.7"},
            MagicMock(),
        )

        assert result.status_code == 200
        body = json.loads(result.body)

        assert body["export_type"] == "gauntlet"
        assert body["parameters"]["persona"] == "gdpr"
        assert body["parameters"]["min_severity"] == 0.7

    def test_parameter_validation_confidence(self, handler):
        """Should clamp confidence to valid range."""
        with patch.object(handler, "_get_sft_exporter") as mock:
            mock_exporter = MagicMock()
            mock_exporter.export.return_value = []
            mock.return_value = mock_exporter

            # Test clamping to max
            handler.handle_export_sft(
                "/api/training/export/sft",
                {"min_confidence": "1.5"},
                MagicMock(),
            )
            call_args = mock_exporter.export.call_args
            assert call_args.kwargs["min_confidence"] == 1.0

            # Test clamping to min
            handler.handle_export_sft(
                "/api/training/export/sft",
                {"min_confidence": "-0.5"},
                MagicMock(),
            )
            call_args = mock_exporter.export.call_args
            assert call_args.kwargs["min_confidence"] == 0.0

    def test_parameter_validation_limit(self, handler):
        """Should clamp limit to valid range."""
        with patch.object(handler, "_get_sft_exporter") as mock:
            mock_exporter = MagicMock()
            mock_exporter.export.return_value = []
            mock.return_value = mock_exporter

            # Test clamping to max
            handler.handle_export_sft(
                "/api/training/export/sft",
                {"limit": "50000"},
                MagicMock(),
            )
            call_args = mock_exporter.export.call_args
            assert call_args.kwargs["limit"] == 10000

            # Test clamping to min
            handler.handle_export_sft(
                "/api/training/export/sft",
                {"limit": "0"},
                MagicMock(),
            )
            call_args = mock_exporter.export.call_args
            assert call_args.kwargs["limit"] == 1


class TestDPOExporter:
    """Tests for DPOExporter."""

    def test_import(self):
        """Should be importable."""
        from aragora.training import DPOExporter

        assert DPOExporter is not None

    def test_exporter_type(self):
        """Should have correct exporter type."""
        from aragora.training import DPOExporter

        assert DPOExporter.exporter_type == "dpo"


class TestGauntletExporter:
    """Tests for GauntletExporter."""

    def test_import(self):
        """Should be importable."""
        from aragora.training import GauntletExporter

        assert GauntletExporter is not None

    def test_exporter_type(self):
        """Should have correct exporter type."""
        from aragora.training import GauntletExporter

        assert GauntletExporter.exporter_type == "gauntlet"


class TestTrainingModuleExports:
    """Tests for training module exports."""

    def test_all_exports_available(self):
        """All expected exports should be available."""
        from aragora.training import (
            TinkerClient,
            TinkerConfig,
            TinkerModel,
            TrainingScheduler,
            TrainingJob,
            TinkerEvaluator,
            ABTestResult,
            EvaluationMetrics,
            ModelRegistry,
            ModelMetadata,
            get_registry,
            SFTExporter,
            DPOExporter,
            GauntletExporter,
            BaseExporter,
        )

        # All should be classes or functions
        assert TinkerClient is not None
        assert SFTExporter is not None
        assert DPOExporter is not None
        assert GauntletExporter is not None


class TestEdgeCases:
    """Edge case tests."""

    def test_training_record_empty_strings(self):
        """Should handle empty strings."""
        record = TrainingRecord(instruction="", response="")

        assert record.instruction == ""
        assert record.response == ""

    def test_training_record_unicode(self):
        """Should handle unicode content."""
        record = TrainingRecord(
            instruction="Analyze: 你好世界",
            response="Response with emoji: 🎉",
        )

        d = record.to_dict()
        assert "你好" in d["instruction"]
        assert "🎉" in d["response"]

    def test_preference_record_long_content(self):
        """Should handle long content."""
        long_text = "x" * 10000
        record = PreferenceRecord(
            prompt=long_text,
            chosen=long_text,
            rejected=long_text,
        )

        assert len(record.prompt) == 10000
        assert len(record.to_jsonl()) > 30000
