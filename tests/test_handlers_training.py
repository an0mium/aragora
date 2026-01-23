"""
Tests for training handler (training.py).

Tests cover:
- Route handling (can_handle)
- SFT export endpoint
- DPO export endpoint
- Gauntlet export endpoint
- Stats endpoint
- Formats endpoint
- Parameter validation and clamping
- Error handling when exporters unavailable
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


class MockHandler:
    """Mock HTTP request handler."""

    def __init__(self):
        self.headers = {"Content-Type": "application/json"}
        self.path = "/api/training/stats"
        self.command = "GET"


def parse_result(result):
    """Parse HandlerResult to get JSON body and status."""
    body = json.loads(result.body.decode())
    return body, result.status_code


@pytest.fixture
def mock_handler():
    return MockHandler()


@pytest.fixture
def training_handler(tmp_path):
    """Create TrainingHandler instance with temp export directory."""
    from aragora.server.handlers.training import TrainingHandler

    # Use temp path for exports
    with patch.dict("os.environ", {"ARAGORA_TRAINING_EXPORT_DIR": str(tmp_path)}):
        ctx = {"storage": MagicMock()}
        handler = TrainingHandler(ctx)
        # Clear any cached exporters
        handler._exporters = {}
        return handler


# ============================================================================
# Route Handling Tests
# ============================================================================


class TestTrainingRouteHandling:
    """Test route handling and can_handle."""

    def test_can_handle_export_sft(self, training_handler):
        """Test can_handle recognizes SFT export path."""
        assert training_handler.can_handle("/api/v1/training/export/sft") is True

    def test_can_handle_export_dpo(self, training_handler):
        """Test can_handle recognizes DPO export path."""
        assert training_handler.can_handle("/api/v1/training/export/dpo") is True

    def test_can_handle_export_gauntlet(self, training_handler):
        """Test can_handle recognizes Gauntlet export path."""
        assert training_handler.can_handle("/api/v1/training/export/gauntlet") is True

    def test_can_handle_stats(self, training_handler):
        """Test can_handle recognizes stats path."""
        assert training_handler.can_handle("/api/v1/training/stats") is True

    def test_can_handle_formats(self, training_handler):
        """Test can_handle recognizes formats path."""
        assert training_handler.can_handle("/api/v1/training/formats") is True

    def test_cannot_handle_unknown_path(self, training_handler):
        """Test can_handle rejects unknown paths."""
        assert training_handler.can_handle("/api/v1/training/unknown") is False
        assert training_handler.can_handle("/api/v1/other/path") is False

    def test_handle_returns_none_for_unknown_path(self, training_handler, mock_handler):
        """Test handle returns None for unrecognized paths."""
        result = training_handler.handle("/api/training/unknown", {}, mock_handler)
        assert result is None


# ============================================================================
# Formats Endpoint Tests
# ============================================================================


class TestFormatsEndpoint:
    """Test formats endpoint."""

    def test_formats_returns_schema_info(self, training_handler, mock_handler):
        """Test formats endpoint returns format schemas."""
        result = training_handler.handle_formats("/api/training/formats", {}, mock_handler)
        body, status = parse_result(result)

        assert status == 200
        assert "formats" in body
        assert "sft" in body["formats"]
        assert "dpo" in body["formats"]
        assert "gauntlet" in body["formats"]

    def test_formats_includes_output_formats(self, training_handler, mock_handler):
        """Test formats endpoint includes output format options."""
        result = training_handler.handle_formats("/api/training/formats", {}, mock_handler)
        body, status = parse_result(result)

        assert status == 200
        assert "output_formats" in body
        assert "json" in body["output_formats"]
        assert "jsonl" in body["output_formats"]

    def test_formats_includes_endpoints(self, training_handler, mock_handler):
        """Test formats endpoint includes API endpoint references."""
        result = training_handler.handle_formats("/api/training/formats", {}, mock_handler)
        body, status = parse_result(result)

        assert status == 200
        assert "endpoints" in body
        assert body["endpoints"]["sft"] == "/api/training/export/sft"
        assert body["endpoints"]["dpo"] == "/api/training/export/dpo"
        assert body["endpoints"]["gauntlet"] == "/api/training/export/gauntlet"

    def test_formats_sft_schema_complete(self, training_handler, mock_handler):
        """Test SFT format schema is complete."""
        result = training_handler.handle_formats("/api/training/formats", {}, mock_handler)
        body, status = parse_result(result)

        sft = body["formats"]["sft"]
        assert "description" in sft
        assert "schema" in sft
        assert "instruction" in sft["schema"]
        assert "response" in sft["schema"]
        assert "use_case" in sft

    def test_formats_dpo_schema_complete(self, training_handler, mock_handler):
        """Test DPO format schema is complete."""
        result = training_handler.handle_formats("/api/training/formats", {}, mock_handler)
        body, status = parse_result(result)

        dpo = body["formats"]["dpo"]
        assert "description" in dpo
        assert "schema" in dpo
        assert "prompt" in dpo["schema"]
        assert "chosen" in dpo["schema"]
        assert "rejected" in dpo["schema"]

    def test_formats_gauntlet_schema_complete(self, training_handler, mock_handler):
        """Test Gauntlet format schema is complete."""
        result = training_handler.handle_formats("/api/training/formats", {}, mock_handler)
        body, status = parse_result(result)

        gauntlet = body["formats"]["gauntlet"]
        assert "description" in gauntlet
        assert "schema" in gauntlet
        assert "instruction" in gauntlet["schema"]
        assert "response" in gauntlet["schema"]


# ============================================================================
# Stats Endpoint Tests
# ============================================================================


class TestStatsEndpoint:
    """Test stats endpoint."""

    def test_stats_returns_available_exporters(self, training_handler, mock_handler):
        """Test stats returns list of available exporters."""
        result = training_handler.handle_stats("/api/training/stats", {}, mock_handler)
        body, status = parse_result(result)

        assert status == 200
        assert "available_exporters" in body
        assert isinstance(body["available_exporters"], list)

    def test_stats_returns_export_directory(self, training_handler, mock_handler):
        """Test stats returns export directory path."""
        result = training_handler.handle_stats("/api/training/stats", {}, mock_handler)
        body, status = parse_result(result)

        assert status == 200
        assert "export_directory" in body

    def test_stats_returns_exported_files(self, training_handler, mock_handler, tmp_path):
        """Test stats returns list of exported files."""
        # Create a test export file
        test_file = tmp_path / "test_export.jsonl"
        test_file.write_text('{"test": true}')

        training_handler._export_dir = tmp_path
        result = training_handler.handle_stats("/api/training/stats", {}, mock_handler)
        body, status = parse_result(result)

        assert status == 200
        assert "exported_files" in body
        assert len(body["exported_files"]) == 1
        assert body["exported_files"][0]["name"] == "test_export.jsonl"

    def test_stats_file_info_complete(self, training_handler, mock_handler, tmp_path):
        """Test exported file info includes size and timestamps."""
        test_file = tmp_path / "test.jsonl"
        test_file.write_text('{"data": "test"}')

        training_handler._export_dir = tmp_path
        result = training_handler.handle_stats("/api/training/stats", {}, mock_handler)
        body, status = parse_result(result)

        file_info = body["exported_files"][0]
        assert "size_bytes" in file_info
        assert "created_at" in file_info
        assert "modified_at" in file_info

    def test_stats_with_sft_exporter_available(self, training_handler, mock_handler):
        """Test stats checks SFT exporter availability."""
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = [{"test": True}]

        with patch.object(training_handler, "_get_sft_exporter", return_value=mock_exporter):
            result = training_handler.handle_stats("/api/training/stats", {}, mock_handler)
            body, status = parse_result(result)

        assert status == 200
        assert "sft" in body["available_exporters"]
        assert body["sft_available"] is True


# ============================================================================
# SFT Export Tests
# ============================================================================


class TestSFTExport:
    """Test SFT export endpoint."""

    def test_sft_export_exporter_unavailable(self, training_handler, mock_handler):
        """Test SFT export returns 500 when exporter unavailable."""
        with patch.object(training_handler, "_get_sft_exporter", return_value=None):
            result = training_handler.handle_export_sft(
                "/api/training/export/sft", {}, mock_handler
            )
            body, status = parse_result(result)

        assert status == 500
        assert "not available" in body["error"]["message"]

    def test_sft_export_success(self, training_handler, mock_handler):
        """Test successful SFT export."""
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = [{"instruction": "test", "response": "result"}]

        with patch.object(training_handler, "_get_sft_exporter", return_value=mock_exporter):
            result = training_handler.handle_export_sft(
                "/api/training/export/sft", {}, mock_handler
            )
            body, status = parse_result(result)

        assert status == 200
        assert body["export_type"] == "sft"
        assert body["total_records"] == 1
        assert "records" in body
        assert body["format"] == "json"

    def test_sft_export_with_parameters(self, training_handler, mock_handler):
        """Test SFT export respects query parameters."""
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = []

        params = {
            "min_confidence": "0.8",
            "min_success_rate": "0.7",
            "limit": "500",
            "offset": "10",
            "include_critiques": "false",
            "include_patterns": "true",
            "include_debates": "false",
        }

        with patch.object(training_handler, "_get_sft_exporter", return_value=mock_exporter):
            result = training_handler.handle_export_sft(
                "/api/training/export/sft", params, mock_handler
            )
            body, status = parse_result(result)

        assert status == 200
        assert body["parameters"]["min_confidence"] == 0.8
        assert body["parameters"]["min_success_rate"] == 0.7
        assert body["parameters"]["limit"] == 500
        assert body["parameters"]["offset"] == 10
        assert body["parameters"]["include_critiques"] is False
        assert body["parameters"]["include_debates"] is False

    def test_sft_export_jsonl_format(self, training_handler, mock_handler):
        """Test SFT export with JSONL output format."""
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = [
            {"instruction": "q1", "response": "a1"},
            {"instruction": "q2", "response": "a2"},
        ]

        with patch.object(training_handler, "_get_sft_exporter", return_value=mock_exporter):
            result = training_handler.handle_export_sft(
                "/api/training/export/sft", {"format": "jsonl"}, mock_handler
            )
            body, status = parse_result(result)

        assert status == 200
        assert body["format"] == "jsonl"
        assert "data" in body
        # JSONL should have newline-separated JSON
        assert "\n" in body["data"]

    def test_sft_export_parameter_clamping(self, training_handler, mock_handler):
        """Test SFT export clamps parameters to valid ranges."""
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = []

        # Test extreme values
        params = {
            "min_confidence": "1.5",  # Should clamp to 1.0
            "min_success_rate": "-0.5",  # Should clamp to 0.0
            "limit": "99999",  # Should clamp to 10000
            "offset": "-10",  # Should clamp to 0
        }

        with patch.object(training_handler, "_get_sft_exporter", return_value=mock_exporter):
            result = training_handler.handle_export_sft(
                "/api/training/export/sft", params, mock_handler
            )
            body, status = parse_result(result)

        assert status == 200
        assert body["parameters"]["min_confidence"] == 1.0
        assert body["parameters"]["min_success_rate"] == 0.0
        assert body["parameters"]["limit"] == 10000
        assert body["parameters"]["offset"] == 0

    def test_sft_export_handles_exception(self, training_handler, mock_handler):
        """Test SFT export handles exporter exceptions."""
        mock_exporter = MagicMock()
        mock_exporter.export.side_effect = Exception("Database error")

        with patch.object(training_handler, "_get_sft_exporter", return_value=mock_exporter):
            result = training_handler.handle_export_sft(
                "/api/training/export/sft", {}, mock_handler
            )
            body, status = parse_result(result)

        assert status == 500
        assert "Export failed" in body["error"]


# ============================================================================
# DPO Export Tests
# ============================================================================


class TestDPOExport:
    """Test DPO export endpoint."""

    def test_dpo_export_exporter_unavailable(self, training_handler, mock_handler):
        """Test DPO export returns 500 when exporter unavailable."""
        with patch.object(training_handler, "_get_dpo_exporter", return_value=None):
            result = training_handler.handle_export_dpo(
                "/api/training/export/dpo", {}, mock_handler
            )
            body, status = parse_result(result)

        assert status == 500
        assert "not available" in body["error"]["message"]

    def test_dpo_export_success(self, training_handler, mock_handler):
        """Test successful DPO export."""
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = [{"prompt": "q", "chosen": "good", "rejected": "bad"}]

        with patch.object(training_handler, "_get_dpo_exporter", return_value=mock_exporter):
            result = training_handler.handle_export_dpo(
                "/api/training/export/dpo", {}, mock_handler
            )
            body, status = parse_result(result)

        assert status == 200
        assert body["export_type"] == "dpo"
        assert body["total_records"] == 1

    def test_dpo_export_with_parameters(self, training_handler, mock_handler):
        """Test DPO export respects query parameters."""
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = []

        params = {
            "min_confidence_diff": "0.2",
            "limit": "250",
        }

        with patch.object(training_handler, "_get_dpo_exporter", return_value=mock_exporter):
            result = training_handler.handle_export_dpo(
                "/api/training/export/dpo", params, mock_handler
            )
            body, status = parse_result(result)

        assert status == 200
        assert body["parameters"]["min_confidence_diff"] == 0.2
        assert body["parameters"]["limit"] == 250

    def test_dpo_export_jsonl_format(self, training_handler, mock_handler):
        """Test DPO export with JSONL output format."""
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = [
            {"prompt": "q1", "chosen": "a1", "rejected": "b1"},
        ]

        with patch.object(training_handler, "_get_dpo_exporter", return_value=mock_exporter):
            result = training_handler.handle_export_dpo(
                "/api/training/export/dpo", {"format": "jsonl"}, mock_handler
            )
            body, status = parse_result(result)

        assert status == 200
        assert body["format"] == "jsonl"
        assert "data" in body

    def test_dpo_export_parameter_clamping(self, training_handler, mock_handler):
        """Test DPO export clamps parameters to valid ranges."""
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = []

        params = {
            "min_confidence_diff": "2.0",  # Should clamp to 1.0
            "limit": "10000",  # Should clamp to 5000
        }

        with patch.object(training_handler, "_get_dpo_exporter", return_value=mock_exporter):
            result = training_handler.handle_export_dpo(
                "/api/training/export/dpo", params, mock_handler
            )
            body, status = parse_result(result)

        assert status == 200
        assert body["parameters"]["min_confidence_diff"] == 1.0
        assert body["parameters"]["limit"] == 5000


# ============================================================================
# Gauntlet Export Tests
# ============================================================================


class TestGauntletExport:
    """Test Gauntlet export endpoint."""

    def test_gauntlet_export_exporter_unavailable(self, training_handler, mock_handler):
        """Test Gauntlet export returns 500 when exporter unavailable."""
        with patch.object(training_handler, "_get_gauntlet_exporter", return_value=None):
            result = training_handler.handle_export_gauntlet(
                "/api/training/export/gauntlet", {}, mock_handler
            )
            body, status = parse_result(result)

        assert status == 500
        assert "not available" in body["error"]["message"]

    def test_gauntlet_export_success(self, training_handler, mock_handler):
        """Test successful Gauntlet export."""
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = [{"instruction": "adversarial", "response": "safe"}]

        with patch.object(training_handler, "_get_gauntlet_exporter", return_value=mock_exporter):
            result = training_handler.handle_export_gauntlet(
                "/api/training/export/gauntlet", {}, mock_handler
            )
            body, status = parse_result(result)

        assert status == 200
        assert body["export_type"] == "gauntlet"
        assert body["total_records"] == 1

    def test_gauntlet_export_with_persona_filter(self, training_handler, mock_handler):
        """Test Gauntlet export with persona filter."""
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = []

        params = {
            "persona": "gdpr",
            "min_severity": "0.7",
            "limit": "100",
        }

        with patch.object(training_handler, "_get_gauntlet_exporter", return_value=mock_exporter):
            result = training_handler.handle_export_gauntlet(
                "/api/training/export/gauntlet", params, mock_handler
            )
            body, status = parse_result(result)

        assert status == 200
        assert body["parameters"]["persona"] == "gdpr"
        assert body["parameters"]["min_severity"] == 0.7

        # Verify exporter was called with persona
        mock_exporter.export.assert_called_once()
        call_kwargs = mock_exporter.export.call_args[1]
        assert call_kwargs["persona"] == "gdpr"

    def test_gauntlet_export_all_personas(self, training_handler, mock_handler):
        """Test Gauntlet export with all personas (default)."""
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = []

        with patch.object(training_handler, "_get_gauntlet_exporter", return_value=mock_exporter):
            result = training_handler.handle_export_gauntlet(
                "/api/training/export/gauntlet", {}, mock_handler
            )
            body, status = parse_result(result)

        assert status == 200
        assert body["parameters"]["persona"] == "all"

        # Verify exporter was called without persona filter
        call_kwargs = mock_exporter.export.call_args[1]
        assert "persona" not in call_kwargs

    def test_gauntlet_export_jsonl_format(self, training_handler, mock_handler):
        """Test Gauntlet export with JSONL output format."""
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = [
            {"instruction": "test", "response": "safe"},
        ]

        with patch.object(training_handler, "_get_gauntlet_exporter", return_value=mock_exporter):
            result = training_handler.handle_export_gauntlet(
                "/api/training/export/gauntlet", {"format": "jsonl"}, mock_handler
            )
            body, status = parse_result(result)

        assert status == 200
        assert body["format"] == "jsonl"


# ============================================================================
# Exporter Caching Tests
# ============================================================================


class TestExporterCaching:
    """Test exporter lazy initialization and caching."""

    def test_sft_exporter_cached(self, training_handler):
        """Test SFT exporter is cached after first creation."""
        mock_exporter = MagicMock()

        with patch("aragora.training.SFTExporter", return_value=mock_exporter) as mock_class:
            # First call creates exporter
            result1 = training_handler._get_sft_exporter()
            # Second call returns cached
            result2 = training_handler._get_sft_exporter()

        # Only one instance created
        assert mock_class.call_count == 1
        assert result1 is result2

    def test_dpo_exporter_cached(self, training_handler):
        """Test DPO exporter is cached after first creation."""
        mock_exporter = MagicMock()

        with patch("aragora.training.DPOExporter", return_value=mock_exporter) as mock_class:
            result1 = training_handler._get_dpo_exporter()
            result2 = training_handler._get_dpo_exporter()

        assert mock_class.call_count == 1
        assert result1 is result2

    def test_gauntlet_exporter_cached(self, training_handler):
        """Test Gauntlet exporter is cached after first creation."""
        mock_exporter = MagicMock()

        with patch("aragora.training.GauntletExporter", return_value=mock_exporter) as mock_class:
            result1 = training_handler._get_gauntlet_exporter()
            result2 = training_handler._get_gauntlet_exporter()

        assert mock_class.call_count == 1
        assert result1 is result2

    def test_exporter_returns_none_on_import_error(self, training_handler):
        """Test exporter returns None when import fails."""
        with patch.dict("sys.modules", {"aragora.training": None}):
            # Force re-check by clearing cache
            training_handler._exporters = {}

            # These should return None due to ImportError
            assert training_handler._get_sft_exporter() is None
            assert training_handler._get_dpo_exporter() is None
            assert training_handler._get_gauntlet_exporter() is None


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling in training handler."""

    def test_stats_handles_exception(self, training_handler, mock_handler):
        """Test stats endpoint handles exceptions gracefully."""
        with patch.object(
            training_handler, "_get_sft_exporter", side_effect=Exception("Unexpected error")
        ):
            result = training_handler.handle_stats("/api/training/stats", {}, mock_handler)
            body, status = parse_result(result)

        assert status == 500
        assert "Failed to get stats" in body["error"]

    def test_invalid_float_parameter(self, training_handler, mock_handler):
        """Test handler handles invalid float parameters."""
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = []

        params = {"min_confidence": "not_a_number"}

        with patch.object(training_handler, "_get_sft_exporter", return_value=mock_exporter):
            result = training_handler.handle_export_sft(
                "/api/training/export/sft", params, mock_handler
            )
            body, status = parse_result(result)

        # Should return 500 due to ValueError in float conversion
        assert status == 500

    def test_invalid_int_parameter(self, training_handler, mock_handler):
        """Test handler handles invalid int parameters."""
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = []

        params = {"limit": "not_a_number"}

        with patch.object(training_handler, "_get_sft_exporter", return_value=mock_exporter):
            result = training_handler.handle_export_sft(
                "/api/training/export/sft", params, mock_handler
            )
            body, status = parse_result(result)

        # Should return 500 due to ValueError in int conversion
        assert status == 500


# ============================================================================
# Export Directory Tests
# ============================================================================


class TestExportDirectory:
    """Test export directory handling."""

    def test_export_dir_created_on_init(self, tmp_path):
        """Test export directory is created during handler initialization."""
        from aragora.server.handlers.training import TrainingHandler

        export_dir = tmp_path / "new_export_dir"
        assert not export_dir.exists()

        with patch.dict("os.environ", {"ARAGORA_TRAINING_EXPORT_DIR": str(export_dir)}):
            ctx = {"storage": MagicMock()}
            handler = TrainingHandler(ctx)

        assert export_dir.exists()

    def test_export_dir_default(self, tmp_path):
        """Test default export directory is used when env not set."""
        from aragora.server.handlers.training import TrainingHandler

        # Clear environment variable
        with patch.dict("os.environ", {}, clear=True):
            ctx = {"storage": MagicMock()}
            handler = TrainingHandler(ctx)

        assert ".nomic/training_exports" in str(handler._export_dir)
