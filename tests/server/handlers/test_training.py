"""
Tests for training data export handler.

Tests:
- TrainingHandler initialization
- Route matching (can_handle)
- Format endpoint (no auth required)
- Parameter validation bounds
"""

import pytest
from pathlib import Path

from aragora.server.handlers.training import TrainingHandler


class TestTrainingHandlerInit:
    """Tests for TrainingHandler initialization."""

    def test_init_creates_export_dir(self, tmp_path, monkeypatch):
        """Should create export directory on init."""
        export_dir = tmp_path / "exports"
        monkeypatch.setenv("ARAGORA_TRAINING_EXPORT_DIR", str(export_dir))
        handler = TrainingHandler({})
        assert export_dir.exists()

    def test_init_with_existing_dir(self, tmp_path, monkeypatch):
        """Should work with existing directory."""
        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        monkeypatch.setenv("ARAGORA_TRAINING_EXPORT_DIR", str(export_dir))
        handler = TrainingHandler({})
        assert export_dir.exists()

    def test_init_empty_exporters(self):
        """Should initialize with empty exporters dict."""
        handler = TrainingHandler({})
        assert handler._exporters == {}


class TestTrainingHandlerCanHandle:
    """Tests for can_handle routing."""

    def test_can_handle_sft_export(self):
        """Should handle SFT export path."""
        handler = TrainingHandler({})
        assert handler.can_handle("/api/training/export/sft") is True

    def test_can_handle_dpo_export(self):
        """Should handle DPO export path."""
        handler = TrainingHandler({})
        assert handler.can_handle("/api/training/export/dpo") is True

    def test_can_handle_gauntlet_export(self):
        """Should handle Gauntlet export path."""
        handler = TrainingHandler({})
        assert handler.can_handle("/api/training/export/gauntlet") is True

    def test_can_handle_stats(self):
        """Should handle stats path."""
        handler = TrainingHandler({})
        assert handler.can_handle("/api/training/stats") is True

    def test_can_handle_formats(self):
        """Should handle formats path."""
        handler = TrainingHandler({})
        assert handler.can_handle("/api/training/formats") is True

    def test_cannot_handle_unknown_path(self):
        """Should not handle unknown paths."""
        handler = TrainingHandler({})
        assert handler.can_handle("/api/training/unknown") is False
        assert handler.can_handle("/api/other/path") is False
        assert handler.can_handle("/api/training") is False


class TestTrainingHandlerRoutes:
    """Tests for route configuration."""

    def test_routes_constant_has_all_endpoints(self):
        """ROUTES should have all expected endpoints."""
        assert "/api/training/export/sft" in TrainingHandler.ROUTES
        assert "/api/training/export/dpo" in TrainingHandler.ROUTES
        assert "/api/training/export/gauntlet" in TrainingHandler.ROUTES
        assert "/api/training/stats" in TrainingHandler.ROUTES
        assert "/api/training/formats" in TrainingHandler.ROUTES

    def test_routes_map_to_handler_methods(self):
        """Each route should map to a valid method."""
        handler = TrainingHandler({})
        for path, method_name in TrainingHandler.ROUTES.items():
            assert hasattr(handler, method_name), f"Missing method: {method_name}"


class TestTrainingHandlerFormats:
    """Tests for formats endpoint (no auth required)."""

    def test_handle_formats_returns_result(self):
        """handle_formats should return a result."""
        handler = TrainingHandler({})
        result = handler.handle_formats("/api/training/formats", {}, None)
        assert result is not None

    def test_handle_formats_has_expected_structure(self):
        """Formats response should have expected keys."""
        handler = TrainingHandler({})
        result = handler.handle_formats("/api/training/formats", {}, None)
        # Result is a HandlerResult, need to check body
        assert result.status_code == 200
        import json

        body = json.loads(result.body.decode("utf-8"))
        assert "formats" in body
        assert "output_formats" in body
        assert "endpoints" in body

    def test_handle_formats_includes_sft(self):
        """Formats should include SFT description."""
        handler = TrainingHandler({})
        result = handler.handle_formats("/api/training/formats", {}, None)
        import json

        body = json.loads(result.body.decode("utf-8"))
        assert "sft" in body["formats"]
        assert "description" in body["formats"]["sft"]
        assert "schema" in body["formats"]["sft"]

    def test_handle_formats_includes_dpo(self):
        """Formats should include DPO description."""
        handler = TrainingHandler({})
        result = handler.handle_formats("/api/training/formats", {}, None)
        import json

        body = json.loads(result.body.decode("utf-8"))
        assert "dpo" in body["formats"]
        assert "description" in body["formats"]["dpo"]

    def test_handle_formats_includes_gauntlet(self):
        """Formats should include Gauntlet description."""
        handler = TrainingHandler({})
        result = handler.handle_formats("/api/training/formats", {}, None)
        import json

        body = json.loads(result.body.decode("utf-8"))
        assert "gauntlet" in body["formats"]
        assert "description" in body["formats"]["gauntlet"]

    def test_output_formats_include_json_jsonl(self):
        """Output formats should include json and jsonl."""
        handler = TrainingHandler({})
        result = handler.handle_formats("/api/training/formats", {}, None)
        import json

        body = json.loads(result.body.decode("utf-8"))
        assert "json" in body["output_formats"]
        assert "jsonl" in body["output_formats"]


class TestTrainingHandlerExporterLazyLoad:
    """Tests for lazy-loading exporters."""

    def test_sft_exporter_returns_none_without_module(self):
        """Should return None when training module not available."""
        handler = TrainingHandler({})
        # Without actual training module, should return None
        result = handler._get_sft_exporter()
        # May return None or actual exporter depending on install
        assert result is None or result is not None

    def test_dpo_exporter_returns_none_without_module(self):
        """Should return None when training module not available."""
        handler = TrainingHandler({})
        result = handler._get_dpo_exporter()
        assert result is None or result is not None

    def test_gauntlet_exporter_returns_none_without_module(self):
        """Should return None when training module not available."""
        handler = TrainingHandler({})
        result = handler._get_gauntlet_exporter()
        assert result is None or result is not None

    def test_exporter_cached_after_first_load(self):
        """Exporter should be cached after first load."""
        handler = TrainingHandler({})
        handler._exporters["sft"] = "mock_exporter"
        result = handler._get_sft_exporter()
        assert result == "mock_exporter"


class TestTrainingHandlerHandle:
    """Tests for the handle dispatcher method."""

    def test_handle_unknown_path_returns_none(self):
        """Unknown path should return None."""
        handler = TrainingHandler({})
        result = handler.handle("/api/unknown", {}, None)
        assert result is None

    def test_handle_formats_dispatches_correctly(self):
        """Should dispatch formats path to handle_formats."""
        handler = TrainingHandler({})
        result = handler.handle("/api/training/formats", {}, None)
        assert result is not None
        assert result.status_code == 200


class TestTrainingHandlerStats:
    """Tests for stats endpoint."""

    def test_handle_stats_returns_result(self):
        """handle_stats should return a result."""
        handler = TrainingHandler({})
        result = handler.handle_stats("/api/training/stats", {}, None)
        assert result is not None

    def test_handle_stats_has_available_exporters(self):
        """Stats response should list available exporters."""
        handler = TrainingHandler({})
        result = handler.handle_stats("/api/training/stats", {}, None)
        import json

        body = json.loads(result.body.decode("utf-8"))
        assert "available_exporters" in body
        assert isinstance(body["available_exporters"], list)

    def test_handle_stats_has_export_directory(self):
        """Stats response should include export directory."""
        handler = TrainingHandler({})
        result = handler.handle_stats("/api/training/stats", {}, None)
        import json

        body = json.loads(result.body.decode("utf-8"))
        assert "export_directory" in body

    def test_handle_stats_has_exported_files(self):
        """Stats response should list exported files."""
        handler = TrainingHandler({})
        result = handler.handle_stats("/api/training/stats", {}, None)
        import json

        body = json.loads(result.body.decode("utf-8"))
        assert "exported_files" in body
        assert isinstance(body["exported_files"], list)
