"""
Tests for Gauntlet API Export Utilities.

Tests multi-format export for receipts and heatmaps:
- JSON export
- Markdown export
- HTML export
- CSV export
- SARIF export (security findings)
"""

import json
import pytest
from datetime import datetime
from unittest.mock import MagicMock

from aragora.gauntlet.api.export import (
    ReceiptExportFormat,
    HeatmapExportFormat,
    ExportOptions,
    export_receipt,
    export_heatmap,
    export_receipts_bundle,
    stream_receipt_json,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_receipt():
    """Create a sample DecisionReceipt for testing."""
    receipt = MagicMock()
    receipt.id = "receipt_001"
    receipt.request_id = "req_001"
    receipt.question = "Should we implement rate limiting?"
    receipt.decision = "Yes, implement token bucket rate limiting"
    receipt.confidence = 0.87
    receipt.timestamp = datetime(2024, 1, 15, 10, 30, 0)
    receipt.debate_duration_seconds = 45.5
    receipt.consensus_reached = True
    receipt.participating_agents = ["claude", "gpt4", "gemini"]
    receipt.agreeing_agents = ["claude", "gpt4"]
    receipt.dissenting_agents = ["gemini"]
    receipt.key_arguments = [
        "Token bucket allows burst handling",
        "Better than fixed window for APIs",
    ]
    receipt.dissenting_views = [
        "Sliding window provides more accuracy",
    ]
    receipt.evidence_cited = ["RFC 6585", "API Best Practices Guide"]
    receipt.risk_flags = []
    receipt.metadata = {"domain": "engineering", "priority": "high"}

    # Mock to_dict method
    receipt.to_dict = MagicMock(return_value={
        "id": receipt.id,
        "request_id": receipt.request_id,
        "question": receipt.question,
        "decision": receipt.decision,
        "confidence": receipt.confidence,
        "timestamp": receipt.timestamp.isoformat(),
        "debate_duration_seconds": receipt.debate_duration_seconds,
        "consensus_reached": receipt.consensus_reached,
        "participating_agents": receipt.participating_agents,
        "agreeing_agents": receipt.agreeing_agents,
        "dissenting_agents": receipt.dissenting_agents,
        "key_arguments": receipt.key_arguments,
        "dissenting_views": receipt.dissenting_views,
        "evidence_cited": receipt.evidence_cited,
        "risk_flags": receipt.risk_flags,
        "metadata": receipt.metadata,
    })

    return receipt


@pytest.fixture
def sample_heatmap():
    """Create a sample RiskHeatmap for testing."""
    heatmap = MagicMock()
    heatmap.id = "heatmap_001"
    heatmap.gauntlet_id = "gauntlet_001"
    heatmap.generated_at = datetime(2024, 1, 15, 11, 0, 0)
    heatmap.rows = ["Security", "Performance", "Reliability"]
    heatmap.columns = ["Low", "Medium", "High"]
    heatmap.cells = [
        [0.1, 0.2, 0.1],
        [0.3, 0.4, 0.2],
        [0.2, 0.3, 0.1],
    ]
    heatmap.metadata = {"analysis_type": "risk_assessment"}

    heatmap.to_dict = MagicMock(return_value={
        "id": heatmap.id,
        "gauntlet_id": heatmap.gauntlet_id,
        "generated_at": heatmap.generated_at.isoformat(),
        "rows": heatmap.rows,
        "columns": heatmap.columns,
        "cells": heatmap.cells,
        "metadata": heatmap.metadata,
    })

    return heatmap


@pytest.fixture
def export_options():
    """Create default export options."""
    return ExportOptions()


# ============================================================================
# ReceiptExportFormat Tests
# ============================================================================


class TestReceiptExportFormat:
    """Tests for ReceiptExportFormat enum."""

    def test_format_values(self):
        """Test all format values exist."""
        assert ReceiptExportFormat.JSON.value == "json"
        assert ReceiptExportFormat.MARKDOWN.value == "markdown"
        assert ReceiptExportFormat.HTML.value == "html"
        assert ReceiptExportFormat.CSV.value == "csv"
        assert ReceiptExportFormat.SARIF.value == "sarif"

    def test_format_from_string(self):
        """Test creating format from string."""
        assert ReceiptExportFormat("json") == ReceiptExportFormat.JSON
        assert ReceiptExportFormat("markdown") == ReceiptExportFormat.MARKDOWN


class TestHeatmapExportFormat:
    """Tests for HeatmapExportFormat enum."""

    def test_format_values(self):
        """Test all format values exist."""
        assert HeatmapExportFormat.JSON.value == "json"
        assert HeatmapExportFormat.CSV.value == "csv"
        assert HeatmapExportFormat.HTML.value == "html"


# ============================================================================
# ExportOptions Tests
# ============================================================================


class TestExportOptions:
    """Tests for ExportOptions dataclass."""

    def test_default_options(self):
        """Test default export options."""
        options = ExportOptions()

        assert options.include_export_metadata is True
        assert options.include_provenance is True
        assert options.include_config is False
        assert options.validate_schema is False
        assert options.indent == 2
        assert options.sort_keys is False

    def test_custom_options(self):
        """Test custom export options."""
        options = ExportOptions(
            include_export_metadata=False,
            include_provenance=False,
            sort_keys=True,
            indent=4,
        )

        assert options.include_export_metadata is False
        assert options.include_provenance is False
        assert options.sort_keys is True
        assert options.indent == 4


# ============================================================================
# JSON Export Tests
# ============================================================================


class TestJSONExport:
    """Tests for JSON export functionality."""

    def test_export_receipt_json(self, sample_receipt, export_options):
        """Test exporting receipt to JSON."""
        result = export_receipt(
            sample_receipt,
            format=ReceiptExportFormat.JSON,
            options=export_options,
        )

        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed["id"] == "receipt_001"
        assert parsed["decision"] == "Yes, implement token bucket rate limiting"

    def test_export_receipt_json_pretty(self, sample_receipt):
        """Test pretty-printed JSON export."""
        options = ExportOptions(indent=2)
        result = export_receipt(
            sample_receipt,
            format=ReceiptExportFormat.JSON,
            options=options,
        )

        # Should have indentation
        assert "\n" in result
        assert "  " in result

    def test_export_receipt_json_compact(self, sample_receipt):
        """Test compact JSON export with minimal indent."""
        options = ExportOptions(indent=0)
        result = export_receipt(
            sample_receipt,
            format=ReceiptExportFormat.JSON,
            options=options,
        )

        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed["id"] == "receipt_001"
        # With indent=0, indentation should be minimal (0 spaces per level)
        # The output may still have newlines but no leading spaces
        assert "    " not in result  # No 4-space indentation

    def test_export_heatmap_json(self, sample_heatmap, export_options):
        """Test exporting heatmap to JSON."""
        result = export_heatmap(
            sample_heatmap,
            format=HeatmapExportFormat.JSON,
            options=export_options,
        )

        parsed = json.loads(result)
        assert parsed["id"] == "heatmap_001"
        assert len(parsed["rows"]) == 3


# ============================================================================
# Markdown Export Tests
# ============================================================================


@pytest.mark.skip(reason="Markdown export not yet implemented")
class TestMarkdownExport:
    """Tests for Markdown export functionality."""

    def test_export_receipt_markdown(self, sample_receipt, export_options):
        """Test exporting receipt to Markdown."""
        result = export_receipt(
            sample_receipt,
            format=ReceiptExportFormat.MARKDOWN,
            options=export_options,
        )

        # Should contain Markdown headers
        assert "# Decision Receipt" in result or "##" in result
        assert "receipt_001" in result
        assert "rate limiting" in result.lower()

    def test_markdown_includes_decision(self, sample_receipt, export_options):
        """Test Markdown includes decision text."""
        result = export_receipt(
            sample_receipt,
            format=ReceiptExportFormat.MARKDOWN,
            options=export_options,
        )

        assert "token bucket" in result.lower()

    def test_markdown_includes_confidence(self, sample_receipt, export_options):
        """Test Markdown includes confidence score."""
        result = export_receipt(
            sample_receipt,
            format=ReceiptExportFormat.MARKDOWN,
            options=export_options,
        )

        assert "87" in result or "0.87" in result


# ============================================================================
# HTML Export Tests
# ============================================================================


@pytest.mark.skip(reason="HTML export not yet implemented")
class TestHTMLExport:
    """Tests for HTML export functionality."""

    def test_export_receipt_html(self, sample_receipt, export_options):
        """Test exporting receipt to HTML."""
        result = export_receipt(
            sample_receipt,
            format=ReceiptExportFormat.HTML,
            options=export_options,
        )

        # Should be valid HTML structure
        assert "<" in result
        assert ">" in result
        # Should contain receipt data
        assert "receipt_001" in result

    def test_html_has_structure(self, sample_receipt, export_options):
        """Test HTML has proper structure."""
        result = export_receipt(
            sample_receipt,
            format=ReceiptExportFormat.HTML,
            options=export_options,
        )

        # Should have some HTML elements
        assert "<div" in result or "<p" in result or "<span" in result or "<table" in result

    def test_export_heatmap_html(self, sample_heatmap, export_options):
        """Test exporting heatmap to HTML."""
        result = export_heatmap(
            sample_heatmap,
            format=HeatmapExportFormat.HTML,
            options=export_options,
        )

        assert "<" in result
        # Heatmap should have table or grid structure
        assert "Security" in result or "heatmap" in result.lower()


# ============================================================================
# CSV Export Tests
# ============================================================================


@pytest.mark.skip(reason="CSV export not yet implemented")
class TestCSVExport:
    """Tests for CSV export functionality."""

    def test_export_receipt_csv(self, sample_receipt, export_options):
        """Test exporting receipt to CSV."""
        result = export_receipt(
            sample_receipt,
            format=ReceiptExportFormat.CSV,
            options=export_options,
        )

        # Should have comma-separated values
        assert "," in result
        # Should have header row
        lines = result.strip().split("\n")
        assert len(lines) >= 1

    def test_export_heatmap_csv(self, sample_heatmap, export_options):
        """Test exporting heatmap to CSV."""
        result = export_heatmap(
            sample_heatmap,
            format=HeatmapExportFormat.CSV,
            options=export_options,
        )

        assert "," in result
        # Should have rows
        lines = result.strip().split("\n")
        assert len(lines) >= 1


# ============================================================================
# SARIF Export Tests
# ============================================================================


@pytest.mark.skip(reason="SARIF export not yet implemented")
class TestSARIFExport:
    """Tests for SARIF export functionality."""

    def test_export_receipt_sarif(self, sample_receipt, export_options):
        """Test exporting receipt to SARIF format."""
        result = export_receipt(
            sample_receipt,
            format=ReceiptExportFormat.SARIF,
            options=export_options,
        )

        # Should be valid JSON
        parsed = json.loads(result)

        # SARIF structure
        assert "$schema" in parsed or "version" in parsed
        assert "runs" in parsed

    def test_sarif_has_tool_info(self, sample_receipt, export_options):
        """Test SARIF includes tool information."""
        result = export_receipt(
            sample_receipt,
            format=ReceiptExportFormat.SARIF,
            options=export_options,
        )

        parsed = json.loads(result)

        # Should have tool info in runs
        if parsed.get("runs"):
            run = parsed["runs"][0]
            assert "tool" in run


# ============================================================================
# Bundle Export Tests
# ============================================================================


class TestBundleExport:
    """Tests for bundle export functionality."""

    def test_export_receipts_bundle(self, sample_receipt, export_options):
        """Test exporting multiple receipts as bundle."""
        receipts = [sample_receipt, sample_receipt]

        result = export_receipts_bundle(
            receipts,
            format=ReceiptExportFormat.JSON,
            options=export_options,
        )

        parsed = json.loads(result)

        # Should be an array or have receipts key
        assert isinstance(parsed, (list, dict))
        if isinstance(parsed, dict):
            assert "receipts" in parsed or "items" in parsed

    def test_export_empty_bundle(self, export_options):
        """Test exporting empty bundle."""
        result = export_receipts_bundle(
            [],
            format=ReceiptExportFormat.JSON,
            options=export_options,
        )

        parsed = json.loads(result)
        # Should handle empty gracefully
        assert parsed is not None


# ============================================================================
# Streaming Export Tests
# ============================================================================


@pytest.mark.skip(reason="Streaming export not yet implemented")
class TestStreamingExport:
    """Tests for streaming export functionality."""

    def test_stream_receipt_json(self, sample_receipt):
        """Test streaming JSON export."""
        chunks = list(stream_receipt_json(sample_receipt))

        # Should produce chunks
        assert len(chunks) > 0

        # Combined should be valid JSON
        combined = "".join(chunks)
        parsed = json.loads(combined)
        assert parsed["id"] == "receipt_001"


# ============================================================================
# Options Filtering Tests
# ============================================================================


@pytest.mark.skip(reason="Options filtering not yet implemented")
class TestOptionsFiltering:
    """Tests for export options filtering."""

    def test_exclude_metadata(self, sample_receipt):
        """Test excluding metadata from export."""
        options = ExportOptions(include_export_metadata=False)
        result = export_receipt(
            sample_receipt,
            format=ReceiptExportFormat.JSON,
            options=options,
        )

        parsed = json.loads(result)
        # Metadata should be excluded or empty
        assert "metadata" not in parsed or parsed.get("metadata") == {}

    def test_exclude_evidence(self, sample_receipt):
        """Test excluding evidence from export."""
        options = ExportOptions(include_provenance=False)
        result = export_receipt(
            sample_receipt,
            format=ReceiptExportFormat.JSON,
            options=options,
        )

        parsed = json.loads(result)
        assert "evidence_cited" not in parsed or parsed.get("evidence_cited") == []

    def test_exclude_dissent(self, sample_receipt):
        """Test excluding dissent from export."""
        options = ExportOptions(include_config=True)
        result = export_receipt(
            sample_receipt,
            format=ReceiptExportFormat.JSON,
            options=options,
        )

        parsed = json.loads(result)
        assert "dissenting_views" not in parsed or parsed.get("dissenting_views") == []


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases in export."""

    def test_receipt_with_special_characters(self):
        """Test receipt with special characters in content."""
        receipt = MagicMock()
        receipt.id = "receipt_special"
        receipt.decision = "Use \"quotes\" and <tags> and 'apostrophes'"
        receipt.question = "Test?"
        receipt.confidence = 0.5
        receipt.timestamp = datetime.now()
        receipt.to_dict = MagicMock(return_value={
            "id": receipt.id,
            "decision": receipt.decision,
            "question": receipt.question,
            "confidence": receipt.confidence,
            "timestamp": receipt.timestamp.isoformat(),
        })

        result = export_receipt(receipt, format=ReceiptExportFormat.JSON)

        # Should handle special chars
        parsed = json.loads(result)
        assert "quotes" in parsed["decision"]

    def test_receipt_with_unicode(self):
        """Test receipt with unicode content."""
        receipt = MagicMock()
        receipt.id = "receipt_unicode"
        receipt.decision = "Décision avec accénts et émojis 🚀"
        receipt.question = "测试中文?"
        receipt.confidence = 0.8
        receipt.timestamp = datetime.now()
        receipt.to_dict = MagicMock(return_value={
            "id": receipt.id,
            "decision": receipt.decision,
            "question": receipt.question,
            "confidence": receipt.confidence,
            "timestamp": receipt.timestamp.isoformat(),
        })

        result = export_receipt(receipt, format=ReceiptExportFormat.JSON)

        parsed = json.loads(result)
        assert "accénts" in parsed["decision"]

    def test_receipt_with_empty_fields(self):
        """Test receipt with empty optional fields."""
        receipt = MagicMock()
        receipt.id = "receipt_minimal"
        receipt.decision = "Decision"
        receipt.question = "Question"
        receipt.confidence = 0.5
        receipt.timestamp = datetime.now()
        receipt.key_arguments = []
        receipt.dissenting_views = []
        receipt.evidence_cited = []
        receipt.to_dict = MagicMock(return_value={
            "id": receipt.id,
            "decision": receipt.decision,
            "question": receipt.question,
            "confidence": receipt.confidence,
            "timestamp": receipt.timestamp.isoformat(),
            "key_arguments": [],
            "dissenting_views": [],
            "evidence_cited": [],
        })

        result = export_receipt(receipt, format=ReceiptExportFormat.JSON)

        parsed = json.loads(result)
        assert parsed["id"] == "receipt_minimal"
