"""
Tests for GauntletHeatmapMixin (aragora/server/handlers/gauntlet/heatmap.py).

Covers:
- _get_heatmap: JSON/SVG/ASCII output formats
- In-memory run lookup (completed, not completed)
- Persistent storage fallback (found, not found, storage errors)
- Heatmap generation from result_obj (from_mode_result path)
- Heatmap generation from findings data (manual cell building path)
- Heatmap generation from vulnerabilities fallback
- Edge cases: empty findings, unknown categories, mixed severities
- Query parameter handling (format param)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.gauntlet.heatmap import (
    GauntletHeatmapMixin,
    _get_storage_proxy,
)
from aragora.server.handlers.gauntlet.storage import get_gauntlet_runs
from aragora.server.handlers.utils.responses import HandlerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: HandlerResult) -> dict[str, Any]:
    """Decode a HandlerResult JSON body into a dict."""
    return json.loads(result.body.decode("utf-8"))


def _status(result: HandlerResult) -> int:
    """Extract status code from HandlerResult."""
    return result.status_code


def _body_raw(result: HandlerResult) -> str:
    """Get body as decoded string."""
    return result.body.decode("utf-8")


# ---------------------------------------------------------------------------
# Fake objects
# ---------------------------------------------------------------------------


@dataclass
class FakeFinding:
    """Mimics a gauntlet finding for from_mode_result."""

    finding_id: str
    category: str
    severity_level: str


@dataclass
class FakeResultObj:
    """Mimics a GauntletResult with all_findings attribute."""

    all_findings: list[FakeFinding] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Mixin wrapper - instantiable test class
# ---------------------------------------------------------------------------


class HeatmapHandler(GauntletHeatmapMixin):
    """Minimal concrete class that includes the heatmap mixin."""

    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a HeatmapHandler instance."""
    return HeatmapHandler()


@pytest.fixture(autouse=True)
def clear_gauntlet_runs():
    """Ensure gauntlet runs storage is clean for each test."""
    runs = get_gauntlet_runs()
    runs.clear()
    yield
    runs.clear()


# ===========================================================================
# In-memory: run found, completed, JSON format (default)
# ===========================================================================


class TestGetHeatmapInMemoryCompleted:
    """Tests for _get_heatmap when run is found in-memory and completed."""

    @pytest.mark.asyncio
    async def test_json_format_default(self, handler):
        """Default format should return JSON with heatmap data."""
        runs = get_gauntlet_runs()
        runs["g-001"] = {
            "status": "completed",
            "result": {
                "findings": [
                    {"category": "injection", "severity_level": "high"},
                    {"category": "injection", "severity_level": "critical"},
                    {"category": "auth", "severity_level": "medium"},
                ],
                "total_findings": 3,
            },
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-001", {})
        assert _status(result) == 200
        body = _body(result)
        assert "cells" in body
        assert "categories" in body
        assert "severities" in body
        assert body["total_findings"] == 3

    @pytest.mark.asyncio
    async def test_json_format_explicit(self, handler):
        """Explicit format=json should return JSON."""
        runs = get_gauntlet_runs()
        runs["g-002"] = {
            "status": "completed",
            "result": {"findings": [], "total_findings": 0},
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-002", {"format": "json"})
        assert _status(result) == 200
        assert result.content_type == "application/json"

    @pytest.mark.asyncio
    async def test_svg_format(self, handler):
        """format=svg should return SVG image."""
        runs = get_gauntlet_runs()
        runs["g-003"] = {
            "status": "completed",
            "result": {
                "findings": [
                    {"category": "xss", "severity_level": "high"},
                ],
                "total_findings": 1,
            },
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-003", {"format": "svg"})
        assert _status(result) == 200
        assert result.content_type == "image/svg+xml"
        svg_content = _body_raw(result)
        assert "<svg" in svg_content

    @pytest.mark.asyncio
    async def test_ascii_format(self, handler):
        """format=ascii should return text/plain ASCII table."""
        runs = get_gauntlet_runs()
        runs["g-004"] = {
            "status": "completed",
            "result": {
                "findings": [
                    {"category": "auth", "severity_level": "low"},
                ],
                "total_findings": 1,
            },
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-004", {"format": "ascii"})
        assert _status(result) == 200
        assert result.content_type == "text/plain"
        ascii_content = _body_raw(result)
        assert "auth" in ascii_content

    @pytest.mark.asyncio
    async def test_empty_findings(self, handler):
        """Empty findings should still return valid heatmap."""
        runs = get_gauntlet_runs()
        runs["g-005"] = {
            "status": "completed",
            "result": {"findings": [], "total_findings": 0},
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-005", {})
        assert _status(result) == 200
        body = _body(result)
        assert body["cells"] == []
        assert body["categories"] == []
        assert body["total_findings"] == 0

    @pytest.mark.asyncio
    async def test_categories_sorted_alphabetically(self, handler):
        """Categories in heatmap should be sorted alphabetically."""
        runs = get_gauntlet_runs()
        runs["g-006"] = {
            "status": "completed",
            "result": {
                "findings": [
                    {"category": "zzz", "severity_level": "high"},
                    {"category": "aaa", "severity_level": "low"},
                    {"category": "mmm", "severity_level": "medium"},
                ],
                "total_findings": 3,
            },
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-006", {})
        body = _body(result)
        assert body["categories"] == ["aaa", "mmm", "zzz"]

    @pytest.mark.asyncio
    async def test_severities_order(self, handler):
        """Severities should always be critical, high, medium, low."""
        runs = get_gauntlet_runs()
        runs["g-007"] = {
            "status": "completed",
            "result": {
                "findings": [{"category": "test", "severity_level": "low"}],
                "total_findings": 1,
            },
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-007", {})
        body = _body(result)
        assert body["severities"] == ["critical", "high", "medium", "low"]

    @pytest.mark.asyncio
    async def test_cell_counts_correct(self, handler):
        """Each cell should have the correct count of findings."""
        runs = get_gauntlet_runs()
        runs["g-008"] = {
            "status": "completed",
            "result": {
                "findings": [
                    {"category": "injection", "severity_level": "high"},
                    {"category": "injection", "severity_level": "high"},
                    {"category": "injection", "severity_level": "high"},
                    {"category": "injection", "severity_level": "critical"},
                ],
                "total_findings": 4,
            },
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-008", {})
        body = _body(result)

        cells_by_key = {(c["category"], c["severity"]): c["count"] for c in body["cells"]}
        assert cells_by_key[("injection", "high")] == 3
        assert cells_by_key[("injection", "critical")] == 1
        assert cells_by_key[("injection", "medium")] == 0
        assert cells_by_key[("injection", "low")] == 0

    @pytest.mark.asyncio
    async def test_multiple_categories(self, handler):
        """Heatmap should handle multiple categories correctly."""
        runs = get_gauntlet_runs()
        runs["g-009"] = {
            "status": "completed",
            "result": {
                "findings": [
                    {"category": "auth", "severity_level": "high"},
                    {"category": "injection", "severity_level": "medium"},
                    {"category": "xss", "severity_level": "low"},
                ],
                "total_findings": 3,
            },
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-009", {})
        body = _body(result)
        assert body["categories"] == ["auth", "injection", "xss"]
        # 3 categories x 4 severities = 12 cells
        assert len(body["cells"]) == 12

    @pytest.mark.asyncio
    async def test_cells_per_category_always_four_severities(self, handler):
        """Each category should produce exactly 4 cells (one per severity)."""
        runs = get_gauntlet_runs()
        runs["g-010"] = {
            "status": "completed",
            "result": {
                "findings": [
                    {"category": "a", "severity_level": "high"},
                    {"category": "b", "severity_level": "critical"},
                ],
                "total_findings": 2,
            },
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-010", {})
        body = _body(result)
        # 2 categories x 4 severities
        assert len(body["cells"]) == 8

    @pytest.mark.asyncio
    async def test_unknown_category_default(self, handler):
        """Findings without a category should use 'unknown'."""
        runs = get_gauntlet_runs()
        runs["g-011"] = {
            "status": "completed",
            "result": {
                "findings": [
                    {"severity_level": "high"},  # no 'category' key
                ],
                "total_findings": 1,
            },
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-011", {})
        body = _body(result)
        assert "unknown" in body["categories"]

    @pytest.mark.asyncio
    async def test_severity_defaults_to_medium(self, handler):
        """Findings without severity_level should default to 'medium'."""
        runs = get_gauntlet_runs()
        runs["g-012"] = {
            "status": "completed",
            "result": {
                "findings": [
                    {"category": "test"},  # no 'severity_level'
                ],
                "total_findings": 1,
            },
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-012", {})
        body = _body(result)
        cells_by_key = {(c["category"], c["severity"]): c["count"] for c in body["cells"]}
        assert cells_by_key[("test", "medium")] == 1

    @pytest.mark.asyncio
    async def test_severity_case_insensitive(self, handler):
        """Severity level should be lowercased."""
        runs = get_gauntlet_runs()
        runs["g-013"] = {
            "status": "completed",
            "result": {
                "findings": [
                    {"category": "test", "severity_level": "HIGH"},
                    {"category": "test", "severity_level": "High"},
                ],
                "total_findings": 2,
            },
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-013", {})
        body = _body(result)
        cells_by_key = {(c["category"], c["severity"]): c["count"] for c in body["cells"]}
        assert cells_by_key[("test", "high")] == 2


# ===========================================================================
# In-memory: run not completed
# ===========================================================================


class TestGetHeatmapNotCompleted:
    """Tests for _get_heatmap when run is in-memory but not completed."""

    @pytest.mark.asyncio
    async def test_pending_returns_400(self, handler):
        """Pending run should return 400 error."""
        runs = get_gauntlet_runs()
        runs["g-pending"] = {"status": "pending"}

        result = await handler._get_heatmap("g-pending", {})
        assert _status(result) == 400
        body = _body(result)
        assert "not completed" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_running_returns_400(self, handler):
        """Running state should also return 400."""
        runs = get_gauntlet_runs()
        runs["g-running"] = {"status": "running"}

        result = await handler._get_heatmap("g-running", {})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_failed_returns_400(self, handler):
        """Failed status should return 400."""
        runs = get_gauntlet_runs()
        runs["g-failed"] = {"status": "failed"}

        result = await handler._get_heatmap("g-failed", {})
        assert _status(result) == 400


# ===========================================================================
# Persistent storage fallback
# ===========================================================================


class TestGetHeatmapPersistentStorage:
    """Tests for _get_heatmap falling back to persistent storage."""

    @pytest.mark.asyncio
    async def test_storage_found(self, handler):
        """When not in-memory, should check persistent storage."""
        mock_storage = MagicMock()
        mock_storage.get.return_value = {
            "findings": [
                {"category": "logic", "severity_level": "medium"},
            ],
            "total_findings": 1,
        }

        with patch(
            "aragora.server.handlers.gauntlet.heatmap._get_storage_proxy",
            return_value=mock_storage,
        ):
            result = await handler._get_heatmap("g-stored", {})

        assert _status(result) == 200
        body = _body(result)
        assert "logic" in body["categories"]

    @pytest.mark.asyncio
    async def test_storage_not_found(self, handler):
        """When not in storage either, should return 404."""
        mock_storage = MagicMock()
        mock_storage.get.return_value = None

        with patch(
            "aragora.server.handlers.gauntlet.heatmap._get_storage_proxy",
            return_value=mock_storage,
        ):
            result = await handler._get_heatmap("g-missing", {})

        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_storage_os_error(self, handler):
        """OSError from storage should return 404 gracefully."""
        mock_storage = MagicMock()
        mock_storage.get.side_effect = OSError("disk read failure")

        with patch(
            "aragora.server.handlers.gauntlet.heatmap._get_storage_proxy",
            return_value=mock_storage,
        ):
            result = await handler._get_heatmap("g-oserr", {})

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_storage_runtime_error(self, handler):
        """RuntimeError from storage should return 404 gracefully."""
        mock_storage = MagicMock()
        mock_storage.get.side_effect = RuntimeError("connection lost")

        with patch(
            "aragora.server.handlers.gauntlet.heatmap._get_storage_proxy",
            return_value=mock_storage,
        ):
            result = await handler._get_heatmap("g-rterr", {})

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_storage_value_error(self, handler):
        """ValueError from storage should return 404 gracefully."""
        mock_storage = MagicMock()
        mock_storage.get.side_effect = ValueError("corrupt data")

        with patch(
            "aragora.server.handlers.gauntlet.heatmap._get_storage_proxy",
            return_value=mock_storage,
        ):
            result = await handler._get_heatmap("g-valerr", {})

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_storage_result_used_for_heatmap(self, handler):
        """Storage result should be used to build heatmap cells."""
        mock_storage = MagicMock()
        mock_storage.get.return_value = {
            "findings": [
                {"category": "crypto", "severity_level": "critical"},
                {"category": "crypto", "severity_level": "critical"},
                {"category": "network", "severity_level": "low"},
            ],
            "total_findings": 3,
        }

        with patch(
            "aragora.server.handlers.gauntlet.heatmap._get_storage_proxy",
            return_value=mock_storage,
        ):
            result = await handler._get_heatmap("g-stored2", {})

        body = _body(result)
        cells_by_key = {(c["category"], c["severity"]): c["count"] for c in body["cells"]}
        assert cells_by_key[("crypto", "critical")] == 2
        assert cells_by_key[("network", "low")] == 1


# ===========================================================================
# result_obj path (from_mode_result)
# ===========================================================================


class TestGetHeatmapFromResultObj:
    """Tests for _get_heatmap using result_obj (from_mode_result path)."""

    @pytest.mark.asyncio
    async def test_result_obj_used_when_present(self, handler):
        """When result_obj is present, should use RiskHeatmap.from_mode_result."""
        fake_result = FakeResultObj(
            all_findings=[
                FakeFinding("f-1", "injection", "high"),
                FakeFinding("f-2", "injection", "critical"),
                FakeFinding("f-3", "xss", "medium"),
            ]
        )

        runs = get_gauntlet_runs()
        runs["g-obj"] = {
            "status": "completed",
            "result": {"findings": [], "total_findings": 0},
            "result_obj": fake_result,
        }

        result = await handler._get_heatmap("g-obj", {})
        assert _status(result) == 200
        body = _body(result)
        assert "injection" in body["categories"]
        assert "xss" in body["categories"]

    @pytest.mark.asyncio
    async def test_result_obj_svg_format(self, handler):
        """result_obj path should also support SVG output."""
        fake_result = FakeResultObj(
            all_findings=[
                FakeFinding("f-1", "auth", "high"),
            ]
        )

        runs = get_gauntlet_runs()
        runs["g-obj-svg"] = {
            "status": "completed",
            "result": {"findings": [], "total_findings": 0},
            "result_obj": fake_result,
        }

        result = await handler._get_heatmap("g-obj-svg", {"format": "svg"})
        assert _status(result) == 200
        assert result.content_type == "image/svg+xml"
        assert "<svg" in _body_raw(result)

    @pytest.mark.asyncio
    async def test_result_obj_ascii_format(self, handler):
        """result_obj path should also support ASCII output."""
        fake_result = FakeResultObj(
            all_findings=[
                FakeFinding("f-1", "auth", "low"),
            ]
        )

        runs = get_gauntlet_runs()
        runs["g-obj-ascii"] = {
            "status": "completed",
            "result": {"findings": [], "total_findings": 0},
            "result_obj": fake_result,
        }

        result = await handler._get_heatmap("g-obj-ascii", {"format": "ascii"})
        assert _status(result) == 200
        assert result.content_type == "text/plain"
        assert "auth" in _body_raw(result)

    @pytest.mark.asyncio
    async def test_result_obj_empty_findings(self, handler):
        """result_obj with no findings should return empty heatmap."""
        fake_result = FakeResultObj(all_findings=[])

        runs = get_gauntlet_runs()
        runs["g-obj-empty"] = {
            "status": "completed",
            "result": {"findings": [], "total_findings": 0},
            "result_obj": fake_result,
        }

        result = await handler._get_heatmap("g-obj-empty", {})
        body = _body(result)
        assert body["categories"] == []
        assert body["cells"] == []

    @pytest.mark.asyncio
    async def test_result_obj_cell_counts(self, handler):
        """result_obj heatmap should have correct cell counts."""
        fake_result = FakeResultObj(
            all_findings=[
                FakeFinding("f-1", "rce", "critical"),
                FakeFinding("f-2", "rce", "critical"),
                FakeFinding("f-3", "rce", "high"),
            ]
        )

        runs = get_gauntlet_runs()
        runs["g-obj-count"] = {
            "status": "completed",
            "result": {"findings": [], "total_findings": 0},
            "result_obj": fake_result,
        }

        result = await handler._get_heatmap("g-obj-count", {})
        body = _body(result)
        cells_by_key = {(c["category"], c["severity"]): c["count"] for c in body["cells"]}
        assert cells_by_key[("rce", "critical")] == 2
        assert cells_by_key[("rce", "high")] == 1
        assert cells_by_key[("rce", "medium")] == 0


# ===========================================================================
# Vulnerabilities fallback (compatibility)
# ===========================================================================


class TestGetHeatmapVulnerabilitiesFallback:
    """Tests for using 'vulnerabilities' key when 'findings' is absent."""

    @pytest.mark.asyncio
    async def test_vulnerabilities_fallback(self, handler):
        """When findings key is missing, should use vulnerabilities."""
        runs = get_gauntlet_runs()
        runs["g-vuln"] = {
            "status": "completed",
            "result": {
                "vulnerabilities": [
                    {"category": "sqli", "severity_level": "critical"},
                ],
                "total_findings": 1,
            },
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-vuln", {})
        assert _status(result) == 200
        body = _body(result)
        assert "sqli" in body["categories"]

    @pytest.mark.asyncio
    async def test_findings_empty_list_uses_vulnerabilities(self, handler):
        """When findings is empty list, should fallback to vulnerabilities."""
        runs = get_gauntlet_runs()
        runs["g-vuln2"] = {
            "status": "completed",
            "result": {
                "findings": [],
                "vulnerabilities": [
                    {"category": "csrf", "severity_level": "high"},
                ],
                "total_findings": 1,
            },
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-vuln2", {})
        body = _body(result)
        # Falsy findings = [], so `or` kicks in → uses vulnerabilities
        assert "csrf" in body["categories"]

    @pytest.mark.asyncio
    async def test_findings_none_uses_vulnerabilities(self, handler):
        """When findings is None, should fallback to vulnerabilities."""
        runs = get_gauntlet_runs()
        runs["g-vuln3"] = {
            "status": "completed",
            "result": {
                "findings": None,
                "vulnerabilities": [
                    {"category": "ssrf", "severity_level": "medium"},
                ],
                "total_findings": 1,
            },
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-vuln3", {})
        body = _body(result)
        assert "ssrf" in body["categories"]

    @pytest.mark.asyncio
    async def test_no_findings_no_vulnerabilities(self, handler):
        """When neither findings nor vulnerabilities exist, should return empty heatmap."""
        runs = get_gauntlet_runs()
        runs["g-nothing"] = {
            "status": "completed",
            "result": {"total_findings": 0},
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-nothing", {})
        body = _body(result)
        assert body["cells"] == []
        assert body["categories"] == []


# ===========================================================================
# Format edge cases
# ===========================================================================


class TestGetHeatmapFormatEdgeCases:
    """Tests for format query parameter edge cases."""

    @pytest.mark.asyncio
    async def test_unknown_format_defaults_to_json(self, handler):
        """Unknown format value should default to JSON."""
        runs = get_gauntlet_runs()
        runs["g-fmt"] = {
            "status": "completed",
            "result": {"findings": [], "total_findings": 0},
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-fmt", {"format": "xml"})
        assert _status(result) == 200
        assert result.content_type == "application/json"

    @pytest.mark.asyncio
    async def test_format_param_list_value(self, handler):
        """Format param as list (from query string) should take first value."""
        runs = get_gauntlet_runs()
        runs["g-fmtlist"] = {
            "status": "completed",
            "result": {"findings": [], "total_findings": 0},
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-fmtlist", {"format": ["svg", "json"]})
        assert _status(result) == 200
        assert result.content_type == "image/svg+xml"

    @pytest.mark.asyncio
    async def test_svg_empty_findings_no_data_svg(self, handler):
        """SVG with no data should return 'No data' SVG."""
        runs = get_gauntlet_runs()
        runs["g-svgempty"] = {
            "status": "completed",
            "result": {"findings": [], "total_findings": 0},
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-svgempty", {"format": "svg"})
        assert _status(result) == 200
        svg = _body_raw(result)
        assert "No data" in svg

    @pytest.mark.asyncio
    async def test_ascii_empty_findings_message(self, handler):
        """ASCII with no data should return 'No findings to display'."""
        runs = get_gauntlet_runs()
        runs["g-ascempty"] = {
            "status": "completed",
            "result": {"findings": [], "total_findings": 0},
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-ascempty", {"format": "ascii"})
        assert _status(result) == 200
        assert "No findings to display" in _body_raw(result)


# ===========================================================================
# _get_storage_proxy
# ===========================================================================


class TestGetStorageProxy:
    """Tests for the _get_storage_proxy helper."""

    def test_returns_storage_instance(self):
        """_get_storage_proxy should call _get_storage from the package."""
        mock_storage = MagicMock()
        with patch("aragora.server.handlers.gauntlet.heatmap._get_storage_proxy") as mock_proxy:
            mock_proxy.return_value = mock_storage
            result = mock_proxy()
            assert result is mock_storage

    def test_actual_proxy_calls_get_storage(self):
        """Verify _get_storage_proxy delegates to package _get_storage."""
        mock_storage = MagicMock()
        with patch(
            "aragora.server.handlers.gauntlet._get_storage",
            return_value=mock_storage,
        ):
            result = _get_storage_proxy()
            assert result is mock_storage


# ===========================================================================
# Additional edge cases and content verification
# ===========================================================================


class TestGetHeatmapContentDetails:
    """Detailed content verification tests."""

    @pytest.mark.asyncio
    async def test_json_contains_matrix(self, handler):
        """JSON output should include the matrix field from to_dict()."""
        runs = get_gauntlet_runs()
        runs["g-mat"] = {
            "status": "completed",
            "result": {
                "findings": [
                    {"category": "auth", "severity_level": "high"},
                ],
                "total_findings": 1,
            },
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-mat", {})
        body = _body(result)
        assert "matrix" in body
        # 1 category, should be a list with one row
        assert len(body["matrix"]) == 1
        # Each row has 4 severity columns
        assert len(body["matrix"][0]) == 4

    @pytest.mark.asyncio
    async def test_total_findings_from_result(self, handler):
        """total_findings should come from the result dict."""
        runs = get_gauntlet_runs()
        runs["g-total"] = {
            "status": "completed",
            "result": {
                "findings": [
                    {"category": "a", "severity_level": "low"},
                ],
                "total_findings": 42,
            },
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-total", {})
        body = _body(result)
        assert body["total_findings"] == 42

    @pytest.mark.asyncio
    async def test_total_findings_default_zero(self, handler):
        """Missing total_findings key should default to 0."""
        runs = get_gauntlet_runs()
        runs["g-nocount"] = {
            "status": "completed",
            "result": {
                "findings": [
                    {"category": "a", "severity_level": "low"},
                ],
            },
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-nocount", {})
        body = _body(result)
        assert body["total_findings"] == 0

    @pytest.mark.asyncio
    async def test_svg_contains_category_labels(self, handler):
        """SVG should include category names as text labels."""
        runs = get_gauntlet_runs()
        runs["g-svglabel"] = {
            "status": "completed",
            "result": {
                "findings": [
                    {"category": "injection", "severity_level": "high"},
                ],
                "total_findings": 1,
            },
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-svglabel", {"format": "svg"})
        svg = _body_raw(result)
        assert "injection" in svg

    @pytest.mark.asyncio
    async def test_svg_contains_severity_headers(self, handler):
        """SVG should include severity headers."""
        runs = get_gauntlet_runs()
        runs["g-svghdr"] = {
            "status": "completed",
            "result": {
                "findings": [
                    {"category": "test", "severity_level": "high"},
                ],
                "total_findings": 1,
            },
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-svghdr", {"format": "svg"})
        svg = _body_raw(result)
        assert "CRITICAL" in svg
        assert "HIGH" in svg

    @pytest.mark.asyncio
    async def test_ascii_contains_total_row(self, handler):
        """ASCII output should contain a TOTAL row."""
        runs = get_gauntlet_runs()
        runs["g-asctotal"] = {
            "status": "completed",
            "result": {
                "findings": [
                    {"category": "test", "severity_level": "high"},
                ],
                "total_findings": 1,
            },
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-asctotal", {"format": "ascii"})
        ascii_text = _body_raw(result)
        assert "TOTAL" in ascii_text

    @pytest.mark.asyncio
    async def test_heatmap_cell_intensity_in_json(self, handler):
        """JSON cells should include an intensity field."""
        runs = get_gauntlet_runs()
        runs["g-intensity"] = {
            "status": "completed",
            "result": {
                "findings": [
                    {"category": "test", "severity_level": "high"},
                ],
                "total_findings": 1,
            },
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-intensity", {})
        body = _body(result)
        # Find the cell with count > 0
        non_zero_cells = [c for c in body["cells"] if c["count"] > 0]
        assert len(non_zero_cells) == 1
        assert non_zero_cells[0]["intensity"] > 0

    @pytest.mark.asyncio
    async def test_zero_count_cells_have_zero_intensity(self, handler):
        """Cells with count=0 should have intensity=0."""
        runs = get_gauntlet_runs()
        runs["g-zeroint"] = {
            "status": "completed",
            "result": {
                "findings": [
                    {"category": "test", "severity_level": "high"},
                ],
                "total_findings": 1,
            },
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-zeroint", {})
        body = _body(result)
        zero_cells = [c for c in body["cells"] if c["count"] == 0]
        assert len(zero_cells) > 0
        for cell in zero_cells:
            assert cell["intensity"] == 0.0

    @pytest.mark.asyncio
    async def test_duplicate_categories_collapsed(self, handler):
        """Multiple findings in same category should not duplicate the category."""
        runs = get_gauntlet_runs()
        runs["g-dup"] = {
            "status": "completed",
            "result": {
                "findings": [
                    {"category": "auth", "severity_level": "high"},
                    {"category": "auth", "severity_level": "low"},
                    {"category": "auth", "severity_level": "critical"},
                ],
                "total_findings": 3,
            },
            "result_obj": None,
        }

        result = await handler._get_heatmap("g-dup", {})
        body = _body(result)
        assert body["categories"] == ["auth"]
        assert len(body["cells"]) == 4  # 1 category x 4 severities

    @pytest.mark.asyncio
    async def test_error_response_contains_gauntlet_id(self, handler):
        """404 error response should include the gauntlet ID."""
        mock_storage = MagicMock()
        mock_storage.get.return_value = None

        with patch(
            "aragora.server.handlers.gauntlet.heatmap._get_storage_proxy",
            return_value=mock_storage,
        ):
            result = await handler._get_heatmap("g-xyz-789", {})

        body = _body(result)
        assert "g-xyz-789" in body.get("error", "")
