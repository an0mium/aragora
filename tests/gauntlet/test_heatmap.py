"""
Tests for aragora.gauntlet.heatmap module.

Covers HeatmapCell and RiskHeatmap dataclasses including:
- Cell creation, intensity calculation, serialization
- Heatmap construction from GauntletResult and mode results
- Cell lookup, category/severity totals
- Matrix, JSON, SVG, and ASCII export
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock

import pytest

from aragora.gauntlet.heatmap import HeatmapCell, RiskHeatmap
from aragora.gauntlet.types import RiskSummary, SeverityLevel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vulnerability(
    vuln_id: str,
    severity: SeverityLevel,
    category: str,
) -> MagicMock:
    """Create a mock Vulnerability for GauntletResult.from_result."""
    vuln = MagicMock()
    vuln.id = vuln_id
    vuln.severity = severity
    vuln.category = category
    return vuln


def _make_gauntlet_result(vulns: list[MagicMock], total: int | None = None) -> MagicMock:
    """Create a mock GauntletResult with vulnerabilities and risk_summary."""
    result = MagicMock()
    result.vulnerabilities = vulns
    result.risk_summary = MagicMock()
    result.risk_summary.total = total if total is not None else len(vulns)
    return result


def _make_mode_finding(
    finding_id: str,
    severity_level: str,
    category: str,
) -> MagicMock:
    """Create a mock finding for from_mode_result (generic result)."""
    finding = MagicMock()
    finding.finding_id = finding_id
    finding.severity_level = severity_level
    finding.category = category
    return finding


def _make_mode_result(findings: list[MagicMock]) -> MagicMock:
    """Create a mock result with all_findings attribute."""
    result = MagicMock()
    result.all_findings = findings
    return result


# ===========================================================================
# HeatmapCell Tests
# ===========================================================================


class TestHeatmapCell:
    """Tests for HeatmapCell dataclass."""

    def test_creation_with_defaults(self):
        """HeatmapCell can be created with required fields; vulnerabilities defaults to empty list."""
        cell = HeatmapCell(category="injection", severity="critical", count=5)
        assert cell.category == "injection"
        assert cell.severity == "critical"
        assert cell.count == 5
        assert cell.vulnerabilities == []

    def test_creation_with_vulnerabilities(self):
        """HeatmapCell stores vulnerability IDs when provided."""
        ids = ["v-1", "v-2"]
        cell = HeatmapCell(category="xss", severity="high", count=2, vulnerabilities=ids)
        assert cell.vulnerabilities == ids

    # ------- intensity property ------- #

    def test_intensity_count_zero(self):
        """count=0 yields intensity 0.0."""
        cell = HeatmapCell(category="a", severity="low", count=0)
        assert cell.intensity == 0.0

    def test_intensity_count_one(self):
        """count=1 yields log10(2)/2."""
        cell = HeatmapCell(category="a", severity="low", count=1)
        expected = math.log10(2) / 2
        assert cell.intensity == pytest.approx(expected)

    def test_intensity_count_nine(self):
        """count=9 yields log10(10)/2 = 0.5."""
        cell = HeatmapCell(category="a", severity="low", count=9)
        expected = math.log10(10) / 2  # 0.5
        assert cell.intensity == pytest.approx(0.5)

    def test_intensity_count_ninety_nine(self):
        """count=99 yields log10(100)/2 = 1.0."""
        cell = HeatmapCell(category="a", severity="low", count=99)
        expected = math.log10(100) / 2  # 1.0
        assert cell.intensity == pytest.approx(1.0)

    def test_intensity_capped_at_one(self):
        """Counts above 99 still cap intensity at 1.0."""
        cell = HeatmapCell(category="a", severity="low", count=1000)
        assert cell.intensity == 1.0

    def test_intensity_large_count(self):
        """Very large count still returns 1.0."""
        cell = HeatmapCell(category="a", severity="low", count=999999)
        assert cell.intensity == 1.0

    # ------- to_dict ------- #

    def test_to_dict_includes_all_fields(self):
        """to_dict returns category, severity, count, intensity, vulnerabilities."""
        cell = HeatmapCell(
            category="auth",
            severity="medium",
            count=9,
            vulnerabilities=["v-10"],
        )
        d = cell.to_dict()
        assert d["category"] == "auth"
        assert d["severity"] == "medium"
        assert d["count"] == 9
        assert d["intensity"] == pytest.approx(0.5)
        assert d["vulnerabilities"] == ["v-10"]

    def test_to_dict_zero_count(self):
        """to_dict correctly reports intensity 0.0 for zero-count cell."""
        cell = HeatmapCell(category="a", severity="low", count=0)
        assert cell.to_dict()["intensity"] == 0.0


# ===========================================================================
# RiskHeatmap — basic construction
# ===========================================================================


class TestRiskHeatmapCreation:
    """Tests for RiskHeatmap default construction."""

    def test_default_fields(self):
        """RiskHeatmap has sensible defaults."""
        hm = RiskHeatmap()
        assert hm.cells == []
        assert hm.categories == []
        assert hm.severities == ["critical", "high", "medium", "low"]
        assert hm.total_findings == 0
        assert hm.highest_risk_category is None
        assert hm.highest_risk_severity is None

    def test_custom_fields(self):
        """RiskHeatmap accepts custom values."""
        cell = HeatmapCell(category="xss", severity="high", count=3)
        hm = RiskHeatmap(
            cells=[cell],
            categories=["xss"],
            total_findings=3,
            highest_risk_category="xss",
            highest_risk_severity="high",
        )
        assert len(hm.cells) == 1
        assert hm.total_findings == 3


# ===========================================================================
# RiskHeatmap.from_result
# ===========================================================================


class TestRiskHeatmapFromResult:
    """Tests for RiskHeatmap.from_result classmethod."""

    def test_from_result_basic(self):
        """from_result builds cells from vulnerabilities and identifies highest risk."""
        vulns = [
            _make_vulnerability("v1", SeverityLevel.CRITICAL, "injection"),
            _make_vulnerability("v2", SeverityLevel.HIGH, "injection"),
            _make_vulnerability("v3", SeverityLevel.MEDIUM, "xss"),
        ]
        result = _make_gauntlet_result(vulns, total=3)

        hm = RiskHeatmap.from_result(result)

        # Categories sorted alphabetically
        assert hm.categories == ["injection", "xss"]
        assert hm.severities == ["critical", "high", "medium", "low"]
        assert hm.total_findings == 3

        # Check individual cells
        cell_crit_inj = hm.get_cell("injection", "critical")
        assert cell_crit_inj is not None
        assert cell_crit_inj.count == 1
        assert "v1" in cell_crit_inj.vulnerabilities

        cell_high_inj = hm.get_cell("injection", "high")
        assert cell_high_inj is not None
        assert cell_high_inj.count == 1
        assert "v2" in cell_high_inj.vulnerabilities

        cell_med_xss = hm.get_cell("xss", "medium")
        assert cell_med_xss is not None
        assert cell_med_xss.count == 1

        # Zero-count cells should still exist
        cell_low_inj = hm.get_cell("injection", "low")
        assert cell_low_inj is not None
        assert cell_low_inj.count == 0

    def test_from_result_highest_risk_category(self):
        """highest_risk_category is the category with most findings."""
        vulns = [
            _make_vulnerability("v1", SeverityLevel.HIGH, "auth"),
            _make_vulnerability("v2", SeverityLevel.HIGH, "auth"),
            _make_vulnerability("v3", SeverityLevel.LOW, "crypto"),
        ]
        result = _make_gauntlet_result(vulns, total=3)
        hm = RiskHeatmap.from_result(result)
        assert hm.highest_risk_category == "auth"

    def test_from_result_highest_severity_critical(self):
        """highest_risk_severity is 'critical' when critical findings exist."""
        vulns = [
            _make_vulnerability("v1", SeverityLevel.CRITICAL, "a"),
            _make_vulnerability("v2", SeverityLevel.LOW, "b"),
        ]
        result = _make_gauntlet_result(vulns, total=2)
        hm = RiskHeatmap.from_result(result)
        assert hm.highest_risk_severity == "critical"

    def test_from_result_highest_severity_high(self):
        """highest_risk_severity is 'high' when only high (no critical) findings exist."""
        vulns = [
            _make_vulnerability("v1", SeverityLevel.HIGH, "a"),
            _make_vulnerability("v2", SeverityLevel.MEDIUM, "a"),
        ]
        result = _make_gauntlet_result(vulns, total=2)
        hm = RiskHeatmap.from_result(result)
        assert hm.highest_risk_severity == "high"

    def test_from_result_highest_severity_none_when_only_medium(self):
        """highest_risk_severity is None when only medium/low findings exist."""
        vulns = [
            _make_vulnerability("v1", SeverityLevel.MEDIUM, "a"),
        ]
        result = _make_gauntlet_result(vulns, total=1)
        hm = RiskHeatmap.from_result(result)
        assert hm.highest_risk_severity is None

    def test_from_result_empty_vulnerabilities(self):
        """from_result with no vulnerabilities yields empty heatmap."""
        result = _make_gauntlet_result([], total=0)
        hm = RiskHeatmap.from_result(result)
        assert hm.cells == []
        assert hm.categories == []
        assert hm.total_findings == 0
        assert hm.highest_risk_category is None
        assert hm.highest_risk_severity is None

    def test_from_result_total_from_risk_summary(self):
        """total_findings comes from result.risk_summary.total, not len(vulns)."""
        vulns = [
            _make_vulnerability("v1", SeverityLevel.LOW, "a"),
        ]
        # risk_summary.total can differ from len(vulnerabilities)
        result = _make_gauntlet_result(vulns, total=42)
        hm = RiskHeatmap.from_result(result)
        assert hm.total_findings == 42

    def test_from_result_cells_count(self):
        """Number of cells = categories * severities."""
        vulns = [
            _make_vulnerability("v1", SeverityLevel.HIGH, "cat_a"),
            _make_vulnerability("v2", SeverityLevel.LOW, "cat_b"),
            _make_vulnerability("v3", SeverityLevel.MEDIUM, "cat_c"),
        ]
        result = _make_gauntlet_result(vulns, total=3)
        hm = RiskHeatmap.from_result(result)
        # 3 categories x 4 severities = 12 cells
        assert len(hm.cells) == 12


# ===========================================================================
# RiskHeatmap.from_mode_result
# ===========================================================================


class TestRiskHeatmapFromModeResult:
    """Tests for RiskHeatmap.from_mode_result classmethod."""

    def test_from_mode_result_basic(self):
        """from_mode_result builds cells from all_findings attribute."""
        findings = [
            _make_mode_finding("f1", "Critical", "injection"),
            _make_mode_finding("f2", "High", "injection"),
            _make_mode_finding("f3", "Medium", "xss"),
        ]
        result = _make_mode_result(findings)

        hm = RiskHeatmap.from_mode_result(result)

        assert hm.categories == ["injection", "xss"]
        assert hm.total_findings == 3

        cell_crit_inj = hm.get_cell("injection", "critical")
        assert cell_crit_inj is not None
        assert cell_crit_inj.count == 1
        assert "f1" in cell_crit_inj.vulnerabilities

    def test_from_mode_result_empty_findings(self):
        """from_mode_result with empty all_findings gives empty heatmap."""
        result = _make_mode_result([])
        hm = RiskHeatmap.from_mode_result(result)
        assert hm.cells == []
        assert hm.categories == []
        assert hm.total_findings == 0

    def test_from_mode_result_no_all_findings_attr(self):
        """from_mode_result handles result without all_findings (falls back to [])."""
        result = MagicMock(spec=[])  # no attributes
        hm = RiskHeatmap.from_mode_result(result)
        assert hm.cells == []
        assert hm.total_findings == 0

    def test_from_mode_result_highest_risk(self):
        """from_mode_result correctly identifies highest risk category and severity."""
        findings = [
            _make_mode_finding("f1", "Critical", "auth"),
            _make_mode_finding("f2", "Critical", "auth"),
            _make_mode_finding("f3", "High", "crypto"),
        ]
        result = _make_mode_result(findings)
        hm = RiskHeatmap.from_mode_result(result)
        assert hm.highest_risk_category == "auth"
        assert hm.highest_risk_severity == "critical"

    def test_from_mode_result_severity_case_insensitive(self):
        """from_mode_result lowercases severity_level for matching."""
        findings = [
            _make_mode_finding("f1", "HIGH", "a"),
            _make_mode_finding("f2", "High", "a"),
            _make_mode_finding("f3", "high", "a"),
        ]
        result = _make_mode_result(findings)
        hm = RiskHeatmap.from_mode_result(result)
        cell = hm.get_cell("a", "high")
        assert cell is not None
        assert cell.count == 3


# ===========================================================================
# RiskHeatmap.get_cell
# ===========================================================================


class TestRiskHeatmapGetCell:
    """Tests for RiskHeatmap.get_cell."""

    def test_get_cell_found(self):
        """get_cell returns matching cell."""
        cell = HeatmapCell(category="xss", severity="high", count=2)
        hm = RiskHeatmap(cells=[cell], categories=["xss"])
        found = hm.get_cell("xss", "high")
        assert found is cell

    def test_get_cell_not_found(self):
        """get_cell returns None when no match."""
        cell = HeatmapCell(category="xss", severity="high", count=2)
        hm = RiskHeatmap(cells=[cell], categories=["xss"])
        assert hm.get_cell("xss", "low") is None
        assert hm.get_cell("injection", "high") is None

    def test_get_cell_empty_heatmap(self):
        """get_cell returns None on empty heatmap."""
        hm = RiskHeatmap()
        assert hm.get_cell("any", "any") is None


# ===========================================================================
# RiskHeatmap.get_category_total / get_severity_total
# ===========================================================================


class TestRiskHeatmapTotals:
    """Tests for get_category_total and get_severity_total."""

    @pytest.fixture()
    def populated_heatmap(self) -> RiskHeatmap:
        """Heatmap with known cell counts."""
        cells = [
            HeatmapCell(category="injection", severity="critical", count=3),
            HeatmapCell(category="injection", severity="high", count=2),
            HeatmapCell(category="injection", severity="medium", count=0),
            HeatmapCell(category="injection", severity="low", count=1),
            HeatmapCell(category="xss", severity="critical", count=0),
            HeatmapCell(category="xss", severity="high", count=1),
            HeatmapCell(category="xss", severity="medium", count=4),
            HeatmapCell(category="xss", severity="low", count=0),
        ]
        return RiskHeatmap(
            cells=cells,
            categories=["injection", "xss"],
            total_findings=11,
        )

    def test_get_category_total(self, populated_heatmap: RiskHeatmap):
        """get_category_total sums counts for a given category."""
        assert populated_heatmap.get_category_total("injection") == 6  # 3+2+0+1
        assert populated_heatmap.get_category_total("xss") == 5  # 0+1+4+0

    def test_get_category_total_unknown(self, populated_heatmap: RiskHeatmap):
        """get_category_total returns 0 for unknown category."""
        assert populated_heatmap.get_category_total("nonexistent") == 0

    def test_get_severity_total(self, populated_heatmap: RiskHeatmap):
        """get_severity_total sums counts for a given severity."""
        assert populated_heatmap.get_severity_total("critical") == 3  # 3+0
        assert populated_heatmap.get_severity_total("high") == 3  # 2+1
        assert populated_heatmap.get_severity_total("medium") == 4  # 0+4
        assert populated_heatmap.get_severity_total("low") == 1  # 1+0

    def test_get_severity_total_unknown(self, populated_heatmap: RiskHeatmap):
        """get_severity_total returns 0 for unknown severity."""
        assert populated_heatmap.get_severity_total("info") == 0


# ===========================================================================
# RiskHeatmap._to_matrix
# ===========================================================================


class TestRiskHeatmapToMatrix:
    """Tests for RiskHeatmap._to_matrix."""

    def test_matrix_dimensions(self):
        """_to_matrix produces len(categories) x len(severities) 2D list."""
        cells = [
            HeatmapCell(category="a", severity="critical", count=1),
            HeatmapCell(category="a", severity="high", count=2),
            HeatmapCell(category="a", severity="medium", count=3),
            HeatmapCell(category="a", severity="low", count=4),
            HeatmapCell(category="b", severity="critical", count=5),
            HeatmapCell(category="b", severity="high", count=6),
            HeatmapCell(category="b", severity="medium", count=7),
            HeatmapCell(category="b", severity="low", count=8),
        ]
        hm = RiskHeatmap(cells=cells, categories=["a", "b"])
        matrix = hm._to_matrix()
        assert len(matrix) == 2
        assert all(len(row) == 4 for row in matrix)

    def test_matrix_values(self):
        """_to_matrix contains correct count values."""
        cells = [
            HeatmapCell(category="a", severity="critical", count=10),
            HeatmapCell(category="a", severity="high", count=20),
            HeatmapCell(category="a", severity="medium", count=30),
            HeatmapCell(category="a", severity="low", count=40),
            HeatmapCell(category="b", severity="critical", count=0),
            HeatmapCell(category="b", severity="high", count=5),
            HeatmapCell(category="b", severity="medium", count=0),
            HeatmapCell(category="b", severity="low", count=15),
        ]
        hm = RiskHeatmap(cells=cells, categories=["a", "b"])
        matrix = hm._to_matrix()
        assert matrix[0] == [10, 20, 30, 40]
        assert matrix[1] == [0, 5, 0, 15]

    def test_matrix_empty(self):
        """_to_matrix returns empty list for empty heatmap."""
        hm = RiskHeatmap()
        assert hm._to_matrix() == []

    def test_matrix_missing_cell_defaults_to_zero(self):
        """_to_matrix returns 0 for missing cells."""
        # Categories listed but no matching cells
        hm = RiskHeatmap(categories=["orphan"])
        matrix = hm._to_matrix()
        assert matrix == [[0, 0, 0, 0]]


# ===========================================================================
# RiskHeatmap.to_dict
# ===========================================================================


class TestRiskHeatmapToDict:
    """Tests for RiskHeatmap.to_dict."""

    def test_to_dict_keys(self):
        """to_dict includes all expected top-level keys."""
        hm = RiskHeatmap()
        d = hm.to_dict()
        expected_keys = {
            "cells",
            "categories",
            "severities",
            "total_findings",
            "highest_risk_category",
            "highest_risk_severity",
            "matrix",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_includes_matrix(self):
        """to_dict 'matrix' field matches _to_matrix output."""
        cells = [
            HeatmapCell(category="a", severity="critical", count=1),
            HeatmapCell(category="a", severity="high", count=0),
            HeatmapCell(category="a", severity="medium", count=2),
            HeatmapCell(category="a", severity="low", count=0),
        ]
        hm = RiskHeatmap(cells=cells, categories=["a"])
        d = hm.to_dict()
        assert d["matrix"] == [[1, 0, 2, 0]]

    def test_to_dict_cells_serialized(self):
        """to_dict serializes each cell via to_dict."""
        cell = HeatmapCell(category="x", severity="low", count=3, vulnerabilities=["v1"])
        hm = RiskHeatmap(cells=[cell], categories=["x"])
        d = hm.to_dict()
        assert len(d["cells"]) == 1
        assert d["cells"][0]["category"] == "x"
        assert d["cells"][0]["count"] == 3
        assert "intensity" in d["cells"][0]

    def test_to_dict_preserves_summary_fields(self):
        """to_dict preserves total_findings and highest_risk fields."""
        hm = RiskHeatmap(
            total_findings=7,
            highest_risk_category="auth",
            highest_risk_severity="critical",
        )
        d = hm.to_dict()
        assert d["total_findings"] == 7
        assert d["highest_risk_category"] == "auth"
        assert d["highest_risk_severity"] == "critical"


# ===========================================================================
# RiskHeatmap.to_json
# ===========================================================================


class TestRiskHeatmapToJson:
    """Tests for RiskHeatmap.to_json."""

    def test_to_json_valid(self):
        """to_json returns valid JSON string."""
        cell = HeatmapCell(category="a", severity="low", count=1)
        hm = RiskHeatmap(cells=[cell], categories=["a"], total_findings=1)
        raw = hm.to_json()
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)
        assert parsed["total_findings"] == 1

    def test_to_json_matches_to_dict(self):
        """to_json roundtrips to same structure as to_dict."""
        cell = HeatmapCell(category="b", severity="high", count=5, vulnerabilities=["x"])
        hm = RiskHeatmap(cells=[cell], categories=["b"], total_findings=5)
        parsed = json.loads(hm.to_json())
        expected = hm.to_dict()
        assert parsed == expected

    def test_to_json_empty_heatmap(self):
        """to_json works for empty heatmap."""
        hm = RiskHeatmap()
        parsed = json.loads(hm.to_json())
        assert parsed["cells"] == []
        assert parsed["matrix"] == []


# ===========================================================================
# RiskHeatmap.to_svg
# ===========================================================================


class TestRiskHeatmapToSvg:
    """Tests for RiskHeatmap.to_svg."""

    def test_to_svg_empty_categories(self):
        """to_svg returns 'No data' SVG when categories is empty."""
        hm = RiskHeatmap()
        svg = hm.to_svg()
        assert "<svg" in svg
        assert "No data" in svg

    def test_to_svg_with_data(self):
        """to_svg returns full SVG with rect elements for cells."""
        vulns = [
            _make_vulnerability("v1", SeverityLevel.HIGH, "auth"),
            _make_vulnerability("v2", SeverityLevel.MEDIUM, "auth"),
        ]
        result = _make_gauntlet_result(vulns, total=2)
        hm = RiskHeatmap.from_result(result)
        svg = hm.to_svg()

        assert "<svg" in svg
        assert "</svg>" in svg
        assert "<rect" in svg
        # Should contain severity headers
        assert "CRITICAL" in svg
        assert "HIGH" in svg
        assert "MEDIUM" in svg
        assert "LOW" in svg
        # Should contain category label
        assert "auth" in svg

    def test_to_svg_custom_dimensions(self):
        """to_svg respects width and height parameters."""
        vulns = [_make_vulnerability("v1", SeverityLevel.LOW, "a")]
        result = _make_gauntlet_result(vulns, total=1)
        hm = RiskHeatmap.from_result(result)
        svg = hm.to_svg(width=800, height=600)
        assert 'width="800"' in svg
        assert 'height="600"' in svg

    def test_to_svg_contains_style(self):
        """to_svg includes style block."""
        vulns = [_make_vulnerability("v1", SeverityLevel.LOW, "x")]
        result = _make_gauntlet_result(vulns, total=1)
        hm = RiskHeatmap.from_result(result)
        svg = hm.to_svg()
        assert "<style>" in svg

    def test_to_svg_shows_nonzero_counts(self):
        """to_svg includes text elements with count values for non-zero cells."""
        vulns = [
            _make_vulnerability("v1", SeverityLevel.HIGH, "net"),
            _make_vulnerability("v2", SeverityLevel.HIGH, "net"),
        ]
        result = _make_gauntlet_result(vulns, total=2)
        hm = RiskHeatmap.from_result(result)
        svg = hm.to_svg()
        # The count "2" should appear as text in the SVG
        assert ">2</text>" in svg


# ===========================================================================
# RiskHeatmap.to_ascii
# ===========================================================================


class TestRiskHeatmapToAscii:
    """Tests for RiskHeatmap.to_ascii."""

    def test_to_ascii_empty(self):
        """to_ascii returns 'No findings to display' for empty heatmap."""
        hm = RiskHeatmap()
        assert hm.to_ascii() == "No findings to display"

    def test_to_ascii_with_data(self):
        """to_ascii produces table with headers, rows, and totals."""
        cells = [
            HeatmapCell(category="injection", severity="critical", count=2),
            HeatmapCell(category="injection", severity="high", count=1),
            HeatmapCell(category="injection", severity="medium", count=0),
            HeatmapCell(category="injection", severity="low", count=3),
        ]
        hm = RiskHeatmap(cells=cells, categories=["injection"], total_findings=6)
        ascii_out = hm.to_ascii()

        # Header should contain severity abbreviations
        assert "CRIT" in ascii_out
        assert "HIGH" in ascii_out
        assert "MEDI" in ascii_out
        assert "LOW" in ascii_out

        # Category name
        assert "injection" in ascii_out

        # Totals row
        assert "TOTAL" in ascii_out

    def test_to_ascii_contains_separator_lines(self):
        """to_ascii includes separator lines made of dashes."""
        cells = [
            HeatmapCell(category="a", severity="critical", count=1),
            HeatmapCell(category="a", severity="high", count=0),
            HeatmapCell(category="a", severity="medium", count=0),
            HeatmapCell(category="a", severity="low", count=0),
        ]
        hm = RiskHeatmap(cells=cells, categories=["a"])
        ascii_out = hm.to_ascii()
        lines = ascii_out.split("\n")
        separator_lines = [line for line in lines if set(line.strip()) == {"-"}]
        assert len(separator_lines) >= 2  # before and after data rows

    def test_to_ascii_multiple_categories(self):
        """to_ascii includes all categories."""
        cells = [
            HeatmapCell(category="auth", severity="critical", count=1),
            HeatmapCell(category="auth", severity="high", count=0),
            HeatmapCell(category="auth", severity="medium", count=0),
            HeatmapCell(category="auth", severity="low", count=0),
            HeatmapCell(category="xss", severity="critical", count=0),
            HeatmapCell(category="xss", severity="high", count=2),
            HeatmapCell(category="xss", severity="medium", count=0),
            HeatmapCell(category="xss", severity="low", count=0),
        ]
        hm = RiskHeatmap(cells=cells, categories=["auth", "xss"], total_findings=3)
        ascii_out = hm.to_ascii()
        assert "auth" in ascii_out
        assert "xss" in ascii_out

    def test_to_ascii_truncates_long_category(self):
        """to_ascii truncates category names longer than 20 chars."""
        long_name = "a_very_long_category_name_that_exceeds"
        cells = [
            HeatmapCell(category=long_name, severity="critical", count=1),
            HeatmapCell(category=long_name, severity="high", count=0),
            HeatmapCell(category=long_name, severity="medium", count=0),
            HeatmapCell(category=long_name, severity="low", count=0),
        ]
        hm = RiskHeatmap(cells=cells, categories=[long_name])
        ascii_out = hm.to_ascii()
        # Category should be truncated to 20 chars
        assert long_name[:20] in ascii_out
        # Full name should NOT appear as a continuous string in the row
        data_lines = [
            line
            for line in ascii_out.split("\n")
            if "|" in line and "Category" not in line and "TOTAL" not in line
        ]
        for line in data_lines:
            category_part = line.split("|")[0]
            assert len(category_part.strip()) <= 20


# ===========================================================================
# Integration: from_result round-trip through serialization
# ===========================================================================


class TestRiskHeatmapIntegration:
    """End-to-end tests combining from_result with serialization."""

    def test_from_result_to_json_roundtrip(self):
        """Build heatmap from result and verify JSON roundtrip."""
        vulns = [
            _make_vulnerability("v1", SeverityLevel.CRITICAL, "injection"),
            _make_vulnerability("v2", SeverityLevel.CRITICAL, "injection"),
            _make_vulnerability("v3", SeverityLevel.HIGH, "xss"),
            _make_vulnerability("v4", SeverityLevel.LOW, "config"),
        ]
        result = _make_gauntlet_result(vulns, total=4)
        hm = RiskHeatmap.from_result(result)

        parsed = json.loads(hm.to_json())

        assert parsed["total_findings"] == 4
        assert sorted(parsed["categories"]) == ["config", "injection", "xss"]
        # Matrix: 3 categories x 4 severities
        assert len(parsed["matrix"]) == 3
        assert all(len(row) == 4 for row in parsed["matrix"])

    def test_from_result_svg_and_ascii_not_empty(self):
        """Both SVG and ASCII outputs are non-trivial for populated heatmap."""
        vulns = [
            _make_vulnerability("v1", SeverityLevel.MEDIUM, "data"),
        ]
        result = _make_gauntlet_result(vulns, total=1)
        hm = RiskHeatmap.from_result(result)

        svg = hm.to_svg()
        assert len(svg) > 100
        assert "<rect" in svg

        ascii_out = hm.to_ascii()
        assert "data" in ascii_out
        assert "TOTAL" in ascii_out

    def test_single_vuln_per_category_severity_pair(self):
        """One vulnerability per unique (category, severity) pair produces count=1 cells."""
        vulns = [
            _make_vulnerability("v1", SeverityLevel.CRITICAL, "a"),
            _make_vulnerability("v2", SeverityLevel.HIGH, "b"),
            _make_vulnerability("v3", SeverityLevel.MEDIUM, "c"),
            _make_vulnerability("v4", SeverityLevel.LOW, "d"),
        ]
        result = _make_gauntlet_result(vulns, total=4)
        hm = RiskHeatmap.from_result(result)

        for vuln in vulns:
            cell = hm.get_cell(vuln.category, vuln.severity.value)
            assert cell is not None
            assert cell.count == 1
            assert vuln.id in cell.vulnerabilities
