"""Tests for StrategicScanner — deep codebase assessment for self-improvement."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.nomic.strategic_scanner import (
    CATEGORY_COMPLEX,
    CATEGORY_INTEGRATION_GAP,
    CATEGORY_STALE,
    CATEGORY_UNTESTED,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    StrategicAssessment,
    StrategicFinding,
    StrategicScanner,
    _track_for_path,
)


# ---------------------------------------------------------------------------
# StrategicFinding dataclass
# ---------------------------------------------------------------------------


class TestStrategicFinding:
    """Test StrategicFinding dataclass."""

    def test_fields(self):
        f = StrategicFinding(
            category="untested",
            severity="high",
            file_path="aragora/debate/consensus.py",
            description="No test file",
            evidence="No tests/*/test_consensus.py found",
            suggested_action="Create tests",
            track="core",
        )
        assert f.category == "untested"
        assert f.severity == "high"
        assert f.file_path == "aragora/debate/consensus.py"
        assert f.track == "core"

    def test_all_categories(self):
        for cat in ("untested", "complex", "stale", "integration_gap", "dead_code"):
            f = StrategicFinding(
                category=cat,
                severity="low",
                file_path="a.py",
                description="d",
                evidence="e",
                suggested_action="s",
                track="qa",
            )
            assert f.category == cat


# ---------------------------------------------------------------------------
# StrategicAssessment dataclass
# ---------------------------------------------------------------------------


class TestStrategicAssessment:
    """Test StrategicAssessment dataclass."""

    def test_defaults(self):
        a = StrategicAssessment()
        assert a.findings == []
        assert a.metrics == {}
        assert a.focus_areas == []
        assert a.objective == ""
        assert a.timestamp == 0.0

    def test_with_values(self):
        finding = StrategicFinding(
            category="untested",
            severity="high",
            file_path="a.py",
            description="d",
            evidence="e",
            suggested_action="s",
            track="core",
        )
        a = StrategicAssessment(
            findings=[finding],
            metrics={"total_modules": 10},
            focus_areas=["[core] untested: 1 findings (score 2)"],
            objective="test coverage",
            timestamp=1000.0,
        )
        assert len(a.findings) == 1
        assert a.metrics["total_modules"] == 10
        assert a.objective == "test coverage"


# ---------------------------------------------------------------------------
# _track_for_path helper
# ---------------------------------------------------------------------------


class TestTrackForPath:
    """Test the path-to-track mapping function."""

    def test_debate_maps_to_core(self):
        assert _track_for_path("aragora/debate/consensus.py") == "core"

    def test_server_maps_to_developer(self):
        assert _track_for_path("aragora/server/handlers/foo.py") == "developer"

    def test_auth_maps_to_security(self):
        assert _track_for_path("aragora/auth/oidc.py") == "security"

    def test_billing_maps_to_sme(self):
        assert _track_for_path("aragora/billing/cost_tracker.py") == "sme"

    def test_ops_maps_to_self_hosted(self):
        assert _track_for_path("aragora/ops/deploy.py") == "self_hosted"

    def test_unknown_defaults_to_developer(self):
        assert _track_for_path("aragora/unknown_pkg/foo.py") == "developer"

    def test_single_segment_defaults_to_core(self):
        assert _track_for_path("setup.py") == "core"

    def test_tests_maps_to_qa(self):
        assert _track_for_path("tests/test_foo.py") == "qa"


# ---------------------------------------------------------------------------
# StrategicScanner._find_untested_modules
# ---------------------------------------------------------------------------


class TestFindUntestedModules:
    """Test untested module detection."""

    def test_finds_module_without_test(self, tmp_path):
        src = tmp_path / "aragora" / "billing"
        src.mkdir(parents=True)
        (src / "cost_tracker.py").write_text("x = 1\n" * 50)
        tests = tmp_path / "tests"
        tests.mkdir()
        # No test_cost_tracker.py exists

        scanner = StrategicScanner(repo_path=tmp_path)
        findings = scanner._find_untested_modules()

        assert len(findings) == 1
        assert findings[0].category == CATEGORY_UNTESTED
        assert "cost_tracker" in findings[0].description

    def test_skips_tested_module(self, tmp_path):
        src = tmp_path / "aragora" / "billing"
        src.mkdir(parents=True)
        (src / "cost_tracker.py").write_text("x = 1\n")
        tests = tmp_path / "tests" / "billing"
        tests.mkdir(parents=True)
        (tests / "test_cost_tracker.py").write_text("def test_a(): pass\n")

        scanner = StrategicScanner(repo_path=tmp_path)
        findings = scanner._find_untested_modules()
        assert len(findings) == 0

    def test_skips_dunder_files(self, tmp_path):
        src = tmp_path / "aragora"
        src.mkdir()
        (src / "__init__.py").write_text("")
        (src / "__main__.py").write_text("")
        (tmp_path / "tests").mkdir()

        scanner = StrategicScanner(repo_path=tmp_path)
        findings = scanner._find_untested_modules()
        assert len(findings) == 0

    def test_severity_based_on_loc(self, tmp_path):
        src = tmp_path / "aragora"
        src.mkdir()
        # Small module -> medium severity
        (src / "small.py").write_text("x = 1\n" * 10)
        # Large module -> high severity
        (src / "large.py").write_text("x = 1\n" * 250)
        (tmp_path / "tests").mkdir()

        scanner = StrategicScanner(repo_path=tmp_path)
        findings = scanner._find_untested_modules()
        severities = {f.file_path.split("/")[-1]: f.severity for f in findings}
        assert severities["small.py"] == SEVERITY_MEDIUM
        assert severities["large.py"] == SEVERITY_HIGH

    def test_empty_src_root(self, tmp_path):
        scanner = StrategicScanner(repo_path=tmp_path)
        assert scanner._find_untested_modules() == []


# ---------------------------------------------------------------------------
# StrategicScanner._find_complexity_hotspots
# ---------------------------------------------------------------------------


class TestFindComplexityHotspots:
    """Test complexity hotspot detection."""

    def test_detects_high_loc(self, tmp_path):
        src = tmp_path / "aragora"
        src.mkdir()
        (src / "big.py").write_text("x = 1\n" * 600)

        scanner = StrategicScanner(repo_path=tmp_path)
        findings = scanner._find_complexity_hotspots()
        assert len(findings) == 1
        assert findings[0].category == CATEGORY_COMPLEX
        assert "LOC" in findings[0].evidence

    def test_detects_deep_nesting(self, tmp_path):
        src = tmp_path / "aragora"
        src.mkdir()
        # 7 levels deep (> 6 threshold)
        code = "def f():\n" + "    " * 7 + "x = 1\n"
        (src / "nested.py").write_text(code)

        scanner = StrategicScanner(repo_path=tmp_path)
        findings = scanner._find_complexity_hotspots()
        assert len(findings) == 1
        assert "indent" in findings[0].description

    def test_detects_many_functions(self, tmp_path):
        src = tmp_path / "aragora"
        src.mkdir()
        funcs = "\n".join(f"def func_{i}(): pass\n" for i in range(25))
        (src / "funcs.py").write_text(funcs)

        scanner = StrategicScanner(repo_path=tmp_path)
        findings = scanner._find_complexity_hotspots()
        assert len(findings) == 1
        assert "functions" in findings[0].description

    def test_critical_for_very_large_file(self, tmp_path):
        src = tmp_path / "aragora"
        src.mkdir()
        (src / "huge.py").write_text("x = 1\n" * 1100)

        scanner = StrategicScanner(repo_path=tmp_path)
        findings = scanner._find_complexity_hotspots()
        assert findings[0].severity == SEVERITY_CRITICAL

    def test_no_findings_for_simple_file(self, tmp_path):
        src = tmp_path / "aragora"
        src.mkdir()
        (src / "simple.py").write_text("x = 1\ny = 2\n")

        scanner = StrategicScanner(repo_path=tmp_path)
        assert scanner._find_complexity_hotspots() == []

    def test_skips_dunder(self, tmp_path):
        src = tmp_path / "aragora"
        src.mkdir()
        (src / "__init__.py").write_text("x = 1\n" * 600)

        scanner = StrategicScanner(repo_path=tmp_path)
        assert scanner._find_complexity_hotspots() == []


# ---------------------------------------------------------------------------
# StrategicScanner._find_stale_todos
# ---------------------------------------------------------------------------


class TestFindStaleTodos:
    """Test stale TODO/FIXME/HACK detection."""

    def test_finds_stale_todo(self, tmp_path):
        src = tmp_path / "aragora"
        src.mkdir()
        (src / "old.py").write_text("# TODO fix this thing\nx = 1\n")

        scanner = StrategicScanner(repo_path=tmp_path)
        # Mock git returning old timestamps
        old_ts = time.time() - (90 * 86400)  # 90 days ago
        with patch.object(scanner, "_git_file_mod_times", return_value={"aragora/old.py": old_ts}):
            findings = scanner._find_stale_todos()

        assert len(findings) == 1
        assert findings[0].category == CATEGORY_STALE
        assert "TODO" in findings[0].description

    def test_skips_recently_modified(self, tmp_path):
        src = tmp_path / "aragora"
        src.mkdir()
        (src / "fresh.py").write_text("# TODO do something\nx = 1\n")

        scanner = StrategicScanner(repo_path=tmp_path)
        recent_ts = time.time() - (5 * 86400)  # 5 days ago
        with patch.object(
            scanner, "_git_file_mod_times", return_value={"aragora/fresh.py": recent_ts}
        ):
            findings = scanner._find_stale_todos()

        assert len(findings) == 0

    def test_fixme_is_high_severity(self, tmp_path):
        src = tmp_path / "aragora"
        src.mkdir()
        (src / "fix.py").write_text("# FIXME critical bug\n")

        scanner = StrategicScanner(repo_path=tmp_path)
        old_ts = time.time() - (90 * 86400)
        with patch.object(scanner, "_git_file_mod_times", return_value={"aragora/fix.py": old_ts}):
            findings = scanner._find_stale_todos()

        assert findings[0].severity == SEVERITY_HIGH

    def test_hack_is_medium_severity(self, tmp_path):
        src = tmp_path / "aragora"
        src.mkdir()
        (src / "hack.py").write_text("# HACK workaround\n")

        scanner = StrategicScanner(repo_path=tmp_path)
        old_ts = time.time() - (90 * 86400)
        with patch.object(scanner, "_git_file_mod_times", return_value={"aragora/hack.py": old_ts}):
            findings = scanner._find_stale_todos()

        assert findings[0].severity == SEVERITY_MEDIUM

    def test_no_git_data_still_finds_stale(self, tmp_path):
        """If git returns no data, files with TODOs are assumed stale."""
        src = tmp_path / "aragora"
        src.mkdir()
        (src / "unknown.py").write_text("# TODO clean up\n")

        scanner = StrategicScanner(repo_path=tmp_path)
        with patch.object(scanner, "_git_file_mod_times", return_value={}):
            findings = scanner._find_stale_todos()

        assert len(findings) == 1


# ---------------------------------------------------------------------------
# StrategicScanner._find_integration_gaps
# ---------------------------------------------------------------------------


class TestFindIntegrationGaps:
    """Test integration gap detection."""

    def test_finds_unused_package(self, tmp_path):
        pkg = tmp_path / "aragora" / "orphan"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text('__all__ = ["Orphan"]\n')
        (pkg / "core.py").write_text("class Orphan: pass\n")

        # Another module that does NOT import orphan
        other = tmp_path / "aragora" / "other"
        other.mkdir(parents=True)
        (other / "__init__.py").write_text("")
        (other / "main.py").write_text("import os\n")

        scanner = StrategicScanner(repo_path=tmp_path)
        findings = scanner._find_integration_gaps()

        orphan_findings = [f for f in findings if "orphan" in f.file_path]
        assert len(orphan_findings) == 1
        assert orphan_findings[0].category == CATEGORY_INTEGRATION_GAP

    def test_no_gap_when_imported(self, tmp_path):
        pkg = tmp_path / "aragora" / "used"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text('__all__ = ["UsedClass"]\n')
        (pkg / "core.py").write_text("class UsedClass: pass\n")

        consumer = tmp_path / "aragora" / "consumer"
        consumer.mkdir(parents=True)
        (consumer / "__init__.py").write_text("")
        (consumer / "main.py").write_text("from aragora.used import UsedClass\n")

        scanner = StrategicScanner(repo_path=tmp_path)
        findings = scanner._find_integration_gaps()

        used_findings = [f for f in findings if "used" in f.file_path]
        assert len(used_findings) == 0

    def test_empty_exports_no_finding(self, tmp_path):
        pkg = tmp_path / "aragora" / "empty"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("# empty\n")

        scanner = StrategicScanner(repo_path=tmp_path)
        findings = scanner._find_integration_gaps()
        empty_findings = [f for f in findings if "empty" in f.file_path]
        assert len(empty_findings) == 0


# ---------------------------------------------------------------------------
# StrategicScanner._rank_findings
# ---------------------------------------------------------------------------


class TestRankFindings:
    """Test finding ranking by severity and objective."""

    def _make_finding(self, severity="medium", category="untested", path="a.py", track="core"):
        return StrategicFinding(
            category=category,
            severity=severity,
            file_path=path,
            description=f"{category} in {path}",
            evidence="e",
            suggested_action="s",
            track=track,
        )

    def test_severity_ordering(self):
        low = self._make_finding(severity="low")
        high = self._make_finding(severity="high")
        critical = self._make_finding(severity="critical")

        scanner = StrategicScanner()
        ranked = scanner._rank_findings([low, high, critical], "")

        assert ranked[0].severity == "critical"
        assert ranked[1].severity == "high"
        assert ranked[2].severity == "low"

    def test_objective_keyword_boost(self):
        test_finding = self._make_finding(
            severity="low", category="untested", path="aragora/debate/test_thing.py"
        )
        other_finding = self._make_finding(
            severity="low", category="complex", path="aragora/billing/cost.py"
        )

        scanner = StrategicScanner()
        ranked = scanner._rank_findings([other_finding, test_finding], "untested debate")

        # test_finding mentions "untested" and "debate" -> should rank higher
        assert ranked[0].file_path == test_finding.file_path

    def test_empty_objective_uses_severity_only(self):
        high = self._make_finding(severity="high")
        low = self._make_finding(severity="low")

        scanner = StrategicScanner()
        ranked = scanner._rank_findings([low, high], "")
        assert ranked[0].severity == "high"

    def test_empty_findings(self):
        scanner = StrategicScanner()
        assert scanner._rank_findings([], "anything") == []


# ---------------------------------------------------------------------------
# StrategicScanner.scan (full integration)
# ---------------------------------------------------------------------------


class TestScanIntegration:
    """Test full scan integration."""

    def test_scan_returns_assessment(self, tmp_path):
        src = tmp_path / "aragora"
        src.mkdir()
        (src / "module_a.py").write_text("x = 1\n" * 10)
        (tmp_path / "tests").mkdir()

        scanner = StrategicScanner(repo_path=tmp_path)
        with patch.object(scanner, "_git_file_mod_times", return_value={}):
            assessment = scanner.scan("improve coverage")

        assert isinstance(assessment, StrategicAssessment)
        assert assessment.objective == "improve coverage"
        assert assessment.timestamp > 0
        assert "total_modules" in assessment.metrics
        assert assessment.metrics["total_modules"] >= 1

    def test_scan_with_no_source(self, tmp_path):
        scanner = StrategicScanner(repo_path=tmp_path)
        with patch.object(scanner, "_git_file_mod_times", return_value={}):
            assessment = scanner.scan()

        assert assessment.findings == []
        assert assessment.metrics["total_modules"] == 0

    def test_scan_focus_areas_limited_to_5(self, tmp_path):
        src = tmp_path / "aragora"
        src.mkdir()
        # Create many untested modules in different packages
        for pkg in ("debate", "billing", "auth", "server", "ops", "cli", "mcp"):
            d = src / pkg
            d.mkdir(parents=True)
            (d / "mod.py").write_text("x = 1\n" * 300)
        (tmp_path / "tests").mkdir()

        scanner = StrategicScanner(repo_path=tmp_path)
        with patch.object(scanner, "_git_file_mod_times", return_value={}):
            assessment = scanner.scan()

        assert len(assessment.focus_areas) <= 5

    def test_scan_metrics_findings_by_category(self, tmp_path):
        src = tmp_path / "aragora"
        src.mkdir()
        (src / "untest.py").write_text("x = 1\n")
        (src / "big.py").write_text("x = 1\n" * 600)
        (tmp_path / "tests").mkdir()

        scanner = StrategicScanner(repo_path=tmp_path)
        with patch.object(scanner, "_git_file_mod_times", return_value={}):
            assessment = scanner.scan()

        cats = assessment.metrics.get("findings_by_category", {})
        assert CATEGORY_UNTESTED in cats or CATEGORY_COMPLEX in cats


# ---------------------------------------------------------------------------
# Helper methods
# ---------------------------------------------------------------------------


class TestHelperMethods:
    """Test static helper methods."""

    def test_count_lines(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("a\n\nb\nc\n")
        assert StrategicScanner._count_lines(f) == 3  # blank line excluded

    def test_count_lines_missing_file(self, tmp_path):
        assert StrategicScanner._count_lines(tmp_path / "nonexistent.py") == 0

    def test_max_indent_depth(self):
        content = "def f():\n    if True:\n        x = 1\n"
        assert StrategicScanner._max_indent_depth(content) == 2

    def test_max_indent_depth_ignores_comments(self):
        content = "        # deep comment\ndef f():\n    pass\n"
        assert StrategicScanner._max_indent_depth(content) == 1

    def test_count_functions_ast(self):
        content = "def a(): pass\ndef b(): pass\nasync def c(): pass\n"
        assert StrategicScanner._count_functions(content) == 3

    def test_count_functions_syntax_error_fallback(self):
        content = "def a(): pass\ndef b():\n  invalid syntax {{{\n"
        # Falls back to counting 'def ' lines
        assert StrategicScanner._count_functions(content) >= 2
