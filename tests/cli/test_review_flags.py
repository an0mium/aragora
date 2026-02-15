"""
Tests for --sarif and --gauntlet flags on the review CLI command.

Tests:
- --sarif flag produces valid SARIF 2.1.0 output
- --gauntlet flag runs adversarial stress-testing
- Both flags work together
- Flags work with mock review results (no real API calls)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.cli.review import (
    cmd_review,
    create_review_parser,
    findings_to_sarif,
    get_demo_findings,
    run_gauntlet_on_diff,
)


# ===========================================================================
# Test Fixtures and Mock Classes
# ===========================================================================


@dataclass
class MockCritique:
    """Mock critique for testing."""

    agent: str = "anthropic-api"
    target_agent: str = "openai-api"
    issues: list = field(default_factory=lambda: ["SQL injection risk"])
    suggestions: list = field(default_factory=lambda: ["Use parameterized queries"])
    severity: float = 0.8


@dataclass
class MockMessage:
    """Mock message for testing."""

    agent: str = "anthropic-api"
    content: str = "Found a security issue"
    role: str = "reviewer"


@dataclass
class MockVote:
    """Mock vote for testing."""

    voter: str = "anthropic-api"
    choice: str = "reject"
    confidence: float = 0.9


@dataclass
class MockDebateResult:
    """Mock debate result for testing."""

    final_answer: str = "Review complete: 2 issues found"
    confidence: float = 0.85
    consensus_reached: bool = True
    winner: str = "anthropic-api"
    rounds_used: int = 2
    duration_seconds: float = 10.0
    messages: list = field(default_factory=lambda: [MockMessage()])
    votes: list = field(default_factory=lambda: [MockVote()])
    critiques: list = field(
        default_factory=lambda: [
            MockCritique(severity=0.95),
            MockCritique(agent="openai-api", severity=0.5),
        ]
    )
    participants: list = field(default_factory=lambda: ["anthropic-api", "openai-api"])


def make_args(**overrides) -> argparse.Namespace:
    """Create a Namespace with all review defaults plus overrides."""
    defaults = {
        "pr_url": None,
        "diff_file": None,
        "agents": "anthropic-api,openai-api",
        "rounds": 2,
        "focus": "security,performance,quality",
        "output_format": "github",
        "output_dir": None,
        "demo": False,
        "share": False,
        "sarif": None,
        "gauntlet": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def mock_findings() -> dict:
    """Create mock review findings for testing."""
    return {
        "unanimous_critiques": [
            "SQL injection vulnerability in user search",
        ],
        "split_opinions": [],
        "risk_areas": ["Error handling in payment flow"],
        "agreement_score": 0.8,
        "agent_alignment": {},
        "critical_issues": [
            {
                "agent": "anthropic-api",
                "issue": "SQL injection in search_users()",
                "target": "api/users.py:45",
                "suggestions": ["Use parameterized queries"],
            },
        ],
        "high_issues": [
            {
                "agent": "openai-api",
                "issue": "Missing CSRF protection",
                "target": "api/routes.py",
                "suggestions": [],
            },
        ],
        "medium_issues": [
            {
                "agent": "anthropic-api",
                "issue": "Unbounded query results",
                "target": "api/products.py:102",
                "suggestions": ["Add pagination"],
            },
        ],
        "low_issues": [],
        "all_critiques": [],
        "final_summary": "Review found 2 critical issues.",
        "agents_used": ["anthropic-api", "openai-api"],
    }


# ===========================================================================
# Tests for findings_to_sarif
# ===========================================================================


class TestFindingsToSarif:
    """Tests for the SARIF conversion function."""

    def test_sarif_schema_version(self):
        """SARIF output has correct schema and version."""
        findings = mock_findings()
        sarif = findings_to_sarif(findings)

        assert sarif["version"] == "2.1.0"
        assert "$schema" in sarif
        assert "sarif-schema-2.1.0" in sarif["$schema"]

    def test_sarif_has_runs(self):
        """SARIF output has a runs array with one entry."""
        findings = mock_findings()
        sarif = findings_to_sarif(findings)

        assert "runs" in sarif
        assert len(sarif["runs"]) == 1

    def test_sarif_tool_info(self):
        """SARIF run contains tool driver info."""
        findings = mock_findings()
        sarif = findings_to_sarif(findings)

        driver = sarif["runs"][0]["tool"]["driver"]
        assert driver["name"] == "Aragora Review"
        assert driver["version"] == "1.0.0"
        assert "informationUri" in driver

    def test_sarif_custom_tool_name(self):
        """SARIF accepts a custom tool name."""
        findings = mock_findings()
        sarif = findings_to_sarif(findings, tool_name="Custom Tool")

        driver = sarif["runs"][0]["tool"]["driver"]
        assert driver["name"] == "Custom Tool"

    def test_sarif_results_from_critical_issues(self):
        """Critical issues appear as error-level results."""
        findings = mock_findings()
        sarif = findings_to_sarif(findings)

        results = sarif["runs"][0]["results"]
        critical_results = [r for r in results if r["properties"].get("severity") == "CRITICAL"]
        assert len(critical_results) == 1
        assert critical_results[0]["level"] == "error"
        assert "SQL injection" in critical_results[0]["message"]["text"]

    def test_sarif_results_from_high_issues(self):
        """High issues appear as error-level results."""
        findings = mock_findings()
        sarif = findings_to_sarif(findings)

        results = sarif["runs"][0]["results"]
        high_results = [r for r in results if r["properties"].get("severity") == "HIGH"]
        assert len(high_results) == 1
        assert high_results[0]["level"] == "error"

    def test_sarif_results_from_medium_issues(self):
        """Medium issues appear as warning-level results."""
        findings = mock_findings()
        sarif = findings_to_sarif(findings)

        results = sarif["runs"][0]["results"]
        medium_results = [r for r in results if r["properties"].get("severity") == "MEDIUM"]
        assert len(medium_results) == 1
        assert medium_results[0]["level"] == "warning"

    def test_sarif_unanimous_critiques_included(self):
        """Unanimous critiques are included as error-level results."""
        findings = mock_findings()
        sarif = findings_to_sarif(findings)

        results = sarif["runs"][0]["results"]
        unanimous_results = [r for r in results if r["properties"].get("unanimous") is True]
        assert len(unanimous_results) == 1
        assert "SQL injection" in unanimous_results[0]["message"]["text"]

    def test_sarif_has_rules(self):
        """SARIF output includes rule definitions."""
        findings = mock_findings()
        sarif = findings_to_sarif(findings)

        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        assert len(rules) > 0
        # Rules should have IDs starting with ARAGORA-REVIEW-
        for rule in rules:
            assert rule["id"].startswith("ARAGORA-REVIEW-")

    def test_sarif_results_have_fingerprints(self):
        """Each result has a unique fingerprint for deduplication."""
        findings = mock_findings()
        sarif = findings_to_sarif(findings)

        results = sarif["runs"][0]["results"]
        for result in results:
            assert "fingerprints" in result
            assert "aragora/v1" in result["fingerprints"]

    def test_sarif_results_have_locations(self):
        """Each result has a location."""
        findings = mock_findings()
        sarif = findings_to_sarif(findings)

        results = sarif["runs"][0]["results"]
        for result in results:
            assert "locations" in result
            assert len(result["locations"]) > 0

    def test_sarif_target_in_location(self):
        """Issue target is used as artifact URI."""
        findings = mock_findings()
        sarif = findings_to_sarif(findings)

        results = sarif["runs"][0]["results"]
        critical_results = [r for r in results if r["properties"].get("severity") == "CRITICAL"]
        assert len(critical_results) == 1
        uri = critical_results[0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        assert uri == "api/users.py:45"

    def test_sarif_suggestions_as_fixes(self):
        """Issues with suggestions get fix entries."""
        findings = mock_findings()
        sarif = findings_to_sarif(findings)

        results = sarif["runs"][0]["results"]
        critical_results = [r for r in results if r["properties"].get("severity") == "CRITICAL"]
        assert len(critical_results) == 1
        assert "fixes" in critical_results[0]
        assert "parameterized" in critical_results[0]["fixes"][0]["description"]["text"]

    def test_sarif_empty_findings(self):
        """SARIF output is valid even with empty findings."""
        findings = {
            "unanimous_critiques": [],
            "critical_issues": [],
            "high_issues": [],
            "medium_issues": [],
            "low_issues": [],
        }
        sarif = findings_to_sarif(findings)

        assert sarif["version"] == "2.1.0"
        assert len(sarif["runs"]) == 1
        assert sarif["runs"][0]["results"] == []

    def test_sarif_json_serializable(self):
        """SARIF output can be serialized to JSON without errors."""
        findings = mock_findings()
        sarif = findings_to_sarif(findings)

        # Should not raise
        json_str = json.dumps(sarif, indent=2)
        assert len(json_str) > 0

        # Should round-trip cleanly
        parsed = json.loads(json_str)
        assert parsed["version"] == "2.1.0"

    def test_sarif_with_demo_findings(self):
        """SARIF works with the demo findings fixture."""
        findings = get_demo_findings()
        sarif = findings_to_sarif(findings)

        assert sarif["version"] == "2.1.0"
        results = sarif["runs"][0]["results"]
        # Demo has 1 critical + 1 high + 1 medium + 2 unanimous = 5 results
        assert len(results) >= 4


# ===========================================================================
# Tests for --sarif flag in cmd_review
# ===========================================================================


class TestSarifFlag:
    """Tests for the --sarif CLI flag."""

    def test_sarif_flag_demo_mode(self, tmp_path):
        """--sarif flag works in demo mode."""
        sarif_path = tmp_path / "output.sarif"
        args = make_args(demo=True, sarif=str(sarif_path))

        result = cmd_review(args)
        assert result == 0

        # Verify SARIF file was created
        assert sarif_path.exists()
        sarif_data = json.loads(sarif_path.read_text())
        assert sarif_data["version"] == "2.1.0"
        assert len(sarif_data["runs"]) == 1

    def test_sarif_flag_default_path_demo(self, tmp_path, monkeypatch):
        """--sarif with no path uses default filename."""
        monkeypatch.chdir(tmp_path)
        args = make_args(demo=True, sarif="review-results.sarif")

        result = cmd_review(args)
        assert result == 0

        sarif_path = tmp_path / "review-results.sarif"
        assert sarif_path.exists()

    def test_sarif_flag_creates_parent_dirs(self, tmp_path):
        """--sarif creates parent directories if needed."""
        sarif_path = tmp_path / "nested" / "dir" / "output.sarif"
        args = make_args(demo=True, sarif=str(sarif_path))

        result = cmd_review(args)
        assert result == 0
        assert sarif_path.exists()

    def test_sarif_contents_match_findings(self, tmp_path):
        """SARIF output contains the expected findings from demo mode."""
        sarif_path = tmp_path / "output.sarif"
        args = make_args(demo=True, sarif=str(sarif_path))

        cmd_review(args)

        sarif_data = json.loads(sarif_path.read_text())
        results = sarif_data["runs"][0]["results"]

        # Demo findings should produce results
        assert len(results) > 0

        # Check that at least one critical issue is present
        error_results = [r for r in results if r["level"] == "error"]
        assert len(error_results) > 0

    @patch("aragora.cli.review.run_review_debate")
    @patch("aragora.cli.review.extract_review_findings")
    @patch("aragora.cli.review.get_available_agents")
    def test_sarif_flag_with_diff_file(self, mock_agents, mock_extract, mock_debate, tmp_path):
        """--sarif works with real review flow (mocked)."""
        # Setup mocks
        mock_agents.return_value = "anthropic-api,openai-api"
        mock_debate.return_value = MockDebateResult()
        mock_extract.return_value = mock_findings()

        # Create diff file
        diff_file = tmp_path / "test.diff"
        diff_file.write_text("--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n")

        sarif_path = tmp_path / "output.sarif"
        args = make_args(
            diff_file=str(diff_file),
            sarif=str(sarif_path),
        )

        result = cmd_review(args)
        assert result == 0
        assert sarif_path.exists()

        sarif_data = json.loads(sarif_path.read_text())
        assert sarif_data["version"] == "2.1.0"


# ===========================================================================
# Tests for --gauntlet flag
# ===========================================================================


class TestGauntletFlag:
    """Tests for the --gauntlet CLI flag."""

    @patch("aragora.cli.review.run_review_debate")
    @patch("aragora.cli.review.extract_review_findings")
    @patch("aragora.cli.review.get_available_agents")
    @patch("aragora.cli.review.run_gauntlet_on_diff")
    def test_gauntlet_flag_invokes_runner(
        self, mock_gauntlet, mock_agents, mock_extract, mock_debate, tmp_path
    ):
        """--gauntlet flag triggers the gauntlet runner."""
        # Setup mocks
        mock_agents.return_value = "anthropic-api,openai-api"
        mock_debate.return_value = MockDebateResult()
        base_findings = mock_findings()
        mock_extract.return_value = base_findings

        # Gauntlet returns findings with gauntlet section added
        gauntlet_findings = {**base_findings}
        gauntlet_findings["gauntlet"] = {
            "gauntlet_id": "gauntlet-abc123",
            "gauntlet_verdict": "conditional",
            "gauntlet_robustness": 0.75,
            "gauntlet_vulnerabilities": [],
        }
        mock_gauntlet.return_value = gauntlet_findings

        diff_file = tmp_path / "test.diff"
        diff_file.write_text("--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n")

        args = make_args(diff_file=str(diff_file), gauntlet=True)

        result = cmd_review(args)
        assert result == 0

        # Verify gauntlet was called
        mock_gauntlet.assert_called_once()

    @patch("aragora.cli.review.run_review_debate")
    @patch("aragora.cli.review.extract_review_findings")
    @patch("aragora.cli.review.get_available_agents")
    @patch("aragora.cli.review.run_gauntlet_on_diff")
    def test_gauntlet_failure_does_not_fail_review(
        self, mock_gauntlet, mock_agents, mock_extract, mock_debate, tmp_path
    ):
        """Gauntlet failure is a warning, not a hard error."""
        mock_agents.return_value = "anthropic-api,openai-api"
        mock_debate.return_value = MockDebateResult()
        mock_extract.return_value = mock_findings()
        mock_gauntlet.side_effect = RuntimeError("Gauntlet exploded")

        diff_file = tmp_path / "test.diff"
        diff_file.write_text("--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n")

        args = make_args(diff_file=str(diff_file), gauntlet=True)

        # Should still return 0 despite gauntlet failure
        result = cmd_review(args)
        assert result == 0

    @patch("aragora.cli.review.run_review_debate")
    @patch("aragora.cli.review.extract_review_findings")
    @patch("aragora.cli.review.get_available_agents")
    def test_no_gauntlet_when_flag_not_set(self, mock_agents, mock_extract, mock_debate, tmp_path):
        """Gauntlet does not run when --gauntlet is not set."""
        mock_agents.return_value = "anthropic-api,openai-api"
        mock_debate.return_value = MockDebateResult()
        mock_extract.return_value = mock_findings()

        diff_file = tmp_path / "test.diff"
        diff_file.write_text("--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n")

        args = make_args(diff_file=str(diff_file), gauntlet=False)

        with patch("aragora.cli.review.run_gauntlet_on_diff") as mock_gauntlet:
            result = cmd_review(args)
            assert result == 0
            mock_gauntlet.assert_not_called()


# ===========================================================================
# Tests for --sarif and --gauntlet together
# ===========================================================================


class TestSarifAndGauntletCombined:
    """Tests for using both flags together."""

    @patch("aragora.cli.review.run_review_debate")
    @patch("aragora.cli.review.extract_review_findings")
    @patch("aragora.cli.review.get_available_agents")
    @patch("aragora.cli.review.run_gauntlet_on_diff")
    def test_both_flags_produce_combined_sarif(
        self, mock_gauntlet, mock_agents, mock_extract, mock_debate, tmp_path
    ):
        """Using --sarif and --gauntlet together includes gauntlet findings in SARIF."""
        mock_agents.return_value = "anthropic-api,openai-api"
        mock_debate.return_value = MockDebateResult()
        base_findings = mock_findings()
        mock_extract.return_value = base_findings

        # Gauntlet adds extra critical issue
        gauntlet_findings = {**base_findings}
        gauntlet_findings["critical_issues"] = base_findings["critical_issues"] + [
            {
                "agent": "gauntlet-red-team",
                "issue": "Race condition in concurrent writes",
                "target": "api/database.py:88",
                "suggestions": ["Add locking"],
            },
        ]
        gauntlet_findings["gauntlet"] = {
            "gauntlet_id": "gauntlet-test123",
            "gauntlet_verdict": "fail",
            "gauntlet_robustness": 0.4,
            "gauntlet_vulnerabilities": [
                {"id": "vuln-0001", "title": "Race condition", "severity": "CRITICAL"},
            ],
        }
        mock_gauntlet.return_value = gauntlet_findings

        diff_file = tmp_path / "test.diff"
        diff_file.write_text("--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n")

        sarif_path = tmp_path / "combined.sarif"
        args = make_args(
            diff_file=str(diff_file),
            gauntlet=True,
            sarif=str(sarif_path),
        )

        result = cmd_review(args)
        assert result == 0

        # Verify SARIF includes gauntlet findings
        assert sarif_path.exists()
        sarif_data = json.loads(sarif_path.read_text())
        results = sarif_data["runs"][0]["results"]

        # Should have original + gauntlet critical issues
        critical_results = [
            r for r in results if r.get("properties", {}).get("severity") == "CRITICAL"
        ]
        assert len(critical_results) == 2  # Original + gauntlet

    @patch("aragora.cli.review.run_review_debate")
    @patch("aragora.cli.review.extract_review_findings")
    @patch("aragora.cli.review.get_available_agents")
    @patch("aragora.cli.review.run_gauntlet_on_diff")
    def test_gauntlet_failure_still_produces_sarif(
        self, mock_gauntlet, mock_agents, mock_extract, mock_debate, tmp_path
    ):
        """If gauntlet fails, SARIF still gets written with base findings."""
        mock_agents.return_value = "anthropic-api,openai-api"
        mock_debate.return_value = MockDebateResult()
        mock_extract.return_value = mock_findings()
        mock_gauntlet.side_effect = RuntimeError("Gauntlet unavailable")

        diff_file = tmp_path / "test.diff"
        diff_file.write_text("--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n")

        sarif_path = tmp_path / "output.sarif"
        args = make_args(
            diff_file=str(diff_file),
            gauntlet=True,
            sarif=str(sarif_path),
        )

        result = cmd_review(args)
        assert result == 0

        # SARIF should still exist with base findings
        assert sarif_path.exists()
        sarif_data = json.loads(sarif_path.read_text())
        assert sarif_data["version"] == "2.1.0"
        assert len(sarif_data["runs"][0]["results"]) > 0


# ===========================================================================
# Tests for argparse integration
# ===========================================================================


class TestArgparse:
    """Tests that the flags are properly registered in argparse."""

    def test_sarif_flag_registered(self):
        """The --sarif flag is recognized by the parser."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_review_parser(subparsers)

        args = parser.parse_args(["review", "--demo", "--sarif"])
        assert args.sarif == "review-results.sarif"

    def test_sarif_flag_with_path(self):
        """The --sarif flag accepts a custom path."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_review_parser(subparsers)

        args = parser.parse_args(["review", "--demo", "--sarif", "/tmp/custom.sarif"])
        assert args.sarif == "/tmp/custom.sarif"

    def test_sarif_flag_not_set(self):
        """When --sarif is not provided, it is None."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_review_parser(subparsers)

        args = parser.parse_args(["review", "--demo"])
        assert args.sarif is None

    def test_gauntlet_flag_registered(self):
        """The --gauntlet flag is recognized by the parser."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_review_parser(subparsers)

        args = parser.parse_args(["review", "--demo", "--gauntlet"])
        assert args.gauntlet is True

    def test_gauntlet_flag_not_set(self):
        """When --gauntlet is not provided, it is False."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_review_parser(subparsers)

        args = parser.parse_args(["review", "--demo"])
        assert args.gauntlet is False

    def test_both_flags_together(self):
        """Both --sarif and --gauntlet can be used together."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_review_parser(subparsers)

        args = parser.parse_args(["review", "--demo", "--sarif", "--gauntlet"])
        assert args.sarif == "review-results.sarif"
        assert args.gauntlet is True


# ===========================================================================
# Tests for run_gauntlet_on_diff
# ===========================================================================


class TestRunGauntletOnDiff:
    """Tests for the gauntlet runner helper."""

    @pytest.mark.asyncio
    @patch("aragora.gauntlet.runner.GauntletRunner")
    @patch("aragora.gauntlet.templates.get_template")
    async def test_gauntlet_uses_code_review_template(self, mock_get_template, mock_runner_cls):
        """Gauntlet helper uses the CODE_REVIEW template."""
        mock_config = MagicMock()
        mock_get_template.return_value = mock_config

        mock_runner = MagicMock()
        mock_result = MagicMock()
        mock_result.gauntlet_id = "gauntlet-test"
        mock_result.verdict.value = "pass"
        mock_result.attack_summary.robustness_score = 0.9
        mock_result.vulnerabilities = []
        mock_runner.run = AsyncMock(return_value=mock_result)
        mock_runner_cls.return_value = mock_runner

        findings = mock_findings()
        result = await run_gauntlet_on_diff("some diff", findings, "anthropic-api")

        # Template was fetched
        mock_get_template.assert_called_once()
        # Runner was created with the config
        mock_runner_cls.assert_called_once_with(config=mock_config)
        # Run was called with the diff
        mock_runner.run.assert_called_once()

        # Gauntlet section was added to findings
        assert "gauntlet" in result
        assert result["gauntlet"]["gauntlet_id"] == "gauntlet-test"

    @pytest.mark.asyncio
    @patch("aragora.gauntlet.runner.GauntletRunner")
    @patch("aragora.gauntlet.templates.get_template")
    async def test_gauntlet_merges_vulnerabilities(self, mock_get_template, mock_runner_cls):
        """Gauntlet vulnerabilities get merged into findings severity buckets."""
        mock_config = MagicMock()
        mock_get_template.return_value = mock_config

        # Create a mock vulnerability
        mock_vuln = MagicMock()
        mock_vuln.severity.value = "HIGH"
        mock_vuln.agent_name = "red-team"
        mock_vuln.source = "red_team"
        mock_vuln.description = "Buffer overflow in parser"
        mock_vuln.category = "security"
        mock_vuln.mitigation = "Bounds checking"
        mock_vuln.to_dict.return_value = {"id": "vuln-001", "severity": "HIGH"}

        mock_runner = MagicMock()
        mock_result = MagicMock()
        mock_result.gauntlet_id = "gauntlet-merge-test"
        mock_result.verdict.value = "conditional"
        mock_result.attack_summary.robustness_score = 0.6
        mock_result.vulnerabilities = [mock_vuln]
        mock_runner.run = AsyncMock(return_value=mock_result)
        mock_runner_cls.return_value = mock_runner

        findings = mock_findings()
        original_high_count = len(findings.get("high_issues", []))

        result = await run_gauntlet_on_diff("diff content", findings, "anthropic-api")

        # High issues should have grown
        assert len(result["high_issues"]) == original_high_count + 1
        # The new issue should have the gauntlet vulnerability data
        new_issue = result["high_issues"][-1]
        assert new_issue["issue"] == "Buffer overflow in parser"
        assert new_issue["suggestions"] == ["Bounds checking"]
