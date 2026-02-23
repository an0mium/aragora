"""Tests for the `aragora explain <debate_id>` CLI command.

Tests:
1. Parser registration (flag is recognized, --verbose, --format, --api-url, --api-key)
2. API-first path returns explanation
3. Local fallback when API is unavailable
4. Error when both API and local fail
5. JSON output format
6. Text output format (standard and verbose)
7. Verbose mode (belief changes, summary metrics, untruncated content)
8. Error handling (debate not found, etc.)
"""

from __future__ import annotations

import argparse
import json
from unittest.mock import MagicMock, patch

import pytest

from aragora.cli.commands.explain import (
    add_explain_parser,
    cmd_explain,
    _try_api_explanation,
    _try_local_explanation,
    _print_text_explanation,
)


def _make_args(**overrides) -> argparse.Namespace:
    """Create a Namespace with explain defaults plus overrides."""
    defaults = {
        "debate_id": "test-debate-123",
        "output_format": "text",
        "verbose": False,
        "api_url": None,
        "api_key": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _mock_explanation() -> dict:
    """Create a mock explanation dict."""
    return {
        "debate_id": "test-debate-123",
        "decision": "Use microservices architecture",
        "confidence": 0.85,
        "summary": "After 3 rounds of debate, agents reached consensus on microservices.",
        "factors": [
            {
                "name": "Scalability",
                "description": "Independent scaling of services",
                "weight": 0.6,
                "evidence": ["Service isolation allows targeted scaling"],
                "source_agents": ["anthropic-api", "openai-api"],
            },
        ],
        "evidence_chain": [
            {
                "content": "Microservices enable independent deployment",
                "source": "proposal",
                "confidence": 0.9,
                "round_number": 1,
                "agent_id": "anthropic-api",
            },
        ],
        "vote_pivots": [
            {
                "agent_id": "openai-api",
                "vote_value": "approve",
                "confidence": 0.88,
                "influence_score": 0.75,
                "reasoning": "Strong scalability argument",
                "changed_outcome": True,
            },
        ],
        "counterfactuals": [
            {
                "scenario": "Remove openai-api vote",
                "description": "Without the OpenAI vote the result would differ",
                "alternative_outcome": "No consensus reached",
                "probability": 0.6,
                "key_differences": ["Missing critical scaling perspective"],
            },
        ],
    }


def _mock_verbose_explanation() -> dict:
    """Create a mock explanation dict with all verbose-mode fields populated."""
    base = _mock_explanation()
    base.update(
        {
            "consensus_reached": True,
            "consensus_type": "majority",
            "rounds_used": 3,
            "agents_participated": ["anthropic-api", "openai-api", "gemini-api"],
            "task": "Design a rate limiter",
            "domain": "engineering",
            "evidence_quality_score": 0.78,
            "agent_agreement_score": 0.90,
            "belief_stability_score": 0.65,
            "belief_changes": [
                {
                    "agent": "openai-api",
                    "round": 2,
                    "topic": "Architecture choice",
                    "prior_belief": "Monolith preferred",
                    "posterior_belief": "Microservices preferred",
                    "prior_confidence": 0.6,
                    "posterior_confidence": 0.85,
                    "confidence_delta": 0.25,
                    "trigger": "critique",
                    "trigger_source": "anthropic-api",
                },
            ],
        }
    )
    # Add verbose-only fields to existing sections
    base["factors"][0]["raw_value"] = 0.82
    base["evidence_chain"][0]["grounding_type"] = "argument"
    base["evidence_chain"][0]["quality_scores"] = {"relevance": 0.9, "authority": 0.7}
    base["evidence_chain"][0]["cited_by"] = ["openai-api"]
    base["vote_pivots"][0]["weight"] = 1.5
    base["vote_pivots"][0]["elo_rating"] = 1250.0
    base["vote_pivots"][0]["calibration_adjustment"] = 0.032
    base["counterfactuals"][0]["sensitivity"] = 0.75
    return base


# ===========================================================================
# Parser registration
# ===========================================================================


class TestExplainParser:
    """Tests that the explain command is properly registered."""

    def test_parser_registered(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_explain_parser(subparsers)

        args = parser.parse_args(["explain", "debate-abc"])
        assert args.debate_id == "debate-abc"

    def test_parser_format_json(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_explain_parser(subparsers)

        args = parser.parse_args(["explain", "debate-abc", "--format", "json"])
        assert args.output_format == "json"

    def test_parser_format_text_default(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_explain_parser(subparsers)

        args = parser.parse_args(["explain", "debate-abc"])
        assert args.output_format == "text"

    def test_parser_api_url(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_explain_parser(subparsers)

        args = parser.parse_args(
            ["explain", "debate-abc", "--api-url", "http://example.com:9090"]
        )
        assert args.api_url == "http://example.com:9090"

    def test_parser_verbose_flag(self):
        """--verbose flag is recognized and defaults to False."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_explain_parser(subparsers)

        args_default = parser.parse_args(["explain", "debate-abc"])
        assert args_default.verbose is False

        args_verbose = parser.parse_args(["explain", "debate-abc", "--verbose"])
        assert args_verbose.verbose is True

    def test_parser_api_key(self):
        """--api-key is accepted."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_explain_parser(subparsers)

        args = parser.parse_args(
            ["explain", "debate-abc", "--api-key", "sk-test-123"]
        )
        assert args.api_key == "sk-test-123"


# ===========================================================================
# API path
# ===========================================================================


class TestExplainAPIPath:
    """Tests for the API-first explanation path."""

    def test_api_success_returns_dict(self):
        """When API call succeeds, returns explanation dict."""
        mock_explanation = MagicMock()
        mock_explanation.debate_id = "d1"
        mock_explanation.decision = "Use microservices"
        mock_explanation.confidence = 0.85
        mock_explanation.summary = "Consensus reached"
        mock_explanation.factors = []
        mock_explanation.evidence_chain = []
        mock_explanation.vote_pivots = []
        mock_explanation.counterfactuals = []

        mock_client = MagicMock()
        mock_client.explainability.get_explanation.return_value = mock_explanation

        with patch(
            "aragora.cli.commands.explain.AragoraClient", create=True
        ) as mock_cls:
            # Patch at the module level where it's imported
            pass

        with patch("aragora.client.client.AragoraClient") as mock_cls:
            mock_cls.return_value = mock_client
            args = _make_args()
            result = _try_api_explanation("d1", args)

        assert result is not None
        assert result["debate_id"] == "d1"
        assert result["decision"] == "Use microservices"

    def test_api_import_error_returns_none(self):
        """When client not available, returns None."""
        original_import = __import__

        def block_client(name, *args, **kwargs):
            if "aragora.client.client" in name:
                raise ImportError("client not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=block_client):
            result = _try_api_explanation("d1", _make_args())

        assert result is None

    def test_api_connection_error_returns_none(self):
        """When API unreachable, returns None."""
        mock_client = MagicMock()
        mock_client.explainability.get_explanation.side_effect = ConnectionError(
            "refused"
        )

        with patch("aragora.client.client.AragoraClient") as mock_cls:
            mock_cls.return_value = mock_client
            result = _try_api_explanation("d1", _make_args())

        assert result is None

    def test_api_runtime_error_returns_none(self):
        """When API returns a runtime error, returns None."""
        mock_client = MagicMock()
        mock_client.explainability.get_explanation.side_effect = RuntimeError(
            "server error"
        )

        with patch("aragora.client.client.AragoraClient") as mock_cls:
            mock_cls.return_value = mock_client
            result = _try_api_explanation("d1", _make_args())

        assert result is None

    def test_api_uses_custom_url_and_key(self):
        """API path uses --api-url and --api-key from args."""
        mock_explanation = MagicMock()
        mock_explanation.debate_id = "d1"
        mock_explanation.decision = "ok"
        mock_explanation.confidence = 0.5
        mock_explanation.summary = ""
        mock_explanation.factors = []
        mock_explanation.evidence_chain = []
        mock_explanation.vote_pivots = []
        mock_explanation.counterfactuals = []

        mock_client = MagicMock()
        mock_client.explainability.get_explanation.return_value = mock_explanation

        with patch("aragora.client.client.AragoraClient") as mock_cls:
            mock_cls.return_value = mock_client
            args = _make_args(api_url="http://custom:9090", api_key="sk-custom")
            _try_api_explanation("d1", args)

        mock_cls.assert_called_once_with(
            base_url="http://custom:9090", api_key="sk-custom"
        )


# ===========================================================================
# Local fallback
# ===========================================================================


class TestExplainLocalFallback:
    """Tests for the local ExplanationBuilder fallback."""

    def test_local_import_error_returns_none(self):
        """When ExplanationBuilder not available, returns None."""
        original_import = __import__

        def block_builder(name, *args, **kwargs):
            if "explainability.builder" in name:
                raise ImportError("builder not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=block_builder):
            result = _try_local_explanation("d1")

        assert result is None

    def test_local_no_debate_data_returns_none(self):
        """When no local debate data exists, returns None."""
        with patch(
            "aragora.cli.commands.explain._load_local_debate", return_value=None
        ):
            result = _try_local_explanation("nonexistent-debate")

        assert result is None

    def test_local_builder_exception_returns_none(self):
        """When ExplanationBuilder.build() raises, returns None gracefully."""
        mock_debate = MagicMock()

        with patch(
            "aragora.cli.commands.explain._load_local_debate",
            return_value=mock_debate,
        ), patch(
            "aragora.explainability.builder.ExplanationBuilder.build",
            side_effect=ValueError("invalid debate structure"),
        ):
            result = _try_local_explanation("bad-debate")

        assert result is None


# ===========================================================================
# cmd_explain integration
# ===========================================================================


class TestCmdExplain:
    """Integration tests for cmd_explain."""

    def test_success_with_api(self, capsys):
        """cmd_explain returns 0 and prints explanation when API succeeds."""
        explanation = _mock_explanation()

        with patch(
            "aragora.cli.commands.explain._try_api_explanation",
            return_value=explanation,
        ):
            args = _make_args()
            result = cmd_explain(args)

        assert result == 0
        output = capsys.readouterr().out
        assert "Decision Explanation" in output
        assert "Use microservices" in output

    def test_success_with_local_fallback(self, capsys):
        """cmd_explain falls back to local when API fails."""
        explanation = _mock_explanation()

        with patch(
            "aragora.cli.commands.explain._try_api_explanation", return_value=None
        ), patch(
            "aragora.cli.commands.explain._try_local_explanation",
            return_value=explanation,
        ):
            args = _make_args()
            result = cmd_explain(args)

        assert result == 0
        output = capsys.readouterr().out
        assert "Decision Explanation" in output

    def test_failure_when_both_fail(self, capsys):
        """cmd_explain returns 1 when both API and local fail."""
        with patch(
            "aragora.cli.commands.explain._try_api_explanation", return_value=None
        ), patch(
            "aragora.cli.commands.explain._try_local_explanation", return_value=None
        ):
            args = _make_args()
            result = cmd_explain(args)

        assert result == 1
        err = capsys.readouterr().err
        assert "Could not retrieve explanation" in err

    def test_json_output_format(self, capsys):
        """cmd_explain outputs valid JSON when --format json."""
        explanation = _mock_explanation()

        with patch(
            "aragora.cli.commands.explain._try_api_explanation",
            return_value=explanation,
        ):
            args = _make_args(output_format="json")
            result = cmd_explain(args)

        assert result == 0
        output = capsys.readouterr().out
        parsed = json.loads(output)
        assert parsed["debate_id"] == "test-debate-123"
        assert parsed["confidence"] == 0.85
        assert len(parsed["factors"]) == 1
        assert len(parsed["counterfactuals"]) == 1

    def test_verbose_text_output(self, capsys):
        """cmd_explain with --verbose passes verbose to text formatter."""
        explanation = _mock_verbose_explanation()

        with patch(
            "aragora.cli.commands.explain._try_api_explanation",
            return_value=explanation,
        ):
            args = _make_args(verbose=True)
            result = cmd_explain(args)

        assert result == 0
        output = capsys.readouterr().out
        # Verbose sections
        assert "Belief Changes" in output
        assert "Summary Metrics" in output
        assert "Consensus:" in output or "Reached" in output

    def test_error_message_includes_debate_id(self, capsys):
        """Error message mentions the specific debate ID."""
        with patch(
            "aragora.cli.commands.explain._try_api_explanation", return_value=None
        ), patch(
            "aragora.cli.commands.explain._try_local_explanation", return_value=None
        ):
            args = _make_args(debate_id="missing-debate-xyz")
            cmd_explain(args)

        err = capsys.readouterr().err
        assert "missing-debate-xyz" in err


# ===========================================================================
# Text formatting
# ===========================================================================


class TestTextFormatting:
    """Tests for the text output formatting."""

    def test_text_includes_decision(self, capsys):
        explanation = _mock_explanation()
        _print_text_explanation("d1", explanation)
        output = capsys.readouterr().out
        assert "Use microservices architecture" in output

    def test_text_includes_confidence(self, capsys):
        explanation = _mock_explanation()
        _print_text_explanation("d1", explanation)
        output = capsys.readouterr().out
        assert "85" in output  # 85% confidence

    def test_text_includes_factors(self, capsys):
        explanation = _mock_explanation()
        _print_text_explanation("d1", explanation)
        output = capsys.readouterr().out
        assert "Scalability" in output
        assert "Contributing Factors" in output

    def test_text_includes_evidence(self, capsys):
        explanation = _mock_explanation()
        _print_text_explanation("d1", explanation)
        output = capsys.readouterr().out
        assert "Evidence Chain" in output
        assert "anthropic-api" in output

    def test_text_includes_vote_pivots(self, capsys):
        explanation = _mock_explanation()
        _print_text_explanation("d1", explanation)
        output = capsys.readouterr().out
        assert "Vote Pivots" in output
        assert "CHANGED OUTCOME" in output

    def test_text_includes_counterfactuals(self, capsys):
        explanation = _mock_explanation()
        _print_text_explanation("d1", explanation)
        output = capsys.readouterr().out
        assert "Counterfactuals" in output
        assert "No consensus reached" in output

    def test_text_empty_sections_omitted(self, capsys):
        explanation = {
            "debate_id": "d1",
            "decision": "Result",
            "confidence": 0.5,
            "summary": "",
            "factors": [],
            "evidence_chain": [],
            "vote_pivots": [],
            "counterfactuals": [],
        }
        _print_text_explanation("d1", explanation)
        output = capsys.readouterr().out
        assert "Contributing Factors" not in output
        assert "Evidence Chain" not in output
        assert "Vote Pivots" not in output
        assert "Counterfactuals" not in output

    def test_text_truncates_long_evidence_in_standard_mode(self, capsys):
        """In non-verbose mode, evidence content > 120 chars is truncated."""
        explanation = {
            "debate_id": "d1",
            "decision": "Result",
            "confidence": 0.5,
            "summary": "",
            "factors": [],
            "evidence_chain": [
                {
                    "content": "A" * 200,
                    "source": "proposal",
                    "confidence": 0.8,
                    "round_number": 1,
                    "agent_id": "agent1",
                },
            ],
            "vote_pivots": [],
            "counterfactuals": [],
        }
        _print_text_explanation("d1", explanation, verbose=False)
        output = capsys.readouterr().out
        assert "..." in output
        # The full 200 A's should not appear
        assert "A" * 200 not in output


# ===========================================================================
# Verbose mode
# ===========================================================================


class TestVerboseMode:
    """Tests specific to --verbose output."""

    def test_verbose_shows_consensus_info(self, capsys):
        explanation = _mock_verbose_explanation()
        _print_text_explanation("d1", explanation, verbose=True)
        output = capsys.readouterr().out
        assert "Consensus:" in output
        assert "Reached" in output
        assert "majority" in output

    def test_verbose_shows_rounds_and_agents(self, capsys):
        explanation = _mock_verbose_explanation()
        _print_text_explanation("d1", explanation, verbose=True)
        output = capsys.readouterr().out
        assert "Rounds:" in output
        assert "3" in output
        assert "Agents:" in output
        assert "anthropic-api" in output
        assert "gemini-api" in output

    def test_verbose_shows_task_and_domain(self, capsys):
        explanation = _mock_verbose_explanation()
        _print_text_explanation("d1", explanation, verbose=True)
        output = capsys.readouterr().out
        assert "Task:" in output
        assert "Design a rate limiter" in output
        assert "Domain:" in output
        assert "engineering" in output

    def test_verbose_shows_belief_changes(self, capsys):
        explanation = _mock_verbose_explanation()
        _print_text_explanation("d1", explanation, verbose=True)
        output = capsys.readouterr().out
        assert "Belief Changes" in output
        assert "openai-api" in output
        assert "Monolith preferred" in output
        assert "Microservices preferred" in output
        assert "Trigger: critique" in output

    def test_verbose_shows_summary_metrics(self, capsys):
        explanation = _mock_verbose_explanation()
        _print_text_explanation("d1", explanation, verbose=True)
        output = capsys.readouterr().out
        assert "Summary Metrics" in output
        assert "Evidence quality:" in output
        assert "0.78" in output
        assert "Agent agreement:" in output
        assert "0.90" in output
        assert "Belief stability:" in output
        assert "0.65" in output

    def test_verbose_shows_untruncated_evidence(self, capsys):
        """Verbose mode shows full evidence content without truncation."""
        long_content = "B" * 200
        explanation = {
            "debate_id": "d1",
            "decision": "Result",
            "confidence": 0.5,
            "summary": "",
            "factors": [],
            "evidence_chain": [
                {
                    "content": long_content,
                    "source": "proposal",
                    "confidence": 0.8,
                    "round_number": 1,
                    "agent_id": "agent1",
                    "grounding_type": "argument",
                    "quality_scores": {"relevance": 0.95, "authority": 0.80},
                    "cited_by": ["agent2", "agent3"],
                },
            ],
            "vote_pivots": [],
            "counterfactuals": [],
        }
        _print_text_explanation("d1", explanation, verbose=True)
        output = capsys.readouterr().out
        # Full content preserved
        assert long_content in output
        # Quality scores shown
        assert "Quality:" in output
        assert "relevance: 0.95" in output
        # Grounding type shown
        assert "Type: argument" in output
        # Cited by shown
        assert "Cited by: agent2, agent3" in output

    def test_verbose_shows_vote_weight_and_elo(self, capsys):
        """Verbose mode shows ELO, weight, and calibration for vote pivots."""
        explanation = _mock_verbose_explanation()
        _print_text_explanation("d1", explanation, verbose=True)
        output = capsys.readouterr().out
        assert "Weight:" in output
        assert "1.50" in output
        assert "ELO:" in output
        assert "1250" in output
        assert "Calibration adj:" in output
        assert "0.032" in output

    def test_verbose_shows_counterfactual_sensitivity(self, capsys):
        """Verbose mode shows sensitivity for counterfactuals."""
        explanation = _mock_verbose_explanation()
        _print_text_explanation("d1", explanation, verbose=True)
        output = capsys.readouterr().out
        assert "Sensitivity:" in output
        assert "0.75" in output

    def test_verbose_shows_factor_raw_value(self, capsys):
        """Verbose mode shows raw_value for contributing factors."""
        explanation = _mock_verbose_explanation()
        _print_text_explanation("d1", explanation, verbose=True)
        output = capsys.readouterr().out
        assert "Raw value:" in output
        assert "0.82" in output

    def test_non_verbose_hides_belief_changes(self, capsys):
        """Non-verbose mode omits belief changes section."""
        explanation = _mock_verbose_explanation()
        _print_text_explanation("d1", explanation, verbose=False)
        output = capsys.readouterr().out
        assert "Belief Changes" not in output
        assert "Summary Metrics" not in output

    def test_verbose_omits_domain_general(self, capsys):
        """Verbose mode skips 'general' domain since it's not informative."""
        explanation = _mock_verbose_explanation()
        explanation["domain"] = "general"
        _print_text_explanation("d1", explanation, verbose=True)
        output = capsys.readouterr().out
        assert "Domain:" not in output
