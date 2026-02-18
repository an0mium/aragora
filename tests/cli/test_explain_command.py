"""Tests for the `aragora explain <debate_id>` CLI command.

Tests:
1. Parser registration (flag is recognized)
2. API-first path returns explanation
3. Local fallback when API is unavailable
4. Error when both API and local fail
5. JSON output format
6. Text output format
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
