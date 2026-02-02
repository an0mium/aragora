"""Tests for Explainability namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestExplainability:
    """Tests for decision explainability operations."""

    def test_explain_decision_default(self) -> None:
        """Get explanation with default parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "summary": "The decision was approved based on...",
                "factors": [{"name": "cost", "weight": 0.4}],
                "evidence": [{"source": "data-analysis", "confidence": 0.9}],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.explainability.explain("decision_123")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/explainability/decisions/decision_123",
                params={
                    "audience": "technical",
                    "include_factors": True,
                    "include_evidence": True,
                },
            )
            assert "summary" in result
            client.close()

    def test_explain_decision_executive_audience(self) -> None:
        """Get explanation for executive audience."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"summary": "Executive summary"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.explainability.explain(
                "decision_123",
                audience="executive",
                include_factors=False,
                include_evidence=False,
            )

            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["audience"] == "executive"
            assert params["include_factors"] is False
            assert params["include_evidence"] is False
            client.close()

    def test_explain_decision_compliance_audience(self) -> None:
        """Get explanation for compliance audience."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"summary": "Compliance summary"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.explainability.explain("decision_123", audience="compliance")

            call_args = mock_request.call_args
            assert call_args[1]["params"]["audience"] == "compliance"
            client.close()


class TestExplainabilityFactors:
    """Tests for factor decomposition operations."""

    def test_get_factors_default(self) -> None:
        """Get all factors for a decision."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "factors": [
                    {"name": "cost", "weight": 0.4, "description": "Cost analysis"},
                    {"name": "risk", "weight": 0.3, "description": "Risk assessment"},
                ]
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.explainability.get_factors("decision_123")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/explainability/decisions/decision_123/factors",
                params={},
            )
            assert len(result["factors"]) == 2
            client.close()

    def test_get_factors_with_min_weight(self) -> None:
        """Get factors above minimum weight."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"factors": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.explainability.get_factors("decision_123", min_weight=0.25)

            call_args = mock_request.call_args
            assert call_args[1]["params"]["min_weight"] == 0.25
            client.close()


class TestExplainabilityCounterfactuals:
    """Tests for counterfactual analysis operations."""

    def test_get_counterfactuals_default(self) -> None:
        """Get counterfactuals with default parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "counterfactuals": [
                    {"change": "Reduce cost by 20%", "outcome": "REJECTED"},
                ]
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.explainability.get_counterfactuals("decision_123")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/explainability/decisions/decision_123/counterfactuals",
                params={"limit": 5},
            )
            assert "counterfactuals" in result
            client.close()

    def test_get_counterfactuals_with_target(self) -> None:
        """Get counterfactuals for a specific outcome."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"counterfactuals": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.explainability.get_counterfactuals(
                "decision_123",
                target_outcome="APPROVED",
                limit=10,
            )

            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["target_outcome"] == "APPROVED"
            assert params["limit"] == 10
            client.close()


class TestExplainabilitySummary:
    """Tests for summary generation operations."""

    def test_generate_summary_default(self) -> None:
        """Generate summary with default parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "summary": "The decision was based on multiple factors...",
                "format": "paragraph",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.explainability.generate_summary("decision_123")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/explainability/decisions/decision_123/summary",
                params={"format": "paragraph"},
            )
            assert result["format"] == "paragraph"
            client.close()

    def test_generate_summary_bullets(self) -> None:
        """Generate bullet point summary."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"summary": "- Point 1\n- Point 2"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.explainability.generate_summary("decision_123", format="bullets")

            call_args = mock_request.call_args
            assert call_args[1]["params"]["format"] == "bullets"
            client.close()

    def test_generate_summary_with_max_length(self) -> None:
        """Generate summary with length constraint."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"summary": "Short summary"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.explainability.generate_summary(
                "decision_123",
                format="executive",
                max_length=100,
            )

            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["format"] == "executive"
            assert params["max_length"] == 100
            client.close()


class TestExplainabilityEvidence:
    """Tests for evidence chain operations."""

    def test_get_evidence_chain(self) -> None:
        """Get evidence chain for a decision."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "evidence": [
                    {"source": "document_1", "citation": "Page 5"},
                    {"source": "debate_123", "citation": "Round 2"},
                ],
                "chain_length": 2,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.explainability.get_evidence_chain("decision_123")

            mock_request.assert_called_once_with(
                "GET", "/api/v1/explainability/decisions/decision_123/evidence"
            )
            assert result["chain_length"] == 2
            client.close()


class TestExplainabilityContributions:
    """Tests for agent contribution operations."""

    def test_get_agent_contributions(self) -> None:
        """Get agent contributions for a decision."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "contributions": [
                    {"agent": "claude", "impact_score": 0.45},
                    {"agent": "gpt-4", "impact_score": 0.35},
                ]
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.explainability.get_agent_contributions("decision_123")

            mock_request.assert_called_once_with(
                "GET", "/api/v1/explainability/decisions/decision_123/contributions"
            )
            assert len(result["contributions"]) == 2
            client.close()


class TestExplainabilityConfidence:
    """Tests for confidence breakdown operations."""

    def test_get_confidence_breakdown(self) -> None:
        """Get confidence score breakdown."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "overall_confidence": 0.87,
                "by_source": {"document": 0.9, "debate": 0.85},
                "by_factor": {"cost": 0.92, "risk": 0.80},
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.explainability.get_confidence_breakdown("decision_123")

            mock_request.assert_called_once_with(
                "GET", "/api/v1/explainability/decisions/decision_123/confidence"
            )
            assert result["overall_confidence"] == 0.87
            client.close()


class TestExplainabilityComparison:
    """Tests for decision comparison operations."""

    def test_compare_decisions(self) -> None:
        """Compare two decisions."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "decision_1": "decision_123",
                "decision_2": "decision_456",
                "differences": [
                    {"factor": "cost", "delta": 0.15},
                ],
                "similarity_score": 0.72,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.explainability.compare_decisions("decision_123", "decision_456")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/explainability/compare",
                params={"decision_1": "decision_123", "decision_2": "decision_456"},
            )
            assert result["similarity_score"] == 0.72
            client.close()


class TestAsyncExplainability:
    """Tests for async explainability API."""

    @pytest.mark.asyncio
    async def test_async_explain(self) -> None:
        """Get explanation asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"summary": "Async explanation"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.explainability.explain("decision_123")

                mock_request.assert_called_once_with(
                    "GET",
                    "/api/v1/explainability/decisions/decision_123",
                    params={
                        "audience": "technical",
                        "include_factors": True,
                        "include_evidence": True,
                    },
                )
                assert result["summary"] == "Async explanation"

    @pytest.mark.asyncio
    async def test_async_get_factors(self) -> None:
        """Get factors asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"factors": []}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.explainability.get_factors("decision_123")

                assert "factors" in result

    @pytest.mark.asyncio
    async def test_async_get_counterfactuals(self) -> None:
        """Get counterfactuals asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"counterfactuals": []}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.explainability.get_counterfactuals(
                    "decision_123", target_outcome="REJECTED"
                )

                call_args = mock_request.call_args
                assert call_args[1]["params"]["target_outcome"] == "REJECTED"
                assert "counterfactuals" in result

    @pytest.mark.asyncio
    async def test_async_generate_summary(self) -> None:
        """Generate summary asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"summary": "Async summary"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.explainability.generate_summary(
                    "decision_123", format="bullets", max_length=50
                )

                assert "summary" in result

    @pytest.mark.asyncio
    async def test_async_compare_decisions(self) -> None:
        """Compare decisions asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"similarity_score": 0.85}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.explainability.compare_decisions(
                    "decision_123", "decision_456"
                )

                mock_request.assert_called_once_with(
                    "GET",
                    "/api/v1/explainability/compare",
                    params={"decision_1": "decision_123", "decision_2": "decision_456"},
                )
                assert result["similarity_score"] == 0.85
