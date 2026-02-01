"""Tests for Uncertainty namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient

# =========================================================================
# Estimate Uncertainty Operations
# =========================================================================


class TestUncertaintyEstimate:
    """Tests for uncertainty estimation operations."""

    def test_estimate_basic(self) -> None:
        """Estimate uncertainty with basic parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "overall_uncertainty": 0.35,
                "confidence_interval": [0.25, 0.45],
                "cruxes": [
                    {
                        "description": "Market conditions are volatile",
                        "importance": 0.8,
                        "uncertainty": 0.6,
                    }
                ],
                "methodology": "ensemble_calibrated",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.uncertainty.estimate(content="The market will grow by 15% next quarter")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/uncertainty/estimate",
                json={"content": "The market will grow by 15% next quarter"},
            )
            assert result["overall_uncertainty"] == 0.35
            assert len(result["cruxes"]) == 1
            client.close()

    def test_estimate_with_context(self) -> None:
        """Estimate uncertainty with context."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"overall_uncertainty": 0.2}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.uncertainty.estimate(
                content="Revenue will double", context="Based on historical trends"
            )

            call_args = mock_request.call_args
            json_data = call_args[1]["json"]
            assert json_data["content"] == "Revenue will double"
            assert json_data["context"] == "Based on historical trends"
            client.close()

    def test_estimate_with_all_options(self) -> None:
        """Estimate uncertainty with all options."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"overall_uncertainty": 0.5}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.uncertainty.estimate(
                content="Prediction X",
                context="Context Y",
                debate_id="deb_123",
                config={"method": "bayesian"},
            )

            call_args = mock_request.call_args
            json_data = call_args[1]["json"]
            assert json_data["content"] == "Prediction X"
            assert json_data["context"] == "Context Y"
            assert json_data["debate_id"] == "deb_123"
            assert json_data["config"] == {"method": "bayesian"}
            client.close()


# =========================================================================
# Debate Metrics Operations
# =========================================================================


class TestUncertaintyDebateMetrics:
    """Tests for debate uncertainty metrics operations."""

    def test_get_debate_metrics(self) -> None:
        """Get uncertainty metrics for a debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "debate_id": "deb_456",
                "overall_uncertainty": 0.45,
                "round_uncertainties": [
                    {"round": 1, "uncertainty": 0.6},
                    {"round": 2, "uncertainty": 0.5},
                    {"round": 3, "uncertainty": 0.45},
                ],
                "agent_uncertainties": {"claude": 0.4, "gpt-4": 0.5},
                "convergence_trend": [0.6, 0.5, 0.45],
                "cruxes": [{"description": "Definition of success", "importance": 0.9}],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.uncertainty.get_debate_metrics("deb_456")

            mock_request.assert_called_once_with("GET", "/api/v1/uncertainty/debate/deb_456")
            assert result["debate_id"] == "deb_456"
            assert result["overall_uncertainty"] == 0.45
            assert len(result["round_uncertainties"]) == 3
            client.close()


# =========================================================================
# Agent Profile Operations
# =========================================================================


class TestUncertaintyAgentProfile:
    """Tests for agent calibration profile operations."""

    def test_get_agent_profile(self) -> None:
        """Get calibration profile for an agent."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "agent_id": "claude",
                "calibration_score": 0.85,
                "overconfidence_bias": -0.05,
                "accuracy_by_confidence": [
                    {"confidence_bucket": 0.5, "actual_accuracy": 0.52, "sample_count": 100},
                    {"confidence_bucket": 0.7, "actual_accuracy": 0.68, "sample_count": 150},
                    {"confidence_bucket": 0.9, "actual_accuracy": 0.87, "sample_count": 80},
                ],
                "total_predictions": 330,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.uncertainty.get_agent_profile("claude")

            mock_request.assert_called_once_with("GET", "/api/v1/uncertainty/agent/claude")
            assert result["agent_id"] == "claude"
            assert result["calibration_score"] == 0.85
            assert len(result["accuracy_by_confidence"]) == 3
            client.close()


# =========================================================================
# Follow-up Generation Operations
# =========================================================================


class TestUncertaintyFollowups:
    """Tests for follow-up generation operations."""

    def test_generate_followups_with_debate_id(self) -> None:
        """Generate follow-ups for a debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "suggestions": [
                    {
                        "question": "What assumptions are we making about market conditions?",
                        "rationale": "This addresses the key uncertainty around market volatility",
                        "priority": 0.9,
                        "related_crux": "Market conditions",
                    },
                    {
                        "question": "How sensitive is this conclusion to interest rate changes?",
                        "rationale": "Interest rates were mentioned but not deeply explored",
                        "priority": 0.7,
                        "related_crux": "Economic factors",
                    },
                ]
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.uncertainty.generate_followups(debate_id="deb_789")

            mock_request.assert_called_once_with(
                "POST", "/api/v1/uncertainty/followups", json={"debate_id": "deb_789"}
            )
            assert len(result["suggestions"]) == 2
            assert result["suggestions"][0]["priority"] == 0.9
            client.close()

    def test_generate_followups_with_cruxes(self) -> None:
        """Generate follow-ups from cruxes."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"suggestions": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            cruxes = [
                {"description": "Technology risk", "importance": 0.8},
                {"description": "Adoption timeline", "importance": 0.6},
            ]
            client.uncertainty.generate_followups(cruxes=cruxes)

            call_args = mock_request.call_args
            json_data = call_args[1]["json"]
            assert json_data["cruxes"] == cruxes
            client.close()

    def test_generate_followups_with_max_suggestions(self) -> None:
        """Generate follow-ups with max suggestions limit."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"suggestions": [{"question": "Q1"}]}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.uncertainty.generate_followups(debate_id="deb_101", max_suggestions=5)

            call_args = mock_request.call_args
            json_data = call_args[1]["json"]
            assert json_data["debate_id"] == "deb_101"
            assert json_data["max_suggestions"] == 5
            client.close()


# =========================================================================
# Async Tests
# =========================================================================


class TestAsyncUncertainty:
    """Tests for async Uncertainty API."""

    @pytest.mark.asyncio
    async def test_async_estimate(self) -> None:
        """Estimate uncertainty asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"overall_uncertainty": 0.3}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.uncertainty.estimate(content="Test prediction")

                mock_request.assert_called_once_with(
                    "POST",
                    "/api/v1/uncertainty/estimate",
                    json={"content": "Test prediction"},
                )
                assert result["overall_uncertainty"] == 0.3

    @pytest.mark.asyncio
    async def test_async_get_debate_metrics(self) -> None:
        """Get debate metrics asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "debate_id": "deb_async",
                "overall_uncertainty": 0.4,
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.uncertainty.get_debate_metrics("deb_async")

                mock_request.assert_called_once_with("GET", "/api/v1/uncertainty/debate/deb_async")
                assert result["overall_uncertainty"] == 0.4

    @pytest.mark.asyncio
    async def test_async_get_agent_profile(self) -> None:
        """Get agent profile asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "agent_id": "gpt-4",
                "calibration_score": 0.82,
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.uncertainty.get_agent_profile("gpt-4")

                mock_request.assert_called_once_with("GET", "/api/v1/uncertainty/agent/gpt-4")
                assert result["calibration_score"] == 0.82

    @pytest.mark.asyncio
    async def test_async_generate_followups(self) -> None:
        """Generate follow-ups asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"suggestions": [{"question": "Async Q1", "priority": 0.8}]}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.uncertainty.generate_followups(debate_id="deb_async2")

                mock_request.assert_called_once_with(
                    "POST",
                    "/api/v1/uncertainty/followups",
                    json={"debate_id": "deb_async2"},
                )
                assert len(result["suggestions"]) == 1
