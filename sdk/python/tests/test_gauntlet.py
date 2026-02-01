"""Tests for Gauntlet namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient


class TestGauntletRun:
    """Tests for running Gauntlet validations."""

    def test_run_with_debate_id(self) -> None:
        """Run Gauntlet on an existing debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "gauntlet_id": "gnt_123",
                "verdict": "PASS",
                "confidence": 0.92,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.gauntlet.run(debate_id="dbt_123")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/gauntlet/run",
                json={"attack_rounds": 3, "debate_id": "dbt_123"},
            )
            assert result["verdict"] == "PASS"
            client.close()

    def test_run_with_task(self) -> None:
        """Run Gauntlet with a new task."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"gauntlet_id": "gnt_123", "verdict": "CONDITIONAL"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.gauntlet.run(task="Should we deploy to production?")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/gauntlet/run",
                json={"attack_rounds": 3, "task": "Should we deploy to production?"},
            )
            client.close()

    def test_run_with_custom_rounds(self) -> None:
        """Run Gauntlet with custom attack rounds."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"gauntlet_id": "gnt_123"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.gauntlet.run(debate_id="dbt_123", attack_rounds=5)

            call_args = mock_request.call_args
            assert call_args[1]["json"]["attack_rounds"] == 5
            client.close()

    def test_run_with_agents(self) -> None:
        """Run Gauntlet with specific agents."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"gauntlet_id": "gnt_123"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.gauntlet.run(
                task="Test decision",
                proposer_agent="claude",
                attacker_agents=["gpt-4", "gemini"],
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["proposer_agent"] == "claude"
            assert call_args[1]["json"]["attacker_agents"] == ["gpt-4", "gemini"]
            client.close()


class TestGauntletResults:
    """Tests for Gauntlet result operations."""

    def test_get_result(self) -> None:
        """Get a Gauntlet result by ID."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "gauntlet_id": "gnt_123",
                "verdict": "PASS",
                "findings_count": 2,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.gauntlet.get_result("gnt_123")

            mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/gnt_123")
            assert result["gauntlet_id"] == "gnt_123"
            client.close()

    def test_list_results(self) -> None:
        """List Gauntlet results."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "results": [{"gauntlet_id": "gnt_1"}],
                "total": 10,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.gauntlet.list_results(limit=20, offset=0)

            mock_request.assert_called_once_with(
                "GET", "/api/v1/gauntlet", params={"limit": 20, "offset": 0}
            )
            assert result["total"] == 10
            client.close()

    def test_list_results_with_verdict_filter(self) -> None:
        """List Gauntlet results filtered by verdict."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"results": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.gauntlet.list_results(verdict="FAIL")

            call_args = mock_request.call_args
            assert call_args[1]["params"]["verdict"] == "FAIL"
            client.close()

    def test_get_findings(self) -> None:
        """Get findings from a Gauntlet run."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "findings": [
                    {"id": "fnd_1", "severity": "high", "description": "Missing error handling"}
                ]
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.gauntlet.get_findings("gnt_123")

            mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/gnt_123/findings")
            assert len(result["findings"]) == 1
            client.close()

    def test_get_attacks(self) -> None:
        """Get attack details from a Gauntlet run."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "attacks": [{"round": 1, "attacker": "gpt-4", "argument": "Counter-argument"}]
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.gauntlet.get_attacks("gnt_123")

            mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/gnt_123/attacks")
            assert len(result["attacks"]) == 1
            client.close()

    def test_get_stats(self) -> None:
        """Get Gauntlet statistics."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "total_runs": 100,
                "pass_rate": 0.85,
                "common_findings": ["error_handling", "security"],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.gauntlet.get_stats()

            mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/stats")
            assert result["pass_rate"] == 0.85
            client.close()


class TestGauntletReceipts:
    """Tests for Gauntlet receipt operations."""

    def test_get_receipt(self) -> None:
        """Get the decision receipt for a Gauntlet run."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "receipt_id": "rcp_123",
                "hash": "sha256:abc123",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.gauntlet.get_receipt("gnt_123")

            mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/gnt_123/receipt")
            assert result["receipt_id"] == "rcp_123"
            client.close()

    def test_verify_receipt(self) -> None:
        """Verify a Gauntlet receipt's integrity."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "valid": True,
                "hash": "sha256:abc123",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.gauntlet.verify_receipt("gnt_123")

            mock_request.assert_called_once_with("POST", "/api/v1/gauntlet/gnt_123/receipt/verify")
            assert result["valid"] is True
            client.close()

    def test_export_receipt_json(self) -> None:
        """Export a Gauntlet receipt as JSON."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"data": {}}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.gauntlet.export_receipt("gnt_123", format="json")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/gauntlet/gnt_123/receipt/export",
                params={"format": "json"},
            )
            client.close()

    def test_export_receipt_sarif(self) -> None:
        """Export a Gauntlet receipt as SARIF."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"data": {}}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.gauntlet.export_receipt("gnt_123", format="sarif")

            call_args = mock_request.call_args
            assert call_args[1]["params"]["format"] == "sarif"
            client.close()

    def test_list_receipts(self) -> None:
        """List all receipts with pagination."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"receipts": [], "total": 0}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.gauntlet.list_receipts(limit=50, offset=10)

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/gauntlet/receipts",
                params={"limit": 50, "offset": 10},
            )
            client.close()

    def test_get_receipt_by_id(self) -> None:
        """Get a receipt by ID."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"receipt_id": "rcp_123"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.gauntlet.get_receipt_by_id("rcp_123")

            mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/receipts/rcp_123")
            client.close()

    def test_export_receipt_by_id(self) -> None:
        """Export a receipt by ID."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"data": {}}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.gauntlet.export_receipt_by_id("rcp_123", format="html")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/gauntlet/receipts/rcp_123/export",
                params={"format": "html"},
            )
            client.close()

    def test_stream_receipt(self) -> None:
        """Stream a receipt's data."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"stream": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.gauntlet.stream_receipt("rcp_123")

            mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/receipts/rcp_123/stream")
            client.close()

    def test_export_receipts_bundle(self) -> None:
        """Export multiple receipts as a bundle."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"bundle_url": "https://..."}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.gauntlet.export_receipts_bundle(["rcp_1", "rcp_2", "rcp_3"])

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/gauntlet/receipts/export/bundle",
                json={"receipt_ids": ["rcp_1", "rcp_2", "rcp_3"]},
            )
            client.close()


class TestGauntletHeatmaps:
    """Tests for Gauntlet heatmap operations."""

    def test_list_heatmaps(self) -> None:
        """List all heatmaps with pagination."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"heatmaps": [], "total": 0}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.gauntlet.list_heatmaps(limit=10, offset=0)

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/gauntlet/heatmaps",
                params={"limit": 10, "offset": 0},
            )
            client.close()

    def test_get_heatmap(self) -> None:
        """Get a heatmap by ID."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "heatmap_id": "hm_123",
                "data": [[0.1, 0.5], [0.8, 0.3]],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.gauntlet.get_heatmap("hm_123")

            mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/heatmaps/hm_123")
            assert result["heatmap_id"] == "hm_123"
            client.close()

    def test_export_heatmap(self) -> None:
        """Export a heatmap."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"data": {}}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.gauntlet.export_heatmap("hm_123", format="csv")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/gauntlet/heatmaps/hm_123/export",
                params={"format": "csv"},
            )
            client.close()


class TestAsyncGauntlet:
    """Tests for async Gauntlet API."""

    @pytest.mark.asyncio
    async def test_async_run(self) -> None:
        """Run Gauntlet asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"gauntlet_id": "gnt_123", "verdict": "PASS"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.gauntlet.run(task="Test decision")

                mock_request.assert_called_once_with(
                    "POST",
                    "/api/v1/gauntlet/run",
                    json={"attack_rounds": 3, "task": "Test decision"},
                )
                assert result["verdict"] == "PASS"

    @pytest.mark.asyncio
    async def test_async_get_result(self) -> None:
        """Get result asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"gauntlet_id": "gnt_123"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.gauntlet.get_result("gnt_123")

                mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/gnt_123")

    @pytest.mark.asyncio
    async def test_async_list_results(self) -> None:
        """List results asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"results": []}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.gauntlet.list_results(verdict="CONDITIONAL")

                call_args = mock_request.call_args
                assert call_args[1]["params"]["verdict"] == "CONDITIONAL"

    @pytest.mark.asyncio
    async def test_async_get_receipt(self) -> None:
        """Get receipt asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"receipt_id": "rcp_123"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.gauntlet.get_receipt("gnt_123")

                mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/gnt_123/receipt")

    @pytest.mark.asyncio
    async def test_async_verify_receipt(self) -> None:
        """Verify receipt asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"valid": True}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.gauntlet.verify_receipt("gnt_123")

                mock_request.assert_called_once_with(
                    "POST", "/api/v1/gauntlet/gnt_123/receipt/verify"
                )
                assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_async_get_findings(self) -> None:
        """Get findings asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"findings": []}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.gauntlet.get_findings("gnt_123")

                mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/gnt_123/findings")

    @pytest.mark.asyncio
    async def test_async_get_attacks(self) -> None:
        """Get attacks asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"attacks": []}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.gauntlet.get_attacks("gnt_123")

                mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/gnt_123/attacks")

    @pytest.mark.asyncio
    async def test_async_get_stats(self) -> None:
        """Get stats asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"total_runs": 50}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.gauntlet.get_stats()

                mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/stats")
                assert result["total_runs"] == 50

    @pytest.mark.asyncio
    async def test_async_get_heatmap(self) -> None:
        """Get heatmap asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"heatmap_id": "hm_123"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.gauntlet.get_heatmap("hm_123")

                mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/heatmaps/hm_123")
