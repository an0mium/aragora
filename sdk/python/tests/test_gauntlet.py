"""Tests for Gauntlet namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient


class TestGauntletRun:
    """Tests for Gauntlet run operations."""

    def test_run_gauntlet_with_debate_id(self) -> None:
        """Run Gauntlet validation on existing debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "gauntlet_id": "gnt_123",
                "verdict": "PASS",
                "confidence": 0.92,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.gauntlet.run(debate_id="dbt_123")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/gauntlet/run",
                json={"attack_rounds": 3, "debate_id": "dbt_123"},
            )
            assert result["verdict"] == "PASS"
            assert result["confidence"] == 0.92
            client.close()

    def test_run_gauntlet_with_task(self) -> None:
        """Run Gauntlet validation with new task."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"gauntlet_id": "gnt_456", "verdict": "CONDITIONAL"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.gauntlet.run(task="Should we deploy to production?")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/gauntlet/run",
                json={"attack_rounds": 3, "task": "Should we deploy to production?"},
            )
            assert result["verdict"] == "CONDITIONAL"
            client.close()

    def test_run_gauntlet_custom_rounds(self) -> None:
        """Run Gauntlet with custom attack rounds."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"gauntlet_id": "gnt_789"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.gauntlet.run(debate_id="dbt_1", attack_rounds=5)
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/gauntlet/run",
                json={"attack_rounds": 5, "debate_id": "dbt_1"},
            )
            client.close()

    def test_run_gauntlet_with_proposer_agent(self) -> None:
        """Run Gauntlet with specified proposer agent."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"gauntlet_id": "gnt_abc"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.gauntlet.run(debate_id="dbt_1", proposer_agent="claude")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/gauntlet/run",
                json={"attack_rounds": 3, "debate_id": "dbt_1", "proposer_agent": "claude"},
            )
            client.close()

    def test_run_gauntlet_with_attacker_agents(self) -> None:
        """Run Gauntlet with specified attacker agents."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"gauntlet_id": "gnt_def"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            attackers = ["gpt-4", "gemini", "mistral"]
            client.gauntlet.run(debate_id="dbt_1", attacker_agents=attackers)
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/gauntlet/run",
                json={"attack_rounds": 3, "debate_id": "dbt_1", "attacker_agents": attackers},
            )
            client.close()

    def test_run_gauntlet_full_config(self) -> None:
        """Run Gauntlet with all configuration options."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"gauntlet_id": "gnt_full", "verdict": "FAIL"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.gauntlet.run(
                task="Complex decision",
                attack_rounds=10,
                proposer_agent="claude",
                attacker_agents=["gpt-4", "gemini"],
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/gauntlet/run",
                json={
                    "attack_rounds": 10,
                    "task": "Complex decision",
                    "proposer_agent": "claude",
                    "attacker_agents": ["gpt-4", "gemini"],
                },
            )
            assert result["verdict"] == "FAIL"
            client.close()


class TestGauntletResults:
    """Tests for Gauntlet result retrieval."""

    def test_get_result(self) -> None:
        """Get Gauntlet result by ID."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "gauntlet_id": "gnt_123",
                "verdict": "PASS",
                "findings": [],
                "attack_rounds_completed": 3,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.gauntlet.get_result("gnt_123")
            mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/gnt_123")
            assert result["verdict"] == "PASS"
            assert result["attack_rounds_completed"] == 3
            client.close()

    def test_list_results_default(self) -> None:
        """List Gauntlet results with defaults."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"results": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.gauntlet.list_results()
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/gauntlet",
                params={"limit": 20, "offset": 0},
            )
            assert result["total"] == 0
            client.close()

    def test_list_results_with_verdict_filter(self) -> None:
        """List Gauntlet results filtered by verdict."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"results": [{"verdict": "PASS"}], "total": 5}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.gauntlet.list_results(verdict="PASS")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/gauntlet",
                params={"limit": 20, "offset": 0, "verdict": "PASS"},
            )
            assert result["total"] == 5
            client.close()

    def test_list_results_with_pagination(self) -> None:
        """List Gauntlet results with custom pagination."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"results": [], "total": 100}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.gauntlet.list_results(limit=50, offset=25)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/gauntlet",
                params={"limit": 50, "offset": 25},
            )
            client.close()


class TestGauntletReceipts:
    """Tests for Gauntlet receipt operations."""

    def test_get_receipt(self) -> None:
        """Get receipt for a Gauntlet run."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "receipt_id": "rcp_123",
                "hash": "sha256:abc123...",
                "timestamp": "2024-01-15T10:30:00Z",
                "verdict": "PASS",
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.gauntlet.get_receipt("gnt_123")
            mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/gnt_123/receipt")
            assert result["receipt_id"] == "rcp_123"
            assert "hash" in result
            client.close()

    def test_verify_receipt(self) -> None:
        """Verify a Gauntlet receipt's integrity."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "valid": True,
                "hash": "sha256:abc123...",
                "verified_at": "2024-01-15T10:35:00Z",
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.gauntlet.verify_receipt("gnt_123")
            mock_request.assert_called_once_with("POST", "/api/v1/gauntlet/gnt_123/receipt/verify")
            assert result["valid"] is True
            client.close()

    def test_export_receipt_json(self) -> None:
        """Export receipt as JSON."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"data": {"receipt": "..."}, "format": "json"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.gauntlet.export_receipt("gnt_123", format="json")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/gauntlet/gnt_123/receipt/export",
                params={"format": "json"},
            )
            assert result["format"] == "json"
            client.close()

    def test_export_receipt_pdf(self) -> None:
        """Export receipt as PDF."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"url": "https://cdn.aragora.ai/receipts/...pdf"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.gauntlet.export_receipt("gnt_123", format="pdf")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/gauntlet/gnt_123/receipt/export",
                params={"format": "pdf"},
            )
            assert "url" in result
            client.close()

    def test_export_receipt_sarif(self) -> None:
        """Export receipt as SARIF format."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"sarif": {"version": "2.1.0"}}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.gauntlet.export_receipt("gnt_123", format="sarif")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/gauntlet/gnt_123/receipt/export",
                params={"format": "sarif"},
            )
            assert "sarif" in result
            client.close()

    def test_list_receipts(self) -> None:
        """List all receipts."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"receipts": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.gauntlet.list_receipts()
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/gauntlet/receipts",
                params={"limit": 20, "offset": 0},
            )
            assert result["total"] == 0
            client.close()

    def test_get_receipt_by_id(self) -> None:
        """Get receipt by direct ID."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "rcp_456", "gauntlet_id": "gnt_123"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.gauntlet.get_receipt_by_id("rcp_456")
            mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/receipts/rcp_456")
            assert result["id"] == "rcp_456"
            client.close()

    def test_export_receipt_by_id(self) -> None:
        """Export receipt by direct ID."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"data": "..."}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.gauntlet.export_receipt_by_id("rcp_456", format="markdown")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/gauntlet/receipts/rcp_456/export",
                params={"format": "markdown"},
            )
            client.close()

    def test_stream_receipt(self) -> None:
        """Stream receipt data."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"stream_url": "wss://..."}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.gauntlet.stream_receipt("rcp_456")
            mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/receipts/rcp_456/stream")
            assert "stream_url" in result
            client.close()

    def test_export_receipts_bundle(self) -> None:
        """Export multiple receipts as bundle."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"bundle_url": "https://..."}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            receipt_ids = ["rcp_1", "rcp_2", "rcp_3"]
            result = client.gauntlet.export_receipts_bundle(receipt_ids)
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/gauntlet/receipts/export/bundle",
                json={"receipt_ids": receipt_ids},
            )
            assert "bundle_url" in result
            client.close()


class TestGauntletFindings:
    """Tests for Gauntlet findings operations."""

    def test_get_findings(self) -> None:
        """Get findings from a Gauntlet run."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "findings": [
                    {
                        "id": "fnd_1",
                        "severity": "high",
                        "description": "Assumption not validated",
                    },
                    {
                        "id": "fnd_2",
                        "severity": "medium",
                        "description": "Edge case not considered",
                    },
                ]
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.gauntlet.get_findings("gnt_123")
            mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/gnt_123/findings")
            assert len(result["findings"]) == 2
            assert result["findings"][0]["severity"] == "high"
            client.close()


class TestGauntletAttacks:
    """Tests for Gauntlet attack details."""

    def test_get_attacks(self) -> None:
        """Get attack details from a Gauntlet run."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "attacks": [
                    {
                        "round": 1,
                        "attacker": "gpt-4",
                        "argument": "What about scalability?",
                        "outcome": "defended",
                    },
                    {
                        "round": 2,
                        "attacker": "gemini",
                        "argument": "Cost implications unclear",
                        "outcome": "partially_defended",
                    },
                ]
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.gauntlet.get_attacks("gnt_123")
            mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/gnt_123/attacks")
            assert len(result["attacks"]) == 2
            assert result["attacks"][0]["outcome"] == "defended"
            client.close()


class TestGauntletStats:
    """Tests for Gauntlet statistics."""

    def test_get_stats(self) -> None:
        """Get Gauntlet statistics."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "total_runs": 150,
                "pass_rate": 0.72,
                "average_rounds": 3.5,
                "common_findings": ["assumption_validation", "edge_cases"],
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.gauntlet.get_stats()
            mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/stats")
            assert result["total_runs"] == 150
            assert result["pass_rate"] == 0.72
            client.close()


class TestGauntletHeatmaps:
    """Tests for Gauntlet heatmap operations."""

    def test_list_heatmaps(self) -> None:
        """List all heatmaps."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"heatmaps": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.gauntlet.list_heatmaps()
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/gauntlet/heatmaps",
                params={"limit": 20, "offset": 0},
            )
            assert result["total"] == 0
            client.close()

    def test_list_heatmaps_with_pagination(self) -> None:
        """List heatmaps with custom pagination."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"heatmaps": [{"id": "hm_1"}], "total": 10}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.gauntlet.list_heatmaps(limit=5, offset=5)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/gauntlet/heatmaps",
                params={"limit": 5, "offset": 5},
            )
            assert len(result["heatmaps"]) == 1
            client.close()

    def test_get_heatmap(self) -> None:
        """Get a specific heatmap."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "id": "hm_123",
                "data": [[0.1, 0.5], [0.8, 0.3]],
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.gauntlet.get_heatmap("hm_123")
            mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/heatmaps/hm_123")
            assert result["id"] == "hm_123"
            client.close()

    def test_export_heatmap(self) -> None:
        """Export a heatmap."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"url": "https://..."}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.gauntlet.export_heatmap("hm_123", format="csv")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/gauntlet/heatmaps/hm_123/export",
                params={"format": "csv"},
            )
            assert "url" in result
            client.close()


class TestAsyncGauntlet:
    """Tests for async Gauntlet methods."""

    @pytest.mark.asyncio
    async def test_async_run_gauntlet(self) -> None:
        """Run Gauntlet asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"gauntlet_id": "gnt_async", "verdict": "PASS"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.gauntlet.run(debate_id="dbt_1")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/gauntlet/run",
                json={"attack_rounds": 3, "debate_id": "dbt_1"},
            )
            assert result["verdict"] == "PASS"
            await client.close()

    @pytest.mark.asyncio
    async def test_async_get_result(self) -> None:
        """Get Gauntlet result asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"gauntlet_id": "gnt_1", "verdict": "CONDITIONAL"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.gauntlet.get_result("gnt_1")
            mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/gnt_1")
            assert result["verdict"] == "CONDITIONAL"
            await client.close()

    @pytest.mark.asyncio
    async def test_async_list_results(self) -> None:
        """List Gauntlet results asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"results": [], "total": 0}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.gauntlet.list_results(verdict="FAIL")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/gauntlet",
                params={"limit": 20, "offset": 0, "verdict": "FAIL"},
            )
            assert result["total"] == 0
            await client.close()

    @pytest.mark.asyncio
    async def test_async_get_receipt(self) -> None:
        """Get receipt asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"receipt_id": "rcp_async"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.gauntlet.get_receipt("gnt_1")
            mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/gnt_1/receipt")
            assert result["receipt_id"] == "rcp_async"
            await client.close()

    @pytest.mark.asyncio
    async def test_async_verify_receipt(self) -> None:
        """Verify receipt asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"valid": True}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.gauntlet.verify_receipt("gnt_1")
            mock_request.assert_called_once_with("POST", "/api/v1/gauntlet/gnt_1/receipt/verify")
            assert result["valid"] is True
            await client.close()

    @pytest.mark.asyncio
    async def test_async_get_findings(self) -> None:
        """Get findings asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"findings": [{"severity": "low"}]}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.gauntlet.get_findings("gnt_1")
            mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/gnt_1/findings")
            assert len(result["findings"]) == 1
            await client.close()

    @pytest.mark.asyncio
    async def test_async_get_attacks(self) -> None:
        """Get attacks asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"attacks": []}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.gauntlet.get_attacks("gnt_1")
            mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/gnt_1/attacks")
            assert "attacks" in result
            await client.close()

    @pytest.mark.asyncio
    async def test_async_get_stats(self) -> None:
        """Get stats asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"total_runs": 100}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.gauntlet.get_stats()
            mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/stats")
            assert result["total_runs"] == 100
            await client.close()

    @pytest.mark.asyncio
    async def test_async_export_receipt(self) -> None:
        """Export receipt asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"url": "https://..."}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.gauntlet.export_receipt("gnt_1", format="html")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/gauntlet/gnt_1/receipt/export",
                params={"format": "html"},
            )
            assert "url" in result
            await client.close()

    @pytest.mark.asyncio
    async def test_async_list_heatmaps(self) -> None:
        """List heatmaps asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"heatmaps": [], "total": 0}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.gauntlet.list_heatmaps()
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/gauntlet/heatmaps",
                params={"limit": 20, "offset": 0},
            )
            assert result["total"] == 0
            await client.close()

    @pytest.mark.asyncio
    async def test_async_get_heatmap(self) -> None:
        """Get heatmap asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"id": "hm_async", "data": []}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.gauntlet.get_heatmap("hm_async")
            mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/heatmaps/hm_async")
            assert result["id"] == "hm_async"
            await client.close()

    @pytest.mark.asyncio
    async def test_async_export_receipts_bundle(self) -> None:
        """Export receipts bundle asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"bundle_url": "https://..."}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.gauntlet.export_receipts_bundle(["rcp_1", "rcp_2"])
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/gauntlet/receipts/export/bundle",
                json={"receipt_ids": ["rcp_1", "rcp_2"]},
            )
            assert "bundle_url" in result
            await client.close()
