"""Tests for Nomic namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestNomicState:
    """Tests for nomic loop state and monitoring."""

    def test_get_state(self) -> None:
        """Get current nomic loop state."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "running": True,
                "cycle": 3,
                "phase": "implement",
                "paused": False,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.nomic.get_state()

            mock_request.assert_called_once_with("GET", "/api/v1/nomic/state")
            assert result["running"] is True
            assert result["cycle"] == 3
            client.close()

    def test_get_health(self) -> None:
        """Get nomic loop health with stall detection."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "status": "healthy",
                "cycle": 2,
                "phase": "debate",
                "stall_duration_seconds": 0,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.nomic.get_health()

            mock_request.assert_called_once_with("GET", "/api/v1/nomic/health")
            assert result["status"] == "healthy"
            client.close()

    def test_get_metrics(self) -> None:
        """Get nomic loop metrics summary."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "summary": {"total_cycles": 10, "successful_cycles": 9},
                "status": "healthy",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.nomic.get_metrics()

            mock_request.assert_called_once_with("GET", "/api/v1/nomic/metrics")
            assert result["summary"]["total_cycles"] == 10
            client.close()

    def test_get_logs(self) -> None:
        """Get recent nomic loop log lines."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "lines": ["Starting cycle 3", "Phase: debate"],
                "total": 100,
                "showing": 2,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.nomic.get_logs(lines=50)

            mock_request.assert_called_once_with("GET", "/api/v1/nomic/log", params={"lines": 50})
            assert len(result["lines"]) == 2
            client.close()

    def test_get_risk_register(self) -> None:
        """Get risk register entries."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "risks": [{"id": "risk_1", "severity": "high"}],
                "total": 5,
                "critical_count": 1,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.nomic.get_risk_register(limit=10)

            mock_request.assert_called_once_with(
                "GET", "/api/v1/nomic/risk-register", params={"limit": 10}
            )
            assert result["critical_count"] == 1
            client.close()

    def test_get_risk_register_no_limit(self) -> None:
        """Get risk register without limit."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"risks": [], "total": 0}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.nomic.get_risk_register()

            mock_request.assert_called_once_with("GET", "/api/v1/nomic/risk-register", params={})
            client.close()


class TestNomicControl:
    """Tests for nomic loop control operations."""

    def test_start(self) -> None:
        """Start the nomic loop."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "status": "started",
                "pid": 12345,
                "target_cycles": 5,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.nomic.start(cycles=5, auto_approve=False)

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/nomic/control/start",
                json={"auto_approve": False, "dry_run": False, "cycles": 5},
            )
            assert result["status"] == "started"
            client.close()

    def test_start_with_max_cycles(self) -> None:
        """Start with deprecated max_cycles parameter."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"status": "started"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.nomic.start(max_cycles=3)

            call_args = mock_request.call_args
            assert call_args[1]["json"]["max_cycles"] == 3
            client.close()

    def test_start_dry_run(self) -> None:
        """Start nomic loop in dry run mode."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"status": "dry_run"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.nomic.start(dry_run=True)

            call_args = mock_request.call_args
            assert call_args[1]["json"]["dry_run"] is True
            client.close()

    def test_stop(self) -> None:
        """Stop the running nomic loop."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"status": "stopped", "pid": 12345}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.nomic.stop()

            mock_request.assert_called_once_with(
                "POST", "/api/v1/nomic/control/stop", json={"graceful": True}
            )
            assert result["status"] == "stopped"
            client.close()

    def test_stop_force(self) -> None:
        """Force stop the nomic loop."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"status": "stopped"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.nomic.stop(graceful=False)

            mock_request.assert_called_once_with(
                "POST", "/api/v1/nomic/control/stop", json={"graceful": False}
            )
            client.close()

    def test_pause(self) -> None:
        """Pause the nomic loop."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "status": "paused",
                "cycle": 2,
                "phase": "design",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.nomic.pause()

            mock_request.assert_called_once_with("POST", "/api/v1/nomic/control/pause")
            assert result["status"] == "paused"
            client.close()

    def test_resume(self) -> None:
        """Resume a paused nomic loop."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "status": "resumed",
                "cycle": 2,
                "phase": "design",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.nomic.resume()

            mock_request.assert_called_once_with("POST", "/api/v1/nomic/control/resume")
            assert result["status"] == "resumed"
            client.close()

    def test_skip_phase(self) -> None:
        """Skip the current phase."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "status": "skipped",
                "previous_phase": "debate",
                "next_phase": "design",
                "cycle": 2,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.nomic.skip_phase()

            mock_request.assert_called_once_with("POST", "/api/v1/nomic/control/skip-phase")
            assert result["previous_phase"] == "debate"
            client.close()


class TestNomicProposals:
    """Tests for proposal management."""

    def test_get_proposals(self) -> None:
        """Get pending improvement proposals."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "proposals": [{"id": "prop_1", "title": "Improve error handling"}],
                "total": 3,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.nomic.get_proposals()

            mock_request.assert_called_once_with("GET", "/api/v1/nomic/proposals")
            assert len(result["proposals"]) == 1
            client.close()

    def test_approve_proposal(self) -> None:
        """Approve a pending proposal."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"status": "approved", "proposal_id": "prop_1"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.nomic.approve_proposal("prop_1", approved_by="admin")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/nomic/proposals/approve",
                json={"proposal_id": "prop_1", "approved_by": "admin"},
            )
            assert result["status"] == "approved"
            client.close()

    def test_reject_proposal(self) -> None:
        """Reject a pending proposal."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"status": "rejected", "proposal_id": "prop_1"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.nomic.reject_proposal(
                "prop_1", rejected_by="admin", reason="Out of scope"
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/nomic/proposals/reject",
                json={
                    "proposal_id": "prop_1",
                    "rejected_by": "admin",
                    "reason": "Out of scope",
                },
            )
            assert result["status"] == "rejected"
            client.close()

    def test_get_proposal_by_id(self) -> None:
        """Get a specific proposal by ID."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "proposals": [
                    {"id": "prop_1", "title": "First"},
                    {"id": "prop_2", "title": "Second"},
                ]
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.nomic.get_proposal("prop_2")

            assert result is not None
            assert result["id"] == "prop_2"
            assert result["title"] == "Second"
            client.close()

    def test_get_proposal_not_found(self) -> None:
        """Get a proposal that doesn't exist."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"proposals": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.nomic.get_proposal("nonexistent")

            assert result is None
            client.close()


class TestNomicModes:
    """Tests for operational modes."""

    def test_get_modes(self) -> None:
        """Get available operational modes."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "modes": [
                    {"name": "default", "description": "Default mode"},
                    {"name": "aggressive", "description": "More frequent cycles"},
                ],
                "total": 2,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.nomic.get_modes()

            mock_request.assert_called_once_with("GET", "/api/v1/modes")
            assert result["total"] == 2
            client.close()


class TestNomicGasTown:
    """Tests for Gas Town witness and mayor monitoring."""

    def test_get_witness_status(self) -> None:
        """Get Gas Town witness patrol status."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "active": True,
                "patrol_count": 15,
                "violations_detected": 2,
                "witnesses": ["witness_1", "witness_2"],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.nomic.get_witness_status()

            mock_request.assert_called_once_with("GET", "/api/v1/nomic/witness/status")
            assert result["active"] is True
            assert result["patrol_count"] == 15
            client.close()

    def test_get_mayor_current(self) -> None:
        """Get current Gas Town mayor information."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "current_mayor": "mayor_alice",
                "approval_rating": 0.85,
                "policies_enacted": 3,
                "vetoes": 1,
                "emergency_powers_active": False,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.nomic.get_mayor_current()

            mock_request.assert_called_once_with("GET", "/api/v1/nomic/mayor/current")
            assert result["current_mayor"] == "mayor_alice"
            assert result["approval_rating"] == 0.85
            client.close()


class TestNomicConvenience:
    """Tests for convenience methods."""

    def test_state_alias(self) -> None:
        """Test state() alias for get_state()."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"running": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.nomic.state()

            mock_request.assert_called_once_with("GET", "/api/v1/nomic/state")
            assert result["running"] is True
            client.close()

    def test_health_alias(self) -> None:
        """Test health() alias for get_health()."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"status": "healthy"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.nomic.health()

            mock_request.assert_called_once_with("GET", "/api/v1/nomic/health")
            assert result["status"] == "healthy"
            client.close()

    def test_is_running_true(self) -> None:
        """Check if nomic loop is running."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"running": True, "paused": False}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.nomic.is_running()

            assert result is True
            client.close()

    def test_is_running_paused(self) -> None:
        """Check if nomic loop is paused (not actively running)."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"running": True, "paused": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.nomic.is_running()

            assert result is False
            client.close()

    def test_is_running_stopped(self) -> None:
        """Check if nomic loop is stopped."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"running": False, "paused": False}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.nomic.is_running()

            assert result is False
            client.close()


class TestAsyncNomic:
    """Tests for async nomic API."""

    @pytest.mark.asyncio
    async def test_async_get_state(self) -> None:
        """Get state asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"running": True, "cycle": 5}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.nomic.get_state()

                mock_request.assert_called_once_with("GET", "/api/v1/nomic/state")
                assert result["cycle"] == 5

    @pytest.mark.asyncio
    async def test_async_get_health(self) -> None:
        """Get health asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"status": "healthy"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.nomic.get_health()

                mock_request.assert_called_once_with("GET", "/api/v1/nomic/health")
                assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_async_start(self) -> None:
        """Start nomic loop asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"status": "started"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.nomic.start(cycles=3, auto_approve=True)

                mock_request.assert_called_once_with(
                    "POST",
                    "/api/v1/nomic/control/start",
                    json={"auto_approve": True, "dry_run": False, "cycles": 3},
                )

    @pytest.mark.asyncio
    async def test_async_stop(self) -> None:
        """Stop nomic loop asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"status": "stopped"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.nomic.stop()

                mock_request.assert_called_once_with(
                    "POST", "/api/v1/nomic/control/stop", json={"graceful": True}
                )

    @pytest.mark.asyncio
    async def test_async_pause(self) -> None:
        """Pause nomic loop asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"status": "paused"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.nomic.pause()

                mock_request.assert_called_once_with("POST", "/api/v1/nomic/control/pause")

    @pytest.mark.asyncio
    async def test_async_resume(self) -> None:
        """Resume nomic loop asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"status": "resumed"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.nomic.resume()

                mock_request.assert_called_once_with("POST", "/api/v1/nomic/control/resume")

    @pytest.mark.asyncio
    async def test_async_get_proposals(self) -> None:
        """Get proposals asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"proposals": [], "total": 0}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.nomic.get_proposals()

                mock_request.assert_called_once_with("GET", "/api/v1/nomic/proposals")

    @pytest.mark.asyncio
    async def test_async_is_running(self) -> None:
        """Check if running asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"running": True, "paused": False}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.nomic.is_running()

                assert result is True
