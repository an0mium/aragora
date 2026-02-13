"""
Tests for NomicAPI resource.

Tests cover:
- NomicAPI read-only methods: state, health, metrics, log, risk_register,
  witness_status, mayor_current, proposals, modes
- NomicAPI control methods: start, stop, pause, resume, skip_phase,
  approve_proposal, reject_proposal
- Both sync and async variants for every method
- Parameter passing (lines, limit, config, payload, proposal_id, reason)
- Default values for optional parameters
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.client.client import AragoraClient
from aragora.client.resources.nomic import NomicAPI


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_client() -> AragoraClient:
    """Create a mock AragoraClient."""
    client = MagicMock(spec=AragoraClient)
    return client


@pytest.fixture
def nomic_api(mock_client: AragoraClient) -> NomicAPI:
    """Create a NomicAPI with mock client."""
    return NomicAPI(mock_client)


# ============================================================================
# NomicAPI.state() Tests
# ============================================================================


class TestNomicAPIState:
    """Tests for NomicAPI.state() method."""

    def test_state(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test state() returns current nomic loop state."""
        mock_client._get.return_value = {
            "phase": "debate",
            "cycle": 3,
            "running": True,
        }

        result = nomic_api.state()

        assert result["phase"] == "debate"
        assert result["cycle"] == 3
        assert result["running"] is True
        mock_client._get.assert_called_once_with("/api/v1/nomic/state")


class TestNomicAPIStateAsync:
    """Tests for NomicAPI.state_async() method."""

    @pytest.mark.asyncio
    async def test_state_async(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test state_async() returns current nomic loop state."""
        mock_client._get_async = AsyncMock(
            return_value={
                "phase": "implement",
                "cycle": 5,
                "running": True,
            }
        )

        result = await nomic_api.state_async()

        assert result["phase"] == "implement"
        assert result["cycle"] == 5
        mock_client._get_async.assert_called_once_with("/api/v1/nomic/state")


# ============================================================================
# NomicAPI.health() Tests
# ============================================================================


class TestNomicAPIHealth:
    """Tests for NomicAPI.health() method."""

    def test_health(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test health() returns nomic loop health with stall detection."""
        mock_client._get.return_value = {
            "healthy": True,
            "stalled": False,
            "uptime_seconds": 3600,
        }

        result = nomic_api.health()

        assert result["healthy"] is True
        assert result["stalled"] is False
        assert result["uptime_seconds"] == 3600
        mock_client._get.assert_called_once_with("/api/v1/nomic/health")


class TestNomicAPIHealthAsync:
    """Tests for NomicAPI.health_async() method."""

    @pytest.mark.asyncio
    async def test_health_async(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test health_async() returns nomic loop health."""
        mock_client._get_async = AsyncMock(
            return_value={"healthy": False, "stalled": True}
        )

        result = await nomic_api.health_async()

        assert result["healthy"] is False
        assert result["stalled"] is True
        mock_client._get_async.assert_called_once_with("/api/v1/nomic/health")


# ============================================================================
# NomicAPI.metrics() Tests
# ============================================================================


class TestNomicAPIMetrics:
    """Tests for NomicAPI.metrics() method."""

    def test_metrics(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test metrics() returns nomic loop metrics summary."""
        mock_client._get.return_value = {
            "total_cycles": 42,
            "successful_cycles": 40,
            "average_cycle_time_seconds": 120.5,
        }

        result = nomic_api.metrics()

        assert result["total_cycles"] == 42
        assert result["successful_cycles"] == 40
        assert result["average_cycle_time_seconds"] == 120.5
        mock_client._get.assert_called_once_with("/api/v1/nomic/metrics")


class TestNomicAPIMetricsAsync:
    """Tests for NomicAPI.metrics_async() method."""

    @pytest.mark.asyncio
    async def test_metrics_async(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test metrics_async() returns nomic loop metrics."""
        mock_client._get_async = AsyncMock(
            return_value={"total_cycles": 10, "successful_cycles": 9}
        )

        result = await nomic_api.metrics_async()

        assert result["total_cycles"] == 10
        mock_client._get_async.assert_called_once_with("/api/v1/nomic/metrics")


# ============================================================================
# NomicAPI.log() Tests
# ============================================================================


class TestNomicAPILog:
    """Tests for NomicAPI.log() method."""

    def test_log_default_lines(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test log() with default lines parameter (100)."""
        mock_client._get.return_value = {
            "lines": ["[INFO] Cycle 3 started", "[INFO] Phase: debate"],
            "total": 2,
        }

        result = nomic_api.log()

        assert len(result["lines"]) == 2
        mock_client._get.assert_called_once_with(
            "/api/v1/nomic/log", params={"lines": 100}
        )

    def test_log_custom_lines(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test log() with custom lines parameter."""
        mock_client._get.return_value = {"lines": ["[ERROR] Something failed"], "total": 1}

        result = nomic_api.log(lines=50)

        assert result["total"] == 1
        mock_client._get.assert_called_once_with(
            "/api/v1/nomic/log", params={"lines": 50}
        )


class TestNomicAPILogAsync:
    """Tests for NomicAPI.log_async() method."""

    @pytest.mark.asyncio
    async def test_log_async_default(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test log_async() with default lines parameter."""
        mock_client._get_async = AsyncMock(
            return_value={"lines": ["async log line"], "total": 1}
        )

        result = await nomic_api.log_async()

        assert result["lines"] == ["async log line"]
        mock_client._get_async.assert_called_once_with(
            "/api/v1/nomic/log", params={"lines": 100}
        )

    @pytest.mark.asyncio
    async def test_log_async_custom_lines(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test log_async() with custom lines parameter."""
        mock_client._get_async = AsyncMock(
            return_value={"lines": [], "total": 0}
        )

        result = await nomic_api.log_async(lines=25)

        assert result["total"] == 0
        mock_client._get_async.assert_called_once_with(
            "/api/v1/nomic/log", params={"lines": 25}
        )


# ============================================================================
# NomicAPI.risk_register() Tests
# ============================================================================


class TestNomicAPIRiskRegister:
    """Tests for NomicAPI.risk_register() method."""

    def test_risk_register_default_limit(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test risk_register() with default limit (50)."""
        mock_client._get.return_value = {
            "risks": [
                {"id": "risk-1", "severity": "high", "description": "Stale model weights"},
            ],
            "total": 1,
        }

        result = nomic_api.risk_register()

        assert len(result["risks"]) == 1
        assert result["risks"][0]["severity"] == "high"
        mock_client._get.assert_called_once_with(
            "/api/v1/nomic/risk-register", params={"limit": 50}
        )

    def test_risk_register_custom_limit(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test risk_register() with custom limit."""
        mock_client._get.return_value = {"risks": [], "total": 0}

        result = nomic_api.risk_register(limit=10)

        assert result["total"] == 0
        mock_client._get.assert_called_once_with(
            "/api/v1/nomic/risk-register", params={"limit": 10}
        )


class TestNomicAPIRiskRegisterAsync:
    """Tests for NomicAPI.risk_register_async() method."""

    @pytest.mark.asyncio
    async def test_risk_register_async_default(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test risk_register_async() with default limit."""
        mock_client._get_async = AsyncMock(
            return_value={"risks": [{"id": "r-async"}], "total": 1}
        )

        result = await nomic_api.risk_register_async()

        assert result["total"] == 1
        mock_client._get_async.assert_called_once_with(
            "/api/v1/nomic/risk-register", params={"limit": 50}
        )

    @pytest.mark.asyncio
    async def test_risk_register_async_custom_limit(
        self, nomic_api: NomicAPI, mock_client: MagicMock
    ):
        """Test risk_register_async() with custom limit."""
        mock_client._get_async = AsyncMock(return_value={"risks": [], "total": 0})

        await nomic_api.risk_register_async(limit=5)

        mock_client._get_async.assert_called_once_with(
            "/api/v1/nomic/risk-register", params={"limit": 5}
        )


# ============================================================================
# NomicAPI.witness_status() Tests
# ============================================================================


class TestNomicAPIWitnessStatus:
    """Tests for NomicAPI.witness_status() method."""

    def test_witness_status(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test witness_status() returns current witness patrol status."""
        mock_client._get.return_value = {
            "active": True,
            "patrol_count": 12,
            "last_patrol_at": "2026-02-12T10:00:00Z",
        }

        result = nomic_api.witness_status()

        assert result["active"] is True
        assert result["patrol_count"] == 12
        mock_client._get.assert_called_once_with("/api/v1/nomic/witness/status")


class TestNomicAPIWitnessStatusAsync:
    """Tests for NomicAPI.witness_status_async() method."""

    @pytest.mark.asyncio
    async def test_witness_status_async(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test witness_status_async() returns witness patrol status."""
        mock_client._get_async = AsyncMock(
            return_value={"active": False, "patrol_count": 0}
        )

        result = await nomic_api.witness_status_async()

        assert result["active"] is False
        mock_client._get_async.assert_called_once_with("/api/v1/nomic/witness/status")


# ============================================================================
# NomicAPI.mayor_current() Tests
# ============================================================================


class TestNomicAPIMayorCurrent:
    """Tests for NomicAPI.mayor_current() method."""

    def test_mayor_current(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test mayor_current() returns current mayor information."""
        mock_client._get.return_value = {
            "agent_id": "claude-opus",
            "term_start": "2026-02-01T00:00:00Z",
            "proposals_approved": 15,
        }

        result = nomic_api.mayor_current()

        assert result["agent_id"] == "claude-opus"
        assert result["proposals_approved"] == 15
        mock_client._get.assert_called_once_with("/api/v1/nomic/mayor/current")


class TestNomicAPIMayorCurrentAsync:
    """Tests for NomicAPI.mayor_current_async() method."""

    @pytest.mark.asyncio
    async def test_mayor_current_async(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test mayor_current_async() returns current mayor information."""
        mock_client._get_async = AsyncMock(
            return_value={"agent_id": "gpt-4", "term_start": "2026-01-15T00:00:00Z"}
        )

        result = await nomic_api.mayor_current_async()

        assert result["agent_id"] == "gpt-4"
        mock_client._get_async.assert_called_once_with("/api/v1/nomic/mayor/current")


# ============================================================================
# NomicAPI.proposals() Tests
# ============================================================================


class TestNomicAPIProposals:
    """Tests for NomicAPI.proposals() method."""

    def test_proposals(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test proposals() returns pending nomic proposals."""
        mock_client._get.return_value = {
            "proposals": [
                {"id": "prop-1", "title": "Add retry logic", "status": "pending"},
                {"id": "prop-2", "title": "Improve logging", "status": "pending"},
            ],
            "total": 2,
        }

        result = nomic_api.proposals()

        assert len(result["proposals"]) == 2
        assert result["proposals"][0]["title"] == "Add retry logic"
        mock_client._get.assert_called_once_with("/api/v1/nomic/proposals")


class TestNomicAPIProposalsAsync:
    """Tests for NomicAPI.proposals_async() method."""

    @pytest.mark.asyncio
    async def test_proposals_async(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test proposals_async() returns pending proposals."""
        mock_client._get_async = AsyncMock(
            return_value={"proposals": [], "total": 0}
        )

        result = await nomic_api.proposals_async()

        assert result["total"] == 0
        mock_client._get_async.assert_called_once_with("/api/v1/nomic/proposals")


# ============================================================================
# NomicAPI.modes() Tests
# ============================================================================


class TestNomicAPIModes:
    """Tests for NomicAPI.modes() method."""

    def test_modes(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test modes() returns available operational modes."""
        mock_client._get.return_value = {
            "modes": ["architect", "coder", "reviewer", "tester"],
            "active": "coder",
        }

        result = nomic_api.modes()

        assert "coder" in result["modes"]
        assert result["active"] == "coder"
        mock_client._get.assert_called_once_with("/api/v1/modes")


class TestNomicAPIModesAsync:
    """Tests for NomicAPI.modes_async() method."""

    @pytest.mark.asyncio
    async def test_modes_async(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test modes_async() returns available operational modes."""
        mock_client._get_async = AsyncMock(
            return_value={"modes": ["architect"], "active": "architect"}
        )

        result = await nomic_api.modes_async()

        assert result["active"] == "architect"
        mock_client._get_async.assert_called_once_with("/api/v1/modes")


# ============================================================================
# NomicAPI.start() Tests
# ============================================================================


class TestNomicAPIStart:
    """Tests for NomicAPI.start() method."""

    def test_start_no_config(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test start() with no configuration."""
        mock_client._post.return_value = {"status": "started", "cycle": 1}

        result = nomic_api.start()

        assert result["status"] == "started"
        mock_client._post.assert_called_once_with(
            "/api/v1/nomic/control/start", data={}
        )

    def test_start_with_config(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test start() with custom configuration."""
        config = {"cycles": 5, "require_approval": True}
        mock_client._post.return_value = {
            "status": "started",
            "cycle": 1,
            "config": config,
        }

        result = nomic_api.start(config=config)

        assert result["status"] == "started"
        assert result["config"]["cycles"] == 5
        mock_client._post.assert_called_once_with(
            "/api/v1/nomic/control/start", data=config
        )

    def test_start_none_config_sends_empty_dict(
        self, nomic_api: NomicAPI, mock_client: MagicMock
    ):
        """Test start(config=None) sends empty dict as data."""
        mock_client._post.return_value = {"status": "started"}

        nomic_api.start(config=None)

        mock_client._post.assert_called_once_with(
            "/api/v1/nomic/control/start", data={}
        )


class TestNomicAPIStartAsync:
    """Tests for NomicAPI.start_async() method."""

    @pytest.mark.asyncio
    async def test_start_async_no_config(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test start_async() with no configuration."""
        mock_client._post_async = AsyncMock(
            return_value={"status": "started", "cycle": 1}
        )

        result = await nomic_api.start_async()

        assert result["status"] == "started"
        mock_client._post_async.assert_called_once_with(
            "/api/v1/nomic/control/start", data={}
        )

    @pytest.mark.asyncio
    async def test_start_async_with_config(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test start_async() with configuration."""
        config = {"cycles": 10}
        mock_client._post_async = AsyncMock(
            return_value={"status": "started", "config": config}
        )

        result = await nomic_api.start_async(config=config)

        assert result["config"]["cycles"] == 10
        mock_client._post_async.assert_called_once_with(
            "/api/v1/nomic/control/start", data=config
        )


# ============================================================================
# NomicAPI.stop() Tests
# ============================================================================


class TestNomicAPIStop:
    """Tests for NomicAPI.stop() method."""

    def test_stop_no_payload(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test stop() with no payload."""
        mock_client._post.return_value = {"status": "stopped", "cycle": 3}

        result = nomic_api.stop()

        assert result["status"] == "stopped"
        mock_client._post.assert_called_once_with(
            "/api/v1/nomic/control/stop", data={}
        )

    def test_stop_with_payload(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test stop() with payload (e.g., reason)."""
        payload = {"reason": "maintenance", "force": True}
        mock_client._post.return_value = {"status": "stopped", "reason": "maintenance"}

        result = nomic_api.stop(payload=payload)

        assert result["status"] == "stopped"
        mock_client._post.assert_called_once_with(
            "/api/v1/nomic/control/stop", data=payload
        )

    def test_stop_none_payload_sends_empty_dict(
        self, nomic_api: NomicAPI, mock_client: MagicMock
    ):
        """Test stop(payload=None) sends empty dict."""
        mock_client._post.return_value = {"status": "stopped"}

        nomic_api.stop(payload=None)

        mock_client._post.assert_called_once_with(
            "/api/v1/nomic/control/stop", data={}
        )


class TestNomicAPIStopAsync:
    """Tests for NomicAPI.stop_async() method."""

    @pytest.mark.asyncio
    async def test_stop_async(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test stop_async() stops the nomic loop."""
        mock_client._post_async = AsyncMock(
            return_value={"status": "stopped"}
        )

        result = await nomic_api.stop_async()

        assert result["status"] == "stopped"
        mock_client._post_async.assert_called_once_with(
            "/api/v1/nomic/control/stop", data={}
        )

    @pytest.mark.asyncio
    async def test_stop_async_with_payload(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test stop_async() with payload."""
        payload = {"reason": "scheduled"}
        mock_client._post_async = AsyncMock(return_value={"status": "stopped"})

        await nomic_api.stop_async(payload=payload)

        mock_client._post_async.assert_called_once_with(
            "/api/v1/nomic/control/stop", data=payload
        )


# ============================================================================
# NomicAPI.pause() Tests
# ============================================================================


class TestNomicAPIPause:
    """Tests for NomicAPI.pause() method."""

    def test_pause(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test pause() pauses the nomic loop."""
        mock_client._post.return_value = {"status": "paused", "phase": "debate"}

        result = nomic_api.pause()

        assert result["status"] == "paused"
        assert result["phase"] == "debate"
        mock_client._post.assert_called_once_with(
            "/api/v1/nomic/control/pause", {}
        )


class TestNomicAPIPauseAsync:
    """Tests for NomicAPI.pause_async() method."""

    @pytest.mark.asyncio
    async def test_pause_async(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test pause_async() pauses the nomic loop."""
        mock_client._post_async = AsyncMock(
            return_value={"status": "paused", "phase": "implement"}
        )

        result = await nomic_api.pause_async()

        assert result["status"] == "paused"
        mock_client._post_async.assert_called_once_with(
            "/api/v1/nomic/control/pause", {}
        )


# ============================================================================
# NomicAPI.resume() Tests
# ============================================================================


class TestNomicAPIResume:
    """Tests for NomicAPI.resume() method."""

    def test_resume(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test resume() resumes the nomic loop."""
        mock_client._post.return_value = {"status": "running", "phase": "verify"}

        result = nomic_api.resume()

        assert result["status"] == "running"
        mock_client._post.assert_called_once_with(
            "/api/v1/nomic/control/resume", {}
        )


class TestNomicAPIResumeAsync:
    """Tests for NomicAPI.resume_async() method."""

    @pytest.mark.asyncio
    async def test_resume_async(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test resume_async() resumes the nomic loop."""
        mock_client._post_async = AsyncMock(
            return_value={"status": "running", "phase": "design"}
        )

        result = await nomic_api.resume_async()

        assert result["status"] == "running"
        mock_client._post_async.assert_called_once_with(
            "/api/v1/nomic/control/resume", {}
        )


# ============================================================================
# NomicAPI.skip_phase() Tests
# ============================================================================


class TestNomicAPISkipPhase:
    """Tests for NomicAPI.skip_phase() method."""

    def test_skip_phase_no_payload(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test skip_phase() with no payload."""
        mock_client._post.return_value = {
            "skipped": "debate",
            "current_phase": "design",
        }

        result = nomic_api.skip_phase()

        assert result["skipped"] == "debate"
        assert result["current_phase"] == "design"
        mock_client._post.assert_called_once_with(
            "/api/v1/nomic/control/skip-phase", data={}
        )

    def test_skip_phase_with_payload(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test skip_phase() with payload (e.g., reason for skipping)."""
        payload = {"reason": "phase already completed manually"}
        mock_client._post.return_value = {"skipped": "implement", "current_phase": "verify"}

        result = nomic_api.skip_phase(payload=payload)

        assert result["skipped"] == "implement"
        mock_client._post.assert_called_once_with(
            "/api/v1/nomic/control/skip-phase", data=payload
        )

    def test_skip_phase_none_payload_sends_empty_dict(
        self, nomic_api: NomicAPI, mock_client: MagicMock
    ):
        """Test skip_phase(payload=None) sends empty dict."""
        mock_client._post.return_value = {"skipped": "verify"}

        nomic_api.skip_phase(payload=None)

        mock_client._post.assert_called_once_with(
            "/api/v1/nomic/control/skip-phase", data={}
        )


class TestNomicAPISkipPhaseAsync:
    """Tests for NomicAPI.skip_phase_async() method."""

    @pytest.mark.asyncio
    async def test_skip_phase_async(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test skip_phase_async() skips the current phase."""
        mock_client._post_async = AsyncMock(
            return_value={"skipped": "context", "current_phase": "debate"}
        )

        result = await nomic_api.skip_phase_async()

        assert result["skipped"] == "context"
        mock_client._post_async.assert_called_once_with(
            "/api/v1/nomic/control/skip-phase", data={}
        )

    @pytest.mark.asyncio
    async def test_skip_phase_async_with_payload(
        self, nomic_api: NomicAPI, mock_client: MagicMock
    ):
        """Test skip_phase_async() with payload."""
        payload = {"reason": "timeout"}
        mock_client._post_async = AsyncMock(
            return_value={"skipped": "design"}
        )

        await nomic_api.skip_phase_async(payload=payload)

        mock_client._post_async.assert_called_once_with(
            "/api/v1/nomic/control/skip-phase", data=payload
        )


# ============================================================================
# NomicAPI.approve_proposal() Tests
# ============================================================================


class TestNomicAPIApproveProposal:
    """Tests for NomicAPI.approve_proposal() method."""

    def test_approve_proposal_with_id(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test approve_proposal() with a specific proposal_id."""
        mock_client._post.return_value = {
            "approved": True,
            "proposal_id": "prop-42",
        }

        result = nomic_api.approve_proposal(proposal_id="prop-42")

        assert result["approved"] is True
        assert result["proposal_id"] == "prop-42"
        mock_client._post.assert_called_once_with(
            "/api/v1/nomic/proposals/approve",
            data={"proposal_id": "prop-42"},
        )

    def test_approve_proposal_no_id(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test approve_proposal() without a proposal_id sends empty payload."""
        mock_client._post.return_value = {"approved": True}

        result = nomic_api.approve_proposal()

        assert result["approved"] is True
        mock_client._post.assert_called_once_with(
            "/api/v1/nomic/proposals/approve", data={}
        )

    def test_approve_proposal_none_id(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test approve_proposal(proposal_id=None) sends empty payload."""
        mock_client._post.return_value = {"approved": True}

        nomic_api.approve_proposal(proposal_id=None)

        mock_client._post.assert_called_once_with(
            "/api/v1/nomic/proposals/approve", data={}
        )


class TestNomicAPIApproveProposalAsync:
    """Tests for NomicAPI.approve_proposal_async() method."""

    @pytest.mark.asyncio
    async def test_approve_proposal_async_with_id(
        self, nomic_api: NomicAPI, mock_client: MagicMock
    ):
        """Test approve_proposal_async() with a specific proposal_id."""
        mock_client._post_async = AsyncMock(
            return_value={"approved": True, "proposal_id": "prop-async"}
        )

        result = await nomic_api.approve_proposal_async(proposal_id="prop-async")

        assert result["approved"] is True
        mock_client._post_async.assert_called_once_with(
            "/api/v1/nomic/proposals/approve",
            data={"proposal_id": "prop-async"},
        )

    @pytest.mark.asyncio
    async def test_approve_proposal_async_no_id(
        self, nomic_api: NomicAPI, mock_client: MagicMock
    ):
        """Test approve_proposal_async() without a proposal_id."""
        mock_client._post_async = AsyncMock(return_value={"approved": True})

        await nomic_api.approve_proposal_async()

        mock_client._post_async.assert_called_once_with(
            "/api/v1/nomic/proposals/approve", data={}
        )


# ============================================================================
# NomicAPI.reject_proposal() Tests
# ============================================================================


class TestNomicAPIRejectProposal:
    """Tests for NomicAPI.reject_proposal() method."""

    def test_reject_proposal_with_id_and_reason(
        self, nomic_api: NomicAPI, mock_client: MagicMock
    ):
        """Test reject_proposal() with both proposal_id and reason."""
        mock_client._post.return_value = {
            "rejected": True,
            "proposal_id": "prop-99",
        }

        result = nomic_api.reject_proposal(
            proposal_id="prop-99", reason="Too risky"
        )

        assert result["rejected"] is True
        mock_client._post.assert_called_once_with(
            "/api/v1/nomic/proposals/reject",
            data={"proposal_id": "prop-99", "reason": "Too risky"},
        )

    def test_reject_proposal_with_id_only(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test reject_proposal() with only proposal_id, no reason."""
        mock_client._post.return_value = {"rejected": True}

        nomic_api.reject_proposal(proposal_id="prop-50")

        mock_client._post.assert_called_once_with(
            "/api/v1/nomic/proposals/reject",
            data={"proposal_id": "prop-50"},
        )

    def test_reject_proposal_with_reason_only(
        self, nomic_api: NomicAPI, mock_client: MagicMock
    ):
        """Test reject_proposal() with only reason, no proposal_id."""
        mock_client._post.return_value = {"rejected": True}

        nomic_api.reject_proposal(reason="Does not align with goals")

        mock_client._post.assert_called_once_with(
            "/api/v1/nomic/proposals/reject",
            data={"reason": "Does not align with goals"},
        )

    def test_reject_proposal_no_args(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test reject_proposal() with no arguments sends empty payload."""
        mock_client._post.return_value = {"rejected": True}

        nomic_api.reject_proposal()

        mock_client._post.assert_called_once_with(
            "/api/v1/nomic/proposals/reject", data={}
        )


class TestNomicAPIRejectProposalAsync:
    """Tests for NomicAPI.reject_proposal_async() method."""

    @pytest.mark.asyncio
    async def test_reject_proposal_async_full(
        self, nomic_api: NomicAPI, mock_client: MagicMock
    ):
        """Test reject_proposal_async() with proposal_id and reason."""
        mock_client._post_async = AsyncMock(
            return_value={"rejected": True, "proposal_id": "prop-async-rej"}
        )

        result = await nomic_api.reject_proposal_async(
            proposal_id="prop-async-rej", reason="Insufficient testing"
        )

        assert result["rejected"] is True
        mock_client._post_async.assert_called_once_with(
            "/api/v1/nomic/proposals/reject",
            data={"proposal_id": "prop-async-rej", "reason": "Insufficient testing"},
        )

    @pytest.mark.asyncio
    async def test_reject_proposal_async_no_args(
        self, nomic_api: NomicAPI, mock_client: MagicMock
    ):
        """Test reject_proposal_async() with no arguments."""
        mock_client._post_async = AsyncMock(return_value={"rejected": True})

        await nomic_api.reject_proposal_async()

        mock_client._post_async.assert_called_once_with(
            "/api/v1/nomic/proposals/reject", data={}
        )


# ============================================================================
# Integration-like Tests
# ============================================================================


class TestNomicAPIIntegration:
    """Integration-like tests for NomicAPI."""

    def test_full_lifecycle(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test full nomic loop lifecycle: start -> state -> pause -> resume -> stop."""
        # Start the loop
        mock_client._post.return_value = {"status": "started", "cycle": 1}
        start_result = nomic_api.start(config={"cycles": 3})
        assert start_result["status"] == "started"

        # Check state
        mock_client._get.return_value = {
            "phase": "debate",
            "cycle": 1,
            "running": True,
        }
        state_result = nomic_api.state()
        assert state_result["running"] is True

        # Pause
        mock_client._post.return_value = {"status": "paused", "phase": "debate"}
        pause_result = nomic_api.pause()
        assert pause_result["status"] == "paused"

        # Resume
        mock_client._post.return_value = {"status": "running", "phase": "debate"}
        resume_result = nomic_api.resume()
        assert resume_result["status"] == "running"

        # Stop
        mock_client._post.return_value = {"status": "stopped", "cycle": 2}
        stop_result = nomic_api.stop()
        assert stop_result["status"] == "stopped"

    def test_proposal_workflow(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test proposal workflow: list -> approve/reject."""
        # List proposals
        mock_client._get.return_value = {
            "proposals": [
                {"id": "prop-1", "title": "Add caching", "status": "pending"},
                {"id": "prop-2", "title": "Remove dead code", "status": "pending"},
            ],
            "total": 2,
        }
        proposals = nomic_api.proposals()
        assert proposals["total"] == 2

        # Approve first proposal
        mock_client._post.return_value = {"approved": True, "proposal_id": "prop-1"}
        approve_result = nomic_api.approve_proposal(proposal_id="prop-1")
        assert approve_result["approved"] is True

        # Reject second proposal
        mock_client._post.return_value = {"rejected": True, "proposal_id": "prop-2"}
        reject_result = nomic_api.reject_proposal(
            proposal_id="prop-2", reason="Not a priority"
        )
        assert reject_result["rejected"] is True

    def test_monitoring_workflow(self, nomic_api: NomicAPI, mock_client: MagicMock):
        """Test monitoring workflow: health -> metrics -> log -> risk_register."""
        # Check health
        mock_client._get.return_value = {"healthy": True, "stalled": False}
        health = nomic_api.health()
        assert health["healthy"] is True

        # Get metrics
        mock_client._get.return_value = {"total_cycles": 50, "successful_cycles": 48}
        metrics = nomic_api.metrics()
        assert metrics["total_cycles"] == 50

        # Get logs
        mock_client._get.return_value = {
            "lines": ["[INFO] Cycle 50 completed"],
            "total": 1,
        }
        log = nomic_api.log(lines=10)
        assert log["total"] == 1

        # Check risk register
        mock_client._get.return_value = {"risks": [], "total": 0}
        risks = nomic_api.risk_register(limit=20)
        assert risks["total"] == 0
