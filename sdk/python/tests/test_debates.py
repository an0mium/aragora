"""Tests for Debates namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient


class TestDebatesCreate:
    """Tests for debate creation."""

    def test_create_debate_with_task(self) -> None:
        """Create a debate with just a task."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "debate_id": "deb_123",
                "task": "Should we adopt microservices?",
                "status": "pending",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.create(task="Should we adopt microservices?")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/debates",
                json={"task": "Should we adopt microservices?"},
            )
            assert result["debate_id"] == "deb_123"
            client.close()

    def test_create_debate_with_agents(self) -> None:
        """Create a debate with specific agents."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"debate_id": "deb_123"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.debates.create(task="Debate topic", agents=["claude", "gpt-4", "gemini"])

            call_args = mock_request.call_args
            assert call_args[1]["json"]["agents"] == ["claude", "gpt-4", "gemini"]
            client.close()

    def test_create_debate_with_protocol(self) -> None:
        """Create a debate with protocol configuration."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"debate_id": "deb_123"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            protocol = {"rounds": 5, "consensus": "unanimous"}
            client.debates.create(task="Debate topic", protocol=protocol)

            call_args = mock_request.call_args
            assert call_args[1]["json"]["protocol"] == protocol
            client.close()

    def test_create_debate_with_extra_kwargs(self) -> None:
        """Create a debate with additional keyword arguments."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"debate_id": "deb_123"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.debates.create(task="Debate topic", timeout=300, workspace_id="ws_123")

            call_args = mock_request.call_args
            assert call_args[1]["json"]["timeout"] == 300
            assert call_args[1]["json"]["workspace_id"] == "ws_123"
            client.close()


class TestDebatesGet:
    """Tests for getting debate details."""

    def test_get_debate_by_id(self) -> None:
        """Get a debate by its ID."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "debate_id": "deb_123",
                "task": "Test topic",
                "status": "completed",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.get("deb_123")

            mock_request.assert_called_once_with("GET", "/api/v1/debates/deb_123")
            assert result["debate_id"] == "deb_123"
            client.close()

    def test_get_debate_by_slug(self) -> None:
        """Get a debate by its slug."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "debate_id": "deb_123",
                "slug": "microservices-debate",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.debates.get_by_slug("microservices-debate")

            mock_request.assert_called_once_with("GET", "/api/v1/debates/slug/microservices-debate")
            client.close()


class TestDebatesList:
    """Tests for listing debates."""

    def test_list_debates_default_pagination(self) -> None:
        """List debates with default pagination."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "debates": [],
                "total": 0,
                "limit": 20,
                "offset": 0,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.debates.list()

            mock_request.assert_called_once_with(
                "GET", "/api/v1/debates", params={"limit": 20, "offset": 0}
            )
            client.close()

    def test_list_debates_custom_pagination(self) -> None:
        """List debates with custom pagination."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"debates": [], "total": 100}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.debates.list(limit=50, offset=25)

            mock_request.assert_called_once_with(
                "GET", "/api/v1/debates", params={"limit": 50, "offset": 25}
            )
            client.close()

    def test_list_debates_with_status_filter(self) -> None:
        """List debates filtered by status."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"debates": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.debates.list(status="completed")

            call_args = mock_request.call_args
            assert call_args[1]["params"]["status"] == "completed"
            client.close()


class TestDebatesConsensus:
    """Tests for consensus operations."""

    def test_get_consensus(self) -> None:
        """Get consensus information for a debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "reached": True,
                "confidence": 0.95,
                "position": "Adopt microservices for new projects",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.get_consensus("deb_123")

            mock_request.assert_called_once_with("GET", "/api/v1/debates/deb_123/consensus")
            assert result["reached"] is True
            assert result["confidence"] == 0.95
            client.close()


class TestDebatesMessages:
    """Tests for debate message operations."""

    def test_get_messages(self) -> None:
        """Get messages from a debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "messages": [
                    {"id": "msg_1", "content": "First message", "role": "agent"},
                ]
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.get_messages("deb_123")

            mock_request.assert_called_once_with("GET", "/api/v1/debates/deb_123/messages")
            assert len(result["messages"]) == 1
            client.close()

    def test_add_message(self) -> None:
        """Add a message to a debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "id": "msg_new",
                "content": "User input",
                "role": "user",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.debates.add_message("deb_123", content="User input", role="user")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/debates/deb_123/messages",
                json={"content": "User input", "role": "user"},
            )
            client.close()


class TestDebatesExport:
    """Tests for debate export."""

    def test_export_debate_json(self) -> None:
        """Export debate as JSON."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"data": "exported_content"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.debates.get_export("deb_123", format="json")

            mock_request.assert_called_once_with(
                "GET", "/api/v1/debates/deb_123/export", params={"format": "json"}
            )
            client.close()

    def test_export_debate_pdf(self) -> None:
        """Export debate as PDF."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"url": "https://cdn.aragora.ai/exports/..."}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.debates.get_export("deb_123", format="pdf")

            call_args = mock_request.call_args
            assert call_args[1]["params"]["format"] == "pdf"
            client.close()


class TestDebatesCancel:
    """Tests for cancelling debates."""

    def test_cancel_debate(self) -> None:
        """Cancel a debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"cancelled": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.cancel("deb_123")

            mock_request.assert_called_once_with("POST", "/api/v1/debates/deb_123/cancel")
            assert result["cancelled"] is True
            client.close()


class TestDebatesExplainability:
    """Tests for explainability features."""

    def test_get_explainability(self) -> None:
        """Get explainability data for a debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"factors": [], "narrative": ""}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.debates.get_explainability("deb_123")

            mock_request.assert_called_once_with("GET", "/api/v1/debates/deb_123/explainability")
            client.close()

    def test_get_explainability_factors(self) -> None:
        """Get factor decomposition for a debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"factors": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.debates.get_explainability_factors("deb_123")

            mock_request.assert_called_once_with(
                "GET", "/api/v1/debates/deb_123/explainability/factors"
            )
            client.close()

    def test_create_counterfactual(self) -> None:
        """Create a counterfactual scenario."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"outcome": "different"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            changes = {"remove_agent": "claude"}
            client.debates.create_counterfactual("deb_123", changes)

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/debates/deb_123/explainability/counterfactual",
                json=changes,
            )
            client.close()


class TestAsyncDebates:
    """Tests for async debates API."""

    @pytest.mark.asyncio
    async def test_async_create_debate(self) -> None:
        """Create a debate asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"debate_id": "deb_123"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.debates.create(task="Async debate")

                mock_request.assert_called_once_with(
                    "POST", "/api/v1/debates", json={"task": "Async debate"}
                )
                assert result["debate_id"] == "deb_123"

    @pytest.mark.asyncio
    async def test_async_get_debate(self) -> None:
        """Get a debate asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"debate_id": "deb_123"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.debates.get("deb_123")

                mock_request.assert_called_once_with("GET", "/api/v1/debates/deb_123")

    @pytest.mark.asyncio
    async def test_async_list_debates(self) -> None:
        """List debates asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"debates": [], "total": 0}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.debates.list(limit=10, offset=5)

                mock_request.assert_called_once_with(
                    "GET", "/api/v1/debates", params={"limit": 10, "offset": 5}
                )

    @pytest.mark.asyncio
    async def test_async_get_consensus(self) -> None:
        """Get consensus asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"reached": True}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.debates.get_consensus("deb_123")

                mock_request.assert_called_once_with("GET", "/api/v1/debates/deb_123/consensus")
                assert result["reached"] is True


class TestDebatesAdvancedFeatures:
    """Tests for advanced debate features."""

    def test_capability_probe(self) -> None:
        """Run a capability probe debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"debate_id": "deb_probe"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.debates.capability_probe(task="Test capability", agents=["claude"])

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/debates/capability-probe",
                json={"task": "Test capability", "agents": ["claude"]},
            )
            client.close()

    def test_deep_audit(self) -> None:
        """Run a deep audit debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"debate_id": "deb_audit"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.debates.deep_audit(task="Audit this decision")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/debates/deep-audit",
                json={"task": "Audit this decision"},
            )
            client.close()

    def test_fork_debate(self) -> None:
        """Fork a debate with changes."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"debate_id": "deb_forked"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.debates.fork("deb_123", changes={"agents": ["gpt-4"]})

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/debates/deb_123/fork",
                json={"agents": ["gpt-4"]},
            )
            client.close()

    def test_broadcast_debate(self) -> None:
        """Broadcast debate to channels."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"broadcasted": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.debates.broadcast("deb_123", channels=["slack", "email"])

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/debates/deb_123/broadcast",
                json={"channels": ["slack", "email"]},
            )
            client.close()


class TestDebatesLifecycle:
    """Tests for debate lifecycle methods."""

    def test_start_debate(self) -> None:
        """Start a debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"success": True, "status": "running"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.start("deb_123")

            mock_request.assert_called_once_with("POST", "/api/v1/debates/deb_123/start")
            assert result["success"] is True
            client.close()

    def test_stop_debate(self) -> None:
        """Stop a running debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"success": True, "status": "stopped"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.stop("deb_123")

            mock_request.assert_called_once_with("POST", "/api/v1/debates/deb_123/stop")
            assert result["success"] is True
            client.close()

    def test_pause_debate(self) -> None:
        """Pause a running debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"success": True, "status": "paused"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.pause("deb_123")

            mock_request.assert_called_once_with("POST", "/api/v1/debates/deb_123/pause")
            assert result["success"] is True
            client.close()

    def test_resume_debate(self) -> None:
        """Resume a paused debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"success": True, "status": "running"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.resume("deb_123")

            mock_request.assert_called_once_with("POST", "/api/v1/debates/deb_123/resume")
            assert result["success"] is True
            client.close()

    def test_delete_debate(self) -> None:
        """Delete a debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"success": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.delete("deb_123")

            mock_request.assert_called_once_with("DELETE", "/api/v1/debates/deb_123")
            assert result["success"] is True
            client.close()

    def test_clone_debate(self) -> None:
        """Clone a debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"debate_id": "deb_cloned"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.clone("deb_123", preserve_agents=True)

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/debates/deb_123/clone",
                json={"preserveAgents": True, "preserveContext": False},
            )
            assert result["debate_id"] == "deb_cloned"
            client.close()

    def test_archive_debate(self) -> None:
        """Archive a debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"success": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.archive("deb_123")

            mock_request.assert_called_once_with("POST", "/api/v1/debates/deb_123/archive")
            assert result["success"] is True
            client.close()


class TestDebatesAnalysis:
    """Tests for debate analysis methods."""

    def test_get_rhetorical(self) -> None:
        """Get rhetorical pattern observations."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"observations": [], "summary": {}}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.debates.get_rhetorical("deb_123")

            mock_request.assert_called_once_with("GET", "/api/v1/debates/deb_123/rhetorical")
            client.close()

    def test_get_trickster(self) -> None:
        """Get trickster hollow consensus detection."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "hollow_consensus_detected": False,
                "confidence": 0.85,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.get_trickster("deb_123")

            mock_request.assert_called_once_with("GET", "/api/v1/debates/deb_123/trickster")
            assert result["hollow_consensus_detected"] is False
            client.close()

    def test_get_summary(self) -> None:
        """Get a debate summary."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"verdict": "Adopt microservices", "confidence": 0.9}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.get_summary("deb_123")

            mock_request.assert_called_once_with("GET", "/api/v1/debates/deb_123/summary")
            assert result["verdict"] == "Adopt microservices"
            client.close()

    def test_verify_claim(self) -> None:
        """Verify a specific claim."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"verified": True, "confidence": 0.95}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.verify_claim(
                "deb_123", "claim_456", evidence="Supporting evidence"
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/debates/deb_123/verify",
                json={"claim_id": "claim_456", "evidence": "Supporting evidence"},
            )
            assert result["verified"] is True
            client.close()


class TestDebatesRoundsAgentsVotes:
    """Tests for rounds, agents, and votes methods."""

    def test_get_rounds(self) -> None:
        """Get rounds from a debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"rounds": [{"number": 1, "proposals": []}]}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.get_rounds("deb_123")

            mock_request.assert_called_once_with("GET", "/api/v1/debates/deb_123/rounds")
            assert len(result["rounds"]) == 1
            client.close()

    def test_get_agents(self) -> None:
        """Get agents participating in a debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"agents": [{"name": "claude", "role": "proposer"}]}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.get_agents("deb_123")

            mock_request.assert_called_once_with("GET", "/api/v1/debates/deb_123/agents")
            assert result["agents"][0]["name"] == "claude"
            client.close()

    def test_get_votes(self) -> None:
        """Get votes from a debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"votes": [{"agent": "claude", "position": "pro"}]}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.get_votes("deb_123")

            mock_request.assert_called_once_with("GET", "/api/v1/debates/deb_123/votes")
            assert result["votes"][0]["agent"] == "claude"
            client.close()

    def test_add_user_input(self) -> None:
        """Add user input to a debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"input_id": "inp_123", "success": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.add_user_input("deb_123", "Consider scalability", "suggestion")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/debates/deb_123/user-input",
                json={"input": "Consider scalability", "type": "suggestion"},
            )
            assert result["success"] is True
            client.close()

    def test_get_timeline(self) -> None:
        """Get the timeline of events."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"timeline": [{"type": "round_start"}]}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.get_timeline("deb_123")

            mock_request.assert_called_once_with("GET", "/api/v1/debates/deb_123/timeline")
            assert result["timeline"][0]["type"] == "round_start"
            client.close()

    def test_add_evidence(self) -> None:
        """Add evidence to a debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"evidence_id": "ev_123", "success": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.add_evidence(
                "deb_123", "Studies show...", source="research-paper"
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/debates/deb_123/evidence",
                json={"evidence": "Studies show...", "source": "research-paper"},
            )
            assert result["evidence_id"] == "ev_123"
            client.close()


class TestDebatesBatchOperations:
    """Tests for batch operations."""

    def test_submit_batch(self) -> None:
        """Submit multiple debates for batch processing."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"batch_id": "batch_123", "total_jobs": 2}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.submit_batch(
                [
                    {"task": "Debate 1"},
                    {"task": "Debate 2"},
                ]
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/debates/batch",
                json={"requests": [{"task": "Debate 1"}, {"task": "Debate 2"}]},
            )
            assert result["total_jobs"] == 2
            client.close()

    def test_get_batch_status(self) -> None:
        """Get the status of a batch job."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"batch_id": "batch_123", "status": "running"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.get_batch_status("batch_123")

            mock_request.assert_called_once_with("GET", "/api/v1/debates/batch/batch_123/status")
            assert result["status"] == "running"
            client.close()

    def test_get_queue_status(self) -> None:
        """Get the current queue status."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"pending_count": 5, "running_count": 2}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.get_queue_status()

            mock_request.assert_called_once_with("GET", "/api/v1/debates/queue/status")
            assert result["pending_count"] == 5
            client.close()


class TestDebatesGraph:
    """Tests for graph and visualization methods."""

    def test_get_graph(self) -> None:
        """Get the argument graph for a debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"nodes": [], "edges": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.get_graph("deb_123")

            mock_request.assert_called_once_with("GET", "/api/v1/debates/graph/deb_123")
            assert "nodes" in result
            client.close()

    def test_get_graph_branches(self) -> None:
        """Get branches in the argument graph."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"branches": [{"branch_id": "br_1"}]}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.get_graph_branches("deb_123")

            mock_request.assert_called_once_with("GET", "/api/v1/debates/graph/deb_123/branches")
            assert len(result["branches"]) == 1
            client.close()

    def test_get_matrix_comparison(self) -> None:
        """Get matrix comparison for a multi-scenario debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"scenarios": [], "comparison_matrix": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.get_matrix_comparison("deb_123")

            mock_request.assert_called_once_with("GET", "/api/v1/debates/matrix/deb_123")
            assert "scenarios" in result
            client.close()


class TestDebatesSearch:
    """Tests for search functionality."""

    def test_search_debates(self) -> None:
        """Search across all debates."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"debates": [], "total": 0}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.debates.search("microservices", limit=10, status="completed")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/search",
                params={
                    "query": "microservices",
                    "limit": 10,
                    "offset": 0,
                    "status": "completed",
                },
            )
            assert "debates" in result
            client.close()
