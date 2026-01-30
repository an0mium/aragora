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
