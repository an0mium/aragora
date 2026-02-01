"""Tests for Orchestration namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient


class TestOrchestrationDeliberate:
    """Tests for async deliberation submission."""

    def test_deliberate_minimal(self) -> None:
        """Submit deliberation with just a question."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "request_id": "req_123",
                "status": "pending",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.orchestration.deliberate(question="Should we migrate to Kubernetes?")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/orchestration/deliberate",
                json={"question": "Should we migrate to Kubernetes?"},
            )
            assert result["request_id"] == "req_123"
            assert result["status"] == "pending"
            client.close()

    def test_deliberate_with_knowledge_sources(self) -> None:
        """Submit deliberation with knowledge sources."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_456"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.orchestration.deliberate(
                question="What's our best cloud strategy?",
                knowledge_sources=["confluence:12345", "slack:C123456"],
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["knowledge_sources"] == [
                "confluence:12345",
                "slack:C123456",
            ]
            client.close()

    def test_deliberate_with_knowledge_sources_as_dicts(self) -> None:
        """Submit deliberation with knowledge sources as dicts."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_457"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            sources = [
                {"type": "confluence", "id": "12345", "filters": {"space": "engineering"}},
                {"type": "slack", "channel": "C123456"},
            ]
            client.orchestration.deliberate(
                question="What's our best cloud strategy?",
                knowledge_sources=sources,
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["knowledge_sources"] == sources
            client.close()

    def test_deliberate_with_workspaces(self) -> None:
        """Submit deliberation with workspace IDs."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_458"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.orchestration.deliberate(
                question="Test question",
                workspaces=["ws_abc", "ws_def"],
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["workspaces"] == ["ws_abc", "ws_def"]
            client.close()

    def test_deliberate_with_team_strategy(self) -> None:
        """Submit deliberation with non-default team strategy."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_459"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.orchestration.deliberate(
                question="Test question",
                team_strategy="diverse",
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["team_strategy"] == "diverse"
            client.close()

    def test_deliberate_with_specified_agents(self) -> None:
        """Submit deliberation with explicit agent list."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_460"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.orchestration.deliberate(
                question="Test question",
                agents=["claude", "gpt-4", "gemini"],
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["agents"] == ["claude", "gpt-4", "gemini"]
            client.close()

    def test_deliberate_with_output_channels(self) -> None:
        """Submit deliberation with output channel routing."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_461"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.orchestration.deliberate(
                question="Test question",
                output_channels=["slack:C789", "email:team@example.com"],
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["output_channels"] == [
                "slack:C789",
                "email:team@example.com",
            ]
            client.close()

    def test_deliberate_with_output_format(self) -> None:
        """Submit deliberation with non-default output format."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_462"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.orchestration.deliberate(
                question="Test question",
                output_format="decision_receipt",
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["output_format"] == "decision_receipt"
            client.close()

    def test_deliberate_without_consensus(self) -> None:
        """Submit deliberation without requiring consensus."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_463"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.orchestration.deliberate(
                question="Test question",
                require_consensus=False,
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["require_consensus"] is False
            client.close()

    def test_deliberate_with_high_priority(self) -> None:
        """Submit deliberation with high priority."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_464"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.orchestration.deliberate(
                question="Test question",
                priority="critical",
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["priority"] == "critical"
            client.close()

    def test_deliberate_with_custom_rounds(self) -> None:
        """Submit deliberation with custom max rounds."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_465"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.orchestration.deliberate(
                question="Test question",
                max_rounds=5,
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["max_rounds"] == 5
            client.close()

    def test_deliberate_with_custom_timeout(self) -> None:
        """Submit deliberation with custom timeout."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_466"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.orchestration.deliberate(
                question="Test question",
                timeout_seconds=600.0,
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["timeout_seconds"] == 600.0
            client.close()

    def test_deliberate_with_template(self) -> None:
        """Submit deliberation using a template."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_467"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.orchestration.deliberate(
                question="Test question",
                template="security-review",
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["template"] == "security-review"
            client.close()

    def test_deliberate_with_metadata(self) -> None:
        """Submit deliberation with custom metadata."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_468"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            metadata = {"project": "infrastructure", "requestor": "alice"}
            client.orchestration.deliberate(
                question="Test question",
                metadata=metadata,
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["metadata"] == metadata
            client.close()

    def test_deliberate_full_options(self) -> None:
        """Submit deliberation with all options specified."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "request_id": "req_full",
                "status": "pending",
                "estimated_time": 120,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.orchestration.deliberate(
                question="Complex architecture decision",
                knowledge_sources=["confluence:arch-docs"],
                workspaces=["ws_main"],
                team_strategy="diverse",
                agents=["claude", "gpt-4"],
                output_channels=["slack:C999"],
                output_format="decision_receipt",
                require_consensus=False,
                priority="high",
                max_rounds=5,
                timeout_seconds=600.0,
                template="architecture-review",
                metadata={"urgency": "high"},
            )

            call_args = mock_request.call_args
            json_data = call_args[1]["json"]
            assert json_data["question"] == "Complex architecture decision"
            assert json_data["knowledge_sources"] == ["confluence:arch-docs"]
            assert json_data["workspaces"] == ["ws_main"]
            assert json_data["team_strategy"] == "diverse"
            assert json_data["agents"] == ["claude", "gpt-4"]
            assert json_data["output_channels"] == ["slack:C999"]
            assert json_data["output_format"] == "decision_receipt"
            assert json_data["require_consensus"] is False
            assert json_data["priority"] == "high"
            assert json_data["max_rounds"] == 5
            assert json_data["timeout_seconds"] == 600.0
            assert json_data["template"] == "architecture-review"
            assert json_data["metadata"] == {"urgency": "high"}
            assert result["request_id"] == "req_full"
            client.close()

    def test_deliberate_default_values_not_sent(self) -> None:
        """Verify default values are not sent in the request."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_defaults"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.orchestration.deliberate(question="Test question")

            call_args = mock_request.call_args
            json_data = call_args[1]["json"]
            # Only question should be present when using defaults
            assert json_data == {"question": "Test question"}
            client.close()


class TestOrchestrationDeliberateSync:
    """Tests for synchronous deliberation submission."""

    def test_deliberate_sync_minimal(self) -> None:
        """Submit synchronous deliberation with just a question."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "request_id": "req_sync_123",
                "status": "completed",
                "decision": "Proceed with migration",
                "confidence": 0.95,
                "consensus_reached": True,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.orchestration.deliberate_sync(
                question="Should we migrate to Kubernetes?"
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/orchestration/deliberate/sync",
                json={"question": "Should we migrate to Kubernetes?"},
            )
            assert result["status"] == "completed"
            assert result["decision"] == "Proceed with migration"
            assert result["consensus_reached"] is True
            client.close()

    def test_deliberate_sync_with_knowledge_sources(self) -> None:
        """Submit synchronous deliberation with knowledge sources."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "request_id": "req_sync_456",
                "decision": "Hybrid approach recommended",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.orchestration.deliberate_sync(
                question="Cloud strategy?",
                knowledge_sources=["confluence:cloud-docs", "jira:CLOUD-123"],
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["knowledge_sources"] == [
                "confluence:cloud-docs",
                "jira:CLOUD-123",
            ]
            client.close()

    def test_deliberate_sync_with_all_options(self) -> None:
        """Submit synchronous deliberation with all options."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "request_id": "req_sync_full",
                "decision": "Approved",
                "reasoning": ["Point 1", "Point 2"],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.orchestration.deliberate_sync(
                question="Security approval needed",
                knowledge_sources=["security:policies"],
                workspaces=["ws_security"],
                team_strategy="specified",
                agents=["claude", "security-bot"],
                output_channels=["slack:C-sec"],
                output_format="summary",
                require_consensus=True,
                priority="critical",
                max_rounds=7,
                timeout_seconds=900.0,
                template="security-approval",
                metadata={"ticket": "SEC-123"},
            )

            call_args = mock_request.call_args
            json_data = call_args[1]["json"]
            assert json_data["team_strategy"] == "specified"
            assert json_data["agents"] == ["claude", "security-bot"]
            assert json_data["priority"] == "critical"
            assert json_data["max_rounds"] == 7
            assert result["decision"] == "Approved"
            client.close()

    def test_deliberate_sync_github_review_format(self) -> None:
        """Submit synchronous deliberation with GitHub review format."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "request_id": "req_github",
                "review": {"approval": "APPROVED", "comments": []},
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.orchestration.deliberate_sync(
                question="Review this PR",
                output_format="github_review",
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["output_format"] == "github_review"
            client.close()

    def test_deliberate_sync_slack_message_format(self) -> None:
        """Submit synchronous deliberation with Slack message format."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "request_id": "req_slack",
                "slack_message": {"blocks": []},
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.orchestration.deliberate_sync(
                question="Status update needed",
                output_format="slack_message",
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["output_format"] == "slack_message"
            client.close()


class TestOrchestrationGetStatus:
    """Tests for getting deliberation status."""

    def test_get_status_pending(self) -> None:
        """Get status of a pending deliberation."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "request_id": "req_123",
                "status": "pending",
                "created_at": "2024-01-01T00:00:00Z",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.orchestration.get_status("req_123")

            mock_request.assert_called_once_with("GET", "/api/v1/orchestration/status/req_123")
            assert result["status"] == "pending"
            client.close()

    def test_get_status_running(self) -> None:
        """Get status of a running deliberation."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "request_id": "req_456",
                "status": "running",
                "current_round": 2,
                "max_rounds": 5,
                "agents_participating": ["claude", "gpt-4", "gemini"],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.orchestration.get_status("req_456")

            assert result["status"] == "running"
            assert result["current_round"] == 2
            assert len(result["agents_participating"]) == 3
            client.close()

    def test_get_status_completed(self) -> None:
        """Get status of a completed deliberation."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "request_id": "req_789",
                "status": "completed",
                "result": {
                    "decision": "Approved",
                    "confidence": 0.92,
                    "consensus_reached": True,
                    "rounds_completed": 3,
                },
                "completed_at": "2024-01-01T01:00:00Z",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.orchestration.get_status("req_789")

            assert result["status"] == "completed"
            assert result["result"]["decision"] == "Approved"
            assert result["result"]["consensus_reached"] is True
            client.close()

    def test_get_status_failed(self) -> None:
        """Get status of a failed deliberation."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "request_id": "req_fail",
                "status": "failed",
                "error": "Timeout exceeded",
                "failed_at": "2024-01-01T01:30:00Z",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.orchestration.get_status("req_fail")

            assert result["status"] == "failed"
            assert result["error"] == "Timeout exceeded"
            client.close()

    def test_get_status_cancelled(self) -> None:
        """Get status of a cancelled deliberation."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "request_id": "req_cancel",
                "status": "cancelled",
                "cancelled_by": "user_abc",
                "cancelled_at": "2024-01-01T00:30:00Z",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.orchestration.get_status("req_cancel")

            assert result["status"] == "cancelled"
            assert result["cancelled_by"] == "user_abc"
            client.close()


class TestOrchestrationListTemplates:
    """Tests for listing deliberation templates."""

    def test_list_templates_empty(self) -> None:
        """List templates when none are available."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "templates": [],
                "count": 0,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.orchestration.list_templates()

            mock_request.assert_called_once_with("GET", "/api/v1/orchestration/templates")
            assert result["count"] == 0
            assert result["templates"] == []
            client.close()

    def test_list_templates_with_results(self) -> None:
        """List templates with multiple results."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "templates": [
                    {
                        "id": "tpl_security",
                        "name": "Security Review",
                        "description": "Multi-agent security assessment",
                        "default_agents": ["claude", "security-bot"],
                        "default_rounds": 5,
                    },
                    {
                        "id": "tpl_arch",
                        "name": "Architecture Review",
                        "description": "Architecture decision review",
                        "default_agents": ["claude", "gpt-4"],
                        "default_rounds": 3,
                    },
                    {
                        "id": "tpl_code",
                        "name": "Code Review",
                        "description": "Automated code review",
                        "default_agents": ["codex", "claude"],
                        "default_rounds": 2,
                    },
                ],
                "count": 3,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.orchestration.list_templates()

            assert result["count"] == 3
            assert len(result["templates"]) == 3
            assert result["templates"][0]["id"] == "tpl_security"
            assert result["templates"][1]["name"] == "Architecture Review"
            assert result["templates"][2]["default_rounds"] == 2
            client.close()


class TestAsyncOrchestrationDeliberate:
    """Tests for async client deliberation."""

    @pytest.mark.asyncio
    async def test_async_deliberate_minimal(self) -> None:
        """Submit async deliberation with just a question."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "request_id": "req_async_123",
                "status": "pending",
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.orchestration.deliberate(
                    question="Should we adopt microservices?"
                )

                mock_request.assert_called_once_with(
                    "POST",
                    "/api/v1/orchestration/deliberate",
                    json={"question": "Should we adopt microservices?"},
                )
                assert result["request_id"] == "req_async_123"

    @pytest.mark.asyncio
    async def test_async_deliberate_with_options(self) -> None:
        """Submit async deliberation with various options."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_async_456"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.orchestration.deliberate(
                    question="Infrastructure decision",
                    knowledge_sources=["docs:infrastructure"],
                    team_strategy="fast",
                    agents=["claude"],
                    priority="high",
                    max_rounds=2,
                )

                call_args = mock_request.call_args
                json_data = call_args[1]["json"]
                assert json_data["team_strategy"] == "fast"
                assert json_data["agents"] == ["claude"]
                assert json_data["priority"] == "high"
                assert json_data["max_rounds"] == 2

    @pytest.mark.asyncio
    async def test_async_deliberate_full_options(self) -> None:
        """Submit async deliberation with all options."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_async_full"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.orchestration.deliberate(
                    question="Full async test",
                    knowledge_sources=["kb:test"],
                    workspaces=["ws_async"],
                    team_strategy="random",
                    agents=["agent1", "agent2"],
                    output_channels=["webhook:endpoint"],
                    output_format="summary",
                    require_consensus=False,
                    priority="low",
                    max_rounds=10,
                    timeout_seconds=1200.0,
                    template="async-template",
                    metadata={"async": True},
                )

                call_args = mock_request.call_args
                json_data = call_args[1]["json"]
                assert json_data["team_strategy"] == "random"
                assert json_data["timeout_seconds"] == 1200.0
                assert json_data["template"] == "async-template"
                assert result["request_id"] == "req_async_full"


class TestAsyncOrchestrationDeliberateSync:
    """Tests for async client synchronous deliberation."""

    @pytest.mark.asyncio
    async def test_async_deliberate_sync_minimal(self) -> None:
        """Submit sync deliberation via async client with just a question."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "request_id": "req_async_sync_123",
                "status": "completed",
                "decision": "Approved",
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.orchestration.deliberate_sync(
                    question="Quick decision needed"
                )

                mock_request.assert_called_once_with(
                    "POST",
                    "/api/v1/orchestration/deliberate/sync",
                    json={"question": "Quick decision needed"},
                )
                assert result["status"] == "completed"
                assert result["decision"] == "Approved"

    @pytest.mark.asyncio
    async def test_async_deliberate_sync_with_options(self) -> None:
        """Submit sync deliberation via async client with options."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "request_id": "req_async_sync_456",
                "decision": "Proceed",
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.orchestration.deliberate_sync(
                    question="Sync decision via async",
                    knowledge_sources=["kb:sync"],
                    team_strategy="best_for_domain",
                    require_consensus=True,
                    max_rounds=3,
                )

                call_args = mock_request.call_args
                json_data = call_args[1]["json"]
                # best_for_domain is default so not sent
                assert "team_strategy" not in json_data
                # require_consensus=True is default so not sent
                assert "require_consensus" not in json_data
                # max_rounds=3 is default so not sent
                assert "max_rounds" not in json_data
                assert json_data["knowledge_sources"] == ["kb:sync"]


class TestAsyncOrchestrationGetStatus:
    """Tests for async client status retrieval."""

    @pytest.mark.asyncio
    async def test_async_get_status_pending(self) -> None:
        """Get status of pending deliberation via async client."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "request_id": "req_async_status",
                "status": "pending",
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.orchestration.get_status("req_async_status")

                mock_request.assert_called_once_with(
                    "GET", "/api/v1/orchestration/status/req_async_status"
                )
                assert result["status"] == "pending"

    @pytest.mark.asyncio
    async def test_async_get_status_completed(self) -> None:
        """Get status of completed deliberation via async client."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "request_id": "req_async_done",
                "status": "completed",
                "result": {
                    "decision": "Move forward",
                    "confidence": 0.88,
                },
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.orchestration.get_status("req_async_done")

                assert result["status"] == "completed"
                assert result["result"]["confidence"] == 0.88


class TestAsyncOrchestrationListTemplates:
    """Tests for async client template listing."""

    @pytest.mark.asyncio
    async def test_async_list_templates(self) -> None:
        """List templates via async client."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "templates": [
                    {"id": "tpl_async_1", "name": "Async Template 1"},
                    {"id": "tpl_async_2", "name": "Async Template 2"},
                ],
                "count": 2,
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.orchestration.list_templates()

                mock_request.assert_called_once_with("GET", "/api/v1/orchestration/templates")
                assert result["count"] == 2
                assert len(result["templates"]) == 2

    @pytest.mark.asyncio
    async def test_async_list_templates_empty(self) -> None:
        """List templates via async client when none exist."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "templates": [],
                "count": 0,
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.orchestration.list_templates()

                assert result["count"] == 0
                assert result["templates"] == []


class TestOrchestrationTeamStrategies:
    """Tests for different team strategy options."""

    def test_team_strategy_specified(self) -> None:
        """Test specified team strategy."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_strat_1"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.orchestration.deliberate(question="Test", team_strategy="specified")

            call_args = mock_request.call_args
            assert call_args[1]["json"]["team_strategy"] == "specified"
            client.close()

    def test_team_strategy_best_for_domain_not_sent(self) -> None:
        """Test best_for_domain strategy is not sent (default)."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_strat_2"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.orchestration.deliberate(question="Test", team_strategy="best_for_domain")

            call_args = mock_request.call_args
            assert "team_strategy" not in call_args[1]["json"]
            client.close()

    def test_team_strategy_diverse(self) -> None:
        """Test diverse team strategy."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_strat_3"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.orchestration.deliberate(question="Test", team_strategy="diverse")

            call_args = mock_request.call_args
            assert call_args[1]["json"]["team_strategy"] == "diverse"
            client.close()

    def test_team_strategy_fast(self) -> None:
        """Test fast team strategy."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_strat_4"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.orchestration.deliberate(question="Test", team_strategy="fast")

            call_args = mock_request.call_args
            assert call_args[1]["json"]["team_strategy"] == "fast"
            client.close()

    def test_team_strategy_random(self) -> None:
        """Test random team strategy."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_strat_5"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.orchestration.deliberate(question="Test", team_strategy="random")

            call_args = mock_request.call_args
            assert call_args[1]["json"]["team_strategy"] == "random"
            client.close()


class TestOrchestrationOutputFormats:
    """Tests for different output format options."""

    def test_output_format_standard_not_sent(self) -> None:
        """Test standard format is not sent (default)."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_fmt_1"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.orchestration.deliberate(question="Test", output_format="standard")

            call_args = mock_request.call_args
            assert "output_format" not in call_args[1]["json"]
            client.close()

    def test_output_format_decision_receipt(self) -> None:
        """Test decision_receipt output format."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_fmt_2"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.orchestration.deliberate(question="Test", output_format="decision_receipt")

            call_args = mock_request.call_args
            assert call_args[1]["json"]["output_format"] == "decision_receipt"
            client.close()

    def test_output_format_summary(self) -> None:
        """Test summary output format."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_fmt_3"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.orchestration.deliberate(question="Test", output_format="summary")

            call_args = mock_request.call_args
            assert call_args[1]["json"]["output_format"] == "summary"
            client.close()

    def test_output_format_github_review(self) -> None:
        """Test github_review output format."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_fmt_4"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.orchestration.deliberate(question="Test", output_format="github_review")

            call_args = mock_request.call_args
            assert call_args[1]["json"]["output_format"] == "github_review"
            client.close()

    def test_output_format_slack_message(self) -> None:
        """Test slack_message output format."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"request_id": "req_fmt_5"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.orchestration.deliberate(question="Test", output_format="slack_message")

            call_args = mock_request.call_args
            assert call_args[1]["json"]["output_format"] == "slack_message"
            client.close()
