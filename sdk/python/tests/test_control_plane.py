"""Tests for Control Plane namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient


class TestControlPlaneAgentManagement:
    """Tests for agent registry operations."""

    def test_list_agents_default(self) -> None:
        """List registered agents with defaults."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "agents": [
                    {"agent_id": "agent-1", "status": "available"},
                    {"agent_id": "agent-2", "status": "available"},
                ],
                "total": 2,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.control_plane.list_agents()

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/control-plane/agents",
                params={"available": "true"},
            )
            assert result["total"] == 2
            client.close()

    def test_list_agents_with_capability_filter(self) -> None:
        """List agents filtered by capability."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"agents": [], "total": 0}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.control_plane.list_agents(capability="reasoning")

            call_args = mock_request.call_args
            assert call_args[1]["params"]["capability"] == "reasoning"
            client.close()

    def test_list_agents_include_unavailable(self) -> None:
        """List agents including unavailable ones."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"agents": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.control_plane.list_agents(available=False)

            call_args = mock_request.call_args
            assert call_args[1]["params"]["available"] == "false"
            client.close()

    def test_register_agent(self) -> None:
        """Register a new agent."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "agent_id": "agent-new",
                "registered": True,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.control_plane.register_agent(
                agent_id="agent-new",
                capabilities=["reasoning", "coding"],
                model="claude-3-opus",
                provider="anthropic",
                metadata={"version": "1.0"},
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/control-plane/agents",
                json={
                    "agent_id": "agent-new",
                    "capabilities": ["reasoning", "coding"],
                    "model": "claude-3-opus",
                    "provider": "anthropic",
                    "metadata": {"version": "1.0"},
                },
            )
            assert result["registered"] is True
            client.close()

    def test_register_agent_minimal(self) -> None:
        """Register agent with minimal parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"agent_id": "agent-min"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.control_plane.register_agent(agent_id="agent-min")

            call_args = mock_request.call_args
            assert call_args[1]["json"]["agent_id"] == "agent-min"
            assert call_args[1]["json"]["capabilities"] == []
            assert call_args[1]["json"]["model"] == "unknown"
            client.close()

    def test_get_agent(self) -> None:
        """Get agent details."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "agent_id": "agent-1",
                "status": "available",
                "last_heartbeat": "2024-01-01T00:00:00Z",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.control_plane.get_agent("agent-1")

            mock_request.assert_called_once_with("GET", "/api/v1/control-plane/agents/agent-1")
            assert result["agent_id"] == "agent-1"
            client.close()

    def test_unregister_agent(self) -> None:
        """Unregister an agent."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"unregistered": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.control_plane.unregister_agent("agent-1")

            mock_request.assert_called_once_with("DELETE", "/api/v1/control-plane/agents/agent-1")
            assert result["unregistered"] is True
            client.close()

    def test_heartbeat(self) -> None:
        """Send agent heartbeat."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"acknowledged": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.control_plane.heartbeat("agent-1", status="idle")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/control-plane/agents/agent-1/heartbeat",
                json={"status": "idle"},
            )
            assert result["acknowledged"] is True
            client.close()


class TestControlPlaneTaskManagement:
    """Tests for task scheduling operations."""

    def test_submit_task(self) -> None:
        """Submit a new task."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"task_id": "task-123"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.control_plane.submit_task(
                task_type="debate",
                payload={"topic": "AI safety"},
                required_capabilities=["reasoning"],
                priority="high",
                timeout_seconds=300,
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/control-plane/tasks",
                json={
                    "task_type": "debate",
                    "payload": {"topic": "AI safety"},
                    "required_capabilities": ["reasoning"],
                    "priority": "high",
                    "timeout_seconds": 300,
                    "metadata": {},
                },
            )
            assert result["task_id"] == "task-123"
            client.close()

    def test_submit_task_defaults(self) -> None:
        """Submit task with default parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"task_id": "task-456"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.control_plane.submit_task(task_type="analysis")

            call_args = mock_request.call_args
            assert call_args[1]["json"]["priority"] == "normal"
            assert call_args[1]["json"]["required_capabilities"] == []
            client.close()

    def test_get_task(self) -> None:
        """Get task details."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "task_id": "task-123",
                "status": "running",
                "assigned_to": "agent-1",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.control_plane.get_task("task-123")

            mock_request.assert_called_once_with("GET", "/api/v1/control-plane/tasks/task-123")
            assert result["status"] == "running"
            client.close()

    def test_claim_task(self) -> None:
        """Claim a task for an agent."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "task_id": "task-123",
                "claimed": True,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.control_plane.claim_task(
                agent_id="agent-1",
                capabilities=["reasoning"],
                block_ms=10000,
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/control-plane/tasks/claim",
                json={
                    "agent_id": "agent-1",
                    "capabilities": ["reasoning"],
                    "block_ms": 10000,
                },
            )
            client.close()

    def test_complete_task(self) -> None:
        """Mark task as completed."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"completed": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.control_plane.complete_task(
                task_id="task-123",
                result={"output": "Done"},
                agent_id="agent-1",
                latency_ms=1500,
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/control-plane/tasks/task-123/complete",
                json={
                    "result": {"output": "Done"},
                    "agent_id": "agent-1",
                    "latency_ms": 1500,
                },
            )
            assert result["completed"] is True
            client.close()

    def test_fail_task(self) -> None:
        """Mark task as failed."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"failed": True, "requeued": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.control_plane.fail_task(
                task_id="task-123",
                error="Agent timeout",
                agent_id="agent-1",
                requeue=True,
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/control-plane/tasks/task-123/fail",
                json={
                    "error": "Agent timeout",
                    "requeue": True,
                    "agent_id": "agent-1",
                },
            )
            client.close()

    def test_cancel_task(self) -> None:
        """Cancel a task."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"cancelled": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.control_plane.cancel_task("task-123")

            mock_request.assert_called_once_with(
                "POST", "/api/v1/control-plane/tasks/task-123/cancel"
            )
            assert result["cancelled"] is True
            client.close()


class TestControlPlaneDeliberations:
    """Tests for deliberation operations."""

    def test_submit_deliberation_sync(self) -> None:
        """Submit synchronous deliberation."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "decision": "approved",
                "confidence": 0.95,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.control_plane.submit_deliberation(
                content="Should we deploy to production?",
                async_mode=False,
                priority="high",
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/control-plane/deliberations",
                json={
                    "content": "Should we deploy to production?",
                    "async": False,
                    "priority": "high",
                    "required_capabilities": ["deliberation"],
                },
            )
            assert result["decision"] == "approved"
            client.close()

    def test_submit_deliberation_async(self) -> None:
        """Submit asynchronous deliberation."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "task_id": "task-delib",
                "request_id": "req-123",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.control_plane.submit_deliberation(
                content="Complex decision",
                async_mode=True,
                timeout_seconds=600,
                context={"background": "info"},
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["async"] is True
            assert call_args[1]["json"]["timeout_seconds"] == 600
            assert result["request_id"] == "req-123"
            client.close()

    def test_get_deliberation(self) -> None:
        """Get deliberation result."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "request_id": "req-123",
                "status": "completed",
                "result": {"decision": "approved"},
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.control_plane.get_deliberation("req-123")

            mock_request.assert_called_once_with(
                "GET", "/api/v1/control-plane/deliberations/req-123"
            )
            assert result["status"] == "completed"
            client.close()

    def test_get_deliberation_status(self) -> None:
        """Get deliberation status for polling."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "status": "in_progress",
                "progress": 0.5,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.control_plane.get_deliberation_status("req-123")

            mock_request.assert_called_once_with(
                "GET", "/api/v1/control-plane/deliberations/req-123/status"
            )
            assert result["status"] == "in_progress"
            client.close()


class TestControlPlaneHealth:
    """Tests for health monitoring operations."""

    def test_get_system_health(self) -> None:
        """Get system health status."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "status": "healthy",
                "agents": {"total": 10, "available": 8},
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.control_plane.get_system_health()

            mock_request.assert_called_once_with("GET", "/api/v1/control-plane/health")
            assert result["status"] == "healthy"
            client.close()

    def test_get_detailed_health(self) -> None:
        """Get detailed system health."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "status": "healthy",
                "uptime": 86400,
                "version": "1.0.0",
                "components": {"database": "healthy", "cache": "healthy"},
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.control_plane.get_detailed_health()

            mock_request.assert_called_once_with("GET", "/api/v1/control-plane/health/detailed")
            assert "components" in result
            client.close()

    def test_get_agent_health(self) -> None:
        """Get health for specific agent."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "agent_id": "agent-1",
                "status": "healthy",
                "latency_ms": 50,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.control_plane.get_agent_health("agent-1")

            mock_request.assert_called_once_with("GET", "/api/v1/control-plane/health/agent-1")
            assert result["status"] == "healthy"
            client.close()

    def test_get_circuit_breakers(self) -> None:
        """Get circuit breaker states."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "breakers": [
                    {"name": "openai", "state": "closed"},
                    {"name": "anthropic", "state": "open"},
                ]
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.control_plane.get_circuit_breakers()

            mock_request.assert_called_once_with("GET", "/api/v1/control-plane/breakers")
            assert len(result["breakers"]) == 2
            client.close()


class TestControlPlaneStatistics:
    """Tests for statistics and metrics."""

    def test_get_stats(self) -> None:
        """Get control plane statistics."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "tasks_completed": 1000,
                "tasks_failed": 50,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.control_plane.get_stats()

            mock_request.assert_called_once_with("GET", "/api/v1/control-plane/stats")
            assert result["tasks_completed"] == 1000
            client.close()

    def test_get_queue(self) -> None:
        """Get job queue."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "jobs": [],
                "total": 0,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.control_plane.get_queue(limit=100)

            mock_request.assert_called_once_with(
                "GET", "/api/v1/control-plane/queue", params={"limit": 100}
            )
            client.close()

    def test_get_queue_metrics(self) -> None:
        """Get queue metrics."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "pending": 10,
                "running": 5,
                "throughput": 100,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.control_plane.get_queue_metrics()

            mock_request.assert_called_once_with("GET", "/api/v1/control-plane/queue/metrics")
            assert result["throughput"] == 100
            client.close()

    def test_get_metrics(self) -> None:
        """Get dashboard metrics."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "active_jobs": 5,
                "queued_jobs": 10,
                "agents": 8,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.control_plane.get_metrics()

            mock_request.assert_called_once_with("GET", "/api/v1/control-plane/metrics")
            assert result["active_jobs"] == 5
            client.close()


class TestControlPlaneAudit:
    """Tests for audit log operations."""

    def test_query_audit_logs_default(self) -> None:
        """Query audit logs with defaults."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"entries": [], "total": 0}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.control_plane.query_audit_logs()

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/control-plane/audit",
                params={"limit": 100, "offset": 0},
            )
            client.close()

    def test_query_audit_logs_with_filters(self) -> None:
        """Query audit logs with filters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"entries": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.control_plane.query_audit_logs(
                start_time="2024-01-01T00:00:00Z",
                end_time="2024-01-31T23:59:59Z",
                actions=["create", "update"],
                actor_types=["user", "agent"],
                limit=50,
            )

            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["start_time"] == "2024-01-01T00:00:00Z"
            assert params["actions"] == "create,update"
            assert params["actor_types"] == "user,agent"
            client.close()

    def test_get_audit_stats(self) -> None:
        """Get audit statistics."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "total_entries": 10000,
                "storage_backend": "postgres",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.control_plane.get_audit_stats()

            mock_request.assert_called_once_with("GET", "/api/v1/control-plane/audit/stats")
            assert result["total_entries"] == 10000
            client.close()

    def test_verify_audit_integrity(self) -> None:
        """Verify audit log integrity."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "valid": True,
                "message": "All entries verified",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.control_plane.verify_audit_integrity(start_seq=0, end_seq=1000)

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/control-plane/audit/verify",
                params={"start_seq": 0, "end_seq": 1000},
            )
            assert result["valid"] is True
            client.close()


class TestControlPlanePolicyViolations:
    """Tests for policy violation management."""

    def test_list_policy_violations(self) -> None:
        """List policy violations."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"violations": [], "total": 0}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.control_plane.list_policy_violations(
                status="open",
                violation_type="rate_limit",
            )

            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["status"] == "open"
            assert params["violation_type"] == "rate_limit"
            client.close()

    def test_get_policy_violation(self) -> None:
        """Get specific violation."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "violation_id": "viol-123",
                "status": "open",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.control_plane.get_policy_violation("viol-123")

            mock_request.assert_called_once_with(
                "GET", "/api/v1/control-plane/policies/violations/viol-123"
            )
            assert result["violation_id"] == "viol-123"
            client.close()

    def test_update_policy_violation(self) -> None:
        """Update violation status."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"updated": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.control_plane.update_policy_violation(
                violation_id="viol-123",
                status="resolved",
                resolution_notes="False positive",
            )

            mock_request.assert_called_once_with(
                "PATCH",
                "/api/v1/control-plane/policies/violations/viol-123",
                json={"status": "resolved", "resolution_notes": "False positive"},
            )
            assert result["updated"] is True
            client.close()


class TestAsyncControlPlane:
    """Tests for async control plane API."""

    @pytest.mark.asyncio
    async def test_async_get_system_health(self) -> None:
        """Get system health asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"status": "healthy"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.control_plane.get_system_health()

                mock_request.assert_called_once_with("GET", "/api/v1/control-plane/health")
                assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_async_list_agents(self) -> None:
        """List agents asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"agents": [], "total": 0}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.control_plane.list_agents()

                assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_async_submit_task(self) -> None:
        """Submit task asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"task_id": "task-async"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.control_plane.submit_task(
                    task_type="debate",
                    payload={"topic": "async test"},
                )

                assert result["task_id"] == "task-async"

    @pytest.mark.asyncio
    async def test_async_submit_deliberation(self) -> None:
        """Submit deliberation asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"decision": "approved"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.control_plane.submit_deliberation(
                    content="Async deliberation"
                )

                assert result["decision"] == "approved"
