"""Tests for prometheus_control_plane metrics recording functions."""

import pytest


class TestDeliberationMetricsRecording:
    """Test deliberation metrics recording functions."""

    def test_record_deliberation_complete_import(self):
        """Verify record_deliberation_complete can be imported."""
        from aragora.server.prometheus_control_plane import record_deliberation_complete

        assert callable(record_deliberation_complete)

    def test_record_deliberation_complete_success(self):
        """Test recording a successful deliberation."""
        from aragora.server.prometheus_control_plane import record_deliberation_complete

        # Should not raise
        record_deliberation_complete(
            duration_seconds=45.5,
            status="completed",
            consensus_reached=True,
            confidence=0.85,
            round_count=3,
            agent_count=4,
        )

    def test_record_deliberation_complete_failed(self):
        """Test recording a failed deliberation."""
        from aragora.server.prometheus_control_plane import record_deliberation_complete

        # Should not raise
        record_deliberation_complete(
            duration_seconds=120.0,
            status="failed",
            consensus_reached=False,
            confidence=0.0,
            round_count=5,
            agent_count=3,
        )

    def test_record_deliberation_complete_timeout(self):
        """Test recording a timed out deliberation."""
        from aragora.server.prometheus_control_plane import record_deliberation_complete

        # Should not raise
        record_deliberation_complete(
            duration_seconds=300.0,
            status="timeout",
            consensus_reached=False,
            confidence=0.0,
            round_count=0,
            agent_count=2,
        )


class TestSLAMetricsRecording:
    """Test SLA metrics recording functions."""

    def test_record_deliberation_sla_import(self):
        """Verify record_deliberation_sla can be imported."""
        from aragora.server.prometheus_control_plane import record_deliberation_sla

        assert callable(record_deliberation_sla)

    def test_record_deliberation_sla_compliant(self):
        """Test recording compliant SLA."""
        from aragora.server.prometheus_control_plane import record_deliberation_sla

        # Should not raise
        record_deliberation_sla("compliant")

    def test_record_deliberation_sla_warning(self):
        """Test recording SLA warning."""
        from aragora.server.prometheus_control_plane import record_deliberation_sla

        record_deliberation_sla("warning")

    def test_record_deliberation_sla_critical(self):
        """Test recording SLA critical."""
        from aragora.server.prometheus_control_plane import record_deliberation_sla

        record_deliberation_sla("critical")

    def test_record_deliberation_sla_violated(self):
        """Test recording SLA violated."""
        from aragora.server.prometheus_control_plane import record_deliberation_sla

        record_deliberation_sla("violated")


class TestAgentUtilizationMetrics:
    """Test agent utilization metrics recording."""

    def test_record_agent_utilization_import(self):
        """Verify record_agent_utilization can be imported."""
        from aragora.server.prometheus_control_plane import record_agent_utilization

        assert callable(record_agent_utilization)

    def test_record_agent_utilization(self):
        """Test recording agent utilization."""
        from aragora.server.prometheus_control_plane import record_agent_utilization

        record_agent_utilization("claude-3-opus", 0.75)
        record_agent_utilization("gpt-4", 0.5)
        record_agent_utilization("gemini-pro", 1.0)

    def test_record_agent_utilization_boundary_values(self):
        """Test boundary utilization values."""
        from aragora.server.prometheus_control_plane import record_agent_utilization

        record_agent_utilization("agent-idle", 0.0)
        record_agent_utilization("agent-full", 1.0)


class TestPolicyDecisionMetrics:
    """Test policy decision metrics recording."""

    def test_record_policy_decision_import(self):
        """Verify record_policy_decision can be imported."""
        from aragora.server.prometheus_control_plane import record_policy_decision

        assert callable(record_policy_decision)

    def test_record_policy_decision_allow(self):
        """Test recording allow decision."""
        from aragora.server.prometheus_control_plane import record_policy_decision

        record_policy_decision(decision="allow", policy_type="all")

    def test_record_policy_decision_deny(self):
        """Test recording deny decisions."""
        from aragora.server.prometheus_control_plane import record_policy_decision

        record_policy_decision(decision="deny", policy_type="agent_restriction")
        record_policy_decision(decision="deny", policy_type="region_restriction")

    def test_record_policy_decision_warn(self):
        """Test recording warn decision."""
        from aragora.server.prometheus_control_plane import record_policy_decision

        record_policy_decision(decision="warn", policy_type="sla_violation")


class TestExistingControlPlaneMetrics:
    """Test existing control plane metrics still work."""

    def test_record_control_plane_task_submitted(self):
        """Test task submission recording."""
        from aragora.server.prometheus_control_plane import record_control_plane_task_submitted

        record_control_plane_task_submitted(task_type="debate", priority="high")

    def test_record_control_plane_task_status(self):
        """Test task status recording."""
        from aragora.server.prometheus_control_plane import record_control_plane_task_status

        record_control_plane_task_status(status="pending", count=5)
        record_control_plane_task_status(status="running", count=2)

    def test_record_control_plane_task_completed(self):
        """Test task completion recording."""
        from aragora.server.prometheus_control_plane import record_control_plane_task_completed

        record_control_plane_task_completed(
            task_type="debate",
            outcome="completed",
            duration_seconds=45.0,
        )

    def test_record_control_plane_queue_depth(self):
        """Test queue depth recording."""
        from aragora.server.prometheus_control_plane import record_control_plane_queue_depth

        record_control_plane_queue_depth(priority="high", depth=3)

    def test_record_control_plane_agents(self):
        """Test agent count recording."""
        from aragora.server.prometheus_control_plane import record_control_plane_agents

        record_control_plane_agents(status="available", count=5)
        record_control_plane_agents(status="busy", count=2)


class TestModuleExports:
    """Test module exports are correct."""

    def test_all_exports_present(self):
        """Verify all expected exports are in __all__."""
        from aragora.server import prometheus_control_plane

        expected = [
            "record_control_plane_task_submitted",
            "record_control_plane_task_status",
            "record_control_plane_task_completed",
            "record_control_plane_queue_depth",
            "record_control_plane_agents",
            "record_control_plane_agent_health",
            "record_control_plane_agent_latency",
            "record_control_plane_task_retry",
            "record_control_plane_dead_letter_queue",
            "record_control_plane_claim_latency",
            "record_deliberation_complete",
            "record_deliberation_sla",
            "record_agent_utilization",
            "record_policy_decision",
        ]

        for name in expected:
            assert name in prometheus_control_plane.__all__, f"Missing export: {name}"
            assert hasattr(prometheus_control_plane, name), f"Missing attribute: {name}"
