"""Tests for Enterprise Deployment Validator."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from aragora.ops.deployment_validator import ComponentStatus, Severity
from aragora.ops.enterprise_validator import (
    get_enterprise_health_summary,
    validate_enterprise_deployment,
)


class TestEnterpriseValidator:
    """Tests for enterprise deployment validation."""

    @pytest.mark.asyncio
    async def test_validate_enterprise_deployment_default(self):
        """Test basic enterprise validation runs without errors."""
        result = await validate_enterprise_deployment()

        assert result is not None
        assert hasattr(result, "ready")
        assert hasattr(result, "live")
        assert hasattr(result, "issues")
        assert hasattr(result, "components")
        assert hasattr(result, "validated_at")

    @pytest.mark.asyncio
    async def test_validate_enterprise_deployment_components(self):
        """Test that all expected components are validated."""
        result = await validate_enterprise_deployment()

        component_names = {c.name for c in result.components}

        # Should validate these components
        expected_components = {
            "rbac",
            "audit",
            "tenancy",
            "control_plane",
            "channels",
            "observability",
        }

        for expected in expected_components:
            assert expected in component_names, f"Missing component: {expected}"

    @pytest.mark.asyncio
    async def test_audit_disabled_warning(self):
        """Test warning when audit logging is disabled."""
        with patch.dict(os.environ, {"ARAGORA_AUDIT_ENABLED": "false"}):
            result = await validate_enterprise_deployment()

            audit_issues = [i for i in result.issues if i.component == "audit"]
            disabled_issues = [i for i in audit_issues if "disabled" in i.message.lower()]
            assert len(disabled_issues) > 0

    @pytest.mark.asyncio
    async def test_audit_retention_warning(self):
        """Test warning for short audit retention period."""
        with patch.dict(os.environ, {"ARAGORA_AUDIT_RETENTION_DAYS": "30"}):
            result = await validate_enterprise_deployment()

            audit_issues = [i for i in result.issues if i.component == "audit"]
            retention_issues = [i for i in audit_issues if "retention" in i.message.lower()]
            assert len(retention_issues) > 0

    @pytest.mark.asyncio
    async def test_redis_warning_for_multi_instance(self):
        """Test critical error when multi-instance without Redis."""
        with patch.dict(
            os.environ,
            {
                "ARAGORA_MULTI_INSTANCE": "true",
                "REDIS_URL": "",
                "ARAGORA_REDIS_URL": "",
            },
            clear=False,
        ):
            # Clear Redis URLs
            env_copy = os.environ.copy()
            if "REDIS_URL" in env_copy:
                del env_copy["REDIS_URL"]
            if "ARAGORA_REDIS_URL" in env_copy:
                del env_copy["ARAGORA_REDIS_URL"]

            with patch.dict(os.environ, env_copy, clear=True):
                os.environ["ARAGORA_MULTI_INSTANCE"] = "true"

                result = await validate_enterprise_deployment()

                cp_issues = [i for i in result.issues if i.component == "control_plane"]
                critical_issues = [i for i in cp_issues if i.severity == Severity.CRITICAL]

                # Should have a critical issue about Redis
                assert any("redis" in i.message.lower() for i in critical_issues) or any(
                    "multi-instance" in i.message.lower() for i in cp_issues
                )

    @pytest.mark.asyncio
    async def test_slack_incomplete_config_warning(self):
        """Test warning for incomplete Slack configuration."""
        with patch.dict(
            os.environ,
            {"SLACK_BOT_TOKEN": "xoxb-test-token", "SLACK_SIGNING_SECRET": ""},
            clear=False,
        ):
            result = await validate_enterprise_deployment()

            channel_issues = [i for i in result.issues if i.component == "channels"]
            slack_issues = [i for i in channel_issues if "slack" in i.message.lower()]
            assert len(slack_issues) > 0

    @pytest.mark.asyncio
    async def test_debug_logging_warning(self):
        """Test warning for debug logging in production."""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            result = await validate_enterprise_deployment()

            obs_issues = [i for i in result.issues if i.component == "observability"]
            debug_issues = [i for i in obs_issues if "debug" in i.message.lower()]
            assert len(debug_issues) > 0

    @pytest.mark.asyncio
    async def test_ready_when_no_critical_issues(self):
        """Test ready is True when no critical issues."""
        # With default config (minimal issues expected)
        result = await validate_enterprise_deployment()

        # Count critical issues
        critical_count = sum(1 for i in result.issues if i.severity == Severity.CRITICAL)

        # Ready should be True if no critical issues
        assert result.ready == (critical_count == 0)

    @pytest.mark.asyncio
    async def test_validated_at_is_set(self):
        """Test that validated_at timestamp is set."""
        result = await validate_enterprise_deployment()

        assert result.validated_at > 0
        assert isinstance(result.validated_at, float)


class TestEnterpriseHealthSummary:
    """Tests for quick health summary."""

    def test_get_enterprise_health_summary_keys(self):
        """Test health summary returns expected keys."""
        summary = get_enterprise_health_summary()

        expected_keys = {
            "rbac_available",
            "audit_enabled",
            "redis_configured",
            "multi_instance",
            "metrics_enabled",
            "sentry_configured",
            "otel_configured",
        }

        assert set(summary.keys()) == expected_keys

    def test_get_enterprise_health_summary_types(self):
        """Test health summary returns correct types."""
        summary = get_enterprise_health_summary()

        for key, value in summary.items():
            assert isinstance(value, bool), f"{key} should be bool, got {type(value)}"

    def test_health_summary_reflects_environment(self):
        """Test that summary reflects environment variables."""
        with patch.dict(
            os.environ,
            {
                "ARAGORA_AUDIT_ENABLED": "false",
                "ARAGORA_MULTI_INSTANCE": "true",
                "SENTRY_DSN": "https://test@sentry.io/123",
            },
        ):
            summary = get_enterprise_health_summary()

            assert summary["audit_enabled"] is False
            assert summary["multi_instance"] is True
            assert summary["sentry_configured"] is True
