"""
Tests for audit logging in skill invocations.

Covers:
- Successful invocation emits expected log messages
- Failed invocation emits error details in logs
- Audit event emission via AuditLog
- Audit logging failure does not break skill execution
"""

from __future__ import annotations

import logging
import time
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.skills.base import (
    Skill,
    SkillCapability,
    SkillContext,
    SkillManifest,
    SkillResult,
    SkillStatus,
)
from aragora.skills.registry import (
    SkillRegistry,
    _ensure_audit_imports,
    _skill_status_to_audit_outcome,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _MockSkill(Skill):
    """Simple mock skill for audit tests."""

    def __init__(
        self,
        name: str = "audit_test_skill",
        required_permissions: list[str] | None = None,
    ):
        self._name = name
        self._perms = required_permissions or []
        self._execute_result = SkillResult.create_success({"ok": True})

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name=self._name,
            version="2.0.0",
            capabilities=[SkillCapability.WEB_SEARCH],
            input_schema={"query": {"type": "string", "required": True}},
            required_permissions=self._perms,
            tags=["test"],
        )

    async def execute(self, input_data: dict[str, Any], context: SkillContext) -> SkillResult:
        return self._execute_result


class _FailingSkill(Skill):
    """Skill that always raises."""

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="failing_audit_skill",
            version="1.0.0",
            capabilities=[],
            input_schema={},
        )

    async def execute(self, input_data: dict[str, Any], context: SkillContext) -> SkillResult:
        raise RuntimeError("intentional failure for audit test")


@pytest.fixture
def registry() -> SkillRegistry:
    return SkillRegistry()


@pytest.fixture
def context() -> SkillContext:
    return SkillContext(
        user_id="audit_user",
        tenant_id="audit_tenant",
        permissions=["skill:execute"],
        correlation_id="corr-123",
    )


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestSkillStatusToAuditOutcome:
    """Tests for _skill_status_to_audit_outcome mapping."""

    def test_success_maps_to_success(self):
        assert _skill_status_to_audit_outcome(SkillStatus.SUCCESS) == "success"

    def test_permission_denied_maps_to_denied(self):
        assert _skill_status_to_audit_outcome(SkillStatus.PERMISSION_DENIED) == "denied"

    def test_failure_maps_to_failure(self):
        assert _skill_status_to_audit_outcome(SkillStatus.FAILURE) == "failure"

    def test_timeout_maps_to_failure(self):
        assert _skill_status_to_audit_outcome(SkillStatus.TIMEOUT) == "failure"

    def test_rate_limited_maps_to_failure(self):
        assert _skill_status_to_audit_outcome(SkillStatus.RATE_LIMITED) == "failure"


# ---------------------------------------------------------------------------
# Invocation audit log message tests
# ---------------------------------------------------------------------------


class TestInvokeAuditLogMessages:
    """Verify that invoke() emits the expected structured log messages."""

    @pytest.mark.asyncio
    async def test_successful_invocation_logs_start_and_completion(
        self, registry: SkillRegistry, context: SkillContext, caplog
    ):
        """Successful invocation should log both start and completion."""
        skill = _MockSkill()
        registry.register(skill)

        with caplog.at_level(logging.INFO, logger="aragora.skills.registry"):
            result = await registry.invoke("audit_test_skill", {"query": "test"}, context)

        assert result.success is True

        # Check invocation start log
        start_msgs = [r for r in caplog.records if "Skill invocation:" in r.message]
        assert len(start_msgs) == 1
        assert "skill=audit_test_skill" in start_msgs[0].message
        assert "user=audit_user" in start_msgs[0].message
        assert "tenant=audit_tenant" in start_msgs[0].message

        # Check completion log
        done_msgs = [r for r in caplog.records if "Skill completed:" in r.message]
        assert len(done_msgs) == 1
        assert "status=success" in done_msgs[0].message
        assert "duration=" in done_msgs[0].message

    @pytest.mark.asyncio
    async def test_failed_invocation_logs_error_details(
        self, registry: SkillRegistry, context: SkillContext, caplog
    ):
        """Failed invocation should log the error status."""
        skill = _FailingSkill()
        registry.register(skill)

        with caplog.at_level(logging.INFO, logger="aragora.skills.registry"):
            result = await registry.invoke("failing_audit_skill", {}, context)

        assert result.success is False

        done_msgs = [r for r in caplog.records if "Skill completed:" in r.message]
        assert len(done_msgs) == 1
        assert "status=failure" in done_msgs[0].message

    @pytest.mark.asyncio
    async def test_permission_denied_logs_correctly(self, registry: SkillRegistry, caplog):
        """Permission denied invocations should log with permission_denied status."""
        skill = _MockSkill(
            required_permissions=["admin:secret"],
        )
        registry.register(skill)
        ctx = SkillContext(
            user_id="unprivileged",
            tenant_id="t1",
            permissions=["basic:read"],
        )

        with caplog.at_level(logging.INFO, logger="aragora.skills.registry"):
            result = await registry.invoke("audit_test_skill", {"query": "x"}, ctx)

        assert result.status == SkillStatus.PERMISSION_DENIED

        done_msgs = [r for r in caplog.records if "Skill completed:" in r.message]
        assert len(done_msgs) == 1
        assert "status=permission_denied" in done_msgs[0].message


# ---------------------------------------------------------------------------
# Structured audit event emission tests
# ---------------------------------------------------------------------------


class TestInvokeAuditEventEmission:
    """Verify that invoke() emits structured audit events via AuditLog."""

    @pytest.mark.asyncio
    async def test_successful_invoke_emits_audit_event(
        self, registry: SkillRegistry, context: SkillContext
    ):
        """Successful invocation should emit an AuditEvent with outcome=success."""
        skill = _MockSkill()
        registry.register(skill)

        mock_audit_log_instance = MagicMock()
        mock_audit_log_cls = MagicMock(return_value=mock_audit_log_instance)

        with (
            patch("aragora.skills.registry._audit_log_cls", mock_audit_log_cls),
            patch("aragora.skills.registry._audit_event_cls") as mock_event_cls,
            patch("aragora.skills.registry._audit_category_cls") as mock_cat_cls,
            patch("aragora.skills.registry._audit_outcome_cls") as mock_outcome_cls,
            patch("aragora.skills.registry._ensure_audit_imports", return_value=True),
        ):
            mock_event = MagicMock()
            mock_event_cls.return_value = mock_event

            result = await registry.invoke("audit_test_skill", {"query": "test"}, context)

        assert result.success is True

        # Verify AuditEvent was constructed
        mock_event_cls.assert_called_once()
        call_kwargs = mock_event_cls.call_args
        # Could be positional or keyword args
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs["action"] == "skill_invoke"
            assert call_kwargs.kwargs["actor_id"] == "audit_user"
            assert call_kwargs.kwargs["resource_type"] == "skill"
            assert call_kwargs.kwargs["resource_id"] == "audit_test_skill"
            assert call_kwargs.kwargs["org_id"] == "audit_tenant"
            assert call_kwargs.kwargs["correlation_id"] == "corr-123"
            details = call_kwargs.kwargs["details"]
            assert details["skill_name"] == "audit_test_skill"
            assert details["skill_version"] == "2.0.0"
            assert details["status"] == "success"
            assert "duration_seconds" in details

        # Verify log() was called
        mock_audit_log_instance.log.assert_called_once_with(mock_event)

    @pytest.mark.asyncio
    async def test_failed_invoke_emits_audit_event_with_error(
        self, registry: SkillRegistry, context: SkillContext
    ):
        """Failed invocation should emit an AuditEvent with error details."""
        skill = _FailingSkill()
        registry.register(skill)

        mock_audit_log_instance = MagicMock()
        mock_audit_log_cls = MagicMock(return_value=mock_audit_log_instance)

        with (
            patch("aragora.skills.registry._audit_log_cls", mock_audit_log_cls),
            patch("aragora.skills.registry._audit_event_cls") as mock_event_cls,
            patch("aragora.skills.registry._audit_category_cls") as mock_cat_cls,
            patch("aragora.skills.registry._audit_outcome_cls") as mock_outcome_cls,
            patch("aragora.skills.registry._ensure_audit_imports", return_value=True),
        ):
            mock_event = MagicMock()
            mock_event_cls.return_value = mock_event

            result = await registry.invoke("failing_audit_skill", {}, context)

        assert result.success is False

        mock_event_cls.assert_called_once()
        call_kwargs = mock_event_cls.call_args
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs["reason"] != ""
            details = call_kwargs.kwargs["details"]
            assert "error" in details

        mock_audit_log_instance.log.assert_called_once()

    @pytest.mark.asyncio
    async def test_permission_denied_emits_audit_event(self, registry: SkillRegistry):
        """Permission denied should emit an AuditEvent with outcome=denied."""
        skill = _MockSkill(required_permissions=["admin:secret"])
        registry.register(skill)
        ctx = SkillContext(
            user_id="bad_user",
            tenant_id="t1",
            permissions=["basic:read"],
        )

        mock_audit_log_instance = MagicMock()
        mock_audit_log_cls = MagicMock(return_value=mock_audit_log_instance)

        with (
            patch("aragora.skills.registry._audit_log_cls", mock_audit_log_cls),
            patch("aragora.skills.registry._audit_event_cls") as mock_event_cls,
            patch("aragora.skills.registry._audit_category_cls") as mock_cat_cls,
            patch("aragora.skills.registry._audit_outcome_cls") as mock_outcome_cls,
            patch("aragora.skills.registry._ensure_audit_imports", return_value=True),
        ):
            mock_event = MagicMock()
            mock_event_cls.return_value = mock_event

            result = await registry.invoke("audit_test_skill", {"query": "x"}, ctx)

        assert result.status == SkillStatus.PERMISSION_DENIED
        mock_outcome_cls.assert_called_with("denied")
        mock_audit_log_instance.log.assert_called_once()


# ---------------------------------------------------------------------------
# Audit failure resilience tests
# ---------------------------------------------------------------------------


class TestAuditFailureResilience:
    """Verify that audit logging failures never break skill execution."""

    @pytest.mark.asyncio
    async def test_audit_import_failure_doesnt_break_invocation(
        self, registry: SkillRegistry, context: SkillContext
    ):
        """If audit imports fail, skill execution should still succeed."""
        skill = _MockSkill()
        registry.register(skill)

        with patch(
            "aragora.skills.registry._ensure_audit_imports",
            return_value=False,
        ):
            result = await registry.invoke("audit_test_skill", {"query": "test"}, context)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_audit_log_exception_doesnt_break_invocation(
        self, registry: SkillRegistry, context: SkillContext
    ):
        """If AuditLog.log() raises, skill execution should still succeed."""
        skill = _MockSkill()
        registry.register(skill)

        mock_audit_log_instance = MagicMock()
        mock_audit_log_instance.log.side_effect = RuntimeError("audit DB down")
        mock_audit_log_cls = MagicMock(return_value=mock_audit_log_instance)

        with (
            patch("aragora.skills.registry._audit_log_cls", mock_audit_log_cls),
            patch("aragora.skills.registry._audit_event_cls", MagicMock()),
            patch("aragora.skills.registry._audit_category_cls", MagicMock()),
            patch("aragora.skills.registry._audit_outcome_cls", MagicMock()),
            patch("aragora.skills.registry._ensure_audit_imports", return_value=True),
        ):
            result = await registry.invoke("audit_test_skill", {"query": "test"}, context)

        # Skill should still succeed even though audit blew up
        assert result.success is True

    @pytest.mark.asyncio
    async def test_audit_event_construction_failure_doesnt_break_invocation(
        self, registry: SkillRegistry, context: SkillContext
    ):
        """If AuditEvent constructor raises, skill execution should still succeed."""
        skill = _MockSkill()
        registry.register(skill)

        mock_event_cls = MagicMock(side_effect=TypeError("bad event args"))
        mock_audit_log_cls = MagicMock()

        with (
            patch("aragora.skills.registry._audit_log_cls", mock_audit_log_cls),
            patch("aragora.skills.registry._audit_event_cls", mock_event_cls),
            patch("aragora.skills.registry._audit_category_cls", MagicMock()),
            patch("aragora.skills.registry._audit_outcome_cls", MagicMock()),
            patch("aragora.skills.registry._ensure_audit_imports", return_value=True),
        ):
            result = await registry.invoke("audit_test_skill", {"query": "test"}, context)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_failed_skill_still_emits_audit_even_when_audit_fails(
        self, registry: SkillRegistry, context: SkillContext
    ):
        """A failing skill combined with failing audit should still return a result."""
        skill = _FailingSkill()
        registry.register(skill)

        mock_audit_log_instance = MagicMock()
        mock_audit_log_instance.log.side_effect = RuntimeError("double failure")
        mock_audit_log_cls = MagicMock(return_value=mock_audit_log_instance)

        with (
            patch("aragora.skills.registry._audit_log_cls", mock_audit_log_cls),
            patch("aragora.skills.registry._audit_event_cls", MagicMock()),
            patch("aragora.skills.registry._audit_category_cls", MagicMock()),
            patch("aragora.skills.registry._audit_outcome_cls", MagicMock()),
            patch("aragora.skills.registry._ensure_audit_imports", return_value=True),
        ):
            result = await registry.invoke("failing_audit_skill", {}, context)

        # Skill failure result should still be returned
        assert result.success is False
        assert "intentional failure" in result.error_message
