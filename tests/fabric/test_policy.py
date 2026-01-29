"""Tests for Agent Fabric PolicyEngine."""

from __future__ import annotations

import asyncio
import pytest

from aragora.fabric.policy import PolicyEngine
from aragora.fabric.models import (
    Policy,
    PolicyContext,
    PolicyDecision,
    PolicyEffect,
    PolicyRule,
)


@pytest.fixture
def engine():
    return PolicyEngine()


def make_context(agent_id: str = "a1", action: str = "tool:shell:execute") -> PolicyContext:
    return PolicyContext(agent_id=agent_id, action=action)


def allow_all_policy() -> Policy:
    return Policy(
        id="allow-all",
        name="Allow All",
        rules=[PolicyRule(action_pattern="*", effect=PolicyEffect.ALLOW)],
        priority=0,
    )


def deny_shell_policy() -> Policy:
    return Policy(
        id="deny-shell",
        name="Deny Shell",
        rules=[
            PolicyRule(
                action_pattern="tool:shell:*",
                effect=PolicyEffect.DENY,
                description="No shell access",
            )
        ],
        priority=10,
    )


class TestPolicyEvaluation:
    @pytest.mark.asyncio
    async def test_default_deny(self, engine):
        decision = await engine.check("some:action", make_context())
        assert not decision.allowed
        assert decision.effect == PolicyEffect.DENY

    @pytest.mark.asyncio
    async def test_allow_all(self, engine):
        await engine.add_policy(allow_all_policy())
        decision = await engine.check("anything", make_context())
        assert decision.allowed

    @pytest.mark.asyncio
    async def test_deny_shell(self, engine):
        await engine.add_policy(deny_shell_policy())
        decision = await engine.check("tool:shell:execute", make_context())
        assert not decision.allowed
        assert decision.matching_policy == "deny-shell"

    @pytest.mark.asyncio
    async def test_priority_ordering(self, engine):
        # Lower priority allow-all
        await engine.add_policy(allow_all_policy())
        # Higher priority deny-shell
        await engine.add_policy(deny_shell_policy())

        # Shell should be denied (higher priority wins)
        decision = await engine.check("tool:shell:execute", make_context())
        assert not decision.allowed

        # Non-shell should still be allowed
        decision = await engine.check("tool:browser:navigate", make_context())
        assert decision.allowed

    @pytest.mark.asyncio
    async def test_require_approval(self, engine):
        policy = Policy(
            id="approve-deploy",
            name="Approve Deployments",
            rules=[
                PolicyRule(
                    action_pattern="deploy:*",
                    effect=PolicyEffect.REQUIRE_APPROVAL,
                    description="Deployments require approval",
                )
            ],
        )
        await engine.add_policy(policy)
        decision = await engine.check("deploy:production", make_context())
        assert decision.requires_approval
        assert decision.effect == PolicyEffect.REQUIRE_APPROVAL

    @pytest.mark.asyncio
    async def test_disabled_policy_skipped(self, engine):
        policy = Policy(
            id="disabled",
            name="Disabled",
            rules=[PolicyRule(action_pattern="*", effect=PolicyEffect.ALLOW)],
            enabled=False,
        )
        await engine.add_policy(policy)
        # Should fall through to default deny
        decision = await engine.check("anything", make_context())
        assert not decision.allowed

    @pytest.mark.asyncio
    async def test_condition_matching(self, engine):
        policy = Policy(
            id="tenant-specific",
            name="Tenant Specific",
            rules=[
                PolicyRule(
                    action_pattern="data:read",
                    effect=PolicyEffect.ALLOW,
                    conditions={"tenant_id": "tenant-1"},
                )
            ],
        )
        await engine.add_policy(policy)

        # Matching tenant
        ctx = PolicyContext(agent_id="a1", tenant_id="tenant-1")
        decision = await engine.check("data:read", ctx)
        assert decision.allowed

        # Non-matching tenant
        ctx = PolicyContext(agent_id="a1", tenant_id="tenant-2")
        decision = await engine.check("data:read", ctx)
        assert not decision.allowed

    @pytest.mark.asyncio
    async def test_glob_patterns(self, engine):
        policy = Policy(
            id="tool-access",
            name="Tool Access",
            rules=[
                PolicyRule(action_pattern="tool:*:read", effect=PolicyEffect.ALLOW),
                PolicyRule(action_pattern="tool:*:write", effect=PolicyEffect.DENY),
            ],
        )
        await engine.add_policy(policy)

        decision = await engine.check("tool:file:read", make_context())
        assert decision.allowed

        decision = await engine.check("tool:file:write", make_context())
        assert not decision.allowed


class TestPolicyManagement:
    @pytest.mark.asyncio
    async def test_add_policy(self, engine):
        await engine.add_policy(allow_all_policy())
        policy = await engine.get_policy("allow-all")
        assert policy is not None
        assert policy.name == "Allow All"

    @pytest.mark.asyncio
    async def test_remove_policy(self, engine):
        await engine.add_policy(allow_all_policy())
        result = await engine.remove_policy("allow-all")
        assert result is True
        policy = await engine.get_policy("allow-all")
        assert policy is None

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self, engine):
        result = await engine.remove_policy("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_policies(self, engine):
        await engine.add_policy(allow_all_policy())
        await engine.add_policy(deny_shell_policy())
        policies = await engine.list_policies()
        assert len(policies) == 2

    @pytest.mark.asyncio
    async def test_update_policy(self, engine):
        await engine.add_policy(allow_all_policy())
        updated = Policy(
            id="allow-all",
            name="Updated Allow All",
            rules=[PolicyRule(action_pattern="tool:*", effect=PolicyEffect.ALLOW)],
        )
        await engine.add_policy(updated)
        policy = await engine.get_policy("allow-all")
        assert policy.name == "Updated Allow All"


class TestApprovalWorkflow:
    @pytest.mark.asyncio
    async def test_request_and_approve(self, engine):
        ctx = make_context()

        # Start approval request in background
        async def request():
            return await engine.require_approval(
                "deploy:prod", ctx, approvers=["admin1"], timeout_seconds=5.0
            )

        # Start request
        request_task = asyncio.create_task(request())
        await asyncio.sleep(0.05)  # Let request register

        # Find and approve
        pending = await engine.list_pending_approvals(approver_id="admin1")
        assert len(pending) == 1
        await engine.approve(pending[0].id, "admin1")

        result = await request_task
        assert result.approved

    @pytest.mark.asyncio
    async def test_request_and_deny(self, engine):
        ctx = make_context()

        async def request():
            return await engine.require_approval(
                "deploy:prod", ctx, approvers=["admin1"], timeout_seconds=5.0
            )

        request_task = asyncio.create_task(request())
        await asyncio.sleep(0.05)

        pending = await engine.list_pending_approvals()
        assert len(pending) == 1
        await engine.deny(pending[0].id, "admin1")

        result = await request_task
        assert not result.approved

    @pytest.mark.asyncio
    async def test_approval_timeout(self, engine):
        ctx = make_context()
        result = await engine.require_approval(
            "deploy:prod", ctx, approvers=["admin1"], timeout_seconds=0.1
        )
        assert not result.approved
        assert result.waited_seconds >= 0.05

    @pytest.mark.asyncio
    async def test_approve_nonexistent(self, engine):
        result = await engine.approve("nonexistent", "admin1")
        assert result is False

    @pytest.mark.asyncio
    async def test_approve_unauthorized(self, engine):
        ctx = make_context()

        async def request():
            return await engine.require_approval(
                "deploy:prod", ctx, approvers=["admin1"], timeout_seconds=2.0
            )

        task = asyncio.create_task(request())
        await asyncio.sleep(0.05)

        pending = await engine.list_pending_approvals()
        result = await engine.approve(pending[0].id, "unauthorized-user")
        assert result is False

        # Clean up - cancel the request
        await engine.deny(pending[0].id, "admin1")
        await task


class TestStats:
    @pytest.mark.asyncio
    async def test_stats(self, engine):
        await engine.add_policy(allow_all_policy())
        await engine.check("action1", make_context())
        await engine.check("action2", make_context())

        stats = await engine.get_stats()
        assert stats["policies"] == 1
        assert stats["decisions_allowed"] == 2
        assert stats["decisions_denied"] == 0
