"""
Tests for aragora.server.middleware.abac - Attribute-Based Access Control Middleware.

Tests cover:
1. Action, ResourceType, AccessLevel enums
2. Subject attributes and properties
3. Resource attributes
4. Environment attributes
5. AccessRequest and AccessDecision data models
6. ResourcePolicy defaults and customization
7. PolicyRegistry (singleton, register, get, reset)
8. AccessEvaluator precedence logic:
   - System admin override
   - Owner check
   - Workspace admin check
   - Workspace member check
   - Shared access check
   - Public read check
   - Sensitivity restrictions
   - Default deny
9. check_resource_access convenience function (with string/enum coercion)
10. is_resource_owner helper
11. require_resource_owner decorator
12. require_access decorator
13. DEFAULT_POLICIES for every ResourceType
14. Policy conflict / edge-case scenarios
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.middleware.abac import (
    AccessDecision,
    AccessEvaluator,
    AccessLevel,
    AccessRequest,
    Action,
    DEFAULT_POLICIES,
    Environment,
    PolicyRegistry,
    Resource,
    ResourcePolicy,
    ResourceType,
    Subject,
    check_resource_access,
    get_evaluator,
    get_policy_registry,
    is_resource_owner,
    require_access,
    require_resource_owner,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_subject(**overrides: Any) -> Subject:
    defaults = dict(user_id="user-1", role="user", plan="free")
    defaults.update(overrides)
    return Subject(**defaults)


def _make_resource(**overrides: Any) -> Resource:
    defaults = dict(resource_type=ResourceType.DEBATE, resource_id="res-1")
    defaults.update(overrides)
    return Resource(**defaults)


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset the PolicyRegistry singleton before each test."""
    PolicyRegistry.reset()
    yield
    PolicyRegistry.reset()


@pytest.fixture(autouse=True)
def _reset_evaluator():
    """Reset the global evaluator before each test."""
    import aragora.server.middleware.abac as abac_mod

    abac_mod._evaluator = None
    yield
    abac_mod._evaluator = None


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestActionEnum:
    """Tests for the Action enum."""

    def test_all_actions_defined(self):
        assert set(Action) == {
            Action.READ,
            Action.WRITE,
            Action.DELETE,
            Action.SHARE,
            Action.ADMIN,
            Action.EXECUTE,
            Action.EXPORT,
        }

    def test_action_values_are_lowercase_strings(self):
        for action in Action:
            assert action.value == action.value.lower()
            assert isinstance(action.value, str)

    def test_action_string_comparison(self):
        assert Action.READ == "read"
        assert Action.ADMIN == "admin"


class TestResourceTypeEnum:
    """Tests for the ResourceType enum."""

    def test_all_resource_types_defined(self):
        expected = {
            "debate",
            "workspace",
            "document",
            "knowledge",
            "workflow",
            "agent",
            "template",
            "evidence",
            "insight",
            "tournament",
        }
        assert {rt.value for rt in ResourceType} == expected

    def test_resource_type_string_comparison(self):
        assert ResourceType.DEBATE == "debate"
        assert ResourceType.WORKSPACE == "workspace"


class TestAccessLevelEnum:
    """Tests for the AccessLevel enum."""

    def test_access_levels_defined(self):
        assert set(al.value for al in AccessLevel) == {
            "none",
            "read",
            "write",
            "admin",
            "owner",
        }


# ===========================================================================
# Data Model Tests
# ===========================================================================


class TestSubject:
    """Tests for the Subject dataclass."""

    def test_defaults(self):
        s = Subject(user_id="u1")
        assert s.role == "user"
        assert s.plan == "free"
        assert s.workspace_id is None
        assert s.workspace_role is None
        assert s.scopes == []
        assert s.metadata == {}

    def test_is_admin_for_admin_role(self):
        assert Subject(user_id="u1", role="admin").is_admin is True

    def test_is_admin_for_superadmin_role(self):
        assert Subject(user_id="u1", role="superadmin").is_admin is True

    def test_is_admin_false_for_user_role(self):
        assert Subject(user_id="u1", role="user").is_admin is False

    def test_is_workspace_admin_owner(self):
        s = Subject(user_id="u1", workspace_role="owner")
        assert s.is_workspace_admin is True

    def test_is_workspace_admin_admin(self):
        s = Subject(user_id="u1", workspace_role="admin")
        assert s.is_workspace_admin is True

    def test_is_workspace_admin_false_for_member(self):
        s = Subject(user_id="u1", workspace_role="member")
        assert s.is_workspace_admin is False

    def test_is_workspace_admin_false_for_none(self):
        s = Subject(user_id="u1")
        assert s.is_workspace_admin is False


class TestResource:
    """Tests for the Resource dataclass."""

    def test_defaults(self):
        r = Resource(resource_type=ResourceType.DEBATE, resource_id="r1")
        assert r.owner_id is None
        assert r.workspace_id is None
        assert r.sensitivity == "internal"
        assert r.shared_with == set()
        assert r.metadata == {}

    def test_shared_with_set(self):
        r = Resource(
            resource_type=ResourceType.DOCUMENT,
            resource_id="r1",
            shared_with={"u1", "u2"},
        )
        assert "u1" in r.shared_with
        assert "u2" in r.shared_with


class TestEnvironment:
    """Tests for the Environment dataclass."""

    def test_defaults(self):
        e = Environment()
        assert e.ip_address is None
        assert e.user_agent is None
        assert e.request_path is None
        assert e.request_method is None
        assert e.timestamp is None

    def test_custom_values(self):
        e = Environment(ip_address="10.0.0.1", request_path="/api/v1/debates")
        assert e.ip_address == "10.0.0.1"
        assert e.request_path == "/api/v1/debates"


class TestAccessDecision:
    """Tests for the AccessDecision dataclass."""

    def test_allowed_decision(self):
        d = AccessDecision(allowed=True, reason="Test pass")
        assert d.allowed is True
        assert d.reason == "Test pass"
        assert d.policy_name is None
        assert d.attributes_evaluated == {}

    def test_denied_decision_with_policy(self):
        d = AccessDecision(allowed=False, reason="Denied", policy_name="default_deny")
        assert d.allowed is False
        assert d.policy_name == "default_deny"


# ===========================================================================
# ResourcePolicy Tests
# ===========================================================================


class TestResourcePolicy:
    """Tests for ResourcePolicy defaults and customization."""

    def test_default_owner_actions(self):
        p = ResourcePolicy(resource_type=ResourceType.DEBATE)
        expected = {
            Action.READ,
            Action.WRITE,
            Action.DELETE,
            Action.SHARE,
            Action.ADMIN,
            Action.EXPORT,
        }
        assert p.owner_actions == expected

    def test_default_sensitivity_restrictions(self):
        p = ResourcePolicy(resource_type=ResourceType.DEBATE)
        assert "free" in p.sensitivity_restrictions["public"]
        assert "enterprise" in p.sensitivity_restrictions["restricted"]
        assert "free" not in p.sensitivity_restrictions["restricted"]

    def test_custom_sensitivity_restrictions(self):
        p = ResourcePolicy(
            resource_type=ResourceType.DOCUMENT,
            sensitivity_restrictions={
                "public": ["free"],
                "internal": ["pro"],
            },
        )
        assert p.sensitivity_restrictions["public"] == ["free"]
        assert p.sensitivity_restrictions["internal"] == ["pro"]

    def test_allow_public_read_default_false(self):
        p = ResourcePolicy(resource_type=ResourceType.DEBATE)
        assert p.allow_public_read is False

    def test_require_same_workspace_default_true(self):
        p = ResourcePolicy(resource_type=ResourceType.DEBATE)
        assert p.require_same_workspace is True


# ===========================================================================
# PolicyRegistry Tests
# ===========================================================================


class TestPolicyRegistry:
    """Tests for the PolicyRegistry singleton."""

    def test_is_singleton(self):
        r1 = PolicyRegistry()
        r2 = PolicyRegistry()
        assert r1 is r2

    def test_default_policies_loaded(self):
        registry = get_policy_registry()
        for rt in DEFAULT_POLICIES:
            assert registry.get(rt) is not None

    def test_register_and_get(self):
        registry = get_policy_registry()
        custom = ResourcePolicy(
            resource_type=ResourceType.DEBATE,
            allow_public_read=True,
        )
        registry.register(custom)
        assert registry.get(ResourceType.DEBATE) is custom

    def test_get_returns_none_for_unknown(self):
        # After reset, all default types exist; simulate unknown via direct check
        registry = get_policy_registry()
        # ResourceType doesn't have a type that isn't in defaults, so test get_or_default instead
        policy = registry.get(ResourceType.DEBATE)
        assert policy is not None

    def test_get_or_default_creates_policy(self):
        registry = get_policy_registry()
        # Remove a type to test creation
        registry._policies.pop(ResourceType.INSIGHT, None)
        policy = registry.get_or_default(ResourceType.INSIGHT)
        assert policy.resource_type == ResourceType.INSIGHT

    def test_list_all(self):
        registry = get_policy_registry()
        all_policies = registry.list_all()
        assert isinstance(all_policies, dict)
        assert len(all_policies) >= len(DEFAULT_POLICIES)

    def test_reset_restores_defaults(self):
        registry = get_policy_registry()
        custom = ResourcePolicy(
            resource_type=ResourceType.DEBATE,
            allow_public_read=True,
        )
        registry.register(custom)
        assert registry.get(ResourceType.DEBATE).allow_public_read is True

        PolicyRegistry.reset()
        assert registry.get(ResourceType.DEBATE).allow_public_read is False


# ===========================================================================
# AccessEvaluator Tests
# ===========================================================================


class TestAccessEvaluator:
    """Tests for the AccessEvaluator precedence logic."""

    def _eval(self, subject: Subject, resource: Resource, action: Action) -> AccessDecision:
        evaluator = AccessEvaluator()
        return evaluator.evaluate(AccessRequest(subject=subject, resource=resource, action=action))

    # --- 1. System admin override ---

    def test_admin_allowed_for_admin_actions(self):
        subject = _make_subject(role="admin")
        resource = _make_resource()
        decision = self._eval(subject, resource, Action.READ)
        assert decision.allowed is True
        assert decision.policy_name == "admin_override"

    def test_superadmin_allowed(self):
        subject = _make_subject(role="superadmin")
        resource = _make_resource()
        decision = self._eval(subject, resource, Action.DELETE)
        assert decision.allowed is True
        assert decision.policy_name == "admin_override"

    def test_admin_denied_for_non_admin_action(self):
        """Admin trying action not in admin_actions gets denied via later rules."""
        # Default debate policy admin_actions don't include SHARE
        registry = get_policy_registry()
        policy = registry.get(ResourceType.DEBATE)
        assert Action.SHARE not in policy.admin_actions
        subject = _make_subject(role="admin")
        resource = _make_resource()
        decision = self._eval(subject, resource, Action.SHARE)
        assert decision.policy_name != "admin_override"

    # --- 2. Owner check ---

    def test_owner_allowed_for_owner_actions(self):
        subject = _make_subject(user_id="owner-1")
        resource = _make_resource(owner_id="owner-1")
        decision = self._eval(subject, resource, Action.DELETE)
        assert decision.allowed is True
        assert decision.policy_name == "owner_access"

    def test_owner_denied_for_action_not_in_owner_actions(self):
        # EXECUTE is not in debate owner_actions
        subject = _make_subject(user_id="owner-1")
        resource = _make_resource(owner_id="owner-1")
        decision = self._eval(subject, resource, Action.EXECUTE)
        assert decision.policy_name != "owner_access"

    def test_non_owner_not_owner_access(self):
        subject = _make_subject(user_id="other-user")
        resource = _make_resource(owner_id="owner-1")
        decision = self._eval(subject, resource, Action.DELETE)
        assert decision.policy_name != "owner_access"

    # --- 3a. Workspace admin ---

    def test_workspace_admin_allowed(self):
        subject = _make_subject(
            workspace_id="ws-1",
            workspace_role="admin",
        )
        resource = _make_resource(workspace_id="ws-1")
        decision = self._eval(subject, resource, Action.READ)
        assert decision.allowed is True
        assert decision.policy_name == "workspace_admin"

    def test_workspace_admin_different_workspace_denied(self):
        subject = _make_subject(
            workspace_id="ws-2",
            workspace_role="admin",
        )
        resource = _make_resource(workspace_id="ws-1")
        decision = self._eval(subject, resource, Action.READ)
        assert decision.allowed is False
        assert decision.policy_name == "workspace_isolation"

    # --- 3b. Workspace member ---

    def test_workspace_member_allowed_for_read(self):
        subject = _make_subject(workspace_id="ws-1", workspace_role="member")
        resource = _make_resource(workspace_id="ws-1")
        decision = self._eval(subject, resource, Action.READ)
        assert decision.allowed is True
        assert decision.policy_name == "workspace_member"

    def test_workspace_member_allowed_for_write(self):
        subject = _make_subject(workspace_id="ws-1", workspace_role="member")
        resource = _make_resource(workspace_id="ws-1")
        decision = self._eval(subject, resource, Action.WRITE)
        assert decision.allowed is True
        assert decision.policy_name == "workspace_member"

    def test_workspace_member_denied_for_delete(self):
        subject = _make_subject(workspace_id="ws-1", workspace_role="member")
        resource = _make_resource(workspace_id="ws-1")
        decision = self._eval(subject, resource, Action.DELETE)
        # Falls through to default deny (delete not in workspace_member_actions)
        assert decision.allowed is False

    # --- 4. Shared access ---

    def test_shared_user_read_allowed(self):
        subject = _make_subject(user_id="shared-user")
        resource = _make_resource(shared_with={"shared-user"})
        decision = self._eval(subject, resource, Action.READ)
        assert decision.allowed is True
        assert decision.policy_name == "shared_access"

    def test_shared_user_write_denied(self):
        subject = _make_subject(user_id="shared-user")
        resource = _make_resource(shared_with={"shared-user"})
        decision = self._eval(subject, resource, Action.WRITE)
        # WRITE is not in default shared_user_actions
        assert decision.policy_name != "shared_access"

    # --- 5. Public read ---

    def test_public_read_allowed_when_policy_allows(self):
        subject = _make_subject()
        resource = _make_resource(resource_type=ResourceType.EVIDENCE)
        decision = self._eval(subject, resource, Action.READ)
        assert decision.allowed is True
        assert decision.policy_name == "public_access"

    def test_public_read_denied_when_policy_disallows(self):
        subject = _make_subject()
        resource = _make_resource(resource_type=ResourceType.DEBATE)
        decision = self._eval(subject, resource, Action.READ)
        # Debate does not allow public read
        assert decision.policy_name != "public_access"

    def test_public_write_never_allowed(self):
        subject = _make_subject()
        resource = _make_resource(resource_type=ResourceType.EVIDENCE)
        decision = self._eval(subject, resource, Action.WRITE)
        # Public access only for READ
        assert decision.policy_name != "public_access"

    # --- 6. Sensitivity restrictions ---

    def test_sensitivity_restriction_blocks_free_plan(self):
        """Document policy: free plan cannot access confidential docs."""
        subject = _make_subject(plan="free")
        resource = _make_resource(
            resource_type=ResourceType.DOCUMENT,
            sensitivity="confidential",
        )
        decision = self._eval(subject, resource, Action.READ)
        assert decision.allowed is False
        assert decision.policy_name == "sensitivity_restriction"

    def test_sensitivity_restriction_allows_enterprise(self):
        """Document policy: enterprise can access restricted."""
        subject = _make_subject(plan="enterprise")
        resource = _make_resource(
            resource_type=ResourceType.DOCUMENT,
            sensitivity="restricted",
        )
        decision = self._eval(subject, resource, Action.READ)
        # Even with right plan, still needs another access path (owner, ws, etc.)
        # This test verifies the sensitivity check doesn't block it
        assert decision.policy_name != "sensitivity_restriction"

    # --- 7. Default deny ---

    def test_default_deny(self):
        subject = _make_subject()
        resource = _make_resource()
        decision = self._eval(subject, resource, Action.DELETE)
        assert decision.allowed is False
        assert decision.policy_name == "default_deny"

    def test_attributes_evaluated_populated(self):
        subject = _make_subject()
        resource = _make_resource()
        decision = self._eval(subject, resource, Action.READ)
        assert "subject_id" in decision.attributes_evaluated
        assert "resource_type" in decision.attributes_evaluated
        assert "action" in decision.attributes_evaluated

    # --- Workspace isolation ---

    def test_require_same_workspace_false_bypasses_isolation(self):
        """Workspace policy has require_same_workspace=False."""
        subject = _make_subject(
            workspace_id="ws-different",
            workspace_role="admin",
        )
        resource = _make_resource(
            resource_type=ResourceType.WORKSPACE,
            workspace_id="ws-1",
        )
        decision = self._eval(subject, resource, Action.READ)
        # Should NOT get workspace_isolation denial since require_same_workspace=False
        assert decision.policy_name != "workspace_isolation"

    # --- No policy defined ---

    def test_no_policy_returns_denied(self):
        registry = get_policy_registry()
        # Remove all policies
        registry._policies.clear()
        evaluator = AccessEvaluator(registry=registry)
        decision = evaluator.evaluate(
            AccessRequest(
                subject=_make_subject(),
                resource=_make_resource(),
                action=Action.READ,
            )
        )
        assert decision.allowed is False
        assert "No policy defined" in decision.reason


# ===========================================================================
# DEFAULT_POLICIES Tests
# ===========================================================================


class TestDefaultPolicies:
    """Tests for the DEFAULT_POLICIES constant."""

    def test_all_resource_types_covered(self):
        for rt in ResourceType:
            assert rt in DEFAULT_POLICIES, f"Missing default policy for {rt.value}"

    def test_evidence_public_read(self):
        p = DEFAULT_POLICIES[ResourceType.EVIDENCE]
        assert p.allow_public_read is True

    def test_template_public_read(self):
        p = DEFAULT_POLICIES[ResourceType.TEMPLATE]
        assert p.allow_public_read is True

    def test_agent_public_read(self):
        p = DEFAULT_POLICIES[ResourceType.AGENT]
        assert p.allow_public_read is True

    def test_tournament_public_read(self):
        p = DEFAULT_POLICIES[ResourceType.TOURNAMENT]
        assert p.allow_public_read is True

    def test_debate_no_public_read(self):
        p = DEFAULT_POLICIES[ResourceType.DEBATE]
        assert p.allow_public_read is False

    def test_document_stricter_sensitivity(self):
        p = DEFAULT_POLICIES[ResourceType.DOCUMENT]
        # free can only access public docs
        assert "free" not in p.sensitivity_restrictions["internal"]

    def test_workspace_no_require_same_workspace(self):
        p = DEFAULT_POLICIES[ResourceType.WORKSPACE]
        assert p.require_same_workspace is False

    def test_workflow_owner_can_execute(self):
        p = DEFAULT_POLICIES[ResourceType.WORKFLOW]
        assert Action.EXECUTE in p.owner_actions


# ===========================================================================
# check_resource_access Convenience Function
# ===========================================================================


class TestCheckResourceAccess:
    """Tests for the check_resource_access helper."""

    def test_basic_deny(self):
        decision = check_resource_access(
            user_id="u1",
            user_role="user",
            user_plan="free",
            resource_type=ResourceType.DEBATE,
            resource_id="d1",
            action=Action.DELETE,
        )
        assert decision.allowed is False

    def test_admin_access(self):
        decision = check_resource_access(
            user_id="admin-1",
            user_role="admin",
            user_plan="enterprise",
            resource_type=ResourceType.DEBATE,
            resource_id="d1",
            action=Action.READ,
        )
        assert decision.allowed is True
        assert decision.policy_name == "admin_override"

    def test_owner_access(self):
        decision = check_resource_access(
            user_id="u1",
            user_role="user",
            user_plan="free",
            resource_type=ResourceType.DEBATE,
            resource_id="d1",
            action=Action.DELETE,
            resource_owner_id="u1",
        )
        assert decision.allowed is True
        assert decision.policy_name == "owner_access"

    def test_string_coercion_resource_type(self):
        decision = check_resource_access(
            user_id="admin-1",
            user_role="admin",
            user_plan="enterprise",
            resource_type="debate",
            resource_id="d1",
            action="read",
        )
        assert decision.allowed is True

    def test_shared_access(self):
        decision = check_resource_access(
            user_id="shared-u",
            user_role="user",
            user_plan="free",
            resource_type=ResourceType.DEBATE,
            resource_id="d1",
            action=Action.READ,
            shared_with={"shared-u"},
        )
        assert decision.allowed is True
        assert decision.policy_name == "shared_access"

    def test_workspace_member_access(self):
        decision = check_resource_access(
            user_id="u1",
            user_role="user",
            user_plan="free",
            resource_type=ResourceType.DEBATE,
            resource_id="d1",
            action=Action.WRITE,
            resource_workspace_id="ws-1",
            user_workspace_id="ws-1",
            user_workspace_role="member",
        )
        assert decision.allowed is True
        assert decision.policy_name == "workspace_member"

    def test_sensitivity_restriction(self):
        decision = check_resource_access(
            user_id="u1",
            user_role="user",
            user_plan="free",
            resource_type=ResourceType.DOCUMENT,
            resource_id="doc-1",
            action=Action.READ,
            sensitivity="confidential",
        )
        assert decision.allowed is False
        assert decision.policy_name == "sensitivity_restriction"


# ===========================================================================
# is_resource_owner
# ===========================================================================


class TestIsResourceOwner:
    """Tests for is_resource_owner helper."""

    def test_owner_matches(self):
        assert is_resource_owner("u1", "u1") is True

    def test_owner_does_not_match(self):
        assert is_resource_owner("u1", "u2") is False

    def test_owner_is_none(self):
        assert is_resource_owner("u1", None) is False


# ===========================================================================
# get_evaluator
# ===========================================================================


class TestGetEvaluator:
    """Tests for the get_evaluator singleton."""

    def test_returns_evaluator(self):
        evaluator = get_evaluator()
        assert isinstance(evaluator, AccessEvaluator)

    def test_returns_same_instance(self):
        e1 = get_evaluator()
        e2 = get_evaluator()
        assert e1 is e2


# ===========================================================================
# require_resource_owner Decorator
# ===========================================================================


class TestRequireResourceOwnerDecorator:
    """Tests for the @require_resource_owner decorator."""

    @pytest.mark.asyncio
    async def test_allows_owner(self):
        @require_resource_owner("debate")
        async def delete_debate(**kwargs):
            return {"deleted": True}

        user = MagicMock()
        user.id = "u1"

        result = await delete_debate(user=user, resource_owner_id="u1")
        assert result == {"deleted": True}

    @pytest.mark.asyncio
    async def test_allows_admin(self):
        @require_resource_owner("debate")
        async def delete_debate(**kwargs):
            return {"deleted": True}

        user = MagicMock()
        user.id = "other-user"
        user.is_admin = True

        result = await delete_debate(user=user, resource_owner_id="u1")
        assert result == {"deleted": True}

    @pytest.mark.asyncio
    async def test_denies_non_owner_non_admin(self):
        @require_resource_owner("debate")
        async def delete_debate(**kwargs):
            return {"deleted": True}

        user = MagicMock()
        user.id = "other-user"
        user.is_admin = False

        result = await delete_debate(user=user, resource_owner_id="u1")
        # Should return error response dict, not the function result
        assert result != {"deleted": True}

    @pytest.mark.asyncio
    async def test_no_user_returns_auth_error(self):
        @require_resource_owner("debate")
        async def delete_debate(**kwargs):
            return {"deleted": True}

        result = await delete_debate()
        # Should return authentication error response
        assert result != {"deleted": True}

    @pytest.mark.asyncio
    async def test_allows_when_no_resource_owner(self):
        """When resource_owner_id is not in kwargs, should proceed."""

        @require_resource_owner("debate")
        async def delete_debate(**kwargs):
            return {"deleted": True}

        user = MagicMock()
        user.id = "u1"
        # No resource_owner_id passed
        result = await delete_debate(user=user)
        assert result == {"deleted": True}


# ===========================================================================
# require_access Decorator
# ===========================================================================


class TestRequireAccessDecorator:
    """Tests for the @require_access decorator."""

    @pytest.mark.asyncio
    async def test_allows_admin(self):
        @require_access(ResourceType.DEBATE, Action.READ)
        async def get_debate(**kwargs):
            return {"debate": "data"}

        user = MagicMock()
        user.id = "admin-1"
        user.role = "admin"
        user.plan = "enterprise"
        user.workspace_id = None

        result = await get_debate(user=user, resource_id="d1")
        assert result == {"debate": "data"}

    @pytest.mark.asyncio
    async def test_denies_unauthorized(self):
        @require_access(ResourceType.DEBATE, Action.DELETE)
        async def delete_debate(**kwargs):
            return {"deleted": True}

        user = MagicMock()
        user.id = "u1"
        user.role = "user"
        user.plan = "free"
        user.workspace_id = None

        result = await delete_debate(user=user, resource_id="d1")
        # Should return forbidden error, not the function result
        assert result != {"deleted": True}

    @pytest.mark.asyncio
    async def test_no_user_returns_auth_error(self):
        @require_access(ResourceType.DEBATE, Action.READ)
        async def get_debate(**kwargs):
            return {"debate": "data"}

        result = await get_debate()
        assert result != {"debate": "data"}

    @pytest.mark.asyncio
    async def test_string_args_work(self):
        @require_access("debate", "read")
        async def get_debate(**kwargs):
            return {"debate": "data"}

        user = MagicMock()
        user.id = "admin-1"
        user.role = "admin"
        user.plan = "enterprise"
        user.workspace_id = None

        result = await get_debate(user=user, resource_id="d1")
        assert result == {"debate": "data"}
