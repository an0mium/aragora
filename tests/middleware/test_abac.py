"""Tests for ABAC (Attribute-Based Access Control) middleware."""

import pytest

from aragora.server.middleware.abac import (
    Action,
    AccessDecision,
    AccessEvaluator,
    AccessRequest,
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
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def registry():
    """Fresh policy registry for testing."""
    PolicyRegistry.reset()
    return get_policy_registry()


@pytest.fixture
def evaluator(registry):
    """Fresh evaluator for testing."""
    return AccessEvaluator(registry)


@pytest.fixture
def admin_subject():
    """System admin subject."""
    return Subject(
        user_id="admin-001",
        role="admin",
        plan="enterprise",
        workspace_id="ws-001",
        workspace_role="owner",
    )


@pytest.fixture
def owner_subject():
    """Resource owner subject."""
    return Subject(
        user_id="user-001",
        role="user",
        plan="pro",
        workspace_id="ws-001",
        workspace_role="owner",
    )


@pytest.fixture
def member_subject():
    """Workspace member subject."""
    return Subject(
        user_id="user-002",
        role="user",
        plan="pro",
        workspace_id="ws-001",
        workspace_role="member",
    )


@pytest.fixture
def external_subject():
    """External user (different workspace)."""
    return Subject(
        user_id="user-003",
        role="user",
        plan="free",
        workspace_id="ws-002",
        workspace_role="member",
    )


@pytest.fixture
def debate_resource():
    """Sample debate resource."""
    return Resource(
        resource_type=ResourceType.DEBATE,
        resource_id="debate-001",
        owner_id="user-001",
        workspace_id="ws-001",
        sensitivity="internal",
    )


@pytest.fixture
def public_resource():
    """Public template resource."""
    return Resource(
        resource_type=ResourceType.TEMPLATE,
        resource_id="template-001",
        owner_id="user-001",
        workspace_id="ws-001",
        sensitivity="public",
    )


# =============================================================================
# Action Enum Tests
# =============================================================================


class TestAction:
    """Tests for Action enum."""

    def test_all_actions_exist(self):
        """Verify all expected actions exist."""
        expected = ["READ", "WRITE", "DELETE", "SHARE", "ADMIN", "EXECUTE", "EXPORT"]
        for action in expected:
            assert hasattr(Action, action)

    def test_action_values(self):
        """Verify action string values."""
        assert Action.READ.value == "read"
        assert Action.WRITE.value == "write"
        assert Action.DELETE.value == "delete"


# =============================================================================
# ResourceType Enum Tests
# =============================================================================


class TestResourceType:
    """Tests for ResourceType enum."""

    def test_all_resource_types_exist(self):
        """Verify all expected resource types exist."""
        expected = [
            "DEBATE",
            "WORKSPACE",
            "DOCUMENT",
            "KNOWLEDGE",
            "WORKFLOW",
            "AGENT",
            "TEMPLATE",
            "EVIDENCE",
            "INSIGHT",
            "TOURNAMENT",
        ]
        for rt in expected:
            assert hasattr(ResourceType, rt)


# =============================================================================
# Subject Tests
# =============================================================================


class TestSubject:
    """Tests for Subject dataclass."""

    def test_default_values(self):
        """Test default subject creation."""
        subject = Subject(user_id="test-user")
        assert subject.user_id == "test-user"
        assert subject.role == "user"
        assert subject.plan == "free"
        assert subject.workspace_id is None
        assert subject.scopes == []

    def test_is_admin_property(self):
        """Test is_admin property."""
        regular = Subject(user_id="u1", role="user")
        admin = Subject(user_id="u2", role="admin")
        superadmin = Subject(user_id="u3", role="superadmin")

        assert not regular.is_admin
        assert admin.is_admin
        assert superadmin.is_admin

    def test_is_workspace_admin_property(self):
        """Test is_workspace_admin property."""
        owner = Subject(user_id="u1", workspace_role="owner")
        admin = Subject(user_id="u2", workspace_role="admin")
        member = Subject(user_id="u3", workspace_role="member")
        viewer = Subject(user_id="u4", workspace_role="viewer")

        assert owner.is_workspace_admin
        assert admin.is_workspace_admin
        assert not member.is_workspace_admin
        assert not viewer.is_workspace_admin


# =============================================================================
# Resource Tests
# =============================================================================


class TestResource:
    """Tests for Resource dataclass."""

    def test_default_values(self):
        """Test default resource creation."""
        resource = Resource(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-001",
        )
        assert resource.sensitivity == "internal"
        assert resource.shared_with == set()
        assert resource.owner_id is None

    def test_with_shared_users(self):
        """Test resource with shared users."""
        resource = Resource(
            resource_type=ResourceType.DOCUMENT,
            resource_id="doc-001",
            shared_with={"user-a", "user-b"},
        )
        assert "user-a" in resource.shared_with
        assert "user-b" in resource.shared_with
        assert "user-c" not in resource.shared_with


# =============================================================================
# PolicyRegistry Tests
# =============================================================================


class TestPolicyRegistry:
    """Tests for PolicyRegistry."""

    def test_singleton_pattern(self, registry):
        """Test registry is a singleton."""
        registry2 = get_policy_registry()
        assert registry is registry2

    def test_default_policies_loaded(self, registry):
        """Test default policies are loaded."""
        debate_policy = registry.get(ResourceType.DEBATE)
        assert debate_policy is not None
        assert debate_policy.resource_type == ResourceType.DEBATE

    def test_register_custom_policy(self, registry):
        """Test registering a custom policy."""
        custom_policy = ResourcePolicy(
            resource_type=ResourceType.INSIGHT,
            allow_public_read=True,
        )
        registry.register(custom_policy)

        retrieved = registry.get(ResourceType.INSIGHT)
        assert retrieved.allow_public_read is True

    def test_list_all_policies(self, registry):
        """Test listing all policies."""
        policies = registry.list_all()
        assert len(policies) > 0
        assert ResourceType.DEBATE in policies


# =============================================================================
# AccessEvaluator Tests - Admin Access
# =============================================================================


class TestAccessEvaluatorAdmin:
    """Tests for admin access scenarios."""

    def test_admin_can_read_any_resource(self, evaluator, admin_subject, debate_resource):
        """System admins can read any resource."""
        request = AccessRequest(
            subject=admin_subject,
            resource=debate_resource,
            action=Action.READ,
        )
        decision = evaluator.evaluate(request)

        assert decision.allowed is True
        assert decision.policy_name == "admin_override"

    def test_admin_can_delete_any_resource(self, evaluator, admin_subject, debate_resource):
        """System admins can delete any resource."""
        request = AccessRequest(
            subject=admin_subject,
            resource=debate_resource,
            action=Action.DELETE,
        )
        decision = evaluator.evaluate(request)

        assert decision.allowed is True
        assert decision.policy_name == "admin_override"


# =============================================================================
# AccessEvaluator Tests - Owner Access
# =============================================================================


class TestAccessEvaluatorOwner:
    """Tests for owner access scenarios."""

    def test_owner_can_read_own_resource(self, evaluator, owner_subject, debate_resource):
        """Resource owners can read their resources."""
        request = AccessRequest(
            subject=owner_subject,
            resource=debate_resource,
            action=Action.READ,
        )
        decision = evaluator.evaluate(request)

        assert decision.allowed is True
        assert decision.policy_name == "owner_access"

    def test_owner_can_delete_own_resource(self, evaluator, owner_subject, debate_resource):
        """Resource owners can delete their resources."""
        request = AccessRequest(
            subject=owner_subject,
            resource=debate_resource,
            action=Action.DELETE,
        )
        decision = evaluator.evaluate(request)

        assert decision.allowed is True
        assert decision.policy_name == "owner_access"

    def test_owner_can_share_own_resource(self, evaluator, owner_subject, debate_resource):
        """Resource owners can share their resources."""
        request = AccessRequest(
            subject=owner_subject,
            resource=debate_resource,
            action=Action.SHARE,
        )
        decision = evaluator.evaluate(request)

        assert decision.allowed is True
        assert decision.policy_name == "owner_access"


# =============================================================================
# AccessEvaluator Tests - Workspace Access
# =============================================================================


class TestAccessEvaluatorWorkspace:
    """Tests for workspace-based access scenarios."""

    def test_workspace_member_can_read(self, evaluator, member_subject, debate_resource):
        """Workspace members can read resources in their workspace."""
        request = AccessRequest(
            subject=member_subject,
            resource=debate_resource,
            action=Action.READ,
        )
        decision = evaluator.evaluate(request)

        assert decision.allowed is True
        assert decision.policy_name == "workspace_member"

    def test_workspace_member_can_write(self, evaluator, member_subject, debate_resource):
        """Workspace members can write to resources in their workspace."""
        request = AccessRequest(
            subject=member_subject,
            resource=debate_resource,
            action=Action.WRITE,
        )
        decision = evaluator.evaluate(request)

        assert decision.allowed is True
        assert decision.policy_name == "workspace_member"

    def test_workspace_member_cannot_delete(self, evaluator, member_subject, debate_resource):
        """Workspace members cannot delete resources they don't own."""
        request = AccessRequest(
            subject=member_subject,
            resource=debate_resource,
            action=Action.DELETE,
        )
        decision = evaluator.evaluate(request)

        assert decision.allowed is False

    def test_external_user_cannot_access(self, evaluator, external_subject, debate_resource):
        """Users from different workspaces cannot access resources."""
        request = AccessRequest(
            subject=external_subject,
            resource=debate_resource,
            action=Action.READ,
        )
        decision = evaluator.evaluate(request)

        assert decision.allowed is False
        assert decision.policy_name == "workspace_isolation"


# =============================================================================
# AccessEvaluator Tests - Shared Access
# =============================================================================


class TestAccessEvaluatorShared:
    """Tests for shared access scenarios."""

    def test_shared_user_can_read(self, evaluator, external_subject):
        """Users with shared access can read resources."""
        resource = Resource(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-001",
            owner_id="user-001",
            workspace_id="ws-001",
            shared_with={external_subject.user_id},
        )

        # Need to disable workspace isolation for this test
        registry = get_policy_registry()
        policy = registry.get(ResourceType.DEBATE)
        original_require = policy.require_same_workspace
        policy.require_same_workspace = False

        try:
            request = AccessRequest(
                subject=external_subject,
                resource=resource,
                action=Action.READ,
            )
            decision = evaluator.evaluate(request)

            assert decision.allowed is True
            assert decision.policy_name == "shared_access"
        finally:
            policy.require_same_workspace = original_require

    def test_shared_user_cannot_write(self, evaluator, external_subject):
        """Users with shared access cannot write by default."""
        resource = Resource(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-001",
            owner_id="user-001",
            workspace_id="ws-001",
            shared_with={external_subject.user_id},
        )

        registry = get_policy_registry()
        policy = registry.get(ResourceType.DEBATE)
        original_require = policy.require_same_workspace
        policy.require_same_workspace = False

        try:
            request = AccessRequest(
                subject=external_subject,
                resource=resource,
                action=Action.WRITE,
            )
            decision = evaluator.evaluate(request)

            assert decision.allowed is False
        finally:
            policy.require_same_workspace = original_require


# =============================================================================
# AccessEvaluator Tests - Public Access
# =============================================================================


class TestAccessEvaluatorPublic:
    """Tests for public access scenarios."""

    def test_public_read_allowed(self, evaluator, external_subject, public_resource):
        """Public resources can be read by anyone."""
        registry = get_policy_registry()
        policy = registry.get(ResourceType.TEMPLATE)
        original_require = policy.require_same_workspace
        policy.require_same_workspace = False

        try:
            request = AccessRequest(
                subject=external_subject,
                resource=public_resource,
                action=Action.READ,
            )
            decision = evaluator.evaluate(request)

            assert decision.allowed is True
            assert decision.policy_name == "public_access"
        finally:
            policy.require_same_workspace = original_require

    def test_public_write_not_allowed(self, evaluator, external_subject, public_resource):
        """Public resources cannot be written by non-owners."""
        registry = get_policy_registry()
        policy = registry.get(ResourceType.TEMPLATE)
        original_require = policy.require_same_workspace
        policy.require_same_workspace = False

        try:
            request = AccessRequest(
                subject=external_subject,
                resource=public_resource,
                action=Action.WRITE,
            )
            decision = evaluator.evaluate(request)

            assert decision.allowed is False
        finally:
            policy.require_same_workspace = original_require


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_check_resource_access_owner(self):
        """Test check_resource_access for owner."""
        decision = check_resource_access(
            user_id="user-001",
            user_role="user",
            user_plan="pro",
            resource_type="debate",
            resource_id="debate-001",
            action="read",
            resource_owner_id="user-001",
            resource_workspace_id="ws-001",
            user_workspace_id="ws-001",
        )

        assert decision.allowed is True
        assert decision.policy_name == "owner_access"

    def test_check_resource_access_denied(self):
        """Test check_resource_access for denied access."""
        decision = check_resource_access(
            user_id="user-002",
            user_role="user",
            user_plan="free",
            resource_type="debate",
            resource_id="debate-001",
            action="delete",
            resource_owner_id="user-001",
            resource_workspace_id="ws-001",
            user_workspace_id="ws-002",
        )

        assert decision.allowed is False

    def test_is_resource_owner_true(self):
        """Test is_resource_owner returns true for owner."""
        assert is_resource_owner("user-001", "user-001") is True

    def test_is_resource_owner_false(self):
        """Test is_resource_owner returns false for non-owner."""
        assert is_resource_owner("user-002", "user-001") is False

    def test_is_resource_owner_none(self):
        """Test is_resource_owner returns false for None owner."""
        assert is_resource_owner("user-001", None) is False


# =============================================================================
# AccessDecision Tests
# =============================================================================


class TestAccessDecision:
    """Tests for AccessDecision dataclass."""

    def test_allowed_decision(self):
        """Test allowed decision creation."""
        decision = AccessDecision(
            allowed=True,
            reason="Owner access",
            policy_name="owner_access",
        )
        assert decision.allowed is True
        assert decision.reason == "Owner access"

    def test_denied_decision(self):
        """Test denied decision creation."""
        decision = AccessDecision(
            allowed=False,
            reason="Access denied",
            policy_name="default_deny",
        )
        assert decision.allowed is False

    def test_decision_with_attributes(self):
        """Test decision with evaluated attributes."""
        decision = AccessDecision(
            allowed=True,
            reason="Test",
            attributes_evaluated={
                "subject_id": "user-001",
                "action": "read",
            },
        )
        assert decision.attributes_evaluated["subject_id"] == "user-001"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unknown_resource_type(self, evaluator):
        """Test handling of unregistered resource type."""
        # Create a resource with a type that has no policy
        PolicyRegistry.reset()
        registry = get_policy_registry()

        # Remove a policy to simulate unknown type
        if ResourceType.INSIGHT in registry._policies:
            del registry._policies[ResourceType.INSIGHT]

        subject = Subject(user_id="user-001")
        resource = Resource(
            resource_type=ResourceType.INSIGHT,
            resource_id="insight-001",
        )

        request = AccessRequest(
            subject=subject,
            resource=resource,
            action=Action.READ,
        )

        evaluator = AccessEvaluator(registry)
        decision = evaluator.evaluate(request)

        assert decision.allowed is False
        assert "No policy defined" in decision.reason

    def test_empty_shared_with_set(self, evaluator, external_subject):
        """Test resource with empty shared_with set."""
        resource = Resource(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-001",
            owner_id="user-001",
            workspace_id="ws-001",
            shared_with=set(),
        )

        request = AccessRequest(
            subject=external_subject,
            resource=resource,
            action=Action.READ,
        )
        decision = evaluator.evaluate(request)

        assert decision.allowed is False

    def test_null_workspace_id(self, evaluator, owner_subject):
        """Test resource without workspace."""
        resource = Resource(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-001",
            owner_id=owner_subject.user_id,
            workspace_id=None,  # No workspace
        )

        request = AccessRequest(
            subject=owner_subject,
            resource=resource,
            action=Action.READ,
        )
        decision = evaluator.evaluate(request)

        # Owner should still have access
        assert decision.allowed is True
        assert decision.policy_name == "owner_access"
