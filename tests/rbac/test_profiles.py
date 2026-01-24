"""Tests for RBAC Profile System."""

import pytest

from aragora.rbac.profiles import (
    RBACProfile,
    ProfileConfig,
    PROFILE_CONFIGS,
    get_profile_config,
    get_profile_roles,
    get_default_role,
    get_available_roles_for_assignment,
    can_upgrade_profile,
    get_migration_plan,
    get_lite_role_summary,
)


class TestProfileConfig:
    """Tests for profile configuration."""

    def test_all_profiles_defined(self):
        """All profile enums have configurations."""
        for profile in RBACProfile:
            assert profile in PROFILE_CONFIGS

    def test_lite_profile_has_three_roles(self):
        """Lite profile has exactly 3 roles."""
        config = PROFILE_CONFIGS[RBACProfile.LITE]
        assert len(config.roles) == 3
        assert set(config.roles) == {"owner", "admin", "member"}

    def test_standard_profile_has_five_roles(self):
        """Standard profile has 5 roles."""
        config = PROFILE_CONFIGS[RBACProfile.STANDARD]
        assert len(config.roles) == 5
        assert "analyst" in config.roles
        assert "viewer" in config.roles

    def test_enterprise_profile_has_all_roles(self):
        """Enterprise profile has all 8 roles."""
        config = PROFILE_CONFIGS[RBACProfile.ENTERPRISE]
        assert len(config.roles) == 8
        assert "compliance_officer" in config.roles
        assert "team_lead" in config.roles

    def test_all_profiles_have_member_default(self):
        """All profiles default to member role."""
        for profile in RBACProfile:
            config = PROFILE_CONFIGS[profile]
            assert config.default_role == "member"


class TestGetProfileConfig:
    """Tests for get_profile_config function."""

    def test_get_by_enum(self):
        """Can get config by enum value."""
        config = get_profile_config(RBACProfile.LITE)
        assert config.name == "Lite"

    def test_get_by_string(self):
        """Can get config by string name."""
        config = get_profile_config("lite")
        assert config.name == "Lite"

    def test_get_by_string_case_insensitive(self):
        """String lookup is case-insensitive."""
        config = get_profile_config("STANDARD")
        assert config.name == "Standard"

    def test_invalid_profile_raises_error(self):
        """Unknown profile raises ValueError."""
        with pytest.raises(ValueError, match="Unknown profile"):
            get_profile_config("invalid_profile")


class TestGetProfileRoles:
    """Tests for get_profile_roles function."""

    def test_lite_returns_three_roles(self):
        """Lite profile returns 3 Role objects."""
        roles = get_profile_roles("lite")
        assert len(roles) == 3
        assert "owner" in roles
        assert "admin" in roles
        assert "member" in roles

    def test_roles_are_valid_role_objects(self):
        """Returned roles are valid Role objects."""
        roles = get_profile_roles("enterprise")
        for name, role in roles.items():
            assert role.name == name
            assert role.is_system is True


class TestGetDefaultRole:
    """Tests for get_default_role function."""

    def test_default_is_member(self):
        """Default role is member for all profiles."""
        for profile in ["lite", "standard", "enterprise"]:
            role = get_default_role(profile)
            assert role.name == "member"

    def test_returns_role_object(self):
        """Returns actual Role object."""
        role = get_default_role("lite")
        assert hasattr(role, "permissions")
        assert hasattr(role, "priority")


class TestRoleAssignment:
    """Tests for role assignment permissions."""

    def test_owner_can_assign_all_except_owner(self):
        """Owner can assign all roles except owner."""
        assignable = get_available_roles_for_assignment("lite", "owner")
        assert "admin" in assignable
        assert "member" in assignable
        assert "owner" not in assignable

    def test_admin_can_assign_limited_roles(self):
        """Admin can only assign member, analyst, viewer."""
        assignable = get_available_roles_for_assignment("standard", "admin")
        assert "member" in assignable
        assert "analyst" in assignable
        assert "viewer" in assignable
        assert "admin" not in assignable
        assert "owner" not in assignable

    def test_member_cannot_assign(self):
        """Members cannot assign roles."""
        assignable = get_available_roles_for_assignment("lite", "member")
        assert len(assignable) == 0


class TestProfileUpgrade:
    """Tests for profile upgrade validation."""

    def test_can_upgrade_lite_to_standard(self):
        """Can upgrade from lite to standard."""
        assert can_upgrade_profile("lite", "standard") is True

    def test_can_upgrade_standard_to_enterprise(self):
        """Can upgrade from standard to enterprise."""
        assert can_upgrade_profile("standard", "enterprise") is True

    def test_can_upgrade_lite_to_enterprise(self):
        """Can upgrade from lite directly to enterprise."""
        assert can_upgrade_profile("lite", "enterprise") is True

    def test_cannot_downgrade_enterprise_to_lite(self):
        """Cannot downgrade from enterprise to lite."""
        assert can_upgrade_profile("enterprise", "lite") is False

    def test_can_stay_same_profile(self):
        """Same profile is valid (no-op upgrade)."""
        assert can_upgrade_profile("standard", "standard") is True


class TestMigrationPlan:
    """Tests for profile migration planning."""

    def test_lite_to_standard_adds_roles(self):
        """Upgrading lite to standard adds analyst and viewer."""
        plan = get_migration_plan("lite", "standard")
        assert "analyst" in plan["add"]
        assert "viewer" in plan["add"]
        assert "owner" in plan["keep"]
        assert "admin" in plan["keep"]
        assert "member" in plan["keep"]

    def test_standard_to_enterprise_adds_roles(self):
        """Upgrading standard to enterprise adds more roles."""
        plan = get_migration_plan("standard", "enterprise")
        assert "compliance_officer" in plan["add"]
        assert "team_lead" in plan["add"]
        assert "debate_creator" in plan["add"]

    def test_same_profile_no_changes(self):
        """Same profile migration has no adds."""
        plan = get_migration_plan("lite", "lite")
        assert len(plan["add"]) == 0
        assert len(plan["remove"]) == 0
        assert len(plan["keep"]) == 3


class TestLiteRoleSummary:
    """Tests for lite role UI summary."""

    def test_returns_three_roles(self):
        """Returns summary for 3 roles."""
        summary = get_lite_role_summary()
        assert len(summary) == 3

    def test_has_required_fields(self):
        """Each summary has name, display_name, description."""
        summary = get_lite_role_summary()
        for role in summary:
            assert "name" in role
            assert "display_name" in role
            assert "description" in role

    def test_owner_first(self):
        """Owner is first in the list (highest permission)."""
        summary = get_lite_role_summary()
        assert summary[0]["name"] == "owner"

    def test_descriptions_are_user_friendly(self):
        """Descriptions are short and user-friendly."""
        summary = get_lite_role_summary()
        for role in summary:
            # Descriptions should be under 100 chars for UI
            assert len(role["description"]) < 100
