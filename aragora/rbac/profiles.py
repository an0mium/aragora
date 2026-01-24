"""
RBAC Profile System.

Provides simplified role configurations for different use cases:
- lite: 3 roles (owner, admin, member) for simple workspaces
- standard: 5 roles (adds analyst, viewer) for growing teams
- enterprise: All 8 roles for full governance
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from .defaults import (
    SYSTEM_ROLES,
)

if TYPE_CHECKING:
    from .models import Role


class RBACProfile(str, Enum):
    """Available RBAC profile configurations."""

    LITE = "lite"
    STANDARD = "standard"
    ENTERPRISE = "enterprise"


@dataclass
class ProfileConfig:
    """Configuration for an RBAC profile."""

    name: str
    description: str
    roles: list[str]
    default_role: str
    features: set[str]


# Profile definitions
PROFILE_CONFIGS: dict[RBACProfile, ProfileConfig] = {
    RBACProfile.LITE: ProfileConfig(
        name="Lite",
        description="Simple 3-role setup for small teams and quick onboarding",
        roles=["owner", "admin", "member"],
        default_role="member",
        features={"basic_debates", "basic_workflows"},
    ),
    RBACProfile.STANDARD: ProfileConfig(
        name="Standard",
        description="5 roles with analysts and viewers for growing teams",
        roles=["owner", "admin", "member", "analyst", "viewer"],
        default_role="member",
        features={"basic_debates", "basic_workflows", "analytics", "read_only_access"},
    ),
    RBACProfile.ENTERPRISE: ProfileConfig(
        name="Enterprise",
        description="Full 8-role configuration with compliance and team leads",
        roles=[
            "owner",
            "admin",
            "compliance_officer",
            "debate_creator",
            "team_lead",
            "member",
            "analyst",
            "viewer",
        ],
        default_role="member",
        features={
            "basic_debates",
            "basic_workflows",
            "analytics",
            "read_only_access",
            "compliance",
            "team_management",
            "audit_trails",
        },
    ),
}


def get_profile_config(profile: RBACProfile | str) -> ProfileConfig:
    """
    Get the configuration for an RBAC profile.

    Args:
        profile: Profile name or enum value

    Returns:
        ProfileConfig for the requested profile

    Raises:
        ValueError: If profile is not recognized
    """
    if isinstance(profile, str):
        try:
            profile = RBACProfile(profile.lower())
        except ValueError:
            valid = ", ".join(p.value for p in RBACProfile)
            raise ValueError(f"Unknown profile '{profile}'. Valid profiles: {valid}")

    return PROFILE_CONFIGS[profile]


def get_profile_roles(profile: RBACProfile | str) -> dict[str, "Role"]:
    """
    Get the roles available in a profile.

    Args:
        profile: Profile name or enum value

    Returns:
        Dictionary of role name to Role object
    """
    config = get_profile_config(profile)
    return {name: SYSTEM_ROLES[name] for name in config.roles if name in SYSTEM_ROLES}


def get_default_role(profile: RBACProfile | str) -> "Role":
    """
    Get the default role for new users in a profile.

    Args:
        profile: Profile name or enum value

    Returns:
        Default Role for the profile
    """
    config = get_profile_config(profile)
    return SYSTEM_ROLES[config.default_role]


def get_available_roles_for_assignment(
    profile: RBACProfile | str,
    assigner_role: str,
) -> list[str]:
    """
    Get roles that a user with the given role can assign.

    Args:
        profile: Current profile
        assigner_role: Role of the user trying to assign

    Returns:
        List of role names that can be assigned
    """
    config = get_profile_config(profile)
    available_roles = config.roles

    # Role assignment rules:
    # - owner: can assign all roles except owner
    # - admin: can assign member, analyst, viewer
    # - others: cannot assign roles

    if assigner_role == "owner":
        return [r for r in available_roles if r != "owner"]
    elif assigner_role == "admin":
        return [r for r in available_roles if r in ("member", "analyst", "viewer")]
    else:
        return []


def can_upgrade_profile(current: RBACProfile | str, target: RBACProfile | str) -> bool:
    """
    Check if a profile upgrade is valid.

    Args:
        current: Current profile
        target: Target profile

    Returns:
        True if upgrade is valid
    """
    order = [RBACProfile.LITE, RBACProfile.STANDARD, RBACProfile.ENTERPRISE]

    if isinstance(current, str):
        current = RBACProfile(current.lower())
    if isinstance(target, str):
        target = RBACProfile(target.lower())

    current_idx = order.index(current)
    target_idx = order.index(target)

    # Can upgrade to higher tier or stay same
    return target_idx >= current_idx


def get_migration_plan(
    current: RBACProfile | str,
    target: RBACProfile | str,
) -> dict[str, list[str]]:
    """
    Get the roles that need to be added/kept during profile migration.

    Args:
        current: Current profile
        target: Target profile

    Returns:
        Dictionary with 'add' and 'keep' role lists
    """
    current_config = get_profile_config(current)
    target_config = get_profile_config(target)

    current_roles = set(current_config.roles)
    target_roles = set(target_config.roles)

    return {
        "keep": list(current_roles & target_roles),
        "add": list(target_roles - current_roles),
        "remove": list(current_roles - target_roles),
    }


# Convenience mappings for lite profile
LITE_ROLE_DESCRIPTIONS = {
    "owner": "Full control - billing, users, all settings",
    "admin": "Manage users and debates - no billing access",
    "member": "Create and run debates - standard access",
}


def get_lite_role_summary() -> list[dict[str, str]]:
    """
    Get a summary of lite profile roles for UI display.

    Returns:
        List of role summaries with name, display_name, and description
    """
    return [
        {
            "name": "owner",
            "display_name": "Owner",
            "description": LITE_ROLE_DESCRIPTIONS["owner"],
        },
        {
            "name": "admin",
            "display_name": "Admin",
            "description": LITE_ROLE_DESCRIPTIONS["admin"],
        },
        {
            "name": "member",
            "display_name": "Member",
            "description": LITE_ROLE_DESCRIPTIONS["member"],
        },
    ]
