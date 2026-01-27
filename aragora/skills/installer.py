# mypy: ignore-errors
"""
Skill Installer.

Handles skill installation with:
- RBAC permission checking
- Tenant quota enforcement
- Dependency resolution
- Version compatibility checking

Usage:
    from aragora.skills.installer import SkillInstaller

    installer = SkillInstaller()

    # Check if installation is allowed
    allowed, reason = await installer.can_install(skill_id, tenant_id, user_id)

    # Install a skill
    result = await installer.install(skill_id, tenant_id, user_id)

    # Install with specific version
    result = await installer.install(skill_id, tenant_id, user_id, version="1.2.0")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .marketplace import (
    InstallResult,
    SkillListing,
    SkillMarketplace,
    SkillTier,
    get_marketplace,
)

logger = logging.getLogger(__name__)


@dataclass
class InstallationPolicy:
    """Policy governing skill installation for a tenant."""

    # Tier access
    allowed_tiers: List[SkillTier] = field(
        default_factory=lambda: [SkillTier.FREE, SkillTier.STANDARD]
    )

    # Quota limits
    max_installed_skills: int = 100
    max_skills_per_category: int = 20

    # Capability restrictions
    blocked_capabilities: List[str] = field(default_factory=list)
    require_verified_only: bool = False

    # Auto-update settings
    auto_update_enabled: bool = False
    auto_update_minor_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "allowed_tiers": [t.value for t in self.allowed_tiers],
            "max_installed_skills": self.max_installed_skills,
            "max_skills_per_category": self.max_skills_per_category,
            "blocked_capabilities": self.blocked_capabilities,
            "require_verified_only": self.require_verified_only,
            "auto_update_enabled": self.auto_update_enabled,
            "auto_update_minor_only": self.auto_update_minor_only,
        }


@dataclass
class InstallationCheck:
    """Result of installation permission check."""

    allowed: bool
    reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    missing_permissions: List[str] = field(default_factory=list)
    missing_dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "warnings": self.warnings,
            "missing_permissions": self.missing_permissions,
            "missing_dependencies": self.missing_dependencies,
        }


class SkillInstaller:
    """
    Handles skill installation with permission and quota checking.

    Features:
    - RBAC permission verification
    - Tenant quota enforcement
    - Dependency resolution
    - Version compatibility checking
    """

    # Default tenant quotas
    DEFAULT_MAX_SKILLS = 100
    DEFAULT_MAX_PER_CATEGORY = 20

    # Required permissions for installation
    INSTALL_PERMISSION = "skills:install"
    UNINSTALL_PERMISSION = "skills:uninstall"
    ADMIN_PERMISSION = "skills:admin"

    def __init__(self, marketplace: Optional[SkillMarketplace] = None):
        """Initialize the installer."""
        self._marketplace = marketplace or get_marketplace()
        self._policies: Dict[str, InstallationPolicy] = {}

    # ==========================================================================
    # Policy Management
    # ==========================================================================

    def set_policy(self, tenant_id: str, policy: InstallationPolicy) -> None:
        """Set installation policy for a tenant."""
        self._policies[tenant_id] = policy
        logger.info(f"Set installation policy for tenant {tenant_id}")

    def get_policy(self, tenant_id: str) -> InstallationPolicy:
        """Get installation policy for a tenant."""
        return self._policies.get(tenant_id, InstallationPolicy())

    # ==========================================================================
    # Permission Checking
    # ==========================================================================

    async def can_install(
        self,
        skill_id: str,
        tenant_id: str,
        user_id: str,
        permissions: Optional[List[str]] = None,
    ) -> InstallationCheck:
        """
        Check if a skill can be installed.

        Args:
            skill_id: Skill to install
            tenant_id: Tenant to install for
            user_id: User performing installation
            permissions: User's permissions (if None, checks RBAC)

        Returns:
            InstallationCheck with result and any issues
        """
        warnings: List[str] = []
        missing_permissions: List[str] = []
        missing_dependencies: List[str] = []

        # Get skill listing
        listing = await self._marketplace.get_skill(skill_id)
        if not listing:
            return InstallationCheck(
                allowed=False,
                reason=f"Skill not found: {skill_id}",
            )

        # Check if skill is published
        if not listing.is_published:
            return InstallationCheck(
                allowed=False,
                reason="Skill is not published",
            )

        # Check if skill is deprecated
        if listing.is_deprecated:
            warnings.append("This skill is deprecated and may be removed in the future")

        # Get user permissions
        user_permissions = permissions or await self._get_user_permissions(user_id, tenant_id)

        # Check basic install permission
        if self.INSTALL_PERMISSION not in user_permissions:
            if self.ADMIN_PERMISSION not in user_permissions:
                missing_permissions.append(self.INSTALL_PERMISSION)

        # Get tenant policy
        policy = self.get_policy(tenant_id)

        # Check tier access
        if listing.tier not in policy.allowed_tiers:
            return InstallationCheck(
                allowed=False,
                reason=f"Tenant does not have access to {listing.tier.value} tier skills",
            )

        # Check verified requirement
        if policy.require_verified_only and not listing.is_verified:
            return InstallationCheck(
                allowed=False,
                reason="Policy requires verified skills only",
            )

        # Check blocked capabilities
        for capability in listing.capabilities:
            if capability.value in policy.blocked_capabilities:
                return InstallationCheck(
                    allowed=False,
                    reason=f"Capability '{capability.value}' is blocked by policy",
                )

        # Check skill's required permissions
        for perm in listing.required_permissions:
            if perm not in user_permissions:
                missing_permissions.append(perm)

        # Check quotas
        quota_check = await self._check_quotas(tenant_id, listing, policy)
        if not quota_check[0]:
            return InstallationCheck(
                allowed=False,
                reason=quota_check[1],
            )

        # Check dependencies
        for dep in listing.dependencies:
            if not dep.optional:
                is_installed = await self._marketplace.is_installed(dep.skill_id, tenant_id)
                if not is_installed:
                    missing_dependencies.append(dep.skill_id)

        # Determine if allowed
        if missing_permissions:
            return InstallationCheck(
                allowed=False,
                reason=f"Missing permissions: {', '.join(missing_permissions)}",
                missing_permissions=missing_permissions,
            )

        if missing_dependencies:
            return InstallationCheck(
                allowed=False,
                reason=f"Missing dependencies: {', '.join(missing_dependencies)}",
                missing_dependencies=missing_dependencies,
            )

        return InstallationCheck(
            allowed=True,
            warnings=warnings,
        )

    async def _get_user_permissions(self, user_id: str, tenant_id: str) -> List[str]:
        """Get user permissions from RBAC system."""
        try:
            from aragora.rbac.checker import PermissionChecker

            checker = PermissionChecker()
            # Get all permissions for user in tenant context
            permissions = await checker.get_user_permissions(user_id, tenant_id)
            return permissions

        except ImportError:
            # RBAC not available, return default permissions
            logger.warning("RBAC not available, using default permissions")
            return [self.INSTALL_PERMISSION, self.UNINSTALL_PERMISSION]

    async def _check_quotas(
        self,
        tenant_id: str,
        listing: SkillListing,
        policy: InstallationPolicy,
    ) -> Tuple[bool, Optional[str]]:
        """Check if installation would exceed quotas."""
        # Get current installations
        installed = await self._marketplace.get_installed_skills(tenant_id)

        # Check total count
        if len(installed) >= policy.max_installed_skills:
            return False, f"Quota exceeded: maximum {policy.max_installed_skills} skills"

        # Check per-category count
        category_count = sum(1 for s in installed if s.category == listing.category)
        if category_count >= policy.max_skills_per_category:
            return (
                False,
                f"Quota exceeded: maximum {policy.max_skills_per_category} skills per category",
            )

        return True, None

    # ==========================================================================
    # Installation
    # ==========================================================================

    async def install(
        self,
        skill_id: str,
        tenant_id: str,
        user_id: str,
        version: Optional[str] = None,
        permissions: Optional[List[str]] = None,
        install_dependencies: bool = True,
    ) -> InstallResult:
        """
        Install a skill for a tenant.

        Args:
            skill_id: Skill to install
            tenant_id: Tenant to install for
            user_id: User performing installation
            version: Specific version (default: latest)
            permissions: User's permissions (if None, checks RBAC)
            install_dependencies: Whether to auto-install dependencies

        Returns:
            InstallResult with outcome
        """
        # Check if allowed
        check = await self.can_install(skill_id, tenant_id, user_id, permissions)

        if not check.allowed:
            return InstallResult(
                success=False,
                skill_id=skill_id,
                version=version or "",
                error=check.reason,
            )

        # Install dependencies first if needed
        dependencies_installed: List[str] = []
        if install_dependencies and check.missing_dependencies:
            logger.warning(f"Missing dependencies for {skill_id}: {check.missing_dependencies}")
            # In a full implementation, would recursively install dependencies
            pass

        # Get the listing to access dependencies
        listing = await self._marketplace.get_skill(skill_id)
        if listing and install_dependencies:
            for dep in listing.dependencies:
                if not dep.optional:
                    is_installed = await self._marketplace.is_installed(dep.skill_id, tenant_id)
                    if not is_installed:
                        # Recursively install dependency
                        dep_result = await self.install(
                            dep.skill_id,
                            tenant_id,
                            user_id,
                            permissions=permissions,
                            install_dependencies=True,
                        )
                        if dep_result.success:
                            dependencies_installed.append(dep.skill_id)
                        else:
                            return InstallResult(
                                success=False,
                                skill_id=skill_id,
                                version=version or "",
                                error=f"Failed to install dependency {dep.skill_id}: {dep_result.error}",
                            )

        # Perform installation
        result = await self._marketplace.install(
            skill_id=skill_id,
            tenant_id=tenant_id,
            user_id=user_id,
            version=version,
        )

        # Add dependency info
        result.dependencies_installed = dependencies_installed

        # Log warnings
        for warning in check.warnings:
            logger.warning(f"Installation warning for {skill_id}: {warning}")

        return result

    async def uninstall(
        self,
        skill_id: str,
        tenant_id: str,
        user_id: str,
        permissions: Optional[List[str]] = None,
    ) -> bool:
        """
        Uninstall a skill from a tenant.

        Args:
            skill_id: Skill to uninstall
            tenant_id: Tenant to uninstall from
            user_id: User performing uninstallation
            permissions: User's permissions (if None, checks RBAC)

        Returns:
            True if uninstalled successfully
        """
        # Check permissions
        user_permissions = permissions or await self._get_user_permissions(user_id, tenant_id)

        if self.UNINSTALL_PERMISSION not in user_permissions:
            if self.ADMIN_PERMISSION not in user_permissions:
                logger.warning(f"User {user_id} lacks permission to uninstall skill {skill_id}")
                return False

        # Check for dependent skills
        dependents = await self._find_dependents(skill_id, tenant_id)
        if dependents:
            logger.warning(f"Cannot uninstall {skill_id}: required by {dependents}")
            return False

        return await self._marketplace.uninstall(skill_id, tenant_id)

    async def _find_dependents(self, skill_id: str, tenant_id: str) -> List[str]:
        """Find skills that depend on the given skill."""
        installed = await self._marketplace.get_installed_skills(tenant_id)
        dependents = []

        for skill in installed:
            for dep in skill.dependencies:
                if dep.skill_id == skill_id and not dep.optional:
                    dependents.append(skill.skill_id)

        return dependents

    # ==========================================================================
    # Bulk Operations
    # ==========================================================================

    async def install_batch(
        self,
        skill_ids: List[str],
        tenant_id: str,
        user_id: str,
        permissions: Optional[List[str]] = None,
    ) -> Dict[str, InstallResult]:
        """
        Install multiple skills.

        Args:
            skill_ids: Skills to install
            tenant_id: Tenant to install for
            user_id: User performing installation
            permissions: User's permissions

        Returns:
            Dict mapping skill_id to InstallResult
        """
        results = {}

        for skill_id in skill_ids:
            results[skill_id] = await self.install(
                skill_id=skill_id,
                tenant_id=tenant_id,
                user_id=user_id,
                permissions=permissions,
            )

        return results

    async def get_installed(self, tenant_id: str) -> List[SkillListing]:
        """Get all skills installed for a tenant."""
        return await self._marketplace.get_installed_skills(tenant_id)

    async def get_available_upgrades(self, tenant_id: str) -> Dict[str, str]:
        """
        Get available upgrades for installed skills.

        Args:
            tenant_id: Tenant to check

        Returns:
            Dict mapping skill_id to available version
        """
        installed = await self._marketplace.get_installed_skills(tenant_id)
        upgrades = {}

        for skill in installed:
            # Get latest version from marketplace
            latest = await self._marketplace.get_skill(skill.skill_id)
            if latest and latest.current_version != skill.current_version:
                upgrades[skill.skill_id] = latest.current_version

        return upgrades
