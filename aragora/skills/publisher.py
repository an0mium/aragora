"""
Skill Publisher.

Validates and publishes skills to the marketplace with:
- Manifest validation
- Security scanning
- Version management
- Dependency resolution

Usage:
    from aragora.skills.publisher import SkillPublisher

    publisher = SkillPublisher()

    # Validate a skill before publishing
    result = await publisher.validate(skill)

    # Publish a skill
    listing = await publisher.publish(skill, author_id="user-1")

    # Publish a new version
    listing = await publisher.publish_version(skill_id, skill, changelog="Bug fixes")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .base import Skill, SkillCapability, SkillManifest
from .marketplace import (
    SkillCategory,
    SkillListing,
    SkillMarketplace,
    SkillTier,
    get_marketplace,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """A validation issue found during skill validation."""

    severity: str  # error, warning, info
    code: str
    message: str
    field: Optional[str] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationResult:
    """Result of skill validation."""

    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    manifest: Optional[SkillManifest] = None

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get error-level issues."""
        return [i for i in self.issues if i.severity == "error"]

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return any(i.severity == "error" for i in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "is_valid": self.is_valid,
            "issues": [i.to_dict() for i in self.issues],
            "warnings": self.warnings,
            "error_count": len(self.errors),
        }


class SkillPublisher:
    """
    Validates and publishes skills to the marketplace.

    Performs comprehensive validation including:
    - Manifest completeness and correctness
    - Version format validation (semver)
    - Security capability checks
    - Dependency resolution
    """

    # Dangerous capabilities that require extra verification
    SENSITIVE_CAPABILITIES = {
        SkillCapability.SHELL_EXECUTION,
        SkillCapability.CODE_EXECUTION,
        SkillCapability.WRITE_LOCAL,
        SkillCapability.WRITE_DATABASE,
        SkillCapability.NETWORK,
    }

    # Reserved skill name prefixes
    RESERVED_PREFIXES = ["aragora", "system", "admin", "internal"]

    def __init__(self, marketplace: Optional[SkillMarketplace] = None):
        """Initialize the publisher."""
        self._marketplace = marketplace or get_marketplace()

    # ==========================================================================
    # Validation
    # ==========================================================================

    async def validate(self, skill: Skill) -> ValidationResult:
        """
        Validate a skill for publishing.

        Args:
            skill: Skill instance to validate

        Returns:
            ValidationResult with any issues found
        """
        issues: List[ValidationIssue] = []
        manifest = skill.manifest

        # Validate manifest fields
        issues.extend(self._validate_manifest(manifest))

        # Validate version format
        issues.extend(self._validate_version(manifest.version))

        # Validate capabilities
        issues.extend(self._validate_capabilities(manifest))

        # Validate input schema
        issues.extend(self._validate_input_schema(manifest))

        # Check for sensitive capabilities
        issues.extend(self._check_sensitive_capabilities(manifest))

        # Check reserved names
        issues.extend(self._check_reserved_names(manifest))

        is_valid = not any(i.severity == "error" for i in issues)
        warnings = [i.message for i in issues if i.severity == "warning"]

        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            warnings=warnings,
            manifest=manifest if is_valid else None,
        )

    def _validate_manifest(self, manifest: SkillManifest) -> List[ValidationIssue]:
        """Validate manifest required fields."""
        issues = []

        # Name validation
        if not manifest.name:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="MISSING_NAME",
                    message="Skill name is required",
                    field="name",
                )
            )
        elif not re.match(r"^[a-z][a-z0-9_-]*$", manifest.name):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="INVALID_NAME",
                    message="Skill name must be lowercase, start with a letter, and contain only a-z, 0-9, -, _",
                    field="name",
                    suggestion="Use a name like 'my-skill' or 'web_search'",
                )
            )
        elif len(manifest.name) > 50:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="NAME_TOO_LONG",
                    message="Skill name must be 50 characters or less",
                    field="name",
                )
            )

        # Version validation
        if not manifest.version:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="MISSING_VERSION",
                    message="Skill version is required",
                    field="version",
                )
            )

        # Description
        if not manifest.description:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="MISSING_DESCRIPTION",
                    message="Skill description is recommended for discoverability",
                    field="description",
                    suggestion="Add a clear description of what the skill does",
                )
            )
        elif len(manifest.description) > 1000:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="DESCRIPTION_TOO_LONG",
                    message="Description exceeds 1000 characters and may be truncated",
                    field="description",
                )
            )

        # Capabilities
        if not manifest.capabilities:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="NO_CAPABILITIES",
                    message="Skill declares no capabilities",
                    field="capabilities",
                    suggestion="Declare the capabilities your skill uses",
                )
            )

        # Author
        if not manifest.author:
            issues.append(
                ValidationIssue(
                    severity="info",
                    code="MISSING_AUTHOR",
                    message="Author information is helpful for users",
                    field="author",
                )
            )

        return issues

    def _validate_version(self, version: str) -> List[ValidationIssue]:
        """Validate semantic version format."""
        issues = []

        # Semver pattern: MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
        semver_pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$"

        if not re.match(semver_pattern, version):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="INVALID_VERSION",
                    message=f"Version '{version}' is not valid semver format",
                    field="version",
                    suggestion="Use semver format: MAJOR.MINOR.PATCH (e.g., 1.0.0)",
                )
            )

        return issues

    def _validate_capabilities(self, manifest: SkillManifest) -> List[ValidationIssue]:
        """Validate declared capabilities."""
        issues: List[ValidationIssue] = []

        for capability in manifest.capabilities:
            if not isinstance(capability, SkillCapability):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="INVALID_CAPABILITY",
                        message=f"Unknown capability: {capability}",
                        field="capabilities",
                    )
                )

        return issues

    def _validate_input_schema(self, manifest: SkillManifest) -> List[ValidationIssue]:
        """Validate input schema structure."""
        issues: List[ValidationIssue] = []
        schema = manifest.input_schema

        if not schema:
            return issues

        for field_name, field_spec in schema.items():
            if not isinstance(field_spec, dict):
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="INVALID_SCHEMA_FIELD",
                        message=f"Field '{field_name}' should be an object with type information",
                        field="input_schema",
                    )
                )
                continue

            if "type" not in field_spec:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="MISSING_FIELD_TYPE",
                        message=f"Field '{field_name}' should specify a type",
                        field="input_schema",
                    )
                )

        return issues

    def _check_sensitive_capabilities(self, manifest: SkillManifest) -> List[ValidationIssue]:
        """Check for sensitive capabilities that need review."""
        issues = []

        sensitive = set(manifest.capabilities) & self.SENSITIVE_CAPABILITIES
        if sensitive:
            cap_names = ", ".join(c.value for c in sensitive)
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="SENSITIVE_CAPABILITIES",
                    message=f"Skill uses sensitive capabilities: {cap_names}",
                    field="capabilities",
                    suggestion="Skills with sensitive capabilities may require manual review",
                )
            )

        return issues

    def _check_reserved_names(self, manifest: SkillManifest) -> List[ValidationIssue]:
        """Check for reserved name prefixes."""
        issues = []

        for prefix in self.RESERVED_PREFIXES:
            if manifest.name.startswith(prefix):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="RESERVED_NAME",
                        message=f"Skill name cannot start with reserved prefix: {prefix}",
                        field="name",
                    )
                )
                break

        return issues

    # ==========================================================================
    # Publishing
    # ==========================================================================

    async def publish(
        self,
        skill: Skill,
        author_id: str,
        author_name: str,
        category: SkillCategory = SkillCategory.CUSTOM,
        tier: SkillTier = SkillTier.FREE,
        changelog: str = "Initial release",
        **kwargs: Any,
    ) -> Tuple[bool, Optional[SkillListing], List[ValidationIssue]]:
        """
        Publish a skill to the marketplace.

        Args:
            skill: Skill to publish
            author_id: Publisher's user ID
            author_name: Publisher's display name
            category: Skill category
            tier: Access tier
            changelog: Version changelog
            **kwargs: Additional listing metadata

        Returns:
            Tuple of (success, listing, issues)
        """
        # Validate first
        validation = await self.validate(skill)
        if not validation.is_valid:
            return False, None, validation.issues

        # Publish to marketplace
        try:
            listing = await self._marketplace.publish(
                skill=skill,
                author_id=author_id,
                author_name=author_name,
                category=category,
                tier=tier,
                changelog=changelog,
                **kwargs,
            )

            logger.info(f"Published skill: {listing.skill_id} v{listing.current_version}")
            return True, listing, validation.issues

        except Exception as e:
            logger.error(f"Failed to publish skill: {e}")
            return (
                False,
                None,
                [
                    ValidationIssue(
                        severity="error",
                        code="PUBLISH_FAILED",
                        message=f"Failed to publish: {str(e)}",
                    )
                ],
            )

    async def publish_version(
        self,
        skill_id: str,
        skill: Skill,
        author_id: str,
        changelog: str = "",
    ) -> Tuple[bool, Optional[SkillListing], List[ValidationIssue]]:
        """
        Publish a new version of an existing skill.

        Args:
            skill_id: Existing skill ID
            skill: Updated skill instance
            author_id: Author ID (for authorization)
            changelog: Version changelog

        Returns:
            Tuple of (success, listing, issues)
        """
        # Check existing skill
        existing = await self._marketplace.get_skill(skill_id)
        if not existing:
            return (
                False,
                None,
                [
                    ValidationIssue(
                        severity="error",
                        code="SKILL_NOT_FOUND",
                        message=f"Skill not found: {skill_id}",
                    )
                ],
            )

        # Verify ownership
        if existing.author_id != author_id:
            return (
                False,
                None,
                [
                    ValidationIssue(
                        severity="error",
                        code="NOT_AUTHORIZED",
                        message="You are not authorized to update this skill",
                    )
                ],
            )

        # Validate new version is higher
        new_version = skill.manifest.version
        if not self._is_higher_version(new_version, existing.current_version):
            return (
                False,
                None,
                [
                    ValidationIssue(
                        severity="error",
                        code="VERSION_NOT_HIGHER",
                        message=f"New version {new_version} must be higher than current {existing.current_version}",
                        field="version",
                    )
                ],
            )

        # Validate the skill
        validation = await self.validate(skill)
        if not validation.is_valid:
            return False, None, validation.issues

        # Publish update
        try:
            listing = await self._marketplace.publish(
                skill=skill,
                author_id=author_id,
                author_name=existing.author_name,
                category=existing.category,
                tier=existing.tier,
                changelog=changelog,
            )

            logger.info(f"Published new version: {skill_id} v{new_version}")
            return True, listing, validation.issues

        except Exception as e:
            logger.error(f"Failed to publish version: {e}")
            return (
                False,
                None,
                [
                    ValidationIssue(
                        severity="error",
                        code="PUBLISH_FAILED",
                        message=f"Failed to publish: {str(e)}",
                    )
                ],
            )

    def _is_higher_version(self, new_version: str, current_version: str) -> bool:
        """Check if new version is higher than current."""
        try:
            new_parts = [int(p) for p in new_version.split("-")[0].split(".")]
            current_parts = [int(p) for p in current_version.split("-")[0].split(".")]

            # Pad to same length
            while len(new_parts) < 3:
                new_parts.append(0)
            while len(current_parts) < 3:
                current_parts.append(0)

            return new_parts > current_parts

        except (ValueError, IndexError):
            # If we can't parse, assume new is higher
            return True

    # ==========================================================================
    # Deprecation
    # ==========================================================================

    async def deprecate(
        self,
        skill_id: str,
        author_id: str,
        replacement_skill_id: Optional[str] = None,
        message: str = "",
    ) -> bool:
        """
        Mark a skill as deprecated.

        Args:
            skill_id: Skill to deprecate
            author_id: Author ID (for authorization)
            replacement_skill_id: Optional replacement skill
            message: Deprecation message

        Returns:
            True if deprecated successfully
        """
        existing = await self._marketplace.get_skill(skill_id)
        if not existing:
            return False

        if existing.author_id != author_id:
            logger.warning(f"Unauthorized deprecation attempt for {skill_id}")
            return False

        # Update in database (would need to add this to marketplace)
        logger.info(f"Deprecated skill: {skill_id}")
        return True
