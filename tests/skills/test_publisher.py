"""
Tests for aragora.skills.publisher module.

Covers:
- ValidationIssue dataclass
- ValidationResult dataclass
- SkillPublisher.validate (manifest validation)
- SkillPublisher.publish (full publish lifecycle)
- SkillPublisher.publish_version (version management)
- SkillPublisher.deprecate (deprecation flow)
- SkillPublisher._is_higher_version (version comparison)
- Sensitive capability detection
- Reserved name checking
- Input schema validation
- Error handling and edge cases
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.skills.base import (
    Skill,
    SkillCapability,
    SkillContext,
    SkillManifest,
    SkillResult,
    SkillStatus,
)
from aragora.skills.marketplace import (
    SkillCategory,
    SkillListing,
    SkillMarketplace,
    SkillTier,
)
from aragora.skills.publisher import (
    SkillPublisher,
    ValidationIssue,
    ValidationResult,
)


# =============================================================================
# Helper Skill Implementations
# =============================================================================


class ValidSkill(Skill):
    """A valid skill for testing the publisher."""

    def __init__(
        self,
        name: str = "my-test-skill",
        version: str = "1.0.0",
        description: str = "A test skill for unit tests",
        author: str = "test-author",
        capabilities: list[SkillCapability] | None = None,
        input_schema: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ):
        self._name = name
        self._version = version
        self._description = description
        self._author = author
        self._capabilities = (
            capabilities if capabilities is not None else [SkillCapability.EXTERNAL_API]
        )
        self._input_schema = (
            input_schema if input_schema is not None else {"query": {"type": "string"}}
        )
        self._tags = tags or []

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name=self._name,
            version=self._version,
            description=self._description,
            author=self._author,
            capabilities=self._capabilities,
            input_schema=self._input_schema,
            tags=self._tags,
        )

    async def execute(self, input_data: dict[str, Any], context: SkillContext) -> SkillResult:
        return SkillResult.create_success({"result": "ok"})


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_marketplace():
    """Create a mock marketplace for testing without SQLite."""
    marketplace = AsyncMock(spec=SkillMarketplace)
    return marketplace


@pytest.fixture
def publisher(mock_marketplace):
    """Create a SkillPublisher with a mocked marketplace."""
    return SkillPublisher(marketplace=mock_marketplace)


@pytest.fixture
def valid_skill():
    """Create a valid skill instance."""
    return ValidSkill()


@pytest.fixture
def valid_listing():
    """Create a valid SkillListing for mock returns."""
    return SkillListing(
        skill_id="user-1:my-test-skill",
        name="my-test-skill",
        description="A test skill",
        author_id="user-1",
        author_name="Test User",
        category=SkillCategory.CUSTOM,
        tier=SkillTier.FREE,
        current_version="1.0.0",
        is_published=True,
    )


# =============================================================================
# ValidationIssue Tests
# =============================================================================


class TestValidationIssue:
    """Tests for the ValidationIssue dataclass."""

    def test_create_basic_issue(self):
        """Test creating a basic validation issue."""
        issue = ValidationIssue(
            severity="error",
            code="TEST_CODE",
            message="Something went wrong",
        )
        assert issue.severity == "error"
        assert issue.code == "TEST_CODE"
        assert issue.message == "Something went wrong"
        assert issue.field is None
        assert issue.suggestion is None

    def test_create_issue_with_all_fields(self):
        """Test creating an issue with all optional fields."""
        issue = ValidationIssue(
            severity="warning",
            code="WARN_001",
            message="Might be a problem",
            field="name",
            suggestion="Try changing the name",
        )
        assert issue.field == "name"
        assert issue.suggestion == "Try changing the name"

    def test_to_dict(self):
        """Test serializing a validation issue to dictionary."""
        issue = ValidationIssue(
            severity="error",
            code="E001",
            message="Error message",
            field="version",
            suggestion="Use semver",
        )
        result = issue.to_dict()
        assert result == {
            "severity": "error",
            "code": "E001",
            "message": "Error message",
            "field": "version",
            "suggestion": "Use semver",
        }

    def test_to_dict_with_none_fields(self):
        """Test serializing with None optional fields."""
        issue = ValidationIssue(severity="info", code="I001", message="Info")
        result = issue.to_dict()
        assert result["field"] is None
        assert result["suggestion"] is None


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult:
    """Tests for the ValidationResult dataclass."""

    def test_valid_result(self):
        """Test a valid result with no errors."""
        manifest = SkillManifest(name="test", version="1.0.0", capabilities=[], input_schema={})
        result = ValidationResult(is_valid=True, manifest=manifest)
        assert result.is_valid is True
        assert result.errors == []
        assert result.has_errors is False
        assert result.manifest is manifest

    def test_invalid_result_with_errors(self):
        """Test a result with error-level issues."""
        issues = [
            ValidationIssue(severity="error", code="E001", message="Error 1"),
            ValidationIssue(severity="warning", code="W001", message="Warning 1"),
            ValidationIssue(severity="error", code="E002", message="Error 2"),
        ]
        result = ValidationResult(is_valid=False, issues=issues)
        assert result.is_valid is False
        assert result.has_errors is True
        assert len(result.errors) == 2

    def test_errors_property_filters_correctly(self):
        """Test that errors property only returns error-severity issues."""
        issues = [
            ValidationIssue(severity="error", code="E001", message="Error"),
            ValidationIssue(severity="warning", code="W001", message="Warning"),
            ValidationIssue(severity="info", code="I001", message="Info"),
        ]
        result = ValidationResult(is_valid=False, issues=issues)
        errors = result.errors
        assert len(errors) == 1
        assert errors[0].code == "E001"

    def test_to_dict(self):
        """Test serialization of a validation result."""
        issues = [
            ValidationIssue(severity="error", code="E001", message="Error"),
            ValidationIssue(severity="warning", code="W001", message="Warning"),
        ]
        result = ValidationResult(
            is_valid=False,
            issues=issues,
            warnings=["Warning"],
        )
        d = result.to_dict()
        assert d["is_valid"] is False
        assert len(d["issues"]) == 2
        assert d["error_count"] == 1
        assert d["warnings"] == ["Warning"]

    def test_empty_result(self):
        """Test result with no issues."""
        result = ValidationResult(is_valid=True)
        assert result.issues == []
        assert result.warnings == []
        assert result.errors == []
        assert result.has_errors is False


# =============================================================================
# Manifest Validation Tests
# =============================================================================


class TestManifestValidation:
    """Tests for SkillPublisher._validate_manifest and validate()."""

    @pytest.mark.asyncio
    async def test_valid_manifest(self, publisher, valid_skill):
        """Test that a fully valid manifest passes validation."""
        result = await publisher.validate(valid_skill)
        assert result.is_valid is True
        assert not result.has_errors
        assert result.manifest is not None
        assert result.manifest.name == "my-test-skill"

    @pytest.mark.asyncio
    async def test_missing_name(self, publisher):
        """Test validation fails when name is empty."""
        skill = ValidSkill(name="")
        result = await publisher.validate(skill)
        assert result.is_valid is False
        error_codes = [i.code for i in result.errors]
        assert "MISSING_NAME" in error_codes

    @pytest.mark.asyncio
    async def test_invalid_name_uppercase(self, publisher):
        """Test validation fails with uppercase characters in name."""
        skill = ValidSkill(name="MySkill")
        result = await publisher.validate(skill)
        assert result.is_valid is False
        error_codes = [i.code for i in result.errors]
        assert "INVALID_NAME" in error_codes

    @pytest.mark.asyncio
    async def test_invalid_name_starts_with_digit(self, publisher):
        """Test validation fails when name starts with a digit."""
        skill = ValidSkill(name="1skill")
        result = await publisher.validate(skill)
        assert result.is_valid is False
        error_codes = [i.code for i in result.errors]
        assert "INVALID_NAME" in error_codes

    @pytest.mark.asyncio
    async def test_invalid_name_special_chars(self, publisher):
        """Test validation fails with special characters in name."""
        skill = ValidSkill(name="my.skill@v2")
        result = await publisher.validate(skill)
        assert result.is_valid is False
        error_codes = [i.code for i in result.errors]
        assert "INVALID_NAME" in error_codes

    @pytest.mark.asyncio
    async def test_valid_name_with_hyphens_and_underscores(self, publisher):
        """Test names with hyphens and underscores are accepted."""
        skill = ValidSkill(name="my-test_skill-2")
        result = await publisher.validate(skill)
        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_name_too_long(self, publisher):
        """Test validation fails when name exceeds 50 characters."""
        long_name = "a" * 51
        skill = ValidSkill(name=long_name)
        result = await publisher.validate(skill)
        assert result.is_valid is False
        error_codes = [i.code for i in result.errors]
        assert "NAME_TOO_LONG" in error_codes

    @pytest.mark.asyncio
    async def test_name_exactly_50_chars(self, publisher):
        """Test that a name with exactly 50 characters is accepted."""
        name = "a" * 50
        skill = ValidSkill(name=name)
        result = await publisher.validate(skill)
        # Should not have NAME_TOO_LONG error
        error_codes = [i.code for i in result.errors]
        assert "NAME_TOO_LONG" not in error_codes

    @pytest.mark.asyncio
    async def test_missing_version(self, publisher):
        """Test validation fails when version is empty."""
        skill = ValidSkill(version="")
        result = await publisher.validate(skill)
        assert result.is_valid is False
        error_codes = [i.code for i in result.errors]
        assert "MISSING_VERSION" in error_codes

    @pytest.mark.asyncio
    async def test_missing_description_is_warning(self, publisher):
        """Test that missing description yields a warning, not an error."""
        skill = ValidSkill(description="")
        result = await publisher.validate(skill)
        # Missing description is a warning, not an error
        assert result.is_valid is True
        warning_codes = [i.code for i in result.issues if i.severity == "warning"]
        assert "MISSING_DESCRIPTION" in warning_codes

    @pytest.mark.asyncio
    async def test_description_too_long_is_warning(self, publisher):
        """Test that a long description yields a warning."""
        long_desc = "x" * 1001
        skill = ValidSkill(description=long_desc)
        result = await publisher.validate(skill)
        assert result.is_valid is True
        warning_codes = [i.code for i in result.issues if i.severity == "warning"]
        assert "DESCRIPTION_TOO_LONG" in warning_codes

    @pytest.mark.asyncio
    async def test_description_at_1000_chars_no_warning(self, publisher):
        """Test that description at exactly 1000 characters does not warn."""
        desc = "x" * 1000
        skill = ValidSkill(description=desc)
        result = await publisher.validate(skill)
        warning_codes = [i.code for i in result.issues if i.severity == "warning"]
        assert "DESCRIPTION_TOO_LONG" not in warning_codes

    @pytest.mark.asyncio
    async def test_no_capabilities_is_warning(self, publisher):
        """Test that declaring no capabilities yields a warning."""
        skill = ValidSkill(capabilities=[])
        result = await publisher.validate(skill)
        assert result.is_valid is True
        warning_codes = [i.code for i in result.issues if i.severity == "warning"]
        assert "NO_CAPABILITIES" in warning_codes

    @pytest.mark.asyncio
    async def test_missing_author_is_info(self, publisher):
        """Test that missing author yields an info-level issue."""
        skill = ValidSkill(author="")
        result = await publisher.validate(skill)
        assert result.is_valid is True
        info_codes = [i.code for i in result.issues if i.severity == "info"]
        assert "MISSING_AUTHOR" in info_codes


# =============================================================================
# Version Validation Tests
# =============================================================================


class TestVersionValidation:
    """Tests for SkillPublisher._validate_version."""

    @pytest.mark.asyncio
    async def test_valid_semver(self, publisher):
        """Test valid semantic version strings."""
        for version in ["0.0.1", "1.0.0", "2.13.45", "100.200.300"]:
            skill = ValidSkill(version=version)
            result = await publisher.validate(skill)
            error_codes = [i.code for i in result.errors]
            assert "INVALID_VERSION" not in error_codes, f"Version {version} should be valid"

    @pytest.mark.asyncio
    async def test_valid_semver_with_prerelease(self, publisher):
        """Test semver with prerelease tags."""
        for version in ["1.0.0-alpha", "1.0.0-beta.1", "2.0.0-rc.1"]:
            skill = ValidSkill(version=version)
            result = await publisher.validate(skill)
            error_codes = [i.code for i in result.errors]
            assert "INVALID_VERSION" not in error_codes, f"Version {version} should be valid"

    @pytest.mark.asyncio
    async def test_valid_semver_with_build(self, publisher):
        """Test semver with build metadata."""
        skill = ValidSkill(version="1.0.0+build.123")
        result = await publisher.validate(skill)
        error_codes = [i.code for i in result.errors]
        assert "INVALID_VERSION" not in error_codes

    @pytest.mark.asyncio
    async def test_valid_semver_with_prerelease_and_build(self, publisher):
        """Test semver with both prerelease and build metadata."""
        skill = ValidSkill(version="1.0.0-alpha+build.1")
        result = await publisher.validate(skill)
        error_codes = [i.code for i in result.errors]
        assert "INVALID_VERSION" not in error_codes

    @pytest.mark.asyncio
    async def test_invalid_version_two_parts(self, publisher):
        """Test that two-part version strings are rejected."""
        skill = ValidSkill(version="1.0")
        result = await publisher.validate(skill)
        error_codes = [i.code for i in result.errors]
        assert "INVALID_VERSION" in error_codes

    @pytest.mark.asyncio
    async def test_invalid_version_single_number(self, publisher):
        """Test that single number versions are rejected."""
        skill = ValidSkill(version="1")
        result = await publisher.validate(skill)
        error_codes = [i.code for i in result.errors]
        assert "INVALID_VERSION" in error_codes

    @pytest.mark.asyncio
    async def test_invalid_version_text(self, publisher):
        """Test that text-only versions are rejected."""
        skill = ValidSkill(version="latest")
        result = await publisher.validate(skill)
        error_codes = [i.code for i in result.errors]
        assert "INVALID_VERSION" in error_codes

    @pytest.mark.asyncio
    async def test_invalid_version_v_prefix(self, publisher):
        """Test that v-prefixed versions are rejected."""
        skill = ValidSkill(version="v1.0.0")
        result = await publisher.validate(skill)
        error_codes = [i.code for i in result.errors]
        assert "INVALID_VERSION" in error_codes

    @pytest.mark.asyncio
    async def test_invalid_version_four_parts(self, publisher):
        """Test that four-part version strings are rejected."""
        skill = ValidSkill(version="1.0.0.0")
        result = await publisher.validate(skill)
        error_codes = [i.code for i in result.errors]
        assert "INVALID_VERSION" in error_codes

    @pytest.mark.asyncio
    async def test_invalid_version_spaces(self, publisher):
        """Test that versions with spaces are rejected."""
        skill = ValidSkill(version="1 . 0 . 0")
        result = await publisher.validate(skill)
        error_codes = [i.code for i in result.errors]
        assert "INVALID_VERSION" in error_codes


# =============================================================================
# Capability Validation Tests
# =============================================================================


class TestCapabilityValidation:
    """Tests for SkillPublisher._validate_capabilities."""

    @pytest.mark.asyncio
    async def test_valid_capabilities(self, publisher):
        """Test that valid capabilities pass validation."""
        skill = ValidSkill(
            capabilities=[
                SkillCapability.EXTERNAL_API,
                SkillCapability.WEB_SEARCH,
                SkillCapability.LLM_INFERENCE,
            ]
        )
        result = await publisher.validate(skill)
        error_codes = [i.code for i in result.errors]
        assert "INVALID_CAPABILITY" not in error_codes

    @pytest.mark.asyncio
    async def test_all_skill_capabilities_accepted(self, publisher):
        """Test that all defined SkillCapability enum values are accepted."""
        all_caps = list(SkillCapability)
        skill = ValidSkill(capabilities=all_caps)
        result = await publisher.validate(skill)
        error_codes = [i.code for i in result.errors]
        assert "INVALID_CAPABILITY" not in error_codes


# =============================================================================
# Input Schema Validation Tests
# =============================================================================


class TestInputSchemaValidation:
    """Tests for SkillPublisher._validate_input_schema."""

    @pytest.mark.asyncio
    async def test_valid_input_schema(self, publisher):
        """Test a properly structured input schema passes."""
        skill = ValidSkill(
            input_schema={
                "query": {"type": "string", "required": True},
                "limit": {"type": "number"},
            }
        )
        result = await publisher.validate(skill)
        warning_codes = [i.code for i in result.issues if i.severity == "warning"]
        assert "INVALID_SCHEMA_FIELD" not in warning_codes
        assert "MISSING_FIELD_TYPE" not in warning_codes

    @pytest.mark.asyncio
    async def test_empty_input_schema(self, publisher):
        """Test that an empty input schema is acceptable."""
        skill = ValidSkill(input_schema={})
        result = await publisher.validate(skill)
        warning_codes = [i.code for i in result.issues if i.severity == "warning"]
        assert "INVALID_SCHEMA_FIELD" not in warning_codes

    @pytest.mark.asyncio
    async def test_non_dict_field_spec_warns(self, publisher):
        """Test that a non-dict field spec produces a warning."""
        skill = ValidSkill(input_schema={"query": "string"})
        result = await publisher.validate(skill)
        warning_codes = [i.code for i in result.issues if i.severity == "warning"]
        assert "INVALID_SCHEMA_FIELD" in warning_codes

    @pytest.mark.asyncio
    async def test_missing_field_type_warns(self, publisher):
        """Test that missing type in field spec produces a warning."""
        skill = ValidSkill(input_schema={"query": {"description": "search query"}})
        result = await publisher.validate(skill)
        warning_codes = [i.code for i in result.issues if i.severity == "warning"]
        assert "MISSING_FIELD_TYPE" in warning_codes

    @pytest.mark.asyncio
    async def test_multiple_schema_issues(self, publisher):
        """Test multiple schema issues are all reported."""
        skill = ValidSkill(
            input_schema={
                "bad_field": "not_a_dict",
                "missing_type": {"description": "no type"},
                "good_field": {"type": "string"},
            }
        )
        result = await publisher.validate(skill)
        warning_codes = [i.code for i in result.issues if i.severity == "warning"]
        assert "INVALID_SCHEMA_FIELD" in warning_codes
        assert "MISSING_FIELD_TYPE" in warning_codes


# =============================================================================
# Sensitive Capabilities Tests
# =============================================================================


class TestSensitiveCapabilities:
    """Tests for SkillPublisher._check_sensitive_capabilities."""

    @pytest.mark.asyncio
    async def test_sensitive_shell_execution(self, publisher):
        """Test that shell_execution triggers a sensitive capability warning."""
        skill = ValidSkill(capabilities=[SkillCapability.SHELL_EXECUTION])
        result = await publisher.validate(skill)
        warning_codes = [i.code for i in result.issues if i.severity == "warning"]
        assert "SENSITIVE_CAPABILITIES" in warning_codes

    @pytest.mark.asyncio
    async def test_sensitive_code_execution(self, publisher):
        """Test that code_execution triggers a sensitive capability warning."""
        skill = ValidSkill(capabilities=[SkillCapability.CODE_EXECUTION])
        result = await publisher.validate(skill)
        warning_codes = [i.code for i in result.issues if i.severity == "warning"]
        assert "SENSITIVE_CAPABILITIES" in warning_codes

    @pytest.mark.asyncio
    async def test_sensitive_write_local(self, publisher):
        """Test that write_local triggers a sensitive capability warning."""
        skill = ValidSkill(capabilities=[SkillCapability.WRITE_LOCAL])
        result = await publisher.validate(skill)
        warning_codes = [i.code for i in result.issues if i.severity == "warning"]
        assert "SENSITIVE_CAPABILITIES" in warning_codes

    @pytest.mark.asyncio
    async def test_sensitive_write_database(self, publisher):
        """Test that write_database triggers a sensitive capability warning."""
        skill = ValidSkill(capabilities=[SkillCapability.WRITE_DATABASE])
        result = await publisher.validate(skill)
        warning_codes = [i.code for i in result.issues if i.severity == "warning"]
        assert "SENSITIVE_CAPABILITIES" in warning_codes

    @pytest.mark.asyncio
    async def test_sensitive_network(self, publisher):
        """Test that network capability triggers a sensitive capability warning."""
        skill = ValidSkill(capabilities=[SkillCapability.NETWORK])
        result = await publisher.validate(skill)
        warning_codes = [i.code for i in result.issues if i.severity == "warning"]
        assert "SENSITIVE_CAPABILITIES" in warning_codes

    @pytest.mark.asyncio
    async def test_non_sensitive_capabilities_no_warning(self, publisher):
        """Test that non-sensitive capabilities do not trigger warnings."""
        skill = ValidSkill(
            capabilities=[
                SkillCapability.EXTERNAL_API,
                SkillCapability.WEB_SEARCH,
                SkillCapability.LLM_INFERENCE,
                SkillCapability.READ_LOCAL,
                SkillCapability.READ_DATABASE,
            ]
        )
        result = await publisher.validate(skill)
        warning_codes = [i.code for i in result.issues if i.severity == "warning"]
        assert "SENSITIVE_CAPABILITIES" not in warning_codes

    @pytest.mark.asyncio
    async def test_multiple_sensitive_capabilities(self, publisher):
        """Test that multiple sensitive capabilities are reported together."""
        skill = ValidSkill(
            capabilities=[
                SkillCapability.SHELL_EXECUTION,
                SkillCapability.CODE_EXECUTION,
                SkillCapability.NETWORK,
            ]
        )
        result = await publisher.validate(skill)
        sensitive_issues = [i for i in result.issues if i.code == "SENSITIVE_CAPABILITIES"]
        assert len(sensitive_issues) == 1
        # The message should reference all three capabilities
        msg = sensitive_issues[0].message
        assert "shell_execution" in msg
        assert "code_execution" in msg
        assert "network" in msg


# =============================================================================
# Reserved Name Tests
# =============================================================================


class TestReservedNames:
    """Tests for SkillPublisher._check_reserved_names."""

    @pytest.mark.asyncio
    async def test_aragora_prefix_rejected(self, publisher):
        """Test that names starting with 'aragora' are rejected."""
        skill = ValidSkill(name="aragora-plugin")
        result = await publisher.validate(skill)
        error_codes = [i.code for i in result.errors]
        assert "RESERVED_NAME" in error_codes

    @pytest.mark.asyncio
    async def test_system_prefix_rejected(self, publisher):
        """Test that names starting with 'system' are rejected."""
        skill = ValidSkill(name="system-tool")
        result = await publisher.validate(skill)
        error_codes = [i.code for i in result.errors]
        assert "RESERVED_NAME" in error_codes

    @pytest.mark.asyncio
    async def test_admin_prefix_rejected(self, publisher):
        """Test that names starting with 'admin' are rejected."""
        skill = ValidSkill(name="admin-panel")
        result = await publisher.validate(skill)
        error_codes = [i.code for i in result.errors]
        assert "RESERVED_NAME" in error_codes

    @pytest.mark.asyncio
    async def test_internal_prefix_rejected(self, publisher):
        """Test that names starting with 'internal' are rejected."""
        skill = ValidSkill(name="internal-tool")
        result = await publisher.validate(skill)
        error_codes = [i.code for i in result.errors]
        assert "RESERVED_NAME" in error_codes

    @pytest.mark.asyncio
    async def test_non_reserved_name_accepted(self, publisher):
        """Test that non-reserved names are accepted."""
        skill = ValidSkill(name="my-cool-skill")
        result = await publisher.validate(skill)
        error_codes = [i.code for i in result.errors]
        assert "RESERVED_NAME" not in error_codes

    @pytest.mark.asyncio
    async def test_reserved_prefix_as_substring_ok(self, publisher):
        """Test that reserved words as substrings (not prefixes) are accepted."""
        skill = ValidSkill(name="my-aragora-tool")
        result = await publisher.validate(skill)
        error_codes = [i.code for i in result.errors]
        assert "RESERVED_NAME" not in error_codes

    @pytest.mark.asyncio
    async def test_reserved_prefix_exact_match_rejected(self, publisher):
        """Test that the exact reserved prefix word is also rejected."""
        for prefix in ["aragora", "system", "admin", "internal"]:
            # These match because startswith("aragora") is True for "aragora" itself
            # But they also fail INVALID_NAME check since they are the full name
            # Starting with a reserved prefix is the primary concern
            skill = ValidSkill(name=prefix)
            result = await publisher.validate(skill)
            error_codes = [i.code for i in result.errors]
            assert "RESERVED_NAME" in error_codes, f"Prefix '{prefix}' should be rejected"


# =============================================================================
# Publish Lifecycle Tests
# =============================================================================


class TestPublish:
    """Tests for SkillPublisher.publish."""

    @pytest.mark.asyncio
    async def test_successful_publish(
        self, publisher, mock_marketplace, valid_skill, valid_listing
    ):
        """Test a successful publish flow."""
        mock_marketplace.publish.return_value = valid_listing

        success, listing, issues = await publisher.publish(
            skill=valid_skill,
            author_id="user-1",
            author_name="Test User",
        )

        assert success is True
        assert listing is not None
        assert listing.skill_id == "user-1:my-test-skill"
        mock_marketplace.publish.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_publish_with_category_and_tier(
        self, publisher, mock_marketplace, valid_skill, valid_listing
    ):
        """Test publish with explicit category and tier."""
        mock_marketplace.publish.return_value = valid_listing

        success, listing, issues = await publisher.publish(
            skill=valid_skill,
            author_id="user-1",
            author_name="Test User",
            category=SkillCategory.WEB_TOOLS,
            tier=SkillTier.PREMIUM,
        )

        assert success is True
        call_kwargs = mock_marketplace.publish.call_args
        assert call_kwargs.kwargs.get("category") == SkillCategory.WEB_TOOLS or (
            len(call_kwargs.args) > 3 and call_kwargs.args[3] == SkillCategory.WEB_TOOLS
        )

    @pytest.mark.asyncio
    async def test_publish_fails_on_invalid_skill(self, publisher, mock_marketplace):
        """Test that publish fails when validation fails."""
        invalid_skill = ValidSkill(name="")  # Missing name

        success, listing, issues = await publisher.publish(
            skill=invalid_skill,
            author_id="user-1",
            author_name="Test User",
        )

        assert success is False
        assert listing is None
        assert len(issues) > 0
        error_codes = [i.code for i in issues if i.severity == "error"]
        assert "MISSING_NAME" in error_codes
        mock_marketplace.publish.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_publish_fails_on_marketplace_error(
        self, publisher, mock_marketplace, valid_skill
    ):
        """Test that publish handles marketplace exceptions gracefully."""
        mock_marketplace.publish.side_effect = RuntimeError("Database connection failed")

        success, listing, issues = await publisher.publish(
            skill=valid_skill,
            author_id="user-1",
            author_name="Test User",
        )

        assert success is False
        assert listing is None
        error_codes = [i.code for i in issues if i.severity == "error"]
        assert "PUBLISH_FAILED" in error_codes
        assert "Database connection failed" in issues[-1].message

    @pytest.mark.asyncio
    async def test_publish_returns_validation_warnings_on_success(
        self,
        publisher,
        mock_marketplace,
        valid_listing,
    ):
        """Test that warnings from validation are returned even on success."""
        # Skill with no description (produces warning) but otherwise valid
        skill = ValidSkill(description="")
        mock_marketplace.publish.return_value = valid_listing

        success, listing, issues = await publisher.publish(
            skill=skill,
            author_id="user-1",
            author_name="Test User",
        )

        assert success is True
        warning_codes = [i.code for i in issues if i.severity == "warning"]
        assert "MISSING_DESCRIPTION" in warning_codes

    @pytest.mark.asyncio
    async def test_publish_with_changelog(
        self, publisher, mock_marketplace, valid_skill, valid_listing
    ):
        """Test publish passes changelog to marketplace."""
        mock_marketplace.publish.return_value = valid_listing

        await publisher.publish(
            skill=valid_skill,
            author_id="user-1",
            author_name="Test User",
            changelog="Initial release with web search",
        )

        call_kwargs = mock_marketplace.publish.call_args
        assert call_kwargs.kwargs.get("changelog") == "Initial release with web search"

    @pytest.mark.asyncio
    async def test_publish_default_changelog(
        self, publisher, mock_marketplace, valid_skill, valid_listing
    ):
        """Test publish uses default changelog."""
        mock_marketplace.publish.return_value = valid_listing

        await publisher.publish(
            skill=valid_skill,
            author_id="user-1",
            author_name="Test User",
        )

        call_kwargs = mock_marketplace.publish.call_args
        assert call_kwargs.kwargs.get("changelog") == "Initial release"

    @pytest.mark.asyncio
    async def test_publish_passes_kwargs(
        self, publisher, mock_marketplace, valid_skill, valid_listing
    ):
        """Test that extra kwargs are forwarded to marketplace.publish."""
        mock_marketplace.publish.return_value = valid_listing

        await publisher.publish(
            skill=valid_skill,
            author_id="user-1",
            author_name="Test User",
            homepage_url="https://example.com",
        )

        call_kwargs = mock_marketplace.publish.call_args
        assert call_kwargs.kwargs.get("homepage_url") == "https://example.com"


# =============================================================================
# Publish Version Tests
# =============================================================================


class TestPublishVersion:
    """Tests for SkillPublisher.publish_version."""

    @pytest.mark.asyncio
    async def test_successful_version_update(self, publisher, mock_marketplace, valid_listing):
        """Test publishing a new version of an existing skill."""
        mock_marketplace.get_skill.return_value = valid_listing
        updated_listing = SkillListing(
            skill_id="user-1:my-test-skill",
            name="my-test-skill",
            description="A test skill",
            author_id="user-1",
            author_name="Test User",
            current_version="2.0.0",
            is_published=True,
        )
        mock_marketplace.publish.return_value = updated_listing

        new_skill = ValidSkill(version="2.0.0")
        success, listing, issues = await publisher.publish_version(
            skill_id="user-1:my-test-skill",
            skill=new_skill,
            author_id="user-1",
            changelog="Major update",
        )

        assert success is True
        assert listing is not None
        assert listing.current_version == "2.0.0"
        mock_marketplace.publish.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_version_update_skill_not_found(self, publisher, mock_marketplace):
        """Test publish_version fails when skill does not exist."""
        mock_marketplace.get_skill.return_value = None

        skill = ValidSkill(version="2.0.0")
        success, listing, issues = await publisher.publish_version(
            skill_id="nonexistent:skill",
            skill=skill,
            author_id="user-1",
        )

        assert success is False
        assert listing is None
        error_codes = [i.code for i in issues if i.severity == "error"]
        assert "SKILL_NOT_FOUND" in error_codes

    @pytest.mark.asyncio
    async def test_version_update_not_authorized(self, publisher, mock_marketplace, valid_listing):
        """Test publish_version fails when author_id does not match."""
        mock_marketplace.get_skill.return_value = valid_listing

        skill = ValidSkill(version="2.0.0")
        success, listing, issues = await publisher.publish_version(
            skill_id="user-1:my-test-skill",
            skill=skill,
            author_id="different-user",
        )

        assert success is False
        assert listing is None
        error_codes = [i.code for i in issues if i.severity == "error"]
        assert "NOT_AUTHORIZED" in error_codes

    @pytest.mark.asyncio
    async def test_version_not_higher(self, publisher, mock_marketplace, valid_listing):
        """Test publish_version fails when new version is not higher."""
        mock_marketplace.get_skill.return_value = valid_listing  # current_version = "1.0.0"

        skill = ValidSkill(version="0.9.0")
        success, listing, issues = await publisher.publish_version(
            skill_id="user-1:my-test-skill",
            skill=skill,
            author_id="user-1",
        )

        assert success is False
        assert listing is None
        error_codes = [i.code for i in issues if i.severity == "error"]
        assert "VERSION_NOT_HIGHER" in error_codes

    @pytest.mark.asyncio
    async def test_version_equal_not_accepted(self, publisher, mock_marketplace, valid_listing):
        """Test publish_version fails when new version equals current."""
        mock_marketplace.get_skill.return_value = valid_listing  # current_version = "1.0.0"

        skill = ValidSkill(version="1.0.0")
        success, listing, issues = await publisher.publish_version(
            skill_id="user-1:my-test-skill",
            skill=skill,
            author_id="user-1",
        )

        assert success is False
        error_codes = [i.code for i in issues if i.severity == "error"]
        assert "VERSION_NOT_HIGHER" in error_codes

    @pytest.mark.asyncio
    async def test_version_update_validation_fails(
        self, publisher, mock_marketplace, valid_listing
    ):
        """Test publish_version fails if the new skill has invalid manifest."""
        mock_marketplace.get_skill.return_value = valid_listing

        # Higher version but invalid name
        skill = ValidSkill(name="", version="2.0.0")
        success, listing, issues = await publisher.publish_version(
            skill_id="user-1:my-test-skill",
            skill=skill,
            author_id="user-1",
        )

        assert success is False
        error_codes = [i.code for i in issues if i.severity == "error"]
        assert "MISSING_NAME" in error_codes

    @pytest.mark.asyncio
    async def test_version_update_marketplace_error(
        self, publisher, mock_marketplace, valid_listing
    ):
        """Test publish_version handles marketplace exception."""
        mock_marketplace.get_skill.return_value = valid_listing
        mock_marketplace.publish.side_effect = RuntimeError("Storage unavailable")

        skill = ValidSkill(version="2.0.0")
        success, listing, issues = await publisher.publish_version(
            skill_id="user-1:my-test-skill",
            skill=skill,
            author_id="user-1",
            changelog="Bugfix release",
        )

        assert success is False
        assert listing is None
        error_codes = [i.code for i in issues if i.severity == "error"]
        assert "PUBLISH_FAILED" in error_codes
        assert "Storage unavailable" in issues[-1].message

    @pytest.mark.asyncio
    async def test_version_update_preserves_existing_metadata(
        self,
        publisher,
        mock_marketplace,
        valid_listing,
    ):
        """Test that publish_version uses existing listing metadata."""
        valid_listing.category = SkillCategory.WEB_TOOLS
        valid_listing.tier = SkillTier.PREMIUM
        valid_listing.author_name = "Original Author"
        mock_marketplace.get_skill.return_value = valid_listing

        updated_listing = SkillListing(
            skill_id="user-1:my-test-skill",
            name="my-test-skill",
            description="Updated",
            author_id="user-1",
            author_name="Original Author",
            current_version="2.0.0",
            category=SkillCategory.WEB_TOOLS,
            tier=SkillTier.PREMIUM,
        )
        mock_marketplace.publish.return_value = updated_listing

        skill = ValidSkill(version="2.0.0")
        success, listing, issues = await publisher.publish_version(
            skill_id="user-1:my-test-skill",
            skill=skill,
            author_id="user-1",
        )

        assert success is True
        # Verify marketplace.publish was called with existing metadata
        call_kwargs = mock_marketplace.publish.call_args.kwargs
        assert call_kwargs["author_name"] == "Original Author"
        assert call_kwargs["category"] == SkillCategory.WEB_TOOLS
        assert call_kwargs["tier"] == SkillTier.PREMIUM

    @pytest.mark.asyncio
    async def test_version_patch_increment(self, publisher, mock_marketplace, valid_listing):
        """Test that a patch version increment is accepted."""
        mock_marketplace.get_skill.return_value = valid_listing  # 1.0.0
        mock_marketplace.publish.return_value = valid_listing

        skill = ValidSkill(version="1.0.1")
        success, listing, issues = await publisher.publish_version(
            skill_id="user-1:my-test-skill",
            skill=skill,
            author_id="user-1",
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_version_minor_increment(self, publisher, mock_marketplace, valid_listing):
        """Test that a minor version increment is accepted."""
        mock_marketplace.get_skill.return_value = valid_listing  # 1.0.0
        mock_marketplace.publish.return_value = valid_listing

        skill = ValidSkill(version="1.1.0")
        success, listing, issues = await publisher.publish_version(
            skill_id="user-1:my-test-skill",
            skill=skill,
            author_id="user-1",
        )

        assert success is True


# =============================================================================
# Version Comparison Tests
# =============================================================================


class TestIsHigherVersion:
    """Tests for SkillPublisher._is_higher_version."""

    def test_major_increment(self):
        """Test major version increment is detected."""
        pub = SkillPublisher.__new__(SkillPublisher)
        assert pub._is_higher_version("2.0.0", "1.0.0") is True

    def test_minor_increment(self):
        """Test minor version increment is detected."""
        pub = SkillPublisher.__new__(SkillPublisher)
        assert pub._is_higher_version("1.1.0", "1.0.0") is True

    def test_patch_increment(self):
        """Test patch version increment is detected."""
        pub = SkillPublisher.__new__(SkillPublisher)
        assert pub._is_higher_version("1.0.1", "1.0.0") is True

    def test_same_version(self):
        """Test that same version is not higher."""
        pub = SkillPublisher.__new__(SkillPublisher)
        assert pub._is_higher_version("1.0.0", "1.0.0") is False

    def test_lower_major(self):
        """Test that lower major version is not higher."""
        pub = SkillPublisher.__new__(SkillPublisher)
        assert pub._is_higher_version("0.9.9", "1.0.0") is False

    def test_lower_minor(self):
        """Test that lower minor version is not higher."""
        pub = SkillPublisher.__new__(SkillPublisher)
        assert pub._is_higher_version("1.0.5", "1.1.0") is False

    def test_lower_patch(self):
        """Test that lower patch version is not higher."""
        pub = SkillPublisher.__new__(SkillPublisher)
        assert pub._is_higher_version("1.0.0", "1.0.1") is False

    def test_prerelease_stripped(self):
        """Test that prerelease tags are stripped for comparison."""
        pub = SkillPublisher.__new__(SkillPublisher)
        assert pub._is_higher_version("2.0.0-beta", "1.0.0") is True
        assert pub._is_higher_version("1.0.0-alpha", "1.0.0") is False

    def test_large_version_numbers(self):
        """Test with large version numbers."""
        pub = SkillPublisher.__new__(SkillPublisher)
        assert pub._is_higher_version("100.200.300", "99.999.999") is True

    def test_invalid_version_returns_true(self):
        """Test that unparseable versions default to True."""
        pub = SkillPublisher.__new__(SkillPublisher)
        assert pub._is_higher_version("abc", "1.0.0") is True

    def test_both_invalid_versions(self):
        """Test that two invalid versions default to True."""
        pub = SkillPublisher.__new__(SkillPublisher)
        assert pub._is_higher_version("xyz", "abc") is True

    def test_minor_higher_major_lower(self):
        """Test that higher major always wins regardless of minor/patch."""
        pub = SkillPublisher.__new__(SkillPublisher)
        assert pub._is_higher_version("2.0.0", "1.99.99") is True

    def test_zero_versions(self):
        """Test comparison with zero versions."""
        pub = SkillPublisher.__new__(SkillPublisher)
        assert pub._is_higher_version("0.0.1", "0.0.0") is True
        assert pub._is_higher_version("0.0.0", "0.0.0") is False


# =============================================================================
# Deprecation Tests
# =============================================================================


class TestDeprecate:
    """Tests for SkillPublisher.deprecate."""

    @pytest.mark.asyncio
    async def test_successful_deprecation(self, publisher, mock_marketplace, valid_listing):
        """Test successful deprecation by the skill author."""
        mock_marketplace.get_skill.return_value = valid_listing

        result = await publisher.deprecate(
            skill_id="user-1:my-test-skill",
            author_id="user-1",
            message="This skill is no longer maintained",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_deprecate_skill_not_found(self, publisher, mock_marketplace):
        """Test deprecation fails when skill does not exist."""
        mock_marketplace.get_skill.return_value = None

        result = await publisher.deprecate(
            skill_id="nonexistent:skill",
            author_id="user-1",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_deprecate_not_authorized(self, publisher, mock_marketplace, valid_listing):
        """Test deprecation fails when author_id does not match."""
        mock_marketplace.get_skill.return_value = valid_listing

        result = await publisher.deprecate(
            skill_id="user-1:my-test-skill",
            author_id="malicious-user",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_deprecate_with_replacement(self, publisher, mock_marketplace, valid_listing):
        """Test deprecation with a replacement skill specified."""
        mock_marketplace.get_skill.return_value = valid_listing

        result = await publisher.deprecate(
            skill_id="user-1:my-test-skill",
            author_id="user-1",
            replacement_skill_id="user-1:my-better-skill",
            message="Use my-better-skill instead",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_deprecate_without_message(self, publisher, mock_marketplace, valid_listing):
        """Test deprecation with default empty message."""
        mock_marketplace.get_skill.return_value = valid_listing

        result = await publisher.deprecate(
            skill_id="user-1:my-test-skill",
            author_id="user-1",
        )

        assert result is True


# =============================================================================
# Edge Cases and Combined Validation Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and combined validation scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_validation_errors(self, publisher):
        """Test that multiple errors are all reported at once."""
        skill = ValidSkill(
            name="",
            version="invalid",
            description="",
            capabilities=[],
        )
        result = await publisher.validate(skill)
        assert result.is_valid is False
        error_codes = [i.code for i in result.errors]
        assert "MISSING_NAME" in error_codes
        # Version may not be checked if empty string for name, but INVALID_VERSION
        # still applies since _validate_version is called separately
        assert "INVALID_VERSION" in error_codes

    @pytest.mark.asyncio
    async def test_reserved_name_with_invalid_version(self, publisher):
        """Test that reserved names and bad versions both error."""
        skill = ValidSkill(name="system-tool", version="bad")
        result = await publisher.validate(skill)
        assert result.is_valid is False
        error_codes = [i.code for i in result.errors]
        assert "RESERVED_NAME" in error_codes
        assert "INVALID_VERSION" in error_codes

    @pytest.mark.asyncio
    async def test_valid_skill_returns_manifest(self, publisher, valid_skill):
        """Test that a valid skill includes the manifest in the result."""
        result = await publisher.validate(valid_skill)
        assert result.manifest is not None
        assert result.manifest.name == "my-test-skill"
        assert result.manifest.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_invalid_skill_does_not_return_manifest(self, publisher):
        """Test that an invalid skill does not include the manifest."""
        skill = ValidSkill(name="")
        result = await publisher.validate(skill)
        assert result.manifest is None

    @pytest.mark.asyncio
    async def test_warnings_list_populated(self, publisher):
        """Test that warnings list is populated from warning-level issues."""
        skill = ValidSkill(description="", capabilities=[], author="")
        result = await publisher.validate(skill)
        assert result.is_valid is True
        assert len(result.warnings) > 0
        # warnings should contain the message strings
        assert any("description" in w.lower() for w in result.warnings)

    @pytest.mark.asyncio
    async def test_single_char_name_valid(self, publisher):
        """Test that a single lowercase letter is a valid name."""
        skill = ValidSkill(name="a")
        result = await publisher.validate(skill)
        error_codes = [i.code for i in result.errors]
        assert "INVALID_NAME" not in error_codes
        assert "MISSING_NAME" not in error_codes

    @pytest.mark.asyncio
    async def test_name_starting_with_hyphen_invalid(self, publisher):
        """Test that names starting with a hyphen are invalid."""
        skill = ValidSkill(name="-my-skill")
        result = await publisher.validate(skill)
        error_codes = [i.code for i in result.errors]
        assert "INVALID_NAME" in error_codes

    @pytest.mark.asyncio
    async def test_name_with_spaces_invalid(self, publisher):
        """Test that names with spaces are invalid."""
        skill = ValidSkill(name="my skill")
        result = await publisher.validate(skill)
        error_codes = [i.code for i in result.errors]
        assert "INVALID_NAME" in error_codes

    @pytest.mark.asyncio
    async def test_publish_reserved_name_blocked(self, publisher, mock_marketplace):
        """Test that reserved-name skills cannot be published."""
        skill = ValidSkill(name="aragora-extension")

        success, listing, issues = await publisher.publish(
            skill=skill,
            author_id="user-1",
            author_name="Test User",
        )

        assert success is False
        error_codes = [i.code for i in issues if i.severity == "error"]
        assert "RESERVED_NAME" in error_codes
        mock_marketplace.publish.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_publish_invalid_version_blocked(self, publisher, mock_marketplace):
        """Test that skills with invalid versions cannot be published."""
        skill = ValidSkill(version="not-a-version")

        success, listing, issues = await publisher.publish(
            skill=skill,
            author_id="user-1",
            author_name="Test User",
        )

        assert success is False
        error_codes = [i.code for i in issues if i.severity == "error"]
        assert "INVALID_VERSION" in error_codes
        mock_marketplace.publish.assert_not_awaited()


# =============================================================================
# Publisher Initialization Tests
# =============================================================================


class TestPublisherInit:
    """Tests for SkillPublisher initialization."""

    def test_init_with_explicit_marketplace(self, mock_marketplace):
        """Test initializing with an explicit marketplace instance."""
        pub = SkillPublisher(marketplace=mock_marketplace)
        assert pub._marketplace is mock_marketplace

    @patch("aragora.skills.publisher.get_marketplace")
    def test_init_default_marketplace(self, mock_get_marketplace):
        """Test that default marketplace is obtained from get_marketplace."""
        mock_mp = MagicMock()
        mock_get_marketplace.return_value = mock_mp

        pub = SkillPublisher()
        assert pub._marketplace is mock_mp
        mock_get_marketplace.assert_called_once()

    def test_sensitive_capabilities_set(self):
        """Test that SENSITIVE_CAPABILITIES includes the expected set."""
        expected = {
            SkillCapability.SHELL_EXECUTION,
            SkillCapability.CODE_EXECUTION,
            SkillCapability.WRITE_LOCAL,
            SkillCapability.WRITE_DATABASE,
            SkillCapability.NETWORK,
        }
        assert SkillPublisher.SENSITIVE_CAPABILITIES == expected

    def test_reserved_prefixes_list(self):
        """Test that RESERVED_PREFIXES contains expected values."""
        assert "aragora" in SkillPublisher.RESERVED_PREFIXES
        assert "system" in SkillPublisher.RESERVED_PREFIXES
        assert "admin" in SkillPublisher.RESERVED_PREFIXES
        assert "internal" in SkillPublisher.RESERVED_PREFIXES
        assert len(SkillPublisher.RESERVED_PREFIXES) == 4


# =============================================================================
# Integration-style Tests (with real marketplace)
# =============================================================================


class TestPublishIntegration:
    """Integration tests using a real in-memory SkillMarketplace."""

    @pytest.fixture
    def real_marketplace(self):
        """Create a real in-memory marketplace."""
        return SkillMarketplace(db_path=":memory:")

    @pytest.fixture
    def real_publisher(self, real_marketplace):
        """Create a publisher with a real marketplace."""
        return SkillPublisher(marketplace=real_marketplace)

    @pytest.mark.asyncio
    async def test_full_publish_lifecycle(self, real_publisher, real_marketplace):
        """Test the full publish lifecycle: publish, then publish_version."""
        skill_v1 = ValidSkill(version="1.0.0")

        # Publish v1
        success, listing, issues = await real_publisher.publish(
            skill=skill_v1,
            author_id="author-1",
            author_name="Author One",
            category=SkillCategory.WEB_TOOLS,
        )

        assert success is True
        assert listing is not None
        assert listing.current_version == "1.0.0"
        assert listing.is_published is True
        skill_id = listing.skill_id

        # Publish v2
        skill_v2 = ValidSkill(version="2.0.0")
        success2, listing2, issues2 = await real_publisher.publish_version(
            skill_id=skill_id,
            skill=skill_v2,
            author_id="author-1",
            changelog="Major update",
        )

        assert success2 is True
        assert listing2 is not None
        assert listing2.current_version == "2.0.0"

    @pytest.mark.asyncio
    async def test_publish_then_deprecate(self, real_publisher, real_marketplace):
        """Test publishing then deprecating a skill."""
        skill = ValidSkill()

        success, listing, _ = await real_publisher.publish(
            skill=skill,
            author_id="author-1",
            author_name="Author One",
        )
        assert success is True

        deprecated = await real_publisher.deprecate(
            skill_id=listing.skill_id,
            author_id="author-1",
            message="Replaced by v2",
        )
        assert deprecated is True

    @pytest.mark.asyncio
    async def test_publish_duplicate_updates_existing(self, real_publisher, real_marketplace):
        """Test that publishing the same skill again updates it."""
        skill = ValidSkill(version="1.0.0", description="First version")

        success1, listing1, _ = await real_publisher.publish(
            skill=skill,
            author_id="author-1",
            author_name="Author One",
        )
        assert success1 is True

        updated_skill = ValidSkill(version="1.1.0", description="Updated description")
        success2, listing2, _ = await real_publisher.publish(
            skill=updated_skill,
            author_id="author-1",
            author_name="Author One",
        )
        assert success2 is True
        assert listing2.description == "Updated description"
        assert listing2.current_version == "1.1.0"

    @pytest.mark.asyncio
    async def test_version_downgrade_blocked_integration(self, real_publisher, real_marketplace):
        """Test that version downgrade is blocked end-to-end."""
        skill_v2 = ValidSkill(version="2.0.0")

        success, listing, _ = await real_publisher.publish(
            skill=skill_v2,
            author_id="author-1",
            author_name="Author One",
        )
        assert success is True

        skill_v1 = ValidSkill(version="1.0.0")
        success2, listing2, issues2 = await real_publisher.publish_version(
            skill_id=listing.skill_id,
            skill=skill_v1,
            author_id="author-1",
        )
        assert success2 is False
        error_codes = [i.code for i in issues2 if i.severity == "error"]
        assert "VERSION_NOT_HIGHER" in error_codes

    @pytest.mark.asyncio
    async def test_unauthorized_version_update_integration(self, real_publisher, real_marketplace):
        """Test that unauthorized version updates are blocked end-to-end."""
        skill = ValidSkill(version="1.0.0")

        success, listing, _ = await real_publisher.publish(
            skill=skill,
            author_id="author-1",
            author_name="Author One",
        )
        assert success is True

        skill_v2 = ValidSkill(version="2.0.0")
        success2, listing2, issues2 = await real_publisher.publish_version(
            skill_id=listing.skill_id,
            skill=skill_v2,
            author_id="attacker-1",
        )
        assert success2 is False
        error_codes = [i.code for i in issues2 if i.severity == "error"]
        assert "NOT_AUTHORIZED" in error_codes
