"""
Tests for aragora.skills.base module.

Covers:
- SkillCapability enum
- SkillStatus enum
- SkillManifest dataclass
- SkillResult dataclass
- SkillContext dataclass
- Skill abstract base class
- SyncSkill base class
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock

import pytest

from aragora.skills.base import (
    Skill,
    SkillCapability,
    SkillContext,
    SkillManifest,
    SkillResult,
    SkillStatus,
    SyncSkill,
)


# =============================================================================
# SkillCapability Tests
# =============================================================================


class TestSkillCapability:
    """Tests for SkillCapability enum."""

    def test_data_access_capabilities(self):
        """Test data access capability values."""
        assert SkillCapability.READ_LOCAL.value == "read_local"
        assert SkillCapability.WRITE_LOCAL.value == "write_local"
        assert SkillCapability.READ_DATABASE.value == "read_database"
        assert SkillCapability.WRITE_DATABASE.value == "write_database"

    def test_external_capabilities(self):
        """Test external capability values."""
        assert SkillCapability.EXTERNAL_API.value == "external_api"
        assert SkillCapability.WEB_SEARCH.value == "web_search"
        assert SkillCapability.WEB_FETCH.value == "web_fetch"

    def test_execution_capabilities(self):
        """Test execution capability values."""
        assert SkillCapability.CODE_EXECUTION.value == "code_execution"
        assert SkillCapability.SHELL_EXECUTION.value == "shell_execution"

    def test_ai_capabilities(self):
        """Test AI capability values."""
        assert SkillCapability.LLM_INFERENCE.value == "llm_inference"
        assert SkillCapability.EMBEDDING.value == "embedding"

    def test_debate_capabilities(self):
        """Test debate-specific capability values."""
        assert SkillCapability.DEBATE_CONTEXT.value == "debate_context"
        assert SkillCapability.EVIDENCE_COLLECTION.value == "evidence_collection"
        assert SkillCapability.KNOWLEDGE_QUERY.value == "knowledge_query"

    def test_system_capabilities(self):
        """Test system capability values."""
        assert SkillCapability.SYSTEM_INFO.value == "system_info"
        assert SkillCapability.NETWORK.value == "network"

    def test_capability_is_string_enum(self):
        """Test that capabilities are string enums."""
        for cap in SkillCapability:
            assert isinstance(cap, str)
            assert cap == cap.value


# =============================================================================
# SkillStatus Tests
# =============================================================================


class TestSkillStatus:
    """Tests for SkillStatus enum."""

    def test_success_status(self):
        """Test success status value."""
        assert SkillStatus.SUCCESS.value == "success"

    def test_failure_status(self):
        """Test failure status value."""
        assert SkillStatus.FAILURE.value == "failure"

    def test_partial_status(self):
        """Test partial status value."""
        assert SkillStatus.PARTIAL.value == "partial"

    def test_timeout_status(self):
        """Test timeout status value."""
        assert SkillStatus.TIMEOUT.value == "timeout"

    def test_rate_limited_status(self):
        """Test rate limited status value."""
        assert SkillStatus.RATE_LIMITED.value == "rate_limited"

    def test_permission_denied_status(self):
        """Test permission denied status value."""
        assert SkillStatus.PERMISSION_DENIED.value == "permission_denied"

    def test_invalid_input_status(self):
        """Test invalid input status value."""
        assert SkillStatus.INVALID_INPUT.value == "invalid_input"

    def test_not_implemented_status(self):
        """Test not implemented status value."""
        assert SkillStatus.NOT_IMPLEMENTED.value == "not_implemented"


# =============================================================================
# SkillManifest Tests
# =============================================================================


class TestSkillManifest:
    """Tests for SkillManifest dataclass."""

    @pytest.fixture
    def basic_manifest(self) -> SkillManifest:
        """Create a basic manifest for testing."""
        return SkillManifest(
            name="test_skill",
            version="1.0.0",
            capabilities=[SkillCapability.WEB_SEARCH],
            input_schema={"query": {"type": "string", "required": True}},
        )

    @pytest.fixture
    def full_manifest(self) -> SkillManifest:
        """Create a full manifest with all fields."""
        return SkillManifest(
            name="full_skill",
            version="2.0.0",
            capabilities=[SkillCapability.WEB_SEARCH, SkillCapability.EXTERNAL_API],
            input_schema={
                "query": {"type": "string", "required": True},
                "max_results": {"type": "number", "default": 10},
            },
            description="A fully configured test skill",
            author="Test Author",
            tags=["test", "search", "web"],
            required_permissions=["skill:execute", "web:search"],
            required_env_vars=["API_KEY"],
            required_packages=["httpx"],
            max_execution_time_seconds=30.0,
            max_retries=5,
            rate_limit_per_minute=60,
            debate_compatible=True,
            requires_debate_context=False,
            output_schema={"results": {"type": "array"}},
        )

    def test_basic_manifest_creation(self, basic_manifest: SkillManifest):
        """Test creating a basic manifest."""
        assert basic_manifest.name == "test_skill"
        assert basic_manifest.version == "1.0.0"
        assert SkillCapability.WEB_SEARCH in basic_manifest.capabilities
        assert "query" in basic_manifest.input_schema

    def test_manifest_defaults(self, basic_manifest: SkillManifest):
        """Test manifest default values."""
        assert basic_manifest.description == ""
        assert basic_manifest.author == ""
        assert basic_manifest.tags == []
        assert basic_manifest.required_permissions == []
        assert basic_manifest.required_env_vars == []
        assert basic_manifest.required_packages == []
        assert basic_manifest.max_execution_time_seconds == 60.0
        assert basic_manifest.max_retries == 3
        assert basic_manifest.rate_limit_per_minute is None
        assert basic_manifest.debate_compatible is True
        assert basic_manifest.requires_debate_context is False
        assert basic_manifest.output_schema is None

    def test_full_manifest_creation(self, full_manifest: SkillManifest):
        """Test creating a fully configured manifest."""
        assert full_manifest.name == "full_skill"
        assert full_manifest.description == "A fully configured test skill"
        assert full_manifest.author == "Test Author"
        assert "test" in full_manifest.tags
        assert "skill:execute" in full_manifest.required_permissions
        assert full_manifest.max_execution_time_seconds == 30.0
        assert full_manifest.max_retries == 5
        assert full_manifest.rate_limit_per_minute == 60

    def test_to_function_schema(self, basic_manifest: SkillManifest):
        """Test converting manifest to function schema."""
        schema = basic_manifest.to_function_schema()

        assert schema["name"] == "test_skill"
        assert "description" in schema
        assert "parameters" in schema
        assert schema["parameters"]["type"] == "object"
        assert "query" in schema["parameters"]["properties"]
        assert "query" in schema["parameters"]["required"]

    def test_to_function_schema_with_description(self, full_manifest: SkillManifest):
        """Test function schema includes custom description."""
        schema = full_manifest.to_function_schema()
        assert schema["description"] == "A fully configured test skill"

    def test_to_dict(self, basic_manifest: SkillManifest):
        """Test serializing manifest to dict."""
        data = basic_manifest.to_dict()

        assert data["name"] == "test_skill"
        assert data["version"] == "1.0.0"
        assert "web_search" in data["capabilities"]
        assert "query" in data["input_schema"]
        assert data["max_execution_time_seconds"] == 60.0

    def test_from_dict(self):
        """Test deserializing manifest from dict."""
        data = {
            "name": "loaded_skill",
            "version": "1.0.0",
            "capabilities": ["web_search", "external_api"],
            "input_schema": {"query": {"type": "string"}},
            "description": "Loaded from dict",
            "max_execution_time_seconds": 45.0,
        }

        manifest = SkillManifest.from_dict(data)

        assert manifest.name == "loaded_skill"
        assert manifest.version == "1.0.0"
        assert SkillCapability.WEB_SEARCH in manifest.capabilities
        assert SkillCapability.EXTERNAL_API in manifest.capabilities
        assert manifest.description == "Loaded from dict"
        assert manifest.max_execution_time_seconds == 45.0

    def test_from_dict_with_defaults(self):
        """Test deserializing manifest uses defaults for missing fields."""
        data = {
            "name": "minimal_skill",
            "version": "1.0.0",
            "capabilities": [],
            "input_schema": {},
        }

        manifest = SkillManifest.from_dict(data)

        assert manifest.description == ""
        assert manifest.tags == []
        assert manifest.max_retries == 3

    def test_roundtrip_serialization(self, full_manifest: SkillManifest):
        """Test that to_dict and from_dict are inverse operations."""
        data = full_manifest.to_dict()
        restored = SkillManifest.from_dict(data)

        assert restored.name == full_manifest.name
        assert restored.version == full_manifest.version
        assert restored.capabilities == full_manifest.capabilities
        assert restored.description == full_manifest.description
        assert restored.max_execution_time_seconds == full_manifest.max_execution_time_seconds


# =============================================================================
# SkillResult Tests
# =============================================================================


class TestSkillResult:
    """Tests for SkillResult dataclass."""

    def test_create_success(self):
        """Test creating a successful result."""
        result = SkillResult.create_success(
            data={"value": 42},
            source="test",
        )

        assert result.status == SkillStatus.SUCCESS
        assert result.data == {"value": 42}
        assert result.success is True
        assert result.error_message is None
        assert result.completed_at is not None
        assert result.metadata.get("source") == "test"

    def test_create_failure(self):
        """Test creating a failure result."""
        result = SkillResult.create_failure(
            error_message="Something went wrong",
            error_code="TEST_ERROR",
        )

        assert result.status == SkillStatus.FAILURE
        assert result.success is False
        assert result.error_message == "Something went wrong"
        assert result.error_code == "TEST_ERROR"
        assert result.data is None

    def test_create_failure_with_custom_status(self):
        """Test creating a failure result with custom status."""
        result = SkillResult.create_failure(
            error_message="Rate limited",
            status=SkillStatus.RATE_LIMITED,
        )

        assert result.status == SkillStatus.RATE_LIMITED
        assert result.success is False

    def test_create_timeout(self):
        """Test creating a timeout result."""
        result = SkillResult.create_timeout(30.0)

        assert result.status == SkillStatus.TIMEOUT
        assert result.success is False
        assert "30" in result.error_message
        assert "timed out" in result.error_message.lower()

    def test_create_permission_denied(self):
        """Test creating a permission denied result."""
        result = SkillResult.create_permission_denied("skill:execute")

        assert result.status == SkillStatus.PERMISSION_DENIED
        assert result.success is False
        assert "skill:execute" in result.error_message
        assert result.error_code == "permission_denied"

    def test_execution_id_generated(self):
        """Test that execution ID is auto-generated."""
        result1 = SkillResult(status=SkillStatus.SUCCESS)
        result2 = SkillResult(status=SkillStatus.SUCCESS)

        assert result1.execution_id is not None
        assert result2.execution_id is not None
        assert result1.execution_id != result2.execution_id

    def test_duration_seconds(self):
        """Test duration calculation."""
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 12, 0, 30, tzinfo=timezone.utc)

        result = SkillResult(
            status=SkillStatus.SUCCESS,
            started_at=start,
            completed_at=end,
        )

        assert result.duration_seconds == 30.0

    def test_duration_seconds_none_when_incomplete(self):
        """Test duration is None when timestamps missing."""
        result = SkillResult(status=SkillStatus.SUCCESS)
        assert result.duration_seconds is None

    def test_to_dict(self):
        """Test serializing result to dict."""
        result = SkillResult.create_success(
            data={"value": 42},
            custom_meta="test_value",
        )

        data = result.to_dict()

        assert data["status"] == "success"
        assert data["data"] == {"value": 42}
        assert data["execution_id"] is not None
        assert "custom_meta" in data["metadata"]

    def test_warnings_list(self):
        """Test warnings can be added to result."""
        result = SkillResult(
            status=SkillStatus.PARTIAL,
            data={"partial": True},
            warnings=["Some items failed", "Retried 3 times"],
        )

        assert len(result.warnings) == 2
        assert "Some items failed" in result.warnings

    def test_usage_tracking(self):
        """Test usage tracking fields."""
        result = SkillResult(
            status=SkillStatus.SUCCESS,
            tokens_used=1500,
            cost_estimate=0.05,
        )

        assert result.tokens_used == 1500
        assert result.cost_estimate == 0.05


# =============================================================================
# SkillContext Tests
# =============================================================================


class TestSkillContext:
    """Tests for SkillContext dataclass."""

    @pytest.fixture
    def basic_context(self) -> SkillContext:
        """Create a basic context for testing."""
        return SkillContext(
            user_id="user123",
            tenant_id="tenant456",
            permissions=["skill:execute", "web:search"],
        )

    @pytest.fixture
    def debate_context(self) -> SkillContext:
        """Create a context with debate info."""
        return SkillContext(
            user_id="user123",
            debate_id="debate789",
            debate_context={"topic": "Climate change", "round": 2},
            agent_name="claude",
            permissions=["skill:execute", "debate:participate"],
        )

    def test_basic_context_creation(self, basic_context: SkillContext):
        """Test creating a basic context."""
        assert basic_context.user_id == "user123"
        assert basic_context.tenant_id == "tenant456"
        assert "skill:execute" in basic_context.permissions

    def test_context_defaults(self):
        """Test context default values."""
        context = SkillContext()

        assert context.user_id is None
        assert context.tenant_id is None
        assert context.session_id is None
        assert context.permissions == []
        assert context.environment == "development"
        assert context.config == {}
        assert context.previous_results == {}

    def test_has_permission(self, basic_context: SkillContext):
        """Test checking single permission."""
        assert basic_context.has_permission("skill:execute") is True
        assert basic_context.has_permission("web:search") is True
        assert basic_context.has_permission("admin:delete") is False

    def test_has_all_permissions(self, basic_context: SkillContext):
        """Test checking multiple permissions."""
        assert basic_context.has_all_permissions(["skill:execute"]) is True
        assert basic_context.has_all_permissions(["skill:execute", "web:search"]) is True
        assert basic_context.has_all_permissions(["skill:execute", "admin:delete"]) is False

    def test_get_config(self, basic_context: SkillContext):
        """Test getting config values."""
        basic_context.config = {"timeout": 30, "retries": 3}

        assert basic_context.get_config("timeout") == 30
        assert basic_context.get_config("retries") == 3
        assert basic_context.get_config("missing") is None
        assert basic_context.get_config("missing", default=10) == 10

    def test_debate_context_fields(self, debate_context: SkillContext):
        """Test debate-related context fields."""
        assert debate_context.debate_id == "debate789"
        assert debate_context.debate_context["topic"] == "Climate change"
        assert debate_context.agent_name == "claude"

    def test_previous_results(self):
        """Test storing previous skill results."""
        context = SkillContext()
        result = SkillResult.create_success({"value": 42})
        context.previous_results["web_search"] = result

        assert "web_search" in context.previous_results
        assert context.previous_results["web_search"].data == {"value": 42}

    def test_request_metadata(self):
        """Test request metadata fields."""
        context = SkillContext(
            request_id="req123",
            correlation_id="corr456",
        )

        assert context.request_id == "req123"
        assert context.correlation_id == "corr456"


# =============================================================================
# Skill Abstract Base Class Tests
# =============================================================================


class ConcreteSkill(Skill):
    """Concrete implementation for testing."""

    def __init__(self, name: str = "test_skill"):
        self._name = name

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name=self._name,
            version="1.0.0",
            capabilities=[SkillCapability.WEB_SEARCH],
            input_schema={
                "query": {"type": "string", "required": True},
                "limit": {"type": "number"},
            },
            required_permissions=["skill:execute"],
        )

    async def execute(
        self,
        input_data: Dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        query = input_data.get("query", "")
        return SkillResult.create_success({"results": [f"Result for: {query}"]})


class TestSkill:
    """Tests for Skill abstract base class."""

    @pytest.fixture
    def skill(self) -> ConcreteSkill:
        """Create a concrete skill for testing."""
        return ConcreteSkill()

    @pytest.fixture
    def context(self) -> SkillContext:
        """Create a context for testing."""
        return SkillContext(
            user_id="user123",
            permissions=["skill:execute"],
        )

    def test_manifest_property(self, skill: ConcreteSkill):
        """Test accessing skill manifest."""
        manifest = skill.manifest
        assert manifest.name == "test_skill"
        assert manifest.version == "1.0.0"
        assert SkillCapability.WEB_SEARCH in manifest.capabilities

    @pytest.mark.asyncio
    async def test_execute(self, skill: ConcreteSkill, context: SkillContext):
        """Test skill execution."""
        result = await skill.execute({"query": "test"}, context)

        assert result.success is True
        assert "Result for: test" in result.data["results"]

    @pytest.mark.asyncio
    async def test_validate_input_valid(self, skill: ConcreteSkill):
        """Test input validation with valid input."""
        is_valid, error = await skill.validate_input({"query": "test"})
        assert is_valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_input_missing_required(self, skill: ConcreteSkill):
        """Test input validation with missing required field."""
        is_valid, error = await skill.validate_input({})
        assert is_valid is False
        assert "query" in error

    @pytest.mark.asyncio
    async def test_validate_input_wrong_type_string(self, skill: ConcreteSkill):
        """Test input validation with wrong type for string field."""
        is_valid, error = await skill.validate_input({"query": 123})
        assert is_valid is False
        assert "string" in error.lower()

    @pytest.mark.asyncio
    async def test_validate_input_wrong_type_number(self, skill: ConcreteSkill):
        """Test input validation with wrong type for number field."""
        is_valid, error = await skill.validate_input({"query": "test", "limit": "abc"})
        assert is_valid is False
        assert "number" in error.lower()

    @pytest.mark.asyncio
    async def test_check_permissions_has_permission(
        self, skill: ConcreteSkill, context: SkillContext
    ):
        """Test permission check when permission exists."""
        has_perm, missing = await skill.check_permissions(context)
        assert has_perm is True
        assert missing is None

    @pytest.mark.asyncio
    async def test_check_permissions_missing_permission(self, skill: ConcreteSkill):
        """Test permission check when permission missing."""
        context = SkillContext(permissions=[])
        has_perm, missing = await skill.check_permissions(context)
        assert has_perm is False
        assert missing == "skill:execute"

    def test_repr(self, skill: ConcreteSkill):
        """Test skill string representation."""
        repr_str = repr(skill)
        assert "ConcreteSkill" in repr_str
        assert "test_skill" in repr_str


# =============================================================================
# SyncSkill Tests
# =============================================================================


class ConcreteSyncSkill(SyncSkill):
    """Concrete sync skill for testing."""

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="sync_skill",
            version="1.0.0",
            capabilities=[SkillCapability.READ_LOCAL],
            input_schema={"path": {"type": "string", "required": True}},
        )

    def execute_sync(
        self,
        input_data: Dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        path = input_data.get("path", "")
        return SkillResult.create_success({"content": f"Contents of: {path}"})


class TestSyncSkill:
    """Tests for SyncSkill base class."""

    @pytest.fixture
    def skill(self) -> ConcreteSyncSkill:
        """Create a sync skill for testing."""
        return ConcreteSyncSkill()

    @pytest.fixture
    def context(self) -> SkillContext:
        """Create a context for testing."""
        return SkillContext()

    @pytest.mark.asyncio
    async def test_execute_wraps_sync(self, skill: ConcreteSyncSkill, context: SkillContext):
        """Test that async execute wraps sync execution."""
        result = await skill.execute({"path": "/test/file.txt"}, context)

        assert result.success is True
        assert "Contents of: /test/file.txt" in result.data["content"]

    def test_execute_sync_directly(self, skill: ConcreteSyncSkill, context: SkillContext):
        """Test calling execute_sync directly."""
        result = skill.execute_sync({"path": "/test/file.txt"}, context)

        assert result.success is True
        assert "Contents of: /test/file.txt" in result.data["content"]


# =============================================================================
# Input Validation Edge Cases
# =============================================================================


class TypeValidationSkill(Skill):
    """Skill with various type validations for testing."""

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="type_validation_skill",
            version="1.0.0",
            capabilities=[],
            input_schema={
                "string_field": {"type": "string"},
                "number_field": {"type": "number"},
                "boolean_field": {"type": "boolean"},
                "array_field": {"type": "array"},
                "object_field": {"type": "object"},
            },
        )

    async def execute(
        self,
        input_data: Dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        return SkillResult.create_success(input_data)


class TestInputValidationEdgeCases:
    """Tests for input validation edge cases."""

    @pytest.fixture
    def skill(self) -> TypeValidationSkill:
        return TypeValidationSkill()

    @pytest.mark.asyncio
    async def test_validate_boolean_type(self, skill: TypeValidationSkill):
        """Test boolean type validation."""
        is_valid, _ = await skill.validate_input({"boolean_field": True})
        assert is_valid is True

        is_valid, error = await skill.validate_input({"boolean_field": "true"})
        assert is_valid is False
        assert "boolean" in error.lower()

    @pytest.mark.asyncio
    async def test_validate_array_type(self, skill: TypeValidationSkill):
        """Test array type validation."""
        is_valid, _ = await skill.validate_input({"array_field": [1, 2, 3]})
        assert is_valid is True

        is_valid, error = await skill.validate_input({"array_field": "not an array"})
        assert is_valid is False
        assert "array" in error.lower()

    @pytest.mark.asyncio
    async def test_validate_object_type(self, skill: TypeValidationSkill):
        """Test object type validation."""
        is_valid, _ = await skill.validate_input({"object_field": {"key": "value"}})
        assert is_valid is True

        is_valid, error = await skill.validate_input({"object_field": [1, 2, 3]})
        assert is_valid is False
        assert "object" in error.lower()

    @pytest.mark.asyncio
    async def test_validate_number_accepts_int_and_float(self, skill: TypeValidationSkill):
        """Test number type accepts both int and float."""
        is_valid, _ = await skill.validate_input({"number_field": 42})
        assert is_valid is True

        is_valid, _ = await skill.validate_input({"number_field": 3.14})
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_empty_schema(self):
        """Test validation with empty schema accepts anything."""

        class EmptySchemaSkill(Skill):
            @property
            def manifest(self) -> SkillManifest:
                return SkillManifest(
                    name="empty_schema",
                    version="1.0.0",
                    capabilities=[],
                    input_schema={},
                )

            async def execute(
                self, input_data: Dict[str, Any], context: SkillContext
            ) -> SkillResult:
                return SkillResult.create_success({})

        skill = EmptySchemaSkill()
        is_valid, _ = await skill.validate_input({"anything": "goes"})
        assert is_valid is True
