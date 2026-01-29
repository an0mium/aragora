"""
Tests for aragora.server.openapi.schemas module.

Covers:
- COMMON_SCHEMAS dictionary structure and required schemas
- ok_response helper function
- array_response helper function
- error_response helper function
- STANDARD_ERRORS definitions
"""

from __future__ import annotations

import pytest

from aragora.server.openapi.schemas import (
    COMMON_SCHEMAS,
    STANDARD_ERRORS,
    array_response,
    error_response,
    ok_response,
)


# =============================================================================
# TestCommonSchemasStructure
# =============================================================================


class TestCommonSchemasStructure:
    """Tests for COMMON_SCHEMAS dictionary structure."""

    def test_common_schemas_is_dict(self):
        """COMMON_SCHEMAS should be a dictionary."""
        assert isinstance(COMMON_SCHEMAS, dict)

    def test_common_schemas_not_empty(self):
        """COMMON_SCHEMAS should contain schemas."""
        assert len(COMMON_SCHEMAS) > 0

    def test_all_schemas_are_dicts(self):
        """All schema values should be dictionaries."""
        for name, schema in COMMON_SCHEMAS.items():
            assert isinstance(schema, dict), f"Schema '{name}' is not a dict"


# =============================================================================
# TestRequiredSchemas
# =============================================================================


class TestRequiredSchemas:
    """Tests for required schema definitions."""

    @pytest.mark.parametrize(
        "schema_name",
        [
            "Error",
            "PaginatedResponse",
            "Agent",
            "DebateStatus",
            "ConsensusResult",
            "DebateCreateRequest",
            "DebateCreateResponse",
            "Debate",
            "Message",
            "HealthCheck",
            "Round",
        ],
    )
    def test_required_schema_exists(self, schema_name: str):
        """Required schemas should exist."""
        assert schema_name in COMMON_SCHEMAS

    def test_error_schema_has_required_error_field(self):
        """Error schema should have 'error' as required field."""
        error_schema = COMMON_SCHEMAS["Error"]
        assert "required" in error_schema
        assert "error" in error_schema["required"]

    def test_error_schema_has_error_codes(self):
        """Error schema should define error codes enum."""
        error_schema = COMMON_SCHEMAS["Error"]
        code_prop = error_schema["properties"]["code"]
        assert "enum" in code_prop
        assert "INVALID_JSON" in code_prop["enum"]
        assert "NOT_FOUND" in code_prop["enum"]
        assert "RATE_LIMITED" in code_prop["enum"]

    def test_debate_create_request_has_task_required(self):
        """DebateCreateRequest should require 'task' field."""
        schema = COMMON_SCHEMAS["DebateCreateRequest"]
        assert "required" in schema
        assert "task" in schema["required"]

    def test_debate_status_enum_values(self):
        """DebateStatus should have expected enum values."""
        schema = COMMON_SCHEMAS["DebateStatus"]
        assert schema["type"] == "string"
        assert "enum" in schema
        statuses = schema["enum"]
        assert "created" in statuses
        assert "running" in statuses
        assert "completed" in statuses
        assert "failed" in statuses


# =============================================================================
# TestSchemaProperties
# =============================================================================


class TestSchemaProperties:
    """Tests for schema property definitions."""

    def test_agent_schema_has_elo(self):
        """Agent schema should have elo property."""
        agent = COMMON_SCHEMAS["Agent"]
        assert "elo" in agent["properties"]
        assert agent["properties"]["elo"]["type"] == "number"

    def test_consensus_result_has_reached(self):
        """ConsensusResult should have 'reached' boolean."""
        schema = COMMON_SCHEMAS["ConsensusResult"]
        assert "reached" in schema["properties"]
        assert schema["properties"]["reached"]["type"] == "boolean"

    def test_health_check_has_status_enum(self):
        """HealthCheck should have status with enum values."""
        schema = COMMON_SCHEMAS["HealthCheck"]
        status_prop = schema["properties"]["status"]
        assert "enum" in status_prop
        assert "healthy" in status_prop["enum"]
        assert "degraded" in status_prop["enum"]
        assert "unhealthy" in status_prop["enum"]

    def test_message_has_role_enum(self):
        """Message schema should have role enum."""
        schema = COMMON_SCHEMAS["Message"]
        role_prop = schema["properties"]["role"]
        assert "enum" in role_prop
        assert "system" in role_prop["enum"]
        assert "user" in role_prop["enum"]
        assert "assistant" in role_prop["enum"]


# =============================================================================
# TestOkResponse
# =============================================================================


class TestOkResponse:
    """Tests for ok_response helper function."""

    def test_ok_response_basic(self):
        """ok_response should create basic response."""
        result = ok_response("Success")
        assert result["description"] == "Success"

    def test_ok_response_without_schema(self):
        """ok_response without schema should not have content."""
        result = ok_response("Success")
        assert "content" not in result

    def test_ok_response_with_schema_ref(self):
        """ok_response with schema should have content with $ref."""
        result = ok_response("Agent retrieved", "Agent")
        assert "content" in result
        assert "application/json" in result["content"]
        schema = result["content"]["application/json"]["schema"]
        assert "$ref" in schema
        assert schema["$ref"] == "#/components/schemas/Agent"

    def test_ok_response_schema_ref_format(self):
        """Schema reference should follow OpenAPI format."""
        result = ok_response("Test", "DebateCreateResponse")
        ref = result["content"]["application/json"]["schema"]["$ref"]
        assert ref.startswith("#/components/schemas/")


# =============================================================================
# TestArrayResponse
# =============================================================================


class TestArrayResponse:
    """Tests for array_response helper function."""

    def test_array_response_structure(self):
        """array_response should create proper structure."""
        result = array_response("List of agents", "Agent")
        assert result["description"] == "List of agents"
        assert "content" in result

    def test_array_response_has_items_array(self):
        """array_response should define items as array."""
        result = array_response("List", "Agent")
        content = result["content"]["application/json"]["schema"]
        assert content["type"] == "object"
        assert "items" in content["properties"]
        items_prop = content["properties"]["items"]
        assert items_prop["type"] == "array"

    def test_array_response_has_total(self):
        """array_response should include total count property."""
        result = array_response("List", "Agent")
        content = result["content"]["application/json"]["schema"]
        assert "total" in content["properties"]
        assert content["properties"]["total"]["type"] == "integer"

    def test_array_response_items_ref(self):
        """array_response should reference schema in items."""
        result = array_response("List", "Debate")
        items_def = result["content"]["application/json"]["schema"]["properties"]["items"]
        assert items_def["items"]["$ref"] == "#/components/schemas/Debate"


# =============================================================================
# TestErrorResponse
# =============================================================================


class TestErrorResponse:
    """Tests for error_response helper function."""

    def test_error_response_description(self):
        """error_response should set description."""
        result = error_response("404", "Not found")
        assert result["description"] == "Not found"

    def test_error_response_references_error_schema(self):
        """error_response should reference Error schema."""
        result = error_response("400", "Bad request")
        schema = result["content"]["application/json"]["schema"]
        assert schema["$ref"] == "#/components/schemas/Error"

    def test_error_response_content_type(self):
        """error_response should use application/json content type."""
        result = error_response("500", "Server error")
        assert "application/json" in result["content"]


# =============================================================================
# TestStandardErrors
# =============================================================================


class TestStandardErrors:
    """Tests for STANDARD_ERRORS definitions."""

    def test_standard_errors_is_dict(self):
        """STANDARD_ERRORS should be a dictionary."""
        assert isinstance(STANDARD_ERRORS, dict)

    @pytest.mark.parametrize("status_code", ["400", "401", "404", "429", "500"])
    def test_standard_errors_has_common_codes(self, status_code: str):
        """STANDARD_ERRORS should include common HTTP error codes."""
        assert status_code in STANDARD_ERRORS

    def test_standard_errors_402_for_quota(self):
        """STANDARD_ERRORS should include 402 for quota exceeded."""
        assert "402" in STANDARD_ERRORS

    def test_all_errors_reference_error_schema(self):
        """All standard errors should reference Error schema."""
        for status, response in STANDARD_ERRORS.items():
            schema = response["content"]["application/json"]["schema"]
            assert schema["$ref"] == "#/components/schemas/Error", f"Error {status} wrong ref"


# =============================================================================
# TestSpecializedSchemas
# =============================================================================


class TestSpecializedSchemas:
    """Tests for specialized schema definitions."""

    def test_decision_receipt_schema_exists(self):
        """DecisionReceipt schema should exist."""
        assert "DecisionReceipt" in COMMON_SCHEMAS
        schema = COMMON_SCHEMAS["DecisionReceipt"]
        assert "id" in schema["properties"]
        assert "debate_id" in schema["properties"]
        assert "verdict" in schema["properties"]

    def test_control_plane_agent_schema_exists(self):
        """ControlPlaneAgent schema should exist."""
        assert "ControlPlaneAgent" in COMMON_SCHEMAS
        schema = COMMON_SCHEMAS["ControlPlaneAgent"]
        assert "agent_id" in schema["properties"]
        assert "capabilities" in schema["properties"]
        assert "status" in schema["properties"]

    def test_workflow_schema_exists(self):
        """Workflow schema should exist."""
        assert "Workflow" in COMMON_SCHEMAS
        schema = COMMON_SCHEMAS["Workflow"]
        assert "id" in schema["properties"]
        assert "name" in schema["properties"]
        assert "steps" in schema["properties"]

    def test_decision_explanation_schema_exists(self):
        """DecisionExplanation schema should exist."""
        assert "DecisionExplanation" in COMMON_SCHEMAS
        schema = COMMON_SCHEMAS["DecisionExplanation"]
        assert "debate_id" in schema["properties"]
        assert "narrative" in schema["properties"]

    def test_evidence_chain_schema_exists(self):
        """EvidenceChain schema should exist."""
        assert "EvidenceChain" in COMMON_SCHEMAS
        schema = COMMON_SCHEMAS["EvidenceChain"]
        assert "evidence_count" in schema["properties"]


# =============================================================================
# TestSchemaReferences
# =============================================================================


class TestSchemaReferences:
    """Tests for schema cross-references."""

    def test_debate_references_consensus_result(self):
        """Debate schema should reference ConsensusResult."""
        debate = COMMON_SCHEMAS["Debate"]
        consensus_prop = debate["properties"]["consensus"]
        assert "$ref" in consensus_prop
        assert "ConsensusResult" in consensus_prop["$ref"]

    def test_debate_references_debate_status(self):
        """Debate schema should reference DebateStatus."""
        debate = COMMON_SCHEMAS["Debate"]
        status_prop = debate["properties"]["status"]
        assert "$ref" in status_prop
        assert "DebateStatus" in status_prop["$ref"]

    def test_round_references_message(self):
        """Round schema should reference Message."""
        round_schema = COMMON_SCHEMAS["Round"]
        messages_prop = round_schema["properties"]["messages"]
        assert messages_prop["type"] == "array"
        assert "Message" in messages_prop["items"]["$ref"]
