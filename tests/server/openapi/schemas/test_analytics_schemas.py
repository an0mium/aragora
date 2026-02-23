"""Tests for analytics OpenAPI schema definitions."""

import pytest

from aragora.server.openapi.schemas.analytics import ANALYTICS_SCHEMAS


class TestAnalyticsSchemasStructure:
    """Verify all expected schemas exist and have correct structure."""

    def test_schemas_dict_exists(self):
        assert isinstance(ANALYTICS_SCHEMAS, dict)
        assert len(ANALYTICS_SCHEMAS) > 0

    def test_all_schemas_are_dicts(self):
        for name, schema in ANALYTICS_SCHEMAS.items():
            assert isinstance(schema, dict), f"{name} is not a dict"

    def test_all_object_schemas_have_properties(self):
        for name, schema in ANALYTICS_SCHEMAS.items():
            if schema.get("type") == "object":
                assert "properties" in schema, f"{name} object missing 'properties'"


class TestDisagreementStats:
    def test_schema_exists(self):
        assert "DisagreementStats" in ANALYTICS_SCHEMAS

    def test_has_required_fields(self):
        props = ANALYTICS_SCHEMAS["DisagreementStats"]["properties"]
        assert "total_debates" in props
        assert "with_disagreements" in props
        assert "unanimous" in props
        assert "disagreement_types" in props

    def test_integer_fields(self):
        props = ANALYTICS_SCHEMAS["DisagreementStats"]["properties"]
        assert props["total_debates"]["type"] == "integer"
        assert props["with_disagreements"]["type"] == "integer"
        assert props["unanimous"]["type"] == "integer"


class TestRoleRotationStats:
    def test_schema_exists(self):
        assert "RoleRotationStats" in ANALYTICS_SCHEMAS

    def test_has_total_rotations(self):
        props = ANALYTICS_SCHEMAS["RoleRotationStats"]["properties"]
        assert "total_rotations" in props
        assert props["total_rotations"]["type"] == "integer"

    def test_by_agent_is_object(self):
        props = ANALYTICS_SCHEMAS["RoleRotationStats"]["properties"]
        assert props["by_agent"]["type"] == "object"


class TestEarlyStopStats:
    def test_schema_exists(self):
        assert "EarlyStopStats" in ANALYTICS_SCHEMAS

    def test_has_required_fields(self):
        props = ANALYTICS_SCHEMAS["EarlyStopStats"]["properties"]
        assert "total_early_stops" in props
        assert "by_reason" in props
        assert "average_rounds_saved" in props

    def test_average_is_number(self):
        props = ANALYTICS_SCHEMAS["EarlyStopStats"]["properties"]
        assert props["average_rounds_saved"]["type"] == "number"


class TestRankingStats:
    def test_schema_exists(self):
        assert "RankingStats" in ANALYTICS_SCHEMAS

    def test_has_required_fields(self):
        props = ANALYTICS_SCHEMAS["RankingStats"]["properties"]
        assert "total_agents" in props
        assert "average_elo" in props
        assert "highest_elo" in props
        assert "lowest_elo" in props
        assert "total_matches" in props


class TestPositionFlip:
    def test_schema_exists(self):
        assert "PositionFlip" in ANALYTICS_SCHEMAS

    def test_has_required_fields(self):
        props = ANALYTICS_SCHEMAS["PositionFlip"]["properties"]
        assert "debate_id" in props
        assert "agent" in props
        assert "round" in props
        assert "old_position" in props
        assert "new_position" in props
        assert "reason" in props
        assert "conviction_delta" in props
        assert "timestamp" in props

    def test_timestamp_format(self):
        props = ANALYTICS_SCHEMAS["PositionFlip"]["properties"]
        assert props["timestamp"]["format"] == "date-time"


class TestSchemaPropertyTypes:
    """Validate that property types are valid OpenAPI types."""

    VALID_TYPES = {"string", "integer", "number", "boolean", "array", "object"}

    def test_all_property_types_valid(self):
        for schema_name, schema in ANALYTICS_SCHEMAS.items():
            if schema.get("type") != "object":
                continue
            for prop_name, prop in schema.get("properties", {}).items():
                if "type" in prop:
                    # OpenAPI 3.1 allows type as list for nullable: ["string", "null"]
                    prop_type = prop["type"]
                    if isinstance(prop_type, list):
                        types = set(prop_type) - {"null"}
                        assert types <= self.VALID_TYPES, (
                            f"{schema_name}.{prop_name} has invalid type: {prop_type}"
                        )
                    else:
                        assert prop_type in self.VALID_TYPES, (
                            f"{schema_name}.{prop_name} has invalid type: {prop_type}"
                        )

    def test_array_properties_have_items(self):
        for schema_name, schema in ANALYTICS_SCHEMAS.items():
            if schema.get("type") != "object":
                continue
            for prop_name, prop in schema.get("properties", {}).items():
                if prop.get("type") == "array":
                    assert "items" in prop, f"{schema_name}.{prop_name} array missing 'items'"

    def test_descriptions_present(self):
        """Schemas should have descriptions for documentation."""
        schemas_with_desc = sum(1 for s in ANALYTICS_SCHEMAS.values() if "description" in s)
        # At least some schemas should have descriptions
        assert schemas_with_desc > 0
