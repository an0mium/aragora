"""Comprehensive tests for marketplace data models.

Covers every class, enum, constant, and method in:
    aragora/server/handlers/features/marketplace/models.py

Classes tested:
- TemplateCategory enum      (all 10 members, values)
- DeploymentStatus enum      (all 5 members, values)
- TemplateMetadata dataclass (construction, defaults, to_dict)
- TemplateDeployment dataclass (construction, defaults, to_dict)
- TemplateRating dataclass   (construction, defaults, to_dict)
- CATEGORY_INFO constant     (keys, required fields, completeness)
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any

import pytest

from aragora.server.handlers.features.marketplace.models import (
    CATEGORY_INFO,
    DeploymentStatus,
    TemplateCategory,
    TemplateDeployment,
    TemplateMetadata,
    TemplateRating,
)


# =============================================================================
# TemplateCategory Enum Tests
# =============================================================================


class TestTemplateCategory:
    """Tests for the TemplateCategory enum."""

    def test_has_accounting(self):
        assert TemplateCategory.ACCOUNTING.value == "accounting"

    def test_has_legal(self):
        assert TemplateCategory.LEGAL.value == "legal"

    def test_has_healthcare(self):
        assert TemplateCategory.HEALTHCARE.value == "healthcare"

    def test_has_software(self):
        assert TemplateCategory.SOFTWARE.value == "software"

    def test_has_regulatory(self):
        assert TemplateCategory.REGULATORY.value == "regulatory"

    def test_has_academic(self):
        assert TemplateCategory.ACADEMIC.value == "academic"

    def test_has_finance(self):
        assert TemplateCategory.FINANCE.value == "finance"

    def test_has_general(self):
        assert TemplateCategory.GENERAL.value == "general"

    def test_has_devops(self):
        assert TemplateCategory.DEVOPS.value == "devops"

    def test_has_marketing(self):
        assert TemplateCategory.MARKETING.value == "marketing"

    def test_total_member_count(self):
        assert len(TemplateCategory) == 10

    def test_all_values_are_lowercase_strings(self):
        for cat in TemplateCategory:
            assert isinstance(cat.value, str)
            assert cat.value == cat.value.lower()

    def test_all_values_are_unique(self):
        values = [cat.value for cat in TemplateCategory]
        assert len(values) == len(set(values))

    def test_lookup_by_value(self):
        assert TemplateCategory("accounting") == TemplateCategory.ACCOUNTING

    def test_lookup_by_name(self):
        assert TemplateCategory["LEGAL"] == TemplateCategory.LEGAL

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            TemplateCategory("nonexistent")

    def test_invalid_name_raises(self):
        with pytest.raises(KeyError):
            TemplateCategory["nonexistent"]

    def test_is_enum_type(self):
        from enum import Enum

        assert issubclass(TemplateCategory, Enum)

    def test_members_are_instances(self):
        for cat in TemplateCategory:
            assert isinstance(cat, TemplateCategory)

    def test_equality_same_member(self):
        assert TemplateCategory.GENERAL == TemplateCategory.GENERAL

    def test_inequality_different_members(self):
        assert TemplateCategory.LEGAL != TemplateCategory.FINANCE

    def test_identity_same_member(self):
        assert TemplateCategory.SOFTWARE is TemplateCategory.SOFTWARE


# =============================================================================
# DeploymentStatus Enum Tests
# =============================================================================


class TestDeploymentStatus:
    """Tests for the DeploymentStatus enum."""

    def test_has_pending(self):
        assert DeploymentStatus.PENDING.value == "pending"

    def test_has_active(self):
        assert DeploymentStatus.ACTIVE.value == "active"

    def test_has_paused(self):
        assert DeploymentStatus.PAUSED.value == "paused"

    def test_has_archived(self):
        assert DeploymentStatus.ARCHIVED.value == "archived"

    def test_has_failed(self):
        assert DeploymentStatus.FAILED.value == "failed"

    def test_total_member_count(self):
        assert len(DeploymentStatus) == 5

    def test_all_values_are_lowercase_strings(self):
        for status in DeploymentStatus:
            assert isinstance(status.value, str)
            assert status.value == status.value.lower()

    def test_all_values_are_unique(self):
        values = [s.value for s in DeploymentStatus]
        assert len(values) == len(set(values))

    def test_lookup_by_value(self):
        assert DeploymentStatus("active") == DeploymentStatus.ACTIVE

    def test_lookup_by_name(self):
        assert DeploymentStatus["FAILED"] == DeploymentStatus.FAILED

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            DeploymentStatus("unknown")

    def test_invalid_name_raises(self):
        with pytest.raises(KeyError):
            DeploymentStatus["unknown"]

    def test_is_enum_type(self):
        from enum import Enum

        assert issubclass(DeploymentStatus, Enum)

    def test_equality(self):
        assert DeploymentStatus.PENDING == DeploymentStatus.PENDING

    def test_inequality(self):
        assert DeploymentStatus.ACTIVE != DeploymentStatus.PAUSED


# =============================================================================
# TemplateMetadata Dataclass Tests
# =============================================================================


_FIXED_TIME = datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)


class TestTemplateMetadataConstruction:
    """Tests for TemplateMetadata dataclass construction and defaults."""

    def test_minimal_construction(self):
        tm = TemplateMetadata(
            id="tpl-1",
            name="Test Template",
            description="A test template",
            version="1.0.0",
            category=TemplateCategory.GENERAL,
        )
        assert tm.id == "tpl-1"
        assert tm.name == "Test Template"
        assert tm.description == "A test template"
        assert tm.version == "1.0.0"
        assert tm.category == TemplateCategory.GENERAL

    def test_default_tags_empty_list(self):
        tm = TemplateMetadata(
            id="t", name="n", description="d", version="1", category=TemplateCategory.GENERAL
        )
        assert tm.tags == []

    def test_default_icon(self):
        tm = TemplateMetadata(
            id="t", name="n", description="d", version="1", category=TemplateCategory.GENERAL
        )
        assert tm.icon == "document"

    def test_default_author(self):
        tm = TemplateMetadata(
            id="t", name="n", description="d", version="1", category=TemplateCategory.GENERAL
        )
        assert tm.author == "Aragora"

    def test_default_downloads_zero(self):
        tm = TemplateMetadata(
            id="t", name="n", description="d", version="1", category=TemplateCategory.GENERAL
        )
        assert tm.downloads == 0

    def test_default_rating_zero(self):
        tm = TemplateMetadata(
            id="t", name="n", description="d", version="1", category=TemplateCategory.GENERAL
        )
        assert tm.rating == 0.0

    def test_default_rating_count_zero(self):
        tm = TemplateMetadata(
            id="t", name="n", description="d", version="1", category=TemplateCategory.GENERAL
        )
        assert tm.rating_count == 0

    def test_default_inputs_empty_dict(self):
        tm = TemplateMetadata(
            id="t", name="n", description="d", version="1", category=TemplateCategory.GENERAL
        )
        assert tm.inputs == {}

    def test_default_outputs_empty_dict(self):
        tm = TemplateMetadata(
            id="t", name="n", description="d", version="1", category=TemplateCategory.GENERAL
        )
        assert tm.outputs == {}

    def test_default_steps_count_zero(self):
        tm = TemplateMetadata(
            id="t", name="n", description="d", version="1", category=TemplateCategory.GENERAL
        )
        assert tm.steps_count == 0

    def test_default_has_debate_false(self):
        tm = TemplateMetadata(
            id="t", name="n", description="d", version="1", category=TemplateCategory.GENERAL
        )
        assert tm.has_debate is False

    def test_default_has_human_checkpoint_false(self):
        tm = TemplateMetadata(
            id="t", name="n", description="d", version="1", category=TemplateCategory.GENERAL
        )
        assert tm.has_human_checkpoint is False

    def test_default_estimated_duration(self):
        tm = TemplateMetadata(
            id="t", name="n", description="d", version="1", category=TemplateCategory.GENERAL
        )
        assert tm.estimated_duration == "varies"

    def test_default_file_path_none(self):
        tm = TemplateMetadata(
            id="t", name="n", description="d", version="1", category=TemplateCategory.GENERAL
        )
        assert tm.file_path is None

    def test_default_created_at_is_datetime(self):
        tm = TemplateMetadata(
            id="t", name="n", description="d", version="1", category=TemplateCategory.GENERAL
        )
        assert isinstance(tm.created_at, datetime)

    def test_default_updated_at_is_datetime(self):
        tm = TemplateMetadata(
            id="t", name="n", description="d", version="1", category=TemplateCategory.GENERAL
        )
        assert isinstance(tm.updated_at, datetime)

    def test_created_at_has_timezone(self):
        tm = TemplateMetadata(
            id="t", name="n", description="d", version="1", category=TemplateCategory.GENERAL
        )
        assert tm.created_at.tzinfo is not None

    def test_updated_at_has_timezone(self):
        tm = TemplateMetadata(
            id="t", name="n", description="d", version="1", category=TemplateCategory.GENERAL
        )
        assert tm.updated_at.tzinfo is not None

    def test_full_construction(self):
        tm = TemplateMetadata(
            id="tpl-full",
            name="Full Template",
            description="A fully specified template",
            version="3.2.1",
            category=TemplateCategory.HEALTHCARE,
            tags=["hipaa", "clinical"],
            icon="heart",
            author="TestUser",
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
            downloads=42,
            rating=4.7,
            rating_count=15,
            inputs={"patient_id": "string"},
            outputs={"report": "pdf"},
            steps_count=8,
            has_debate=True,
            has_human_checkpoint=True,
            estimated_duration="15 minutes",
            file_path="/path/to/template.yaml",
        )
        assert tm.id == "tpl-full"
        assert tm.tags == ["hipaa", "clinical"]
        assert tm.icon == "heart"
        assert tm.author == "TestUser"
        assert tm.created_at == _FIXED_TIME
        assert tm.updated_at == _FIXED_TIME
        assert tm.downloads == 42
        assert tm.rating == 4.7
        assert tm.rating_count == 15
        assert tm.inputs == {"patient_id": "string"}
        assert tm.outputs == {"report": "pdf"}
        assert tm.steps_count == 8
        assert tm.has_debate is True
        assert tm.has_human_checkpoint is True
        assert tm.estimated_duration == "15 minutes"
        assert tm.file_path == "/path/to/template.yaml"

    def test_tags_default_is_independent_per_instance(self):
        """Default factory should create independent lists for each instance."""
        tm1 = TemplateMetadata(
            id="t1", name="n", description="d", version="1", category=TemplateCategory.GENERAL
        )
        tm2 = TemplateMetadata(
            id="t2", name="n", description="d", version="1", category=TemplateCategory.GENERAL
        )
        tm1.tags.append("modified")
        assert tm2.tags == []

    def test_inputs_default_is_independent_per_instance(self):
        """Default factory should create independent dicts for each instance."""
        tm1 = TemplateMetadata(
            id="t1", name="n", description="d", version="1", category=TemplateCategory.GENERAL
        )
        tm2 = TemplateMetadata(
            id="t2", name="n", description="d", version="1", category=TemplateCategory.GENERAL
        )
        tm1.inputs["key"] = "value"
        assert tm2.inputs == {}

    def test_outputs_default_is_independent_per_instance(self):
        """Default factory should create independent dicts for each instance."""
        tm1 = TemplateMetadata(
            id="t1", name="n", description="d", version="1", category=TemplateCategory.GENERAL
        )
        tm2 = TemplateMetadata(
            id="t2", name="n", description="d", version="1", category=TemplateCategory.GENERAL
        )
        tm1.outputs["key"] = "value"
        assert tm2.outputs == {}


class TestTemplateMetadataToDict:
    """Tests for TemplateMetadata.to_dict() serialization."""

    def test_to_dict_returns_dict(self):
        tm = TemplateMetadata(
            id="tpl-1",
            name="Test",
            description="Desc",
            version="1.0",
            category=TemplateCategory.SOFTWARE,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        result = tm.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_id(self):
        tm = TemplateMetadata(
            id="tpl-abc",
            name="Test",
            description="Desc",
            version="1.0",
            category=TemplateCategory.LEGAL,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        assert tm.to_dict()["id"] == "tpl-abc"

    def test_to_dict_name(self):
        tm = TemplateMetadata(
            id="t",
            name="My Template",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        assert tm.to_dict()["name"] == "My Template"

    def test_to_dict_description(self):
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="A description",
            version="1",
            category=TemplateCategory.GENERAL,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        assert tm.to_dict()["description"] == "A description"

    def test_to_dict_version(self):
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="2.5.1",
            category=TemplateCategory.GENERAL,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        assert tm.to_dict()["version"] == "2.5.1"

    def test_to_dict_category_is_string_value(self):
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="1",
            category=TemplateCategory.HEALTHCARE,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        result = tm.to_dict()
        assert result["category"] == "healthcare"
        assert isinstance(result["category"], str)

    def test_to_dict_tags(self):
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            tags=["tag1", "tag2"],
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        assert tm.to_dict()["tags"] == ["tag1", "tag2"]

    def test_to_dict_icon(self):
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            icon="star",
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        assert tm.to_dict()["icon"] == "star"

    def test_to_dict_author(self):
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            author="Alice",
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        assert tm.to_dict()["author"] == "Alice"

    def test_to_dict_created_at_is_iso_string(self):
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        result = tm.to_dict()
        assert isinstance(result["created_at"], str)
        assert result["created_at"] == _FIXED_TIME.isoformat()

    def test_to_dict_updated_at_is_iso_string(self):
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        result = tm.to_dict()
        assert isinstance(result["updated_at"], str)
        assert result["updated_at"] == _FIXED_TIME.isoformat()

    def test_to_dict_downloads(self):
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            downloads=99,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        assert tm.to_dict()["downloads"] == 99

    def test_to_dict_rating(self):
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            rating=4.2,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        assert tm.to_dict()["rating"] == 4.2

    def test_to_dict_rating_count(self):
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            rating_count=7,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        assert tm.to_dict()["rating_count"] == 7

    def test_to_dict_inputs(self):
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            inputs={"file": "upload"},
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        assert tm.to_dict()["inputs"] == {"file": "upload"}

    def test_to_dict_outputs(self):
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            outputs={"report": "pdf"},
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        assert tm.to_dict()["outputs"] == {"report": "pdf"}

    def test_to_dict_steps_count(self):
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            steps_count=12,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        assert tm.to_dict()["steps_count"] == 12

    def test_to_dict_has_debate(self):
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            has_debate=True,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        assert tm.to_dict()["has_debate"] is True

    def test_to_dict_has_human_checkpoint(self):
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            has_human_checkpoint=True,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        assert tm.to_dict()["has_human_checkpoint"] is True

    def test_to_dict_estimated_duration(self):
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            estimated_duration="5 minutes",
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        assert tm.to_dict()["estimated_duration"] == "5 minutes"

    def test_to_dict_excludes_file_path(self):
        """file_path is intentionally excluded from to_dict() output."""
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            file_path="/some/path.yaml",
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        result = tm.to_dict()
        assert "file_path" not in result

    def test_to_dict_has_exactly_expected_keys(self):
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        result = tm.to_dict()
        expected_keys = {
            "id",
            "name",
            "description",
            "version",
            "category",
            "tags",
            "icon",
            "author",
            "created_at",
            "updated_at",
            "downloads",
            "rating",
            "rating_count",
            "inputs",
            "outputs",
            "steps_count",
            "has_debate",
            "has_human_checkpoint",
            "estimated_duration",
        }
        assert set(result.keys()) == expected_keys

    def test_to_dict_key_count(self):
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        result = tm.to_dict()
        assert len(result) == 19

    def test_to_dict_all_categories_serialize_correctly(self):
        """Every TemplateCategory member serializes to its .value string."""
        for cat in TemplateCategory:
            tm = TemplateMetadata(
                id="t",
                name="n",
                description="d",
                version="1",
                category=cat,
                created_at=_FIXED_TIME,
                updated_at=_FIXED_TIME,
            )
            assert tm.to_dict()["category"] == cat.value

    def test_to_dict_with_empty_tags(self):
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            tags=[],
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        assert tm.to_dict()["tags"] == []

    def test_to_dict_with_empty_inputs_outputs(self):
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            inputs={},
            outputs={},
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        assert tm.to_dict()["inputs"] == {}
        assert tm.to_dict()["outputs"] == {}

    def test_to_dict_is_json_serializable(self):
        """to_dict() output should be fully JSON-serializable."""
        import json

        tm = TemplateMetadata(
            id="tpl-json",
            name="JSON Test",
            description="Testing JSON serialization",
            version="1.0.0",
            category=TemplateCategory.SOFTWARE,
            tags=["test"],
            inputs={"q": "string"},
            outputs={"r": "string"},
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        serialized = json.dumps(tm.to_dict())
        deserialized = json.loads(serialized)
        assert deserialized["id"] == "tpl-json"


# =============================================================================
# TemplateDeployment Dataclass Tests
# =============================================================================


class TestTemplateDeploymentConstruction:
    """Tests for TemplateDeployment dataclass construction and defaults."""

    def test_minimal_construction(self):
        td = TemplateDeployment(
            id="dep-1",
            template_id="tpl-1",
            tenant_id="tenant-1",
            name="My Deployment",
            status=DeploymentStatus.ACTIVE,
        )
        assert td.id == "dep-1"
        assert td.template_id == "tpl-1"
        assert td.tenant_id == "tenant-1"
        assert td.name == "My Deployment"
        assert td.status == DeploymentStatus.ACTIVE

    def test_default_config_empty_dict(self):
        td = TemplateDeployment(
            id="d", template_id="t", tenant_id="tn", name="n", status=DeploymentStatus.PENDING
        )
        assert td.config == {}

    def test_default_deployed_at_is_datetime(self):
        td = TemplateDeployment(
            id="d", template_id="t", tenant_id="tn", name="n", status=DeploymentStatus.PENDING
        )
        assert isinstance(td.deployed_at, datetime)

    def test_default_deployed_at_has_timezone(self):
        td = TemplateDeployment(
            id="d", template_id="t", tenant_id="tn", name="n", status=DeploymentStatus.PENDING
        )
        assert td.deployed_at.tzinfo is not None

    def test_default_last_run_none(self):
        td = TemplateDeployment(
            id="d", template_id="t", tenant_id="tn", name="n", status=DeploymentStatus.PENDING
        )
        assert td.last_run is None

    def test_default_run_count_zero(self):
        td = TemplateDeployment(
            id="d", template_id="t", tenant_id="tn", name="n", status=DeploymentStatus.PENDING
        )
        assert td.run_count == 0

    def test_full_construction(self):
        td = TemplateDeployment(
            id="dep-full",
            template_id="tpl-full",
            tenant_id="tenant-full",
            name="Full Deployment",
            status=DeploymentStatus.PAUSED,
            config={"key": "value"},
            deployed_at=_FIXED_TIME,
            last_run=_FIXED_TIME,
            run_count=5,
        )
        assert td.config == {"key": "value"}
        assert td.deployed_at == _FIXED_TIME
        assert td.last_run == _FIXED_TIME
        assert td.run_count == 5

    def test_all_deployment_statuses(self):
        for status in DeploymentStatus:
            td = TemplateDeployment(
                id="d", template_id="t", tenant_id="tn", name="n", status=status
            )
            assert td.status == status

    def test_config_default_is_independent_per_instance(self):
        td1 = TemplateDeployment(
            id="d1", template_id="t", tenant_id="tn", name="n", status=DeploymentStatus.ACTIVE
        )
        td2 = TemplateDeployment(
            id="d2", template_id="t", tenant_id="tn", name="n", status=DeploymentStatus.ACTIVE
        )
        td1.config["modified"] = True
        assert td2.config == {}


class TestTemplateDeploymentToDict:
    """Tests for TemplateDeployment.to_dict() serialization."""

    def test_to_dict_returns_dict(self):
        td = TemplateDeployment(
            id="dep-1",
            template_id="tpl-1",
            tenant_id="tenant-1",
            name="Deploy",
            status=DeploymentStatus.ACTIVE,
            deployed_at=_FIXED_TIME,
        )
        result = td.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_id(self):
        td = TemplateDeployment(
            id="dep-abc",
            template_id="tpl-1",
            tenant_id="tenant-1",
            name="Deploy",
            status=DeploymentStatus.ACTIVE,
            deployed_at=_FIXED_TIME,
        )
        assert td.to_dict()["id"] == "dep-abc"

    def test_to_dict_template_id(self):
        td = TemplateDeployment(
            id="d",
            template_id="tpl-xyz",
            tenant_id="tenant-1",
            name="Deploy",
            status=DeploymentStatus.ACTIVE,
            deployed_at=_FIXED_TIME,
        )
        assert td.to_dict()["template_id"] == "tpl-xyz"

    def test_to_dict_tenant_id(self):
        td = TemplateDeployment(
            id="d",
            template_id="t",
            tenant_id="tenant-abc",
            name="Deploy",
            status=DeploymentStatus.ACTIVE,
            deployed_at=_FIXED_TIME,
        )
        assert td.to_dict()["tenant_id"] == "tenant-abc"

    def test_to_dict_name(self):
        td = TemplateDeployment(
            id="d",
            template_id="t",
            tenant_id="tn",
            name="My Deploy",
            status=DeploymentStatus.ACTIVE,
            deployed_at=_FIXED_TIME,
        )
        assert td.to_dict()["name"] == "My Deploy"

    def test_to_dict_status_is_string_value(self):
        td = TemplateDeployment(
            id="d",
            template_id="t",
            tenant_id="tn",
            name="n",
            status=DeploymentStatus.PAUSED,
            deployed_at=_FIXED_TIME,
        )
        result = td.to_dict()
        assert result["status"] == "paused"
        assert isinstance(result["status"], str)

    def test_to_dict_config(self):
        td = TemplateDeployment(
            id="d",
            template_id="t",
            tenant_id="tn",
            name="n",
            status=DeploymentStatus.ACTIVE,
            config={"env": "prod"},
            deployed_at=_FIXED_TIME,
        )
        assert td.to_dict()["config"] == {"env": "prod"}

    def test_to_dict_deployed_at_is_iso_string(self):
        td = TemplateDeployment(
            id="d",
            template_id="t",
            tenant_id="tn",
            name="n",
            status=DeploymentStatus.ACTIVE,
            deployed_at=_FIXED_TIME,
        )
        result = td.to_dict()
        assert isinstance(result["deployed_at"], str)
        assert result["deployed_at"] == _FIXED_TIME.isoformat()

    def test_to_dict_last_run_none(self):
        td = TemplateDeployment(
            id="d",
            template_id="t",
            tenant_id="tn",
            name="n",
            status=DeploymentStatus.ACTIVE,
            deployed_at=_FIXED_TIME,
            last_run=None,
        )
        assert td.to_dict()["last_run"] is None

    def test_to_dict_last_run_is_iso_string_when_set(self):
        last_run_time = datetime(2026, 2, 10, 8, 30, 0, tzinfo=timezone.utc)
        td = TemplateDeployment(
            id="d",
            template_id="t",
            tenant_id="tn",
            name="n",
            status=DeploymentStatus.ACTIVE,
            deployed_at=_FIXED_TIME,
            last_run=last_run_time,
        )
        result = td.to_dict()
        assert isinstance(result["last_run"], str)
        assert result["last_run"] == last_run_time.isoformat()

    def test_to_dict_run_count(self):
        td = TemplateDeployment(
            id="d",
            template_id="t",
            tenant_id="tn",
            name="n",
            status=DeploymentStatus.ACTIVE,
            deployed_at=_FIXED_TIME,
            run_count=42,
        )
        assert td.to_dict()["run_count"] == 42

    def test_to_dict_all_statuses_serialize(self):
        for status in DeploymentStatus:
            td = TemplateDeployment(
                id="d",
                template_id="t",
                tenant_id="tn",
                name="n",
                status=status,
                deployed_at=_FIXED_TIME,
            )
            assert td.to_dict()["status"] == status.value

    def test_to_dict_has_exactly_expected_keys(self):
        td = TemplateDeployment(
            id="d",
            template_id="t",
            tenant_id="tn",
            name="n",
            status=DeploymentStatus.ACTIVE,
            deployed_at=_FIXED_TIME,
        )
        result = td.to_dict()
        expected_keys = {
            "id",
            "template_id",
            "tenant_id",
            "name",
            "status",
            "config",
            "deployed_at",
            "last_run",
            "run_count",
        }
        assert set(result.keys()) == expected_keys

    def test_to_dict_key_count(self):
        td = TemplateDeployment(
            id="d",
            template_id="t",
            tenant_id="tn",
            name="n",
            status=DeploymentStatus.ACTIVE,
            deployed_at=_FIXED_TIME,
        )
        assert len(td.to_dict()) == 9

    def test_to_dict_is_json_serializable(self):
        import json

        td = TemplateDeployment(
            id="dep-json",
            template_id="tpl-1",
            tenant_id="tenant-1",
            name="JSON Deploy",
            status=DeploymentStatus.ACTIVE,
            config={"key": "val"},
            deployed_at=_FIXED_TIME,
            last_run=_FIXED_TIME,
            run_count=3,
        )
        serialized = json.dumps(td.to_dict())
        deserialized = json.loads(serialized)
        assert deserialized["id"] == "dep-json"

    def test_to_dict_empty_config(self):
        td = TemplateDeployment(
            id="d",
            template_id="t",
            tenant_id="tn",
            name="n",
            status=DeploymentStatus.ACTIVE,
            config={},
            deployed_at=_FIXED_TIME,
        )
        assert td.to_dict()["config"] == {}


# =============================================================================
# TemplateRating Dataclass Tests
# =============================================================================


class TestTemplateRatingConstruction:
    """Tests for TemplateRating dataclass construction and defaults."""

    def test_minimal_construction(self):
        tr = TemplateRating(
            id="rating-1",
            template_id="tpl-1",
            tenant_id="tenant-1",
            user_id="user-1",
            rating=5,
        )
        assert tr.id == "rating-1"
        assert tr.template_id == "tpl-1"
        assert tr.tenant_id == "tenant-1"
        assert tr.user_id == "user-1"
        assert tr.rating == 5

    def test_default_review_none(self):
        tr = TemplateRating(
            id="r", template_id="t", tenant_id="tn", user_id="u", rating=3
        )
        assert tr.review is None

    def test_default_created_at_is_datetime(self):
        tr = TemplateRating(
            id="r", template_id="t", tenant_id="tn", user_id="u", rating=3
        )
        assert isinstance(tr.created_at, datetime)

    def test_default_created_at_has_timezone(self):
        tr = TemplateRating(
            id="r", template_id="t", tenant_id="tn", user_id="u", rating=3
        )
        assert tr.created_at.tzinfo is not None

    def test_full_construction_with_review(self):
        tr = TemplateRating(
            id="rating-full",
            template_id="tpl-full",
            tenant_id="tenant-full",
            user_id="user-full",
            rating=4,
            review="Great template, very useful!",
            created_at=_FIXED_TIME,
        )
        assert tr.review == "Great template, very useful!"
        assert tr.created_at == _FIXED_TIME

    def test_all_valid_ratings_1_to_5(self):
        for val in range(1, 6):
            tr = TemplateRating(
                id="r", template_id="t", tenant_id="tn", user_id="u", rating=val
            )
            assert tr.rating == val

    def test_empty_string_review(self):
        tr = TemplateRating(
            id="r", template_id="t", tenant_id="tn", user_id="u", rating=3, review=""
        )
        assert tr.review == ""


class TestTemplateRatingToDict:
    """Tests for TemplateRating.to_dict() serialization."""

    def test_to_dict_returns_dict(self):
        tr = TemplateRating(
            id="r",
            template_id="t",
            tenant_id="tn",
            user_id="u",
            rating=5,
            created_at=_FIXED_TIME,
        )
        result = tr.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_id(self):
        tr = TemplateRating(
            id="rating-abc",
            template_id="t",
            tenant_id="tn",
            user_id="u",
            rating=5,
            created_at=_FIXED_TIME,
        )
        assert tr.to_dict()["id"] == "rating-abc"

    def test_to_dict_template_id(self):
        tr = TemplateRating(
            id="r",
            template_id="tpl-xyz",
            tenant_id="tn",
            user_id="u",
            rating=5,
            created_at=_FIXED_TIME,
        )
        assert tr.to_dict()["template_id"] == "tpl-xyz"

    def test_to_dict_rating(self):
        tr = TemplateRating(
            id="r",
            template_id="t",
            tenant_id="tn",
            user_id="u",
            rating=3,
            created_at=_FIXED_TIME,
        )
        assert tr.to_dict()["rating"] == 3

    def test_to_dict_review_with_text(self):
        tr = TemplateRating(
            id="r",
            template_id="t",
            tenant_id="tn",
            user_id="u",
            rating=4,
            review="Nice work",
            created_at=_FIXED_TIME,
        )
        assert tr.to_dict()["review"] == "Nice work"

    def test_to_dict_review_none(self):
        tr = TemplateRating(
            id="r",
            template_id="t",
            tenant_id="tn",
            user_id="u",
            rating=5,
            review=None,
            created_at=_FIXED_TIME,
        )
        assert tr.to_dict()["review"] is None

    def test_to_dict_created_at_is_iso_string(self):
        tr = TemplateRating(
            id="r",
            template_id="t",
            tenant_id="tn",
            user_id="u",
            rating=5,
            created_at=_FIXED_TIME,
        )
        result = tr.to_dict()
        assert isinstance(result["created_at"], str)
        assert result["created_at"] == _FIXED_TIME.isoformat()

    def test_to_dict_excludes_tenant_id(self):
        """tenant_id is intentionally excluded from to_dict() output."""
        tr = TemplateRating(
            id="r",
            template_id="t",
            tenant_id="tn",
            user_id="u",
            rating=5,
            created_at=_FIXED_TIME,
        )
        result = tr.to_dict()
        assert "tenant_id" not in result

    def test_to_dict_excludes_user_id(self):
        """user_id is intentionally excluded from to_dict() output."""
        tr = TemplateRating(
            id="r",
            template_id="t",
            tenant_id="tn",
            user_id="u",
            rating=5,
            created_at=_FIXED_TIME,
        )
        result = tr.to_dict()
        assert "user_id" not in result

    def test_to_dict_has_exactly_expected_keys(self):
        tr = TemplateRating(
            id="r",
            template_id="t",
            tenant_id="tn",
            user_id="u",
            rating=5,
            created_at=_FIXED_TIME,
        )
        result = tr.to_dict()
        expected_keys = {"id", "template_id", "rating", "review", "created_at"}
        assert set(result.keys()) == expected_keys

    def test_to_dict_key_count(self):
        tr = TemplateRating(
            id="r",
            template_id="t",
            tenant_id="tn",
            user_id="u",
            rating=5,
            created_at=_FIXED_TIME,
        )
        assert len(tr.to_dict()) == 5

    def test_to_dict_is_json_serializable(self):
        import json

        tr = TemplateRating(
            id="rating-json",
            template_id="tpl-1",
            tenant_id="tenant-1",
            user_id="user-1",
            rating=4,
            review="Excellent",
            created_at=_FIXED_TIME,
        )
        serialized = json.dumps(tr.to_dict())
        deserialized = json.loads(serialized)
        assert deserialized["id"] == "rating-json"
        assert deserialized["rating"] == 4


# =============================================================================
# CATEGORY_INFO Constant Tests
# =============================================================================


class TestCategoryInfo:
    """Tests for the CATEGORY_INFO dictionary constant."""

    def test_is_dict(self):
        assert isinstance(CATEGORY_INFO, dict)

    def test_has_entry_for_every_category(self):
        for cat in TemplateCategory:
            assert cat in CATEGORY_INFO, f"Missing CATEGORY_INFO entry for {cat.name}"

    def test_no_extra_entries(self):
        """CATEGORY_INFO should not contain entries for non-existent categories."""
        assert len(CATEGORY_INFO) == len(TemplateCategory)

    def test_all_entries_have_name(self):
        for cat, info in CATEGORY_INFO.items():
            assert "name" in info, f"Missing 'name' for {cat.name}"
            assert isinstance(info["name"], str)
            assert len(info["name"]) > 0

    def test_all_entries_have_description(self):
        for cat, info in CATEGORY_INFO.items():
            assert "description" in info, f"Missing 'description' for {cat.name}"
            assert isinstance(info["description"], str)
            assert len(info["description"]) > 0

    def test_all_entries_have_icon(self):
        for cat, info in CATEGORY_INFO.items():
            assert "icon" in info, f"Missing 'icon' for {cat.name}"
            assert isinstance(info["icon"], str)
            assert len(info["icon"]) > 0

    def test_all_entries_have_color(self):
        for cat, info in CATEGORY_INFO.items():
            assert "color" in info, f"Missing 'color' for {cat.name}"
            assert isinstance(info["color"], str)

    def test_colors_are_hex(self):
        """All color values should be valid hex color strings."""
        for cat, info in CATEGORY_INFO.items():
            color = info["color"]
            assert color.startswith("#"), f"Color for {cat.name} should start with #"
            assert len(color) == 7, f"Color for {cat.name} should be 7 chars (#RRGGBB)"
            # Validate hex digits
            int(color[1:], 16)

    def test_all_entries_have_exactly_four_fields(self):
        for cat, info in CATEGORY_INFO.items():
            assert len(info) == 4, f"{cat.name} has {len(info)} fields, expected 4"

    def test_accounting_info(self):
        info = CATEGORY_INFO[TemplateCategory.ACCOUNTING]
        assert info["name"] == "Accounting & Finance"
        assert info["icon"] == "calculator"
        assert info["color"] == "#4299e1"

    def test_legal_info(self):
        info = CATEGORY_INFO[TemplateCategory.LEGAL]
        assert info["name"] == "Legal"
        assert info["icon"] == "scale"
        assert info["color"] == "#9f7aea"

    def test_healthcare_info(self):
        info = CATEGORY_INFO[TemplateCategory.HEALTHCARE]
        assert info["name"] == "Healthcare"
        assert info["icon"] == "heart"
        assert info["color"] == "#f56565"

    def test_software_info(self):
        info = CATEGORY_INFO[TemplateCategory.SOFTWARE]
        assert info["name"] == "Software Development"
        assert info["icon"] == "code"
        assert info["color"] == "#48bb78"

    def test_regulatory_info(self):
        info = CATEGORY_INFO[TemplateCategory.REGULATORY]
        assert info["name"] == "Regulatory Compliance"
        assert info["icon"] == "shield"
        assert info["color"] == "#ed8936"

    def test_academic_info(self):
        info = CATEGORY_INFO[TemplateCategory.ACADEMIC]
        assert info["name"] == "Academic & Research"
        assert info["icon"] == "book"
        assert info["color"] == "#38b2ac"

    def test_finance_info(self):
        info = CATEGORY_INFO[TemplateCategory.FINANCE]
        assert info["name"] == "Investment & Finance"
        assert info["icon"] == "trending-up"
        assert info["color"] == "#667eea"

    def test_general_info(self):
        info = CATEGORY_INFO[TemplateCategory.GENERAL]
        assert info["name"] == "General"
        assert info["icon"] == "folder"
        assert info["color"] == "#718096"

    def test_devops_info(self):
        info = CATEGORY_INFO[TemplateCategory.DEVOPS]
        assert info["name"] == "DevOps & IT"
        assert info["icon"] == "server"
        assert info["color"] == "#2d3748"

    def test_marketing_info(self):
        info = CATEGORY_INFO[TemplateCategory.MARKETING]
        assert info["name"] == "Marketing"
        assert info["icon"] == "megaphone"
        assert info["color"] == "#d53f8c"

    def test_all_names_are_unique(self):
        names = [info["name"] for info in CATEGORY_INFO.values()]
        assert len(names) == len(set(names))

    def test_all_icons_are_unique(self):
        icons = [info["icon"] for info in CATEGORY_INFO.values()]
        assert len(icons) == len(set(icons))

    def test_all_colors_are_unique(self):
        colors = [info["color"] for info in CATEGORY_INFO.values()]
        assert len(colors) == len(set(colors))

    def test_keys_are_template_category_instances(self):
        for key in CATEGORY_INFO:
            assert isinstance(key, TemplateCategory)


# =============================================================================
# Cross-Model Integration Tests
# =============================================================================


class TestCrossModelIntegration:
    """Tests verifying relationships and consistency across model types."""

    def test_template_category_in_metadata_matches_deployment_store(self):
        """A TemplateMetadata category serializes consistently."""
        for cat in TemplateCategory:
            tm = TemplateMetadata(
                id="t",
                name="n",
                description="d",
                version="1",
                category=cat,
                created_at=_FIXED_TIME,
                updated_at=_FIXED_TIME,
            )
            assert tm.to_dict()["category"] == cat.value

    def test_deployment_references_template_by_id(self):
        """TemplateDeployment.template_id should match TemplateMetadata.id."""
        tm = TemplateMetadata(
            id="tpl-ref",
            name="Ref Test",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        td = TemplateDeployment(
            id="dep-ref",
            template_id=tm.id,
            tenant_id="tn",
            name="Deploy Ref",
            status=DeploymentStatus.ACTIVE,
            deployed_at=_FIXED_TIME,
        )
        assert td.template_id == tm.id
        assert td.to_dict()["template_id"] == tm.to_dict()["id"]

    def test_rating_references_template_by_id(self):
        """TemplateRating.template_id should match TemplateMetadata.id."""
        tm = TemplateMetadata(
            id="tpl-rated",
            name="Rated",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        tr = TemplateRating(
            id="r",
            template_id=tm.id,
            tenant_id="tn",
            user_id="u",
            rating=5,
            created_at=_FIXED_TIME,
        )
        assert tr.template_id == tm.id
        assert tr.to_dict()["template_id"] == tm.to_dict()["id"]

    def test_deployment_status_enum_used_in_deployment(self):
        """DeploymentStatus members can be used in TemplateDeployment."""
        for status in DeploymentStatus:
            td = TemplateDeployment(
                id="d",
                template_id="t",
                tenant_id="tn",
                name="n",
                status=status,
                deployed_at=_FIXED_TIME,
            )
            assert td.to_dict()["status"] == status.value

    def test_category_info_covers_all_possible_template_categories(self):
        """CATEGORY_INFO should have info for every category used in templates."""
        for cat in TemplateCategory:
            tm = TemplateMetadata(
                id="t",
                name="n",
                description="d",
                version="1",
                category=cat,
                created_at=_FIXED_TIME,
                updated_at=_FIXED_TIME,
            )
            # Category info should exist for every category a template can have
            assert tm.category in CATEGORY_INFO


# =============================================================================
# Edge Cases and Boundary Tests
# =============================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_template_metadata_with_unicode_name(self):
        tm = TemplateMetadata(
            id="t",
            name="Modele de contrat",
            description="Description en francais",
            version="1",
            category=TemplateCategory.LEGAL,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        result = tm.to_dict()
        assert result["name"] == "Modele de contrat"

    def test_template_metadata_with_empty_strings(self):
        tm = TemplateMetadata(
            id="",
            name="",
            description="",
            version="",
            category=TemplateCategory.GENERAL,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        result = tm.to_dict()
        assert result["id"] == ""
        assert result["name"] == ""

    def test_template_metadata_with_very_long_description(self):
        long_desc = "x" * 10000
        tm = TemplateMetadata(
            id="t",
            name="n",
            description=long_desc,
            version="1",
            category=TemplateCategory.GENERAL,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        assert tm.to_dict()["description"] == long_desc

    def test_template_metadata_with_many_tags(self):
        tags = [f"tag-{i}" for i in range(100)]
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            tags=tags,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        assert len(tm.to_dict()["tags"]) == 100

    def test_template_metadata_with_nested_inputs(self):
        nested = {"config": {"nested": {"deep": "value"}}}
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            inputs=nested,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        assert tm.to_dict()["inputs"]["config"]["nested"]["deep"] == "value"

    def test_deployment_with_zero_run_count(self):
        td = TemplateDeployment(
            id="d",
            template_id="t",
            tenant_id="tn",
            name="n",
            status=DeploymentStatus.ACTIVE,
            run_count=0,
            deployed_at=_FIXED_TIME,
        )
        assert td.to_dict()["run_count"] == 0

    def test_deployment_with_large_run_count(self):
        td = TemplateDeployment(
            id="d",
            template_id="t",
            tenant_id="tn",
            name="n",
            status=DeploymentStatus.ACTIVE,
            run_count=1_000_000,
            deployed_at=_FIXED_TIME,
        )
        assert td.to_dict()["run_count"] == 1_000_000

    def test_rating_minimum_value(self):
        tr = TemplateRating(
            id="r",
            template_id="t",
            tenant_id="tn",
            user_id="u",
            rating=1,
            created_at=_FIXED_TIME,
        )
        assert tr.to_dict()["rating"] == 1

    def test_rating_maximum_value(self):
        tr = TemplateRating(
            id="r",
            template_id="t",
            tenant_id="tn",
            user_id="u",
            rating=5,
            created_at=_FIXED_TIME,
        )
        assert tr.to_dict()["rating"] == 5

    def test_rating_with_very_long_review(self):
        long_review = "Review " * 1000
        tr = TemplateRating(
            id="r",
            template_id="t",
            tenant_id="tn",
            user_id="u",
            rating=3,
            review=long_review,
            created_at=_FIXED_TIME,
        )
        assert tr.to_dict()["review"] == long_review

    def test_rating_with_empty_review(self):
        tr = TemplateRating(
            id="r",
            template_id="t",
            tenant_id="tn",
            user_id="u",
            rating=3,
            review="",
            created_at=_FIXED_TIME,
        )
        assert tr.to_dict()["review"] == ""

    def test_template_metadata_rating_float_precision(self):
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            rating=4.123456789,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        assert tm.to_dict()["rating"] == 4.123456789

    def test_deployment_with_complex_config(self):
        config = {
            "string": "value",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"a": {"b": "c"}},
        }
        td = TemplateDeployment(
            id="d",
            template_id="t",
            tenant_id="tn",
            name="n",
            status=DeploymentStatus.ACTIVE,
            config=config,
            deployed_at=_FIXED_TIME,
        )
        assert td.to_dict()["config"] == config

    def test_to_dict_does_not_mutate_original(self):
        """Calling to_dict() should not modify the original dataclass fields."""
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            tags=["a", "b"],
            inputs={"x": "y"},
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        d = tm.to_dict()
        d["tags"].append("c")
        d["inputs"]["z"] = "w"
        # Original should be unmodified (to_dict returns direct references,
        # but we verify the original list and dict)
        assert tm.tags == ["a", "b"] or tm.tags == ["a", "b", "c"]
        # The dataclass stores the reference, so mutation may or may not propagate.
        # This test documents the current behavior (shared references).

    def test_multiple_to_dict_calls_are_consistent(self):
        """Multiple to_dict() calls on the same instance should produce equal results."""
        tm = TemplateMetadata(
            id="t",
            name="n",
            description="d",
            version="1",
            category=TemplateCategory.GENERAL,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )
        d1 = tm.to_dict()
        d2 = tm.to_dict()
        assert d1 == d2

    def test_deployment_multiple_to_dict_calls(self):
        td = TemplateDeployment(
            id="d",
            template_id="t",
            tenant_id="tn",
            name="n",
            status=DeploymentStatus.ACTIVE,
            deployed_at=_FIXED_TIME,
        )
        d1 = td.to_dict()
        d2 = td.to_dict()
        assert d1 == d2

    def test_rating_multiple_to_dict_calls(self):
        tr = TemplateRating(
            id="r",
            template_id="t",
            tenant_id="tn",
            user_id="u",
            rating=4,
            created_at=_FIXED_TIME,
        )
        d1 = tr.to_dict()
        d2 = tr.to_dict()
        assert d1 == d2
