"""
Tests for Workflow Template Packaging System.

Tests cover:
- Package creation and serialization
- Metadata handling
- Version management
- Package registry operations
- Checksum verification
"""

import json
import pytest
from datetime import datetime


# =============================================================================
# TemplateMetadata Tests
# =============================================================================


class TestTemplateMetadata:
    """Tests for TemplateMetadata dataclass."""

    def test_create_minimal_metadata(self):
        """Should create metadata with required fields."""
        from aragora.workflow.templates.package import TemplateMetadata

        metadata = TemplateMetadata(
            id="test-template",
            name="Test Template",
            version="1.0.0",
        )

        assert metadata.id == "test-template"
        assert metadata.name == "Test Template"
        assert metadata.version == "1.0.0"

    def test_create_full_metadata(self):
        """Should create metadata with all fields."""
        from aragora.workflow.templates.package import (
            TemplateMetadata,
            TemplateAuthor,
            TemplateCategory,
            TemplateStatus,
        )

        metadata = TemplateMetadata(
            id="legal/contract-review",
            name="Contract Review",
            version="2.1.0",
            description="Review contracts for risks",
            category=TemplateCategory.LEGAL,
            tags=["legal", "contracts", "compliance"],
            status=TemplateStatus.STABLE,
            author=TemplateAuthor(name="Test Author", email="test@example.com"),
            complexity="medium",
            estimated_duration="10-15 minutes",
        )

        assert metadata.category == TemplateCategory.LEGAL
        assert metadata.status == TemplateStatus.STABLE
        assert "legal" in metadata.tags
        assert metadata.author.email == "test@example.com"

    def test_metadata_to_dict(self):
        """Should convert metadata to dictionary."""
        from aragora.workflow.templates.package import (
            TemplateMetadata,
            TemplateCategory,
        )

        metadata = TemplateMetadata(
            id="test",
            name="Test",
            version="1.0.0",
            category=TemplateCategory.CODE,
        )

        data = metadata.to_dict()
        assert data["id"] == "test"
        assert data["category"] == "code"  # Enum converted to value

    def test_metadata_from_dict(self):
        """Should create metadata from dictionary."""
        from aragora.workflow.templates.package import (
            TemplateMetadata,
            TemplateCategory,
            TemplateStatus,
        )

        data = {
            "id": "test",
            "name": "Test",
            "version": "1.0.0",
            "category": "healthcare",
            "status": "stable",
        }

        metadata = TemplateMetadata.from_dict(data)
        assert metadata.id == "test"
        assert metadata.category == TemplateCategory.HEALTHCARE
        assert metadata.status == TemplateStatus.STABLE


# =============================================================================
# TemplatePackage Tests
# =============================================================================


class TestTemplatePackage:
    """Tests for TemplatePackage class."""

    def test_create_package(self):
        """Should create a template package."""
        from aragora.workflow.templates.package import (
            TemplatePackage,
            TemplateMetadata,
        )

        metadata = TemplateMetadata(
            id="test-package",
            name="Test Package",
            version="1.0.0",
        )

        template = {
            "name": "Test Template",
            "steps": [
                {"name": "step1", "type": "debate"},
            ],
        }

        package = TemplatePackage(
            metadata=metadata,
            template=template,
            readme="# Test Package\nA test workflow template.",
        )

        assert package.id == "test-package"
        assert package.version == "1.0.0"
        assert package.template["name"] == "Test Template"
        assert "# Test Package" in package.readme

    def test_package_checksum(self):
        """Should compute checksum on creation."""
        from aragora.workflow.templates.package import (
            TemplatePackage,
            TemplateMetadata,
        )

        metadata = TemplateMetadata(id="test", name="Test", version="1.0.0")
        template = {"steps": []}

        package = TemplatePackage(metadata=metadata, template=template)

        assert package.checksum is not None
        assert len(package.checksum) == 16  # Truncated SHA-256

    def test_package_verify_checksum(self):
        """Should verify checksum integrity."""
        from aragora.workflow.templates.package import (
            TemplatePackage,
            TemplateMetadata,
        )

        metadata = TemplateMetadata(id="test", name="Test", version="1.0.0")
        template = {"steps": [{"name": "step1"}]}

        package = TemplatePackage(metadata=metadata, template=template)

        # Valid checksum
        assert package.verify_checksum() is True

        # Tamper with template
        package.template["steps"].append({"name": "tampered"})
        assert package.verify_checksum() is False

    def test_package_to_dict(self):
        """Should convert package to dictionary."""
        from aragora.workflow.templates.package import (
            TemplatePackage,
            TemplateMetadata,
        )

        metadata = TemplateMetadata(id="test", name="Test", version="1.0.0")
        package = TemplatePackage(
            metadata=metadata,
            template={"steps": []},
            readme="Test readme",
        )

        data = package.to_dict()
        assert data["package_version"] == "1.0"
        assert "metadata" in data
        assert "template" in data
        assert data["readme"] == "Test readme"

    def test_package_to_json(self):
        """Should serialize to JSON."""
        from aragora.workflow.templates.package import (
            TemplatePackage,
            TemplateMetadata,
        )

        metadata = TemplateMetadata(id="test", name="Test", version="1.0.0")
        package = TemplatePackage(metadata=metadata, template={"steps": []})

        json_str = package.to_json()
        assert isinstance(json_str, str)

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["metadata"]["id"] == "test"

    def test_package_from_dict(self):
        """Should create package from dictionary."""
        from aragora.workflow.templates.package import TemplatePackage

        data = {
            "package_version": "1.0",
            "metadata": {
                "id": "restored",
                "name": "Restored Package",
                "version": "2.0.0",
            },
            "template": {"steps": [{"name": "step1"}]},
            "readme": "Restored readme",
            "checksum": "abc123",
        }

        package = TemplatePackage.from_dict(data)
        assert package.id == "restored"
        assert package.version == "2.0.0"
        assert package.readme == "Restored readme"

    def test_package_from_json(self):
        """Should deserialize from JSON."""
        from aragora.workflow.templates.package import TemplatePackage

        json_str = json.dumps(
            {
                "package_version": "1.0",
                "metadata": {"id": "json-test", "name": "JSON Test", "version": "1.0.0"},
                "template": {"steps": []},
            }
        )

        package = TemplatePackage.from_json(json_str)
        assert package.id == "json-test"

    def test_package_is_stable(self):
        """Should check stable status."""
        from aragora.workflow.templates.package import (
            TemplatePackage,
            TemplateMetadata,
            TemplateStatus,
        )

        stable_metadata = TemplateMetadata(
            id="stable",
            name="Stable",
            version="1.0.0",
            status=TemplateStatus.STABLE,
        )
        stable_package = TemplatePackage(metadata=stable_metadata, template={})
        assert stable_package.is_stable is True

        draft_metadata = TemplateMetadata(
            id="draft",
            name="Draft",
            version="0.1.0",
            status=TemplateStatus.DRAFT,
        )
        draft_package = TemplatePackage(metadata=draft_metadata, template={})
        assert draft_package.is_stable is False

    def test_package_is_deprecated(self):
        """Should check deprecated status."""
        from aragora.workflow.templates.package import (
            TemplatePackage,
            TemplateMetadata,
            TemplateStatus,
        )

        deprecated_metadata = TemplateMetadata(
            id="old",
            name="Old Template",
            version="1.0.0",
            status=TemplateStatus.DEPRECATED,
            successor="new-template",
        )
        package = TemplatePackage(metadata=deprecated_metadata, template={})
        assert package.is_deprecated is True


# =============================================================================
# create_package Tests
# =============================================================================


class TestCreatePackage:
    """Tests for create_package helper function."""

    def test_create_package_from_template(self):
        """Should create package from template dict."""
        from aragora.workflow.templates.package import create_package

        template = {
            "id": "my-template",
            "name": "My Template",
            "description": "A test template",
            "steps": [{"name": "step1", "type": "debate"}],
        }

        package = create_package(
            template=template,
            version="1.0.0",
            author="Test Author",
        )

        assert package.id == "my-template"
        assert package.metadata.author.name == "Test Author"

    def test_create_package_infers_id(self):
        """Should infer ID from name if not provided."""
        from aragora.workflow.templates.package import create_package

        template = {"name": "Contract Review Template"}

        package = create_package(template=template, version="1.0.0")

        assert package.id == "contract-review-template"

    def test_create_package_with_category(self):
        """Should set category correctly."""
        from aragora.workflow.templates.package import (
            create_package,
            TemplateCategory,
        )

        template = {"name": "Test"}

        package = create_package(
            template=template,
            version="1.0.0",
            category="legal",
        )

        assert package.metadata.category == TemplateCategory.LEGAL


# =============================================================================
# Package Registry Tests
# =============================================================================


class TestPackageRegistry:
    """Tests for package registry functions."""

    def test_register_package(self):
        """Should register a package in the registry."""
        from aragora.workflow.templates.package import (
            TemplatePackage,
            TemplateMetadata,
            register_package,
            get_package,
            _template_registry,
        )

        # Clear registry for test
        _template_registry.clear()

        metadata = TemplateMetadata(
            id="registry-test",
            name="Registry Test",
            version="1.0.0",
        )
        package = TemplatePackage(metadata=metadata, template={})

        register_package(package)

        # Should be retrievable
        retrieved = get_package("registry-test")
        assert retrieved is not None
        assert retrieved.id == "registry-test"

    def test_register_multiple_versions(self):
        """Should handle multiple versions of same template."""
        from aragora.workflow.templates.package import (
            TemplatePackage,
            TemplateMetadata,
            TemplateStatus,
            register_package,
            get_package,
            _template_registry,
        )

        _template_registry.clear()

        # Register v1
        v1_meta = TemplateMetadata(
            id="versioned",
            name="Versioned",
            version="1.0.0",
            status=TemplateStatus.DEPRECATED,
        )
        v1 = TemplatePackage(metadata=v1_meta, template={"version": 1})
        register_package(v1)

        # Register v2
        v2_meta = TemplateMetadata(
            id="versioned",
            name="Versioned",
            version="2.0.0",
            status=TemplateStatus.STABLE,
        )
        v2 = TemplatePackage(metadata=v2_meta, template={"version": 2})
        register_package(v2)

        # Should get v2 (stable) by default
        latest = get_package("versioned")
        assert latest.version == "2.0.0"

        # Should get specific version
        specific = get_package("versioned", version="1.0.0")
        assert specific.version == "1.0.0"

    def test_list_packages(self):
        """Should list packages with filters."""
        from aragora.workflow.templates.package import (
            TemplatePackage,
            TemplateMetadata,
            TemplateCategory,
            TemplateStatus,
            register_package,
            list_packages,
            _template_registry,
        )

        _template_registry.clear()

        # Register packages
        for i, cat in enumerate([TemplateCategory.LEGAL, TemplateCategory.CODE]):
            meta = TemplateMetadata(
                id=f"pkg-{i}",
                name=f"Package {i}",
                version="1.0.0",
                category=cat,
                tags=["test", f"tag{i}"],
            )
            pkg = TemplatePackage(metadata=meta, template={})
            register_package(pkg)

        # List all
        all_packages = list_packages()
        assert len(all_packages) == 2

        # Filter by category
        legal = list_packages(category="legal")
        assert len(legal) == 1
        assert legal[0].metadata.category == TemplateCategory.LEGAL

        # Filter by tag
        tagged = list_packages(tags=["tag0"])
        assert len(tagged) == 1


# =============================================================================
# package_all_templates Tests
# =============================================================================


class TestPackageAllTemplates:
    """Tests for package_all_templates function."""

    def test_package_all_templates(self):
        """Should package all registered templates."""
        from aragora.workflow.templates.package import package_all_templates

        packages = package_all_templates()

        # Should have packages
        assert len(packages) > 0

        # Each should be a proper package
        for template_id, package in packages.items():
            assert package.id is not None
            assert package.version == "1.0.0"
            assert package.template is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestTemplatePackagingIntegration:
    """Integration tests for template packaging."""

    def test_roundtrip_serialization(self):
        """Should serialize and deserialize correctly."""
        from aragora.workflow.templates.package import (
            TemplatePackage,
            TemplateMetadata,
            TemplateAuthor,
            TemplateCategory,
        )

        original = TemplatePackage(
            metadata=TemplateMetadata(
                id="roundtrip-test",
                name="Roundtrip Test",
                version="1.2.3",
                description="Test description",
                category=TemplateCategory.AI_ML,
                author=TemplateAuthor(
                    name="Test Author",
                    email="test@example.com",
                    organization="Test Org",
                ),
                tags=["test", "roundtrip"],
            ),
            template={
                "name": "Roundtrip Template",
                "steps": [
                    {"name": "step1", "type": "debate"},
                    {"name": "step2", "type": "gauntlet"},
                ],
            },
            readme="# Roundtrip Test\n\nTest documentation.",
            changelog="## 1.2.3\n- Initial release",
        )

        # Serialize
        json_str = original.to_json()

        # Deserialize
        restored = TemplatePackage.from_json(json_str)

        # Verify
        assert restored.id == original.id
        assert restored.version == original.version
        assert restored.metadata.category == original.metadata.category
        assert restored.metadata.author.name == original.metadata.author.name
        assert len(restored.template["steps"]) == len(original.template["steps"])
        assert restored.readme == original.readme

    def test_template_checksum_stability(self):
        """Checksum should be stable for same content."""
        from aragora.workflow.templates.package import (
            TemplatePackage,
            TemplateMetadata,
        )

        template = {"steps": [{"name": "step1"}, {"name": "step2"}]}

        pkg1 = TemplatePackage(
            metadata=TemplateMetadata(id="t1", name="T1", version="1.0.0"),
            template=template,
        )

        pkg2 = TemplatePackage(
            metadata=TemplateMetadata(id="t2", name="T2", version="2.0.0"),
            template=template,
        )

        # Same template content should produce same checksum
        assert pkg1.checksum == pkg2.checksum
