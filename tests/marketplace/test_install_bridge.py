"""Tests for the marketplace install bridge.

Verifies that ``MarketplaceInstaller`` correctly bridges catalog install
operations to the SkillRegistry and workflow TemplateRegistry.
"""

from __future__ import annotations

import pytest

from aragora.marketplace.catalog import MarketplaceCatalog, MarketplaceItem
from aragora.marketplace.installer import MarketplaceInstaller


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_catalog(extra_items: list[MarketplaceItem] | None = None) -> MarketplaceCatalog:
    """Build a seeded catalog, optionally with extra items."""
    catalog = MarketplaceCatalog(seed=True)
    for item in extra_items or []:
        catalog._items[item.id] = item
    return catalog


# ---------------------------------------------------------------------------
# Skill installation
# ---------------------------------------------------------------------------


class TestSkillInstallBridge:
    """Skill-type items should be registered in the SkillRegistry."""

    def test_skill_install_registers_in_skill_registry(self) -> None:
        """Installing a skill item bridges it into the SkillRegistry."""
        from aragora.skills.registry import SkillRegistry

        catalog = _make_catalog()
        skill_reg = SkillRegistry(enable_metrics=False, enable_rate_limiting=False)

        installer = MarketplaceInstaller(
            catalog=catalog,
            skill_registry=skill_reg,
        )

        result = installer.install("skill-summarize")

        assert result.success
        assert result.registered_in == "skill_registry"
        assert result.registry_id == "skill-summarize"
        assert skill_reg.has_skill("skill-summarize")

    def test_skill_install_replaces_on_duplicate(self) -> None:
        """Re-installing a skill replaces the existing registration."""
        from aragora.skills.registry import SkillRegistry

        catalog = _make_catalog()
        skill_reg = SkillRegistry(enable_metrics=False, enable_rate_limiting=False)

        installer = MarketplaceInstaller(
            catalog=catalog,
            skill_registry=skill_reg,
        )

        result1 = installer.install("skill-summarize")
        assert result1.success

        result2 = installer.install("skill-summarize")
        assert result2.success
        assert skill_reg.has_skill("skill-summarize")

    def test_skill_uninstall_removes_from_registry(self) -> None:
        """Uninstalling a skill removes it from the SkillRegistry."""
        from aragora.skills.registry import SkillRegistry

        catalog = _make_catalog()
        skill_reg = SkillRegistry(enable_metrics=False, enable_rate_limiting=False)

        installer = MarketplaceInstaller(
            catalog=catalog,
            skill_registry=skill_reg,
        )

        installer.install("skill-summarize")
        assert skill_reg.has_skill("skill-summarize")

        removed = installer.uninstall("skill-summarize")
        assert removed is True
        assert not skill_reg.has_skill("skill-summarize")

    def test_skill_manifest_inherits_metadata(self) -> None:
        """The proxy skill manifest should carry marketplace metadata."""
        from aragora.skills.registry import SkillRegistry

        catalog = _make_catalog()
        skill_reg = SkillRegistry(enable_metrics=False, enable_rate_limiting=False)

        installer = MarketplaceInstaller(
            catalog=catalog,
            skill_registry=skill_reg,
        )

        installer.install("skill-summarize")

        skill = skill_reg.get("skill-summarize")
        assert skill is not None
        manifest = skill.manifest
        assert manifest.name == "skill-summarize"
        assert manifest.version == "1.0.0"
        assert manifest.author == "aragora"
        assert "summarize" in manifest.tags


# ---------------------------------------------------------------------------
# Template installation
# ---------------------------------------------------------------------------


class TestTemplateInstallBridge:
    """Template-type items should be registered in the workflow TemplateRegistry."""

    def test_template_install_registers_in_template_registry(self, tmp_path) -> None:
        """Installing a template item bridges it into the TemplateRegistry."""
        from aragora.workflow.templates.registry import TemplateRegistry

        catalog = _make_catalog()
        tpl_reg = TemplateRegistry(db_path=str(tmp_path / "tpl.db"))

        installer = MarketplaceInstaller(
            catalog=catalog,
            template_registry=tpl_reg,
        )

        result = installer.install("tpl-code-review")

        assert result.success
        assert result.registered_in == "workflow_template_registry"
        assert result.registry_id is not None

        # Verify the listing is searchable and approved
        listings = tpl_reg.search(query="Code Review Pipeline", status="approved")
        assert len(listings) >= 1
        found = listings[0]
        assert found.template_data.get("marketplace_id") == "tpl-code-review"

    def test_template_duplicate_install_is_idempotent(self, tmp_path) -> None:
        """Re-installing a template does not create a duplicate listing."""
        from aragora.workflow.templates.registry import TemplateRegistry

        catalog = _make_catalog()
        tpl_reg = TemplateRegistry(db_path=str(tmp_path / "tpl.db"))

        installer = MarketplaceInstaller(
            catalog=catalog,
            template_registry=tpl_reg,
        )

        result1 = installer.install("tpl-code-review")
        assert result1.success

        # The second install should detect the existing entry
        # (first call writes an entry; second finds it via search)
        result2 = installer.install("tpl-code-review")
        assert result2.success

    def test_template_uninstall_rejects_listing(self, tmp_path) -> None:
        """Uninstalling a template rejects its listing in the registry."""
        from aragora.workflow.templates.registry import TemplateRegistry

        catalog = _make_catalog()
        tpl_reg = TemplateRegistry(db_path=str(tmp_path / "tpl.db"))

        installer = MarketplaceInstaller(
            catalog=catalog,
            template_registry=tpl_reg,
        )

        installer.install("tpl-code-review")
        removed = installer.uninstall("tpl-code-review")
        assert removed is True

        # The listing should no longer show as approved
        approved = tpl_reg.search(query="Code Review Pipeline", status="approved")
        marketplace_entries = [
            l for l in approved if l.template_data.get("marketplace_id") == "tpl-code-review"
        ]
        assert len(marketplace_entries) == 0

    def test_template_auto_approved(self, tmp_path) -> None:
        """Marketplace templates are auto-approved on install."""
        from aragora.workflow.templates.registry import TemplateRegistry

        catalog = _make_catalog()
        tpl_reg = TemplateRegistry(db_path=str(tmp_path / "tpl.db"))

        installer = MarketplaceInstaller(
            catalog=catalog,
            template_registry=tpl_reg,
        )

        result = installer.install("tpl-doc-analysis")
        assert result.success

        listing_id = result.registry_id
        assert listing_id is not None

        listing = tpl_reg.get(listing_id)
        assert listing is not None
        assert listing.approved_by == "marketplace-installer"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestInstallValidation:
    """Items must meet schema requirements before registration."""

    def test_nonexistent_item_fails(self) -> None:
        """Attempting to install a non-existent item returns failure."""
        catalog = _make_catalog()
        installer = MarketplaceInstaller(catalog=catalog)

        result = installer.install("nonexistent-item")
        assert not result.success
        assert "not found" in result.catalog_result.errors[0].lower()

    def test_item_with_empty_name_fails(self) -> None:
        """Items with an empty name fail validation."""
        bad_item = MarketplaceItem(
            id="bad-item",
            name="",
            type="skill",
            description="A bad item",
            author="test",
            version="1.0.0",
        )
        catalog = _make_catalog(extra_items=[bad_item])
        installer = MarketplaceInstaller(catalog=catalog)

        result = installer.install("bad-item")
        assert not result.success
        assert any("name" in e.lower() for e in result.errors)

    def test_skill_without_description_fails(self) -> None:
        """A skill with empty description fails validation."""
        bad_skill = MarketplaceItem(
            id="bad-skill",
            name="Bad Skill",
            type="skill",
            description="",
            author="test",
            version="1.0.0",
        )
        catalog = _make_catalog(extra_items=[bad_skill])
        installer = MarketplaceInstaller(catalog=catalog)

        result = installer.install("bad-skill")
        assert not result.success
        assert any("description" in e.lower() for e in result.errors)

    def test_template_without_description_fails(self) -> None:
        """A template with empty description fails validation."""
        bad_tpl = MarketplaceItem(
            id="bad-tpl",
            name="Bad Template",
            type="template",
            description="",
            author="test",
            version="1.0.0",
        )
        catalog = _make_catalog(extra_items=[bad_tpl])
        installer = MarketplaceInstaller(catalog=catalog)

        result = installer.install("bad-tpl")
        assert not result.success
        assert any("description" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# Unsupported types
# ---------------------------------------------------------------------------


class TestUnsupportedTypes:
    """Non-bridged item types should succeed at catalog level but skip registry."""

    def test_agent_pack_install_succeeds_without_registry(self) -> None:
        """Agent pack items install in catalog but have no registry bridge."""
        from aragora.skills.registry import SkillRegistry

        catalog = _make_catalog()
        skill_reg = SkillRegistry(enable_metrics=False, enable_rate_limiting=False)

        installer = MarketplaceInstaller(
            catalog=catalog,
            skill_registry=skill_reg,
        )

        result = installer.install("pack-speed")
        assert result.catalog_result.success
        assert result.registered_in is None

    def test_uninstall_untracked_item_returns_false(self) -> None:
        """Uninstalling an item that was never installed returns False."""
        catalog = _make_catalog()
        installer = MarketplaceInstaller(catalog=catalog)

        assert installer.uninstall("skill-summarize") is False


# ---------------------------------------------------------------------------
# to_dict serialization
# ---------------------------------------------------------------------------


class TestInstallBridgeResultSerialization:
    """InstallBridgeResult.to_dict round-trips correctly."""

    def test_to_dict_includes_bridge_fields(self) -> None:
        from aragora.skills.registry import SkillRegistry

        catalog = _make_catalog()
        skill_reg = SkillRegistry(enable_metrics=False, enable_rate_limiting=False)

        installer = MarketplaceInstaller(
            catalog=catalog,
            skill_registry=skill_reg,
        )

        result = installer.install("skill-summarize")
        d = result.to_dict()

        assert d["success"] is True
        assert d["registered_in"] == "skill_registry"
        assert d["registry_id"] == "skill-summarize"
        assert d["bridge_errors"] == []
