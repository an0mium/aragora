"""End-to-end marketplace lifecycle tests.

Validates the full browse -> install -> use -> uninstall flow, ensuring
installed items are visible in the appropriate registries and can be
invoked or queried after installation.
"""

from __future__ import annotations

import asyncio

import pytest

from aragora.marketplace.catalog import MarketplaceCatalog
from aragora.marketplace.installer import MarketplaceInstaller
from aragora.marketplace.service import MarketplaceService
from aragora.skills.registry import SkillRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def catalog() -> MarketplaceCatalog:
    return MarketplaceCatalog(seed=True)


@pytest.fixture()
def skill_registry() -> SkillRegistry:
    return SkillRegistry(enable_metrics=False, enable_rate_limiting=False)


@pytest.fixture()
def template_registry(tmp_path):
    from aragora.workflow.templates.registry import TemplateRegistry

    return TemplateRegistry(db_path=str(tmp_path / "tpl_e2e.db"))


@pytest.fixture()
def installer(catalog, skill_registry, template_registry) -> MarketplaceInstaller:
    return MarketplaceInstaller(
        catalog=catalog,
        skill_registry=skill_registry,
        template_registry=template_registry,
    )


@pytest.fixture()
def service(catalog) -> MarketplaceService:
    return MarketplaceService(catalog=catalog)


# ---------------------------------------------------------------------------
# Full lifecycle: skill
# ---------------------------------------------------------------------------


class TestSkillLifecycle:
    """Browse -> install -> verify in registry -> invoke -> uninstall."""

    def test_browse_find_install_invoke_uninstall(
        self, service, installer, skill_registry
    ) -> None:
        """Complete skill lifecycle from browsing to uninstallation."""
        # 1. Browse -- find skills in the catalog
        listings = service.list_listings(item_type="skill")
        skill_ids = [item["id"] for item in listings["items"]]
        assert "skill-summarize" in skill_ids
        assert listings["total"] >= 5  # seed catalog has 5 skills

        # 2. Get detail on a specific skill
        detail = service.get_listing("skill-summarize")
        assert detail is not None
        assert detail["type"] == "skill"
        assert "summarize" in detail["name"].lower()

        # 3. Install the skill
        result = installer.install("skill-summarize")
        assert result.success
        assert result.registered_in == "skill_registry"

        # 4. Verify the skill appears in SkillRegistry.list_skills()
        manifests = skill_registry.list_skills()
        manifest_names = [m.name for m in manifests]
        assert "skill-summarize" in manifest_names

        # 5. Verify the skill is retrievable by name
        skill = skill_registry.get("skill-summarize")
        assert skill is not None
        assert skill.manifest.description == detail["description"]

        # 6. Invoke the skill (proxy returns failure, but proves invocation works)
        from aragora.skills.base import SkillContext

        ctx = SkillContext(user_id="e2e-test", permissions=["*"])
        invoke_result = asyncio.get_event_loop().run_until_complete(
            skill_registry.invoke("skill-summarize", {"text": "hello"}, ctx)
        )
        # Proxy skill returns failure with known message
        assert invoke_result.error_message is not None
        assert "proxy" in invoke_result.error_message.lower()

        # 7. Uninstall
        removed = installer.uninstall("skill-summarize")
        assert removed is True

        # 8. Verify it's gone from the registry
        assert not skill_registry.has_skill("skill-summarize")

    def test_install_multiple_skills(self, installer, skill_registry) -> None:
        """Installing multiple skills registers all of them."""
        items = ["skill-summarize", "skill-translate", "skill-extract"]
        for item_id in items:
            result = installer.install(item_id)
            assert result.success, f"Failed to install {item_id}"

        manifests = skill_registry.list_skills()
        names = {m.name for m in manifests}
        for item_id in items:
            assert item_id in names

        # Uninstall one and verify the others remain
        installer.uninstall("skill-translate")
        remaining = {m.name for m in skill_registry.list_skills()}
        assert "skill-translate" not in remaining
        assert "skill-summarize" in remaining
        assert "skill-extract" in remaining


# ---------------------------------------------------------------------------
# Full lifecycle: template
# ---------------------------------------------------------------------------


class TestTemplateLifecycle:
    """Browse -> install -> verify in registry -> uninstall."""

    def test_browse_find_install_query_uninstall(
        self, service, installer, template_registry
    ) -> None:
        """Complete template lifecycle from browsing to uninstallation."""
        # 1. Browse templates
        listings = service.list_listings(item_type="template")
        tpl_ids = [item["id"] for item in listings["items"]]
        assert "tpl-code-review" in tpl_ids

        # 2. Install
        result = installer.install("tpl-code-review")
        assert result.success
        assert result.registered_in == "workflow_template_registry"

        # 3. Verify the template is searchable in the workflow registry
        found = template_registry.search(query="Code Review Pipeline", status="approved")
        marketplace_entries = [
            l for l in found if l.template_data.get("marketplace_id") == "tpl-code-review"
        ]
        assert len(marketplace_entries) == 1
        entry = marketplace_entries[0]
        assert entry.template_data["source"] == "marketplace"
        assert entry.approved_by == "marketplace-installer"

        # 4. Uninstall
        removed = installer.uninstall("tpl-code-review")
        assert removed is True

        # 5. Verify it's no longer approved
        after = template_registry.search(query="Code Review Pipeline", status="approved")
        marketplace_after = [
            l for l in after if l.template_data.get("marketplace_id") == "tpl-code-review"
        ]
        assert len(marketplace_after) == 0

    def test_install_multiple_templates(self, installer, template_registry) -> None:
        """Installing multiple templates registers all of them."""
        items = ["tpl-code-review", "tpl-doc-analysis", "tpl-risk-assessment"]
        registry_ids = []
        for item_id in items:
            result = installer.install(item_id)
            assert result.success, f"Failed to install {item_id}"
            registry_ids.append(result.registry_id)

        # All should be findable as approved
        all_approved = template_registry.search(status="approved")
        mp_ids = {
            l.template_data.get("marketplace_id")
            for l in all_approved
            if l.template_data.get("source") == "marketplace"
        }
        for item_id in items:
            assert item_id in mp_ids


# ---------------------------------------------------------------------------
# Mixed lifecycle
# ---------------------------------------------------------------------------


class TestMixedLifecycle:
    """Install both skills and templates in the same session."""

    def test_mixed_install_and_uninstall(
        self, installer, skill_registry, template_registry
    ) -> None:
        """Skills and templates coexist correctly."""
        # Install one skill and one template
        skill_result = installer.install("skill-classify")
        tpl_result = installer.install("tpl-brainstorm")

        assert skill_result.success
        assert tpl_result.success

        # Verify both registries
        assert skill_registry.has_skill("skill-classify")
        tpl_found = template_registry.search(query="Brainstorm Session", status="approved")
        assert any(
            l.template_data.get("marketplace_id") == "tpl-brainstorm" for l in tpl_found
        )

        # Uninstall skill -- template should remain
        installer.uninstall("skill-classify")
        assert not skill_registry.has_skill("skill-classify")

        tpl_still = template_registry.search(query="Brainstorm Session", status="approved")
        assert any(
            l.template_data.get("marketplace_id") == "tpl-brainstorm" for l in tpl_still
        )


# ---------------------------------------------------------------------------
# Service-level install tracking
# ---------------------------------------------------------------------------


class TestServiceInstallTracking:
    """MarketplaceService tracks user installs alongside registry bridging."""

    def test_service_install_and_bridge(
        self, catalog, skill_registry, template_registry
    ) -> None:
        """Service install_listing tracks user + bridges via installer."""
        service = MarketplaceService(catalog=catalog)
        installer = MarketplaceInstaller(
            catalog=catalog,
            skill_registry=skill_registry,
            template_registry=template_registry,
        )

        # Service install (tracks user)
        svc_result = service.install_listing("skill-summarize", user_id="user-42")
        assert svc_result.success
        assert "skill-summarize" in service.get_user_installs("user-42")

        # Bridge install (registers in registry)
        bridge_result = installer.install("skill-summarize")
        assert bridge_result.success
        assert skill_registry.has_skill("skill-summarize")


# ---------------------------------------------------------------------------
# Rating integration
# ---------------------------------------------------------------------------


class TestRatingIntegration:
    """Ratings work alongside installation."""

    def test_rate_after_install(self, service, installer) -> None:
        """Users can rate items after installing them."""
        installer.install("skill-compare")

        rating = service.rate_listing(
            "skill-compare", user_id="user-1", score=5, review="Great skill!"
        )
        assert rating["success"]
        assert rating["average_rating"] == 5.0

        # Stats reflect the rating
        stats = service.get_stats()
        assert stats["total_ratings"] >= 1


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    """Repeated operations are safe."""

    def test_double_install_skill(self, installer, skill_registry) -> None:
        """Installing the same skill twice is idempotent."""
        r1 = installer.install("skill-extract")
        r2 = installer.install("skill-extract")
        assert r1.success and r2.success
        assert skill_registry.skill_count >= 1

    def test_double_uninstall_skill(self, installer) -> None:
        """Uninstalling an already-uninstalled skill returns False the second time."""
        installer.install("skill-extract")
        assert installer.uninstall("skill-extract") is True
        assert installer.uninstall("skill-extract") is False
