"""Tests for the marketplace install bridge (Sprint 1 / T3)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.marketplace.catalog import InstallResult, MarketplaceItem
from aragora.marketplace.installer import (
    InstallBridgeResult,
    MarketplaceInstaller,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def skill_item() -> MarketplaceItem:
    return MarketplaceItem(
        id="skill-summarize",
        name="Summarizer",
        type="skill",
        description="Summarizes debate transcripts into key points",
        author="aragora-team",
        version="1.0.0",
        tags=["nlp", "summarization"],
    )


@pytest.fixture
def template_item() -> MarketplaceItem:
    return MarketplaceItem(
        id="tpl-code-review",
        name="Code Review Template",
        type="template",
        description="Multi-agent code review debate template",
        author="aragora-team",
        version="1.0.0",
        tags=["code", "review"],
    )


@pytest.fixture
def connector_item() -> MarketplaceItem:
    return MarketplaceItem(
        id="conn-jira",
        name="Jira Connector",
        type="connector",
        description="Connect Aragora to Jira for issue tracking",
        author="community",
        version="0.5.0",
    )


@pytest.fixture
def mock_catalog(skill_item, template_item, connector_item):
    """Mock catalog with get_item and install_item."""
    catalog = MagicMock()
    items = {
        skill_item.id: skill_item,
        template_item.id: template_item,
        connector_item.id: connector_item,
    }
    catalog.get_item.side_effect = lambda item_id: items.get(item_id)
    catalog.install_item.return_value = InstallResult(
        success=True, item_id="", installed_path="/installed"
    )
    return catalog


@pytest.fixture
def mock_skill_registry():
    registry = MagicMock()
    registry.has_skill.return_value = False
    return registry


@pytest.fixture
def mock_template_registry():
    registry = MagicMock()
    registry.get.return_value = None
    registry.submit.return_value = "listing-123"
    return registry


@pytest.fixture
def installer(mock_catalog, mock_skill_registry, mock_template_registry):
    return MarketplaceInstaller(
        catalog=mock_catalog,
        skill_registry=mock_skill_registry,
        template_registry=mock_template_registry,
    )


# ---------------------------------------------------------------------------
# InstallBridgeResult
# ---------------------------------------------------------------------------


class TestInstallBridgeResult:
    def test_success_when_catalog_succeeds(self):
        result = InstallBridgeResult(
            catalog_result=InstallResult(success=True, item_id="test"),
        )
        assert result.success is True

    def test_failure_when_catalog_fails(self):
        result = InstallBridgeResult(
            catalog_result=InstallResult(success=False, item_id="test", errors=["not found"]),
        )
        assert result.success is False

    def test_success_even_with_bridge_errors(self):
        """Bridge errors are non-fatal — catalog success is what matters."""
        result = InstallBridgeResult(
            catalog_result=InstallResult(success=True, item_id="test"),
            errors=["registry unavailable"],
        )
        assert result.success is True

    def test_to_dict(self):
        result = InstallBridgeResult(
            catalog_result=InstallResult(success=True, item_id="test"),
            registered_in="skill_registry",
            registry_id="skill-test",
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["registered_in"] == "skill_registry"
        assert d["registry_id"] == "skill-test"


# ---------------------------------------------------------------------------
# Install: item not found
# ---------------------------------------------------------------------------


class TestInstallNotFound:
    def test_item_not_found(self, installer, mock_catalog):
        mock_catalog.get_item.return_value = None
        result = installer.install("nonexistent")
        assert not result.success
        assert "not found" in result.catalog_result.errors[0].lower()

    def test_catalog_install_failure(self, installer, mock_catalog):
        mock_catalog.install_item.return_value = InstallResult(
            success=False, item_id="skill-summarize", errors=["disk full"]
        )
        result = installer.install("skill-summarize")
        assert not result.success


# ---------------------------------------------------------------------------
# Install: skill items
# ---------------------------------------------------------------------------


class TestInstallSkill:
    def test_installs_skill_to_registry(self, installer, mock_skill_registry):
        result = installer.install("skill-summarize")
        assert result.success
        assert result.registered_in == "skill_registry"
        assert result.registry_id == "skill-summarize"
        mock_skill_registry.register.assert_called_once()

    def test_replaces_existing_skill(self, installer, mock_skill_registry):
        mock_skill_registry.has_skill.return_value = True
        result = installer.install("skill-summarize")
        assert result.success
        mock_skill_registry.register.assert_called_once()

    def test_tracks_installed_skill(self, installer):
        installer.install("skill-summarize")
        assert "skill-summarize" in installer._installed
        assert installer._installed["skill-summarize"] == "skill_registry"


# ---------------------------------------------------------------------------
# Install: template items
# ---------------------------------------------------------------------------


class TestInstallTemplate:
    def test_installs_template_to_registry(self, installer, mock_template_registry):
        result = installer.install("tpl-code-review")
        assert result.success
        assert result.registered_in == "workflow_template_registry"
        assert result.registry_id == "listing-123"
        mock_template_registry.submit.assert_called_once()
        mock_template_registry.approve.assert_called_once_with(
            "listing-123", approved_by="marketplace-installer"
        )

    def test_skips_duplicate_template(self, installer, mock_template_registry):
        mock_template_registry.get.return_value = MagicMock()  # Already exists
        result = installer.install("tpl-code-review")
        assert result.success
        assert result.registered_in == "workflow_template_registry"
        mock_template_registry.submit.assert_not_called()

    def test_template_category_from_tags(self, installer, mock_template_registry):
        installer.install("tpl-code-review")
        call_kwargs = mock_template_registry.submit.call_args
        assert call_kwargs.kwargs.get("category") == "code"

    def test_tracks_installed_template(self, installer):
        installer.install("tpl-code-review")
        assert "tpl-code-review" in installer._installed
        assert installer._installed["tpl-code-review"] == "workflow_template_registry"


# ---------------------------------------------------------------------------
# Install: unsupported types
# ---------------------------------------------------------------------------


class TestInstallUnsupported:
    def test_connector_passes_through(self, installer):
        result = installer.install("conn-jira")
        assert result.success
        assert result.registered_in is None  # No registry bridge


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_skill_without_description(
        self, mock_catalog, mock_skill_registry, mock_template_registry
    ):
        bad_item = MarketplaceItem(
            id="skill-bad",
            name="Bad Skill",
            type="skill",
            description="",  # Empty
            author="test",
            version="1.0.0",
        )
        mock_catalog.get_item.side_effect = None
        mock_catalog.get_item.return_value = bad_item
        inst = MarketplaceInstaller(
            catalog=mock_catalog,
            skill_registry=mock_skill_registry,
            template_registry=mock_template_registry,
        )
        result = inst.install("skill-bad")
        assert not result.success
        assert any("description" in e.lower() for e in result.errors)

    def test_template_without_description(
        self, mock_catalog, mock_skill_registry, mock_template_registry
    ):
        bad_item = MarketplaceItem(
            id="tpl-bad",
            name="Bad Template",
            type="template",
            description="  ",  # Whitespace only
            author="test",
            version="1.0.0",
        )
        mock_catalog.get_item.side_effect = None
        mock_catalog.get_item.return_value = bad_item
        inst = MarketplaceInstaller(
            catalog=mock_catalog,
            skill_registry=mock_skill_registry,
            template_registry=mock_template_registry,
        )
        result = inst.install("tpl-bad")
        assert not result.success

    def test_item_without_name(self, mock_catalog, mock_skill_registry, mock_template_registry):
        bad_item = MarketplaceItem(
            id="skill-noname",
            name="",
            type="skill",
            description="Has description",
            author="test",
            version="1.0.0",
        )
        mock_catalog.get_item.side_effect = None
        mock_catalog.get_item.return_value = bad_item
        inst = MarketplaceInstaller(
            catalog=mock_catalog,
            skill_registry=mock_skill_registry,
            template_registry=mock_template_registry,
        )
        result = inst.install("skill-noname")
        assert not result.success
        assert any("name" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------


class TestUninstall:
    def test_uninstall_skill(self, installer, mock_skill_registry):
        installer.install("skill-summarize")
        mock_skill_registry.unregister.return_value = True
        assert installer.uninstall("skill-summarize") is True
        mock_skill_registry.unregister.assert_called_once_with("skill-summarize")

    def test_uninstall_template(self, installer, mock_template_registry):
        installer.install("tpl-code-review")

        mock_listing = MagicMock()
        mock_listing.id = "listing-123"
        mock_listing.template_data = {"marketplace_id": "tpl-code-review"}
        mock_template_registry.search.return_value = [mock_listing]

        assert installer.uninstall("tpl-code-review") is True
        mock_template_registry.reject.assert_called_once()

    def test_uninstall_not_tracked(self, installer):
        assert installer.uninstall("never-installed") is False

    def test_uninstall_item_removed_from_catalog(self, installer, mock_catalog):
        installer.install("skill-summarize")
        mock_catalog.get_item.side_effect = None
        mock_catalog.get_item.return_value = None  # Removed from catalog
        assert installer.uninstall("skill-summarize") is False


# ---------------------------------------------------------------------------
# Registry error handling
# ---------------------------------------------------------------------------


class TestRegistryErrors:
    def test_skill_registry_import_error(self, mock_catalog, skill_item):
        """Skill bridging fails gracefully on ImportError."""
        mock_catalog.get_item.return_value = skill_item
        mock_catalog.install_item.return_value = InstallResult(success=True, item_id=skill_item.id)

        broken_registry = MagicMock()
        broken_registry.has_skill.side_effect = RuntimeError("registry down")

        installer = MarketplaceInstaller(
            catalog=mock_catalog,
            skill_registry=broken_registry,
        )
        result = installer.install(skill_item.id)
        # Catalog install succeeded; bridge error is non-fatal
        assert result.success
        assert len(result.errors) > 0
        assert result.registered_in is None

    def test_template_registry_submit_error(self, mock_catalog, template_item):
        """Template bridging fails gracefully on submit error."""
        mock_catalog.get_item.return_value = template_item
        mock_catalog.install_item.return_value = InstallResult(
            success=True, item_id=template_item.id
        )

        broken_registry = MagicMock()
        broken_registry.get.return_value = None
        broken_registry.submit.side_effect = ValueError("invalid schema")

        installer = MarketplaceInstaller(
            catalog=mock_catalog,
            template_registry=broken_registry,
        )
        result = installer.install(template_item.id)
        assert result.success  # Catalog succeeded
        assert len(result.errors) > 0


# ---------------------------------------------------------------------------
# E2E lifecycle
# ---------------------------------------------------------------------------


class TestE2ELifecycle:
    def test_full_skill_lifecycle(self, installer, mock_skill_registry):
        """Install → verify tracked → uninstall → verify removed."""
        # Install
        result = installer.install("skill-summarize")
        assert result.success
        assert "skill-summarize" in installer._installed

        # Verify registered
        mock_skill_registry.register.assert_called_once()

        # Uninstall
        mock_skill_registry.unregister.return_value = True
        assert installer.uninstall("skill-summarize") is True
        assert "skill-summarize" not in installer._installed

    def test_full_template_lifecycle(self, installer, mock_template_registry):
        """Install → verify tracked → uninstall → verify removed."""
        # Install
        result = installer.install("tpl-code-review")
        assert result.success
        assert result.registered_in == "workflow_template_registry"

        # Verify submitted and approved
        mock_template_registry.submit.assert_called_once()
        mock_template_registry.approve.assert_called_once()

        # Uninstall
        mock_listing = MagicMock()
        mock_listing.id = "listing-123"
        mock_listing.template_data = {"marketplace_id": "tpl-code-review"}
        mock_template_registry.search.return_value = [mock_listing]

        assert installer.uninstall("tpl-code-review") is True
        assert "tpl-code-review" not in installer._installed

    def test_install_multiple_types(self, installer):
        """Install skill, template, and connector — each bridges correctly."""
        r1 = installer.install("skill-summarize")
        r2 = installer.install("tpl-code-review")
        r3 = installer.install("conn-jira")

        assert r1.registered_in == "skill_registry"
        assert r2.registered_in == "workflow_template_registry"
        assert r3.registered_in is None  # No bridge for connectors

        assert len(installer._installed) == 2  # Only bridged items tracked
