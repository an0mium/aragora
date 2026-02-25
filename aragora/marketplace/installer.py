"""
Marketplace Install Bridge.

Bridges marketplace catalog installation to the actual skill and workflow
template registries so that ``install_item()`` has real side-effects beyond
incrementing a download counter.

Usage:
    from aragora.marketplace.installer import MarketplaceInstaller

    installer = MarketplaceInstaller()

    # Install a marketplace item -- registers in the appropriate registry
    result = installer.install("skill-summarize")

    # Uninstall -- removes from the registry
    result = installer.uninstall("skill-summarize")
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from .catalog import InstallResult, MarketplaceCatalog, MarketplaceItem

logger = logging.getLogger(__name__)


class InstallError(Exception):
    """Raised when marketplace installation fails."""


@dataclass
class InstallBridgeResult:
    """Extended install result with registry details."""

    catalog_result: InstallResult
    registered_in: str | None = None  # "skill_registry" | "workflow_template_registry" | None
    registry_id: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True if the catalog-level install succeeded.

        Bridge errors (e.g. registry unavailable) are non-fatal and recorded
        in :attr:`errors` but do **not** prevent the install from being
        considered successful.  Pre-install validation failures set
        ``catalog_result.success`` to False.
        """
        return self.catalog_result.success

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.catalog_result.to_dict(),
            "registered_in": self.registered_in,
            "registry_id": self.registry_id,
            "bridge_errors": self.errors,
        }


class MarketplaceInstaller:
    """
    Bridges marketplace catalog install/uninstall to live registries.

    For ``skill`` items: registers a lightweight proxy Skill in the
    :class:`~aragora.skills.registry.SkillRegistry`.

    For ``template`` items: submits and auto-approves the template data in the
    :class:`~aragora.workflow.templates.registry.TemplateRegistry`.

    Other item types (``agent_pack``, ``connector``) are passed through to the
    catalog without registry bridging (logged as unsupported).
    """

    def __init__(
        self,
        catalog: MarketplaceCatalog | None = None,
        skill_registry: Any | None = None,
        template_registry: Any | None = None,
    ) -> None:
        self._catalog = catalog or MarketplaceCatalog(seed=True)
        self._skill_registry = skill_registry
        self._template_registry = template_registry
        # Track which marketplace items have been bridged to registries
        self._installed: dict[str, str] = {}  # item_id -> registry name

    def _get_skill_registry(self) -> Any:
        if self._skill_registry is not None:
            return self._skill_registry
        from aragora.skills.registry import get_skill_registry

        return get_skill_registry()

    def _get_template_registry(self) -> Any:
        if self._template_registry is not None:
            return self._template_registry
        from aragora.workflow.templates.registry import get_template_registry

        return get_template_registry()

    # ------------------------------------------------------------------
    # Install
    # ------------------------------------------------------------------

    def install(self, item_id: str) -> InstallBridgeResult:
        """Install a marketplace item and bridge to the appropriate registry.

        Args:
            item_id: The marketplace catalog item ID.

        Returns:
            InstallBridgeResult with catalog result and registry details.
        """
        # Validate item exists before touching the catalog
        item = self._catalog.get_item(item_id)
        if item is None:
            catalog_result = InstallResult(
                success=False,
                item_id=item_id,
                errors=[f"Item not found: {item_id}"],
            )
            return InstallBridgeResult(catalog_result=catalog_result)

        # Validate required fields based on type
        validation_errors = self._validate_item(item)
        if validation_errors:
            catalog_result = InstallResult(
                success=False,
                item_id=item_id,
                errors=validation_errors,
            )
            return InstallBridgeResult(catalog_result=catalog_result, errors=validation_errors)

        # Run catalog install (increments downloads, etc.)
        catalog_result = self._catalog.install_item(item_id)
        if not catalog_result.success:
            return InstallBridgeResult(catalog_result=catalog_result)

        # Bridge to the appropriate registry
        bridge_result = InstallBridgeResult(catalog_result=catalog_result)

        if item.type == "skill":
            self._bridge_skill(item, bridge_result)
        elif item.type == "template":
            self._bridge_template(item, bridge_result)
        else:
            # agent_pack, connector -- no registry bridge yet
            logger.info(
                "Item %s is type '%s'; no registry bridge available",
                item_id,
                item.type,
            )

        if bridge_result.success and bridge_result.registered_in:
            self._installed[item_id] = bridge_result.registered_in

        return bridge_result

    # ------------------------------------------------------------------
    # Uninstall
    # ------------------------------------------------------------------

    def uninstall(self, item_id: str) -> bool:
        """Remove a previously installed item from its registry.

        Args:
            item_id: The marketplace catalog item ID.

        Returns:
            True if the item was found and removed from a registry.
        """
        registry_name = self._installed.pop(item_id, None)
        if registry_name is None:
            logger.warning("Item %s not tracked as installed", item_id)
            return False

        item = self._catalog.get_item(item_id)
        if item is None:
            logger.warning("Item %s no longer in catalog", item_id)
            return False

        if registry_name == "skill_registry":
            return self._unbridge_skill(item)
        elif registry_name == "workflow_template_registry":
            return self._unbridge_template(item)

        return False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_item(self, item: MarketplaceItem) -> list[str]:
        """Validate that an item meets schema requirements for its type."""
        errors: list[str] = []

        if not item.name or not item.name.strip():
            errors.append("Item name is required")
        if not item.version or not item.version.strip():
            errors.append("Item version is required")

        if item.type == "skill":
            # Skills need an id and description at minimum
            if not item.description or not item.description.strip():
                errors.append("Skill description is required")

        elif item.type == "template":
            if not item.description or not item.description.strip():
                errors.append("Template description is required")

        return errors

    # ------------------------------------------------------------------
    # Skill bridging
    # ------------------------------------------------------------------

    def _bridge_skill(self, item: MarketplaceItem, result: InstallBridgeResult) -> None:
        """Register a marketplace skill item in the SkillRegistry."""
        try:
            registry = self._get_skill_registry()

            # Check for duplicate
            if registry.has_skill(item.id):
                logger.info("Skill %s already registered, replacing", item.id)

            from aragora.skills.base import (
                Skill,
                SkillCapability,
                SkillContext,
                SkillManifest,
                SkillResult,
            )

            # Build a lightweight proxy skill from marketplace metadata
            manifest = SkillManifest(
                name=item.id,
                version=item.version,
                capabilities=[SkillCapability.LLM_INFERENCE],
                input_schema={"text": {"type": "string"}},
                description=item.description,
                author=item.author,
                tags=list(item.tags),
            )

            class _MarketplaceSkillProxy(Skill):
                """Auto-generated proxy for a marketplace skill item."""

                def __init__(self, m: SkillManifest) -> None:
                    self._manifest = m

                @property
                def manifest(self) -> SkillManifest:
                    return self._manifest

                async def execute(
                    self, input_data: dict[str, Any], context: SkillContext
                ) -> SkillResult:
                    return SkillResult.create_failure(
                        "Marketplace proxy skill -- not yet implemented"
                    )

            proxy = _MarketplaceSkillProxy(manifest)
            registry.register(proxy, replace=True)

            result.registered_in = "skill_registry"
            result.registry_id = item.id
            logger.info("Bridged marketplace skill %s to SkillRegistry", item.id)

        except (ImportError, ValueError, TypeError, RuntimeError) as exc:
            msg = f"Failed to bridge skill {item.id}: {exc}"
            logger.warning(msg)
            result.errors.append(msg)

    def _unbridge_skill(self, item: MarketplaceItem) -> bool:
        """Remove a skill from the SkillRegistry."""
        try:
            registry = self._get_skill_registry()
            return registry.unregister(item.id)
        except (ImportError, ValueError, RuntimeError) as exc:
            logger.warning("Failed to unbridge skill %s: %s", item.id, exc)
            return False

    # ------------------------------------------------------------------
    # Template bridging
    # ------------------------------------------------------------------

    def _bridge_template(self, item: MarketplaceItem, result: InstallBridgeResult) -> None:
        """Register a marketplace template in the workflow TemplateRegistry."""
        try:
            registry = self._get_template_registry()

            # Check for duplicate by looking up existing listing
            existing = registry.get(item.id)
            if existing is not None:
                logger.info("Template %s already registered, skipping duplicate", item.id)
                result.registered_in = "workflow_template_registry"
                result.registry_id = item.id
                return

            template_data = {
                "name": item.name,
                "description": item.description,
                "version": item.version,
                "author": item.author,
                "tags": item.tags,
                "source": "marketplace",
                "marketplace_id": item.id,
            }

            # Determine category from tags
            category = "general"
            tag_to_category = {
                "code": "code",
                "legal": "legal",
                "healthcare": "healthcare",
                "compliance": "general",
                "risk": "general",
            }
            for tag in item.tags:
                if tag.lower() in tag_to_category:
                    category = tag_to_category[tag.lower()]
                    break

            listing_id = registry.submit(
                template_data=template_data,
                name=item.name,
                description=item.description,
                category=category,
                author_id=item.author,
                tags=item.tags,
                version=item.version,
            )

            # Auto-approve marketplace items
            registry.approve(listing_id, approved_by="marketplace-installer")

            result.registered_in = "workflow_template_registry"
            result.registry_id = listing_id
            logger.info(
                "Bridged marketplace template %s to TemplateRegistry as %s",
                item.id,
                listing_id,
            )

        except (ImportError, ValueError, TypeError, RuntimeError, sqlite3.Error) as exc:
            msg = f"Failed to bridge template {item.id}: {exc}"
            logger.warning(msg)
            result.errors.append(msg)

    def _unbridge_template(self, item: MarketplaceItem) -> bool:
        """Remove a template from the workflow TemplateRegistry.

        Note: The workflow TemplateRegistry does not expose a delete API
        for individual listings, so we archive it by rejecting.
        """
        try:
            registry = self._get_template_registry()

            # Search for listings that came from this marketplace item
            listings = registry.search(
                query=item.name,
                status="approved",
            )
            removed = False
            for listing in listings:
                if listing.template_data.get("marketplace_id") == item.id:
                    registry.reject(listing.id, reason="Uninstalled from marketplace")
                    removed = True
                    logger.info(
                        "Rejected template listing %s (marketplace item %s)",
                        listing.id,
                        item.id,
                    )

            return removed
        except (ImportError, ValueError, RuntimeError, sqlite3.Error) as exc:
            logger.warning("Failed to unbridge template %s: %s", item.id, exc)
            return False
