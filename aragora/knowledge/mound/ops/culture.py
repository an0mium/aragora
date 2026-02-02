"""
Culture Operations Mixin for Knowledge Mound.

Provides culture management operations:
- get_culture_profile: Get workspace culture patterns
- observe_debate: Extract patterns from debates
- recommend_agents: Agent recommendations based on culture
- Organization culture management

NOTE: This is a mixin class designed to be composed with KnowledgeMound.
Attribute accesses like self._ensure_initialized, self._culture_accumulator, self._cache, etc.
are provided by the composed class. Type safety is achieved by using CultureProtocol
as the base class during type checking (via TYPE_CHECKING conditional).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, Protocol

if TYPE_CHECKING:
    from aragora.knowledge.mound.types import CulturePattern, CultureProfile, MoundConfig
    from aragora.knowledge.mound.culture import (
        CultureDocument,
        OrganizationCulture,
        OrganizationCultureManager,
    )

logger = logging.getLogger(__name__)


class CultureProtocol(Protocol):
    """Protocol defining expected interface for Culture mixin."""

    config: "MoundConfig"
    workspace_id: str
    _culture_accumulator: Any | None
    _cache: Any | None
    _initialized: bool
    _org_culture_manager: Optional["OrganizationCultureManager"]

    def _ensure_initialized(self) -> None: ...

    def get_org_culture_manager(self) -> "OrganizationCultureManager": ...


# Use Protocol as base class only for type checking
if TYPE_CHECKING:
    _CultureMixinBase = CultureProtocol
else:
    _CultureMixinBase = object


class CultureOperationsMixin(_CultureMixinBase):
    """Mixin providing culture management for KnowledgeMound."""

    async def get_culture_profile(
        self,
        workspace_id: str | None = None,
    ) -> "CultureProfile":
        """Get aggregated culture profile for a workspace."""
        from aragora.knowledge.mound.types import CultureProfile

        self._ensure_initialized()

        ws_id = workspace_id or self.workspace_id

        # Check cache
        if self._cache:
            cached = await self._cache.get_culture(ws_id)
            if cached:
                return cached

        if not self._culture_accumulator:
            return CultureProfile(
                workspace_id=ws_id,
                patterns={},
                generated_at=datetime.now(),
                total_observations=0,
            )

        profile = await self._culture_accumulator.get_profile(ws_id)

        # Cache result
        if self._cache:
            await self._cache.set_culture(ws_id, profile)

        return profile

    async def observe_debate(
        self,
        debate_result: Any,
    ) -> list["CulturePattern"]:
        """Extract and store cultural patterns from a completed debate."""
        self._ensure_initialized()

        if not self._culture_accumulator:
            return []

        return await self._culture_accumulator.observe_debate(debate_result, self.workspace_id)

    async def recommend_agents(
        self,
        task_type: str,
        workspace_id: str | None = None,
    ) -> list[str]:
        """Recommend agents based on cultural patterns."""
        self._ensure_initialized()

        if not self._culture_accumulator:
            return []

        ws_id = workspace_id or self.workspace_id
        return await self._culture_accumulator.recommend_agents(task_type, ws_id)

    # =========================================================================
    # Organization-Level Culture
    # =========================================================================

    def get_org_culture_manager(self) -> "OrganizationCultureManager":
        """
        Get the organization culture manager.

        Returns:
            OrganizationCultureManager instance
        """
        self._ensure_initialized()

        # Access _org_culture_manager attribute from composed class
        manager: Optional["OrganizationCultureManager"] = getattr(
            self, "_org_culture_manager", None
        )
        if manager is None:
            from aragora.knowledge.mound.culture import OrganizationCultureManager

            # Mixin pattern: self is the composed KnowledgeMound which satisfies
            # the manager's mound interface at runtime.
            manager = OrganizationCultureManager(
                mound=self,  # type: ignore[arg-type]
                culture_accumulator=self._culture_accumulator,
            )
            self._org_culture_manager = manager

        return manager

    async def get_org_culture(
        self,
        org_id: str,
        workspace_ids: Optional[list[str]] = None,
    ) -> "OrganizationCulture":
        """
        Get the organization culture profile.

        Aggregates patterns from all workspaces plus explicit culture documents.

        Args:
            org_id: Organization ID
            workspace_ids: Optional list of workspaces to include

        Returns:
            Complete organization culture profile
        """
        manager = self.get_org_culture_manager()
        return await manager.get_organization_culture(org_id, workspace_ids)

    async def add_culture_document(
        self,
        org_id: str,
        category: str,
        title: str,
        content: str,
        created_by: str,
    ) -> "CultureDocument":
        """
        Add an explicit culture document.

        Args:
            org_id: Organization ID
            category: Document category (values, practices, standards, policies, learnings)
            title: Document title
            content: Document content
            created_by: User creating the document

        Returns:
            Created culture document
        """
        from aragora.knowledge.mound.culture import CultureDocumentCategory

        manager = self.get_org_culture_manager()
        return await manager.add_document(
            org_id=org_id,
            category=CultureDocumentCategory(category),
            title=title,
            content=content,
            created_by=created_by,
        )

    async def promote_to_culture(
        self,
        workspace_id: str,
        pattern_id: str,
        promoted_by: str,
        title: str | None = None,
    ) -> "CultureDocument":
        """
        Promote a workspace pattern to organization culture.

        Args:
            workspace_id: Workspace containing the pattern
            pattern_id: Pattern ID to promote
            promoted_by: User promoting the pattern
            title: Optional title override

        Returns:
            Created culture document
        """
        manager = self.get_org_culture_manager()
        return await manager.promote_pattern_to_culture(
            workspace_id=workspace_id,
            pattern_id=pattern_id,
            promoted_by=promoted_by,
            title=title,
        )

    async def get_culture_context(
        self,
        org_id: str,
        task: str,
        max_documents: int = 3,
    ) -> str:
        """
        Get relevant culture context for a task.

        This is used to inject organizational knowledge into agent prompts.

        Args:
            org_id: Organization ID
            task: Task description
            max_documents: Maximum documents to include

        Returns:
            Formatted context string
        """
        manager = self.get_org_culture_manager()
        return await manager.get_relevant_context(org_id, task, max_documents)

    def register_workspace_org(
        self,
        workspace_id: str,
        org_id: str,
    ) -> None:
        """
        Register a workspace's organization for culture aggregation.

        Args:
            workspace_id: Workspace ID
            org_id: Organization ID
        """
        manager = self.get_org_culture_manager()
        manager.register_workspace(workspace_id, org_id)
