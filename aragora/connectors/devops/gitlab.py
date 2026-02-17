"""GitLab Connector.

Provides integration with GitLab for repos, pipelines, and issues.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from aragora.connectors.base import BaseConnector, Evidence
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)

CONFIG_ENV_VARS = ("GITLAB_TOKEN",)


class GitLabConnector(BaseConnector):
    """GitLab connector for repository, pipeline, and issue data."""

    def __init__(self) -> None:
        super().__init__()
        self._configured = all(os.environ.get(v) for v in CONFIG_ENV_VARS)

    @property
    def name(self) -> str:
        return "gitlab"

    @property
    def source_type(self) -> SourceType:
        return SourceType.CODE_ANALYSIS

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[Evidence]:
        """Search GitLab for repos, issues, and merge requests."""
        if not self._configured:
            logger.debug("GitLab connector not configured")
            return []
        # TODO: Implement GitLab API search
        return []

    async def fetch(self, evidence_id: str, **kwargs: Any) -> Evidence | None:
        """Fetch a specific repo, issue, or MR from GitLab."""
        if not self._configured:
            return None
        # TODO: Implement GitLab API fetch
        return None
