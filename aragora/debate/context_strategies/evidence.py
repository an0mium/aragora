"""
Evidence Collection Strategy.

Gathers evidence from web, GitHub, and local documentation connectors
using the EvidenceCollector system.

Connectors:
- WebConnector: DuckDuckGo search (if duckduckgo_search installed)
- GitHubConnector: Code/docs from GitHub (if GITHUB_TOKEN set)
- LocalDocsConnector: Local documentation files
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from .base import CachingStrategy

if TYPE_CHECKING:
    from aragora.evidence.collector import EvidencePack

logger = logging.getLogger(__name__)

# Cache size limit
MAX_EVIDENCE_CACHE_SIZE = int(os.getenv("ARAGORA_MAX_EVIDENCE_CACHE", "100"))
EVIDENCE_TIMEOUT = float(os.getenv("ARAGORA_EVIDENCE_TIMEOUT", "30.0"))


class EvidenceStrategy(CachingStrategy):
    """
    Gather evidence from multiple connectors.

    Uses EvidenceCollector with available connectors to search
    web, GitHub, and local documentation for relevant evidence.
    """

    name = "evidence"
    default_timeout = EVIDENCE_TIMEOUT
    max_cache_size = MAX_EVIDENCE_CACHE_SIZE

    def __init__(
        self,
        project_root: Path | None = None,
        prompt_builder: Any = None,
        evidence_store_callback: Callable[[list, str], None] | None = None,
    ) -> None:
        super().__init__()
        self._project_root = project_root or Path.cwd()
        self._prompt_builder = prompt_builder
        self._evidence_store_callback = evidence_store_callback
        self._evidence_packs: dict[str, EvidencePack] = {}

    def set_prompt_builder(self, prompt_builder: Any) -> None:
        """Set the prompt builder for evidence injection."""
        self._prompt_builder = prompt_builder

    def get_evidence_pack(self, task: str) -> EvidencePack | None:
        """Get cached evidence pack for a task."""
        key = self._get_cache_key(task)
        return self._evidence_packs.get(key)

    def is_available(self) -> bool:
        """Check if evidence collector is available."""
        try:
            from aragora.evidence.collector import EvidenceCollector  # noqa: F401

            return True
        except ImportError:
            return False

    async def gather(self, task: str, **kwargs: Any) -> str | None:
        """
        Gather evidence from web, GitHub, and local docs connectors.

        Args:
            task: The debate topic/task description.

        Returns:
            Formatted evidence context, or None if unavailable.
        """
        try:
            from aragora.evidence.collector import EvidenceCollector

            collector = EvidenceCollector()
            enabled_connectors = []

            # Add web connector if available
            try:
                from aragora.connectors.web import DDGS_AVAILABLE, WebConnector

                if DDGS_AVAILABLE:
                    collector.add_connector("web", WebConnector())
                    enabled_connectors.append("web")
            except ImportError:
                logger.debug("WebConnector not available: duckduckgo_search not installed")

            # Add GitHub connector if available
            try:
                from aragora.connectors.github import GitHubConnector

                if os.environ.get("GITHUB_TOKEN"):
                    collector.add_connector("github", GitHubConnector())
                    enabled_connectors.append("github")
            except ImportError:
                logger.debug("GitHubConnector not available")

            # Add local docs connector
            try:
                from aragora.connectors.local_docs import LocalDocsConnector

                collector.add_connector(
                    "local_docs",
                    LocalDocsConnector(
                        root_path=str(self._project_root / "docs"),
                        file_types="docs",
                    ),
                )
                enabled_connectors.append("local_docs")
            except ImportError:
                logger.debug("LocalDocsConnector not available")

            if not enabled_connectors:
                return None

            evidence_pack = await collector.collect_evidence(
                task, enabled_connectors=enabled_connectors
            )

            if evidence_pack.snippets:
                # Cache evidence pack
                key = self._get_cache_key(task)
                self._evidence_packs[key] = evidence_pack

                # Update prompt builder if available
                if self._prompt_builder:
                    self._prompt_builder.set_evidence_pack(evidence_pack)

                # Store evidence via callback if provided
                if self._evidence_store_callback and callable(self._evidence_store_callback):
                    self._evidence_store_callback(evidence_pack.snippets, task)

                return f"## EVIDENCE CONTEXT\n{evidence_pack.to_context_string()}"

        except ImportError as e:
            logger.debug("Evidence collector not available: %s", e)
        except (ConnectionError, OSError) as e:
            logger.warning("Evidence collection network/IO error: %s", e)
        except (ValueError, RuntimeError) as e:
            logger.warning("Evidence collection failed: %s", e)
        except Exception as e:  # noqa: BLE001
            logger.warning("Unexpected error in evidence collection: %s", e)

        return None
