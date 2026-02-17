"""RLM Hierarchy <-> Knowledge Mound Graph bridge.

Maps RLM's hierarchical context (summary layers) to KM graph nodes
with 'summarizes' relationships.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BridgeResult:
    nodes_created: int
    relationships_created: int
    errors: int


class RLMKMBridge:
    def __init__(self, km_backend: Any):
        self._km = km_backend

    async def map_hierarchy(
        self,
        content_id: str,
        full_content: str,
        compressed_content: str,
        source_type: str = "rlm_compression",
    ) -> BridgeResult:
        """Map RLM full and compressed versions as linked KM graph nodes."""
        nodes_created = 0
        relationships_created = 0
        errors = 0

        try:
            # Store full content node
            full_node_id = f"rlm_full_{content_id}"
            if hasattr(self._km, "store_knowledge"):
                store_fn = getattr(self._km, "store_knowledge")
                if asyncio.iscoroutinefunction(store_fn):
                    await store_fn(
                        content=full_content,
                        source=source_type,
                        source_id=full_node_id,
                        confidence=1.0,
                        metadata={"level": "full", "content_id": content_id},
                    )
                else:
                    store_fn(
                        content=full_content,
                        source=source_type,
                        source_id=full_node_id,
                        confidence=1.0,
                        metadata={"level": "full", "content_id": content_id},
                    )
                nodes_created += 1

            # Store compressed/summary node
            summary_node_id = f"rlm_summary_{content_id}"
            if hasattr(self._km, "store_knowledge"):
                store_fn = getattr(self._km, "store_knowledge")
                if asyncio.iscoroutinefunction(store_fn):
                    await store_fn(
                        content=compressed_content,
                        source=source_type,
                        source_id=summary_node_id,
                        confidence=0.9,
                        metadata={
                            "level": "summary",
                            "content_id": content_id,
                            "summarizes": full_node_id,
                        },
                    )
                else:
                    store_fn(
                        content=compressed_content,
                        source=source_type,
                        source_id=summary_node_id,
                        confidence=0.9,
                        metadata={
                            "level": "summary",
                            "content_id": content_id,
                            "summarizes": full_node_id,
                        },
                    )
                nodes_created += 1

            # Create relationship if KM supports it
            if hasattr(self._km, "add_relationship"):
                rel_fn = getattr(self._km, "add_relationship")
                if asyncio.iscoroutinefunction(rel_fn):
                    await rel_fn(
                        source_id=summary_node_id,
                        target_id=full_node_id,
                        relationship_type="summarizes",
                    )
                else:
                    rel_fn(
                        source_id=summary_node_id,
                        target_id=full_node_id,
                        relationship_type="summarizes",
                    )
                relationships_created += 1

        except (RuntimeError, ValueError, OSError, AttributeError, TypeError) as exc:
            logger.warning("RLM-KM bridge mapping failed: %s", exc)
            errors += 1

        return BridgeResult(nodes_created, relationships_created, errors)
