"""
Deduplication operations for Knowledge Mound.

Provides similarity-based deduplication (beyond exact hash matching):
- Find near-duplicates by semantic similarity
- Merge duplicate clusters
- Dedup visibility/reporting
"""

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    pass


@dataclass
class DuplicateMatch:
    """A potential duplicate match."""

    node_id: str
    similarity: float
    content_preview: str
    created_at: datetime
    tier: str
    confidence: float


@dataclass
class DuplicateCluster:
    """A cluster of potential duplicate nodes."""

    cluster_id: str
    primary_node_id: str
    duplicates: list[DuplicateMatch]
    avg_similarity: float
    recommended_action: str  # "merge", "review", "keep_separate"


@dataclass
class MergeResult:
    """Result of a merge operation."""

    kept_node_id: str
    merged_node_ids: list[str]
    archived_count: int
    updated_relationships: int


@dataclass
class DedupReport:
    """Report of deduplication analysis."""

    workspace_id: str
    generated_at: datetime
    total_nodes_analyzed: int
    duplicate_clusters_found: int
    clusters: list[DuplicateCluster]
    estimated_reduction_percent: float


class DedupOperationsMixin:
    """Mixin providing deduplication operations for Knowledge Mound."""

    # Type stubs for mixin - actual implementation provided by composed class
    # Uses adapter methods from KnowledgeMoundCore

    async def find_duplicates(
        self,
        workspace_id: str,
        similarity_threshold: float = 0.9,
        limit: int = 100,
    ) -> list[DuplicateCluster]:
        """Find potential duplicate clusters in the knowledge mound.

        Uses semantic similarity to identify near-duplicates that
        may not be caught by exact hash matching.

        Args:
            workspace_id: Workspace to analyze
            similarity_threshold: Minimum similarity (0-1) to consider duplicates
            limit: Maximum clusters to return

        Returns:
            List of duplicate clusters sorted by similarity
        """
        clusters = []

        # Get all nodes for workspace using adapter method
        nodes = await self._get_nodes_for_workspace(  # type: ignore[attr-defined]
            workspace_id=workspace_id,
            limit=1000,  # Sample for performance
        )

        if not nodes:
            return []

        # Build similarity matrix using embeddings
        processed_ids: set[str] = set()
        cluster_id = 0

        for node in nodes:
            if node.id in processed_ids:
                continue

            # Find similar nodes using semantic search via adapter method
            similar = await self._search_similar(  # type: ignore[attr-defined]
                workspace_id=workspace_id,
                embedding=node.embedding if hasattr(node, "embedding") else None,
                query=node.content[:500],  # Fallback to content search
                top_k=20,
                min_score=similarity_threshold,
            )

            # Filter to actual duplicates (excluding self)
            duplicates = [
                DuplicateMatch(
                    node_id=s.id,
                    similarity=s.score,
                    content_preview=s.content[:200] if hasattr(s, "content") else "",
                    created_at=s.created_at if hasattr(s, "created_at") else datetime.now(),
                    tier=s.tier if hasattr(s, "tier") else "medium",
                    confidence=s.confidence if hasattr(s, "confidence") else 0.5,
                )
                for s in similar
                if s.id != node.id and s.id not in processed_ids
            ]

            if duplicates:
                avg_sim = sum(d.similarity for d in duplicates) / len(duplicates)

                # Determine recommended action
                if avg_sim >= 0.95:
                    action = "merge"
                elif avg_sim >= 0.85:
                    action = "review"
                else:
                    action = "keep_separate"

                clusters.append(
                    DuplicateCluster(
                        cluster_id=f"cluster_{cluster_id}",
                        primary_node_id=node.id,
                        duplicates=duplicates,
                        avg_similarity=avg_sim,
                        recommended_action=action,
                    )
                )
                cluster_id += 1

                # Mark all as processed
                processed_ids.add(node.id)
                for d in duplicates:
                    processed_ids.add(d.node_id)

            if len(clusters) >= limit:
                break

        # Sort by average similarity (highest first)
        clusters.sort(key=lambda c: c.avg_similarity, reverse=True)
        return clusters[:limit]

    async def generate_dedup_report(
        self,
        workspace_id: str,
        similarity_threshold: float = 0.9,
    ) -> DedupReport:
        """Generate a deduplication analysis report.

        Args:
            workspace_id: Workspace to analyze
            similarity_threshold: Minimum similarity for duplicates

        Returns:
            Comprehensive dedup report
        """
        clusters = await self.find_duplicates(  # type: ignore[attr-defined]
            workspace_id=workspace_id,
            similarity_threshold=similarity_threshold,
            limit=500,
        )

        # Count total nodes using adapter method
        total_nodes = await self._count_nodes(workspace_id=workspace_id)  # type: ignore[attr-defined]

        # Calculate potential reduction
        duplicate_count = sum(len(c.duplicates) for c in clusters)
        reduction_percent = (duplicate_count / total_nodes * 100) if total_nodes > 0 else 0

        return DedupReport(
            workspace_id=workspace_id,
            generated_at=datetime.now(),
            total_nodes_analyzed=total_nodes,
            duplicate_clusters_found=len(clusters),
            clusters=clusters,
            estimated_reduction_percent=round(reduction_percent, 2),
        )

    async def merge_duplicates(
        self,
        workspace_id: str,
        cluster_id: str,
        primary_node_id: Optional[str] = None,
        archive: bool = True,
    ) -> MergeResult:
        """Merge a cluster of duplicate nodes into one.

        Keeps the primary node and archives/deletes the duplicates.
        Relationships from duplicates are transferred to the primary.

        Args:
            workspace_id: Workspace containing the nodes
            cluster_id: Cluster ID from find_duplicates
            primary_node_id: Which node to keep (defaults to cluster's primary)
            archive: Whether to archive instead of delete

        Returns:
            Result of the merge operation
        """
        # Find the cluster
        clusters = await self.find_duplicates(  # type: ignore[attr-defined]
            workspace_id=workspace_id,
            similarity_threshold=0.8,  # Lower threshold to find the cluster
            limit=500,
        )

        cluster = next((c for c in clusters if c.cluster_id == cluster_id), None)
        if not cluster:
            raise ValueError(f"Cluster {cluster_id} not found")

        keep_id = primary_node_id or cluster.primary_node_id
        merged_ids = [d.node_id for d in cluster.duplicates if d.node_id != keep_id]

        # Include original primary if different from keep_id
        if cluster.primary_node_id != keep_id:
            merged_ids.append(cluster.primary_node_id)

        # Transfer relationships from duplicates to primary using adapter method
        updated_rels = 0
        for merge_id in merged_ids:
            # Get relationships from duplicate
            rels = await self._get_node_relationships_for_ops(merge_id, workspace_id)  # type: ignore[attr-defined]
            for rel in rels:
                # Update relationship to point to/from primary
                source = getattr(rel, "source_id", None) or getattr(rel, "from_node_id", None)
                target = getattr(rel, "target_id", None) or getattr(rel, "to_node_id", None)
                rel_type = (
                    getattr(rel, "type", None) or getattr(rel, "relationship", None) or "related_to"
                )
                if hasattr(rel_type, "value"):
                    rel_type = rel_type.value
                if source == merge_id:
                    await self._create_relationship(  # type: ignore[attr-defined]
                        source_id=keep_id,
                        target_id=target,
                        relationship_type=rel_type,
                        workspace_id=workspace_id,
                    )
                elif target == merge_id:
                    await self._create_relationship(  # type: ignore[attr-defined]
                        source_id=source,
                        target_id=keep_id,
                        relationship_type=rel_type,
                        workspace_id=workspace_id,
                    )
                updated_rels += 1

        # Archive or delete duplicates using adapter methods
        archived = 0
        for merge_id in merged_ids:
            if archive:
                await self._archive_node_with_reason(  # type: ignore[attr-defined]
                    node_id=merge_id,
                    workspace_id=workspace_id,
                    reason=f"merged_into_{keep_id}",
                )
            else:
                await self._delete_node(merge_id)  # type: ignore[attr-defined]
            archived += 1

        return MergeResult(
            kept_node_id=keep_id,
            merged_node_ids=merged_ids,
            archived_count=archived,
            updated_relationships=updated_rels,
        )

    async def auto_merge_exact_duplicates(
        self,
        workspace_id: str,
        dry_run: bool = True,
    ) -> dict:
        """Automatically merge exact duplicates (hash matches).

        Only merges when content hashes match exactly.

        Args:
            workspace_id: Workspace to process
            dry_run: If True, report what would be merged without doing it

        Returns:
            Summary of merges (or would-be merges if dry_run)
        """
        # Get nodes grouped by content hash using adapter method
        hash_groups = await self._get_nodes_by_content_hash(workspace_id)  # type: ignore[attr-defined]

        duplicates_found = 0
        merges_performed = 0
        merged_ids = []

        for content_hash, node_ids in hash_groups.items():
            if len(node_ids) <= 1:
                continue

            duplicates_found += len(node_ids) - 1

            if not dry_run:
                # Keep the oldest node - get nodes using adapter
                nodes = [await self._get_node(nid) for nid in node_ids]  # type: ignore[attr-defined]
                nodes = [n for n in nodes if n is not None]
                nodes_sorted = sorted(nodes, key=lambda n: n.created_at)
                keep = nodes_sorted[0]
                merge = nodes_sorted[1:]

                for m in merge:
                    await self._archive_node_with_reason(  # type: ignore[attr-defined]
                        node_id=m.id,
                        workspace_id=workspace_id,
                        reason=f"exact_duplicate_of_{keep.id}",
                    )
                    merged_ids.append(m.id)
                    merges_performed += 1

        return {
            "duplicates_found": duplicates_found,
            "merges_performed": merges_performed if not dry_run else 0,
            "merged_ids": merged_ids if not dry_run else [],
            "dry_run": dry_run,
        }
