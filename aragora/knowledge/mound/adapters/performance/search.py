"""
Search methods and mixin implementations for the PerformanceAdapter.

Handles:
- FusionMixin implementation (cross-adapter fusion)
- SemanticSearchMixin implementation (vector search)
- Keyword search across ratings, expertise, and matches
- Record retrieval and filtering
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class _FusionHostProtocol(Protocol):
    """Protocol for host class of FusionImplementationMixin."""

    def _emit_event(self, event_type: str, data: dict[str, Any]) -> None: ...


class _SemanticSearchHostProtocol(Protocol):
    """Protocol for host class of SemanticSearchImplementationMixin."""

    ELO_PREFIX: str
    EXPERTISE_PREFIX: str
    _ratings: dict[str, dict[str, Any]]
    _expertise: dict[str, dict[str, Any]]
    _matches: dict[str, dict[str, Any]]
    _calibrations: dict[str, dict[str, Any]]


class _SearchHostProtocol(Protocol):
    """Protocol for host class of SearchMixin."""

    _ratings: dict[str, dict[str, Any]]
    _expertise: dict[str, dict[str, Any]]
    _matches: dict[str, dict[str, Any]]
    _calibrations: dict[str, dict[str, Any]]

    def _get_record_by_id(self, record_id: str) -> dict[str, Any] | None: ...


class FusionImplementationMixin:
    """Implements FusionMixin abstract methods for PerformanceAdapter.

    NOTE: Does NOT inherit from Protocol to preserve cooperative inheritance.

    Expects the following attributes on the host class:
    - _emit_event(event_type, data): method for event emission
    """

    # Attribute declarations for mypy (provided by host class)
    _emit_event: Any

    def _get_fusion_sources(self) -> list[str]:
        """Return list of adapter names this adapter can fuse data from.

        PerformanceAdapter can fuse validations from Consensus (debate outcomes),
        Evidence (supporting data), and Belief (claim confidence) adapters.
        """
        return ["consensus", "evidence", "belief", "continuum"]

    def _extract_fusible_data(self, km_item: dict[str, Any]) -> dict[str, Any] | None:
        """Extract data from a KM item that can be used for fusion.

        Args:
            km_item: Knowledge Mound item dict

        Returns:
            Dict with fusible fields, or None if not fusible
        """
        metadata = km_item.get("metadata", {})

        # Extract performance-relevant fields
        item_id = (
            metadata.get("source_id")
            or metadata.get("agent_name")
            or metadata.get("rating_id")
            or km_item.get("id")
        )

        if not item_id:
            return None

        confidence = km_item.get("confidence", 0.5)
        if isinstance(confidence, str):
            confidence = {"low": 0.3, "medium": 0.6, "high": 0.9}.get(confidence.lower(), 0.5)

        return {
            "item_id": item_id,
            "confidence": confidence,
            "source_adapter": metadata.get("source_adapter", "unknown"),
            "agent_name": metadata.get("agent_name"),
            "domain": metadata.get("domain"),
            "calibration_score": metadata.get("calibration_score"),
            "validation_count": metadata.get("validation_count", 1),
        }

    def _apply_fusion_result(
        self,
        record: Any,
        fusion_result: Any,  # FusedValidation from ops.fusion
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Apply a fusion result to an agent rating or expertise record.

        Args:
            record: The rating/expertise dict or object to update
            fusion_result: FusedValidation with fused confidence/validity
            metadata: Optional additional metadata

        Returns:
            True if successfully applied, False otherwise
        """
        try:
            # Handle both dict and object records
            if isinstance(record, dict):
                record["fusion_applied"] = True
                record["fused_confidence"] = fusion_result.fused_confidence
                record["fusion_is_valid"] = fusion_result.is_valid
                record["fusion_strategy"] = fusion_result.strategy_used.value
                record["fusion_source_count"] = len(fusion_result.source_validations)
                record["fusion_timestamp"] = datetime.now(timezone.utc).isoformat()
                if metadata:
                    record["fusion_metadata"] = metadata
            else:
                # Object with metadata attribute (like AgentRating)
                if hasattr(record, "metadata") and isinstance(record.metadata, dict):
                    record.metadata["fusion_applied"] = True
                    record.metadata["fused_confidence"] = fusion_result.fused_confidence
                    record.metadata["fusion_is_valid"] = fusion_result.is_valid
                    record.metadata["fusion_strategy"] = fusion_result.strategy_used.value
                    record.metadata["fusion_source_count"] = len(fusion_result.source_validations)
                    record.metadata["fusion_timestamp"] = datetime.now(timezone.utc).isoformat()
                    if metadata:
                        record.metadata["fusion_metadata"] = metadata

            # Emit event for fusion application
            self._emit_event(
                "km_adapter_fusion_applied",
                {
                    "adapter": "performance",
                    "record_id": record.get("id")
                    if isinstance(record, dict)
                    else getattr(record, "agent_name", "unknown"),
                    "fused_confidence": fusion_result.fused_confidence,
                    "is_valid": fusion_result.is_valid,
                    "source_count": len(fusion_result.source_validations),
                },
            )

            logger.debug(
                f"Applied fusion to performance record: "
                f"confidence={fusion_result.fused_confidence:.2f}, "
                f"sources={len(fusion_result.source_validations)}"
            )

            return True

        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.warning("Failed to apply fusion result: %s", e)
            return False


class SemanticSearchImplementationMixin:
    """Implements SemanticSearchMixin abstract methods for PerformanceAdapter.

    NOTE: Does NOT inherit from Protocol to preserve cooperative inheritance.

    Expects the following attributes on the host class:
    - ELO_PREFIX: str
    - EXPERTISE_PREFIX: str
    - _ratings: dict[str, dict[str, Any]]
    - _expertise: dict[str, dict[str, Any]]
    - _matches: dict[str, dict[str, Any]]
    - _calibrations: dict[str, dict[str, Any]]
    """

    # Attribute declarations for mypy (provided by host class)
    ELO_PREFIX: str
    EXPERTISE_PREFIX: str
    _ratings: dict[str, dict[str, Any]]
    _expertise: dict[str, dict[str, Any]]
    _matches: dict[str, dict[str, Any]]
    _calibrations: dict[str, dict[str, Any]]

    def _get_record_by_id(self, record_id: str) -> dict[str, Any] | None:
        """Get a performance record by ID (required by SemanticSearchMixin).

        Supports both ELO rating IDs (el_ prefix) and expertise IDs (ex_ prefix).

        Args:
            record_id: The record identifier.

        Returns:
            The full record dict, or None if not found.
        """
        # Check ELO ratings
        if record_id.startswith(self.ELO_PREFIX) or record_id in self._ratings:
            return self._ratings.get(record_id)

        # Check expertise records
        if record_id.startswith(self.EXPERTISE_PREFIX) or record_id in self._expertise:
            return self._expertise.get(record_id)

        # Check matches
        if record_id in self._matches:
            return self._matches.get(record_id)

        # Check calibrations
        if record_id in self._calibrations:
            return self._calibrations.get(record_id)

        return None

    def _record_to_dict(self, record: dict[str, Any], similarity: float = 0.0) -> dict[str, Any]:
        """Convert a performance record to dict (required by SemanticSearchMixin).

        Args:
            record: The record dict to convert.
            similarity: Optional similarity score to include.

        Returns:
            Dictionary representation with similarity.
        """
        result = dict(record)
        if similarity > 0:
            result["similarity"] = similarity
        return result

    def _extract_record_id(self, source_id: str) -> str:
        """Extract record ID from prefixed source ID (override for SemanticSearchMixin).

        Args:
            source_id: The source ID from SemanticStore.

        Returns:
            The actual record ID for lookup.
        """
        # Performance records use el_, ex_, dm_ prefixes
        # SemanticStore may add its own prefix, so strip both if needed
        if source_id.startswith("performance:"):
            source_id = source_id[12:]  # len("performance:")
        return source_id


class SearchMixin:
    """Mixin providing search and record retrieval methods.

    NOTE: Does NOT inherit from Protocol to preserve cooperative inheritance.

    Expects the following attributes on the host class:
    - _ratings: dict[str, dict[str, Any]]
    - _expertise: dict[str, dict[str, Any]]
    - _matches: dict[str, dict[str, Any]]
    - _calibrations: dict[str, dict[str, Any]]
    - _get_record_by_id(record_id): method from SemanticSearchImplementationMixin
    """

    # Attribute declarations for mypy (provided by host class)
    _ratings: dict[str, dict[str, Any]]
    _expertise: dict[str, dict[str, Any]]
    _matches: dict[str, dict[str, Any]]
    _calibrations: dict[str, dict[str, Any]]
    _get_record_by_id: Any  # Method from SemanticSearchImplementationMixin

    def get(self, record_id: str) -> dict[str, Any] | None:
        """Get a single performance record by ID.

        Args:
            record_id: The record identifier.

        Returns:
            The record dict, or None if not found.
        """
        return self._get_record_by_id(record_id)

    def search_by_keyword(
        self,
        keyword: str,
        limit: int = 20,
        record_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search performance data by keyword.

        Args:
            keyword: The search keyword.
            limit: Maximum results to return.
            record_type: Optional filter for record type ("rating", "expertise", "match").

        Returns:
            List of matching records with relevance scores.
        """
        results: list[dict[str, Any]] = []
        keyword_lower = keyword.lower()

        # Search ratings
        if record_type is None or record_type == "rating":
            for rating_id, rating in self._ratings.items():
                agent_name = rating.get("agent_name", "").lower()
                reason = rating.get("reason", "").lower()
                if keyword_lower in agent_name or keyword_lower in reason:
                    result = dict(rating)
                    result["_match_type"] = "rating"
                    result["_relevance"] = 1.0 if keyword_lower in agent_name else 0.7
                    results.append(result)

        # Search expertise
        if record_type is None or record_type == "expertise":
            for exp_key, expertise in self._expertise.items():
                agent_name = expertise.get("agent_name", "").lower()
                domain = expertise.get("domain", "").lower()
                if keyword_lower in agent_name or keyword_lower in domain:
                    result = dict(expertise)
                    result["_match_type"] = "expertise"
                    result["_relevance"] = (
                        1.0
                        if keyword_lower in domain
                        else 0.8
                        if keyword_lower in agent_name
                        else 0.5
                    )
                    results.append(result)

        # Search matches
        if record_type is None or record_type == "match":
            for match_id, match in self._matches.items():
                winner = match.get("winner", "").lower()
                loser = match.get("loser", "").lower()
                if keyword_lower in winner or keyword_lower in loser:
                    result = dict(match)
                    result["_match_type"] = "match"
                    result["_relevance"] = 0.7
                    results.append(result)

        # Sort by relevance and limit
        results.sort(key=lambda x: x.get("_relevance", 0), reverse=True)
        return results[:limit]

    def search_similar(
        self,
        query: str,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Search for similar performance records (fallback for semantic search).

        Uses keyword matching as a fallback when vector search is unavailable.

        Args:
            query: The search query.
            limit: Maximum results to return.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of matching records.
        """
        # Use keyword search as fallback
        results = self.search_by_keyword(query, limit=limit * 2)

        # Filter by confidence if applicable
        if min_confidence > 0:
            results = [
                r
                for r in results
                if r.get("confidence", r.get("calibration_accuracy", 1.0)) >= min_confidence
            ]

        return results[:limit]

    def get_all_records(
        self,
        record_type: str | None = None,
        agent_name: str | None = None,
        domain: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get all performance records with optional filtering.

        Args:
            record_type: Filter by type ("rating", "expertise", "match", "calibration").
            agent_name: Filter by agent name.
            domain: Filter by domain.
            limit: Maximum results to return.

        Returns:
            List of matching records.
        """
        results: list[dict[str, Any]] = []

        # Collect ratings
        if record_type is None or record_type == "rating":
            for rating in self._ratings.values():
                if agent_name and rating.get("agent_name") != agent_name:
                    continue
                # Domain filter for domain_elos
                if domain and domain not in rating.get("domain_elos", {}):
                    continue
                results.append(dict(rating))

        # Collect expertise
        if record_type is None or record_type == "expertise":
            for expertise in self._expertise.values():
                if agent_name and expertise.get("agent_name") != agent_name:
                    continue
                if domain and expertise.get("domain") != domain:
                    continue
                results.append(dict(expertise))

        # Collect matches
        if record_type is None or record_type == "match":
            for match in self._matches.values():
                if agent_name and agent_name not in (match.get("winner"), match.get("loser")):
                    continue
                results.append(dict(match))

        # Collect calibrations
        if record_type is None or record_type == "calibration":
            for calibration in self._calibrations.values():
                if agent_name and calibration.get("agent_name") != agent_name:
                    continue
                results.append(dict(calibration))

        return results[:limit]


__all__ = [
    "FusionImplementationMixin",
    "SemanticSearchImplementationMixin",
    "SearchMixin",
]
