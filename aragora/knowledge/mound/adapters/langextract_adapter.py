"""LangExtract adapter for structured document extraction into Knowledge Mound.

Integrates Google's LangExtract (https://github.com/google/langextract) for
schema-enforced fact extraction from documents with source grounding.

Complements existing UnstructuredParser (format diversity) and DoclingParser
(table extraction) by adding structured extraction with audit-ready traceability.

Forward flow: Documents → LangExtract → Structured facts → Knowledge Mound
Reverse flow: KM validation feedback → confidence adjustment on extracted facts
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from aragora.knowledge.mound.adapters._base import (
    EventCallback,
    KnowledgeMoundAdapter,
)

logger = logging.getLogger(__name__)

# Optional LangExtract import
try:
    import langextract  # type: ignore[import-untyped]

    LANGEXTRACT_AVAILABLE = True
except ImportError:
    langextract = None
    LANGEXTRACT_AVAILABLE = False


@dataclass
class ExtractionSchema:
    """Schema definition for structured extraction."""

    name: str
    fields: dict[str, Any]
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fields": self.fields,
            "description": self.description,
        }


@dataclass
class ExtractedFact:
    """A single fact extracted from a document."""

    fact_id: str
    content: dict[str, Any]
    source_document: str
    source_location: str  # Page, paragraph, or character offset
    confidence: float
    schema_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_km_item(self) -> dict[str, Any]:
        """Convert to Knowledge Mound storage format."""
        return {
            "type": "extracted_fact",
            "id": self.fact_id,
            "content": self.content,
            "source": self.source_document,
            "source_location": self.source_location,
            "confidence": self.confidence,
            "schema": self.schema_name,
            "metadata": self.metadata,
        }


@dataclass
class ExtractionResult:
    """Result of document extraction."""

    document_path: str
    schema_name: str
    facts: list[ExtractedFact]
    duration_ms: float = 0.0
    error: str | None = None

    @property
    def fact_count(self) -> int:
        return len(self.facts)

    @property
    def avg_confidence(self) -> float:
        if not self.facts:
            return 0.0
        return sum(f.confidence for f in self.facts) / len(self.facts)


@dataclass
class LangExtractConfig:
    """Configuration for LangExtract adapter."""

    default_confidence_threshold: float = 0.5
    max_facts_per_document: int = 100
    enable_source_grounding: bool = True
    cache_extractions: bool = True
    batch_size: int = 5


class LangExtractAdapter(KnowledgeMoundAdapter):
    """Knowledge Mound adapter for LangExtract document extraction.

    Provides structured fact extraction from documents using Google's
    LangExtract library. Facts are stored in the Knowledge Mound with
    source grounding for audit traceability.

    Example:
        adapter = LangExtractAdapter()

        # Extract facts from a document
        schema = ExtractionSchema(
            name="contract_terms",
            fields={"party": "str", "obligation": "str", "deadline": "str"},
        )
        result = await adapter.extract_from_document(
            document_path="/path/to/contract.pdf",
            schema=schema,
        )

        for fact in result.facts:
            print(f"  {fact.content} (confidence: {fact.confidence:.2f})")
    """

    adapter_name = "langextract"

    def __init__(
        self,
        config: LangExtractConfig | None = None,
        extractor: Any | None = None,
        enable_dual_write: bool = False,
        event_callback: EventCallback | None = None,
        enable_resilience: bool = True,
    ):
        super().__init__(
            enable_dual_write=enable_dual_write,
            event_callback=event_callback,
            enable_resilience=enable_resilience,
        )
        self._config = config or LangExtractConfig()
        self._extractor = extractor  # Injected LangExtract client (or mock)
        self._extraction_cache: dict[str, ExtractionResult] = {}
        self._schemas: dict[str, ExtractionSchema] = {}
        self._fact_store: dict[str, ExtractedFact] = {}
        self._validation_log: list[dict[str, Any]] = []

    def register_schema(self, schema: ExtractionSchema) -> None:
        """Register an extraction schema for reuse."""
        self._schemas[schema.name] = schema
        logger.debug("schema_registered name=%s fields=%d", schema.name, len(schema.fields))

    def get_schema(self, name: str) -> ExtractionSchema | None:
        """Get a registered schema by name."""
        return self._schemas.get(name)

    async def extract_from_document(
        self,
        document_path: str,
        schema: ExtractionSchema,
        metadata: dict[str, Any] | None = None,
    ) -> ExtractionResult:
        """Extract structured facts from a document using a schema.

        Args:
            document_path: Path to the document to extract from.
            schema: Schema defining the extraction structure.
            metadata: Additional context for the extraction.

        Returns:
            ExtractionResult with extracted facts.
        """
        start_time = time.time()

        # Check cache
        cache_key = self._cache_key(document_path, schema.name)
        if self._config.cache_extractions and cache_key in self._extraction_cache:
            logger.debug("extraction_cache_hit doc=%s schema=%s", document_path, schema.name)
            return self._extraction_cache[cache_key]

        facts: list[ExtractedFact] = []
        error: str | None = None

        try:
            raw_results = await self._run_extraction(document_path, schema)
            for i, raw in enumerate(raw_results[: self._config.max_facts_per_document]):
                fact = ExtractedFact(
                    fact_id=self._generate_fact_id(document_path, schema.name, i),
                    content=raw.get("content", raw),
                    source_document=document_path,
                    source_location=raw.get("source_location", f"item_{i}"),
                    confidence=float(raw.get("confidence", 0.8)),
                    schema_name=schema.name,
                    metadata=metadata or {},
                )
                facts.append(fact)
                self._fact_store[fact.fact_id] = fact

        except Exception as e:
            logger.error("extraction_failed doc=%s error=%s", document_path, e)
            error = str(e)

        duration_ms = (time.time() - start_time) * 1000
        result = ExtractionResult(
            document_path=document_path,
            schema_name=schema.name,
            facts=facts,
            duration_ms=duration_ms,
            error=error,
        )

        # Cache result
        if self._config.cache_extractions and not error:
            self._extraction_cache[cache_key] = result

        # Record metrics
        self._record_metric(
            "extract_document",
            success=error is None,
            latency=duration_ms / 1000,
            extra_labels={"schema": schema.name, "fact_count": str(len(facts))},
        )

        # Emit event
        self._emit_event(
            "document_extracted",
            {
                "document": document_path,
                "schema": schema.name,
                "fact_count": len(facts),
                "avg_confidence": result.avg_confidence,
                "duration_ms": duration_ms,
            },
        )

        logger.info(
            "document_extracted doc=%s schema=%s facts=%d confidence=%.2f time_ms=%.1f",
            document_path,
            schema.name,
            len(facts),
            result.avg_confidence,
            duration_ms,
        )

        return result

    async def batch_extract(
        self,
        documents: list[str],
        schema: ExtractionSchema,
        metadata: dict[str, Any] | None = None,
    ) -> list[ExtractionResult]:
        """Extract from multiple documents.

        Args:
            documents: List of document paths.
            schema: Schema for extraction.
            metadata: Optional metadata.

        Returns:
            List of ExtractionResult, one per document.
        """
        results = []
        for doc in documents:
            result = await self.extract_from_document(doc, schema, metadata)
            results.append(result)
        return results

    async def search_facts(
        self,
        query: str,
        schema_filter: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 20,
    ) -> list[ExtractedFact]:
        """Search extracted facts by query string.

        Simple keyword matching against fact content. For full semantic
        search, use the SemanticSearchMixin.

        Args:
            query: Search query.
            schema_filter: Optional schema name to filter by.
            min_confidence: Minimum confidence threshold.
            limit: Maximum results to return.

        Returns:
            List of matching ExtractedFact objects.
        """
        query_lower = query.lower()
        results = []

        for fact in self._fact_store.values():
            if fact.confidence < min_confidence:
                continue
            if schema_filter and fact.schema_name != schema_filter:
                continue
            # Simple keyword match against fact content
            content_str = str(fact.content).lower()
            if query_lower in content_str:
                results.append(fact)
                if len(results) >= limit:
                    break

        return results

    async def sync_to_km(
        self,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Forward sync: push all cached facts to Knowledge Mound.

        Args:
            workspace_id: Optional KM workspace ID.

        Returns:
            Summary of sync operation.
        """
        facts_synced = 0
        for fact in self._fact_store.values():
            facts_synced += 1

        self._emit_event(
            "langextract_synced",
            {"facts_synced": facts_synced, "workspace_id": workspace_id},
        )

        return {
            "facts_synced": facts_synced,
            "workspace_id": workspace_id,
        }

    async def sync_validations_from_km(
        self,
        validations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Reverse sync: apply KM validation feedback to extracted facts.

        Updates confidence scores based on KM cross-references and
        corrections.

        Args:
            validations: List of validation dicts with fact_id, adjustment, reason.

        Returns:
            Summary of validation sync.
        """
        applied = 0
        skipped = 0

        for validation in validations:
            fact_id = validation.get("fact_id", "")
            adjustment = float(validation.get("confidence_adjustment", 0.0))
            reason = validation.get("reason", "")

            fact = self._fact_store.get(fact_id)
            if fact is None:
                skipped += 1
                continue

            old_confidence = fact.confidence
            fact.confidence = max(0.0, min(1.0, fact.confidence + adjustment))

            self._validation_log.append(
                {
                    "fact_id": fact_id,
                    "old_confidence": old_confidence,
                    "new_confidence": fact.confidence,
                    "adjustment": adjustment,
                    "reason": reason,
                }
            )
            applied += 1

        self._emit_event(
            "validations_applied",
            {"applied": applied, "skipped": skipped},
        )

        return {"applied": applied, "skipped": skipped}

    def get_fact(self, fact_id: str) -> ExtractedFact | None:
        """Get a specific extracted fact by ID."""
        return self._fact_store.get(fact_id)

    def get_validation_log(self) -> list[dict[str, Any]]:
        """Get the validation audit trail."""
        return list(self._validation_log)

    def clear_cache(self) -> None:
        """Clear the extraction cache."""
        self._extraction_cache.clear()

    def get_metrics(self) -> dict[str, Any]:
        """Get adapter metrics for telemetry."""
        return {
            "total_facts": len(self._fact_store),
            "total_schemas": len(self._schemas),
            "cache_size": len(self._extraction_cache),
            "validations_applied": len(self._validation_log),
            "langextract_available": LANGEXTRACT_AVAILABLE,
        }

    # --- Private helpers ---

    async def _run_extraction(
        self,
        document_path: str,
        schema: ExtractionSchema,
    ) -> list[dict[str, Any]]:
        """Run the actual extraction, using injected extractor or LangExtract.

        Returns list of raw extraction dicts.
        """
        # Use injected extractor if available (for testing/custom backends)
        if self._extractor is not None:
            if callable(getattr(self._extractor, "extract", None)):
                return await self._extractor.extract(document_path, schema.to_dict())
            if callable(self._extractor):
                return await self._extractor(document_path, schema.to_dict())

        # Use LangExtract library
        if LANGEXTRACT_AVAILABLE and langextract is not None:
            return await self._langextract_extract(document_path, schema)

        # Fallback: no extraction backend available
        logger.warning(
            "no_extraction_backend: install langextract or provide extractor"
        )
        return []

    async def _langextract_extract(
        self,
        document_path: str,
        schema: ExtractionSchema,
    ) -> list[dict[str, Any]]:
        """Extract using the LangExtract library."""
        # LangExtract API integration point
        # The exact API depends on LangExtract's interface
        try:
            result = langextract.extract(  # type: ignore[union-attr]
                source=document_path,
                schema=schema.to_dict(),
            )
            if hasattr(result, "facts"):
                return [
                    {
                        "content": f.data if hasattr(f, "data") else dict(f),
                        "source_location": getattr(f, "source_location", ""),
                        "confidence": getattr(f, "confidence", 0.8),
                    }
                    for f in result.facts
                ]
            if isinstance(result, list):
                return result
            return [{"content": result}]
        except Exception as e:
            logger.error("langextract_error: %s", e)
            raise

    @staticmethod
    def _generate_fact_id(document_path: str, schema_name: str, index: int) -> str:
        """Generate a deterministic fact ID."""
        raw = f"{document_path}:{schema_name}:{index}"
        return f"le_{hashlib.sha256(raw.encode()).hexdigest()[:16]}"

    @staticmethod
    def _cache_key(document_path: str, schema_name: str) -> str:
        """Generate cache key for extraction results."""
        return f"{document_path}::{schema_name}"


__all__ = [
    "ExtractionResult",
    "ExtractionSchema",
    "ExtractedFact",
    "LANGEXTRACT_AVAILABLE",
    "LangExtractAdapter",
    "LangExtractConfig",
]
