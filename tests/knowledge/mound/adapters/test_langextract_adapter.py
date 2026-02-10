"""Tests for LangExtract Knowledge Mound adapter."""

from __future__ import annotations

import pytest

from aragora.knowledge.mound.adapters.langextract_adapter import (
    ExtractionResult,
    ExtractionSchema,
    ExtractedFact,
    LangExtractAdapter,
    LangExtractConfig,
)


# --- Fixtures ---


def _make_schema(name: str = "test_schema") -> ExtractionSchema:
    return ExtractionSchema(
        name=name,
        fields={"title": "str", "amount": "float", "date": "str"},
        description="Test extraction schema",
    )


def _make_extractor(results: list[dict] | None = None):
    """Create a mock extractor that returns canned results."""

    async def extract(document_path: str, schema: dict) -> list[dict]:
        if results is not None:
            return results
        return [
            {
                "content": {"title": "Contract A", "amount": 1000.0, "date": "2026-01-15"},
                "source_location": "page_1:para_3",
                "confidence": 0.92,
            },
            {
                "content": {"title": "Contract B", "amount": 2500.0, "date": "2026-02-01"},
                "source_location": "page_2:para_1",
                "confidence": 0.85,
            },
        ]

    return extract


# --- Schema Tests ---


class TestExtractionSchema:
    def test_to_dict(self):
        schema = _make_schema()
        d = schema.to_dict()
        assert d["name"] == "test_schema"
        assert "title" in d["fields"]

    def test_description_optional(self):
        schema = ExtractionSchema(name="minimal", fields={"x": "int"})
        assert schema.description == ""


# --- ExtractedFact Tests ---


class TestExtractedFact:
    def test_to_km_item(self):
        fact = ExtractedFact(
            fact_id="le_abc123",
            content={"title": "Test"},
            source_document="/tmp/doc.pdf",
            source_location="page_1",
            confidence=0.9,
            schema_name="test",
        )
        item = fact.to_km_item()
        assert item["type"] == "extracted_fact"
        assert item["id"] == "le_abc123"
        assert item["confidence"] == 0.9
        assert item["source"] == "/tmp/doc.pdf"

    def test_metadata_default_empty(self):
        fact = ExtractedFact(
            fact_id="x",
            content={},
            source_document="",
            source_location="",
            confidence=0.5,
            schema_name="s",
        )
        assert fact.metadata == {}


# --- ExtractionResult Tests ---


class TestExtractionResult:
    def test_fact_count(self):
        result = ExtractionResult(
            document_path="/tmp/doc.pdf",
            schema_name="test",
            facts=[
                ExtractedFact("a", {}, "", "", 0.8, "test"),
                ExtractedFact("b", {}, "", "", 0.6, "test"),
            ],
        )
        assert result.fact_count == 2

    def test_avg_confidence(self):
        result = ExtractionResult(
            document_path="/tmp/doc.pdf",
            schema_name="test",
            facts=[
                ExtractedFact("a", {}, "", "", 0.8, "test"),
                ExtractedFact("b", {}, "", "", 0.6, "test"),
            ],
        )
        assert result.avg_confidence == pytest.approx(0.7)

    def test_avg_confidence_empty(self):
        result = ExtractionResult(
            document_path="/tmp/doc.pdf",
            schema_name="test",
            facts=[],
        )
        assert result.avg_confidence == 0.0


# --- Adapter Core Tests ---


class TestLangExtractAdapter:
    def test_default_config(self):
        adapter = LangExtractAdapter()
        assert adapter.adapter_name == "langextract"
        assert adapter._config.default_confidence_threshold == 0.5
        assert adapter._config.max_facts_per_document == 100

    def test_custom_config(self):
        config = LangExtractConfig(max_facts_per_document=10, cache_extractions=False)
        adapter = LangExtractAdapter(config=config)
        assert adapter._config.max_facts_per_document == 10
        assert adapter._config.cache_extractions is False

    def test_register_schema(self):
        adapter = LangExtractAdapter()
        schema = _make_schema("contracts")
        adapter.register_schema(schema)
        assert adapter.get_schema("contracts") is schema
        assert adapter.get_schema("nonexistent") is None

    def test_get_metrics_initial(self):
        adapter = LangExtractAdapter()
        metrics = adapter.get_metrics()
        assert metrics["total_facts"] == 0
        assert metrics["total_schemas"] == 0
        assert metrics["cache_size"] == 0


# --- Extraction Tests ---


class TestExtraction:
    @pytest.mark.asyncio
    async def test_extract_from_document(self):
        adapter = LangExtractAdapter(extractor=_make_extractor())
        schema = _make_schema()

        result = await adapter.extract_from_document("/tmp/contract.pdf", schema)

        assert result.fact_count == 2
        assert result.error is None
        assert result.document_path == "/tmp/contract.pdf"
        assert result.schema_name == "test_schema"
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_extract_populates_fact_store(self):
        adapter = LangExtractAdapter(extractor=_make_extractor())
        schema = _make_schema()

        result = await adapter.extract_from_document("/tmp/doc.pdf", schema)

        assert len(adapter._fact_store) == 2
        # Can retrieve by ID
        fact = adapter.get_fact(result.facts[0].fact_id)
        assert fact is not None
        assert fact.source_document == "/tmp/doc.pdf"

    @pytest.mark.asyncio
    async def test_extract_caches_result(self):
        call_count = 0

        async def counting_extractor(doc, schema):
            nonlocal call_count
            call_count += 1
            return [{"content": {"x": 1}, "confidence": 0.9, "source_location": "p1"}]

        adapter = LangExtractAdapter(extractor=counting_extractor)
        schema = _make_schema()

        # First call
        r1 = await adapter.extract_from_document("/tmp/doc.pdf", schema)
        assert call_count == 1

        # Second call - should use cache
        r2 = await adapter.extract_from_document("/tmp/doc.pdf", schema)
        assert call_count == 1  # Not called again
        assert r1.fact_count == r2.fact_count

    @pytest.mark.asyncio
    async def test_extract_cache_disabled(self):
        call_count = 0

        async def counting_extractor(doc, schema):
            nonlocal call_count
            call_count += 1
            return [{"content": {"x": 1}, "confidence": 0.9, "source_location": "p1"}]

        config = LangExtractConfig(cache_extractions=False)
        adapter = LangExtractAdapter(config=config, extractor=counting_extractor)
        schema = _make_schema()

        await adapter.extract_from_document("/tmp/doc.pdf", schema)
        await adapter.extract_from_document("/tmp/doc.pdf", schema)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_extract_handles_error(self):
        async def failing_extractor(doc, schema):
            raise RuntimeError("Extraction failed")

        adapter = LangExtractAdapter(extractor=failing_extractor)
        schema = _make_schema()

        result = await adapter.extract_from_document("/tmp/doc.pdf", schema)

        assert result.fact_count == 0
        assert result.error is not None
        assert "Extraction failed" in result.error

    @pytest.mark.asyncio
    async def test_extract_respects_max_facts(self):
        async def many_results(doc, schema):
            return [
                {"content": {"i": i}, "confidence": 0.9, "source_location": f"p{i}"}
                for i in range(200)
            ]

        config = LangExtractConfig(max_facts_per_document=5)
        adapter = LangExtractAdapter(config=config, extractor=many_results)
        schema = _make_schema()

        result = await adapter.extract_from_document("/tmp/doc.pdf", schema)
        assert result.fact_count == 5

    @pytest.mark.asyncio
    async def test_extract_no_backend_returns_empty(self):
        adapter = LangExtractAdapter()  # No extractor, no langextract
        schema = _make_schema()

        result = await adapter.extract_from_document("/tmp/doc.pdf", schema)
        assert result.fact_count == 0
        assert result.error is None  # Graceful fallback, not an error


# --- Batch Extraction Tests ---


class TestBatchExtraction:
    @pytest.mark.asyncio
    async def test_batch_extract(self):
        adapter = LangExtractAdapter(extractor=_make_extractor())
        schema = _make_schema()

        results = await adapter.batch_extract(
            ["/tmp/a.pdf", "/tmp/b.pdf"],
            schema,
        )

        assert len(results) == 2
        assert all(r.fact_count == 2 for r in results)


# --- Search Tests ---


class TestFactSearch:
    @pytest.mark.asyncio
    async def test_search_by_keyword(self):
        adapter = LangExtractAdapter(extractor=_make_extractor())
        schema = _make_schema()
        await adapter.extract_from_document("/tmp/doc.pdf", schema)

        results = await adapter.search_facts("Contract A")
        assert len(results) == 1
        assert results[0].content["title"] == "Contract A"

    @pytest.mark.asyncio
    async def test_search_with_confidence_filter(self):
        adapter = LangExtractAdapter(extractor=_make_extractor())
        schema = _make_schema()
        await adapter.extract_from_document("/tmp/doc.pdf", schema)

        # Both facts have confidence >= 0.85
        results = await adapter.search_facts("Contract", min_confidence=0.9)
        assert len(results) == 1  # Only the 0.92 one

    @pytest.mark.asyncio
    async def test_search_with_schema_filter(self):
        adapter = LangExtractAdapter(extractor=_make_extractor())
        await adapter.extract_from_document("/tmp/doc.pdf", _make_schema("schema_a"))

        results = await adapter.search_facts("Contract", schema_filter="schema_b")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_with_limit(self):
        adapter = LangExtractAdapter(extractor=_make_extractor())
        schema = _make_schema()
        await adapter.extract_from_document("/tmp/doc.pdf", schema)

        results = await adapter.search_facts("Contract", limit=1)
        assert len(results) == 1


# --- Validation Sync Tests ---


class TestValidationSync:
    @pytest.mark.asyncio
    async def test_apply_validations(self):
        adapter = LangExtractAdapter(extractor=_make_extractor())
        schema = _make_schema()
        result = await adapter.extract_from_document("/tmp/doc.pdf", schema)

        fact_id = result.facts[0].fact_id
        original_confidence = result.facts[0].confidence

        sync_result = await adapter.sync_validations_from_km([
            {"fact_id": fact_id, "confidence_adjustment": -0.1, "reason": "contradicted"},
        ])

        assert sync_result["applied"] == 1
        assert sync_result["skipped"] == 0

        updated_fact = adapter.get_fact(fact_id)
        assert updated_fact is not None
        assert updated_fact.confidence == pytest.approx(original_confidence - 0.1)

    @pytest.mark.asyncio
    async def test_validation_clamps_confidence(self):
        adapter = LangExtractAdapter(extractor=_make_extractor())
        schema = _make_schema()
        result = await adapter.extract_from_document("/tmp/doc.pdf", schema)

        fact_id = result.facts[0].fact_id

        await adapter.sync_validations_from_km([
            {"fact_id": fact_id, "confidence_adjustment": -5.0, "reason": "wrong"},
        ])

        fact = adapter.get_fact(fact_id)
        assert fact is not None
        assert fact.confidence == 0.0  # Clamped at 0

    @pytest.mark.asyncio
    async def test_validation_skips_unknown_facts(self):
        adapter = LangExtractAdapter()

        sync_result = await adapter.sync_validations_from_km([
            {"fact_id": "nonexistent", "confidence_adjustment": 0.1},
        ])

        assert sync_result["applied"] == 0
        assert sync_result["skipped"] == 1

    @pytest.mark.asyncio
    async def test_validation_log(self):
        adapter = LangExtractAdapter(extractor=_make_extractor())
        schema = _make_schema()
        result = await adapter.extract_from_document("/tmp/doc.pdf", schema)

        fact_id = result.facts[0].fact_id
        await adapter.sync_validations_from_km([
            {"fact_id": fact_id, "confidence_adjustment": 0.05, "reason": "confirmed"},
        ])

        log = adapter.get_validation_log()
        assert len(log) == 1
        assert log[0]["fact_id"] == fact_id
        assert log[0]["reason"] == "confirmed"
        assert log[0]["adjustment"] == 0.05


# --- Sync to KM Tests ---


class TestSyncToKM:
    @pytest.mark.asyncio
    async def test_sync_to_km(self):
        adapter = LangExtractAdapter(extractor=_make_extractor())
        schema = _make_schema()
        await adapter.extract_from_document("/tmp/doc.pdf", schema)

        result = await adapter.sync_to_km(workspace_id="ws-123")

        assert result["facts_synced"] == 2
        assert result["workspace_id"] == "ws-123"


# --- Utility Tests ---


class TestUtilities:
    def test_generate_fact_id_deterministic(self):
        id1 = LangExtractAdapter._generate_fact_id("/tmp/doc.pdf", "schema", 0)
        id2 = LangExtractAdapter._generate_fact_id("/tmp/doc.pdf", "schema", 0)
        assert id1 == id2
        assert id1.startswith("le_")

    def test_generate_fact_id_unique_per_index(self):
        id1 = LangExtractAdapter._generate_fact_id("/tmp/doc.pdf", "schema", 0)
        id2 = LangExtractAdapter._generate_fact_id("/tmp/doc.pdf", "schema", 1)
        assert id1 != id2

    def test_cache_key(self):
        key = LangExtractAdapter._cache_key("/tmp/doc.pdf", "schema_a")
        assert key == "/tmp/doc.pdf::schema_a"

    def test_clear_cache(self):
        adapter = LangExtractAdapter()
        adapter._extraction_cache["test"] = ExtractionResult("/tmp/doc.pdf", "s", [])
        adapter.clear_cache()
        assert len(adapter._extraction_cache) == 0

    def test_get_metrics_after_extraction(self):
        adapter = LangExtractAdapter()
        adapter.register_schema(_make_schema())
        adapter._fact_store["a"] = ExtractedFact("a", {}, "", "", 0.5, "test")

        metrics = adapter.get_metrics()
        assert metrics["total_facts"] == 1
        assert metrics["total_schemas"] == 1


# --- Factory Registration Tests ---


class TestFactoryRegistration:
    def test_adapter_registered_in_factory(self):
        from aragora.knowledge.mound.adapters.factory import ADAPTER_SPECS

        assert "langextract" in ADAPTER_SPECS
        spec = ADAPTER_SPECS["langextract"]
        assert spec.name == "langextract"
        assert spec.enabled_by_default is False
        assert spec.forward_method == "sync_to_km"
        assert spec.reverse_method == "sync_validations_from_km"
        assert spec.priority == 72

    def test_adapter_name_unique(self):
        """Verify adapter_name is unique across all registered adapters."""
        from aragora.knowledge.mound.adapters.factory import ADAPTER_SPECS

        names = list(ADAPTER_SPECS.keys())
        assert len(names) == len(set(names))
        assert "langextract" in names
