"""Tests for Input Extension Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest
from aragora.pipeline.input_extension import (
    Constraint,
    ExtendedInput,
    Implication,
    InputExtensionEngine,
    PriorArt,
)


class TestImplication:
    def test_basic(self):
        imp = Implication(statement="Users expect pagination", confidence=0.8)
        assert imp.confidence == 0.8
        assert imp.category == "general"


class TestConstraint:
    def test_defaults(self):
        con = Constraint(description="Use HTTPS", reason="Security")
        assert con.severity == "recommended"


class TestPriorArt:
    def test_basic(self):
        pa = PriorArt(
            title="Rate Limiter Design", description="Token bucket approach", relevance=0.9
        )
        assert pa.relevance == 0.9


class TestExtendedInput:
    def test_empty(self):
        ext = ExtendedInput(original_prompt="build a widget")
        assert not ext.has_extensions
        assert ext.to_context_block() == ""

    def test_with_extensions(self):
        ext = ExtendedInput(
            original_prompt="build a widget",
            implications=[Implication(statement="Needs error handling", confidence=0.8)],
            constraints=[
                Constraint(
                    description="Use TypeScript", reason="Team standard", severity="required"
                )
            ],
        )
        assert ext.has_extensions

    def test_high_confidence_filter(self):
        ext = ExtendedInput(
            original_prompt="test",
            implications=[
                Implication(statement="high", confidence=0.9),
                Implication(statement="low", confidence=0.3),
                Implication(statement="medium", confidence=0.7),
            ],
        )
        high = ext.high_confidence_implications
        assert len(high) == 2

    def test_required_constraints(self):
        ext = ExtendedInput(
            original_prompt="test",
            constraints=[
                Constraint(description="must", reason="r", severity="required"),
                Constraint(description="should", reason="r", severity="recommended"),
            ],
        )
        assert len(ext.required_constraints) == 1

    def test_context_block_formatting(self):
        ext = ExtendedInput(
            original_prompt="test",
            implications=[Implication(statement="imp1", confidence=0.8, category="technical")],
            constraints=[Constraint(description="con1", reason="reason1", severity="required")],
            prior_art=[PriorArt(title="PA1", description="desc1", relevance=0.9)],
            risk_factors=["Data loss risk"],
        )
        block = ext.to_context_block()
        assert "## Implied Requirements" in block
        assert "## Recommended Constraints" in block
        assert "## Prior Art" in block
        assert "## Risk Factors" in block


class TestInputExtensionEngine:
    @pytest.mark.asyncio
    async def test_basic_extension(self):
        engine = InputExtensionEngine()
        result = await engine.extend("build a login page", domain="security")
        assert isinstance(result, ExtendedInput)
        assert result.original_prompt == "build a login page"
        # Security domain should add constraints
        assert len(result.constraints) > 0

    @pytest.mark.asyncio
    async def test_domain_constraints_security(self):
        engine = InputExtensionEngine()
        result = await engine.extend("anything", domain="security")
        descriptions = [c.description for c in result.constraints]
        assert any("validation" in d.lower() for d in descriptions)

    @pytest.mark.asyncio
    async def test_domain_constraints_healthcare(self):
        engine = InputExtensionEngine()
        result = await engine.extend("patient records", domain="healthcare")
        assert any("HIPAA" in c.description for c in result.constraints)

    @pytest.mark.asyncio
    async def test_risk_detection(self):
        engine = InputExtensionEngine()
        result = await engine.extend("migrate the database and delete old records")
        assert len(result.risk_factors) >= 2

    @pytest.mark.asyncio
    async def test_no_domain_no_constraints(self):
        engine = InputExtensionEngine()
        result = await engine.extend("hello world")
        assert len(result.constraints) == 0

    @pytest.mark.asyncio
    async def test_km_prior_art(self):
        km = MagicMock()
        km.query.return_value = [
            {"title": "Previous API Design", "content": "RESTful approach", "relevance": 0.8}
        ]
        engine = InputExtensionEngine(knowledge_mound=km)
        result = await engine.extend("design an API")
        assert len(result.prior_art) == 1
        assert result.prior_art[0].source_type == "internal"

    @pytest.mark.asyncio
    async def test_research_context_implications(self):
        @dataclass
        class MockResult:
            content: str = "Relevant finding"
            relevance: float = 0.8
            source: str = "km"

        @dataclass
        class MockContext:
            results: list = field(default_factory=list)

        ctx = MockContext(results=[MockResult()])
        engine = InputExtensionEngine()
        result = await engine.extend("test", research_context=ctx)
        assert len(result.implications) == 1

    @pytest.mark.asyncio
    async def test_km_failure_graceful(self):
        km = MagicMock()
        km.query.side_effect = RuntimeError("fail")
        engine = InputExtensionEngine(knowledge_mound=km)
        result = await engine.extend("test")
        assert len(result.prior_art) == 0
