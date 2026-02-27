"""Tests for the interrogation engine facade."""

import pytest
from unittest.mock import AsyncMock
from aragora.interrogation.engine import (
    InterrogationEngine,
    InterrogationResult,
    InterrogationState,
)


class TestInterrogationEngine:
    @pytest.fixture
    def engine(self):
        return InterrogationEngine()

    @pytest.mark.asyncio
    async def test_start_returns_interrogation_state(self, engine):
        result = await engine.start("Make aragora more powerful")
        assert isinstance(result, InterrogationState)
        assert result.prompt == "Make aragora more powerful"
        assert len(result.dimensions) >= 1
        assert len(result.questions) >= 0

    @pytest.mark.asyncio
    async def test_answer_updates_state(self, engine):
        state = await engine.start("Improve test coverage")
        if state.questions:
            q = state.questions[0]
            updated = engine.answer(state, q.text, "Unit tests for new modules")
            assert q.text in updated.answers

    @pytest.mark.asyncio
    async def test_crystallize_produces_spec(self, engine):
        state = await engine.start("Add dark mode")
        result = await engine.crystallize(state)
        assert isinstance(result, InterrogationResult)
        assert result.spec.problem_statement

    @pytest.mark.asyncio
    async def test_full_flow(self, engine):
        state = await engine.start("Improve performance")
        for q in state.questions:
            state = engine.answer(state, q.text, "Yes, do it")
        result = await engine.crystallize(state)
        assert result.spec.requirements

    @pytest.mark.asyncio
    async def test_start_with_km(self):
        mock_km = AsyncMock()
        mock_km.query.return_value = AsyncMock(items=[])
        engine = InterrogationEngine(knowledge_mound=mock_km)
        state = await engine.start("Test", sources=["knowledge_mound"])
        assert isinstance(state, InterrogationState)
