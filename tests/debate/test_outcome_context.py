"""Tests for outcome context injection into debate prompts.

Verifies that past decision outcomes (successes/failures) from the
OutcomeAdapter are properly injected into debate context when
``enable_outcome_context`` is True.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Patch target for OutcomeAdapter - must patch at the source module
_ADAPTER_MODULE = "aragora.knowledge.mound.adapters.outcome_adapter"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeKnowledgeItem:
    """Minimal KnowledgeItem stand-in for tests."""

    id: str = "outc_abc123"
    content: str = "[Outcome:success] Vendor selection went well"
    metadata: dict = field(default_factory=lambda: {
        "outcome_type": "success",
        "impact_score": 0.85,
        "lessons_learned": "Always compare at least 3 vendors",
        "kpi_deltas": {"cost_reduction": -0.15, "satisfaction": 0.2},
        "tags": ["decision_outcome", "type:success"],
    })


@dataclass
class FakeEnv:
    task: str = "Choose a cloud vendor for our infrastructure"
    context: str = ""


@dataclass
class FakeDebateContext:
    env: FakeEnv = field(default_factory=FakeEnv)
    _prompt_builder: MagicMock | None = None


def _run(coro):
    """Run a coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_mock_adapter(outcomes=None):
    """Create a mock OutcomeAdapter class that returns the given outcomes."""
    mock_cls = MagicMock()
    mock_instance = MagicMock()
    mock_instance.find_similar_outcomes = AsyncMock(
        return_value=outcomes if outcomes is not None else []
    )
    mock_cls.return_value = mock_instance
    return mock_cls


# ---------------------------------------------------------------------------
# PromptBuilder outcome context tests
# ---------------------------------------------------------------------------

class TestPromptBuilderOutcomeContext:
    """Tests for get/set_outcome_context on PromptBuilder/PromptContextMixin."""

    def _make_builder(self):
        from aragora.debate.prompt_builder import PromptBuilder

        protocol = MagicMock()
        protocol.rounds = 3
        protocol.agreement_intensity = None
        protocol.asymmetric_stances = False
        protocol.enable_trending_injection = False
        protocol.deliberation_template = None
        env = MagicMock()
        env.task = "Test task"
        env.context = ""
        return PromptBuilder(protocol=protocol, env=env)

    def test_default_empty(self):
        builder = self._make_builder()
        assert builder.get_outcome_context() == ""

    def test_set_and_get(self):
        builder = self._make_builder()
        builder.set_outcome_context("## PAST DECISION OUTCOMES\nSome outcomes")
        assert "PAST DECISION OUTCOMES" in builder.get_outcome_context()

    def test_set_none_yields_empty(self):
        builder = self._make_builder()
        builder.set_outcome_context(None)
        assert builder.get_outcome_context() == ""

    def test_set_empty_string(self):
        builder = self._make_builder()
        builder.set_outcome_context("content")
        builder.set_outcome_context("")
        assert builder.get_outcome_context() == ""


# ---------------------------------------------------------------------------
# ContextInitializer outcome injection tests
# ---------------------------------------------------------------------------

class TestContextInitializerOutcomeInjection:
    """Tests for _inject_outcome_context in ContextInitializer."""

    def _make_initializer(self, knowledge_mound=None, enable=True):
        from aragora.debate.phases.context_init import ContextInitializer

        return ContextInitializer(
            knowledge_mound=knowledge_mound,
            enable_outcome_context=enable,
        )

    def test_flag_default_true(self):
        init = self._make_initializer()
        assert init.enable_outcome_context is True

    def test_flag_can_disable(self):
        init = self._make_initializer(enable=False)
        assert init.enable_outcome_context is False

    def test_skips_when_no_knowledge_mound(self):
        """No knowledge mound -> no outcome injection."""
        init = self._make_initializer(knowledge_mound=None)
        ctx = FakeDebateContext()
        _run(init._inject_outcome_context(ctx))
        assert "PAST DECISION OUTCOMES" not in ctx.env.context

    def test_skips_when_no_outcomes_found(self):
        """No similar outcomes -> no context added."""
        mound = MagicMock()
        mock_cls = _make_mock_adapter(outcomes=[])

        with patch(f"{_ADAPTER_MODULE}.OutcomeAdapter", mock_cls):
            init = self._make_initializer(knowledge_mound=mound)
            ctx = FakeDebateContext()
            _run(init._inject_outcome_context(ctx))
            assert "PAST DECISION OUTCOMES" not in ctx.env.context

    def test_injects_outcome_into_env_context(self):
        """When outcomes found and no prompt builder, injects into env.context."""
        mound = MagicMock()
        items = [FakeKnowledgeItem()]
        mock_cls = _make_mock_adapter(outcomes=items)

        with patch(f"{_ADAPTER_MODULE}.OutcomeAdapter", mock_cls):
            init = self._make_initializer(knowledge_mound=mound)
            ctx = FakeDebateContext()
            _run(init._inject_outcome_context(ctx))

            assert "PAST DECISION OUTCOMES" in ctx.env.context
            assert "SUCCESS" in ctx.env.context
            assert "85%" in ctx.env.context

    def test_injects_outcome_onto_prompt_builder(self):
        """When prompt builder is available, uses set_outcome_context."""
        mound = MagicMock()
        items = [FakeKnowledgeItem()]
        mock_cls = _make_mock_adapter(outcomes=items)

        with patch(f"{_ADAPTER_MODULE}.OutcomeAdapter", mock_cls):
            init = self._make_initializer(knowledge_mound=mound)
            builder = MagicMock()
            builder.set_outcome_context = MagicMock()
            ctx = FakeDebateContext(_prompt_builder=builder)
            _run(init._inject_outcome_context(ctx))

            builder.set_outcome_context.assert_called_once()
            call_text = builder.set_outcome_context.call_args[0][0]
            assert "PAST DECISION OUTCOMES" in call_text

    def test_includes_lessons_learned(self):
        """Lessons learned from outcomes are included in the context."""
        mound = MagicMock()
        items = [FakeKnowledgeItem()]
        mock_cls = _make_mock_adapter(outcomes=items)

        with patch(f"{_ADAPTER_MODULE}.OutcomeAdapter", mock_cls):
            init = self._make_initializer(knowledge_mound=mound)
            ctx = FakeDebateContext()
            _run(init._inject_outcome_context(ctx))

            assert "Always compare at least 3 vendors" in ctx.env.context

    def test_includes_kpi_deltas(self):
        """KPI delta information from outcomes is included."""
        mound = MagicMock()
        items = [FakeKnowledgeItem()]
        mock_cls = _make_mock_adapter(outcomes=items)

        with patch(f"{_ADAPTER_MODULE}.OutcomeAdapter", mock_cls):
            init = self._make_initializer(knowledge_mound=mound)
            ctx = FakeDebateContext()
            _run(init._inject_outcome_context(ctx))

            assert "cost_reduction" in ctx.env.context
            assert "KPI changes" in ctx.env.context

    def test_graceful_on_import_error(self):
        """When OutcomeAdapter is not importable, degrades gracefully."""
        mound = MagicMock()
        init = self._make_initializer(knowledge_mound=mound)
        ctx = FakeDebateContext()

        with patch.dict("sys.modules", {
            _ADAPTER_MODULE: None,
        }):
            # Should not raise
            _run(init._inject_outcome_context(ctx))
            assert "PAST DECISION OUTCOMES" not in ctx.env.context

    def test_graceful_on_timeout(self):
        """Timeout during outcome query doesn't break the debate."""
        mound = MagicMock()
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.find_similar_outcomes = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )
        mock_cls.return_value = mock_instance

        with patch(f"{_ADAPTER_MODULE}.OutcomeAdapter", mock_cls):
            init = self._make_initializer(knowledge_mound=mound)
            ctx = FakeDebateContext()
            # Should not raise
            _run(init._inject_outcome_context(ctx))
            assert "PAST DECISION OUTCOMES" not in ctx.env.context

    def test_appends_to_existing_env_context(self):
        """When env.context already has content, outcome is appended."""
        mound = MagicMock()
        items = [FakeKnowledgeItem()]
        mock_cls = _make_mock_adapter(outcomes=items)

        with patch(f"{_ADAPTER_MODULE}.OutcomeAdapter", mock_cls):
            init = self._make_initializer(knowledge_mound=mound)
            ctx = FakeDebateContext()
            ctx.env.context = "Existing context here"
            _run(init._inject_outcome_context(ctx))

            assert ctx.env.context.startswith("Existing context here")
            assert "PAST DECISION OUTCOMES" in ctx.env.context

    def test_empty_task_skips_injection(self):
        """Empty task string -> skip outcome injection."""
        mound = MagicMock()
        init = self._make_initializer(knowledge_mound=mound)
        ctx = FakeDebateContext()
        ctx.env.task = ""

        with patch(f"{_ADAPTER_MODULE}.OutcomeAdapter", _make_mock_adapter()):
            _run(init._inject_outcome_context(ctx))
            assert "PAST DECISION OUTCOMES" not in ctx.env.context


# ---------------------------------------------------------------------------
# Config wiring tests
# ---------------------------------------------------------------------------

class TestOutcomeContextConfig:
    """Tests for enable_outcome_context config flag wiring."""

    def test_knowledge_mound_sub_config_default(self):
        from aragora.debate.arena_sub_configs import KnowledgeMoundConfig

        cfg = KnowledgeMoundConfig()
        assert cfg.enable_outcome_context is True

    def test_knowledge_config_default(self):
        from aragora.debate.arena_primary_configs import KnowledgeConfig

        cfg = KnowledgeConfig()
        assert cfg.enable_outcome_context is True

    def test_knowledge_config_can_disable(self):
        from aragora.debate.arena_primary_configs import KnowledgeConfig

        cfg = KnowledgeConfig(enable_outcome_context=False)
        assert cfg.enable_outcome_context is False


# ---------------------------------------------------------------------------
# Prompt assembler integration tests
# ---------------------------------------------------------------------------

class TestPromptAssemblerOutcomeSection:
    """Tests verifying outcome context appears in assembled prompts."""

    def _make_builder_with_outcome(self, outcome_text=""):
        from aragora.debate.prompt_builder import PromptBuilder

        protocol = MagicMock()
        protocol.rounds = 3
        protocol.agreement_intensity = None
        protocol.asymmetric_stances = False
        protocol.enable_trending_injection = False
        protocol.deliberation_template = None
        protocol.enable_privacy_anonymization = False
        env = MagicMock()
        env.task = "Test task"
        env.context = ""
        builder = PromptBuilder(protocol=protocol, env=env)
        if outcome_text:
            builder.set_outcome_context(outcome_text)
        return builder

    def test_proposal_prompt_includes_outcome(self):
        outcome = "## PAST DECISION OUTCOMES\n- [SUCCESS, 85% impact] Good vendor choice"
        builder = self._make_builder_with_outcome(outcome)
        agent = MagicMock()
        agent.name = "claude"
        agent.role = "proposer"
        prompt = builder.build_proposal_prompt(agent)
        assert "PAST DECISION OUTCOMES" in prompt
        assert "SUCCESS" in prompt

    def test_proposal_prompt_omits_when_empty(self):
        builder = self._make_builder_with_outcome("")
        agent = MagicMock()
        agent.name = "claude"
        agent.role = "proposer"
        prompt = builder.build_proposal_prompt(agent)
        assert "PAST DECISION OUTCOMES" not in prompt

    def test_revision_prompt_includes_outcome(self):
        from aragora.core import Critique

        outcome = "## PAST DECISION OUTCOMES\n- [FAILURE, 30% impact] Bad migration"
        builder = self._make_builder_with_outcome(outcome)
        agent = MagicMock()
        agent.name = "claude"
        agent.role = "proposer"
        critique = MagicMock(spec=Critique)
        critique.to_prompt.return_value = "Some critique"
        prompt = builder.build_revision_prompt(
            agent=agent,
            original="Original proposal",
            critiques=[critique],
        )
        assert "PAST DECISION OUTCOMES" in prompt
        assert "FAILURE" in prompt
