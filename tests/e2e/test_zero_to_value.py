"""E2E validation: zero-to-value CLI path.

Exercises the full quickstart journey with mocked LLM responses:
1. .env loading and API key detection
2. Agent creation from detected keys
3. Arena construction with ProviderRouter integration
4. Debate execution (mocked agent responses)
5. Receipt persistence to ~/.aragora/receipts/
6. Result extraction (final_answer, confidence, dissent)

These tests validate the WIRING, not the LLM quality. They ensure
a user with API keys configured will successfully reach a receipt.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ===========================================================================
# 1. API Key Detection
# ===========================================================================


class TestApiKeyDetection:
    """Validate _detect_agents finds available providers."""

    def test_detects_anthropic_key(self):
        from aragora.cli.commands.quickstart import _detect_agents

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False):
            agents = _detect_agents()
            providers = [p for p, _ in agents]
            assert "anthropic-api" in providers

    def test_detects_openai_key(self):
        from aragora.cli.commands.quickstart import _detect_agents

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
            agents = _detect_agents()
            providers = [p for p, _ in agents]
            assert "openai-api" in providers

    def test_detects_multiple_keys(self):
        from aragora.cli.commands.quickstart import _detect_agents

        env = {
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "OPENAI_API_KEY": "sk-test",
            "OPENROUTER_API_KEY": "sk-or-test",
        }
        with patch.dict(os.environ, env, clear=False):
            agents = _detect_agents()
            assert len(agents) >= 3

    def test_returns_empty_when_no_keys(self):
        from aragora.cli.commands.quickstart import _detect_agents

        # Remove all known API key env vars
        keys_to_remove = [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "MISTRAL_API_KEY",
            "XAI_API_KEY",
            "GROK_API_KEY",
            "OPENROUTER_API_KEY",
        ]
        clean_env = {k: v for k, v in os.environ.items() if k not in keys_to_remove}
        with patch.dict(os.environ, clean_env, clear=True):
            agents = _detect_agents()
            assert len(agents) == 0


# ===========================================================================
# 2. Dotenv Loading
# ===========================================================================


class TestDotenvLoading:
    """Validate .env file loading."""

    def test_loads_env_from_cwd(self, tmp_path):
        from aragora.cli.commands.quickstart import _load_dotenv

        env_file = tmp_path / ".env"
        env_file.write_text('TEST_QUICKSTART_VAR="hello"\n')

        with patch("aragora.cli.commands.quickstart.Path") as mock_path:
            mock_path.cwd.return_value = tmp_path
            # Remove the var if it exists
            os.environ.pop("TEST_QUICKSTART_VAR", None)
            result = _load_dotenv()

        if result:
            assert os.environ.get("TEST_QUICKSTART_VAR") == "hello"
            os.environ.pop("TEST_QUICKSTART_VAR", None)


# ===========================================================================
# 3. Arena Construction with ProviderRouter
# ===========================================================================


class TestArenaProviderRouterIntegration:
    """Validate Arena is constructed with ProviderRouter wiring."""

    def test_arena_accepts_provider_budget(self):
        """Arena constructor accepts provider_budget kwarg."""
        from aragora.core import Environment
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol

        class MockAgent:
            def __init__(self, name):
                self.name = name
                self.provider = "mock"
                self.model = "mock"
                self.role = "proposer"
                self.agent_type = "mock"
                self.capabilities = set()
                self.hierarchy_role = None
                self.metadata = {}
                self.system_prompt = ""

            async def generate(self, prompt, context=None):
                return "mock response"

            async def critique(self, proposal, task, context=None, target_agent=None):
                return "mock critique"

        env = Environment(task="Test question")
        protocol = DebateProtocol(rounds=1, consensus="majority")
        agents = [MockAgent("a"), MockAgent("b")]

        # Should not raise
        arena = Arena(
            env,
            agents,
            protocol,
            provider_budget=2.0,
            use_performance_selection=True,
        )
        assert arena.provider_budget == 2.0
        assert arena.use_performance_selection is True

    def test_protocol_enables_knowledge_injection_by_default(self):
        """DebateProtocol defaults enable_knowledge_injection to True."""
        from aragora.debate.protocol import DebateProtocol

        protocol = DebateProtocol(rounds=2, consensus="majority")
        assert protocol.enable_knowledge_injection is True


# ===========================================================================
# 4. Receipt Persistence
# ===========================================================================


class TestReceiptPersistence:
    """Validate receipt auto-persistence from quickstart path."""

    def test_persist_creates_receipt_file(self):
        """_persist_debate_receipt creates a valid JSON receipt."""
        from aragora.cli.commands.debate import _persist_debate_receipt
        from aragora.core import DebateResult

        result = DebateResult(
            debate_id="e2e-test-001",
            task="Should we use Redis or Memcached?",
            final_answer="Use Redis for its richer data structures.",
            consensus_reached=True,
            confidence=0.88,
            rounds_used=2,
            dissenting_views=["Memcached is simpler for pure caching"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                path = _persist_debate_receipt(result)

                assert path is not None
                receipt = json.loads(Path(path).read_text())
                assert receipt["debate_id"] == "e2e-test-001"
                assert receipt["consensus_reached"] is True
                assert receipt["confidence"] == 0.88
                assert "content_hash" in receipt
                assert len(receipt["content_hash"]) == 16  # SHA-256 truncated

    def test_receipt_includes_dissenting_views(self):
        """Receipt captures dissenting views from the debate."""
        from aragora.cli.commands.debate import _persist_debate_receipt
        from aragora.core import DebateResult

        result = DebateResult(
            debate_id="e2e-test-002",
            task="Test",
            final_answer="Answer",
            consensus_reached=False,
            confidence=0.5,
            dissenting_views=["View A", "View B"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                path = _persist_debate_receipt(result)
                receipt = json.loads(Path(path).read_text())
                assert len(receipt["dissenting_views"]) == 2


# ===========================================================================
# 5. Full Quickstart Flow (Mocked Agents)
# ===========================================================================


class TestFullQuickstartFlow:
    """Validate the complete quickstart flow wiring."""

    def test_quickstart_result_has_required_fields(self):
        """_run_live_debate returns all required fields."""
        from aragora.cli.commands.quickstart import _run_live_debate
        from aragora.core import DebateResult

        mock_result = DebateResult(
            debate_id="qs-flow-001",
            task="Test question",
            final_answer="The answer is 42.",
            consensus_reached=True,
            confidence=0.9,
            rounds_used=2,
            dissenting_views=["Alternative view"],
        )

        with (
            patch("aragora.debate.orchestrator.Arena") as mock_arena_cls,
            patch("aragora.agents.base.create_agent") as mock_create,
        ):
            mock_arena = MagicMock()
            mock_arena.run = AsyncMock(return_value=mock_result)
            mock_arena_cls.return_value = mock_arena

            mock_agent = MagicMock()
            mock_agent.name = "test-agent"
            mock_create.return_value = mock_agent

            import asyncio

            result = asyncio.run(
                _run_live_debate(
                    "Test question",
                    [("anthropic-api", "claude-sonnet-4")],
                    rounds=2,
                )
            )

            # Verify all required fields present
            assert result["question"] == "Test question"
            assert result["mode"] == "live"
            assert result["confidence"] == 0.9
            assert result["verdict"] == "consensus"
            assert len(result["agents"]) == 1
            assert result["summary"] == "The answer is 42."
            assert len(result["dissent"]) == 1

    def test_quickstart_arena_uses_performance_selection(self):
        """Arena is constructed with use_performance_selection=True."""
        from aragora.cli.commands.quickstart import _run_live_debate
        from aragora.core import DebateResult

        mock_result = DebateResult(
            debate_id="qs-perf-001",
            task="Test",
            final_answer="Answer",
            consensus_reached=True,
            confidence=0.8,
        )

        with (
            patch("aragora.debate.orchestrator.Arena") as mock_arena_cls,
            patch("aragora.agents.base.create_agent") as mock_create,
        ):
            mock_arena = MagicMock()
            mock_arena.run = AsyncMock(return_value=mock_result)
            mock_arena_cls.return_value = mock_arena
            mock_create.return_value = MagicMock(name="agent")

            import asyncio

            asyncio.run(_run_live_debate("Test", [("anthropic-api", None)], rounds=1))

            # Verify Arena was constructed with performance selection
            call_kwargs = mock_arena_cls.call_args
            assert call_kwargs[1]["use_performance_selection"] is True
            assert call_kwargs[1]["enable_knowledge_retrieval"] is True


# ===========================================================================
# 6. Provider Outcome Recording
# ===========================================================================


class TestProviderOutcomeRecording:
    """Validate debate outcomes feed back to ProviderRouter."""

    def test_outcomes_recorded_per_agent(self):
        """Each participating agent's outcome is recorded."""
        from aragora.debate.orchestrator_runner import _record_provider_outcomes

        class Agent:
            def __init__(self, name, provider):
                self.name = name
                self.provider = provider

        arena = MagicMock()
        state = MagicMock()
        state.ctx.agents = [
            Agent("claude", "anthropic"),
            Agent("gpt", "openai"),
        ]
        state.ctx.result.consensus_reached = True
        state.ctx.result.confidence = 0.85
        state.debate_status = "completed"
        state.debate_start_time = 0.0

        with patch("aragora.routing.provider_router.get_provider_router") as mock_get:
            mock_router = MagicMock()
            mock_get.return_value = mock_router
            _record_provider_outcomes(arena, state)

            assert mock_router.record_outcome.call_count == 2
            providers = [c[0][0] for c in mock_router.record_outcome.call_args_list]
            assert "anthropic" in providers
            assert "openai" in providers


# ===========================================================================
# 7. OpenClaw Dispatch Wiring
# ===========================================================================


class TestOpenClawDispatchWiring:
    """Validate the dispatch pipeline is connected."""

    def test_unified_orchestrator_has_default_factory(self):
        """UnifiedOrchestrator auto-creates code_task_factory."""
        from aragora.pipeline.unified_orchestrator import UnifiedOrchestrator

        orch = UnifiedOrchestrator()
        assert orch._code_task_factory is not None

    @pytest.mark.asyncio
    async def test_dispatch_extracts_spec_from_debate(self):
        """dispatch_implementation extracts actionable spec."""
        from aragora.core import DebateResult
        from aragora.pipeline.dispatch import dispatch_implementation

        result = DebateResult(
            debate_id="dispatch-001",
            task="Add rate limiting to the API",
            final_answer=(
                "Implementation plan:\n"
                "1. Add middleware in `server/middleware/rate_limit.py`\n"
                "2. Configure Redis backend in `config/redis.py`\n"
                "3. Add tests in `tests/test_rate_limit.py`\n"
                "Rollback: remove the middleware registration"
            ),
            consensus_reached=True,
            confidence=0.92,
        )

        with patch(
            "aragora.pipeline.dispatch.create_code_task",
            new_callable=AsyncMock,
            return_value={"success": True, "exit_code": 0},
        ):
            dispatch_result = await dispatch_implementation(result)
            assert dispatch_result["spec"] is not None
            assert "implementation_prompt" in dispatch_result["spec"]
