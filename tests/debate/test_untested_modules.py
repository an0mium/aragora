"""Tests for previously untested debate modules.

Covers: sanitization, safety, optional_imports, orchestrator_factory,
lifecycle_manager, context_compressor, feedback_persona,
compliance_artifact_hook, embeddings, traces_database.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# OutputSanitizer (sanitization.py)
# ---------------------------------------------------------------------------


class TestOutputSanitizer:
    def test_sanitize_removes_null_bytes(self):
        from aragora.debate.sanitization import OutputSanitizer

        result = OutputSanitizer.sanitize_agent_output("hello\x00world", "agent1")
        assert result == "hello world" or result == "helloworld"
        assert "\x00" not in result

    def test_sanitize_removes_control_chars(self):
        from aragora.debate.sanitization import OutputSanitizer

        result = OutputSanitizer.sanitize_agent_output("hello\x01\x02\x03world", "a")
        assert "\x01" not in result
        assert "\x02" not in result
        assert "\x03" not in result

    def test_sanitize_preserves_newlines_and_tabs(self):
        from aragora.debate.sanitization import OutputSanitizer

        result = OutputSanitizer.sanitize_agent_output("line1\nline2\ttab", "a")
        assert "\n" in result
        assert "\t" in result

    def test_sanitize_returns_placeholder_for_empty(self):
        from aragora.debate.sanitization import OutputSanitizer

        result = OutputSanitizer.sanitize_agent_output("\x00\x01", "a")
        assert "empty" in result.lower() or "error" in result.lower()

    def test_sanitize_returns_placeholder_for_non_string(self):
        from aragora.debate.sanitization import OutputSanitizer

        result = OutputSanitizer.sanitize_agent_output(12345, "a")  # type: ignore[arg-type]
        assert "error" in result.lower() or "type" in result.lower()

    def test_sanitize_normal_text_unchanged(self):
        from aragora.debate.sanitization import OutputSanitizer

        result = OutputSanitizer.sanitize_agent_output("Normal output text.", "a")
        assert result == "Normal output text."

    def test_sanitize_prompt_removes_null_bytes(self):
        from aragora.debate.sanitization import OutputSanitizer

        result = OutputSanitizer.sanitize_prompt("hello\x00world")
        assert "\x00" not in result

    def test_sanitize_prompt_preserves_newlines(self):
        from aragora.debate.sanitization import OutputSanitizer

        result = OutputSanitizer.sanitize_prompt("line1\nline2")
        assert "\n" in result

    def test_sanitize_prompt_empty_string(self):
        from aragora.debate.sanitization import OutputSanitizer

        result = OutputSanitizer.sanitize_prompt("")
        assert result == ""

    def test_sanitize_prompt_non_string(self):
        from aragora.debate.sanitization import OutputSanitizer

        result = OutputSanitizer.sanitize_prompt(42)  # type: ignore[arg-type]
        assert result == "42"


# ---------------------------------------------------------------------------
# Safety gates (safety.py)
# ---------------------------------------------------------------------------


class TestSafetyGates:
    def test_resolve_auto_evolve_false_when_not_requested(self):
        from aragora.debate.safety import resolve_auto_evolve

        assert resolve_auto_evolve(False) is False

    def test_resolve_auto_evolve_false_without_env(self):
        from aragora.debate.safety import resolve_auto_evolve

        with patch.dict("os.environ", {}, clear=True):
            assert resolve_auto_evolve(True) is False

    def test_resolve_auto_evolve_true_with_env(self):
        from aragora.debate.safety import resolve_auto_evolve

        with patch.dict("os.environ", {"ARAGORA_ALLOW_AUTO_EVOLVE": "1"}):
            assert resolve_auto_evolve(True) is True

    def test_resolve_prompt_evolution_false_when_not_requested(self):
        from aragora.debate.safety import resolve_prompt_evolution

        assert resolve_prompt_evolution(False) is False

    def test_resolve_prompt_evolution_false_without_env(self):
        from aragora.debate.safety import resolve_prompt_evolution

        with patch.dict("os.environ", {}, clear=True):
            assert resolve_prompt_evolution(True) is False

    def test_resolve_prompt_evolution_true_with_env(self):
        from aragora.debate.safety import resolve_prompt_evolution

        with patch.dict("os.environ", {"ARAGORA_ALLOW_PROMPT_EVOLVE": "1"}):
            assert resolve_prompt_evolution(True) is True


# ---------------------------------------------------------------------------
# OptionalImports (optional_imports.py)
# ---------------------------------------------------------------------------


class TestOptionalImports:
    def setup_method(self):
        from aragora.debate.optional_imports import OptionalImports

        OptionalImports.clear_cache()

    def test_get_cached_returns_none_on_import_error(self):
        from aragora.debate.optional_imports import OptionalImports

        result = OptionalImports._get_cached(
            "nonexistent", "aragora.nonexistent.module", "FakeClass"
        )
        assert result is None

    def test_get_cached_caches_result(self):
        from aragora.debate.optional_imports import OptionalImports

        # First call should cache
        result1 = OptionalImports._get_cached("test_key", "aragora.nonexistent_module_xyz", "Fake")
        assert result1 is None
        assert "test_key" in OptionalImports._cache

        # Second call should use cache
        result2 = OptionalImports._get_cached("test_key", "aragora.nonexistent_module_xyz", "Fake")
        assert result2 is None

    def test_get_position_tracker(self):
        from aragora.debate.optional_imports import OptionalImports

        # Should return the class or None (depending on imports)
        result = OptionalImports.get_position_tracker()
        assert result is None or callable(result)

    def test_get_calibration_tracker(self):
        from aragora.debate.optional_imports import OptionalImports

        result = OptionalImports.get_calibration_tracker()
        assert result is None or callable(result)

    def test_get_belief_analyzer_returns_tuple(self):
        from aragora.debate.optional_imports import OptionalImports

        result = OptionalImports.get_belief_analyzer()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_clear_cache(self):
        from aragora.debate.optional_imports import OptionalImports

        OptionalImports._cache["test"] = "value"
        OptionalImports.clear_cache()
        assert "test" not in OptionalImports._cache

    def test_get_argument_cartographer(self):
        from aragora.debate.optional_imports import OptionalImports

        result = OptionalImports.get_argument_cartographer()
        assert result is None or callable(result)

    def test_get_critique_store(self):
        from aragora.debate.optional_imports import OptionalImports

        result = OptionalImports.get_critique_store()
        assert result is None or callable(result)

    def test_get_citation_extractor(self):
        from aragora.debate.optional_imports import OptionalImports

        result = OptionalImports.get_citation_extractor()
        assert result is None or callable(result)

    def test_get_insight_extractor(self):
        from aragora.debate.optional_imports import OptionalImports

        result = OptionalImports.get_insight_extractor()
        assert result is None or callable(result)


# ---------------------------------------------------------------------------
# OrchestratorFactory (orchestrator_factory.py)
# ---------------------------------------------------------------------------


class TestOrchestratorFactory:
    def test_from_config_creates_arena(self):
        from aragora.debate.orchestrator_factory import from_config

        mock_cls = MagicMock()
        mock_env = MagicMock()
        mock_agents = [MagicMock()]

        with patch("aragora.debate.feature_validator.validate_and_warn"):
            from_config(mock_cls, mock_env, mock_agents)

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args
        assert call_kwargs.kwargs["environment"] is mock_env
        assert call_kwargs.kwargs["agents"] is mock_agents

    def test_from_configs_passes_kwargs(self):
        from aragora.debate.orchestrator_factory import from_configs

        mock_cls = MagicMock()
        mock_env = MagicMock()
        mock_agents = [MagicMock()]

        from_configs(mock_cls, mock_env, mock_agents, debate_config="test_config")

        mock_cls.assert_called_once()
        assert mock_cls.call_args.kwargs["debate_config"] == "test_config"

    def test_create_merges_config_and_kwargs(self):
        from aragora.debate.orchestrator_factory import create

        mock_cls = MagicMock()
        mock_env = MagicMock()
        mock_agents = [MagicMock()]
        mock_config = MagicMock()
        mock_config.to_arena_kwargs.return_value = {"rounds": 3, "timeout": 60}

        create(
            mock_cls,
            mock_env,
            mock_agents,
            config=mock_config,
            debate_config="override",
        )

        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["debate_config"] == "override"
        assert kwargs["timeout"] == 60

    def test_create_without_config(self):
        from aragora.debate.orchestrator_factory import create

        mock_cls = MagicMock()
        create(mock_cls, MagicMock(), [MagicMock()])
        mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# LifecycleManager (lifecycle_manager.py)
# ---------------------------------------------------------------------------


class TestLifecycleManager:
    def test_is_arena_task_true(self):
        from aragora.debate.lifecycle_manager import LifecycleManager

        mgr = LifecycleManager()
        task = MagicMock()
        task.get_name.return_value = "arena_run_debate"
        assert mgr.is_arena_task(task) is True

    def test_is_arena_task_false(self):
        from aragora.debate.lifecycle_manager import LifecycleManager

        mgr = LifecycleManager()
        task = MagicMock()
        task.get_name.return_value = "unrelated_task"
        assert mgr.is_arena_task(task) is False

    def test_is_arena_task_debate_prefix(self):
        from aragora.debate.lifecycle_manager import LifecycleManager

        mgr = LifecycleManager()
        task = MagicMock()
        task.get_name.return_value = "debate_round_1"
        assert mgr.is_arena_task(task) is True

    def test_count_open_circuit_breakers_none(self):
        from aragora.debate.lifecycle_manager import LifecycleManager

        mgr = LifecycleManager(circuit_breaker=None)
        assert mgr.count_open_circuit_breakers() == 0

    def test_count_open_circuit_breakers_some_open(self):
        from aragora.debate.lifecycle_manager import LifecycleManager

        cb = MagicMock()
        state1 = MagicMock(is_open=True)
        state2 = MagicMock(is_open=False)
        state3 = MagicMock(is_open=True)
        cb._agent_states = {"a": state1, "b": state2, "c": state3}

        mgr = LifecycleManager(circuit_breaker=cb)
        assert mgr.count_open_circuit_breakers() == 2

    def test_clear_cache(self):
        from aragora.debate.lifecycle_manager import LifecycleManager

        cache = MagicMock()
        mgr = LifecycleManager(cache=cache)
        mgr.clear_cache()
        cache.clear.assert_called_once()

    def test_clear_cache_no_cache(self):
        from aragora.debate.lifecycle_manager import LifecycleManager

        mgr = LifecycleManager(cache=None)
        mgr.clear_cache()  # Should not raise

    def test_log_phase_failures_success(self):
        from aragora.debate.lifecycle_manager import LifecycleManager

        mgr = LifecycleManager()
        result = MagicMock(success=True)
        mgr.log_phase_failures(result)  # Should not raise

    def test_log_phase_failures_with_failures(self):
        from aragora.debate.lifecycle_manager import LifecycleManager

        mgr = LifecycleManager()
        phase1 = MagicMock(phase_name="proposal", status=MagicMock(value="failed"))
        phase2 = MagicMock(phase_name="critique", status=MagicMock(value="completed"))
        result = MagicMock(success=False, phases=[phase1, phase2])
        mgr.log_phase_failures(result)  # Should log warning

    @pytest.mark.asyncio
    async def test_cancel_task(self):
        from aragora.debate.lifecycle_manager import LifecycleManager

        mgr = LifecycleManager()

        # Create a real asyncio task that we can cancel
        async def noop():
            await asyncio.sleep(10)

        task = asyncio.create_task(noop())
        await mgr.cancel_task(task)
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_close_checkpoint_manager(self):
        from aragora.debate.lifecycle_manager import LifecycleManager

        cp = MagicMock()
        cp.close = MagicMock(return_value=None)  # Sync close
        mgr = LifecycleManager(checkpoint_manager=cp)
        await mgr.close_checkpoint_manager()
        cp.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_checkpoint_manager_none(self):
        from aragora.debate.lifecycle_manager import LifecycleManager

        mgr = LifecycleManager(checkpoint_manager=None)
        await mgr.close_checkpoint_manager()  # Should not raise


class TestArenaContextManager:
    @pytest.mark.asyncio
    async def test_context_manager_protocol(self):
        from aragora.debate.lifecycle_manager import ArenaContextManager

        mgr = ArenaContextManager()
        async with mgr as ctx:
            assert ctx is mgr


# ---------------------------------------------------------------------------
# ContextCompressor (phases/context_compressor.py)
# ---------------------------------------------------------------------------


class TestContextCompressor:
    def test_init_defaults(self):
        from aragora.debate.phases.context_compressor import ContextCompressor

        c = ContextCompressor()
        assert c._min_messages == 10
        assert c._timeout == 30.0

    def test_should_compress_false_no_callback(self):
        from aragora.debate.phases.context_compressor import ContextCompressor

        c = ContextCompressor(compress_callback=None)
        ctx = MagicMock()
        ctx.context_messages = list(range(20))
        assert c.should_compress(ctx) is False

    def test_should_compress_false_few_messages(self):
        from aragora.debate.phases.context_compressor import ContextCompressor

        c = ContextCompressor(compress_callback=lambda: None, min_messages=10)
        ctx = MagicMock()
        ctx.context_messages = list(range(5))
        assert c.should_compress(ctx) is False

    def test_should_compress_true(self):
        from aragora.debate.phases.context_compressor import ContextCompressor

        c = ContextCompressor(compress_callback=lambda: None, min_messages=5)
        ctx = MagicMock()
        ctx.context_messages = list(range(10))
        assert c.should_compress(ctx) is True

    @pytest.mark.asyncio
    async def test_compress_context_no_callback(self):
        from aragora.debate.phases.context_compressor import ContextCompressor

        c = ContextCompressor(compress_callback=None)
        ctx = MagicMock()
        ctx.context_messages = list(range(20))
        result = await c.compress_context(ctx, 1, [])
        assert result == (0, 0)

    @pytest.mark.asyncio
    async def test_compress_context_below_threshold(self):
        from aragora.debate.phases.context_compressor import ContextCompressor

        c = ContextCompressor(compress_callback=AsyncMock(), min_messages=10)
        ctx = MagicMock()
        ctx.context_messages = list(range(5))
        result = await c.compress_context(ctx, 1, [])
        assert result == (0, 0)

    @pytest.mark.asyncio
    async def test_compress_context_successful(self):
        from aragora.debate.phases.context_compressor import ContextCompressor

        original = list(range(15))
        compressed = list(range(8))

        async def mock_compress(messages, critiques):
            return compressed, []

        c = ContextCompressor(compress_callback=mock_compress, min_messages=5)
        ctx = MagicMock()
        ctx.context_messages = original.copy()
        result = await c.compress_context(ctx, 2, [])
        assert result == (15, 8)
        assert ctx.context_messages == compressed


# ---------------------------------------------------------------------------
# PersonaFeedback (phases/feedback_persona.py)
# ---------------------------------------------------------------------------


class TestPersonaFeedback:
    def test_init_defaults(self):
        from aragora.debate.phases.feedback_persona import PersonaFeedback

        pf = PersonaFeedback()
        assert pf.persona_manager is None
        assert pf.event_emitter is None

    def test_update_persona_no_manager(self):
        from aragora.debate.phases.feedback_persona import PersonaFeedback

        pf = PersonaFeedback(persona_manager=None)
        ctx = MagicMock()
        pf.update_persona_performance(ctx)  # Should not raise

    def test_detect_emerging_traits_no_stats(self):
        from aragora.debate.phases.feedback_persona import PersonaFeedback

        pm = MagicMock()
        del pm.get_performance_stats  # No stats method
        pf = PersonaFeedback(persona_manager=pm)
        ctx = MagicMock()
        traits = pf.detect_emerging_traits("agent1", ctx)
        assert traits == []

    def test_detect_emerging_traits_domain_specialist(self):
        from aragora.debate.phases.feedback_persona import PersonaFeedback

        pm = MagicMock()
        pm.get_performance_stats.return_value = {
            "domain_wins": {"security": 5},
            "prediction_accuracy": 0.6,
            "total_predictions": 3,
            "win_rate": 0.5,
            "total_debates": 3,
        }
        pf = PersonaFeedback(persona_manager=pm)
        ctx = MagicMock()
        ctx.domain = "security"

        traits = pf.detect_emerging_traits("agent1", ctx)
        trait_names = [t["name"] for t in traits]
        assert "security_specialist" in trait_names

    def test_detect_emerging_traits_well_calibrated(self):
        from aragora.debate.phases.feedback_persona import PersonaFeedback

        pm = MagicMock()
        pm.get_performance_stats.return_value = {
            "domain_wins": {},
            "prediction_accuracy": 0.9,
            "total_predictions": 10,
            "win_rate": 0.5,
            "total_debates": 3,
        }
        pf = PersonaFeedback(persona_manager=pm)
        ctx = MagicMock()
        ctx.domain = "general"

        traits = pf.detect_emerging_traits("agent1", ctx)
        trait_names = [t["name"] for t in traits]
        assert "well_calibrated" in trait_names

    def test_detect_emerging_traits_consistent_winner(self):
        from aragora.debate.phases.feedback_persona import PersonaFeedback

        pm = MagicMock()
        pm.get_performance_stats.return_value = {
            "domain_wins": {},
            "prediction_accuracy": 0.5,
            "total_predictions": 2,
            "win_rate": 0.8,
            "total_debates": 10,
        }
        pf = PersonaFeedback(persona_manager=pm)
        ctx = MagicMock()
        ctx.domain = "general"

        traits = pf.detect_emerging_traits("agent1", ctx)
        trait_names = [t["name"] for t in traits]
        assert "consistent_winner" in trait_names

    def test_check_trait_emergence_no_manager(self):
        from aragora.debate.phases.feedback_persona import PersonaFeedback

        pf = PersonaFeedback(persona_manager=None)
        ctx = MagicMock()
        pf.check_trait_emergence(ctx)  # Should not raise


# ---------------------------------------------------------------------------
# ComplianceArtifactHook (hooks/compliance_artifact_hook.py)
# ---------------------------------------------------------------------------


class TestComplianceArtifactHook:
    def test_init_defaults(self):
        from aragora.debate.hooks.compliance_artifact_hook import ComplianceArtifactHook

        hook = ComplianceArtifactHook()
        assert hook.frameworks == ["eu_ai_act"]
        assert hook.min_risk_level == "HIGH"
        assert hook.enabled is True

    def test_meets_risk_threshold_high(self):
        from aragora.debate.hooks.compliance_artifact_hook import ComplianceArtifactHook

        hook = ComplianceArtifactHook(min_risk_level="HIGH")
        assert hook._meets_risk_threshold("HIGH") is True
        assert hook._meets_risk_threshold("CRITICAL") is True
        assert hook._meets_risk_threshold("MEDIUM") is False
        assert hook._meets_risk_threshold("LOW") is False

    def test_meets_risk_threshold_medium(self):
        from aragora.debate.hooks.compliance_artifact_hook import ComplianceArtifactHook

        hook = ComplianceArtifactHook(min_risk_level="MEDIUM")
        assert hook._meets_risk_threshold("MEDIUM") is True
        assert hook._meets_risk_threshold("HIGH") is True
        assert hook._meets_risk_threshold("LOW") is False

    def test_meets_risk_threshold_invalid(self):
        from aragora.debate.hooks.compliance_artifact_hook import ComplianceArtifactHook

        hook = ComplianceArtifactHook(min_risk_level="HIGH")
        assert hook._meets_risk_threshold("INVALID") is False

    def test_on_post_debate_disabled(self):
        from aragora.debate.hooks.compliance_artifact_hook import ComplianceArtifactHook

        hook = ComplianceArtifactHook(enabled=False)
        ctx = MagicMock()
        result = MagicMock()
        hook.on_post_debate(ctx, result)  # Should return early

    def test_build_receipt_dict(self):
        from aragora.debate.hooks.compliance_artifact_hook import ComplianceArtifactHook

        hook = ComplianceArtifactHook()
        ctx = MagicMock()
        ctx.debate_id = "d123"
        result = MagicMock()
        result.task = "Evaluate risk"
        result.consensus_reached = True
        result.confidence = 0.85
        result.agents = [MagicMock(name="claude"), MagicMock(name="gpt")]
        result.total_rounds = 3
        result.final_answer = "Accept proposal"

        receipt = hook._build_receipt_dict(ctx, result)
        assert receipt["debate_id"] == "d123"
        assert receipt["verdict"] == "PASS"
        assert receipt["confidence"] == 0.85
        assert receipt["rounds"] == 3

    def test_classify_risk_fallback(self):
        from aragora.debate.hooks.compliance_artifact_hook import ComplianceArtifactHook

        hook = ComplianceArtifactHook()
        # Low confidence → HIGH risk
        assert hook._classify_risk({"confidence": 0.3}) == "HIGH"
        # Higher confidence → MEDIUM risk
        assert hook._classify_risk({"confidence": 0.7}) == "MEDIUM"

    def test_create_compliance_artifact_hook_factory(self):
        from aragora.debate.hooks.compliance_artifact_hook import (
            create_compliance_artifact_hook,
        )

        hook = create_compliance_artifact_hook(
            frameworks=["soc2"],
            min_risk_level="MEDIUM",
            enabled=False,
        )
        assert hook.frameworks == ["soc2"]
        assert hook.min_risk_level == "MEDIUM"
        assert hook.enabled is False


# ---------------------------------------------------------------------------
# DebateEmbeddingsDatabase (embeddings.py)
# ---------------------------------------------------------------------------


class TestDebateEmbeddingsDatabase:
    def test_transcript_to_text(self):
        from aragora.debate.embeddings import DebateEmbeddingsDatabase

        # Patch the __init__ to avoid needing actual DB
        with patch.object(DebateEmbeddingsDatabase, "__init__", lambda self, *a, **kw: None):
            db = DebateEmbeddingsDatabase.__new__(DebateEmbeddingsDatabase)
            transcript = [
                {"agent": "claude", "type": "proposal", "content": "I suggest X"},
                {"agent": "gpt", "type": "critique", "content": "But what about Y?"},
            ]
            text = db._transcript_to_text(transcript)
            assert "claude (proposal): I suggest X" in text
            assert "gpt (critique): But what about Y?" in text

    def test_transcript_to_text_empty(self):
        from aragora.debate.embeddings import DebateEmbeddingsDatabase

        with patch.object(DebateEmbeddingsDatabase, "__init__", lambda self, *a, **kw: None):
            db = DebateEmbeddingsDatabase.__new__(DebateEmbeddingsDatabase)
            assert db._transcript_to_text([]) == ""

    def test_transcript_to_text_missing_fields(self):
        from aragora.debate.embeddings import DebateEmbeddingsDatabase

        with patch.object(DebateEmbeddingsDatabase, "__init__", lambda self, *a, **kw: None):
            db = DebateEmbeddingsDatabase.__new__(DebateEmbeddingsDatabase)
            transcript = [{"other_key": "value"}]
            text = db._transcript_to_text(transcript)
            assert "Unknown (message)" in text


# ---------------------------------------------------------------------------
# TracesDatabase (traces_database.py)
# ---------------------------------------------------------------------------


class TestTracesDatabase:
    def test_repr(self):
        from aragora.debate.traces_database import TracesDatabase

        with patch.object(TracesDatabase, "__init__", lambda self, *a, **kw: None):
            db = TracesDatabase.__new__(TracesDatabase)
            from pathlib import Path

            db.db_path = Path("/tmp/test.db")
            assert "test.db" in repr(db)

    def test_fetch_one(self, tmp_path):
        """TracesDatabase.fetch_one works with real SQLite."""
        import sqlite3

        db_path = tmp_path / "traces.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO test VALUES (1, 'hello')")
        conn.commit()
        conn.close()

        from aragora.debate.traces_database import TracesDatabase

        with patch("aragora.debate.traces_database.DatabaseManager") as MockDM:
            mock_mgr = MagicMock()

            # Make fresh_connection() return a real sqlite connection context manager
            from contextlib import contextmanager

            @contextmanager
            def fresh_conn():
                c = sqlite3.connect(str(db_path))
                try:
                    yield c
                    c.commit()
                finally:
                    c.close()

            mock_mgr.fresh_connection = fresh_conn
            MockDM.get_instance.return_value = mock_mgr

            db = TracesDatabase(str(db_path))
            row = db.fetch_one("SELECT * FROM test WHERE id = ?", (1,))
            assert row == (1, "hello")

    def test_fetch_all(self, tmp_path):
        """TracesDatabase.fetch_all returns multiple rows."""
        import sqlite3

        db_path = tmp_path / "traces.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.execute("INSERT INTO test VALUES (1)")
        conn.execute("INSERT INTO test VALUES (2)")
        conn.commit()
        conn.close()

        from aragora.debate.traces_database import TracesDatabase

        with patch("aragora.debate.traces_database.DatabaseManager") as MockDM:
            mock_mgr = MagicMock()

            from contextlib import contextmanager

            @contextmanager
            def fresh_conn():
                c = sqlite3.connect(str(db_path))
                try:
                    yield c
                    c.commit()
                finally:
                    c.close()

            mock_mgr.fresh_connection = fresh_conn
            MockDM.get_instance.return_value = mock_mgr

            db = TracesDatabase(str(db_path))
            rows = db.fetch_all("SELECT * FROM test ORDER BY id")
            assert len(rows) == 2
            assert rows[0] == (1,)
            assert rows[1] == (2,)


# ---------------------------------------------------------------------------
# Callback timeout helper
# ---------------------------------------------------------------------------


class TestCallbackTimeout:
    @pytest.mark.asyncio
    async def test_with_callback_timeout_success(self):
        from aragora.debate.phases.context_compressor import _with_callback_timeout

        async def quick():
            return 42

        result = await _with_callback_timeout(quick(), timeout=5.0)
        assert result == 42

    @pytest.mark.asyncio
    async def test_with_callback_timeout_returns_default(self):
        from aragora.debate.phases.context_compressor import _with_callback_timeout

        async def slow():
            await asyncio.sleep(10)

        result = await _with_callback_timeout(slow(), timeout=0.01, default="fallback")
        assert result == "fallback"
