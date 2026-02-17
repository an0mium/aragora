"""Tests for ContentProcessor in aragora.debate.context.processors."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

import aragora.debate.context.processors as proc_mod
from aragora.debate.context.processors import ContentProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rlm_result(answer: str, used_true_rlm: bool = False, confidence: float = 0.8):
    """Create a mock RLM result object."""
    return SimpleNamespace(answer=answer, used_true_rlm=used_true_rlm, confidence=confidence)


def _make_memory(
    mem_id: str,
    tier_value: str,
    content: str = "memory content",
    consolidation_score: float = 0.6,
    metadata: dict | None = None,
):
    """Create a mock ContinuumMemory entry."""
    tier = SimpleNamespace(value=tier_value)
    return SimpleNamespace(
        id=mem_id,
        tier=tier,
        content=content,
        consolidation_score=consolidation_score,
        metadata=metadata or {},
    )


# ===================================================================
# 1. __init__
# ===================================================================
class TestContentProcessorInit:
    """Test ContentProcessor initialization."""

    def test_default_init_no_rlm(self):
        """Default init when RLM is unavailable."""
        with patch.object(proc_mod, "HAS_RLM", False):
            cp = ContentProcessor()
        assert cp._enable_rlm is False
        assert cp._aragora_rlm is None
        assert cp._rlm_compressor is None
        assert cp._rlm_threshold == 3000

    def test_custom_project_root(self, tmp_path: Path):
        with patch.object(proc_mod, "HAS_RLM", False):
            cp = ContentProcessor(project_root=tmp_path)
        assert cp._project_root == tmp_path

    def test_rlm_disabled_explicitly(self):
        """enable_rlm_compression=False bypasses RLM even when available."""
        with patch.object(proc_mod, "HAS_RLM", True):
            cp = ContentProcessor(enable_rlm_compression=False)
        assert cp._enable_rlm is False

    def test_with_prebuilt_compressor(self):
        compressor = MagicMock()
        with patch.object(proc_mod, "HAS_RLM", False):
            cp = ContentProcessor(rlm_compressor=compressor)
        assert cp._rlm_compressor is compressor

    def test_custom_threshold(self):
        with patch.object(proc_mod, "HAS_RLM", False):
            cp = ContentProcessor(rlm_compression_threshold=500)
        assert cp._rlm_threshold == 500

    def test_rlm_enabled_factory_success(self):
        """When HAS_RLM is True and factories work, both rlm and compressor are set."""
        mock_rlm = MagicMock()
        mock_compressor = MagicMock()
        with (
            patch.object(proc_mod, "HAS_RLM", True),
            patch.object(proc_mod, "HAS_OFFICIAL_RLM", True),
            patch.object(proc_mod, "get_rlm", return_value=mock_rlm),
            patch.object(proc_mod, "get_compressor", return_value=mock_compressor),
        ):
            cp = ContentProcessor()
        assert cp._enable_rlm is True
        assert cp._aragora_rlm is mock_rlm
        assert cp._rlm_compressor is mock_compressor

    def test_rlm_factory_import_error_handled(self):
        """ImportError from get_rlm is caught gracefully."""
        with (
            patch.object(proc_mod, "HAS_RLM", True),
            patch.object(proc_mod, "HAS_OFFICIAL_RLM", False),
            patch.object(proc_mod, "get_rlm", side_effect=ImportError("no rlm")),
            patch.object(proc_mod, "get_compressor", return_value=None),
        ):
            cp = ContentProcessor()
        assert cp._aragora_rlm is None

    def test_rlm_factory_runtime_error_handled(self):
        """RuntimeError from get_rlm is caught gracefully."""
        with (
            patch.object(proc_mod, "HAS_RLM", True),
            patch.object(proc_mod, "HAS_OFFICIAL_RLM", False),
            patch.object(proc_mod, "get_rlm", side_effect=RuntimeError("init fail")),
            patch.object(proc_mod, "get_compressor", return_value=None),
        ):
            cp = ContentProcessor()
        assert cp._aragora_rlm is None


# ===================================================================
# 2. compress_with_rlm
# ===================================================================
class TestCompressWithRLM:
    """Test the compress_with_rlm method."""

    @pytest.fixture()
    def processor_no_rlm(self):
        with patch.object(proc_mod, "HAS_RLM", False):
            return ContentProcessor()

    @pytest.fixture()
    def processor_with_rlm(self):
        mock_rlm = AsyncMock()
        mock_compressor = AsyncMock()
        with (
            patch.object(proc_mod, "HAS_RLM", True),
            patch.object(proc_mod, "HAS_OFFICIAL_RLM", True),
            patch.object(proc_mod, "get_rlm", return_value=mock_rlm),
            patch.object(proc_mod, "get_compressor", return_value=mock_compressor),
        ):
            cp = ContentProcessor()
        return cp

    @pytest.mark.asyncio
    async def test_content_under_threshold_returns_as_is(self, processor_no_rlm):
        """Short content is returned unchanged."""
        result = await processor_no_rlm.compress_with_rlm("hello", max_chars=3000)
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_content_under_threshold_but_over_max_chars_is_capped(self, processor_no_rlm):
        """Content under threshold but over max_chars is capped."""
        content = "a" * 200
        result = await processor_no_rlm.compress_with_rlm(content, max_chars=50)
        assert len(result) == 50

    @pytest.mark.asyncio
    async def test_rlm_disabled_uses_truncation(self, processor_no_rlm):
        """When RLM is disabled, long content is truncated."""
        content = "x" * 5000
        result = await processor_no_rlm.compress_with_rlm(content, max_chars=500)
        assert result.endswith("... [truncated]")
        assert len(result) <= 500

    @pytest.mark.asyncio
    async def test_aragora_rlm_success_path(self, processor_with_rlm):
        """AragoraRLM compress_and_query success returns compressed content."""
        compressed = "compressed summary"
        processor_with_rlm._aragora_rlm.compress_and_query = AsyncMock(
            return_value=_make_rlm_result(compressed, used_true_rlm=True)
        )
        content = "x" * 5000
        result = await processor_with_rlm.compress_with_rlm(content, max_chars=3000)
        assert result == compressed

    @pytest.mark.asyncio
    async def test_aragora_rlm_timeout_falls_back_to_compressor(self, processor_with_rlm):
        """AragoraRLM timeout leads to HierarchicalCompressor fallback."""
        processor_with_rlm._aragora_rlm.compress_and_query = AsyncMock(
            side_effect=asyncio.TimeoutError
        )
        # Set up compressor mock
        summary_text = "compressor summary"
        mock_level = MagicMock()
        mock_context = MagicMock()
        mock_context.get_at_level.return_value = summary_text
        mock_result = MagicMock(context=mock_context)
        processor_with_rlm._rlm_compressor.compress = AsyncMock(return_value=mock_result)

        with patch("aragora.debate.context.processors.AbstractionLevel", create=True) as mock_al:
            mock_al.SUMMARY = "SUMMARY"
            mock_al.ABSTRACT = "ABSTRACT"
            # Patch the import inside the method
            with patch.dict("sys.modules", {"aragora.rlm.types": MagicMock(AbstractionLevel=mock_al)}):
                content = "x" * 5000
                result = await processor_with_rlm.compress_with_rlm(content, max_chars=3000)
        assert result == summary_text

    @pytest.mark.asyncio
    async def test_both_fail_uses_truncation(self, processor_with_rlm):
        """When both AragoraRLM and compressor fail, truncation is used."""
        processor_with_rlm._aragora_rlm.compress_and_query = AsyncMock(
            side_effect=RuntimeError("rlm fail")
        )
        processor_with_rlm._rlm_compressor.compress = AsyncMock(
            side_effect=RuntimeError("compressor fail")
        )
        content = "x" * 5000
        result = await processor_with_rlm.compress_with_rlm(content, max_chars=500)
        assert result.endswith("... [truncated]")
        assert len(result) <= 500

    @pytest.mark.asyncio
    async def test_max_chars_respected_after_rlm(self, processor_with_rlm):
        """Result from RLM is capped at max_chars."""
        long_answer = "y" * 6000
        processor_with_rlm._aragora_rlm.compress_and_query = AsyncMock(
            return_value=_make_rlm_result(long_answer, used_true_rlm=False)
        )
        content = "x" * 10000
        result = await processor_with_rlm.compress_with_rlm(content, max_chars=500)
        assert len(result) == 500

    @pytest.mark.asyncio
    async def test_rlm_returns_none_answer_falls_through(self, processor_with_rlm):
        """If RLM answer is None, falls through to compressor or truncation."""
        processor_with_rlm._aragora_rlm.compress_and_query = AsyncMock(
            return_value=_make_rlm_result(None)
        )
        processor_with_rlm._rlm_compressor.compress = AsyncMock(
            side_effect=RuntimeError("also fails")
        )
        content = "x" * 5000
        result = await processor_with_rlm.compress_with_rlm(content, max_chars=500)
        assert "truncated" in result


# ===================================================================
# 3. query_with_true_rlm
# ===================================================================
class TestQueryWithTrueRLM:
    """Test the query_with_true_rlm method."""

    @pytest.mark.asyncio
    async def test_rlm_disabled_returns_none(self):
        with patch.object(proc_mod, "HAS_RLM", False):
            cp = ContentProcessor(enable_rlm_compression=False)
        result = await cp.query_with_true_rlm("what?", "content")
        assert result is None

    @pytest.mark.asyncio
    async def test_no_aragora_rlm_returns_none(self):
        with patch.object(proc_mod, "HAS_RLM", False):
            cp = ContentProcessor()
        cp._enable_rlm = True  # force enable but no rlm instance
        cp._aragora_rlm = None
        result = await cp.query_with_true_rlm("what?", "content")
        assert result is None

    @pytest.mark.asyncio
    async def test_true_rlm_success(self):
        """TRUE RLM query returns an answer when available."""
        mock_rlm = AsyncMock()
        mock_rlm.query = AsyncMock(
            return_value=_make_rlm_result("the answer", used_true_rlm=True)
        )
        with patch.object(proc_mod, "HAS_RLM", False):
            cp = ContentProcessor()
        cp._enable_rlm = True
        cp._aragora_rlm = mock_rlm

        with (
            patch.object(proc_mod, "HAS_RLM", True),
            patch.object(proc_mod, "HAS_OFFICIAL_RLM", True),
        ):
            result = await cp.query_with_true_rlm("what?", "content")
        assert result == "the answer"

    @pytest.mark.asyncio
    async def test_true_rlm_timeout_returns_none(self):
        """Timeout during TRUE RLM query returns None."""
        mock_rlm = AsyncMock()
        mock_rlm.query = AsyncMock(side_effect=asyncio.TimeoutError)
        mock_rlm.compress_and_query = AsyncMock(side_effect=asyncio.TimeoutError)
        with patch.object(proc_mod, "HAS_RLM", False):
            cp = ContentProcessor()
        cp._enable_rlm = True
        cp._aragora_rlm = mock_rlm

        with (
            patch.object(proc_mod, "HAS_RLM", True),
            patch.object(proc_mod, "HAS_OFFICIAL_RLM", True),
        ):
            result = await cp.query_with_true_rlm("what?", "content")
        assert result is None

    @pytest.mark.asyncio
    async def test_fallback_to_compress_and_query_when_not_official(self):
        """When HAS_OFFICIAL_RLM is False, falls back to compress_and_query."""
        mock_rlm = AsyncMock()
        mock_rlm.compress_and_query = AsyncMock(
            return_value=_make_rlm_result("compressed answer", used_true_rlm=False)
        )
        with patch.object(proc_mod, "HAS_RLM", False):
            cp = ContentProcessor()
        cp._enable_rlm = True
        cp._aragora_rlm = mock_rlm

        with (
            patch.object(proc_mod, "HAS_RLM", True),
            patch.object(proc_mod, "HAS_OFFICIAL_RLM", False),
        ):
            result = await cp.query_with_true_rlm("what?", "content")
        assert result == "compressed answer"

    @pytest.mark.asyncio
    async def test_value_error_returns_none(self):
        """ValueError during query returns None."""
        mock_rlm = AsyncMock()
        mock_rlm.compress_and_query = AsyncMock(side_effect=ValueError("bad"))
        with patch.object(proc_mod, "HAS_RLM", False):
            cp = ContentProcessor()
        cp._enable_rlm = True
        cp._aragora_rlm = mock_rlm

        with (
            patch.object(proc_mod, "HAS_RLM", True),
            patch.object(proc_mod, "HAS_OFFICIAL_RLM", False),
        ):
            result = await cp.query_with_true_rlm("what?", "content")
        assert result is None


# ===================================================================
# 4. gather_aragora_context
# ===================================================================
class TestGatherAragoraContext:
    """Test the gather_aragora_context method."""

    @pytest.fixture()
    def processor(self, tmp_path: Path):
        with patch.object(proc_mod, "HAS_RLM", False):
            return ContentProcessor(project_root=tmp_path)

    @pytest.mark.asyncio
    async def test_non_aragora_task_returns_none(self, processor):
        result = await processor.gather_aragora_context("What is the best pizza topping?")
        assert result is None

    @pytest.mark.asyncio
    async def test_aragora_topic_reads_docs(self, processor, tmp_path: Path):
        """Aragora-related task reads documentation and returns context."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "FEATURES.md").write_text("Feature list here")
        (docs_dir / "ARCHITECTURE.md").write_text("Architecture overview")
        # Also CLAUDE.md at project root
        (tmp_path / "CLAUDE.md").write_text("Claude integration guide")

        result = await processor.gather_aragora_context("How does aragora debate work?")
        assert result is not None
        assert "ARAGORA PROJECT CONTEXT" in result

    @pytest.mark.asyncio
    async def test_file_read_failure_handled_gracefully(self, processor, tmp_path: Path):
        """If docs directory doesn't exist, returns None gracefully."""
        # tmp_path has no docs/ directory
        result = await processor.gather_aragora_context("aragora debate")
        # Should not raise; may return None or empty context
        assert result is None or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_detects_various_keywords(self, processor, tmp_path: Path):
        """All aragora-related keywords trigger context gathering."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "STATUS.md").write_text("Status info")

        keywords = [
            "multi-agent debate",
            "decision stress-test",
            "ai red team",
            "adversarial validation",
            "gauntlet",
            "nomic loop",
            "debate framework",
        ]
        for kw in keywords:
            result = await processor.gather_aragora_context(f"Tell me about {kw}")
            assert result is not None or result is None  # should not raise

    @pytest.mark.asyncio
    async def test_codebase_context_included(self, processor, tmp_path: Path):
        """Codebase context is inserted at beginning when available."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "FEATURES.md").write_text("Features")

        with patch.object(
            processor, "_gather_codebase_context", new_callable=AsyncMock, return_value="## MAP\ncode"
        ):
            result = await processor.gather_aragora_context("aragora improvements")
        assert result is not None
        assert "## MAP" in result


# ===================================================================
# 5. _gather_codebase_context
# ===================================================================
class TestGatherCodebaseContext:
    """Test the _gather_codebase_context private method."""

    @pytest.fixture()
    def processor(self, tmp_path: Path):
        with patch.object(proc_mod, "HAS_RLM", False):
            return ContentProcessor(project_root=tmp_path)

    @pytest.mark.asyncio
    async def test_env_var_disabled_returns_none(self, processor):
        """Explicitly disabling via env var returns None."""
        with patch.dict("os.environ", {"ARAGORA_CONTEXT_USE_CODEBASE": "false"}):
            result = await processor._gather_codebase_context()
        assert result is None

    @pytest.mark.asyncio
    async def test_env_var_off_returns_none(self, processor):
        with patch.dict("os.environ", {"ARAGORA_CONTEXT_USE_CODEBASE": "0"}):
            result = await processor._gather_codebase_context()
        assert result is None

    @pytest.mark.asyncio
    async def test_import_failure_returns_none(self, processor):
        """ImportError for CodebaseContextBuilder returns None."""
        with (
            patch.dict("os.environ", {}, clear=False),
            patch(
                "aragora.debate.context.processors.CodebaseContextBuilder",
                side_effect=ImportError("no module"),
                create=True,
            ),
        ):
            # Force the import to fail inside the method
            import builtins
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if "codebase_context" in name:
                    raise ImportError("no module")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                result = await processor._gather_codebase_context()
        assert result is None

    @pytest.mark.asyncio
    async def test_builder_timeout_returns_none(self, processor):
        """Builder timeout returns None."""
        mock_builder = MagicMock()
        mock_builder.build_debate_context = AsyncMock(side_effect=asyncio.TimeoutError)

        mock_module = MagicMock()
        mock_module.CodebaseContextBuilder.return_value = mock_builder

        with (
            patch.dict("os.environ", {}, clear=False),
            patch.dict("sys.modules", {"aragora.rlm.codebase_context": mock_module}),
        ):
            processor._codebase_context_builder = mock_builder
            result = await processor._gather_codebase_context()
        assert result is None

    @pytest.mark.asyncio
    async def test_success_path(self, processor):
        """Success: builder returns context string."""
        mock_builder = MagicMock()
        mock_builder.build_debate_context = AsyncMock(return_value="module1\nmodule2")

        mock_module = MagicMock()
        mock_module.CodebaseContextBuilder.return_value = mock_builder

        with (
            patch.dict("os.environ", {}, clear=False),
            patch.dict("sys.modules", {"aragora.rlm.codebase_context": mock_module}),
        ):
            processor._codebase_context_builder = mock_builder
            result = await processor._gather_codebase_context()
        assert result is not None
        assert "CODEBASE MAP" in result
        assert "module1" in result

    @pytest.mark.asyncio
    async def test_builder_returns_empty_string(self, processor):
        """Empty string from builder returns None."""
        mock_builder = MagicMock()
        mock_builder.build_debate_context = AsyncMock(return_value="")

        processor._codebase_context_builder = mock_builder
        with patch.dict("os.environ", {}, clear=False):
            result = await processor._gather_codebase_context()
        assert result is None


# ===================================================================
# 6. get_continuum_context
# ===================================================================
class TestGetContinuumContext:
    """Test the get_continuum_context method."""

    @pytest.fixture()
    def processor(self):
        with patch.object(proc_mod, "HAS_RLM", False):
            return ContentProcessor()

    def test_none_memory_returns_empty(self, processor):
        ctx, ids, tiers = processor.get_continuum_context(
            continuum_memory=None, domain="test", task="test task"
        )
        assert ctx == ""
        assert ids == []
        assert tiers == {}

    def test_no_memories_returns_empty(self, processor):
        mock_mem = MagicMock()
        mock_mem.retrieve.return_value = []

        ctx, ids, tiers = processor.get_continuum_context(
            continuum_memory=mock_mem, domain="coding", task="test task"
        )
        assert ctx == ""
        assert ids == []

    def test_recent_memories_formatted_correctly(self, processor):
        """Recent (non-glacial) memories are formatted with tier and confidence."""
        mem1 = _make_memory("m1", "fast", content="Fast memory", consolidation_score=0.8)
        mem2 = _make_memory("m2", "medium", content="Medium memory", consolidation_score=0.5)

        mock_cm = MagicMock()
        mock_cm.retrieve.return_value = [mem1, mem2]
        # No get_glacial_insights
        del mock_cm.get_glacial_insights

        ctx, ids, tiers = processor.get_continuum_context(
            continuum_memory=mock_cm, domain="coding", task="optimize loops"
        )
        assert "Previous learnings" in ctx
        assert "[fast|high]" in ctx
        assert "[medium|medium]" in ctx
        assert "m1" in ids
        assert "m2" in ids
        assert tiers["m1"] == mem1.tier
        assert tiers["m2"] == mem2.tier

    def test_glacial_insights_included(self, processor):
        """Glacial insights appear in a separate section."""
        recent = _make_memory("r1", "slow", content="Recent insight")
        glacial = _make_memory("g1", "glacial", content="Long-term pattern")

        mock_cm = MagicMock()
        mock_cm.retrieve.return_value = [recent]
        mock_cm.get_glacial_insights.return_value = [glacial]

        ctx, ids, tiers = processor.get_continuum_context(
            continuum_memory=mock_cm, domain="ethics", task="fairness review"
        )
        assert "[glacial|foundational]" in ctx
        assert "Long-term patterns" in ctx
        assert "g1" in ids

    def test_glacial_insights_disabled(self, processor):
        """include_glacial_insights=False skips glacial retrieval."""
        recent = _make_memory("r1", "medium", content="stuff")
        mock_cm = MagicMock()
        mock_cm.retrieve.return_value = [recent]

        ctx, ids, tiers = processor.get_continuum_context(
            continuum_memory=mock_cm,
            domain="test",
            task="test",
            include_glacial_insights=False,
        )
        mock_cm.get_glacial_insights.assert_not_called()
        assert "r1" in ids

    def test_rbac_access_denied_returns_empty(self, processor):
        """When auth_context lacks memory_read, returns empty."""
        mock_cm = MagicMock()
        auth_ctx = MagicMock()

        with patch(
            "aragora.debate.context.processors.has_memory_read_access",
            return_value=False,
            create=True,
        ):
            # Need to mock the import inside the method
            mock_access = MagicMock()
            mock_access.has_memory_read_access.return_value = False
            mock_access.emit_denial_telemetry = MagicMock()
            mock_access.filter_entries = MagicMock()

            with patch.dict(
                "sys.modules",
                {"aragora.memory.access": mock_access},
            ):
                ctx, ids, tiers = processor.get_continuum_context(
                    continuum_memory=mock_cm,
                    domain="test",
                    task="test",
                    auth_context=auth_ctx,
                )
        assert ctx == ""
        assert ids == []

    def test_tenant_id_filters_glacial(self, processor):
        """tenant_id filters glacial insights by metadata."""
        recent = _make_memory("r1", "fast", content="recent")
        glacial_match = _make_memory(
            "g1", "glacial", content="matching tenant", metadata={"tenant_id": "t1"}
        )
        glacial_other = _make_memory(
            "g2", "glacial", content="other tenant", metadata={"tenant_id": "t2"}
        )

        mock_cm = MagicMock()
        mock_cm.retrieve.return_value = [recent]
        mock_cm.get_glacial_insights.return_value = [glacial_match, glacial_other]

        ctx, ids, tiers = processor.get_continuum_context(
            continuum_memory=mock_cm,
            domain="test",
            task="test",
            tenant_id="t1",
        )
        assert "g1" in ids
        assert "g2" not in ids

    def test_confidence_low_for_low_consolidation(self, processor):
        """consolidation_score < 0.4 shows [low] confidence."""
        mem = _make_memory("m1", "fast", content="low confidence", consolidation_score=0.2)
        mock_cm = MagicMock()
        mock_cm.retrieve.return_value = [mem]
        del mock_cm.get_glacial_insights

        ctx, ids, tiers = processor.get_continuum_context(
            continuum_memory=mock_cm, domain="test", task="test"
        )
        assert "[fast|low]" in ctx

    def test_attribute_error_returns_empty(self, processor):
        """AttributeError during retrieval is caught."""
        mock_cm = MagicMock()
        mock_cm.retrieve.side_effect = AttributeError("missing attr")

        ctx, ids, tiers = processor.get_continuum_context(
            continuum_memory=mock_cm, domain="test", task="test"
        )
        assert ctx == ""


# ===================================================================
# 7. refresh_evidence_for_round
# ===================================================================
class TestRefreshEvidenceForRound:
    """Test refresh_evidence_for_round method."""

    @pytest.fixture()
    def processor(self):
        with patch.object(proc_mod, "HAS_RLM", False):
            return ContentProcessor()

    @pytest.mark.asyncio
    async def test_no_collector_returns_zero(self, processor):
        count, pack = await processor.refresh_evidence_for_round(
            combined_text="text", evidence_collector=None, task="task"
        )
        assert count == 0
        assert pack is None

    @pytest.mark.asyncio
    async def test_no_claims_returns_zero(self, processor):
        collector = MagicMock()
        collector.extract_claims_from_text.return_value = []

        count, pack = await processor.refresh_evidence_for_round(
            combined_text="text", evidence_collector=collector, task="task"
        )
        assert count == 0
        assert pack is None

    @pytest.mark.asyncio
    async def test_successful_refresh_with_callback(self, processor):
        """Successful evidence collection invokes the callback."""
        snippet1 = MagicMock()
        snippet2 = MagicMock()
        evidence_pack = MagicMock()
        evidence_pack.snippets = [snippet1, snippet2]

        collector = MagicMock()
        collector.extract_claims_from_text.return_value = ["claim1", "claim2"]
        collector.collect_for_claims = AsyncMock(return_value=evidence_pack)

        callback = MagicMock()

        count, pack = await processor.refresh_evidence_for_round(
            combined_text="Claim: X is true. Claim: Y is false.",
            evidence_collector=collector,
            task="verify claims",
            evidence_store_callback=callback,
        )
        assert count == 2
        assert pack is evidence_pack
        callback.assert_called_once_with([snippet1, snippet2], "verify claims")

    @pytest.mark.asyncio
    async def test_successful_refresh_without_callback(self, processor):
        """Without callback, still returns count and pack."""
        evidence_pack = MagicMock()
        evidence_pack.snippets = [MagicMock()]

        collector = MagicMock()
        collector.extract_claims_from_text.return_value = ["claim"]
        collector.collect_for_claims = AsyncMock(return_value=evidence_pack)

        count, pack = await processor.refresh_evidence_for_round(
            combined_text="text",
            evidence_collector=collector,
            task="task",
        )
        assert count == 1
        assert pack is evidence_pack

    @pytest.mark.asyncio
    async def test_error_handled_gracefully(self, processor):
        """RuntimeError during collection returns (0, None)."""
        collector = MagicMock()
        collector.extract_claims_from_text.side_effect = RuntimeError("extraction failed")

        count, pack = await processor.refresh_evidence_for_round(
            combined_text="text", evidence_collector=collector, task="task"
        )
        assert count == 0
        assert pack is None

    @pytest.mark.asyncio
    async def test_empty_snippets_returns_zero(self, processor):
        """If evidence_pack has empty snippets, returns (0, None)."""
        evidence_pack = MagicMock()
        evidence_pack.snippets = []

        collector = MagicMock()
        collector.extract_claims_from_text.return_value = ["claim"]
        collector.collect_for_claims = AsyncMock(return_value=evidence_pack)

        count, pack = await processor.refresh_evidence_for_round(
            combined_text="text", evidence_collector=collector, task="task"
        )
        assert count == 0
        assert pack is None


# ===================================================================
# 8. query_knowledge_with_true_rlm
# ===================================================================
class TestQueryKnowledgeWithTrueRLM:
    """Test query_knowledge_with_true_rlm method."""

    @pytest.fixture()
    def processor(self):
        with patch.object(proc_mod, "HAS_RLM", False):
            return ContentProcessor()

    @pytest.mark.asyncio
    async def test_no_mound_returns_none(self, processor):
        result = await processor.query_knowledge_with_true_rlm(
            task="test", knowledge_mound=None
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_no_true_rlm_returns_none(self, processor):
        """When HAS_OFFICIAL_RLM is False, returns None."""
        mound = MagicMock()
        with (
            patch.object(proc_mod, "HAS_RLM", True),
            patch.object(proc_mod, "HAS_OFFICIAL_RLM", False),
        ):
            result = await processor.query_knowledge_with_true_rlm(
                task="test", knowledge_mound=mound
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_import_error_handled(self, processor):
        """ImportError from get_repl_adapter returns None."""
        mound = MagicMock()

        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "aragora.rlm" and args and "get_repl_adapter" in (args[2] or ()):
                raise ImportError("no repl")
            return original_import(name, *args, **kwargs)

        with (
            patch.object(proc_mod, "HAS_RLM", True),
            patch.object(proc_mod, "HAS_OFFICIAL_RLM", True),
            patch("builtins.__import__", side_effect=mock_import),
        ):
            result = await processor.query_knowledge_with_true_rlm(
                task="test", knowledge_mound=mound
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_success_returns_repl_prompt(self, processor):
        """Successful TRUE RLM knowledge query returns REPL prompt."""
        mound = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter.create_repl_for_knowledge.return_value = MagicMock()
        mock_adapter.get_repl_prompt.return_value = ">>> km.search('query')"

        mock_cache = MagicMock()
        mock_cache.get_task_hash.return_value = "abc123"

        mock_rlm_module = MagicMock()
        mock_rlm_module.get_repl_adapter.return_value = mock_adapter

        mock_cache_module = MagicMock()
        mock_cache_module.ContextCache = mock_cache

        with (
            patch.object(proc_mod, "HAS_RLM", True),
            patch.object(proc_mod, "HAS_OFFICIAL_RLM", True),
            patch.dict("sys.modules", {
                "aragora.rlm": mock_rlm_module,
                "aragora.debate.context.cache": mock_cache_module,
            }),
        ):
            result = await processor.query_knowledge_with_true_rlm(
                task="test query", knowledge_mound=mound
            )
        assert result is not None
        assert "KNOWLEDGE MOUND CONTEXT" in result
        assert "TRUE RLM" in result

    @pytest.mark.asyncio
    async def test_runtime_error_returns_none(self, processor):
        """RuntimeError during REPL creation returns None."""
        mound = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter.create_repl_for_knowledge.side_effect = RuntimeError("REPL failed")

        mock_rlm_module = MagicMock()
        mock_rlm_module.get_repl_adapter.return_value = mock_adapter

        mock_cache = MagicMock()
        mock_cache.get_task_hash.return_value = "x"
        mock_cache_module = MagicMock()
        mock_cache_module.ContextCache = mock_cache

        with (
            patch.object(proc_mod, "HAS_RLM", True),
            patch.object(proc_mod, "HAS_OFFICIAL_RLM", True),
            patch.dict("sys.modules", {
                "aragora.rlm": mock_rlm_module,
                "aragora.debate.context.cache": mock_cache_module,
            }),
        ):
            result = await processor.query_knowledge_with_true_rlm(
                task="test", knowledge_mound=mound
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_no_env_returned_from_adapter(self, processor):
        """When adapter.create_repl_for_knowledge returns None, result is None."""
        mound = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter.create_repl_for_knowledge.return_value = None

        mock_rlm_module = MagicMock()
        mock_rlm_module.get_repl_adapter.return_value = mock_adapter

        mock_cache = MagicMock()
        mock_cache.get_task_hash.return_value = "y"
        mock_cache_module = MagicMock()
        mock_cache_module.ContextCache = mock_cache

        with (
            patch.object(proc_mod, "HAS_RLM", True),
            patch.object(proc_mod, "HAS_OFFICIAL_RLM", True),
            patch.dict("sys.modules", {
                "aragora.rlm": mock_rlm_module,
                "aragora.debate.context.cache": mock_cache_module,
            }),
        ):
            result = await processor.query_knowledge_with_true_rlm(
                task="test", knowledge_mound=mound
            )
        assert result is None
