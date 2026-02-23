"""
Tests for aragora/debate/context_gatherer/compression.py

Tests the CompressionMixin class, which provides RLM compression and
TRUE RLM query methods used by the main ContextGatherer.
"""

import asyncio
import hashlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.context_gatherer.compression import (
    CompressionMixin,
    _has_official_rlm,
)


# ---------------------------------------------------------------------------
# Concrete test class
# ---------------------------------------------------------------------------


class ConcreteCompression(CompressionMixin):
    """Minimal concrete subclass that satisfies the mixin's type contract."""

    def __init__(self):
        self._enable_rlm = False
        self._rlm_compressor = None
        self._aragora_rlm = None
        self._rlm_threshold = 3000
        self._enable_knowledge_grounding = False
        self._knowledge_mound = None
        self._knowledge_workspace_id = "test-ws"

    def _get_task_hash(self, task: str) -> str:
        return hashlib.sha256(task.encode()).hexdigest()[:16]

    async def gather_knowledge_mound_context(self, task: str) -> str | None:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rlm_result(answer: str, used_true_rlm: bool = False, confidence: float = 0.9):
    """Build a mock RLM result object."""
    return SimpleNamespace(answer=answer, used_true_rlm=used_true_rlm, confidence=confidence)


def _long_content(n: int = 5000) -> str:
    """Return a string of length *n* that exceeds default thresholds."""
    return "x" * n


# ===========================================================================
# _has_official_rlm()
# ===========================================================================


class TestHasOfficialRlm:
    """Tests for the module-level _has_official_rlm() helper."""

    def test_returns_package_override_when_set_true(self):
        """When the package module has HAS_OFFICIAL_RLM=True, return True."""
        fake_pkg = SimpleNamespace(HAS_OFFICIAL_RLM=True)
        with patch.dict(sys.modules, {"aragora.debate.context_gatherer": fake_pkg}):
            assert _has_official_rlm() is True

    def test_returns_package_override_when_set_false(self):
        """When the package module has HAS_OFFICIAL_RLM=False, return False."""
        fake_pkg = SimpleNamespace(HAS_OFFICIAL_RLM=False)
        with patch.dict(sys.modules, {"aragora.debate.context_gatherer": fake_pkg}):
            assert _has_official_rlm() is False

    def test_falls_back_to_constant_when_no_package_attr(self):
        """When the package module lacks HAS_OFFICIAL_RLM attr, fall back to constant."""
        import aragora.debate.context_gatherer.compression as comp_mod

        # Remove the attr from the package so hasattr returns False
        pkg = sys.modules["aragora.debate.context_gatherer"]
        had_attr = hasattr(pkg, "HAS_OFFICIAL_RLM")
        if had_attr:
            saved_val = getattr(pkg, "HAS_OFFICIAL_RLM")
            delattr(pkg, "HAS_OFFICIAL_RLM")
        try:
            # Now it should fall back to the module-level constant
            original = comp_mod.HAS_OFFICIAL_RLM
            try:
                comp_mod.HAS_OFFICIAL_RLM = True
                assert _has_official_rlm() is True
                comp_mod.HAS_OFFICIAL_RLM = False
                assert _has_official_rlm() is False
            finally:
                comp_mod.HAS_OFFICIAL_RLM = original
        finally:
            if had_attr:
                setattr(pkg, "HAS_OFFICIAL_RLM", saved_val)

    def test_truthy_coercion(self):
        """Non-bool truthy values are coerced to bool."""
        fake_pkg = SimpleNamespace(HAS_OFFICIAL_RLM=1)
        with patch.dict(sys.modules, {"aragora.debate.context_gatherer": fake_pkg}):
            assert _has_official_rlm() is True


# ===========================================================================
# _compress_with_rlm
# ===========================================================================


class TestCompressWithRlm:
    """Tests for CompressionMixin._compress_with_rlm."""

    @pytest.mark.asyncio
    async def test_content_under_threshold_returned_as_is(self):
        """Short content (<=threshold) is returned unchanged."""
        obj = ConcreteCompression()
        obj._rlm_threshold = 3000
        content = "short content"
        result = await obj._compress_with_rlm(content, max_chars=3000)
        assert result == content

    @pytest.mark.asyncio
    async def test_content_under_threshold_but_over_max_chars_is_truncated(self):
        """Content within threshold but exceeding max_chars is sliced."""
        obj = ConcreteCompression()
        obj._rlm_threshold = 3000
        content = "a" * 2000
        result = await obj._compress_with_rlm(content, max_chars=500)
        assert len(result) == 500
        assert result == "a" * 500

    @pytest.mark.asyncio
    async def test_rlm_disabled_simple_truncation(self):
        """When RLM is disabled and content is over threshold, simple truncation."""
        obj = ConcreteCompression()
        obj._enable_rlm = False
        obj._rlm_threshold = 100
        content = "b" * 5000
        max_chars = 3000
        result = await obj._compress_with_rlm(content, max_chars=max_chars)
        assert result.endswith("... [truncated]")
        # content[:max_chars - 30] + "... [truncated]" (15 chars suffix)
        suffix = "... [truncated]"
        expected_len = max_chars - 30 + len(suffix)
        assert len(result) == expected_len

    @pytest.mark.asyncio
    async def test_rlm_disabled_content_under_max_chars_but_over_threshold(self):
        """RLM disabled, content over threshold but under max_chars -- returned as-is."""
        obj = ConcreteCompression()
        obj._enable_rlm = False
        obj._rlm_threshold = 100
        content = "c" * 200
        result = await obj._compress_with_rlm(content, max_chars=3000)
        assert result == content

    @pytest.mark.asyncio
    async def test_aragora_rlm_success(self):
        """AragoraRLM compress_and_query success returns result.answer."""
        obj = ConcreteCompression()
        obj._enable_rlm = True
        obj._rlm_threshold = 100
        obj._aragora_rlm = MagicMock()
        obj._aragora_rlm.compress_and_query = AsyncMock(
            return_value=_make_rlm_result("compressed answer")
        )
        content = _long_content(5000)
        result = await obj._compress_with_rlm(content, max_chars=3000)
        assert result == "compressed answer"
        obj._aragora_rlm.compress_and_query.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aragora_rlm_result_truncated_to_max_chars(self):
        """If RLM answer exceeds max_chars, it is truncated."""
        obj = ConcreteCompression()
        obj._enable_rlm = True
        obj._rlm_threshold = 100
        obj._aragora_rlm = MagicMock()
        big_answer = "z" * 5000
        obj._aragora_rlm.compress_and_query = AsyncMock(return_value=_make_rlm_result(big_answer))
        content = _long_content(10000)
        result = await obj._compress_with_rlm(content, max_chars=3000)
        assert len(result) == 3000

    @pytest.mark.asyncio
    async def test_aragora_rlm_empty_answer_falls_through(self):
        """If RLM returns empty answer, falls through to compressor/truncation."""
        obj = ConcreteCompression()
        obj._enable_rlm = True
        obj._rlm_threshold = 100
        obj._aragora_rlm = MagicMock()
        obj._aragora_rlm.compress_and_query = AsyncMock(return_value=_make_rlm_result(""))
        obj._rlm_compressor = None
        content = _long_content(5000)
        result = await obj._compress_with_rlm(content, max_chars=3000)
        # Falls to final truncation
        assert result.endswith("... [truncated]")

    @pytest.mark.asyncio
    async def test_aragora_rlm_answer_not_shorter_falls_through(self):
        """If RLM answer is not shorter than content, falls through."""
        obj = ConcreteCompression()
        obj._enable_rlm = True
        obj._rlm_threshold = 100
        obj._aragora_rlm = MagicMock()
        content = _long_content(5000)
        # Answer same length as content -- not shorter
        obj._aragora_rlm.compress_and_query = AsyncMock(return_value=_make_rlm_result(content))
        obj._rlm_compressor = None
        result = await obj._compress_with_rlm(content, max_chars=3000)
        assert result.endswith("... [truncated]")

    @pytest.mark.asyncio
    async def test_aragora_rlm_timeout_falls_to_compressor(self):
        """AragoraRLM timeout falls through to HierarchicalCompressor."""
        obj = ConcreteCompression()
        obj._enable_rlm = True
        obj._rlm_threshold = 100
        obj._aragora_rlm = MagicMock()
        obj._aragora_rlm.compress_and_query = AsyncMock(side_effect=asyncio.TimeoutError)
        obj._rlm_compressor = None
        content = _long_content(5000)
        result = await obj._compress_with_rlm(content, max_chars=3000)
        assert result.endswith("... [truncated]")

    @pytest.mark.asyncio
    async def test_aragora_rlm_value_error_falls_through(self):
        """AragoraRLM ValueError falls through."""
        obj = ConcreteCompression()
        obj._enable_rlm = True
        obj._rlm_threshold = 100
        obj._aragora_rlm = MagicMock()
        obj._aragora_rlm.compress_and_query = AsyncMock(side_effect=ValueError("bad input"))
        obj._rlm_compressor = None
        content = _long_content(5000)
        result = await obj._compress_with_rlm(content, max_chars=3000)
        assert result.endswith("... [truncated]")

    @pytest.mark.asyncio
    async def test_aragora_rlm_type_error_falls_through(self):
        """AragoraRLM TypeError (unexpected) falls through."""
        obj = ConcreteCompression()
        obj._enable_rlm = True
        obj._rlm_threshold = 100
        obj._aragora_rlm = MagicMock()
        obj._aragora_rlm.compress_and_query = AsyncMock(side_effect=TypeError("bad type"))
        obj._rlm_compressor = None
        content = _long_content(5000)
        result = await obj._compress_with_rlm(content, max_chars=3000)
        assert result.endswith("... [truncated]")

    @pytest.mark.asyncio
    async def test_compressor_success_summary_level(self):
        """HierarchicalCompressor returns SUMMARY level content."""
        obj = ConcreteCompression()
        obj._enable_rlm = True
        obj._rlm_threshold = 100
        obj._aragora_rlm = None  # skip primary path

        mock_context = MagicMock()
        mock_context.get_at_level.return_value = "summary result"
        mock_compression_result = MagicMock()
        mock_compression_result.context = mock_context

        obj._rlm_compressor = MagicMock()
        obj._rlm_compressor.compress = AsyncMock(return_value=mock_compression_result)

        content = _long_content(5000)

        with patch(
            "aragora.debate.context_gatherer.compression.AbstractionLevel",
            create=True,
        ) as mock_level:
            mock_level.SUMMARY = "SUMMARY"
            mock_level.ABSTRACT = "ABSTRACT"
            with patch.dict(
                "sys.modules",
                {"aragora.rlm.types": MagicMock(AbstractionLevel=mock_level)},
            ):
                result = await obj._compress_with_rlm(content, max_chars=3000)

        assert result == "summary result"

    @pytest.mark.asyncio
    async def test_compressor_summary_too_long_falls_to_abstract(self):
        """If SUMMARY exceeds max_chars, tries ABSTRACT level."""
        obj = ConcreteCompression()
        obj._enable_rlm = True
        obj._rlm_threshold = 100
        obj._aragora_rlm = None

        long_summary = "s" * 4000  # exceeds max_chars=3000
        short_abstract = "abstract result"

        mock_context = MagicMock()

        def get_at_level_side_effect(level):
            if level == "SUMMARY":
                return long_summary
            if level == "ABSTRACT":
                return short_abstract
            return None

        mock_context.get_at_level.side_effect = get_at_level_side_effect
        mock_compression_result = MagicMock()
        mock_compression_result.context = mock_context

        obj._rlm_compressor = MagicMock()
        obj._rlm_compressor.compress = AsyncMock(return_value=mock_compression_result)

        content = _long_content(5000)

        with patch(
            "aragora.debate.context_gatherer.compression.AbstractionLevel",
            create=True,
        ) as mock_level:
            mock_level.SUMMARY = "SUMMARY"
            mock_level.ABSTRACT = "ABSTRACT"
            with patch.dict(
                "sys.modules",
                {"aragora.rlm.types": MagicMock(AbstractionLevel=mock_level)},
            ):
                result = await obj._compress_with_rlm(content, max_chars=3000)

        assert result == short_abstract

    @pytest.mark.asyncio
    async def test_compressor_import_error_for_abstraction_level(self):
        """ImportError when importing AbstractionLevel falls to truncation."""
        obj = ConcreteCompression()
        obj._enable_rlm = True
        obj._rlm_threshold = 100
        obj._aragora_rlm = None

        mock_compression_result = MagicMock()

        obj._rlm_compressor = MagicMock()
        obj._rlm_compressor.compress = AsyncMock(return_value=mock_compression_result)

        content = _long_content(5000)

        # Patch the import to raise ImportError
        with patch.dict("sys.modules", {"aragora.rlm.types": None}):
            result = await obj._compress_with_rlm(content, max_chars=3000)

        # summary is None from ImportError -> falls to final truncation
        assert result.endswith("... [truncated]")

    @pytest.mark.asyncio
    async def test_compressor_timeout_falls_to_truncation(self):
        """HierarchicalCompressor timeout falls to final truncation."""
        obj = ConcreteCompression()
        obj._enable_rlm = True
        obj._rlm_threshold = 100
        obj._aragora_rlm = None

        obj._rlm_compressor = MagicMock()
        obj._rlm_compressor.compress = AsyncMock(side_effect=asyncio.TimeoutError)

        content = _long_content(5000)
        result = await obj._compress_with_rlm(content, max_chars=3000)
        assert result.endswith("... [truncated]")

    @pytest.mark.asyncio
    async def test_compressor_runtime_error_falls_to_truncation(self):
        """HierarchicalCompressor RuntimeError falls to final truncation."""
        obj = ConcreteCompression()
        obj._enable_rlm = True
        obj._rlm_threshold = 100
        obj._aragora_rlm = None

        obj._rlm_compressor = MagicMock()
        obj._rlm_compressor.compress = AsyncMock(side_effect=RuntimeError("fail"))

        content = _long_content(5000)
        result = await obj._compress_with_rlm(content, max_chars=3000)
        assert result.endswith("... [truncated]")

    @pytest.mark.asyncio
    async def test_all_rlm_fails_final_truncation(self):
        """When both AragoraRLM and compressor fail, final truncation is used."""
        obj = ConcreteCompression()
        obj._enable_rlm = True
        obj._rlm_threshold = 100
        obj._aragora_rlm = MagicMock()
        obj._aragora_rlm.compress_and_query = AsyncMock(side_effect=RuntimeError("rlm fail"))
        obj._rlm_compressor = MagicMock()
        obj._rlm_compressor.compress = AsyncMock(side_effect=RuntimeError("compressor fail"))
        content = _long_content(5000)
        max_chars = 3000
        result = await obj._compress_with_rlm(content, max_chars=max_chars)
        assert result.endswith("... [truncated]")
        suffix = "... [truncated]"
        expected_len = max_chars - 30 + len(suffix)
        assert len(result) == expected_len


# ===========================================================================
# _query_with_true_rlm
# ===========================================================================


class TestQueryWithTrueRlm:
    """Tests for CompressionMixin._query_with_true_rlm."""

    @pytest.mark.asyncio
    async def test_rlm_disabled_returns_none(self):
        """Returns None when RLM is disabled."""
        obj = ConcreteCompression()
        obj._enable_rlm = False
        result = await obj._query_with_true_rlm("query", "content")
        assert result is None

    @pytest.mark.asyncio
    async def test_no_aragora_rlm_returns_none(self):
        """Returns None when _aragora_rlm is None."""
        obj = ConcreteCompression()
        obj._enable_rlm = True
        obj._aragora_rlm = None
        result = await obj._query_with_true_rlm("query", "content")
        assert result is None

    @pytest.mark.asyncio
    async def test_true_rlm_available_and_succeeds(self):
        """When TRUE RLM is available, uses .query() and returns answer."""
        obj = ConcreteCompression()
        obj._enable_rlm = True
        obj._aragora_rlm = MagicMock()
        obj._aragora_rlm.query = AsyncMock(
            return_value=_make_rlm_result("true rlm answer", used_true_rlm=True)
        )

        with (
            patch("aragora.debate.context_gatherer.compression.HAS_RLM", True),
            patch(
                "aragora.debate.context_gatherer.compression._has_official_rlm",
                return_value=True,
            ),
        ):
            result = await obj._query_with_true_rlm("query", "content")

        assert result == "true rlm answer"
        obj._aragora_rlm.query.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_true_rlm_query_returns_non_true_rlm_falls_to_compress(self):
        """If .query() result has used_true_rlm=False, falls to compress_and_query."""
        obj = ConcreteCompression()
        obj._enable_rlm = True
        obj._aragora_rlm = MagicMock()
        obj._aragora_rlm.query = AsyncMock(
            return_value=_make_rlm_result("not true", used_true_rlm=False)
        )
        obj._aragora_rlm.compress_and_query = AsyncMock(
            return_value=_make_rlm_result("fallback answer")
        )

        with (
            patch("aragora.debate.context_gatherer.compression.HAS_RLM", True),
            patch(
                "aragora.debate.context_gatherer.compression._has_official_rlm",
                return_value=True,
            ),
        ):
            result = await obj._query_with_true_rlm("query", "content")

        assert result == "fallback answer"
        obj._aragora_rlm.compress_and_query.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_true_rlm_not_available_uses_compress_and_query(self):
        """When TRUE RLM not available (HAS_RLM=False), uses compress_and_query."""
        obj = ConcreteCompression()
        obj._enable_rlm = True
        obj._aragora_rlm = MagicMock()
        obj._aragora_rlm.compress_and_query = AsyncMock(
            return_value=_make_rlm_result("compressed answer")
        )

        with patch("aragora.debate.context_gatherer.compression.HAS_RLM", False):
            result = await obj._query_with_true_rlm("query", "content")

        assert result == "compressed answer"
        obj._aragora_rlm.compress_and_query.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self):
        """Timeout during TRUE RLM query returns None."""
        obj = ConcreteCompression()
        obj._enable_rlm = True
        obj._aragora_rlm = MagicMock()
        obj._aragora_rlm.compress_and_query = AsyncMock(side_effect=asyncio.TimeoutError)

        with patch("aragora.debate.context_gatherer.compression.HAS_RLM", False):
            result = await obj._query_with_true_rlm("query", "content")

        assert result is None

    @pytest.mark.asyncio
    async def test_value_error_returns_none(self):
        """ValueError during query returns None."""
        obj = ConcreteCompression()
        obj._enable_rlm = True
        obj._aragora_rlm = MagicMock()
        obj._aragora_rlm.compress_and_query = AsyncMock(side_effect=ValueError("bad"))

        with patch("aragora.debate.context_gatherer.compression.HAS_RLM", False):
            result = await obj._query_with_true_rlm("query", "content")

        assert result is None

    @pytest.mark.asyncio
    async def test_attribute_error_returns_none(self):
        """AttributeError during query returns None."""
        obj = ConcreteCompression()
        obj._enable_rlm = True
        obj._aragora_rlm = MagicMock()
        obj._aragora_rlm.compress_and_query = AsyncMock(side_effect=AttributeError("missing"))

        with patch("aragora.debate.context_gatherer.compression.HAS_RLM", False):
            result = await obj._query_with_true_rlm("query", "content")

        assert result is None

    @pytest.mark.asyncio
    async def test_compress_and_query_empty_answer_returns_none(self):
        """compress_and_query returning empty answer returns None."""
        obj = ConcreteCompression()
        obj._enable_rlm = True
        obj._aragora_rlm = MagicMock()
        obj._aragora_rlm.compress_and_query = AsyncMock(return_value=_make_rlm_result(""))

        with patch("aragora.debate.context_gatherer.compression.HAS_RLM", False):
            result = await obj._query_with_true_rlm("query", "content")

        assert result is None


# ===========================================================================
# query_knowledge_with_true_rlm
# ===========================================================================


class TestQueryKnowledgeWithTrueRlm:
    """Tests for CompressionMixin.query_knowledge_with_true_rlm."""

    @pytest.mark.asyncio
    async def test_knowledge_grounding_disabled_returns_none(self):
        """Returns None when knowledge grounding is disabled."""
        obj = ConcreteCompression()
        obj._enable_knowledge_grounding = False
        result = await obj.query_knowledge_with_true_rlm("task")
        assert result is None

    @pytest.mark.asyncio
    async def test_no_knowledge_mound_returns_none(self):
        """Returns None when knowledge_mound is None."""
        obj = ConcreteCompression()
        obj._enable_knowledge_grounding = True
        obj._knowledge_mound = None
        result = await obj.query_knowledge_with_true_rlm("task")
        assert result is None

    @pytest.mark.asyncio
    async def test_true_rlm_not_available_delegates_to_standard(self):
        """When TRUE RLM not available, delegates to gather_knowledge_mound_context."""
        obj = ConcreteCompression()
        obj._enable_knowledge_grounding = True
        obj._knowledge_mound = MagicMock()
        obj.gather_knowledge_mound_context = AsyncMock(return_value="standard result")

        with patch("aragora.debate.context_gatherer.compression.HAS_RLM", False):
            result = await obj.query_knowledge_with_true_rlm("task")

        assert result == "standard result"
        obj.gather_knowledge_mound_context.assert_awaited_once_with("task")

    @pytest.mark.asyncio
    async def test_true_rlm_available_creates_repl_env(self):
        """When TRUE RLM available, creates REPL env and returns formatted prompt."""
        obj = ConcreteCompression()
        obj._enable_knowledge_grounding = True
        obj._knowledge_mound = MagicMock()

        mock_adapter = MagicMock()
        mock_adapter.create_repl_for_knowledge.return_value = {"env": "ok"}
        mock_adapter.get_repl_prompt.return_value = "Use search_km() to query."

        with (
            patch("aragora.debate.context_gatherer.compression.HAS_RLM", True),
            patch(
                "aragora.debate.context_gatherer.compression._has_official_rlm",
                return_value=True,
            ),
            patch(
                "aragora.rlm.get_repl_adapter",
                return_value=mock_adapter,
            ),
        ):
            result = await obj.query_knowledge_with_true_rlm("my task")

        assert result is not None
        assert "KNOWLEDGE MOUND CONTEXT (TRUE RLM)" in result
        assert "Use search_km() to query." in result

        # Verify adapter was called with correct args
        task_hash = obj._get_task_hash("my task")
        mock_adapter.create_repl_for_knowledge.assert_called_once_with(
            mound=obj._knowledge_mound,
            workspace_id="test-ws",
            content_id=f"km_{task_hash}",
        )
        mock_adapter.get_repl_prompt.assert_called_once_with(
            content_id=f"km_{task_hash}",
            content_type="knowledge",
        )

    @pytest.mark.asyncio
    async def test_repl_creation_returns_none_falls_to_standard(self):
        """If create_repl_for_knowledge returns None, falls back to standard."""
        obj = ConcreteCompression()
        obj._enable_knowledge_grounding = True
        obj._knowledge_mound = MagicMock()
        obj.gather_knowledge_mound_context = AsyncMock(return_value="fallback km")

        mock_adapter = MagicMock()
        mock_adapter.create_repl_for_knowledge.return_value = None

        with (
            patch("aragora.debate.context_gatherer.compression.HAS_RLM", True),
            patch(
                "aragora.debate.context_gatherer.compression._has_official_rlm",
                return_value=True,
            ),
            patch(
                "aragora.rlm.get_repl_adapter",
                return_value=mock_adapter,
            ),
        ):
            result = await obj.query_knowledge_with_true_rlm("task")

        assert result == "fallback km"
        obj.gather_knowledge_mound_context.assert_awaited_once_with("task")

    @pytest.mark.asyncio
    async def test_import_error_on_get_repl_adapter_falls_to_standard(self):
        """ImportError when importing get_repl_adapter falls back."""
        obj = ConcreteCompression()
        obj._enable_knowledge_grounding = True
        obj._knowledge_mound = MagicMock()
        obj.gather_knowledge_mound_context = AsyncMock(return_value="import fallback")

        with (
            patch("aragora.debate.context_gatherer.compression.HAS_RLM", True),
            patch(
                "aragora.debate.context_gatherer.compression._has_official_rlm",
                return_value=True,
            ),
            patch(
                "builtins.__import__",
                side_effect=_make_import_raiser("aragora.rlm"),
            ),
        ):
            result = await obj.query_knowledge_with_true_rlm("task")

        assert result == "import fallback"

    @pytest.mark.asyncio
    async def test_runtime_error_in_repl_falls_to_standard(self):
        """RuntimeError during REPL creation falls back to standard query."""
        obj = ConcreteCompression()
        obj._enable_knowledge_grounding = True
        obj._knowledge_mound = MagicMock()
        obj.gather_knowledge_mound_context = AsyncMock(return_value="runtime fallback")

        mock_adapter = MagicMock()
        mock_adapter.create_repl_for_knowledge.side_effect = RuntimeError("boom")

        with (
            patch("aragora.debate.context_gatherer.compression.HAS_RLM", True),
            patch(
                "aragora.debate.context_gatherer.compression._has_official_rlm",
                return_value=True,
            ),
            patch(
                "aragora.rlm.get_repl_adapter",
                return_value=mock_adapter,
            ),
        ):
            result = await obj.query_knowledge_with_true_rlm("task")

        assert result == "runtime fallback"

    @pytest.mark.asyncio
    async def test_has_rlm_true_but_official_false_delegates_standard(self):
        """HAS_RLM=True but _has_official_rlm()=False delegates to standard."""
        obj = ConcreteCompression()
        obj._enable_knowledge_grounding = True
        obj._knowledge_mound = MagicMock()
        obj.gather_knowledge_mound_context = AsyncMock(return_value="no official")

        with (
            patch("aragora.debate.context_gatherer.compression.HAS_RLM", True),
            patch(
                "aragora.debate.context_gatherer.compression._has_official_rlm",
                return_value=False,
            ),
        ):
            result = await obj.query_knowledge_with_true_rlm("task")

        assert result == "no official"
        obj.gather_knowledge_mound_context.assert_awaited_once_with("task")


# ---------------------------------------------------------------------------
# Helper for import error simulation
# ---------------------------------------------------------------------------

_real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__


def _make_import_raiser(blocked_module: str):
    """Return an __import__ replacement that raises ImportError for a module."""

    def _import(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"No module named '{name}'")
        return _real_import(name, *args, **kwargs)

    return _import
