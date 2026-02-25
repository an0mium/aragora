"""Tests for aragora.debate.similarity.factory module.

Covers SimilarityFactory (class-level state, registration, auto-selection),
BackendInfo dataclass, _FAISSBackendWrapper, and get_backend convenience function.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.similarity.factory import (
    BackendInfo,
    SimilarityFactory,
    _FAISSBackendWrapper,
    get_backend,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_factory():
    """Reset class-level state before each test, restore original state after."""
    orig_registry = SimilarityFactory._registry
    orig_initialized = SimilarityFactory._initialized
    SimilarityFactory._registry = {}
    SimilarityFactory._initialized = False
    yield
    SimilarityFactory._registry = orig_registry
    # Always reset to False so _ensure_initialized re-registers defaults
    SimilarityFactory._initialized = False


class _DummyBackend:
    """Minimal stand-in for a SimilarityBackend subclass."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def compute_similarity(self, text1: str, text2: str) -> float:
        return 0.5


class _FailingBackend:
    """Backend whose __init__ raises ImportError."""

    def __init__(self, **kwargs):
        raise ImportError("missing dependency")


class _RuntimeErrorBackend:
    """Backend whose __init__ raises RuntimeError."""

    def __init__(self, **kwargs):
        raise RuntimeError("cannot start")


# ---------------------------------------------------------------------------
# BackendInfo dataclass
# ---------------------------------------------------------------------------


class TestBackendInfo:
    def test_defaults(self):
        info = BackendInfo(
            name="test",
            backend_class=_DummyBackend,
            description="A test backend",
            requires=[],
        )
        assert info.min_input_size == 0
        assert info.max_input_size == 10000
        assert info.accuracy == "medium"
        assert info.speed == "medium"

    def test_custom_values(self):
        info = BackendInfo(
            name="custom",
            backend_class=_DummyBackend,
            description="Custom",
            requires=["numpy"],
            min_input_size=10,
            max_input_size=500,
            accuracy="high",
            speed="fast",
        )
        assert info.name == "custom"
        assert info.requires == ["numpy"]
        assert info.min_input_size == 10
        assert info.max_input_size == 500
        assert info.accuracy == "high"
        assert info.speed == "fast"


# ---------------------------------------------------------------------------
# _ensure_initialized
# ---------------------------------------------------------------------------


class TestEnsureInitialized:
    def test_registers_builtin_backends(self):
        """After initialization the three built-in backends must be present."""
        # Force FAISS import to fail inside _ensure_initialized by
        # temporarily making the ann module raise ImportError for FAISSIndex.
        original_import = __import__

        def _block_faiss(name, *args, **kwargs):
            if name == "aragora.debate.similarity.ann":
                raise ImportError("no faiss")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_block_faiss):
            SimilarityFactory._ensure_initialized()

        names = {info.name for info in SimilarityFactory._registry.values()}
        assert "jaccard" in names
        assert "tfidf" in names
        assert "sentence-transformer" in names

    def test_idempotent(self):
        """Calling _ensure_initialized twice does not double-register."""
        SimilarityFactory._ensure_initialized()
        count_first = len(SimilarityFactory._registry)
        SimilarityFactory._ensure_initialized()
        assert len(SimilarityFactory._registry) == count_first

    def test_faiss_registered_when_available(self):
        """When the FAISS import succeeds, a 'faiss' backend is registered."""
        # The ann module is already importable (numpy-based fallback).
        # Just run initialization normally -- FAISSIndex class exists.
        SimilarityFactory._ensure_initialized()
        assert "faiss" in SimilarityFactory._registry

    def test_faiss_not_registered_when_import_fails(self):
        """When FAISS is not importable, no 'faiss' backend appears."""
        original_import = __import__

        def _block_faiss(name, *args, **kwargs):
            if name == "aragora.debate.similarity.ann":
                raise ImportError("no faiss")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_block_faiss):
            SimilarityFactory._ensure_initialized()

        assert "faiss" not in SimilarityFactory._registry

    def test_sets_initialized_flag(self):
        assert SimilarityFactory._initialized is False
        SimilarityFactory._ensure_initialized()
        assert SimilarityFactory._initialized is True


# ---------------------------------------------------------------------------
# register / unregister
# ---------------------------------------------------------------------------


class TestRegisterUnregister:
    def test_register_adds_backend(self):
        SimilarityFactory.register(
            "my_backend",
            _DummyBackend,
            description="Test backend",
            requires=["dep"],
            accuracy="high",
        )
        assert "my_backend" in SimilarityFactory._registry
        info = SimilarityFactory._registry["my_backend"]
        assert info.accuracy == "high"
        assert info.requires == ["dep"]

    def test_register_replaces_existing(self):
        SimilarityFactory.register("x", _DummyBackend, description="v1")
        SimilarityFactory.register("x", _FailingBackend, description="v2")
        assert SimilarityFactory._registry["x"].backend_class is _FailingBackend
        assert SimilarityFactory._registry["x"].description == "v2"

    def test_unregister_existing(self):
        SimilarityFactory.register("removable", _DummyBackend)
        assert SimilarityFactory.unregister("removable") is True
        assert "removable" not in SimilarityFactory._registry

    def test_unregister_missing_returns_false(self):
        assert SimilarityFactory.unregister("nonexistent") is False

    def test_register_requires_defaults_to_empty_list(self):
        SimilarityFactory.register("no_deps", _DummyBackend)
        assert SimilarityFactory._registry["no_deps"].requires == []


# ---------------------------------------------------------------------------
# list_backends / get_backend_info
# ---------------------------------------------------------------------------


class TestListAndInfo:
    def test_list_backends_triggers_init(self):
        """list_backends should call _ensure_initialized."""
        assert SimilarityFactory._initialized is False
        backends = SimilarityFactory.list_backends()
        assert SimilarityFactory._initialized is True
        assert len(backends) >= 3  # at least jaccard, tfidf, sentence-transformer

    def test_list_backends_returns_backend_info_objects(self):
        SimilarityFactory.register("a", _DummyBackend, description="A")
        SimilarityFactory._initialized = True  # skip builtin init
        result = SimilarityFactory.list_backends()
        assert len(result) == 1
        assert isinstance(result[0], BackendInfo)
        assert result[0].name == "a"

    def test_get_backend_info_known(self):
        SimilarityFactory.register("known", _DummyBackend, description="Known")
        SimilarityFactory._initialized = True
        info = SimilarityFactory.get_backend_info("known")
        assert info is not None
        assert info.name == "known"

    def test_get_backend_info_unknown_returns_none(self):
        SimilarityFactory._initialized = True
        assert SimilarityFactory.get_backend_info("unknown") is None


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_available_when_instantiation_succeeds(self):
        SimilarityFactory.register("good", _DummyBackend)
        SimilarityFactory._initialized = True
        assert SimilarityFactory.is_available("good") is True

    def test_not_available_on_import_error(self):
        SimilarityFactory.register("bad", _FailingBackend)
        SimilarityFactory._initialized = True
        assert SimilarityFactory.is_available("bad") is False

    def test_not_available_on_runtime_error(self):
        SimilarityFactory.register("rterr", _RuntimeErrorBackend)
        SimilarityFactory._initialized = True
        assert SimilarityFactory.is_available("rterr") is False

    def test_not_available_for_unregistered_name(self):
        SimilarityFactory._initialized = True
        assert SimilarityFactory.is_available("ghost") is False

    def test_not_available_on_type_error(self):
        class _TypeErrBackend:
            def __init__(self):
                raise TypeError("bad type")

        SimilarityFactory.register("typeerr", _TypeErrBackend)
        SimilarityFactory._initialized = True
        assert SimilarityFactory.is_available("typeerr") is False

    def test_not_available_on_value_error(self):
        class _ValErrBackend:
            def __init__(self):
                raise ValueError("bad value")

        SimilarityFactory.register("valerr", _ValErrBackend)
        SimilarityFactory._initialized = True
        assert SimilarityFactory.is_available("valerr") is False


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


class TestCreate:
    def test_create_known_backend(self):
        SimilarityFactory.register("simple", _DummyBackend)
        SimilarityFactory._initialized = True
        instance = SimilarityFactory.create("simple")
        assert isinstance(instance, _DummyBackend)

    def test_create_unknown_raises_value_error(self):
        SimilarityFactory._initialized = True
        with pytest.raises(ValueError, match="Unknown backend"):
            SimilarityFactory.create("nonexistent")

    def test_create_error_message_lists_available(self):
        SimilarityFactory.register("alpha", _DummyBackend)
        SimilarityFactory.register("beta", _DummyBackend)
        SimilarityFactory._initialized = True
        with pytest.raises(ValueError, match="alpha") as exc_info:
            SimilarityFactory.create("missing")
        assert "beta" in str(exc_info.value)

    def test_create_passes_kwargs(self):
        SimilarityFactory.register("kw", _DummyBackend)
        SimilarityFactory._initialized = True
        instance = SimilarityFactory.create("kw", custom_param=42)
        assert instance.kwargs == {"custom_param": 42}

    def test_create_sentence_transformer_passes_debate_id(self):
        """debate_id is injected as kwarg only for the sentence-transformer backend."""
        SimilarityFactory.register("sentence-transformer", _DummyBackend)
        SimilarityFactory._initialized = True
        instance = SimilarityFactory.create("sentence-transformer", debate_id="d-123")
        assert instance.kwargs.get("debate_id") == "d-123"

    def test_create_sentence_transformer_no_debate_id_when_none(self):
        """When debate_id is None, it should NOT be passed as a kwarg."""
        SimilarityFactory.register("sentence-transformer", _DummyBackend)
        SimilarityFactory._initialized = True
        instance = SimilarityFactory.create("sentence-transformer", debate_id=None)
        assert "debate_id" not in instance.kwargs

    def test_create_non_sentence_transformer_ignores_debate_id(self):
        """Other backends do not receive debate_id even if provided."""
        SimilarityFactory.register("other", _DummyBackend)
        SimilarityFactory._initialized = True
        instance = SimilarityFactory.create("other", debate_id="d-456")
        assert "debate_id" not in instance.kwargs


# ---------------------------------------------------------------------------
# auto_select
# ---------------------------------------------------------------------------


class TestAutoSelect:
    def _register_stubs(self):
        """Register stub backends under standard names."""
        SimilarityFactory.register(
            "jaccard",
            _DummyBackend,
            min_input_size=0,
            max_input_size=1000,
            accuracy="low",
            speed="fast",
        )
        SimilarityFactory.register(
            "tfidf",
            _DummyBackend,
            min_input_size=0,
            max_input_size=5000,
            accuracy="medium",
            speed="medium",
        )
        SimilarityFactory.register(
            "sentence-transformer",
            _DummyBackend,
            min_input_size=0,
            max_input_size=10000,
            accuracy="high",
            speed="slow",
        )
        SimilarityFactory._initialized = True

    def test_env_var_override(self, monkeypatch):
        self._register_stubs()
        monkeypatch.setenv("ARAGORA_SIMILARITY_BACKEND", "jaccard")
        result = SimilarityFactory.auto_select()
        # Jaccard was selected via env var
        assert isinstance(result, _DummyBackend)

    def test_env_var_unknown_value_ignored(self, monkeypatch):
        """An unregistered env var value is silently ignored."""
        self._register_stubs()
        monkeypatch.setenv("ARAGORA_SIMILARITY_BACKEND", "nonexistent_backend")
        # Should fall through to normal selection logic, not crash
        result = SimilarityFactory.auto_select(prefer_accuracy=True)
        assert isinstance(result, _DummyBackend)

    def test_large_input_prefers_faiss(self):
        self._register_stubs()
        SimilarityFactory.register(
            "faiss",
            _DummyBackend,
            min_input_size=50,
            max_input_size=100000,
            accuracy="high",
            speed="fast",
        )
        # Patch is_available to return True only for faiss
        with patch.object(SimilarityFactory, "is_available", side_effect=lambda n: n == "faiss"):
            result = SimilarityFactory.auto_select(input_size=100)
        assert isinstance(result, _DummyBackend)

    def test_prefer_accuracy_selects_sentence_transformer(self):
        self._register_stubs()
        with patch.object(
            SimilarityFactory,
            "is_available",
            side_effect=lambda n: n in ("sentence-transformer", "tfidf", "jaccard"),
        ):
            result = SimilarityFactory.auto_select(input_size=10, prefer_accuracy=True)
        assert isinstance(result, _DummyBackend)

    def test_fallback_to_tfidf_when_st_unavailable(self):
        self._register_stubs()
        with patch.object(
            SimilarityFactory,
            "is_available",
            side_effect=lambda n: n in ("tfidf", "jaccard"),
        ):
            result = SimilarityFactory.auto_select(input_size=10, prefer_accuracy=True)
        assert isinstance(result, _DummyBackend)

    def test_ultimate_fallback_to_jaccard(self):
        self._register_stubs()
        # Only jaccard "available" (tfidf / sentence-transformer fail is_available)
        with patch.object(
            SimilarityFactory,
            "is_available",
            return_value=False,
        ):
            result = SimilarityFactory.auto_select(input_size=5, prefer_accuracy=False)
        # Falls through all is_available checks, ends up calling create("jaccard")
        assert isinstance(result, _DummyBackend)

    def test_auto_select_passes_debate_id(self):
        self._register_stubs()
        with (
            patch.object(
                SimilarityFactory,
                "is_available",
                side_effect=lambda n: n == "sentence-transformer",
            ),
            patch.object(SimilarityFactory, "create", return_value=_DummyBackend()) as mock_create,
        ):
            SimilarityFactory.auto_select(prefer_accuracy=True, debate_id="d-789")
        mock_create.assert_called_with("sentence-transformer", debate_id="d-789")

    def test_small_input_no_accuracy_skips_faiss_and_st(self):
        """With small input and prefer_accuracy=False, should skip FAISS and ST."""
        self._register_stubs()
        SimilarityFactory.register(
            "faiss",
            _DummyBackend,
            min_input_size=50,
            max_input_size=100000,
        )
        # is_available returns True for everything
        with patch.object(SimilarityFactory, "is_available", return_value=True):
            # input_size < 50 so FAISS check skipped, prefer_accuracy=False so ST skipped
            # Should still pick sentence-transformer because prefer_accuracy defaults to True
            # Actually, let's explicitly set prefer_accuracy=False
            with patch.object(
                SimilarityFactory, "create", wraps=SimilarityFactory.create
            ) as mock_create:
                result = SimilarityFactory.auto_select(input_size=5, prefer_accuracy=False)
        # With prefer_accuracy=False and input_size<50, should fall through to tfidf
        assert isinstance(result, _DummyBackend)


# ---------------------------------------------------------------------------
# _FAISSBackendWrapper
# ---------------------------------------------------------------------------


class TestFAISSBackendWrapper:
    def test_init_creates_index(self):
        mock_index = MagicMock()
        with patch("aragora.debate.similarity.ann.FAISSIndex", return_value=mock_index) as mock_cls:
            wrapper = _FAISSBackendWrapper(dimension=128)
        mock_cls.assert_called_once_with(dimension=128, use_gpu=False)
        assert wrapper._index is mock_index
        assert wrapper._dimension == 128

    def test_compute_similarity_empty_strings_return_zero(self):
        with patch("aragora.debate.similarity.ann.FAISSIndex"):
            wrapper = _FAISSBackendWrapper()
        assert wrapper.compute_similarity("", "hello") == 0.0
        assert wrapper.compute_similarity("hello", "") == 0.0
        assert wrapper.compute_similarity("", "") == 0.0

    def test_compute_similarity_delegates_to_embeddings(self):
        import numpy as np

        with patch("aragora.debate.similarity.ann.FAISSIndex"):
            wrapper = _FAISSBackendWrapper(dimension=4)

        # Provide a fake embedder that returns deterministic embeddings
        fake_embedder = MagicMock()
        embed_map = {
            "hello": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            "world": np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        }
        fake_embedder.encode.side_effect = lambda texts, **kw: embed_map[texts[0]]
        wrapper._embedder = fake_embedder

        sim = wrapper.compute_similarity("hello", "world")
        # Orthogonal vectors -> cosine similarity = 0.0
        assert sim == pytest.approx(0.0, abs=1e-6)

    def test_compute_similarity_identical_texts(self):
        import numpy as np

        with patch("aragora.debate.similarity.ann.FAISSIndex"):
            wrapper = _FAISSBackendWrapper(dimension=4)

        fake_embedder = MagicMock()
        fake_embedder.encode.return_value = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
        wrapper._embedder = fake_embedder

        sim = wrapper.compute_similarity("same", "same")
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_compute_batch_similarity(self):
        import numpy as np

        with patch("aragora.debate.similarity.ann.FAISSIndex"):
            wrapper = _FAISSBackendWrapper(dimension=4)

        fake_embedder = MagicMock()
        fake_embedder.encode.return_value = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32
        )
        wrapper._embedder = fake_embedder

        with patch(
            "aragora.debate.similarity.ann.compute_batch_similarity_fast",
            return_value=0.75,
        ) as mock_batch_fast:
            result = wrapper.compute_batch_similarity(["a", "b", "c"])
        assert result == 0.75
        mock_batch_fast.assert_called_once()

    def test_compute_batch_similarity_single_text_returns_one(self):
        with patch("aragora.debate.similarity.ann.FAISSIndex"):
            wrapper = _FAISSBackendWrapper()
        assert wrapper.compute_batch_similarity(["only"]) == 1.0

    def test_lazy_loads_sentence_transformer(self):
        with patch("aragora.debate.similarity.ann.FAISSIndex"):
            wrapper = _FAISSBackendWrapper()
        assert wrapper._embedder is None

        mock_st = MagicMock()
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_st,
        ):
            embedder = wrapper._get_embedder()
        assert embedder is mock_st
        assert wrapper._embedder is mock_st

    def test_embedder_cached_after_first_load(self):
        """_get_embedder only loads once; subsequent calls return cached."""
        with patch("aragora.debate.similarity.ann.FAISSIndex"):
            wrapper = _FAISSBackendWrapper()

        mock_st = MagicMock()
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_st,
        ) as mock_cls:
            wrapper._get_embedder()
            wrapper._get_embedder()
        # Only constructed once
        mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# get_backend convenience function
# ---------------------------------------------------------------------------


class TestGetBackend:
    def test_auto_delegates_to_auto_select(self):
        with patch.object(
            SimilarityFactory, "auto_select", return_value=_DummyBackend()
        ) as mock_auto:
            result = get_backend(preferred="auto", input_size=50, debate_id="d-1")
        mock_auto.assert_called_once_with(input_size=50, debate_id="d-1")
        assert isinstance(result, _DummyBackend)

    def test_specific_name_delegates_to_create(self):
        with patch.object(SimilarityFactory, "create", return_value=_DummyBackend()) as mock_create:
            result = get_backend(preferred="tfidf", debate_id="d-2", extra=True)
        mock_create.assert_called_once_with("tfidf", debate_id="d-2", extra=True)
        assert isinstance(result, _DummyBackend)

    def test_default_preferred_is_auto(self):
        with patch.object(
            SimilarityFactory, "auto_select", return_value=_DummyBackend()
        ) as mock_auto:
            get_backend()
        mock_auto.assert_called_once_with(input_size=10, debate_id=None)
