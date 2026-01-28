"""Tests for SimilarityFactory with registration and auto-selection."""

import pytest
from unittest.mock import patch, MagicMock

from tests.conftest import requires_sklearn, REQUIRES_SKLEARN

from aragora.debate.similarity.factory import (
    SimilarityFactory,
    BackendInfo,
    get_backend,
)
from aragora.debate.similarity.backends import (
    SimilarityBackend,
    JaccardBackend,
    TFIDFBackend,
)


class TestBackendInfo:
    """Tests for BackendInfo dataclass."""

    def test_creation(self):
        """Test creating BackendInfo."""
        info = BackendInfo(
            name="test",
            backend_class=JaccardBackend,
            description="Test backend",
            requires=[],
            min_input_size=0,
            max_input_size=100,
            accuracy="low",
            speed="fast",
        )
        assert info.name == "test"
        assert info.backend_class == JaccardBackend
        assert info.accuracy == "low"

    def test_defaults(self):
        """Test BackendInfo default values."""
        info = BackendInfo(
            name="test",
            backend_class=JaccardBackend,
            description="Test",
            requires=[],
        )
        assert info.min_input_size == 0
        assert info.max_input_size == 10000
        assert info.accuracy == "medium"
        assert info.speed == "medium"


class TestSimilarityFactory:
    """Tests for SimilarityFactory class."""

    @pytest.fixture(autouse=True)
    def reset_factory(self):
        """Reset factory state before each test."""
        # Clear registry and reset initialization flag
        SimilarityFactory._registry.clear()
        SimilarityFactory._initialized = False
        yield

    def test_ensure_initialized(self):
        """Test factory auto-initializes with default backends."""
        SimilarityFactory._ensure_initialized()
        assert SimilarityFactory._initialized is True
        assert "jaccard" in SimilarityFactory._registry

    def test_register_backend(self):
        """Test registering a custom backend."""

        class CustomBackend(SimilarityBackend):
            def compute_similarity(self, text1: str, text2: str) -> float:
                return 0.5

        SimilarityFactory.register(
            "custom",
            CustomBackend,
            description="Custom test backend",
            requires=[],
            accuracy="high",
            speed="slow",
        )

        assert "custom" in SimilarityFactory._registry
        info = SimilarityFactory._registry["custom"]
        assert info.accuracy == "high"
        assert info.speed == "slow"

    def test_unregister_backend(self):
        """Test unregistering a backend."""
        SimilarityFactory._ensure_initialized()

        # Unregister jaccard
        result = SimilarityFactory.unregister("jaccard")
        assert result is True
        assert "jaccard" not in SimilarityFactory._registry

        # Unregister non-existent
        result = SimilarityFactory.unregister("nonexistent")
        assert result is False

    def test_list_backends(self):
        """Test listing all backends."""
        backends = SimilarityFactory.list_backends()

        assert len(backends) >= 2  # At least jaccard and tfidf
        names = [b.name for b in backends]
        assert "jaccard" in names
        assert "tfidf" in names

    def test_get_backend_info(self):
        """Test getting info for specific backend."""
        info = SimilarityFactory.get_backend_info("jaccard")
        assert info is not None
        assert info.name == "jaccard"
        assert "Jaccard" in info.description

        # Non-existent
        info = SimilarityFactory.get_backend_info("nonexistent")
        assert info is None

    def test_create_jaccard_backend(self):
        """Test creating jaccard backend."""
        backend = SimilarityFactory.create("jaccard")
        assert isinstance(backend, JaccardBackend)

    def test_create_tfidf_backend(self):
        """Test creating tfidf backend."""
        backend = SimilarityFactory.create("tfidf")
        assert isinstance(backend, TFIDFBackend)

    def test_create_unknown_backend_raises(self):
        """Test creating unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            SimilarityFactory.create("nonexistent")

    def test_is_available_jaccard(self):
        """Test checking if jaccard is available."""
        # Jaccard has no dependencies
        assert SimilarityFactory.is_available("jaccard") is True

    def test_is_available_nonexistent(self):
        """Test checking if nonexistent backend is available."""
        assert SimilarityFactory.is_available("nonexistent") is False

    def test_auto_select_returns_backend(self):
        """Test auto_select returns a valid backend."""
        backend = SimilarityFactory.auto_select(input_size=5)
        assert backend is not None
        assert hasattr(backend, "compute_similarity")

    def test_auto_select_prefers_accuracy(self):
        """Test auto_select with prefer_accuracy=True."""
        # Without sentence-transformer, should fall back to tfidf or jaccard
        backend = SimilarityFactory.auto_select(
            input_size=5,
            prefer_accuracy=True,
        )
        assert backend is not None

    @patch.dict("os.environ", {"ARAGORA_SIMILARITY_BACKEND": "jaccard"})
    def test_auto_select_respects_env_override(self):
        """Test auto_select respects ARAGORA_SIMILARITY_BACKEND env var."""
        backend = SimilarityFactory.auto_select()
        assert isinstance(backend, JaccardBackend)


class TestGetBackend:
    """Tests for get_backend convenience function."""

    @pytest.fixture(autouse=True)
    def reset_factory(self):
        """Reset factory state before each test."""
        SimilarityFactory._registry.clear()
        SimilarityFactory._initialized = False
        yield

    def test_get_backend_auto(self):
        """Test get_backend with auto selection."""
        backend = get_backend(preferred="auto")
        assert backend is not None

    def test_get_backend_specific(self):
        """Test get_backend with specific backend."""
        backend = get_backend(preferred="jaccard")
        assert isinstance(backend, JaccardBackend)

    def test_get_backend_with_input_size(self):
        """Test get_backend uses input_size for auto selection."""
        backend = get_backend(preferred="auto", input_size=100)
        assert backend is not None

    def test_get_backend_with_debate_id(self):
        """Test get_backend passes debate_id for caching."""
        backend = get_backend(preferred="jaccard", debate_id="test-123")
        assert backend is not None


class TestBackendFunctionality:
    """Tests for actual similarity computation via factory-created backends."""

    @pytest.fixture(autouse=True)
    def reset_factory(self):
        """Reset factory state before each test."""
        SimilarityFactory._registry.clear()
        SimilarityFactory._initialized = False
        yield

    def test_jaccard_compute_similarity(self):
        """Test jaccard backend computes similarity."""
        backend = SimilarityFactory.create("jaccard")
        sim = backend.compute_similarity("hello world", "hello there")
        assert 0.0 <= sim <= 1.0

    def test_jaccard_identical_texts(self):
        """Test jaccard returns 1.0 for identical texts."""
        backend = SimilarityFactory.create("jaccard")
        sim = backend.compute_similarity("hello world", "hello world")
        assert sim == 1.0

    def test_jaccard_empty_texts(self):
        """Test jaccard handles empty texts."""
        backend = SimilarityFactory.create("jaccard")
        sim = backend.compute_similarity("", "")
        assert sim == 0.0

    def test_tfidf_compute_similarity(self):
        """Test tfidf backend computes similarity."""
        backend = SimilarityFactory.create("tfidf")
        sim = backend.compute_similarity(
            "The quick brown fox jumps over the lazy dog",
            "A fast brown fox leaps over a sleepy dog",
        )
        assert 0.0 <= sim <= 1.0
        # Should have some similarity due to shared words
        assert sim > 0.1

    def test_tfidf_identical_texts(self):
        """Test tfidf returns 1.0 for identical texts."""
        backend = SimilarityFactory.create("tfidf")
        sim = backend.compute_similarity(
            "The quick brown fox",
            "The quick brown fox",
        )
        assert abs(sim - 1.0) < 0.01


class TestFactoryExtensibility:
    """Tests for factory extensibility with custom backends."""

    @pytest.fixture(autouse=True)
    def reset_factory(self):
        """Reset factory state before each test."""
        SimilarityFactory._registry.clear()
        SimilarityFactory._initialized = False
        yield

    def test_register_and_use_custom_backend(self):
        """Test registering and using a custom backend."""

        class ConstantBackend(SimilarityBackend):
            """Backend that always returns 0.42."""

            def compute_similarity(self, text1: str, text2: str) -> float:
                return 0.42

        SimilarityFactory.register(
            "constant",
            ConstantBackend,
            description="Always returns 0.42",
            requires=[],
        )

        backend = SimilarityFactory.create("constant")
        assert backend.compute_similarity("any", "text") == 0.42

    def test_override_builtin_backend(self):
        """Test overriding a built-in backend."""
        SimilarityFactory._ensure_initialized()

        class BetterJaccard(SimilarityBackend):
            def compute_similarity(self, text1: str, text2: str) -> float:
                return 0.99

        # Override jaccard
        SimilarityFactory.register(
            "jaccard",
            BetterJaccard,
            description="Better Jaccard",
            requires=[],
        )

        backend = SimilarityFactory.create("jaccard")
        assert backend.compute_similarity("a", "b") == 0.99

    def test_custom_backend_in_auto_select(self):
        """Test custom backend can be used in auto_select via env var."""

        class PriorityBackend(SimilarityBackend):
            def compute_similarity(self, text1: str, text2: str) -> float:
                return 0.77

        SimilarityFactory.register(
            "priority",
            PriorityBackend,
            description="Priority backend",
            requires=[],
        )

        with patch.dict("os.environ", {"ARAGORA_SIMILARITY_BACKEND": "priority"}):
            backend = SimilarityFactory.auto_select()
            assert backend.compute_similarity("a", "b") == 0.77
