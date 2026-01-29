"""Tests for aragora.embeddings module imports and basic structure."""

import importlib
import inspect

import pytest


class TestModuleImports:
    """Verify the embeddings module can be imported and exposes expected symbols."""

    def test_module_importable(self):
        mod = importlib.import_module("aragora.embeddings")
        assert mod is not None

    def test_all_exports_defined(self):
        from aragora.embeddings import __all__

        expected = {
            "EmbeddingProvider",
            "OpenAIEmbedding",
            "GeminiEmbedding",
            "OllamaEmbedding",
            "get_embedding_provider",
            "EmbeddingCache",
            "SemanticRetriever",
            "EmbeddingBackend",
            "EmbeddingProviderProtocol",
            "embed_text",
            "embed_batch",
            "reset_default_provider",
        }
        assert expected == set(__all__)


class TestCoreExports:
    """Verify key classes and functions exist and are the correct type."""

    def test_embedding_provider_is_class(self):
        from aragora.embeddings import EmbeddingProvider

        assert inspect.isclass(EmbeddingProvider)

    def test_embedding_cache_is_class(self):
        from aragora.embeddings import EmbeddingCache

        assert inspect.isclass(EmbeddingCache)

    def test_semantic_retriever_is_class(self):
        from aragora.embeddings import SemanticRetriever

        assert inspect.isclass(SemanticRetriever)

    def test_openai_embedding_is_class(self):
        from aragora.embeddings import OpenAIEmbedding

        assert inspect.isclass(OpenAIEmbedding)

    def test_gemini_embedding_is_class(self):
        from aragora.embeddings import GeminiEmbedding

        assert inspect.isclass(GeminiEmbedding)

    def test_ollama_embedding_is_class(self):
        from aragora.embeddings import OllamaEmbedding

        assert inspect.isclass(OllamaEmbedding)

    def test_embedding_backend_exists(self):
        from aragora.embeddings import EmbeddingBackend

        assert EmbeddingBackend is not None

    def test_embedding_provider_protocol_is_class(self):
        from aragora.embeddings import EmbeddingProviderProtocol

        assert inspect.isclass(EmbeddingProviderProtocol)


class TestFunctions:
    """Verify module-level functions exist and are callable."""

    def test_get_embedding_provider_callable(self):
        from aragora.embeddings import get_embedding_provider

        assert callable(get_embedding_provider)

    def test_reset_default_provider_callable(self):
        from aragora.embeddings import reset_default_provider

        assert callable(reset_default_provider)

    def test_embed_text_callable(self):
        from aragora.embeddings import embed_text

        assert callable(embed_text)

    def test_embed_batch_callable(self):
        from aragora.embeddings import embed_batch

        assert callable(embed_batch)


class TestBasicInstantiation:
    """Test instantiation of classes that don't require API keys."""

    def test_embedding_provider_default(self):
        from aragora.embeddings import EmbeddingProvider

        provider = EmbeddingProvider()
        assert provider.dimension == 256

    def test_embedding_provider_custom_dimension(self):
        from aragora.embeddings import EmbeddingProvider

        provider = EmbeddingProvider(dimension=768)
        assert provider.dimension == 768

    def test_embedding_provider_has_embed_method(self):
        from aragora.embeddings import EmbeddingProvider

        provider = EmbeddingProvider()
        assert hasattr(provider, "embed")
        assert callable(provider.embed)

    def test_embedding_provider_has_embed_batch_method(self):
        from aragora.embeddings import EmbeddingProvider

        provider = EmbeddingProvider()
        assert hasattr(provider, "embed_batch")
        assert callable(provider.embed_batch)

    def test_get_embedding_provider_returns_provider(self):
        from aragora.embeddings import EmbeddingProvider, get_embedding_provider

        provider = get_embedding_provider()
        assert isinstance(provider, EmbeddingProvider)

    def test_reset_default_provider_runs(self):
        from aragora.embeddings import reset_default_provider

        # Should not raise
        reset_default_provider()

    def test_embedding_provider_protocol_runtime_checkable(self):
        from aragora.embeddings import EmbeddingProvider, EmbeddingProviderProtocol

        provider = EmbeddingProvider()
        assert isinstance(provider, EmbeddingProviderProtocol)


class TestSubclassRelationships:
    """Verify provider subclass hierarchy."""

    def test_openai_is_subclass(self):
        from aragora.embeddings import EmbeddingProvider, OpenAIEmbedding

        assert issubclass(OpenAIEmbedding, EmbeddingProvider)

    def test_gemini_is_subclass(self):
        from aragora.embeddings import EmbeddingProvider, GeminiEmbedding

        assert issubclass(GeminiEmbedding, EmbeddingProvider)

    def test_ollama_is_subclass(self):
        from aragora.embeddings import EmbeddingProvider, OllamaEmbedding

        assert issubclass(OllamaEmbedding, EmbeddingProvider)
