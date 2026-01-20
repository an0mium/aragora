"""Unified embedding service for all Aragora subsystems.

This module provides a single, consistent interface for generating text embeddings
across the entire codebase. It consolidates previously fragmented implementations
from memory, debate, and knowledge modules.

Usage:
    from aragora.core.embeddings import get_embedding_service, EmbeddingConfig

    # Get default service (auto-detects best available provider)
    service = get_embedding_service()

    # Or configure explicitly
    config = EmbeddingConfig(provider="openai", model="text-embedding-3-small")
    service = get_embedding_service(config)

    # Generate embeddings
    embedding = await service.embed("Your text here")
    embeddings = await service.embed_batch(["Text 1", "Text 2"])

    # Compute similarity
    similarity = await service.similarity("Hello", "Hi there")
"""

from aragora.core.embeddings.types import EmbeddingConfig, EmbeddingResult
from aragora.core.embeddings.backends import EmbeddingBackend
from aragora.core.embeddings.service import UnifiedEmbeddingService, get_embedding_service

__all__ = [
    "EmbeddingBackend",
    "EmbeddingConfig",
    "EmbeddingResult",
    "UnifiedEmbeddingService",
    "get_embedding_service",
]
