"""
Memory and pattern storage module.

Provides:
- CritiqueStore: SQLite-based storage for debate results and patterns
- SemanticRetriever: Embedding-based similarity search
- Pattern: Dataclass for critique patterns
"""

from aagora.memory.store import CritiqueStore, Pattern
from aagora.memory.embeddings import (
    SemanticRetriever,
    OpenAIEmbedding,
    GeminiEmbedding,
    OllamaEmbedding,
)

__all__ = [
    "CritiqueStore",
    "Pattern",
    "SemanticRetriever",
    "OpenAIEmbedding",
    "GeminiEmbedding",
    "OllamaEmbedding",
]
