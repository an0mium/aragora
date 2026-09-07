"""Deprecated module: backend protocol definitions moved to ``aragora.protocols``.

This module is a backward-compatibility shim. The backend protocol definitions
now live in ``aragora.protocols`` (see ``aragora.protocols.backend_protocols``).
Import them from the canonical surface instead::

    from aragora.protocols import StorageBackend, MemoryBackend

Importing from ``aragora.core_protocols`` emits a ``DeprecationWarning`` and the
module will be removed in a future release.
"""

from __future__ import annotations

import warnings

from aragora.protocols import (
    Agent,
    AgentRating,
    AgentRecord,
    AuthenticatedUser,
    ConsensusBackend,
    CritiqueBackend,
    DebateRecord,
    EloBackend,
    EmbeddingBackend,
    GenesisBackend,
    HTTPHeaders,
    HTTPRequestHandler,
    MemoryBackend,
    MemoryRecord,
    PathSegments,
    PersonaBackend,
    QueryParams,
    StorageBackend,
)

warnings.warn(
    "aragora.core_protocols is deprecated; import backend protocols from "
    "aragora.protocols instead (e.g. `from aragora.protocols import StorageBackend`).",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    # Storage backends
    "StorageBackend",
    "MemoryBackend",
    "EloBackend",
    "EmbeddingBackend",
    "ConsensusBackend",
    "CritiqueBackend",
    "PersonaBackend",
    "GenesisBackend",
    # HTTP/Auth
    "HTTPHeaders",
    "HTTPRequestHandler",
    "AuthenticatedUser",
    # Agent basics
    "Agent",
    "AgentRating",
    # Type aliases
    "DebateRecord",
    "MemoryRecord",
    "AgentRecord",
    "QueryParams",
    "PathSegments",
]
