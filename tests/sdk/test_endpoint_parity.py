from aragora.server.handlers.consensus import ConsensusHandler
from aragora.server.handlers.knowledge_base.handler import KnowledgeHandler
from aragora.server.handlers.memory.memory import MemoryHandler


def test_consensus_v1_paths_supported() -> None:
    handler = ConsensusHandler({})
    assert handler.can_handle("/api/v1/consensus/similar")
    assert handler.can_handle("/api/v1/consensus/settled")
    assert handler.can_handle("/api/v1/consensus/stats")
    assert handler.can_handle("/api/v1/consensus/dissents")
    assert handler.can_handle("/api/v1/consensus/contrarian-views")
    assert handler.can_handle("/api/v1/consensus/risk-warnings")
    assert handler.can_handle("/api/v1/consensus/domain/security")


def test_knowledge_v1_paths_supported() -> None:
    handler = KnowledgeHandler({})
    assert handler.can_handle("/api/v1/knowledge/search")
    assert handler.can_handle("/api/v1/knowledge/query")
    assert handler.can_handle("/api/v1/knowledge/facts")
    assert handler.can_handle("/api/v1/knowledge/stats")
    assert handler.can_handle("/api/v1/knowledge/facts/fact-123")


def test_memory_v1_paths_supported() -> None:
    handler = MemoryHandler({})
    assert handler.can_handle("/api/v1/memory/continuum/retrieve")
    assert handler.can_handle("/api/v1/memory/search")
    assert handler.can_handle("/api/v1/memory/tier-stats")
    assert handler.can_handle("/api/v1/memory/archive-stats")
    assert handler.can_handle("/api/v1/memory/pressure")
    assert handler.can_handle("/api/v1/memory/tiers")
    assert handler.can_handle("/api/v1/memory/critiques")
