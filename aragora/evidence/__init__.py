"""Evidence collection, verification, and persistence module."""

from aragora.evidence.collector import (
    EvidenceCollector,
    EvidenceSnippet,
    EvidencePack,
)
from aragora.evidence.metadata import (
    EnrichedMetadata,
    MetadataEnricher,
    Provenance,
    SourceType,
    enrich_evidence_snippet,
)
from aragora.evidence.quality import (
    QualityContext,
    QualityFilter,
    QualityScorer,
    QualityScores,
    QualityTier,
    score_evidence_snippet,
)
from aragora.evidence.store import (
    EvidenceStore,
    InMemoryEvidenceStore,
)

# Import Evidence from connectors where it's defined
from aragora.connectors.base import Evidence

# Import EvidenceType from reasoning where it's defined
from aragora.reasoning.claims import EvidenceType

__all__ = [
    # Collector
    "EvidenceCollector",
    "EvidenceSnippet",
    "EvidencePack",
    # Metadata
    "EnrichedMetadata",
    "MetadataEnricher",
    "Provenance",
    "SourceType",
    "enrich_evidence_snippet",
    # Quality
    "QualityContext",
    "QualityFilter",
    "QualityScorer",
    "QualityScores",
    "QualityTier",
    "score_evidence_snippet",
    # Store
    "EvidenceStore",
    "InMemoryEvidenceStore",
    # External
    "Evidence",
    "EvidenceType",
]
