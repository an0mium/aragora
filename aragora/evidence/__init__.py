"""Evidence collection, verification, and persistence module."""

# Import Evidence from connectors where it's defined
from aragora.connectors.base import Evidence
from aragora.evidence.attribution import (
    AttributionChain,
    AttributionChainEntry,
    ReputationScorer,
    ReputationTier,
    SourceReputation,
    SourceReputationManager,
    VerificationOutcome,
    VerificationRecord,
)
from aragora.evidence.collector import (
    EvidenceCollector,
    EvidencePack,
    EvidenceSnippet,
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
    # Attribution
    "AttributionChain",
    "AttributionChainEntry",
    "ReputationScorer",
    "ReputationTier",
    "SourceReputation",
    "SourceReputationManager",
    "VerificationOutcome",
    "VerificationRecord",
    # External
    "Evidence",
    "EvidenceType",
]
