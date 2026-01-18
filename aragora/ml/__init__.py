"""
Machine Learning Module for Aragora.

Provides local ML capabilities:
- Local embedding models (sentence-transformers)
- Quality scoring
- Consensus prediction
- Agent routing
- Local fine-tuning (PEFT/LoRA)

These models run locally without external API dependencies.
"""

from aragora.ml.embeddings import (
    LocalEmbeddingService,
    LocalEmbeddingConfig,
    EmbeddingModel,
    get_embedding_service,
)
from aragora.ml.quality_scorer import (
    QualityScorer,
    QualityScorerConfig,
    QualityScore,
    get_quality_scorer,
)
from aragora.ml.consensus_predictor import (
    ConsensusPredictor,
    ConsensusPrediction,
    get_consensus_predictor,
)
from aragora.ml.agent_router import (
    AgentRouter,
    RoutingDecision,
    TaskType,
    get_agent_router,
)
from aragora.ml.local_finetuning import (
    LocalFineTuner,
    DPOFineTuner,
    FineTuneConfig,
    FineTuneResult,
    TrainingData,
    TrainingExample,
    create_fine_tuner,
)

__all__ = [
    # Embeddings
    "LocalEmbeddingService",
    "LocalEmbeddingConfig",
    "EmbeddingModel",
    "get_embedding_service",
    # Quality Scoring
    "QualityScorer",
    "QualityScorerConfig",
    "QualityScore",
    "get_quality_scorer",
    # Consensus Prediction
    "ConsensusPredictor",
    "ConsensusPrediction",
    "get_consensus_predictor",
    # Agent Routing
    "AgentRouter",
    "RoutingDecision",
    "TaskType",
    "get_agent_router",
    # Fine-tuning
    "LocalFineTuner",
    "DPOFineTuner",
    "FineTuneConfig",
    "FineTuneResult",
    "TrainingData",
    "TrainingExample",
    "create_fine_tuner",
]
