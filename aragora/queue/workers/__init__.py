"""
Job queue workers for async task processing.

Workers:
- GauntletWorker: Processes gauntlet stress-testing jobs
- TranscriptionWorker: Processes audio/video transcription jobs
- RoutingWorker: Processes debate result routing jobs
"""

from aragora.queue.workers.gauntlet_worker import (
    GauntletWorker,
    JOB_TYPE_GAUNTLET,
    enqueue_gauntlet_job,
    recover_interrupted_gauntlets,
)
from aragora.queue.workers.transcription_worker import (
    TranscriptionWorker,
    JOB_TYPE_TRANSCRIPTION,
    JOB_TYPE_TRANSCRIPTION_AUDIO,
    JOB_TYPE_TRANSCRIPTION_VIDEO,
    JOB_TYPE_TRANSCRIPTION_YOUTUBE,
    enqueue_transcription_job,
    recover_interrupted_transcriptions,
)
from aragora.queue.workers.routing_worker import (
    RoutingWorker,
    JOB_TYPE_ROUTING,
    JOB_TYPE_ROUTING_DEBATE,
    JOB_TYPE_ROUTING_EMAIL,
    enqueue_routing_job,
    recover_interrupted_routing,
)
from aragora.queue.workers.consensus_healing_worker import (
    ConsensusHealingWorker,
    HealingAction,
    HealingCandidate,
    HealingConfig,
    HealingReason,
    HealingResult,
    get_consensus_healing_worker,
    start_consensus_healing,
    stop_consensus_healing,
)

__all__ = [
    # Gauntlet
    "GauntletWorker",
    "JOB_TYPE_GAUNTLET",
    "enqueue_gauntlet_job",
    "recover_interrupted_gauntlets",
    # Transcription
    "TranscriptionWorker",
    "JOB_TYPE_TRANSCRIPTION",
    "JOB_TYPE_TRANSCRIPTION_AUDIO",
    "JOB_TYPE_TRANSCRIPTION_VIDEO",
    "JOB_TYPE_TRANSCRIPTION_YOUTUBE",
    "enqueue_transcription_job",
    "recover_interrupted_transcriptions",
    # Routing
    "RoutingWorker",
    "JOB_TYPE_ROUTING",
    "JOB_TYPE_ROUTING_DEBATE",
    "JOB_TYPE_ROUTING_EMAIL",
    "enqueue_routing_job",
    "recover_interrupted_routing",
    # Consensus Healing
    "ConsensusHealingWorker",
    "HealingAction",
    "HealingCandidate",
    "HealingConfig",
    "HealingReason",
    "HealingResult",
    "get_consensus_healing_worker",
    "start_consensus_healing",
    "stop_consensus_healing",
]
