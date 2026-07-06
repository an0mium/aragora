"""
Job queue workers for async task processing.

Workers:
- TranscriptionWorker: Processes audio/video transcription jobs
- RoutingWorker: Processes debate result routing jobs

GauntletWorker and ConsensusHealingWorker live outside this package:
``aragora.server.workers.gauntlet_worker`` (interface layer, imports
``server.stream.gauntlet_emitter``) and
``aragora.memory.consensus_healing_worker`` (domain layer, imports
``memory.consensus``) respectively
(docs/architecture/P4A_EVENTS_QUEUE_INVERSION.md §6.2, §10 Q3).
"""

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

__all__ = [
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
]
