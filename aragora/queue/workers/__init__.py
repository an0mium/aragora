"""
Job queue workers for async task processing.

Workers:
- TranscriptionWorker: Processes audio/video transcription jobs

GauntletWorker and ConsensusHealingWorker live outside this package:
``aragora.server.workers.gauntlet_worker`` (interface layer, imports
``server.stream.gauntlet_emitter``) and
``aragora.memory.consensus_healing_worker`` (domain layer, imports
``memory.consensus``) respectively (docs/architecture/P4A_EVENTS_QUEUE_INVERSION.md
§6.2, §10 Q3). RoutingWorker and TestFixerWorker also live outside this
package: ``aragora.server.workers.routing_worker`` (interface layer, imports
``server.debate_origin`` + ``integrations.email_reply_loop``) and
``aragora.nomic.testfixer.queue_worker`` (application layer, imports
``nomic.testfixer.http_api``) respectively (§10 Q4).
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

__all__ = [
    # Transcription
    "TranscriptionWorker",
    "JOB_TYPE_TRANSCRIPTION",
    "JOB_TYPE_TRANSCRIPTION_AUDIO",
    "JOB_TYPE_TRANSCRIPTION_VIDEO",
    "JOB_TYPE_TRANSCRIPTION_YOUTUBE",
    "enqueue_transcription_job",
    "recover_interrupted_transcriptions",
]
