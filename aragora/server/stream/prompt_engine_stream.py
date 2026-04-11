"""WebSocket stream handler for prompt engine pipeline events.

Provides real-time streaming of prompt-to-specification pipeline stages
to connected WebSocket clients.

Event types:
- prompt_engine_start      - Pipeline begins
- prompt_engine_stage      - Stage started (decompose/interrogate/research/specify)
- prompt_engine_intent     - Decomposition complete
- prompt_engine_questions  - Clarifying questions generated
- prompt_engine_research   - Research complete
- prompt_engine_spec       - Specification built
- prompt_engine_validation - Validation complete
- prompt_engine_complete   - Pipeline finished
- prompt_engine_error      - Error occurred
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from aiohttp import WSMsgType, web

logger = logging.getLogger(__name__)


@dataclass
class PromptEngineStreamClient:
    """A connected WebSocket client for prompt engine events."""

    ws: web.WebSocketResponse
    client_id: str
    session_id: str
    connected_at: float = field(default_factory=time.time)


class PromptEngineStreamEmitter:
    """Emitter for prompt engine pipeline events.

    Broadcasts pipeline stage events to connected WebSocket clients,
    filtered by session_id so each client only receives events for
    their own pipeline run.
    """

    def __init__(self) -> None:
        self._clients: dict[str, PromptEngineStreamClient] = {}
        self._client_counter = 0

    def add_client(
        self,
        ws: web.WebSocketResponse,
        session_id: str,
    ) -> str:
        self._client_counter += 1
        client_id = f"pe_{self._client_counter}_{int(time.time())}"
        self._clients[client_id] = PromptEngineStreamClient(
            ws=ws,
            client_id=client_id,
            session_id=session_id,
        )
        logger.info("Prompt engine client connected: %s (session %s)", client_id, session_id)
        return client_id

    def remove_client(self, client_id: str) -> None:
        if client_id in self._clients:
            del self._clients[client_id]
            logger.info("Prompt engine client disconnected: %s", client_id)

    async def emit(
        self,
        session_id: str,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Emit an event to all clients watching a specific session."""
        message = {
            "type": event_type,
            "session_id": session_id,
            "timestamp": time.time(),
            **data,
        }

        disconnected: list[str] = []
        for cid, client in self._clients.items():
            if client.session_id != session_id:
                continue
            try:
                await client.ws.send_json(message)
            except (ConnectionError, RuntimeError):
                disconnected.append(cid)

        for cid in disconnected:
            self.remove_client(cid)

    @property
    def client_count(self) -> int:
        return len(self._clients)


# Module-level singleton
_emitter: PromptEngineStreamEmitter | None = None


def get_prompt_engine_emitter() -> PromptEngineStreamEmitter:
    global _emitter
    if _emitter is None:
        _emitter = PromptEngineStreamEmitter()
    return _emitter


def set_prompt_engine_emitter(emitter: PromptEngineStreamEmitter) -> None:
    global _emitter
    _emitter = emitter


async def _run_pipeline(
    emitter: PromptEngineStreamEmitter,
    session_id: str,
    prompt: str,
    profile: str,
    context: dict[str, Any] | None,
) -> None:
    """Run the prompt engine pipeline, emitting events at each stage."""
    from aragora.prompt_engine import (
        ConductorConfig,
        PROMPT_ENGINE_TARGET_DURATION_MS,
        PromptConductor,
        PipelineTiming,
        SpecValidator,
    )
    from aragora.prompt_engine.timing import elapsed_ms, start_timer

    try:
        config = ConductorConfig.from_profile(profile)
        conductor = PromptConductor(config=config)
        pipeline_start = start_timer()
        stage_durations_ms: dict[str, float] = {}
        stages_completed: list[str] = []

        # Stage 1: Decompose
        stage_start = start_timer()
        await emitter.emit(
            session_id,
            "prompt_engine_stage",
            {
                "stage": "decompose",
                "status": "started",
            },
        )
        stage_start = start_timer()
        intent = await conductor.decompose_only(prompt, context)
        stage_durations_ms["decompose"] = elapsed_ms(stage_start)
        stages_completed.append("decompose")
        await emitter.emit(
            session_id,
            "prompt_engine_intent",
            {
                "intent": intent.to_dict(),
                "stage_duration_ms": round(stage_durations_ms["decompose"], 2),
            },
        )

        # Stage 2: Interrogate
        if intent.needs_clarification and not config.skip_interrogation:
            stage_start = start_timer()
            await emitter.emit(
                session_id,
                "prompt_engine_stage",
                {
                    "stage": "interrogate",
                    "status": "started",
                },
            )
            stage_start = start_timer()
            questions = await conductor.interrogate_only(intent)
            stage_durations_ms["interrogate"] = elapsed_ms(stage_start)
            stages_completed.append("interrogate")
            await emitter.emit(
                session_id,
                "prompt_engine_questions",
                {
                    "questions": [q.to_dict() for q in questions],
                    "stage_duration_ms": round(stage_durations_ms["interrogate"], 2),
                },
            )
        else:
            questions = []

        # Stage 3: Research
        if not config.skip_research:
            stage_start = start_timer()
            await emitter.emit(
                session_id,
                "prompt_engine_stage",
                {
                    "stage": "research",
                    "status": "started",
                },
            )
            stage_start = start_timer()
            research = await conductor.research_only(intent, questions or None)
            stage_durations_ms["research"] = elapsed_ms(stage_start)
            stages_completed.append("research")
            await emitter.emit(
                session_id,
                "prompt_engine_research",
                {
                    "research": research.to_dict(),
                    "stage_duration_ms": round(stage_durations_ms["research"], 2),
                },
            )
        else:
            research = None

        # Stage 4: Specify
        stage_start = start_timer()
        await emitter.emit(
            session_id,
            "prompt_engine_stage",
            {
                "stage": "specify",
                "status": "started",
            },
        )
        stage_start = start_timer()
        spec = await conductor.specify_only(intent, questions or None, research)
        stage_durations_ms["specify"] = elapsed_ms(stage_start)
        stages_completed.append("specify")
        await emitter.emit(
            session_id,
            "prompt_engine_spec",
            {
                "specification": spec.to_dict(),
                "stage_duration_ms": round(stage_durations_ms["specify"], 2),
            },
        )

        # Validation
        validator = SpecValidator()
        validation_start = start_timer()
        validation = validator.validate_heuristic(spec)
        validator_operation_timings = list(getattr(validator, "last_operation_timings", None) or [])
        validation_duration_ms = elapsed_ms(validation_start)
        latency_target_ms = getattr(config, "latency_target_ms", PROMPT_ENGINE_TARGET_DURATION_MS)
        if not isinstance(latency_target_ms, (int, float)):
            latency_target_ms = PROMPT_ENGINE_TARGET_DURATION_MS

        validation_timing = PipelineTiming(
            total_duration_ms=validation_duration_ms,
            stage_durations_ms={"validate": validation_duration_ms},
            operation_timings=validator_operation_timings,
            target_duration_ms=latency_target_ms,
        )
        await emitter.emit(
            session_id,
            "prompt_engine_validation",
            {
                "validation": validation.to_dict(),
                "timing": validation_timing.to_dict(),
            },
        )

        target_duration_ms = getattr(
            config, "latency_target_ms", PipelineTiming().target_duration_ms
        )
        if not isinstance(target_duration_ms, int | float):
            target_duration_ms = PipelineTiming().target_duration_ms

        stage_operation_timings = getattr(conductor, "stage_operation_timings", {})
        if not isinstance(stage_operation_timings, dict):
            stage_operation_timings = {}
        operation_timings = [
            timing
            for timings in stage_operation_timings.values()
            if isinstance(timings, list | tuple)
            for timing in timings
        ]
        operation_timings.extend(validator_operation_timings)
        timing = PipelineTiming(
            total_duration_ms=elapsed_ms(pipeline_start),
            stage_durations_ms=stage_durations_ms,
            operation_timings=operation_timings,
            target_duration_ms=float(target_duration_ms),
        )

        # Complete
        await emitter.emit(
            session_id,
            "prompt_engine_complete",
            {
                "stages_completed": stages_completed,
                "timing": timing.to_dict(),
            },
        )

    except (
        KeyError,
        ValueError,
        TypeError,
        AttributeError,
        RuntimeError,
        OSError,
        ImportError,
    ) as exc:
        logger.exception("Prompt engine pipeline error: %s", exc)
        await emitter.emit(
            session_id,
            "prompt_engine_error",
            {
                "error": "Pipeline failed",
            },
        )


async def prompt_engine_websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """WebSocket handler for prompt engine pipeline streaming."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    emitter = get_prompt_engine_emitter()
    session_id = str(uuid.uuid4())
    client_id = emitter.add_client(ws, session_id)

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    action = data.get("action", "run")

                    if action == "run":
                        prompt = data.get("prompt", "").strip()
                        if not prompt:
                            await ws.send_json(
                                {
                                    "type": "prompt_engine_error",
                                    "error": "prompt is required",
                                }
                            )
                            continue

                        profile = data.get("profile", "founder")
                        context = data.get("context")

                        await emitter.emit(
                            session_id,
                            "prompt_engine_start",
                            {
                                "prompt": prompt,
                                "profile": profile,
                            },
                        )

                        # Run pipeline as a background task so we can
                        # continue receiving messages
                        asyncio.create_task(
                            _run_pipeline(emitter, session_id, prompt, profile, context)
                        )

                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "Invalid JSON"})

            elif msg.type == WSMsgType.ERROR:
                logger.error("Prompt engine WS error: %s", ws.exception())
                break
    finally:
        emitter.remove_client(client_id)

    return ws


def register_prompt_engine_stream_routes(app: web.Application) -> None:
    """Register the prompt engine stream WebSocket route."""
    app.router.add_get("/ws/prompt-engine", prompt_engine_websocket_handler)


__all__ = [
    "PromptEngineStreamClient",
    "PromptEngineStreamEmitter",
    "get_prompt_engine_emitter",
    "set_prompt_engine_emitter",
    "prompt_engine_websocket_handler",
    "register_prompt_engine_stream_routes",
]
