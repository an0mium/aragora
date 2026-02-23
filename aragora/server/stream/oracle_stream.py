"""
Oracle Real-Time Streaming — WebSocket endpoint for the Shoggoth Oracle.

Replaces the batch request/response flow with streaming tokens and audio:
  1. Reflex phase: fast small-model acknowledgment (~2-3s)
  2. Deep phase: full essay-informed response with streaming TTS
  3. Tentacles: parallel multi-model perspectives
  4. Think-while-listening: pre-builds prompts from interim transcripts

Protocol:
  Client → Server (JSON text frames):
    {"type": "ask", "question": "...", "mode": "consult|divine|commune"}
    {"type": "interim", "text": "..."}   (partial speech transcript)
    {"type": "stop"}
    {"type": "ping"}

  Server → Client (JSON text frames):
    {"type": "connected"}
    {"type": "reflex_start"}
    {"type": "token", "text": "...", "phase": "reflex|deep", "sentence_complete": false}
    {"type": "sentence_ready", "text": "full sentence", "phase": "reflex|deep"}
    {"type": "phase_done", "phase": "reflex|deep", "full_text": "..."}
    {"type": "tentacle_start", "agent": "..."}
    {"type": "tentacle_token", "agent": "...", "text": "..."}
    {"type": "tentacle_done", "agent": "...", "full_text": "..."}
    {"type": "synthesis", "text": "..."}
    {"type": "error", "message": "..."}
    {"type": "pong"}

  Server → Client (binary frames):
    1-byte phase tag + raw mp3 chunk
    Phase tags: 0x00=reflex, 0x01=deep, 0x02=tentacle, 0x03=synthesis
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import aiohttp
from aiohttp import WSMsgType, web

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — reuse from playground handler
# ---------------------------------------------------------------------------

_TTS_VOICE_ID = "flHkNRp1BlvT73UL6gyz"
_TTS_MODEL = "eleven_multilingual_v2"

_PHASE_TAG_REFLEX = 0x00
_PHASE_TAG_DEEP = 0x01
_PHASE_TAG_TENTACLE = 0x02
_PHASE_TAG_SYNTHESIS = 0x03

# Sentence boundary pattern: ends with . ! or ? followed by space, newline, or end
_SENTENCE_BOUNDARY = re.compile(r"[.!?](?:\s|\n|$)")

# Reflex model — fast, cheap, low-latency
_REFLEX_MODEL_OPENROUTER = "anthropic/claude-3-5-haiku-20241022"
_REFLEX_MODEL_OPENAI = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# API key + model helpers (import from playground at runtime to avoid
# circular imports; fall back to env vars)
# ---------------------------------------------------------------------------


def _get_api_key(name: str) -> str | None:
    """Get an API key from secrets manager or env."""
    try:
        from aragora.config.secrets import get_secret
        return get_secret(name)
    except ImportError:
        return os.environ.get(name)


def _get_oracle_models() -> tuple[str, str, str]:
    """Return (openrouter_model, anthropic_model, openai_model) for deep phase."""
    try:
        from aragora.server.handlers.playground import (
            _ORACLE_MODEL_OPENROUTER,
            _ORACLE_MODEL_ANTHROPIC,
            _ORACLE_MODEL_OPENAI,
        )
        return _ORACLE_MODEL_OPENROUTER, _ORACLE_MODEL_ANTHROPIC, _ORACLE_MODEL_OPENAI
    except ImportError:
        return "anthropic/claude-opus-4.6", "claude-sonnet-4-6", "gpt-5.2"


def _get_tentacle_models() -> list[dict[str, str]]:
    """Return available tentacle model configs."""
    try:
        from aragora.server.handlers.playground import _get_available_tentacle_models
        return _get_available_tentacle_models()
    except ImportError:
        return []


def _build_oracle_prompt(mode: str, question: str) -> str:
    """Build the full Oracle prompt with essay context."""
    try:
        from aragora.server.handlers.playground import _build_oracle_prompt as _build
        return _build(mode, question)
    except ImportError:
        return question


# ---------------------------------------------------------------------------
# Session state for think-while-listening
# ---------------------------------------------------------------------------


@dataclass
class OracleSession:
    """Per-connection session state."""

    mode: str = "consult"
    last_interim: str = ""
    prebuilt_prompt: str | None = None
    active_task: asyncio.Task[Any] | None = None
    cancelled: bool = False
    created_at: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Streaming LLM — async generators yielding token strings
# ---------------------------------------------------------------------------

_SSE_DATA_PREFIX = "data: "


async def _stream_openrouter(
    model: str,
    prompt: str,
    max_tokens: int = 2000,
    timeout: float = 45.0,
) -> AsyncGenerator[str, None]:
    """Stream tokens from OpenRouter (OpenAI-compatible SSE)."""
    key = _get_api_key("OPENROUTER_API_KEY")
    if not key:
        return

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://aragora.ai",
        "X-Title": "Aragora Oracle",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    logger.warning("OpenRouter stream error: %d", resp.status)
                    return
                async for line in resp.content:
                    text = line.decode("utf-8", errors="replace").strip()
                    if not text.startswith(_SSE_DATA_PREFIX):
                        continue
                    data_str = text[len(_SSE_DATA_PREFIX):]
                    if data_str == "[DONE]":
                        return
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        logger.warning("OpenRouter stream failed: %s", exc)


async def _stream_anthropic(
    model: str,
    prompt: str,
    max_tokens: int = 2000,
    timeout: float = 45.0,
) -> AsyncGenerator[str, None]:
    """Stream tokens from Anthropic Messages API (SSE)."""
    key = _get_api_key("ANTHROPIC_API_KEY")
    if not key:
        return

    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    logger.warning("Anthropic stream error: %d", resp.status)
                    return
                async for line in resp.content:
                    text = line.decode("utf-8", errors="replace").strip()
                    if not text.startswith(_SSE_DATA_PREFIX):
                        continue
                    data_str = text[len(_SSE_DATA_PREFIX):]
                    try:
                        data = json.loads(data_str)
                        if data.get("type") == "content_block_delta":
                            delta = data.get("delta", {})
                            content = delta.get("text")
                            if content:
                                yield content
                    except (json.JSONDecodeError, KeyError):
                        continue
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        logger.warning("Anthropic stream failed: %s", exc)


async def _stream_openai_compat(
    base_url: str,
    key: str,
    model: str,
    prompt: str,
    max_tokens: int = 2000,
    timeout: float = 45.0,
) -> AsyncGenerator[str, None]:
    """Stream tokens from any OpenAI-compatible API (SSE)."""
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    logger.warning("OpenAI-compat stream error (%s): %d", base_url, resp.status)
                    return
                async for line in resp.content:
                    text = line.decode("utf-8", errors="replace").strip()
                    if not text.startswith(_SSE_DATA_PREFIX):
                        continue
                    data_str = text[len(_SSE_DATA_PREFIX):]
                    if data_str == "[DONE]":
                        return
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        logger.warning("OpenAI-compat stream failed (%s): %s", base_url, exc)


async def _call_provider_llm_stream(
    provider: str,
    model: str,
    prompt: str,
    max_tokens: int = 2000,
    timeout: float = 45.0,
) -> AsyncGenerator[str, None]:
    """Unified streaming LLM dispatcher. Yields token strings."""
    if provider == "openrouter":
        async for token in _stream_openrouter(model, prompt, max_tokens, timeout):
            yield token
    elif provider == "anthropic":
        async for token in _stream_anthropic(model, prompt, max_tokens, timeout):
            yield token
    elif provider == "openai":
        key = _get_api_key("OPENAI_API_KEY")
        if key:
            async for token in _stream_openai_compat(
                "https://api.openai.com/v1", key, model, prompt, max_tokens, timeout
            ):
                yield token
    elif provider == "xai":
        key = _get_api_key("XAI_API_KEY")
        if key:
            async for token in _stream_openai_compat(
                "https://api.x.ai/v1", key, model, prompt, max_tokens, timeout
            ):
                yield token
    elif provider == "google":
        # Google doesn't support standard SSE — fall back to non-streaming
        try:
            from aragora.server.handlers.playground import _call_provider_llm
            result = await asyncio.to_thread(_call_provider_llm, provider, model, prompt, max_tokens, timeout)
            if result:
                yield result
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Streaming TTS — ElevenLabs chunked mp3
# ---------------------------------------------------------------------------


async def _stream_tts(
    ws: web.WebSocketResponse,
    text: str,
    phase_tag: int = _PHASE_TAG_DEEP,
) -> None:
    """Stream TTS audio as binary WebSocket frames with phase tag prefix."""
    key = _get_api_key("ELEVENLABS_API_KEY")
    if not key:
        return

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{_TTS_VOICE_ID}/stream"
    headers = {
        "xi-api-key": key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": _TTS_MODEL,
        "voice_settings": {
            "stability": 0.4,
            "similarity_boost": 0.8,
            "style": 0.6,
            "use_speaker_boost": True,
        },
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    logger.warning("ElevenLabs stream error: %d", resp.status)
                    return
                tag_byte = bytes([phase_tag])
                async for chunk in resp.content.iter_chunked(4096):
                    if ws.closed:
                        return
                    await ws.send_bytes(tag_byte + chunk)
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        logger.warning("ElevenLabs stream failed: %s", exc)


# ---------------------------------------------------------------------------
# Sentence accumulator — detects boundaries and triggers TTS
# ---------------------------------------------------------------------------


class SentenceAccumulator:
    """Accumulates tokens and emits complete sentences."""

    def __init__(self) -> None:
        self._buffer = ""
        self._sentences: list[str] = []

    def add(self, token: str) -> str | None:
        """Add a token. Returns a complete sentence if boundary detected."""
        self._buffer += token
        match = _SENTENCE_BOUNDARY.search(self._buffer)
        if match:
            end = match.end()
            sentence = self._buffer[:end].strip()
            self._buffer = self._buffer[end:]
            if sentence:
                self._sentences.append(sentence)
                return sentence
        return None

    def flush(self) -> str | None:
        """Flush any remaining text as a final sentence."""
        remaining = self._buffer.strip()
        self._buffer = ""
        if remaining:
            self._sentences.append(remaining)
            return remaining
        return None

    @property
    def full_text(self) -> str:
        return " ".join(self._sentences)


# ---------------------------------------------------------------------------
# Reflex prompt — fast acknowledgment
# ---------------------------------------------------------------------------

_REFLEX_PROMPT = """You are the Oracle's quick-response system. Given the question below,
provide a 2-3 sentence immediate acknowledgment that shows you understand the question
and gives a preview of the direction you'll explore. Be warm, confident, and specific
to the question — never generic. Do NOT answer the full question.

Question: {question}"""


# ---------------------------------------------------------------------------
# Phase streaming functions
# ---------------------------------------------------------------------------


async def _stream_phase(
    ws: web.WebSocketResponse,
    prompt: str,
    phase: str,
    phase_tag: int,
    session: OracleSession,
    provider: str = "openrouter",
    model: str | None = None,
    max_tokens: int = 2000,
) -> str:
    """Stream an LLM response through the WebSocket, returning full text.

    Sends token events, sentence_ready events, and streams TTS per sentence.
    """
    if model is None:
        models = _get_oracle_models()
        model = models[0]  # openrouter default

    accumulator = SentenceAccumulator()
    tts_tasks: list[asyncio.Task[None]] = []

    async for token in _call_provider_llm_stream(provider, model, prompt, max_tokens):
        if session.cancelled or ws.closed:
            break

        # Send token event
        await ws.send_json({
            "type": "token",
            "text": token,
            "phase": phase,
            "sentence_complete": False,
        })

        # Check for complete sentence
        sentence = accumulator.add(token)
        if sentence:
            await ws.send_json({
                "type": "sentence_ready",
                "text": sentence,
                "phase": phase,
            })
            # Stream TTS for this sentence (fire and forget, bounded)
            task = asyncio.create_task(_stream_tts(ws, sentence, phase_tag))
            tts_tasks.append(task)

    # Flush remaining text
    remainder = accumulator.flush()
    if remainder and not session.cancelled and not ws.closed:
        await ws.send_json({
            "type": "sentence_ready",
            "text": remainder,
            "phase": phase,
        })
        task = asyncio.create_task(_stream_tts(ws, remainder, phase_tag))
        tts_tasks.append(task)

    # Wait for all TTS to finish
    if tts_tasks:
        await asyncio.gather(*tts_tasks, return_exceptions=True)

    full_text = accumulator.full_text

    if not session.cancelled and not ws.closed:
        await ws.send_json({
            "type": "phase_done",
            "phase": phase,
            "full_text": full_text,
        })

    return full_text


async def _stream_reflex(
    ws: web.WebSocketResponse,
    question: str,
    session: OracleSession,
) -> str:
    """Stream the reflex (quick acknowledgment) phase."""
    await ws.send_json({"type": "reflex_start"})

    prompt = _REFLEX_PROMPT.format(question=question)

    # Try OpenRouter with Haiku first, then OpenAI mini
    key_or = _get_api_key("OPENROUTER_API_KEY")
    if key_or:
        return await _stream_phase(
            ws, prompt, "reflex", _PHASE_TAG_REFLEX, session,
            provider="openrouter", model=_REFLEX_MODEL_OPENROUTER, max_tokens=300,
        )

    key_oai = _get_api_key("OPENAI_API_KEY")
    if key_oai:
        return await _stream_phase(
            ws, prompt, "reflex", _PHASE_TAG_REFLEX, session,
            provider="openai", model=_REFLEX_MODEL_OPENAI, max_tokens=300,
        )

    return ""


async def _stream_deep(
    ws: web.WebSocketResponse,
    prompt: str,
    session: OracleSession,
) -> str:
    """Stream the deep (full response) phase."""
    or_model, anth_model, oai_model = _get_oracle_models()

    # Try OpenRouter → Anthropic → OpenAI
    if _get_api_key("OPENROUTER_API_KEY"):
        result = await _stream_phase(
            ws, prompt, "deep", _PHASE_TAG_DEEP, session,
            provider="openrouter", model=or_model, max_tokens=2000,
        )
        if result:
            return result

    if _get_api_key("ANTHROPIC_API_KEY"):
        result = await _stream_phase(
            ws, prompt, "deep", _PHASE_TAG_DEEP, session,
            provider="anthropic", model=anth_model, max_tokens=2000,
        )
        if result:
            return result

    if _get_api_key("OPENAI_API_KEY"):
        return await _stream_phase(
            ws, prompt, "deep", _PHASE_TAG_DEEP, session,
            provider="openai", model=oai_model, max_tokens=2000,
        )

    return ""


async def _stream_tentacles(
    ws: web.WebSocketResponse,
    question: str,
    mode: str,
    session: OracleSession,
) -> None:
    """Stream tentacle perspectives from multiple models in parallel."""
    models = _get_tentacle_models()
    if not models:
        return

    prompt = _build_oracle_prompt(mode, question)

    async def run_tentacle(m: dict[str, str]) -> None:
        name = m["name"]
        if session.cancelled or ws.closed:
            return

        await ws.send_json({"type": "tentacle_start", "agent": name})

        full_text = ""
        try:
            async for token in _call_provider_llm_stream(
                m["provider"], m["model"], prompt, max_tokens=1000, timeout=30.0,
            ):
                if session.cancelled or ws.closed:
                    return
                full_text += token
                await ws.send_json({
                    "type": "tentacle_token",
                    "agent": name,
                    "text": token,
                })
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError, ConnectionError, TimeoutError):
            logger.warning("Tentacle %s failed", name, exc_info=True)

        if full_text and not session.cancelled and not ws.closed:
            await ws.send_json({
                "type": "tentacle_done",
                "agent": name,
                "full_text": full_text,
            })

    # Run tentacles concurrently (max 5)
    tasks = [asyncio.create_task(run_tentacle(m)) for m in models[:5]]
    await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# Main ask handler — orchestrates reflex → deep → tentacles
# ---------------------------------------------------------------------------


async def _handle_ask(
    ws: web.WebSocketResponse,
    question: str,
    mode: str,
    session: OracleSession,
) -> None:
    """Handle a complete Oracle consultation."""
    session.mode = mode
    session.cancelled = False

    # Start reflex immediately + build deep prompt concurrently
    reflex_task = asyncio.create_task(_stream_reflex(ws, question, session))

    # Use prebuilt prompt from interim if available, otherwise build now
    deep_prompt = session.prebuilt_prompt or _build_oracle_prompt(mode, question)
    session.prebuilt_prompt = None  # consumed

    await reflex_task

    if session.cancelled:
        return

    # Stream deep response
    await _stream_deep(ws, deep_prompt, session)

    if session.cancelled:
        return

    # Stream tentacles
    await _stream_tentacles(ws, question, mode, session)

    if session.cancelled or ws.closed:
        return

    # Send synthesis summary
    synthesis = (
        f"The Oracle has spoken. {len(_get_tentacle_models())} perspectives weighed. "
        f"The deep analysis is complete."
    )
    await ws.send_json({"type": "synthesis", "text": synthesis})


# ---------------------------------------------------------------------------
# Think-while-listening — process interim transcripts
# ---------------------------------------------------------------------------


def _handle_interim(session: OracleSession, text: str) -> None:
    """Process an interim transcript — pre-build the Oracle prompt."""
    session.last_interim = text
    session.prebuilt_prompt = _build_oracle_prompt(session.mode, text)


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------


async def oracle_websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """WebSocket handler for real-time Oracle streaming.

    Endpoint: /ws/oracle
    """
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)

    session = OracleSession()

    await ws.send_json({"type": "connected", "timestamp": time.time()})

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    msg_type = data.get("type")

                    if msg_type == "ping":
                        await ws.send_json({"type": "pong", "timestamp": time.time()})

                    elif msg_type == "ask":
                        question = str(data.get("question", "")).strip()
                        mode = str(data.get("mode", "consult"))
                        if not question:
                            await ws.send_json({
                                "type": "error",
                                "message": "Missing question",
                            })
                            continue

                        # Cancel any running task
                        if session.active_task and not session.active_task.done():
                            session.cancelled = True
                            session.active_task.cancel()
                            try:
                                await session.active_task
                            except (asyncio.CancelledError, Exception):
                                pass

                        # Start new consultation
                        session.active_task = asyncio.create_task(
                            _handle_ask(ws, question, mode, session)
                        )

                    elif msg_type == "interim":
                        text = str(data.get("text", "")).strip()
                        if text:
                            _handle_interim(session, text)

                    elif msg_type == "stop":
                        session.cancelled = True
                        if session.active_task and not session.active_task.done():
                            session.active_task.cancel()
                            try:
                                await session.active_task
                            except (asyncio.CancelledError, Exception):
                                pass

                except json.JSONDecodeError:
                    await ws.send_json({
                        "type": "error",
                        "message": "Invalid JSON",
                    })

            elif msg.type == WSMsgType.ERROR:
                logger.error("Oracle WebSocket error: %s", ws.exception())
                break

    finally:
        session.cancelled = True
        if session.active_task and not session.active_task.done():
            session.active_task.cancel()
            try:
                await session.active_task
            except (asyncio.CancelledError, Exception):
                pass

    return ws


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def register_oracle_stream_routes(app: web.Application) -> None:
    """Register the Oracle streaming WebSocket route."""
    app.router.add_get("/ws/oracle", oracle_websocket_handler)


__all__ = [
    "oracle_websocket_handler",
    "register_oracle_stream_routes",
    "OracleSession",
    "SentenceAccumulator",
]
