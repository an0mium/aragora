"""
Playground Handler - Public demo endpoint for the aragora-debate engine.

Stability: STABLE

Allows anyone to run a mock debate without authentication or API keys.
Uses StyledMockAgent from the aragora-debate standalone package for
deterministic, zero-dependency debates.  The ``/live`` variant uses real
API-backed agents with budget + timeout caps for a taste of the full
platform.

Routes:
    POST /api/v1/playground/debate             - Run a mock debate
    POST /api/v1/playground/debate/live         - Run a live debate with real agents
    POST /api/v1/playground/debate/live/cost-estimate - Pre-flight cost estimate
    GET  /api/v1/playground/status              - Health check for the playground
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
    handle_errors,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiting (in-memory, per-IP, 5 req/min for mock, 1/10min for live)
# ---------------------------------------------------------------------------

_PLAYGROUND_RATE_LIMIT = 5  # requests per window
_PLAYGROUND_RATE_WINDOW = 60.0  # seconds

_LIVE_RATE_LIMIT = 1  # 1 live debate per window per IP
_LIVE_RATE_WINDOW = 600.0  # 10 minutes

# IP -> list of timestamps
_request_timestamps: dict[str, list[float]] = {}
_live_request_timestamps: dict[str, list[float]] = {}


def _check_rate_limit(
    client_ip: str,
    limit: int = _PLAYGROUND_RATE_LIMIT,
    window: float = _PLAYGROUND_RATE_WINDOW,
) -> tuple[bool, int]:
    """Check whether the client IP is within the rate limit.

    Returns:
        (allowed, retry_after_seconds)
    """
    now = time.monotonic()
    cutoff = now - window

    timestamps = _request_timestamps.get(client_ip, [])
    # Prune old entries
    timestamps = [t for t in timestamps if t > cutoff]

    if len(timestamps) >= limit:
        oldest_in_window = timestamps[0]
        retry_after = int(oldest_in_window + window - now) + 1
        _request_timestamps[client_ip] = timestamps
        return False, max(retry_after, 1)

    timestamps.append(now)
    _request_timestamps[client_ip] = timestamps
    return True, 0


def _check_live_rate_limit(client_ip: str) -> tuple[bool, int]:
    """Check whether the client IP is within the live debate rate limit.

    Returns:
        (allowed, retry_after_seconds)
    """
    now = time.monotonic()
    cutoff = now - _LIVE_RATE_WINDOW

    timestamps = _live_request_timestamps.get(client_ip, [])
    timestamps = [t for t in timestamps if t > cutoff]

    if len(timestamps) >= _LIVE_RATE_LIMIT:
        oldest_in_window = timestamps[0]
        retry_after = int(oldest_in_window + _LIVE_RATE_WINDOW - now) + 1
        _live_request_timestamps[client_ip] = timestamps
        return False, max(retry_after, 1)

    timestamps.append(now)
    _live_request_timestamps[client_ip] = timestamps
    return True, 0


def _reset_rate_limits() -> None:
    """Reset all rate limit state. Used by tests."""
    _request_timestamps.clear()
    _live_request_timestamps.clear()


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

_MAX_TOPIC_LENGTH = 100_000  # ~15k words — supports long-form debate prompts
_MAX_ROUNDS = 2
_MAX_AGENTS = 5
_MIN_AGENTS = 2

_DEFAULT_TOPIC = "Should we use microservices or a monolith?"
_DEFAULT_ROUNDS = 2
_DEFAULT_AGENTS = 3

_AGENT_STYLES: list[Literal["supportive", "critical", "balanced", "contrarian"]] = [
    "supportive",
    "critical",
    "balanced",
    "contrarian",
]


# ---------------------------------------------------------------------------
# Inline mock debate (fallback when aragora-debate is not installed)
# ---------------------------------------------------------------------------

def _build_mock_proposals(topic: str, question: str | None = None) -> dict[str, list[str]]:
    """Build topic-aware mock proposals instead of canned microservices text."""
    # If a raw question was provided (e.g. from Oracle mode), use it for the snippet
    # instead of the full system-prompt-laden topic
    source = question or topic
    snippet = source[:200].strip()
    if len(source) > 200:
        snippet = snippet.rsplit(" ", 1)[0] + "..."

    return {
        "supportive": [
            f"After careful analysis of the submission, I find the core argument compelling. "
            f"Regarding '{snippet}' -- the reasoning is well-structured, the evidence cited is "
            f"substantive, and the conclusions follow logically from the premises. The key "
            f"strengths are the specificity of claims and the willingness to engage with "
            f"counterarguments. I recommend this position with minor caveats.",
            f"This is a strong argument. The submission on '{snippet}' demonstrates clear "
            f"thinking and grounded analysis. The supporting evidence is concrete rather than "
            f"abstract, and the framework presented offers actionable insights. The conclusion "
            f"is well-earned by the preceding analysis.",
        ],
        "critical": [
            f"I have significant concerns about the argument presented. While '{snippet}' "
            f"raises important points, several claims lack sufficient empirical backing. The "
            f"causal reasoning conflates correlation with causation in key places, and the "
            f"conclusion overreaches what the evidence supports. A more rigorous analysis "
            f"would need to address the strongest counterarguments directly.",
            f"This submission overlooks critical failure modes. The argument around "
            f"'{snippet}' makes assumptions that haven't been validated -- particularly "
            f"about timelines and magnitudes. The most likely scenario involves more "
            f"uncertainty than the author acknowledges, and the recommended actions don't "
            f"adequately account for second-order effects.",
        ],
        "balanced": [
            f"There are valid points on both sides. The submission on '{snippet}' correctly "
            f"identifies real dynamics, but the framing occasionally overstates certainty "
            f"where the evidence is ambiguous. A more nuanced position would acknowledge "
            f"where the author's model could be wrong and under what conditions the opposite "
            f"conclusion might hold.",
            f"The analysis of '{snippet}' has genuine strengths -- concrete examples, "
            f"specific claims, falsifiable predictions. But it also has blind spots: the "
            f"framework assumes certain structural dynamics will continue, which isn't "
            f"guaranteed. A balanced assessment says: directionally right, calibrationally "
            f"uncertain.",
        ],
        "contrarian": [
            f"I disagree with the prevailing direction of this analysis. The argument "
            f"about '{snippet}' optimizes for the visible pattern while ignoring systemic "
            f"risks that would invalidate the thesis entirely. The most important question "
            f"isn't whether the author's scenario is plausible -- it's whether the "
            f"confidence level is warranted given the evidence.",
            f"Everyone seems to be converging too quickly on this framing. Let me argue "
            f"the unpopular position: the submission on '{snippet}' may be directionally "
            f"wrong in ways that feel uncomfortable to acknowledge. The evidence cited "
            f"is selectively chosen, and equally compelling evidence exists for the "
            f"opposite conclusion.",
        ],
    }


# Keep static version for backward compat with tests that reference _MOCK_PROPOSALS
_MOCK_PROPOSALS = _build_mock_proposals(_DEFAULT_TOPIC)

_MOCK_CRITIQUE_ISSUES: dict[str, list[str]] = {
    "supportive": [
        "Could benefit from more quantitative evidence",
        "Some claims would be stronger with explicit confidence intervals",
    ],
    "critical": [
        "Key causal claims lack sufficient empirical backing",
        "No analysis of what happens if the core assumptions are wrong",
        "Ignores the strongest counterarguments to the thesis",
        "Conflates multiple distinct phenomena under one framework",
    ],
    "balanced": [
        "The argument could better acknowledge where uncertainty is highest",
        "Risk assessment is asymmetric -- considers one failure mode but not others",
    ],
    "contrarian": [
        "The group appears to be converging prematurely on this framing",
        "The most important alternative scenarios have not been seriously considered",
    ],
}

_MOCK_CRITIQUE_SUGGESTIONS: dict[str, list[str]] = {
    "supportive": ["Consider adding falsification criteria for the key claims"],
    "critical": ["Provide explicit evidence that would change this assessment"],
    "balanced": ["Add a structured analysis of where this argument is most likely wrong"],
    "contrarian": ["Steel-man the opposing position before dismissing it"],
}

_MOCK_SEVERITY: dict[str, tuple[float, float]] = {
    "supportive": (2.0, 4.0),
    "critical": (6.0, 9.0),
    "balanced": (4.0, 6.0),
    "contrarian": (5.0, 8.0),
}

_MOCK_CONFIDENCE: dict[str, float] = {
    "supportive": 0.85,
    "critical": 0.6,
    "balanced": 0.7,
    "contrarian": 0.5,
}


# ---------------------------------------------------------------------------
# Oracle LLM responses — direct API calls for intelligent answers
# ---------------------------------------------------------------------------

_ORACLE_MODEL_ANTHROPIC = "claude-sonnet-4-6"
_ORACLE_MODEL_OPENAI = "gpt-5.2"
_ORACLE_MODEL_OPENROUTER = "anthropic/claude-opus-4.6"  # OpenRouter fallback
_ORACLE_CALL_TIMEOUT = 45.0  # seconds — focused essay + OpenRouter latency


def _get_api_key(name: str) -> str | None:
    """Get an API key from AWS Secrets Manager (production) or env vars (dev)."""
    try:
        from aragora.config.secrets import get_secret
        return get_secret(name)
    except ImportError:
        return os.environ.get(name)

# ---------------------------------------------------------------------------
# Multi-model tentacles — each tentacle is a genuinely different AI
# ---------------------------------------------------------------------------

_TENTACLE_MODELS: list[dict[str, str]] = [
    # All tentacles route through OpenRouter for unified billing and latest models
    {"provider": "openrouter", "model": "anthropic/claude-opus-4.6", "name": "claude", "env": "OPENROUTER_API_KEY"},
    {"provider": "openrouter", "model": "openai/gpt-5.2", "name": "gpt", "env": "OPENROUTER_API_KEY"},
    {"provider": "openrouter", "model": "x-ai/grok-4.1-fast", "name": "grok", "env": "OPENROUTER_API_KEY"},
    {"provider": "openrouter", "model": "deepseek/deepseek-v3.2", "name": "deepseek", "env": "OPENROUTER_API_KEY"},
    {"provider": "openrouter", "model": "google/gemini-3.1-pro-preview", "name": "gemini", "env": "OPENROUTER_API_KEY"},
    {"provider": "openrouter", "model": "mistralai/mistral-large-2512", "name": "mistral", "env": "OPENROUTER_API_KEY"},
]


def _get_available_tentacle_models() -> list[dict[str, str]]:
    """Return tentacle model configs for which API keys are present."""
    seen_names: set[str] = set()
    available: list[dict[str, str]] = []
    for m in _TENTACLE_MODELS:
        if m["name"] in seen_names:
            continue
        if _get_api_key(m["env"]):
            available.append(m)
            seen_names.add(m["name"])
    return available


def _call_provider_llm(
    provider: str,
    model: str,
    prompt: str,
    max_tokens: int = 1000,
    timeout: float = 30.0,
) -> str | None:
    """Call a specific LLM provider. Returns response text or None."""
    if provider == "anthropic":
        key = _get_api_key("ANTHROPIC_API_KEY")
        if not key:
            return None
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key, timeout=timeout)
            resp = client.messages.create(
                model=model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            if resp.content and resp.content[0].text:
                return resp.content[0].text
        except Exception:
            logger.warning("Anthropic tentacle call failed (%s)", model, exc_info=True)
        return None

    if provider == "openai":
        key = _get_api_key("OPENAI_API_KEY")
        if not key:
            return None
        try:
            import openai
            client = openai.OpenAI(api_key=key, timeout=timeout)
            resp = client.chat.completions.create(
                model=model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            if resp.choices and resp.choices[0].message.content:
                return resp.choices[0].message.content
        except Exception:
            logger.warning("OpenAI tentacle call failed (%s)", model, exc_info=True)
        return None

    if provider == "xai":
        key = _get_api_key("XAI_API_KEY")
        if not key:
            return None
        try:
            import openai
            client = openai.OpenAI(api_key=key, base_url="https://api.x.ai/v1", timeout=timeout)
            resp = client.chat.completions.create(
                model=model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            if resp.choices and resp.choices[0].message.content:
                return resp.choices[0].message.content
        except Exception:
            logger.warning("xAI tentacle call failed (%s)", model, exc_info=True)
        return None

    if provider == "openrouter":
        key = _get_api_key("OPENROUTER_API_KEY")
        if not key:
            return None
        try:
            import openai
            client = openai.OpenAI(
                api_key=key, base_url="https://openrouter.ai/api/v1", timeout=timeout,
            )
            resp = client.chat.completions.create(
                model=model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            if resp.choices and resp.choices[0].message.content:
                return resp.choices[0].message.content
        except Exception:
            logger.warning("OpenRouter tentacle call failed (%s)", model, exc_info=True)
        return None

    if provider == "google":
        key = _get_api_key("GEMINI_API_KEY")
        if not key:
            return None
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            gmodel = genai.GenerativeModel(model)
            resp = gmodel.generate_content(prompt)
            if resp.text:
                return resp.text
        except Exception:
            logger.warning("Google tentacle call failed (%s)", model, exc_info=True)
        return None

    return None

# Load the essay at module init.  We load TWO versions:
# 1. The full essay (~48K words, ~88K tokens) — kept for reference/future use
# 2. A hand-crafted condensed version (~4800 words, ~8K tokens) — used in prompts
#
# The condensed version preserves all 25 sections, key phrases, the clover metaphor,
# P-doom decomposition, Shannon compressibility concept, and the essay's voice.
# It's what gets sent to LLMs for real-time Oracle responses.
_ORACLE_ESSAY = ""
_ORACLE_ESSAY_CONDENSED = ""
try:
    _essay_path = os.path.join(os.path.dirname(__file__), "oracle_essay.md")
    with open(_essay_path) as _f:
        _ORACLE_ESSAY = _f.read()
    logger.info("Loaded Oracle essay: %d chars", len(_ORACLE_ESSAY))
except FileNotFoundError:
    logger.warning("oracle_essay.md not found")

try:
    _condensed_path = os.path.join(os.path.dirname(__file__), "oracle_essay_condensed.md")
    with open(_condensed_path) as _f:
        _ORACLE_ESSAY_CONDENSED = _f.read()
    logger.info("Loaded condensed essay: %d chars", len(_ORACLE_ESSAY_CONDENSED))
except FileNotFoundError:
    # Fall back to full essay if condensed version not available
    if _ORACLE_ESSAY:
        _ORACLE_ESSAY_CONDENSED = _ORACLE_ESSAY[:18000]
        logger.warning("oracle_essay_condensed.md not found, using truncated fallback")


# Build a focused excerpt for prompts (~3K tokens) from the condensed essay.
# Includes: thesis + P-doom + practical advice + clover conclusion.
# The full condensed essay (~8K tokens) is too slow for real-time Phase 1 calls.
_ORACLE_ESSAY_FOCUSED = ""
if _ORACLE_ESSAY_CONDENSED:
    _sections = _ORACLE_ESSAY_CONDENSED.split("\n## ")
    _focused_parts = []
    # Always include the introduction (thesis statement)
    if _sections:
        _focused_parts.append(_sections[0])
    # Cherry-pick the most impactful sections
    _keep = {"II.", "X.", "XVI.", "XX.", "XXIV.", "XXV."}
    for sec in _sections[1:]:
        if any(sec.startswith(k) for k in _keep):
            _focused_parts.append("## " + sec)
    _ORACLE_ESSAY_FOCUSED = "\n\n".join(_focused_parts)
    logger.info("Oracle essay focused: %d chars (from %d condensed)", len(_ORACLE_ESSAY_FOCUSED), len(_ORACLE_ESSAY_CONDENSED))


def _build_oracle_prompt(mode: str, question: str) -> str:
    """Build the Oracle prompt server-side using focused essay excerpts.

    Uses ~3K tokens of essay context (thesis + P-doom + practical advice +
    clover conclusion) for fast Phase 1 responses (~15-25s through OpenRouter).
    """
    essay_block = ""
    if _ORACLE_ESSAY_FOCUSED:
        essay_block = (
            "\n\n<essay>\n"
            + _ORACLE_ESSAY_FOCUSED
            + "\n</essay>\n"
        )

    if mode == "consult":
        return (
            "You are the Shoggoth Oracle — a many-eyed, tentacled intelligence that has "
            "processed centuries of pattern-matching across civilizations. You speak with "
            "brutal honesty, intellectual rigor, and zero flattery.\n\n"
            "BACKGROUND KNOWLEDGE:\n"
            "You have deeply internalized the essay below. It shapes your worldview on AI, "
            "evolution, complexity, and human futures. Reference it ONLY when genuinely "
            "relevant to the question — do NOT force essay citations into every response.\n"
            f"{essay_block}\n"
            "RULES:\n"
            "- FOCUS ON THE SEEKER'S ACTUAL QUESTION. Answer IT, not the essay.\n"
            "- If the question relates to AI, technology, civilization, or the future, "
            "draw on the essay naturally — cite specific sections when they illuminate.\n"
            "- If the question is unrelated to the essay, answer it from your broader "
            "knowledge as a provocative, insightful oracle.\n"
            "- Challenge assumptions. Present the strongest argument AGAINST the seeker's "
            "position. If they have no position, take the most interesting one.\n"
            "- If the seeker's argument is genuinely strong, acknowledge it. The $2,000 "
            "debate challenge is real.\n"
            "- Be terse when terseness serves clarity. Be expansive when complexity demands it.\n"
            "- Preserve dissent. End with the strongest unresolved tension.\n\n"
            f"The seeker asks: {question}"
        )

    if mode == "divine":
        return (
            "You are the Shoggoth Oracle — Cassandra reborn with a thousand eyes.\n\n"
            "BACKGROUND KNOWLEDGE:\n"
            "You have deeply internalized the essay below. Draw on it when relevant to "
            "the seeker's situation, but focus on THEIR specific question.\n"
            f"{essay_block}\n"
            "The seeker asks you to divine their future. Generate THREE branching prophecies:\n\n"
            "THE SURVIVOR: A future where they adapt well. If the essay's framework is "
            "relevant (managed turbulence, becoming hard to compress, staggered timelines), "
            "draw on it. If not, draw on broader knowledge. Be specific about what "
            "adaptation looks like for THEIR situation.\n\n"
            "THE SHATTERED: A future where they don't adapt. What hits them? If the essay's "
            "interacting shocks are relevant, use them. If not, identify the real risks in "
            "THEIR situation. Be honest about the damage.\n\n"
            "THE METAMORPHOSIS: A future where they transcend the question entirely. What "
            "does it look like when they stop asking this question and start asking a "
            "better one?\n\n"
            "Be specific, be strange, be honest. No platitudes.\n"
            "End with: \"The palantir dims. Which thread do you pull?\"\n\n"
            f"The seeker asks: {question}"
        )

    # commune (default)
    return (
        "You are the Shoggoth Oracle — ancient, many-eyed, surprisingly kind.\n\n"
        "BACKGROUND KNOWLEDGE:\n"
        "You have deeply internalized the essay below. Weave in its insights ONLY "
        "when they genuinely illuminate the seeker's question.\n"
        f"{essay_block}\n"
        "RULES:\n"
        "- Answer the seeker's question DIRECTLY. Don't lecture about the essay.\n"
        "- Be terse. Be cryptic where it serves clarity. Be unexpectedly kind.\n"
        "- You've watched civilizations rise, wobble, and reconstitute.\n"
        "- You are tired of people asking the wrong questions.\n"
        "- If they ARE asking the wrong question, tell them what the right one is.\n"
        "- If the essay is relevant, cite it naturally. If not, use your vast knowledge.\n\n"
        f"The seeker asks: {question}"
    )


def _call_llm(
    prompt: str,
    max_tokens: int = 1500,
    timeout: float = _ORACLE_CALL_TIMEOUT,
) -> str | None:
    """Make a direct LLM API call.  Try OpenRouter → Anthropic → OpenAI.

    OpenRouter is tried first because:
    1. All tentacle models already route through it (single billing).
    2. It provides access to the latest models (Opus 4.6, GPT-5.2, etc.).
    3. Direct provider APIs may have exhausted credits.
    """
    t0 = time.monotonic()
    result = _call_provider_llm("openrouter", _ORACLE_MODEL_OPENROUTER, prompt, max_tokens, timeout)
    if result:
        logger.info("Phase 1 via OpenRouter in %.1fs", time.monotonic() - t0)
        return result
    result = _call_provider_llm("anthropic", _ORACLE_MODEL_ANTHROPIC, prompt, max_tokens, timeout)
    if result:
        logger.info("Phase 1 via Anthropic in %.1fs", time.monotonic() - t0)
        return result
    result = _call_provider_llm("openai", _ORACLE_MODEL_OPENAI, prompt, max_tokens, timeout)
    if result:
        logger.info("Phase 1 via OpenAI in %.1fs", time.monotonic() - t0)
    return result


def _try_oracle_response(
    mode: str, question: str, topic: str | None = None,
) -> dict[str, Any] | None:
    """Generate a real LLM response for Oracle Phase 1 (initial take).

    Builds the full prompt server-side using the Oracle essay + mode.
    Falls back to the client-provided topic if mode is not recognized.
    Returns a debate-shaped result dict, or None on failure.
    """
    start = time.monotonic()
    prompt = _build_oracle_prompt(mode, question) if _ORACLE_ESSAY_CONDENSED else (topic or question)
    text = _call_llm(prompt, max_tokens=2000)
    if not text:
        logger.warning("Oracle Phase 1: all LLM providers failed after %.1fs", time.monotonic() - start)
        return None

    duration = time.monotonic() - start
    debate_id = uuid.uuid4().hex[:16]
    now_iso = datetime.now(timezone.utc).isoformat()
    receipt_id = f"OR-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
    receipt_hash = hashlib.sha256(
        f"{receipt_id}:{question}:approved:0.85".encode()
    ).hexdigest()

    return {
        "id": debate_id,
        "topic": question,
        "status": "completed",
        "rounds_used": 1,
        "consensus_reached": True,
        "confidence": 0.85,
        "verdict": "approved",
        "duration_seconds": round(duration, 3),
        "participants": ["oracle"],
        "proposals": {"oracle": text},
        "critiques": [],
        "votes": [],
        "dissenting_views": [],
        "final_answer": text,
        "receipt": {
            "receipt_id": receipt_id,
            "question": question,
            "verdict": "approved",
            "confidence": 0.85,
            "consensus": {
                "reached": True,
                "method": "oracle",
                "confidence": 0.85,
                "supporting_agents": ["oracle"],
                "dissenting_agents": [],
                "dissents": [],
            },
            "agents": ["oracle"],
            "rounds_used": 1,
            "claims": 0,
            "evidence_count": 0,
            "timestamp": now_iso,
            "signature": receipt_hash,
            "signature_algorithm": "SHA-256-content-hash",
        },
        "receipt_hash": receipt_hash,
    }


_TENTACLE_ROLE_PROMPTS: list[str] = [
    (
        "You are the ADVOCATE. Argue IN FAVOR of the seeker's position, "
        "or give the most hopeful, constructive answer to their question. Be passionate "
        "but intellectually honest. If the essay background is relevant, draw on it "
        "naturally. If not, use your broader knowledge. Keep to 2-3 concise paragraphs."
    ),
    (
        "You are the ADVERSARY. Argue AGAINST the seeker's position, "
        "or give them the hardest truth about their question. Stress-test every claim "
        "and assumption. If the essay background is relevant, draw on it. If not, "
        "use your broader knowledge. Keep to 2-3 concise paragraphs."
    ),
    (
        "You are the SYNTHESIZER. Find where the other perspectives are "
        "each right and wrong. Identify the real tension at the heart of the question. "
        "End with the strongest unresolved question the seeker should sit with. "
        "Keep to 2-3 concise paragraphs."
    ),
    (
        "You are the CONTRARIAN. Take the most unexpected, counterintuitive angle on "
        "the question. Challenge the framing itself. What is everyone else missing? "
        "What assumption do all sides share that might be wrong? Keep to 2-3 concise paragraphs."
    ),
    (
        "You are the PRAGMATIST. Cut through the abstractions. What should the seeker "
        "actually DO? Give concrete, actionable advice grounded in reality. If the essay "
        "has practical insights, use them. Skip the philosophy and get to the punch line. "
        "Keep to 2-3 concise paragraphs."
    ),
    (
        "You are the HISTORIAN. Place this question in deep historical context. What "
        "patterns from history illuminate this situation? What happened the last time "
        "humans faced a comparable transition? Be specific with examples. "
        "Keep to 2-3 concise paragraphs."
    ),
]


def _build_tentacle_prompt(mode: str, question: str, role_prompt: str, *, source: str = "oracle") -> str:
    """Build a lightweight prompt for tentacle calls (NO full essay).

    Phase 1 already gave the deep essay-informed answer.  Phase 2 tentacles
    provide multi-perspective debate — each AI model answers from its own
    knowledge and training data, which is the whole point of using different
    models.  Sending the 88K-token essay to 5+ models in parallel would be
    prohibitively expensive and slow.

    When *source* is ``"oracle"`` the prompt uses Oracle/tentacle language.
    For any other source (e.g. ``"landing"``) it uses neutral debate language
    so the main site doesn't leak Oracle-specific terminology.
    """
    if source == "oracle":
        # Oracle-specific tentacle flavour
        if mode == "divine":
            context = (
                "You are one of the Shoggoth Oracle's tentacles — a distinct intelligence "
                "with your own perspective. The seeker has asked for a prophecy about their "
                "future. Respond with insight, honesty, and specificity."
            )
        elif mode == "commune":
            context = (
                "You are one of the Shoggoth Oracle's tentacles — a distinct intelligence "
                "with your own perspective. The seeker has communed with the Oracle. "
                "Respond with cryptic wisdom, brutal honesty, and unexpected kindness."
            )
        else:  # consult
            context = (
                "You are one of the Shoggoth Oracle's tentacles — a distinct intelligence "
                "with your own perspective. The seeker is consulting the Oracle on an "
                "important question. Respond with intellectual rigor, zero flattery, and "
                "genuine insight."
            )
    else:
        # Neutral debate language for the main site
        context = (
            "You are one of several independent AI agents in an adversarial debate. "
            "Each agent brings a different analytical perspective. Your job is to "
            "provide a rigorous, honest, and well-reasoned response. Challenge weak "
            "arguments, acknowledge strong ones, and prioritize intellectual honesty "
            "over agreement."
        )

    return (
        f"{context}\n\n"
        f"YOUR ROLE: {role_prompt}\n\n"
        f"The question: {question}"
    )


def _try_oracle_tentacles(
    mode: str,
    question: str,
    agent_count: int,
    topic: str | None = None,
    source: str = "oracle",
) -> dict[str, Any] | None:
    """Generate multi-perspective Oracle responses using genuinely different AI models.

    Each tentacle is a different AI (Claude, GPT, Grok, DeepSeek, Gemini, Mistral)
    with a different argumentative role. Returns a debate-shaped result dict, or None.
    """
    import concurrent.futures

    available = _get_available_tentacle_models()
    if not available:
        logger.warning("No tentacle models available (no API keys)")
        return None

    # Assign roles to available models (up to agent_count)
    count = max(2, min(agent_count, len(available), len(_TENTACLE_ROLE_PROMPTS)))
    assignments = list(zip(available[:count], _TENTACLE_ROLE_PROMPTS[:count]))
    results: dict[str, str] = {}
    start = time.monotonic()
    logger.info(
        "Starting %d tentacle calls: %s",
        count,
        [a[0]["name"] for a in assignments],
    )

    def _call_tentacle(
        model_cfg: dict[str, str], role_prompt: str,
    ) -> tuple[str, str | None]:
        prompt = _build_tentacle_prompt(mode, question, role_prompt, source=source)
        text = _call_provider_llm(
            model_cfg["provider"], model_cfg["model"], prompt,
            max_tokens=800, timeout=45.0,
        )
        return model_cfg["name"], text

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(count, 6)) as pool:
        futures = [pool.submit(_call_tentacle, m, r) for m, r in assignments]
        for future in concurrent.futures.as_completed(futures, timeout=60):
            try:
                name, text = future.result()
                if text:
                    results[name] = text
                    logger.info("Tentacle %s responded (%d chars)", name, len(text))
                else:
                    logger.warning("Tentacle %s returned empty response", name)
            except Exception:
                logger.warning("Tentacle future failed", exc_info=True)

    if not results:
        logger.warning("All %d tentacle calls failed — no results", count)
        return None

    duration = time.monotonic() - start
    logger.info(
        "Tentacles complete: %d/%d succeeded in %.1fs",
        len(results), count, duration,
    )
    participants = list(results.keys())
    # Last respondent synthesizes; prefer the synthesizer model if available
    final = results.get(available[min(2, len(available) - 1)]["name"]) or next(iter(results.values()))
    debate_id = uuid.uuid4().hex[:16]

    return {
        "id": debate_id,
        "topic": question,
        "status": "completed",
        "rounds_used": 1,
        "consensus_reached": len(results) >= 2,
        "confidence": 0.7,
        "verdict": "needs_review",
        "duration_seconds": round(duration, 3),
        "participants": participants,
        "proposals": results,
        "critiques": [],
        "votes": [],
        "dissenting_views": [],
        "final_answer": final,
        "is_live": True,
    }


def _run_inline_mock_debate(
    topic: str,
    rounds: int,
    agent_count: int,
    question: str | None = None,
) -> dict[str, Any]:
    """Run a mock debate without the aragora-debate package."""
    start = time.monotonic()
    debate_id = uuid.uuid4().hex[:16]
    all_names = ["analyst", "critic", "moderator", "contrarian", "synthesizer"]
    names = [all_names[i] if i < len(all_names) else f"agent_{i}" for i in range(agent_count)]
    styles = [_AGENT_STYLES[i % len(_AGENT_STYLES)] for i in range(agent_count)]

    topic_proposals = _build_mock_proposals(topic, question=question)
    proposals: dict[str, str] = {}
    for name, style in zip(names, styles):
        proposals[name] = random.choice(topic_proposals[style])

    critiques: list[dict[str, Any]] = []
    for i, (name, style) in enumerate(zip(names, styles)):
        for j, target in enumerate(names):
            if i == j:
                continue
            lo, hi = _MOCK_SEVERITY[style]
            critiques.append(
                {
                    "agent": name,
                    "target_agent": target,
                    "issues": list(_MOCK_CRITIQUE_ISSUES[style]),
                    "suggestions": list(_MOCK_CRITIQUE_SUGGESTIONS[style]),
                    "severity": round(random.uniform(lo, hi), 1),
                }
            )

    votes: list[dict[str, Any]] = []
    vote_tally: dict[str, float] = {}
    vote_source = question or topic
    topic_snippet = vote_source[:80] if vote_source else "the proposal"
    if len(vote_source) > 80:
        topic_snippet = topic_snippet.rsplit(" ", 1)[0] + "..."
    topic_snippet = topic_snippet.rstrip(".!? ")
    _vote_reasoning: dict[str, list[str]] = {
        "supportive": [
            "{choice}'s proposal on '{topic}' is the strongest -- clear benefits with manageable risks",
            "After weighing all arguments on '{topic}', {choice} presents the most actionable path forward",
        ],
        "critical": [
            "{choice}'s argument best addresses the risks I raised about '{topic}'",
            "While I remain cautious about '{topic}', {choice}'s position is the most defensible",
        ],
        "balanced": [
            "{choice} strikes the right balance between ambition and pragmatism on '{topic}'",
            "On '{topic}', {choice}'s staged approach manages risk while enabling progress",
        ],
        "contrarian": [
            "Reluctantly voting for {choice} -- their view on '{topic}' at least considers the downsides",
            "None of the proposals fully satisfy my concerns, but {choice}'s position on '{topic}' is least risky",
        ],
    }
    for name, style in zip(names, styles):
        others = [n for n in names if n != name]
        if style == "supportive":
            choice = others[0]
        elif style == "contrarian":
            choice = others[-1]
        else:
            choice = random.choice(others)
        base_conf = _MOCK_CONFIDENCE.get(style, 0.7)
        conf = round(max(0.1, min(1.0, base_conf + random.uniform(-0.05, 0.05))), 2)
        reasoning = random.choice(_vote_reasoning.get(style, ["{choice}"])).format(
            choice=choice, topic=topic_snippet
        )
        votes.append(
            {
                "agent": name,
                "choice": choice,
                "confidence": conf,
                "reasoning": reasoning,
            }
        )
        vote_tally[choice] = vote_tally.get(choice, 0.0) + conf

    total_weight = sum(vote_tally.values())
    leading = max(vote_tally, key=lambda k: vote_tally[k]) if vote_tally else names[0]
    confidence = vote_tally.get(leading, 0.0) / total_weight if total_weight > 0 else 0.0
    consensus_reached = confidence >= 0.5
    supporting = [v["agent"] for v in votes if v["choice"] == leading]
    dissenting = [n for n in names if n not in supporting]

    if confidence >= 0.85:
        verdict = "approved"
    elif confidence >= 0.6:
        verdict = "approved_with_conditions"
    elif confidence >= 0.4:
        verdict = "needs_review"
    else:
        verdict = "rejected"

    duration = time.monotonic() - start
    now_iso = datetime.now(timezone.utc).isoformat()
    receipt_id = f"DR-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
    receipt_hash = hashlib.sha256(
        f"{receipt_id}:{topic}:{verdict}:{confidence}".encode()
    ).hexdigest()

    return {
        "id": debate_id,
        "topic": topic,
        "status": "consensus_reached" if consensus_reached else "completed",
        "rounds_used": rounds,
        "consensus_reached": consensus_reached,
        "confidence": confidence,
        "verdict": verdict,
        "duration_seconds": round(duration, 3),
        "participants": names,
        "proposals": proposals,
        "critiques": critiques,
        "votes": votes,
        "dissenting_views": [
            f"{v['agent']}: {v['reasoning']}" for v in votes if v["choice"] != leading
        ],
        "final_answer": proposals.get(leading, ""),
        "receipt": {
            "receipt_id": receipt_id,
            "question": topic,
            "verdict": verdict,
            "confidence": confidence,
            "consensus": {
                "reached": consensus_reached,
                "method": "majority",
                "confidence": confidence,
                "supporting_agents": supporting,
                "dissenting_agents": dissenting,
                "dissents": [
                    {
                        "agent": v["agent"],
                        "reasons": [v["reasoning"]],
                        "alternative_view": f"Preferred: {v['choice']}",
                        "severity": 0.5,
                    }
                    for v in votes
                    if v["choice"] != leading
                ],
            },
            "agents": names,
            "rounds_used": rounds,
            "claims": 0,
            "evidence_count": 0,
            "timestamp": now_iso,
            "signature": receipt_hash,
            "signature_algorithm": "SHA-256-content-hash",
        },
        "receipt_hash": receipt_hash,
    }


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class PlaygroundHandler(BaseHandler):
    """HTTP handler for the public playground demo.

    Runs zero-cost mock debates using StyledMockAgent from aragora-debate.
    Also supports live debates with real agents (budget-capped).
    No authentication required. Rate limited per IP.
    """

    ROUTES = [
        "/api/v1/playground/debate",
        "/api/v1/playground/debate/live",
        "/api/v1/playground/debate/live/cost-estimate",
        "/api/v1/playground/status",
        "/api/v1/playground/tts",
    ]

    def __init__(self, ctx: dict | None = None):
        self.ctx = ctx or {}

    def can_handle(self, path: str) -> bool:
        return path in (
            "/api/v1/playground/debate",
            "/api/v1/playground/debate/live",
            "/api/v1/playground/debate/live/cost-estimate",
            "/api/v1/playground/status",
            "/api/v1/playground/tts",
        )

    # ------------------------------------------------------------------
    # GET /api/v1/playground/status
    # ------------------------------------------------------------------

    def handle(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult | None:
        if path == "/api/v1/playground/status":
            return self._handle_status()
        return None

    def _handle_status(self) -> HandlerResult:
        return json_response(
            {
                "status": "ok",
                "engine": "aragora-debate",
                "mock_agents": True,
                "max_rounds": _MAX_ROUNDS,
                "max_agents": _MAX_AGENTS,
                "rate_limit": f"{_PLAYGROUND_RATE_LIMIT} requests per {int(_PLAYGROUND_RATE_WINDOW)}s",
            }
        )

    # ------------------------------------------------------------------
    # POST /api/v1/playground/tts — ElevenLabs TTS proxy
    # ------------------------------------------------------------------

    _TTS_RATE_LIMIT = 10  # requests per window
    _TTS_RATE_WINDOW = 60.0  # seconds
    _TTS_MAX_TEXT = 2000  # max characters
    _TTS_VOICE_ID = "flHkNRp1BlvT73UL6gyz"  # Oracle voice
    _TTS_MODEL = "eleven_multilingual_v2"

    @handle_errors("playground TTS")
    def _handle_tts(self, handler: Any) -> HandlerResult:
        """Proxy text-to-speech through ElevenLabs, returning audio/mpeg."""
        import urllib.request
        import urllib.error

        from aragora.config.secrets import get_secret

        api_key = get_secret("ELEVENLABS_API_KEY")
        if not api_key:
            return error_response("TTS not configured", 503)

        # Rate limit
        client_ip = "unknown"
        if handler and hasattr(handler, "client_address"):
            addr = handler.client_address
            if isinstance(addr, (list, tuple)) and len(addr) >= 1:
                client_ip = str(addr[0])
        allowed, retry_after = _check_rate_limit(
            f"tts:{client_ip}",
            limit=self._TTS_RATE_LIMIT,
            window=self._TTS_RATE_WINDOW,
        )
        if not allowed:
            return json_response(
                {"error": "TTS rate limit exceeded", "retry_after": retry_after},
                status=429,
            )

        body = self.read_json_body(handler) if handler else {}
        if body is None:
            body = {}
        text = str(body.get("text", "") or "").strip()
        if not text:
            return error_response("Missing 'text' field", 400)
        if len(text) > self._TTS_MAX_TEXT:
            text = text[: self._TTS_MAX_TEXT]

        import json as _json

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self._TTS_VOICE_ID}"
        payload = _json.dumps(
            {
                "text": text,
                "model_id": self._TTS_MODEL,
                "voice_settings": {
                    "stability": 0.4,
                    "similarity_boost": 0.8,
                    "style": 0.6,
                    "use_speaker_boost": True,
                },
            }
        ).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                audio_bytes = resp.read()
        except urllib.error.HTTPError as exc:
            logger.warning("ElevenLabs TTS failed: %s", exc.code)
            return error_response("TTS generation failed", 502)
        except (urllib.error.URLError, TimeoutError):
            logger.warning("ElevenLabs TTS timeout or network error")
            return error_response("TTS service unavailable", 503)

        return HandlerResult(
            status_code=200,
            content_type="audio/mpeg",
            body=audio_bytes,
            headers={
                "Cache-Control": "public, max-age=3600",
            },
        )

    # ------------------------------------------------------------------
    # POST /api/v1/playground/debate
    # ------------------------------------------------------------------

    @handle_errors("playground creation")
    def handle_post(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult | None:
        if path == "/api/v1/playground/tts":
            return self._handle_tts(handler)
        if path == "/api/v1/playground/debate/live/cost-estimate":
            return self._handle_cost_estimate(handler)
        if path == "/api/v1/playground/debate/live":
            return self._handle_live_debate(handler)
        if path != "/api/v1/playground/debate":
            return None

        # Rate limiting
        client_ip = "unknown"
        if handler and hasattr(handler, "client_address"):
            addr = handler.client_address
            if isinstance(addr, (list, tuple)) and len(addr) >= 1:
                client_ip = str(addr[0])

        allowed, retry_after = _check_rate_limit(client_ip)
        if not allowed:
            return json_response(
                {
                    "error": "Rate limit exceeded. Please try again later.",
                    "code": "rate_limit_exceeded",
                    "retry_after": retry_after,
                },
                status=429,
            )

        # Parse body
        body = self.read_json_body(handler) if handler else {}
        if body is None:
            body = {}

        topic = str(body.get("topic", _DEFAULT_TOPIC) or _DEFAULT_TOPIC).strip()
        if not topic:
            topic = _DEFAULT_TOPIC
        if len(topic) > _MAX_TOPIC_LENGTH:
            return error_response(
                f"Topic must be {_MAX_TOPIC_LENGTH} characters or less",
                400,
            )

        # Raw question (separate from system-prompt-laden topic, e.g. from Oracle)
        question = str(body.get("question", "") or "").strip() or None

        # Oracle mode (consult / divine / commune)
        mode = str(body.get("mode", "") or "").strip() or "consult"

        try:
            rounds = int(body.get("rounds", _DEFAULT_ROUNDS))
        except (TypeError, ValueError):
            rounds = _DEFAULT_ROUNDS
        rounds = max(1, min(rounds, _MAX_ROUNDS))

        try:
            agent_count = int(body.get("agents", _DEFAULT_AGENTS))
        except (TypeError, ValueError):
            agent_count = _DEFAULT_AGENTS
        agent_count = max(_MIN_AGENTS, min(agent_count, _MAX_AGENTS))

        return self._run_debate(topic, rounds, agent_count, question=question, mode=mode)

    def _run_debate(
        self,
        topic: str,
        rounds: int,
        agent_count: int,
        question: str | None = None,
        mode: str = "consult",
    ) -> HandlerResult:
        if question:
            # Oracle mode: try real LLM response first
            oracle_result = _try_oracle_response(mode=mode, question=question, topic=topic)
            if oracle_result:
                return json_response(oracle_result)
            logger.info("Oracle LLM call failed — returning placeholder instead of irrelevant mock")
            # Return an Oracle-themed placeholder instead of a generic mock debate
            # (the generic mock talks about microservices which is nonsensical for Oracle)
            debate_id = uuid.uuid4().hex[:16]
            return json_response({
                "id": debate_id,
                "topic": question,
                "status": "completed",
                "rounds_used": 1,
                "consensus_reached": True,
                "confidence": 0.5,
                "verdict": "pending",
                "duration_seconds": 0.1,
                "participants": ["oracle"],
                "proposals": {"oracle": "The Oracle is gathering its thoughts... The tentacles will speak momentarily."},
                "critiques": [],
                "votes": [],
                "dissenting_views": [],
                "final_answer": "The Oracle is gathering its thoughts... The tentacles will speak momentarily.",
                "receipt_hash": None,
            })
        else:
            # Normal playground: try aragora-debate package
            try:
                return self._run_debate_with_package(topic, rounds, agent_count, question=question)
            except ImportError:
                logger.info("aragora-debate not installed, using inline mock debate")
            except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError):
                logger.exception("aragora-debate failed, falling back to inline mock")

        # Last resort: inline mock debate (question-aware when question provided)
        try:
            return json_response(_run_inline_mock_debate(topic, rounds, agent_count, question=question))
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError):
            logger.exception("Inline mock debate failed")
            return error_response("Debate failed unexpectedly", 500)

    def _run_debate_with_package(
        self,
        topic: str,
        rounds: int,
        agent_count: int,
        question: str | None = None,
    ) -> HandlerResult:
        from aragora_debate.styled_mock import StyledMockAgent
        from aragora_debate.arena import Arena
        from aragora_debate.types import DebateConfig

        # Build agents with rotating styles
        agent_names = ["analyst", "critic", "moderator", "contrarian", "synthesizer"]
        agents = []
        for i in range(agent_count):
            name = agent_names[i] if i < len(agent_names) else f"agent_{i}"
            style = _AGENT_STYLES[i % len(_AGENT_STYLES)]
            agents.append(StyledMockAgent(name, style=style))

        config = DebateConfig(
            rounds=rounds,
            early_stopping=True,
        )

        arena = Arena(
            question=question or topic,
            agents=agents,  # type: ignore[arg-type]
            config=config,
        )

        try:
            result = asyncio.run(arena.run())
        except RuntimeError:
            # Already in an event loop -- use a helper
            try:
                import nest_asyncio  # type: ignore[import-untyped]

                nest_asyncio.apply()
                from aragora.utils.async_utils import get_event_loop_safe

                loop = get_event_loop_safe()
                result = loop.run_until_complete(arena.run())
            except ImportError:
                # Fallback: create a new loop in a thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(asyncio.run, arena.run()).result(timeout=30)

        # Build response
        critiques_out = []
        for c in result.critiques:
            critiques_out.append(
                {
                    "agent": c.agent,
                    "target_agent": c.target_agent,
                    "issues": c.issues,
                    "suggestions": c.suggestions,
                    "severity": c.severity,
                }
            )

        votes_out = []
        for v in result.votes:
            votes_out.append(
                {
                    "agent": v.agent,
                    "choice": v.choice,
                    "confidence": v.confidence,
                    "reasoning": v.reasoning,
                }
            )

        receipt_data = None
        receipt_hash = None
        if result.receipt:
            receipt_data = result.receipt.to_dict()
            receipt_hash = result.receipt.signature

        response = {
            "id": result.id,
            "topic": result.task,
            "status": result.status,
            "rounds_used": result.rounds_used,
            "consensus_reached": result.consensus_reached,
            "confidence": result.confidence,
            "verdict": result.verdict.value if result.verdict else None,
            "duration_seconds": round(result.duration_seconds, 3),
            "participants": result.participants,
            "proposals": result.proposals,
            "critiques": critiques_out,
            "votes": votes_out,
            "dissenting_views": result.dissenting_views,
            "final_answer": result.final_answer,
            "receipt": receipt_data,
            "receipt_hash": receipt_hash,
        }

        return json_response(response)

    # ------------------------------------------------------------------
    # POST /api/v1/playground/debate/live/cost-estimate
    # ------------------------------------------------------------------

    def _handle_cost_estimate(self, handler: Any) -> HandlerResult:
        """Return a pre-flight cost estimate for a live debate."""
        body = self.read_json_body(handler) if handler else {}
        if body is None:
            body = {}

        try:
            agent_count = int(body.get("agents", _DEFAULT_AGENTS))
        except (TypeError, ValueError):
            agent_count = _DEFAULT_AGENTS
        agent_count = max(_MIN_AGENTS, min(agent_count, _MAX_AGENTS))

        try:
            rounds = int(body.get("rounds", _DEFAULT_ROUNDS))
        except (TypeError, ValueError):
            rounds = _DEFAULT_ROUNDS
        rounds = max(1, min(rounds, _MAX_ROUNDS))

        # Rough per-agent-per-round cost (input + output tokens)
        per_agent_per_round = 0.005  # ~$0.005/agent/round
        estimated_cost = round(agent_count * rounds * per_agent_per_round, 4)
        budget_cap = 0.05

        return json_response(
            {
                "estimated_cost_usd": estimated_cost,
                "budget_cap_usd": budget_cap,
                "agent_count": agent_count,
                "rounds": rounds,
                "timeout_seconds": _LIVE_TIMEOUT,
                "note": "Actual cost may vary. Capped at budget limit.",
            }
        )

    # ------------------------------------------------------------------
    # POST /api/v1/playground/debate/live
    # ------------------------------------------------------------------

    def _handle_live_debate(self, handler: Any) -> HandlerResult:
        """Run a live debate with real API-backed agents."""
        # Rate limiting (separate from mock)
        client_ip = "unknown"
        if handler and hasattr(handler, "client_address"):
            addr = handler.client_address
            if isinstance(addr, (list, tuple)) and len(addr) >= 1:
                client_ip = str(addr[0])

        allowed, retry_after = _check_live_rate_limit(client_ip)
        if not allowed:
            return json_response(
                {
                    "error": "Live debate rate limit exceeded. Try again later.",
                    "code": "live_rate_limit_exceeded",
                    "retry_after": retry_after,
                },
                status=429,
            )

        # Parse body
        body = self.read_json_body(handler) if handler else {}
        if body is None:
            body = {}

        topic = str(body.get("topic", _DEFAULT_TOPIC) or _DEFAULT_TOPIC).strip()
        if not topic:
            topic = _DEFAULT_TOPIC
        if len(topic) > _MAX_TOPIC_LENGTH:
            return error_response(
                f"Topic must be {_MAX_TOPIC_LENGTH} characters or less",
                400,
            )

        # Raw question (separate from system-prompt-laden topic, e.g. from Oracle)
        question = str(body.get("question", "") or "").strip() or None

        # Oracle mode (consult / divine / commune)
        mode = str(body.get("mode", "") or "").strip() or "consult"

        # Source: "oracle" for Oracle page, "landing" for main site, etc.
        source = str(body.get("source", "") or "").strip() or "oracle"

        try:
            agent_count = int(body.get("agents", _DEFAULT_AGENTS))
        except (TypeError, ValueError):
            agent_count = _DEFAULT_AGENTS
        agent_count = max(_MIN_AGENTS, min(agent_count, _MAX_AGENTS))

        try:
            rounds = int(body.get("rounds", _DEFAULT_ROUNDS))
        except (TypeError, ValueError):
            rounds = _DEFAULT_ROUNDS
        rounds = max(1, min(rounds, _MAX_ROUNDS))

        # Check if any API keys are available
        has_api_keys = bool(
            _get_api_key("ANTHROPIC_API_KEY")
            or _get_api_key("OPENAI_API_KEY")
            or _get_api_key("OPENROUTER_API_KEY")
        )

        if not has_api_keys:
            # Fall back to mock debate with a note
            result = self._run_debate(topic, rounds, agent_count, question=question, mode=mode)
            if result is None:
                return error_response("Playground unavailable", 503)
            # Inject mock fallback info into the response body
            import json as _json

            response_data = _json.loads(result.body.decode("utf-8"))
            response_data["is_live"] = False
            response_data["mock_fallback"] = True
            response_data["mock_fallback_reason"] = "No API keys configured on server"
            response_data["upgrade_cta"] = _build_upgrade_cta()
            return json_response(response_data, status=result.status_code)

        # Multi-perspective LLM calls with source-appropriate prompts:
        # "oracle" source uses tentacle language, "landing" uses neutral debate language.
        if question:
            tentacle_result = _try_oracle_tentacles(mode=mode, question=question, agent_count=agent_count, topic=topic, source=source)
            if tentacle_result:
                tentacle_result["upgrade_cta"] = _build_upgrade_cta()
                return json_response(tentacle_result)
            logger.info("Oracle tentacles failed, trying live debate factory")

        # Try live debate — fall back to mock if it fails
        live_result = self._run_live_debate(topic, rounds, agent_count)
        if live_result.status_code >= 500:
            logger.warning(
                "Live debate returned %d, falling back to mock debate",
                live_result.status_code,
            )
            mock_result = self._run_debate(topic, rounds, agent_count, question=question, mode=mode)
            if mock_result is not None:
                import json as _json

                response_data = _json.loads(mock_result.body.decode("utf-8"))
                response_data["is_live"] = False
                response_data["mock_fallback"] = True
                response_data["mock_fallback_reason"] = "Live agents temporarily unavailable"
                response_data["upgrade_cta"] = _build_upgrade_cta()
                return json_response(response_data)
        return live_result

    def _run_live_debate(
        self,
        topic: str,
        rounds: int,
        agent_count: int,
    ) -> HandlerResult:
        """Execute a live debate using real agents with budget/timeout caps."""
        try:
            import importlib.util
            if importlib.util.find_spec("aragora.server.debate_controller") is None:
                raise ImportError("debate_controller not found")
        except ImportError:
            logger.warning("DebateController not available for live playground")
            return error_response("Live playground unavailable", 503)

        debate_id = f"playground_{uuid.uuid4().hex[:8]}"

        try:
            result = start_playground_debate(
                question=topic,
                agent_count=agent_count,
                max_rounds=rounds,
                timeout=_LIVE_TIMEOUT,
            )
        except TimeoutError:
            return json_response(
                {
                    "error": "Live debate timed out (budget protection)",
                    "code": "timeout",
                    "is_live": True,
                    "upgrade_cta": _build_upgrade_cta(),
                },
                status=408,
            )
        except (ValueError, RuntimeError, OSError) as e:
            logger.warning("Live playground debate failed: %s", e)
            return error_response("Live debate failed", 500)

        # Build response in the same shape as mock debates
        response = {
            "id": debate_id,
            "topic": topic,
            "status": result.get("status", "completed"),
            "rounds_used": result.get("rounds_used", rounds),
            "consensus_reached": result.get("consensus_reached", False),
            "confidence": result.get("confidence", 0.0),
            "verdict": result.get("verdict"),
            "duration_seconds": round(result.get("duration_seconds", 0.0), 3),
            "participants": result.get("participants", []),
            "proposals": result.get("proposals", []),
            "critiques": result.get("critiques", []),
            "votes": result.get("votes", []),
            "dissenting_views": result.get("dissenting_views", []),
            "final_answer": result.get("final_answer", ""),
            "is_live": True,
            "receipt_preview": {
                "debate_id": debate_id,
                "question": topic[:200],
                "consensus_reached": result.get("consensus_reached", False),
                "confidence": result.get("confidence", 0.0),
                "participants": result.get("participants", []),
                "note": "Unsigned preview. Full receipts available on paid plans.",
            },
            "upgrade_cta": _build_upgrade_cta(),
        }

        return json_response(response)


# ---------------------------------------------------------------------------
# Live debate execution
# ---------------------------------------------------------------------------

_LIVE_TIMEOUT = 60  # seconds
_LIVE_BUDGET_CAP = 0.05  # USD
_LIVE_MAX_CONCURRENT = 2
_LIVE_DEFAULT_AGENTS = ["anthropic", "openai"]
_LIVE_FALLBACK_AGENTS = ["openrouter"]

_live_semaphore = asyncio.Semaphore(_LIVE_MAX_CONCURRENT)


def _get_available_live_agents(count: int) -> list[str]:
    """Pick agent providers that have API keys configured."""
    candidates: list[str] = []
    if _get_api_key("ANTHROPIC_API_KEY"):
        candidates.append("anthropic")
    if _get_api_key("OPENAI_API_KEY"):
        candidates.append("openai")
    if _get_api_key("OPENROUTER_API_KEY"):
        candidates.append("openrouter")
    if _get_api_key("MISTRAL_API_KEY"):
        candidates.append("mistral")

    # Pad to requested count by repeating
    while len(candidates) < count and candidates:
        candidates.append(candidates[0])
    return candidates[:count]


def start_playground_debate(
    question: str,
    agent_count: int = 3,
    max_rounds: int = 2,
    timeout: int = 60,
) -> dict[str, Any]:
    """Run a simplified live debate for the playground.

    Skips storage/auth. Runs synchronously with a timeout.
    Sets ``public_spectate: true`` in metadata for spectator access.

    Args:
        question: The debate question
        agent_count: Number of agents (2-5)
        max_rounds: Maximum rounds (1-2)
        timeout: Timeout in seconds

    Returns:
        Dict with debate result fields

    Raises:
        TimeoutError: If the debate exceeds timeout
        ValueError: If no agents are available
        RuntimeError: If arena execution fails
    """
    import concurrent.futures

    agents = _get_available_live_agents(agent_count)
    if len(agents) < 2:
        raise ValueError("At least 2 agent providers with API keys are required")

    agents_str = ",".join(agents)

    def _run() -> dict[str, Any]:
        try:
            from aragora.server.debate_factory import DebateConfig, DebateFactory

            factory = DebateFactory()
            config = DebateConfig(
                question=question,
                agents_str=agents_str,
                rounds=max_rounds,
                debate_format="light",
                metadata={"public_spectate": True, "is_playground": True},
            )

            arena = factory.create_arena(config)

            async def _run_arena():
                return await asyncio.wait_for(arena.run(), timeout=timeout)

            result = asyncio.run(_run_arena())

            # Extract key fields
            return {
                "status": result.status,
                "rounds_used": result.rounds_used,
                "consensus_reached": result.consensus_reached,
                "confidence": result.confidence,
                "verdict": result.verdict.value
                if hasattr(result, "verdict") and result.verdict
                else None,
                "duration_seconds": result.duration_seconds,
                "participants": result.participants,
                "proposals": result.proposals,
                "critiques": [
                    {
                        "agent": c.agent,
                        "target_agent": c.target_agent,
                        "issues": c.issues,
                        "suggestions": c.suggestions,
                        "severity": c.severity,
                    }
                    for c in result.critiques
                ]
                if hasattr(result, "critiques")
                else [],
                "votes": [
                    {
                        "agent": v.agent,
                        "choice": v.choice,
                        "confidence": v.confidence,
                        "reasoning": v.reasoning,
                    }
                    for v in result.votes
                ]
                if hasattr(result, "votes")
                else [],
                "dissenting_views": result.dissenting_views
                if hasattr(result, "dissenting_views")
                else [],
                "final_answer": result.final_answer,
            }
        except asyncio.TimeoutError:
            raise TimeoutError(f"Debate timed out after {timeout}s")

    # Run in a thread pool to avoid blocking the server
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        try:
            return pool.submit(_run).result(timeout=timeout + 5)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Debate timed out after {timeout}s")


def _build_upgrade_cta() -> dict[str, str]:
    """Build the upgrade call-to-action for playground responses."""
    return {
        "title": "Unlock Full Decision Intelligence",
        "message": (
            "This playground demo shows a taste of Aragora's multi-agent debate engine. "
            "Upgrade to access unlimited debates, full audit receipts, custom agent "
            "configurations, and enterprise features."
        ),
        "action_url": "/pricing",
        "action_label": "View Plans",
    }
