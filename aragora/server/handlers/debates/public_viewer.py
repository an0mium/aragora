"""
Public Debate Viewer Handler.

Serves debate results publicly (no auth required) for shared links,
plus OG metadata for social previews when sharing on Twitter/Slack/LinkedIn.

Routes:
    GET /api/v1/debates/public/{debate_id}      - Public debate JSON
    GET /api/v1/debates/public/{debate_id}/og    - OG meta tags HTML
"""

from __future__ import annotations

import html as html_mod
import logging
import re
import time
from typing import Any

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiting: 30 req/min per IP for public viewer
# ---------------------------------------------------------------------------

_PUBLIC_VIEWER_RATE_LIMIT = 30
_PUBLIC_VIEWER_RATE_WINDOW = 60.0  # seconds
_public_viewer_timestamps: dict[str, list[float]] = {}


def _check_public_viewer_rate_limit(client_ip: str) -> tuple[bool, int]:
    """Check rate limit for public viewer endpoints.

    Returns:
        (allowed, retry_after_seconds)
    """
    now = time.monotonic()
    cutoff = now - _PUBLIC_VIEWER_RATE_WINDOW

    timestamps = _public_viewer_timestamps.get(client_ip, [])
    timestamps = [t for t in timestamps if t > cutoff]

    if len(timestamps) >= _PUBLIC_VIEWER_RATE_LIMIT:
        oldest_in_window = timestamps[0]
        retry_after = int(oldest_in_window + _PUBLIC_VIEWER_RATE_WINDOW - now) + 1
        _public_viewer_timestamps[client_ip] = timestamps
        return False, max(retry_after, 1)

    timestamps.append(now)
    _public_viewer_timestamps[client_ip] = timestamps
    return True, 0


def _reset_public_viewer_rate_limits() -> None:
    """Reset rate limit state. Used by tests."""
    _public_viewer_timestamps.clear()


# ---------------------------------------------------------------------------
# Debate retrieval helpers
# ---------------------------------------------------------------------------

# Debate IDs can be:
# - Playground IDs: hex strings or playground_<hex>
# - Stored debate IDs: slug/UUID-like safe identifiers
_DEBATE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")

# Also support playground-prefixed IDs like playground_abcd1234
_PLAYGROUND_ID_RE = re.compile(r"^playground_[a-f0-9]{8,16}$")


def _is_valid_debate_id(debate_id: str) -> bool:
    """Validate debate ID format to prevent path traversal."""
    return bool(_DEBATE_ID_RE.match(debate_id) or _PLAYGROUND_ID_RE.match(debate_id))


def _get_debate_result(debate_id: str) -> dict[str, Any] | None:
    """Retrieve a debate from the debate store.

    Returns the full result dict, or None if not found/expired.
    """
    try:
        from aragora.storage.debate_store import get_debate_store

        store = get_debate_store()
        return store.get(debate_id)
    except (ImportError, RuntimeError, OSError) as exc:
        logger.debug("Debate store unavailable: %s", exc)
        return None


def _get_storage_debate_result(storage: Any, debate_id: str) -> dict[str, Any] | None:
    """Retrieve a debate from primary storage when it has been made public."""
    try:
        is_public = False
        is_public_fn = getattr(storage, "is_public", None)
        if callable(is_public_fn):
            is_public = bool(is_public_fn(debate_id))

        if not is_public:
            from aragora.server.handlers.debates.share import is_publicly_shared

            is_public = is_publicly_shared(debate_id)

        if not is_public:
            return None

        for getter_name in ("get_debate", "get", "get_by_id"):
            getter = getattr(storage, getter_name, None)
            if not callable(getter):
                continue
            result = getter(debate_id)
            if not isinstance(result, dict):
                continue

            hydrated = dict(result)
            hydrated.setdefault("id", hydrated.get("debate_id", debate_id))
            hydrated.setdefault("debate_id", hydrated.get("id", debate_id))
            hydrated.setdefault("share_url", f"/debate/{debate_id}")
            hydrated["is_public"] = True
            return hydrated
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        logger.debug("Primary debate storage unavailable for %s: %s", debate_id, exc)

    return None


def _normalize_messages(messages: Any) -> list[dict[str, Any]]:
    """Normalize debate messages for the public read-only viewer."""
    if not isinstance(messages, list):
        return []

    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue

        content = message.get("content")
        if content is None:
            continue

        normalized_message = {
            "agent": str(
                message.get("agent")
                or message.get("author")
                or message.get("name")
                or message.get("role")
                or "unknown"
            ),
            "role": str(message.get("role") or message.get("position") or "message"),
            "content": str(content),
            "round": message.get("round", 0),
        }

        if message.get("timestamp") is not None:
            normalized_message["timestamp"] = message.get("timestamp")

        normalized.append(normalized_message)

    return normalized


def _normalize_critiques(critiques: Any, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize critique records across playground and debate-storage formats."""
    normalized: list[dict[str, Any]] = []

    if isinstance(critiques, list):
        for critique in critiques:
            if not isinstance(critique, dict):
                continue

            issues = critique.get("issues")
            if not isinstance(issues, list):
                legacy_reason = (
                    critique.get("reasoning") or critique.get("content") or critique.get("text")
                )
                issues = [str(legacy_reason)] if legacy_reason else []

            suggestions = critique.get("suggestions")
            if not isinstance(suggestions, list):
                suggestions = []

            severity = critique.get("severity", 0.0)
            if not isinstance(severity, int | float):
                severity = 0.0

            normalized.append(
                {
                    "agent": str(critique.get("agent") or critique.get("author") or "unknown"),
                    "target_agent": str(
                        critique.get("target_agent") or critique.get("target") or ""
                    ),
                    "issues": [str(issue) for issue in issues if issue is not None],
                    "suggestions": [
                        str(suggestion) for suggestion in suggestions if suggestion is not None
                    ],
                    "severity": float(severity),
                }
            )

    if normalized:
        return normalized

    for message in messages:
        role = str(message.get("role") or "").lower()
        if "critic" not in role:
            continue

        normalized.append(
            {
                "agent": str(message.get("agent") or "unknown"),
                "target_agent": str(message.get("target_agent") or message.get("target") or ""),
                "issues": [str(message.get("content", ""))],
                "suggestions": [],
                "severity": 0.0,
            }
        )

    return normalized


def _normalize_votes(votes: Any) -> list[dict[str, Any]]:
    """Normalize vote records for the standalone public viewer."""
    if not isinstance(votes, list):
        return []

    normalized: list[dict[str, Any]] = []
    for vote in votes:
        if not isinstance(vote, dict):
            continue

        confidence = vote.get("confidence", 0.0)
        if not isinstance(confidence, int | float):
            confidence = 0.0

        normalized_vote = {
            "agent": str(vote.get("agent") or "unknown"),
            "choice": str(vote.get("choice") or ""),
            "confidence": float(confidence),
        }

        reasoning = vote.get("reasoning")
        if reasoning is not None:
            normalized_vote["reasoning"] = str(reasoning)

        normalized.append(normalized_vote)

    return normalized


def _derive_participants(result: dict[str, Any], messages: list[dict[str, Any]]) -> list[str]:
    """Return debate participants from the most explicit available source."""
    participants = result.get("participants")
    if isinstance(participants, list) and all(isinstance(agent, str) for agent in participants):
        return list(participants)

    agents = result.get("agents")
    if isinstance(agents, list) and all(isinstance(agent, str) for agent in agents):
        return list(agents)

    seen: set[str] = set()
    derived: list[str] = []
    for message in messages:
        agent = message.get("agent")
        if isinstance(agent, str) and agent and agent not in seen:
            seen.add(agent)
            derived.append(agent)
    return derived


def _derive_proposals(result: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, str]:
    """Derive per-agent positions when explicit proposals are unavailable."""
    proposals = result.get("proposals")
    if isinstance(proposals, dict):
        return {
            str(agent): str(text)
            for agent, text in proposals.items()
            if isinstance(agent, str) and text is not None
        }

    derived: dict[str, str] = {}
    for message in messages:
        agent = message.get("agent")
        role = str(message.get("role") or "").lower()
        content = message.get("content")
        if not isinstance(agent, str) or not isinstance(content, str):
            continue
        if "propos" in role or "argument" in role or "synth" in role:
            derived[agent] = content
    return derived


def _coerce_confidence(result: dict[str, Any]) -> float:
    """Extract the best available confidence value from a debate payload."""
    for candidate in (
        result.get("confidence"),
        result.get("agreement"),
        (result.get("consensus") or {}).get("confidence")
        if isinstance(result.get("consensus"), dict)
        else None,
        (result.get("consensus_proof") or {}).get("confidence")
        if isinstance(result.get("consensus_proof"), dict)
        else None,
    ):
        if isinstance(candidate, int | float):
            return float(candidate)
    return 0.0


def _coerce_final_answer(result: dict[str, Any]) -> str:
    """Extract the best available final answer text from a debate payload."""
    for candidate in (
        result.get("final_answer"),
        result.get("conclusion"),
        result.get("winning_proposal"),
        (result.get("consensus") or {}).get("final_answer")
        if isinstance(result.get("consensus"), dict)
        else None,
        (result.get("consensus_proof") or {}).get("final_answer")
        if isinstance(result.get("consensus_proof"), dict)
        else None,
    ):
        if isinstance(candidate, str) and candidate:
            return candidate
    return ""


def _normalize_public_view_payload(result: dict[str, Any], debate_id: str) -> dict[str, Any]:
    """Normalize debate payloads to the standalone public viewer contract."""
    messages = _normalize_messages(result.get("messages"))
    participants = _derive_participants(result, messages)
    final_answer = _coerce_final_answer(result)
    receipt = result.get("receipt") if isinstance(result.get("receipt"), dict) else {}

    normalized = dict(result)
    normalized["id"] = str(result.get("id") or result.get("debate_id") or debate_id)
    normalized["topic"] = str(
        result.get("topic") or result.get("task") or result.get("question") or f"Debate {debate_id}"
    )
    normalized["status"] = str(result.get("status") or "completed")
    normalized["consensus_reached"] = bool(
        result.get("consensus_reached")
        or (
            isinstance(result.get("consensus"), dict) and result["consensus"].get("reached")  # type: ignore[index]
        )
        or (
            isinstance(result.get("consensus_proof"), dict)
            and result["consensus_proof"].get("reached")  # type: ignore[index]
        )
    )
    normalized["confidence"] = _coerce_confidence(result)
    normalized["verdict"] = str(result.get("verdict") or final_answer)
    duration = result.get("duration_seconds", 0.0)
    normalized["duration_seconds"] = float(duration) if isinstance(duration, int | float) else 0.0
    normalized["participants"] = participants
    normalized["proposals"] = _derive_proposals(result, messages)
    normalized["critiques"] = _normalize_critiques(result.get("critiques"), messages)
    normalized["votes"] = _normalize_votes(result.get("votes"))
    normalized["final_answer"] = final_answer
    normalized["receipt_hash"] = result.get("receipt_hash") or receipt.get("signature")
    normalized["messages"] = messages
    normalized.setdefault("share_url", f"/debate/{debate_id}")
    if result.get("is_public"):
        normalized["is_public"] = True

    return normalized


def _is_shareable(result: dict[str, Any]) -> bool:
    """Check whether a debate result is allowed to be viewed publicly.

    A debate is shareable if:
    - It has a share_url field (set by _persist_and_respond in playground.py)
    - OR it has visibility == "public" (set by _persist_playground_debate)
    - OR its source is "playground", "landing", "oracle", or "demo"
    """
    if result.get("share_url"):
        return True
    if result.get("is_public") is True:
        return True
    if result.get("visibility") == "public":
        return True
    source = result.get("source", "")
    if source in ("playground", "landing", "oracle", "demo"):
        return True
    return False


# ---------------------------------------------------------------------------
# OG metadata rendering
# ---------------------------------------------------------------------------

_DEFAULT_OG_IMAGE = "https://aragora.ai/og-card.png"


def _render_og_html(debate: dict[str, Any], debate_id: str) -> str:
    """Render an HTML page with Open Graph meta tags for social previews."""
    esc = html_mod.escape

    topic = debate.get("topic", "Untitled Debate")
    # Truncate to 60 chars for OG title
    if len(topic) > 60:
        og_title = topic[:57] + "..."
    else:
        og_title = topic
    og_title = f"Aragora Debate: {og_title}"

    verdict = debate.get("verdict") or debate.get("final_answer") or "Pending"
    confidence = debate.get("confidence", 0.0)
    participants = debate.get("participants", [])
    agent_count = len(participants)
    consensus = debate.get("consensus_reached", False)

    # Build description
    desc_parts = []
    if verdict and verdict != "Pending":
        verdict_preview = verdict[:120] if len(str(verdict)) > 120 else verdict
        desc_parts.append(f"Verdict: {verdict_preview}")
    desc_parts.append(f"Confidence: {confidence:.0%}")
    desc_parts.append(f"{agent_count} AI agents")
    if consensus:
        desc_parts.append("Consensus reached")
    og_description = " | ".join(desc_parts)

    og_image = _DEFAULT_OG_IMAGE
    canonical_url = f"https://aragora.ai/debate/{debate_id}/"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{esc(og_title)}</title>

    <!-- Open Graph -->
    <meta property="og:type" content="article">
    <meta property="og:title" content="{esc(og_title)}">
    <meta property="og:description" content="{esc(og_description)}">
    <meta property="og:image" content="{esc(og_image)}">
    <meta property="og:url" content="{esc(canonical_url)}">
    <meta property="og:site_name" content="Aragora">

    <!-- Twitter Card -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="{esc(og_title)}">
    <meta name="twitter:description" content="{esc(og_description)}">
    <meta name="twitter:image" content="{esc(og_image)}">

    <meta name="description" content="{esc(og_description)}">
    <link rel="canonical" href="{esc(canonical_url)}">

    <!-- Redirect to the live viewer after a brief delay for crawlers -->
    <meta http-equiv="refresh" content="0;url={esc(canonical_url)}">
</head>
<body>
    <h1>{esc(og_title)}</h1>
    <p>{esc(og_description)}</p>
    <p><a href="{esc(canonical_url)}">View this debate on Aragora</a></p>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class PublicDebateViewerHandler(BaseHandler):
    """Handler for public debate viewing and OG metadata.

    No authentication required. Rate limited to 30 req/min per IP.
    """

    ROUTES = [
        "/api/v1/debates/public/*",
        "/api/v1/debates/public/*/og",
    ]

    def __init__(self, ctx: dict | None = None):
        self.ctx = ctx or {}

    def can_handle(self, path: str) -> bool:
        """Match /api/v1/debates/public/{id} and /api/v1/debates/public/{id}/og."""
        if not path.startswith("/api/v1/debates/public/"):
            return False
        parts = path.rstrip("/").split("/")
        # /api/v1/debates/public/{id} -> 6 parts
        # /api/v1/debates/public/{id}/og -> 7 parts
        if len(parts) == 6:
            return True
        if len(parts) == 7 and parts[6] == "og":
            return True
        return False

    def _extract_client_ip(self, handler: Any) -> str:
        """Extract client IP from the handler."""
        if handler and hasattr(handler, "client_address"):
            addr = handler.client_address
            if isinstance(addr, (list, tuple)) and len(addr) >= 1:
                return str(addr[0])
        return "unknown"

    def _extract_debate_id(self, path: str) -> str | None:
        """Extract debate ID from path.

        /api/v1/debates/public/{id} -> parts[5]
        /api/v1/debates/public/{id}/og -> parts[5]
        """
        parts = path.rstrip("/").split("/")
        if len(parts) >= 6:
            return parts[5]
        return None

    def _load_debate_result(self, debate_id: str) -> dict[str, Any] | None:
        """Load a debate from public stores and normalize it for the standalone viewer."""
        result = _get_debate_result(debate_id)
        if result is None:
            storage = self.get_storage()
            if storage is not None:
                result = _get_storage_debate_result(storage, debate_id)

        if result is None:
            return None

        return _normalize_public_view_payload(result, debate_id)

    # ------------------------------------------------------------------
    # GET /api/v1/debates/public/{id}
    # GET /api/v1/debates/public/{id}/og
    # ------------------------------------------------------------------

    @handle_errors("public debate viewer")
    def handle(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult | None:
        # Rate limit
        client_ip = self._extract_client_ip(handler)
        allowed, retry_after = _check_public_viewer_rate_limit(client_ip)
        if not allowed:
            return json_response(
                {
                    "error": "Rate limit exceeded. Please try again later.",
                    "retry_after": retry_after,
                },
                status=429,
            )

        debate_id = self._extract_debate_id(path)
        if not debate_id:
            return error_response("Missing debate ID", 400)

        if not _is_valid_debate_id(debate_id):
            return error_response("Invalid debate ID format", 400)

        parts = path.rstrip("/").split("/")

        # OG endpoint: /api/v1/debates/public/{id}/og
        if len(parts) == 7 and parts[6] == "og":
            return self._handle_og(debate_id)

        # JSON endpoint: /api/v1/debates/public/{id}
        return self._handle_public_debate(debate_id)

    def _handle_public_debate(self, debate_id: str) -> HandlerResult:
        """Return the debate result JSON for a publicly shared debate."""
        result = self._load_debate_result(debate_id)
        if result is None:
            return error_response("Debate not found", 404)

        if not _is_shareable(result):
            return error_response("Debate not found", 404)

        return json_response(result)

    def _handle_og(self, debate_id: str) -> HandlerResult:
        """Return HTML with Open Graph meta tags for social previews."""
        result = self._load_debate_result(debate_id)
        if result is None:
            return error_response("Debate not found", 404)

        if not _is_shareable(result):
            return error_response("Debate not found", 404)

        html_content = _render_og_html(result, debate_id)

        return HandlerResult(
            body=html_content.encode("utf-8"),
            status_code=200,
            content_type="text/html; charset=utf-8",
            headers={"Cache-Control": "public, max-age=300"},
        )


__all__ = [
    "PublicDebateViewerHandler",
    "_reset_public_viewer_rate_limits",
]
