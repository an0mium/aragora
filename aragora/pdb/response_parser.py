"""Robust JSON parser for PDB Mode 3 provider responses.

Every provider slot prompt in :mod:`aragora.pdb.prompts` asks the model
to return a compact JSON object. Real-world models return that JSON
wrapped in code fences, prefixed with prose, truncated mid-string, or
with stray ``//`` comments. This module exposes pure-function parsers
that coerce any of those into the canonical shapes the executor's
response dataclasses need.

Contract:

- ``extract_json_object(text)`` strips code fences, strips leading and
  trailing prose, heuristically removes ``//`` / ``#`` comments, and
  returns the first parseable JSON object. Returns ``None`` on
  irrecoverable failure.
- ``parse_findings_response(text, *, slot_id)`` coerces to the
  findings-phase payload shape the executor expects, filling in safe
  defaults (``confidence=0.0``, empty ``top_findings``) when fields are
  missing. Never raises.
- ``parse_critique_response(text, *, slot_id)`` coerces to the critique
  payload shape. Never raises.
- ``parse_synthesis_response(text)`` coerces to the synthesis payload
  shape. Never raises.

Every parser records a descriptive ``reason`` / ``summary`` (or
``top_line``) when the response was malformed so the caller can
surface the failure without invoking the model again.

These parsers are deliberately permissive. The
:class:`aragora.pdb.protocol.ProviderInvoker` contract is that failures
propagate as exceptions from the provider call itself — a parsed-but-
empty response should still produce a valid dataclass so the executor
can either treat the slot as present-with-no-findings or log-and-move-
on per its own rules.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from aragora.review.protocol import DissentPosition

__all__ = [
    "PARSE_FAILURE_REASON",
    "extract_json_object",
    "parse_critique_response",
    "parse_findings_response",
    "parse_synthesis_response",
    "position_from_string",
]


logger = logging.getLogger(__name__)

PARSE_FAILURE_REASON = "response_parse_failed"


# Regex that removes ``// ...`` and ``# ...`` single-line comments but is
# conservative enough not to eat them inside strings. We only strip
# comments that appear at the start of a (post-whitespace) line; this
# is sufficient for the "model added stray commentary" case without
# risking corrupting URLs like ``https://``.
_COMMENT_LINE_RE = re.compile(r"^\s*(?://|#)[^\n]*$", re.MULTILINE)

# Regex that catches ```json ... ``` and ``` ... ``` code fences.
_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*\n?(?P<body>.*?)\n?```",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the first parseable top-level JSON object in ``text``.

    Tries, in order:

    1. Strip ``// ...`` / ``# ...`` comment lines.
    2. Look for a fenced code block; parse its body first.
    3. Parse the whole string as JSON.
    4. Find the first ``{`` and the last ``}`` and parse that slice.

    Returns ``None`` if every attempt fails. Never raises.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    cleaned = _COMMENT_LINE_RE.sub("", text).strip()

    # 1. Fenced block.
    for match in _FENCE_RE.finditer(cleaned):
        body = match.group("body").strip()
        parsed = _try_loads(body)
        if isinstance(parsed, dict):
            return parsed

    # 2. Whole string.
    parsed = _try_loads(cleaned)
    if isinstance(parsed, dict):
        return parsed

    # 3. Slice from first { to last }. Handles prose prefix + truncated
    # suffix as long as a complete closing brace is present.
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        slice_ = cleaned[first_brace : last_brace + 1]
        parsed = _try_loads(slice_)
        if isinstance(parsed, dict):
            return parsed

    return None


def _try_loads(body: str) -> Any:
    try:
        return json.loads(body)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


_POSITION_MAP = {
    "approve": DissentPosition.APPROVE,
    "approve_candidate": DissentPosition.APPROVE,
    "approved": DissentPosition.APPROVE,
    "request_changes": DissentPosition.REQUEST_CHANGES,
    "request-changes": DissentPosition.REQUEST_CHANGES,
    "requestchanges": DissentPosition.REQUEST_CHANGES,
    "repair_first": DissentPosition.REQUEST_CHANGES,
    "changes_requested": DissentPosition.REQUEST_CHANGES,
    "block": DissentPosition.REQUEST_CHANGES,
    "defer": DissentPosition.DEFER,
    "needs_human_attention": DissentPosition.DEFER,
    "escalate": DissentPosition.DEFER,
    "abstain": DissentPosition.DEFER,
}


def position_from_string(
    value: Any, *, default: DissentPosition = DissentPosition.DEFER
) -> DissentPosition:
    """Coerce a model-returned string into a :class:`DissentPosition`.

    Unknown values fall back to ``default`` so malformed responses do
    not crash the executor.
    """
    if not isinstance(value, str):
        return default
    key = value.strip().lower().replace(" ", "_")
    return _POSITION_MAP.get(key, default)


def _coerce_float(value: Any, *, default: float = 0.0, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp ``value`` to ``[lo, hi]``; fall back to ``default`` if it is not numeric."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN check — NaN != NaN is the canonical test.
        return default
    return max(lo, min(hi, number))


def _coerce_str(value: Any, *, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip() or default
    return str(value).strip() or default


def _coerce_str_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif item is not None:
                as_str = str(item).strip()
                if as_str:
                    out.append(as_str)
        return tuple(out)
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    return ()


_VALID_SEVERITIES = ("low", "medium", "high")


def _coerce_severity(value: Any) -> str:
    as_str = _coerce_str(value, default="medium").lower()
    return as_str if as_str in _VALID_SEVERITIES else "medium"


# ---------------------------------------------------------------------------
# Findings-phase parser
# ---------------------------------------------------------------------------


def parse_findings_response(text: str, *, slot_id: str) -> dict[str, Any]:
    """Parse a findings-phase model response.

    Returns a dict with these keys (always present):

    - ``position`` (:class:`DissentPosition`)
    - ``confidence`` (float in [0.0, 1.0])
    - ``summary`` (str)
    - ``top_findings`` (list of dicts: finding_id/category/severity/summary/evidence)
    - ``contested_finding_ids`` (tuple[str, ...])
    - ``reason`` (str)
    - ``parsed`` (bool) — ``True`` if the raw text was parseable JSON

    On parse failure every field gets a safe default and ``parsed`` is
    ``False`` so the caller can label the slot's confidence accordingly.
    """
    parsed = extract_json_object(text)
    if parsed is None:
        logger.warning(
            "pdb.response_parser: findings JSON unparseable for slot %s (len=%d)",
            slot_id,
            len(text or ""),
        )
        return {
            "position": DissentPosition.DEFER,
            "confidence": 0.0,
            "summary": f"{slot_id}: {PARSE_FAILURE_REASON}",
            "top_findings": [],
            "contested_finding_ids": (),
            "reason": PARSE_FAILURE_REASON,
            "parsed": False,
            "raw_text_preview": (text or "")[:500],
        }

    position = position_from_string(parsed.get("recommendation"))
    confidence = _coerce_float(parsed.get("confidence"), default=0.0)
    reason = _coerce_str(parsed.get("reason"), default="")
    summary = _coerce_str(parsed.get("summary"), default=reason)
    if not summary:
        summary = f"{slot_id}: no summary provided"

    top_findings_raw = parsed.get("top_findings") or []
    top_findings: list[dict[str, Any]] = []
    if isinstance(top_findings_raw, list):
        for idx, item in enumerate(top_findings_raw[:5]):
            if not isinstance(item, dict):
                continue
            finding_id = _coerce_str(item.get("finding_id"), default=f"{slot_id}-F{idx + 1}")
            category = _coerce_str(item.get("category"), default="general")
            severity = _coerce_severity(item.get("severity"))
            f_summary = _coerce_str(item.get("summary"), default="(no summary)")
            evidence = _coerce_str_tuple(item.get("evidence"))
            top_findings.append(
                {
                    "finding_id": finding_id,
                    "category": category,
                    "severity": severity,
                    "summary": f_summary,
                    "evidence": list(evidence),
                }
            )

    contested = _coerce_str_tuple(parsed.get("contested_finding_ids"))

    return {
        "position": position,
        "confidence": confidence,
        "summary": summary,
        "top_findings": top_findings,
        "contested_finding_ids": contested,
        "reason": reason or summary,
        "parsed": True,
    }


# ---------------------------------------------------------------------------
# Critique-phase parser
# ---------------------------------------------------------------------------


def parse_critique_response(text: str, *, slot_id: str) -> dict[str, Any]:
    """Parse a critique-phase model response.

    Returns a dict with these keys (always present):

    - ``position`` (:class:`DissentPosition`)
    - ``confidence`` (float)
    - ``reason`` (str)
    - ``agrees_with`` (tuple[str, ...])
    - ``disagrees_with`` (tuple[str, ...])
    - ``contested_finding_ids`` (tuple[str, ...])
    - ``parsed`` (bool)
    """
    parsed = extract_json_object(text)
    if parsed is None:
        logger.warning(
            "pdb.response_parser: critique JSON unparseable for slot %s (len=%d)",
            slot_id,
            len(text or ""),
        )
        return {
            "position": DissentPosition.DEFER,
            "confidence": 0.0,
            "reason": PARSE_FAILURE_REASON,
            "agrees_with": (),
            "disagrees_with": (),
            "contested_finding_ids": (),
            "parsed": False,
            "raw_text_preview": (text or "")[:500],
        }

    return {
        "position": position_from_string(parsed.get("recommendation")),
        "confidence": _coerce_float(parsed.get("confidence"), default=0.0),
        "reason": _coerce_str(parsed.get("reason"), default=""),
        "agrees_with": _coerce_str_tuple(parsed.get("agrees_with")),
        "disagrees_with": _coerce_str_tuple(parsed.get("disagrees_with")),
        "contested_finding_ids": _coerce_str_tuple(parsed.get("contested_finding_ids")),
        "parsed": True,
    }


# ---------------------------------------------------------------------------
# Synthesis-phase parser
# ---------------------------------------------------------------------------


def parse_synthesis_response(text: str) -> dict[str, Any]:
    """Parse a synthesis-phase model response.

    Returns a dict with these keys (always present):

    - ``top_line`` (str)
    - ``validation_summary`` (str)
    - ``preserved_dissent`` (tuple of dicts with slot_id/lens/position/reason)
    - ``parsed`` (bool)
    """
    parsed = extract_json_object(text)
    if parsed is None:
        logger.warning(
            "pdb.response_parser: synthesis JSON unparseable (len=%d)",
            len(text or ""),
        )
        return {
            "top_line": f"Synthesis {PARSE_FAILURE_REASON}; see raw response for detail.",
            "validation_summary": "",
            "preserved_dissent": (),
            "parsed": False,
            "raw_text_preview": (text or "")[:500],
        }

    top_line = _coerce_str(
        parsed.get("top_line"),
        default="(synthesis returned no top_line)",
    )
    validation_summary = _coerce_str(parsed.get("validation_summary"), default="")

    preserved_raw = parsed.get("preserved_dissent") or []
    preserved: list[dict[str, str]] = []
    if isinstance(preserved_raw, list):
        for item in preserved_raw:
            if not isinstance(item, dict):
                continue
            preserved.append(
                {
                    "slot_id": _coerce_str(item.get("slot_id"), default="unknown"),
                    "lens": _coerce_str(item.get("lens"), default="unknown"),
                    "position": _coerce_str(item.get("position"), default="defer"),
                    "reason": _coerce_str(item.get("reason"), default=""),
                }
            )

    return {
        "top_line": top_line,
        "validation_summary": validation_summary,
        "preserved_dissent": tuple(preserved),
        "parsed": True,
    }
