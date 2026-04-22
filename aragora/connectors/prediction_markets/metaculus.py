"""Metaculus read-only adapter (AGT-03 Phase 1 companion venue).

Injectable-HTTP pattern — callers supply ``http_client(method, url, headers)
→ (status_code, body_text)``; no implicit network calls.

Resolution: 1.0→YES, 0.0→NO, -1/other/None→inconclusive.
Out of scope: prediction submission, per-agent Brier (AGT-05).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)

METACULUS_API_BASE = "https://www.metaculus.com/api2"
DEFAULT_MIN_WINDOW_DAYS = 30
HttpClient = Callable[..., tuple[int, str]]


class MetaculusError(RuntimeError):
    pass


@dataclass(frozen=True)
class MetaculusQuestion:
    question_id: int
    title: str
    question_type: str
    created_time: str | None
    close_time: str | None
    resolve_time: str | None
    active_state: str
    resolution: float | None
    community_q2: float | None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_resolved(self) -> bool:
        return self.active_state == "resolved"

    @classmethod
    def from_api_payload(cls, payload: dict[str, Any]) -> "MetaculusQuestion":
        qid = payload.get("id")
        if qid is None:
            raise MetaculusError("Metaculus question payload missing 'id'")
        try:
            qid = int(qid)
        except (TypeError, ValueError) as exc:
            raise MetaculusError(f"Metaculus question 'id' is not an integer: {qid!r}") from exc
        community_q2: float | None = None
        full = (
            (payload.get("community_prediction") or {}).get("full")
            if isinstance(payload.get("community_prediction"), dict)
            else None
        )
        if isinstance(full, dict) and full.get("q2") is not None:
            try:
                community_q2 = float(full["q2"])
            except (TypeError, ValueError):
                pass
        resolution: float | None = None
        if payload.get("resolution") is not None:
            try:
                resolution = float(payload["resolution"])
            except (TypeError, ValueError):
                pass
        return cls(
            question_id=qid,
            title=str(payload.get("title") or ""),
            question_type=str(payload.get("question_type") or "binary"),
            created_time=payload.get("created_time"),
            close_time=payload.get("close_time"),
            resolve_time=payload.get("resolve_time"),
            active_state=str(payload.get("active_state") or "active"),
            resolution=resolution,
            community_q2=community_q2,
            raw=dict(payload),
        )


@dataclass(frozen=True)
class MetaculusResolution:
    question_id: int
    outcome: str  # "yes" | "no" | "inconclusive"
    resolved_at: str | None
    community_q2: float | None
    raw: dict[str, Any] = field(default_factory=dict)


def _normalize_outcome(resolution: float | None) -> str:
    if resolution == 1.0:
        return "yes"
    if resolution == 0.0:
        return "no"
    return "inconclusive"


@dataclass
class MetaculusAdapter:
    """Read-only Metaculus API v2 adapter."""

    http_client: HttpClient
    api_base: str = METACULUS_API_BASE
    min_window_days: int = DEFAULT_MIN_WINDOW_DAYS

    def _get(self, path: str) -> Any:
        url = f"{self.api_base.rstrip('/')}/{path.lstrip('/')}"
        try:
            status, body = self.http_client("GET", url, {"Accept": "application/json"})
        except Exception as exc:  # noqa: BLE001
            raise MetaculusError(f"metaculus transport error for {path}: {exc}") from exc
        if status >= 400:
            raise MetaculusError(f"metaculus {path} returned HTTP {status}: {body[:200]}")
        try:
            return json.loads(body or "null")
        except json.JSONDecodeError as exc:
            raise MetaculusError(f"metaculus {path} returned non-JSON: {exc}") from exc

    def fetch_question(self, question_id: int | str) -> MetaculusQuestion:
        qid = int(question_id)
        payload = self._get(f"questions/{qid}/")
        if not isinstance(payload, dict):
            raise MetaculusError(f"metaculus questions/{qid}/ returned non-object payload")
        return MetaculusQuestion.from_api_payload(payload)

    def list_questions(
        self,
        *,
        limit: int = 50,
        status: str | None = None,
        question_type: str = "binary",
        offset: int = 0,
    ) -> list[MetaculusQuestion]:
        if limit < 1 or limit > 100:
            raise MetaculusError("limit must be in [1, 100]")
        path = f"questions/?limit={int(limit)}&offset={int(offset)}"
        if status:
            path += f"&status={status}"
        if question_type:
            path += f"&type={question_type}"
        payload = self._get(path)
        if isinstance(payload, dict) and "results" in payload:
            entries = payload["results"]
        elif isinstance(payload, list):
            entries = payload
        else:
            raise MetaculusError("metaculus questions endpoint returned unexpected shape")
        out: list[MetaculusQuestion] = []
        for entry in entries:
            if isinstance(entry, dict):
                try:
                    out.append(MetaculusQuestion.from_api_payload(entry))
                except MetaculusError:
                    logger.debug("skipping malformed metaculus question entry")
        return out

    def discover_open_binary_questions(
        self, *, limit: int = 50, now: datetime | None = None
    ) -> list[MetaculusQuestion]:
        """Return open binary questions with at least min_window_days until close."""
        questions = self.list_questions(limit=limit, status="active", question_type="binary")
        reference = now or datetime.now(tz=UTC)
        threshold_s = self.min_window_days * 24 * 3600
        out: list[MetaculusQuestion] = []
        for q in questions:
            if q.close_time is None:
                continue
            try:
                close_dt = datetime.fromisoformat(q.close_time.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            if (close_dt - reference).total_seconds() >= threshold_s:
                out.append(q)
        return out

    def fetch_resolution(self, question_id: int | str) -> MetaculusResolution | None:
        question = self.fetch_question(question_id)
        if not question.is_resolved:
            return None
        return MetaculusResolution(
            question_id=question.question_id,
            outcome=_normalize_outcome(question.resolution),
            resolved_at=question.resolve_time,
            community_q2=question.community_q2,
            raw={"resolution": question.resolution, "question_type": question.question_type},
        )


def metaculus_to_market_resolution(
    resolution: MetaculusResolution, *, resolved_at: datetime | None = None
) -> Any:
    """Bridge to the AGT-04 ResolutionEvent shape (lazy import)."""
    from aragora.markets.types import ResolutionEvent as MR

    when = resolved_at
    if when is None and resolution.resolved_at is not None:
        try:
            when = datetime.fromisoformat(resolution.resolved_at.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            pass
    market_id = f"metaculus:{resolution.question_id}"
    evidence: dict[str, Any] = dict(resolution.raw)
    if resolution.community_q2 is not None:
        evidence["community_q2"] = resolution.community_q2
    factory = {"yes": MR.yes, "no": MR.no}.get(resolution.outcome, MR.inconclusive)
    return factory(
        market_id=market_id, resolution_source="metaculus", evidence=evidence, resolved_at=when
    )


__all__ = [
    "DEFAULT_MIN_WINDOW_DAYS",
    "METACULUS_API_BASE",
    "MetaculusAdapter",
    "MetaculusError",
    "MetaculusQuestion",
    "MetaculusResolution",
    "metaculus_to_market_resolution",
]
