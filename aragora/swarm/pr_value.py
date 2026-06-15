"""PR and issue value-composition classification for automation admission.

The classifier is deliberately heuristic: it reads titles and labels to decide
whether a PR/issue looks like loop self-maintenance, product work, infra, or
unknown. Use it as a composition signal, not a per-item authority.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DEFAULT_STALE_DAYS = 4
DEFAULT_MAX_MAINTENANCE_RATIO = 0.5
SAMPLE_PER_CLASS = 5

MAINTENANCE_LABEL = "codex-automation"
FEATURE_LABEL = "feature"

CLASS_MAINTENANCE = "maintenance"
CLASS_PRODUCT = "product"
CLASS_INFRA = "infra"
CLASS_UNKNOWN = "unknown"
CLASS_ORDER = (CLASS_MAINTENANCE, CLASS_PRODUCT, CLASS_INFRA, CLASS_UNKNOWN)

MAINTENANCE_TITLE_PATTERNS = (
    r"outbox-harvest",
    r"refresh generated",
    r"regenerate",
    r"resync",
    r"module_tiers",
    r"metrics drift",
    r"stale[- ]quorum",
    r"salvage",
    r"\brepair\b",
    r"drift",
    r"preflight",
    r"reconcile",
    r"janitor",
    r"backpressure",
    r"sentinel",
    r"lane",
    r"gate",
    r"quorum",
)

PRODUCT_TITLE_PATTERNS = (
    r"ODR",
    r"crux",
    r"calibration",
    r"receipt",
    r"vertical",
    r"\bAPI\b",
    r"\bSDK\b",
    r"handler",
    r"endpoint",
    r"debate",
    r"consensus",
)

INFRA_TITLE_PATTERNS = (
    r"test",
    r"mypy",
    r"lint",
    r"ruff",
    r"ci",
    r"packaging",
    r"docs",
    r"import",
)


def _compile(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


_MAINTENANCE_RE = _compile(MAINTENANCE_TITLE_PATTERNS)
_PRODUCT_RE = _compile(PRODUCT_TITLE_PATTERNS)
_INFRA_RE = _compile(INFRA_TITLE_PATTERNS)


def parse_iso_datetime(value: Any) -> datetime | None:
    """Parse GitHub ISO timestamps; return ``None`` for absent/bad values."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def label_names(record: dict[str, Any]) -> set[str]:
    """Return lower-cased label names from GitHub-style dicts or strings."""
    labels = record.get("labels") or []
    names: set[str] = set()
    if not isinstance(labels, list):
        return names
    for label in labels:
        if isinstance(label, dict):
            name = label.get("name")
        else:
            name = label
        if name:
            names.add(str(name).lower())
    return names


def _any_match(title: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(title) for pattern in patterns)


def classify_value_record(record: dict[str, Any]) -> str:
    """Classify one PR/issue-like record into one value-composition class.

    Precedence is fixed and first-match-wins:
    maintenance label/title, product label/title, infra title, unknown.
    """
    labels = label_names(record)
    title = str(record.get("title") or "")

    if MAINTENANCE_LABEL in labels or _any_match(title, _MAINTENANCE_RE):
        return CLASS_MAINTENANCE
    if FEATURE_LABEL in labels or _any_match(title, _PRODUCT_RE):
        return CLASS_PRODUCT
    if _any_match(title, _INFRA_RE):
        return CLASS_INFRA
    return CLASS_UNKNOWN


def ratio(part: int, total: int) -> float:
    return 0.0 if total <= 0 else round(part / total, 4)


def build_value_report(
    records: list[dict[str, Any]],
    *,
    stale_days: int = DEFAULT_STALE_DAYS,
    max_maintenance_ratio: float = DEFAULT_MAX_MAINTENANCE_RATIO,
    now: datetime,
    annotations: list[str] | None = None,
) -> dict[str, Any]:
    """Build a pure value-composition report from PR/issue-like records."""
    by_class = dict.fromkeys(CLASS_ORDER, 0)
    sample: dict[str, list[dict[str, Any]]] = {name: [] for name in CLASS_ORDER}
    drafts = 0
    stale_count = 0
    stale_cutoff = timedelta(days=max(0, stale_days))

    for record in records:
        if not isinstance(record, dict):
            continue
        value_class = classify_value_record(record)
        by_class[value_class] += 1
        if len(sample[value_class]) < SAMPLE_PER_CLASS:
            sample[value_class].append(
                {"number": record.get("number"), "title": str(record.get("title") or "")}
            )
        if bool(record.get("isDraft")):
            drafts += 1
        created = parse_iso_datetime(record.get("createdAt") or record.get("created_at"))
        if created is not None and now - created > stale_cutoff:
            stale_count += 1

    total = sum(by_class.values())
    return {
        "total": total,
        "by_class": by_class,
        "maintenance_ratio": ratio(by_class[CLASS_MAINTENANCE], total),
        "product_ratio": ratio(by_class[CLASS_PRODUCT], total),
        "drafts": drafts,
        "stale_count": stale_count,
        "threshold": {
            "max_maintenance_ratio": max_maintenance_ratio,
            "stale_days": stale_days,
        },
        "annotations": list(annotations or []),
        "sample": sample,
    }


def summary_line(report: dict[str, Any]) -> str:
    by_class = report["by_class"]
    return (
        f"PR value: total={report['total']} "
        f"maint={by_class[CLASS_MAINTENANCE]}({report['maintenance_ratio']:.0%}) "
        f"product={by_class[CLASS_PRODUCT]}({report['product_ratio']:.0%}) "
        f"infra={by_class[CLASS_INFRA]} unknown={by_class[CLASS_UNKNOWN]} "
        f"drafts={report['drafts']} stale={report['stale_count']}"
    )


def read_backpressure_withheld_classes(path: str | Path | None) -> set[str]:
    """Read explicit backpressure admission withholding classes.

    Legacy ``mode: shepherd`` signals without an ``admission`` block stay
    advisory-only. Missing, malformed, or non-object files fail open because
    the binding policy must be explicit.
    """
    if not path:
        return set()
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return set()
    if not isinstance(payload, dict):
        return set()
    admission = payload.get("admission")
    if not isinstance(admission, dict):
        return set()
    raw = admission.get("withhold_classes")
    if not isinstance(raw, list):
        return set()
    return {
        str(item).strip().lower() for item in raw if str(item).strip().lower() in set(CLASS_ORDER)
    }
