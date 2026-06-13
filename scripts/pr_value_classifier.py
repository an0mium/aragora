#!/usr/bin/env python3
"""PR value-composition classifier for the autonomous PR pipeline (READ-ONLY).

The autonomous loop has a failure mode where it drifts toward self-maintenance:
it generates PRs that maintain its own machinery (outbox harvests, drift repairs,
quorum/lane/gate fixes) faster than it ships external product value. A hand-count
at build time found 196 open PRs, ~57% maintenance-labeled, only ~1.5% product
PRs -- a drift that stayed invisible because nothing measured it. This script
turns "is the loop building product value or just maintaining itself?" into a
number the boss loop / operator can steer on.

It takes one read-only snapshot per run:

1. One ``gh pr list --state open`` call (capped at ``--limit``, default 300),
   returning number/title/labels/isDraft/createdAt/updatedAt.
2. Each open PR is classified into exactly one of four classes
   (``maintenance``, ``product``, ``infra``, ``unknown``) by documented,
   case-insensitive heuristics checked in a fixed precedence (see below).

Emits JSON to stdout (and atomically to ``--out`` when given)::

    {generated_at, total, by_class: {maintenance, product, infra, unknown},
     maintenance_ratio, product_ratio, drafts, stale_count, threshold,
     annotations[], sample: {<class>: [up to 5 {number, title}]}}

``stale_count`` counts open PRs older than ``--stale-days`` (default 4).

Exit codes (sentinel-friendly): 0 normal, 3 when
``maintenance_ratio > --max-maintenance-ratio`` (default 0.5) so the boss loop
can branch on it to bias generation toward the product/feature backlog, 1 on
failure.

Classification is ADVISORY, not authoritative: it keys off PR titles and
labels, which can mislead (a refactor titled "fix API handler" reads as
product). It exists to surface composition drift, not to gate individual PRs.

Safety model (mirrors ``funnel_stage_metrics.py`` / ``boss_pr_janitor.py``):
read-only against GitHub, no mutations; the only file this script can write is
the optional ``--out`` snapshot (mkstemp + ``os.replace``). Stdlib-only by
design so it can run anywhere ``gh`` is authenticated.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

DEFAULT_REPO = "synaptent/aragora"
DEFAULT_LIMIT = 300
DEFAULT_STALE_DAYS = 4
DEFAULT_MAX_MAINTENANCE_RATIO = 0.5
GH_TIMEOUT_SECONDS = 120
SAMPLE_PER_CLASS = 5

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_BREACH = 3

_GH_FIELDS = "number,title,labels,isDraft,createdAt,updatedAt"

# Label that unconditionally marks a PR as the loop maintaining its own machinery.
MAINTENANCE_LABEL = "codex-automation"
# Label that marks external-facing product value.
FEATURE_LABEL = "feature"

# --- Classification heuristics (HEURISTIC + TUNABLE) ----------------------------------
# These keyword lists are deliberately heuristic and meant to be tuned over time
# as the loop's title conventions shift. They are matched case-insensitively
# against the PR title (as regex fragments) in the precedence documented in
# ``classify_pr``. Classification is advisory, not authoritative -- titles can
# mislead -- so treat the output as a composition signal, not a per-PR verdict.

# Maintenance: the loop maintaining its own machinery.
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

# Product: external-facing value.
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

# Infra: engineering health that is not pure self-maintenance. Kept distinct so
# it is not lumped with maintenance or product.
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

CLASS_MAINTENANCE = "maintenance"
CLASS_PRODUCT = "product"
CLASS_INFRA = "infra"
CLASS_UNKNOWN = "unknown"
CLASS_ORDER = (CLASS_MAINTENANCE, CLASS_PRODUCT, CLASS_INFRA, CLASS_UNKNOWN)


def _compile(patterns: tuple[str, ...]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_MAINTENANCE_RE = _compile(MAINTENANCE_TITLE_PATTERNS)
_PRODUCT_RE = _compile(PRODUCT_TITLE_PATTERNS)
_INFRA_RE = _compile(INFRA_TITLE_PATTERNS)


# --- Parsing helpers ------------------------------------------------------------------


def _parse_iso(ts: Any) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _label_names(pr: dict[str, Any]) -> set[str]:
    """Lower-cased label names from a ``gh`` PR record (labels is a list of dicts)."""
    raw = pr.get("labels") or []
    names: set[str] = set()
    if isinstance(raw, list):
        for label in raw:
            if isinstance(label, dict):
                name = label.get("name")
                if name:
                    names.add(str(name).lower())
            elif label:
                names.add(str(label).lower())
    return names


def _any_match(title: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(p.search(title) for p in patterns)


def classify_pr(pr: dict[str, Any]) -> str:
    """Classify one open PR into exactly one class by heuristic precedence.

    Precedence (first match wins):

    1. ``maintenance`` -- ``codex-automation`` label OR a maintenance title
       pattern. The label takes precedence over everything: a PR titled
       "feat(api): new endpoint" but labeled ``codex-automation`` classifies
       as ``maintenance``, because the label is the strongest signal that this
       PR came from the loop maintaining its own machinery.
    2. ``product`` -- a product title pattern OR the ``feature`` label.
    3. ``infra`` -- an infra title pattern.
    4. ``unknown`` -- anything unmatched.

    Advisory only: titles can mislead. The point is composition, not verdicts.
    """
    labels = _label_names(pr)
    title = str(pr.get("title") or "")

    if MAINTENANCE_LABEL in labels or _any_match(title, _MAINTENANCE_RE):
        return CLASS_MAINTENANCE
    if FEATURE_LABEL in labels or _any_match(title, _PRODUCT_RE):
        return CLASS_PRODUCT
    if _any_match(title, _INFRA_RE):
        return CLASS_INFRA
    return CLASS_UNKNOWN


# --- Inputs (read-only) ---------------------------------------------------------------


def default_list_prs(repo: str, limit: int) -> list[dict[str, Any]]:
    """One ``gh pr list`` call for open PRs (read-only, capped at ``limit``)."""
    command = [
        "gh",
        "pr",
        "list",
        "--repo",
        repo,
        "--state",
        "open",
        "--json",
        _GH_FIELDS,
        "--limit",
        str(limit),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=GH_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh pr list failed (exit {result.returncode}): {result.stderr.strip()}")
    payload = json.loads(result.stdout or "[]")
    if not isinstance(payload, list):
        raise RuntimeError("gh pr list returned unexpected payload (expected a list)")
    return payload


# --- Output ---------------------------------------------------------------------------


def atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    """Write temp + ``os.replace`` so readers never observe a partial file."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except OSError:  # never mask the primary error with a cleanup failure
            pass


# --- Snapshot -------------------------------------------------------------------------


def _ratio(part: int, total: int) -> float:
    return 0.0 if total <= 0 else round(part / total, 4)


def build_report(
    prs: list[dict[str, Any]],
    *,
    stale_days: int,
    max_maintenance_ratio: float,
    now: datetime,
    annotations: list[str],
) -> dict[str, Any]:
    """Pure report builder: classify PRs and compute composition. No I/O."""
    by_class = dict.fromkeys(CLASS_ORDER, 0)
    sample: dict[str, list[dict[str, Any]]] = {name: [] for name in CLASS_ORDER}
    drafts = 0
    stale_count = 0
    stale_cutoff = timedelta(days=max(0, stale_days))

    for pr in prs:
        if not isinstance(pr, dict):
            continue
        cls = classify_pr(pr)
        by_class[cls] += 1
        if len(sample[cls]) < SAMPLE_PER_CLASS:
            sample[cls].append({"number": pr.get("number"), "title": str(pr.get("title") or "")})
        if bool(pr.get("isDraft")):
            drafts += 1
        created = _parse_iso(pr.get("createdAt"))
        if created is not None and now - created > stale_cutoff:
            stale_count += 1

    total = sum(by_class.values())
    return {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total": total,
        "by_class": by_class,
        "maintenance_ratio": _ratio(by_class[CLASS_MAINTENANCE], total),
        "product_ratio": _ratio(by_class[CLASS_PRODUCT], total),
        "drafts": drafts,
        "stale_count": stale_count,
        "threshold": {
            "max_maintenance_ratio": max_maintenance_ratio,
            "stale_days": stale_days,
        },
        "annotations": annotations,
        "sample": sample,
    }


def _summary_line(report: dict[str, Any]) -> str:
    bc = report["by_class"]
    return (
        f"PR value: total={report['total']} "
        f"maint={bc[CLASS_MAINTENANCE]}({report['maintenance_ratio']:.0%}) "
        f"product={bc[CLASS_PRODUCT]}({report['product_ratio']:.0%}) "
        f"infra={bc[CLASS_INFRA]} unknown={bc[CLASS_UNKNOWN]} "
        f"drafts={report['drafts']} stale={report['stale_count']}"
    )


def run_classifier(
    *,
    list_prs: Callable[[], list[dict[str, Any]]],
    limit: int = DEFAULT_LIMIT,
    stale_days: int = DEFAULT_STALE_DAYS,
    max_maintenance_ratio: float = DEFAULT_MAX_MAINTENANCE_RATIO,
    out_file: str | None = None,
    summary: bool = False,
    now: datetime | None = None,
    log: Callable[[str], None] = print,
) -> int:
    """Build one value-composition report; return 0 / 3 (breach) / 1 (failure)."""
    if now is None:
        now = datetime.now(timezone.utc)
    annotations: list[str] = []

    try:
        raw = list_prs()
        if not isinstance(raw, list):
            raise RuntimeError("list_prs returned unexpected payload (expected a list)")
        prs = [pr for pr in raw if isinstance(pr, dict)]
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(json.dumps({"action": "error", "error": str(exc)[:500]}), file=sys.stderr)
        return EXIT_FAILURE

    # ``gh pr list`` truncates at the limit; a full payload means counts are a
    # floor, not the true total -- surface that so ratios are read with care.
    if len(prs) >= limit:
        annotations.append(f"list_truncated:>={limit}")

    report = build_report(
        prs,
        stale_days=stale_days,
        max_maintenance_ratio=max_maintenance_ratio,
        now=now,
        annotations=annotations,
    )

    if summary:
        log(_summary_line(report))
    else:
        log(json.dumps(report, sort_keys=True))

    if out_file:
        try:
            atomic_write_json(out_file, report)
        except OSError as exc:
            print(
                json.dumps({"action": "out_write_failed", "error": str(exc)[:300]}),
                file=sys.stderr,
            )
            return EXIT_FAILURE

    if report["maintenance_ratio"] > max_maintenance_ratio:
        return EXIT_BREACH
    return EXIT_OK


_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def repo_arg(value: str) -> str:
    """Argparse type: validate ``owner/name`` at parse time, before any gh call."""
    if not _REPO_RE.match(value):
        raise argparse.ArgumentTypeError(f"--repo must look like owner/name, got {value!r}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only PR value-composition classifier: tags open PRs "
            "product/maintenance/infra/unknown so the loop can see when it has "
            "drifted into self-maintenance. Exits 0 normally, 3 when "
            "maintenance_ratio exceeds the threshold, 1 on failure."
        )
    )
    parser.add_argument(
        "--repo", default=DEFAULT_REPO, type=repo_arg, help="GitHub repo (owner/name)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Per-run cap on open PRs fetched (default {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--stale-days",
        type=int,
        default=DEFAULT_STALE_DAYS,
        help=f"Open PRs older than this many days count as stale (default {DEFAULT_STALE_DAYS})",
    )
    parser.add_argument(
        "--max-maintenance-ratio",
        type=float,
        default=DEFAULT_MAX_MAINTENANCE_RATIO,
        help=f"Breach (exit 3) when maintenance_ratio exceeds this "
        f"(default {DEFAULT_MAX_MAINTENANCE_RATIO})",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Emit a one-line human summary instead of JSON to stdout",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to atomically write the report JSON",
    )
    args = parser.parse_args(argv)

    return run_classifier(
        list_prs=lambda: default_list_prs(args.repo, args.limit),
        limit=args.limit,
        stale_days=args.stale_days,
        max_maintenance_ratio=args.max_maintenance_ratio,
        out_file=args.out,
        summary=args.summary,
    )


if __name__ == "__main__":
    sys.exit(main())
