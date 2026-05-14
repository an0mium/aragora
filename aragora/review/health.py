"""Operational health surfaces for the review queue and proof loop.

Reports staleness, count, and age across the write-side daemons that close
the proof loop:

- settlement receipts (``.aragora/review-queue/receipts``)
- briefs (``.aragora/review-queue/briefs``)
- boss-metrics ledger (``.aragora/overnight/boss_metrics.jsonl``)
- automation receipts (``.aragora/automation-receipts``)
- boss-loop launchd log (``.aragora/overnight/boss-loop-launchd.log``)
- watchdog log (``.aragora/overnight/watchdog.log``)
- B0 benchmark publication (``docs/status/B0_BENCHMARK_TRUTH_STATUS.md``)
- TW-03 rescue productization (``docs/status/TW03_RESCUE_PRODUCTIZATION_STATUS.md``)

Intentionally read-only and network-free; one call should answer
"can I trust today's proof-loop output?" without hitting GitHub.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aragora.review.invalidation_event_source import RECEIPTS_SUBDIR

UTC = timezone.utc

DEFAULT_BRIEFS_SUBDIR = "briefs"
DEFAULT_OVERNIGHT_REL = Path(".aragora") / "overnight"
DEFAULT_AUTOMATION_RECEIPTS_REL = Path(".aragora") / "automation-receipts"
DEFAULT_BOSS_METRICS_REL = DEFAULT_OVERNIGHT_REL / "boss_metrics.jsonl"
DEFAULT_BOSS_LOG_REL = DEFAULT_OVERNIGHT_REL / "boss-loop-launchd.log"
DEFAULT_WATCHDOG_LOG_REL = DEFAULT_OVERNIGHT_REL / "watchdog.log"
DEFAULT_B0_STATUS_REL = Path("docs") / "status" / "B0_BENCHMARK_TRUTH_STATUS.md"
DEFAULT_TW03_STATUS_REL = Path("docs") / "status" / "TW03_RESCUE_PRODUCTIZATION_STATUS.md"

# Status severities (ordered ascending so max() returns worst).
STATUS_FRESH = "fresh"
STATUS_AGING = "aging"
STATUS_STALE = "stale"
STATUS_MISSING = "missing"
STATUS_EMPTY = "empty"

_SEVERITY_RANK = {
    STATUS_FRESH: 0,
    STATUS_AGING: 1,
    STATUS_STALE: 2,
    STATUS_EMPTY: 3,
    STATUS_MISSING: 4,
}


@dataclass(frozen=True)
class SurfaceCheck:
    """A single health observation for one write-side surface."""

    name: str
    status: str
    count: int | None = None
    latest_mtime: datetime | None = None
    age_hours: float | None = None
    path: str | None = None
    detail: str | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "name": self.name,
            "status": self.status,
            "count": self.count,
            "latest_mtime": (
                self.latest_mtime.isoformat() if self.latest_mtime is not None else None
            ),
            "age_hours": (round(self.age_hours, 2) if self.age_hours is not None else None),
            "path": self.path,
            "detail": self.detail,
        }
        if self.extra:
            out["extra"] = dict(self.extra)
        return out


@dataclass(frozen=True)
class HealthReport:
    """Aggregated health across all write-side surfaces."""

    generated_at: datetime
    overall_status: str
    surfaces: list[SurfaceCheck]

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "overall_status": self.overall_status,
            "surfaces": [s.to_dict() for s in self.surfaces],
        }


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _file_mtime(path: Path) -> datetime | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    except OSError:
        return None
    return datetime.fromtimestamp(stat.st_mtime, tz=UTC)


def _age_hours(mtime: datetime, now: datetime) -> float:
    return (now - mtime).total_seconds() / 3600.0


def _classify_age(age_hours: float, warn_h: float, crit_h: float) -> str:
    if age_hours <= warn_h:
        return STATUS_FRESH
    if age_hours <= crit_h:
        return STATUS_AGING
    return STATUS_STALE


def _newest_in_dir(directory: Path, glob: str = "*") -> tuple[Path | None, datetime | None, int]:
    """Return (newest_path, newest_mtime, count) for files matching glob in directory."""
    if not directory.exists() or not directory.is_dir():
        return None, None, 0
    newest_mtime: datetime | None = None
    newest_path: Path | None = None
    count = 0
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        if glob != "*" and not entry.match(glob):
            continue
        count += 1
        mtime = _file_mtime(entry)
        if mtime is None:
            continue
        if newest_mtime is None or mtime > newest_mtime:
            newest_mtime = mtime
            newest_path = entry
    return newest_path, newest_mtime, count


def _check_directory_freshness(
    *,
    name: str,
    directory: Path,
    warn_h: float,
    crit_h: float,
    expect_nonempty: bool = True,
    glob: str = "*",
) -> SurfaceCheck:
    """Build a SurfaceCheck for a directory containing dated artifacts."""
    if not directory.exists():
        return SurfaceCheck(
            name=name,
            status=STATUS_MISSING,
            count=0,
            path=str(directory),
            detail=f"directory does not exist: {directory}",
        )
    newest_path, newest_mtime, count = _newest_in_dir(directory, glob=glob)
    if count == 0:
        return SurfaceCheck(
            name=name,
            status=STATUS_EMPTY if expect_nonempty else STATUS_FRESH,
            count=0,
            path=str(directory),
            detail="no files present",
        )
    now = _now()
    age = _age_hours(newest_mtime, now) if newest_mtime is not None else None
    status = _classify_age(age, warn_h, crit_h) if age is not None else STATUS_AGING
    return SurfaceCheck(
        name=name,
        status=status,
        count=count,
        latest_mtime=newest_mtime,
        age_hours=age,
        path=str(directory),
        detail=f"newest: {newest_path.name}" if newest_path is not None else None,
    )


def _check_file_freshness(
    *,
    name: str,
    path: Path,
    warn_h: float,
    crit_h: float,
    extra_counter: Callable[[Path], Mapping[str, Any]] | None = None,
) -> SurfaceCheck:
    mtime = _file_mtime(path)
    if mtime is None:
        return SurfaceCheck(
            name=name,
            status=STATUS_MISSING,
            path=str(path),
            detail=f"file does not exist: {path}",
        )
    now = _now()
    age = _age_hours(mtime, now)
    status = _classify_age(age, warn_h, crit_h)
    extra: dict[str, object] = {}
    if extra_counter is not None:
        try:
            extra.update(extra_counter(path))
        except Exception as exc:  # noqa: BLE001 - never block health on counter failure
            extra["counter_error"] = str(exc)
    return SurfaceCheck(
        name=name,
        status=status,
        latest_mtime=mtime,
        age_hours=age,
        path=str(path),
        extra=extra,
    )


def _count_jsonl_rows(path: Path) -> dict[str, object]:
    """Return total row count and rows-in-last-7d for a JSONL ledger."""
    total = 0
    recent_7d = 0
    cutoff = _now().timestamp() - 7 * 24 * 3600
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                total += 1
                # Best-effort parse for recorded_at_ts; missing rows skipped silently.
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = row.get("recorded_at_ts") or row.get("ts") or row.get("timestamp")
                if isinstance(ts, (int, float)) and ts >= cutoff:
                    recent_7d += 1
                elif isinstance(ts, str):
                    try:
                        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        if parsed.timestamp() >= cutoff:
                            recent_7d += 1
                    except ValueError:
                        continue
    except OSError:
        return {"row_count_total": total, "row_count_7d": recent_7d, "error": "io_error"}
    return {"row_count_total": total, "row_count_7d": recent_7d}


_LAST_UPDATED_RE = re.compile(r"^Last updated:\s*(\S+)", re.MULTILINE)


def _parse_status_doc_last_updated(path: Path) -> datetime | None:
    """Read the 'Last updated:' line from a status markdown doc."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _LAST_UPDATED_RE.search(text)
    if not match:
        return None
    token = match.group(1).strip().rstrip(".")
    # Common forms: 2026-04-30T16:01:56Z, 2026-04-30, 2026-04-30T03:27:10Z
    try:
        if "T" in token:
            return datetime.fromisoformat(token.replace("Z", "+00:00"))
        # bare date -> midnight UTC
        return datetime.fromisoformat(token).replace(tzinfo=UTC)
    except ValueError:
        return None


def _check_status_doc(
    *,
    name: str,
    path: Path,
    warn_h: float,
    crit_h: float,
) -> SurfaceCheck:
    if not path.exists():
        return SurfaceCheck(
            name=name,
            status=STATUS_MISSING,
            path=str(path),
            detail=f"status doc not present: {path}",
        )
    last = _parse_status_doc_last_updated(path)
    if last is None:
        return SurfaceCheck(
            name=name,
            status=STATUS_AGING,
            path=str(path),
            detail="could not parse 'Last updated:' line",
        )
    now = _now()
    age = _age_hours(last, now)
    status = _classify_age(age, warn_h, crit_h)
    return SurfaceCheck(
        name=name,
        status=status,
        latest_mtime=last,
        age_hours=age,
        path=str(path),
    )


_CRASH_PATTERN = re.compile(r"ModuleNotFoundError")
_EXIT_OK_PATTERN = re.compile(r"Boss loop exited with status 0")
_EXIT_FAIL_PATTERN = re.compile(r"Boss loop exited with status 1")


def _check_boss_loop_log(path: Path, warn_h: float, crit_h: float) -> SurfaceCheck:
    if not path.exists():
        return SurfaceCheck(
            name="boss_loop_log",
            status=STATUS_MISSING,
            path=str(path),
            detail="boss-loop launchd log not present",
        )
    mtime = _file_mtime(path)
    now = _now()
    age = _age_hours(mtime, now) if mtime is not None else None
    status = _classify_age(age, warn_h, crit_h) if age is not None else STATUS_AGING

    crashes_total = 0
    exits_ok_total = 0
    exits_fail_total = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if _CRASH_PATTERN.search(line):
                    crashes_total += 1
                elif _EXIT_OK_PATTERN.search(line):
                    exits_ok_total += 1
                elif _EXIT_FAIL_PATTERN.search(line):
                    exits_fail_total += 1
    except OSError:
        pass

    extra: dict[str, object] = {
        "crashes_total": crashes_total,
        "exits_ok_total": exits_ok_total,
        "exits_fail_total": exits_fail_total,
    }
    detail = f"crashes={crashes_total} ok={exits_ok_total} fail={exits_fail_total}"
    return SurfaceCheck(
        name="boss_loop_log",
        status=status,
        latest_mtime=mtime,
        age_hours=age,
        path=str(path),
        detail=detail,
        extra=extra,
    )


def _resolve_repo_root(override: Path | None) -> Path:
    if override is not None:
        return override
    cwd = Path.cwd()
    for candidate in (cwd, *cwd.parents):
        if (candidate / ".git").exists() or (candidate / ".aragora").exists():
            return candidate
    return cwd


def _resolve_aragora_state_dir(repo: Path) -> Path:
    """Return the shared ``.aragora`` state directory for repo-local health.

    Disposable worktrees usually do not carry the shared runtime state tree,
    so falling back to ``repo/.aragora`` would make all write-side surfaces
    look missing. This mirrors the local automation scripts' state-root
    convention while staying network-free and avoiding any git subprocess.
    """
    local_state = repo / ".aragora"
    if local_state.is_dir():
        return local_state

    candidates: list[Path] = []
    configured = os.environ.get("ARAGORA_AUTOMATION_STATE_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(Path.home() / "Development" / "aragora")

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved.name == ".aragora" and resolved.is_dir():
            return resolved
        candidate_state = resolved / ".aragora"
        if candidate_state.is_dir():
            return candidate_state
    return local_state


def gather_health(
    *,
    repo_root: Path | None = None,
    review_queue_root: Path | None = None,
    overnight_root: Path | None = None,
    automation_receipts_root: Path | None = None,
    warn_hours_settlement: float = 168.0,  # 7d - settlement is sporadic
    warn_hours_briefs: float = 168.0,
    warn_hours_boss_metrics: float = 48.0,
    warn_hours_automation: float = 24.0,
    warn_hours_status_doc: float = 168.0,  # 7d for weekly publications
    warn_hours_boss_log: float = 24.0,
    crit_multiplier: float = 4.0,
) -> HealthReport:
    """Gather all surface checks into a single HealthReport.

    All thresholds are configurable so tests can pin specific values.
    Critical thresholds are derived as ``warn_h * crit_multiplier`` unless
    overridden by callers needing per-surface critical tuning.
    """
    repo = _resolve_repo_root(repo_root)
    state_dir = _resolve_aragora_state_dir(repo)
    env_review_queue_root = os.environ.get("ARAGORA_REVIEW_QUEUE_ROOT")
    rq_root = (
        review_queue_root
        if review_queue_root is not None
        else Path(env_review_queue_root).expanduser()
        if env_review_queue_root
        else state_dir / "review-queue"
    )
    on_root = overnight_root if overnight_root is not None else state_dir / "overnight"
    auto_root = (
        automation_receipts_root
        if automation_receipts_root is not None
        else state_dir / "automation-receipts"
    )

    surfaces: list[SurfaceCheck] = []

    # 1. Settlement receipts
    surfaces.append(
        _check_directory_freshness(
            name="settlement_receipts",
            directory=rq_root / RECEIPTS_SUBDIR,
            warn_h=warn_hours_settlement,
            crit_h=warn_hours_settlement * crit_multiplier,
            glob="pr-*.json",
        )
    )

    # 2. Briefs (founder-dogfood; empty/old is informational, not critical)
    surfaces.append(
        _check_directory_freshness(
            name="briefs",
            directory=rq_root / DEFAULT_BRIEFS_SUBDIR,
            warn_h=warn_hours_briefs,
            crit_h=warn_hours_briefs * crit_multiplier,
            expect_nonempty=False,
            glob="pr-*.json",
        )
    )

    # 3. Boss metrics ledger
    surfaces.append(
        _check_file_freshness(
            name="boss_metrics",
            path=on_root / "boss_metrics.jsonl",
            warn_h=warn_hours_boss_metrics,
            crit_h=warn_hours_boss_metrics * crit_multiplier,
            extra_counter=_count_jsonl_rows,
        )
    )

    # 4. Automation receipts
    surfaces.append(
        _check_directory_freshness(
            name="automation_receipts",
            directory=auto_root,
            warn_h=warn_hours_automation,
            crit_h=warn_hours_automation * crit_multiplier,
            expect_nonempty=False,
        )
    )

    # 5. Boss-loop launchd log
    surfaces.append(
        _check_boss_loop_log(
            on_root / "boss-loop-launchd.log",
            warn_h=warn_hours_boss_log,
            crit_h=warn_hours_boss_log * crit_multiplier,
        )
    )

    # 6. Watchdog log
    surfaces.append(
        _check_file_freshness(
            name="watchdog_log",
            path=on_root / "watchdog.log",
            warn_h=warn_hours_boss_log,
            crit_h=warn_hours_boss_log * crit_multiplier,
        )
    )

    # 7. B0 benchmark publication status
    surfaces.append(
        _check_status_doc(
            name="b0_publication",
            path=repo / DEFAULT_B0_STATUS_REL,
            warn_h=warn_hours_status_doc,
            crit_h=warn_hours_status_doc * crit_multiplier,
        )
    )

    # 8. TW-03 rescue productization
    surfaces.append(
        _check_status_doc(
            name="tw03_rescue",
            path=repo / DEFAULT_TW03_STATUS_REL,
            warn_h=warn_hours_status_doc,
            crit_h=warn_hours_status_doc * crit_multiplier,
        )
    )

    overall = STATUS_FRESH
    for surface in surfaces:
        if _SEVERITY_RANK[surface.status] > _SEVERITY_RANK[overall]:
            overall = surface.status

    return HealthReport(
        generated_at=_now(),
        overall_status=overall,
        surfaces=surfaces,
    )


def render_text(report: HealthReport) -> str:
    """Render a HealthReport as a compact human-readable block."""
    lines = []
    lines.append("Aragora Review-Queue Health")
    lines.append(f"  generated_at:   {report.generated_at.isoformat()}")
    lines.append(f"  overall_status: {report.overall_status.upper()}")
    lines.append("")
    lines.append(f"{'surface':<24}{'status':<10}{'age':<14}{'count':<8}{'detail'}")
    lines.append("-" * 80)
    for surface in report.surfaces:
        age_str = (
            f"{surface.age_hours:.1f}h"
            if surface.age_hours is not None and surface.age_hours < 96
            else (f"{surface.age_hours / 24:.1f}d" if surface.age_hours is not None else "n/a")
        )
        count_str = str(surface.count) if surface.count is not None else "-"
        detail = surface.detail or ""
        lines.append(f"{surface.name:<24}{surface.status:<10}{age_str:<14}{count_str:<8}{detail}")
    return "\n".join(lines)


__all__ = [
    "HealthReport",
    "STATUS_AGING",
    "STATUS_EMPTY",
    "STATUS_FRESH",
    "STATUS_MISSING",
    "STATUS_STALE",
    "SurfaceCheck",
    "gather_health",
    "render_text",
]
