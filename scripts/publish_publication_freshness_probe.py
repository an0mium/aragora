"""Publish a recurring publication-freshness probe receipt.

Aggregates three existing repo-tracked drift signals into a single
machine-readable receipt + human-readable status surface so an
operator can answer "are the published surfaces aligned with the live
state of the repo right now?" in one command instead of three.

Inputs
------
1. ``scripts/check_canonical_metrics.py --all`` (--json) for the
   canonical-claim ledger. Failures here mean repo-tracked docs are
   numerically out of sync with the live codebase.
2. ``scripts/reconcile_status_docs.py --json`` for status-doc
   reconciliation findings (GA_CHECKLIST, ROADMAP, connectors STATUS,
   etc.). Warnings here mean docs are aging past their freshness
   policy.
3. ``docs/status/generated/benchmark_truth_artifacts/*/latest.json``
   for benchmark truth artifact age. Stale truth artifacts mean B0
   measurements were not refreshed within the configured window.

Outputs
-------
- ``docs/status/generated/publication_freshness_probe/latest.json``:
  canonical machine-readable payload.
- ``docs/status/generated/publication_freshness_probe/probe-<ts>.json``:
  timestamped historical snapshot.
- ``docs/status/PUBLICATION_FRESHNESS_PROBE_STATUS.md`` (when
  ``--render-markdown`` is given): stable status surface.

This script is intentionally additive and read-only relative to the
sources it inspects. It never mutates the underlying canonical
metrics, status docs, or benchmark artifacts; it only writes its own
probe receipts into a dedicated published directory.

Non-goals
~~~~~~~~~
- Does not regenerate any of the source data.
- Does not import the ``aragora`` package.
- Does not require any network or git access.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLISHED_ROOT = (
    DEFAULT_REPO_ROOT / "docs" / "status" / "generated" / "publication_freshness_probe"
)
DEFAULT_STATUS_MD = DEFAULT_REPO_ROOT / "docs" / "status" / "PUBLICATION_FRESHNESS_PROBE_STATUS.md"
DEFAULT_BENCHMARK_TRUTH_ROOT = (
    DEFAULT_REPO_ROOT / "docs" / "status" / "generated" / "benchmark_truth_artifacts"
)
DEFAULT_STALE_HOURS = 48.0


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _iso(value: dt.datetime) -> str:
    return value.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(value: Any) -> dt.datetime | None:
    if not value or not isinstance(value, str):
        return None
    candidate = value.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def _hours_between(a: dt.datetime, b: dt.datetime) -> float:
    return abs((a - b).total_seconds()) / 3600.0


def run_canonical_metrics_probe(
    *,
    repo_root: Path,
    runner: Any = None,
) -> dict[str, Any]:
    """Run check_canonical_metrics.py --all --json and summarise."""
    script = repo_root / "scripts" / "check_canonical_metrics.py"
    if not script.exists():
        return {
            "available": False,
            "reason": "script not present",
            "summary": {},
            "results": [],
        }
    runner = runner or subprocess.run
    try:
        proc = runner(
            [sys.executable, str(script), "--all"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=repo_root,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        return {
            "available": False,
            "reason": f"invocation failed: {exc}",
            "summary": {},
            "results": [],
        }
    if proc.returncode not in (0, 1):
        return {
            "available": True,
            "reason": f"non-zero exit ({proc.returncode}): {proc.stderr.strip()[:200]}",
            "summary": {},
            "results": [],
        }
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "available": True,
            "reason": "non-JSON output",
            "summary": {},
            "results": [],
        }
    results = payload.get("results") or []
    counts = {"pass": 0, "warn": 0, "fail": 0, "skip": 0}
    drift_records: list[dict[str, Any]] = []
    for entry in results:
        status = str(entry.get("status") or "").lower()
        if status in counts:
            counts[status] += 1
        if status in ("fail", "warn"):
            drift_records.append(
                {
                    "claim_id": entry.get("claim_id"),
                    "status": status,
                    "observed": entry.get("observed"),
                    "claimed": entry.get("claimed"),
                    "tolerance": entry.get("tolerance"),
                    "message": entry.get("message"),
                }
            )
    return {
        "available": True,
        "manifest_id": payload.get("manifest_id"),
        "summary": counts,
        "results": results,
        "drift_records": drift_records,
        "drift_count": len(drift_records),
    }


def run_reconcile_status_docs_probe(
    *,
    repo_root: Path,
    runner: Any = None,
) -> dict[str, Any]:
    """Run reconcile_status_docs.py --json and summarise."""
    script = repo_root / "scripts" / "reconcile_status_docs.py"
    if not script.exists():
        return {"available": False, "reason": "script not present", "summary": {}, "findings": []}
    runner = runner or subprocess.run
    try:
        proc = runner(
            [sys.executable, str(script), "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=repo_root,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        return {
            "available": False,
            "reason": f"invocation failed: {exc}",
            "summary": {},
            "findings": [],
        }
    if proc.returncode not in (0, 1):
        return {
            "available": True,
            "reason": f"non-zero exit ({proc.returncode}): {proc.stderr.strip()[:200]}",
            "summary": {},
            "findings": [],
        }
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "available": True,
            "reason": "non-JSON output",
            "summary": {},
            "findings": [],
        }
    findings = payload.get("findings") or []
    summary = payload.get("summary") or {}
    drift_records = [
        f for f in findings if str(f.get("severity") or "").lower() in {"critical", "warning"}
    ]
    return {
        "available": True,
        "generated": payload.get("generated"),
        "summary": summary,
        "findings": findings,
        "drift_records": drift_records,
        "drift_count": len(drift_records),
    }


def scan_benchmark_truth_artifacts(
    *,
    truth_root: Path,
    stale_hours: float,
    now: dt.datetime,
) -> dict[str, Any]:
    """Inspect every ``latest.json`` under benchmark_truth_artifacts/*."""
    if not truth_root.exists():
        return {
            "available": False,
            "reason": "truth root not present",
            "corpora": [],
            "drift_records": [],
            "drift_count": 0,
        }
    corpora: list[dict[str, Any]] = []
    drift: list[dict[str, Any]] = []
    for child in sorted(truth_root.iterdir()):
        if not child.is_dir():
            continue
        latest = child / "latest.json"
        if not latest.exists():
            continue
        try:
            payload = json.loads(latest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        generated_at = _parse_iso(payload.get("generated_at"))
        age_hours = _hours_between(now, generated_at) if generated_at else None
        is_stale = age_hours is not None and age_hours > stale_hours
        row = {
            "corpus_id": child.name,
            "latest_path": str(latest.relative_to(truth_root.parent.parent.parent)),
            "generated_at": payload.get("generated_at"),
            "age_hours": round(age_hours, 2) if age_hours is not None else None,
            "is_stale": bool(is_stale),
            "stale_threshold_hours": stale_hours,
            "coverage_status": (payload.get("coverage") or {}).get("status"),
        }
        corpora.append(row)
        if is_stale:
            drift.append(row)
    return {
        "available": True,
        "stale_threshold_hours": stale_hours,
        "corpora": corpora,
        "drift_records": drift,
        "drift_count": len(drift),
    }


def build_published_report(
    *,
    repo_root: Path = DEFAULT_REPO_ROOT,
    truth_root: Path = DEFAULT_BENCHMARK_TRUTH_ROOT,
    stale_hours: float = DEFAULT_STALE_HOURS,
    now: dt.datetime | None = None,
    canonical_runner: Any = None,
    reconcile_runner: Any = None,
) -> dict[str, Any]:
    """Build the full freshness-probe payload."""
    moment = now or _utc_now()
    canonical = run_canonical_metrics_probe(repo_root=repo_root, runner=canonical_runner)
    reconcile = run_reconcile_status_docs_probe(repo_root=repo_root, runner=reconcile_runner)
    benchmark = scan_benchmark_truth_artifacts(
        truth_root=truth_root, stale_hours=stale_hours, now=moment
    )

    total_drift = (
        int(canonical.get("drift_count") or 0)
        + int(reconcile.get("drift_count") or 0)
        + int(benchmark.get("drift_count") or 0)
    )

    if total_drift == 0:
        verdict = "fresh"
    elif total_drift <= 3:
        verdict = "minor_drift"
    else:
        verdict = "drift"

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _iso(moment),
        "repo_root": str(repo_root),
        "stale_threshold_hours": stale_hours,
        "verdict": verdict,
        "total_drift": total_drift,
        "sources": {
            "canonical_metrics": canonical,
            "reconcile_status_docs": reconcile,
            "benchmark_truth_artifacts": benchmark,
        },
    }


def publish_report_bundle(
    report: dict[str, Any],
    *,
    out_root: Path = DEFAULT_PUBLISHED_ROOT,
    snapshot_prefix: str = "probe",
) -> dict[str, Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    ts = report.get("generated_at") or _iso(_utc_now())
    sanitized = ts.replace(":", "").replace("-", "")
    snapshot_name = f"{snapshot_prefix}-{sanitized}.json"
    snapshot_path = out_root / snapshot_name
    latest_path = out_root / "latest.json"
    body = json.dumps(report, indent=2, sort_keys=True) + "\n"
    snapshot_path.write_text(body, encoding="utf-8")
    latest_path.write_text(body, encoding="utf-8")
    return {"snapshot": snapshot_path, "latest": latest_path}


def render_status_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Publication Freshness Probe Status",
        "",
        f"Generated at: {report.get('generated_at')}",
        f"Verdict: **{report.get('verdict')}**  (total drift: {report.get('total_drift')})",
        f"Stale threshold: {report.get('stale_threshold_hours')}h",
        "",
        "## Canonical Metrics",
        "",
    ]
    canonical = (report.get("sources") or {}).get("canonical_metrics") or {}
    if canonical.get("available"):
        summary = canonical.get("summary") or {}
        lines.append(
            "pass={pass} warn={warn} fail={fail} skip={skip}; drift_count={dc}".format(
                **{
                    "pass": summary.get("pass", 0),
                    "warn": summary.get("warn", 0),
                    "fail": summary.get("fail", 0),
                    "skip": summary.get("skip", 0),
                    "dc": canonical.get("drift_count", 0),
                }
            )
        )
        for record in canonical.get("drift_records") or []:
            lines.append(
                f"- [{record.get('status')}] {record.get('claim_id')}: "
                f"observed={record.get('observed')}, claimed={record.get('claimed')}"
            )
    else:
        lines.append(f"unavailable: {canonical.get('reason')}")

    lines.extend(["", "## Status-doc Reconciliation", ""])
    reconcile = (report.get("sources") or {}).get("reconcile_status_docs") or {}
    if reconcile.get("available"):
        summary = reconcile.get("summary") or {}
        lines.append(
            "critical={c} warning={w} info={i} total={t}; drift_count={dc}".format(
                c=summary.get("critical", 0),
                w=summary.get("warning", 0),
                i=summary.get("info", 0),
                t=summary.get("total", 0),
                dc=reconcile.get("drift_count", 0),
            )
        )
        for finding in reconcile.get("drift_records") or []:
            lines.append(
                f"- [{finding.get('severity')}] {finding.get('source')}: {finding.get('message')}"
            )
    else:
        lines.append(f"unavailable: {reconcile.get('reason')}")

    lines.extend(["", "## Benchmark Truth Artifacts", ""])
    benchmark = (report.get("sources") or {}).get("benchmark_truth_artifacts") or {}
    if benchmark.get("available"):
        lines.append(
            f"stale_threshold={benchmark.get('stale_threshold_hours')}h; "
            f"drift_count={benchmark.get('drift_count')}"
        )
        for row in benchmark.get("corpora") or []:
            marker = "STALE" if row.get("is_stale") else "ok"
            lines.append(
                f"- [{marker:5}] {row.get('corpus_id')}: age={row.get('age_hours')}h, "
                f"coverage={row.get('coverage_status')}"
            )
    else:
        lines.append(f"unavailable: {benchmark.get('reason')}")

    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help=f"Repository root (default: {DEFAULT_REPO_ROOT}).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_PUBLISHED_ROOT,
        help=f"Published directory (default: {DEFAULT_PUBLISHED_ROOT}).",
    )
    parser.add_argument(
        "--status-md",
        type=Path,
        default=DEFAULT_STATUS_MD,
        help=f"Markdown status surface (default: {DEFAULT_STATUS_MD}).",
    )
    parser.add_argument(
        "--truth-root",
        type=Path,
        default=DEFAULT_BENCHMARK_TRUTH_ROOT,
        help=f"Benchmark truth artifact root (default: {DEFAULT_BENCHMARK_TRUTH_ROOT}).",
    )
    parser.add_argument(
        "--stale-hours",
        type=float,
        default=DEFAULT_STALE_HOURS,
        help=f"Benchmark truth artifact stale threshold in hours (default: {DEFAULT_STALE_HOURS}).",
    )
    parser.add_argument("--json", action="store_true", help="Print the payload to stdout as JSON.")
    parser.add_argument(
        "--render-markdown",
        action="store_true",
        help="Also write the rendered markdown status surface to --status-md.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write any output files; print to stdout only.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_published_report(
        repo_root=args.repo_root,
        truth_root=args.truth_root,
        stale_hours=args.stale_hours,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    paths = publish_report_bundle(report, out_root=args.out_root)
    if args.render_markdown:
        args.status_md.parent.mkdir(parents=True, exist_ok=True)
        args.status_md.write_text(render_status_markdown(report), encoding="utf-8")
    if not args.json:
        print(
            f"published: latest={paths['latest']}; snapshot={paths['snapshot']}; "
            f"verdict={report.get('verdict')}; total_drift={report.get('total_drift')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
