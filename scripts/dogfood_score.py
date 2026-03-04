#!/usr/bin/env python3
"""Score baseline vs enhanced dogfood debate outputs with deterministic metrics."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aragora.debate.output_quality import OutputContract, validate_output_against_contract
from aragora.debate.repo_grounding import assess_repo_grounding, extract_repo_paths


TIMEOUT_PREFIX = "ARAGORA_TIMEOUT_JSON="
CREATE_ACTION_RE = re.compile(
    r"(?i)\b(create|add|build|scaffold|introduce|implement new|spin up)\b"
)

STANDARD_SECTIONS = [
    "Ranked High-Level Tasks",
    "Suggested Subtasks",
    "Owner module / file paths",
    "Test Plan",
    "Rollback Plan",
    "Gate Criteria",
]


@dataclass
class RunMetrics:
    name: str
    timed_out: bool
    timeout_seconds: int | None
    elapsed_seconds: float | None
    final_answer_chars: int
    quality_score_10: float | None
    practicality_score_10: float | None
    verified_paths_ratio: float | None
    duplicate_existing_create_ratio: float | None
    mentioned_paths: int
    existing_paths: int
    missing_paths: int
    new_paths: int
    defects: list[str]
    timeout_payload: dict[str, Any] | None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _load_timeout_payload(report_path: Path | None, stdout_text: str) -> dict[str, Any] | None:
    if report_path and report_path.exists():
        raw = _read_text(report_path).strip()
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass

    for line in stdout_text.splitlines():
        if not line.startswith(TIMEOUT_PREFIX):
            continue
        raw = line[len(TIMEOUT_PREFIX) :].strip()
        if not raw:
            continue
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            continue
    return None


def _extract_final_answer(stdout_text: str) -> str:
    marker = "FINAL ANSWER:"
    if marker not in stdout_text:
        return ""
    idx = stdout_text.find(marker)
    chunk = stdout_text[idx + len(marker) :]
    lines = chunk.splitlines()
    # Remove leading separators/empty lines after marker.
    while lines and (not lines[0].strip() or set(lines[0].strip()) <= {"="}):
        lines.pop(0)

    stop_prefixes = (
        "[path-check]",
        "[quality]",
        "WHY THIS ANSWER:",
        "DISSENTING VIEWS:",
        TIMEOUT_PREFIX,
    )
    out_lines: list[str] = []
    for line in lines:
        if line.strip().startswith(stop_prefixes):
            break
        out_lines.append(line)
    return "\n".join(out_lines).strip()


def _duplicate_existing_create_ratio(answer: str, repo_root: Path) -> float | None:
    if not answer.strip():
        return None

    total_create_paths = 0
    duplicate_existing = 0
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line or not CREATE_ACTION_RE.search(line):
            continue
        paths = extract_repo_paths(line)
        if not paths:
            continue
        for rel in paths:
            total_create_paths += 1
            if (repo_root / rel).exists():
                duplicate_existing += 1

    if total_create_paths == 0:
        return None
    return round(duplicate_existing / total_create_paths, 4)


def _score_run(
    name: str,
    stdout_path: Path,
    timeout_report_path: Path | None,
    repo_root: Path,
) -> RunMetrics:
    stdout_text = _read_text(stdout_path)
    timeout_payload = _load_timeout_payload(timeout_report_path, stdout_text)
    final_answer = _extract_final_answer(stdout_text)

    timed_out = bool(timeout_payload and timeout_payload.get("status") == "timeout")
    timeout_seconds = (
        int(timeout_payload["timeout_seconds"])
        if timeout_payload and "timeout_seconds" in timeout_payload
        else None
    )
    elapsed_seconds = (
        float(timeout_payload["elapsed_seconds"])
        if timeout_payload and "elapsed_seconds" in timeout_payload
        else None
    )

    if not final_answer:
        return RunMetrics(
            name=name,
            timed_out=timed_out,
            timeout_seconds=timeout_seconds,
            elapsed_seconds=elapsed_seconds,
            final_answer_chars=0,
            quality_score_10=None,
            practicality_score_10=None,
            verified_paths_ratio=None,
            duplicate_existing_create_ratio=None,
            mentioned_paths=0,
            existing_paths=0,
            missing_paths=0,
            new_paths=0,
            defects=["no_final_answer_payload"],
            timeout_payload=timeout_payload,
        )

    contract = OutputContract(
        required_sections=STANDARD_SECTIONS,
        require_json_payload=False,
        require_gate_thresholds=True,
        require_rollback_triggers=True,
        require_owner_paths=True,
        require_repo_path_existence=True,
        require_practicality_checks=True,
    )
    quality = validate_output_against_contract(final_answer, contract, repo_root=str(repo_root))
    grounding = assess_repo_grounding(
        final_answer, repo_root=str(repo_root), require_owner_paths=True
    )

    total_paths = len(grounding.mentioned_paths)
    verified_ratio = (
        round(len(grounding.existing_paths) / total_paths, 4) if total_paths > 0 else 0.0
    )
    duplicate_ratio = _duplicate_existing_create_ratio(final_answer, repo_root)

    return RunMetrics(
        name=name,
        timed_out=timed_out,
        timeout_seconds=timeout_seconds,
        elapsed_seconds=elapsed_seconds,
        final_answer_chars=len(final_answer),
        quality_score_10=quality.quality_score_10,
        practicality_score_10=quality.practicality_score_10,
        verified_paths_ratio=verified_ratio,
        duplicate_existing_create_ratio=duplicate_ratio,
        mentioned_paths=total_paths,
        existing_paths=len(grounding.existing_paths),
        missing_paths=len(grounding.missing_paths),
        new_paths=len(grounding.new_paths),
        defects=list(quality.defects),
        timeout_payload=timeout_payload,
    )


def _composite_score(run: RunMetrics) -> float:
    if run.timed_out or run.final_answer_chars == 0:
        return 0.0

    verified = run.verified_paths_ratio if run.verified_paths_ratio is not None else 0.0
    dup_penalty = (
        1.0 - run.duplicate_existing_create_ratio
        if run.duplicate_existing_create_ratio is not None
        else 1.0
    )
    quality = (run.quality_score_10 or 0.0) / 10.0
    practical = (run.practicality_score_10 or 0.0) / 10.0

    # Weighted blend emphasizing path grounding and practical quality.
    score = (0.35 * verified) + (0.25 * dup_penalty) + (0.2 * quality) + (0.2 * practical)
    return round(score, 4)


def _build_summary(baseline: RunMetrics, enhanced: RunMetrics) -> dict[str, Any]:
    baseline_score = _composite_score(baseline)
    enhanced_score = _composite_score(enhanced)
    timeout_rate = round(
        (int(baseline.timed_out) + int(enhanced.timed_out)) / 2.0,
        4,
    )

    winner = "tie"
    if enhanced_score > baseline_score:
        winner = "enhanced"
    elif baseline_score > enhanced_score:
        winner = "baseline"

    return {
        "timeout_rate": timeout_rate,
        "composite_scores": {"baseline": baseline_score, "enhanced": enhanced_score},
        "deltas": {
            "quality_score_10": (
                None
                if baseline.quality_score_10 is None or enhanced.quality_score_10 is None
                else round(enhanced.quality_score_10 - baseline.quality_score_10, 4)
            ),
            "practicality_score_10": (
                None
                if baseline.practicality_score_10 is None or enhanced.practicality_score_10 is None
                else round(enhanced.practicality_score_10 - baseline.practicality_score_10, 4)
            ),
            "verified_paths_ratio": (
                None
                if baseline.verified_paths_ratio is None or enhanced.verified_paths_ratio is None
                else round(enhanced.verified_paths_ratio - baseline.verified_paths_ratio, 4)
            ),
            "duplicate_existing_create_ratio": (
                None
                if baseline.duplicate_existing_create_ratio is None
                or enhanced.duplicate_existing_create_ratio is None
                else round(
                    enhanced.duplicate_existing_create_ratio
                    - baseline.duplicate_existing_create_ratio,
                    4,
                )
            ),
        },
        "winner": winner,
    }


def _write_markdown(
    path: Path, baseline: RunMetrics, enhanced: RunMetrics, summary: dict[str, Any]
) -> None:
    lines = [
        "# Dogfood A/B Score",
        "",
        "| Metric | Baseline | Enhanced |",
        "|---|---:|---:|",
        f"| timed_out | {baseline.timed_out} | {enhanced.timed_out} |",
        f"| quality_score_10 | {baseline.quality_score_10} | {enhanced.quality_score_10} |",
        f"| practicality_score_10 | {baseline.practicality_score_10} | {enhanced.practicality_score_10} |",
        f"| verified_paths_ratio | {baseline.verified_paths_ratio} | {enhanced.verified_paths_ratio} |",
        f"| duplicate_existing_create_ratio | {baseline.duplicate_existing_create_ratio} | {enhanced.duplicate_existing_create_ratio} |",
        f"| composite_score | {summary['composite_scores']['baseline']} | {summary['composite_scores']['enhanced']} |",
        "",
        f"- timeout_rate: {summary['timeout_rate']}",
        f"- winner: {summary['winner']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-stdout", required=True)
    parser.add_argument("--enhanced-stdout", required=True)
    parser.add_argument("--baseline-timeout-report")
    parser.add_argument("--enhanced-timeout-report")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()

    baseline = _score_run(
        name="baseline",
        stdout_path=Path(args.baseline_stdout).expanduser(),
        timeout_report_path=(
            Path(args.baseline_timeout_report).expanduser()
            if args.baseline_timeout_report
            else None
        ),
        repo_root=repo_root,
    )
    enhanced = _score_run(
        name="enhanced",
        stdout_path=Path(args.enhanced_stdout).expanduser(),
        timeout_report_path=(
            Path(args.enhanced_timeout_report).expanduser()
            if args.enhanced_timeout_report
            else None
        ),
        repo_root=repo_root,
    )
    summary = _build_summary(baseline, enhanced)

    payload = {
        "baseline": asdict(baseline),
        "enhanced": asdict(enhanced),
        "summary": summary,
    }

    output_json = Path(args.output_json).expanduser()
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        _write_markdown(Path(args.output_md).expanduser(), baseline, enhanced, summary)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
