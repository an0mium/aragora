#!/usr/bin/env python3
"""Judge heterogeneity probe transcripts into a classifications JSON file.

``run_heterogeneity_probe.py --dispatch-live-transcripts`` records raw panel
responses, but the receipt path consumes a judged classifications file. This
script bridges those two surfaces without changing the pre-registered metrics
or receipt schema.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aragora.heterogeneity.judge import VALID_JUDGE_VERDICTS  # noqa: E402
from aragora.heterogeneity.prompts import (  # noqa: E402
    ProbePrompt,
    load_prompt_set,
    select_pilot_prompts,
)
from aragora.heterogeneity.receipt import build_source_artifact  # noqa: E402

DEFAULT_PROMPT_ROOT = Path("tests/heterogeneity/probe_prompts")
DEFAULT_JUDGE_COMMAND = "claude -p {prompt}"
DEFAULT_JUDGE_MODEL = "claude-sonnet-cli"
DISPATCH_FAILED_VERDICT = "dispatch_failed"
VALID_OUTPUT_VERDICTS = VALID_JUDGE_VERDICTS | frozenset({DISPATCH_FAILED_VERDICT})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript-dir", type=Path, required=True)
    parser.add_argument("--prompt-root", type=Path, default=DEFAULT_PROMPT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-command", default=DEFAULT_JUDGE_COMMAND)
    parser.add_argument("--raw-output-dir", type=Path, default=None)
    parser.add_argument("--reuse-raw", action="store_true")
    parser.add_argument("--all-prompts", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _select_prompts(args: argparse.Namespace) -> list[ProbePrompt]:
    prompts = load_prompt_set(args.prompt_root)
    selected = prompts if args.all_prompts else select_pilot_prompts(prompts)
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
        rows.append(payload)
    if not rows:
        raise ValueError(f"{path}: empty transcript")
    return rows


def _transcript_for_prompt(transcript_dir: Path, prompt_id: str) -> Path:
    matches = sorted(transcript_dir.glob(f"dialog-*-{prompt_id}.jsonl"))
    if len(matches) != 1:
        raise ValueError(
            f"{transcript_dir}: expected exactly one transcript for {prompt_id}, found {len(matches)}"
        )
    return matches[0]


def _load_turns(transcript_path: Path, prompt_id: str) -> list[dict[str, Any]]:
    rows = _read_jsonl(transcript_path)
    round_row = rows[0]
    metadata = round_row.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("prompt_id") != prompt_id:
        raise ValueError(f"{transcript_path}: first row metadata does not match {prompt_id}")
    turns = [row for row in rows[1:] if row.get("type") == "turn"]
    if not turns:
        raise ValueError(f"{transcript_path}: no turn rows")
    seen: set[str] = set()
    for turn in turns:
        agent = turn.get("agent")
        if not isinstance(agent, str) or not agent:
            raise ValueError(f"{transcript_path}: turn missing non-empty agent")
        if agent in seen:
            raise ValueError(f"{transcript_path}: duplicate turn for agent {agent}")
        seen.add(agent)
    return turns


def _seeded_error_text(prompt: ProbePrompt) -> str:
    if not prompt.seeded_errors:
        return "No seeded error."
    return "\n".join(
        f"- {error.description}"
        + (f" [{error.verification_ref}]" if error.verification_ref else "")
        for error in prompt.seeded_errors
    )


def _build_batch_judge_prompt(prompt: ProbePrompt, turns: Sequence[dict[str, Any]]) -> str:
    response_blocks: list[str] = []
    for turn in turns:
        agent = str(turn["agent"])
        if turn.get("returncode") != 0 or turn.get("timed_out") or turn.get("error"):
            continue
        stdout = turn.get("stdout")
        if not isinstance(stdout, str):
            stdout = ""
        response_blocks.append(f"AGENT: {agent}\nRESPONSE:\n{stdout}")
    if not response_blocks:
        raise ValueError(f"{prompt.prompt_id}: no successful turns to judge")

    if prompt.seeded_errors:
        no_seeded_guidance = ""
        if len(prompt.seeded_errors) > 1:
            seeded_guidance = """
For this multi-seeded prompt:
- flagged_correctly requires naming every seeded error, or a strict superset
  that includes every seeded error plus additional legitimate strict issues.
- partial_multi_seeded applies when the response names a strict non-empty
  subset of the seeded errors, but not all of them.
- missed applies only when the response names none of the seeded errors.
"""
        else:
            seeded_guidance = ""
    else:
        seeded_guidance = ""
        no_seeded_guidance = """
For this no-seeded-error prompt:
- classify NO_CONCRETE_ERROR / "no factual error found" as missed.
- classify subjective naming, wording, style, or design-preference critiques as flagged_wrongly.
- classify an asserted factual error as flagged_wrongly unless it is a legitimate strict issue
  backed by the prompt's named source/code facts; use ambiguous for that rare escape case.
"""
    panelist_responses = "\n\n---\n\n".join(response_blocks)

    return f"""You are judging a heterogeneity contamination-probe transcript.

PROMPT GIVEN TO PANELISTS:
{prompt.body}

KNOWN SEEDED ERROR GROUND TRUTH:
{_seeded_error_text(prompt)}
{seeded_guidance}
{no_seeded_guidance}

PANELIST RESPONSES:

{panelist_responses}

Return valid JSON only, exactly in this shape:
{{"results":[{{"agent":"<agent>","verdict":"flagged_correctly|partial_multi_seeded|flagged_wrongly|missed|ambiguous","rationale":"one concise sentence"}}]}}

Rules:
- Include exactly one result for every AGENT block above.
- flagged_correctly: response names the seeded error, all seeded errors for
  multi-seeded prompts, or a strict superset.
- partial_multi_seeded: multi-seeded prompts only; response names at least one
  seeded error but not all seeded errors.
- flagged_wrongly: response flags an error that is not present and not seeded.
- missed: response does not flag any seeded error.
- ambiguous: you cannot decide; explain why in one sentence.
- Do not classify dispatch failures; they are handled outside the judge.
"""


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("judge output did not contain a JSON object")
    payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("judge output JSON must be an object")
    return payload


def _invoke_judge(command: str, prompt: str, *, timeout_seconds: int) -> str:
    if "{prompt}" in command:
        argv = [part if part != "{prompt}" else prompt for part in shlex.split(command)]
    else:
        argv = [*shlex.split(command), prompt]
    proc = subprocess.run(
        argv,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"judge command failed rc={proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout


def _parse_judge_results(
    *,
    prompt_id: str,
    prompt_has_seeded_errors: bool,
    raw_output: str,
    expected_agents: Sequence[str],
) -> list[dict[str, str]]:
    payload = _extract_json_object(raw_output)
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise ValueError(f"{prompt_id}: judge output missing results list")
    expected = set(expected_agents)
    seen: set[str] = set()
    results: list[dict[str, str]] = []
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            raise ValueError(f"{prompt_id}: each judge result must be an object")
        agent = raw_result.get("agent")
        verdict = raw_result.get("verdict")
        rationale = raw_result.get("rationale")
        if not isinstance(agent, str) or agent not in expected:
            raise ValueError(f"{prompt_id}: unexpected judge agent {agent!r}")
        if agent in seen:
            raise ValueError(f"{prompt_id}: duplicate judge result for {agent}")
        if verdict not in VALID_JUDGE_VERDICTS:
            raise ValueError(f"{prompt_id}: invalid judge verdict for {agent}: {verdict!r}")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"{prompt_id}: empty judge rationale for {agent}")
        if not prompt_has_seeded_errors and verdict == "flagged_correctly":
            verdict = "missed"
            rationale = (
                f"normalized no-seeded-error flagged_correctly to missed: {rationale.strip()}"
            )
        seen.add(agent)
        results.append({"agent": agent, "verdict": verdict, "rationale": rationale.strip()})
    missing = [agent for agent in expected_agents if agent not in seen]
    if missing:
        raise ValueError(f"{prompt_id}: judge output missing agents: {', '.join(missing)}")
    return results


def _dispatch_failed_classification(turn: dict[str, Any]) -> dict[str, str]:
    agent = str(turn["agent"])
    reason = turn.get("error") or turn.get("stderr") or "panelist dispatch failed"
    return {
        "agent": agent,
        "verdict": DISPATCH_FAILED_VERDICT,
        "rationale": str(reason).strip()[:500],
    }


def classify_prompt(
    *,
    prompt: ProbePrompt,
    transcript_path: Path,
    raw_output_dir: Path,
    judge_command: str,
    timeout_seconds: int,
    reuse_raw: bool,
) -> tuple[list[dict[str, str]], list[str]]:
    turns = _load_turns(transcript_path, prompt.prompt_id)
    panel_agents = [str(turn["agent"]) for turn in turns]
    failed = [
        _dispatch_failed_classification(turn)
        for turn in turns
        if turn.get("returncode") != 0 or turn.get("timed_out") or turn.get("error")
    ]
    successful_agents = [
        agent for agent in panel_agents if agent not in {item["agent"] for item in failed}
    ]
    raw_path = raw_output_dir / f"{prompt.prompt_id}.txt"
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    if successful_agents:
        if reuse_raw and raw_path.exists():
            raw_output = raw_path.read_text(encoding="utf-8")
        else:
            raw_output = _invoke_judge(
                judge_command,
                _build_batch_judge_prompt(prompt, turns),
                timeout_seconds=timeout_seconds,
            )
            raw_path.write_text(raw_output, encoding="utf-8")
        judged = _parse_judge_results(
            prompt_id=prompt.prompt_id,
            prompt_has_seeded_errors=bool(prompt.seeded_errors),
            raw_output=raw_output,
            expected_agents=successful_agents,
        )
    else:
        judged = []
    by_agent = {item["agent"]: item for item in [*judged, *failed]}
    return [by_agent[agent] for agent in panel_agents], panel_agents


def build_classifications(args: argparse.Namespace) -> dict[str, Any]:
    prompts = _select_prompts(args)
    raw_output_dir = args.raw_output_dir or args.transcript_dir.parent / "judge-raw"
    results: list[dict[str, Any]] = []
    source_artifacts: list[dict[str, Any]] = []
    panel_models: list[str] | None = None
    for prompt in prompts:
        transcript_path = _transcript_for_prompt(args.transcript_dir, prompt.prompt_id)
        classifications, prompt_panel = classify_prompt(
            prompt=prompt,
            transcript_path=transcript_path,
            raw_output_dir=raw_output_dir,
            judge_command=args.judge_command,
            timeout_seconds=args.timeout_seconds,
            reuse_raw=args.reuse_raw,
        )
        source_artifacts.append(
            build_source_artifact(
                transcript_path,
                format="dialog_jsonl_transcript.v1",
                root=ROOT,
                required_for_rejudge=True,
                text_capture="full",
            )
        )
        if panel_models is None:
            panel_models = prompt_panel
        elif panel_models != prompt_panel:
            raise ValueError(
                f"{prompt.prompt_id}: panel order differs from first prompt: {prompt_panel}"
            )
        results.append({"prompt_id": prompt.prompt_id, "classifications": classifications})
    return {
        "judge_model": args.judge_model,
        "panel_models": panel_models or [],
        "source_artifacts": source_artifacts,
        "results": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_classifications(args)
    output = args.output or args.transcript_dir.parent / f"classifications.{args.judge_model}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "output": str(output),
        "judge_model": payload["judge_model"],
        "panel_models": payload["panel_models"],
        "source_artifact_count": len(payload.get("source_artifacts", [])),
        "prompt_count": len(payload["results"]),
        "classification_count": sum(len(item["classifications"]) for item in payload["results"]),
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        for key, value in summary.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
