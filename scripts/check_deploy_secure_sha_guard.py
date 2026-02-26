#!/usr/bin/env python3
"""Guard deploy-secure SHA verification hardening against workflow regressions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Violation:
    path: str
    message: str


WORKFLOW_PATH = Path(".github/workflows/deploy-secure.yml")
SHA_STEP_NAME = "Post-deploy SHA verification"
ROLLBACK_GATE_STEP_NAME = "Determine rollback requirement"
ROLLBACK_STEP_NAME = "Rollback on failure"

REQUIRED_MARKERS: dict[str, str] = {
    "ec2_user_command": "sudo -u ec2-user git -C /home/ec2-user/aragora rev-parse HEAD",
    "safe_directory_fallback": "git -C /home/ec2-user/aragora -c safe.directory=/home/ec2-user/aragora rev-parse HEAD",
    "ssm_timeout": "--timeout-seconds 60",
    "stdout_diagnostics": "::warning::SHA stdout for $INST_ID:",
    "stderr_diagnostics": "::warning::SHA stderr for $INST_ID:",
}


ROLLBACK_GATE_REQUIRED_MARKERS: dict[str, str] = {
    "rollback_gate_id": "id: rollback_gate",
    "rollback_gate_condition": "if: always() && steps.deploy.conclusion != 'skipped'",
    "rollback_should_output": 'echo "should_rollback=$SHOULD_ROLLBACK" >> "$GITHUB_OUTPUT"',
    "rollback_reason_output": 'echo "rollback_reason=$ROLLBACK_REASON" >> "$GITHUB_OUTPUT"',
}


ROLLBACK_STEP_REQUIRED_MARKERS: dict[str, str] = {
    "rollback_condition": "if: steps.rollback_gate.outputs.should_rollback == 'true'",
    "rollback_reason_reference": "steps.rollback_gate.outputs.rollback_reason",
    "rollback_state_file": "/tmp/aragora_deploy_state",
    "rollback_previous_commit": "git checkout $PREVIOUS_COMMIT",
}


def _extract_step_blocks(workflow_text: str, step_name: str) -> list[str]:
    lines = workflow_text.splitlines()
    blocks: list[str] = []
    step_marker = f"- name: {step_name}"
    for start_idx, line in enumerate(lines):
        if line.strip() == step_marker:
            start_indent = len(line) - len(line.lstrip())
            block_lines: list[str] = []
            for inner in lines[start_idx + 1 :]:
                stripped = inner.lstrip()
                indent = len(inner) - len(stripped)
                if stripped.startswith("- name:") and indent == start_indent:
                    break
                if indent <= max(start_indent - 2, 0) and re.match(
                    r"^[A-Za-z0-9_-]+:\s*$", stripped
                ):
                    break
                block_lines.append(inner)
            blocks.append("\n".join(block_lines))
    return blocks


def find_sha_verification_violations(workflow_text: str) -> list[str]:
    violations: list[str] = []
    blocks = _extract_step_blocks(workflow_text, SHA_STEP_NAME)
    if not blocks:
        return ["missing `Post-deploy SHA verification` step"]
    block = blocks[-1]

    for name, marker in REQUIRED_MARKERS.items():
        if marker not in block:
            violations.append(f"missing required marker `{name}`: {marker}")
    return violations


def find_rollback_guard_violations(workflow_text: str) -> list[str]:
    violations: list[str] = []

    gate_blocks = _extract_step_blocks(workflow_text, ROLLBACK_GATE_STEP_NAME)
    if not gate_blocks:
        violations.append("missing `Determine rollback requirement` step")
    else:
        if len(gate_blocks) < 2:
            violations.append(
                "expected rollback gate parity for staging+production (>=2 `Determine rollback requirement` steps)"
            )
        for index, gate_block in enumerate(gate_blocks, start=1):
            for name, marker in ROLLBACK_GATE_REQUIRED_MARKERS.items():
                if marker not in gate_block:
                    violations.append(
                        f"gate step #{index} missing required marker `{name}`: {marker}"
                    )
        if not any("steps.sha_verify.conclusion" in block for block in gate_blocks):
            violations.append(
                "missing production rollback-gate marker `steps.sha_verify.conclusion`"
            )

    rollback_blocks = _extract_step_blocks(workflow_text, ROLLBACK_STEP_NAME)
    if not rollback_blocks:
        violations.append("missing `Rollback on failure` step")
    else:
        if len(rollback_blocks) < 2:
            violations.append(
                "expected rollback step parity for staging+production (>=2 `Rollback on failure` steps)"
            )
        for index, rollback_block in enumerate(rollback_blocks, start=1):
            for name, marker in ROLLBACK_STEP_REQUIRED_MARKERS.items():
                if marker not in rollback_block:
                    violations.append(
                        f"rollback step #{index} missing required marker `{name}`: {marker}"
                    )

    return violations


def check_repo(repo_root: Path) -> list[Violation]:
    workflow_file = repo_root / WORKFLOW_PATH
    if not workflow_file.exists():
        return [Violation(path=str(WORKFLOW_PATH), message="missing workflow file")]

    text = workflow_file.read_text(encoding="utf-8")
    messages = find_sha_verification_violations(text)
    messages.extend(find_rollback_guard_violations(text))
    return [Violation(path=str(WORKFLOW_PATH), message=message) for message in messages]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enforce post-deploy SHA verification hardening in deploy-secure workflow."
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root to check",
    )
    args = parser.parse_args()

    violations = check_repo(Path(args.repo_root).resolve())
    if not violations:
        print("Deploy secure SHA guard check passed")
        return 0

    print("Deploy secure SHA guard violations detected:")
    for v in violations:
        print(f"- {v.path}: {v.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
